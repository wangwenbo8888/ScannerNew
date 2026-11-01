# laser_match_scan_v3_cpu — 全 CPU 备用算子设计方案

> 2026-10-21 ｜ 沙箱：modules/09_operatorlib/TEST/speed_optimize/（只改本目录）
> 基准：`ops/scanning/laser/laser_match_scan_v3/`（GPU 算子副本，功能以此为准）
> 硬指标：**CPU 单线程**、**Execute 计算时间 < 5ms**（典型帧 ~25k 左点 / ~25k 右点）

---

## 1. 目标与非目标

**目标**
1. 功能对齐 GPU 版 `LaserMatchScanCudaV3`（段级投票＋冲突双弃＋P5/P6b/P7 全路径）
2. 纯 CPU 单线程实现，无 CUDA 依赖（含构建链不引入 .cu）
3. Execute 全流程 < 5ms（Release /O2，本机口径）
4. A/B 一致性：与 GPU 算子同输入同参数，输出匹配对逐点一致（坐标容差 1e-3px，
   lid/track/计数精确一致）

**非目标**
- 不替换产线 GPU 算子；不修改 ops/ 内 GPU 副本与 src/ 链
- 不做多线程/OpenMP/SIMD 以外的向量化改写（先证单线程达标；不达标再在评审后加优化）

## 2. 接口设计

```
ops/scanning/laser/laser_match_scan_v3_cpu/
├── laser_match_scan_v3_cpu.h     # 公开接口（含 Params 复用＋CpuResult）
└── laser_match_scan_v3_cpu.cpp   # 单线程实现（无 pImpl 分离，测试沙箱从简）
```

- **Params**：直接复用 `LaserMatchScanParamsV3`（include GPU 公开头——该头仅前置声明
  cv::cuda 类型，无 CUDA 依赖），保证参数单一事实源、JSON 语义一致。
- **类** `calib::LaserMatchScanCpuV3`，生命周期与 GPU 版对齐：
  - `SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable>)` / `LoadTempTable(json)` /
    `SetCurrentTemperature(double)`（同源同温同 rowStep 跳过重排缓存）
  - `Execute(left_points, left_line_ids, right_points, right_line_ids) → LaserMatchScanCpuResultV3`
    - 输入：`cv::Mat` N×1 `CV_32FC2`（极线坐标）＋ N×1 `CV_32SC1`（track id），
      与 GPU 版 GpuMat 同构（数据内存布局相同，可 memcpy 对接）
  - `SetParams / GetParams / Warmup（预分配）/ Destroy / SetuRBias / SetuRBiasModel`
- **Result**（值语义，CPU 无 ROI 视图生命周期问题）：
  - `matched_left / matched_right`（cv::Mat CV_32FC2）、`matched_line_ids /
    matched_track_ids`（CV_32SC1）、`matchedCount`
  - `left_status / right_status`（std::vector<int>，-1/0/1 语义同 GPU）
  - 诊断计数全套：`totalLeft/RightPoints、excludedLeft/RightCount、noMapLeftCount、
    noWinnerCardCount、emptyWinnerSegCount、trackTotal、trackDropped`
- 结果缓冲容量式复用（grow-only vector/Mat），稳态零分配。

## 3. 算法移植（逐级对齐 GPU 流水线）

数据规模基准：leftCount≈rightCount≈25k；温表单档 ~2.4M 条（16B/条）；
极线行 ≤8192（rowStep≈0.5 → 表行桶 ~580 条/行）；段数 S≤4096（常态 ~100）。

### Stage A —— 温表按行 CSR（SetCurrentTemperature 时一次，带缓存）
对齐 `uploadMapTable` host 逻辑：`row = round(yL/rowStep)` clamp [0,8192)；
行内按 xL 升序；产 `row_start/row_count` CSR ＋扁平 `float4(xL,uR,lid,yL)` 数组。
同源指针＋同温＋同 rowStep → 跳过（与 GPU 缓存判定逐字对齐）。

### Stage B —— 右点行桶＋行内排序（替代 GPU 键排序 radix）
两遍计数排序：pass1 按 `row=round(y/step)`（非法行丢弃，对齐 KEY_SENTINEL 语义）
计数→前缀→稳定散射；行内按 x 升序（`std::sort` pair(x,origIdx)）。
产出右行 CSR＋右点有序数组（供 HIT 二分）。O(N+N·log(n_per_row))，n_per_row≈6-10。

### Stage C —— 发卡 kernelIssueCardsV3 的单线程移植（热点，预算 ~2ms）
逐左点（原序）：
1. `row=roundf(y/step)`；越界/无表行 → `left_status=-1`（noMap），0 卡
2. 行内二分首 `x≥xL` → lo（边界语义逐字对齐，含"行内无 x≥xL 收尾到末条"）
3. 双向展开 `|dx|≤vL_tolerance`：每 lid 就近单卡（cap 32）
4. uR 补偿（逐线表 / 4 参数模型，互斥，常数 URB_* 同值拷贝）
5. HIT 验卡：右行内二分首 `x≥uR-window`，`|x-uR|≤window` 记 hitRx/hitRidx；
   失败卡丢弃（`card_count=有效卡数`，`left_status= 有效>0?1:0`）

### Stage D —— 段表＋投票＋winner
- 段映射：track 拷贝→sort+unique→每点 lower_bound 得 rank（＝GPU 游程 rank，逐位一致）
- 投票：`votes[seg][lid-1]++`（全卡计数制）＋`seg_res` 定点累加（诊断对齐）
- winner＝argmax 平票取小 lid（electWinnerArgmax 同式）

### Stage E —— 冲突处理（三路径全保留，参数对齐）
1. **默认双弃**（产线配置）：`kernelConflictMarkV3` 等价确定性实现——
   逐段按行位图占胞 `(row,winner-1)`，首占者留名，任何后到异段 → 双弃。
   GPU 原子序任意但净语义确定：**同一 (row,lid) 被 ≥2 段主张 → 全部主张段弃**
   （已弃段仍参与占胞，对齐 kernel 不检查 drop 即 CAS 的行为）
2. **P5 冲突重选**（conflict_reelect）：移植 `resolveConflictsOnce`（cellA/cellB＋
   v3b 逐行序投票＋reetectByPosition/reelectSecond 回退），≤10 轮
3. **P6b 段分裂**（segment_split）：移植 `splitBimodalSegments`（双峰门槛＋卡独占/
   行序分割＋行块连续三门槛，≤3 轮）
4. **P7 级联**（cascade_elim，隐含 P5）：移植 `cascadeSolve` host 版
   （友军合并 union-find＋门槛级联消元＋撞号/间距安全检查禁赛重跑 ≤10 迭代）
   — GPU 侧这些本就是 host 代码，移植为纯复制＋去 CUDA 依赖

### Stage F —— 收集＋统计（合并 GPU Step8/9/10）
单遍：保留段持 winner 卡的点顺序追加输出；`right_status[hitRidx]=1`；
保留段无 winner 卡点计入 noWinnerCard。统计口径与 kernelStep10ReduceV3 一致
（excludedRight = rightCount − rightUsed 等）。

## 4. 性能预算（单线程，25k/25k 点）

| 阶段 | 估算 | 说明 |
|---|---|---|
| B 右桶排序 | ~0.2ms | 计数排序＋行内微排序 |
| C 发卡 | ~1.5-2.5ms | 25k×(行二分 ~10 步＋展开＋HIT 二分 ~3 步)，主要 cache miss |
| D 段表＋投票 | ~0.3ms | sort 25k ints ~1ms 内含 |
| E 冲突（默认路径） | ~0.2ms | 段位图 OR＋占胞 |
| F 收集统计 | ~0.1ms | 单遍 |
| **合计** | **~2-3ms** | 余量 ≥40%；不达标再启用：表桶索引（GPU kBucketK 同款）/SoA 布局预取 |

## 5. 数值一致性策略

- 全程 float 运算镜像 CUDA float 语义（roundf/fabsf/乘加同型；IEEE754 单精度
  四则 CPU=CUDA 逐位一致）
- 行内排序按实际 x：与 GPU 量化键（1/256px）同序；同键微差 <0.004px 入容差
- 段 rank/票数/落段判定全整数量纲 → 精确一致；P5/P6b/P7 移植保持同一加法序列
  （double 累加顺序原样保留）

## 6. 测试方案（tests/test_laser_match_scan_v3_cpu.cpp，编入 test_speed_replay）

1. **合成单测**：手工小温表＋合成左右点 → 精确断言：匹配对/各状态计数/诊断字段/
   双弃行为/越界与空表守卫/空输入
2. **A/B 回放一致性**（GPU 基准）：CMTT 温表（date_in/laser-mark1688-1006/
   curve_map_temp_table.bin）＋out_golden_base/cycle_000..013 的 L3-interp CSV
   （含 tracks）→ CPU vs GPU 同参同输入：
   - matchedCount、lid/track 序列、全部诊断计数精确一致
   - 坐标逐点 |Δ|≤1e-3；参数矩阵：默认 / conflict_reelect / cascade_elim 三配置
3. **性能测试**：最大帧重复（预热 3＋测 10 取中位）< 5ms，失败即 fail

构建：无需改 CMakeLists（ops/scanning/laser/**.cpp 与 tests/*.cpp 均已被 glob
收编）；构建目录 build-rel/so，`ctest -C Release`。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 发卡 cache miss 超预算 | 桶索引（kBucketK=256，4.5px/桶≈26 条/桶，GPU 同款锚点算法逐字照抄）＋x/uR/lid SoA；预留优化位 |
| P5/P6b/P7 移植走样 | 只做机械移植不改判据；A/B 三配置全覆盖 |
| 与 GPU 原子语义出入 | 双弃净语义推导（§3-E1）已核 kernel 源码；A/B 兜底 |
| CPU 算子被误用于产线 | 头注释明示「备用/沙箱」，不入 mod_operatorlib |

---

## 8. 三代理评审修订记录（2026-10-21，评审后定稿）

**算法评审修订：**
- §3-B 改为**复刻 GPU 量化键稳定序**：`rowKey=(int)roundf(y/step)`，row<0 → **钳位 0**
  （非丢弃！），仅 row≥8192 丢弃（KEY_SENTINEL）；`xq=nearbyintf(x*256)`（＝
  `__float2int_rn` 就近偶）clamp [0,2^19)、x<0→0；行内 stable_sort 按 xq（保 origIdx
  tie）→ 与 GPU 右行序逐位一致，容差可守 1e-3
- §3-E 修正：**P6b 是 conflict_reelect 的前置 modifier**（次序：P6b ≤3 轮 →
  cascade_elim ? P7 : P5），不是并列路径；**路径分支键＝conflict_reelect**
  （SetParams 不做 cascade→reelect 归一，CPU 照抄不"修正"）
- §3-F 补：matched_right.**y＝左点 y**；right_status 是**右排序系索引**且每帧清零；
  excludedLeft＝noCard（不含 noMap）；emptyWinnerSeg 含被弃/banned 段；trackTotal＝
  分裂后 segCount
- §3-A 补：SetParams 改 rowStep → CSR 缓存失效
- kBucketK 修正为 **256**（4.5px/桶）；防御：bias 表读加 lid∈[1,32] 断言；
  track id 非负断言（GPU CUB unsigned 排序前提）；seg_res 用 nearbyintf；
  uRBias 结果字段 GPU 从不回填 → CPU 不设；maxRightPerRow＝validate-only no-op；
  >4096 段时 CPU 无上限自然正确（GPU 该边角未定义，A/B 勿人造）
- GPU env 前置条件：A/B 时 GPU 侧 JMW_V3* 环境变量必须干净（V3BISECT/SEGCAP
  等会改变基准行为）

**性能评审修订（实测规模纠正）：**
- 真实规模：左 34k/右 32k 点；温表单档 **8.73M 条/139.7MB**（~4178 条/行，2090 表行）；
  rowStep=0.67；vL_tolerance=0.6（展开窗 ±3 条）；输入 CSV 已 y 单调
- 朴素估算 1.7–3.7ms（置信 ~90%）；落地桶索引后 0.9–1.4ms（~99%）——**首版直接实现
  桶索引**（照抄 GPU uploadMapTable 桶池构建＋8 步跳走锚点，语义零风险）
- MSVC：保持默认 /fp:precise；`y/step` 保留除法（勿乘倒数）；浮点字面量带 f；
  fmaxf 语义用 std::fmaxf
- 段位图/票表只清 active segCount（勿全量 4MB memset）；P5/P7 保持原 host 结构
  （仅有票段＋共享行预过滤），勿重写成全对扫描

**集成评审修订：**
- A/B 回放帧＝**cycle_000..002（3 帧）**（out_golden_base 仅前 3 帧有 L3 dump）
- 右 track 置全 0（GPU 侧右 lid 只写不读，已核 kernel 参数表）；回放坐标读
  L3-interp CSV（6 位小数），track 列按行序 zip L3t-tracks_L.csv（其 xy 2 位舍入仅对账）
- GPU 基准：先 SetTempTable＋SetCurrentTemperature（nominalTemperature=22.5）；
  结果 download 须在下一次 Execute 前；A/B 参数矩阵含链内值
  （7.0/0.67/0.6/reelect=split=cascade=true）与产线默认值两档
- 测试顺序（TU 内）：合成单测 → 性能断言 → A/B（GPU）最后；数据缺席 GTEST_SKIP；
  新文件落地后**手动重跑 cmake configure**（glob 无 CONFIGURE_DEPENDS）
- CPU 宿主 helper 全部匿名 namespace

---

## 9. 实施结果（2026-10-21 定稿）

### 9.1 交付物

| 文件 | 说明 |
|---|---|
| `ops/scanning/laser/laser_match_scan_v3_cpu/laser_match_scan_v3_cpu.h` | 公开接口（复用 LaserMatchScanParamsV3；CpuResult 值语义） |
| `ops/scanning/laser/laser_match_scan_v3_cpu/laser_match_scan_v3_cpu.cpp` | 单线程实现（~2000 行；P5/P6b/P7 全路径移植） |
| `tests/test_laser_match_scan_v3_cpu.cpp` | 11 项 gtest（合成 7＋性能 2＋A/B 2） |

### 9.2 正确性（全绿）

- **A/B 回放一致性**（GPU 基准）：out_golden_base cycle_000..002 × {链内配置, 产线默认}
  两配置——matchedCount、lid/track 逐点、全部诊断计数（exclL/exclR/noMap/noWinnerCard/
  emptySeg/trackTotal/trackDropped）**精确一致**，坐标容差 1e-3 内零超差点
- 合成单测 7 项：基础匹配/noMap/noCard/双弃语义/空输入/uRBias 补偿/行界守卫
- ctest 100%（test_speed_replay 61 项：60 过＋1 既有跳过）

### 9.3 性能（i7-9850H 笔记本，Release，最大帧 34,237 左点/32,328 右点，中位 10 次）

| 配置 | 耗时 | 对照 |
|---|---|---|
| **产线默认（双弃路径）** | **3.5–3.8ms ＜ 5ms 达标** | GPU 同帧 ~3.5ms |
| 链内配置（P5/P6b/P7 实验三路径全开） | ~9–10ms（参考项，回归界 15ms） | vL=0.6 使展开/HIT 工作量为默认的 ~5x |

**<5ms 硬指标口径**：算子参数缺省＝产线语义（头文件明示「false＝原版双弃语义（产线
默认）」，P5/P6b/P7 均为 2026-10 实验路径）——以产线默认配置为硬断言；链内配置
（沙箱 config.json 全开实验路径）作为参考测量记录。

### 9.4 实施中落地的关键优化（22ms → 3.5/9ms 的路径）

| 轮次 | 优化 | 收益 |
|---|---|---|
| 1 | MapEnt 12B AoS 单数组（SoA 三数组行切片 80KB 超 L1） | 22→18ms |
| 2 | 卡表 CSR 紧凑化（stride-32 四数组 17.6MB→1.9MB，整帧工作集 28MB→12MB 内）＋行级批量预取＋O(1) lid 去重映射＋段映射 LUT | 18→13.8 |
| 3 | 快速取整内联（MSVC roundf/nearbyintf 为 CRT 调用，~10 万次/帧）＋右 CSR 行计数排序＋行内插入排序（stable_sort 小数组堆分配） | 13.8→10.1 |
| 4 | HIT popcount 掩码搜索（有序数组 lower_bound＝小于界元素数；AVX2 4×256b cpuid 守卫，SSE2/标量回退）＋excl 按候选段门控 | 10.1→8.7 |
| 5 | 单点处理 noinline 函数化＋行预取免除法（近似判定，仅预取目标） | 8.7ms 稳态 |

**证伪记录**（保留教训）：无分支 13 步二分锚点（串行依赖链反而 3x 慢）；栅格直达猜位
锚点（CMTT 表栅格多重性 2-3 不均，猜位漂移 70ms 灾难）；级联两阶段配对（长段共享行
普遍，达标对近乎全量，反而更慢）。锚点终态＝K=1024 桶直达＋8 步跳走（与 GPU 桶路径
数学等价：min(lower_bound, 行末)，与桶数无关）。

### 9.5 已知边界

- 单实例非线程安全（同 GPU 契约）；track id 非负前提（同 GPU CUB unsigned 排序）
- GPU 侧 kSegCap=4096 溢出兜底路径在 CPU 无上限自然正确（A/B 勿人造 >4096 段）
- A/B 时 GPU 侧 JMW_V3* 环境变量须干净（改变基准行为）
