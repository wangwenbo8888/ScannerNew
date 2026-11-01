# laser_match_scan_v3 流水线拆分（GPU 前段算子 + CPU 后段算子）设计方案

> 2026-10-21 ｜ 沙箱：modules/09_operatorlib/TEST/speed_optimize/（只改本目录＋原算子目录增量）
> 基准：`ops/scanning/laser/laser_match_scan_v3/`（GPU 算子，功能以此为准，**不删除不改动既有行为**）
> 硬指标：**CPU 后段算子计算 < 2ms**（典型最大帧 ~34k 左点 / ~32k 右点）
> 需求：原算子保留为备用；全程自动化实施至测试通过。

---

## 1. 目标与非目标

**目标**
1. 把 `LaserMatchScanCudaV3::Execute` 的 10 步流水线拆成两个可独立调用的算子：
   - **GPU 前段算子**：Step 0～6（右排序/CSR、发卡、GPU 段表）＋ 中间产物下载与同步（Step 6.5「取段数」含在前段收尾）
   - **CPU 后段算子**：Step 7～10（段内投票、winner 选举、冲突处理 P5/P6b/P7 全路径、收集、统计），**纯 CPU、无任何 CUDA 调用**
2. 拆分后 [前段+后段] 与原 GPU 算子同输入同参数输出**逐点一致**（坐标容差 1e-3px，lid/track/全部计数精确一致）
3. CPU 后段 < 2ms（产线默认配置硬断言；链内实验配置实测记录）
4. 原 GPU 算子文件**零行为改动**：`process()` 与全部内核不动，既有 61 项 gtest 全绿

**非目标**
- 不替换产线 GPU 算子；不改 src/ 链与 config
- 不做双缓冲流水线重叠调度（拆分本身即交付物，重叠属后续链层工作）
- 不做多线程/SIMD 化后段（先证单线程达标）

**动机**：GPU 前段与 CPU 后段可并行错帧执行（GPU 算 N+1 帧前段时 CPU 收 N 帧后段），
消除原版 Step 7-10 的 GPU 尾部串行＋两次同步点中的第二次。

## 2. 拆分边界与数据流

```
原 Execute 流水线                    拆分后
─────────────────                ─────────────────────────────────
Step 0-4 右排序/CSR   ┐
Step 5    发卡验卡     ├ GPU 前段算子 ExecuteFront() → FrontOutput（host 值语义）
Step 6    GPU 段表    ┘   （含卡表紧缩＋批量 D2H＋单次同步＝Step 6.5）
─────────────────                ─────────────────────────────────
Step 6.5  取段数      ┐
Step 7    投票/选举    ├ CPU 后段算子 Execute(front) → Result（纯 CPU）
Step 7'   冲突处理     │
Step 8/9  收集/压缩    │
Step 10   统计回传    ┘
```

FrontOutput（host，值语义，可跨帧持有）：
| 字段 | 类型 | 说明 |
|---|---|---|
| leftCount/rightCount/segCount/totalCards | int | 规模（counts[2]＝segCount，Step 6.5 产物） |
| leftPts | vector\<Point2f\> N | 左点原序（后段消费序） |
| leftTracks | vector\<int\> N | 左点 track id（原序） |
| segmap | vector\<int\> N | orig→segIdx（GPU 游程 rank） |
| leftStatus | vector\<int\> N | -1 noMap / 0 noCard / 1 有卡（GPU Step5 产物） |
| cardPtr | vector\<int\> N+1 | 每点卡起始偏移（CUB ExclusiveSum(card_count)） |
| cardLid/cardRx/cardRidx | vector N×T | 紧缩卡表 SoA（新内核 kernelTightenCardsSplitV3） |
| success/message/timing | — | 状态与阶段耗时串 |

后段不需要：温表（查表只发生在发卡）、右点坐标（matched_right=(hitRx, 左y)）、右 track
（GPU 侧右 lid 只写不读——已核 kernel 参数表）。

## 3. 交付物与文件布局

```
ops/scanning/laser/laser_match_scan_v3/          （原算子目录——增量）
├── laser_match_scan_v3_cuda.h                   +4 行：前置声明＋公有 ExecuteFront 声明
├── laser_match_scan_v3_cuda_pimpl.h             +~20 行：executeFront 声明＋新缓冲
│                                                   （d_card_off_/d_tight_lid/rx/ridx、
│                                                    PinnedBuf hpin_cardoff_/cardrx_ 等）
└── laser_match_scan_v3_split_front.cu           新增：kernelTightenCardsSplitV3 ＋
                                                    Impl::executeFront（复用既有内核，
                                                    extern 原型跨 TU 发射；process() 零改动）

ops/scanning/laser/laser_match_scan_v3_split/    （新目录）
├── laser_match_scan_v3_split.h                  LaserMatchSplitFrontV3 ＋
│                                                LaserMatchScanSplitGpuV3（前段，组合持有
│                                                LaserMatchScanCudaV3，转发温表/参数/
│                                                Warmup/Destroy/SetuRBias*）＋
│                                                LaserMatchScanSplitCpuV3（后段）
├── laser_match_scan_v3_split_gpu.cpp            前段门面（调原算子 ExecuteFront）
└── laser_match_scan_v3_split_cpu.cpp            后段实现（Stage D/E/F 机械移植自
                                                 laser_match_scan_v3_cpu.cpp 行 259-988＋
                                                 1621-1868：helpers＋投票→冲突→收集→统计）

tests/test_laser_match_v3_split.cpp              新增 gtest（glob 自动收编，需重跑 configure）
```

原算子唯一 API 增量（`ExecuteFront` 公有方法，返回 void＋出参避免不完备类型）：
```cpp
class LaserMatchScanCudaV3 {
    ...
public:
    // 流水线拆分（2026-10-21）：只跑 Step0-6＋产物下载；out 见 laser_match_scan_v3_split.h
    void ExecuteFront(const cv::cuda::GpuMat& d_left_points,
                      const cv::cuda::GpuMat& d_left_line_ids,
                      const cv::cuda::GpuMat& d_right_points,
                      const cv::cuda::GpuMat& d_right_line_ids,
                      cv::cuda::Stream& stream,
                      class LaserMatchSplitFrontV3& out);
};
```

## 4. 前段算子实现要点（Impl::executeFront，新 .cu）

1. 复用 `allocateBuffers`（Impl 成员，外部可链接）＋原内核（全局命名空间、外部链接，
   跨 TU 声明原型后直接发射）：kernelGenKeysIdxV3 / kernelScatterSortedV3 /
   kernelBuildCSRV3 / kernelIssueCardsV3 / kernelCopySeq / kernelRunHeadsV3 /
   kernelSegMapBackV3。发射参数从原 process() 对应块**逐字拷贝**（普通发射，不走
   CUDA Graph；JMW_V3GRAPH 只影响原 Execute，不影响本方法）
2. 发卡参数与原版一致：桶索引路径（bucketK_>0 且未设 JMW_V3BISECT 时启用）、
   uRBias/uRBiasModel 状态、status ping-pong 代际
3. 新内核 kernelTightenCardsSplitV3：输入 d_card_lid_/d_card_hitRx_/d_card_hitRidx_
   （N×32 stride 布局）＋d_card_count_＋d_card_off_（CUB ExclusiveSum 前置），输出
   三路紧缩数组；每点卡内顺序保持 stride 表顺序（对下游判定无影响：每 lid 至多一卡，
   票计数与 winner 卡查找均序无关）
4. 批量异步 D2H（全部走既有/新增 pinned 缓冲）→ **单次 cudaStreamSynchronize** →
   拷入 FrontOutput（值语义脱离生命周期契约）
5. 空输入/未设温表：与原版同语义（空输入 success="No points to match"）
6. 计时串（front 各阶段）供性能观测

## 5. 后段算子实现要点（机械移植，不改判据）

来源：`laser_match_scan_v3_cpu.cpp`（该实现已经 GPU A/B 精确一致验证）。
1. 入口组装：SoA 卡→LmCard AoS（单遍，~T×12B）＋ptRow[i]=froundAwayI(y/rowStep)
   （无 clamp，同 GPU 口径）
2. 段内票直方图＋行位图＋段点集（单遍）→ winner/second argmax（平票取小 lid）
3. 分支键 conflict_reelect：P6b（≤3 轮）→ cascade_elim ? P7 : P5（≤10 轮）——与
   CPU 版逐字一致（helpers 原样拷贝：electWinnerArgmax/reelectSecond/
   reelectByPosition/resolveConflictsOnce/splitBimodalSegments/cascadeSolve）
4. 默认路径双弃：cellOwner 占胞净语义（同一 (row,lid) ≥2 段主张→全弃，已弃段仍占胞）
5. 收集＋统计＋输出 Mats/status——口径同 GPU kernelStep10ReduceV3（exclL=noCard、
   exclR=rightCount−rightUsed、emptySeg 含被弃段、trackTotal=分裂后 segCount）
6. 结果复用 `LaserMatchScanCpuResultV3`（值语义）；LastTimingBreakdown() 暴露阶段耗时

## 6. 性能预算（后段，34k 点，i7-9850H 参照）

| 阶段 | 默认配置估算 | 链内配置估算 |
|---|---|---|
| SoA→AoS＋ptRow | ~0.15ms | ~0.15ms |
| 投票＋位图＋点集 | ~0.25ms | ~0.4ms（vL=0.6 卡数 ~2-3x） |
| winner/second | ~0.05ms | ~0.05ms |
| 冲突（默认双弃） | ~0.15ms | P6b+P7 ~1-2ms（待实测） |
| 收集＋统计＋输出 | ~0.2ms | ~0.3ms |
| **合计** | **~0.8ms ＜ 2ms** | **~2-3ms（实测后定界）** |

硬断言口径：产线默认配置（参数缺省＝双弃语义）< 2ms 必须达标；链内实验配置实测记录
（若 <2ms 则一并硬断言，否则设热安全回归界并注明）。前段耗时单独记录（无硬指标）。

## 7. 测试方案（tests/test_laser_match_v3_split.cpp，编入 test_speed_replay）

1. **合成单测**（GPU 前段＋CPU 后段，手工小温表）：基本匹配／noMap 与 noCard／
   同 lid 同行双弃／空输入／uRBias 补偿——断言口径复用
   test_laser_match_scan_v3_cpu.cpp 的既有用例（GPU 路径）
2. **A/B 一致性**（GPU 基准＝原算子 Execute）：CMTT 温表＋out_golden_base
   cycle_000..002 × {链内配置, 产线默认}——matchedCount／lid/track 序列／全部诊断
   计数精确一致；坐标逐点 |Δ|≤1e-3；GPU 侧 JMW_V3* 环境变量必须干净
3. **性能**：后段中位 10 次（预热 3）——默认配置 <2ms 硬断言；链内配置实测输出；
   前段单独计时输出
4. **全量回归**：ctest test_speed_replay 全绿（证明原算子零扰动）

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 跨 TU 内核原型签名失配 | 原型从原 .cu 定义逐字拷贝；链接期即暴露 |
| GPU 段表（CUB rank）与后段消费语义出入 | 全 CPU 版已证 GPU/CPU 段表一致；本方案直接消费 GPU 段表，A/B 三帧双配置再验 |
| 卡内顺序 GPU/CPU 不同影响判定 | 每lid至多一卡＋票计数序无关——判定不变（评审确认） |
| segCount>4096（kSegCap 溢出） | 前段沿用 d_counts_[3] 溢出标志；后段无段上限（同 CPU 版）；A/B 不人造该边角 |
| 下载量超预算拖慢前段 | 紧缩卡表 ~1MB＋其余 ~1MB ≈ 2MB pinned D2H ≈ 0.3-0.5ms（在前段，无硬指标） |
| glob 不收新文件 | 落地后手动重跑 cmake configure（既有惯例） |
| 后段链内配置超 2ms | 默认配置为硬指标（产线语义）；链内实测后定硬/软界，必要时仅优化分配（段表扁平池化） |

## 9. 三代理评审修订记录（2026-10-21，评审后定稿）

**算法评审修订：**
- 后段算子必须持有与 GPU 前段同一份 `LaserMatchScanParamsV3`（rowStep＋conflict 三开关＋
  cascade 七参），入口检查 `front.success` 短路
- `ExecuteFront` 成功末尾同样 `status_gen_++`（避免与 Execute 交替时提前覆写下游持有的
  status 视图）；头注释契约改为「有效期至下一次 Execute/ExecuteFront/Destroy/setParams」；
  Debug 下加同款 inProcess_ 重入守卫
- segCount>4096：前段**透传溢出标志**（FrontOutput 加 segOverflow 字段）＋照常下载
  segmap/segCount（rank 值仍正确）；后段无段上限自然正确；JMW_V3_SEGCAP 测试通道不带入前段
- 「逐字拷贝」清单补：Step0 counts/diag memset、ExclusiveSum 临时缓冲按需分配块
  （d_scan_temp_ 同款，卡偏移复用同一临时缓冲——同流顺序复用安全）
- ptRow 复算存**原始值**（负/≥8192 不提前 clamp），消费侧统一守卫（P5/P6b/P7 语义不变）

**性能评审修订：**
- 后段三处分配改扁平池化（布局优化，不触碰判据/加法序列）：①segVotes/segRows 扁平池
  只清 active 段；②segPts 两遍计数 CSR（计数→前缀→散射，升序 i 保序）；③P5 cellA/cellB
  与 P6b 缓冲成员化复用（assign 不 realloc）
- 计时三段分报：`front_ms`（内核＋D2H＋同步＋FrontOutput 组装）、`back_ms`（**<2ms 硬断言
  唯一对象**：SoA→AoS＋计算＋输出）、串行总和（对照原版 ~3.5ms）；硬断言绑定 i7-9850H
  参照机＋Release 构建
- 链内配置软回归界 4ms（实测记录）；段池必须 grow-only＋只清 active 段（防 >4096 段帧
  全量清理税）
- 前段提交税（普通发射 ~10-13 次 ≈ 0.1-0.65ms）接受；图化列后续项

**集成评审修订：**
- 同步结构改为**两段式**（与原版同构）：sync#1 下载 counts(4)/diag(4)/cardOff(N)（~140KB）
  → 得 segCount/totalCards → 发主批次 D2H → sync#2
- 新 .cu 顶部集中「逐字拷贝自 impl.cu，勿改签名（含 __restrict__）」原型块；file-static
  常量（BLOCK_SIZE/V3_MAX_CARDS/V3_MAX_LIDS/XQ_*/KEY_SENTINEL）同值副本＋注释；原型勿加 static
- `LaserMatchScanCudaV3::ExecuteFront` 公有方法定义落位 split_front.cu（成员函数外部 TU
  定义）；不改 laser_match_scan_v3_cuda.cpp
- **split 公开头卫生**：不得 include pimpl.h 或 cuda_runtime.h（tests/*.cpp 纯 CXX 编入）；
  GpuMat/Stream 用前置声明
- 测试 TU 顶部加**静态环境守卫**（先于任何算子构造/static env 固化）：对 JMW_V3GRAPH/
  JMW_V3BISECT/JMW_V3_SEGCAP/JMW_V3FULLGPU/JMW_V3GPU/JMW_V3PAIR_AB/JMW_NOTBD/JMW_V3TIME
  逐个 `_putenv("XXX=")` 清空
- split 目录不建 tests/ 子目录（glob 静默排除陷阱）；新文件 UTF-8 无 BOM；落地后重跑
  cmake configure；全绿口径按新增用例数更新（现 61）

## 10. 实施顺序

1. 原算子增量（cuda.h＋pimpl.h＋split_front.cu）→ 编译过
2. split 目录（.h／gpu 门面／cpu 后段）→ 编译过
3. 合成单测 → 绿
4. A/B 一致性 → 绿
5. 性能断言 → <2ms 达标（不达标则按 §6 预算位优化）
6. 全量 ctest → 全绿；本文件追加实施结果

---

## 11. 实施结果（2026-10-21 定稿）

### 11.1 交付物

| 文件 | 说明 |
|---|---|
| `ops/scanning/laser/laser_match_scan_v3/laser_match_scan_v3_cuda.h` | ＋增量：ExecuteFront 公有声明（process/内核零改动） |
| `ops/scanning/laser/laser_match_scan_v3/laser_match_scan_v3_cuda_pimpl.h` | ＋增量：executeFront 声明＋紧缩/下载缓冲 |
| `ops/scanning/laser/laser_match_scan_v3/laser_match_scan_v3_split_front.cu` | 新增：kernelTightenCardsSplitV3＋Impl::executeFront＋公有桥接（extern 原型跨 TU 发射原内核） |
| `ops/scanning/laser/laser_match_scan_v3_split/laser_match_scan_v3_split.h` | FrontV3 产物结构＋SplitGpuV3（前段门面）＋SplitCpuV3（后段） |
| `ops/scanning/laser/laser_match_scan_v3_split/laser_match_scan_v3_split_gpu.cpp` | 前段门面（组合持有原算子，纯转发） |
| `ops/scanning/laser/laser_match_scan_v3_split/laser_match_scan_v3_split_cpu.cpp` | 后段实现（D/E/F 段自全 CPU 版机械移植；P5 池化） |
| `tests/test_laser_match_v3_split.cpp` | 9 项 gtest（合成 5＋性能 2＋A/B 2；含静态环境守卫） |

原算子 `process()` 与全部既有内核**零字符改动**；主树 `modules/09_operatorlib/scanning/` 未触碰。

### 11.2 正确性（全绿）

- **A/B 回放一致性**（基准＝原 GPU 算子 Execute）：out_golden_base cycle_000..002 ×
  {链内配置, 产线默认}——matchedCount、lid/track 逐点、全部诊断计数（exclL/exclR/
  noMap/noWinnerCard/emptySeg/trackTotal/trackDropped）**精确一致**，坐标容差 1e-3 零超差
- 合成单测 5 项：基本匹配／noMap+noCard／同 lid 同行双弃／空输入／uRBias 补偿
- **全量回归**：test_speed_replay 70 项（69 过＋1 既有跳过）——原算子零扰动

### 11.3 性能（i7-9850H＋RTX 5000，Release，最大帧 34,237 左点/32,328 右点/51 段）

| 配置 | back_ms（硬指标对象） | front_ms | 串行总和 |
|---|---|---|---|
| **产线默认（双弃）** | **0.547–0.593ms ＜ 2ms 达标（3.4x 余量）** | ~0.9–1.0ms | ~1.5ms |
| 链内（P5/P6b/P7 全开，67k 卡） | 2.8–3.0ms（软回归界 4ms 内） | ~1.6–2.5ms | ~4.4–5.5ms |

后段分相（默认）：组装 0.14＋投票 0.21＋选举 0.00＋双弃 0.13＋收集 0.11＋输出 0.01。
链内超 2ms 主因＝P7 级联 1.3ms＋P6b 0.4ms（实验路径，非产线语义；与全 CPU 版口径一致）。

### 11.4 评审与终审修订落地

- 三代理设计评审（算法/性能/集成）修订全部落地（§9）
- 终审代理复核：**无阻断项**；3 警告已修——①totalCards==0 跳过紧缩内核与下载
  （消 Debug 断言分叉）②空输入检查顺序对齐 process()（先 tempSet_ 后空查，桥接不再
  提前短路）③segOverflow 后段消费（WARN 日志）＋pinned ensure 逐点判空优雅失败
- 未采纳（记录理由）：段结构扁平池化（默认实测 0.55ms 余量充足，链内属实验路径）；
  Execute/ExecuteFront 交替测试（代码审读已确认代际契约成立）

### 11.5 已知边界

- 前后段参数一致性由调用方保证（FrontV3 无参数指纹；契约见 split.h 头注释）
- segCount>4096 溢出路径等价性未测（与既有「A/B 勿人造」口径一致；后段对
  segOverflow!=0 打 WARN）
- 前段普通发射含 WDDM 提交税（~10 次发射）；图化（复用 fg_/fgD_ 机制）列后续项
