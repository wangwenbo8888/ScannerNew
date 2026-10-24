# 产线集成 speed_optimize（最终形态算子＋五链流水线）· 计划 v1.0

> 日期：2026-10-24 ｜ 状态：**待人工确认——确认前不移植任何代码**
> 来源：`modules/09_operatorlib/TEST/speed_optimize`（外部 Junction `D:\workfold0822\speed_optimize`）
> 目标：把回放沙盒中验证过的最终形态（16 算子链 + 五链并行流水线，399 帧 0 diffs、131→142-158 FPS）
> 集成进主工程，支撑产品四功能：**标定 / 标志点扫描 / 面片扫描（标志点＋激光线）/ 后处理**。

---

## 一、现状对照（主库 vs speed_optimize）

### 1.1 算子层（ops/ vs modules/OperatorLib）

| 域 | 主库现状 | ops 最终形态 | 集成动作 |
|---|---|---|---|
| core/laser | epipolar_interp、steger、undistort_cuda、laser_reconstruct | **steger_fast**（新提取核）、**flat_center**（FlatCenterK1 48 档网格）、**epipolar_interp_dual**（已扫双目）、undistort_cuda、laser_reconstruct | 新增 3 算子＋替换 1 |
| core/marker | 12 算子 | 同 12 ＋ **conic_reconstruct**；zernike_edge/ellipse_fit 有改动（P2 热点） | 新增 1＋改 2＋其余对齐 |
| core/vision | mask_extract、frame_filter、ccl | 同 3（回读路径优化） | 对齐改动 |
| scanning | laser_match_scan（v1）、fusion 四件 | **laser_match_scan_v3 三变体**（v3 GPU / v3_cpu / v3_split 流水线拆分＋deferred）、laser_cloud_fuse_cuda（deferred/RLE/wait 原子删大改） | 新增 3＋大改 1；v1 与 CPU/normal 融合件去留见决策 D3 |
| calibration（运行时查表） | camera 5 件、laser_calib 8 件（**plane_map 系**）、temp 3 件 | 仅扫描运行时所需：**curve_map / curve_map_temp_table**（v3 匹配配套）、laser_label、laser_match、projector_joint_calib、stereo_rectify_temp_table、temp 3 | **plane_map 系 → curve_map 系**（见决策 D2——影响标定产出端） |

### 1.2 流水线层（src/fc_executor vs 07 PipelineManager）

| 维度 | 主库 07 | speed_optimize 五链（v3.1 设计，实测 143-158 FPS） |
|---|---|---|
| 结构 | SchedulerRuntime 多 lane（GPU 链+P 链+E 汇合）＋FuseConsumer | **七执行单元**：CUDA链1a(掩膜 D2D)→CUDA链1b(steger/flat/undist/interp/v3前段)→GT 投递线程(RLE 通道)→CPU链1(标志点)→CPU链2→CPU链3(配准)→CUDA链2(fuse deferred) |
| 跨单元同步 | 队列＋GPU 槽 | 事件（evGate/evCnt/evRuns/evStegSlot[8]）＋ pacing 门 |
| 关键机制 | — | v3 前段**延迟执行**（JMW_V3DEFER）、fuse **ExecuteDeferred**、RLE 行数压缩 D2H、双流拆分 |
| 验证 | 真机联调中（lane 冻结问题待修） | **399 帧逐帧 0 diffs（golden 对拍）＋ 82 单测** |

### 1.3 沙盒已验证战果（不可丢）

- 性能：131→142-158 FPS（每帧省 0.4-1.0ms）；GPU 内核地耗 ~5.8ms/帧（nsys 实证）
- 正确性：优化前后**逐帧完全一致**（82 用例＋399 帧 0 diffs，ABBA 双实测定档）
- 部署三雷（文档《产线移植三雷-部署清单》）：①timeBeginPeriod(1)（否则 ~64 FPS 上限）②Wi-Fi 中断钉核（DPC 抢核0）③电源计划核闲置 100%
- 25 个文件改动均有记录文档背书（43 份归档，含文件行号＋回滚口）

### 1.4 缺口（speed_optimize 不覆盖、主库保留）

- **标定工作流**（01 CalibrationWorkflow / factory_calib 只读区）——沙盒只加载标定产物不产标定
- **后处理**（04 / 07-E 五阶段）——沙盒不跑
- **全局优化 GBA / 观测账本（FrameObsAccumulator）**——沙盒不跑；五链需为 D 线保留逐帧观测出口
- **设备层 08 / UI / 数据仓库 06**——主库现状即产品形态

---

## 二、集成原则

1. **golden 红线**：每一批合入后 399 帧回放对拍 0 diffs ＋ 主库 ctest 100/100（Debug+Release）——不达标即回滚该批
2. **沙盒为源**：算子以 ops/ 为准（主库同名算子的 16 处新改动逐个评估：等价吸收／被取代／需回移）
3. **开关过渡**：编译期开关（如 `JMW_FIVECHAIN`）保留双轨，验收后切默认、删旧轨
4. **不动只读区**：factory_calib 不碰；标定产出端变更属人工任务

---

## 三、决策点（需人工拍板后开工）

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| **D1** | 流水线结构 | A. 五链执行器整体进 07（fc_executor 移植为主执行模型，SchedulerRuntime 淡出）<br>B. 保留 07 lane 模型只换算子内核（损失流水线收益 ~30-50%） | **A**——五链已在真实数据验证；07 现结构还有 lane 冻结未解 |
| **D2** | 激光标定查表 | A. plane_map → curve_map 系整体切换（标定产出端需配套，涉 factory_calib 人工）<br>B. 双表并存过渡 | **A＋过渡期并存**（v3 匹配吃 curve） |
| **D3** | 旧算子去留 | v1 匹配／CPU 融合／normal 件：A. 立删<br>B. 保留一个版本周期 | **B**（编译开关控制，验收后删） |
| **D4** | 五链观测出口 | GBA 需逐帧 R/T＋markerObs＋激光帧缓存——五链须增观测汇聚点（沙盒没有） | 按主库 FrameObsAccumulator 契约在 CPU链3/fuse 尾挂账 |
| **D5** | 时序 | 何时切产品默认 | Phase 4 真机四功能回归全绿后 |

---

## 四、分阶段计划（每阶段有验收门槛，绿灯才进下一段）

### Phase 0 · 基建（估 1-2 天）
- 把 scan_replay 构建为**主工程测试 target**（modules/OperatorLib/test 场，Release-only，同沙盒口径）
- golden（150MB）＋ sparse_30 数据集就位；对拍脚本进 `ctest`（标记 LONG）
- 差异台账冻结：25 改文件 × 主库 16 新改动 → 逐文件处置表（吸收/取代/冲突回移）
- **验收**：主库上跑通 399 帧 0 diffs（旧算子基线）

### Phase 1 · 算子合入（估 3-5 天，5 批）
1. core/laser 三新件（steger_fast/flat_center/epipolar_interp_dual）
2. laser_match_scan_v3 三变体（含 split/deferred 基础设施）
3. mask_separation 8 核形态学＋laser_cloud_fuse_cuda（deferred/RLE）
4. marker 改动（zernike/ellipse P2 优化＋conic_reconstruct）
5. vision/common/scheduler 垫片
- 每批：82 单测＋399 帧 0 diffs＋主库 ctest 全绿＋逐批 commit（可回滚）
- **验收**：主库算子层与 ops 完全等价（回放性能对齐 142-158±10%）

### Phase 2 · 五链执行器进 07（估 3-5 天，最大风险段）
- fc_executor 移植为 07 新执行模型（`JMW_FIVECHAIN` 开关）：七单元＋事件同步＋pacing
- 02 ScanWorkflow 对接（帧源从回放换 08 实时流）；FuseConsumer 语义并入 CUDA链2/fuse deferred
- FrameObs 观测出口（D4）＋暂停/续采水位＋看门狗/1610 诊断链平移
- 部署三雷落地：main.cpp timeBeginPeriod＋安装期中断亲和/电源计划脚本
- **验收**：真机 60fps 吃满不丢帧、激光零降级、laser 段 <40ms；暂停/编辑/急停全链可用

### Phase 3 · 标定链对齐（估 2-4 天，含人工协作）
- curve_map/curve_map_temp_table 替换 plane_map 查表（过渡双轨）
- 01 标定工作流产出端适配（**factory_calib 产出 curve 表＝人工任务，出需求文档**）
- 标定→扫描 readyForScan 门禁字段对齐
- **验收**：真机完整标定→标志点扫描→面片扫描闭环

### Phase 4 · 四功能端到端验收（估 2-3 天）
- 标定／标志点扫描／面片扫描／后处理（04+07-E）真机全流程回归
- GBA（D 线）在新流水线观测账上复跑：RMSE 与旧版同量级、终版云 0 异常
- 性能/稳定性压测（长会话、启停切换、暂停编辑）
- **验收**：四功能清单逐项签字；切默认（D5）

**总计估：11-19 个工作日**（不含 factory_calib 人工协作）

---

## 五、风险清单

| 风险 | 缓解 |
|---|---|
| 主库 16 处新改动与 ops 25 处改动冲突 | Phase 0 逐文件处置表；冲突文件人工裁决 |
| 五链与 02/06/08 接口断裂（FrameResult 结构变化） | Phase 2 出适配层；开关双轨可回退 |
| WDDM 提交税/时钟门（沙盒 Windows 实测 ~2ms 门） | 部署三雷落地＋Linux 复测列后续项 |
| GBA 观测账本缺口（沙盒无此链） | D4 设计先行，Phase 2 实现并单测 |
| factory_calib 只读区依赖（curve 表产出） | 出需求文档交人工；过渡期 plane/curve 双轨 |
| 显存增长（fuse reserve 8M 档 ~320MB＋2^27 槽 ~5GB） | 真机 12GB 预算复核（Phase 2 验收项） |

---

## 六、确认清单（人工勾选后开工）

- [ ] D1 流水线结构＝方案 A（五链进 07）
- [ ] D2 标定查表＝curve_map 切换（含 factory_caliv 人工协作排期）
- [ ] D3 旧算子＝保留一个版本周期后删
- [ ] D4 GBA 观测出口按 FrameObs 契约挂账
- [ ] Phase 0 先行（回放验收基建）即刻启动
- [ ] golden/数据集迁入主工程测试场的存储位置批准（~200MB）
