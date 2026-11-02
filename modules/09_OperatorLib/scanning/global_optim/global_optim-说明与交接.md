> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# global_optim · 全局捆绑优化（GBA）

## 1. 目标

- 功能：全局捆绑优化（GBA）——全部位姿＋标记点一起批算压实漂移，收尾必跑
- 方法/参数：Kabsch 建边＋（可选）PGO 预优化＋Ceres 最多 4 阶段；maxSolveSecondsPerPhase 默认 60s 可限时
- 注意点：批处理模型（秒~分钟级）勿入实时链；软先验 σ 权重属精度红线

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 全部位姿＋标记点一起批算（GBA）：达成｜主类 `GlobalBundleAdjustmentCPU`（global_ba_cpu.h）＋`ba_residuals.h`/`pose_graph_residuals.h`：输入 GlobalBAInput（逐帧 R/t 初值＋markerObs＋高精度点软先验）→输出优化位姿 FramePose＋GlobalMarker
- Kabsch 建边：达成｜相邻帧相对位姿建边，退化守卫：点数 <3 或近共线（条件数 >1e3）跳过；回环边条件帧距 ≥loopFrameGap=30 且共视 ≥minCovisForLoopEdge=5（代码实测＝卡默认）
- （可选）PGO 预优化＋Ceres 最多 4 阶段：达成｜plain/预清洗/Tukey/chi2 四阶段；maxIterations=200、tolerance=1e-10、sigmaObserved=0.01mm、tukeyC=0.03（=3σ）、chiSquareGate=11.34（3-DoF 99%）
- maxSolveSecondsPerPhase 默认 60s 可限时：达成｜261004 加固：默认 60s（0=不限，离线 runner 可置 0），到时 NO_CONVERGENCE 保当前最优；cancelRequested→SOLVER_ABORT（不采半程解）、progressCb 跨 PGO＋4 阶段加权 f∈[0,1]（261017）
- 收尾必跑压实漂移：部分达成｜07 globaloptim/GlobalOptimObject.cpp:301 A/B 两模式均必跑；但真机 GBA 降级（迭代=0 初值兜底）根因未排查（P0 挂账），无回环时仅前向 BA 全局漂移不消
- 约束（批处理勿入实时链／软先验 σ 精度红线）：达成（已固化）｜秒~分钟级定位离线；useSoftPrior=true、defaultPriorSigma=0.001mm，逐点 σ 可覆写（ids 空＝行为同无先验）；so3 log 近似仅 |θ|≲0.3 rad 有效
- 离线工具与测试：达成｜gba_dataset_runner.cpp（离线 exe，CMake 单列不入库）；tests/test_global_ba_cpu.cpp＋test_global_ba_prior.cpp

## 3. 问题与方法

- 〔待定〕问题：真机 GBA 降级（迭代=0 初值兜底）根因未排查——观测帧/先验点条件不满足？｜方法：真机排查根因（P0 挂账）｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
