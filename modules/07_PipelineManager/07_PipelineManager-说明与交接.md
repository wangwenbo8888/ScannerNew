> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# 07_PipelineManager · 流水线管理

## 1. 目标

- 帧处理引擎：并行调度底座＋五条流水线（姿态/标定计算/扫描五链/全局优化/后处理）；PipelineManager.dll 契约热更（改实现不改公共头＝换 DLL 即生效）
- 量化锚｜五链对拍 399/399 全同；全量 ctest 110/110（261026）；拆分匹配 80-85 优于整版 67-73
- 注意点｜调度底座零算子知识（GPU 链经 LaneHooks.gpuChain 注入，frontReady()＝ccl 就绪点，勿破坏）；07_session 已退役勿复活

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 并行调度底座：达成｜`sched/` 在库——SchedulerRuntime（抓帧→占 GPU 槽→pChain 汇合＋看门狗 250ms）/CpuTopology/PCoreBroker/GpuSlotService/FrameResultQueue＋IFrameSource/SchedConfig
- 五条流水线（姿态/标定计算/扫描五链/全局优化/后处理）：达成｜`posture/` PosturePipeline＋`calibcompute/` CalibComputePipeline＋`pipelines/scan/` ScanPipeline＋`fivechain/` 五链执行器＋`globaloptim/` GlobalOptimObject＋`postprocess/` PostProcessPipeline，全编入 PipelineManager.dll
- 契约热更：达成｜`api/jmw_pm_contract.h` 契约指纹机制（与 09 互锁）——改实现不改公共头＝换 DLL 即生效
- 调度底座零算子知识：达成｜GPU 链经 LaneHooks.gpuChain 注入、第 4 参 frontReady()＝ccl 就绪点，机制层不含算子知识
- 07_session 已退役勿复活：达成｜现库会话体＝ScanPipeline 六态状态机；旧对象「以现库为准」不在库〔待核：现状段未点名 07_session〕
- 五链对拍 399/399 全同：达成｜五链与沙盒实证等价（离线对拍 399/399；test_fc_replay＝对拍件）
- CCL 门限接线：达成｜FiveChainAdapter.cpp 读 operators.ccl 配置注入 cclMinArea_（原硬编码 800 不读配置；261102 在线调参修复，标志点链全通 det=19 match=21）
- 全量 ctest 110/110（261026）：达成｜tests/ 21 件测试源（sched 7＋posture 3＋calib 3＋scan 3＋postprocess 2＋globaloptim/eventsink/fc_replay 各 1）双绿
- 拆分匹配 80-85 优于整版 67-73：部分达成｜五链拆分版在库且 399/399 对拍全同；该组匹配率数值现状段无实测记录〔待核〕

## 3. 问题与方法

- 〔待定〕问题：真机五链会话验证未做｜方法：JMW_FIVECHAIN=1 跑面片扫描，对比旧 lane 点云/精度/吞吐｜提出 261102｜解决 —
- 〔待定〕问题：算子/流水线与工作流未解耦（D1-D3）｜方法：api 契约层＋替换工作流（docs/plans/2026-10-25-算子流水线独立替换-解耦目标与计划.md）｜提出 261102｜解决 —
- 〔待定〕问题：激光匹配现为固定视差 30px 近似｜方法：激光查表匹配接入｜提出 261102｜解决 —
- 〔待定〕问题：lane 冻结 dump 待分析｜方法：栈帧分析定位卡死根因｜提出 261102｜解决 —
- 〔待定〕问题：五链 v1 限制未补齐｜方法：A 模式/暂停续扫/检查点/marker 编辑账本逐项补齐｜提出 261102｜解决 —
- 〔待定〕问题：resume 持久水位决策与 v1 现码不符（决策称 resume 不得从 0 重扫；现码 pause＝停止、checkpoint 不在库）｜方法：待人工裁决去留｜提出 261102｜解决 —
- 〔待定〕问题：激光点飞离标志点平面（离线回放 399 帧正确、在线实拍飞出；已排除新旧标定差异/对比度增强/坐标系偏移；嫌疑 CMTT 表(10-13)与 camera_calib.json(10-31) 微差被查表匹配放大，或在线图像特性[增益噪声]对 steger→undistort→interp→match 链的累积影响）｜方法：用在线 dump 帧过离线回放路径跑，定位「图像内容」还是「在线处理路径」问题｜提出 261102｜解决 —

## 4. 日志

- 261102｜在线调参：CCL/椭圆/增益修复＋标志点链打通；激光飞点未解（详见会话总结）｜—｜—
- 261102｜现状改目标达成对照制（对象级细节精简，证据留真名）｜规范 §2.3
- 261102｜转四段新制：目标待人定稿（含原全局决策注意点）；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量文档颗粒度扩充（源：docs/存量档案/模块功能/07＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
