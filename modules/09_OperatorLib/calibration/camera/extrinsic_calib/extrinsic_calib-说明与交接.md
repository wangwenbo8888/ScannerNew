> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# extrinsic_calib · 双目外参标定

## 1. 目标

- 多视角角点经 OpenCV stereoCalibrate（锁内参）解双目外参 R/T/E/F，附对极/重投影误差与质量标记（门禁 RMS＞1.0px/极线＞0.05px 置 Degraded）
- 两条流程：Flow1 已知内参直标；Flow2 calibrateMono=true 先单目补内参再标；供 07 CameraChain 与 01 标定工作流消费
- 注意：Result 为 move-only（勿按值拷贝）；perViewErrors 恒空、flags 被硬编码锁内参为已知缺陷

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 解双目外参 R/T/E/F＋对极/重投影误差与质量标记：达成｜主类 `ExtrinsicCalibCpu` 调 OpenCV stereoCalibrate 锁内参只解 R/T/E/F，门禁 RMS＞1.0px/极线＞0.05px 置 Degraded、内参加检（fx∈[100,10000] 等）异常置 Warning；在用 07 CameraChain.cpp、01 calib_workflow.cpp（已知缺陷：perViewErrors 恒空、flags 被硬编码、Degraded 会盖 Warning）
- 两条流程 Flow1/Flow2：达成｜Flow1 已知内参直接标；Flow2 calibrateMono=true 先单目补内参再标（单目失败兜底 fx=图宽）；下游 R/T 必喂 stereo_rectify、全量版另带 monoLeft/monoRight

## 3. 问题与方法

- 〔待定〕问题：无挂账（如有人工裁定项随时补）｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
