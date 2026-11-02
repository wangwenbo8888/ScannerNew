> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# stereo_rectify · 立体矫正参数

## 1. 目标

- 左右内外参经 OpenCV stereoRectify 产立体矫正参数 R1/R2/P1/P2/Q＋有效 ROI（行对准源头），参数全经 SetParams 注入、Execute() 无参
- 关键参数 alpha=0.0（全裁无效区）、flags 须显式给 0 或 CALIB_ZERO_DISPARITY；供 07 CameraChain 与 01 消费
- 注意：flags 默认值 1 不合法，默认构造过不了 validate（抛 invalid_argument）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 经 OpenCV stereoRectify 产 R1/R2/P1/P2/Q＋有效 ROI：达成｜主类 `StereoRectifyCpu` 包一层 stereoRectify 出 R1/R2/P1/P2/Q＋validRoiL/R（零 ROI 置 Warning），参数全经 SetParams 注入、Execute() 无参；在用 07 CameraChain.cpp、01 calib_workflow.cpp
- 关键参数 alpha=0.0 与 flags 约束：达成｜alpha=0.0 全裁无效区；flags 须显式给 0 或 CALIB_ZERO_DISPARITY，默认值 1 不合法、默认构造过不了 validate（抛 invalid_argument）

## 3. 问题与方法

- 〔待定〕问题：无挂账（如有人工裁定项随时补）｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
