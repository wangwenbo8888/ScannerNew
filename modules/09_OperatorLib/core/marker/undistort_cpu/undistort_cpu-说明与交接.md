> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# undistort_cpu · 标记点去畸变矫正

## 1. 目标

- 功能：双目稀疏点一步去畸变＋立体矫正（标记点链 06 步）
- 方法/参数：stereoRectify 算 R1/R2/P1/P2/Q→undistortPoints 一步完成；或 SetRectifyMatrices 注入外部矩阵
- 注意：矫正后左右 Y 对齐是 marker_match 配对前提；R/P 由温度表内存注入；1 项单测

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 双目稀疏点一步去畸变＋立体矫正（标记点链 06 步）：达成｜主类 MarkerUndistortCPU：stereoRectify→undistortPoints 一步完成，约 0.01 pixel；07 fivechain FiveChainAdapter.cpp:572＋posture PosturePipeline.cpp:118 在用
- SetRectifyMatrices 外部矩阵注入路径：达成｜注入后跳过内部 stereoRectify，矩阵来自 stereo_rectify_temp_table 温度条目内存注入
- 矫正后左右 Y 对齐（marker_match 配对前提）：达成｜对齐语义在册，为下游 08 步极线配对前提
- 单测：达成｜tests/test_undistort_points_cpu.cpp 1 项

## 3. 问题与方法

- 〔待定〕问题：暂无——待人工提出｜方法：—｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
