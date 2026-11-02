> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# mask_separation · 激光/标记点掩膜分离

## 1. 目标

- 功能：同帧掩膜宽度域分离——一张灰度图出激光/标记点两张掩膜（扫描链预处理-01）
- 方法/参数：高斯＋二值化后按宽度差分级腐蚀（laserKillSize=9／markerKillSize=37）＋边缘膨胀补偿；细激光形态 v2＝高斯→分级腐蚀＋top-hat
- 注意点：分离前提「激光细、标记点宽」；A 模式（无激光）未实测，实测前勿默认正确

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 一张灰度图出激光/标记点两张掩膜（扫描链预处理-01）：达成｜主类 `LaserMarkingSeparationCUDA`（laser_markingpoint_mask_separation_cuda.h，pimpl 见 _pimpl.h）：CV_8UC1 → LaserMarkingSeparationResult（d_laserMask＋d_markingPointMask）；在用：07 fivechain FiveChainAdapter sepL_/sepR_（左右相机各一）＋harness＋07 test_fc_replay
- 高斯＋二值化＋宽度差分级腐蚀＋边缘膨胀补偿：达成｜v2 代码实测默认：threshold=80、noiseErodeSize=3；laserKillSize=9（对付激光宽 4-7px）／markerKillSize=37（标记点 ~35px＋2px 余量）；laserEdgeDilateSize=9／markingEdgeDilateSize=7 补回 ~3px/边
- 细激光形态 v2（高斯→分级腐蚀＋top-hat）：达成｜减法＝掩膜−膨胀(腐蚀)、默认圆形结构元；存量卡 v1 六步形态学全自定义 CUDA 内核骨架沿用、参数表已换血（卡「激光粗 20-40px/标记点细 5-10px」预设与 v2 正相反已过时，只报告）
- 约束（分离前提「激光细、标记点宽」/非线程安全）：达成（已固化）｜Debug 有 inProcess_ 并发检测、左右相机各持独立实例
- A 模式（纯标记点、无激光）：未达成｜工况未经设计验证、待联调实测——备选退回 mask_extract 旧前段（《客户端扫描流水线.md》§3.4，问题段挂账）
- 单测：达成｜tests/test_laser_markingpoint_mask_separation_cuda.cpp（example_usage.cpp 被 CMake 排除）

## 3. 问题与方法

- 〔待定〕问题：A 模式（无激光）工况实测定案｜方法：实测走本件或退回 mask_extract 旧前段｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步+全局挂账迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
