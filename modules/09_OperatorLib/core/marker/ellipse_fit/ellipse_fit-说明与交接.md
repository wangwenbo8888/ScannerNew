> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# ellipse_fit · 椭圆拟合中心提取

## 1. 目标

- 功能：RANSAC 剔离群＋fitEllipseAMS 鲁棒拟合椭圆中心（标记点链 07 步）
- 方法/参数：随机采 5 点→fitEllipse→轴长/内点过滤→早停，最佳内点集最终 fitEllipseAMS；ransacThreshold=0.5
- 注意：固定种子 12345 结果可复现；simd/vendor 为内部加速路径（非独立算子）；1 项单测

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- RANSAC 剔离群＋fitEllipseAMS 鲁棒拟合椭圆中心（标记点链 07 步）：达成｜主类 EllipseFitCPU：随机采 5 点→fitEllipse→轴长/内点过滤→早停→fitEllipseAMS（ransacThreshold=0.5），约 0.01 pixel；07 fivechain FiveChainAdapter.cpp:591＋posture PosturePipeline.cpp:129 在用
- 固定种子结果可复现：达成｜RANSAC 固定种子 12345；AMS 失败自动降级 cv::fitEllipse，仍异常 success=false＋Degraded
- simd/vendor 为内部加速路径（非独立算子）：达成｜ellipse_fit_simd/vendor 为 cpu.cpp 内部加速路径，无外部引用
- 单测：达成｜tests/test_ellipse_fit_cpu.cpp 1 项

## 3. 问题与方法

- 〔待定〕问题：暂无——待人工提出｜方法：—｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
