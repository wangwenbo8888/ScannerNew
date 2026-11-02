> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# conic_reconstruct · 椭圆整体三维重建

## 1. 目标

- 功能：双目椭圆整体参数闭式重建三维圆（圆心/法向/半径），一步替代旧边点路径 09+10+11（现役主路径）
- 方法/参数：EllipseFitCPUResult×2＋centerMatches＋注入 K1/K2/C2 闭式解；rowOffsetPx 代偿右目行偏置残差（conic_offline --row-offset scan 定档）
- 注意：重标定后 rowOffsetPx 须重定档，勿跨标定沿用；1 项单测

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 双目椭圆整体参数闭式重建三维圆（圆心/法向/半径）：达成｜主类 ConicReconstructCPU：EllipseFitCPUResult×2＋centerMatches＋注入 K1/K2/C2 一步闭式解，输出 MarkerReconstructResult
- 现役主路径（一步替代旧边点路径 09+10+11）：达成｜07 fivechain FiveChainAdapter.cpp:629＋posture PosturePipeline.cpp:152 在用
- rowOffsetPx 代偿右目行偏置残差：达成｜conic_offline --row-offset scan 定档；重标定后须重定档勿跨标定沿用（在册）
- 单测：达成｜tests/test_conic_reconstruct_cpu.cpp 1 项

## 3. 问题与方法

- 〔待定〕问题：重标定后 rowOffsetPx 须重新定档（无独立挂账，随五链真机验证演进）｜方法：conic_offline --row-offset scan 重定档，勿跨标定沿用｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准；本算子无存量卡，按文件内事实小幅扩写）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
