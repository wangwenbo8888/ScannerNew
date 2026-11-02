> ⚠️ 本目录进度只写在这份文档：更新「现状/日志」须同批派生《开发进度.md》（头部滚动＋问题池），地图按结构变化定点刷——缺一即未完成；目标段与问题取舍由人定，AI 不自行发起更新（规范 §2.3）。

# epipolar_interp_dual · 极线插值（双模式）

## 1. 目标

- 功能：极线插值双模式——Labeled（标定）/Scan（扫描，无号可插），扫描链 06 步
- 方法/参数：InterpMode=Auto/Labeled/Scan 三档；Labeled＝带线号判同线，Scan＝仅凭几何邻近配对插值
- 注意：Scan 不产线号（留给 laser_match_scan）；目录内 0 项单测（覆盖在 07 test_fc_replay 对拍）

〔待人定稿〕——人确认前为 AI 代拟

## 2. 现状

- 极线插值双模式（Labeled/Scan，扫描链 06 步）：达成｜主类 EpipolarInterpDualCuda，07 fivechain harness.h＋FiveChainAdapter.cpp 扫描链在用
- InterpMode=Auto/Labeled/Scan 三档：达成｜Auto 按输入线号自动判；Labeled 带线号判同线，Scan 无号仅凭几何邻近配对插值
- Scan 不产线号（留给 laser_match_scan）：达成｜EpipolarInterpResult 不带扫描线号，结果类型抽至 common/epipolar_types.h 共享
- 单测：部分达成｜目录内 0 项，覆盖在 07 test_fc_replay 对拍

## 3. 问题与方法

- 〔待定〕问题：暂无——待人工提出（原下一步：无独立挂账，随五链真机验证演进）｜方法：—｜提出 261102｜解决 —

## 4. 日志

- 261102｜现状改目标达成对照制｜规范 §2.3
- 261102｜转四段新制：目标待人定稿；问题段自下一步迁入（均〔待定〕）｜规范 §2.3
- 261102｜现状段按存量参考卡颗粒度扩充（源：docs/存量档案/算子说明文档＋代码实测校准；本算子无存量卡，按文件内事实小幅扩写）｜—｜—
- 261102｜目标段升级：颗粒行＋量化锚（规范 §2.3 新规）｜人工触发·AI 代笔｜规范 §2.3
- 261102｜首次建立本说明（人工触发·AI 代笔，事实经代码实读核实）｜基线＝「现状」段｜—
