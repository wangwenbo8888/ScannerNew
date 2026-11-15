# laser_match_scan_v4 AGENTS — 激光线匹配 v4（裁判型后段）

> 先读：本目录 [`laser_match_scan_v4-说明与交接.md`](laser_match_scan_v4-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserMatchScanSplitCpuV4`：v3 前段产物的 CPU 后段——双分数/自适应窗/弃权门三件抗污染机制；V5 的回退臂；261115 自 TEST/fullgpu_optimize 沙盒回迁。

## 本目录禁令

- L0 恒等契约：四开关全关（或温表/温度未就绪）→ 与 LaserMatchScanSplitCpuV3 同输入同参数逐位一致——改动判定链勿破坏此契约
- 票仓不动：v3 整数票仓与判定序列逐字保留；分数票仓（cntQ_/sumRes_）独立数组永不写回 votes
- 仅争议段介入——非争议段逐字节放行＝零漂移
- 温表注入走 SetTempTable；深度门需 SetStereoQ 注入 Q（未注入自动关）
- 通用契约/指纹红线见模块 AGENTS
