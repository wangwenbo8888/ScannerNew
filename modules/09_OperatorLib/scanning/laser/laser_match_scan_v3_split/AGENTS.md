# laser_match_scan_v3_split AGENTS — 激光线匹配 v3（拆分版）

> 先读：本目录 [`laser_match_scan_v3_split-说明与交接.md`](laser_match_scan_v3_split-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserMatchScanSplitGpuV3/CpuV3`：GPU 前段＋CPU 后段错帧并行——产线默认匹配路径（性能 80-85 fps 级）。

## 本目录禁令

- 前段/后段是成对契约——改任一侧须对拍另一侧（ABAB 基准在沙箱）
- FrontV3 池化依赖 resetKeepCapacity——勿在池外 new 前段结果
- 通用契约/指纹红线见模块 AGENTS
