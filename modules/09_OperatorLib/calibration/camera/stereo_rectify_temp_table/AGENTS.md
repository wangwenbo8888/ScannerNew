# stereo_rectify_temp_table AGENTS — 立体矫正温度表

> 先读：本目录 [`stereo_rectify_temp_table-说明与交接.md`](stereo_rectify_temp_table-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`StereoRectifyTempTableCpu`：一次算好各温度档的 R1/R2/P1/P2/Q——扫描链标记点 06/激光 05 查的就是这张表。

## 本目录禁令

- 表生成参数（CTE/温度网格）改动＝影响全链精度，须人工批准并全量回归
- 通用契约/指纹红线见模块 AGENTS
