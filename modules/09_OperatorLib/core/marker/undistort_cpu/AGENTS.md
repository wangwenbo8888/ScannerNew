# undistort_cpu AGENTS — 标记点去畸变矫正

> 先读：本目录 [`undistort_cpu-说明与交接.md`](undistort_cpu-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`MarkerUndistortCPU`：边缘点一步到矫正系——marker_match 极线匹配的前提。

## 本目录禁令

- R1/R2/P1/P2 由标定 05 温度表在内存注入——勿从标定文件直读
- 通用契约/指纹红线见模块 AGENTS
