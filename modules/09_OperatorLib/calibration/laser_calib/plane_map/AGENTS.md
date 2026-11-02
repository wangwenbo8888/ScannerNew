# plane_map AGENTS — 平面映射表

> 先读：本目录 [`plane_map-说明与交接.md`](plane_map-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`PlaneMapCuda`：虚拟像素→左右相机 u 映射——激光标定链的平面表步。

## 本目录禁令

- 深度窗默认 100–5000mm，与 curve_map（100–700mm）口径不同——勿混用参数
- 通用契约/指纹红线见模块 AGENTS
