# point_reconstruct AGENTS — 边缘点三维重建

> 先读：本目录 [`point_reconstruct-说明与交接.md`](point_reconstruct-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`PointReconstructCPU`：边点对 → 圆心 3D＋法线（边点法 11 步）——与 conic_reconstruct 是竞争方案，现产线选了后者。

## 本目录禁令

- 闲置算子：去留待人工裁定；退役走规范 §4.1
- 通用契约/指纹红线见模块 AGENTS
