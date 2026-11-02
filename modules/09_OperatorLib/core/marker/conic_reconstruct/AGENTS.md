# conic_reconstruct AGENTS — 椭圆整体三维重建

> 先读：本目录 [`conic_reconstruct-说明与交接.md`](conic_reconstruct-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`ConicReconstructCPU`：左右椭圆参数 → 圆心/法向/半径一步到位——产线标记点重建主路径。

## 本目录禁令

- rowOffsetPx 是标定残差代偿——重标定后须重定档，勿跨标定沿用
- 通用契约/指纹红线见模块 AGENTS
