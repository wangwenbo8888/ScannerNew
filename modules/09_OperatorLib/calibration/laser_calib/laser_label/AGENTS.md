# laser_label AGENTS — 激光线重编号

> 先读：本目录 [`laser_label-说明与交接.md`](laser_label-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserLabelerCUDA`：掩膜 → 有序线号——标定链 Steger 之前的重编号步。

## 本目录禁令

- 质量门禁（expectedLineCount/maxRunWidth）触发只置 frameRejected、success 仍 true——调用方须整帧弃用 L+R
- 通用契约/指纹红线见模块 AGENTS
