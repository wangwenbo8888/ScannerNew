# frame_filter AGENTS — 帧类型判别

> 先读：本目录 [`frame_filter-说明与交接.md`](frame_filter-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`FrameFilterCUDA`：这帧是标记点帧还是激光帧——姿态链的门神。

## 本目录禁令

- maskRatioThreshold 默认 0＝恒真占位——定真实阈值须人工标定
- 通用契约/指纹红线见模块 AGENTS
