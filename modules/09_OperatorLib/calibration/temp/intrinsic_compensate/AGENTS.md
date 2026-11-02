# intrinsic_compensate AGENTS — 内参温度补偿

> 先读：本目录 [`intrinsic_compensate-说明与交接.md`](intrinsic_compensate-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`IntrinsicCompensateCPU`：内参按温度档线性缩放——stereo_rectify_temp_table 的直接上游。

## 本目录禁令

- validate 要求 cx/cy>0（主点须正）；CTE 改动须人工批准
- 通用契约/指纹红线见模块 AGENTS
