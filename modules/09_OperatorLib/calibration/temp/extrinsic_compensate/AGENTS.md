# extrinsic_compensate AGENTS — 外参温度补偿

> 先读：本目录 [`extrinsic_compensate-说明与交接.md`](extrinsic_compensate-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`ExtrinsicCompensateCPU`：外参按温度档缩放基线——温度表链 02 号。

## 本目录禁令

- validate 强制 T 向量非零（基线必须存在）；CTE 改动须人工批准
- 通用契约/指纹红线见模块 AGENTS
