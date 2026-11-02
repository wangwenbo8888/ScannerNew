# pose_estimate AGENTS — 设备姿态评估

> 先读：本目录 [`pose_estimate-说明与交接.md`](pose_estimate-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`PoseEstimateCPU`：当前位姿 vs 预设目标姿态，命中即回调——07 PosturePipeline 的判定核心。

## 本目录禁令

- 欧拉角约定 ZYX 内旋（度）——改约定＝改契约，须人工批准
- 通用契约/指纹红线见模块 AGENTS
