# frame_fuse AGENTS — 单帧快速配准

> 先读：本目录 [`frame_fuse-说明与交接.md`](frame_fuse-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`FrameFuseCPU`：无对应关系的两帧硬配准（描述子＋SVD）——光流配准失败时的兜底。

## 本目录禁令

- 与 optical_flow_fuse 是并列路径（fuseExecute/acceptFuse 分支）——改分支逻辑在 07 harness，勿在本算子内做链决策
- 通用契约/指纹红线见模块 AGENTS
