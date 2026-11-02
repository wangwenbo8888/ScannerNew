# epipolar_interp_dual AGENTS — 极线插值（双模式）

> 先读：本目录 [`epipolar_interp_dual-说明与交接.md`](epipolar_interp_dual-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`EpipolarInterpDualCuda`：扫描链极线插值——无号扫描点也能配对插值。

## 本目录禁令

- Scan 模式不产线号（留给 laser_match_scan 内部确定）——上游算子勿依赖 line_id 语义
- 通用契约/指纹红线见模块 AGENTS
