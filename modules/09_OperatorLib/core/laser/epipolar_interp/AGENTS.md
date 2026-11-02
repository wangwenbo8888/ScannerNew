# epipolar_interp AGENTS — 极线插值（单模式）

> 先读：本目录 [`epipolar_interp-说明与交接.md`](epipolar_interp-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`EpipolarInterpCuda`：标定链极线插值；lineIdCheck 双语义（true=按线号同线，false=纯几何）。

## 本目录禁令

- 扫描链勿用本件——用 `epipolar_interp_dual`（Scan 模式）
- 通用契约/指纹红线见模块 AGENTS
