# intrinsic_calib AGENTS — 内参标定

> 先读：本目录 [`intrinsic_calib-说明与交接.md`](intrinsic_calib-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`IntrinsicCalibCPU`：角点进、K/dist 出，标定链 CameraChain 的内参步；FileStorage/JSON 双持久化。

## 本目录禁令

- 类名尾缀 CPU 为历史风格，勿"顺手统一"改名（公共头改动＝契约指纹变）
- 通用契约/指纹红线见模块 AGENTS
