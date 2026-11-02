# extrinsic_calib AGENTS — 外参标定

> 先读：本目录 [`extrinsic_calib-说明与交接.md`](extrinsic_calib-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`ExtrinsicCalibCpu`：角点进、R/T/E/F 出，标定链 CameraChain 的外参步。

## 本目录禁令

- Result 是 move-only（规范 §3.3）——勿按值拷贝传递
- 通用契约/指纹红线见模块 AGENTS
