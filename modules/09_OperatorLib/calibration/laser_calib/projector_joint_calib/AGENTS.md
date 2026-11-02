# projector_joint_calib AGENTS — 投射器联合标定

> 先读：本目录 [`projector_joint_calib-说明与交接.md`](projector_joint_calib-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`ProjectorJointCalib`：激光标定链核心——光心＋发射曲线一起解，产出的 ImplicitCurve 喂 curve_map。

## 本目录禁令

- 多线版 F8 右目残差仅 baseline>0 启用——改默认须人工批准
- 通用契约/指纹红线见模块 AGENTS
