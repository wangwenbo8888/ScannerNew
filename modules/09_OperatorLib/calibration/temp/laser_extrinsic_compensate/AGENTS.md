# laser_extrinsic_compensate AGENTS — 激光外参温度补偿

> 先读：本目录 [`laser_extrinsic_compensate-说明与交接.md`](laser_extrinsic_compensate-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserExtrinsicCompensateCPU`：一次出虚拟相机→左右两表——temp 组唯一被 LaserChain 直接引用者。

## 本目录禁令

- 内部组合复用 extrinsic_compensate 的 Execute——改 02 号算子契约会连锁到这
- 通用契约/指纹红线见模块 AGENTS
