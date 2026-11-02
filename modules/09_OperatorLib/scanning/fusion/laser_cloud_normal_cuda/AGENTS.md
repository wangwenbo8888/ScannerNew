# laser_cloud_normal_cuda AGENTS — 激光法线估计（GPU）

> 先读：本目录 [`laser_cloud_normal_cuda-说明与交接.md`](laser_cloud_normal_cuda-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserCloudNormalCuda`：GPU 法线估计——代码侧当前无调用方（法线在扫描链由融合上下文/后处理 CPU 路径覆盖）。

## 本目录禁令

- 闲置算子：接入前先人工确认链路定位；退役走规范 §4.1 并同步修流水线文档
- 通用契约/指纹红线见模块 AGENTS
