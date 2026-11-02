# laser_cloud_fuse AGENTS — 激光点云融合（CPU）

> 先读：本目录 [`laser_cloud_fuse-说明与交接.md`](laser_cloud_fuse-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserCloudFuseCPU`：CPU 版体素融合——主要给后处理 NormalStage 当法线宿主；产线实时用 CUDA 版。

## 本目录禁令

- 产线实时链勿换用本件（性能）——用 `laser_cloud_fuse_cuda`
- 通用契约/指纹红线见模块 AGENTS
