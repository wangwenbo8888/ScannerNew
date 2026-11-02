# laser_cloud_fuse_cuda AGENTS — 激光点云融合（GPU）

> 先读：本目录 [`laser_cloud_fuse_cuda-说明与交接.md`](laser_cloud_fuse_cuda-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserCloudFuseCuda`：产线激光融合——体素哈希常驻显存，与标记点哈希共栅格。

## 本目录禁令

- removePoints 是编辑账本入口（05 D4）——删除语义改动须联测 05/07
- 扩容 Rehash 行为与显存上限强相关——改槽位参数须压测（TEST/laser_fuse_stress）
- 通用契约/指纹红线见模块 AGENTS
