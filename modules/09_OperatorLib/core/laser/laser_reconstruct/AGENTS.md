# laser_reconstruct AGENTS — 激光线三维重建

> 先读：本目录 [`laser_reconstruct-说明与交接.md`](laser_reconstruct-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

双实现一目录：CUDA 版（标定链）＋CPU 版（扫描链五链）——Q 矩阵三角测量出 3D 点云。

## 本目录禁令

- 改任一版本须同步验证另一版本口径一致（Q/深度过滤参数）
- 通用契约/指纹红线见模块 AGENTS
