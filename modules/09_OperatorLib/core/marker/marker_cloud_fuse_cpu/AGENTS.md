# marker_cloud_fuse_cpu AGENTS — 标记点体素融合

> 先读：本目录 [`marker_cloud_fuse_cpu-说明与交接.md`](marker_cloud_fuse_cpu-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`MarkerCloudFuseCPU`：逐帧标记点 → 全局去重累积器——「首个落入者为代表」天然保住 seed 先入点。

## 本目录禁令

- **生命周期强约束**：seed() 预置必须在任何扫描帧之前；停止时 Clear()——顺序错＝基准点被扫描点顶掉
- 通用契约/指纹红线见模块 AGENTS
