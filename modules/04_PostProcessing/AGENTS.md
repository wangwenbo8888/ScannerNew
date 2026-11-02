# 04 AGENTS — 点云后处理

> 先读：本目录 [`04_PostProcessing-说明与交接.md`](04_PostProcessing-说明与交接.md)；后处理流水线在 `../07_PipelineManager/pipelines/postprocess/`；对象速查 [`../../工程地图.md`](../../工程地图.md) §6。

## 一行速记

薄编排壳（`PostProcessWorkflow`）：记账与出口；批算在 07、网格算子等 09。

## 本目录禁令

- 批算/算法不内置——阶段实现在 07 PostProcessPipeline，算子在 09
- 网格四族算子未落地前保持 degraded 桩口径，勿造假实现
