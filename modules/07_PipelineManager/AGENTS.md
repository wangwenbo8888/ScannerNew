# 07 AGENTS — 流水线管理

> 先读：本目录 [`07_PipelineManager-说明与交接.md`](07_PipelineManager-说明与交接.md)；流水线详图 `docs/流水线/`；调度方案 `docs/流水线/扫描并行调度260629.md`；对象速查 [`../../工程地图.md`](../../工程地图.md) §6。

## 一行速记

调度底座（`sched/`，零算子知识）＋五条流水线（`pipelines/`）＋扫描五链（`pipelines/scan/fivechain/`）；出 PipelineManager.dll，契约指纹与 exe 互锁。

## 本目录禁令

- **调度底座零算子知识**：GPU 链经 LaneHooks.gpuChain 注入，第 4 参 `frontReady()` 回调＝ccl 就绪点——勿破坏此契约
- **pause/resume 消费水位持久**：resume 不得从 0 重扫（错误 R/T 污染点云＋obs 重复）
- 算子实现归 09——本目录只编排；改公共头＝契约指纹变＝全量重编（`scripts/replace_dlls.ps1` 护栏）
