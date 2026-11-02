# global_optim AGENTS — 全局捆绑优化（GBA）

> 先读：本目录 [`global_optim-说明与交接.md`](global_optim-说明与交接.md)；模块入口 [`../../AGENTS.md`](../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]；阶段定位 `docs/流水线/客户端扫描流水线.md` §四。

## 一行速记

`GlobalBundleAdjustmentCPU`：Ceres＋PGO 的终局批优化——秒~分钟级、批处理模型，与 200fps 流式不同。

## 本目录禁令

- **不入算子三元组**（关键决策）——独立组件，勿迁就算子规范改形
- pose_graph_residuals 的 so3 log 近似仅 |θ|≲0.3 rad 有效——大角度场景勿依赖
- 高精度点软先验 σ 权重属精度红线——改动须人工批准并全量回归
