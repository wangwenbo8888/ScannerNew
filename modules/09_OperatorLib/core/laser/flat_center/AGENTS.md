# flat_center AGENTS — 平顶激光中心提取

> 先读：本目录 [`flat_center-说明与交接.md`](flat_center-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`FlatCenterExtractor`：平顶激光专用中心提取（steger_fast 的备用路径）。

## 本目录禁令

- lowThreshold 与 steger_fast 的阈值不同域（灰度 vs |λ|）——数值不可照搬
- 通用契约/指纹红线见模块 AGENTS
