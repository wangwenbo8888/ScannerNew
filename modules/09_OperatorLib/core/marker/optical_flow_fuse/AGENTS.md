# optical_flow_fuse AGENTS — 标记点光流配准

> 先读：本目录 [`optical_flow_fuse-说明与交接.md`](optical_flow_fuse-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`MarkerOpticalFlowFuseCPU`：帧间/对地图解 R/T——扫描链配准主路径，globalId 是三阶段（配准→融合→GBA）连接键。

## 本目录禁令

- existingMarkers 匹配全程不回写已有点坐标；useNormalWeight 默认关——翻转须人工批准
- 通用契约/指纹红线见模块 AGENTS
