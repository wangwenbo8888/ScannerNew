# curve_map AGENTS — 极线曲线映射表

> 先读：本目录 [`curve_map-说明与交接.md`](curve_map-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`CurveMapGenerator`/`CurveMapTable`：激光左右相机 u↔u 映射表——扫描链查表匹配（CMTT）的数据源。

## 本目录禁令

- **禁 R1/R2**（矫正系直连）；rowStep 必须从标定 JSON 继承，禁硬编码；FAIL 不出产物
- 通用契约/指纹红线见模块 AGENTS
