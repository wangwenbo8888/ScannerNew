# curve_map_temp_table AGENTS — 曲线映射温度表

> 先读：本目录 [`curve_map_temp_table-说明与交接.md`](curve_map_temp_table-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`CurveMapTempTableGenerator`：61 温度档批量出 CMTT sidecar（.bin），扫描链激光匹配的查表数据。

## 本目录禁令

- 硬规则 #3：参考档失败则整体 sidecar 为空——不得部分出产物
- 档 blob 只驻 sidecarBytes 一份（内存契约），勿另存副本
- 通用契约/指纹红线见模块 AGENTS
