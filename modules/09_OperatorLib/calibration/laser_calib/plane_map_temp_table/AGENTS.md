# plane_map_temp_table AGENTS — 平面映射温度表

> 先读：本目录 [`plane_map_temp_table-说明与交接.md`](plane_map_temp_table-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`PlaneMapTempTable`：逐温度档的激光平面映射表——扫描链 start_scan 门禁要的「激光档表 planeMap」源头。

## 本目录禁令

- 勿与 `core/common/calib_result_types.h` 的 struct LaserPlaneMapTempTable（结果容器）混淆——那是传递件，这才是生成器
- 通用契约/指纹红线见模块 AGENTS
