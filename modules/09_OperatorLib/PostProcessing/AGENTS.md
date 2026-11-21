# PostProcessing AGENTS — 后处理网格四族（PCMP 沙盒移植落点）

> 先读：模块 [`../AGENTS.md`](../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]；移植计划 `sandbox/PointCloudMeshPostProcessing/docs/plans/2026-11-21-PCMP移植主仓09算子库-计划v0.9.md`

## 一行速记

四算子＋统一索引网格（`IndexedMesh`）：法线定向清理 → 滚球封装 → 补洞 → SR 光顺；M0 骨架（261121），实现按 M1~M3 落地。

## 本目录禁令

- **禁止引入 Open3D/CGAL 等新第三方依赖**（移植计划 §3 依赖铁律——SR 零依赖为样板）
- 法线权威源唯一：stage0 出口起 `IndexedMesh.normals` 即权威，禁止旁路重算/外置翻转（R2）
- 阶段间只传 `IndexedMesh`（host 内存），禁止文件落盘中转（R1/R7）
- OpenMP 仅按 D2 口径（运行时开关，`numThreads` 可降 1，线程数无关结果）
- **mesh_smooth_sr 参数标定值在法务书面裁定（计划 §6-C2）前不得随产品发布**
