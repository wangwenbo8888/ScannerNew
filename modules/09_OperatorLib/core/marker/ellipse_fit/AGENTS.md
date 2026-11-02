# ellipse_fit AGENTS — 椭圆拟合中心提取

> 先读：本目录 [`ellipse_fit-说明与交接.md`](ellipse_fit-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`EllipseFitCPU`：边缘点 → 椭圆中心＋参数，标记点链第 07 步；simd/vendor 是内部加速路径非独立算子。

## 本目录禁令

- ellipse_fit_simd/ellipse_fit_vendor 无外部引用——勿当独立算子调用
- 通用契约/指纹红线见模块 AGENTS
