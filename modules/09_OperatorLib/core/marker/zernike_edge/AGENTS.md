# zernike_edge AGENTS — Zernike 边缘亚像素提取

> 先读：本目录 [`zernike_edge-说明与交接.md`](zernike_edge-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`ZernikeEdgeCPU`：小图块 → 椭圆边缘亚像素点，标记点链第 04 步。

## 本目录禁令

- templateSize 只支持 5/7——扩档须验证精度基准
- 通用契约/指纹红线见模块 AGENTS
