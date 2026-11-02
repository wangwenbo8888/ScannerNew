# mask_extract AGENTS — 激光掩膜提取（旧版）

> 先读：本目录 [`mask_extract-说明与交接.md`](mask_extract-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`MaskExtractCUDA`：只提激光掩膜的旧版前段——标定/姿态链在用；扫描链已被 mask_separation 取代。

## 本目录禁令

- 扫描链新代码勿引用本件——用 `scanning/preprocess/mask_separation`
- 通用契约/指纹红线见模块 AGENTS
