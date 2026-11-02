# steger AGENTS — 激光中心提取（原版）

> 先读：本目录 [`steger-说明与交接.md`](steger-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`StegerExtractorCUDA`：经典 Hessian 中心提取——标定链用；扫描链用 `steger_fast`（同契约同精度）。

## 本目录禁令

- 三类共享类型唯一驻扎 common/steger_types.h（261024 ODR 修复）——勿在本目录重定义
- 通用契约/指纹红线见模块 AGENTS
