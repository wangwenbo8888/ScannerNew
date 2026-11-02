# mask_separation AGENTS — 激光/标记点掩膜分离

> 先读：本目录 [`mask_separation-说明与交接.md`](mask_separation-说明与交接.md)；模块入口 [`../../../AGENTS.md`](../../../AGENTS.md)；契约 [`docs/规范与模板/算子规范.md`]。

## 一行速记

`LaserMarkingSeparationCUDA`：一张灰度图 → 激光/标记点两张掩膜——扫描链第一步，宽度域分离。

## 本目录禁令

- 分离前提是「激光细、标记点宽」宽度差——A 模式（无激光）未经设计验证，实测前勿默认正确
- 通用契约/指纹红线见模块 AGENTS
