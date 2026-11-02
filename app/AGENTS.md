# app AGENTS — 应用装配

> 先读：本目录 [`app-说明与交接.md`](app-说明与交接.md)；对象速查 [`../工程地图.md`](../工程地图.md) §6；全局入口 [`../AGENTS.md`](../AGENTS.md)。

## 一行速记

组合根＋主窗：main.cpp → LeadScannerK2.exe；全部对象在 AppContext 创建接线。

## 本目录禁令

- **对象创建集中在 AppContext**——别处不散落 new（组合根纪律）
- `stubs/` 5 个旧头**勿引用**（清理时机待人工裁定）
- 改 07/09 后先关 exe 再编（运行中锁 DLL）
