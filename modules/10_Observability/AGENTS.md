# 10 AGENTS — 可观测

> 先读：本目录 [`10_Observability-说明与交接.md`](10_Observability-说明与交接.md)；对象速查 [`../../工程地图.md`](../../工程地图.md) §6；全局入口 [`../../AGENTS.md`](../../AGENTS.md)。

## 一行速记

监护 DLL（Observability.dll）：状态机/门禁/故障/日志/崩溃/性能——全局唯一状态裁决在这。

## 本目录禁令

- **勿在 EventBus 订阅回调内调 `reportFault`**（transition→publish 重入总线锁死锁）
- **SEH filter 内禁堆分配/锁/日志/zip**（会二次崩）；打包归下次启动 detectResidualDump
- `onStateChange` 回调可能乱序到达——需当前态须查 `getCurrentState()`；命令收尾只走 `notifyCompleted`
