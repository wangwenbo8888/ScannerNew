# 06 AGENTS — 数据管理

> 先读：本目录 [`06_DataManager-说明与交接.md`](06_DataManager-说明与交接.md)；数据归属规则 [`docs/规范与模板/数据管理原则.md`](../../docs/规范与模板/数据管理原则.md)；对象速查 [`../../工程地图.md`](../../工程地图.md) §6。

## 一行速记

仓库＋缓冲＋记账的共享 DLL（DataManager.dll）：标定表/点云/帧环/L4 产物都在这。

## 本目录禁令

- **依赖铁律「06 不链 08」**（08 测试 exe 作叶子消费方例外）
- `SlotRing::pushFrame` 单生产者铁约——仅采集回调单线程调用；槽位构造期定死无 resize
- 写策略勿混：扫描环＝Overwrite、姿态环＝Backpressure；正确用法先 `waitFor(frameId)` 再 `read()`（read 是近似语义）
