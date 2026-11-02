# 08 AGENTS — 设备管理

> 先读：本目录 [`08_DeviceManager-说明与交接.md`](08_DeviceManager-说明与交接.md)；协议取证 `docs/开发需要的信息/下位机和按键/`；对象速查 [`../../工程地图.md`](../../工程地图.md) §6。

## 一行速记

硬件门面 DLL（DeviceManager.dll）：相机/下位机/按键/监控全在这；对外只走 `DeviceManager` 一个口。

## 本目录禁令

- **串口定版 QSerialPort**——勿回 Win32 CreateFile 版（CH343 兼容已栽过）
- **线程纪律 Critical #1**：切模式等变异入口必经 `post()` 任务队列转逻辑线程；无锁零件单一属主＝逻辑线程
- 铁规「不漏零件指针/类型」（MCUDriver/KeySemantics 仅前向声明）
- 真机操作禁 AI 代劳：不启动 exe、不跑真机扫描（AI工作纪律）
