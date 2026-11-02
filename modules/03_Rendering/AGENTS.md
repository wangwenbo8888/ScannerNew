# 03 AGENTS — 渲染显示

> 先读：本目录 [`03_Rendering-说明与交接.md`](03_Rendering-说明与交接.md)；对象速查 [`../../工程地图.md`](../../工程地图.md) §6；全局入口 [`../../AGENTS.md`](../../AGENTS.md)。

## 一行速记

一个大类 `OSGWidget`（OSG＋Qt 视窗）＋`RenderSanity` 纯函数；数据进、像素出，不反向控制任何流程。

## 本目录禁令

- 纯显示：不得从这里发起流程控制/状态切换
- 不依赖 base——fault 出口由 app 桥接 EventBus，勿直接引入
- OSGWidget.cpp 编入 exe：改它＝改主程序，注意编译纪律与真机联调口径
