# 应用层（app/）文档索引

> `app/` 是**应用入口与装配层**：引导 Qt 应用、装配全部运行时组件、托管主窗口 UI。
> 本身不实现业务逻辑，只做组合根（composition root）与依赖注入。产物 `scan_demo.exe`。
> 四份主题文档已按「模块协作文档」模板重构（2026-10-01，基于代码 1f2b19a）——
> 人经各文档 §1.2 追加需求，AI 经 §3/§4 汇报差距与代码现状。

## 文档索引（协作文档）

| 文档 | 主题 | 代码范围 |
|------|------|---------|
| [01-入口与启动.md](01-入口与启动.md) | main.cpp 启动序列/自检门禁/关闭 | `app/main.cpp` |
| [02-AppContext装配.md](02-AppContext装配.md) | 组合根/命令通道点火/会话生命周期/模拟提取 | `app/AppContext.*` `app/WorkflowContext.*` `app/SceneFeedAdapter.*` |
| [03-MainWindow-UI.md](03-MainWindow-UI.md) | 主窗口布局/扫描操作闭环/编辑工具链/信息面板 | `app/MainWindow.*` `app/ScannerWindow.*` |
| [04-构建与依赖.md](04-构建与依赖.md) | scan_demo 编译归属/链接/运行时部署 | `app/CMakeLists.txt` `app/copy_dlls.bat` `app/resources.qrc` |

## 代码清单（2026-10-01 核对）

| 文件 | 职责 |
|------|------|
| `main.cpp`（131 行） | 入口：日志/崩溃前置 → 环境准备 → 装配 → 主窗口 → 启动期弹窗 → 主循环 → 对称关闭 |
| `AppContext.h/.cpp`（227/1125 行） | 装配根＋命令通道点火＋扫描会话生命周期＋自检/故障桥/设备灯＋中段模拟提取 |
| `WorkflowContext.h/.cpp` | 工作流依赖窄接口（DI 聚合器） |
| `SceneFeedAdapter.h/.cpp` | 07 ISceneFeed 首个实现：流水线云快照 queued 推 UI 渲染 |
| `MainWindow.h/.cpp`（176/2714 行） | 无边框主窗口 "LeadScan K2"（布局/槽/编辑/信息面板） |
| `ScannerWindow.h/.cpp/.ui` | 集成测试窗口（设备开闭/预览消费/参数滑条，经 DeviceManager 门面） |
| `stubs/`（5 个桩头） | LEADSCANSeries/CameraControl/标定工作流桩（人工提供，构建门禁依赖） |
| `CMakeLists.txt`＋`copy_dlls.bat`＋`resources.qrc`＋`resources/icons/` | 构建/部署/Qt 资源（三态 SVG 图标） |
| `dark.qss` | 暗色主题（未登记 qrc，暂不生效） |

## 三大关键设计

1. **组合根模式**：AppContext 拥有全部对象，经 WorkflowContext 窄接口注入工作流，UI 经裸指针 getter 取用
2. **命令通道统一点火**：标定/扫描/后处理经 10-CommandGate（pre 门禁＋handler 毫秒级返回），完成经 onFinished 合账——跨模块接线只在 app 完成（08/02/01/04 不反链 10）
3. **UI 与运行时解耦**：MainWindow 不 new 组件；EventBus 回调一律 queued 切主线程（总线锁红线）

## 顶层速记（详见各文档 §3 差距表）

- 标定槽仍走 stubs 空桩（真实链 01 工作流未接 UI）——03-D1
- dark.qss 未登记 qrc，暗色主题不生效——01-D1/03-D3/04-D4
- copy_dlls.bat 全绝对路径且 Galaxy DLL 借自 factory_calib 构建目录——04-D1
- MainWindow.cpp 单文件 2714 行——03-D6
