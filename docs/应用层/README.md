# 应用层（app/）核心要点

> 本目录描述 `app/` 代码工程的核心要点，对应可执行产物 `scan_demo.exe`。
> 内容以代码为准（2026-08-14 核对；framework/ 已整体退役，app/ 链接归属迁移至 `base` + `mod_*`，详见下文）。

## 职责
`app/` 是**应用入口与装配层**：引导 Qt 应用、装配全部运行时组件（Data/Service/HAL/Infra/Workflow，现物理分布在 `base` 与 `modules/`）、托管主窗口 UI。它本身不实现业务逻辑，只做"组合根"（composition root）与依赖注入。

## 架构地位
- 在 5 层架构（UI/Workflow/Service∥Algorithm/Data/HAL）之外——框架整体图 §3.5 明确：主窗口/UI 外壳属 **app/ 层职责**（非模块、非框架元素）。
- 运行期由 `app/main.cpp` 装配；生命周期 = 整个应用。

## 文件清单

### 代码
| 文件 | 行数 | 职责 |
|------|------|------|
| `main.cpp` | 47 | 程序入口（Qt/OSG/spdlog 初始化 → 装配 AppContext → 全屏主窗口 → 主循环） |
| `AppContext.h` / `.cpp` | 80 / 95 | 装配根：创建并拥有全部运行时组件，经 `WorkflowContext` 注入 |
| `MainWindow.h` / `.cpp` | 122 / 1672 | 主窗口 UI（无边框 "LeadScan K2"，分区布局，槽函数驱动） |
| `WorkflowContext.h` / `.cpp` | 76 / 41 | 工作流注入窄接口（自 framework/workflow 迁入 app/，编入 APP_SOURCES） |
| `ScannerWindow.h` / `.cpp` | 74 / 459 | 扫描窗口 UI（自 modules/02 迁入 app/，MOD_SCANNING 现指 app/ 本地） |

### 构建 / 资源
| 文件 | 职责 |
|------|------|
| `CMakeLists.txt` | 产物 `scan_demo.exe`；编入 app + 部分 modules 文件；链接 `base` + `mod_devicemgmt/operatorlib/fileio/calibration/session/observability`（framework/ 退役后不再链 fw_* 库） |
| `copy_dlls.bat` | POST_BUILD 拷贝 Qt/OSG/OpenCV 运行时 DLL |
| `dark.qss` | 暗色主题样式表（⚠ 物理在 app/ 但未登记进 resources.qrc，当前不生效） |
| `resources.qrc` + `resources/icons/*.svg` | Qt 资源（大量三态：黑/红/灰 SVG 图标） |

## 文档索引
- [01-入口与启动.md](01-入口与启动.md) — `main.cpp` 详解
- [02-AppContext装配.md](02-AppContext装配.md) — 装配根与依赖注入
- [03-MainWindow-UI.md](03-MainWindow-UI.md) — 主窗口 UI 结构与槽函数
- [04-构建与依赖.md](04-构建与依赖.md) — CMake、依赖、部署

## 三大关键设计
1. **组合根模式**：`AppContext` 拥有全部对象，通过 `WorkflowContext` 窄接口注入。
2. **分层装配 + 逆序析构**：Infra → Data → Service → HAL → WorkflowContext → Workflow；`shutdown()` 逆序停止。
3. **UI 与运行时解耦**：`MainWindow` 持 `AppContext*` 按需取用，不直接 `new` 组件对象。

## 已知技术债（如实记录）
- **硬编码路径**：`D:/pointcloud_100M.ply`、`E:/workfold/.../calib_debug.log`、`OSG_PLUGIN_PATH=F:/osg3.6.5/...`（`main.cpp` 仍指向别机 F: 盘）。
- **模块文件编入 exe**：`modules/01_calibration`、`02_scanning`、`03_rendering`、`06_fileio` 的部分 `.cpp` 直接编入 `scan_demo`，而非链接 `mod_` 库（见 [04](04-构建与依赖.md)）。
- **工作流层整体闲置**：`AppContext` 装配了 `Scanner::workflow::{Scan,Calibration,PostProcess}Workflow`，但 `MainWindow` 一个都没调用——标定/扫描全走 calibration 模块的 `calib::*Workflow`（见 [03](03-MainWindow-UI.md)）。
- **dark.qss 未生效**：物理存在于 `app/` 但未登记进 `resources.qrc`，`:/icons/dark.qss` 资源路径无效（见 [01](01-入口与启动.md)）。

> 已解决（framework 退役迁移后，记录备查）：
> - **LEADSCANSeries 悬空 include（原无法编译）**：`app/stubs/` 已补齐 5 个头（LEADSCANSeries/CameraControl/camera_calib_workflow/laser_calib_workflow/scan_workflow），`MainWindow` 的 stubs include 不再悬空，`scan_demo` 现可全量构建（`ScannerWindow`↔`LEADSCANSeries` 的 `static_cast` 仍为占位 UB，待真实接入）。
> - **copy_dlls.bat 绝对路径（原指别机）**：三路径已从别机（`C:\Qt`/`F:\osg`/`F:\opencv`）修正为本机真实（`C:\devlibs\Qt-5.15.2` / `C:\devlibs\osg-install` / `C:\opencv-cuda-4.13.0`[-debug]），并补 `exit /b 0`。
