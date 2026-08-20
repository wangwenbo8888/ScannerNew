# 03 · MainWindow 主窗口 UI

> `app/MainWindow.h`（122 行）/ `MainWindow.cpp`（1672 行）。无边框主窗口 "LeadScan K2"，承担全部 UI 布局与操作入口。

## 总体
- 基类 `QMainWindow`，**无边框**（`Qt::FramelessWindowHint`），标题 "LeadScan K2"。
- 持 `AppContext* m_appCtx`，按需取用其运行时容器（`DeviceStateCache` / `PointCloudBuffer` / `FrameBuffer`，不 new）。
- 嵌入 `OSGWidget`（模块3）做 3D 点云渲染视图。
- **构建状态（framework 退役后）**：现可全量构建。`MainWindow.cpp:9-13` 的 `#include "stubs/*.h"` 解析到 `app/stubs/` 下 5 个桩头（LEADSCANSeries / CameraControl / camera_calib_workflow / laser_calib_workflow / scan_workflow），include 不再悬空。`ScannerWindow`（集成测试窗口）已自 modules/02 迁入 `app/`，与 MainWindow 同目录。

## 布局结构（构造函数 127–158）
```
QMainWindow (FramelessWindowHint)
└─ central (QVBoxLayout, 边距0/间距0)
   ├─ createTitleBar()        // 标题栏
   ├─ createNavBar()          // 导航栏
   ├─ createToolBar()         // 工具栏
   └─ QHBoxLayout (contentLayout)
      ├─ createLeftPanel(), stretch=2   // 左侧面板(项目/参数/信息)
      └─ create3DViewArea(), stretch=5  // 3D 视图区
+ createFloatingToolbar()     // 悬浮工具条
+ startInfoTimer()            // 信息面板定时刷新
```

## MainWindow.cpp 文件地图（行号）
| 行 | 内容 |
|----|------|
| 41–58 | `renderSvg()` ×2（SVG → QPixmap，静态） |
| 60–125 | `ArrowSlider`（自定义滑块，带 groove pixmap 绘制） |
| 127–158 | 构造函数（组装布局） |
| 162–172 | `onIntegrateTestClicked()` — 打开集成测试对话框 |
| 174–180 | `onReloadPointCloud()` — 重载点云（⚠ 硬编码 `D:/pointcloud_100M.ply`） |
| 182–403 | `onCalibDeviceClicked()` — 标定流程（221 行，相机标定采集） |
| 404–471 | `onScanClicked()` — 扫描流程 |
| 472–491 | `closeEvent` |
| 492–535 | 悬浮工具条 create/reposition |
| 536–612 | `createTitleBar` |
| 613–721 | `createNavBar` |
| 722–970 | `createToolBar`（最大，248 行） |
| 971–1000 | `createLeftPanel` |
| 1001–1063 | `createProjectSection` |
| 1064–1161 | `createParamSection` |
| 1162–1237 | `createInfoSection` |
| 1238–1295 | `create3DViewArea` |
| 1296–1369 | `createBottomToolBar` |
| 1370–1434 | 按钮工厂（nav/tool/selection + group exclusive） |
| 1435–1497 | `eventFilter` |
| 1498–1535 | `startInfoTimer` |
| 1536–1672 | `updateInfoSection`（周期刷新系统信息） |

## 槽函数 → 业务集成
| 槽 | 触发 | 行为 |
|----|------|------|
| `onIntegrateTestClicked` | 导航栏「集成测试」 | `new ScannerWindow(m_appCtx, this)` 赋给 `m_integrateTestDialog`（现声明为 `QWidget*`，见 MainWindow.h:104）并 show。ScannerWindow 现位于 `app/`（自 modules/02 迁入） |
| `onReloadPointCloud` | 导航栏「加载点云」 | `m_3dView->loadTestDataFromPLY("D:/pointcloud_100M.ply", 0)` |
| `onCalibDeviceClicked` | 工具栏「校准设备」 | 弹 `CalibDialog` → `cameraCalibClicked`/`laserCalibClicked` → 经 `LEADSCANSeries::getCameraControl()`（**stub**）取相机，循环采 15 帧 → `calibration::CameraCalibWorkflow`（**stub**）.run()。stub 的 `getCameraControl()` 返回 nullptr，相机检查恒失败 |
| `onScanClicked` | 导航栏「扫描」 | 隐藏标定板/彩条 → 清空 3D 场景 → 经 `LEADSCANSeries`（**stub**）取相机 → **`calibration::ScanWorkflow`（stub）.processFrameToCloud()** → 结果点云载入 `m_3dView`。stub 返回 0 点，实际不载入 |

## 信息面板（系统监控显示）
- `startInfoTimer()`：`QTimer` 周期触发 `updateInfoSection()`。
- `updateInfoSection()` 读 `m_appCtx->deviceStateCache()` / `pointCloudBuffer()` / `frameBuffer()`，刷新：连接状态、点云数、FPS、温度、CPU、内存。
- 此处是**模块8 HardwareMonitor 采集的数据流向 UI 的出口**。

## 图标系统
- 大量 SVG 图标，三态：**黑 / 红 / 灰**（如 `下一步-灰-13.svg` / `-红-13` / `-黑-13`），按按钮状态切换。
- `renderSvg()` 把 SVG 渲染为 QPixmap；按钮工厂 `createIconButton/createNavButton/createToolButton/createSelectionButton` 统一生成。

## ⚠ 技术债（如实记录）
1. **标定/扫描槽依赖空桩，未接真实算子链**：`onScanClicked` 用 `calibration::ScanWorkflow`、`onCalibDeviceClicked` 用 `calibration::{Camera,Laser}CalibWorkflow`，但这些类型现由 `app/stubs/*.h` 提供——均为返回 "not implemented" / 0 点的**空桩**，未接入 modules/09 算子库或真实标定/扫描流水线。（framework `Scanner::workflow::*` 已随 framework 退役删除，原"框架工作流层未被 UI 调用"之债随之失效，转化为"真实算子链尚未接入 UI"。）
2. **LEADSCANSeries 桩 include（已补齐，现可构建）**：`app/stubs/` 下 5 个桩头（LEADSCANSeries.h / CameraControl.h / camera_calib_workflow.h / laser_calib_workflow.h / scan_workflow.h）已补齐，`MainWindow.cpp:9-13` 的 `#include "stubs/*.h"` 不再悬空，scan_demo 全量可构建（原"悬空无法编译 / WIP"状态已消除）。**残留设计气味**：`m_integrateTestDialog` 声明为 `QWidget*`（MainWindow.h:104），槽内只赋过 `new ScannerWindow(...)`（:165），但标定/扫描槽在 6 处（:193/:198/:244/:249/:434/:437）将其 `static_cast<LEADSCANSeries*>` 使用——对话框存在时该 cast 恒为 UB；成员 `m_series`（MainWindow.h:106）声明后从未赋值使用（死成员）。
3. **硬编码路径**（共 4 处）：`D:/pointcloud_100M.ply`（:179）、`E:/workfold/20260509intergrate/calib_debug.log`（:185）、`E:/workfold/framework/build/JEAMMSCAN.stl`（:300）、`E:/workfold/framework/build/import_debug.log`（:813）（后三者为调试日志/中间产物直接 fopen）。
4. **单文件 1672 行**：UI 构建、槽逻辑、信息面板全堆一处，可考虑拆分（如 ui_builder / slots / info_panel）。
