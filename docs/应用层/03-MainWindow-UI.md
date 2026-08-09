# 03 · MainWindow 主窗口 UI

> `app/MainWindow.h`（122 行）/ `MainWindow.cpp`（1672 行）。无边框主窗口 "LeadScan K2"，承担全部 UI 布局与操作入口。

## 总体
- 基类 `QMainWindow`，**无边框**（`Qt::FramelessWindowHint`），标题 "LeadScan K2"。
- 持 `AppContext* m_appCtx`，按需取用框架对象（不 new）。
- 嵌入 `OSGWidget`（模块3）做 3D 点云渲染视图。

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
| 182–403 | `onCalibDeviceClicked()` — 标定流程（220 行，相机标定采集） |
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
| `onIntegrateTestClicked` | 工具栏按钮 | 打开 `ScannerWindow(m_appCtx, this)`（集成测试） |
| `onReloadPointCloud` | 按钮 | `m_3dView->loadTestDataFromPLY("D:/pointcloud_100M.ply")` |
| `onCalibDeviceClicked` | 按钮 | 弹 `CalibDialog` → `cameraCalibClicked` 信号触发相机标定：经 `LEADSCANSeries::getCameraControl()` 取相机，循环采 15 帧 `cam->GetScannerImages()` |
| `onScanClicked` | 按钮 | 隐藏标定板/彩条 → 清空 3D 场景 → 经 `LEADSCANSeries` 取相机 → **`calibration::ScanWorkflow().processFrameToCloud()`** → 结果点云载入 `m_3dView` |

## 信息面板（系统监控显示）
- `startInfoTimer()`：`QTimer` 周期触发 `updateInfoSection()`。
- `updateInfoSection()` 读 `m_appCtx->deviceStateCache()` / `pointCloudBuffer()` / `frameBuffer()`，刷新：连接状态、点云数、FPS、温度、CPU、内存。
- 此处是**模块8 HardwareMonitor 采集的数据流向 UI 的出口**。

## 图标系统
- 大量 SVG 图标，三态：**黑 / 红 / 灰**（如 `下一步-灰-13.svg` / `-红-13` / `-黑-13`），按按钮状态切换。
- `renderSvg()` 把 SVG 渲染为 QPixmap；按钮工厂 `createIconButton/createNavButton/createToolButton/createSelectionButton` 统一生成。

## ⚠ 技术债（如实记录）
1. **框架工作流层整体闲置**：`onScanClicked` 用 `calibration::ScanWorkflow`、`onCalibDeviceClicked` 用 `calibration::{Camera,Laser}CalibWorkflow`（均 calibration 模块内），**完全没用** AppContext 装配的框架 `Scanner::workflow::*`。
2. **LEADSCANSeries 悬空 include（当前无法编译）**：`MainWindow.cpp:9` `#include "stubs/LEADSCANSeries.h"`，但该头文件**全仓库不存在**（glob 无结果）→ 此文件目前不可编译，属 WIP。`m_integrateTestDialog`（实为 `ScannerWindow*`，而 `ScannerWindow : public QMainWindow`）被 `static_cast<LEADSCANSeries*>` 也是 UB。
3. **硬编码路径**：`D:/pointcloud_100M.ply`、`E:/workfold/20260509intergrate/calib_debug.log`（调试日志直接 fopen）。
4. **单文件 1672 行**：UI 构建、槽逻辑、信息面板全堆一处，可考虑拆分（如 ui_builder / slots / info_panel）。
