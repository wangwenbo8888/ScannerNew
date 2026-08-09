# 02 · AppContext 装配（应用层组合根）

> `app/AppContext.h`（80 行）/ `AppContext.cpp`（95 行）。创建并拥有全部框架对象，是整个应用的**依赖注入根**。

## 职责定位
- **组合根（composition root）**：唯一一处 `new` 出所有框架层对象的地方。
- **拥有关系**：用 `std::unique_ptr` 持有全部对象，生命周期 = 整个应用。
- **注入方式**：通过 `WorkflowContext`（窄接口注入容器，框架 ADR 7.7）把依赖传给 Workflow；UI 通过 `AppContext*` 取用。

## 拥有的对象（按层）
| 层 | 对象 | 类型 |
|----|------|------|
| **Infra** | `eventBus_` | `infra::EventBus` |
| **Data** | `frameBuffer_` | `data::FrameBuffer`（**容量 60 帧**环形缓冲） |
| | `pointCloudBuffer_` | `data::PointCloudBuffer` |
| | `deviceStateCache_` | `data::DeviceStateCache` |
| | `calibStore_` | `data::CalibStore` |
| **Service** | `stateMachine_` | `service::StateMachine`（注入 EventBus） |
| | `paramManager_` | `service::ParameterManager` |
| | `faultHandler_` | `service::FaultHandler` |
| | `sessionService_` | `service::SessionService` |
| **HAL** | `camera_` | `device::CameraControl`（双目：左 0 / 右 1，右图 180°） |
| | `mcu_` | `device::MCUDriver`（115200 baud） |
| | `hwMonitor_` | `device::HardwareMonitor` |
| **Workflow** | `wfCtx_` | `workflow::WorkflowContext` |
| | `scanWf_` / `calibWf_` / `postWf_` | Scan / Calibration / PostProcess Workflow |

## `initialize()` 装配顺序（严格按层）
```
1. Infra   : EventBus
2. Data    : FrameBuffer(60) → PointCloudBuffer → DeviceStateCache → CalibStore
3. Service : StateMachine(EventBus) → ParameterManager → FaultHandler → SessionService
             └ FaultHandler.setEventBus + setStateMachine + start()
4. HAL     : CameraControl(camCfg) → MCUDriver(115200) → HardwareMonitor
             └ HardwareMonitor 注入 stateCache / eventBus / mcu / camera
5. WfCtx   : WorkflowContext 注入 10 项（frame/pointCloud/stateCache/calib/
             stateMachine/param/session/camera/mcu + eventBus）
6. Workflow: ScanWorkflow(wfCtx) → CalibrationWorkflow(wfCtx) → PostProcessWorkflow(wfCtx)
7. 启动监控: hwMonitor_->start(1000)   // 周期 1s 采设备状态
```

## 关键接线（手动 set 的依赖边）
- `FaultHandler` ← `EventBus` + `StateMachine`（聚合故障 → 转状态）。
- `HardwareMonitor` ← `DeviceStateCache` + `EventBus` + `MCU` + `Camera`（周期采集写缓存/发事件）。
- `WorkflowContext` ← 10 项依赖（统一注入给三条工作流）。

## `shutdown()` 逆序停止
```
hwMonitor_.stop() → scanWf_/calibWf_/postWf_.stop()
→ faultHandler_.stop()
→ camera_.stopAsyncCapture() + close()
→ mcu_.close()
```
> 注：析构函数也调 `shutdown()`，双重保护。

## 访问接口
全部对象通过 `xxx()` 裸指针 getter 暴露（如 `camera()`、`scanWorkflow()`、`eventBus()`），供 `MainWindow` 与工作流取用。**返回的是非所有权指针**，调用方不得 delete。

## 设计要点
- **唯一装配点**：所有依赖关系集中在此文件，便于审视全系统拓扑。
- **Data 层先于 Service/HAL**：缓存先就绪，供后续层回填（HAL→Data 依赖倒置 Sink）。
- **WorkflowContext 解耦**：三条工作流不各自找依赖，统一从 `wfCtx_` 取（窄接口）。

## ⚠ 框架工作流层整体未被 UI 调用（技术债）
`AppContext` 装配了框架的 `Scanner::workflow::{Scan,Calibration,PostProcess}Workflow`，但 `MainWindow` 的槽函数**一个都没调用它们**：标定走 `calibration::CameraCalibWorkflow`/`LaserCalibWorkflow`、扫描走 `calibration::ScanWorkflow`（均属 calibration 模块的独立实现，见 [03](03-MainWindow-UI.md)）。即**整套框架工作流层（`Scanner::workflow::*`）被装配却闲置**，UI 与框架工作流当前是两套并行路径。
