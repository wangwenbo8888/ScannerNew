# 02 · AppContext 装配（应用层组合根）

> `app/AppContext.h`（80 行）/ `app/AppContext.cpp`（95 行）。创建并拥有**全部系统对象**（Infra/Data/Service/HAL/Workflow 五类），是整个应用的**依赖注入根**。`framework/` 退役后，这些对象不再集中于一处，而是按逻辑层**物理散布于 `modules/NN_*/`、`base/`、`app/`**，但仍由 AppContext 统一 `new` + `unique_ptr` 持有；命名空间仍沿用 `Scanner::{infra,data,service,device,workflow}::*` 的逻辑分层。

## 职责定位
- **组合根（composition root）**：唯一一处 `new` 出全部系统对象的地方（`framework/` 已删，对象现按逻辑层散布于 `modules/NN` + `base/` + `app/`）。
- **拥有关系**：用 `std::unique_ptr` 持有全部对象，生命周期 = 整个应用。
- **注入方式**：通过 `WorkflowContext`（现位于 `app/` 的 DI 聚合器）把依赖传给 Workflow；UI 通过 `AppContext*` 取用。

## 拥有的对象（按逻辑层，附物理位置）
| 层 | 对象 | 类型 | 物理位置 |
|----|------|------|----------|
| **Infra** | `eventBus_` | `Scanner::infra::EventBus` | `base/` |
| **Data** | `frameBuffer_` | `Scanner::data::FrameBuffer`（**容量 60 帧**环形缓冲） | `modules/06_fileio` |
| | `pointCloudBuffer_` | `Scanner::data::PointCloudBuffer` | `modules/06_fileio` |
| | `deviceStateCache_` | `Scanner::data::DeviceStateCache` | `modules/06_fileio` |
| | `calibStore_` | `Scanner::data::CalibStore` | `modules/01_calibration` |
| **Service** | `stateMachine_` | `Scanner::service::StateMachine`（注入 EventBus） | `modules/07_session` |
| | `paramManager_` | `Scanner::service::ParameterManager` | `modules/06_fileio` |
| | `faultHandler_` | `Scanner::service::FaultHandler` | `modules/10_observability` |
| | `sessionService_` | `Scanner::service::SessionService` | `modules/07_session` |
| **HAL** | `camera_` | `Scanner::device::CameraControl`（双目：左 0 / 右 1，右图 180°） | `modules/08_devicemgmt` |
| | `mcu_` | `Scanner::device::MCUDriver`（115200 baud） | `modules/08_devicemgmt` |
| | `hwMonitor_` | `Scanner::device::HardwareMonitor` | `modules/08_devicemgmt` |
| **Workflow** | `wfCtx_` | `Scanner::workflow::WorkflowContext` | `app/`（本地，DI 聚合器） |
| | `scanWf_` | `Scanner::workflow::ScanWorkflow` | `modules/02_scanning` |
| | `calibWf_` | `Scanner::workflow::CalibrationWorkflow` | `modules/01_calibration` |
| | `postWf_` | `Scanner::workflow::PostProcessWorkflow` | `modules/04_postprocessing` |

> **Include 拓扑**（`AppContext.cpp`，对应上表迁移）：Data/Service 类已扁平化（`FrameBuffer.h`/`StateMachine.h`…，经 include 目录解析到各 `modules/NN`）；`EventBus` 走 `base/EventBus.h`；HAL 三件走显式 `modules/08_devicemgmt/*.h`；`WorkflowContext.h` 为 `app/` 本地相对引用；三条 Workflow 类扁平 include。

## `initialize()` 装配顺序（严格按层）
```
1. Infra   : EventBus
2. Data    : FrameBuffer(60) → PointCloudBuffer → DeviceStateCache → CalibStore
3. Service : StateMachine(EventBus) → ParameterManager → FaultHandler → SessionService
             └ FaultHandler.setEventBus + setStateMachine + start()
4. HAL     : CameraControl(camCfg) → MCUDriver(115200) → HardwareMonitor
             └ HardwareMonitor 注入 stateCache / eventBus / mcu / camera
5. WfCtx   : WorkflowContext（app/ 本地）注入 10 项（frame/pointCloud/stateCache/calib/
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
- **WorkflowContext 解耦**：三条工作流不各自找依赖，统一从 `wfCtx_` 取（窄接口）。`WorkflowContext` 现为 `app/` 本地的 DI 聚合器——**de-ctx/retire 未做**（保留作聚合器是务实取舍），故它仍是 AppContext 的本地一等成员，而非可拆除的过渡件。
- **逻辑分层 × 物理散布**：`framework/` 退役后对象按业务模块物理拆分（Data 多在 06、Service 拆到 06/07/10、HAL 在 08），但 `Scanner::{infra,data,service,device,workflow}` 命名空间分层保留——AppContext 仍按"逻辑层"装配与接线，物理位置不影响装配拓扑。

## ⚠ 框架工作流层整体未被 UI 调用（技术债）
`AppContext` 装配了 `Scanner::workflow::{Scan,Calibration,PostProcess}Workflow`（物理上现分布于 `modules/01·02·04`，原属 `framework/workflow`），但 `MainWindow` 的槽函数**一个都没调用它们**：标定走 `calibration::CameraCalibWorkflow`/`LaserCalibWorkflow`、扫描走 `calibration::ScanWorkflow`（均属 calibration 模块的独立实现，见 [03](03-MainWindow-UI.md)）。即**整套 `Scanner::workflow::*` 工作流层被装配却闲置**，UI 与框架工作流当前是两套并行路径。`framework/` 删除并未消除此债——工作流类迁入业务模块后仍保持 `Scanner::workflow::` 命名空间，且仍不被 UI 调用。
