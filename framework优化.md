# framework/ 层优化记录

> 日期: 2026-08-12
> 操作: 删除 framework/ 共享层，所有代码归属到 11 个模块或 app/
> 结果: 项目结构从 `framework/ + modules/ + app/` 简化为 `modules/ + app/`

---

## 一、背景

原架构三层结构：
```
framework/    ← 共享基础设施（data/service/workflow/hal/infra/common）
modules/      ← 11 个功能模块（依赖 framework/）
app/          ← 应用入口（依赖 framework/ + modules/）
```

问题：framework/ 是承重墙，大量代码不属于任何单一模块，归属不清晰。

优化后两层结构：
```
app/          ← 应用入口 + 公共类型 + 工作流框架
modules/      ← 11 个功能模块（自包含，互不依赖或单向依赖）
```

---

## 二、framework/ 各子目录去向

### framework/common/ → app/ + 各模块（拆分）

| 文件 | 去向 | 说明 |
|------|------|------|
| `types.h` 主体 | **app/types.h** | Result/QualityFlag/Pose/FaultSeverity/FrameId 等 |
| `types.h` EventType/Event | **modules/07_session/EventTypes.h** | 事件类型，EventBus 用 |
| `types.h` DeviceState | **modules/08_devicemgmt/DeviceTypes.h** | 设备状态枚举 |
| `types.h` ScanMode | **modules/02_scanning/ScanConfig.h** | 已有 3 值版（原 types.h 是 2 值版） |
| `common.cpp` | 删除 | 空桩文件 |

### framework/data/ → 按归属拆分到 3 个模块

| 文件 | 去向 | 理由 |
|------|------|------|
| `FrameBuffer.h/.cpp` | **modules/08_devicemgmt/** | 帧缓冲，设备层 |
| `RingBuffer.h` | **modules/08_devicemgmt/** | FrameBuffer 的依赖 |
| `IFrameSink.h` | **modules/08_devicemgmt/** | 随 FrameBuffer |
| `DeviceStateCache.h/.cpp` | **modules/08_devicemgmt/** | 设备状态缓存 |
| `IDeviceStateSink.h` | **modules/08_devicemgmt/** | 随 DeviceStateCache |
| `CalibStore.h/.cpp` | **modules/01_calibration/** | 标定数据存储 |
| `PointCloudBuffer.h/.cpp` | **modules/06_fileio/** | 点云缓冲，数据管理 |
| `IPointCloudSink.h` | **modules/06_fileio/** | 随 PointCloudBuffer |
| `tests/test_ringbuffer.cpp` | 删除 | 依赖已迁移 |

### framework/service/ → 按归属拆分到 3 个模块

| 文件 | 去向 | 理由 |
|------|------|------|
| `StateMachine.h/.cpp` | **modules/07_session/** | 状态机，会话管理 |
| `SessionService.h/.cpp` | **modules/07_session/** | 会话服务 |
| `IState.h` | **modules/07_session/** | 随 StateMachine |
| `FaultHandler.h/.cpp` | **modules/10_observability/** | 故障处理，运维/可观测性 |
| `ParameterManager.h/.cpp` | **modules/06_fileio/** | 参数管理，数据管理 |

### framework/workflow/ → app/

| 文件 | 去向 | 理由 |
|------|------|------|
| `IWorkflow.h` | **app/** | 工作流接口，app 层装配 |
| `Pipeline.h/.cpp` | **app/** | 流水线框架基类 |
| `WorkflowContext.h/.cpp` | **app/** | DI 容器，app 层持有 |
| `tests/test_pipeline.cpp` | 删除 | 依赖已迁移 |

### framework/infra/ → modules/07_session/ + 删除

| 文件 | 去向 | 理由 |
|------|------|------|
| `EventBus.h/.cpp` | **modules/07_session/** | 事件总线，会话管理 |
| `Scheduler.h/.cpp` | 删除 | 死代码（仅测试引用） |
| `Watchdog.h/.cpp` | 删除 | 死代码（完全无引用） |

### framework/hal/ → modules/08_devicemgmt/

| 文件 | 去向 | 理由 |
|------|------|------|
| `IScannerCamera.h` | **modules/08_devicemgmt/** | 相机 HAL 接口，CameraControl 在 08 |
| `IMCU.h` | **modules/08_devicemgmt/** | MCU HAL 接口，MCUDriver 在 08 |

### framework/algorithm/ → 删除

| 文件 | 去向 | 理由 |
|------|------|------|
| `operator_convention.h` | 删除 | 死代码（无任何引用） |

### framework/crosscut/ → 删除

| 文件 | 去向 | 理由 |
|------|------|------|
| `IAuth.h` | 删除 | 死代码（无任何引用） |

### framework/ui/ → 删除

| 文件 | 去向 | 理由 |
|------|------|------|
| `IView.h` | 删除 | 死代码（无任何引用） |

### framework/tests/ → 删除

| 文件 | 去向 | 理由 |
|------|------|------|
| `test_integration.cpp` | 删除 | 依赖已删除的框架 |
| `test_smoke.cpp` | 删除 | 依赖已删除的框架 |

---

## 三、删除的 CMake 库目标

| 库目标 | 原内容 | 替代方案 |
|--------|--------|---------|
| `fw_common` | types.h | app/types.h（头文件，无需库） |
| `fw_data` | FrameBuffer/PointCloudBuffer/DeviceStateCache/CalibStore | 拆到 08/06/01 模块库 |
| `fw_service` | StateMachine/SessionService/FaultHandler/ParameterManager | 拆到 07/10/06 模块库 |
| `fw_workflow` | Pipeline/WorkflowContext + 工作流.cpp | 移到 app/，工作流.cpp 编入 scan_demo |
| `fw_hal` | IScannerCamera/IMCU | 移到 modules/08_devicemgmt/ |
| `fw_infra` | EventBus/Scheduler/Watchdog | EventBus 移到 07，其余删除 |
| `fw_algorithm` | operator_convention.h | 删除 |
| `fw_ui` | IView.h | 删除 |
| `fw_crosscut` | IAuth.h | 删除 |

---

## 四、优化后的项目结构

```
E:\workfold\framework\
├── app/                        ← 应用入口 + 公共类型 + 工作流框架
│   ├── types.h                 ← Result/QualityFlag/Pose（原 framework/common/）
│   ├── IWorkflow.h             ← 工作流接口（原 framework/workflow/）
│   ├── Pipeline.h/.cpp         ← 流水线框架（原 framework/workflow/）
│   ├── WorkflowContext.h/.cpp  ← DI 容器（原 framework/workflow/）
│   ├── main.cpp
│   ├── MainWindow.cpp/h
│   └── AppContext.cpp/h
│
├── modules/                    ← 11 个功能模块（自包含）
│   ├── 01_calibration/         ← 标定（+CalibStore 从 framework/data 迁入）
│   ├── 02_scanning/            ← 扫描（+ScanMode 从 framework/common 拆出）
│   ├── 03_rendering/           ← 渲染显示
│   ├── 04_postprocessing/      ← 后处理
│   ├── 06_fileio/              ← 文件/数据管理（+PointCloudBuffer +ParameterManager +IPointCloudSink）
│   ├── 07_session/             ← 会话管理（+EventBus +EventTypes +StateMachine +SessionService +IState）
│   ├── 08_devicemgmt/          ← 设备管理（+IScannerCamera +IMCU +FrameBuffer +DeviceStateCache +DeviceTypes +RingBuffer +IFrameSink +IDeviceStateSink）
│   ├── 09_operatorlib/         ← 算子库（84cpp + 17cu）
│   ├── 10_observability/       ← 可观测性（+FaultHandler）
│   └── 11_deploy/              ← 安装部署
│
├── docs/                       ← 文档
├── sdk/                        ← 第三方 SDK
├── cmake/                      ← CMake 辅助脚本
└── CMakeLists.txt              ← 顶层（无 framework 子目录）
```

---

## 五、模块间依赖关系（优化后）

```
app/ ← 所有模块
  ↓
modules/07_session (EventBus + 状态机)
  ↑
modules/08_devicemgmt (相机 + MCU + 帧缓冲 + 设备状态) ← 依赖 07
  ↑
modules/06_fileio (点云缓冲 + 参数管理 + 文件IO) ← 依赖 OSG
modules/10_observability (故障处理) ← 依赖 07
modules/09_operatorlib (算子库) ← 独立
modules/01~04 (工作流实现) ← 通过 app 编译
```

### 依赖原则
- 模块间**单向依赖**，无循环
- app/ 持有所有模块的指针（AppContext 装配）
- 模块不依赖 app/（仅头文件 types.h 例外，通过项目根 include 路径访问）
