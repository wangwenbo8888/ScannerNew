# JEAMMWARE 软件架构骨架设计

> **日期**: 2026-07-05
> **状态**: 完整草案，待整体评审
> **工程**: `E:\JEAMMWARE260705`（3D 扫描仪交付级产品软件）
> **架构依据**: `E:\3DSCANNER260622\docs\plans\2026-07-05-交付级框架整体设计-design.md`（01.4，5 层 + 11 模块）
> **环境依据**: `E:\3DSCANNER260622\环境配置汇总.md` + `E:\3DSCANNER260622\CMakeLists.txt`（同一台机、同一套工具链/依赖）
> **本次范围**: **纯骨架**（framework 5 层契约桩 + modules 11 空桩 + CMake + app 占位）。算子迁入、各层实现留后续。

---

## 1. 项目定位与环境继承

### 1.1 定位
- **JEAMMWARE** = 3D 扫描仪**交付级产品软件**（上位机软件域）
- 架构：01.4 的 5 层（UI/Workflow/Service∥Algorithm/Data/HAL）+ 侧边设施 + 横切 + 部署 wrapper + SDK 接入端
- 与 `3DSCANNER260622`（算子研发工程）关系：**后续迁移其算子源码**到 `modules/09_operatorlib/`（本次不做）
- 全系统视角见 `3DSCANNER260622/docs/plans/2026-07-05-全系统多域架构总图-design.md`（JEAMMWARE = 其中「上位机软件域」）

### 1.2 环境（继承 3DSCANNER，同一台机）

| 类别 | 项 | 版本/路径 | 本次骨架是否需要 |
|------|----|----------|----------------|
| 工具链 | VS 2022 / MSVC v144 (14.44) | `C:\Program Files\Microsoft Visual Studio\2022\Community\` | ✅ 必须 |
| 工具链 | CMake ≥ 3.24 | `C:\Program Files\CMake\` | ✅ 必须 |
| 工具链 | Ninja（可选） | `C:\devlibs\ninja\` | 可选 |
| 语言 | **C++20**（`/std:c++20`） | — | ✅ 必须 |
| GPU | CUDA 12.6 + cuDNN 9.21 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\` | ⚠️ 可选（骨架桩不用，BUILD_CUDA 开但桩不依赖） |
| GPU | 架构 sm_75;86;87（RTX5000/sm_86/Orin） | — | 配置就绪 |
| 视觉 | OpenCV 4.13(+CUDA+Contrib) | `C:\opencv-cuda-4.13.0\`（OpenCV_DIR） | ⚠️ 可选（骨架不用，算子迁入后用） |
| 线代 | Eigen 3.4.1 (header-only) | `C:\devlibs\eigen-3.4.1-install\` | ⚠️ 可选（同上） |
| 日志 | spdlog 1.15 | FetchContent 兜底 | ✅ 用（骨架即可有日志） |
| 配置 | nlohmann_json 3.11.3 | FetchContent 兜底 | ✅ 用 |
| 测试 | GTest 1.14 | FetchContent（BUILD_TESTS） | ✅ 用 |
| 优化 | Ceres 2.2.0（GBA 用） | FetchContent（Eigen 稀疏后端） | ⚠️ 算子迁入后用 |
| UI | Qt 5.15.2 | `C:\devlibs\Qt-5.15.2-msvc2019_64\` | ❌ 本次不用（UI 层桩，BUILD_UI=OFF） |
| 其他 | PCL/Open3D/libigl/CGAL/Boost/OSG/OCCT | `C:\devlibs\...` | ❌ 后处理/UI 等后续阶段用 |

**关键**：骨架阶段只硬依赖 spdlog + nlohmann_json + GTest（都 FetchContent，零外部装），**OpenCV/CUDA/Eigen 配置就绪但桩不链接**——保证骨架快速构建验证架构形态；算子迁入/各层实现时再按需 `find_package`。

---

## 2. 目录组织（方案 C：framework 层契约 + modules 模块实现）

```
E:\JEAMMWARE260705\
├── .gitignore
├── AGENTS.md                          # 项目根索引（仿 3DSCANNER AGENTS.md）
├── README.md
├── CMakeLists.txt                     # 顶层 CMake
├── docs/
│   └── architecture/
│       └── software-architecture-skeleton-design.md   # 本设计稿
├── framework/                         # ★ 架构骨架：层契约（抽象基类/接口）
│   ├── CMakeLists.txt
│   ├── common/                        # 跨层共享类型
│   │   ├── types.h                    # Frame/FrameResult/PipelineFrame/Point3d...
│   │   ├── quality_flag.h             # QualityFlag 枚举
│   │   └── CMakeLists.txt             # add_library(fw_common STATIC)
│   ├── ui/                            # UI 层（接入端 A）
│   │   ├── IView.h                    # 视图基类（点云视图/编辑界面/...）
│   │   ├── IUIController.h            # UI 控制器
│   │   └── CMakeLists.txt             # add_library(fw_ui INTERFACE)
│   ├── workflow/                      # Workflow 层（编排）
│   │   ├── IWorkflow.h                # execute/stop/pause/resume/getStatus
│   │   ├── WorkflowContext.h          # 统一入口（窄接口 ADR 7.7）
│   │   ├── Pipeline.h                 # Stage 组合/队列/背压 基类
│   │   ├── Stage.h
│   │   └── CMakeLists.txt             # add_library(fw_workflow INTERFACE)
│   ├── service/                       # Service 层（状态/业务）
│   │   ├── IService.h
│   │   ├── StateMachine.h             # 系统状态机基类
│   │   ├── FaultHandler.h
│   │   └── CMakeLists.txt             # add_library(fw_service INTERFACE)
│   ├── algorithm/                     # Algorithm 层（算子契约，对齐算子规范 v1.9）
│   │   ├── IOperator.h                # Execute/Destroy/Warmup + Result 契约
│   │   ├── AlgorithmRegistry.h        # 算法注册表
│   │   ├── OperatorResult.h           # success/qualityFlag/message
│   │   └── CMakeLists.txt             # add_library(fw_algorithm INTERFACE)
│   ├── data/                          # Data 层
│   │   ├── IDataStore.h
│   │   ├── DataContext.h              # Data 访问封装
│   │   ├── IFrameSink.h               # HAL 回填接口（依赖倒置）
│   │   ├── IDeviceStateSink.h
│   │   ├── WorkflowArtifactStore.h    # 产物交接（ADR 7.10）
│   │   └── CMakeLists.txt             # add_library(fw_data INTERFACE)
│   ├── hal/                           # HAL 层（通讯链边界适配器）
│   │   ├── ICamera.h                  # ICamera1/2
│   │   ├── IMCU.h
│   │   ├── IPlatform.h
│   │   └── CMakeLists.txt             # add_library(fw_hal INTERFACE)
│   ├── infra/                         # 侧边基础设施
│   │   ├── EventBus.h                 # 事件总线（控制/通知）
│   │   ├── GpuRuntime.h               # ICudaContext + ResourceLifecycleManager
│   │   ├── Scheduler.h                # 线程池/CPU 亲和/CUDA Stream
│   │   ├── Watchdog.h                 # Stage 心跳监督
│   │   └── CMakeLists.txt             # add_library(fw_infra INTERFACE)
│   └── crosscut/                      # 横切关注点
│       ├── IAuth.h                    # 用户权限/鉴权
│       ├── ILogger.h                  # 统一日志（spdlog 后端）
│       ├── IPerfMonitor.h             # 性能监控
│       ├── ICrashHandler.h            # 崩溃捕获（minidump）
│       ├── IConfig.h                  # 配置管理
│       └── CMakeLists.txt             # add_library(fw_crosscut INTERFACE)
├── modules/                           # 11 业务模块（空桩，后续填实现）
│   ├── CMakeLists.txt
│   ├── 01_calibration/                # 模块1 标定
│   ├── 02_scanning/                   # 模块2 扫描（双模式）
│   ├── 03_rendering/                  # 模块3 渲染显示
│   ├── 04_postprocess/                # 模块4 后处理
│   ├── 05_editing/                    # 模块5 编辑（含撤销重做）
│   ├── 06_fileio/                     # 模块6 文件管理
│   ├── 07_session/                    # 模块7 会话管理
│   ├── 08_devicemgmt/                 # 模块8 设备管理（含许可证）
│   ├── 09_operatorlib/                # 模块9 算子库 ← 后续迁入 3DSCANNER 算子
│   ├── 10_observability/              # 模块10 可观测性/运维
│   └── 11_deploy/                     # 模块11 安装部署+首配
├── sdk/                               # 接入端 B（二次开发 API 门面）
│   ├── IScannerSDK.h
│   └── CMakeLists.txt                 # add_library(jmw_sdk INTERFACE)
└── app/                               # exe 入口（链接各层）
    ├── main.cpp                       # 占位 main（打印 banner，验证构建链）
    └── CMakeLists.txt                 # add_executable(jeammware)
```

**命名约定**：
- 根命名空间 `jmw::`，子命名空间按层：`jmw::ui` / `jmw::workflow` / `jmw::service` / `jmw::algorithm` / `jmw::data` / `jmw::hal` / `jmw::infra` / `jmw::crosscut` / `jmw::sdk`
- 库目标：`fw_<layer>`（framework 层契约，INTERFACE 库为主）/ `mod_<name>`（modules 实现，STATIC 库）/ `jmw_sdk` / `jeammware`(exe)
- 头文件守卫/pragma once 统一

---

## 3. 各层接口桩契约（framework/）

> 桩 = 纯抽象基类（接口）+ 共享类型。无实现，定义层边界契约。对齐 07-02 ADR + 算子规范 v1.9。

### 3.1 framework/common（跨层共享类型）
```cpp
namespace jmw {
enum class QualityFlag { Normal, Degraded, Warning, Fault };
struct Frame {                                  // HAL 产帧 → Data
    std::shared_ptr<cv::Mat> leftGray, rightGray;
    uint64_t frameId = 0, timestamp = 0;
    double temperature = 25.0;
};
struct FrameResult {                            // Workflow 输出
    uint64_t frameId = 0;
    // R/T + marker/laser points（迁移算子后细化）
};
}
```

### 3.2 framework/algorithm（算子契约，对齐算子规范 v1.9 §2）
```cpp
namespace jmw::algorithm {
struct OperatorResult {
    bool success = false;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::string message;
};
class IOperator {                               // 三元组的 Operator 抽象
public:
    virtual ~IOperator() = default;
    virtual OperatorResult Execute() = 0;       // CUDA 算子末参 Stream&（迁入后细化）
    virtual void Destroy() = 0;                 // idempotent
    virtual void Warmup() = 0;                  // 预分配
};
class AlgorithmRegistry {                       // 编译期类型安全注册（ADR 7.1）
public:
    template<typename T> void registerOperator(const std::string& key);
    IOperator* get(const std::string& key);
};
}
```

### 3.3 framework/workflow（编排，07-02 §3.1/6.1）
```cpp
namespace jmw::workflow {
enum class WorkflowStatus { Idle, Running, Paused, Stopped, Faulted };
class IWorkflow {
public:
    virtual ~IWorkflow() = default;
    virtual void execute() = 0;
    virtual void stop() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual WorkflowStatus getStatus() const = 0;
};
class WorkflowContext {                         // 统一入口（窄接口注入 ADR 7.7）
    // 聚合 DataContext / Service / AlgorithmRegistry / EventBus
    // 对外暴露窄角色接口 IScanContext / ICalibContext / ...
};
class Pipeline { /* Stage 组合 / 有界队列 / 背压 / 丢帧（ADR 7.2）*/ };
class Stage { /* 流水线最小执行单元 */ };
}
```

### 3.4 framework/service（状态/业务）
```cpp
namespace jmw::service {
class IService {
public:
    virtual ~IService() = default;
    virtual const char* name() const = 0;
};
class StateMachine {  /* 待机/标定/扫描/后处理/故障 状态转移 */ };
class FaultHandler {  /* HardwareFault/AlgorithmFault → FaultOccurred 译码（ADR 7.9）*/ };
}
```

### 3.5 framework/data（数据/持久化，依赖倒置）
```cpp
namespace jmw::data {
class IFrameSink {                              // HAL 回填帧（依赖倒置）
public:
    virtual ~IFrameSink() = default;
    virtual void onFrame(const Frame&) = 0;
};
class IDeviceStateSink { virtual void onState(const DeviceState&) = 0; };
class IDataStore {  /* 帧缓冲/点云/标定/文件IO 访问封装 */ };
class DataContext {  /* 隐藏线程同步的读写封装 */ };
class WorkflowArtifactStore {  /* 跨工作流产物交接（ADR 7.10）*/ };
}
```

### 3.6 framework/hal（通讯链边界适配器，07-02 §2.3）
```cpp
namespace jmw::hal {
class ICamera {  /* receiveFrame/getTemperature/configure；不触发 */ };
class IMCU {  /* triggerCapture/controlLaser/controlLight/setLed/readButton/readTemperature */ };
class IPlatform {  /* getPlatformType/getCudaArch/getGpuCount；编译时多态 */ };
}
```

### 3.7 framework/infra（侧边设施）
```cpp
namespace jmw::infra {
class EventBus {  /* subscribe/publish；控制/通知通道，不携带热路径载荷 */ };
class GpuRuntime {  /* ICudaContext + ResourceLifecycleManager（ADR 7.11）*/ };
class Scheduler {  /* 线程池/P-E 核亲和/CUDA Stream 分配 */ };
class Watchdog {  /* Stage 心跳/队列趋势监督（独立于 EventBus 健康监控）*/ };
}
```

### 3.8 framework/crosscut（横切）
```cpp
namespace jmw::crosscut {
class IAuth {  /* 角色/登录/操作授权；管 UI+SDK 接入 */ };
class ILogger {  /* 统一日志（spdlog 后端，分级/轮转）*/ };
class IPerfMonitor {  /* 延迟/帧率/计数器 */ };
class ICrashHandler {  /* minidump/crash dump */ };
class IConfig {  /* 配置管理（热加载）*/ };
}
```

---

## 4. 模块桩（modules/）

每个模块目录含：
- `CMakeLists.txt`：`add_library(mod_<name> STATIC "")`，链接所需 `fw_<layer>`
- `<name>.h`：模块对外接口（空/最小声明）
- `<name>.cpp`：空实现（占位）
- `README.md`：模块职责一句话 + 主层 + 跨层（对齐 01.4 §3 归位表）

模块→层依赖（按 01.4 归位表）：
| 模块 | 链接的 framework 层 |
|------|-------------------|
| 01_calibration | fw_workflow fw_algorithm fw_data |
| 02_scanning | fw_workflow fw_algorithm fw_data fw_infra |
| 03_rendering | fw_ui fw_algorithm fw_data |
| 04_postprocess | fw_workflow fw_algorithm fw_data |
| 05_editing | fw_ui fw_service fw_data |
| 06_fileio | fw_data fw_service |
| 07_session | fw_service fw_workflow fw_ui fw_data |
| 08_devicemgmt | fw_hal fw_service fw_ui fw_data fw_crosscut |
| 09_operatorlib | fw_algorithm（后续迁入算子） |
| 10_observability | fw_crosscut |
| 11_deploy | （部署期，独立） |

---

## 5. CMake 结构（继承 3DSCANNER 模式）

### 5.1 顶层 `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.24)
project(JEAMMWARE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")  # /MD 统一
endif()

# === 选项 ===
option(BUILD_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_UI "Build Qt UI layer (deferred)" OFF)

# === CUDA（配置就绪，骨架桩不强制用）===
if(BUILD_CUDA)
    set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
    if(MSVC) string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8") endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
endif()

# === 第三方路径（CACHE，骨架阶段不强制 find，算子迁入后启用）===
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")
# find_package(OpenCV/Eigen) 延后到 modules/09_operatorlib 迁入时

# === 轻量依赖（FetchContent，骨架即用）===
include(FetchContent)
find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
    FetchContent_MakeAvailable(spdlog)
endif()
find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(nlohmann_json)
endif()
if(BUILD_TESTS)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# === 子工程 ===
add_subdirectory(framework)
add_subdirectory(modules)
add_subdirectory(sdk)
add_subdirectory(app)
```

### 5.2 各层 CMake 模式
- `framework/<layer>/CMakeLists.txt`：`add_library(fw_<layer> INTERFACE)`（纯头文件契约，INTERFACE 库）→ 零编译开销；`fw_common` 为 STATIC（含 types.cpp 若需）
- `modules/<n>_<name>/CMakeLists.txt`：`add_library(mod_<name> STATIC "")`，`target_link_libraries(mod_<name> PUBLIC fw_<layers...>)`
- `app/CMakeLists.txt`：`add_executable(jeammware main.cpp)`，链接 `mod_*` + `jmw_sdk`

---

## 6. 验证标准（骨架完成判据）

```powershell
cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64
cmake --build E:\JEAMMWARE260705\build --config Release
& E:\JEAMMWARE260705\build\app\Release\jeammware.exe   # 占位 main
ctest --test-dir E:\JEAMMWARE260705\build -C Release --output-on-failure
```

**完成判据**：
- [ ] `cmake configure` 成功（FetchContent 拉取 spdlog/json/gtest）
- [ ] `cmake build` 成功：8 个 `fw_*`（INTERFACE）+ 11 个 `mod_*`（STATIC，空）+ `jmw_sdk` + `jeammware.exe` 全部产出
- [ ] `jeammware.exe` 运行打印 banner（`JEAMMWARE v0.1.0 skeleton — 5 layers / 11 modules`）退出码 0
- [ ] 若 BUILD_TESTS：空 GTest 通过（如 `framework/tests/test_smoke.cpp` 验证头文件可 include）
- [ ] 每个层契约头文件可被独立 `#include`（无交叉依赖污染）

---

## 7. Git 工程化

- `git init`（已执行）
- `.gitignore`：`build/` `out/` `*.user` `.vs/` `compile_commands.json`（可选保留）+ IDE 临时
- `README.md`：项目一句话 + 构建/测试命令 + 架构图指引（链 01.4）
- `AGENTS.md`：项目根索引（仿 3DSCANNER AGENTS.md，给 AI/人查阅；含环境/构建/架构骨架说明）
- 大文件 LFS（后续点云/图像）：`.gitattributes` 配 `*.ply *.pcd *.stl *.tif *.bmp`

---

## 8. 后续（骨架之后）

1. **算子迁入**：`3DSCANNER260622` 的 core/calibration/scanning 算子源码 → `modules/09_operatorlib/`，对齐 `framework/algorithm/IOperator` 契约；启用 `find_package(OpenCV/Eigen)` + CUDA 编译
2. **各层实现**：按 `工程缺口分析与出厂路线图` 的 8 切片逐层填实现（切片 2 真 HAL / 切片 3 显示 / ...）
3. **垂直切片**：选最小垂直切片（如 smoke 链）端到端打通，验证架构形态
4. **跨工程同步**：JEAMMWARE 与 3DSCANNER 在算子契约层保持一致（算子规范 v1.9）；3DSCANNER 后续仅作算法研发沙盒，产品代码进 JEAMMWARE

---

## 9. 待决策点

| # | 决策 | 本次默认 | 备注 |
|---|------|---------|------|
| D1 | 命名空间 | `jmw::` | 备选 `scanner::` / `jwear::` |
| D2 | framework 层库类型 | INTERFACE（纯头契约） | 若层需 .cpp 则改 STATIC（如 fw_common） |
| D3 | UI 框架 | Qt 5.15.2（BUILD_UI=OFF 本次） | 环境已装；OSG 故障不用 |
| D4 | 算子迁入时机 | 本次不做 | 骨架验证后单独一步 |
| D5 | 是否含 v2 数据流主轴实验 | 不含（01.4 正式） | v2 留 3DSCANNER 作参考研究 |
