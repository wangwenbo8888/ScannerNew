# JEAMMWARE 软件架构骨架设计

> **日期**: 2026-07-05
> **状态**: 完整草案 v2（经对照 01.4 审查，应用 6 条决策：U1/G1/G2/G3/G4 采纳，G5 改为「算子纯函数不强制基类派生」）
> **工程**: `E:\JEAMMWARE260705`（3D 扫描仪交付级产品软件）
> **架构依据**: `E:\3DSCANNER260622\docs\plans\2026-07-05-交付级框架整体设计-design.md`（01.4，5 层 + 11 模块）
> **环境依据**: `E:\3DSCANNER260622\环境配置汇总.md` + `E:\3DSCANNER260622\CMakeLists.txt`
> **本次范围**: **纯骨架**（framework 层契约桩 + modules 空桩 + CMake + app 占位）。算子迁入、各层实现留后续。

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
| GPU | CUDA 12.6 + cuDNN 9.21 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\` | ⚠️ 可选（骨架桩不用） |
| GPU | 架构 sm_75;86;87 | — | 配置就绪 |
| 视觉 | OpenCV 4.13(+CUDA+Contrib) | `C:\opencv-cuda-4.13.0\`（OpenCV_DIR） | ⚠️ 可选（算子迁入后用） |
| 线代 | Eigen 3.4.1 | `C:\devlibs\eigen-3.4.1-install\` | ⚠️ 可选 |
| 日志 | spdlog 1.15 | FetchContent 兜底 | ✅ 用 |
| 配置 | nlohmann_json 3.11.3 | FetchContent 兜底 | ✅ 用 |
| 测试 | GTest 1.14 | FetchContent（BUILD_TESTS） | ✅ 用 |
| UI | Qt 5.15.2 | `C:\devlibs\Qt-5.15.2-msvc2019_64\` | ❌ 本次不用（BUILD_UI=OFF） |
| 其他 | PCL/Open3D/libigl/CGAL/Boost/OCCT | `C:\devlibs\...` | ❌ 后续阶段用 |

**关键**：骨架阶段只硬依赖 spdlog + nlohmann_json + GTest（FetchContent），**OpenCV/CUDA/Eigen 配置就绪但桩不链接**——保证骨架快速构建验证架构形态。

---

## 2. 目录组织（方案 C：framework 层契约 + modules 模块实现）

```
E:\JEAMMWARE260705\
├── .gitignore
├── AGENTS.md                          # 项目根索引
├── README.md
├── CMakeLists.txt                     # 顶层 CMake
├── docs/architecture/                 # 本设计稿所在
├── framework/                         # ★ 架构骨架：层契约（接口/抽象/共享类型）
│   ├── CMakeLists.txt
│   ├── common/                        # 跨层共享类型（非第 6 层，见 §3.0 说明）
│   │   ├── types.h                    # Frame/FrameResult/PipelineFrame...
│   │   ├── quality_flag.h             # QualityFlag 枚举（跨算子共用）
│   │   └── CMakeLists.txt
│   ├── ui/                            # UI 层（接入端 A）
│   │   ├── IView.h                    # 含 ADR 7.8 IPointCloudReadView 预留注释
│   │   ├── IUIController.h
│   │   └── CMakeLists.txt
│   ├── workflow/                      # Workflow 层
│   │   ├── IWorkflow.h
│   │   ├── WorkflowContext.h          # 含 ADR 7.7 窄角色接口预留注释
│   │   ├── Pipeline.h  Stage.h
│   │   └── CMakeLists.txt
│   ├── service/                       # Service 层
│   │   ├── IService.h  StateMachine.h  FaultHandler.h
│   │   └── CMakeLists.txt
│   ├── algorithm/                     # Algorithm 层（算子契约 = 纯函数约定，非基类，见 §3.2）
│   │   ├── operator_convention.h      # 文档化约定（无 IOperator 基类）
│   │   ├── AlgorithmRegistry.h        # 概念性注册（模板，集成时定）
│   │   └── CMakeLists.txt
│   ├── data/                          # Data 层
│   │   ├── IDataStore.h  DataContext.h
│   │   ├── IFrameSink.h  IDeviceStateSink.h   # 含 ADR 7.6 FusionStateHandle 预留注释
│   │   ├── WorkflowArtifactStore.h    # ADR 7.10
│   │   └── CMakeLists.txt
│   ├── hal/                           # HAL 层（通讯链边界适配器）
│   │   ├── ICamera.h  IMCU.h  IPlatform.h
│   │   └── CMakeLists.txt
│   ├── infra/                         # 侧边设施
│   │   ├── EventBus.h                 # 含 ADR 7.9 AlgorithmFault 事件预留注释
│   │   ├── GpuRuntime.h  Scheduler.h  Watchdog.h
│   │   └── CMakeLists.txt
│   └── crosscut/                      # 横切（含框架结构元素的实现归属，见 §3.9）
│       ├── IAuth.h  ILogger.h  IPerfMonitor.h  ICrashHandler.h  IConfig.h
│       └── CMakeLists.txt
├── modules/                           # 11 业务模块（空桩）
│   ├── 01_calibration/ ... 11_deploy/   # 各含 README 记 01.4 职责
│   └── 04_postprocess/README.md        # 记后处理七阶段（§3.1 模块4）
├── sdk/                               # 接入端 B（SDK 实现归属，见 §3.9）
│   ├── IScannerSDK.h
│   └── CMakeLists.txt
└── app/                               # exe 入口
    ├── main.cpp
    └── CMakeLists.txt
```

**命名约定**：根命名空间 `jmw::`，子命名空间按层（`jmw::ui` / `jmw::workflow` / ... / `jmw::crosscut` / `jmw::sdk`）；库目标 `fw_<layer>` / `mod_<name>` / `jmw_sdk` / `jeammware`。

---

## 3. 各层接口桩契约（framework/）

> 桩 = 抽象/接口 + 共享类型 + ADR 预留注释。无实现。

### 3.0 framework/common 定位说明（G3 决策）
`fw_common` **不是第 6 个架构层**，是「跨层共享类型集中存放」（仿 3DSCANNER `core/common`）。对应 01.4 里隐含的 Data 共享类型 + 跨层契约类型。`Frame` 由 HAL 产、Workflow 消费——天生跨层，归 Data 不合适，故集中放此。

### 3.1 framework/common（跨层共享类型）
```cpp
namespace jmw {
enum class QualityFlag { Normal, Degraded, Warning, Fault };
struct Frame { std::shared_ptr<cv::Mat> leftGray, rightGray; uint64_t frameId=0, timestamp=0; double temperature=25.0; };
struct FrameResult { uint64_t frameId=0; /* 迁入算子后细化 */ };
}
```

### 3.2 framework/algorithm（算子契约 = 纯函数约定，G5 决策）⭐ 重要
**算子是纯函数，不强制基类派生**。framework/algorithm 只提供：
- **共享 `QualityFlag`**（放 common，跨算子 Result 用）
- **文档化约定**（`operator_convention.h`）：每个算子是独立类型/纯函数，提供 `Execute/Destroy/Warmup`，返回**自带 Result**（含 success/qualityFlag/message）。**不强制继承 IOperator/OperatorResult**——实现形态自由（纯函数/独立 struct/CRTP 均可），对齐算子规范 v1.9 三元组（Params/Result/Operator）但**不预设继承体系**。
- **AlgorithmRegistry**（ADR 7.1）：概念性模板注册表，**具体集成时按算子定**。

```cpp
namespace jmw::algorithm {
// operator_convention.h —— 文档化约定（非基类）：
//   算子 = 纯函数/独立类型；提供 Execute/Destroy/Warmup；返回自带 Result(success/qualityFlag/message)。
//   不强制继承。具体形态在 modules/09_operatorlib 集成时按算子定。
template<typename T> class AlgorithmRegistry { /* 占位，集成时定 */ };
}
```

> **影响**：使用算子的模块（01/02/03/04）在集成时**直接依赖 mod_09_operatorlib 的具体算子类型**（组合优于继承，无虚分发）；骨架阶段仅链 fw_algorithm 取共享 QualityFlag。

### 3.3 framework/workflow（编排，07-02 §3.1/6.1）
```cpp
namespace jmw::workflow {
enum class WorkflowStatus { Idle, Running, Paused, Stopped, Faulted };
class IWorkflow { public: virtual ~IWorkflow()=default; virtual void execute()=0; virtual void stop()=0; virtual void pause()=0; virtual void resume()=0; virtual WorkflowStatus getStatus() const=0; };
class WorkflowContext { /* 统一入口；ADR 7.7 窄角色接口(IScanContext/ICalibContext/...) 待实现 — 见头文件预留注释 */ };
class Pipeline { /* Stage 组合/有界队列/背压/丢帧 ADR 7.2 */ };
class Stage { };
}
```

### 3.4 framework/service / data / hal / infra / crosscut / ui
（与 v1 相同，要点：）
- **service**：IService / StateMachine / FaultHandler
- **data**：IDataStore / DataContext / **IFrameSink·IDeviceStateSink（依赖倒置）** / WorkflowArtifactStore（ADR 7.10）；**ADR 7.6 FusionStateHandle 在 IFrameSink 或独立头文件预留注释**
- **hal**：ICamera / IMCU / IPlatform（通讯链边界适配器）
- **infra**：EventBus（**ADR 7.9 AlgorithmFault 事件预留注释**）/ GpuRuntime（ResourceLifecycleManager ADR 7.11）/ Scheduler / Watchdog
- **crosscut**：IAuth / ILogger / IPerfMonitor / ICrashHandler / IConfig
- **ui**：IView（**ADR 7.8 IPointCloudReadView 跨层直读窄接口预留注释**）/ IUIController

### 3.5 ADR 机制预留（G2 决策：注释占位，不建空类）
| ADR | 机制 | 预留位置 | 方式 |
|-----|------|---------|------|
| 7.6 | FusionStateHandle | fw_data 头文件 | 注释 |
| 7.7 | WorkflowContext 窄角色接口 | fw_workflow/WorkflowContext.h | 注释 |
| 7.8 | IPointCloudReadView | fw_ui/IView.h | 注释 |
| 7.9 | AlgorithmFault 事件 | fw_infra/EventBus.h | 注释 |
| 7.10 | WorkflowArtifactStore | fw_data | ✅ 实类 |
| 7.11 | ResourceLifecycleManager | fw_infra/GpuRuntime.h | ✅ 注释+实类 |

### 3.6 模块 README（G4 决策）
每个模块 README 记 01.4 §3 的功能说明一句话。**modules/04_postprocess/README.md** 额外记后处理七阶段：「全局优化→重融合→**法线计算**→封装→补洞→光顺→边界优化（01.4 §3.1）」。

### 3.7 框架结构元素实现归属（G1 决策）⭐ 重要
01.4 §3.4 把 **SDK** 和 **用户权限** 定为「框架结构元素（非模块，不计入 11）」。它们的**实现归属**：
| 元素 | 契约位置 | 实现归属 | 说明 |
|------|---------|---------|------|
| 用户权限/鉴权 | fw_crosscut/IAuth.h | **fw_crosscut 提供**（横切基础设施实现） | 接入端（UI/SDK）调用；非独立模块 |
| SDK（二次开发） | sdk/IScannerSDK.h | **sdk/ 门面实现** | 接入端 B 的 API 门面，组合各层能力 |

**不新增模块**，保持 01.4 的 11 模块边界。

---

## 4. 模块 → framework 层依赖（U1 修正：06_fileio 补 fw_ui）

| 模块 | 链接的 framework 层 | 01.4 对照 |
|------|---------------------|-----------|
| 01_calibration | fw_workflow fw_algorithm fw_data | Workflow\|Algorithm·Data ✓ |
| 02_scanning | fw_workflow fw_algorithm fw_data fw_infra | Workflow\|Algorithm·Data（+infra 给 Scheduler） |
| 03_rendering | fw_ui fw_algorithm fw_data | UI\|Algorithm·Data ✓ |
| 04_postprocess | fw_workflow fw_algorithm fw_data | Workflow\|Algorithm·Data ✓ |
| 05_editing | fw_ui fw_service fw_data | UI\|Service·Data ✓ |
| **06_fileio** | **fw_data fw_service fw_ui** ✅(U1 修正) | Data\|Service·**UI(界面)** ✓ |
| 07_session | fw_service fw_workflow fw_ui fw_data | Service\|Workflow·UI·Data ✓ |
| 08_devicemgmt | fw_hal fw_service fw_ui fw_data fw_crosscut | HAL\|Service·UI·Data（+crosscut 给许可证） |
| 09_operatorlib | fw_algorithm | Algorithm（算子迁入后含具体算子类型） |
| 10_observability | fw_crosscut | 横切\|全局 ✓ |
| 11_deploy | （部署期，独立） | 部署wrapper ✓ |

> **算子使用模块的额外依赖**（集成时）：01/02/03/04 在算子迁入后另需链 `mod_09_operatorlib`（取具体算子类型，纯函数无基类故直链具体）；骨架阶段仅链 fw_algorithm 取共享 QualityFlag。

---

## 5. CMake 结构（继承 3DSCANNER 模式）

### 5.1 顶层
```cmake
cmake_minimum_required(VERSION 3.24)
project(JEAMMWARE VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
option(BUILD_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_UI "Build Qt UI layer (deferred)" OFF)
if(BUILD_CUDA)
    set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
    if(MSVC) string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8") endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
endif()
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")
# find_package(OpenCV/Eigen) 延后到算子迁入
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
add_subdirectory(framework)
add_subdirectory(modules)
add_subdirectory(sdk)
add_subdirectory(app)
```

### 5.2 各层 CMake
- `framework/<layer>/`：`add_library(fw_<layer> INTERFACE)`（纯头契约）；`fw_common` 为 STATIC（含 types）
- `modules/<n>_<name>/`：`add_library(mod_<name> STATIC "")`，链对应 fw_<层>
- `app/`：`add_executable(jeammware main.cpp)`，链 mod_* + jmw_sdk

---

## 6. 验证标准（骨架完成判据）

```powershell
cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64
cmake --build E:\JEAMMWARE260705\build --config Release
& E:\JEAMMWARE260705\build\app\Release\jeammware.exe
ctest --test-dir E:\JEAMMWARE260705\build -C Release --output-on-failure
```
- [ ] configure 成功（FetchContent spdlog/json/gtest）
- [ ] build 成功：8 fw_* + 11 mod_* + jmw_sdk + jeammware.exe
- [ ] exe 打印 banner 退出 0
- [ ] 各层契约头可独立 #include
- [ ] BUILD_TESTS：空 GTest 通过

---

## 7. Git 工程化
- `git init`（已执行，本地身份 `JEAMMWARE Dev <dev@local>`）
- `.gitignore`：build/ out/ *.user .vs/ compile_commands.json
- `README.md` + `AGENTS.md`（项目根索引）
- `.gitattributes`：LFS 配 `*.ply *.pcd *.stl *.tif *.bmp`

---

## 8. 后续
1. 算子迁入：3DSCANNER core/calib/scan → modules/09_operatorlib；启用 find_package(OpenCV/Eigen)+CUDA；算子作纯函数/独立类型，无基类派生
2. 各层实现：按 8 切片路线图
3. 垂直切片：smoke 链端到端打通
4. 跨工程同步：算子契约对齐算子规范 v1.9（三元组，但实现形态自由）

---

## 9. 待决策点（更新）

| # | 决策 | 状态 |
|---|------|------|
| D1 | 命名空间 `jmw::` | 待确认 |
| D2 | framework 层库类型 INTERFACE（fw_common STATIC） | 默认 |
| D3 | UI 框架 Qt 5.15.2（BUILD_UI=OFF 本次） | 默认 |
| D4 | 算子迁入时机 = 骨架后单独一步 | 默认 |
| D5 | 算子实现形态 = **纯函数/独立类型，无基类派生**（G5 已定） | ✅ 已定 |
| D6 | 用户权限实现归 fw_crosscut；SDK 实现归 sdk/（G1 已定） | ✅ 已定 |
