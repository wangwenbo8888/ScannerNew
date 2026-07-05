# JEAMMWARE 软件架构骨架 · 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `E:\JEAMMWARE260705` 搭起纯骨架：framework 8 层契约桩（INTERFACE）+ modules 11 空桩 + sdk + app exe，CMake 继承 3DSCANNER 环境，构建验证通过。

**Architecture:** 方案 C（framework 层契约 + modules 模块实现）。算子纯函数无基类派生（G5）。命名空间 `Scanner::`。

**Tech Stack:** C++20 / MSVC v144 / VS2022 / CMake≥3.24；骨架硬依赖 spdlog+nlohmann_json+GTest（FetchContent）；OpenCV/CUDA/Eigen 配置就绪但桩不链接。

**设计依据:** `E:\JEAMMWARE260705\docs\architecture\software-architecture-skeleton-design.md`（v2，已应用 6 条决策）

**关键约束:**
- 桩为纯 C++（不引 OpenCV/CUDA 头），保证骨架零外部装可构建
- `cv::Mat` 等用占位/前向声明，注释标"迁入后改"
- 各层为 INTERFACE 库（fw_common 为 STATIC）；modules 为 STATIC 空库
- 工作目录 `E:\JEAMMWARE260705`

---

## Task 1: 顶层工程文件（.gitignore / .gitattributes / README / AGENTS）

**Files:**
- Create: `E:\JEAMMWARE260705\.gitignore`
- Create: `E:\JEAMMWARE260705\.gitattributes`
- Create: `E:\JEAMMWARE260705\README.md`
- Create: `E:\JEAMMWARE260705\AGENTS.md`

**内容:**

`.gitignore`:
```
build/
out/
*.user
.vs/
.vscode/
CMakeSettings.json
compile_commands.json
*.tmp
```

`.gitattributes`:
```
*.ply filter=lfs diff=lfs merge=lfs -text
*.pcd filter=lfs diff=lfs merge=lfs -text
*.stl filter=lfs diff=lfs merge=lfs -text
*.obj filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.bmp filter=lfs diff=lfs merge=lfs -text
```

`README.md`:
```markdown
# JEAMMWARE — 3D 扫描仪交付级产品软件

手持式双目多线激光 3D 扫描仪产品软件工程。架构：5 层（UI/Workflow/Service∥Algorithm/Data/HAL）+ 侧边设施 + 横切 + 部署 wrapper + SDK 接入端。

## 构建（需 VS2022 + CMake≥3.24；spdlog/json/gtest 经 FetchContent 自动拉取）
​```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
​```

## 架构
见 `docs/architecture/software-architecture-skeleton-design.md`。
上层架构依据：`E:\3DSCANNER260622\docs\plans\2026-07-05-交付级框架整体设计-design.md`（01.4）。

## 环境
继承 `E:\3DSCANNER260622`（C++20/MSVC v144/CUDA 12.6/OpenCV 4.13/Eigen 3.4.1）。
```

`AGENTS.md`（项目根索引，AI/人查阅；骨架版，后续扩充）:
```markdown
# AGENTS.md — JEAMMWARE 工程索引

> **快照日期**: 2026-07-05 ｜ **阶段**: 骨架搭建中
> **给 AI**: 3D 扫描仪产品软件。架构 = 5 层（framework/）+ 11 模块（modules/）。读 `docs/architecture/software-architecture-skeleton-design.md` 取设计。
> **环境**: 继承 E:\3DSCANNER260622（C++20/MSVC v144/CUDA 12.6/OpenCV 4.13/Eigen）。

## 构建/测试
​```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
​```

## 结构
- `framework/` = 层契约（common/ui/workflow/service/algorithm/data/hal/infra/crosscut）
- `modules/` = 11 业务模块空桩（01-11）
- `sdk/` = 接入端 B ｜ `app/` = exe 入口
- 命名空间 `Scanner::`

## 现状
骨架阶段：层契约桩 + 空模块。算子迁入、各层实现待后续（见设计稿 §8）。
```

**Verify:** 文件存在；内容正确。
**Commit:** `git add -A && git commit -m "chore: 顶层工程文件(.gitignore/.gitattributes/README/AGENTS)"`

---

## Task 2: 顶层 CMakeLists.txt

**Files:** Create `E:\JEAMMWARE260705\CMakeLists.txt`

**内容**（照设计稿 §5.1）:
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
    if(MSVC)
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
endif()

# 第三方路径（CACHE，骨架桩不强制 find；算子迁入后启用）
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")

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

**Verify:** `cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64` → 预期失败（subdir 未建），但 FetchContent 阶段应通过（spdlog/json/gtest 拉取成功）。记录错误为"缺 framework/modules/sdk/app 子目录"。
**不提交**（聚合到 Task 9 后）。

---

## Task 3: framework/ 骨架（CMake + common 共享类型）

**Files:**
- Create: `framework/CMakeLists.txt`
- Create: `framework/common/CMakeLists.txt`
- Create: `framework/common/quality_flag.h`
- Create: `framework/common/types.h`

`framework/common/quality_flag.h`:
```cpp
#pragma once
namespace Scanner {
enum class QualityFlag { Normal, Degraded, Warning, Fault };
}  // namespace Scanner
```

`framework/common/types.h`:
```cpp
#pragma once
#include <cstdint>
#include <memory>
namespace Scanner {
// 占位类型（算子迁入后 leftGray/rightGray 改 std::shared_ptr<cv::Mat>）
struct Frame {
    std::shared_ptr<void> leftGray;
    std::shared_ptr<void> rightGray;
    uint64_t frameId = 0;
    uint64_t timestamp = 0;
    double temperature = 25.0;
};
struct FrameResult {
    uint64_t frameId = 0;
};
}  // namespace Scanner
```

`framework/common/CMakeLists.txt`:
```cmake
add_library(fw_common STATIC "")
target_sources(fw_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/quality_flag.h ${CMAKE_CURRENT_SOURCE_DIR}/types.h)
target_include_directories(fw_common PUBLIC ${CMAKE_SOURCE_DIR})
```

`framework/CMakeLists.txt`:
```cmake
add_subdirectory(common)
add_subdirectory(ui)
add_subdirectory(workflow)
add_subdirectory(service)
add_subdirectory(algorithm)
add_subdirectory(data)
add_subdirectory(hal)
add_subdirectory(infra)
add_subdirectory(crosscut)
```

**Verify:** 暂不单独构建（subdir 未齐）。
**不提交。**

---

## Task 4: framework 各层头文件桩（8 个层，INTERFACE 库）

**Files:** 每层一个头（或多头）+ CMakeLists。内容要点（最小可编译 + ADR 注释）：

`framework/ui/IView.h` + `IUIController.h`:
```cpp
#pragma once
namespace Scanner::ui {
class IView { public: virtual ~IView() = default; virtual void render() = 0; };
// ADR 7.8 预留：IPointCloudReadView（UI 跨层直读 Data 的只读窄接口）待实现
class IUIController { public: virtual ~IUIController() = default; };
}
```

`framework/workflow/IWorkflow.h` + `WorkflowContext.h` + `Pipeline.h` + `Stage.h`（合并到一个 `workflow_fwd.h` 或分文件，骨架可合一）:
```cpp
#pragma once
namespace Scanner::workflow {
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
// WorkflowContext：统一入口；ADR 7.7 窄角色接口（IScanContext/ICalibContext/...）待实现
class WorkflowContext {};
class Pipeline {};   // ADR 7.2 Stage 组合/有界队列/背压/丢帧
class Stage {};
}
```

`framework/service/IService.h` + `StateMachine.h` + `FaultHandler.h`（合一）:
```cpp
#pragma once
namespace Scanner::service {
class IService { public: virtual ~IService() = default; virtual const char* name() const = 0; };
class StateMachine {};   // 待机/标定/扫描/后处理/故障 转移
class FaultHandler {};   // ADR 7.9 HardwareFault/AlgorithmFault → FaultOccurred
}
```

`framework/algorithm/operator_convention.h`（G5：纯函数无基类）:
```cpp
#pragma once
// 算子契约（文档化约定，非基类派生 — G5 决策）：
//   算子 = 纯函数/独立类型；提供 Execute/Destroy/Warmup；
//   返回自带 Result（含 success/qualityFlag/message，QualityFlag 见 Scanner::QualityFlag）。
//   不强制继承。具体形态在 modules/09_operatorlib 集成时按算子定。
//   对齐算子规范 v1.9 三元组（Params/Result/Operator）。
namespace Scanner::algorithm {
// ADR 7.1 AlgorithmRegistry：概念性模板注册，集成时定
template <typename T> class AlgorithmRegistry {
public:
    void registerOperator(const char* /*key*/, T* /*op*/) {}
};
}
```

`framework/data/IDataStore.h`（含 IFrameSink/IDeviceStateSink/WorkflowArtifactStore/ADR 7.6 注释）:
```cpp
#pragma once
#include "../common/types.h"
namespace Scanner::data {
class IFrameSink {  // 依赖倒置：HAL 回填帧
public:
    virtual ~IFrameSink() = default;
    virtual void onFrame(const Frame&) = 0;
};
class IDeviceStateSink { public: virtual ~IDeviceStateSink() = default; };
class IDataStore { public: virtual ~IDataStore() = default; };
class DataContext {};
class WorkflowArtifactStore {};  // ADR 7.10
// ADR 7.6 预留：FusionStateHandle（有状态融合接缝的不透明句柄）待实现
}
```

`framework/hal/ICamera.h` + `IMCU.h` + `IPlatform.h`（合一）:
```cpp
#pragma once
namespace Scanner::hal {
class ICamera { public: virtual ~ICamera() = default; };   // 通讯链边界适配器
class IMCU { public: virtual ~IMCU() = default; };
class IPlatform { public: virtual ~IPlatform() = default; };
}
```

`framework/infra/EventBus.h` + `GpuRuntime.h` + `Scheduler.h` + `Watchdog.h`（合一）:
```cpp
#pragma once
namespace Scanner::infra {
class EventBus { public: /* ADR 7.9 AlgorithmFault 事件预留 */ };
class GpuRuntime { public: /* ADR 7.11 ResourceLifecycleManager */ };
class Scheduler {};
class Watchdog {};
}
```

`framework/crosscut/IAuth.h` + `ILogger.h` + `IPerfMonitor.h` + `ICrashHandler.h` + `IConfig.h`（合一）:
```cpp
#pragma once
namespace Scanner::crosscut {
// 用户权限/鉴权（G1：框架结构元素，实现在 fw_crosscut）
class IAuth { public: virtual ~IAuth() = default; };
class ILogger { public: virtual ~ILogger() = default; };
class IPerfMonitor { public: virtual ~IPerfMonitor() = default; };
class ICrashHandler { public: virtual ~ICrashHandler() = default; };
class IConfig { public: virtual ~IConfig() = default; };
}
```

每层 `CMakeLists.txt`（除 common 外，都是 INTERFACE 库）:
```cmake
add_library(fw_ui INTERFACE)
target_include_directories(fw_ui INTERFACE ${CMAKE_SOURCE_DIR})
# fw_common 链给需要的层
target_link_libraries(fw_ui INTERFACE fw_common)
```
（ui/workflow/service/data/hal/infra/crosscut 同模式，名字替换；algorithm/crosscut 链 fw_common；data 链 fw_common（用 Frame））

**Verify:** `framework/CMakeLists.txt` 现可被 add_subdirectory（8 层齐全）。
**不提交。**

---

## Task 5: modules/ 11 空桩

**Files:**
- `modules/CMakeLists.txt`
- 每个模块 `modules/<NN>_<name>/CMakeLists.txt` + `README.md` + `<name>.h` + `<name>.cpp`（空）

`modules/CMakeLists.txt`:
```cmake
add_subdirectory(01_calibration)
add_subdirectory(02_scanning)
add_subdirectory(03_rendering)
add_subdirectory(04_postprocess)
add_subdirectory(05_editing)
add_subdirectory(06_fileio)
add_subdirectory(07_session)
add_subdirectory(08_devicemgmt)
add_subdirectory(09_operatorlib)
add_subdirectory(10_observability)
add_subdirectory(11_deploy)
```

每个模块 `CMakeLists.txt` 模式（按设计稿 §4 链对应 fw 层）。例 `01_calibration`:
```cmake
add_library(mod_calibration STATIC calibration.cpp)
target_link_libraries(mod_calibration PUBLIC fw_workflow fw_algorithm fw_data)
target_include_directories(mod_calibration PUBLIC ${CMAKE_SOURCE_DIR})
```

模块依赖对照（照设计稿 §4，U1 修正后）:
- 01_calibration: fw_workflow fw_algorithm fw_data
- 02_scanning: fw_workflow fw_algorithm fw_data fw_infra
- 03_rendering: fw_ui fw_algorithm fw_data
- 04_postprocess: fw_workflow fw_algorithm fw_data
- 05_editing: fw_ui fw_service fw_data
- 06_fileio: fw_data fw_service **fw_ui**  ← U1 修正
- 07_session: fw_service fw_workflow fw_ui fw_data
- 08_devicemgmt: fw_hal fw_service fw_ui fw_data fw_crosscut
- 09_operatorlib: fw_algorithm
- 10_observability: fw_crosscut
- 11_deploy: （STATIC 空库，无 fw 链；部署期 wrapper）

每个模块 `README.md` 一句话（取 01.4 §3 功能说明），如 `04_postprocess/README.md`:
```markdown
# 模块4 后处理

离线 batch 七阶段（全局标记点优化→重融合→**法线计算**→封装→补洞→光顺→边界优化），出 STL。
主层：Workflow。跨层：Algorithm · Data(ArtifactStore)。
```

每个模块 `.h`/`.cpp` 空占位:
```cpp
// calibration.h
#pragma once
namespace Scanner::calibration {
// 模块1 标定（空桩，后续实现）
}
```
```cpp
// calibration.cpp（空）
#include "calibration.h"
namespace Scanner::calibration {}
```

11 模块均按此模式（名字替换；09_operatorlib 的 .h 注释标"算子迁入目标"；11_deploy 注释标"部署期 wrapper"）。

**Verify:** `modules/CMakeLists.txt` 可被 add_subdirectory。
**不提交。**

---

## Task 6: sdk/ + app/

**Files:**
- `sdk/IScannerSDK.h` + `sdk/CMakeLists.txt`
- `app/main.cpp` + `app/CMakeLists.txt`

`sdk/IScannerSDK.h`:
```cpp
#pragma once
namespace Scanner::sdk {
// 接入端 B（二次开发 API 门面）。G1：实现在 sdk/，组合各层能力。
class IScannerSDK { public: virtual ~IScannerSDK() = default; };
}
```

`sdk/CMakeLists.txt`:
```cmake
add_library(jmw_sdk INTERFACE)
target_include_directories(jmw_sdk INTERFACE ${CMAKE_SOURCE_DIR})
```

`app/main.cpp`（验证各层头可 include + 打 banner）:
```cpp
#include <iostream>
#include "framework/common/quality_flag.h"
#include "framework/common/types.h"
#include "framework/ui/IView.h"
#include "framework/workflow/IWorkflow.h"
#include "framework/service/IService.h"
#include "framework/algorithm/operator_convention.h"
#include "framework/data/IDataStore.h"
#include "framework/hal/ICamera.h"
#include "framework/infra/EventBus.h"
#include "framework/crosscut/IAuth.h"
#include "sdk/IScannerSDK.h"

int main() {
    std::cout << "JEAMMWARE v0.1.0 skeleton\n";
    std::cout << "Scanner::QualityFlag Normal=" << static_cast<int>(Scanner::QualityFlag::Normal) << "\n";
    std::cout << "5 layers / 11 modules / sdk / app  —  skeleton OK\n";
    return 0;
}
```

`app/CMakeLists.txt`:
```cmake
add_executable(jeammware main.cpp)
target_link_libraries(jeammware PRIVATE
    fw_common fw_ui fw_workflow fw_service fw_algorithm fw_data fw_hal fw_infra fw_crosscut
    mod_calibration mod_scanning mod_rendering mod_postprocess mod_editing
    mod_fileio mod_session mod_devicemgmt mod_operatorlib mod_observability mod_deploy
    jmw_sdk
)
```

**Verify:** 暂不单独构建。
**不提交。**

---

## Task 7: 构建验证（关键里程碑）

**Step 1: configure**
```powershell
cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64
```
Expected: 成功（FetchContent 拉 spdlog/json/gtest；所有 add_subdirectory 解析；INTERFACE/STATIC 目标定义）。若失败，按错误修（常见：层 CMake 名字笔误、include 路径、缺头文件）。

**Step 2: build**
```powershell
cmake --build E:\JEAMMWARE260705\build --config Release
```
Expected: 全部目标构建通过 → `fw_common`(STATIC) + 8 fw_*(INTERFACE) + 11 mod_*(STATIC,空) + jmw_sdk(INTERFACE) + `jeammware.exe`。

**Step 3: run**
```powershell
& E:\JEAMMWARE260705\build\app\Release\jeammware.exe
```
Expected: 打印 banner + `Scanner::QualityFlag Normal=0` + `skeleton OK`，退出码 0。

**Step 4（可选）: 加 smoke GTest**
若 BUILD_TESTS，建 `framework/tests/test_smoke.cpp`:
```cpp
#include <gtest/gtest.h>
#include "framework/common/quality_flag.h"
TEST(Smoke, QualityFlagCompiles) {
    EXPECT_EQ(static_cast<int>(Scanner::QualityFlag::Normal), 0);
}
```
+ `framework/tests/CMakeLists.txt` 注册到 framework/CMakeLists.txt（`if(BUILD_TESTS) add_subdirectory(tests) endif()`）。
```powershell
ctest --test-dir E:\JEAMMWARE260705\build -C Release --output-on-failure
```
Expected: smoke 测试 PASS。

**Step 5: 提交（全部骨架）**
```powershell
cd E:\JEAMMWARE260705
git add -A
git commit -m "feat(skeleton): JEAMMWARE 架构骨架 - 8层framework契约桩 + 11模块空桩 + sdk + app exe(构建验证通过)"
```

---

## 完成判据

- [ ] 顶层文件齐（.gitignore/.gitattributes/README/AGENTS）
- [ ] CMake configure 成功（FetchContent spdlog/json/gtest）
- [ ] build 成功：8 fw_* + 11 mod_* + jmw_sdk + jeammware.exe
- [ ] exe 运行打印 banner 退出 0
- [ ] 各层头可独立 #include（main.cpp 验证）
- [ ] (可选) smoke GTest PASS
- [ ] 已提交

---

## §99 后续（骨架后，另起计划）

1. 算子迁入：3DSCANNER core/calib/scan → modules/09_operatorlib（纯函数/独立类型，无基类）；启用 find_package(OpenCV/Eigen) + CUDA
2. 各层实现：按 3DSCANNER 的 8 切片路线图逐层填
3. 垂直切片：smoke 链端到端打通
4. 跨工程同步算子契约
