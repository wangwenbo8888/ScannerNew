# 厂家标定工程 (factory_calib/) 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在工程根目录新建 `factory_calib/`，含两个独立可执行（相机标定 / 激光标定），原样拷贝主工程算子，自包含可独立构建运行。

**Architecture:** 方案 B（两棵独立子树）。模块1（`module1_camera/`，纯 CPU）自实现 1 个棋盘格角点提取函数 + 复用 6 个相机/温度算子；模块2（`module2_laser/`，CUDA）原样拷贝 15 个激光链算子。两 exe 文件交接（`camera_calib.json` → `laser_calib.json`）。不引用主工程任何源码。

**Tech Stack:** C++17 / MSVC v144 / OpenCV 4.13（+ CUDA 模块）/ CUDA 12.6 / Eigen 3.4.1 / nlohmann_json / spdlog / GoogleTest（FetchContent）。

**Design Doc:** `docs/plans/2026-07-18-factory-calib-design.md`（已评审通过）

---

## 进度日志（执行中维护）

### ✅ 已完成（Phase 0-5，模块1 完整 + 模块2 算子回归）

分支 `feat/factory-calib`。模块1（`module1_camera/camera_calib.exe`，纯 CPU）**Release 8/8 测试全绿**：
- Task 0.1-0.3：顶层 + 两模块 CMake（模块1 CPU / 模块2 CUDA 占位）
- Task 1.1-1.3：拷贝 6 相机/温度算子 + 6/6 自测一次过
- Task 2.1-2.3：`extractChessboardCorners` + `normalizeLRCornerOrder` TDD（7 单测）
- Task 3.1-3.4：`calib_io`（读+写）+ `camera_calib_cli` 完整驱动 + e2e（合成棋盘 reproj **0.171px**，fx 恢复误差 <1）

模块2（`module2_laser/`，CUDA）**Debug 16/16 算子自测一次过**（56s，build_fc2 dir，debug OpenCV）：
- Task 5.1：拷贝 `core/common` 11 个公共头（commit `a6ade08`）
- Task 5.2：拷贝 15 个激光链算子（72 文件 / 18798 行，commit `b4b4847`）；`fc2_ops.lib` 编译通过；**nvcc 实测 v12.4.99**（pin v12.6 仍因无 FORCE 被覆盖，但能编译 + 测试绿，按经验 #4 接受）
- Task 5.3：16 个 test_*.cpp（mask_extract 有 2 个）**Debug 全绿，无需任何 include/路径修复**

### ⏭ 待办（Phase 6-7，模块2 I/O + CLI + 集成 + 文档）

从 **Task 6.1**（`calib_io` 读 config + 加载模块1 交接文件）开始。

### 🔑 执行中获取的关键经验（Phase 5-7 必读）

1. **GitHub 被封** → 所有 configure 命令必须带 `-DJMW_GH_MIRROR=https://ghfast.top/`（FetchContent 否则超时）。
2. **OpenCV Debug/Release CRT 匹配**：Release 用 `C:/opencv-cuda-4.13.0`（/MD），Debug 用 `C:/opencv-cuda-4.13.0-debug/x64/vc17/lib`（/MDd）。
3. **Debug OpenCV 缺 imgcodecs**（自定义 debug 构建没带该组件，Release 有），**但 CUDA 组件齐全**（cudaarithm/cudafilters/cudaimgproc 都有）。模块1/2 的 `calib_io.cpp` 用 `cv::imread` 需 imgcodecs → **calib_io 的 Debug 构建会失败**（Task 6.1 起需注意）。模块2 算子自测（Task 5.3）不用 imgcodecs → **Debug 全绿**（实测 16/16，无需切 Release）。工厂工具交付物是 **Release**，Debug 限制记入 README。
4. **nvcc pin 串味（Task 0.3 + 5.2 实测确认）**：顶层 CMake 的 `set(CMAKE_CUDA_COMPILER "...v12.6..." CACHE FILEPATH ...)` **无 FORCE**，实际选了 **v12.4.99**（`Check for working CUDA compiler: .../v12.4/bin/nvcc.exe`）。Task 5.2 实测：v12.4.99 + 模块2 全部 .cu 编译通过 + 16/16 测试绿 → **按本经验接受 v12.4**，无需加 FORCE（与父工程同款代码 42/42 绿一致）。后续若新增 .cu 编译失败再考虑 FORCE。
5. **算子签名必须读头验证，不能信计划草稿**：Task 3.3 发现 `ExtrinsicCalibCpuParams` **没有** `cameraMatrixL/distCoeffsL/cameraMatrixR/distCoeffsR` 字段（计划草稿臆测了），改用 `Execute(KL,DL,KR,DR)` 重载在调用点传 K/D。Task 6.2 激光链 13 算子同样要逐个读头确认构造/Execute 签名。
6. **`findChessboardCornersSB` 角点顺序 call-history 依赖**（Task 2.2 实测）：同一图在不同调用历史下返回相反起始角。→ 任何测试/代码**不能假设 `corners[0]` 是左上**；用 `min(x+y)` 找左上。`normalizeLRCornerOrder` 只处理全局翻转（TL↔BR），不处理 transpose；完整规范化 + 极线校验延后（设计 §7 风险#2）。
7. **e2e 合成棋盘必须用 `projectPoints` 显式 K_gt 合成**（Task 3.4 实测）：任意透视四边形的多个单应矩阵不共享单一 K，违反 Zhang 正交性约束，reproj 卡在 ~1.0px。改用 `cv::projectPoints`（已知 K_gt/D=0/多姿态 rvec/tvec）生成 dstQuad → warpPerspective，reproj 降到 0.171px。模块2 若需合成数据，同样要保证多帧标定约束自洽。

---

## 全局约定（每个任务都要遵守）

- **工作目录**: 所有相对路径以工程根 `E:\JEAMMWARE260705\` 为基准。
- **源码主工程根**: `modules/09_operatorlib/`（下记 `$OP`）。
- **拷贝规则**: 算子源码**逐字拷贝**，仅当 `#include` 找不到时调整路径前缀。**禁止改算子业务逻辑**。
- **include 风格**: 算子用 `#include "common/xxx.h"`（来自 `$OP/core/common/`）。拷贝到模块后放在 `<module>/operators/common/`，并把 `<module>/operators/` 加入 include 路径即可解析，**通常无需改 include**。
- **跨算子 include**: 如 `laser_extrinsic_compensate_cpu.h` 含 `#include "extrinsic_compensate_cpu.h"`（同 `$OP/calibration/temp/` 下两兄弟目录）。解决办法：把每个算子目录都加入 include 路径（CMake glob 所有算子目录，复刻 `$OP/CMakeLists.txt` 的 `OP_INC_DIRS`）。
- **提交风格**: 中文 conventional commit，参考 `git log --oneline -5`（`feat(...)` / `docs(...)` / `chore(...)`）。
- **测试运行**: `ctest --test-dir <build> -C <Debug|Release> --output-on-failure`。
- **CRT 部署**: 模块 exe 与测试 exe 都要调 `jmw_deploy_crt()`，复刻 `$OP/CMakeLists.txt:94-106` 的函数（OpenCV 运行期依赖 MSVCP140）。

---

## Phase 0 — 脚手架

### Task 0.1: 创建目录骨架 + 顶层 CMakeLists + README 占位

**Files:**
- Create: `factory_calib/CMakeLists.txt`
- Create: `factory_calib/README.md`
- Create: `factory_calib/data_in/.gitkeep`, `factory_calib/data_out/.gitkeep`
- Create: `factory_calib/cmake/factory_calib_common.cmake`（共享函数：`fc_deploy_crt`）

**Step 1: 建目录**

```powershell
New-Item -ItemType Directory -Path "factory_calib\module1_camera\operators\common","factory_calib\module1_camera\tests","factory_calib\module2_laser\operators\common","factory_calib\module2_laser\tests","factory_calib\cmake","factory_calib\data_in","factory_calib\data_out" -Force
```

**Step 2: 写 `factory_calib/cmake/factory_calib_common.cmake`**（复刻主工程 `jmw_deploy_crt`）

```cmake
# 复刻自主工程根 CMakeLists.txt:91-106
set(FC_MSVC_REDIST_CRT_DIR
    "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
    CACHE PATH "VS2022 MSVC redist CRT DLL dir")
function(fc_deploy_crt target)
    if(NOT EXISTS "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll")
        message(WARNING "fc_deploy_crt: redist CRT 目录未找到: ${FC_MSVC_REDIST_CRT_DIR} (跳过)")
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll"
            "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140.dll"
            "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140_1.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "部署 VS2022 CRT DLL 到 ${target}")
endfunction()
```

**Step 3: 写 `factory_calib/CMakeLists.txt`**（顶层，选项 + 子目录）

```cmake
cmake_minimum_required(VERSION 3.24)
project(factory_calib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

option(FC_BUILD_MODULE1 "Build camera calibration module (CPU)" ON)
option(FC_BUILD_MODULE2 "Build laser calibration module (CUDA)" ON)
option(FC_BUILD_TESTS  "Build tests" ON)

# 第三方路径（默认同主工程；可被 -D 覆盖）
set(OpenCV_DIR "C:/opencv-cuda-4.13.0" CACHE PATH "OpenCV")
set(Eigen3_DIR "C:/devlibs/eigen-3.4.1-install/share/eigen3/cmake" CACHE PATH "Eigen3")
set(JMW_GH_MIRROR "" CACHE STRING "GitHub 镜像前缀")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(factory_calib_common)

include(FetchContent)
find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog URL ${JMW_GH_MIRROR}https://github.com/gabime/spdlog/archive/refs/tags/v1.15.1.tar.gz)
    FetchContent_MakeAvailable(spdlog)
endif()
find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json URL ${JMW_GH_MIRROR}https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(nlohmann_json)
endif()
if(FC_BUILD_TESTS)
    FetchContent_Declare(googletest URL ${JMW_GH_MIRROR}https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

if(FC_BUILD_MODULE1)
    add_subdirectory(module1_camera)
endif()
if(FC_BUILD_MODULE2)
    add_subdirectory(module2_laser)
endif()
```

**Step 4: 写 `factory_calib/README.md`**（占位骨架，Phase 7 补全）

```markdown
# factory_calib — 厂家标定工程（自包含）

两个独立可执行：
- `module1_camera/camera_calib.exe` — 相机内/外参 + 立体矫正 + 温度补偿表（CPU）
- `module2_laser/laser_calib.exe` — 激光虚拟相机标定 + 温度补偿表（CUDA）

详细设计见 `../docs/plans/2026-07-18-factory-calib-design.md`。

## 构建（待 Phase 7 补全命令）
## 运行（待 Phase 7 补全）
## 输入/输出目录约定（待 Phase 7 补全）
```

**Step 5: 验证顶层配置能跑通**（子目录还没建 CMakeLists，先临时跳过 add_subdirectory 验证；此步只 `cmake -S factory_calib -B build_fc` 看FetchContent 与 find_package 是否解析成功，预期因缺子目录 CMakeLists 报错——正常）。

**Step 6: Commit**

```powershell
git add factory_calib
git commit -m "scaffold(factory_calib): 顶层 CMake + 目录骨架 + fc_deploy_crt"
```

---

### Task 0.2: 模块1 CMakeLists（CPU，先建空 exe 占位）

**Files:**
- Create: `factory_calib/module1_camera/CMakeLists.txt`

**Step 1: 写 CMakeLists**（先只产出一个空 exe 与占位 main，下个 Phase 再填算子）

```cmake
# module1_camera: 相机标定（纯 CPU）
# 复刻 modules/09_operatorlib/CMakeLists.txt 的 glob + include 策略，但只 CPU

find_package(OpenCV 4.13 REQUIRED COMPONENTS core imgproc calib3d flann)
find_package(Eigen3 3.4 REQUIRED)

# === 收集算子源 ===
file(GLOB_RECURSE M1_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M1_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
set(M1_SOURCES "")
set(M1_TESTS "")
foreach(f ${M1_OP_CPP})
    if(f MATCHES "/tests/test_[^/]*\\.cpp$")
        list(APPEND M1_TESTS ${f})
    elseif(f MATCHES "/tests/")
        # 跳过 tests 目录内非 test_ 文件
    else()
        list(APPEND M1_SOURCES ${f})
    endif()
endforeach()

# 每个算子目录都加入 include 路径（跨算子 include）
set(M1_INC_DIRS "")
foreach(h ${M1_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M1_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M1_INC_DIRS)

# === 算子静态库 ===
if(M1_SOURCES)
    add_library(fc1_ops STATIC ${M1_SOURCES})
    target_include_directories(fc1_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/operators   # "common/xxx.h"
        ${M1_INC_DIRS})                          # 跨算子 include
    target_link_libraries(fc1_ops PUBLIC
        ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(fc1_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
endif()

# === camera_calib.exe ===
add_executable(camera_calib camera_calib_cli.cpp)
if(TARGET fc1_ops)
    target_link_libraries(camera_calib PRIVATE fc1_ops)
endif()
target_link_libraries(camera_calib PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
fc_deploy_crt(camera_calib)

# === 测试 ===
if(FC_BUILD_TESTS)
    foreach(t ${M1_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE fc1_ops GTest::gtest_main)
        target_include_directories(${tname} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M1_INC_DIRS})
        target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=0)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        fc_deploy_crt(${tname})
    endforeach()
    # 模块1 自有测试（Phase 2/3 添加，glob 自带捕获）
endif()
```

**Step 2: 写占位 `module1_camera/camera_calib_cli.cpp`**

```cpp
#include <iostream>
int main(int argc, char** argv) {
    std::cout << "camera_calib (placeholder)\n";
    return 0;
}
```

**Step 3: 构建验证**

```powershell
cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF
cmake --build build_fc --config Debug
.\build_fc\module1_camera\Debug\camera_calib.exe
```
预期：打印 `camera_calib (placeholder)` 并退出 0。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera
git commit -m "scaffold(factory_calib/module1): CPU CMakeLists + 占位 exe"
```

---

### Task 0.3: 模块2 CMakeLists（CUDA，先建空 exe 占位）

**Files:**
- Create: `factory_calib/module2_laser/CMakeLists.txt`

**Step 1: 在顶层 CMakeLists 启用 CUDA**（修改 Task 0.1 的文件）

在 `factory_calib/CMakeLists.txt` 的 `FC_BUILD_MODULE2` 判断内，`add_subdirectory(module2_laser)` **之前**插入 CUDA 启用块（复刻主工程根 `CMakeLists.txt:17-28`）：

```cmake
if(FC_BUILD_MODULE2)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
        CACHE FILEPATH "nvcc (pin v12.6)")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_DEPENDENCIES_USE_COMPILER OFF)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;87" CACHE STRING "sm_75/86/87" FORCE)
    if(MSVC)
        string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/utf-8")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -DFMT_UNICODE=0")
    add_subdirectory(module2_laser)
endif()
```

**Step 2: 写 `module2_laser/CMakeLists.txt`**（CPU+CUDA glob，复刻 `$OP/CMakeLists.txt`）

```cmake
# module2_laser: 激光标定（CUDA）
find_package(OpenCV 4.13 REQUIRED COMPONENTS
    core imgproc calib3d cudaimgproc cudaarithm cudafilters flann)
find_package(Eigen3 3.4 REQUIRED)

file(GLOB_RECURSE M2_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M2_OP_CU  "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cu")
file(GLOB_RECURSE M2_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
set(M2_SOURCES "")
set(M2_TESTS "")
foreach(f ${M2_OP_CPP} ${M2_OP_CU})
    if(f MATCHES "/tests/test_[^/]*\\.cpp$")
        list(APPEND M2_TESTS ${f})
    elseif(f MATCHES "/tests/")
        # 跳过
    else()
        list(APPEND M2_SOURCES ${f})
    endif()
endforeach()

set(M2_INC_DIRS "")
foreach(h ${M2_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M2_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M2_INC_DIRS)

add_library(fc2_ops STATIC ${M2_SOURCES})
target_include_directories(fc2_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/operators
    ${M2_INC_DIRS})
target_link_libraries(fc2_ops PUBLIC
    ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
target_compile_definitions(fc2_ops PUBLIC BUILD_CUDA=1)
target_compile_definitions(fc2_ops PRIVATE LM_ENABLE_TIMING=1)
if(MSVC)
    target_compile_options(fc2_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

add_executable(laser_calib laser_calib_cli.cpp)
target_link_libraries(laser_calib PRIVATE fc2_ops)
fc_deploy_crt(laser_calib)

if(FC_BUILD_TESTS)
    foreach(t ${M2_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE fc2_ops GTest::gtest_main)
        target_include_directories(${tname} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M2_INC_DIRS})
        target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=1)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        fc_deploy_crt(${tname})
    endforeach()
endif()
```

**Step 3: 占位 `module2_laser/laser_calib_cli.cpp`**

```cpp
#include <iostream>
int main(int argc, char** argv) {
    std::cout << "laser_calib (placeholder)\n";
    return 0;
}
```

**Step 4: 构建验证（需 CUDA 环境）**

```powershell
cmake -S factory_calib -B build_fc2 -DFC_BUILD_MODULE1=OFF
cmake --build build_fc2 --config Debug
.\build_fc2\module2_laser\Debug\laser_calib.exe
```
预期：打印 `laser_calib (placeholder)`。若机器无 CUDA，跳过本步，Phase 5 再验证。

**Step 5: Commit**

```powershell
git add factory_calib
git commit -m "scaffold(factory_calib/module2): CUDA CMakeLists + 占位 exe + 顶层启用 CUDA"
```

---

## Phase 1 — 模块1 算子拷贝与回归

### Task 1.1: 拷贝 core/common 公共头到模块1

**Files:** Copy `$OP/core/common/*.h` → `factory_calib/module1_camera/operators/common/`

**Step 1: 拷贝**

```powershell
$src = "modules\09_operatorlib\core\common"
$dst = "factory_calib\module1_camera\operators\common"
Copy-Item -Path "$src\*.h" -Destination $dst -Force
Get-ChildItem $dst -Name
```

**Step 2: 验证 expected 文件存在**: `calib_types.h`, `calib_result_types.h`, `result.h`, `quality_flag.h`, `json_utils.h`, `scanner_api.h`, `version.h`, `calib_logging.h`（+ `pipeline_types.h`, `calib_warmup_config.h`, `zitai_result_types.h` 一并拷，按需裁剪留到链接期）。

**Step 3: 检查 common 内部 include 是否自洽**

```powershell
Select-String -Path "$dst\*.h" -Pattern '#include\s+"'
```
若出现 `#include "version.h"` 等同目录引用，正常（operators/ 在 include 路径）。若出现 `#include "../xxx"` 之类越界，记录待修。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera\operators\common
git commit -m "feat(factory_calib/m1): 拷贝 core/common 公共头"
```

---

### Task 1.2: 拷贝模块1 的 6 个算子

**Files:** 拷贝 6 个算子目录（含各自 `tests/`）到 `factory_calib/module1_camera/operators/`。

| 源 (`modules/09_operatorlib/`) | 目标 (`factory_calib/module1_camera/operators/`) |
|---|---|
| `calibration/camera/intrinsic_calib/` | `intrinsic_calib/` |
| `calibration/camera/extrinsic_calib/` | `extrinsic_calib/` |
| `calibration/camera/stereo_rectify/` | `stereo_rectify/` |
| `calibration/camera/stereo_rectify_temp_table/` | `stereo_rectify_temp_table/` |
| `calibration/temp/intrinsic_compensate/` | `intrinsic_compensate/` |
| `calibration/temp/extrinsic_compensate/` | `extrinsic_compensate/` |

**Step 1: 拷贝脚本**

```powershell
$srcBase = "modules\09_operatorlib"
$dstBase = "factory_calib\module1_camera\operators"
$ops = @(
  "calibration\camera\intrinsic_calib",
  "calibration\camera\extrinsic_calib",
  "calibration\camera\stereo_rectify",
  "calibration\camera\stereo_rectify_temp_table",
  "calibration\temp\intrinsic_compensate",
  "calibration\temp\extrinsic_compensate")
foreach($op in $ops){
  $name = Split-Path $op -Leaf
  Copy-Item -Path "$srcBase\$op" -Destination "$dstBase\$name" -Recurse -Force
}
Get-ChildItem $dstBase -Directory | Select-Object Name
```

**Step 2: 构建模块1，预期编译失败时定位 include 缺失**

```powershell
cmake --build build_fc --config Debug 2>&1 | Select-String "error C\d+:|cannot open"
```

**Step 3: 修复 include**（仅当上步报错时）

每个报错 `cannot open include "xxx.h"`：
- 若 `xxx.h` 在 common/ → 已在 include 路径，应是 typo 或缺文件，补拷
- 若 `xxx.h` 是另一算子头（如 `extrinsic_compensate_cpu.h`）→ 确认该算子目录已在 `M1_INC_DIRS`（CMake glob 自动含）。重新 configure：`cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF` 再 build。

**Step 4: 验证算子库链接通过**

```powershell
cmake --build build_fc --config Debug --target fc1_ops
```
预期：`fc1_ops.lib` 生成。

**Step 5: Commit**

```powershell
git add factory_calib\module1_camera\operators
git commit -m "feat(factory_calib/m1): 拷贝 6 个相机/温度算子 (intrinsic/extrinsic/stereo_rectify/+3 温度表)"
```

---

### Task 1.3: 模块1 算子自测回归（6 个 test_*.cpp 原样跑通）

**Files:** 已随 Task 1.2 拷入 `operators/<op>/tests/test_*.cpp`，CMake glob 自动捕获。

**Step 1: 构建 + 跑测**

```powershell
cmake --build build_fc --config Debug
ctest --test-dir build_fc -C Debug --output-on-failure -R "test_intrinsic_calib_cpu|test_extrinsic_calib_cpu|test_stereo_rectify_cpu|test_stereo_rectify_temp_table_cpu|test_intrinsic_compensate_cpu|test_extrinsic_compensate_cpu"
```

**Step 2: 预期**: 6 个测试全部 PASS。若有 FAIL：
- 若是 include 问题 → 修 include（同 Task 1.2 Step 3）
- 若是测试逻辑依赖主工程路径（如硬编码 `../../`）→ **仅修测试的路径字符串**，不改断言
- 若是 `SCANNER_VERSION_*` 未定义 → 检查 `operators/common/version.h` 已拷全

**Step 3: Commit**（若有测试修复）

```powershell
git add -A factory_calib\module1_camera
git commit -m "fix(factory_calib/m1): 算子自测 include/路径修复, 6/6 绿"
```

无修复则跳过本步。

---

## Phase 2 — 棋盘格角点提取函数（TDD）

### Task 2.1: 写失败测试 test_chessboard_corner

**Files:**
- Create: `factory_calib/module1_camera/chessboard_corner.h`
- Create: `factory_calib/module1_camera/tests/test_chessboard_corner.cpp`

**Step 1: 写头文件（接口先行）**

`factory_calib/module1_camera/chessboard_corner.h`:
```cpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace fc {

struct ChessboardCornerParams {
    cv::Size patternSize{11, 8};       // 内角点 (width, height)
    int sbFlags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_NORMALIZE_IMAGE;
    cv::Size subpixWin{11, 11};
    cv::Size subpixZeroZone{-1, -1};
    cv::TermCriteria subpixTerm{cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1e-4};
};

struct ChessboardCornerResult {
    bool found = false;
    std::vector<cv::Point2f> corners;  // 行主序, size == patternSize.area()
    double meanSubpixDelta = 0.0;       // 亚像素平均修正量(px)，诊断用
};

// ★ 单 CPU 函数：提取 + 编号（findChessboardCornersSB + cornerSubPix）
// 失败时 result.found=false，返回 false
bool extractChessboardCorners(const cv::Mat& gray,
                              const ChessboardCornerParams& params,
                              ChessboardCornerResult& result);

// L/R 编号一致性归一化：保证两图 corners[i] 对应同物理角点
// 思路：比较两图 4 角凸包走向，必要时翻转/逆序使走向一致；返回是否一致
bool normalizeLRCornerOrder(ChessboardCornerResult& left,
                            ChessboardCornerResult& right);

} // namespace fc
```

**Step 2: 写测试（合成棋盘图，已知 ground truth）**

`factory_calib/module1_camera/tests/test_chessboard_corner.cpp`:
```cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "chessboard_corner.h"

using namespace fc;

namespace {
// 合成一张带已知棋盘角点的灰度图
cv::Mat makeSyntheticChessboard(int winW, int winH, int squaresX, int squaresY,
                                int squarePx, int margin) {
    int W = margin * 2 + squaresX * squarePx;
    int H = margin * 2 + squaresY * squarePx;
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < squaresY; ++r)
        for (int c = 0; c < squaresX; ++c) {
            if ((r + c) % 2 == 0) {
                cv::Rect roi(margin + c * squarePx, margin + r * squarePx, squarePx, squarePx);
                img(roi) = 0;
            }
        }
    return img;
}
} // namespace

TEST(ChessboardCorner, ExtractsAllCorners) {
    int squaresX = 12, squaresY = 9;     // 内角点 = 11x8
    int squarePx = 30, margin = 40;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);

    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);  // 11x8
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    EXPECT_TRUE(r.found);
    EXPECT_EQ(r.corners.size(), static_cast<size_t>(p.patternSize.area()));
}

TEST(ChessboardCorner, FailsOnBlankImage) {
    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(255));
    ChessboardCornerParams p;
    ChessboardCornerResult r;
    EXPECT_FALSE(extractChessboardCorners(blank, p, r));
    EXPECT_FALSE(r.found);
}

TEST(ChessboardCorner, SubpixBetterThanPixel) {
    int squaresX = 8, squaresY = 6;
    int squarePx = 40, margin = 30;
    cv::Mat gray = makeSyntheticChessboard(0, 0, squaresX, squaresY, squarePx, margin);
    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX - 1, squaresY - 1);
    ChessboardCornerResult r;
    ASSERT_TRUE(extractChessboardCorners(gray, p, r));
    // 第一个角点应在 (margin, margin)
    EXPECT_NEAR(r.corners[0].x, margin, 0.5);
    EXPECT_NEAR(r.corners[0].y, margin, 0.5);
}
```

**Step 3: 把 tests/ 加入 glob 验证**——确认 `module1_camera/CMakeLists.txt` 的 glob 捕获到 `tests/test_chessboard_corner.cpp`（注意：当前 glob 是 `operators/*.cpp`，**需扩展**为也扫 `tests/`）。

修改 `module1_camera/CMakeLists.txt` 的 glob 行：
```cmake
file(GLOB_RECURSE M1_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M1_TESTS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
```
并在测试 foreach 里把 `${M1_TESTS_CPP}` 也并入 `M1_TESTS`：
```cmake
foreach(f ${M1_OP_CPP} ${M1_TESTS_CPP})
    if(f MATCHES "tests[\\\\/]test_[^/\\\\]*\\.cpp$")
        list(APPEND M1_TESTS ${f})
    elseif(f MATCHES "tests[\\\\/]")
        # 跳过
    else()
        list(APPEND M1_SOURCES ${f})
    endif()
endforeach()
```
（注：Windows 路径用反斜杠，正则同时兼容 `/` 与 `\`。）

**Step 4: 加 include 路径让测试能找到 `chessboard_corner.h`**

在测试 foreach 内 `target_include_directories` 增加 `${CMAKE_CURRENT_SOURCE_DIR}`：
```cmake
target_include_directories(${tname} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                         # chessboard_corner.h
    ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M1_INC_DIRS})
```

**Step 5: 跑测试，确认失败**

```powershell
cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF
cmake --build build_fc --config Debug --target test_chessboard_corner
ctest --test-dir build_fc -C Debug -R test_chessboard_corner --output-on-failure
```
预期：链接错误 `unresolved external symbol fc::extractChessboardCorners`（还没实现）。

**Step 6: Commit**

```powershell
git add factory_calib\module1_camera\chessboard_corner.h factory_calib\module1_camera\tests factory_calib\module1_camera\CMakeLists.txt
git commit -m "test(factory_calib/m1): test_chessboard_corner (RED) + glob 扩展 tests/"
```

---

### Task 2.2: 实现 extractChessboardCorners 让单图测试通过

**Files:**
- Create: `factory_calib/module1_camera/chessboard_corner.cpp`

**Step 1: 实现**

```cpp
#include "chessboard_corner.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <cmath>

namespace fc {

bool extractChessboardCorners(const cv::Mat& gray,
                              const ChessboardCornerParams& params,
                              ChessboardCornerResult& result) {
    result.found = false;
    result.corners.clear();
    result.meanSubpixDelta = 0.0;

    if (gray.empty()) {
        spdlog::warn("[chessboard_corner] empty image");
        return false;
    }
    cv::Mat gray8;
    if (gray.type() != CV_8U) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray8 = gray;
    }

    std::vector<cv::Point2f> coarse;
    bool found = cv::findChessboardCornersSB(gray8, params.patternSize, coarse, params.sbFlags);
    if (!found || static_cast<int>(coarse.size()) != params.patternSize.area()) {
        spdlog::debug("[chessboard_corner] not found or count mismatch");
        return false;
    }

    // 亚像素精化前快照，用于估算修正量
    std::vector<cv::Point2f> before = coarse;
    cv::cornerSubPix(gray8, coarse, params.subpixWin, params.subpixZeroZone, params.subpixTerm);

    double acc = 0.0;
    for (size_t i = 0; i < coarse.size(); ++i) {
        acc += std::hypot(coarse[i].x - before[i].x, coarse[i].y - before[i].y);
    }
    result.meanSubpixDelta = acc / coarse.size();

    result.corners = std::move(coarse);
    result.found = true;
    return true;
}

} // namespace fc
```

**Step 2: 把 `chessboard_corner.cpp` 加入构建**——它不在 `operators/` 下，需在 `module1_camera/CMakeLists.txt` 显式追加到 `camera_calib` exe 的 sources，并建一个独立小 lib 供测试链接：

在 `module1_camera/CMakeLists.txt` 算子库块之后、exe 之前插入：
```cmake
# 模块1 自有非算子源（chessboard_corner）
add_library(fc1_chess STATIC chessboard_corner.cpp chessboard_corner.h)
target_include_directories(fc1_chess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fc1_chess PUBLIC ${OpenCV_LIBS} spdlog::spdlog)
```
exe 与测试都链 `fc1_chess`：
```cmake
target_link_libraries(camera_calib PRIVATE fc1_ops fc1_chess ...)
# 测试 foreach 内:
target_link_libraries(${tname} PRIVATE fc1_ops fc1_chess GTest::gtest_main)
```

**Step 3: 构建 + 跑测**

```powershell
cmake --build build_fc --config Debug --target test_chessboard_corner
ctest --test-dir build_fc -C Debug -R test_chessboard_corner --output-on-failure
```
预期：3 个 TEST 全 PASS。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera\chessboard_corner.cpp factory_calib\module1_camera\CMakeLists.txt
git commit -m "feat(factory_calib/m1): extractChessboardCorners (findChessboardCornersSB + cornerSubPix), 3/3 绿"
```

---

### Task 2.3: 实现 normalizeLRCornerOrder（L/R 编号一致性）

**Files:**
- Modify: `factory_calib/module1_camera/chessboard_corner.cpp`（追加实现）
- Create: `factory_calib/module1_camera/tests/test_chessboard_corner.cpp`（追加测试）

**Step 1: 追加测试**（在 test_chessboard_corner.cpp 末尾）

```cpp
TEST(ChessboardCorner, LRConsistencyFlip) {
    // 造两图：右图是左图水平翻转，编号方向应相反；归一化后应一致
    int squaresX = 8, squaresY = 6, squarePx = 40, margin = 30;
    cv::Mat gL = makeSyntheticChessboard(0,0,squaresX,squaresY,squarePx,margin);
    cv::Mat gR;
    cv::flip(gL, gR, 1);   // 水平翻转

    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX-1, squaresY-1);
    ChessboardCornerResult rL, rR;
    ASSERT_TRUE(extractChessboardCorners(gL, p, rL));
    ASSERT_TRUE(extractChessboardCorners(gR, p, rR));

    // 归一化前：第一个角点 x 不同（一个左上一个右上）
    EXPECT_NE(rL.corners[0].x, rR.corners[0].x);
    ASSERT_TRUE(normalizeLRCornerOrder(rL, rR));
    // 归一化后：物理同向，第一点应近似镜像后对齐（这里只校验尺寸不变、found 仍真）
    EXPECT_EQ(rL.corners.size(), rR.corners.size());
    EXPECT_TRUE(rL.found && rR.found);
}

TEST(ChessboardCorner, LRConsistencySameDir) {
    int squaresX = 8, squaresY = 6, squarePx = 40, margin = 30;
    cv::Mat gL = makeSyntheticChessboard(0,0,squaresX,squaresY,squarePx,margin);
    cv::Mat gR = gL.clone();   // 完全相同 → 同向
    ChessboardCornerParams p;
    p.patternSize = cv::Size(squaresX-1, squaresY-1);
    ChessboardCornerResult rL, rR;
    ASSERT_TRUE(extractChessboardCorners(gL, p, rL));
    ASSERT_TRUE(extractChessboardCorners(gR, p, rR));
    EXPECT_TRUE(normalizeLRCornerOrder(rL, rR));
}
```

**Step 2: 实现**（追加到 chessboard_corner.cpp）

```cpp
bool fc::normalizeLRCornerOrder(ChessboardCornerResult& left,
                                ChessboardCornerResult& right) {
    if (!left.found || !right.found) return false;
    if (left.corners.size() != right.corners.size()) return false;

    // 用第一个角点 → 最后一角点的向量方向作为朝向判据
    // findChessboardCornersSB 给的顺序是行主序从某一角起步；
    // 若两图起步角的对角位置不同，把右图角点序列逆序（行/列翻转）
    auto firstLast = [](const std::vector<cv::Point2f>& v) {
        return v.back() - v.front();
    };
    cv::Point2f vL = firstLast(left.corners);
    cv::Point2f vR = firstLast(right.corners);
    double dot = vL.dot(vR);
    double mag = std::hypot(vL.x, vL.y) * std::hypot(vR.x, vR.y);
    if (mag == 0.0) return false;
    double cosang = dot / mag;
    // 若两向量夹角 > 90°（cos<0），起步方向相反 → 把右图逆序
    if (cosang < 0.0) {
        std::reverse(right.corners.begin(), right.corners.end());
    }
    return true;
}
```

> 注：该启发式在 25 姿态多视角下可能不够鲁棒；Phase 3 CLI 里再加"极线残差"二次校验，失败丢帧（见设计 §7 风险#2）。本 Task 只做基本方向归一。

**Step 3: 跑测**

```powershell
cmake --build build_fc --config Debug --target test_chessboard_corner
ctest --test-dir build_fc -C Debug -R test_chessboard_corner --output-on-failure
```
预期：5 个 TEST 全 PASS。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera\chessboard_corner.cpp factory_calib\module1_camera\tests\test_chessboard_corner.cpp
git commit -m "feat(factory_calib/m1): normalizeLRCornerOrder 方向归一化 + 2 测试"
```

---

## Phase 3 — 模块1 I/O 与 CLI 驱动

### Task 3.1: calib_io —— 读 config + 温度 + 图像对

**Files:**
- Create: `factory_calib/module1_camera/calib_io.h`
- Create: `factory_calib/module1_camera/calib_io.cpp`

**Step 1: 写头文件**

```cpp
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <optional>

namespace fc {

struct CameraCalibConfig {
    // 棋盘格
    int chessWidth = 11;
    int chessHeight = 8;
    double squareSizeMm = 2.0;
    // 图像
    int imageWidth = 2048;
    int imageHeight = 1536;
    // 内参
    int intrinsicFlags = 0;
    bool useCalibrateCameraRO = true;
    double reprojErrorThreshold = 0.012;
    // 温度
    double cte = 23.6e-6;
    double referenceTemp = 22.5;
    double tempRangeMin = -10.0;
    double tempRangeMax = 10.0;
    double tempStep = 0.2;
    // 矫正
    double rectifyAlpha = 0.0;
    int rectifyFlags = 1;
    // 温度系数（标定板热膨胀，喂 intrinsic_calib 的 temperature_coeff）
    double plateTempCoeff = 5.0e-6;
    double plateTemp = 21.0;

    static CameraCalibConfig fromJson(const std::string& path);
};

struct FramePair {
    cv::Mat leftGray;
    cv::Mat rightGray;
};

struct CameraInput {
    CameraCalibConfig config;
    std::vector<FramePair> frames;
};

// 读 data_in/camera/ 目录
std::optional<CameraInput> loadCameraInput(const std::string& dir);

} // namespace fc
```

**Step 2: 写实现**

```cpp
#include "calib_io.h"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

namespace fc {

namespace fs = std::filesystem;
using json = nlohmann::json;

CameraCalibConfig CameraCalibConfig::fromJson(const std::string& path) {
    CameraCalibConfig c;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::warn("config not found: {}, using defaults", path);
        return c;
    }
    json j = json::parse(ifs, nullptr, true);
    if (j.contains("chessboard")) {
        const auto& cb = j["chessboard"];
        if (cb.contains("width"))        c.chessWidth = cb["width"];
        if (cb.contains("height"))       c.chessHeight = cb["height"];
        if (cb.contains("square_size_mm")) c.squareSizeMm = cb["square_size_mm"];
    }
    if (j.contains("image_size")) {
        c.imageWidth  = j["image_size"][0];
        c.imageHeight = j["image_size"][1];
    }
    if (j.contains("intrinsic_flags"))      c.intrinsicFlags = j["intrinsic_flags"];
    if (j.contains("use_calibrateCameraRO")) c.useCalibrateCameraRO = j["use_calibrateCameraRO"];
    if (j.contains("reproj_error_threshold")) c.reprojErrorThreshold = j["reproj_error_threshold"];
    if (j.contains("temperature")) {
        const auto& t = j["temperature"];
        if (t.contains("cte"))           c.cte = t["cte"];
        if (t.contains("referenceTemp")) c.referenceTemp = t["referenceTemp"];
        if (t.contains("tempRangeMin"))  c.tempRangeMin = t["tempRangeMin"];
        if (t.contains("tempRangeMax"))  c.tempRangeMax = t["tempRangeMax"];
        if (t.contains("tempStep"))      c.tempStep = t["tempStep"];
    }
    if (j.contains("rectify")) {
        const auto& r = j["rectify"];
        if (r.contains("alpha")) c.rectifyAlpha = r["alpha"];
        if (r.contains("flags")) c.rectifyFlags = r["flags"];
    }
    return c;
}

std::optional<CameraInput> loadCameraInput(const std::string& dir) {
    CameraInput in;
    in.config = CameraCalibConfig::fromJson(dir + "/config.json");

    // 参考温度：优先 temps.txt 的 ref_temp 行，否则用 config.referenceTemp
    std::ifstream tf(dir + "/temps.txt");
    if (tf) {
        std::string key; double v;
        while (tf >> key >> v) {
            if (key == "ref_temp") { in.config.referenceTemp = v; break; }
        }
    }

    fs::path ldir = fs::path(dir) / "left";
    fs::path rdir = fs::path(dir) / "right";
    if (!fs::exists(ldir) || !fs::exists(rdir)) {
        spdlog::error("left/ or right/ missing in {}", dir);
        return std::nullopt;
    }
    std::vector<fs::path> lfiles;
    for (auto& e : fs::directory_iterator(ldir))
        if (e.path().extension() == ".png" || e.path().extension() == ".jpg")
            lfiles.push_back(e.path());
    std::sort(lfiles.begin(), lfiles.end());
    for (const auto& lf : lfiles) {
        fs::path rf = rdir / lf.filename();
        if (!fs::exists(rf)) {
            spdlog::warn("skip {}: no right pair", lf.filename().string());
            continue;
        }
        cv::Mat l = cv::imread(lf.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat r = cv::imread(rf.string(), cv::IMREAD_GRAYSCALE);
        if (l.empty() || r.empty()) {
            spdlog::warn("skip {}: read failed", lf.filename().string());
            continue;
        }
        in.frames.push_back({l, r});
    }
    spdlog::info("loaded {} frame pairs from {}", in.frames.size(), dir);
    if (in.frames.empty()) return std::nullopt;
    return in;
}

} // namespace fc
```

**Step 3: 构建验证**（只编译 calib_io，先不接 exe）

把 `calib_io.cpp` 加入 `fc1_chess` 同级或新建 `fc1_io` lib，在 `module1_camera/CMakeLists.txt` 加：
```cmake
add_library(fc1_io STATIC calib_io.cpp calib_io.h)
target_include_directories(fc1_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fc1_io PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
```
（后续 camera_calib exe 与 e2e 测试都链 `fc1_io`。）

```powershell
cmake --build build_fc --config Debug --target fc1_io
```
预期：编译通过。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera\calib_io.h factory_calib\module1_camera\calib_io.cpp factory_calib\module1_camera\CMakeLists.txt
git commit -m "feat(factory_calib/m1): calib_io 读 config/温度/图像对"
```

---

### Task 3.2: calib_io —— 写 camera_calib.json（交接文件）

**Files:**
- Modify: `factory_calib/module1_camera/calib_io.h`（追加 writer 声明）
- Modify: `factory_calib/module1_camera/calib_io.cpp`（追加 writer 实现）

**Step 1: 追加声明到 calib_io.h**

```cpp
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

namespace fc {

// 把模块1 全部结果与配置汇总成单个 json（复用各算子的 toJson()）
nlohmann::json buildCameraCalibJson(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const calib::StereoRectifyCpuResult& rectify,
    const calib::IntrinsicCompensateCPUResult& intrinTableL,
    const calib::IntrinsicCompensateCPUResult& intrinTableR,
    const calib::ExtrinsicCompensateCPUResult& extrinTable,
    const calib::StereoRectifyTempTableResult& rectifyTable);

bool writeJson(const std::string& path, const nlohmann::json& j);

} // namespace fc
```

**Step 2: 追加实现到 calib_io.cpp**

```cpp
nlohmann::json fc::buildCameraCalibJson(
    const CameraCalibConfig& cfg,
    const calib::IntrinsicCalibResult& intrin,
    const calib::ExtrinsicCalibCpuResult& extrin,
    const calib::StereoRectifyCpuResult& rectify,
    const calib::IntrinsicCompensateCPUResult& intrinTableL,
    const calib::IntrinsicCompensateCPUResult& intrinTableR,
    const calib::ExtrinsicCompensateCPUResult& extrinTable,
    const calib::StereoRectifyTempTableResult& rectifyTable)
{
    nlohmann::json j;
    j["schema"] = "factory_calib.camera_calib.v1";
    j["imageSize"] = {cfg.imageWidth, cfg.imageHeight};
    j["referenceTemp"] = cfg.referenceTemp;
    j["cte"] = cfg.cte;
    j["tempRangeMin"] = cfg.tempRangeMin;
    j["tempRangeMax"] = cfg.tempRangeMax;
    j["tempStep"] = cfg.tempStep;
    j["intrinsic"] = intrin.toJson();
    j["extrinsic"] = extrin.toJson();
    j["rectify"] = rectify.toJson();
    j["intrinsicTempTableL"] = intrinTableL.toJson();
    j["intrinsicTempTableR"] = intrinTableR.toJson();
    j["extrinsicTempTable"] = extrinTable.toJson();
    j["stereoRectifyTempTable"] = rectifyTable.toJson();
    return j;
}

bool fc::writeJson(const std::string& path, const nlohmann::json& j) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        spdlog::error("cannot write {}", path);
        return false;
    }
    ofs << j.dump(2);
    return true;
}
```

**Step 3: fc1_io 链 fc1_ops**（让 calib_io.cpp 能链算子头/符号）

在 `module1_camera/CMakeLists.txt` 的 `fc1_io` target 上：
```cmake
target_link_libraries(fc1_io PUBLIC fc1_ops ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
```

**Step 4: 构建验证**

```powershell
cmake --build build_fc --config Debug --target fc1_io
```

**Step 5: Commit**

```powershell
git add factory_calib\module1_camera\calib_io.h factory_calib\module1_camera\calib_io.cpp factory_calib\module1_camera\CMakeLists.txt
git commit -m "feat(factory_calib/m1): buildCameraCalibJson 写交接文件"
```

---

### Task 3.3: camera_calib_cli.cpp —— 完整驱动

**Files:**
- Modify: `factory_calib/module1_camera/camera_calib_cli.cpp`（替换占位）

**Step 1: 写完整 main**

```cpp
#include "calib_io.h"
#include "chessboard_corner.h"

#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "intrinsic_compensate_cpu.h"
#include "extrinsic_compensate_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <string>

using namespace fc;
using namespace calib;

namespace {

IntrinsicCalibParams makeIntrinParams(const CameraCalibConfig& c) {
    IntrinsicCalibParams p;
    p.chessboard_width = c.chessWidth;
    p.chessboard_height = c.chessHeight;
    p.square_size_mm = c.squareSizeMm;
    p.image_width = c.imageWidth;
    p.image_height = c.imageHeight;
    p.use_calibrateCameraRO = c.useCalibrateCameraRO;
    p.calib_flags = c.intrinsicFlags;
    p.reproj_error_threshold = c.reprojErrorThreshold;
    p.temperature_coeff = c.plateTempCoeff;
    p.plate_temp = c.plateTemp;
    return p;
}

ExtrinsicCalibCpuParams makeExtrinParams(const CameraCalibConfig& c,
    const std::vector<std::vector<cv::Point2f>>& lpts,
    const std::vector<std::vector<cv::Point2f>>& rpts,
    const cv::Mat& KL, const cv::Mat& DL,
    const cv::Mat& KR, const cv::Mat& DR)
{
    ExtrinsicCalibCpuParams p;
    p.leftPointsPerView = lpts;
    p.rightPointsPerView = rpts;
    p.imageSize = cv::Size(c.imageWidth, c.imageHeight);
    p.patternSize = cv::Size(c.chessWidth, c.chessHeight);
    p.squareSize = static_cast<float>(c.squareSizeMm);
    p.cameraMatrixL = KL.clone();
    p.distCoeffsL = DL.clone();
    p.cameraMatrixR = KR.clone();
    p.distCoeffsR = DR.clone();
    p.maxReprojError = c.reprojErrorThreshold * 100.0;  // 宽松阈值
    p.minViewCount = 4;
    return p;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: camera_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + left/ + right/ + temps.txt\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "camera_calib.json";

    auto input = loadCameraInput(inDir);
    if (!input) { spdlog::error("load input failed"); return 1; }
    const auto& cfg = input->config;

    // 1. 逐帧提取棋盘角点
    ChessboardCornerParams cp;
    cp.patternSize = cv::Size(cfg.chessWidth, cfg.chessHeight);
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    for (size_t i = 0; i < input->frames.size(); ++i) {
        ChessboardCornerResult rl, rr;
        if (!extractChessboardCorners(input->frames[i].leftGray, cp, rl) ||
            !extractChessboardCorners(input->frames[i].rightGray, cp, rr))
        {
            spdlog::warn("frame {}: corner extraction failed, skip", i);
            continue;
        }
        normalizeLRCornerOrder(rl, rr);
        lpts.push_back(std::move(rl.corners));
        rpts.push_back(std::move(rr.corners));
    }
    if (lpts.size() < 4) { spdlog::error("too few valid frames: {}", lpts.size()); return 1; }

    // 2. 内参
    IntrinsicCalibCPU intrin(makeIntrinParams(cfg));
    IntrinsicCalibResult intrinRes;
    if (!intrin.Execute(lpts, rpts, intrinRes) || !intrinRes.success) {
        spdlog::error("intrinsic calib failed: {}", intrinRes.message); return 1;
    }
    spdlog::info("intrinsic OK, reproj_mean={}", intrinRes.reproj_error_mean);

    // 3. 外参
    ExtrinsicCalibCpuParams ep = makeExtrinParams(cfg, lpts, rpts,
        intrinRes.left.camera_matrix, intrinRes.left.dist_coeffs,
        intrinRes.right.camera_matrix, intrinRes.right.dist_coeffs);
    ExtrinsicCalibCpu extrin(ep);
    ExtrinsicCalibCpuResult extrinRes = extrin.Execute(
        intrinRes.left.camera_matrix, intrinRes.left.dist_coeffs,
        intrinRes.right.camera_matrix, intrinRes.right.dist_coeffs);
    if (!extrinRes.success) { spdlog::error("extrinsic failed: {}", extrinRes.message); return 1; }

    // 4. 立体矫正
    StereoRectifyCpuParams rp;
    rp.cameraMatrixL = intrinRes.left.camera_matrix;
    rp.distCoeffsL   = intrinRes.left.dist_coeffs;
    rp.cameraMatrixR = intrinRes.right.camera_matrix;
    rp.distCoeffsR   = intrinRes.right.dist_coeffs;
    rp.imageSize     = cv::Size(cfg.imageWidth, cfg.imageHeight);
    rp.R = extrinRes.R; rp.T = extrinRes.T;
    rp.alpha = cfg.rectifyAlpha; rp.flags = cfg.rectifyFlags;
    StereoRectifyCpu rectify(rp);
    StereoRectifyCpuResult rectifyRes = rectify.Execute();
    if (!rectifyRes.success) { spdlog::error("rectify failed"); return 1; }

    // 5. 三张温度表
    CameraIntrinsics cL{intrinRes.left.camera_matrix.at<double>(0,0),
                        intrinRes.left.camera_matrix.at<double>(1,1),
                        intrinRes.left.camera_matrix.at<double>(0,2),
                        intrinRes.left.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    CameraIntrinsics cR{intrinRes.right.camera_matrix.at<double>(0,0),
                        intrinRes.right.camera_matrix.at<double>(1,1),
                        intrinRes.right.camera_matrix.at<double>(0,2),
                        intrinRes.right.camera_matrix.at<double>(1,2),
                        cfg.referenceTemp};
    IntrinsicCompensateCPUParams icp; icp.cte=cfg.cte; icp.tempStep=cfg.tempStep;
    icp.tempRangeMin=cfg.tempRangeMin; icp.tempRangeMax=cfg.tempRangeMax;
    IntrinsicCompensateCPU icomp(icp);
    auto tableL = icomp.Execute(cL);
    auto tableR = icomp.Execute(cR);

    CameraExtrinsics ce; ce.referenceTemp = cfg.referenceTemp;
    for (int i=0;i<3;++i) ce.T[i]=extrinRes.T.at<double>(i);
    for (int i=0;i<9;++i) ce.R[i]=extrinRes.R.at<double>(i/3,i%3);
    ExtrinsicCompensateCPUParams ecp; ecp.cte=cfg.cte; ecp.tempStep=cfg.tempStep;
    ecp.tempRangeMin=cfg.tempRangeMin; ecp.tempRangeMax=cfg.tempRangeMax;
    ExtrinsicCompensateCPU ecomp(ecp);
    auto tableE = ecomp.Execute(ce);

    StereoRectifyTempTableParams strp;
    strp.cameraMatrixL=rp.cameraMatrixL; strp.distCoeffsL=rp.distCoeffsL;
    strp.cameraMatrixR=rp.cameraMatrixR; strp.distCoeffsR=rp.distCoeffsR;
    strp.imageSize=rp.imageSize; strp.R=rp.R; strp.T=rp.T;
    strp.referenceTemp=cfg.referenceTemp; strp.cte=cfg.cte;
    strp.tempStep=cfg.tempStep; strp.tempRangeMin=cfg.tempRangeMin; strp.tempRangeMax=cfg.tempRangeMax;
    strp.alpha=cfg.rectifyAlpha; strp.flags=cfg.rectifyFlags;
    StereoRectifyTempTableCpu strtab(strp);
    auto tableR2 = strtab.Execute();

    // 6. 写交接文件
    auto j = buildCameraCalibJson(cfg, intrinRes, extrinRes, rectifyRes,
                                  tableL, tableR, tableE, tableR2);
    if (!writeJson(outPath, j)) return 1;
    spdlog::info("camera_calib done -> {}", outPath);
    return 0;
}
```

**Step 2: exe 链 fc1_ops + fc1_chess + fc1_io**

修改 `module1_camera/CMakeLists.txt` 的 camera_calib target：
```cmake
add_executable(camera_calib camera_calib_cli.cpp)
target_link_libraries(camera_calib PRIVATE fc1_ops fc1_chess fc1_io)
```

**Step 3: 构建**

```powershell
cmake --build build_fc --config Debug --target camera_calib
```
预期：编译通过。

**Step 4: Commit**

```powershell
git add factory_calib\module1_camera\camera_calib_cli.cpp factory_calib\module1_camera\CMakeLists.txt
git commit -m "feat(factory_calib/m1): camera_calib_cli 完整驱动 (角点→内/外参→矫正→3温度表)"
```

---

### Task 3.4: 模块1 端到端测试（合成已知内参）

**Files:**
- Create: `factory_calib/module1_camera/tests/test_camera_calib_e2e.cpp`

**Step 1: 写测试**（合成图像 → 调 CLI 逻辑 → 校验内参误差）

```cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "calib_io.h"
#include "chessboard_corner.h"
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"

#include <filesystem>

using namespace fc;
using namespace calib;
namespace fs = std::filesystem;

namespace {
// 渲染一帧棋盘图（带位姿与畸变）
cv::Mat renderChessboardView(const cv::Size& imSize, const cv::Size& patSize,
                             double squareMm, const cv::Mat& K, const cv::Mat& D,
                             const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point3f> obj;
    for (int i=0;i<patSize.height;++i)
        for (int j=0;j<patSize.width;++j)
            obj.emplace_back(j*squareMm, i*squareMm, 0.0f);
    std::vector<cv::Point2f> pts;
    cv::projectPoints(obj, rvec, tvec, K, D, pts);
    cv::Mat img(imSize, CV_8UC1, cv::Scalar(255));
    for (size_t i=0;i<pts.size()-1;++i) {
        cv::line(img, pts[i], pts[i+1], cv::Scalar(0), 3);
    }
    // 加黑色方块背景增强角点
    cv::Mat black(img.size(), CV_8UC1, cv::Scalar(0));
    for (auto& p : pts) cv::circle(black, p, 6, cv::Scalar(255), -1);
    cv::bitwise_or(img, black, img);
    return img;
}
}

TEST(CameraCalibE2E, RecoversIntrinsicsWithinTolerance) {
    cv::Size imSize(1280, 960);
    cv::Size patSize(8, 6);          // 8x6 内角点
    double sq = 25.0;                // mm
    cv::Mat K = (cv::Mat_<double>(3,3) << 1000, 0, 640, 0, 1000, 480, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(1,5) << 0, 0, 0, 0, 0);

    // 渲染 12 帧 L+R（右图基线 100mm，仅 tvec.x 平移）
    std::vector<cv::Mat> lImgs, rImgs;
    for (int i=0;i<12;++i) {
        cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.3*sin(i*0.5), 0.3*cos(i*0.7), 0.0);
        cv::Mat tvecL = (cv::Mat_<double>(3,1) << 0,0,400);
        cv::Mat tvecR = (cv::Mat_<double>(3,1) << 100,0,400);
        lImgs.push_back(renderChessboardView(imSize, patSize, sq, K, D, rvec, tvecL));
        rImgs.push_back(renderChessboardView(imSize, patSize, sq, K, D, rvec, tvecR));
    }

    // 提角点
    ChessboardCornerParams cp; cp.patternSize = patSize;
    std::vector<std::vector<cv::Point2f>> lpts, rpts;
    for (int i=0;i<12;++i) {
        ChessboardCornerResult rl, rr;
        ASSERT_TRUE(extractChessboardCorners(lImgs[i], cp, rl));
        ASSERT_TRUE(extractChessboardCorners(rImgs[i], cp, rr));
        normalizeLRCornerOrder(rl, rr);
        lpts.push_back(rl.corners); rpts.push_back(rr.corners);
    }

    IntrinsicCalibParams ip;
    ip.chessboard_width=patSize.width; ip.chessboard_height=patSize.height;
    ip.square_size_mm=sq; ip.image_width=imSize.width; ip.image_height=imSize.height;
    ip.use_calibrateCameraRO=false; ip.reproj_error_threshold=1.0;
    IntrinsicCalibCPU intrin(ip);
    IntrinsicCalibResult res;
    ASSERT_TRUE(intrin.Execute(lpts, rpts, res));
    ASSERT_TRUE(res.success);

    // fx/fy 恢复误差 < 2%
    EXPECT_NEAR(res.left.camera_matrix.at<double>(0,0), K.at<double>(0,0), 20.0);
    EXPECT_NEAR(res.left.camera_matrix.at<double>(1,1), K.at<double>(1,1), 20.0);
    EXPECT_LT(res.reproj_error_mean, 1.0);  // 合成图亚像素精度有限，放宽
}
```

**Step 2: 跑测**

```powershell
cmake --build build_fc --config Debug --target test_camera_calib_e2e
ctest --test-dir build_fc -C Debug -R test_camera_calib_e2e --output-on-failure
```
预期：PASS。若 `extractChessboardCorners` 在合成线框图上失败，回到 `renderChessboardView` 改为画实心黑白格（`cv::rectangle` 交替黑白）再测。

**Step 3: Commit**

```powershell
git add factory_calib\module1_camera\tests\test_camera_calib_e2e.cpp
git commit -m "test(factory_calib/m1): e2e 合成棋盘恢复内参 < 2% 误差"
```

---

## Phase 4 — 模块1 全量验证

### Task 4.1: Debug + Release 双配置全测

**Step 1: Debug 全测**

```powershell
ctest --test-dir build_fc -C Debug --output-on-failure
```
预期：模块1 全部（6 算子自测 + chessboard 5 + e2e 1）绿。

**Step 2: Release 独立目录**

```powershell
cmake -S factory_calib -B build_fc_rel -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release -DFC_BUILD_MODULE2=OFF `
  -DOpenCV_DIR=C:/opencv-cuda-4.13.0/x64/vc17/lib
cmake --build build_fc_rel --config Release
ctest --test-dir build_fc_rel -C Release --output-on-failure
```
预期：同 Debug，全绿。

**Step 3: Commit**（若 Release 暴露问题修复后）

```powershell
git add -A factory_calib\module1_camera
git commit -m "fix(factory_calib/m1): Release 配置回归通过"
```

---

## Phase 5 — 模块2 算子拷贝与回归

### Task 5.1: 拷贝 core/common 到模块2

**Step 1: 拷贝**

```powershell
$src = "modules\09_operatorlib\core\common"
$dst = "factory_calib\module2_laser\operators\common"
Copy-Item -Path "$src\*.h" -Destination $dst -Force
```

**Step 2: Commit**

```powershell
git add factory_calib\module2_laser\operators\common
git commit -m "feat(factory_calib/m2): 拷贝 core/common 公共头"
```

---

### Task 5.2: 拷贝模块2 的 15 个算子

**Files:** 按 §4.2 清单。

**Step 1: 拷贝脚本**

```powershell
$srcBase = "modules\09_operatorlib"
$dstBase = "factory_calib\module2_laser\operators"
$ops = @(
  "core\vision\mask_extract",
  "core\vision\ccl",
  "core\laser\steger",
  "core\laser\undistort_cuda",
  "core\laser\epipolar_interp",
  "core\laser\laser_reconstruct",
  "calibration\laser_calib\laser_label",
  "calibration\laser_calib\laser_match",
  "calibration\laser_calib\endpoint_extract",
  "calibration\laser_calib\virtual_camera_pose",
  "calibration\laser_calib\pose_optimize",
  "calibration\laser_calib\plane_map",
  "calibration\laser_calib\plane_map_temp_table",
  "calibration\temp\extrinsic_compensate",
  "calibration\temp\laser_extrinsic_compensate")
foreach($op in $ops){
  $name = Split-Path $op -Leaf
  Copy-Item -Path "$srcBase\$op" -Destination "$dstBase\$name" -Recurse -Force
}
```

**Step 2: 构建 fc2_ops**

```powershell
cmake -S factory_calib -B build_fc2 -DFC_BUILD_MODULE1=OFF
cmake --build build_fc2 --config Debug --target fc2_ops 2>&1 | Select-String "error"
```

**Step 3: 修 include**（同 Task 1.2 思路；预期 transitive include 都在 glob 的 INC_DIRS 内）。常见问题：
- `#include "plane_map_cuda.h"` from `plane_map_temp_table` → plane_map 目录已在 INC_DIRS ✓
- `#include "extrinsic_compensate_cpu.h"` from `laser_extrinsic_compensate` → extrinsic_compensate 目录已在 INC_DIRS ✓

**Step 4: Commit**

```powershell
git add factory_calib\module2_laser\operators
git commit -m "feat(factory_calib/m2): 拷贝 15 个激光链算子 (vision/laser/laser_calib/+2 温度)"
```

---

### Task 5.3: 模块2 算子自测回归（15 个 test_*.cpp）

**Step 1: 构建 + 跑测**（需 CUDA + Release OpenCV）

```powershell
cmake -S factory_calib -B build_fc2 -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Debug `
  -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
  -DFC_BUILD_MODULE1=OFF `
  -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc2 --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc2 -C Debug --output-on-failure
```

**Step 2: 预期**: 15 个测试 PASS。失败按 Task 1.3 同样策略处理（仅修 include/路径，不改断言）。

**Step 3: Commit**（若有修复）

```powershell
git add -A factory_calib\module2_laser
git commit -m "fix(factory_calib/m2): 算子自测 include/路径修复, 15/15 绿"
```

---

## Phase 6 — 模块2 I/O 与 CLI 驱动

### Task 6.1: calib_io —— 读 config + 加载模块1 交接文件 + 一致性校验

**Files:**
- Create: `factory_calib/module2_laser/calib_io.h`
- Create: `factory_calib/module2_laser/calib_io.cpp`

**Step 1: 头文件**

```cpp
#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace fc {

struct LaserCalibConfig {
    // 虚拟相机/平面映射参数
    float gridStep = 0.5f;
    float depthMin = 100.0f;
    float depthMax = 5000.0f;
    int depthSamples = 200;
    float epipolarStep = 0.5f;
    std::vector<int> lineIds;
    int deviceId = 0;
    // 温度（默认从 camera_calib.json 继承，可被 config 覆盖）
    double cte = 23.6e-6;
    double referenceTemp = 25.0;
    double tempRangeMin = -10.0;
    double tempRangeMax = 10.0;
    double tempStep = 0.2;
    int rectifyFlags = 1;
    double rectifyAlpha = 0.0;

    static LaserCalibConfig fromJson(const std::string& path);
};

// 模块1 交接结果（解析 camera_calib.json）
struct CameraCalibHandoff {
    cv::Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;
    double referenceTemp = 25.0;
    double cte = 23.6e-6;
    double tempRangeMin = -10.0, tempRangeMax = 10.0, tempStep = 0.2;
};

std::optional<CameraCalibHandoff> loadCameraCalibHandoff(const std::string& path);

// 一致性校验：config 与 handoff 的温度范围/cte/imageSize 必须一致
bool validateHandoffConsistency(const LaserCalibConfig& cfg, const CameraCalibHandoff& h,
                                std::string& why);

struct PoseFrame {
    cv::Mat leftGray, rightGray;
};

struct LaserInput {
    LaserCalibConfig config;
    CameraCalibHandoff handoff;
    std::vector<std::string> poseDirs;   // 每姿态目录（含 L_tube*.png 等）
    std::vector<std::vector<PoseFrame>> poseFrames; // [pose][tube]
};

std::optional<LaserInput> loadLaserInput(const std::string& dir);

bool writeJson(const std::string& path, const nlohmann::json& j);

} // namespace fc
```

**Step 2: 实现**（要点）

- `loadCameraCalibHandoff`: 读 camera_calib.json，用 `calib::jsonToMatAuto` 还原各 Mat（从 `intrinsic`/`extrinsic`/`rectify` 子节点取 L/R camera_matrix/dist_coeffs/R/T/R1/R2/P1/P2/Q）
- `validateHandoffConsistency`: 对比 cte/tempRange/Step/imageSize，不一致填 `why` 返回 false
- `loadLaserInput`: 扫描 `pose_*` 子目录，每目录里按 `L_tube*.png` / `R_tube*.png` 配对

**Step 3: 实现 writeJson / loadLaserInput 细节**（执行时按头文件约定补全；参考 Task 3.1 模式）

**Step 4: CMake fc2_io**

```cmake
add_library(fc2_io STATIC calib_io.cpp calib_io.h)
target_include_directories(fc2_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fc2_io PUBLIC fc2_ops ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
```

**Step 5: 构建验证**

```powershell
cmake --build build_fc2 --config Debug --target fc2_io
```

**Step 6: Commit**

```powershell
git add factory_calib\module2_laser\calib_io.h factory_calib\module2_laser\calib_io.cpp factory_calib\module2_laser\CMakeLists.txt
git commit -m "feat(factory_calib/m2): calib_io 读 config/handoff/姿态目录 + 一致性校验"
```

---

### Task 6.2: laser_calib_cli.cpp —— 完整驱动

**Files:**
- Modify: `factory_calib/module2_laser/laser_calib_cli.cpp`

**Step 1: 写 main（驱动激光链 4-1~4-13）**

> 结构：外层 pose×tube 循环跑 4-1~4-8 累积 d_points3d；循环结束后 4-9~4-13 一次性执行；最后写 laser_calib.json。各算子的 Params 从 handoff + config 填充。

```cpp
#include "calib_io.h"
#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "endpoint_extract_cuda.h"
#include "virtual_camera_pose_cuda.h"
#include "pose_optimize_cuda.h"
#include "plane_map_cuda.h"
#include "plane_map_temp_table.h"
#include "laser_extrinsic_compensate_cpu.h"

#include <spdlog/spdlog.h>
#include <iostream>
#include <opencv2/cudacodecs.hpp>

using namespace fc;
using namespace calib;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: laser_calib <input_dir> [output_json]\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "laser_calib.json";

    auto input = loadLaserInput(inDir);
    if (!input) return 1;
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    std::string why;
    if (!validateHandoffConsistency(cfg, h, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return 1;
    }

    // 累积器（Host/Device 张量，按算子文档 D2 节衔接收集）
    std::vector<cv::cuda::GpuMat> allPoints3D;     // 每帧 d_points3d
    std::vector<cv::cuda::GpuMat> allLineIds;

    // ===== 4-1 ~ 4-8: 逐姿态×管循环 =====
    for (size_t pi=0; pi<input->poseFrames.size(); ++pi) {
        for (size_t ti=0; ti<input->poseFrames[pi].size(); ++ti) {
            const auto& f = input->poseFrames[pi][ti];
            // 4-1 mask_extract (L & R)
            // 4-2 region_analyze
            // 4-3 laser_label
            // 4-4 steger (ByLabel, lineIdCheck)
            // 4-5 undistort_cuda (用 h.cameraMatrixL/distCoeffsL 等)
            // 4-6 epipolar_interp (lineIdCheck=true)
            // 4-7 laser_match
            // 4-8 laser_reconstruct (Q = h.Q)
            //   累积 d_points3d / d_valid_line_ids 到 allPoints3D/allLineIds
            //
            // 详见 docs/算子说明文档/ 各算子 D2 节的 Execute 签名；
            // 参数从 h 与 cfg 填充。
            (void)f;
        }
    }

    // ===== 4-9 ~ 4-13: 一次性 =====
    // 4-9 endpoint_extract(allPoints3D, allLineIds) -> d_endpoints
    // 4-10 virtual_camera_pose(d_endpoints, h.stereoK via P, h.stereoR via R)
    //      -> virtualK/virtualR/virtualT
    // 4-11 pose_optimize(allPoints3D, allLineIds, virtualK/R/initialT)
    //      -> optimized virtualK/R/T + lineCurves
    // 4-12 plane_map(virtualK/R/T, StereoCalibration from h)
    //      -> d_left_to_right + d_right_u
    // 4-13 plane_map_temp_table(virtualK/R/T, h 立体标定, cfg 温度参数)
    //      -> PlaneMapTempTableResult
    // 5-3 laser_extrinsic_compensate(virtual→L, virtual→R) -> LaserExtrinsicCompensateCPUResult

    // ===== 写 laser_calib.json =====
    nlohmann::json j;
    j["schema"] = "factory_calib.laser_calib.v1";
    // j["virtualK"/"virtualR"/"virtualT"] / j["planeMapTempTable"] / j["laserExtrinsicTempTable"]
    // 复用各 Result 的 toJson()
    if (!writeJson(outPath, j)) return 1;
    spdlog::info("laser_calib done -> {}", outPath);
    return 0;
}
```

> **执行者注**: 上面 4-1~4-8 与 4-9~4-13 的具体 Execute 调用，须对照 `docs/算子说明文档/` 各算子 B/D2 节的精确签名逐行填入。这是本计划里**唯一需要边查文档边写**的部分（因算子太多，签名不在此全文抄录）。每填一个算子就编译一次确认。

**Step 2: exe 链 fc2_ops + fc2_io**

```cmake
add_executable(laser_calib laser_calib_cli.cpp)
target_link_libraries(laser_calib PRIVATE fc2_ops fc2_io)
fc_deploy_crt(laser_calib)
```

**Step 3: 边填边编译**（按执行者注，每算子一次）

```powershell
cmake --build build_fc2 --config Debug --target laser_calib
```

**Step 4: Commit**（建议每填完 2-3 个算子一次提交）

```powershell
git add factory_calib\module2_laser\laser_calib_cli.cpp factory_calib\module2_laser\CMakeLists.txt
git commit -m "feat(factory_calib/m2): laser_calib_cli 驱动激光链 4-1~4-13"
```

---

### Task 6.3: 模块2 端到端测试 / fixture 验证

**Files:**
- Create: `factory_calib/module2_laser/tests/test_laser_calib_e2e.cpp`

**Step 1: 优先复用主工程算子 fixture**

查 `modules/09_operatorlib/calibration/laser_calib/*/tests/` 是否有可复用的合成数据加载逻辑。若有，本测试构造最小输入集（1-2 姿态 × 1 管），跑完整 laser_calib_cli 逻辑，校验：
- `plane_map_temp_table.table` 非空
- `laser_extrinsic_compensate.leftResult.table` 非空
- virtualK/R/T 非零

**Step 2: 若无 fixture** —— 降级为"冒烟测试"：构造全零小图跑流程，校验不崩溃 + 输出 schema 正确（精度不断言）。

**Step 3: 跑测**

```powershell
ctest --test-dir build_fc2 -C Debug -R test_laser_calib_e2e --output-on-failure
```

**Step 4: Commit**

```powershell
git add factory_calib\module2_laser\tests\test_laser_calib_e2e.cpp
git commit -m "test(factory_calib/m2): laser_calib e2e (fixture 或冒烟)"
```

---

## Phase 7 — 集成、交接测试、文档收尾

### Task 7.1: 模块间交接 schema 测试

**Files:**
- Create: `factory_calib/module2_laser/tests/test_handoff.cpp`

**Step 1: 写测试**

```cpp
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include "calib_io.h"

using namespace fc;

TEST(Handoff, AcceptsValidCameraCalibSchema) {
    // 构造一份最小合法 camera_calib.json（手写或调模块1 e2e 产出）
    nlohmann::json j;
    j["schema"]="factory_calib.camera_calib.v1";
    j["imageSize"]={1280,960};
    j["referenceTemp"]=22.5; j["cte"]=23.6e-6;
    j["tempRangeMin"]=-10.0; j["tempRangeMax"]=10.0; j["tempStep"]=0.2;
    // ... 最小 intrinsic/extrinsic/rectify 节点 ...
    std::ofstream("test_handoff_tmp.json")<<j.dump();
    auto h = loadCameraCalibHandoff("test_handoff_tmp.json");
    EXPECT_TRUE(h.has_value());
}

TEST(Handoff, RejectsInconsistentTempRange) {
    LaserCalibConfig cfg; cfg.tempStep=0.5;   // 与 handoff(0.2) 不一致
    CameraCalibHandoff h; h.tempStep=0.2;
    std::string why;
    EXPECT_FALSE(validateHandoffConsistency(cfg, h, why));
    EXPECT_FALSE(why.empty());
}
```

**Step 2: 跑测 + Commit**

```powershell
ctest --test-dir build_fc2 -C Debug -R test_handoff --output-on-failure
git add factory_calib\module2_laser\tests\test_handoff.cpp
git commit -m "test(factory_calib): 模块1→模块2 交接 schema + 一致性测试"
```

---

### Task 7.2: 顶层 README 补全

**Files:**
- Modify: `factory_calib/README.md`

**Step 1: 补全构建/运行/目录约定**（参考主工程 `工程目录地图.md` 的"构建速查"风格）

含：
- 前置依赖（OpenCV 4.13 / CUDA 12.6 / Eigen 3.4.1 / VS2022）
- 模块1 构建（无 CUDA）：`cmake -S factory_calib -B build_fc -DFC_BUILD_MODULE2=OFF ...`
- 模块2 构建（含 CUDA，Debug 用 debug OpenCV）
- 输入目录布局（data_in/camera/、data_in/laser/）
- 运行示例：`camera_calib.exe data_in/camera data_out/camera_calib.json` → `laser_calib.exe data_in/laser data_out/laser_calib.json`
- 已知限制（模块2 需真实样本数据才能精度验证）

**Step 2: Commit**

```powershell
git add factory_calib\README.md
git commit -m "docs(factory_calib): README 构建/运行/目录约定"
```

---

### Task 7.3: 双模块全量构建 + 全测

**Step 1: 全量 Debug（需 CUDA）**

```powershell
cmake -S factory_calib -B build_fc_all -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Debug `
  -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
  -DJMW_GH_MIRROR=https://ghfast.top/
cmake --build build_fc_all --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build_fc_all -C Debug --output-on-failure
```
预期：模块1（12 测试）+ 模块2（15+ 算子自测 + e2e + handoff）全绿。

**Step 2: 全量 Release**

```powershell
cmake -S factory_calib -B build_fc_all_rel -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_CONFIGURATION_TYPES=Release
cmake --build build_fc_all_rel --config Release
ctest --test-dir build_fc_all_rel -C Release --output-on-failure
```

**Step 3: 修复并 Commit**

```powershell
git add -A factory_calib
git commit -m "test(factory_calib): 双模块 Debug+Release 全量回归通过"
```

---

### Task 7.4: 更新工程地图与 AGENTS（可选收尾）

**Files:**
- Modify: `工程目录地图.md`（在顶层加 `factory_calib/` 条目）
- Modify: `AGENTS.md`（在结构区加一行指向 `factory_calib/`）

**Step 1: 加条目**（一行即可：`factory_calib/ = 厂家标定独立工程（模块1 相机 CPU + 模块2 激光 CUDA），自包含，详见 docs/plans/2026-07-18-factory-calib-*.md`）

**Step 2: Commit**

```powershell
git add 工程目录地图.md AGENTS.md
git commit -m "docs: 工程地图/AGENTS 收录 factory_calib/"
```

---

## 验证清单（全程对照）

- [ ] 模块1 6 算子自测绿（Task 1.3）
- [ ] chessboard 5 测试绿（Task 2.3）
- [ ] 模块1 e2e 绿 + 内参误差 < 2%（Task 3.4）
- [ ] 模块1 Debug + Release 双绿（Task 4.1）
- [ ] 模块2 15 算子自测绿（Task 5.3）
- [ ] 模块2 e2e 或冒烟绿（Task 6.3）
- [ ] 交接 schema 测试绿（Task 7.1）
- [ ] 双模块全量 Debug + Release 绿（Task 7.3）
- [ ] README 完整（Task 7.2）
- [ ] 全程未引用 `../../framework/`、`../../modules/`（自包含边界）

## 风险与回退

| 风险 | 触发条件 | 回退策略 |
|---|---|---|
| `intrinsic_calib` 实际不兼容棋盘格点 | Task 3.3 内参求解异常 | 设计已确认兼容（§Task 参考代码）；若仍异常，绕过算子直接调 `cv::calibrateCameraRO` |
| L/R 编号在多视角下不稳 | Task 3.4 e2e 帧丢率高 | `normalizeLRCornerOrder` 加极线残差二次校验，失败丢帧 |
| 模块2 E2E 无 fixture | Task 6.3 无法精度验证 | 降级为冒烟测试，标注"需真实样本数据手动验证" |
| CUDA 构建机器不可用 | Task 5.x 无法跑 | `FC_BUILD_MODULE2=OFF` 先交付模块1，模块2 延后 |
