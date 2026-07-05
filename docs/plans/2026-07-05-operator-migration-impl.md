# 算子源码迁移实现计划：3DSCANNER → JEAMMWARE/modules/09_operatorlib

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 3DSCANNER 的全部算子（core/calibration/scanning 的算子部分 + global_optim/GBA）源码迁入 `E:\JEAMMWARE260705\modules\09_operatorlib\`，编进一个 `mod_operatorlib` 库，启用 OpenCV/Eigen/CUDA/Ceres，迁移算子单测，全量 build + ctest 通过。

**关键决策（已定）:**
- 命名空间：**保留 `calib::`**（机械拷贝，零改名）
- 范围：**全部算子 + GBA + tests**（不含 bench）
- 一个 `mod_operatorlib` STATIC 库（内部保留 core/calibration/scanning/global_optim 子目录组织）

**范围边界（重要）:**
- ✅ 迁：core/{common,vision,laser,marker} + calibration/算子 + scanning/{preprocess,laser,fusion,global_optim} + 各算子 tests
- ❌ 不迁：scanning/pipeline/（workflow 编排，归 modules/02_scanning）、calibration/pipeline/（同）、core/scheduler/（infra，归 framework/infra）、各 bench（perf-only）
- ❌ 不迁：3DSCANNER 的 per-operator CMakeLists（3DSCANNER 专属 target_sources(scanner_*) 模式），用 JEAMMWARE 的统一 CMake 取代

**源工程:** `E:\3DSCANNER260622` ｜ **目标:** `E:\JEAMMWARE260705\modules\09_operatorlib`

---

## Task 1: 顶层 CMake 启用算子依赖（OpenCV/Eigen/Ceres/CUDA 编译）

**Files:** Modify `E:\JEAMMWARE260705\CMakeLists.txt`

在现有 FetchContent 段后、`add_subdirectory` 段前，加：

```cmake
# === 算子依赖（算子迁入启用）===
find_package(OpenCV 4.13 REQUIRED COMPONENTS
    core imgproc calib3d cudaimgproc cudaarithm cudafilters flann)
find_package(Eigen3 3.4 REQUIRED)

# Ceres（global_optim/GBA 用；精简配置，免 SuiteSparse，用 Eigen 稀疏）
option(BUILD_GLOBAL_OPTIM "Build global optimization (GBA + Ceres)" ON)
if(BUILD_GLOBAL_OPTIM)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
    set(GFLAGS OFF CACHE BOOL "" FORCE)
    set(MINIGLOG ON CACHE BOOL "" FORCE)
    set(SUITESPARSE OFF CACHE BOOL "" FORCE)
    set(CXSPARSE OFF CACHE BOOL "" FORCE)
    set(ACCELERATESPARSE OFF CACHE BOOL "" FORCE)
    set(EIGENSPARSE ON CACHE BOOL "" FORCE)
    set(LAPACK OFF CACHE BOOL "" FORCE)
    set(SCHUR_SPECIALIZATIONS OFF CACHE BOOL "" FORCE)
    set(USE_CUDA OFF CACHE BOOL "" FORCE)  # Ceres 自带 CUDA 与项目动态 cudart 冲突
    FetchContent_Declare(ceres
        URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz)
    FetchContent_MakeAvailable(ceres)
endif()
```

**Verify:** `cmake -S . -B build -G "Visual Studio 17 2022" -A x64` → OpenCV/Eigen/Ceres 找到/拉取成功。
**不提交。**

---

## Task 2: 复制算子源码到 modules/09_operatorlib/

**目录结构（目标）:**
```
modules/09_operatorlib/
├── CMakeLists.txt              # 统一 CMake（新写，取代 per-operator CMakeLists）
├── README.md
├── core/
│   ├── common/                 # 共享类型（calib_types.h/pipeline_types.h/result.h/...）
│   ├── vision/                 # ccl, mask_extract
│   ├── laser/                  # steger, undistort_cuda, epipolar_interp, laser_reconstruct
│   └── marker/                 # image_split...marker_cloud_fuse_cpu (12 算子)
├── calibration/
│   ├── camera/ laser_calib/ posture/ temp/   # 16 算子（不含 pipeline/）
│   └── （不迁 pipeline/）
├── scanning/
│   ├── preprocess/ laser/ fusion/            # 6 算子（不含 pipeline/）
│   └── global_optim/                         # GBA（含 global_ba_cpu.* + residuals + runner）
└── tests/                      # （可选集中处；或保留各算子 tests/ 原位）
```

**Step 1: 复制源（不含 pipeline/scheduler/build/tests 顶层 + per-operator CMakeLists/bench）**

用 robocopy/Copy-Item 批量复制。**关键过滤**：排除 `pipeline\`、`build\`、`CMakeLists.txt`（per-operator）、`bench_*.cpp`。

```powershell
$src='E:\3DSCANNER260622'; $dst='E:\JEAMMWARE260705\modules\09_operatorlib'

# core/common, vision, laser, marker（不含 scheduler）
robocopy "$src\core\common" "$dst\core\common" *.h *.cpp /S /XF bench_*.cpp
robocopy "$src\core\vision" "$dst\core\vision" *.h *.cpp *.cu /S /XF bench_*.cpp /XD build
robocopy "$src\core\laser"  "$dst\core\laser"  *.h *.cpp *.cu /S /XF bench_*.cpp /XD build
robocopy "$src\core\marker" "$dst\core\marker" *.h *.cpp *.cu /S /XF bench_*.cpp /XD build

# calibration 算子（不含 pipeline）
robocopy "$src\calibration" "$dst\calibration" *.h *.cpp *.cu /S /XF bench_*.cpp /XD build pipeline

# scanning 算子 + global_optim（不含 pipeline）
robocopy "$src\scanning" "$dst\scanning" *.h *.cpp *.cu /S /XF bench_*.cpp /XD build pipeline
```

> robocopy 不复制空目录、排除 build/pipeline/bench。复制后**删除所有 per-operator CMakeLists.txt**（3DSCANNER 专属）：
> `Get-ChildItem $dst -Recurse -Filter CMakeLists.txt | Remove-Item`

**Step 2: 抽查复制结果**
- 文件数：.h ≈ 85、.cpp ≈ 111（减 bench）、.cu ≈ 17
- 无 pipeline/、无 build/、无 bench_*.cpp、无 CMakeLists.txt 残留

**Verify:** 文件数核对；无禁迁内容。
**不提交。**

---

## Task 3: 写 modules/09_operatorlib/CMakeLists.txt（统一构建）

**Files:** Create `modules/09_operatorlib/CMakeLists.txt`

策略：**glob 所有 .cpp/.cu 源**（迁移量大，glob 务实；CMake glob 在配置期生效，可接受）+ 注册每算子 test。

```cmake
# mod_operatorlib：全部算子（core/calibration/scanning/global_optim）一个库
add_library(mod_operatorlib STATIC "")
target_include_directories(mod_operatorlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # 让 #include "core/common/calib_types.h" 可解析
    ${CMAKE_CURRENT_SOURCE_DIR}/core     # 让 #include "common/calib_types.h" 可解析（3DSCANNER 风格）
    ${CMAKE_SOURCE_DIR}                  # 工程根
)

# 收集源（glob .cpp/.cu，排除 tests/bench）
file(GLOB_RECURSE OP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*.cu")
# 排除 tests/bench/example_usage
list(FILTER OP_SOURCES EXCLUDE REGEX "/tests/|bench_|example_usage|gba_dataset_runner")
target_sources(mod_operatorlib PRIVATE ${OP_SOURCES})

target_link_libraries(mod_operatorlib PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)
if(BUILD_CUDA)
    target_compile_definitions(mod_operatorlib PUBLIC BUILD_CUDA=1)
    set_source_files_properties(${OP_SOURCES} PROPERTIES COMPILE_DEFINITIONS "FMT_UNICODE=0")
endif()
if(BUILD_GLOBAL_OPTIM)
    target_sources(mod_operatorlib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/scanning/global_optim/global_ba_cpu.cpp)
    target_link_libraries(mod_operatorlib PUBLIC ceres)
endif()

# === gba_dataset_runner exe（GBA 离线工具，含 main）===
if(BUILD_GLOBAL_OPTIM)
    add_executable(gba_dataset_runner
        ${CMAKE_CURRENT_SOURCE_DIR}/scanning/global_optim/gba_dataset_runner.cpp)
    target_link_libraries(gba_dataset_runner PRIVATE mod_operatorlib ceres Eigen3::Eigen)
endif()

# === 算子单测（glob tests/test_*.cpp，每文件一个 ctest）===
if(BUILD_TESTS)
    file(GLOB_RECURSE OP_TESTS
        "${CMAKE_CURRENT_SOURCE_DIR}/core/tests/test_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/calibration/*/tests/test_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/scanning/*/tests/test_*.cpp")
    foreach(t ${OP_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE mod_operatorlib GTest::gtest_main)
        target_include_directories(${tname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR})
        add_test(NAME ${tname} COMMAND ${tname})
    endforeach()
endif()
```

> **注意**：3DSCANNER 各算子 test 用 `using namespace calib;` + 链 `scanner_core`；这里链 `mod_operatorlib`（含全部算子）+ include core 路径。test 编译期可能有个别 include 路径或符号缺失，需迭代修（见 Task 5）。

**Verify:** CMake 语法正确（configure 时不报错）。
**不提交。**

---

## Task 4: 更新 modules/09_operatorlib/ 原 CMakeLists + README

**Files:** 
- The骨架阶段建的 `modules/09_operatorlib/CMakeLists.txt`（空 mod_operatorlib）和 `operatorlib.cpp/.h` 会被 Task 3 的 CMakeLists 覆盖；删 `operatorlib.cpp/.h`（骨架占位，算子迁入后无用了）
- 更新 `modules/09_operatorlib/README.md`：说明含 42 算子 + GBA，命名空间 `calib::`，来源 3DSCANNER

**Verify:** 无骨架占位残留冲突。
**不提交。**

---

## Task 5: 构建 + 修 include/符号问题（迭代，核心风险任务）

**Step 1: configure**
```powershell
cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64
```
Expected: OpenCV/Eigen/Ceres 找到 + mod_operatorlib 目标定义 + 所有 test 目标注册。修 CMake 错误（路径/glob 模式）。

**Step 2: build mod_operatorlib**
```powershell
cmake --build E:\JEAMMWARE260705\build --config Release --target mod_operatorlib
```
预期可能失败（算子 .cpp/.cu 的 include 路径、缺失符号、CUDA 编译问题）。逐个修：
- include 路径：补 `target_include_directories`（如 calibration 算子 include `../../common/...`）
- CUDA：确认 .cu 走 nvcc、FMT_UNICODE=0
- 缺失符号：核对 OpenCV COMPONENTS 是否全（cudaimgproc 等）
- pImpl 桥接：CUDA 算子的 `_pimpl.h` + `_impl.cu` 结构需 CUDA 语言启用

**Step 3: build 全部 + tests**
```powershell
cmake --build E:\JEAMMWARE260705\build --config Release
```
修 test 链接问题（test 找不到算子符号 → 已链 mod_operatorlib 应可解；个别 test include 路径补）。

**Step 4: ctest**
```powershell
ctest --test-dir E:\JEAMMWARE260705\build -C Release --output-on-failure
```
预期：算子单测 + 框架 smoke 全绿。若个别算子 test 因数据/环境失败，记录但不阻塞（迁移成功 = 编译 + 大部分 test 绿）。

**Step 5: jeammware.exe 仍能 build + run**（算子迁入不能破框架骨架）
```powershell
cmake --build E:\JEAMMWARE260705\build --config Release --target jeammware
& E:\JEAMMWARE260705\build\app\Release\jeammware.exe
```

**Step 6: 提交**
```powershell
cd E:\JEAMMWARE260705
git add -A
git commit -m "feat(operatorlib): 迁入 3DSCANNER 全部算子(core/calib/scan 42个)+GBA 到 mod_operatorlib; 启用 OpenCV/Eigen/CUDA/Ceres; 算子单测迁入"
```

---

## 完成判据

- [ ] mod_operatorlib 编译通过（含 17 .cu CUDA 文件）
- [ ] gba_dataset_runner exe 构建（BUILD_GLOBAL_OPTIM）
- [ ] 算子单测注册到 ctest，大部分通过（个别数据/环境失败可记录）
- [ ] jeammware.exe 仍 build + run（骨架不破）
- [ ] framework_smoke 仍 100% 通过
- [ ] 命名空间保留 `calib::`（无改名）
- [ ] 已提交

---

## §99 风险与未决

- **Ceres 编译耗时**：首次 FetchContent + 编译 Ceres 较慢（~5-10min）
- **CUDA 12.4 vs 12.6**：本机 nvcc 报 12.4；3DSCANNER 用 12.6。若 .cu 编译失败需确认
- **test 数据依赖**：部分算子 test 可能依赖合成数据/特定 OpenCV 版本，迁移后个别可能失败
- **per-operator CMakeLists 的特殊处理**：个别算子（如 laser_match_scan 有 LM_ENABLE_TIMING）的 target 级定义，需在 mod_operatorlib 补 `target_compile_definitions`
- **pipeline/scheduler 不迁**：scanning/pipeline（EcoreOrchestrator）和 core/scheduler 留后续（归 modules/02_scanning 和 framework/infra）；算子不依赖它们即可编译
