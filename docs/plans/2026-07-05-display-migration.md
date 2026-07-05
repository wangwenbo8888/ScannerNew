# display 渲染模块迁移 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `E:\3DSCANNER260622\display\` 的渲染组件（ScannerViewer / LaserCloudRenderer / MarkerCloudRenderer + point_expand_kernel）迁入 `modules/03_rendering/display/`，并入 `mod_rendering` 库，编译链接通过。

**Architecture:** 方案 A——display 源码（已是 `calib::` + pImpl + kLogTag）原样迁入 `modules/03_rendering/display/`，加入 `mod_rendering` 静态库；该库 PUBLIC 链 `mod_operatorlib`（上游点云类型）+ OSG + CUDA-GL + OpenGL；2 个 benchmark 作 BUILD_TESTS exe。保留 `rendering.cpp` 桩。不套算子规范 §2（display 是渲染组件非流水线算子）。

**Tech Stack:** C++17 / CUDA / OSG 3.6.5 / CUDA-GL interop / OpenGL / CMake。

**设计文档:** `docs/plans/2026-07-05-display-migration-design.md`

**验证模型:** OSG 代码无法 headless 单测，验证 = **编译链接通过**（非 TDD）。每个 Task 以"构建目标 X 成功"为通过判据。

**构建环境备注:** 沿用现有 Debug build 目录（已配 Debug OpenCV / CMAKE_CONFIGURATION_TYPES=Debug / JMW_GH_MIRROR）。OSG 已装于 `C:/devlibs/osg-install`。

---

### Task 1: 迁入 display 源码

**Files:**
- Create: `modules/03_rendering/display/` 下 9 个源文件（自 `E:\3DSCANNER260622\display\` 原样复制）

**Step 1: 创建目录并复制源码**

```powershell
$src='E:\3DSCANNER260622\display'
$dst='E:\JEAMMWARE260705\modules\03_rendering\display'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\scanner_viewer.h","$src\scanner_viewer.cpp",
          "$src\laser_cloud_renderer.h","$src\laser_cloud_renderer.cpp",
          "$src\marker_cloud_renderer.h","$src\marker_cloud_renderer.cpp",
          "$src\point_expand_kernel.cu","$src\point_expand_kernel.h",
          "$src\point_expand_math.h",
          "$src\display_benchmark.cpp","$src\marker_cloud_benchmark.cpp" -Destination $dst
```

**Step 2: 验证 11 个文件就位**

```powershell
(Get-ChildItem 'E:\JEAMMWARE260705\modules\03_rendering\display' -File).Count
```
Expected: `11`

**Step 3: 抽查命名空间确为 calib::**

```powershell
Select-String -Path 'E:\JEAMMWARE260705\modules\03_rendering\display\*.h' -Pattern 'namespace calib' -List
```
Expected: 3 个头文件各匹配一处。

---

### Task 2: 改写 03_rendering/CMakeLists.txt

**Files:**
- Modify: `modules/03_rendering/CMakeLists.txt`（整体替换）

**Step 1: 用以下内容整体替换 `modules/03_rendering/CMakeLists.txt`**

```cmake
# mod_rendering：渲染业务模块
#   - rendering.cpp：业务编排桩（Scanner::rendering，未来实现）
#   - display/：渲染组件（calib::，迁自 3DSCANNER260622/display；OSG + CUDA-GL interop）
add_library(mod_rendering STATIC
    rendering.cpp
    display/scanner_viewer.cpp
    display/laser_cloud_renderer.cpp
    display/marker_cloud_renderer.cpp
)

target_include_directories(mod_rendering PUBLIC
    ${CMAKE_SOURCE_DIR}                  # 工程根
    ${CMAKE_CURRENT_SOURCE_DIR}          # 本模块根
    ${CMAKE_CURRENT_SOURCE_DIR}/display  # display 组件互引（scanner_viewer.h → laser_cloud_renderer.h 等）
)

target_link_libraries(mod_rendering PUBLIC
    fw_ui fw_algorithm fw_data           # rendering.cpp 桩的业务层依赖（保留）
    mod_operatorlib                      # 上游：calib::LaserCloudFuseDeviceContext / MarkerCloudPoint
                                        #   其 PUBLIC include 自动传播 core/marker/... 解析路径
)

# === OSG（本模块专属依赖，不上抛全局）===
set(OSG_ROOT "C:/devlibs/osg-install" CACHE PATH "OSG installation directory")
target_include_directories(mod_rendering PUBLIC ${OSG_ROOT}/include)
target_link_libraries(mod_rendering PUBLIC
    ${OSG_ROOT}/lib/osg.lib
    ${OSG_ROOT}/lib/osgDB.lib
    ${OSG_ROOT}/lib/osgViewer.lib
    ${OSG_ROOT}/lib/osgGA.lib
    ${OSG_ROOT}/lib/osgUtil.lib
    ${OSG_ROOT}/lib/OpenThreads.lib
)

# === CUDA 展点核 + CUDA-GL interop ===
if(BUILD_CUDA)
    target_sources(mod_rendering PRIVATE display/point_expand_kernel.cu)
    target_include_directories(mod_rendering PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(mod_rendering PRIVATE CUDA::cudart)
    target_compile_definitions(mod_rendering PUBLIC BUILD_CUDA=1)
    set_source_files_properties(display/point_expand_kernel.cu PROPERTIES
        COMPILE_DEFINITIONS "FMT_UNICODE=0")
endif()

# === OpenGL（Windows 运行时）===
if(WIN32)
    target_link_libraries(mod_rendering PUBLIC opengl32)
endif()

if(MSVC)
    target_compile_options(mod_rendering PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

# === benchmark：验证渲染路径可跑（OSG 需交互式 GPU，手动运行）===
if(BUILD_TESTS)
    add_executable(marker_cloud_benchmark display/marker_cloud_benchmark.cpp)
    target_link_libraries(marker_cloud_benchmark PRIVATE mod_rendering)
    set_target_properties(marker_cloud_benchmark PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=$<SHELL_PATH:${OSG_ROOT}/bin>;%PATH%")
    if(BUILD_CUDA)
        add_executable(display_benchmark display/display_benchmark.cpp)
        target_link_libraries(display_benchmark PRIVATE mod_rendering)
        set_target_properties(display_benchmark PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=$<SHELL_PATH:${OSG_ROOT}/bin>;%PATH%")
    endif()
endif()
```

**Step 2: 重新配置 CMake**

```powershell
cmake -S E:\JEAMMWARE260705 -B E:\JEAMMWARE260705\build -G "Visual Studio 17 2022" -A x64 `
  -DOpenCV_DIR=C:/opencv-cuda-4.13.0-debug/x64/vc17/lib `
  -DCMAKE_CONFIGURATION_TYPES=Debug `
  -DJMW_GH_MIRROR=https://ghfast.top/
```
Expected: `-- Configuring done` / `-- Generating done`，无 error。

**Step 3: 排查（如配置报 `CUDA::cudart` not found）**

若 `CUDA::cudart` 找不到，把 Task2 Step1 中 `target_link_libraries(mod_rendering PRIVATE CUDA::cudart)` 替换为源端写法：
```cmake
target_link_directories(mod_rendering PRIVATE ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(mod_rendering PRIVATE cudart)
```

---

### Task 3: 编译验证 mod_rendering + benchmark

**Step 1: 构建 mod_rendering**

```powershell
cmake --build E:\JEAMMWARE260705\build --config Debug --target mod_rendering
```
Expected: `mod_rendering.vcxproj -> ...\mod_rendering.lib`，无 `error C` / `fatal error` / `LNK2019`。

**Step 2: 排查编译错误（常见）**

- `cannot open source file "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"`：mod_operatorlib 的 PUBLIC include 未传播 → 确认 `target_link_libraries(mod_rendering PUBLIC mod_operatorlib)` 存在。
- `cuda_gl_interop.h` not found：确认 `BUILD_CUDA=ON` 且 `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}` 在 include 中。
- 中文注释 mojibake 导致编码警告：非错误，忽略（或后续清理）。

**Step 3: 构建 2 个 benchmark**

```powershell
cmake --build E:\JEAMMWARE260705\build --config Debug --target marker_cloud_benchmark display_benchmark
```
Expected: 两个 `.exe` 生成，无链接错误。

**Step 4: 验证 exe 生成**

```powershell
Get-ChildItem 'E:\JEAMMWARE260705\build\modules\03_rendering\Debug\*.exe' | Select-Object Name
```
Expected: 含 `marker_cloud_benchmark.exe` 与 `display_benchmark.exe`。

---

### Task 4: 文档收尾

**Files:**
- Modify: `docs/算子说明文档/算子目录.md`（display 节加落点）
- Modify: `docs/算子说明文档/display/{scanner_viewer,laser_cloud_renderer,marker_cloud_renderer}-*.md`（文首补注）
- Modify: `工程目录地图.md`（display 状态更新）

**Step 1: 算子目录.md 的 display 节**

在 `## display/ — 显示渲染模块` 节首补一行：
```
> 实现位置：modules/03_rendering/display/（并入 mod_rendering）。属渲染组件（非流水线算子，不套算子规范 §2）。
```

**Step 2: 3 份 display 文档文首补注**

每份 `display/<op>/<op>-*.md` 的标题下blockquote补：
```
> 实现位置：`modules/03_rendering/display/`。本组件为渲染组件（OSG/CUDA-GL），非流水线算子，不适用算子规范 §2 三元组/Execute 契约。
```

**Step 3: 工程目录地图.md**

把 modules 表中 03_rendering 状态由"桩（display 算子目标落点，未迁入）"改为"✅ display 渲染组件已迁入（mod_rendering）"。

---

### Task 5: 全量构建回归

**Step 1: 全量 Debug 构建**

```powershell
cmake --build E:\JEAMMWARE260705\build --config Debug
```
Expected: 全部目标（含新增 mod_rendering 源 + 2 benchmark）构建成功，无 error。

**Step 2: ctest 回归（确认 42/42 不变）**

```powershell
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir E:\JEAMMWARE260705\build -C Debug
```
Expected: `100% tests passed, 0 tests failed out of 42`（display 未加 gtest，用例数不变；确认无回归）。

---

## 完成判据
- [ ] `modules/03_rendering/display/` 11 源文件就位，命名空间 `calib::`
- [ ] `mod_rendering` 含 display 源码，链 mod_operatorlib + OSG + CUDA-GL + OpenGL
- [ ] `mod_rendering` + `marker_cloud_benchmark` + `display_benchmark` 编译链接通过
- [ ] 文档 3 处更新（算子目录 / display 文档 / 工程目录地图）
- [ ] 全量 Debug 构建成功；ctest 42/42 无回归

## 备注
- benchmark 实际运行需交互式 GPU + OSG DLL 在 PATH（手动跑，非自动化）
- 提交节奏由用户控制（本会话不自动 commit）
