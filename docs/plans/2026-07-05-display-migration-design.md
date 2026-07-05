# display 渲染模块迁移设计

> **日期**: 2026-07-05
> **源**: `E:\3DSCANNER260622\display\`（scanner_display 静态库）
> **目标**: `E:\JEAMMWARE260705\modules\03_rendering\`
> **状态**: 设计已确认，待实施

---

## 1. 背景

JEAMMWARE 工程从 3DSCANNER 继承。算子库（`modules/09_operatorlib`）已迁入 core/calibration/scanning 全部算子，但 **display 渲染模块未迁入**——`docs/算子说明文档/display/` 下 3 份文档已写好（描述"该有什么"），却无一行对应代码；`modules/03_rendering` 仅是空桩。

源端 `3DSCANNER260622/display/` 代码完整存在，本次将其迁入。

## 2. 关键事实（影响设计）

1. **display 不是流水线算子**：源端 3 个组件（`ScannerViewer`/`LaserCloudRenderer`/`MarkerCloudRenderer`）+ 1 个 CUDA 核（`point_expand_kernel`）没有 `Execute()/Params/Result` 三元组，而是**应用层渲染组件**（Viewer 的 `init/frame/done` 渲染循环、Renderer 的 `update/flush/getDrawable`）。故算子规范 §2（三元组/Execute）不适用，仅 §7 工程集成规范适用。
2. **源码约定已与工程吻合**：`namespace calib`、pImpl、`kLogTag`（`"Display-ScannerViewer"` 等）——迁移改动小。
3. **依赖**：OSG 3.6.5（`C:/devlibs/osg-install`）、CUDA-GL interop（`cuda_gl_interop.h` + `cudart`）、OpenGL（`opengl32`）。上游消费 operatorlib 的 `LaserCloudFuseDeviceContext`（laser_cloud_fuse_cuda）与 `MarkerCloudPoint`（marker_cloud_fuse_cpu）。
4. **OSG 已装**（环境配置汇总.md 第十节），运行期 DLL 在 `C:/devlibs/osg-install/bin`（第五节 PATH）。

## 3. 设计决策（已与用户确认）

| 决策 | 选择 | 理由 |
|---|---|---|
| 落点 | `modules/03_rendering/display/` | display 属渲染业务层，非算法算子；不放 09_operatorlib |
| 命名空间 | 保留 `calib::` | 源码已是 calib::，且消费的 upstream 类型都在 calib::，零跨命名空间限定 |
| 库结构（方案 A） | 源码并入 `mod_rendering` | 符合"一模块一库"约定；保留 `rendering.cpp` 桩供未来 `Scanner::rendering` 业务编排 |
| benchmark | 一并迁（2 个 exe） | 验证渲染路径可跑 |

## 4. 文件布局

```
modules/03_rendering/
├── CMakeLists.txt          ← 改写
├── rendering.cpp/.h        ← 保留桩（Scanner::rendering，未来业务编排）
├── README.md               ← 更新
└── display/                ← 新增，源码自 3DSCANNER260622/display 原样迁入
    ├── scanner_viewer.h/.cpp
    ├── laser_cloud_renderer.h/.cpp
    ├── marker_cloud_renderer.h/.cpp
    ├── point_expand_kernel.cu/.h
    ├── point_expand_math.h
    ├── display_benchmark.cpp        (BUILD_TESTS + BUILD_CUDA → exe)
    └── marker_cloud_benchmark.cpp   (BUILD_TESTS → exe)
```

## 5. CMake 接线（03_rendering/CMakeLists.txt 改写要点）

- `mod_rendering` 源文件追加：3 个渲染组件 `.cpp`；CUDA 时追加 `point_expand_kernel.cu`
- **PUBLIC 链 `mod_operatorlib`**：获取上游 `LaserCloudFuseDeviceContext`/`MarkerCloudPoint` 类型；mod_operatorlib 的 PUBLIC include（09_operatorlib 根 + 各算子自目录）自动传播，故 `marker_cloud_renderer.h` 的 `#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"` 无需改动即可解析
- `target_include_directories` 追加 `display/`（组件互引 `scanner_viewer.h → laser_cloud_renderer.h` 等）；CUDA 时追加 `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}`（`cuda_gl_interop.h`）
- **OSG**（本模块专属依赖，不上抛全局）：`OSG_ROOT=C:/devlibs/osg-install` CACHE PATH；链 osg/osgDB/osgViewer/osgGA/osgUtil/OpenThreads
- **CUDA-GL**：链 `cudart`
- **OpenGL**：`if(WIN32) target_link_libraries(... opengl32)`
- `/bigobj`（CXX + CUDA，对齐工程约定）
- benchmark exe：`VS_DEBUGGER_ENVIRONMENT` 注入 `${OSG_ROOT}/bin` 到 PATH，便于调试运行
- 保留原 `fw_ui/fw_algorithm/fw_data` 链接（rendering.cpp 桩的业务层依赖）

## 6. 源码适配（结论：几乎零改动）

- ✅ `namespace calib` / pImpl / `kLogTag` 全保留
- ✅ 上游类型经 mod_operatorlib PUBLIC 传播解析，include 无需改
- ⚠️ 个别 `.cpp/.h` 中文注释若为 mojibake，修编码（不影响编译，可选清理）
- 不套算子规范 §2 三元组——遵 §7 工程集成即可

## 7. 验证策略

- OSG/Viewer 代码需 GL 上下文，**无法 headless 单测**（源端亦无 gtest，仅 benchmark）
- 迁移成功判据：**`mod_rendering` + `marker_cloud_benchmark` + `display_benchmark` 全部编译链接通过**
- benchmark 实际运行需交互式 GPU + OSG DLL 在 PATH（手动跑，非自动化测试）
- 不新增 gtest 单测（与源端一致）

## 8. 文档收尾

- `docs/算子说明文档/display/` 3 份保留（前期审查确认 API 与源码吻合），仅在文首补注"实现在 `modules/03_rendering/display/`，属渲染组件非流水线算子"
- `docs/算子说明文档/算子目录.md` 的 display 节加落点注释

## 9. 风险

| 风险 | 缓解 |
|---|---|
| OSG 头文件与 MSVC v144 / C++17 兼容性 | OSG 已在 3DSCANNER 同环境验证过；先编译验证 |
| CUDA-GL interop 需要 CUDA toolkit 头 | 已装（CUDA 12.6），include 路径接线即可 |
| 源码 mojibake 注释 | 不影响编译；必要时单独清理 |
| mod_operatorlib 在 03 之后才 add_subdirectory | CMake 允许 target_link_libraries 前向引用（目标在配置期末存在即可）|

## 10. 实施结果（2026-07-05 完成）

迁移完成：`mod_rendering` + `display_benchmark` + `marker_cloud_benchmark` 全部编译链接通过；全量 Debug 构建成功；ctest 42/42 无回归（display 不加 gtest）。

**CMake 实际调整（相对 §5）**：`CUDA::cudart` 需 `find_package(CUDAToolkit REQUIRED)`（`enable_language(CUDA)` 不创建 `CUDA::` 目标）——已在 `if(BUILD_CUDA)` 内补加。CUDAToolkit 找到 v12.4（项目 nvcc pin 12.6），CUDA 12.x minor 兼容，无碍。

**迁入中修复的 3 处源码 bug**（正是 display 此前迁移失败的根因；均为命名空间错用）：

| 文件 | 源码 bug | 修复 |
|---|---|---|
| `display/scanner_viewer.h` | 在 `namespace calib {}` 内做 `namespace osg/osgViewer { class ...; }` 前向声明 → 创建 `calib::osg` 嵌套命名空间，遮蔽全局 `::osg`，导致 .cpp 里 `osg::Shader::VERTEX` 等全部解析失败 | 删除该无用嵌套前向声明（公开 API 不暴露 OSG 类型，pImpl 为 opaque） |
| `display/point_expand_kernel.cu` | 顶部用 `using namespace calib;` 而非 `namespace calib {}` 包裹 → `launchExpandLaserPoints` 定义落全局命名空间，与头文件声明的 `calib::launchExpandLaserPoints` 不匹配 → LNK2019 | 改为 `namespace calib { ... }` 包裹整个文件 |
| `display/{marker_cloud_benchmark,display_benchmark}.cpp` | 缺 `using namespace calib;`，在全局作用域引用 `MarkerCloudRenderer`/`expandMarkerPointCPU`/`MarkerCloudFuseCPU` 等 `calib::` 类型 | 补 `using namespace calib;`（与 scanner_viewer.cpp 一致） |

**运行时提醒**：benchmark 实际运行需交互式 GPU + OSG DLL 在 PATH（`C:/devlibs/osg-install/bin`），手动跑、非自动化测试。
