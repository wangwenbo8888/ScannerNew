# AGENTS.md — JEAMMWARE 工程索引

> **快照日期**: 2026-07-06 ｜ **阶段**: 骨架搭建中
> **给 AI**: 3D 扫描仪产品软件。架构 = 5 层（framework/）+ 11 模块（modules/）。读 `docs/architecture/software-architecture-skeleton-design.md` 取设计。
> **环境**: 继承 E:\3DSCANNER260622（C++17/MSVC v144/CUDA 12.6/OpenCV 4.13/Eigen）。

## 构建/测试
> ⚠ **现有 `build/` 是 Debug-only**（`CMAKE_CONFIGURATION_TYPES=Debug` + Debug OpenCV `C:/opencv-cuda-4.13.0-debug`）。直接跑 Release 命令会报 `MSB8013` / `Test not available in Release`。

**Debug（现有 build/，立即可跑，实测 42/42 绿）**：
```powershell
cmake --build build --config Debug
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH   # Debug OpenCV 运行期
ctest --test-dir build -C Debug --output-on-failure
```
**Release（需另建独立目录，不与 Debug 串味）**：
```powershell
cmake -S . -B build-rel -G "Visual Studio 17 2022" -A x64 -DCMAKE_CONFIGURATION_TYPES=Release
cmake --build build-rel --config Release
ctest --test-dir build-rel -C Release --output-on-failure
```

## 结构
- `framework/` = 层契约（9 库：fw_common[STATIC] + fw_ui/workflow/service/algorithm/data/hal/infra/crosscut[INTERFACE]）
- `modules/` = 11 业务模块（01-11）；其中 **`09_operatorlib`（全部算子，库 `mod_operatorlib`，命名空间 `calib::`）** 与 **`03_rendering`（display 渲染组件，库 `mod_rendering`）** 已迁入实现，其余 9 个为空桩
- `sdk/` = 接入端 B ｜ `app/` = exe 入口
- 命名空间：framework 层 `Scanner::`（+ 子空间 algorithm/crosscut/data/hal/infra/service/ui/workflow）；迁移算子（`modules/09_operatorlib`，原 3DSCANNER）保留 `calib::`

## 现状
骨架阶段：framework 层契约桩 + 9 个业务模块空桩。**已迁入实现**：`09_operatorlib`（core/calibration/scanning 全部算子，单库 `mod_operatorlib`，Debug ctest 42/42 绿）、`03_rendering`（display 渲染组件：scanner_viewer / laser_cloud_renderer / marker_cloud_renderer + point_expand_kernel，并入 `mod_rendering`，依赖 OSG + CUDA-GL interop）。算子规范见 `算子规范.md`，逐算子说明见 `docs/算子说明文档/`，目录地图见 `工程目录地图.md`，**开发计划与进程跟踪见 `开发计划与进程跟踪.md`**。其余各层实现待后续（见设计稿 §8）。
