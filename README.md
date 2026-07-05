# JEAMMWARE — 3D 扫描仪交付级产品软件

手持式双目多线激光 3D 扫描仪产品软件工程。架构：5 层（UI/Workflow/Service∥Algorithm/Data/HAL）+ 侧边设施 + 横切 + 部署 wrapper + SDK 接入端。

## 构建（需 VS2022 + CMake≥3.24；spdlog/json/gtest 经 FetchContent 自动拉取）
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## 架构
见 `docs/architecture/software-architecture-skeleton-design.md`。
上层架构依据：`E:\3DSCANNER260622\docs\plans\2026-07-05-交付级框架整体设计-design.md`（01.4）。

## 环境
继承 `E:\3DSCANNER260622`（C++17/MSVC v144/CUDA 12.6/OpenCV 4.13/Eigen 3.4.1）。

## 模块现状
骨架阶段：framework 层契约桩 + 9 个业务模块空桩。已迁入实现：`modules/09_operatorlib`（全部算子，`mod_operatorlib`）、`modules/03_rendering`（display 渲染组件，`mod_rendering`，OSG + CUDA-GL）。详见 `AGENTS.md` / `工程目录地图.md`。
