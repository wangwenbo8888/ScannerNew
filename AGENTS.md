# AGENTS.md — JEAMMWARE 工程索引

> **快照日期**: 2026-07-05 ｜ **阶段**: 骨架搭建中
> **给 AI**: 3D 扫描仪产品软件。架构 = 5 层（framework/）+ 11 模块（modules/）。读 `docs/architecture/software-architecture-skeleton-design.md` 取设计。
> **环境**: 继承 E:\3DSCANNER260622（C++20/MSVC v144/CUDA 12.6/OpenCV 4.13/Eigen）。

## 构建/测试
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## 结构
- `framework/` = 层契约（common/ui/workflow/service/algorithm/data/hal/infra/crosscut）
- `modules/` = 11 业务模块空桩（01-11）
- `sdk/` = 接入端 B ｜ `app/` = exe 入口
- 命名空间 `Scanner::`

## 现状
骨架阶段：层契约桩 + 空模块。算子迁入、各层实现待后续（见设计稿 §8）。
