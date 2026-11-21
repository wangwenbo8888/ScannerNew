# 模块9 算子库（OperatorLib）

3DSCANNER 工程全部算子的机械迁入地，编进单一静态库 `OperatorLib`。

## 内容

- **来源**：`E:\3DSCANNER260622`（算子研发工程），机械拷贝，**命名空间保留 `calib::`**（未改名）
- **规模**：42 算子 + GBA（`scanning/global_optim/`）
- **范围**：
  - `core/{common,vision,laser,marker}` — 共享类型 + 双链算子
  - `calibration/{camera,laser_calib,posture,temp}` — 16 标定算子
  - `scanning/{preprocess,laser,fusion,global_optim}` — 6 扫描算子 + GBA
  - `PostProcessing/{mesh_normal_est,mesh_ball_pivot,mesh_hole_fill,mesh_smooth_sr}` — 后处理网格四族（261121 PCMP 沙盒移植，金标准对拍 6/6；含 mhf/sr 移植库与 ctest 金标准测试）
- **不含**：`pipeline/`（workflow 编排，归 modules/02_scanning 等）、`scheduler/`（infra）、bench/benchmark

## 依赖

OpenCV 4.13（+CUDA）/ Eigen 3.4.1 / CUDA 12（sm_75;86;87）/ spdlog / nlohmann_json / Ceres 2.2.0（GBA 用，BUILD_GLOBAL_OPTIM）

## 算子规范

对齐 `2026-06-21-算子规范-design.md` v1.9 三元组（Params / Result / Operator）+ pImpl 隔离 CUDA + 末参 `cv::cuda::Stream&`。
