# 激光平面映射表生成

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-12 |
| 中文名称 | 激光平面映射表生成 |
| 英文目录名 | plane_map |
| 运行平台 | CUDA (Thrust) + CPU (Eigen) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_virtual_pixels（含u/v/lineId） | CV_32FC3 |
| **输入②** | virtualK, virtualR, virtualT | Matx33d / Vec3d |
| **输入③** | calib::StereoCalibration | 结构体 |
| **输出** | PlaneMapResult 含 d_left_to_right(uL/vL/uR/lid), d_right_u, lineStats, totalPairs | CV_32FC4 / CV_32FC1 |

## C. 算法

**核心流程**：

```
Step 1: 生成虚拟像素网格 (VirtualPixelGenerator)
Step 2: [Projective方法] 对每个虚拟像素，沿射线在[depthMin,depthMax]范围内采样depthSamples个深度点，投影到左右矫正相机，保留在FOV内的(uL,vL,uR)对，按gridStep量化
Step 3: [FundamentalMatrix方法] 计算虚拟相机到左/右相机的F矩阵，沿极线扫描生成匹配对
Step 4: 候选点压缩 → GPU排序去重 → 统计每条线的匹配对范围
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| kernelProjective | Projective方法：射线采样+投影 |
| kernelFundamental | FundamentalMatrix方法：极线扫描 |
| kernelCompactCandidates | 候选点压缩 |
| kernelComputeStats | 统计每条线匹配对范围 |
| gpuSortUnique | GPU排序去重 |
| computeF_VL / computeF_VR | 计算虚拟相机到左/右相机的基础矩阵 |
| kernelExtractRightU | 提取右相机U坐标 |
| kernelMaxLineId | 查找最大线编号 |

## D. 依赖

**上下游算子**：

```
pose_optimize → 本算子 ← plane_map_temp_table（内部调用）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| calib::StereoCalibration | 立体标定结果（R1,R2,P1,P2） |
| VirtualPixelGenerator | 内部子模块，生成虚拟像素网格 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| pose_optimize | 值传递 | virtualK, virtualR, virtualT |
| 立体标定结果 | 结构体传递 | calib::StereoCalibration |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_left_to_right | plane_map_temp_table（内部调用） | PlaneMapCuda::Execute() | 被13内部实例化调用 | 必用 |
| d_right_u | plane_map_temp_table（内部调用） | PlaneMapCuda::Execute() | 被13内部实例化调用 | 必用 |

## E. 架构

**文件结构**：

```
plane_map/
├── plane_map_cuda.h
├── plane_map_cuda.cpp
├── plane_map_cuda_pimpl.h
├── plane_map_cuda_impl.cu
├── virtual_pixel_gen.h
├── virtual_pixel_gen.cu
└── tests/
    └── test_plane_map_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | PlaneMapCuda (pImpl), VirtualPixelGenerator (pImpl) |
| 核心方法 | Execute |
| 预热方法 | Warmup(numVirtualPixels, maxLineId) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | PlaneMapParams, VirtualPixelGenParams |
| 结果结构体 | PlaneMapResult |
| 日志标签 | "12-PlaneMapCuda", "12-VirtualPixelGen" |

**LineMapStats 字段**（每条激光线的映射对统计）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `lineId` | int | 激光线编号（默认 -1） |
| `numPairs` | int | 该线的匹配对数量 |
| `uMin` | float | 左相机 U 最小值 |
| `uMax` | float | 左相机 U 最大值 |
| `vMin` | float | 左相机 V 最小值 |
| `vMax` | float | 左相机 V 最大值 |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA | >= 11.0 | GPU并行计算, --extended-lambda |
| Thrust | CUDA内置 | GPU排序 |
| Eigen | >= 3.3 | 基础矩阵计算 |
| OpenCV | >= 4.x | 矩阵与数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| method | PlaneMapMethod | Projective | 0/1 | 映射方法 |
| gridStep | float | 0.5 | >0 | 网格量化步长 |
| depthMin | float | 100.0 | >0 | 最小深度（单位 mm） |
| depthMax | float | 5000.0 | >depthMin | 最大深度（单位 mm） |
| depthSamples | int | 200 | >0 | 深度采样数 |
| epipolarStep | float | 0.5 | >0 | 极线扫描步长 |
| enableTiming | bool | false | — | 计时开关 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 虚拟像素数 | >0 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型不符 / imageSize 非法 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel 启动失败 | 抛出 std::runtime_error |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 大量GPU缓冲区，复杂度高 | 显存管理复杂 |
| 🟡 中 | Projective方法depthSamples×N个候选点可能占用大量显存 | 高分辨率场景下可能OOM |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
