# 激光中心亚像素点集去畸变+立体矫正

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-5 |
| 中文名称 | 激光中心亚像素点集去畸变+立体矫正 |
| 英文目录名 | undistort_cuda |
| 运行平台 | CUDA |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类（与cv::undistortPoints差异<0.01像素） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 亚像素点集 | d_centerPoints (CV_32FC2 GpuMat) |
| **输入②** | 线标签 | d_line_ids (CV_32SC1 GpuMat, 可选) |
| **输出** | 矫正后点集 | UndistortPointsResult 含 d_rectifiedPoints(CV_32FC2), d_line_ids |

## C. 算法

**核心流程**：

```
Step 1: 像素坐标→归一化坐标
Step 2: 迭代反向去畸变（8参数模型: k1,k2,p1,p2,k3,k4,k5,k6, 5次迭代）
Step 3: 旋转矫正（乘以R矩阵）
Step 4: 投影到矫正后像素坐标（乘以P矩阵）
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| UndistortRectifyKernel | 自定义CUDA kernel，去畸变+矫正一体化 |

## D. 依赖

**上下游算子**：

```
steger → 本算子 → epipolar_interp
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无 | 独立实例 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| steger | shared_ptr<GpuMat> | d_centerPoints (CV_32FC2) |
| steger | shared_ptr<GpuMat> | d_line_ids (CV_32SC1, 可选) |
| **stereo_rectify_temp_table** | **SetParams(params)** | **当前温度条目的 cameraMatrix/distCoeffs/R(3×3)/P(3×4) 矫正矩阵** |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_rectifiedPoints | epipolar_interp | EpipolarInterpCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| d_line_ids | epipolar_interp | EpipolarInterpCuda::Execute() | shared_ptr<GpuMat> | 必用 |

## E. 架构

**文件结构**：

```
undistort_cuda/
├── undistort_points_cuda.h
├── undistort_points_cuda.cpp
├── undistort_points_cuda_pimpl.h
├── undistort_points_cuda_impl.cu
└── tests/
    └── test_undistort_points_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | UndistortPointsCuda (pImpl) |
| 核心方法 | Execute() (4 重载：含/不含 line_ids + Stream 组合) |
| 销毁方法 | Destroy() |
| 预热方法 | Warmup(maxPointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | UndistortPointsParams (含cameraMatrix, distCoeffs, R, P) |
| 结果结构体 | UndistortPointsResult |
| 日志标签 | "11-UndistortPointsCuda" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。
> 注：`Execute(d_points)` 无 line_ids 重载允许 `d_line_ids` 为空（向后兼容），仅校验 `d_points` 非空和类型正确。

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_core, opencv_imgproc, opencv_calib3d, opencv_cudaarithm |
| CUDA Toolkit | 12.6 | sm_75, sm_86, sm_87 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cameraMatrix | cv::Mat | 空 | 3x3 CV_32F/64F | 相机内参（**来自 `stereo_rectify_temp_table` 当前温度补偿内参**） |
| distCoeffs | cv::Mat | 空 | 4~8元素 | 畸变系数（**同上，当前温度补偿**） |
| R | cv::Mat | 空 | 3x3 | 旋转矫正矩阵（**来自 `stereo_rectify_temp_table` 当前温度条目 R1(左)/R2(右)**） |
| P | cv::Mat | 空 | 3x4 | 投影矩阵（**来自 `stereo_rectify_temp_table` 当前温度条目 P1(左)/P2(右)**） |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 畸变模型 | 支持4~8参数畸变模型 |
| warmup策略 | 按点数而非图像尺寸预分配 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 | 返回 success=true（返回空结果，qualityFlag=Normal） |
| 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel 启动失败 | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 浮点精度 | GPU 计算使用 float32，极端畸变条件下可能有亚像素级精度损失 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
