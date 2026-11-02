# 虚拟相机外参优化（LM优化）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-11 |
| 中文名称 | 虚拟相机外参优化（LM优化） |
| 英文目录名 | pose_optimize |
| 运行平台 | CPU (Eigen Levenberg-Marquardt) + CUDA (数据传输) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_points3d | CV_32FC3 |
| **输入②** | d_valid_line_ids | CV_32SC1 |
| **输入③** | virtualK, virtualR, initialT | Matx33d / Vec3d |
| **输出** | PoseOptimizeResult 含 virtualK, virtualR, virtualT(优化后), initialT, totalReprojectionError, initialReprojectionError, numLines, lineCurves | Matx33d / Vec3d |

## C. 算法

**核心流程**：

```
Step 1: GPU→CPU数据传输
Step 2: 按line_id分组点
Step 3: 对每组: 投影到虚拟相机像素 → 二次曲线拟合 (SVD最小二乘)
Step 4: 构建 LM 优化问题: 残差 = 实际像素到二次曲线的距离
Step 5: Eigen Levenberg-Marquardt 求解最优 T (3参数)
Step 6: 数值雅可比（eps=1e-8）
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| Eigen::LevenbergMarquardt | LM非线性最小二乘优化 |
| Eigen::bdcSvd | SVD分解用于二次曲线拟合 |
| LMOptFunctor | 自定义LM优化仿函数 |
| LaserLineCurve | 二次曲线数据结构（含 lineId, coeffs[3], mainAxis, fittingError, pointCount） |

## D. 依赖

**上下游算子**：

```
laser_reconstruct + virtual_camera_pose → 本算子 → plane_map
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| virtualK/R/T | 由10算子初始化，本算子优化T |
| d_points3d, d_line_ids | 复用08算子的三维重建结果 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| laser_reconstruct | GPU显存直传 | d_points3d, d_valid_line_ids |
| virtual_camera_pose | 值传递 | 10 输出 virtualT，作为本算子的 initialT 参数传入；另传 virtualK, virtualR |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| virtualK | plane_map | Execute | 值传递 | 必用 |
| virtualR | plane_map | Execute | 值传递 | 必用 |
| virtualT | plane_map | Execute | 值传递 | 必用 |

## E. 架构

**文件结构**：

```
pose_optimize/
├── pose_optimize_cuda.h
├── pose_optimize_cuda.cpp
├── pose_optimize_cuda_pimpl.h
├── pose_optimize_cuda_impl.cu
└── tests/
    └── test_pose_optimize_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | PoseOptimizeCuda (pImpl) |
| 核心方法 | Execute |
| 预热方法 | Warmup(numPoints, maxLineId) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | PoseOptimizeParams |
| 结果结构体 | PoseOptimizeResult |
| 日志标签 | "11-PoseOptimizeCuda" |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Eigen | >= 3.3 | 线性代数与LM优化 |
| CUDA | >= 11.0 | 数据传输 |
| OpenCV | >= 4.x | 矩阵数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| maxIterations | int | 50 | >0 | LM最大迭代 |
| convergenceThreshold | double | 1e-6 | >0 | 收敛阈值 |
| minLinesForOptimize | int | 3 | >=2 | 最少线数 |
| minPointsPerLine | int | 10 | >=3 | 每线最少点数 |
| enableTiming | bool | false | — | 计时开关 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 优化自由度 | 仅优化平移向量T (3 DOF)，R和K固定 |
| 编译选项 | MSVC需/bigobj编译选项 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 改善比 <= 0.5 |
| Degraded | 降级 | 改善比 0.5~0.9 |
| Warning | 警告 | 改善比 > 0.9（几乎未改善） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| 有效线数不足 (< minLinesForOptimize) | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 数值雅可比效率低于解析雅可比 | 优化速度较慢 |
| 🟢 低 | 仅优化平移，未优化旋转和内参 | 优化自由度有限 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
