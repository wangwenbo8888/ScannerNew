# 激光器虚拟相机光心和初步外参求解

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-10 |
| 中文名称 | 激光器虚拟相机光心和初步外参求解 |
| 英文目录名 | virtual_camera_pose |
| 运行平台 | CPU (Eigen) + CUDA (数据传输) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_endpoints | CV_32FC3 |
| **输入②** | d_line_ids | CV_32SC1 |
| **输入③** | stereoK | Matx33d |
| **输入④** | stereoR | Matx33d |
| **输出** | VirtualCameraPoseResult 含 virtualK, virtualR, virtualT(光心位置), numLines, totalEndpoints, avgLineFittingError, avgDistToCenter, lineInlierCounts | Matx33d / Vec3d |

## C. 算法

**核心流程**：

```
Step 1: GPU→CPU数据传输
Step 2: 按line_id分组端点
Step 3: 逐线 RANSAC + PCA 拟合3D直线（自适应迭代次数）
Step 4: 解析求距离所有直线之和最小的交点: A = Σ(I-d·dᵀ), b = Σ(I-d·dᵀ)·a, C = A⁻¹·b (LDLT求解)
Step 5: 组装输出和诊断信息
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| Eigen::SelfAdjointEigenSolver | PCA主成分分析拟合3D直线方向 |
| Eigen::LDLT | 求解线性方程组求光心 |
| 自定义RANSAC+PCA | 直线拟合中的鲁棒估计 |

## D. 依赖

**上下游算子**：

```
endpoint_extract → 本算子 → pose_optimize
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| Eigen库 | 线性代数运算 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| endpoint_extract | GPU→CPU传输 | d_endpoints, d_line_ids |
| 标定-04 stereoRectify | 值传递 | stereoK, stereoR |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| virtualK | pose_optimize | Execute | 值传递 | 必用 |
| virtualR | pose_optimize | Execute | 值传递 | 必用 |
| virtualT | pose_optimize | Execute | 值传递 | 必用 |
| numLines | pose_optimize | Execute | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
virtual_camera_pose/
├── virtual_camera_pose_cuda.h
├── virtual_camera_pose_cuda.cpp
├── virtual_camera_pose_cuda_pimpl.h
├── virtual_camera_pose_cuda_impl.cu
└── tests/
    └── test_virtual_camera_pose_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | VirtualCameraPoseCuda (pImpl) |
| 核心方法 | Execute |
| 预热方法 | Warmup(numEndpoints, maxLineId) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | VirtualCameraPoseParams |
| 结果结构体 | VirtualCameraPoseResult |
| 日志标签 | "10-VirtualCameraPoseCuda" |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Eigen | >= 3.3 | 线性代数运算 |
| CUDA | >= 11.0 | 数据传输 |
| OpenCV | >= 4.x | 矩阵数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| ransacThreshold | double | 1.0 | >0 | RANSAC距离阈值（单位 mm） |
| ransacConfidence | double | 0.99 | (0,1) | RANSAC置信度 |
| ransacMaxIterations | int | 1000 | >0 | RANSAC最大迭代 |
| minLinesForSolve | int | 3 | >=2 | 求解最少线数 |
| minPointsPerLine | int | 3 | >=2 | 每线最少点数 |
| enableTiming | bool | false | — | 计时开关 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 有效激光线数 | >= minLinesForSolve |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | avgDistToCenter <= 2.0（单位 mm） |
| Degraded | 降级 | avgDistToCenter > 2.0（单位 mm） |
| Warning | 警告 | avgDistToCenter > 5.0（单位 mm） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| 有效线数不足 (< minLinesForSolve) | 返回 success=false |
| 矩阵奇异 (LDLT 求解失败) | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | RANSAC固定种子42可能不适合所有场景 | 特殊场景下直线拟合精度下降 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
