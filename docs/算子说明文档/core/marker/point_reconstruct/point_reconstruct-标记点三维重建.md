# 标记点法线和中心三维重建

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-11 |
| 中文名称 | 标记点法线和中心三维重建 |
| 英文目录名 | point_reconstruct |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ② 整像素/几何类（~0.05 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 边缘匹配结果 | EdgeMatchCPUResult 或 vector\<cv::Point2f\> 左+右 |
| **输入②** | 分组 ID | groupIds |
| **输入③** | 中心匹配关系 | centerMatches |
| **输出** | 三维重建结果 | PointReconstructCPUResult (markerResults: vector\<MarkerReconstructResult\>, 含 leftEllipseIdx, rightEllipseIdx, reconstructedPoints, centerX/Y/Z, normalX/Y/Z, planeFit, circleFit, validPlane, validCircle; statistics: PointReconstructStats) |

## C. 算法

**核心流程**：

1. 构建投影矩阵 P1, P2
2. cv::triangulatePoints 批量三角测量
3. 重投影误差过滤
4. SVD 拟合平面
5. 投影到平面建立 2D 局部坐标
6. Taubin 圆拟合
7. 还原 3D 圆心 + 法线

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::triangulatePoints | 三角测量 |
| cv::SVD::compute | 平面拟合 |
| 自定义 Taubin 圆拟合 | 3D 圆拟合 |

## D. 依赖

**上下游算子**：

```
edge_match → [point_reconstruct] → frame_fuse
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| P1/P2 | 默认自行构建（从 fx/fy/cx/cy + R + T）；**调用 `SetProjectionMatrices(P1,P2,Q)` 后改用外部矫正矩阵**（来自 06 输出或 `stereo_rectify_temp_table`），消除与 06 的口径不一致 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/zitai_result_types.h` | 姿态流程共享结果类型（MarkerReconstructResult, PlaneFitResult 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| edge_match | matchedPairs | 按 marker 分组后三角测量 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| markerResults | frame_fuse | positions + normals | 两帧间配准 | 必用 |

## E. 架构

**文件结构**：

```
point_reconstruct/
├── point_reconstruct_cpu.h
├── point_reconstruct_cpu.cpp
└── tests/
    └── test_point_reconstruct_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | PointReconstructCPU (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | PointReconstructCPUParams |
| 结果结构体 | PointReconstructCPUResult |
| 预热方法 | Warmup(maxMarkerCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 统计方法 | GetStatistics() / ResetStatistics() |
| 日志标签 | "11-PointReconstructCPU" |

**ReconstructedPoint3D 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| x | double | 三维 x 坐标 |
| y | double | 三维 y 坐标 |
| z | double | 三维 z 坐标 |
| leftU | double | 左图重投影 u 坐标 |
| leftV | double | 左图重投影 v 坐标 |
| rightU | double | 右图重投影 u 坐标 |
| rightV | double | 右图重投影 v 坐标 |
| reprojError | float | 重投影误差 |

**PlaneFitResult 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| nx | double | 平面法向量 x 分量 |
| ny | double | 平面法向量 y 分量 |
| nz | double | 平面法向量 z 分量 |
| d | double | 平面方程常数项 |
| centroidX | double | 质心 x 坐标 |
| centroidY | double | 质心 y 坐标 |
| centroidZ | double | 质心 z 坐标 |
| fitError | double | 平面拟合误差 |
| singularValues[3] | double[3] | SVD 平面拟合的三个奇异值，反映平面拟合的几何稳定性 |

**CircleFitResult 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| centerLocalX | double | 局部坐标系圆心 x |
| centerLocalY | double | 局部坐标系圆心 y |
| radius | double | 拟合圆半径 |
| fitError | double | 圆拟合误差 |
| centerX | double | 三维圆心 x 坐标 |
| centerY | double | 三维圆心 y 坐标 |
| centerZ | double | 三维圆心 z 坐标 |

**PointReconstructStats 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| totalTimeMs | double | 总耗时（毫秒） |
| triangulateTimeMs | double | 三角测量耗时（毫秒） |
| planeFitTimeMs | double | 平面拟合耗时（毫秒） |
| circleFitTimeMs | double | 圆拟合耗时（毫秒） |
| projectionTimeMs | double | 投影耗时（毫秒） |
| totalMarkerPairs | size_t | 标记点对总数 |
| validMarkerCount | size_t | 有效标记点数 |
| totalReconstructedPoints | size_t | 重建点总数 |
| avgReprojError | float | 平均重投影误差 |
| avgPlaneFitError | double | 平均平面拟合误差 |
| avgCircleFitError | double | 平均圆拟合误差 |
| avgRadius | double | 平均半径 |
| radiusStd | double | 半径标准差 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (calib3d, core) | >= 4.x | 三角测量、SVD |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| fxLeft/fyLeft | double | 0 | >0 | 左相机内参（焦距） |
| cxLeft/cyLeft | double | 0 | — | 左相机内参（主点） |
| fxRight/fyRight | double | 0 | >0 | 右相机内参（焦距） |
| cxRight/cyRight | double | 0 | — | 右相机内参（主点） |
| R | Matx33d | eye | 旋转矩阵 | 左→右旋转 |
| T | Vec3d | (0,0,0) | 平移向量 | 左→右平移 |
| minPointsForPlaneFit | int | 6 | >= 3 | 平面拟合最少点数 |
| minPointsForCircleFit | int | 6 | >= 3 | 圆拟合最少点数 |
| maxReprojError | double | 2.0 | > 0 | 最大重投影误差 |
| maxMarkerCount | size_t | 1000 | > 0 | 最大标记点数 |
| collectStatistics | bool | true | — | 是否收集统计信息 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 点数要求 | 每组边缘点需 >= minPointsForPlaneFit 才能拟合 |
| 重投影过滤 | 超过 maxReprojError 的点被剔除 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 成功 | 所有 marker 重建成功 |
| Warning | 部分失败 | 部分 marker 未成功重建 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 点数不足 | 跳过该 marker，记录 Warning |
| 三角测量退化 (w≈0) | 该点设为原点，reprojError=9999，后续被重投影过滤 |
| 重投影误差过大 | 过滤异常点（单次过滤，无重试） |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | Taubin 拟合退化 | minPointsForCircleFit 保证最低点数 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
