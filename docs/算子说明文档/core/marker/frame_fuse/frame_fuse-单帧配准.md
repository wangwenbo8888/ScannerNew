# 单帧快速配准（刚体变换估计）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-12 |
| 中文名称 | 单帧快速配准（刚体变换估计） |
| 英文目录名 | frame_fuse |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ②（~0.05mm 位置） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 两帧标记点集（positions + normals） | MarkerPointSet × 2 |
| **输出** | 配准结果 | FrameFuseCPUResult (R: Matx33d, T: Vec3d, transform: Matx44d, rmse, normalRMSE, matchedCount, totalCorrespondences, overlapRatio, correspondences; statistics: FrameFuseStats) |

## C. 算法

**核心流程**：

1. KD-Tree 构建 + KNN 搜索
2. 自适应距离阈值
3. SPFH（简化点特征直方图）描述子计算（Darboux 框架，3 不变量角，分 bin 直方图；仅计算查询点-邻居对，非全邻居对）
4. FLANN 匹配 + Lowe 比值测试
5. 法线预过滤
6. RANSAC 3 点粗配准
7. 法线加权 SVD 精配准
8. 质量评估（RMSE, normalRMSE）

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::flann::Index | KD-Tree 构建与搜索 |
| cv::SVD::compute | 加权 SVD 配准 |
| cv::determinant | 变换矩阵验证 |
| 自定义 SPFH | 简化点特征直方图描述子（SPFH 风格，仅查询点-邻居对） |
| 自定义加权 SVD | 法线加权精配准 |
| 自定义 RANSAC | 3 点粗配准 |

## D. 依赖

**上下游算子**：

```
point_reconstruct → [frame_fuse] → pose_estimate（标定）/ laser_cloud_fuse（扫描）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| KD-Tree | 可跨帧复用（若场景不变） |
| FLANN Index | PFH 描述子索引 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/zitai_result_types.h` | 姿态流程共享结果类型（PointReconstructCPUResult 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| point_reconstruct | markerResults | 组装为 MarkerPointSet（positions + normals）— 当前帧 |
| 全局标记点集（扫描流程） | 外部参考（内存） | 作为第二份 MarkerPointSet — 配准基准（对全局标记点配准，非上一帧） |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| R, T, transform | pose_estimate（标定流程） | 刚体变换矩阵 | 直接用于姿态匹配 | 必用 |
| R, T, transform | laser_cloud_fuse（扫描流程） | 相机系→全局系刚体变换 | 驱动激光点云体素哈希融合前的坐标变换 | 必用 |

## E. 架构

**文件结构**：

```
frame_fuse/
├── frame_fuse_cpu.h
├── frame_fuse_cpu.cpp
└── tests/
    └── test_frame_fuse_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | FrameFuseCPU (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | FrameFuseCPUParams |
| 结果结构体 | FrameFuseCPUResult |
| 预热方法 | Warmup(maxPointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 统计方法 | GetStatistics() / ResetStatistics() |
| 日志标签 | "12-FrameFuseCPU" |

**FrameFuseStats 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| totalTimeMs | double | 总耗时（毫秒） |
| knnTimeMs | double | KNN 搜索耗时（毫秒） |
| descriptorTimeMs | double | 描述子计算耗时（毫秒） |
| matchingTimeMs | double | 匹配耗时（毫秒） |
| ransacTimeMs | double | RANSAC 耗时（毫秒） |
| refineTimeMs | double | 精配准耗时（毫秒） |
| set1PointCount | size_t | 第一帧点数 |
| set2PointCount | size_t | 第二帧点数 |
| descriptorCorrespondences | size_t | 描述子匹配对应数 |
| normalFilteredCorrespondences | size_t | 法线过滤后对应数 |
| ransacInliers | size_t | RANSAC 内点数 |
| ransacIterations | int | RANSAC 迭代次数 |
| refineIterationsActual | int | 精配准实际迭代次数 |
| adaptiveDistThreshCoarse | double | 粗配准自适应距离阈值 |
| adaptiveDistThreshFine | double | 精配准自适应距离阈值 |
| initialRMSE | double | 初始 RMSE |
| finalRMSE | double | 最终 RMSE |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (flann, core) | >= 4.x | KD-Tree、FLANN、SVD |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| knnK | int | 15 | >= 3 | KNN 搜索 K 值 |
| descriptorBins1 | int | 11 | >= 3 | SPFH 直方图 bin 数（角1） |
| descriptorBins2 | int | 11 | >= 3 | SPFH 直方图 bin 数（角2） |
| descriptorBins3 | int | 15 | >= 3 | SPFH 直方图 bin 数（角3） |
| loweRatio | double | 0.8 | (0, 1) | Lowe 比值测试阈值 |
| normalPreFilterAngleDeg | double | 15.0 | (0, 90) | 法线预过滤角度 |
| ransacConfidence | double | 0.999 | (0, 1] | RANSAC 置信度 |
| ransacMaxIterations | int | 5000 | >= 1 | RANSAC 最大迭代次数 |
| minInlierCount | int | 3 | >= 3 | 最少内点数 |
| refineIterations | int | 3 | >= 1 | 精配准迭代次数 |
| refineConvergeRatio | double | 1e-4 | > 0 | 精配准收敛阈值 |
| maxPointCount | size_t | 10000 | > 0 | 最大点数限制 |
| collectStatistics | bool | true | — | 是否收集统计信息 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 最低点数 | 两帧各需 >= 3 个标记点 |
| SPFH 计算复杂度 | O(N×K)，K 大时计算密集 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 重叠度充足 | overlap >= 50% |
| Degraded | 重叠度偏低 | overlap >= 30% |
| Warning | 重叠度不足 | overlap < 30% |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 内点不足 | 返回 success=false |
| RANSAC 失败 | 返回 success=false，直接退出（不使用最佳假设） |
| SVD 奇异 | 中断精配准迭代，使用 RANSAC 结果，质量由重叠度决定 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | SPFH 计算 O(N×K)，K 大时计算密集 | maxPointCount 限制点数 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
