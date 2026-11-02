# 标记点双目立体匹配

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-8 |
| 中文名称 | 标记点双目立体匹配 |
| 英文目录名 | marker_match |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ② 整像素/几何类（~0.05 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机椭圆中心点（立体矫正后） | vector\<cv::Point2f\> |
| **输入②** | 右相机椭圆中心点（立体矫正后） | vector\<cv::Point2f\> |
| **输出** | 匹配结果 | MarkerMatchCPUResult (disparities, valid_flags, confidence, centerMatches, statistics) |

## C. 算法

**核心流程**：

1. 按 Y 坐标排序（SoA 布局）
2. 自适应容差（根据点密度调整）
3. 滑动窗口极线约束匹配（二分查找 + 唯一性检测）
4. 置信度计算：Y 距离 0.7 + 视差一致性 0.3
5. 可选：预计算右参考点 + YZone 索引加速

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| 自定义 SoA 排序 | 按 Y 坐标高效排序 |
| 二分查找 | 极线约束下快速定位候选 |
| YZoneIndex | Y 区间索引加速查找 |
| OpenMP 条件编译 | 可选并行加速 |

## D. 依赖

**上下游算子**：

```
undistort_cpu → [ellipse_fit] → [marker_match]
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| centerMatches | 仅被 10 消费 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| ellipse_fit | center (cv::Point2f) | 左/右相机椭圆中心点分别输入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| centerMatches, disparities | edge_match | 匹配+视差 | 边缘点匹配参考 | 必用 |

## E. 架构

**文件结构**：

```
marker_match/
├── marker_match_cpu.h
├── marker_match_cpu.cpp
└── tests/
    └── test_marker_match_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | MarkerMatchCPU (pImpl) |
| 核心方法 | Execute(), SetReferencePoints(), MatchWithReference(), ClearReferencePoints(), HasReferencePoints() |
| 参数结构体 | MarkerMatchCPUParams |
| 结果结构体 | MarkerMatchCPUResult |
| 统计结构体 | MarkerMatchStats |
| 预热方法 | Warmup(maxPointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 统计方法 | GetStatistics() / ResetStatistics() |
| 日志标签 | "08-MarkerMatchCPU" |

**MarkerMatchStats 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| total_time_ms | double | 总耗时（毫秒） |
| sort_time_ms | double | 排序耗时（毫秒） |
| match_time_ms | double | 匹配耗时（毫秒） |
| total_points | size_t | 总点数 |
| matched_points | size_t | 已匹配点数 |
| ambiguous_points | size_t | 歧义点数 |
| unvisited_points | size_t | 未访问点数 |
| match_rate | float | 匹配率 |
| avg_disparity | float | 平均视差 |
| disparity_std | float | 视差标准差 |
| avg_confidence | float | 平均置信度 |
| disparity_consistency | float | 视差一致性 |
| num_threads_used | int | 实际使用线程数 |
| used_precomputed | bool | 是否使用预计算参考点 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core) | >= 4.x | 数据结构 |
| OpenMP（可选） | — | 并行加速 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| y_tolerance | float | 0.15 | (0, 1] | 极线 Y 容差（单位 px（像素）） |
| enable_parallel | bool | false | — | 是否启用 OpenMP 并行 |
| num_threads | int | 0 | >= 0 | 并行线程数（0=自动） |
| max_points | size_t | 100 | > 0 | 最大匹配点数 |
| density_threshold_high | float | 2.0 | — | 密集区阈值（单位 点/像素行） |
| density_threshold_low | float | 0.5 | — | 稀疏区阈值（单位 点/像素行） |
| dense_tolerance_scale | float | 0.8 | — | 密集区容差缩放 |
| sparse_tolerance_scale | float | 1.2 | — | 稀疏区容差缩放 |
| collect_statistics | bool | true | — | 是否收集统计信息 |
| prealloc_buffer_size | size_t | 128 | >0 | 预分配缓冲区大小 |
| parallel_threshold | size_t | 50 | >0 | 并行阈值 |
| max_buffer_size | size_t | 10000 | >=max_points | 缓冲区上限 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 极线约束 | 立体矫正后左右点应在同一水平线上 |
| 唯一性约束 | 每个左点最多匹配一个右点 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 匹配率 >= 30% 或总点 <= 5 |
| Warning | 匹配率过低 | 匹配率 < 30% 且总点 > 5 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 输入点集为空 | 抛出 std::invalid_argument 异常 |
| 匹配率过低 | 返回 Warning（centerMatches 中未匹配项为 -1） |
| 输入点含 NaN/Inf | 抛出 std::invalid_argument |
| 点数超过 max_points | 抛出 std::invalid_argument |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 匹配主循环始终单线程（matchImpl/MatchWithReferenceImpl 无 `#pragma omp`），enable_parallel 参数未被代码读取，num_threads_used 固定为 1 | 并行加速实际无效 |
| 🟢 低 | OpenMP 基础设施（线程局部缓冲、omp_set_num_threads）已就绪但未接入匹配循环 | 未来可实现并行 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
