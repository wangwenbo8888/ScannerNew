# 椭圆边界边缘点双目匹配

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-10 |
| 中文名称 | 椭圆边界边缘点双目匹配 |
| 英文目录名 | edge_match |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ② 整像素/几何类（~0.05 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左椭圆极线交点 | vector\<EllipseIntersectResult\> |
| **输入②** | 右椭圆极线交点 | vector\<EllipseIntersectResult\> |
| **输入③** | 中心匹配关系 | vector\<int\> centerMatches |
| **输出** | 边缘匹配结果 | EdgeMatchCPUResult (ellipseResults: vector\<EllipseEdgeMatchResult\>, statistics: EdgeMatchStats) |

## C. 算法

**核心流程**：

1. 遍历匹配的椭圆对（由 centerMatches 指导）
2. 按 epipolarIndex 建立 HashMap
3. 共有极线按 X 坐标排序配对
4. 计算视差 + 置信度（Y距离 0.6 + 视差一致性 0.4）

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| std::unordered_map | 极线索引 HashMap |
| 排序配对 | 按 X 坐标排序匹配 |

## D. 依赖

**上下游算子**：

```
epipolar_intersect + marker_match → [edge_match] → point_reconstruct
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| centerMatches | 来自 marker_match |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/zitai_result_types.h` | 姿态流程共享结果类型（EdgeMatchPair, EdgeMatchCPUResult 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| epipolar_intersect | 左/右 ellipseResults | 极线交点 |
| marker_match | centerMatches | 中心匹配关系指导椭圆对匹配 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| matchedPairs | point_reconstruct | 三角测量输入 | 按 marker 分组后重建 | 必用 |

## E. 架构

**文件结构**：

```
edge_match/
├── edge_match_cpu.h
├── edge_match_cpu.cpp
└── tests/
    └── test_edge_match_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | EdgeMatchCPU (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | EdgeMatchCPUParams |
| 结果结构体 | EdgeMatchCPUResult |
| 预热方法 | Warmup(maxEllipsePairs) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 统计方法 | GetStatistics() / ResetStatistics() |
| 日志标签 | "10-EdgeMatchCPU" |

**EdgeMatchPair 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| leftX | double | 左图边缘点 x 坐标 |
| leftY | double | 左图边缘点 y 坐标 |
| rightX | double | 右图边缘点 x 坐标 |
| rightY | double | 右图边缘点 y 坐标 |
| disparity | float | 视差（leftX - rightX） |
| confidence | float | 匹配置信度 |
| epipolarIndex | int | 极线索引 |
| leftEllipseIdx | int | 左椭圆索引 |
| rightEllipseIdx | int | 右椭圆索引 |
| side | int | 同一极线上按 X 坐标排序后的配对序号（从 0 开始） |

**EllipseEdgeMatchResult 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| leftEllipseIdx | int | 左椭圆索引 |
| rightEllipseIdx | int | 右椭圆索引 |
| leftCenterX | double | 左椭圆中心 x 坐标 |
| leftCenterY | double | 左椭圆中心 y 坐标 |
| rightCenterX | double | 右椭圆中心 x 坐标 |
| rightCenterY | double | 右椭圆中心 y 坐标 |
| matchedPairs | vector\<EdgeMatchPair\> | 匹配点对列表 |

**EdgeMatchStats 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| totalTimeMs | double | 总耗时（毫秒） |
| totalEllipsePairs | size_t | 椭圆对总数 |
| totalEpipolarLines | size_t | 极线总数 |
| matchedPairs | size_t | 已匹配对数 |
| skippedPairs | size_t | 跳过对数 |
| matchRate | float | 匹配率 |
| avgDisparity | float | 平均视差 |
| disparityStd | float | 视差标准差 |
| avgConfidence | float | 平均置信度 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| C++ STL | >= C++14 | HashMap、排序 |
| OpenCV (core) | >= 4.x | cv::Point2f 等数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| yTolerance | float | 0.2 | > 0 | Y 方向匹配容差 |
| disparityMaxDiff | float | 10.0 | > 0 | 视差最大差异 |
| maxMatchPairs | size_t | 100000 | > 0 | 最大匹配对数 |
| collectStatistics | bool | true | — | 是否收集统计信息 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 极线约束 | 仅在同一 epipolarIndex 的交点间匹配 |
| 唯一性 | 同一极线上按 X 排序后一对一配对 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 匹配率 >= 30% 或匹配对数 <= 5 |
| Warning | 匹配率过低 | 匹配率 < 30% 且匹配对数 > 5 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 无匹配椭圆对 | 返回 Normal（qualityFlag 不变） |
| 匹配对为空 | 返回 Normal（matchedPairs=0，不触发 Warning 条件） |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | HashMap 内存开销 | maxMatchPairs 限制 |
| 🟡 中 | matchRate 恒为 1.0 | skippedPairs 未被更新，matchRate 恒等于 1.0，Warning 条件（匹配率<30%）当前不可达 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
