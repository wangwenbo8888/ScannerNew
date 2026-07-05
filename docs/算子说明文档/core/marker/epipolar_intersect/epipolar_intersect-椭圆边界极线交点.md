# 椭圆边界极线交点计算

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-9 |
| 中文名称 | 椭圆边界极线交点计算 |
| 英文目录名 | epipolar_intersect |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ① 亚像素级（~0.01 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 椭圆参数 | EllipseFitCPUResult / RotatedRect / 椭圆参数 |
| **输出** | 极线交点结果 | EpipolarIntersectCPUResult (ellipseResults: vector\<EllipseIntersectResult\>, 每个含 centerX/Y, majorAxis, minorAxis, angle, intersectPts) |

## C. 算法

**核心流程**：

1. 计算椭圆 Y 方向投影范围
2. 以 epipolarStep 间距生成水平极线
3. 对每条极线求解椭圆参数方程的二次方程
4. 交点按 (epipolarIndex, X 坐标) 排序

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| std::sqrt | 二次方程求根 |
| std::ceil / std::floor | 极线范围计算 |
| 椭圆参数方程求解 | 极线与椭圆交点 |

## D. 依赖

**上下游算子**：

```
ellipse_fit → [epipolar_intersect] → edge_match
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无特殊共享 | 纯数学计算 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/zitai_result_types.h` | 姿态流程共享结果类型（EllipseIntersectResult 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| ellipse_fit | EllipseFitCPUResult | 椭圆参数直接输入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| ellipseResults | edge_match | 左/右椭圆交点 | 与 centerMatches 一起进行边缘匹配 | 必用 |

## E. 架构

**文件结构**：

```
epipolar_intersect/
├── epipolar_intersect_cpu.h
├── epipolar_intersect_cpu.cpp
└── tests/
    └── test_epipolar_intersect_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | EpipolarIntersectCPU (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | EpipolarIntersectCPUParams |
| 结果结构体 | EpipolarIntersectCPUResult |
| 预热方法 | Warmup(maxEllipseCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "09-EpipolarIntersectCPU" |

**EpipolarIntersectPoint 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| x | double | 交点 x 坐标 |
| y | double | 交点 y 坐标 |
| yEpipolar | double | 极线 y 坐标 |
| epipolarIndex | int | 极线索引 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| C++ STL | >= C++14 | 数学计算 |
| OpenCV (core) | >= 4.x | cv::RotatedRect 等数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| epipolarStep | double | 0.5 | > 0 | 极线间距（像素） |
| maxIntersectionsPerEllipse | int | 1000 | > 0 | 单个椭圆最大交点数 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 纯数学计算 | 无图像依赖，仅依赖椭圆参数 |
| 交点数量 | 受 epipolarStep 和椭圆尺寸影响 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 成功 | 计算成功 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 椭圆参数异常 | 返回空交点集 |
| 二次方程无实根 | 跳过该极线 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | epipolarStep 过小导致交点过多 | maxIntersectionsPerEllipse 限制 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
