# 椭圆拟合和中心提取

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-7 |
| 中文名称 | 椭圆拟合和中心提取 |
| 英文目录名 | ellipse_fit |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ① 亚像素级（~0.01 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 边缘点集 | vector\<EdgePoint\> / Point2f / Point2d |
| **输出** | 椭圆拟合结果 | EllipseFitCPUResult (centerX/Y, majorAxis, minorAxis, angle, inlierCount, totalPointCount, inlierPoints) |

## C. 算法

**核心流程**：

```
Step 1: RANSAC 循环：随机采样 5 点
Step 2: cv::fitEllipse 拟合椭圆
Step 3: 轴长约束过滤（minEllipseAxis, maxAxisRatio）
Step 4: 代数距离计算内点
Step 5: 早停判断（earlyStopRatio）
Step 6: 收集最佳内点集
Step 7: 最终拟合：cv::fitEllipseAMS() 或降级 cv::fitEllipse()
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::fitEllipse | 椭圆拟合 |
| cv::fitEllipseAMS | AMS 椭圆拟合（更鲁棒） |
| cv::boundingRect | 包围盒计算 |

## D. 依赖

**上下游算子**：

```
undistort_cpu → [ellipse_fit] → marker_match + epipolar_intersect
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| EllipseFitCPUResult | 同时被 08 和 09 消费 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/zitai_result_types.h` | 姿态流程共享结果类型（EllipseFitCPUResult 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| undistort_cpu | rectifiedPoints | 按 groupId 分组后逐组拟合 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| center (cv::Point2f) | marker_match | 左/右椭圆中心 | 用于双目匹配 | 必用 |
| EllipseFitCPUResult | epipolar_intersect | 椭圆参数 | 用于极线交点计算 | 必用 |
| center (cv::Point2f) | inverse_distort（3-1） | 姿态合格帧椭圆中心（矫正坐标系） | 逆变换为原始畸变坐标供内参标定 | 必用 |

## E. 架构

**文件结构**：

```
ellipse_fit/
├── ellipse_fit_cpu.h
├── ellipse_fit_cpu.cpp
└── tests/
    └── test_ellipse_fit_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | EllipseFitCPU |
| 核心方法 | Execute() |
| 参数结构体 | EllipseFitCPUParams |
| 结果结构体 | EllipseFitCPUResult |
| 预热方法 | Warmup(maxPointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "07-EllipseFitCPU" |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core, imgproc) | >= 4.x | 椭圆拟合 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| ransacIterations | int | 100 | > 0 | RANSAC 最大迭代次数 |
| ransacThreshold | double | 0.5 | > 0 | RANSAC 内点阈值 |
| minEllipseAxis | double | 2.0 | > 0 | 椭圆最短轴约束 |
| maxAxisRatio | double | 100.0 | > 1 | 长短轴最大比值 |
| minInliers | int | 5 | >= 5 | 最少内点数 |
| earlyStopRatio | double | 0.8 | (0, 1] | RANSAC 早停比例 |
| useAMS | bool | true | — | 是否使用 AMS 拟合 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 最低点数 | 输入点数 < 5 时无法拟合 |
| 椭圆退化 | 轴比过大的拟合结果被过滤 |
| RNG 固定种子 | RANSAC 使用固定种子 12345，结果可复现 |
| 退化采样过滤 | 采样点包围盒面积 < 2 时跳过（避免共点退化拟合） |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 拟合成功 | 成功 |
| Warning | 输入点数 < 5 | 无法拟合 |
| Degraded | 内点不足 | 拟合质量下降 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 点数不足（输入 < 5） | 返回 success=false, Warning |
| RANSAC 后最佳内点数 < minInliers | 返回 success=false, Degraded |
| 最终内点 < 5 | 返回 success=false, Degraded |
| AMS 失败 | 降级使用 cv::fitEllipse |
| AMS+fitEllipse 均异常 | 返回 success=false, Degraded |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 固定种子 RANSAC（12345）可能导致特定数据下采样偏差 | 足够迭代次数+早停机制 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 核心路径及降级路径已覆盖 |
