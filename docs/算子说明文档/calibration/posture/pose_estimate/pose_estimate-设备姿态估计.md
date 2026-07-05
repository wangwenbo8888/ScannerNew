# 设备姿态估计与匹配

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-13 |
| 中文名称 | 设备姿态估计与匹配 |
| 英文目录名 | pose_estimate |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ②（~0.05mm 位置，~0.05° 角度） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 刚体变换（R + T 或 4×4 位姿矩阵） | cv::Matx33d R + cv::Vec3d T 或 cv::Matx44d pose |
| **输出** | 姿态估计结果 | PoseEstimateCPUResult (currentPose, matches, anyMatched, bestMatch) |

## C. 算法

**核心流程**：

1. 根据网格标记点 + 原点/轴配置建立世界坐标系变换
2. 预计算目标姿态 4×4 矩阵
3. 运行时：T_current = T_world × cameraPose
4. 对每个目标：T_delta = T_target⁻¹ × T_current
5. 位置误差 = ‖ΔT‖，旋转误差 = arccos
6. 阈值判定，加权分数选择最佳匹配
7. 匹配时触发回调

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| ZYX 欧拉角→旋转矩阵 | 姿态参数转旋转矩阵 |
| 刚体变换快速求逆 | 4×4 齐次矩阵求逆 |

## D. 依赖

**上下游算子**：

```
frame_fuse → [pose_estimate] → 无（流程终点）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 预计算目标姿态 | 初始化时计算，运行时复用 |

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
| frame_fuse | R, T, transform | 刚体变换矩阵直接输入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| — | 流程终点 | — | 结果输出至外部系统 | — |

## E. 架构

**文件结构**：

```
pose_estimate/
├── pose_estimate_cpu.h
├── pose_estimate_cpu.cpp
└── tests/
    └── test_pose_estimate_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | PoseEstimateCPU (pImpl) |
| 核心方法 | Execute(), SetCallback() |
| 参数结构体 | PoseEstimateCPUParams |
| 结果结构体 | PoseEstimateCPUResult |
| 回调类型 | PoseCallback = std::function\<void(const PoseEstimateCPUResult&)\> |
| 预热方法 | Warmup(maxTargetCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 统计方法 | GetStatistics() / ResetStatistics() |
| 日志标签 | "13-PoseEstimateCPU" |

**PoseMatch 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| targetIndex | int | 目标索引 |
| targetName | string | 目标名称 |
| matched | bool | 是否匹配 |
| positionError | double | 位置误差（mm） |
| rotationError | double | 旋转误差（度） |
| positionThreshold | double | 位置阈值 |
| rotationThreshold | double | 旋转阈值 |

**PoseEstimateStats 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| totalTimeMs | double | 总耗时（毫秒） |
| coordBuildTimeMs | double | 坐标系构建耗时（毫秒） |
| matchTimeMs | double | 匹配耗时（毫秒） |
| targetCount | size_t | 目标总数 |
| matchedCount | size_t | 已匹配目标数 |
| estimateCallCount | size_t | estimate 调用次数 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core) | >= 4.x | 矩阵运算 |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| gridPoints | vector\<vector\<Point3d\>\> | — | — | 网格标记点三维坐标 |
| originRow | int | 0 | — | 原点所在行 |
| originCol | int | 0 | — | 原点所在列 |
| rowAxis | string | "X" | "X" / "Y" | 行方向轴 |
| faceNormal | string | "Z" | "Z" / "-Z" | 面法线方向 |
| poseTargets | vector\<PoseTarget\> | — | — | 目标姿态列表 |
| poseTargets[i].name | string | — | — | 目标名称 |
| poseTargets[i].tx/ty/tz | double | — | — | 目标位置 |
| poseTargets[i].rx/ry/rz | double | — | — | 目标姿态角（ZYX 欧拉角） |
| poseTargets[i].posThreshold | double | 10.0 | > 0 | 位置匹配阈值（单位 mm） |
| poseTargets[i].rotThreshold | double | 5.0 | > 0 | 旋转匹配阈值（单位 度（°）） |
| collectStatistics | bool | true | — | 是否收集统计信息 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 网格配置 | gridPoints 必须正确配置世界坐标系 |
| 预计算 | 目标姿态矩阵在首次 Execute() 调用时延迟构建（构造与 Warmup 均不触发构建） |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 有网格配置且有目标定义（无论是否匹配到目标） |
| Warning | 无法匹配 | 无网格配置或无目标定义 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| acos 参数越界 | 钳位至 [-1, 1] |
| 回调异常 | 静默捕获，不中断流程 |
| 无匹配目标 | anyMatched = false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | acos 可能 NaN | 已钳位 [-1, 1] |
| 🟢 低 | 回调异常 | 静默捕获 |
| 🟡 中 | `coordBuildTimeMs` 和 `matchTimeMs` 统计字段在 `estimateImpl()` 中从未赋值，始终为默认值 0.0（仅 `totalTimeMs` 被实际累计） | 坐标系构建与匹配阶段的分项耗时数据缺失 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 已覆盖核心路径 |
