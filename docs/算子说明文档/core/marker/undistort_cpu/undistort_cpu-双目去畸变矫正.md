# 双目立体去畸变+矫正

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-6 |
| 中文名称 | 双目立体去畸变+矫正 |
| 英文目录名 | undistort_cpu |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ① 亚像素级（~0.01 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机边缘点 | vector\<EdgePoint\> |
| **输入②** | 右相机边缘点 | vector\<EdgePoint\> |
| **输入③** | 分组 ID | groupIds |
| **输出** | 立体矫正结果 | StereoUndistortResult (rectifiedPoints1/2, pointCount1/2, R1/R2/P1/P2/**Q**, groupIds1/2, groupCount1/2) |

> **外部矫正矩阵（温度补偿路径）**：`SetRectifyMatrices(R1,R2,P1,P2,Q)` 设置后，跳过内部 `cv::stereoRectify`，直接用外部矩阵（来自 `stereo_rectify_temp_table` 当前温度条目）做 `cv::undistortPoints`。`ClearRectifyMatrices()` 恢复内部计算。扫描/温度补偿流程必须设置外部矩阵；不设置时向后兼容（标定流程仍走内部 stereoRectify）。

## C. 算法

**核心流程**：

```
Step 1: 构建左/右相机内参矩阵 K1, K2 和畸变系数 D1, D2（根据 distortionModel 选择模型：brown_conrady 使用 5 畸变系数 k1,k2,p1,p2,k3；rational_polynomial 使用 8 系数 k1-k6,p1,p2）
Step 2: cv::stereoRectify 计算 R1, R2, P1, P2, Q
Step 3: cv::undistortPoints 对左右点集去畸变+矫正
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::stereoRectify | 计算立体矫正变换矩阵 |
| cv::undistortPoints | 去畸变+坐标矫正 |

## D. 依赖

**上下游算子**：

```
image_merge → [undistort_cpu] → ellipse_fit
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| R1/R2/P1/P2 | 矫正矩阵可被下游复用 |

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
| image_merge | mergedEdgePoints, groupIds | 左/右相机各自独立调用 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| rectifiedPoints1/2 | ellipse_fit | 矫正后亚像素坐标 | 按 groupId 分组拟合 | 必用 |
| R1/R2/P1/P2 | — | 投影矩阵（仅存于结果中备用，算子11自行构建投影矩阵，不复用此处输出） | — | — |

## E. 架构

**文件结构**：

```
undistort_cpu/
├── undistort_points_cpu.h
├── undistort_points_cpu.cpp
└── tests/
    └── test_undistort_points_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | MarkerUndistortCPU |
| 核心方法 | Execute() |
| 参数结构体 | MarkerUndistortCPUParams |
| 结果结构体 | StereoUndistortResult |
| 辅助方法 | StereoUndistortResult::splitRectifiedPoints1ByGroup() const — 左相机矫正点按分组拆分 |
| | StereoUndistortResult::splitRectifiedPoints2ByGroup() const — 右相机矫正点按分组拆分 |
| 预热方法 | Warmup(maxPointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "06-MarkerUndistortCPU" |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (calib3d) | >= 4.x | 立体标定与矫正 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| fx1/fy1/cx1/cy1 | double | 0 | fx/fy > 0 | 左相机内参 |
| fx2/fy2/cx2/cy2 | double | 0 | fx/fy > 0 | 右相机内参 |
| k1_1 ~ k6_1, p1_1, p2_1 | double | 0 | — | 左相机畸变系数 |
| k1_2 ~ k6_2, p1_2, p2_2 | double | 0 | — | 右相机畸变系数 |
| R | array\<double,9\> | — | 旋转矩阵 | 左→右旋转 |
| T | array\<double,3\> | — | 平移向量 | 左→右平移 |
| imageWidth | int | 0 | > 0 | 图像宽度 |
| imageHeight | int | 0 | > 0 | 图像高度 |
| distortionModel | string | "brown_conrady" | "brown_conrady"(5系数) / "rational_polynomial"(8系数) | 畸变模型；rational_polynomial 使用 8 系数（含 k4-k6） |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 精度依赖 | 内参/外参标定精度直接影响矫正精度 |
| 点数一致 | 左右点集数量可不同，但 groupIds 应对应 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 矫正成功 | 成功 |
| Warning | 输入为空 | 无有效数据 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 内参矩阵非法 | 抛出 cv::Exception（未捕获），由调用方处理 |
| 输入点集为空 | 返回 Warning |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 标定参数不准确导致矫正偏差 | 定期标定校准 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 核心路径已覆盖 |
