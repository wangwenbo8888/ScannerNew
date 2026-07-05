# 标记点图像分割

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-3 |
| 中文名称 | 标记点图像分割 |
| 英文目录名 | image_split |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ② 整像素/几何类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 灰度图像 | cv::Mat (CV_8UC1) |
| **输入②** | ROI 列表 | vector\<cv::Rect\> |
| **输出** | 分割结果 | ImageSplitCPUResult (splitImages: vector\<cv::Mat\> 深拷贝, splitCount) |

## C. 算法

**核心流程**：

```
Step 1: 遍历 ROI 列表
Step 2: 边界检查模式：rect & imgBounds 裁剪
Step 3: 无边界检查模式：越界直接跳过
Step 4: 对有效 ROI 执行深拷贝 clone()
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::Rect 交集 & | 边界裁剪 |
| cv::Mat::clone() | 深拷贝子图 |

## D. 依赖

**上下游算子**：

```
ccl → [image_split] → zernike_edge
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无特殊共享 | 纯 CPU 操作，无共享资源 |

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
| ccl | components[i] 经 toRectList() 转换为 cv::Rect | 直接用作 ROI 列表 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| splitImages[i] (cv::Mat) | zernike_edge | 单个子图 | 每个子图独立进行边缘检测 | 必用 |

## E. 架构

**文件结构**：

```
image_split/
├── image_split_cpu.h
├── image_split_cpu.cpp
└── tests/
    └── test_image_split_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | ImageSplitCPU |
| 核心方法 | Execute() |
| 参数结构体 | ImageSplitCPUParams |
| 结果结构体 | ImageSplitCPUResult |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "08-ImageSplitCPU" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core, imgproc) | >= 4.x | 图像操作 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| enableBoundaryCheck | bool | false | true/false | 是否启用边界检查裁剪 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 内存拷贝 | 使用 clone() 深拷贝，确保输出独立于输入 |
| 越界处理 | 无边界检查模式下越界 ROI 直接跳过 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | splitCount > 0 且无越界裁剪 | 正常 |
| Degraded | 部分 ROI 越界被裁剪 | 部分数据损失 |
| Warning | 所有 ROI 无效 | 无有效输出 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| ROI 越界 | 根据 enableBoundaryCheck 裁剪或跳过 |
| 输入图像为空 | 返回 success=false |
| 输入类型非 CV_8UC1 | 返回 success=false |
| ROI 列表为空 | 返回 success=true, Warning, splitCount=0 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | clone() 内存开销 | ROI 数量有限，开销可控 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 核心路径已覆盖 |
