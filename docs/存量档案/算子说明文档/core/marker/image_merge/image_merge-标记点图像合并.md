# 标记点图像合并

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-5 |
| 中文名称 | 标记点图像合并 |
| 英文目录名 | image_merge |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ②（坐标偏移为整数加法，零浮点误差） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 多组边缘点 | vector\<vector\<EdgePoint\>\> |
| **输入②** | ROI 列表 | vector\<cv::Rect\> |
| **输出** | 合并结果 | ImageMergeCPUResult (mergedEdgePoints, mergedEdgeCount, groupIds, groupCount) |

## C. 算法

**核心流程**：

```
Step 1: 校验边缘点组数与 ROI 数量一致
Step 2: 遍历每对（edgePoints, roi）
Step 3: 坐标偏移：merged.x = ep.x + roi.x, merged.y = ep.y + roi.y, merged.pixelX = ep.pixelX + roi.x, merged.pixelY = ep.pixelY + roi.y（angle/amplitude 直接拷贝）
Step 4: 记录 groupId 标记来源
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| 坐标偏移计算 | 子图坐标还原至原图坐标 |
| groupId 分配 | 标记点来源追踪 |

## D. 依赖

**上下游算子**：

```
zernike_edge → [image_merge] → undistort_cpu
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无特殊共享 | 纯 CPU 坐标变换 |

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
| zernike_edge | edgePoints (vector\<EdgePoint\>) × N 组 | 与对应 ROI 配对输入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| mergedEdgePoints, groupIds | undistort_cpu | 原图坐标下的边缘点 | 直接传递 | 必用 |

## E. 架构

**文件结构**：

```
image_merge/
├── image_merge_cpu.h
├── image_merge_cpu.cpp
└── tests/
    └── test_image_merge_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | ImageMergeCPU |
| 核心方法 | Execute() |
| 参数结构体 | ImageMergeCPUParams |
| 结果结构体 | ImageMergeCPUResult |
| 辅助方法 | ImageMergeCPUResult::splitByGroup() const — 将合并结果按分组拆分 |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "09-ImageMergeCPU" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core) | >= 4.x | 数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| — | — | — | — | 无特殊参数 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 输入校验 | 边缘点组数必须与 ROI 数量一致 |
| 坐标偏移 | 需正确传入对应 ROI 用于坐标还原 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | mergedEdgePoints 非空 | 正常 |
| Warning | 输入为空或无数据 | 无有效输出 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 边缘点组数与 ROI 数不匹配 | 返回 success=false |
| 输入为空 | 返回 success=true, Warning |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 数量不一致导致合并错误 | 前置校验拦截 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 核心路径已覆盖 |
