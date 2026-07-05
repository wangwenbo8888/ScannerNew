# 激光/标记点区域连通域分析

> 本算子为标记点链与激光链共用（原 `姿态-02` / `激光标定-02` 两份说明已合并）。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | —（共用：原标记点链 姿态-2 / 激光链 激光标定-2，仅追溯用） |
| 中文名称 | 激光/标记点区域连通域分析 |
| 英文目录名 | ccl |
| 运行平台 | CUDA (全GPU管线, 固定 connectivity=8) |
| 所属流程 | 姿态估计 / 激光标定（共用） |
| 精度档次 | ② 整像素/几何类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 二值掩膜 | cv::cuda::GpuMat (CV_8UC1 二值掩膜) |
| **输出** | 连通域分析结果 | RegionAnalysisResult 含 d_labeledMask(CV_32SC1, 1-indexed稠密重编号), componentCount, components(各连通域包围盒 ComponentStats), qualityFlag |

## C. 算法

**核心流程**：

```
Step 1: GPU CCL (cv::cuda::connectedComponents, connectivity=8, CV_32SC1)
Step 2: GPU initStatsKernel — 初始化统计缓冲 (5数组 areas/min_x/max_x/min_y/max_y + remap + num_valid)
Step 3: GPU computeStatsKernel — 逐像素 atomic 统计 (面积+bbox, 5原子, 无质心)
Step 4: GPU buildRemapKernel — 面积过滤(minArea/maxArea) + 稠密重编号(1-indexed)
Step 5: GPU relabelKernel — 查 remap 表生成重编号掩膜 (CV_32SC1)
Step 6: GPU compactStatsKernel — 压缩有效连通域的包围盒到紧凑数组
Step 7: D2H download (pinned memory, 2次小传输: count + compact数组)
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::cuda::connectedComponents | GPU连通域标记 (8-连通) |
| initStatsKernel (自定义CUDA核) | 初始化5个统计数组 + remap + num_valid |
| computeStatsKernel (自定义CUDA核) | 逐像素 atomic 统计面积和包围盒 |
| buildRemapKernel (自定义CUDA核) | 面积过滤 + 稠密重编号 |
| relabelKernel (自定义CUDA核) | 全图重编号生成输出掩膜 |
| compactStatsKernel (自定义CUDA核) | 压缩包围盒结果到紧凑数组 |

## D. 依赖

**上下游算子**：

```
mask_separation → 本算子 → image_split（标记点链）
                      └──────→ laser_label（激光链）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无 | 独立实例 |

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
| mask_separation | shared_ptr\<GpuMat\> | 二值掩膜（d_markingPointMask / d_laserMask，视链而定）(LaserMarkingSeparationResult) |

**本算子→下游**：

| 输出字段 | 传递给 | 所属链 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|--------|---------|---------|---------|
| components (vector\<ComponentStats\>) | image_split | 标记点链 | ROI 列表（via toRectList() 转换 boundingBoxX/Y/Width/Height → cv::Rect） | vector\<ComponentStats\> | 必用 |
| d_labeledMask | laser_label | 激光链 | LaserLabelerCUDA::Execute() | GpuMat 引用 (CV_32SC1) | 必用 |

> **注**：标记点链主要消费 `components`（包围盒列表）用于图像分块；激光链主要消费 `d_labeledMask`（重编号掩膜）。`componentCount` 为通用计数输出。

## E. 架构

**文件结构**：

```
ccl/
├── region_analyze_cuda.h
├── region_analyze_cuda.cpp
├── region_analyze_cuda_pimpl.h
├── region_analyze_cuda_impl.cu
└── tests/
    └── test_region_analyze_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | RegionAnalyzerCUDA (pImpl) |
| 核心方法 | Execute() (shared_ptr 和 const GpuMat& 两个重载) |
| 参数结构体 | RegionAnalyzerParams |
| 统计结构体 | ComponentStats (label + 包围盒) |
| 结果结构体 | RegionAnalysisResult |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "07-RegionAnalyzerCUDA" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_core, opencv_imgproc, opencv_cudaimgproc, opencv_cudafilters, opencv_cudaarithm |
| CUDA Toolkit | 12.4 (未在源码中锁定版本) | sm_75 (RTX5000), sm_86, sm_87 (Orin) |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|--------|--------|------|------|
| minArea | int | 100 | >=0 | 最小面积阈值（像素），过滤小连通域 |
| maxArea | int | 100000 | >minArea | 最大面积阈值（像素），过滤大连通域 |
| deviceId | int | 0 | >=0 | GPU设备ID（多GPU场景） |

> **注**：operator_params.json 中存在 `connectivity` 字段，但 `RegionAnalyzerParams` 结构体未包含该字段，`fromJson()` 不解析它。代码固定使用 connectivity=8。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 输入约束 | 仅接受 CV_8UC1 二值掩膜 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 连通域数量 1~200 |
| Warning | 警告 | 连通域数量为 0（仍 success=true，下游可处理空结果） |
| Degraded | 降级 | 连通域数量 > 200（结果可能不可靠） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate: minArea<0, maxArea<=minArea, deviceId<0) | 抛出 std::invalid_argument |
| 输入 shared_ptr 为空 | 返回 success=false |
| 输入为空 / 类型非 CV_8UC1 | 返回 success=false |
| 无 CUDA 设备 (getCudaEnabledDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| OpenCV/CUDA 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 有效连通域数超过 COMPACT_MAX(65536) 时，超出部分静默丢弃：remap 表中不分配编号（relabel 后变为背景0），components 数组 resize 到 num_valid 但仅拷贝前 65536 条，剩余为零值 | 极端场景下 componentCount 报告的数量与实际可用数据不一致（实际场景连通域通常 <200，触发概率极低） |
| 🟢 低 | Execute() 中间步骤使用 cudaDeviceSynchronize() 强制同步，stream 参数仅最终 copyTo 生效 | 流水线并行度受限，异步流水线中可能成为瓶颈 |
| 🟢 低 | operator_params.json 中 connectivity=4 preset 会被静默忽略（代码固定 conn=8） | 配置文件误导使用者认为支持 4-连通 |
| 🟡 中 | config JSON 结构不匹配 | operator_params.json 参数嵌套在 "default_params" 下，fromJson() 在顶层查找键，实际无法读取嵌套参数。当前因默认值巧合相同未暴露 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
