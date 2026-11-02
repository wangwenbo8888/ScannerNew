# 激光掩膜区域提取

> 本算子为标记点链与激光链共用（原 `姿态-01` / `激光标定-01` 两份说明已合并）。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | —（共用：原标记点链 姿态-1 / 激光链 激光标定-1，仅追溯用） |
| 中文名称 | 激光掩膜区域提取 |
| 英文目录名 | mask_extract |
| 运行平台 | CUDA (OpenCV CUDA API) |
| 所属流程 | 姿态估计 / 激光标定（共用） |
| 精度档次 | ② 整像素/几何类（CPU vs CUDA误差<0.1px） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 灰度图 | cv::Mat (CV_8UC1 灰度图, Host端) |
| **输出** | 掩膜提取结果 | MaskExtractResult 含 d_grayImage(GpuMat), d_laserMask(GpuMat), d_cleanedMask(GpuMat) |

> **注**：`d_grayImage` 为 Host→Device 上传结果，供激光链下游 Steger 算子复用（标记点链不消费此字段）。

## C. 算法

**核心流程**：

```
Step 1: Host→Device 上传灰度图到GpuMat
Step 2: GPU二值化 (cv::cuda::threshold)
Step 3: GPU腐蚀去噪 (cv::cuda::createMorphologyFilter MORPH_ERODE, 椭圆核)
Step 4: GPU膨胀恢复形状 (MORPH_DILATE, 椭圆核)
Step 5: 面积过滤（TODO，当前直接复制）
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::cuda::threshold | GPU二值化 |
| cv::cuda::createMorphologyFilter | GPU形态学滤波（腐蚀/膨胀） |
| cv::getStructuringElement | 生成椭圆结构核 |
| cv::cuda::createContinuous | 创建连续GPU内存缓冲区 |

## D. 依赖

**上下游算子**：

```
无（流程入口） → 本算子 → ccl（标记点链 / 激光链）
                          └──────→ steger（激光链，经 d_grayImage）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| d_grayImage | 供激光链后续 Steger 算子使用 |
| 左/右相机实例 | 左右相机各持独立实例 |
| CUDA Stream | 算子内部 GPU 操作共用 |

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
| 外部输入 | cv::Mat 参数传入 | 灰度图 Host 端传入（流程起点） |

**本算子→下游**：

| 输出字段 | 传递给 | 所属链 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|--------|---------|---------|---------|
| d_cleanedMask | ccl | 标记点链 / 激光链 | RegionAnalyzerCUDA | GpuMat 引用 (CV_8UC1) | 必用 |
| d_grayImage | steger | 激光链 | StegerExtractorCUDA | GpuMat 引用 | 必用 |

## E. 架构

**文件结构**：

```
mask_extract/
├── mask_extract_cuda.h
├── mask_extract_cuda.cpp
├── mask_extract_cuda_pimpl.h
├── mask_extract_cuda_impl.cu
└── tests/
    ├── test_mask_extract_cuda.cpp
    └── test_mask_extract_cuda_precision.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | MaskExtractCUDA (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | MaskExtractParams |
| 结果结构体 | MaskExtractResult |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "06-MaskExtractCUDA" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_cudaimgproc, opencv_cudafilters, opencv_cudaarithm, opencv_imgproc, opencv_core |
| CUDA Toolkit | 12.6 | sm_75, sm_86, sm_87 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|--------|--------|------|------|
| threshold | int | 80 | [0,255] | 二值化阈值 |
| erodeSize | int | 5 | 正奇数 | 腐蚀核大小 |
| laserDilateSize | int | 3 | 正奇数 | 激光膨胀核大小 |
| minArea | int | 100 | >=0 | 最小区域面积 |
| maxArea | int | 100000 | >minArea | 最大区域面积 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 线程安全 | 非线程安全，Debug模式 atomic\<bool\> inProcess_ 断言保护 |
| 实例隔离 | 左/右相机各自独立实例 |
| GPU缓冲区 | 需 Warmup() 预分配，5个GpuMat缓冲区 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | Execute() 成功 |
| Degraded | 降级 | OpenCV/运行时异常 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型非 CV_8UC1 | 返回 success=false（qualityFlag 保持 Normal 默认值） |
| 无 CUDA 设备 (getCudaEnabledDeviceCount<=0) | 构造时抛出 std::runtime_error |
| OpenCV/CUDA 异常 | 捕获并返回 Degraded |
| 运行时异常 | 捕获并返回 Degraded |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 面积过滤功能 TODO 未实现（filterByArea 为空壳） | 可能保留噪声小区域，不影响主流程 |
| 🟡 中 | Execute() 中可能违反"禁止cudaMalloc"规范 | 运行时 GPU 内存碎片化风险 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用（面积过滤缺失不影响主流程） |
