# 激光线与标记点掩膜分离

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 预处理-1 |
| 中文名称 | 激光线与标记点掩膜分离 |
| 英文目录名 | mask_separation |
| 运行平台 | CUDA（全自定义内核，无 cv::cuda 形态学滤镜） |
| 所属流程 | 预处理流程（激光标定 + 姿态估计共用入口） |
| 精度档次 | ② 整像素/几何类（掩膜级，无亚像素精度） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 单帧灰度图像（同时含激光线和标记点） | `cv::Mat`（CV_8UC1，Host 端） |
| **输出** | 掩膜分离结果 | `LaserMarkingSeparationResult` 含 `d_laserMask`（激光线掩膜）、`d_markingPointMask`（标记点掩膜）、`d_combinedMask`（合并掩膜）、`timings`（各步骤耗时） |

---

## C. 算法

**核心流程**（6 步形态学流水线，前提取决于激光线与标记点宽度差异）：

```
Step 1: 融合高斯模糊 5×5 + 二值化 → 基础掩膜（fusedGaussianThresholdKernel）
Step 2: 去除小噪点：腐蚀→膨胀（Open 操作）→ d_step2_mask
Step 3: 去除大噪点：腐蚀→膨胀（更大核）→ 减去得 d_combined（两个提取物合并掩膜）
Step 4+5: 融合提取：腐蚀→膨胀（更大核）→ 大提取物（激光线）d_laserMask；
         d_combined 减去大提取物 → 小提取物（标记点）d_marking_raw
Step 6: 小提取物膨胀 → 恢复标记点边缘渐变区域 → d_marking_final
```

**关键 CUDA 内核**：

| 内核 | 用途 |
|------|------|
| `fusedGaussianThresholdKernel` | 融合 5×5 分离式高斯模糊 + 二值化（共享内存瓦片，权重 {19,66,100,66,19}/270） |
| `binaryMorphKernel<KHALF, IS_ERODE>` | 模板化二值腐蚀/膨胀（共享内存，边值填充） |
| `fusedDilateSubtractKernel<KHALF>` | 融合膨胀 + 减法（Step 3 合并掩膜提取） |
| `fusedDilateDualOutputKernel<KHALF>` | 融合膨胀 + 双输出（Step 4/5 同时输出激光线和标记点掩膜） |

> **注**：所有形态学操作均为自定义 CUDA 内核，**未使用 `cv::cuda::createMorphologyFilter`**。使用的 OpenCV CUDA 函数仅限 `cv::cuda::createContinuous`（内存分配）、`cv::cuda::StreamAccessor`（流获取）和 `cv::cuda::getCudaEnabledDeviceCount`（设备检查）。

---

## D. 依赖

**上下游算子**：

```
无（流程入口） → 本算子 → ccl（激光标定/姿态估计共用）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `d_laserMask` | 激光掩膜 → 后续激光线连通域分析 |
| `d_markingPointMask` | 标记点掩膜 → 后续标记点连通域分析 |
| 左右相机实例 | 左右相机各持独立实例 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（`QualityFlag` 等） |
| `common/calib_logging.h` | 日志宏（`CALIB_LOG_*`） |
| `common/calib_warmup_config.h` | 预热配置 `WarmupConfig` |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 外部输入 | `cv::Mat` 参数传入 | 灰度图 Host 端传入，算子内部上传到 GPU |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `d_laserMask` | ccl（激光标定流程） | `RegionAnalyzerCUDA::Execute()` | `shared_ptr<GpuMat>` 引用 | 必用 |
| `d_markingPointMask` | ccl（姿态估计流程） | `RegionAnalyzerCUDA::Execute()` | `shared_ptr<GpuMat>` 引用 | 必用 |
| `d_combinedMask` | Pipeline 调度 | — | `shared_ptr<GpuMat>` 引用 | 可选 |
| `result.success` | Pipeline 调度 | — | 值 | 必用 |
| `result.qualityFlag` | Pipeline 调度 | — | 值 | 可选 |

---

## E. 架构

**文件结构**：

```
mask_separation/
├── laser_markingpoint_mask_separation_cuda.h        # 公开头文件（纯 C++，不含 CUDA 类型）
├── laser_markingpoint_mask_separation_cuda_pimpl.h  # CUDA 桥接头文件（Impl 定义 + GpuMat + Events）
├── laser_markingpoint_mask_separation_cuda.cpp      # 桥接实现（输入校验 + 转发）
├── laser_markingpoint_mask_separation_cuda_impl.cu  # CUDA 内核 + 6 步流水线实现
├── example_usage.cpp                                 # 使用示例
└── tests/
    └── test_laser_markingpoint_mask_separation_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserMarkingSeparationCUDA`（pImpl） |
| 核心方法 | `Execute(const cv::Mat&, cv::cuda::Stream&)` / `Execute(const cv::Mat&)` |
| 预热方法 | `Warmup(rows, cols)` / `Warmup(WarmupConfig)` |
| 参数更新 | `SetParams()` / `GetParams()` |
| 参数结构体 | `LaserMarkingSeparationParams` |
| 结果结构体 | `LaserMarkingSeparationResult`（move-only） |
| 计时结构体 | `MaskSeparationTimings` |
| 日志标签 | `"LaserMarkingSeparationCUDA"` |

**GPU 缓冲区**：8 个 `GpuMat`（`CV_8UC1`），由 `createContinuous` 预分配：

| 缓冲区 | 用途 |
|--------|------|
| `d_inputBuffer` | Host 上传目标 |
| `d_binary` | Step 1 输出 |
| `d_temp` | 腐蚀暂存 |
| `d_step2_mask` | Step 2 输出 |
| `d_combined` | Step 3 输出（合并掩膜） |
| `d_laser_mask` | Step 4 激光线输出 |
| `d_marking_raw` | Step 5 标记点输出（膨胀前） |
| `d_marking_final` | Step 6 标记点最终输出 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | `opencv_core`, `opencv_imgproc`, `opencv_cudaimgproc`, `opencv_cudafilters`, `opencv_cudaarithm`（实际仅 `createContinuous` + `StreamAccessor` + `getCudaEnabledDeviceCount`） |
| CUDA Toolkit | 12.6 | sm_75（RTX 5000），sm_87（Orin） |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

**编译特性**：

- `BUILD_CUDA` 条件编译宏：开启时编译 `.cu`，关闭时仅编译 `.cpp`（`BUILD_CUDA=0`）
- CUDA 源文件额外定义 `FMT_UNICODE=0`（Windows fmtlib 兼容）
- 平台兼容：Windows 10 x64 + Jetson Orin NX/AGX Orin

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `gaussianSize` | int | 5 | 正奇数 | 高斯模糊核大小（当前内核固定 5×5，此参数未实际使用） |
| `threshold` | int | 80 | [0, 255] | 二值化阈值 |
| `step2_erodeSize` | int | 3 | 正奇数 ∈{3,5,7,9,11} | 第 2 步腐蚀核（去小噪点） |
| `step2_dilateSize` | int | 3 | 正奇数 ∈{3,5,7,9,11} | 第 2 步膨胀核 |
| `step3_erodeSize` | int | 5 | 正奇数 ∈{3,5,7,9,11}，> step2_erodeSize | 第 3 步腐蚀核（去除提取物） |
| `step3_dilateSize` | int | 7 | 正奇数 ∈{3,5,7,9,11} | 第 3 步膨胀核 |
| `step4_erodeSize` | int | 5 | 正奇数 ∈{3,5,7,9,11} | 第 4 步腐蚀核（去除小提取物） |
| `step4_dilateSize` | int | 9 | 正奇数 ∈{3,5,7,9,11}，> step3_dilateSize | 第 4 步膨胀核（包含边缘渐变） |
| `step6_dilateSize` | int | 5 | 正奇数 ∈{3,5,7,9,11} | 第 6 步标记点膨胀核（恢复边缘） |

**预设配置**：`default`（激光线宽 20-40px，标记点宽 5-10px） / `bright_scene`（高亮度） / `dark_scene`（低亮度） / `noisy_environment`（高噪声，强去噪核） / `thin_features`（细特征场景，小核）

> **警告**：`thin_features` 预设的 `step2_erodeSize=3` 与 `step3_erodeSize=3` 违反 `validate()` 约束（要求 `step3_erodeSize > step2_erodeSize`），加载该预设会抛出 `std::invalid_argument`，**实际不可用**。

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ② 掩膜级（整像素，无亚像素精度） |
| 内核大小约束 | 所有 `erodeSize`/`dilateSize` 的 `khalf=(size-1)/2` 必须 ∈ {1,2,3,4,5}，即 size ∈ {3,5,7,9,11}；超出抛出 `std::runtime_error` |
| 线程安全 | 非线程安全，Debug 模式 `atomic<bool> inProcess_` 检测并发调用 |
| 实例隔离 | 左右相机各持独立实例 |
| 前提条件 | 激光线和标记点宽度须有明显差异（激光线粗，标记点细） |

---

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 分离成功 |
| `Degraded` | 降级 | OpenCV/运行时异常（捕获后标记） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败（`validate`） | 抛出 `std::invalid_argument` |
| 内核大小非法（khalf ∉ {1..5}） | `Execute()` 中被 try/catch 捕获 → `success=false` + `Degraded`；`Warmup()` 中直接抛出 |
| 无 CUDA 设备 | 构造时抛出 `std::runtime_error` |
| 输入为空 | 返回 `success=false`，`message="Input image is empty"` |
| 输入类型非 CV_8UC1 | 返回 `success=false`，`message="Input must be CV_8UC1 grayscale image"` |
| OpenCV 异常 | 捕获，`success=false`，`qualityFlag=Degraded` |
| 内核大小非法（khalf ∉ {1..5}） | `Execute()` 内抛出 `std::runtime_error`，但被 try/catch 捕获 → `success=false`，`qualityFlag=Degraded`；`Warmup()` 中则直接抛出未捕获 |
| 运行时异常 | 捕获，`success=false`，`qualityFlag=Degraded` |

> **注**：`Execute()` 全程在 `try{}` 块内执行，所有 `std::runtime_error`（包括内核大小非法）均被 `catch` 捕获并转为 `success=false` + `Degraded`，**不会向上层传播异常**。仅 `Warmup()` 中的异常会直接抛出。

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🔴 高 | `thin_features` 预设 `step2_erodeSize=3` 与 `step3_erodeSize=3` 违反 `validate()` 约束，加载即抛异常 | 预设不可用 |
| 🟡 中 | `MaskSeparationTimings.upload_ms` 计时不可靠：`executePipeline()` 重新记录 `event_start_`（在上传之后），导致 `cudaEventElapsedTime(event_start_, event_upload_done_)` 中 start 晚于 end，返回错误值 | `upload_ms` 为垃圾值；`total_pipeline_ms` 不受影响（正确计量纯计算耗时） |
| 🟡 中 | `gaussianSize` 参数在 `validate()` 中校验，但内核固定使用 5×5 高斯权重，修改该参数无效果 | 误导使用者认为高斯核可调 |
| 🟡 中 | Step 4 和 Step 5 融合为单次内核调用，`step5_extract_marking_ms` 计时约为 0 | 无法独立评估 Step 5 耗时 |
| 🟢 低 | 形态学内核大小受限于 {3,5,7,9,11}，更大核会抛异常 | 特殊场景下需修改内核 switch |
| 🟢 低 | 各步无显式 `cudaGetLastError` 检查（Release），内核错误仅在 catch 块中暴露 | 错误定位困难 |
| 🟢 低 | `Execute(grayImage)` 单参重载内部使用 `static thread_local Stream`（流在线程生命周期内持久） | 实现细节，多实例共享同一线程流 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `mask_separation`（`LaserMarkingSeparationCUDA`） |
| **复用方式** | 完整实现，6 步形态学流水线 + 自定义 CUDA 内核，替代旧版 `mask_extract`（`MaskExtractCUDA`） |

---

> **文档结束**
