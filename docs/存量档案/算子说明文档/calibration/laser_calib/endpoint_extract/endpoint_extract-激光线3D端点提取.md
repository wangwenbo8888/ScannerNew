# 激光线3D端点提取

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-9 |
| 中文名称 | 激光线3D端点提取 |
| 英文目录名 | endpoint_extract |
| 运行平台 | CUDA (CUB) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_points3d | CV_32FC3 |
| **输入②** | d_line_ids | CV_32SC1 |
| **输出** | EndpointExtractResult 含 d_endpoints, d_endpoint_ids, d_line_ids, numEndpoints, numLines, totalInput, StepTiming | CV_32FC3 / CV_32SC1 |

## C. 算法

**核心流程**：

```
Step 1: CUB DeviceReduce::Max 找最大line_id
Step 2: kernelComputeLineSums: 每条线点坐标之和与计数
Step 3: kernelNormalizeCentroids: 计算每条线质心
Step 4: kernelFindMaxDist: 找距质心最远点距离平方
Step 5: kernelFindMaxDistIdx: 找距质心最远点索引（端点A）
Step 6: kernelGatherRef: 收集端点A坐标
Step 7: kernelFindMaxDist: 找距端点A最远点距离平方
Step 8: kernelFindMaxDistIdx: 找距端点A最远点索引（端点B）
Step 9: kernelCollectEndpoints: 收集端点并编号
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cub::DeviceReduce::Max | 找最大line_id |
| kernelComputeLineSums | 计算每条线的坐标累加和与点数 |
| kernelNormalizeCentroids | 归一化得到线质心 |
| kernelFindMaxDist | 找最远点距离平方 |
| kernelFindMaxDistIdx | 找最远点索引 |
| kernelGatherRef | 收集参考端点坐标 |
| kernelCollectEndpoints | 收集并编号两端点 |
| atomicAdd / atomicMax | 原子操作用于并行归约 |

## D. 依赖

**上下游算子**：

```
laser_reconstruct → 本算子 → virtual_camera_pose
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| GPU临时缓冲区 | CUB归约/扫描所需工作空间 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| laser_reconstruct | GPU显存直传 | d_points3d, d_valid_line_ids |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_endpoints | virtual_camera_pose | Execute | GPU显存直传 | 必用 |
| d_line_ids | virtual_camera_pose | Execute | GPU显存直传 | 必用 |
| numEndpoints | virtual_camera_pose | Execute | 值传递 | 可选 |
| numLines | virtual_camera_pose | Execute | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
endpoint_extract/
├── endpoint_extract_cuda.h
├── endpoint_extract_cuda.cpp
├── endpoint_extract_cuda_pimpl.h
├── endpoint_extract_cuda_impl.cu
└── tests/
    └── test_endpoint_extract_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | EndpointExtractCuda (pImpl) |
| 核心方法 | Execute |
| 预热方法 | Warmup(pointCount, maxFrameId) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | EndpointExtractParams |
| 结果结构体 | EndpointExtractResult (含 StepTiming) |
| 日志标签 | "09-EndpointExtractCuda" |

**StepTiming 字段**（enableTiming=true 时填充）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `us_findMaxFid` | float | 查找最大 line_id 耗时 (μs) |
| `us_memsetBuffers` | float | 缓冲区清零耗时 (μs) |
| `us_computeLineSums` | float | 每条线坐标累加和与计数耗时 (μs) |
| `us_normalizeCentroids` | float | 归一化得到线质心耗时 (μs) |
| `us_findMaxDistA` | float | 查找距质心最远点距离耗时 (μs) |
| `us_findMaxDistIdxA` | float | 查找距质心最远点索引耗时 (μs) |
| `us_gatherRefA` | float | 收集端点A坐标耗时 (μs) |
| `us_findMaxDistB` | float | 查找距端点A最远点距离耗时 (μs) |
| `us_findMaxDistIdxB` | float | 查找距端点A最远点索引耗时 (μs) |
| `us_collectEndpoints` | float | 收集并编号两端点耗时 (μs) |
| `us_d2hCopy` | float | Device→Host 拷贝耗时 (μs) |
| `us_total` | float | 总耗时 (μs) |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA | >= 11.0 | GPU并行计算 |
| CUB | CUDA内置 | 设备端归约 |
| OpenCV | >= 4.x | 矩阵数据结构 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| deviceId | int | 0 | >=0 | GPU设备ID |
| maxExpectedLines | int | 4096 | >0 | 预期最大线数 |
| enableTiming | bool | false | — | 是否启用详细计时 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 计时精度 | 支持详细分步计时（12个cudaEvent） |
| 原子操作 | atomicMax使用float_as_uint技巧比较浮点距离平方 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel / CUB 操作失败 | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 两趟近似直径算法不保证精确直径（但实际足够） | 端点精度可接受 |
| 🟢 低 | 3D 坐标精度 | 端点坐标使用 float32（CV_32FC3），源自 Q 矩阵（double64）重建后精度降级，大深度值（如 Z>5000mm）可能有亚毫米级精度损失 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
