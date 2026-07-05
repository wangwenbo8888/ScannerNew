# 激光线编号（重编号）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-3 |
| 中文名称 | 激光线编号（重编号） |
| 英文目录名 | laser_label |
| 运行平台 | CUDA (Thrust) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ② 整像素/几何类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 标记掩膜 | cv::cuda::GpuMat (CV_32SC1, 来自02的标记掩膜) |
| **输出** | 重编号结果 | LaserLabelResult 含 d_labeledMask(CV_32SC1, 1-indexed按Y排序), componentCount |

## C. 算法

**核心流程**：

```
Step 1: 输入判断（仅CV_32SC1）
Step 2: 计算中心列 center_x = cols/2 + centerColOffset
Step 3: cv::cuda::minMax获取最大标签值
Step 4: InitScanBuffersKernel: 初始化扫描缓冲区
Step 5: ScanCenterColumnKernel: 中心列扫描取每个标签最小Y值
Step 6: Thrust stable_sort_by_key: 按Y值排序标签
Step 7: BuildMapTableKernel: 构建旧标签→新编号映射
Step 8: RelabelKernel: 全图重编号
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| thrust::stable_sort_by_key | 按Y值稳定排序标签 |
| cv::cuda::minMax | 获取最大标签值 |
| InitScanBuffersKernel | 初始化扫描缓冲区 |
| ScanCenterColumnKernel | 中心列扫描取每个标签最小Y值 |
| BuildMapTableKernel | 构建旧标签→新编号映射 |
| RelabelKernel | 全图重编号 |

## D. 依赖

**上下游算子**：

```
ccl → 本算子 → steger
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无 | 独立实例 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| ccl | GpuMat 引用 | d_labeledMask (CV_32SC1) |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_labeledMask | steger | StegerExtractorCUDA | GpuMat 引用 | 必用 |
| componentCount | steger | StegerExtractorCUDA | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
laser_label/
├── laser_label_cuda.h
├── laser_label_cuda.cpp
├── laser_label_cuda_pimpl.h
├── laser_label_cuda_impl.cu
└── tests/
    └── test_laser_label_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | LaserLabelerCUDA (pImpl) |
| 核心方法 | Execute() |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | LaserLabelParams |
| 结果结构体 | LaserLabelResult |
| 日志标签 | "09-LaserLabelerCUDA" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA Toolkit | 12.6 | Thrust 库, sm_75, sm_86, sm_87 |
| OpenCV + CUDA | 4.13.0 | opencv_core, opencv_imgproc, opencv_cudaimgproc, opencv_cudaarithm |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| maxLabels | int | 256 | [1,4096] | 最大标签数 |
| centerColOffset | int | 0 | [-500,500] | 中心列偏移 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 输入约束 | 仅接受 CV_32SC1 标记掩膜 |
| 标签数限制 | maxLabels 超限直接返回失败 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |
| Degraded | 降级 | componentCount > 200 |
| Warning | 警告 | componentCount = 0 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型非 CV_32SC1 | 返回 success=false |
| 无 CUDA 设备 (getCudaEnabledDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel 启动失败 | 返回 success=false |
| maxLabels超限 | 直接返回失败 |
| 无连通域 | 返回Warning |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | maxLabels超限时直接返回失败 | 下游无法继续处理 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
