# Steger激光中心亚像素提取

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-4 |
| 中文名称 | Steger激光中心亚像素提取 |
| 英文目录名 | steger |
| 运行平台 | CUDA (Thrust + 自定义kernels) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类（0.01~0.1像素级别，CPU vs CUDA<0.05px） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 灰度图 | d_grayImage (CV_8UC1 GpuMat) |
| **输入②** | 标记掩膜 | d_labeledMask (CV_32SC1 GpuMat) |
| **输出** | Steger提取结果 | StegerResult 含 centerPoints(map<int,vector<Point2f>>), d_centerPoints(CV_32FC2), d_line_ids(CV_32SC1), totalPointCount, lineCount |

> **双模式（`GroupMode` 为 `Execute(...)` 方法实参，`enum class` 类型，非 `StegerParams` 字段）**：
> - `ByLabel`（标定，默认）：输入② = CV_32SC1 标签图，按 label 逐线分组输出。
> - `Flat`（扫描）：输入② = CV_8UC1 二值掩膜，内部转均匀标签图（前景→1）后走同一 Hessian 流程，**不分组**，输出统一 `line_id`（占位）。Hessian 内核零改动。扫描流水线采集的激光线断裂+无序，逐线编号推迟到 `laser_match_scan` 全局查表完成。

## C. 算法

**核心流程**：

```
Step 1: CV_8UC1 → CV_32FC1 转换
Step 2: 行列分离高斯卷积计算 Ix, Iy, Ixx, Iyy, Ixy（5次行列分离卷积，10次kernel调用）
Step 3: Hessian矩阵特征值+特征方向计算 + 阈值过滤
Step 4: 泰勒展开亚像素修正（t < 0.5过滤）
Step 5: Thrust stable_sort 按标签排序
Step 6: D2H拷贝构建CPU结果 + GPU输出数组
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| GaussianConvRowKernel | 行方向高斯卷积 |
| GaussianConvColKernel | 列方向高斯卷积 |
| HessianEigenAndTaylorKernel | Hessian特征值计算+泰勒亚像素修正 |
| thrust::stable_sort | 按标签稳定排序 |

## D. 依赖

**上下游算子**：

```
mask_separation (d_laserMask) + 外部灰度图 (d_grayImage) ─┐
                                                              ├→ 本算子 → undistort_cuda
laser_label (d_labeledMask) ───────────────────────────────┘
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| d_grayImage | 外部灰度图（Host/Device 双副本投递），非算子产出 |
| d_labeledMask | 来自laser_label，共享引用 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 外部灰度图 | GpuMat const& | d_grayImage 灰度图（Host/Device 双副本投递，非算子产出） |
| mask_separation | shared_ptr<GpuMat> | d_laserMask 激光线掩膜（扫描 Flat 模式输入②，CV_8UC1） |
| laser_label | shared_ptr<GpuMat> | d_labeledMask 重编号标签图（标定 ByLabel 模式输入②，CV_32SC1） |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_centerPoints | undistort_cuda | UndistortPointsCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| d_line_ids | undistort_cuda | UndistortPointsCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| centerPoints | 外部 | CPU端使用 | map 值拷贝 | 可选 |
| totalPointCount | 外部 | 统计信息 | 值传递 | 可选 |
| lineCount | 外部 | 统计信息 | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
steger/
├── steger_extract_cuda.h
├── steger_extract_cuda.cpp
├── steger_extract_cuda_pimpl.h
├── steger_extract_cuda_impl.cu
└── tests/
    └── test_steger_extract_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | StegerExtractorCUDA (pImpl) |
| 核心方法 | Execute() (4 重载：2参 ByLabel 兼容 / 4参 ByLabel+Flat 双模式) |
| 销毁方法 | Destroy() |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | StegerParams |
| 结果结构体 | StegerResult |
| 日志标签 | "10-StegerExtractorCUDA" |

> 注：日志标签编号沿用全局统一编号方案，与算子文件夹编号不同。
> 注：v2.3 移除 `SE_*` int 错误码常量，统一使用 `success + qualityFlag` 单一错误模型。移除 `getStegerErrorString()` 辅助函数。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_core, opencv_imgproc, opencv_cudaimgproc, opencv_cudaarithm, opencv_cudafilters |
| CUDA Toolkit | 12.6 | Thrust 库, sm_75, sm_86, sm_87, --extended-lambda |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| sigma | float | 1.5 | [0.5,10.0] | 高斯sigma |
| kernelSize | int | 0 | 0(auto)/3/5/7/9 | 卷积核大小 |
| lowThreshold | float | 2.0 | >=0 | Hessian特征值低阈值 |
| highThreshold | float | 0.0 | >=0 | 高阈值(0=无上限) |
| maxLabels | int | 256 | [1,4096] | 最大标签数 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| GPU缓冲区 | 最大为 rows*cols（全图像素），大图可能OOM |
| 编译约束 | 需要 CUDA --extended-lambda 编译支持 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |
| Degraded | 降级 | 某些线点数<10 |
| Warning | 警告 | 提取点数为 0 或 >50%像素（可能噪声） |

**错误处理模式**（单一错误模型 `success + qualityFlag`）：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 / 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (getCudaEnabledDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel 启动失败 | 返回 success=false |
| 线点数过少 | 返回Degraded |
| 提取点数过多 | 返回Warning |

> 注：v2.3 起移除 `SE_*` int 错误码常量，统一使用 `success + qualityFlag` 单一错误模型。

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | GPU缓冲区最大为 rows*cols（全图像素），大图可能OOM | 超大图像处理失败 |
| 🟢 低 | 需要CUDA --extended-lambda编译支持 | 编译环境要求 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
