# 激光中心点极线插值

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-6 |
| 中文名称 | 激光中心点极线插值 |
| 英文目录名 | epipolar_interp |
| 运行平台 | CUDA (CUB) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 矫正后点集 | d_rectifiedPoints (CV_32FC2 GpuMat) |
| **输入②** | 线标签 | d_line_ids (CV_32SC1 GpuMat) |
| **输出** | 极线插值结果 | EpipolarInterpResult 含 d_interpPoints(CV_32FC2), d_interp_line_ids(CV_32SC1), interpCount |

> **双模式（`lineIdCheck` 参数，EpipolarInterpParams 字段）**：
> - `true`（标定，默认）：同线判据 = `line_id` 相同 **且** 几何邻近（`max_x_diff`/`max_y_span`）。
> - `false`（扫描）：跳过 line_id 判据，**仅靠几何邻近**决定相邻点是否同线插值。扫描模式 steger 输出统一 line_id（无标定含义），故必须关闭此判据。

## C. 算法

**核心流程**：

```
Step 1: 对每对相邻点检查是否同帧（line_id相同）
Step 2: 检查X差值<max_x_diff, Y跨度<max_y_span
Step 3: 计算最近极线 y_target = ceil(y_min/step)*step
Step 4: 验证y_target在(y_min, y_max)之间且距离<1.0
Step 5: 线性插值 x_interp = x1 + t*(x2-x1)
Step 6: CUB DeviceSelect::Flagged 压缩输出
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| kernelMarkAndCompute | 标记待插值位置并计算插值结果 |
| cub::DeviceSelect::Flagged | 压缩输出有效插值点 |

## D. 依赖

**上下游算子**：

```
undistort_cuda → 本算子 → laser_match（标定） / laser_match_scan（扫描）
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
| undistort_cuda | shared_ptr<GpuMat> | d_rectifiedPoints (CV_32FC2) |
| undistort_cuda | shared_ptr<GpuMat> | d_line_ids (CV_32SC1) |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_interpPoints | laser_match（标定） | LaserMatchCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| d_interpPoints | laser_match_scan（扫描） | LaserMatchScanCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| d_interp_line_ids | laser_match（标定） | LaserMatchCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| d_interp_line_ids | laser_match_scan（扫描） | LaserMatchScanCuda::Execute() | shared_ptr<GpuMat> | 必用 |
| interpCount | 下游 | — | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
epipolar_interp/
├── epipolar_interp_cuda.h
├── epipolar_interp_cuda.cpp
├── epipolar_interp_cuda_pimpl.h
├── epipolar_interp_cuda_impl.cu
└── tests/
    └── test_epipolar_interp_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | EpipolarInterpCuda (pImpl) |
| 核心方法 | Execute() |
| 销毁方法 | Destroy() |
| 预热方法 | Warmup(pointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | EpipolarInterpParams |
| 结果结构体 | EpipolarInterpResult |
| 日志标签 | "06-EpipolarInterpCuda" |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_core, opencv_cudaarithm |
| CUDA Toolkit | 12.6 | CUB 库, sm_75, sm_86, sm_87 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| epipolar_row_step | float | 0.5 | >0 | 极线行距 |
| max_x_diff | float | 1.0 | >0 | 最大X差值阈值 |
| max_y_span | float | 2.0 | >0 | 最大Y跨度阈值 |
| deviceId | int | 0 | >=0 | GPU设备ID |
| lineIdCheck | bool | true | true/false | 同线判据开关（true=标定，按 line_id 判同线；false=扫描，仅几何邻近） |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 输入点数 | >0 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 正常 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument |
| 输入为空 | 返回 success=true（返回空结果，qualityFlag=Normal） |
| 类型不符 / 尺寸不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel / CUB 操作失败 | 返回 success=false |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | CUB临时存储在 Execute() 中分配（非 warmup 预分配） | 每次调用有少量内存分配开销 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
