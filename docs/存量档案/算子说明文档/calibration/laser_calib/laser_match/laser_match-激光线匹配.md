# 激光线匹配（立体匹配）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-7 |
| 中文名称 | 激光线匹配（立体匹配） |
| 英文目录名 | laser_match |
| 运行平台 | CUDA (CUB) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机点集 | d_left_points(CV_32FC2 GpuMat), d_left_line_ids(CV_32SC1 GpuMat) |
| **输入②** | 右相机点集 | d_right_points(CV_32FC2 GpuMat), d_right_line_ids(CV_32SC1 GpuMat) |
| **输出** | 匹配结果 | LaserMatchResult 含 d_matched_left, d_matched_right, d_matched_line_ids, matchCount |

## C. 算法

**核心流程**：

```
Step 1: 量化左右点集行索引 (kernelQuantizeRowIdx: y/step取整)
Step 2: 构建左点哈希表 (kernelBuildHash: composite key = rowIdx<<16 | frameId, 开放寻址, max_probe≤128)
Step 3: 探测右点匹配 (kernelProbeMatch: 查哈希+视差范围过滤[min_disp, max_disp])
Step 4: CUB DeviceSelect::Flagged 压缩输出
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| kernelQuantizeRowIdx | 量化行索引 y/step 取整 |
| kernelBuildHash | 构建左点哈希表（开放寻址） |
| kernelProbeMatch | 右点探测匹配（查哈希+视差过滤） |
| cub::DeviceSelect::Flagged | 压缩输出有效匹配对 |
| atomicCAS | 哈希表并发插入 |

## D. 依赖

**上下游算子**：

```
epipolar_interp（左相机） ─┐
                                   ├→ 本算子 → laser_reconstruct
epipolar_interp（右相机） ─┘
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
| epipolar_interp（左） | GpuMat 引用 | d_interpPoints, d_interp_line_ids |
| epipolar_interp（右） | GpuMat 引用 | d_interpPoints, d_interp_line_ids |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_matched_left | laser_reconstruct | LaserReconstructCuda | GpuMat 引用 | 必用 |
| d_matched_right | laser_reconstruct | LaserReconstructCuda | GpuMat 引用 | 必用 |
| d_matched_line_ids | laser_reconstruct | LaserReconstructCuda | GpuMat 引用 | 必用 |
| matchCount | laser_reconstruct | LaserReconstructCuda | 值传递 | 可选 |

## E. 架构

**文件结构**：

```
laser_match/
├── laser_match_cuda.h
├── laser_match_cuda.cpp
├── laser_match_cuda_pimpl.h
├── laser_match_cuda_impl.cu
└── tests/
    └── test_laser_match_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | LaserMatchCuda (pImpl) |
| 核心方法 | Execute() |
| 预热方法 | Warmup(leftCount, rightCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | LaserMatchParams |
| 结果结构体 | LaserMatchResult |
| 日志标签 | "07-LaserMatchCuda" |

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
| epipolar_row_step | float | 0.5 | >0 | 极线行距（量化步长） |
| min_disparity | float | 0.0 | >=0 | 最小视差 |
| max_disparity | float | 500.0 | >min | 最大视差 |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 哈希表容量 | nextPowerOf2(2*leftCount), 最小16 |
| 最大探测次数 | min(leftCount, 128) |

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
| 🟡 中 | 哈希表冲突可能导致匹配遗漏 | 匹配率下降 |
| 🟢 低 | 哈希表需每帧memset 0xFF | 每帧有初始化开销 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
