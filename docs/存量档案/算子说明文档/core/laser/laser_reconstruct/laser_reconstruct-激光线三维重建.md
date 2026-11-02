# 激光线三维重建

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-8 |
| 中文名称 | 激光线三维重建 |
| 英文目录名 | laser_reconstruct |
| 运行平台 | CUDA (CUB) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | d_matched_left | CV_32FC2 |
| **输入②** | d_matched_right | CV_32FC2 |
| **输入③** | d_matched_line_ids | CV_32SC1 |
| **输入④** | Q矩阵 | cv::Mat 4×4 CV_64FC1 |
| **输出** | LaserReconstructResult 含 d_points3d, d_valid_line_ids, validCount, totalInput | CV_32FC3 / CV_32SC1 |

## C. 算法

**核心流程**：

```
Step 1: 计算视差 d = left.x - right.x (kernelComputeDisparity)
Step 2: 使用Q矩阵三维重建: X=(Q[0]*x+Q[3])/W, Y=(Q[5]*y+Q[7])/W, Z=Q[11]/W, W=Q[14]*d+Q[15] (kernelReconstruct3D, Q存储在constant memory)
Step 3: 深度范围过滤 [minDepth, maxDepth]
Step 4: CUB DeviceSelect::Flagged 压缩有效点
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| kernelComputeDisparity | 计算左右匹配点视差 |
| kernelReconstruct3D | 利用Q矩阵进行三维重建 |
| \_\_constant\_\_ float c_Qf[16] | Q矩阵常量内存存储 |
| cub::DeviceSelect::Flagged | 压缩有效点集 |

## D. 依赖

**上下游算子**：

```
laser_match（标定）/ laser_match_scan（扫描） → 本算子
                                                    ├→ endpoint_extract（标定）
                                                    ├→ pose_optimize（标定）
                                                    └→ laser_cloud_fuse（扫描）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| Q矩阵 | 立体标定输出的视差转深度矩阵 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_warmup_config.h` | WarmupConfig 结构体 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| laser_match（标定） | shared_ptr<GpuMat> | d_matched_left, d_matched_right, d_matched_line_ids |
| laser_match_scan（扫描） | shared_ptr<GpuMat> | d_matched_left, d_matched_right, d_matched_line_ids |
| **stereo_rectify_temp_table** | **cv::Mat 参数传入** | **Q矩阵（4×4 CV_64FC1），当前温度条目的视差转深度映射矩阵** |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_points3d | endpoint_extract（标定流程） | Execute() | shared_ptr<GpuMat> | 必用 |
| d_valid_line_ids | endpoint_extract（标定流程） | Execute() | shared_ptr<GpuMat> | 必用 |
| d_points3d | pose_optimize（标定流程） | Execute() | shared_ptr<GpuMat> | 必用 |
| d_valid_line_ids | pose_optimize（标定流程） | Execute() | shared_ptr<GpuMat> | 必用 |
| d_points3d（D2H） | laser_cloud_fuse（扫描流程） | Execute() | vector\<Point3f\>（相机系点云，配单帧配准 R/T 做全局变换） | 必用 |

## E. 架构

**文件结构**：

```
laser_reconstruct/
├── laser_reconstruct_cuda.h
├── laser_reconstruct_cuda.cpp
├── laser_reconstruct_cuda_pimpl.h
├── laser_reconstruct_cuda_impl.cu
└── tests/
    └── test_laser_reconstruct_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | LaserReconstructCuda (pImpl) |
| 核心方法 | Execute() |
| 销毁方法 | Destroy() |
| 预热方法 | Warmup(pointCount) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | LaserReconstructParams |
| 结果结构体 | LaserReconstructResult |
| 日志标签 | "08-LaserReconstructCuda" |

**错误模型**：

> 本算子遵循算子规范 §5.1 单一错误模型：`Result` 仅含 `bool success` + `calib::QualityFlag qualityFlag` + `std::string message`，无第二套整型错误码。失败时 `success=false`，错误细节见 `message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA | >= 11.0 | GPU并行计算 |
| OpenCV | >= 4.x | 矩阵与数据结构 |
| CUB | CUDA内置 | 设备端选择压缩 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| minDepth | float | 0.0 | >=0 | 最小深度（单位 mm） |
| maxDepth | float | 10000.0 | >min | 最大深度（单位 mm） |
| deviceId | int | 0 | >=0 | GPU设备ID |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| Q矩阵存储 | 使用constant memory（异步上传） |
| 无效点判定 | W<=0的点标记为无效 |

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
| 类型不符 / Q 矩阵非法 / 元素数不匹配 | 返回 success=false |
| 无 CUDA 设备 (cudaGetDeviceCount<=0) | 构造时抛出 std::runtime_error |
| deviceId 越界 (>=设备数) | 构造时抛出 std::invalid_argument |
| CUDA kernel 启动失败 | 返回 success=false |
| W<=0 | 标记为无效点并过滤 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| - | 无已知TODO | - |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
