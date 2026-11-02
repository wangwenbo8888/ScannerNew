# 激光点云体素哈希融合（CUDA）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 融合-2C |
| 中文名称 | 激光点云体素哈希融合（CUDA） |
| 英文目录名 | laser_cloud_fuse_cuda |
| 运行平台 | CUDA（GPU） |
| 所属流程 | 融合流程（跨帧密集激光点云去重/累积，GPU 加速版） |
| 精度档次 | ③ 体素级（浮点/体素粒度） |
| 对标 CPU 版 | `laser_cloud_fuse`（融合-02） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 单帧激光 3D 点云（相机坐标系，**显存驻留**） | `cv::cuda::GpuMat`（CV_32FC3，1×N） |
| **输入②**（可选） | 刚体变换 R/T（相机系→全局系） | `cv::Matx33d R` + `cv::Vec3d T`；不传时恒等变换 |
| **输出** | 融合结果 | `LaserCloudFuseCudaResult`（统计标量 + 标志） |

> **零拷贝管线**：输入 `GpuMat` 直接来自上游 `laser_reconstruct`（CUDA），全程显存，无 H2D 批量传输。R/T 仅 12 个 float，按值传入 kernel。
>
> **常驻显存**：哈希表（`d_keys` / `d_fusedIdx` / `d_counts`）和点缓冲（`d_fusedXyz` / `d_fusedNormal`）跨帧持久化于显存。新扫描开始时需调用 `Clear()`。

---

## C. 算法

**核心 kernel**（`fuseKernel`，每点一个 CUDA 线程）：

```
Step 1: 坐标变换 — gx = R*p + T（RTParams 结构体按值传入，12 float）
Step 2: 体素量化 — ix/iy/iz = floor(g / voxelSize)
Step 3: 体素键打包 — key = packVoxelKeyDevice(ix, iy, iz)（21 bit/轴 + 2^20 偏置）
Step 4: 哈希 + atomicCAS 探测 —
        storedKey = key + 1（0 保留为空槽标记）
        idx = hash64(key) & mask
        loop:
          old = atomicCAS(&d_keys[idx], 0, storedKey)
          ├─ old == 0 → 新体素！写代表点 + atomicAdd(d_fusedPointCount)
          ├─ old == storedKey → HIT
          └─ 其他 → 碰撞，idx = (idx+1) & mask
Step 5: 统一饱和计数 — oldCount = atomicAdd(&d_counts[idx], 1)
        oldCount < threshold → KEEP；否则 DELETE
```

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `atomicCAS`（64-bit） | 并发哈希表插入：`atomicCAS(&d_keys[idx], 0, storedKey)` 原子地"_claim or find" |
| `atomicAdd`（32-bit） | 统一饱和计数：所有线程（创建者 + 命中者）统一使用，消除竞态 |
| `hash64Device` | splitmix64 finalizer（与 CPU 版一致），device 内联函数 |
| `packVoxelKeyDevice` | 21 bit/轴打包，越界返回 `UINT64_MAX` 跳过 |
| `RTParams` 结构体 | R/T 按值传递给 kernel（CUDA 自动拷贝到常量内存），避免 host 指针解引用 |
| 开放寻址线性探测 | `idx = (idx + 1) & mask`，上限 4096 次探测 |

> **竞态消除**：CPU 版用单线程串行处理避免竞态；GPU 版将"新体素写入"与"计数"分离——创建者先写代表点数据，所有线程统一用 `atomicAdd(&d_counts, 1)` 做饱和判定。第一个 `atomicAdd` 返回 0 的线程算 KEEP，后续线程按返回值判定。这保证 `threshold` 个线程 KEEP，其余 DELETE，与 CPU 版语义一致。

---

## D. 依赖

**上下游算子**：

```
laser_reconstruct (CUDA, GpuMat 输出) → [本算子] → laser_cloud_normal_cuda / 全局点云导出
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `d_keys` / `d_fusedIdx` / `d_counts` | 哈希表 SoA，**常驻显存**，跨帧持久化 |
| `d_fusedXyz` / `d_fusedNormal` | 融合点缓冲，法线由 `laser_cloud_normal_cuda` 写入 |
| `d_fusedPointCount` | 设备端原子计数器，跨帧持久化 |
| `LaserCloudFuseDeviceContext` | POD 结构体，封装上述指针供法线算子 kernel 访问 |

**头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | `QualityFlag` |
| `common/calib_logging.h` | 日志宏 |
| `common/calib_warmup_config.h` | `WarmupConfig` |
| `opencv2/core/cuda_stream_accessor.hpp` | `StreamAccessor::getStream()` |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `laser_reconstruct` | `const cv::cuda::GpuMat&`（CV_32FC3） | 点云已驻留显存，零拷贝 |
| 单帧配准（CPU） | `cv::Matx33d R` + `cv::Vec3d T` | 12 个标量，host 传入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 |
|---------|--------|---------|---------|
| `GetDeviceContext()` | `laser_cloud_normal_cuda` | `LaserCloudFuseDeviceContext`（POD 结构体） | 值传递，含所有设备指针 |
| `d_fusedXyz` / `d_fusedNormal` | 显示/渲染 | 直接读显存 | 零拷贝 |
| `result.newVoxelCount` | 法线算子范围推导 | `int` 标量 D2H | 4 字节 |

**调用示例**：

```cpp
LaserCloudFuseCuda fuse;
LaserCloudNormalCuda normalOp;

// 每帧：
auto fuseResult = fuse.Execute(d_points3d, R, T, stream);

auto normalResult = normalOp.Execute(fuse, fuseResult, stream);

// 显示侧直接读显存：
auto ctx = fuse.GetDeviceContext();
// ctx.d_fusedXyz / ctx.d_fusedNormal 已就绪
```

---

## E. 架构

**文件结构**（4 文件模式）：

```
laser_cloud_fuse_cuda/
├── laser_cloud_fuse_cuda.h              # 公开头文件（无 CUDA 类型）
├── laser_cloud_fuse_cuda_pimpl.h        # Impl 结构体（含设备指针）
├── laser_cloud_fuse_cuda.cpp            # Host bridge（#if BUILD_CUDA）
├── laser_cloud_fuse_cuda_impl.cu        # CUDA kernel + Impl
└── tests/
    └── test_laser_cloud_fuse_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserCloudFuseCuda`（pImpl，**有状态**） |
| 核心方法 | `Execute(GpuMat, stream)` / `Execute(GpuMat, R, T, stream)` |
| 设备上下文 | `GetDeviceContext()` → `LaserCloudFuseDeviceContext`（供法线算子） |
| 重置 | `Clear()` — 清空哈希表 + 点缓冲（GPU kernel 清零） |
| 预分配 | `Reserve(voxelCount)` |
| 预热 | `Warmup(maxPointCount)` |
| 状态查询 | `GetVoxelCount()` / `GetFusedPointCount()` |
| Stream 支持 | `cv::cuda::Stream&` 参数，支持异步 |
| 参数结构体 | `LaserCloudFuseCUDAParams` |
| 结果结构体 | `LaserCloudFuseCudaResult`（move-only） |
| 日志标签 | `"02C-LaserCloudFuseCuda"` |

**设备数据结构（SoA 哈希表，常驻显存）**：

| 数组 | 类型 | 说明 |
|------|------|------|
| `d_keys` | `unsigned long long*` | voxel key + 1（0 = 空槽），atomicCAS 目标 |
| `d_fusedIdx` | `unsigned int*` | → `d_fusedXyz` 索引 |
| `d_counts` | `unsigned int*` | 饱和计数（atomicAdd） |
| `d_fusedXyz` | `float*` | 代表点坐标（maxPoints × 3） |
| `d_fusedNormal` | `float*` | 法线（由法线算子写入） |
| `d_fusedPointCount` | `unsigned int*` | 设备端原子计数器 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA Toolkit | ≥ 12.x | nvcc 编译，sm_75 / sm_86 / sm_87 |
| OpenCV | ≥ 4.x（CUDA 版） | `cuda_stream_accessor.hpp`、`GpuMat` |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

> **不依赖** CUB、Thrust、Eigen、FLANN、PCL — 哈希表和 kernel 均为全自定义实现。

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `voxelSize` | float | 0.5 | > 0 | 体素边长（mm） |
| `saturationThreshold` | int | 5 | ≥ 1 | 每体素最大保留点数 |
| `reserveVoxelCount` | size_t | 2M (2²¹) | ≥ 64 | 哈希表槽数（向上取 2 的幂），决定显存占用 |
| `collectStatistics` | bool | true | — | 是否填充 result 统计 |

**显存占用**：`reserveVoxelCount × 16 字节`（keys 8B + fusedIdx 4B + counts 4B） + `reserveVoxelCount × 24 字节`（fusedXyz 12B + fusedNormal 12B）。默认 2M 槽 = **80MB**。

**预设配置**：`default`（0.5mm, 5 点, 2M 槽） / `fine`（0.2mm, 8 点, 8M 槽） / `coarse`（1.0mm, 3 点, 512K 槽）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 体素级 |
| 吞吐目标 | ~50M 点/秒（50K 点 < 1ms） |
| 体素坐标范围 | 每轴 [-1,048,576, +1,048,575]（21 bit） |
| 线程安全 | **非线程安全**（单个 fuse 实例不可并发调用） |
| 状态管理 | **有状态**（哈希表 + 点缓冲常驻显存，`Clear()` 重置） |
| 负载因子 | 预分配固定容量，不做 GPU rehash；超 4096 次探测判定表满 |
| Stream | 支持 `cv::cuda::Stream` 异步执行 |

---

## K. 质量

**QualityFlag 语义**（基于保留率 = survivingCount / inputCount）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 保留率 ≥ 0.5 |
| `Degraded` | 降级但可用 | 保留率 ∈ [0.1, 0.5) |
| `Warning` | 需关注 | 保留率 < 0.1 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 | 抛出 `std::invalid_argument` |
| 输入格式错误（非 1×N CV_32FC3） | `success=false`，`message` 描述原因 |
| 点缓冲溢出 | 超限点标记为 DELETE，不崩溃 |
| 探测超限（表接近满） | 超限点标记为 DELETE |
| `BUILD_CUDA=OFF` | 桩实现，`success=false`，`message="BUILD_CUDA is OFF"` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 哈希表容量固定，不做 GPU rehash；长时间扫描累积超限时点被丢弃 | 需预估最大唯一体素数，`Reserve()` 充分预留 |
| 🟡 中 | `atomicCAS` 在高负载因子下竞争加剧 | 保持负载因子 < 50%（容量 ≥ 2× 预期体素数） |
| 🟢 低 | `GetDeviceContext()` 返回的设备指针在 `Clear()` / 析构后失效 | 下游不应跨帧持有（每帧重新获取） |
| 🟢 低 | R/T 按值传入 kernel（float 精度），与 CPU 版 double→float 一致 | mm 级坐标精度无损 |
| 🟢 低 | 越界体素坐标（超 21-bit）静默跳过 | 实际扫描范围 ±524m 内不会触发 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `laser_cloud_fuse_cuda`（`LaserCloudFuseCuda`） |
| **复用方式** | 完整实现：atomicCAS 并发哈希表 + 统一饱和计数 + GpuMat I/O + DeviceContext 上下文传递。9 个测试覆盖单点/重复/饱和/网格/跨帧/R-T/Clear/上下文/空帧 |
| **与 CPU 版关系** | 独立共存，接口对应（概念一致，I/O 为 GpuMat），调用方按硬件选择 |

---

## 性能对比

| 场景 | CPU 版 (`laser_cloud_fuse`) | CUDA 版（本算子） | 加速比 |
|------|---------------------------|------------------|--------|
| 50K 点融合 | ~5ms | ~0.5-1ms（预估） | 5-10× |
| H2D/D2H | 需要（~0.5ms） | **零**（数据已在显存） | — |

---

> **文档结束**
