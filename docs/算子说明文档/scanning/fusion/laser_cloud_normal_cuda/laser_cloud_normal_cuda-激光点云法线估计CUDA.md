# 激光点云法线估计（CUDA）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 融合-3C |
| 中文名称 | 激光点云法线估计（CUDA） |
| 英文目录名 | laser_cloud_normal_cuda |
| 运行平台 | CUDA（GPU） |
| 所属流程 | 融合流程（为新建体素计算法线，GPU 加速版，供显示/渲染） |
| 精度档次 | ③ 体素级 |
| 对标 CPU 版 | `laser_cloud_normal`（融合-03） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 融合算子的设备上下文（哈希表 + 点缓冲） | `LaserCloudFuseDeviceContext`（POD 结构体，含设备指针） |
| **输入②** | 新建体素索引范围 `[beginIdx, endIdx)` | `size_t` × 2；便捷重载从 `fuseResult.newVoxelCount` 推导 |
| **输出** | 法线回写 | 直接写入 `ctx.d_fusedNormal[i*3..i*3+2]`（显存） |

> **零拷贝**：通过 `LaserCloudFuseDeviceContext` 直接访问融合算子的显存数据，无任何 H2D/D2H 传输。kernel 只读哈希表 + 点坐标，写法线到同一显存区域。

---

## C. 算法

**核心 kernel**（`computeNormalsKernel`，每体素一个 CUDA 线程）：

```
Step 1: 读取体素代表点 — px/py/pz = d_fusedXyz[i*3..]
Step 2: 体素量化 — ix/iy/iz = floor(p / voxelSize)
Step 3: 邻域查找（3×3×3 = 27 体素）—
        对每个 (dx,dy,dz) ∈ {-1,0,1}³:
          key = packVoxelKeyDevice(ix+dx, iy+dy, iz+dz)
          ptr = lookupVoxelDevice(key, d_keys, d_fusedIdx, d_fusedXyz, mask)
          if found → 收集到 neighbors[]
Step 4: 退化判定 — if count < minNeighbors → 写 fallbackNormal，return
Step 5: 协方差矩阵 — 质心 μ + 3×3 对称协方差 C
Step 6: Jacobi 特征分解 — ≤12 轮旋转 → 最小特征值特征向量
Step 7: 归一化 + 写回 — d_fusedNormal[i*3..] = normalize(normal)
```

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `lookupVoxelDevice` | 只读哈希查找（无 atomic），线性探测 ≤1024 次 |
| `jacobi3x3Device` | `__device__` Jacobi 旋转法，与 CPU 版算法完全一致 |
| `neighbors[27*3]` | 栈上局部数组（324 字节/线程），L1 cache 友好 |

> **与 CPU 版差异**：GPU 版不做自适应 kernel 扩大（固定 3×3×3），邻居不足直接 fallback。速度优先。

---

## D. 依赖

**上下游算子**：

```
laser_cloud_fuse_cuda（融合-02C）→ [本算子] → 全局点云（带法线）→ 显示/渲染
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 融合哈希表（`d_keys` / `d_fusedIdx`） | 通过 `DeviceContext` **只读**访问 |
| `d_fusedXyz` | 通过 `DeviceContext` **只读**访问（邻居坐标） |
| `d_fusedNormal` | 通过 `DeviceContext` **写入**法线 |
| 无独立跨帧状态 | 本算子无状态 |

**头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `laser_cloud_fuse_cuda.h` | `LaserCloudFuseDeviceContext`、`LaserCloudFuseCuda` |
| `common/calib_types.h` | `QualityFlag` |
| `common/calib_logging.h` | 日志宏 |
| `opencv2/core/cuda_stream_accessor.hpp` | `StreamAccessor::getStream()` |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `laser_cloud_fuse_cuda` | `LaserCloudFuseDeviceContext`（POD） | 哈希表 + 点缓冲的设备指针 |
| `laser_cloud_fuse_cuda` result | `const LaserCloudFuseCudaResult&` | `newVoxelCount` 推导范围（便捷重载） |

**本算子→下游**：

| 输出字段 | 传递给 | 方式 |
|---------|--------|------|
| `ctx.d_fusedNormal` | 显示/渲染 | 直接读显存，零拷贝 |
| `result.statistics` | 性能监控 | `LaserCloudNormalCudaResult` |

**调用示例**：

```cpp
LaserCloudFuseCuda fuse;
LaserCloudNormalCuda normalOp;

auto fuseResult = fuse.Execute(d_points3d, R, T, stream);
auto normalResult = normalOp.Execute(fuse, fuseResult, stream);

// d_fusedNormal 已就绪于显存，显示侧直接读取
```

---

## E. 架构

**文件结构**（4 文件模式）：

```
laser_cloud_normal_cuda/
├── laser_cloud_normal_cuda.h             # 公开头文件
├── laser_cloud_normal_cuda_pimpl.h       # Impl 结构体
├── laser_cloud_normal_cuda.cpp           # Host bridge（#if BUILD_CUDA）
├── laser_cloud_normal_cuda_impl.cu       # CUDA kernel + Impl
└── tests/
    └── test_laser_cloud_normal_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserCloudNormalCuda`（pImpl，**无状态**） |
| 核心方法 | `Execute(ctx, beginIdx, endIdx, stream)` / `Execute(fuse, fuseResult, stream)` |
| Stream 支持 | `cv::cuda::Stream&` |
| 参数结构体 | `LaserCloudNormalCUDAParams` |
| 日志标签 | `"03C-LaserCloudNormalCuda"` |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CUDA Toolkit | ≥ 12.x | sm_75 / sm_86 / sm_87 |
| OpenCV | ≥ 4.x（CUDA 版） | `GpuMat`、`Stream` |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

> **不依赖** Eigen、CUB、PCL — Jacobi 特征分解为全自定义 `__device__` 实现。

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `minNeighbors` | int | 3 | ≥ 3 | PCA 最小邻居数（含中心体素） |
| `fallbackNx` | float | 0.0 | 单位向量 | 退化法线 X |
| `fallbackNy` | float | 0.0 | 单位向量 | 退化法线 Y |
| `fallbackNz` | float | 1.0 | 单位向量 | 退化法线 Z |

> **注意**：无 `kernelRadius` 参数（固定为 1 = 3×3×3）。与 CPU 版不同，GPU 版不做自适应扩大。

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 体素级（近似法线，仅供显示/渲染） |
| 吞吐目标 | 10K 新建体素 < 0.5ms（预估） |
| 邻域 | 固定 3×3×3（27 体素），不自适应扩大 |
| 线程安全 | **非线程安全** |
| 状态管理 | **无状态** |
| Stream | 支持 `cv::cuda::Stream` |

---

## K. 质量

**QualityFlag 语义**（基于退化率 = fallbackCount / total）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 退化率 < 10% |
| `Degraded` | 降级 | 退化率 ∈ [10%, 50%) |
| `Warning` | 需关注 | 退化率 ≥ 50% 或空范围 |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `DeviceContext` 中的设备指针在融合算子 `Clear()` / 析构后失效 | 每帧重新获取，不跨帧持有 |
| 🟢 低 | 固定 3×3×3 邻域，稀疏区域退化率较高 | 可接受（速度优先） |
| 🟢 低 | PCA 法线有 ±符号歧义（不定向） | 双面渲染不受影响 |
| 🟢 低 | `BUILD_CUDA=OFF` 时桩实现抛异常 | 仅影响无 GPU 环境 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `laser_cloud_normal_cuda`（`LaserCloudNormalCuda`） |
| **复用方式** | 完整实现：只读哈希查找 + `__device__` Jacobi + DeviceContext 零拷贝。4 个测试覆盖平面法线/退化/空范围/跨帧 |
| **与 CPU 版关系** | 独立共存。GPU 版固定 kernelRadius=1（无自适应扩大），速度优先 |

---

## 性能对比

| 场景 | CPU 版 (`laser_cloud_normal`) | CUDA 版（本算子） | 加速比 |
|------|------------------------------|------------------|--------|
| 10K 体素 | ~3.7ms | ~0.2-0.5ms（预估） | 7-18× |
| H2D/D2H | 需要 | **零**（显存直读直写） | — |

---

> **文档结束**
