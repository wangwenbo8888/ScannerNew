# 激光点云法线估计

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 融合-3 |
| 中文名称 | 激光点云法线估计 |
| 英文目录名 | laser_cloud_normal |
| 运行平台 | CPU（单线程） |
| 所属流程 | 融合流程（为新建体素计算法线，供显示/渲染） |
| 精度档次 | ③ 体素级（浮点/体素粒度） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 融合算子（哈希表 + fusedPoints_ 访问） | `LaserCloudFuseCPU&`（非 const，因 `FusedPointPtr` 非 const） |
| **输入②** | 新建体素索引范围 `[beginIdx, endIdx)` | `size_t` × 2；便捷重载从 `fuseResult.statistics.newVoxelCount` 推导 |
| **输出** | 法线回写 | 直接写入 `fuse.FusedPointPtr(i)->{nx,ny,nz}` |

> **新建体素范围推导**：融合-02 的 `fuseImpl` 中，新体素代表点**总是 `push_back` 到 `fusedPoints_` 末尾**（`laser_cloud_fuse_cpu.cpp:239-240`）。因此本帧新建体素为索引 `[totalCount - newVoxelCount, totalCount)`，无需额外跟踪。
>
> **仅处理新建体素**：每个体素的法线在其创建帧计算一次，后续帧不重复计算。已存在体素获得新邻居时不更新法线（精度足够用于显示/渲染）。

---

## C. 算法

**核心流程**（逐体素处理）：

```
Step 1: 邻域查找 — GatherVoxelNeighbors(pos, kernelRadius=1) → 3×3×3 共 27 体素位置
        在融合-02 哈希表中做 O(1) 查找（packVoxelKey + hash64 + tag 预过滤 + 线性探测）
        收集命中体素的代表点（const CloudPoint*），含中心体素自身
Step 2: 自适应扩大 — 若邻居数 < minNeighbors（默认 3），扩大到 kernelRadius=2（5×5×5=125 体素）
Step 3: 退化判定 — 若扩大后仍 < minNeighbors → 写入 fallbackNormal，跳过 PCA
Step 4: 协方差矩阵 — 对邻居点集计算质心 μ，再算 3×3 对称协方差 C = (1/N)Σ(pᵢ-μ)(pᵢ-μ)ᵀ
Step 5: Jacobi 特征分解 — 对 C 做 Jacobi 旋转（≤12 轮），得特征值 λ₁≥λ₂≥λ₃ 和特征向量
Step 6: 法线提取 — 最小特征值 λ₃ 对应的特征向量 = 法线方向（表面在切平面内方差大，法线方向方差最小）
Step 7: 归一化 + 写回 — 单位化法线，写入 FusedPointPtr(i)->{nx,ny,nz}
```

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `GatherVoxelNeighbors`（融合-02 方法） | 复用哈希表做 O(1) 邻域查找，零额外空间索引构建 |
| `computeCovariance3x3` | 2 趟扫描计算 3×3 对称协方差（上三角 6 元素） |
| `jacobiEigen3x3` | 经典 Jacobi 旋转法，3×3 对称矩阵 ≤12 轮收敛，返回最小特征值索引 |
| 自适应 kernel 扩大 | 邻居不足时自动从 3×3×3 扩到 5×5×5，避免边缘/稀疏区域退化 |

> **PCA 法线原理**：表面上点的协方差矩阵在切平面内有两个大特征值（表面展开方向），在法线方向有一个小特征值（表面厚度方向）。最小特征值对应的特征向量即为法线。
>
> **法线定向**：不做定向（PCA 有 ±符号歧义）。用于双面光照渲染时符号不影响显示效果。如需一致定向，可在下游根据视角翻转。

---

## D. 依赖

**上下游算子**：

```
laser_cloud_fuse（融合-02）→ [本算子] → 全局点云（带法线）→ 显示 / 渲染
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 融合-02 哈希表（`tags_/keys_/fusedIdx_/counts_`） | 通过 `GatherVoxelNeighbors()` 只读访问，**不修改** |
| 融合-02 首点缓冲（`fusedPoints_`） | 通过 `FusedPointPtr(i)` 写入法线，通过 `GetFusedPointCount()` 获取范围 |
| 无独立跨帧状态 | 本算子无状态（每次 `Execute` 从参数推导范围） |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `laser_cloud_fuse_cpu.h` | `CloudPoint`、`LaserCloudFuseCPU`、`LaserCloudFuseCPUResult` |
| `common/calib_types.h` | `QualityFlag` |
| `common/calib_logging.h` | 日志宏 |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `laser_cloud_fuse` | `LaserCloudFuseCPU&` 引用 | 哈希表邻域查找 + fusedPoints_ 法线回写 |
| `laser_cloud_fuse` result | `const LaserCloudFuseCPUResult&` | `statistics.newVoxelCount` 推导新建体素范围（便捷重载） |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `FusedPointPtr(i)->{nx,ny,nz}` | 显示/渲染 | `fuse.GetFusedPoints()` | `const vector<CloudPoint>&` 读取 | 必用 |
| `result.statistics` | 性能监控 | — | `LaserCloudNormalCPUStats` | 可选 |

**调用示例**：

```cpp
LaserCloudFuseCPU fuse;
LaserCloudNormalCPU normalOp;

// 每帧：
auto fuseResult = fuse.Execute(frame, R, T);

auto normalResult = normalOp.Execute(fuse, fuseResult);  // 便捷重载

// 显示侧直接读取：
const auto& points = fuse.GetFusedPoints();  // 每个 CloudPoint 已带 nx/ny/nz
```

---

## E. 架构

**文件结构**：

```
laser_cloud_normal/
├── laser_cloud_normal_cpu.h                 # 公开头文件
├── laser_cloud_normal_cpu.cpp               # 实现
└── tests/
    └── test_laser_cloud_normal_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserCloudNormalCPU`（pImpl，**无状态**） |
| 核心方法 | `Execute(fuse, beginIdx, endIdx)` / `Execute(fuse, fuseResult)`（便捷重载） |
| 参数更新 | `SetParams()` / `GetParams()` |
| 统计 | `GetStatistics()` / `ResetStatistics()` |
| 参数结构体 | `LaserCloudNormalCPUParams` |
| 统计结构体 | `LaserCloudNormalCPUStats` |
| 结果结构体 | `LaserCloudNormalCPUResult`（move-only） |
| 日志标签 | `"03-LaserCloudNormalCPU"` |

**数据结构**：

| 结构 | 说明 |
|------|------|
| `neighborBuf_`（Impl 成员） | `vector<const CloudPoint*>`，预分配 125（5×5×5 最大），跨 `Execute` 调用复用避免重复分配 |
| Jacobi 工作矩阵 | 栈上 3×3 `float` 数组，无堆分配 |
| 协方差上三角 | 栈上 `float[6]`，无堆分配 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | ≥ 4.x | `opencv_core`（仅 `cv::Point3f` 类型） |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

> **不依赖** Eigen、FLANN、PCL — 3×3 特征分解为全自定义 Jacobi 实现，与融合-02 理念一致。

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `kernelRadius` | int | 1 | ≥ 1 | 邻域半径（1=3×3×3=27 体素，2=5×5×5=125 体素） |
| `minNeighbors` | int | 3 | ≥ 3 | PCA 最小邻居数（含中心体素），不足时先扩大 kernelRadius 再退化 |
| `fallbackNx` | float | 0.0 | 单位向量 | 退化法线 X 分量 |
| `fallbackNy` | float | 0.0 | 单位向量 | 退化法线 Y 分量 |
| `fallbackNz` | float | 1.0 | 单位向量 | 退化法线 Z 分量 |

**预设配置**：`default`（3×3×3 邻域，最小 3 邻居） / `smooth`（5×5×5 邻域，最小 5 邻居）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 体素级（法线为近似值，仅供显示/渲染） |
| 吞吐目标 | 10K 新建体素 < 5ms（实测 3.66ms） |
| 单帧预算 | 典型帧（1K-5K 新建体素）< 2ms；首帧/全新区域（10K-30K）< 15ms |
| 邻域半径 | 默认 1 个体素（voxelSize=0.5mm 时 ~1mm 半径） |
| 线程安全 | **非线程安全**（单线程） |
| 状态管理 | **无状态**（范围从参数推导，无跨帧持久化） |
| 处理范围 | 仅本帧新建体素（`fusedPoints_` 末尾连续段） |

---

## K. 质量

**QualityFlag 语义**（基于退化率 = fallbackCount / totalAttempted）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 退化率 < 10% |
| `Degraded` | 精度降级但可用 | 退化率 ∈ [10%, 50%) |
| `Warning` | 需关注 | 退化率 ≥ 50% 或 beginIdx == endIdx（空范围） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败（`validate`） | 抛出 `std::invalid_argument` |
| 空范围（beginIdx >= endIdx） | `success=true`，`qualityFlag=Warning`，`message="No new voxels"` |
| 邻居不足（< minNeighbors） | 写入 `fallbackNormal`，计入 `fallbackCount` |
| 法线归一化失败（长度 ≈ 0） | 写入 `fallbackNormal`，计入 `fallbackCount` |
| 正常处理 | `success=true`，`message="Normal estimation completed"` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `FusedPointPtr()` 返回的指针在下次 `Execute()` 触发 `fusedPoints_` 重分配后失效 | 本算子在 `Execute()` 返回后立即调用，指针有效；但不应跨帧持有 |
| 🟢 低 | 法线在体素创建帧计算后不再更新，后续帧新增邻居不改善精度 | 对显示/渲染可接受；如需更高精度可定期全量重算 |
| 🟢 低 | PCA 法线有 ±符号歧义（不定向），单面光照渲染可能出现法线翻转 | 双面光照渲染不受影响；如需一致定向，下游可按视角翻转 |
| 🟢 低 | 共线点集（如单条激光线扫描）PCA 最小特征值 ≈ 0，法线不稳定 | 自适应扩大到 5×5×5 后多数情况可缓解；仍不足时退化为 fallbackNormal |
| 🟢 低 | `GatherVoxelNeighbors` 中 `packVoxelKey` 可能抛 `overflow_error`（坐标超 21-bit） | 内部 try-catch 捕获并跳过越界体素，不影响合法邻居 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `laser_cloud_normal`（`LaserCloudNormalCPU`） |
| **复用方式** | 完整实现：复用融合-02 哈希表邻域查找 + 手写 3×3 Jacobi 特征分解 + 自适应 kernel 扩大 + 退化兜底。15 个测试覆盖参数、平面/球面法线、退化、性能 |
| **依赖融合-02 改动** | 新增 `GatherVoxelNeighbors()` const 方法（~45 行），不修改既有 `Execute()` 逻辑 |

---

## 性能实测

| 场景 | 新建体素数 | 耗时 | fallback | expanded |
|------|-----------|------|----------|----------|
| 平面 100×100 网格（voxelSize=1.0） | 10,000 | **3.66ms** | 0 | 0 |
| 随机散点（voxelSize=0.5，稀疏） | 14,986 | 128ms | 14,631 | 14,970 |

> 平面场景为代表工况：每体素有 9 个邻居（3×3 平面），PCA 全部成功，无退化。随机散点因大部分体素孤立导致大量 fallback + 扩展，不代表正常扫描工况。

---

> **文档结束**
