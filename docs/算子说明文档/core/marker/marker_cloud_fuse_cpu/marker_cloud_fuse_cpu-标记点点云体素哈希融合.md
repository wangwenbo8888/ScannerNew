# 标记点点云体素哈希融合（CPU）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 融合-02M |
| 中文名称 | 标记点点云体素哈希融合（CPU） |
| 英文目录名 | marker_cloud_fuse_cpu |
| 运行平台 | CPU（单线程） |
| 所属流程 | 融合流程（跨帧标记点去重/累积，供显示渲染） |
| 精度档次 | ③ 体素级（浮点/体素粒度） |
| 对标激光版 | `laser_cloud_fuse_cpu`（融合-02），哈希表算法相同，点结构+输入接口不同 |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 单帧标记点（相机坐标系，含法线+半径） | `const MarkerFuseInput*` + count，或 `vector<MarkerFuseInput>` |
| **输入②**（可选） | 刚体变换 R/T（相机系→全局系） | `cv::Matx33d R` + `cv::Vec3d T`；不传时为恒等变换 |
| **输出** | 融合结果 | `MarkerCloudFuseCPUResult`（`survivingPoints`: 本帧存活点 全局坐标系 `vector<MarkerCloudPoint>`、`statistics` + `qualityFlag`） |
| **累积输出** | 全局标记点集（跨帧持久） | `const vector<MarkerCloudPoint>& GetFusedPoints()` |

> **R/T 双变换**：传入 R/T 时，位置经 `gp = R*p + T` 变换，法线经 `gn = R*n` 变换（无平移），再做体素哈希。`fusedPoints` 存全局坐标系数据。R/T 来源为单帧配准（`optical_flow_fuse` / `frame_fuse`）输出。
>
> **与激光版区别**：激光版输入仅 `cv::Point3f`（位置），法线由后续 `laser_cloud_normal` 算子估计；本算子输入直接携带法线+半径，**无需后续法线估计**。
>
> **跨帧持久化**：哈希表和点缓冲区跨帧持久化，属于有状态算子。新扫描开始时需调用 `Clear()`。

### 输入数据来源

```
point_reconstruct (标记点 11, CPU)
  │  MarkerReconstructResult → 转换为 MarkerFuseInput
  │    centerX/Y/Z     → x/y/z
  │    normalX/Y/Z     → nx/ny/nz
  │    circleFit.radius → whiteRadius
  ▼
单帧配准 → R/T
  ▼
[本算子] Execute(frame, R, T)
```

---

## C. 算法

**核心流程**（与 `laser_cloud_fuse_cpu` 相同的 SoA 哈希表 + tag 预过滤）：

```
Step 1: R/T 双变换
        位置: gx = R(0,0)*x + R(0,1)*y + R(0,2)*z + T(0)  (gy/gz 同理)
        法线: gnx = R(0,0)*nx + R(0,1)*ny + R(0,2)*nz     (gny/gnz 同理, 无平移)

Step 2: 体素量化 — ix/iy/iz = floor(g / voxelSize)

Step 3: 体素键打包 — key = packVoxelKey(ix, iy, iz)
        21 bit/轴 + 2^20 偏置, 打包为 uint64

Step 4: 哈希 + tag 预过滤 + 开放寻址探测
        h = hash64(key)           // splitmix64 finalizer
        tag = tagFromHash(h)      // 高 8 位, 0 保留给空槽
        idx = h & mask
        loop:
          tags[idx]==0  → 空槽, MISS
          tags[idx]==tag && keys[idx]==key → HIT
          否则 → 碰撞, idx = (idx+1) & mask

Step 5: 分支判定
        HIT (已有体素):
          counts[idx] ≥ threshold → DELETE
          counts[idx] <  threshold → ++counts, KEEP
        MISS (新体素):
          若负载 > 0.7 → 2 倍 rehash + 刷新缓存指针
          写入 MarkerCloudPoint{gx,gy,gz, gnx,gny,gnz, whiteRadius}
          插入哈希条目 (tag/key/fusedIdx/count=1)
```

**关键函数/技术**（与激光版完全相同）：

| 函数/技术 | 用途 |
|-----------|------|
| `hash64`（splitmix64 finalizer） | 64-bit 体素键哈希（三轮混合） |
| `tagFromHash` | 取哈希值高 8 位做标签预过滤（1-byte 比较，减少缓存未命中） |
| `packVoxelKey` | 21 bit/轴偏置打包，BIAS=2^20 |
| 开放寻址线性探测 | `idx = (idx+1) & mask_`，负载因子上限 0.7 |
| `ensureCapacityForOneMore` | 负载 > 0.7 时 2 倍扩容 rehash |
| SoA 平坦哈希表 | 4 个并行数组（tags/keys/fusedIdx/counts） |
| `uint32_t fusedIdx_` | 4B 索引替代 8B 指针 |

> **饱和语义**：`saturationThreshold=99` 时，同一体素被命中 99 次后，后续落入该体素的当前帧点被丢弃（`++deletedCount`）。代表点始终是首个落入该体素的点，后续命中仅 `++counts`。未超限的当前帧存活点收集到 `result.survivingPoints`（全局坐标系）。

---

## D. 依赖

**上下游算子**：

```
point_reconstruct (标记点 11) → MarkerFuseInput
单帧配准 (配准-01 / 标记点 12) → R/T
    ↓
[本算子] MarkerCloudFuseCPU
    ↓ GetFusedPoints()
显示: MarkerCloudRenderer (update + flush + 渲染)
```

**头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | `QualityFlag` |
| `common/calib_logging.h` | 日志宏（`CALIB_LOG_*`） |
| `<opencv2/core.hpp>` | `cv::Matx33d` / `cv::Vec3d` |
| `<nlohmann/json.hpp>` | 参数序列化 |

> **模块独立**：不依赖 `laser_cloud_fuse_cpu` 或 `laser_cloud_normal`，仅复用算法逻辑（代码内联在 .cpp 中）。

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `point_reconstruct` | `MarkerFuseInput`（转换自 `MarkerReconstructResult`） | 位置+法线+半径，相机坐标系 |
| 单帧配准 | `cv::Matx33d R` + `cv::Vec3d T` | 12 个标量 |

**本算子→下游**：

| 输出 | 传递给 | 方式 |
|------|--------|------|
| `GetFusedPoints()` | `MarkerCloudRenderer` | `const vector<MarkerCloudPoint>&` 引用 |

**调用示例**：

```cpp
MarkerCloudFuseCPU fuse;  // 默认参数 (0.5mm, threshold=99, 1024 slots)

// 每帧：
std::vector<MarkerFuseInput> frame = convertFromReconstruct(result);
MarkerCloudFuseCPUResult fuseResult = fuse.Execute(frame, R, T);

// 显示侧读取累积结果：
const auto& markers = fuse.GetFusedPoints();
renderer.update(markers);
renderer.flush();

// 新扫描开始：
fuse.Clear();
```

---

## E. 架构

**文件结构**：

```
core/marker/marker_cloud_fuse_cpu/
├── marker_cloud_fuse_cpu.h
└── marker_cloud_fuse_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `MarkerCloudFuseCPU`（pImpl，**有状态**） |
| 融合方法 | `Execute(MarkerFuseInput*, count, R, T)` |
| 融合方法（vector） | `Execute(vector<MarkerFuseInput>&, R, T)` |
| 融合方法（无变换） | `Execute(vector<MarkerFuseInput>&)` — 已在全球坐标系 |
| 累积查询 | `GetFusedPoints()` → `const vector<MarkerCloudPoint>&` |
| 重置 | `Clear()` — 清空哈希表 + 点缓冲 |
| 预分配 | `Reserve(voxelCount)` |
| 状态查询 | `GetVoxelCount()` / `GetFusedPointCount()` |
| 日志标签 | `"02M-MarkerCloudFuseCPU"` |

**数据结构**：

| 结构 | 字段 | 说明 |
|------|------|------|
| `MarkerFuseInput` | x/y/z, nx/ny/nz, whiteRadius (7 floats) | 输入（相机坐标系） |
| `MarkerCloudPoint` | x/y/z, nx/ny/nz, whiteRadius (7 floats) | 累积输出（全局坐标系） |
| `MarkerCloudFuseCPUParams` | voxelSize, saturationThreshold, reserveVoxelCount, collectStatistics | 参数 |
| `MarkerCloudFuseCPUStats` | totalTimeMs, hashTimeMs, inputCount, survivingCount, deletedCount, newVoxelCount, totalVoxelCount | 统计 |

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `voxelSize` | float | 0.5 | > 0 | 体素边长（mm） |
| `saturationThreshold` | int | 99 | ≥ 1 | 每体素最大保留点数（超过则丢弃当前帧点） |
| `reserveVoxelCount` | size_t | 1024 | ≥ 64 | 哈希表初始槽数（向上取 2 的幂） |
| `collectStatistics` | bool | true | — | 是否填充 result 统计 |

> **与激光版默认值对比**：激光版 `saturationThreshold=5`、`reserveVoxelCount=65536`；本算子因标记点数量少（≤1024），`threshold=99`、`reserve=1024`。

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 体素级（0.5mm 格去重） |
| 标记点数量 | 典型 <100/帧，累积 ≤1024 |
| 线程安全 | **非线程安全**（单实例不可并发调用） |
| 状态管理 | **有状态**（哈希表跨帧持久，`Clear()` 重置） |
| 负载因子 | 上限 0.7，超限自动 2 倍 rehash |
| 体素坐标范围 | 每轴 [-1,048,576, +1,048,575]（21 bit） |
| 法线来源 | 输入直接携带（point_reconstruct 输出），无需后续估计 |

---

## K. 质量

**QualityFlag 语义**（基于保留率 = survivingCount / inputCount）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 保留率 ≥ 0.5 |
| `Degraded` | 降级但可用 | 保留率 ∈ [0.1, 0.5) |
| `Warning` | 需关注 | 保留率 < 0.1 |

**错误处理**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 | 抛出 `std::invalid_argument` |
| 空帧（count=0） | `success=false`，`message="Empty frame"` |
| 体素坐标越界 | 抛出 `std::overflow_error` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 哈希表自动扩容（rehash）时偶发帧延迟 | 标记点量小（≤1024），rehash 一次 <0.01ms |
| 🟢 低 | 先到先得语义：首个观测点永久占据体素 | R/T 配准误差导致位置偏差时无法用后续帧修正 |
| 🟢 低 | 仅位置去重，不做法线去重 | 同位置正反两面标记点（薄板）可能被误判为重复 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `core/marker/marker_cloud_fuse_cpu`（`MarkerCloudFuseCPU`） |
| **复用方式** | 完整实现：SoA 哈希表 + tag 预过滤 + R/T 双变换 + 距离去重。算法逻辑从 `laser_cloud_fuse_cpu` 复制，点结构/输入接口适配标记点。 |
| **与激光版关系** | 算法相同（hash64/tag/open-addressing/saturation），差异仅在点结构（+whiteRadius）、输入（+normal+radius）、法线处理（输入携带 vs 后续估计）、默认参数（threshold=99, reserve=1024） |

---

> **文档结束**
