# 激光点云体素哈希融合

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 融合-2 |
| 中文名称 | 激光点云体素哈希融合 |
| 英文目录名 | laser_cloud_fuse |
| 运行平台 | CPU（单线程） |
| 所属流程 | 融合流程（跨帧密集激光点云去重/累积） |
| 精度档次 | ③ 体素级（浮点/体素粒度） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 单帧激光 3D 点云（相机坐标系） | `vector<Point3f>` 或 `const Point3f*` + count |
| **输入②**（可选） | 刚体变换 R/T（相机系→全局系） | `cv::Matx33d R` + `cv::Vec3d T`；不传时为恒等变换（直接按原坐标融合） |
| **输出** | 融合结果 | `LaserCloudFuseCPUResult`（`survivingPoints`: 本帧存活点 **全局坐标系** `vector<Point3f>`、`statistics`） |

> **R/T 变换**：传入 R/T 时，每个输入点先经 `p_global = R * p_camera + T` 变换到全局系，再做体素哈希。`fusedPoints_` 与 `survivingPoints` 均存全局系坐标。来源为单帧配准（`optical_flow_fuse` / `frame_fuse`）输出的 R/T。
>
> **注**：哈希表和首点缓冲区**跨帧持久化**，属于有状态算子。新扫描开始时需调用 `Clear()`。

---

## C. 算法

**核心流程**（每点逐一处理）：

```
Step 1: 体素坐标量化 — ix/iy/iz = floor(p / voxelSize)（预计算倒数，乘法替代除法）
Step 2: 64-bit 体素键打包 — key = bx | (by<<21) | (bz<<42)，每轴 21 bit + 2^20 偏置
Step 3: 哈希 + 探测 — hash64(key)（splitmix64 finalizer）→ tag = top8(hash)（1-byte 预过滤）
         → 开放寻址线性探测（mask 取模环绕）
Step 4: 分支判定：
  HIT (tag+key 匹配):
    counts[idx] ≥ threshold → DELETE（丢弃点）
    counts[idx] <  threshold → ++counts, KEEP（保留点）
  MISS (空槽):
    新体素 → 写入 tag/key/fusedIdx/count=1, push 首点到 fusedPoints_, KEEP
```

**关键函数/技术**：

| 函数/技术 | 用途 |
|-----------|------|
| `hash64`（splitmix64 finalizer） | 64-bit 体素键哈希（x^=x>>33; x*=...; 三轮混合） |
| `tagFromHash` | 取哈希值高 8 位做标签预过滤（0 保留为空槽标记，1-255 有效） |
| `packVoxelKey` | 21 bit/轴偏置打包，偏置 BIAS=2^20，严格可逆零碰撞 |
| 开放寻址线性探测 | `idx = (idx+1) & mask_`，负载因子上限 0.7 |
| `ensureCapacityForOneMore` | 负载 > 0.7 时 2 倍扩容 rehash，返回 true 通知刷新缓存指针 |
| SoA 平坦哈希表 | 4 个并行数组（tags/keys/fusedIdx/counts），标签预过滤减少缓存未命中 |
| 索引链接 | 用 `uint32_t fusedIdx_`（4B）替代指针（8B），`vector` 重分配不影响索引有效性 |

> **饱和语义**：每个体素保留前 threshold 个点（count < T 则保留并 ++count；count ≥ T 则丢弃）。count 上限为 T，不溢出。`threshold=1` 为纯去重模式。

---

## D. 依赖

**上下游算子**：

```
laser_reconstruct → [本算子] → 下游（全局点云导出 / 法线计算 / Mesh 重建）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 哈希表（`tags_/keys_/fusedIdx_/counts_`） | **跨帧持久化**，仅 `Clear()` 重置 |
| 首点缓冲（`fusedPoints_`） | **跨帧持久化**，每个体素记录第一个落入的点 |
| 无外部共享 | 独立实例 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | `QualityFlag` |
| `common/calib_logging.h` | 日志宏 |
| `common/calib_warmup_config.h` | `WarmupConfig` |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `laser_reconstruct` | `vector<Point3f>` 或 `const Point3f*` | 每帧三维激光点（mm），法线后续通过 `FusedPointPtr()` 写入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `result.survivingPoints` | 本帧存活点导出 | — | `vector<Point3f>` | 必用 |
| `GetFusedPoints()` | 全局累积点云 | `const vector<CloudPoint>&` | const 引用 | 必用 |
| `FusedPointPtr(i)` | 法线回写 | `CloudPoint*` | 可写指针 | 可选 |
| `SnapshotVoxels()` | 体素统计导出 | `void SnapshotVoxels(vector<VoxelInfo>& out)` | 输出引用填充 | 可选 |
| `GetVoxelCount()` | 体素计数 | — | `size_t` 值 | 可选 |

---

## E. 架构

**文件结构**：

```
laser_cloud_fuse/
├── laser_cloud_fuse_cpu.h                 # 公开头文件
├── laser_cloud_fuse_cpu.cpp               # 实现
└── tests/
    └── test_laser_cloud_fuse_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserCloudFuseCPU`（pImpl，**有状态**） |
| 核心方法 | `Execute(const vector<Point3f>&)` / `Execute(const Point3f*, size_t)` |
| 预分配 | `Reserve(voxelCount)` — 哈希表扩容（向上取 2 的幂） |
| 预热 | `Warmup(maxPointCount)` — 仅置标志（实际预分配在构造函数 `reserveVoxelCount`） |
| 重置 | `Clear()` — 清空哈希表 + 点缓冲 + 重置统计（保留容量） |
| 状态查询 | `GetVoxelCount()` / `GetFusedPointCount()` / `GetFusedPoints()` / `FusedPointPtr(i)` / `SnapshotVoxels()` |
| 参数更新 | `SetParams()` / `GetParams()` |
| 统计 | `GetStatistics()` / `ResetStatistics()` |
| 参数结构体 | `LaserCloudFuseCPUParams` |
| 点结构体 | `CloudPoint`（xyz + nxnynz，24 字节） |
| 快照结构体 | `VoxelInfo`（key + firstPoint* + count） |
| 统计结构体 | `LaserCloudFuseCPUStats` |
| 结果结构体 | `LaserCloudFuseCPUResult`（move-only） |
| 日志标签 | `"02-LaserCloudFuseCPU"` |

**数据结构（SoA 平坦哈希表）**：

| 数组 | 类型 | 说明 |
|------|------|------|
| `tags_` | `vector<uint8_t>` | 0=空槽，1-255=哈希高 8 位（标签预过滤） |
| `keys_` | `vector<uint64_t>` | 体素键（仅 tag 匹配时比较） |
| `fusedIdx_` | `vector<uint32_t>` | 首点索引 → `fusedPoints_` |
| `counts_` | `vector<uint32_t>` | 命中计数（上限 threshold） |
| `fusedPoints_` | `vector<CloudPoint>` | 连续首点缓冲（`Reserve` 预分配） |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | ≥ 4.x | `opencv_core`（仅 `cv::Point3f` 类型） |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

> **不依赖** Eigen、FLANN、PCL — 纯标准库 + OpenCV 点类型。哈希表为全自定义实现。

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `voxelSize` | float | 0.5 | > 0 | 体素边长（mm） |
| `saturationThreshold` | int | 5 | ≥ 1 | 每体素最大保留点数（超出丢弃） |
| `reserveVoxelCount` | size_t | 65536 | ≥ 64 | 哈希表预分配槽数（向上取 2 的幂） |
| `collectStatistics` | bool | true | — | 是否填充 result.statistics（正常帧内部 `stats_` 始终更新；**空帧例外**：`stats_` 不更新，`GetStatistics()` 返回上一帧的陈旧值） |

**预设配置**：`default`（0.5mm, 保留 5 点/体素, 64K 槽） / `fine`（0.2mm, 8 点, 256K 槽） / `coarse`（1.0mm, 3 点, 16K 槽）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 体素级（`float` 精度，`voxelSize` 决定分辨率） |
| 吞吐目标 | ~8M 点/秒（200fps × 40K 点/帧） |
| 单帧预算 | 40K 点: avg < 5ms, p99 < 10ms |
| 体素坐标范围 | 每轴 [-1,048,576, +1,048,575] 体素（21 bit），voxelSize=0.5mm 时 ±524m |
| 线程安全 | **非线程安全**（单线程）；Debug 模式声明 `atomic<bool> inProcess_` 但实际未在 `Execute()` 中设置，并发检测为死代码 |
| 状态管理 | **有状态**（哈希表 + 点缓冲跨帧持久化，`Clear()` 重置新扫描） |
| 负载因子 | 上限 0.7，超限自动 2 倍扩容 rehash |

---

## K. 质量

**QualityFlag 语义**（基于保留率 = survivingCount / inputCount）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 保留率 ≥ 0.5 |
| `Degraded` | 精度降级但可用 | 保留率 ∈ [0.1, 0.5) |
| `Warning` | 需关注 | 保留率 < 0.1 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败（`validate`） | 抛出 `std::invalid_argument` |
| 体素坐标超 21-bit 范围 | 抛出 `std::overflow_error`（`Execute()` 内未捕获） |
| 空帧（count == 0） | `success=false`，`message="Empty frame"` |
| 正常处理 | `success=true`，`message="Frame fusion completed"` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `Execute()` 内未捕获 `std::overflow_error`，体素坐标超范围时异常直接传播 | 调用方需 try-catch 或保证坐标范围 |
| 🟡 中 | `FusedPointPtr()` 返回的指针在下次 `Execute()` 触发 `fusedPoints_` 重分配后失效 | 下游不应跨 `Execute()` 调用持有指针 |
| 🟡 中 | 实现与 DESIGN.md 有分歧（`vector` 非 `deque`、索引非指针、SoA+tag 预过滤） | DESIGN.md 过时，以代码为准 |
| 🟡 中 | `SetParams()` 修改 `reserveVoxelCount` 后**不自动生效**：`Clear()` 不重 hash，`Reserve()` 需显式传参；`reserveVoxelCount` 仅在构造函数中消费一次 | 改参后须手动调 `Reserve(newCount)` |
| 🟢 低 | `Warmup()` 为空操作（仅置标志），预分配实际由构造函数 `reserveVoxelCount` 完成 | 文档误导 |
| 🟢 低 | `SnapshotVoxels()` 为 O(capacity) 而非 O(size)，大表时较慢 | 实际 size/size 比高时可接受 |
| 🟢 低 | Debug 模式 `inProcess_` 原子标志声明但 `Execute()` 中从未设置 true，并发检测为死代码 | 无实际并发保护 |
| 🟢 低 | 空帧（n==0）路径提前返回，不更新成员 `stats_`，`GetStatistics()` 返回上一帧陈旧值 | 空帧后读统计需注意 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `laser_cloud_fuse`（`LaserCloudFuseCPU`） |
| **复用方式** | 完整实现：自定义 SoA 平坦哈希表 + splitmix64 哈希 + tag 预过滤 + 索引链接 + 饱和去重。经过 1000/10000 帧 + 100K 帧压力测试（40 亿点）验证 |

---

> **文档结束**
