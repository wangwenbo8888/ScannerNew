# 激光线匹配扫描（温度补偿查表匹配）

> **重要**：本文档替代过时的 `激光标定-07_激光线匹配_cuda.md`。旧文档描述的哈希表 + 视差范围过滤算法（`kernelBuildHash`、`kernelProbeMatch`、`atomicCAS`）在本代码中**完全不存在**。实际代码使用**温度补偿查表**匹配，依赖第 13 步输出的映射表。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-7 |
| 中文名称 | 激光线匹配扫描（温度补偿查表匹配） |
| 英文目录名 | laser_match_scan |
| 运行平台 | CUDA（全 GPU 管线，CUB + Thrust） |
| 所属流程 | 激光标定流程第 7 步（`epipolar_interp` 之后） |
| 精度档次 | ③ 亚像素/浮点类 |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机激光中心点集 + 线 ID | `GpuMat`（`CV_32FC2` 点坐标）+ `GpuMat`（`CV_32SC1` 线 ID） |
| **输入②** | 右相机激光中心点集 + 线 ID | `GpuMat`（`CV_32FC2` 点坐标）+ `GpuMat`（`CV_32SC1` 线 ID） |
| **输入③** | 温度补偿映射表（第 13 步输出） | `SetTempTable(shared_ptr<const LaserPlaneMapTempTable>)` 注入（§3.6 首选）；或 `LoadTempTable(jsonPath)` 从 JSON 加载 |
| **输入④** | 当前温度值 | `SetCurrentTemperature(double)` 设置 |
| **输出** | 匹配结果 | `LaserMatchScanResult`：`d_matched_left`、`d_matched_right`、`d_matched_line_ids`（紧凑匹配对）、`d_left_status`/`d_right_status`（各点状态）、`matchedCount`、`excludedLeftCount`/`excludedRightCount` |

---

## C. 算法

**核心思想**：从温度补偿映射表（第 13 步 `plane_map_temp_table` 输出）中，按当前温度选取对应条目，查表预测每个左相机点在右相机的期望 u 坐标（`uR_expected`），然后在右相机同极线行、同线 ID 的点中，在 `match_threshold` 阈值范围内寻找唯一匹配。

**核心流程**（8 步 GPU 管线）：

```
Step 1a: 生成排序键 — kernelGenerateSortKeys
         sortKey = (rowKey << 32) | floatBits(x)，确保按行再按 u 排序
         注意：负 y 行键 clamp 到 0，负 x 位模式 clamp 到 0（负坐标点会碰撞）
         同时 atomicMax 找出左右最大行键（确定极线行数）

Step 1b: CUB RadixSort — 按排序键对左右点集排序（thrust::sequence 生成原始索引）

Step 1c: 重排数据 — kernelScatterSorted
         按排序索引 gather 点坐标和线 ID 到连续缓冲

Step 2:  构建 CSR 行索引 — kernelBuildCSRGPU
         对左右排序后数据分别构建 row_start / row_count（atomicMin + atomicAdd）
         CSR 数组预分配 MAX_EPIPOLAR_ROWS=8192

Step 3:  初始化状态 — 左右 status 清零，match_count 清零

Step 4:  查表预测 — kernelLookupTableScan（全局按行搜索）
         温度表在 uploadMapTable() 时重组为按极线行(row)的 CSR 视图（d_map_byrow_ + d_map_row_start/count）
         对每个排序后的左点 (xL,yL)：算 row=round(yL/step) → 查该行桶 → 桶内按 xL 二分搜索最近条目
         输出 uR_expected **和标定 line_id**（line_id 为输出，由表条目的 .z 分量确定；超出容差输出 NAN + line_id=-1）
         随后 kernelScatterLineIds 将 line_id 从排序索引散列回原始索引（供 Step7 提取）

> **注**：原 kernelLookupTable（按输入 line_id 选段）已被全局按行搜索取代。扫描输入点无标定 line_id（统一占位），必须全局搜索才能确定每点属哪条标定线。当输入 line_id 正确时（标定语义），全局搜索结果与按线选段等价（每 (row,x) 唯一映射一条线）。

Step 4b: 同步点 — 下载 max_row_key 获取实际极线行数 + cudaGetLastError 检查

Step 5:  逐行匹配 — kernelMatch（每极线行一个 block，threadIdx.x==0 串行处理）
         1. 将右点 u/线ID/占用标志/原始索引加载到共享内存
         2. 遍历左点：查 uR_expected，二分搜索右点 u 在 [uR-thresh, uR+thresh] 范围
         3. 过滤同线ID + 未占用候选，统计候选数
            - 恰好 1 候选 → 匹配成功（atomicAdd 取 pos，`pos < maxMatchPairs` 守卫后写入匹配对）
            - 0/≥2 候选 → 全部排除（标记 status=-1，占用右点防止后续误匹配）

Step 6:  下载匹配计数

Step 7:  提取匹配坐标 — kernelExtractMatchedCoords
         从紧凑匹配对索引 gather 左右点坐标和线 ID

Step 8:  统计排除数 — kernelCountStatus（统计 status==-1 的点数）
```

**关键 CUDA 内核 / 第三方函数**：

| 内核/函数 | 用途 |
|-----------|------|
| `kernelGenerateSortKeys` | 生成 64-bit 复合排序键（行键<<32 \| x 浮点位），`atomicMax` 找最大行 |
| `cub::DeviceRadixSort::SortPairs` | GPU 基数排序（64-bit 键 + 32-bit 值索引） |
| `thrust::sequence` | 生成 0,1,2,...N-1 原始索引序列 |
| `kernelScatterSorted` | 按排序索引 gather 点坐标 + 线 ID |
| `kernelBuildCSRGPU` | 构建稀疏行索引（CSR 格式：row_start + row_count） |
| `kernelLookupTableScan` | 全局按行搜索查温度映射表，输出 uR_expected + line_id（原 `kernelLookupTable` 已废弃未调用） |
| `kernelMatch` | 逐极线行串行匹配（共享内存 + 二分搜索 + 占用标记 + 唯一性判定 + `maxMatchPairs` 边界守卫） |
| `kernelExtractMatchedCoords` | 提取紧凑匹配对坐标 |
| `kernelCountStatus` | 统计排除点数 |

---

## D. 依赖

**上下游算子**：

```
epipolar_interp（左相机） ─┐
                                  ├→ 本算子 → laser_reconstruct
epipolar_interp（右相机） ─┘

plane_map_temp_table → 温度补偿映射表（JSON） → 本算子 LoadTempTable()
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 温度映射表 | 第 13 步输出的 JSON 文件，通过 `LoadTempTable()` 加载 |
| 左右相机实例 | 左右相机点集分别输入同一实例 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型（`QualityFlag` 等） |
| `common/calib_warmup_config.h` | `WarmupConfig` 结构体 |
| `common/json_utils.h` | JSON 辅助函数 |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `epipolar_interp`（左） | `GpuMat` 引用（`CV_32FC2` + `CV_32SC1`） | 极线插值后的左相机点集 + 线 ID |
| `epipolar_interp`（右） | `GpuMat` 引用（`CV_32FC2` + `CV_32SC1`） | 极线插值后的右相机点集 + 线 ID |
| `plane_map_temp_table` | JSON 文件路径 | `LoadTempTable(jsonPath)` 加载，`SetCurrentTemperature(temp)` 选条目 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `d_matched_left` | laser_reconstruct | `LaserReconstructCuda::Execute()` | `shared_ptr<GpuMat>`（`CV_32FC2`） | 必用 |
| `d_matched_right` | laser_reconstruct | `LaserReconstructCuda::Execute()` | `shared_ptr<GpuMat>`（`CV_32FC2`） | 必用 |
| `d_matched_line_ids` | laser_reconstruct | `LaserReconstructCuda::Execute()` | `shared_ptr<GpuMat>`（`CV_32SC1`） | 必用 |
| `matchedCount` | 下游点数校验 | — | 值 | 必用 |
| `d_left_status`/`d_right_status` | 调试/质量分析 | — | `shared_ptr<GpuMat>`（`CV_32SC1`，0=未处理/1=匹配/-1=排除） | 可选 |
| `result.success` | Pipeline 调度 | — | 值 | 必用 |

---

## E. 架构

**文件结构**：

```
laser_match_scan/
├── laser_match_scan_cuda.h               # 公开头文件（纯 C++，不含 CUDA 类型）+ Params + Result
├── laser_match_scan_cuda_pimpl.h         # CUDA 桥接头文件（Impl 完整定义 + GpuMat 缓冲区 + TempTableEntry）
├── laser_match_scan_cuda.cpp             # 桥接实现（输入校验 + BUILD_CUDA 开关）
├── laser_match_scan_cuda_impl.cu         # CUDA 内核 + 8 步流水线
└── tests/
    └── test_laser_match_scan_cuda.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `LaserMatchScanCuda`（pImpl） |
| 表加载 | `SetTempTable(shared_ptr<const LaserPlaneMapTempTable>)` → `bool`（§3.6 只读注入，首选）；`LoadTempTable(jsonPath)` → `bool`（JSON 便利重载） |
| 温度设置 | `SetCurrentTemperature(temperature)` |
| 核心方法 | `Execute(left_pts, left_ids, right_pts, right_ids, stream)` / `Execute(...)`（2 重载，带/不带 Stream） |
| 预热方法 | `Warmup(maxLeftPoints, maxRightPoints)` / `Warmup(WarmupConfig)` |
| 参数更新 | `SetParams()` / `GetParams()` |
| 参数结构体 | `LaserMatchScanParams` |
| 结果结构体 | `LaserMatchScanResult`（move-only） |
| 日志标签 | `"07-LaserMatchScanCuda"` |

**GPU 缓冲区**（`MAX_EPIPOLAR_ROWS = 8192`）：

| 缓冲区 | 用途 |
|--------|------|
| `d_left/right_sorted_pts_` | 排序后点坐标 |
| `d_left/right_sorted_lids_` | 排序后线 ID |
| `d_left/right_sorted_idx_` | 排序索引（→原始索引） |
| `d_left_uR_expected_` | 左点期望右 u 坐标（查表结果） |
| `d_left/right_keys_in/out_` | CUB 排序键（`CV_64FC1`，64-bit 复合键） |
| `d_left/right_idx_in_` | CUB 排序值（0..N-1 序列） |
| `d_left/right_row_start/count_` | CSR 行索引（8192 槽） |
| `d_left/right_status_` | 各点匹配状态（0/1/-1） |
| `d_match_left_idx_` | 匹配对索引数组（作为 `MatchPair*` 使用，每对 8 字节 = 2 int；分配 `leftCount * 2` 个 int = leftCount 对，100% 匹配率亦不溢出；kernelMatch 内 `pos < maxMatchPairs` 边界守卫） |
| `d_match_count_` | 匹配计数 |
| `d_map_table_` | 温度映射表（`CV_32FC4`：xL, yL, uR, lineId） |
| `d_map_line_start/count_` | 映射表按线 ID 的 CSR 索引 |
| `d_max_row_key_` | 最大行键（2 元素：左/右） |
| `d_cub_temp_` | CUB 排序临时存储 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | `opencv_core`、`opencv_cudaarithm`（仅 GpuMat + StreamAccessor） |
| CUDA Toolkit | 12.6 | sm_75（RTX 5000），sm_87（Orin） |
| CUB | 随 CUDA Toolkit | `cub::DeviceRadixSort::SortPairs` |
| Thrust | 随 CUDA Toolkit | `thrust::sequence`、`thrust::device_ptr` |
| nlohmann_json | ≥ 3.11 | 温度表 JSON 解析 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

**编译特性**：

- `BUILD_CUDA` 条件编译：开启时编译 `.cu`（正常工作），关闭时所有方法抛 `std::runtime_error`
- `LM_ENABLE_TIMING=1`（PRIVATE）：启用每步计时日志（`LM_TIMER_MARK`）
- CUDA 源文件 `FMT_UNICODE=0`（Windows fmtlib 兼容）
- MSVC 下 `/bigobj`（大量 GpuMat 成员导致目标文件膨胀）
- 平台兼容：Windows 10 x64 + Jetson Orin NX/AGX Orin

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `match_threshold` | float | 1.0 | > 0 | 匹配阈值（像素），右点 u 与 uR_expected 差值在此范围内才为候选 |
| `epipolar_row_step` | float | 0.5 | > 0 | 极线行间距，用于量化 y 坐标到行索引（`round(y/step)`） |
| `max_right_per_row` | int | 1024 | > 0 | 每极线行最大右点数（共享内存分配上限，实际取 min(右点数, 此值)） |
| `vL_tolerance` | float | 0.01 | ≥ 0 | 查表 y 容差（左点 yL 与映射表 y 的最大允许偏差，超出则 NAN） |
| `deviceId` | int | 0 | ≥ 0 | GPU 设备 ID |

**预设配置**：`default`（1.0px 匹配阈值） / `strict`（0.5px 严格） / `loose`（2.0px 宽松）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ③ 亚像素级（依赖映射表精度 + 匹配阈值） |
| 极线行上限 | `MAX_EPIPOLAR_ROWS = 8192`（CSR 数组大小）；但 `kernelMatch` 的 grid 维度 = 实际最大行键 + 1，**未 clamp 到 8192**，图像高度 ≥ 4096px（step=0.5）时 grid 超出 CSR 数组，导致越界读 |
| 共享内存 | `max_right_per_row × (sizeof(float) + 3×sizeof(int))` 字节/行 |
| 线性化 | `kernelMatch` 中每行 threadIdx.x==0 串行处理左点（block 内串行匹配） |
| 线程安全 | 非线程安全，Debug 模式 `atomic<bool> inProcess_` 检测并发调用 |
| 实例隔离 | 左右相机点集输入同一实例 |
| 前置条件 | `LoadTempTable()` + `SetCurrentTemperature()` 必须在 `Execute()` 前调用 |

---

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 正常 | 匹配流程成功完成（无论匹配数为 0 或更多） |

> **注**：代码中 `qualityFlag` 默认为 `Normal`，`Execute()` 未设置其他值。质量评估通过 `matchedCount`、`excludedLeftCount`、`excludedRightCount` 数值体现（返回在 `message` 中）。

**错误处理模式**（`Execute()` 通过 `success/message/qualityFlag` 返回错误）：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败（`validate`） | 抛出 `std::invalid_argument` |
| 无 CUDA 设备 | 构造时抛出 `std::runtime_error` |
| deviceId 越界 | 构造时抛出 `std::invalid_argument` |
| 温度表未加载 → `SetCurrentTemperature()` | 抛出 `std::runtime_error` |
| 温度超范围 → `SetCurrentTemperature()` | 警告日志 + clamp 到最近条目（不失败） |
| `Execute()` 前未设温度 | `success=false`，`message="Temperature not set: call SetCurrentTemperature() first"` |
| 输入 GpuMat 为空（`.empty()`） | `success=true`，`message="Empty input, no points to match"` |
| 输入非空但点数为 0 | `success=true`，`message="No points to match"` |
| 输入类型非 CV_32FC2/CV_32SC1 | `success=false` |
| 左右点数与线 ID 数不匹配 | `success=false` |
| GPU 管线错误（cudaGetLastError） | `success=false`，`message="GPU pipeline failed: ..."` |
| `kernelMatch` 启动失败 | `success=false`，`message="kernelMatch failed: ..."` |
| `BUILD_CUDA=OFF` | 所有操作抛出 `std::runtime_error` |
| 异常捕获 | `success=false`，`message="Exception: ..."` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| ✅ 已修复 | ~~`d_match_left_idx_` 容量仅 `leftCount/2` 对，匹配率 > 50% 越界~~ → 3 处修复：①分配改 `leftCount * 2` 个 int（= leftCount 对 MatchPair）；②删除死缓冲 `d_match_right_idx_`（从未使用）；③kernelMatch 新增 `maxMatchPairs` 参数 + `pos < maxMatchPairs` 边界守卫（2026-06-23，12 测试通过） | — |
| 🟡 中 | `kernelMatch` grid 维度 = 实际最大行键 + 1，**未 clamp 到 MAX_EPIPOLAR_ROWS(8192)**；CSR 数组仅 8192 槽，图像 ≥ 4096px 时 grid 超出 → 越界读 `d_row_start/count` | 大图传感器需扩大 CSR 或 clamp grid |
| 🟡 中 | 负坐标处理：`kernelGenerateSortKeys` 将负 y clamp 到行 0、负 x clamp 到位模式 0，导致负坐标点互相碰撞排序 | 若上游存在坐标系偏移产生负坐标，匹配质量受损 |
| 🟡 中 | 旧文档（`激光标定-07_激光线匹配_cuda.md`）描述哈希表算法，与代码完全不符 | 文档误导，需以本文档为准 |
| 🟡 中 | `kernelMatch` 中每极线行由 threadIdx.x==0 串行处理全部左点，行内点多时成为瓶颈 | 密集极线行匹配耗时增加 |
| 🟡 中 | `Execute()` 内有 **5 次** `cudaStreamSynchronize`（Step 4b/5/6/7/8），打断异步流水线 | 流水线并行度受限 |
| 🟡 中 | `allocateBuffers` 每次 `Execute()` 调用都重新创建所有 GpuMat，即使 warmup 过 | 未检查 warmed_up_ 标志，存在重复分配开销 |
| 🟢 低 | `SetParams()` 会检测 deviceId 变化并 `cudaSetDevice`，且**重置 warmup 状态**（`warmed_up_=false`） | 改参后需重新 warmup |
| 🟢 低 | 温度表按行 ID CSR 构建在 CPU（`uploadMapTable`），行数多时上传耗时 | 实际线数 < 256，可接受 |
| 🟢 低 | `max_right_per_row` 通过共享内存限制，超出的右点静默丢弃 | 极端密集场景丢点 |
| 🟢 低 | CUB 排序临时存储按 max(leftCount, rightCount) 分配，未检查 warmup 尺寸 | 首次大帧可能 cudaMalloc |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `laser_match_scan`（`LaserMatchScanCuda`） |
| **复用方式** | 完整实现：温度补偿查表 + CSR 行索引 + 二分搜索匹配 + 唯一性占用规则。替代旧文档描述的哈希表方案 |
| **与旧文档差异** | 旧文档描述的 `kernelBuildHash`/`kernelProbeMatch`/`atomicCAS`/`min_disp`/`max_disp` 在代码中**不存在**；实际使用 `LoadTempTable`/`SetCurrentTemperature`/`kernelLookupTable`/`kernelMatch` 温度查表方案 |

---

> **文档结束**
