# 标记点光流配准（刚体变换估计）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 配准-1 |
| 中文名称 | 标记点光流配准（刚体变换估计） |
| 英文目录名 | optical_flow_fuse |
| 运行平台 | CPU |
| 所属流程 | 配准流程（跨帧标记点轨迹追踪 + 全局坐标对齐） |
| 精度档次 | ②（~0.05mm 位置） |

---

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 当前帧标记点 3D 坐标 + 法线 | `vector<Point3d>` + `vector<Vec3d>`（或 `PointReconstructCPUResult`） |
| **输入②** | 上一帧状态（原始坐标、法线、全局 ID、R、T） | `PrevFrameState`（调用方维护） |
| **输入③**（可选） | 全局标记点集（外部全局配准） | `GlobalMarkerSet`（positions + normals） |
| **输出** | 配准结果 | `MarkerOpticalFlowFuseCPUResult`（R: `Matx33d`、T: `Vec3d`、transform: `Matx44d`、markers: `vector<TransformedMarker>`、statistics） |

---

## C. 算法

**核心流程**（利用相邻帧标记点空间相近性，~1mm 间距 vs 100~200mm 标记点间距）：

```
Step 1: 法线归一化（单位化）
Step 2: 暴力最近邻匹配（距离 + 法线角度双门限）
  - 后续帧: matchDistThresh 门限
  - 首帧配准: matchDistThresh × 10 门限（容许大初始偏移）
  - 法线角度采用双面语义: acos(min(1, |n1·n2|))，即 n 和 −n 等价（角度 ∈ [0°, 90°]）
Step 3: 全局 ID 继承（prevState.globalIds[匹配索引]）
Step 4: 法线加权 SVD 刚体变换求解（Kabsch 算法）
  - 权重 wk = (srcNorm·dstNorm)²（点积平方，等价于 cos²）
  - 退化兜底: wTotal < 1e-12 时等权重
  - 反射修正: det(R) < 0 时取反 Vt 第三行
Step 5: 变换当前帧全部标记点至全局坐标系
Step 6: 质量评估（RMSE + 重叠率）
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::SVD::compute` | 加权 SVD 分解（3×3 交叉协方差矩阵 H） |
| `cv::determinant` | 旋转矩阵反射检测（det < 0 修正） |

> **注**：匹配为纯暴力 O(N×M) 最近邻，标记点数通常 < 50，无需 KD-Tree / FLANN。不依赖 Eigen。法线角度使用 `abs(点积)` 实现**双面一致性**（法线方向翻转不影响匹配）。

---

## D. 依赖

**上下游算子**：

```
point_reconstruct → [本算子] → 下游（全局标记点云 / 设备姿态）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `PrevFrameState` | 调用方维护，跨帧传递上一帧状态（原始坐标 + 全局 ID + R/T） |
| `GlobalMarkerSet` | 可选外部全局标记点集，提供独立全局参考 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | `QualityFlag` |
| `common/zitai_result_types.h` | `PointReconstructCPUResult`（输入重载类型） |
| `common/calib_logging.h` | 日志宏 |
| `common/calib_warmup_config.h` | `WarmupConfig` |

---

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| `point_reconstruct` | `PointReconstructCPUResult`（重载②）或 `vector<Point3d>+vector<Vec3d>`（重载①③） | 跳过 `!validCircle` 的标记点，提取 centerX/Y/Z + normalX/Y/Z |
| 调用方状态 | `PrevFrameState` const 引用 | 上一帧的原始坐标 + 全局 ID + 上次 R/T |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| `R`, `T`, `transform` | 下游姿态估计 | — | 值（`Matx33d`/`Vec3d`/`Matx44d`） | 必用 |
| `markers[].transformedPosition` | 全局标记点云 | — | `vector<TransformedMarker>` | 必用 |
| `markers[].globalId` | 轨迹追踪 | — | 值 | 必用 |
| `markers[].rawPosition` | 原始坐标保留 | — | 值 | 可选 |
| `result.success` | Pipeline 调度 | — | 值 | 必用 |
| `result.qualityFlag` | Pipeline 调度 | — | 值 | 可选 |

---

## E. 架构

**文件结构**：

```
optical_flow_fuse/
├── marker_optical_flow_fuse_cpu.h
├── marker_optical_flow_fuse_cpu.cpp
└── tests/
    └── test_marker_optical_flow_fuse_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `MarkerOpticalFlowFuseCPU`（pImpl） |
| 核心方法 | `Execute()` × 3 重载（见下） |
| 预热方法 | `Warmup(maxMarkerCount)` / `Warmup(WarmupConfig)` |
| 参数更新 | `SetParams()` / `GetParams()` |
| 统计查询 | `GetStatistics()` / `ResetStatistics()` |
| 参数结构体 | `MarkerOpticalFlowFuseCPUParams` |
| 状态结构体 | `PrevFrameState`、`GlobalMarkerSet` |
| 结果项结构体 | `TransformedMarker` |
| 统计结构体 | `MarkerOpticalFlowFuseStats` |
| 结果结构体 | `MarkerOpticalFlowFuseCPUResult`（move-only） |
| 日志标签 | `"01-MarkerOpticalFlowFuseCPU"` |

**`Execute()` 三重载**：

| 重载 | 输入 | 用途 |
|------|------|------|
| ① | `positions + normals + prevState` | 基础接口（无全局参考） |
| ② | `PointReconstructCPUResult + prevState` | 便捷接口（直接接 11 号算子输出） |
| ③ | `positions + normals + prevState + globalMarkers` | 带外部全局参考集 |

---

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | ≥ 4.x | `opencv_core`（SVD、Matx、矩阵运算） |
| nlohmann_json | ≥ 3.11 | 参数序列化 |
| spdlog | ≥ 1.15 | 日志 |
| GoogleTest | ≥ 1.14 | 单元测试 |
| C++ 标准 | C++17 | 禁止 C++20 |

> **不依赖** Eigen、FLANN、PCL — 纯 OpenCV + 标准库实现。

---

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| `matchDistThresh` | double | 2.0 | > 0 | 匹配距离阈值（mm）；首帧配准时放大 10× |
| `normalAngleThresh` | double | 15.0 | (0, 90) | 法线角度阈值（度） |
| `minMatchedPoints` | int | 3 | ≥ 3 | 最少匹配点数（低于此值返回失败） |
| `collectStatistics` | bool | true | — | 是否填充统计（实际始终填充） |
| `maxMarkerCount` | size_t | 1000 | > 0 | 最大标记点数（超限返回失败） |

**预设配置**：`default`（2mm, 15°） / `precise`（1mm, 10°, minMatched=4） / `coarse`（5mm, 30°, maxMarkerCount=2000）

---

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 目标精度 | ② ~0.05mm（标记点位置） |
| 匹配复杂度 | O(N×M)，标记点 N/M 通常 < 50 |
| 线程安全 | 非线程安全（Debug 模式 `atomic<bool> inProcess_` 存根） |
| 状态管理 | **无状态**（调用方维护 `PrevFrameState`） |
| 首帧处理 | `prevState.empty()` → 自动初始化（globalId=索引）或外部全局配准 |

---

## K. 质量

**QualityFlag 语义**（基于重叠率 = matchedCount / currentFrameCount）：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| `Normal` | 完全可信 | 重叠率 ≥ 0.8（或首帧自动初始化） |
| `Degraded` | 精度降级但可用 | 重叠率 ∈ [0.5, 0.8) |
| `Warning` | 可用但需关注 | 重叠率 < 0.5 |

> **重要**：以上 QualityFlag 仅在 `success=true` 路径设置。所有 `success=false` 的失败路径**不修改 qualityFlag**，其保持默认值 `Normal`。下游消费者**必须先检查 `success`** 再信任 `qualityFlag`。

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败（`validate`） | 抛出 `std::invalid_argument` |
| 当前帧为空 | `success=false`，`message="Empty current frame"` |
| 标记点数超限 | `success=false`，`message="Point count exceeds maxMarkerCount"` |
| 匹配点不足（< minMatchedPoints） | `success=false`，`message="Insufficient matched points: X < Y"` |
| 后续帧 SVD 失败（匹配点 < 3） | `success=false`，`message="SVD failed"` |
| 首帧 SVD 失败 | `success=false`，`message="First frame: SVD failed"` |
| 首帧配准匹配不足 | `success=false`，`message="First frame: insufficient matches to global markers"` |

---

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `collectStatistics` 参数存在但未实际检查，统计始终填充 | 参数无效但无负面影响 |
| 🟡 中 | `Warmup()` 为空操作（仅置 `warmed_up_=true`），无实际预分配 | 首帧可能略慢 |
| 🟡 中 | 失败路径 `qualityFlag` 保持默认 `Normal`（未设为 Degraded/Warning） | 下游若不检查 `success` 直接读 `qualityFlag` 会误判为可信 |
| 🟢 低 | `matchTimeMs`/`svdTimeMs`/`transformTimeMs` 仅在后续帧路径填充，首帧路径始终为 0 | 首帧无分步计时 |
| 🟢 低 | `inProcess_` 原子标志声明但 `Execute()` 路径未设置 true | 并发检测实际不生效 |
| 🟢 低 | 匹配为暴力 O(N×M)，标记点 > 100 时性能下降 | 实际标记点 < 50，可接受 |
| 🟢 低 | 法线退化兜底（wTotal < 1e-12）时退化为等权重 SVD | 极端场景精度下降 |

---

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
| **现有模块** | `optical_flow_fuse`（`MarkerOpticalFlowFuseCPU`） |
| **复用方式** | 完整实现：暴力最近邻匹配 + 法线加权 SVD（Kabsch）+ 全局 ID 继承 + 首帧自动初始化/外部配准双模式 |

---

> **文档结束**
