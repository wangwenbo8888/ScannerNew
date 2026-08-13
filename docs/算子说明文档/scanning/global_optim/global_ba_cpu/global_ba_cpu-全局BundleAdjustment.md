# 全局 Bundle Adjustment（GBA）

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 全局优化-1 |
| 中文名称 | 全局 Bundle Adjustment |
| 英文目录名 | global_ba_cpu |
| 运行平台 | CPU（Ceres 稀疏求解） |
| 所属流程 | 后处理（离线全局优化，模块4）；非实时扫描链算子 |
| 精度档次 | ②（亚毫米级全局一致性） |

> 说明：本算子位于 `modules/09_operatorlib/scanning/global_optim/`，属"全局优化"类，与实时扫描流水线算子分开。供后处理编排调用。

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 全部帧的标记点观测（本机系3D + globalId）+ 各帧绝对位姿初值 | GlobalBAInput（含 `vector<GlobalBAFrame>`） |
| **输出** | 优化后的全局位姿图 + 标记点3D位置 + 统计 | GlobalBAResult（optimizedPoses / optimizedMarkers / statistics） |

**关键输入结构**：
- `GlobalBAFrame`：frameId、R_init/t_init（相邻帧链乘累积初值）、`vector<GlobalBAMarkerObs>{local, globalId}`。
- `globalId`：同物理标记点共享同一 globalId（由前置全局ID模块赋予），是建立跨帧约束的依据。

## C. 算法

**核心流程**：

1. **相邻帧相对位姿构建**：对共视（covisibility）的相邻帧，用 Kabsch（SVD + 反射修正）求解 `frame-k-local → frame-i-local` 的相对位姿 (R_ik, t_ik)。
   - 退化守卫：点数 < 3、共线/近共线（条件数 > 1e3，即 sMin/sMax < 1e-3）时返回 false，跳过该边（避免污染 PGO）。
2. **位姿图预优化（可选，`enablePoseGraphPreopt`）**：以相对位姿为约束做一次 PGO，得到更稳的初值。
3. **回环检测**：对相隔 ≥ `loopFrameGap` 且共视标记点 ≥ `minCovisForLoopEdge` 的帧对建立回环边。
4. **Ceres 全局 BA**：联合优化所有帧位姿 + 标记点3D位置。
   - 残差：`ba_residuals`（重投影/观测）+ `pose_graph_residuals`（相对位姿）+ 回环边。
   - 鲁棒核：Tukey（`tukeyC = 3σ`）。
   - 外点剔除：χ² 门控（`chiSquareGate = 11.34`，3-DoF 99%）。
5. **统计输出**：initialRMSE / finalRMSE（mm）、ceresIterations、loopDetected、outlierObsIds、loopClosureResidual。
6. **居中**：`centerOrigin=true` 时将优化结果整体平移使质心位于原点。

**关键函数**：

| 函数 | 用途 |
|------|------|
| `kabsch(zi, zk, R, t)` | 成对点集求相对刚体变换（SVD + 反射修正 + 退化守卫） |
| Ceres `Solve` | 稀疏 Levenberg-Marquardt 全局优化 |
| Tukey 鲁棒损失 + χ² 门控 | 外点抑制 |

## D. 依赖

**上下游算子**：

```
point_reconstruct ──(本机系3D点)──┐
optical_flow_fuse / frame_fuse ──(R/T 初值)──┤
全局ID 模块 ──(globalId)──┘ ▶ [global_ba_cpu] ▶ 优化后 R/T → 重融合(后处理)
```

**第三方依赖**：

| 依赖 | 用途 |
|------|------|
| Ceres | 稀疏非线性最小二乘（BA 主求解器） |
| Eigen | SVD / 矩阵运算（Kabsch） |
| OpenCV core | `cv::Point3d` / `cv::Matx33d` 类型载体 |
| spdlog | 日志 |
| nlohmann_json | 参数序列化 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/quality_flag.h` | QualityFlag 枚举 |
| `ba_residuals.h` | BA 残差块 |
| `pose_graph_residuals.h` | 位姿图残差块 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| point_reconstruct | `GlobalBAMarkerObs.local` | 本机系3D坐标（观测 z_ij） |
| optical_flow_fuse/frame_fuse | `GlobalBAFrame.R_init/t_init` | 绝对位姿初值（相邻帧链乘累积） |
| 全局ID 模块 | `GlobalBAMarkerObs.globalId` | 同物理点同 ID，建立跨帧约束 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 必用/可选 |
|---------|--------|---------|---------|
| optimizedPoses (R/T) | 后处理-重融合 | 用优化后位姿重跑 laser_cloud_fuse_cuda | 必用 |

## E. 架构

**文件结构**：

```
global_optim/
├── global_ba_cpu.h
├── global_ba_cpu.cpp
├── ba_residuals.h
├── pose_graph_residuals.h
├── gba_dataset_runner.cpp        # 离线 exe（含 main，不入库）
└── tests/
    └── test_global_ba_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `GlobalBundleAdjustmentCPU`（pImpl） |
| 核心方法 | `Execute(const GlobalBAInput&)` |
| 参数结构体 | `GlobalBAParams` |
| 结果结构体 | `GlobalBAResult`（move-only，禁拷贝） |
| 统计结构体 | `GlobalBAStats` |
| 参数更新 | `SetParams()` / `GetParams()` |
| 统计查询 | `GetStatistics()` |
| 资源释放 | `Destroy()` |
| 日志标签 | `"13-GlobalBundleAdjustmentCPU"` |

**状态模型（算子规范 §4）**：无状态——全量数据按调用传入，实例不持有跨调用累积状态；`SetParams` 仅缓存只读配置。并发策略：每实例非线程安全，多实例并行各自独占。

## F. 参数（GlobalBAParams）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| enablePoseGraphPreopt | bool | true | 是否在 BA 前做一次位姿图预优化 |
| sigmaObserved | double | 0.01 | 观测噪声 σ（mm） |
| tukeyC | double | 0.03 | Tukey 鲁棒核阈值（= 3σ） |
| chiSquareGate | double | 11.34 | χ² 外点门控（3-DoF 99%） |
| tolerance | double | 1e-10 | Ceres 收敛容差 |
| maxIterations | int | 200 | Ceres 最大迭代数 |
| minCovisForLoopEdge | int | 5 | 建立回环边所需最小共视点数（≥3） |
| loopFrameGap | int | 30 | 回环候选帧最小间隔 |
| minPointsPerFrame | int | 3 | 帧有效所需最少标记点数（≥3） |
| centerOrigin | bool | true | 是否将结果质心居中至原点 |

参数支持 `toJson()` / `fromJson()`，`validate()` 在非法值时抛 `std::invalid_argument`。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 最小约束 | minPointsPerFrame ≥ 3、minCovisForLoopEdge ≥ 3（validate 强制） |
| 退化处理 | Kabsch 共线/点数不足时跳过该边，不喂 PGO |
| 求解规模 | 仅适合离线 batch；不要在实时扫描链中调用 |

## K. 质量

**QualityFlag 语义（依 Result.success / message / qualityFlag）**：

| 字段 | 含义 |
|------|------|
| success | Ceres 是否成功求解并产出优化结果 |
| message | 失败/降级原因文本 |
| qualityFlag | Normal / Degraded / Warning / Error（与全局 QualityFlag 一致） |
| statistics.initialRMSE/finalRMSE | 优化前后 RMSE（mm），衡量收益 |
| statistics.loopDetected | 是否检测到回环 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 初值质量差（漂移大）可能导致 Ceres 陷入局部极小 | 需 `enablePoseGraphPreopt=true` 预优化 |
| 🟢 低 | Kabsch 退化已被条件数守卫拦截 | 退化边被跳过，不影响其余 |
| 🟡 中 | 无回环时仅做前向 BA，全局漂移无法消除 | `loopDetected=false` 可据此判断 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 已实现（含 Ceres 求解 + 单测）；ctest `test_global_ba_cpu` 注册 |
| **运行时开关** | CMake `BUILD_GLOBAL_OPTIM`（OFF 时剔除本源，不依赖 Ceres） |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Ceres | 2.2.0 | 仅 `BUILD_GLOBAL_OPTIM=ON` 时链接 |
| Eigen | 3.4.1 | SVD |
| OpenCV | ≥ 4.x | 类型载体 |
| nlohmann_json | ≥ 3.2.0 | 参数序列化 |
| spdlog | 1.x | 日志 |
