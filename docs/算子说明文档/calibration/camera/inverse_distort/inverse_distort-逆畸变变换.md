# 逆畸变+逆矫正变换

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 标定-1 |
| 中文名称 | 逆畸变+逆矫正变换 |
| 英文目录名 | inverse_distort |
| 运行平台 | CPU |
| 所属流程 | 标定流程（**前端阶段**，位于内参标定 3-2 之前；将姿态合格帧的矫正后椭圆中心逆映射为原始畸变坐标，供内参标定使用） |
| 精度档次 | ① 亚像素级（~0.01 pixel），往返误差 < 0.001~0.01px |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 矫正后图像点坐标（姿态合格帧的椭圆中心 `EllipseFitCPUResult.center`，2-7 输出） | `std::vector<cv::Point2f>` |
| **输入②** | 归一化无畸变坐标（可选） | `std::vector<cv::Point2f>` |
| **输入③** | **初始** cameraMatrix, distCoeffs, R1, P1（与 2-6 去畸变参数一致） | 通过参数传入 |
| **输出** | `InverseDistortResult`: originalPoints（原始畸变像素坐标）, iterationsPerPoint, maxIterationsUsed, success, qualityFlag, message | 结构体 |
| **输出** | `RoundTripVerifyResult`: passed, maxError, meanError, perPointErrors, success, qualityFlag, message | 结构体 |

## C. 算法

**核心流程**：

```
Step 1: 从矫正后像素坐标 (P1) 反投影到归一化平面: x_n = (px - cx_rect) / fx_rect
Step 2: 逆 R1 旋转: 克莱默法则解 2x2 线性系统 x_u = R1^{-1} * x_n
Step 3: 正向畸变模型: 径向畸变(k1,k2,k3) + 切向畸变(p1,p2)
Step 4: 像素映射: u = fx * x_d + cx, v = fy * y_d + cy
Step 5: 往返验证: OpenCV cv::undistortPoints 正向 → 自定义逆变换 → 比较误差
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::undistortPoints` | 验证：正向去畸变，与逆变换结果比较 |
| 自定义 `applyDistortionModel()` | 径向+切向正向畸变（单步直接计算，非迭代） |
| 自定义 `InverseRectify()` | 逆 R1 旋转（克莱默法则解 2x2 系统，退化时齐次除法） |

## D. 依赖

**上下游算子**：

```
ellipse_fit（提供矫正后椭圆中心）+ 初始参数（cameraMatrix/distCoeffs/R1/P1，与 undistort_cpu 一致） → 本算子 → intrinsic_calib（内参标定，消费原始畸变中心）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `../../common/json_utils.h` | JSON 序列化/反序列化工具 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| ellipse_fit | 矫正后椭圆中心（EllipseFitCPUResult.center） | 仅姿态合格（命中 25 目标）帧 |
| 初始参数（阶段 0） | cameraMatrix, distCoeffs, R1, P1 | 通过 InverseDistortParams 传入；与 undistort_cpu 严格一致 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| originalPoints | intrinsic_calib（内参标定） | calibrateCameraRO 输入像点 | InverseDistortResult | 必用 |
| RoundTripVerifyResult | 质量检查 | - | 结构体 | 可选 |

## E. 架构

**文件结构**：

```
inverse_distort/
├── inverse_distort_cpu.h
├── inverse_distort_cpu.cpp
└── tests/
    └── test_inverse_distort_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `InverseDistortCPU`（pImpl 模式） |
| 核心方法 | `Execute()`, `ApplyDistortion()`, `InverseRectify()`, `VerifyRoundTrip()` |
| `const InverseDistortParams& GetParams() const noexcept` | 获取当前参数 |
| 参数结构体 | `InverseDistortParams` |
| 结果结构体 | `InverseDistortResult`, `RoundTripVerifyResult` |
| 内部结构体 | `DistortionCoeffs`（k1, k2, p1, p2, k3） |
| 日志标签 | `"[01-InverseDistortCpu]"` |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | 4.x | 图像处理、相机模型 |
| Eigen3 | >= 3.4 | 线性代数（逆变换死代码中使用） |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cameraMatrix | cv::Mat | - | 3x3 | **初始**相机内参矩阵（与 2-6 一致） |
| distCoeffs | cv::Mat | - | 1x5（仅使用 k1,k2,p1,p2,k3，高阶系数被忽略） | **初始**畸变系数 |
| R1 | cv::Mat | - | 3x3 | **初始**矫正旋转矩阵 |
| P1 | cv::Mat | - | 3x4 | **初始**矫正投影矩阵 |
| maxIterations | int | 20 | >0 | 预留迭代次数上限（**当前实现为单步直接计算，此参数实际未使用**） |
| tolerance | double | 1e-10 | >0 | 预留收敛容差（**当前实现为单步直接计算，此参数实际未使用**） |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 往返误差（角落） | < 0.01px |
| 往返误差（中心） | < 0.001px |
| ABI 隔离 | pImpl 模式确保 |
| 参数验证 | `validate()` 检查所有矩阵尺寸 |

## K. 质量

**QualityFlag 语义**：

`qualityFlag` 字段存在（默认 `Normal`）但当前实现未对其赋值；质量反馈通过 `bool success` + `message`（`RoundTripVerifyResult` 另含 `passed`）。

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法 | 抛 `std::invalid_argument` |
| 空输入 | 返回 `false` + warn 日志 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `distortPointIter()` 函数名为 "Iter" 但实际为单步直接计算（非迭代）；`undistortPointIter()` 含完整 Newton 迭代+Eigen 求解器但为**死代码**，无任何公开方法调用 | 开发者可能误以为有迭代优化，实际为单步评估 |
| 🟡 中 | `maxIterations`(20) 和 `tolerance`(1e-10) 参数在当前代码路径中完全未使用 | 参数名称暗示迭代控制，但无实际效果 |
| 🟢 低 | 无 operator_params.json 配置文件 | 参数需硬编码或通过代码构造 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 代码完整，测试充分（25个测试用例），包含往返验证 |
| **v2.1 流程变更** | 本算子由"标定末端（用已标定参数）"改为"内参标定前（用初始参数）"；代码本体不变，仅调用位置与输入参数来源变更。详见《算子说明文档_标定工作流》阶段 3 |
