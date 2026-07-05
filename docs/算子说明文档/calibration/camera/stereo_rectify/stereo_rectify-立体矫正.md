# 立体矫正

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 标定-4 |
| 中文名称 | 立体矫正 |
| 英文目录名 | stereo_rectify |
| 运行平台 | CPU |
| 所属流程 | 标定流程（立体矫正阶段） |
| 精度档次 | ② 整像素/几何类（~0.05 pixel，OpenCV 标准 stereoRectify） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左/右相机内参 | cameraMatrixL/R, distCoeffsL/R（通过参数传入） |
| **输入②** | 旋转和平移 | R(3x3), T(3x1)（通过参数传入） |
| **输入③** | 图像尺寸 | imageSize（通过参数传入） |
| **输出** | `StereoRectifyCpuResult`: R1, R2(3x3), P1, P2(3x4), Q(4x4), validRoiLeft, validRoiRight, qualityFlag | 结构体 |

## C. 算法

**核心流程**：

```
Step 1: 验证参数（所有矩阵尺寸，alpha∈[0,1]，flags 有效性）
Step 2: 调用 cv::stereoRectify — 计算立体矫正变换
Step 3: 输出 R1, R2, P1, P2, Q, validRoi
Step 4: 检查 validRoi 是否为零面积 → Warning
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::stereoRectify` | 计算立体矫正变换矩阵 R1, R2, P1, P2, Q |

## D. 依赖

**上下游算子**：

```
intrinsic_calib（内参）+ extrinsic_calib（R, T） → 本算子 → 温度补偿参数表(3-5) / 运行时实时矫正 / 阶段 4 激光标定
```

> v2.1 变更：原下游 inverse_distort（3-1）已移至内参标定(3-2) 之前，改用初始参数，不再由本算子供给 R1/P1。

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `../common/quality_flag.h` | QualityFlag 枚举定义 |
| `../common/json_utils.h` | JSON 序列化/反序列化工具 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| intrinsic_calib | cameraMatrixL/R, distCoeffsL/R | 通过 StereoRectifyCpuParams 传入 |
| extrinsic_calib | R, T | 通过 StereoRectifyCpuParams 传入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| R1, R2, P1, P2, Q | 温度补偿参数表(3-5)、运行时实时矫正、阶段 4 激光标定 | 矫正参数 | StereoRectifyCpuResult | 必用 |
| qualityFlag | 质量检查 | - | StereoRectifyCpuResult | 必用 |
| validRoiLeft/Right | 有效区域判断 | - | StereoRectifyCpuResult | 可选 |

## E. 架构

**文件结构**：

```
stereo_rectify/
├── stereo_rectify_cpu.h
├── stereo_rectify_cpu.cpp
└── tests/
    └── test_stereo_rectify_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `StereoRectifyCpu`（pImpl 模式） |
| 核心方法 | `Execute()` |
| `void SetParams(const StereoRectifyCpuParams&)` | 设置参数 |
| `const StereoRectifyCpuParams& GetParams() const` | 获取当前参数 |
| 参数结构体 | `StereoRectifyCpuParams` |
| 结果结构体 | `StereoRectifyCpuResult` |
| 日志标签 | `"[04-StereoRectifyCpu]"` |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | 4.x | 立体矫正核心函数 |
| Eigen3 | >= 3.4 | 线性代数（CMake 链接，当前源码未直接使用） |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cameraMatrixL | cv::Mat | - | 3x3 | 左相机内参 |
| distCoeffsL | cv::Mat | - | 1x5 | 左畸变系数 |
| cameraMatrixR | cv::Mat | - | 3x3 | 右相机内参 |
| distCoeffsR | cv::Mat | - | 1x5 | 右畸变系数 |
| imageSize | cv::Size | - | >0 | 图像尺寸 |
| R | cv::Mat | - | 3x3 | 旋转矩阵 |
| T | cv::Mat | - | 3x1 | 平移向量 |
| alpha | double | 0.0 | [0,1] | 裁剪系数（0=全裁剪, 1=保留所有） |
| flags | int | 1 | 0 或 CALIB_ZERO_DISPARITY | 矫正标志 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| validRoi 面积检测 | 零面积 → Warning |
| 异常安全 | 多层 catch |
| 参数验证 | `validate()` 检查所有矩阵尺寸及 alpha、flags |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | validRoi 面积 > 0 |
| Warning | 警告 | validRoi 面积 = 0 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法 | 抛出异常 |
| stereoRectify 失败 | 捕获异常，返回错误信息 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | `validate()` 不检查 distCoeffs 维度；1x5 约束仅在 `fromJson()` 中强制 | 编程构造参数时传入非 1x5 distCoeffs 不会被拦截 |
| 🔴 高 | flags 默认值 1 不合法 | 默认构造的参数 flags=1 既非 0 也非 CALIB_ZERO_DISPARITY(1024)，调用 validate() 或构造算子时必抛 std::invalid_argument。使用时须手动设为 0 或 cv::CALIB_ZERO_DISPARITY |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 简洁完整，测试充分（20个测试用例） |
