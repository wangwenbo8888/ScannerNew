# 双目外参标定

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 标定-3 |
| 中文名称 | 双目外参标定 |
| 英文目录名 | extrinsic_calib |
| 运行平台 | CPU |
| 所属流程 | 标定流程（外参标定阶段） |
| 精度档次 | ① 亚像素级（重投影误差阈值 1.0px，极线误差阈值 0.05px） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机多帧角点 | `std::vector<std::vector<cv::Point2f>>` |
| **输入②** | 右相机多帧角点 | `std::vector<std::vector<cv::Point2f>>` |
| **输入③** | 3D物方点 | `std::vector<cv::Point3f>` |
| **输入④（Flow1）** | cameraMatrixL/R, distCoeffsL/R | 已知内参，通过参数传入 |
| **输入⑤（Flow2）** | calibrateMono=true | 自动标定内参+外参 |
| **输出** | `ExtrinsicCalibCpuResult`: R(3x3), T(3x1), E(3x3), F(3x3), cameraMatrixL/R, distCoeffsL/R, stereoReprojError, epipolarErrorMean/Std, perViewErrors（**注：当前代码中 rvecsStereo/tvecsStereo 未传入 stereoCalibrate，perViewErrors 始终为空，属已知缺陷**）, perViewEpipolarErrors, qualityFlag | 结构体 |
| **输出** | `ExtrinsicCalibCpuFullResult`: + monoLeft/monoRight（Flow2 时） | 结构体 |

## C. 算法

**核心流程（Flow 1 — 固定内参）**：

```
Step 1: 验证输入内参合理性（焦距/主点/宽高比）
Step 2: cv::stereoCalibrate with CALIB_FIX_INTRINSIC — 固定内参只算 R, T, E, F
Step 3: 计算每视图重投影误差（左/右分别投影取平均）
Step 4: 计算极线误差: cv::computeCorrespondEpilines → 点到线距离
Step 5: 质量评估: RMS > threshold → Degraded; 极线误差 > threshold → Degraded
```

**核心流程（Flow 2 — 自动标定内参+外参）**：

```
Step 1: cv::calibrateCameraRO 分别标定左/右单目内参
Step 2: 若单目失败使用粗略内参估计 (fx=imageWidth)
Step 3: 调用 Flow 1 的 Execute
Step 4: 聚合质量: maxQuality(stereo, monoLeft, monoRight)
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::stereoCalibrate` | 双目外参标定，计算 R, T, E, F |
| `cv::calibrateCameraRO` | 单目内参标定（Flow2） |
| `cv::computeCorrespondEpilines` | 计算极线，评估极线误差 |
| `cv::projectPoints` | 重投影，计算重投影误差 |
| `cv::Rodrigues` | 旋转向量与矩阵转换 |

## D. 依赖

**上下游算子**：

```
intrinsic_calib（提供内参）+ 角点检测算子 → 本算子 → stereo_rectify（提供 R, T）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `../common/quality_flag.h` | QualityFlag 枚举定义 |
| `../common/json_utils.h` | JSON 序列化/反序列化工具 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| intrinsic_calib | cameraMatrixL/R, distCoeffsL/R | Flow1: 通过 Execute() 方法参数传入 |
| 角点检测算子 | 左/右角点 + 3D物方点 | `std::vector<std::vector<cv::Point2f>>` |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| R | stereo_rectify | StereoRectifyCpuParams | ExtrinsicCalibCpuResult | 必用 |
| T | stereo_rectify | StereoRectifyCpuParams | ExtrinsicCalibCpuResult | 必用 |
| qualityFlag | 质量检查 | - | ExtrinsicCalibCpuResult | 必用 |
| epipolarErrorMean/Std | 质量检查 | - | ExtrinsicCalibCpuResult | 可选 |

## E. 架构

**文件结构**：

```
extrinsic_calib/
├── extrinsic_calib_cpu.h
├── extrinsic_calib_cpu.cpp
└── tests/
    └── test_extrinsic_calib_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `ExtrinsicCalibCpu`（pImpl 模式） |
| 核心方法 | `Execute()`（Flow1/Flow2 两个重载） |
| `void SetParams(const ExtrinsicCalibCpuParams&)` | 设置参数 |
| `const ExtrinsicCalibCpuParams& GetParams() const` | 获取当前参数 |
| 参数结构体 | `ExtrinsicCalibCpuParams` |
| 结果结构体 | `ExtrinsicCalibCpuResult`, `ExtrinsicCalibCpuMonoResult`, `ExtrinsicCalibCpuFullResult` |
| 日志标签 | `"[03-ExtrinsicCalibCpu]"` |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | 4.x | 立体标定、极线几何、重投影 |
| Eigen3 | >= 3.4 | 线性代数（CMake 链接，当前源码未直接使用） |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| leftPointsPerView | vector\<vector\<Point2f\>\> | - | 非空 | 左相机各视图角点 |
| rightPointsPerView | vector\<vector\<Point2f\>\> | - | 非空 | 右相机各视图角点 |
| objectPoints | vector\<Point3f\> | - | >=4 | 3D物方点 |
| imageSize | cv::Size | - | >0 | 图像尺寸 |
| flags | int | 0 | - | stereoCalibrate 标志（**当前代码硬编码 `CALIB_FIX_INTRINSIC`，此参数实际未使用**） |
| calibrateMono | bool | false | - | 是否同时标定内参（启用 Flow2） |
| patternSize | cv::Size | - | >0 | 棋盘格尺寸（mono 时） |
| squareSize | float | 0.0 | >0 | 方格大小（mono 时） |
| maxReprojError | double | 1.0 | >0 | 最大重投影误差阈值(px) |
| minViewCount | int | 8 | >=3 | 最少视图数 |
| rotateRightImage180 | bool | false | - | 右图是否旋转180° |
| maxEpipolarError | double | 0.05 | — | 最大极线误差阈值(px) |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 重投影误差 | stereoReprojError <= 1.0px |
| 极线误差 | epipolarErrorMean <= 0.05px |
| 最小视图数 | 至少 minViewCount(8) 帧有效视图 |
| 内参合理性 | fx∈[100,10000], 主点偏移<30%图像尺寸, 宽高比∈[0.8,1.2] |
| 异常安全 | 捕获 cv::Exception, bad_alloc, 所有异常 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal (0) | 正常 | 所有误差在阈值内 |
| Degraded (1) | 精度退化 | 重投影误差或极线误差超阈值 |
| Warning (2) | 警告 | 内参参数异常（焦距/主点/宽高比） |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数非法 | 构造/SetParams/fromJson 时抛 `std::invalid_argument` |
| OpenCV 异常 | 捕获 `cv::Exception`，返回错误信息 |
| 内存不足 | 捕获 `bad_alloc` |
| 其他异常 | 全捕获，确保不崩溃 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | `Execute`(Flow2) 中单目失败时使用粗略内参 (fx=imageWidth) | 立体标定精度可能下降 |
| 🟡 中 | `checkIntrinsics` 中如果 K 不是 3x3 直接 return 不报错 | 非法内参可能被静默跳过 |
| 🟡 中 | `Execute()` 中 RMS 超阈值时直接赋值 `qualityFlag = Degraded`，会覆盖此前 `checkIntrinsics` 设置的 Warning(2)，导致严重等级降级 | Warning 被静默降为 Degraded |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 功能完整，双流程支持，测试充分（28个测试用例） |
