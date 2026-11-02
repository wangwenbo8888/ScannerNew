# 双目内参标定

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 标定-2 |
| 中文名称 | 双目内参标定 |
| 英文目录名 | intrinsic_calib |
| 运行平台 | CPU |
| 所属流程 | 标定流程（内参标定阶段） |
| 精度档次 | ① 亚像素级（RMS < 1.0px，重投影误差阈值 0.012px） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左相机多帧角点像素坐标 | `std::vector<std::vector<cv::Point2f>>` |
| **输入②** | 右相机多帧角点像素坐标 | `std::vector<std::vector<cv::Point2f>>` |
| **输出** | `IntrinsicCalibResult` 含 `MonocularCalibResult left/right`: camera_matrix(3x3), dist_coeffs(1xN), rvecs, tvecs, rms_error, per_view_errors, valid_frame_count；`IntrinsicCalibResult` 自身字段: reproj_error_mean, reproj_error_std, total_frames_input, valid_frames_count；两结构均含 §5.1 标准状态字段: success, qualityFlag, message | 结构体 |

## C. 算法

**核心流程**：

```
Step 1: 从 JSON 加载参数（棋盘格尺寸、方格大小、图像尺寸等）
Step 2: 生成3D物方点: actual_size = square_size * (1 + α × (T_plate - 20))
Step 3: 过滤有效帧: 检查点数匹配 + NaN 剔除
Step 4: 分别对左/右相机做单目标定: cv::calibrateCameraRO 或 cv::calibrateCamera
Step 5: 计算每视图重投影误差: cv::projectPoints 投影 → 欧氏距离
Step 6: 计算质量指标: 左右所有视图的 mean/std
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::calibrateCameraRO` | 单目标定（带固定参数版本） |
| `cv::calibrateCamera` | 单目标定（标准版本） |
| `cv::projectPoints` | 重投影，计算重投影误差 |
| `cv::FileStorage` | YAML 格式序列化 |

## D. 依赖

**上下游算子**：

```
inverse_distort（3-1，提供姿态合格帧的原始畸变 left_points/right_points） → 本算子 → extrinsic_calib（提供 cameraMatrix, distCoeffs）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `../common/json_utils.h` | JSON 序列化/反序列化工具 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| inverse_distort（3-1） | 左/右相机多帧原始畸变中心（姿态合格帧） | `std::vector<std::vector<cv::Point2f>>` |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| camera_matrix | extrinsic_calib | 通过 Execute() 方法参数传入 | MonocularCalibResult | 必用 |
| dist_coeffs | extrinsic_calib | 通过 Execute() 方法参数传入 | MonocularCalibResult | 必用 |
| reproj_error_mean/std | 质量评估 | - | IntrinsicCalibResult | 可选 |

## E. 架构

**文件结构**：

```
intrinsic_calib/
├── intrinsic_calib_cpu.h
├── intrinsic_calib_cpu.cpp
└── tests/
    └── test_intrinsic_calib_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `IntrinsicCalibCPU`（pImpl 模式） |
| 核心方法 | `Execute()`（双目/单目两个重载） |
| `const IntrinsicCalibParams& GetParams() const noexcept` | 获取当前参数 |
| `IntrinsicCalibParams::imageSize() const` | 返回 `cv::Size(image_width, image_height)` |
| `IntrinsicCalibParams::totalCorners() const` | 返回 `chessboard_width * chessboard_height`（即内角点行数×列数） |
| `MonocularCalibResult::isValid() const` | 检查标定结果是否有效 |
| `MonocularCalibResult::clear()` | 清空标定结果 |
| `static SaveResult(result, path)` | 保存双目标定结果到 YAML 文件 |
| `static LoadResult(path)` | 从 YAML 文件加载双目标定结果 |
| `static SaveMonoResult(result, path)` | 保存单目标定结果到 YAML 文件 |
| `static LoadMonoResult(path)` | 从 YAML 文件加载单目标定结果 |
| `static SaveResultJson(result, path)` | 保存双目标定结果到 JSON 文件 |
| `static LoadResultJson(path)` | 从 JSON 文件加载双目标定结果 |
| 参数结构体 | `IntrinsicCalibParams` |
| 结果结构体 | `MonocularCalibResult`, `IntrinsicCalibResult` |
| 日志标签 | spdlog::info/warn/error 直接输出 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | 4.x | 相机标定、重投影 |
| Eigen3 | >= 3.4 | 线性代数（CMake 链接，当前源码未直接使用） |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| chessboard_width | int | 11 | >0 | 棋盘格内角点列数 |
| chessboard_height | int | 8 | >0 | 棋盘格内角点行数 |
| square_size_mm | double | 15.0 | >0 | 方格物理尺寸(mm) |
| image_width | int | 2048 | >0 | 图像宽度 |
| image_height | int | 1536 | >0 | 图像高度 |
| use_calibrateCameraRO | bool | true | - | 是否使用 RO 版本 |
| calib_flags | int | 0 | OpenCV flags | 标定标志位 |
| reproj_error_threshold | double | 0.012 | >0 | 重投影误差阈值(px) |
| temperature_coeff | double | 5.0e-6 | - | 温度系数(/°C) |
| plate_temp | double | 21.0 | - | 标定板温度(°C) |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 重投影误差 | reproj_error_mean <= reproj_error_threshold (0.012px) |
| 最小帧数 | 需要至少有效帧才能标定 |
| 序列化格式 | 支持 YAML 和 JSON 两种 |

## K. 质量

**QualityFlag 语义**：

`IntrinsicCalibResult` 与 `MonocularCalibResult` 均含 §5.1 标准状态字段 `calib::QualityFlag qualityFlag`（默认 `QualityFlag::Normal`，与 §B、代码一致）。重投影误差阈值 `reproj_error_threshold` 用于质量评估，但当前仅作信息性日志（详见 §H 风险），未据此改写 `qualityFlag`。

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 帧过滤-点数不匹配 | 跳过该帧 |
| 帧过滤-NaN 点 | 跳过该帧 |
| 左右帧数不匹配 | 返回 `false` + error 日志 |
| 全部帧被过滤（有效帧数为0） | 返回 `false` + error 日志 |
| 空输入 | 返回 `false` + error 日志 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 无显式参数 `validate()`，不像其他算子那样在构造时强制验证 | 非法参数可能在标定时才暴露 |
| 🟢 低 | temperature_coeff 只用于生成 objectPoints | 补偿影响有限 |
| 🟡 中 | 标定板材料假设：temperature_coeff 默认值 5.0e-6/°C 对应玻璃/陶瓷标定板，与温度补偿模块的铝合金 CTE(23.6e-6/°C) 不同 | 混用会导致物方点坐标计算错误 |
| 🟡 中 | 重投影误差阈值未强制执行：reproj_error_threshold(0.012px) 仅为信息性日志（spdlog::info/warn），超阈值时 Execute() 仍返回 true | 典型相机标定 RMS 为 0.3-0.5px，阈值失去质量门禁作用 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 代码完整，支持双格式序列化，测试充分（22个测试用例） |
