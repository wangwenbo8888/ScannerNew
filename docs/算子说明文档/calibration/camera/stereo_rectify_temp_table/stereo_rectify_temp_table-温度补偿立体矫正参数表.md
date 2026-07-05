# 温度补偿立体矫正参数表

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 标定-5 |
| 中文名称 | 温度补偿立体矫正参数表 |
| 英文目录名 | stereo_rectify_temp_table |
| 运行平台 | CPU |
| 所属流程 | 标定流程（温度补偿阶段） |
| 精度档次 | ② 工程级（温度补偿模型: scale = 1 + CTE × ΔT） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 左/右相机内参 | cameraMatrixL/R, distCoeffsL/R（通过参数传入） |
| **输入②** | 旋转和平移 | R(3x3), T(3x1)（通过参数传入） |
| **输入③** | 温度参数 | referenceTemp, cte, tempStep, tempRangeMin/Max（通过参数传入） |
| **输出** | `StereoRectifyTempTableResult`: 结果级别含 referenceTemp(参考温度), cte(热膨胀系数), tableSize(表条目数); table(vector\<StereoRectifyTempEntry\>), 每条含 temperature, deltaT, 补偿后内参/外参, R1, R2, P1, P2, Q, validRoi | 结构体 |

## C. 算法

**核心流程**：

```
Step 1: 遍历温度范围: temp = (refTemp + rangeMin) → (refTemp + rangeMax), 步长 tempStep
Step 2: 温度补偿内参: scale = 1 + CTE × ΔT; fx(T) = fx₀ × scale 等
Step 3: 温度补偿外参: T(T) = T₀ × scale（仅平移，R 不变）
Step 4: 对每个温度点调用 cv::stereoRectify → 生成矫正参数
Step 5: 汇总: 失败计数→Degraded, 零 validRoi 计数→Warning
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| `cv::stereoRectify` | 每个温度点调用一次，生成矫正参数 |

## D. 依赖

**上下游算子**：

```
intrinsic_calib + extrinsic_calib → 本算子 → 生产运行时根据温度查表获取矫正参数
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `common/calib_types.h` | 标定通用类型定义 |
| `../common/json_utils.h` | JSON 序列化/反序列化工具 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| intrinsic_calib + extrinsic_calib | cameraMatrixL/R, distCoeffsL/R, R, T, imageSize | 参考温度下的标定结果 |
| （自包含） | 温度补偿模型自实现 | cte, referenceTemp 为本算子自身参数，不依赖 wendu/01 或 wendu/02 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| table | 生产运行时 | 温度查表 | StereoRectifyTempTableResult | 必用 |
| R1, P1 per temp | 生产运行时 | 矫正映射 | 每条 entry | 必用 |
| qualityFlag | 质量检查 | - | StereoRectifyTempTableResult | 可选 |

## E. 架构

**文件结构**：

```
stereo_rectify_temp_table/
├── stereo_rectify_temp_table_cpu.h
├── stereo_rectify_temp_table_cpu.cpp
└── tests/
    └── test_stereo_rectify_temp_table_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | `StereoRectifyTempTableCpu`（非 pImpl，直接持有 params_） |
| 核心方法 | `Execute()` |
| `void SetParams(const StereoRectifyTempTableParams&)` | 设置参数 |
| `const StereoRectifyTempTableParams& GetParams() const` | 获取当前参数 |
| 参数结构体 | `StereoRectifyTempTableParams` |
| 结果结构体 | `StereoRectifyTempEntry`, `StereoRectifyTempTableResult` |
| 日志标签 | `"05-StereoRectifyTempTableCpu"` |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | 4.x | 立体矫正核心函数 |
| nlohmann_json | >= 3.2.0 | JSON 序列化/反序列化 |
| spdlog | 1.x | 日志输出 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cameraMatrixL/R | cv::Mat | - | 3x3 | 参考温度下左/右内参 |
| distCoeffsL/R | cv::Mat | - | 1x5 | 畸变系数 |
| R | cv::Mat | - | 3x3 | 旋转矩阵 |
| T | cv::Mat | - | 3x1 | 平移向量 |
| imageSize | cv::Size | - | >0 | 图像尺寸 |
| referenceTemp | double | 25.0 | - | 参考温度(°C) |
| cte | double | 23.6e-6 | >0 | 热膨胀系数(/°C) |
| tempStep | double | 0.2 | >0 | 温度步距(°C) |
| tempRangeMin | double | -10.0 | <=tempRangeMax | 最低温度偏移 |
| tempRangeMax | double | 10.0 | >=tempRangeMin | 最高温度偏移 |
| alpha | double | 0.0 | [0,1] | 裁剪系数 |
| flags | int | 1 | 0 或 CALIB_ZERO_DISPARITY | 矫正标志 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 默认性能 | 默认 101 点（range 20°C / step 0.2，即 20/0.2+1=101）约 100ms 级总耗时 |
| 表大小 | (tempRangeMax - tempRangeMin) / tempStep + 1 |
| 结果语义 | 结果不可拷贝（move-only） |
| 设计模式 | 非 pImpl 设计 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 全部温度点成功，validRoi 均非零 |
| Degraded | 降级 | 有 stereoRectify 失败的温度点 |
| Warning | 警告 | 有 validRoi 为零的温度点 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 单点 stereoRectify 失败 | 记录失败计数，继续处理其余温度点 |
| 参数非法 | 抛 `std::invalid_argument`（构造/SetParams/fromJson 时） |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 热膨胀模型简化: 仅缩放焦距和主点，畸变系数不补偿；R 假定不变 | 高温差场景下补偿精度受限 |
| 🟡 中 | cx/cy 也做了缩放 — 物理上主点是否随温度线性缩放需确认 | 补偿模型准确性待验证 |
| 🟢 低 | 非 pImpl 设计 — ABI 不隔离 | 接口变更需重新编译 |
| 🔴 高 | flags 默认值 1 不合法 | 同标定-04，默认 flags=1 无法通过 validate()。config/operator_params.json 中 flags=1 同样会导致 fromJson() 抛异常 |
| 🟡 中 | QualityFlag 覆盖行为 | 当 failCount>0 和 warnCount>0 同时满足时，代码先设 Degraded 再覆盖为 Warning，最终结果为 Warning 而非 Degraded |
| 🔴 高 | T 字段序列化不一致 | toJson() 将 T 序列化为 2D 嵌套数组 [[v],[v],[v]]，fromJson() 按 1D 扁平数组解析 [v,v,v]，往返会抛异常 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 有唯一配置文件 operator_params.json，代码带详细性能计时，测试 9 个用例 |
