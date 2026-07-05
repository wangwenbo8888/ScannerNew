# 相机内参温度补偿

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 温度-1 |
| 中文名称 | 相机内参温度补偿 |
| 英文目录名 | intrinsic_compensate |
| 运行平台 | CPU |
| 所属流程 | 温度补偿流程 |
| 精度档次 | ① 高精度（double浮点，线性热膨胀模型，ΔT=0时bit-exact） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 标定温度下的相机内参 (fx, fy, cx, cy) + 标定温度 referenceTemp | CameraIntrinsics |
| **输出** | 温度补偿表（JSON数组，每条含 temperature, deltaT, fx/fy/cx/cy 补偿值, deltaFx/deltaFy/deltaCx/deltaCy 偏移量）；结果级别含 referenceTemp(参考温度), cte(热膨胀系数), referenceIntrinsics(参考内参，补偿基准） | IntrinsicCompensateCPUResult |

## C. 算法

**核心流程**：

```
Step 1: 验证输入内参 (fx>0, fy>0, cx>0, cy>0) 和算子参数
Step 2: 计算温度范围 T_start = refTemp + rangeMin, T_end = refTemp + rangeMax
Step 3: 对每个温度步距循环：
        ΔT = T - refTemp
        scale = 1 + cte × ΔT（线性热膨胀，6061-T6铝合金）
        补偿内参 = 原始内参 × scale
        偏移量 delta = 原始内参 × cte × ΔT
Step 4: 返回补偿表结果
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| 无（纯算术运算） | — |

## D. 依赖

**上下游算子**：

```
相机标定流程 → 本算子 → 温度补偿应用流程
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_logging.h` | 统一日志宏 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 相机标定流程 | CameraIntrinsics | 内参四元素必须 > 0，referenceTemp 为标定时环境温度 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| IntrinsicCompensateCPUResult | 温度补偿应用流程 | 查表插值 | JSON 补偿表 | 必用 |

## E. 架构

**文件结构**：

```
intrinsic_compensate/
├── intrinsic_compensate_cpu.h
├── intrinsic_compensate_cpu.cpp
└── tests/
    └── test_intrinsic_compensate_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | IntrinsicCompensateCPU |
| 核心方法 | Execute() |
| 参数结构体 | IntrinsicCompensateCPUParams |
| 结果结构体 | IntrinsicCompensateCPUResult, CompensatedEntry |
| 日志标签 | "01-IntrinsicCompensateCPU" |
| void SetParams(const IntrinsicCompensateCPUParams&) | 设置补偿参数 |
| const IntrinsicCompensateCPUParams& GetParams() const | 获取当前补偿参数 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| 编译器 | C++17 | 支持 C++17 |
| 平台 | CPU | 无 GPU 依赖 |
| 第三方库 | OpenCV 4.x / nlohmann_json / spdlog | 矩阵结构、JSON序列化、日志 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cte | double | 23.6e-6 | >0 | 材料线膨胀系数（/°C），6061-T6铝合金 |
| tempStep | double | 0.2 | >0 | 温度步距（°C） |
| tempRangeMin | double | -10.0 | ≤tempRangeMax | 参考温度下方范围（°C） |
| tempRangeMax | double | 10.0 | ≥tempRangeMin | 参考温度上方范围（°C） |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 精度 | double 精度，ΔT=0 时 bit-exact |
| 性能 | 默认 101 条表（~±10°C / 0.2°C 步距），微秒级 |
| 线程安全 | 非线程安全 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal (0) | 计算成功 | 输入验证通过，计算完成 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 输入非法 | `validate()` 抛出 `std::invalid_argument` |
| 计算异常 | 异常传播给调用者 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 浮点循环 t+=step 累积误差 | epsilon=step×1e-6 缓解 |
| 🟡 中 | cx/cy 线性缩放假设 | cx/cy 与 fx/fy 使用相同的热膨胀系数缩放。物理上主点变化由镜头筒膨胀偏移决定，并非简单的线性缩放。此近似在温度变化较小时有效 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
