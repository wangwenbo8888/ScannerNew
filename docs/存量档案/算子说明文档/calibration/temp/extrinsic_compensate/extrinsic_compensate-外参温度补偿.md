# 相机外参温度补偿

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 温度-2 |
| 中文名称 | 相机外参温度补偿 |
| 英文目录名 | extrinsic_compensate |
| 运行平台 | CPU |
| 所属流程 | 温度补偿流程 |
| 精度档次 | ① 高精度（double浮点，基线缩放模型，R不变，ΔT=0时bit-exact） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 标定温度下的双目外参（R[9] 旋转矩阵 3×3 行主序, T[3] 平移向量）+ referenceTemp | CameraExtrinsics |
| **输出** | 温度补偿表（JSON对象，`table` 数组每条含 temperature, deltaT, R[9] 不变, T[3] 补偿值, deltaT_vec[3] 偏移量；结果级别含 referenceTemp(参考温度), cte(热膨胀系数), baselineRef(参考基线长度), referenceExtrinsics(参考外参，补偿基准）） | ExtrinsicCompensateCPUResult |

## C. 算法

**核心流程**：

```
Step 1: 验证输入外参（T 非零，即 baseline > 0）和算子参数
Step 2: 计算温度范围 T_start = refTemp + rangeMin, T_end = refTemp + rangeMax
Step 3: 计算参考基线长度 baseline = |T₀|
Step 4: 对每个温度步距循环：
        scale = 1 + cte × ΔT
        R 不变，原样拷贝（std::copy）
        T_new[i] = T₀[i] × scale
        deltaT_vec[i] = T₀[i] × cte × ΔT
Step 5: 返回补偿表结果
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| 无（纯算术运算） | — |

## D. 依赖

**上下游算子**：

```
双目标定流程 → 本算子 → 温度补偿应用流程
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| ExtrinsicCompensateCPU | 被03算子内部组合复用，作为成员 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_logging.h` | 统一日志宏 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 双目标定流程 | CameraExtrinsics | T 必须非零（baseline > 0），R 为 3×3 行主序旋转矩阵 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| ExtrinsicCompensateCPUResult | 温度补偿应用流程 | 查表插值 | JSON 补偿表 | 必用 |
| ExtrinsicCompensateCPUResult | laser_extrinsic_compensate | Execute() | 直接复用类 | 必用 |

## E. 架构

**文件结构**：

```
extrinsic_compensate/
├── extrinsic_compensate_cpu.h
├── extrinsic_compensate_cpu.cpp
└── tests/
    └── test_extrinsic_compensate_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | ExtrinsicCompensateCPU |
| 核心方法 | Execute() |
| 参数结构体 | ExtrinsicCompensateCPUParams |
| 结果结构体 | ExtrinsicCompensateCPUResult, ExtrinsicCompensatedEntry |
| 日志标签 | "02-ExtrinsicCompensateCPU" |
| void SetParams(const ExtrinsicCompensateCPUParams&) | 设置补偿参数 |
| const ExtrinsicCompensateCPUParams& GetParams() const | 获取当前补偿参数 |

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
| 输入非法 | `validate()` 抛出 `std::invalid_argument`（T 为零向量、cte/tempStep/range 非法时） |
| 计算异常 | 异常传播给调用者 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 模型假设 R 完全不随温度变化，仅 T 缩放 | 物理上近似假设，适用于刚性安装结构 |
| 🟡 中 | T 向量线性缩放假设 | T 向量按 baseline 方向线性缩放，假设相机间距变化与结构材料热膨胀一致（默认 6061-T6 铝合金 CTE=23.6e-6/°C） |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
