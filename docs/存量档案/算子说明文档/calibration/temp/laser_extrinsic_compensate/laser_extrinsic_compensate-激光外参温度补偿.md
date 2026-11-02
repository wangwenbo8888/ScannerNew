# 激光器虚拟相机外参温度补偿

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 温度-3 |
| 中文名称 | 激光器虚拟相机外参温度补偿 |
| 英文目录名 | laser_extrinsic_compensate |
| 运行平台 | CPU |
| 所属流程 | 温度补偿流程 |
| 精度档次 | ① 高精度（复用02算子，double浮点） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 虚拟相机→左相机外参（R, T）+ referenceTemp | CameraExtrinsics |
| **输入②** | 虚拟相机→右相机外参（R, T）+ referenceTemp | CameraExtrinsics |
| **输出** | 两组温度补偿表：leftResult + rightResult；结果级别含 referenceTemp(参考温度), cte(热膨胀系数) | LaserExtrinsicCompensateCPUResult |

## C. 算法

**核心流程**：

```
Step 1: 验证两组输入外参（T 非零）
Step 2: 将参数转换为 ExtrinsicCompensateCPUParams
Step 3: 同步参数到内部补偿器 compensator_
Step 4: 调用 ExtrinsicCompensateCPU::Execute(virtualToLeft) → leftResult
Step 5: 调用 ExtrinsicCompensateCPU::Execute(virtualToRight) → rightResult
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| 无（复用 ExtrinsicCompensateCPU） | — |

## D. 依赖

**上下游算子**：

```
激光器标定流程 → 本算子 → 温度补偿应用流程
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| ExtrinsicCompensateCPU | 直接复用02算子，作为私有成员 compensator_，CMake: PUBLIC extrinsic_compensate_cpu |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |
| `common/calib_logging.h` | 统一日志宏 |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| 激光器标定流程 | CameraExtrinsics (×2) | 两组外参 T 均必须非零；referenceTemp 应一致 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| LaserExtrinsicCompensateCPUResult | 温度补偿应用流程 | 查表插值 | JSON 补偿表 | 必用 |

## E. 架构

**文件结构**：

```
laser_extrinsic_compensate/
├── laser_extrinsic_compensate_cpu.h
├── laser_extrinsic_compensate_cpu.cpp
└── tests/
    └── test_laser_extrinsic_compensate_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | LaserExtrinsicCompensateCPU |
| 核心方法 | Execute(virtualToLeft, virtualToRight) |
| 参数结构体 | LaserExtrinsicCompensateCPUParams |
| 结果结构体 | LaserExtrinsicCompensateCPUResult |
| 日志标签 | "03-LaserExtrinsicCompensateCPU" |
| void SetParams(const LaserExtrinsicCompensateCPUParams&) | 设置补偿参数 |
| const LaserExtrinsicCompensateCPUParams& GetParams() const | 获取当前补偿参数 |

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
| 性能 | 两次 Execute 调用，默认共 202 条表，微秒级 |
| 线程安全 | 非线程安全 |
| 异常安全 | try-catch 包裹每次 compensator_.Execute() 调用 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal (0) | 计算成功 | 输入验证通过，计算完成 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 输入非法 | `validate()` 抛出 `std::invalid_argument`（两组外参 T 为零、cte/tempStep/range 非法时） |
| Execute() 异常 | try-catch 包裹每次 compensator_.Execute() 调用 |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 两组外参 referenceTemp 应一致但未强制 | 上游标定流程应保证两组外参在同一温度下标定 |
| 🟡 中 | 继承 02 算子假设 | 本算子委托 ExtrinsicCompensateCPU::Execute()，继承其 T 向量线性缩放假设 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
