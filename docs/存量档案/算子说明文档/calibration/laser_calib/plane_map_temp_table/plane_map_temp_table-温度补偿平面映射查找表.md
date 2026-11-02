# 温度补偿平面映射查找表

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 激光标定-13 |
| 中文名称 | 温度补偿平面映射查找表 |
| 英文目录名 | plane_map_temp_table |
| 运行平台 | CPU (流程编排) + CUDA (内部调用plane_map) |
| 所属流程 | 激光标定流程 |
| 精度档次 | ③ 亚像素/浮点类 |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入** | PlaneMapTempTableParams（含左右相机内外参、虚拟相机参数、温度范围、热膨胀系数等） | 结构体 |
| **输出** | PlaneMapTempTableResult: 结果级别含 referenceTemp(参考温度), cte(热膨胀系数), tableSize(表条目数); table(vector\<PlaneMapTempEntry\>)，每个entry含温度、补偿后内外参、R1/R2/P1/P2/Q、d_left_to_right、d_right_u | 结构体 |

## C. 算法

**核心流程**：

```
Step 1: 生成虚拟像素网格 (VirtualPixelGenerator)
Step 2: 对每个温度点:
  a. 热膨胀补偿内参: 仅缩放 fx/fy/cx/cy 四个元素 × (1 + cte * deltaT)，保留 K[2,2]=1
  b. 热膨胀补偿基线: Tc = T * (1 + cte * deltaT)
  c. 热膨胀补偿虚拟T
  d. cv::stereoRectify 重新计算 R1/R2/P1/P2/Q
  e. 调用 PlaneMapCuda::Execute 生成该温度下的映射表
Step 3: 收集所有温度点的结果
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::stereoRectify | 重新计算矫正变换矩阵 |
| PlaneMapCuda::Execute | 生成单个温度点的映射表 |
| VirtualPixelGenerator::Execute | 生成虚拟像素网格 |

## D. 依赖

**上下游算子**：

```
pose_optimize + 立体标定结果 → 本算子 → 无（最终输出，用于运行时查表）
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| plane_map | 内部调用，生成单温度点映射表 |
| VirtualPixelGenerator | 虚拟像素网格生成 |
| `common/calib_logging.h` | 统一日志宏 |
| `common/calib_types.h` | 共享类型定义（QualityFlag 等） |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| pose_optimize | 值传递 | virtualK, virtualR, virtualT |
| 立体标定结果 | 结构体传递 | 左右相机内外参 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| table | 运行时查表 | - | 序列化 | 必用 |

## E. 架构

**文件结构**：

```
plane_map_temp_table/
├── plane_map_temp_table.h
├── plane_map_temp_table.cpp
└── tests/
    └── test_plane_map_temp_table.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | PlaneMapTempTable (直接持有params, 无pImpl) |
| 核心方法 | Execute() |
| 参数更新 | SetParams() / GetParams() |
| 参数结构体 | PlaneMapTempTableParams |
| 结果结构体 | PlaneMapTempTableResult (含 PlaneMapTempEntry) |
| 日志标签 | "13-PlaneMapTempTable" |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV | >= 4.x | stereoRectify等 |
| CUDA | >= 11.0 | 内部调用plane_map |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cameraMatrixL/R | cv::Mat | 必需 | 3×3 | 左/右相机内参 |
| distCoeffsL/R | cv::Mat | 必需 | 1×5 | 左/右畸变系数 |
| R | cv::Mat | 必需 | 3×3 | 旋转矩阵 |
| T | cv::Mat | 必需 | 3×1 | 平移向量 |
| virtualK/R/T | Matx33d/Vec3d | 必需 | — | 虚拟相机参数 |
| lineIds | vector\<int\> | 必需 | 非空 | 激光线ID列表 |
| referenceTemp | double | 25.0 | — | 参考温度(°C) |
| cte | double | 23.6e-6 | >0 | 热膨胀系数 |
| tempStep | double | 0.2 | >0 | 温度步长(°C) |
| tempRangeMin | double | -10.0 | — | 温度范围下限 |
| tempRangeMax | double | 10.0 | — | 温度范围上限 |
| alpha | double | 0.0 | [0,1] | stereoRectify alpha |
| flags | int | 1 | 0/CALIB_ZERO_DISPARITY | stereoRectify flags |
| imageSize | cv::Size | - | >0 | 图像尺寸 |
| deviceId | int | 0 | >=0 | GPU设备ID |
| gridStep | float | 0.5 | >0 | 网格量化步长 |
| depthMin | float | 100.0 | >0 | 最小深度 |
| depthMax | float | 5000.0 | >depthMin | 最大深度 |
| depthSamples | int | 200 | >0 | 深度采样数 |
| epipolarStep | float | 0.5 | >0 | 极线扫描步长 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 温度采样点数 | >0 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常 | 全部温度点成功 |
| Degraded | 降级 | 有温度点失败 |
| Warning | 警告 | 有温度点质量非Normal |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate) | 抛出 std::invalid_argument（注意 flags 默认值 1 不合法） |
| BUILD_CUDA=OFF | 返回 success=false |
| stereoRectify 异常 | 跳过该温度点，记录失败计数 |
| planeMap.Execute 失败 | 记录失败计数，继续处理其余温度点 |
| OpenCV / std 异常 | 捕获并返回 success=false |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟡 中 | 温度步长过小会产生大量映射表条目，耗时和显存占用大 | 长时间运行+显存压力 |
| 🔴 高 | **代码缺陷**：`flags` 默认值为 `1`，但 `validate()` 仅允许 `0` 或 `CALIB_ZERO_DISPARITY`(=1024)，默认参数调用必定抛异常 | 需显式传入 `flags=0` 才能正常运行 |
| 🟢 低 | BUILD_CUDA=OFF时直接返回失败 | 无CUDA环境不可用 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 |
