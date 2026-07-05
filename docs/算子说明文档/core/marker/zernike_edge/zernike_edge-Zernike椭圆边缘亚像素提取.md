# Zernike椭圆边缘亚像素提取

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-4 |
| 中文名称 | Zernike椭圆边缘亚像素提取 |
| 英文目录名 | zernike_edge |
| 运行平台 | CPU |
| 所属流程 | 姿态估计流程 |
| 精度档次 | ① 亚像素级（~0.1-0.2 pixel） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 灰度子图 | cv::Mat (CV_8UC1) |
| **输出** | 边缘亚像素提取结果 | ZernikeEdgeCPUResult (edgePoints: vector\<EdgePoint\>, edgeCount, cannyEdgeImage) |

## C. 算法

**核心流程**：

```
Step 1: 高斯模糊（GaussianBlur）降噪
Step 2: Canny 边缘检测提取边缘轮廓（内部使用 Sobel 算子，孔径大小由 sobelApertureSize 配置）
Step 3: findNonZero 提取边缘像素坐标
Step 4: 对每个边缘像素计算 Zernike 矩（T11_re, T11_im, T20）
Step 5: 计算 Z11 幅值和角度
Step 6: 幅值阈值过滤弱边缘（mag < edgeStrengthThreshold 则跳过）
Step 7: 亚像素偏移 l = Z20 / |Z11|, subX = px + l * cos(angle)
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::GaussianBlur | 高斯降噪 |
| cv::Canny | 边缘检测 |
| cv::findNonZero | 提取边缘像素位置 |
| 自定义 Zernike 矩卷积 | 亚像素定位核心计算 |

## D. 依赖

**上下游算子**：

```
image_split → [zernike_edge] → image_merge
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| 无特殊共享 | 纯 CPU 逐像素计算 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| image_split | splitImages[i] | 单个子图直接输入 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| edgePoints (vector\<EdgePoint\>) | image_merge | 多组边缘点 | 与对应 ROI 一起传递 | 必用 |

## E. 架构

**文件结构**：

```
zernike_edge/
├── zernike_edge_cpu.h
├── zernike_edge_cpu.cpp
└── tests/
    └── test_zernike_edge_cpu.cpp
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | ZernikeEdgeCPU |
| 核心方法 | Execute() |
| 参数结构体 | ZernikeEdgeCPUParams |
| 结果结构体 | ZernikeEdgeCPUResult |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | "04-ZernikeEdgeCPU" |

**EdgePoint 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| x | double | 亚像素 x 坐标 |
| y | double | 亚像素 y 坐标 |
| angle | double | 边缘法线方向角（弧度） |
| amplitude | double | 边缘响应幅值 |
| pixelX | int | 像素 x 坐标 |
| pixelY | int | 像素 y 坐标 |

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV (core, imgproc) | >= 4.x | 图像处理 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 |
|--------|------|--------|------|------|
| cannyLowThreshold | double | 50 | > 0 | Canny 低阈值 |
| cannyHighThreshold | double | 150 | > cannyLow | Canny 高阈值 |
| gaussianKernelSize | int | 3 | 正奇数 | 高斯核大小 |
| gaussianSigma | double | 1.0 | > 0 | 高斯核标准差 |
| templateSize | int | 5 | 5 或 7 | Zernike 模板尺寸 |
| edgeStrengthThreshold | double | 20 | >= 0 | 边缘强度阈值 |
| sobelApertureSize | int | 3 | 3/5/7 | Sobel 算子孔径 |

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 精度要求 | 亚像素级精度，依赖 Zernike 矩计算 |
| 边界像素 | 距图像边缘不足 templateSize/2 的像素采用零填充处理（不跳过） |
| 最小尺寸 | 图像最小边 ≥ templateSize |
| 除零保护 | 幅值计算 l = Z20/\|Z11\| 时使用 1e-10 epsilon 防止除零 |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | edgeCount > 0 | 正常 |
| Degraded | 部分边缘被幅值阈值过滤 | 边缘点数量减少 |
| Warning | edgeCount == 0 | 无有效边缘 |

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 输入子图为空 | 返回 success=false（公共接口拦截） |
| 输入类型非 CV_8UC1 | 返回 success=false |
| 图像尺寸 < templateSize | 返回 success=false |
| 无边缘像素 | 返回 Warning |

## H. 风险

| 严重程度 | 风险描述 | 影响 |
|:--------:|---------|------|
| 🟢 低 | 计算量随边缘像素数线性增长 | 子图尺寸有限，开销可控 |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | 可直接使用 — 核心路径已覆盖 |
