# 帧类型过滤

> 本算子**已实现**（`modules/09_operatorlib/core/vision/frame_filter/`，7/7 测试绿）。注：主工程编排层尚为骨架桩（`modules/01_calibration/calibration.cpp` 空命名空间），本算子的**编排层接入**（调用 frame_filter、据 isMarkerFrame 销毁激光线帧）待 framework 实现阶段2 姿态判断链时落地。

## A. 标识

| 字段 | 填写内容 |
|------|---------|
| 算子编号 | 姿态-1b（流水线位置：阶段2 姿态判断链，2-1 `mask_extract` 之后、2-2 `ccl` 之前） |
| 中文名称 | 帧类型过滤 |
| 英文目录名 | frame_filter |
| 运行平台 | CUDA (OpenCV CUDA API) |
| 所属流程 | 姿态估计（阶段2 姿态判断链，标记点链入口） |
| 精度档次 | ① 判断/计数类（非几何精度） |

## B. 数据流

| 方向 | 数据描述 | 数据类型 |
|------|---------|---------|
| **输入①** | 清理后掩膜 | GpuMat (CV_8UC1, ← `mask_extract` 的 `d_cleanedMask`) |
| **输出** | 帧类型过滤结果 | `FrameFilterResult` 含 `isMarkerFrame`(bool) + `maskRatio`(float) + `d_cleanedMask` 透传(GpuMat, 不变) |

> **注**：本算子**仅判断、不修改数据**。`d_cleanedMask` 原样透传给下游 `ccl`，算子只额外输出判断结果。是否销毁帧由**编排层**根据 `isMarkerFrame` 决定（算子本身不销毁）。

## C. 算法

**核心原理**：`mask_extract` 是**专为标记点设计**的掩膜提取算子——对标记点帧能提取出有效掩膜（标记点圆斑，面积可观）；对激光线帧，因激光线并非标记点特征，提取出的掩膜面积很小（激光线被过滤掉）。故用掩膜非零像素占比即可区分两类帧。

**核心流程**：

```
Step 1: 前置同步 stream.waitForCompletion() —— 确保上游 mask_extract 在传入 stream 上写完 d_cleanedMask
Step 2: int count = cv::cuda::countNonZero(*d_cleanedMask) —— 同步版（返回 int，类型确定；本算子为同步屏障，详见 G 节）
Step 3: maskRatio = static_cast<double>(count) / (static_cast<double>(rows) * cols) —— double 除法（防整数截断 + 防溢出）
Step 4: 判定 isMarkerFrame = (maskRatio >= maskRatioThreshold)
        - true  → 标记点帧（通过，走后续 2-2~2-13）
        - false → 激光线帧（编排层销毁，不走后续）
Step 5: 透传 d_cleanedMask（原样 shared_ptr，供下游 ccl 使用）
```

**关键 OpenCV/第三方函数**：

| 函数 | 用途 |
|------|------|
| cv::cuda::countNonZero（**同步版** `int countNonZero(src)`） | GPU 统计 GpuMat 非零像素数，返回 int。选同步版：本算子为同步屏障（isMarkerFrame 需 host），异步版 `void(src,dst,stream)` 的 dst 类型文档未明且多余 |
| cv::cuda::Stream::waitForCompletion | 前置同步，确保上游 stream 写操作完成后再读 d_cleanedMask |
| （透传无额外计算） | d_cleanedMask shared_ptr 引用计数延长生命周期，直接传下游 |

## D. 依赖

**上下游算子**：

```
mask_extract (2-1) → 本算子 → ccl (2-2，标记点链)
```

**共享/复用关系**：

| 共享对象 | 说明 |
|---------|------|
| d_cleanedMask | 从 mask_extract 接收，原样透传给 ccl（本算子不修改） |
| 左/右相机实例 | 左右相机各持独立实例 |
| CUDA Stream | 算子内部 GPU 操作共用 |

**公共头文件依赖**：

| 头文件 | 用途 |
|--------|------|
| `common/calib_types.h` | 共享类型（QualityFlag 等） |
| `common/calib_logging.h` | 日志宏（CALIB_LOG_*） |
| `common/calib_warmup_config.h` | 预热配置 WarmupConfig |
| `common/result.h` | 统一 Result 模板（若遵循规范 §5.1 单一错误模型） |

## D2. 衔接

**上游→本算子**：

| 来源 | 传递方式 | 说明 |
|------|---------|------|
| mask_extract (2-1) | GpuMat 引用 (CV_8UC1) | `d_cleanedMask`，标记点链 / 激光链共用入口之后 |

**本算子→下游**：

| 输出字段 | 传递给 | 下游方法 | 传递方式 | 必用/可选 |
|---------|--------|---------|---------|---------|
| d_cleanedMask | ccl (2-2) | RegionAnalyzerCUDA | GpuMat 引用 (CV_8UC1, 透传不变) | 必用 |
| isMarkerFrame | 编排层 | —（控制流，非算子） | 值传递 | 必用（编排层据此销毁激光线帧/放行标记点帧） |
| maskRatio | 编排层 / 日志 | — | 值传递 | 可选（诊断/监控） |

## E. 架构

**文件结构**（已实现）：

```
frame_filter/
├── frame_filter_cuda.h          # 公开接口（Params/Input/Result/Operator 三元组）
├── frame_filter_cuda.cpp        # pImpl 桥接
├── frame_filter_cuda_pimpl.h    # pImpl 实现声明
├── frame_filter_cuda_impl.cu    # CUDA 实现（countNonZero + 占比 + 判定）
└── tests/
    └── test_frame_filter_cuda.cpp   # 标记点帧/激光线帧/边界阈值用例
```

**核心 API**：

| 项目 | 名称 |
|------|------|
| 核心类 | FrameFilterCUDA (pImpl) |
| 核心方法 | Execute() |
| 参数结构体 | FrameFilterParams |
| 结果结构体 | FrameFilterResult（含 isMarkerFrame + maskRatio + d_cleanedMask 透传） |
| 预热方法 | Warmup(rows, cols) / Warmup(WarmupConfig) |
| 参数更新 | SetParams() / GetParams() |
| 日志标签 | 待分配（建议沿用全局统一编号方案） |

**状态模型**（算子规范 §4）：

| 项目 | 说明 |
|------|------|
| 状态类别 | 无状态 |
| 并发策略 | 每实例非线程安全（§1.4），左/右相机多实例并行各自独占 |

**错误模型**（算子规范 §5.1）：单一错误模型 `bool success + QualityFlag qualityFlag + std::string message`。

## J. 环境

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| OpenCV + CUDA | 4.13.0 | opencv_cudaarithm（countNonZero）, opencv_core |
| CUDA Toolkit | 12.6 | sm_75, sm_86, sm_87 |

## F. 参数

| 参数名 | 类型 | 默认值 | 范围 | 说明 | 备注 |
|--------|------|--------|------|------|------|
| maskRatioThreshold | double | **0.0**（安全占位） | [0, 1) | 掩膜非零像素占比阈值；maskRatio < 此值判为激光线帧 | 默认 0.0 → `maskRatio>=0` 恒真 → 不误过滤任何帧（未标定时的安全行为）；标定后 `SetParams(实测阈值)` 启用过滤 |

> **阈值标定方法**（待实现时做）：采集若干标记点帧 + 激光线帧，分别跑 mask_extract，统计两类帧 d_cleanedMask 的非零占比分布，取两类分布之间的分离点作为阈值。

## G. 约束

| 约束类型 | 指标 |
|---------|------|
| 数据不变性 | **仅判断不修改**：d_cleanedMask 原样透传，算子不产生、不破坏掩膜数据 |
| 前提依赖 | 依赖 `mask_extract` 对标记点帧/激光线帧的**差异化输出**（标记点帧掩膜大、激光线帧掩膜小） |
| **同步屏障** | **本算子是同步屏障**：isMarkerFrame 控制 host 流分支（编排层据此销毁/继续），必须同步返回。countNonZero 用同步版，stream 参数仅用于前置 `waitForCompletion`（确保上游数据就绪）——这是 frame_filter 相对其他纯流式算子的特殊性 |
| 线程安全 | 非线程安全，左/右相机各自独立实例 |
| 销毁职责 | 算子只输出 `isMarkerFrame`；**帧销毁由编排层执行**，算子不销毁 |
| GPU缓冲区 | 无需额外 GPU 缓冲（同步版 countNonZero 原地统计，无 d_countBuf_） |
| Warmup | 空实现（无 GPU 资源需预分配，与"无状态"一致） |

## K. 质量

**QualityFlag 语义**：

| 标记 | 含义 | 触发条件 |
|------|------|---------|
| Normal | 正常完成判断 | Execute() 成功（无论 isMarkerFrame 为 true 或 false，都属正常判定，不是错误） |
| Degraded | 降级 | OpenCV/运行时异常 |

> **重要语义**：`isMarkerFrame=false`（激光线帧）**不是错误**，是预期的过滤判定。算子 `success=true`、`qualityFlag=Normal`，编排层据此销毁该帧。

**错误处理模式**：

| 错误类型 | 处理方式 |
|---------|---------|
| 参数校验失败 (validate，如阈值越界) | 抛出 std::invalid_argument |
| 输入为空（nullptr 或空 GpuMat） | 返回 `success=true, isMarkerFrame=false`（空掩膜=无标记点特征=非标记点帧，编排层销毁） |
| 类型非 CV_8UC1 | 返回 success=false |
| 无 CUDA 设备 (getCudaEnabledDeviceCount<=0) | 构造时抛出 std::runtime_error |
| OpenCV/CUDA 异常 | 捕获并返回 Degraded |

## H. 风险

| 严重程度 | 风险描述 | 影响 | 缓解 |
|:--------:|---------|------|------|
| 🟡 中 | **阈值依赖实测标定**：标记点帧与激光线帧的掩膜占比分布若区分度不足或重叠，判断失效（误销毁标记点帧 / 放行激光线帧） | 标定采集阶段误过滤或误放行 | 上线前用真实产线样本标定 maskRatioThreshold；留监控（maskRatio 落点统计） |
| 🟡 中 | **强依赖 mask_extract 的差异化输出**：若 mask_extract 对激光线帧也提取出较大掩膜（如激光线被误判为标记点区域），本算子前提不成立 | 判断失效 | 需与 mask_extract 联调验证两类帧的掩膜分布 |
| 🟢 低 | 仅判断不修改数据 | 无数据破坏风险 | — |

## I. 状态

| 项目 | 说明 |
|------|------|
| **判定** | **已实现**（7/7 测试绿） |

- 算子代码已落地：`modules/09_operatorlib/core/vision/frame_filter/`（5 文件 + 7 测试）。
- **待办（非算子代码）**：①`maskRatioThreshold` 实测标定（需真实标记点帧/激光线帧跑 mask_extract 统计占比分布）；②编排层接入（framework 实现阶段2 姿态判断链时，调用 frame_filter 并据 isMarkerFrame 销毁激光线帧）；③与 `mask_extract` 联调确认两类帧掩膜分布区分度；④kLogTag 编号全局统一。
- 参见：`docs/流水线/客户端标定流水线.md` 1.1 节流程图（2-1b 框）+ 二节算子表 2-1b 行。
