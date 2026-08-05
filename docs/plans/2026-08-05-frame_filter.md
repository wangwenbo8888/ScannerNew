# frame_filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 frame_filter（姿态判断链 2-1b）算子——在 mask_extract 后、ccl 前，按掩膜非零像素占比过滤激光线帧。

**Architecture:** CUDA 算子，pImpl 模式隔离 CUDA 类型；三元组（Params/Result/Operator）；同步版 `cv::cuda::countNonZero` 统计非零像素；d_cleanedMask 透传不修改；isMarkerFrame 输出给编排层决定销毁。**本算子是同步屏障**（isMarkerFrame 控制 host 流分支）。

**Tech Stack:** C++17 / CUDA 17 / OpenCV CUDA 4.13（cudaarithm）/ nlohmann_json / GoogleTest / sm_75/86/87

---

## 上下文（执行前必读）

- **权威设计**：`docs/算子说明文档/core/vision/frame_filter/frame_filter-帧类型过滤.md`（A–K，含定稿同步版算法、同步屏障约束、阈值默认 0.0）
- **实现模板**：`modules/09_operatorlib/core/vision/mask_extract/`（三元组 + pImpl + BUILD_CUDA 分支模式，照搬结构）
- **下游接口参考**：`ccl/region_analyze_cuda.h` 的 `Execute(const shared_ptr<GpuMat>&, Stream&)` 重载模式
- **CMake**：`modules/09_operatorlib/CMakeLists.txt` 已 `GLOB_RECURSE core/*.cpp *.cu *.h`，**新文件自动收录**，无需改 CMake；但新增文件后需重新 cmake configure
- **构建命令**（Debug，现有 build/）：
  ```powershell
  cmake --build build --config Debug --target mod_operatorlib
  $env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
  ctest --test-dir build -C Debug --output-on-failure -R frame_filter
  ```
- **命名空间**：`calib`（与所有算子一致）

---

## Task 1: 三元组头文件 frame_filter_cuda.h

**Files:**
- Create: `modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda.h`

**Step 1: 写头文件**（照搬 mask_extract_cuda.h 结构，前向声明 GpuMat/Stream 隔离 CUDA）

```cpp
#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

namespace cv { namespace cuda { class GpuMat; class Stream; } }
struct WarmupConfig;

struct FrameFilterParams {
    double maskRatioThreshold = 0.0;   ///< 安全占位（>=0 恒真，不过滤）
    void validate() const {
        if (maskRatioThreshold < 0.0 || maskRatioThreshold >= 1.0)
            throw std::invalid_argument("FrameFilterParams::maskRatioThreshold must be [0, 1)");
    }
    nlohmann::json toJson() const { return {{"maskRatioThreshold", maskRatioThreshold}}; }
    static FrameFilterParams fromJson(const nlohmann::json& j) {
        FrameFilterParams p;
        if (j.contains("maskRatioThreshold")) p.maskRatioThreshold = j.at("maskRatioThreshold").get<double>();
        p.validate();
        return p;
    }
};

struct FrameFilterResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    bool isMarkerFrame = false;          ///< true=标记点帧(通过)，false=激光线帧(编排层销毁)
    double maskRatio = 0.0;              ///< 掩膜非零像素占比 [0,1]
    std::shared_ptr<cv::cuda::GpuMat> d_cleanedMask;  ///< 透传(不变)

    FrameFilterResult() = default;
    ~FrameFilterResult() = default;
    FrameFilterResult(FrameFilterResult&&) = default;
    FrameFilterResult& operator=(FrameFilterResult&&) = default;
    FrameFilterResult(const FrameFilterResult&) = delete;
    FrameFilterResult& operator=(const FrameFilterResult&) = delete;
};

// ===== 算子规范 §4 状态模型：无状态；同步屏障(详见算子文档 G 节) =====
class SCANNER_API FrameFilterCUDA {
public:
    static constexpr const char* kLogTag = "FF-FrameFilterCUDA";  // 编号待全局统一
    explicit FrameFilterCUDA(const FrameFilterParams& params = {});
    ~FrameFilterCUDA();
    FrameFilterCUDA(const FrameFilterCUDA&) = delete;
    FrameFilterCUDA& operator=(const FrameFilterCUDA&) = delete;

    void Destroy();

    FrameFilterResult Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_cleanedMask,
                              cv::cuda::Stream& stream);
    FrameFilterResult Execute(const cv::cuda::GpuMat& d_cleanedMask,
                              cv::cuda::Stream& stream);
    FrameFilterResult Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_cleanedMask);
    FrameFilterResult Execute(const cv::cuda::GpuMat& d_cleanedMask);

    void Warmup(int rows, int cols);        // 空实现(无 GPU 缓冲)
    void Warmup(const WarmupConfig& config);
    void SetParams(const FrameFilterParams& params);
    const FrameFilterParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getFrameFilterCUDAInfo();

} // namespace calib
```

**Step 2: Commit**
```bash
git add modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda.h
git commit -m "feat(frame_filter): add triplet header (Params/Result/Operator)"
```

---

## Task 2: pImpl 声明 frame_filter_cuda_pimpl.h

**Files:**
- Create: `modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda_pimpl.h`

**Step 1: 写 pImpl 声明**（Impl 极薄——无 GPU 缓冲，只持 params + Debug 断言）

```cpp
#pragma once

#include <opencv2/core/cuda.hpp>
#include <memory>
#include <atomic>
#ifndef NDEBUG
#include <cuda_runtime.h>
#endif

#include "frame_filter_cuda.h"

namespace calib {

struct FrameFilterCUDA::Impl {
    FrameFilterParams params_;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};  // 线程安全断言(§2.2)
#endif

    explicit Impl(const FrameFilterParams& params) : params_(params) { params_.validate(); }
    ~Impl() = default;

    FrameFilterResult Execute(const cv::cuda::GpuMat& d_cleanedMask, cv::cuda::Stream& stream);
    void SetParams(const FrameFilterParams& params) { params_ = params; params_.validate(); }
    void Warmup(int /*rows*/, int /*cols*/) { /* 空：无 GPU 缓冲需预分配 */ }
    const FrameFilterParams& GetParams() const { return params_; }
};

} // namespace calib
```

**Step 2: Commit**
```bash
git add modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda_pimpl.h
git commit -m "feat(frame_filter): add pImpl declaration"
```

---

## Task 3: 测试 test_frame_filter_cuda.cpp（TDD - 先写测试）

**Files:**
- Create: `modules/09_operatorlib/core/vision/frame_filter/tests/test_frame_filter_cuda.cpp`

**Step 1: 写测试**（合成 GpuMat 已知占比，验证判定/透传/边界/错误）

```cpp
#include "frame_filter_cuda.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>

using calib::FrameFilterCUDA;
using calib::FrameFilterParams;

// 高占比 → 标记点帧通过
TEST(FrameFilterCUDA, MarkerFrameHighRatio) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Rect(10, 10, 50, 50), 255, cv::FILLED);  // 2500 非零 → 0.25
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff(FrameFilterParams{0.1});
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.isMarkerFrame);
    EXPECT_NEAR(r.maskRatio, 0.25, 0.002);
}

// 低占比 → 激光线帧
TEST(FrameFilterCUDA, LaserLineFrameLowRatio) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::line(mask, {0, 50}, {99, 50}, 255, 1);  // ~100 非零 → 0.01
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff(FrameFilterParams{0.05});
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.isMarkerFrame);
}

// shared_ptr 透传：Result.d_cleanedMask 与传入同一对象
TEST(FrameFilterCUDA, PassthroughShared) {
    auto d_mask = std::make_shared<cv::cuda::GpuMat>(100, 100, CV_8UC1);
    d_mask->setTo(255);
    FrameFilterCUDA ff;
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.d_cleanedMask.get(), d_mask.get());
}

// 默认阈值 0.0 → 全部通过(含全黑帧)
TEST(FrameFilterCUDA, DefaultThresholdPassAll) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);  // 全黑，占比 0
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff;  // 默认 0.0
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.isMarkerFrame);  // 0 >= 0.0
}

// 错误类型 → success=false
TEST(FrameFilterCUDA, WrongTypeFails) {
    cv::cuda::GpuMat d_mask(100, 100, CV_32FC1);
    FrameFilterCUDA ff;
    auto r = ff.Execute(d_mask);
    EXPECT_FALSE(r.success);
}
```

**Step 2: Commit**
```bash
git add modules/09_operatorlib/core/vision/frame_filter/tests/test_frame_filter_cuda.cpp
git commit -m "test(frame_filter): add TDD tests (ratio判定/透传/边界/错误)"
```

> 此时测试无法编译链接（FrameFilterCUDA::Execute 未实现）—— 这是 TDD 的"红"，下两 Task 让它变绿。

---

## Task 4: pImpl 桥接 frame_filter_cuda.cpp

**Files:**
- Create: `modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda.cpp`

**Step 1: 写桥接**（照搬 mask_extract_cuda.cpp 结构：BUILD_CUDA 分支 + OFF 兜底 throw）

```cpp
#include "frame_filter_cuda.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getFrameFilterCUDAInfo() {
    return OperatorInfo{"FrameFilterCUDA", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(FF, FrameFilterCUDA);  // 标签名待全局统一编号

#if BUILD_CUDA

#include "frame_filter_cuda_pimpl.h"

FrameFilterCUDA::FrameFilterCUDA(const FrameFilterParams& params)
    : pImpl_(std::make_unique<Impl>(params)) {
    CALIB_LOG_INFO("FrameFilterCUDA initialized: maskRatioThreshold={}", params.maskRatioThreshold);
}

FrameFilterCUDA::~FrameFilterCUDA() = default;

FrameFilterResult FrameFilterCUDA::Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_cleanedMask,
                                           cv::cuda::Stream& stream) {
    if (!d_cleanedMask || d_cleanedMask->empty()) {
        FrameFilterResult r;
        r.success = true;
        r.message = "empty input, treated as non-marker";
        r.isMarkerFrame = false;
        return r;
    }
    FrameFilterResult r = pImpl_->Execute(*d_cleanedMask, stream);
    r.d_cleanedMask = d_cleanedMask;  // 透传 shared_ptr
    return r;
}

FrameFilterResult FrameFilterCUDA::Execute(const cv::cuda::GpuMat& d_cleanedMask,
                                           cv::cuda::Stream& stream) {
    return pImpl_->Execute(d_cleanedMask, stream);
}

FrameFilterResult FrameFilterCUDA::Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_cleanedMask) {
    cv::cuda::Stream stream;
    return Execute(d_cleanedMask, stream);
}

FrameFilterResult FrameFilterCUDA::Execute(const cv::cuda::GpuMat& d_cleanedMask) {
    cv::cuda::Stream stream;
    return Execute(d_cleanedMask, stream);
}

void FrameFilterCUDA::Destroy() {}

void FrameFilterCUDA::Warmup(int rows, int cols) { pImpl_->Warmup(rows, cols); }
void FrameFilterCUDA::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.maxPointCount > 0 ? config.maxPointCount : 10000, 10000);
}
void FrameFilterCUDA::SetParams(const FrameFilterParams& params) { pImpl_->SetParams(params); }
const FrameFilterParams& FrameFilterCUDA::GetParams() const { return pImpl_->GetParams(); }

#else  // BUILD_CUDA=OFF

struct FrameFilterCUDA::Impl {};
FrameFilterCUDA::FrameFilterCUDA(const FrameFilterParams& params) : pImpl_(std::make_unique<Impl>()) {
    params.validate();
    CALIB_LOG_WARN("FrameFilterCUDA: BUILD_CUDA=OFF, all ops will throw");
}
FrameFilterCUDA::~FrameFilterCUDA() = default;
void FrameFilterCUDA::Destroy() {}
FrameFilterResult FrameFilterCUDA::Execute(const std::shared_ptr<cv::cuda::GpuMat>&, cv::cuda::Stream&) {
    throw std::runtime_error("[FF-FrameFilterCUDA] CUDA not available (BUILD_CUDA=OFF)");
}
FrameFilterResult FrameFilterCUDA::Execute(const cv::cuda::GpuMat&, cv::cuda::Stream&) {
    throw std::runtime_error("[FF-FrameFilterCUDA] CUDA not available (BUILD_CUDA=OFF)");
}
FrameFilterResult FrameFilterCUDA::Execute(const std::shared_ptr<cv::cuda::GpuMat>&) {
    throw std::runtime_error("[FF-FrameFilterCUDA] CUDA not available (BUILD_CUDA=OFF)");
}
FrameFilterResult FrameFilterCUDA::Execute(const cv::cuda::GpuMat&) {
    throw std::runtime_error("[FF-FrameFilterCUDA] CUDA not available (BUILD_CUDA=OFF)");
}
void FrameFilterCUDA::Warmup(int, int) { throw std::runtime_error("[FF-FrameFilterCUDA] CUDA OFF"); }
void FrameFilterCUDA::Warmup(const calib::WarmupConfig&) { throw std::runtime_error("[FF-FrameFilterCUDA] CUDA OFF"); }
void FrameFilterCUDA::SetParams(const FrameFilterParams&) { throw std::runtime_error("[FF-FrameFilterCUDA] CUDA OFF"); }
const FrameFilterParams& FrameFilterCUDA::GetParams() const { throw std::runtime_error("[FF-FrameFilterCUDA] CUDA OFF"); }

#endif
```

**Step 2: Commit**
```bash
git add modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda.cpp
git commit -m "feat(frame_filter): add pImpl bridge (BUILD_CUDA branches)"
```

---

## Task 5: CUDA 实现 frame_filter_cuda_impl.cu（TDD - 让测试绿）

**Files:**
- Create: `modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda_impl.cu`

**Step 1: 写 CUDA 实现**（同步版 countNonZero + static_cast + waitForCompletion）

```cpp
#include "frame_filter_cuda_pimpl.h"
#include "common/calib_logging.h"
#include <opencv2/cudaarithm.hpp>  // cv::cuda::countNonZero

namespace calib {

FrameFilterResult FrameFilterCUDA::Impl::Execute(const cv::cuda::GpuMat& d_cleanedMask,
                                                  cv::cuda::Stream& stream) {
    FrameFilterResult result;

    // 输入校验
    if (d_cleanedMask.empty()) {
        result.success = true;
        result.message = "empty input, treated as non-marker";
        result.isMarkerFrame = false;
        return result;
    }
    if (d_cleanedMask.type() != CV_8UC1) {
        CALIB_LOG_ERROR("d_cleanedMask type={} must be CV_8UC1", d_cleanedMask.type());
        result.success = false;
        result.message = "d_cleanedMask must be CV_8UC1";
        return result;
    }

    // 同步屏障：等上游 stream 写完 d_cleanedMask
    stream.waitForCompletion();

    // 同步版 countNonZero（返回 int，类型确定；异步版 dst 类型文档未明且多余）
    int count = cv::cuda::countNonZero(d_cleanedMask);

    // double 除法（防整数截断 + 防 rows*cols 溢出）
    double maskRatio = static_cast<double>(count)
                     / (static_cast<double>(d_cleanedMask.rows) * d_cleanedMask.cols);

    result.maskRatio = maskRatio;
    result.isMarkerFrame = (maskRatio >= params_.maskRatioThreshold);
    result.success = true;
    result.qualityFlag = QualityFlag::Normal;  // 无论 isMarkerFrame true/false 都属正常判定

    CALIB_LOG_DEBUG("countNonZero={} maskRatio={:.6f} isMarker={} threshold={}",
                    count, maskRatio, result.isMarkerFrame, params_.maskRatioThreshold);
    return result;
}

} // namespace calib
```

**Step 2: Commit**
```bash
git add modules/09_operatorlib/core/vision/frame_filter/frame_filter_cuda_impl.cu
git commit -m "feat(frame_filter): add CUDA impl (sync countNonZero + waitForCompletion)"
```

---

## Task 6: 编译 + 测试（TDD 绿）

**Step 1: 重新 cmake configure（让 GLOB 收录新文件）**
```powershell
cmake -S . -B build  # 重新检测 glob 文件（无需完整重配，仅刷新 glob）
```

**Step 2: 编译 mod_operatorlib**
```powershell
cmake --build build --config Debug --target mod_operatorlib
```
Expected: 编译成功（frame_filter 5 文件 + 测试）

**Step 3: 运行测试**
```powershell
$env:PATH = 'C:\opencv-cuda-4.13.0-debug\x64\vc17\bin;' + $env:PATH
ctest --test-dir build -C Debug --output-on-failure -R frame_filter
```
Expected: **5/5 PASS**（MarkerFrameHighRatio / LaserLineFrameLowRatio / PassthroughShared / DefaultThresholdPassAll / WrongTypeFails）

**Step 4: 若失败**——按算子说明文档 G 节核对（同步屏障语义、static_cast、透传）；常见问题：CALIB_DEFINE_LOG_TAG 宏签名、OperatorType::CUDA 枚举名（对照 mask_extract 确认）。

**Step 5: Commit（测试通过）**
```bash
git add -A
git commit -m "test(frame_filter): all 5 tests green"
```

---

## Task 7: 更新算子目录索引（可选收尾）

**Files:**
- Modify: `docs/算子说明文档/算子目录.md`（core/vision 下补 frame_filter）
- Modify: `工程目录地图.md`（算子说明文档数 44→45，core/vision 补 frame_filter）

**Step 1: 算子目录.md 补 frame_filter 行**（core/vision 分支下，ccl/mask_extract 旁）

**Step 2: 工程目录地图.md** 算子说明文档计数 44→45

**Step 3: Commit**
```bash
git add docs/算子说明文档/算子目录.md 工程目录地图.md
git commit -m "docs(frame_filter): update index & dirmap (44→45)"
```

---

## 验收标准

- [ ] 5 文件齐全：`frame_filter_cuda.h` / `.cpp` / `_pimpl.h` / `_impl.cu` / `tests/test_*.cpp`
- [ ] `mod_operatorlib` 编译通过（Debug）
- [ ] ctest `-R frame_filter` 5/5 绿
- [ ] d_cleanedMask 透传不变（PassthroughShared 测试保证）
- [ ] maskRatio 用 double 除法（无整数截断）
- [ ] 同步版 countNonZero（非异步版）
- [ ] 阈值默认 0.0（未标定不误过滤）

## 后续（非本计划范围）

- `maskRatioThreshold` 实测标定（需真实标记点帧/激光线帧跑 mask_extract，统计占比分布）
- 编排层接入（framework/workflow 实现阶段2 姿态判断链时，调用 frame_filter 并据 isMarkerFrame 销毁激光线帧）
- kLogTag 编号全局统一
