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
