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
    if (!d_cleanedMask) {  // nullptr 检查（空 GpuMat 由 Impl 层统一处理，避免重复）
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

// [算子规范 §1.6 豁免] 无 stream 重载仅供测试/调试；defaultStream 为 thread_local，每线程独立，
// 无跨实例/跨线程共享，不构成可变共享状态（与 mask_extract 一致）
FrameFilterResult FrameFilterCUDA::Execute(const std::shared_ptr<cv::cuda::GpuMat>& d_cleanedMask) {
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(d_cleanedMask, defaultStream);
}

FrameFilterResult FrameFilterCUDA::Execute(const cv::cuda::GpuMat& d_cleanedMask) {
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(d_cleanedMask, defaultStream);
}

void FrameFilterCUDA::Destroy() {}

void FrameFilterCUDA::Warmup(int rows, int cols) { pImpl_->Warmup(rows, cols); }
void FrameFilterCUDA::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.rows, config.cols);
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
