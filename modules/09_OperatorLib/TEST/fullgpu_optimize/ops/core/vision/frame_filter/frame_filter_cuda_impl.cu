#include "frame_filter_cuda_pimpl.h"
#include "common/calib_logging.h"
#include <opencv2/cudaarithm.hpp>  // cv::cuda::countNonZero
#include <string>

using namespace calib;

CALIB_DEFINE_LOG_TAG(FF, FrameFilterCUDA);  // 标签名待全局统一编号

FrameFilterResult FrameFilterCUDA::Impl::Execute(const cv::cuda::GpuMat& d_cleanedMask,
                                                  cv::cuda::Stream& stream) {
#ifndef NDEBUG
    // Debug 并发断言（§2.2）：单实例非线程安全，防 Execute() 并发调用
    assert(!inProcess_.load() && "Concurrent Execute() calls detected - NOT thread-safe!");
    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() { flag->store(false); }
    };
    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);
#endif

    FrameFilterResult result;

    // 输入校验（非 GPU 操作，不入 try）
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

    try {
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
    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        result.qualityFlag = QualityFlag::Degraded;
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        result.qualityFlag = QualityFlag::Degraded;
    }

    return result;
}
