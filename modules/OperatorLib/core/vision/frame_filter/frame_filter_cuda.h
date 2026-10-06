#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

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
