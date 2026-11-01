/**
 * @file image_merge_cpu.cpp
 * @brief 标记点图像合并算子 - 实现文件
 *
 * CPU 算子无需 pImpl 隔离（无 CUDA 类型），但为与框架风格统一仍保留 Impl 结构。
 * 所有核心逻辑为纯坐标偏移运算。
 */

#include "image_merge_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <atomic>
#include <cassert>

using namespace calib;

OperatorInfo getImageMergeCPUInfo() {
    return OperatorInfo{"ImageMergeCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(09, ImageMergeCPU);

struct ImageMergeCPU::Impl {
    ImageMergeCPUParams params_;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const ImageMergeCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    ~Impl() = default;

    ImageMergeCPUResult Execute(const std::vector<std::vector<EdgePoint>>& edgePointsPerSubImage,
                               const std::vector<cv::Rect>& roiRects)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent Execute() calls detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        ImageMergeCPUResult result;

        if (edgePointsPerSubImage.empty() || roiRects.empty()) {
            result.success = true;
            result.mergedEdgeCount = 0;
            if (edgePointsPerSubImage.empty()) {
                result.message = "Edge points list is empty, nothing to merge";
                result.qualityFlag = calib::QualityFlag::Warning;
            } else {
                result.message = "ROI list is empty, nothing to merge";
                result.qualityFlag = calib::QualityFlag::Warning;
            }
            return result;
        }

        if (edgePointsPerSubImage.size() != roiRects.size()) {
            result.success = false;
            result.message = "Edge points count does not match ROI count";
            return result;
        }

        size_t totalPoints = 0;
        for (const auto& pts : edgePointsPerSubImage) {
            totalPoints += pts.size();
        }

        result.mergedEdgePoints.reserve(totalPoints);
        result.groupIds.reserve(totalPoints);

        for (size_t i = 0; i < edgePointsPerSubImage.size(); ++i) {
            const auto& pts = edgePointsPerSubImage[i];
            const auto& roi = roiRects[i];

            for (const auto& ep : pts) {
                EdgePoint merged;
                merged.x = ep.x + roi.x;
                merged.y = ep.y + roi.y;
                merged.angle = ep.angle;
                merged.amplitude = ep.amplitude;
                merged.pixelX = ep.pixelX + roi.x;
                merged.pixelY = ep.pixelY + roi.y;

                result.mergedEdgePoints.push_back(merged);
                result.groupIds.push_back(static_cast<int>(i));
            }
        }

        result.mergedEdgeCount = static_cast<int>(result.mergedEdgePoints.size());
        result.groupCount = static_cast<int>(edgePointsPerSubImage.size());
        result.success = true;

        if (result.mergedEdgeCount == 0) {
            result.message = "All sub-images have no edge points, merged result is empty";
            result.qualityFlag = calib::QualityFlag::Warning;
        } else {
            result.message = "Merge successful";
            result.qualityFlag = calib::QualityFlag::Normal;
        }

        return result;
    }

    void Warmup(int rows, int cols)
    {
        warmup_rows_ = rows;
        warmup_cols_ = cols;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: rows={}, cols={}", rows, cols);
    }

    void SetParams(const ImageMergeCPUParams& params)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() called while Execute() is running - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const ImageMergeCPUParams& GetParams() const { return params_; }
};

// ============================================================
// 构造函数
// ============================================================
ImageMergeCPU::ImageMergeCPU(const ImageMergeCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("ImageMergeCPU initialized");
}

// ============================================================
// 析构函数
// ============================================================
ImageMergeCPU::~ImageMergeCPU() = default;

void ImageMergeCPU::Destroy() { }

// ============================================================
// Execute() - 核心接口
// ============================================================
ImageMergeCPUResult ImageMergeCPU::Execute(const std::vector<std::vector<EdgePoint>>& edgePointsPerSubImage,
                                          const std::vector<cv::Rect>& roiRects)
{
    CALIB_LOG_DEBUG("Execute() called: subImageCount={}, roiCount={}",
                    edgePointsPerSubImage.size(), roiRects.size());

    if (edgePointsPerSubImage.size() != roiRects.size()) {
        CALIB_LOG_ERROR("Execute() failed: edgePoints count ({}) != roi count ({})",
                        edgePointsPerSubImage.size(), roiRects.size());
        ImageMergeCPUResult result;
        result.success = false;
        result.message = "Edge points count does not match ROI count";
        return result;
    }

    return pImpl_->Execute(edgePointsPerSubImage, roiRects);
}

// ============================================================
// warmup(int, int)
// ============================================================
void ImageMergeCPU::Warmup(int rows, int cols)
{
    CALIB_LOG_INFO("warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
}

// ============================================================
// warmup(WarmupConfig)
// ============================================================
void ImageMergeCPU::Warmup(const calib::WarmupConfig& config)
{
    CALIB_LOG_INFO("warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

// ============================================================
// setParams()
// ============================================================
void ImageMergeCPU::SetParams(const ImageMergeCPUParams& params)
{
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

// ============================================================
// GetParams()
// ============================================================
const ImageMergeCPUParams& ImageMergeCPU::GetParams() const
{
    return pImpl_->GetParams();
}
