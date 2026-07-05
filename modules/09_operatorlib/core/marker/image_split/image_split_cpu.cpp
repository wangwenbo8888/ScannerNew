/**
 * @file image_split_cpu.cpp
 * @brief 标记点图像分割算子 - 实现文件
 *
 * CPU 算子无需 pImpl 隔离（无 CUDA 类型），但为与框架风格统一仍保留 Impl 结构。
 * 所有核心逻辑直接使用 OpenCV CPU API。
 */

#include "image_split_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <cassert>

using namespace calib;

OperatorInfo getImageSplitCPUInfo() {
    return OperatorInfo{"ImageSplitCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(08, ImageSplitCPU);

struct ImageSplitCPU::Impl {
    ImageSplitCPUParams params_;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const ImageSplitCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    ~Impl() = default;

    ImageSplitCPUResult Execute(const cv::Mat& srcImage,
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

        ImageSplitCPUResult result;

        if (srcImage.empty() || roiRects.empty()) {
            result.success = true;
            result.splitCount = 0;
            if (srcImage.empty()) {
                result.message = "Input image is empty, no regions to split";
                result.qualityFlag = calib::QualityFlag::Warning;
            } else {
                result.message = "ROI list is empty, no regions to split";
                result.qualityFlag = calib::QualityFlag::Warning;
            }
            return result;
        }

        if (srcImage.type() != CV_8UC1) {
            result.success = false;
            result.message = "Input must be CV_8UC1 grayscale image";
            return result;
        }

        result.splitImages.reserve(roiRects.size());

        const cv::Rect imgBounds(0, 0, srcImage.cols, srcImage.rows);
        bool hasClipped = false;

        for (const auto& rect : roiRects) {
            cv::Rect validRect = rect;

            if (params_.enableBoundaryCheck) {
                validRect = rect & imgBounds;

                if (validRect.width <= 0 || validRect.height <= 0) {
                    continue;
                }

                if (validRect != rect) {
                    hasClipped = true;
                }
            } else {
                if (validRect.x < 0 || validRect.y < 0 ||
                    validRect.x + validRect.width > srcImage.cols ||
                    validRect.y + validRect.height > srcImage.rows ||
                    validRect.width <= 0 || validRect.height <= 0) {
                    continue;
                }
            }

            result.splitImages.emplace_back(srcImage(validRect).clone());
        }

        result.splitCount = static_cast<int>(result.splitImages.size());
        result.success = true;

        if (result.splitCount == 0) {
            result.message = "All ROI regions are invalid or out of bounds";
            result.qualityFlag = calib::QualityFlag::Warning;
        } else if (hasClipped) {
            result.message = "Some ROI regions were clipped to image bounds";
            result.qualityFlag = calib::QualityFlag::Degraded;
        } else {
            result.message = "Split successful";
            result.qualityFlag = calib::QualityFlag::Normal;
        }

        return result;
    }

    void Warmup(int rows, int cols)
    {
        cv::Mat dummy = cv::Mat::zeros(rows, cols, CV_8UC1);
        cv::Rect testRect(0, 0, std::min(cols, 1), std::min(rows, 1));
        if (testRect.width > 0 && testRect.height > 0) {
            cv::Mat roi = dummy(testRect).clone();
            (void)roi;
        }
        warmup_rows_ = rows;
        warmup_cols_ = cols;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: rows={}, cols={}", rows, cols);
    }

    void SetParams(const ImageSplitCPUParams& params)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() called while Execute() is running - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const ImageSplitCPUParams& GetParams() const { return params_; }
};

// ============================================================
// 构造函数
// ============================================================
ImageSplitCPU::ImageSplitCPU(const ImageSplitCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("ImageSplitCPU initialized: enableBoundaryCheck={}",
                   params.enableBoundaryCheck);
}

// ============================================================
// 析构函数
// ============================================================
ImageSplitCPU::~ImageSplitCPU() = default;

void ImageSplitCPU::Destroy() { }

// ============================================================
// Execute() - 核心接口
// ============================================================
ImageSplitCPUResult ImageSplitCPU::Execute(const cv::Mat& srcImage,
                                         const std::vector<cv::Rect>& roiRects)
{
    CALIB_LOG_DEBUG("Execute() called: image size={}x{}, roiCount={}",
                    srcImage.cols, srcImage.rows, roiRects.size());

    if (srcImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input image is empty");
        ImageSplitCPUResult result;
        result.success = false;
        result.message = "Input image is empty";
        return result;
    }

    if (srcImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input must be CV_8UC1, got type {}", srcImage.type());
        ImageSplitCPUResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 grayscale image";
        return result;
    }

    return pImpl_->Execute(srcImage, roiRects);
}

// ============================================================
// warmup(int, int)
// ============================================================
void ImageSplitCPU::Warmup(int rows, int cols)
{
    CALIB_LOG_INFO("warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
}

// ============================================================
// warmup(WarmupConfig)
// ============================================================
void ImageSplitCPU::Warmup(const calib::WarmupConfig& config)
{
    CALIB_LOG_INFO("warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

// ============================================================
// setParams()
// ============================================================
void ImageSplitCPU::SetParams(const ImageSplitCPUParams& params)
{
    CALIB_LOG_INFO("setParams() called: enableBoundaryCheck={}", params.enableBoundaryCheck);
    pImpl_->SetParams(params);
}

// ============================================================
// GetParams()
// ============================================================
const ImageSplitCPUParams& ImageSplitCPU::GetParams() const
{
    return pImpl_->GetParams();
}
