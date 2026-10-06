/**
 * @file zernike_edge_cpu.cpp
 * @brief Zernike椭圆边缘亚像素提取算子 - 实现文件
 *
 * CPU 算子无需 pImpl 隔离（无 CUDA 类型），但为与框架风格统一仍保留 Impl 结构。
 * 算法流水线：GaussianBlur → Canny → 边缘像素遍历 → Zernike 矩卷积 → 亚像素定位
 */

#include "zernike_edge_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <cassert>
#include <array>
#include <stdexcept>
#include <cmath>

using namespace calib;

OperatorInfo getZernikeEdgeCPUInfo() {
    return OperatorInfo{"ZernikeEdgeCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(04, ZernikeEdgeCPU);

namespace {

struct ZernikeTemplates {
    int size;
    std::vector<double> T00;
    std::vector<double> T11_re;
    std::vector<double> T11_im;
    std::vector<double> T20;

    static ZernikeTemplates create(int templateSize) {
        ZernikeTemplates t;
        t.size = templateSize;
        int N = templateSize;
        int half = N / 2;
        int count = N * N;

        t.T00.resize(count);
        t.T11_re.resize(count);
        t.T11_im.resize(count);
        t.T20.resize(count);

        double normRadius = static_cast<double>(half);

        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx) {
                int idx = (dy + half) * N + (dx + half);
                double x = static_cast<double>(dx) / normRadius;
                double y = static_cast<double>(dy) / normRadius;
                double r2 = x * x + y * y;

                if (r2 > 1.0) {
                    t.T00[idx] = 0.0;
                    t.T11_re[idx] = 0.0;
                    t.T11_im[idx] = 0.0;
                    t.T20[idx] = 0.0;
                    continue;
                }

                double r = std::sqrt(r2);
                double theta = std::atan2(y, x);
                double w = (r2 <= 1.0) ? 1.0 : 0.0;

                double pi = 3.14159265358979323846;

                t.T00[idx] = w * 1.0 / pi;

                t.T11_re[idx] = w * (2.0 / pi) * r * std::cos(theta);

                t.T11_im[idx] = w * (2.0 / pi) * r * std::sin(theta);

                t.T20[idx] = w * (std::sqrt(6.0) / pi) * (2.0 * r2 - 1.0);
            }
        }

        return t;
    }
};

double computeZernikeMoment(const cv::Mat& blurred, int cx, int cy,
                            const std::vector<double>& templateData, int templateSize)
{
    int half = templateSize / 2;
    double moment = 0.0;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int iy = cy + dy;
            int ix = cx + dx;
            double pixel = 0.0;
            if (iy >= 0 && iy < blurred.rows && ix >= 0 && ix < blurred.cols) {
                pixel = static_cast<double>(blurred.at<uchar>(iy, ix));
            }
            moment += pixel * templateData[(dy + half) * templateSize + (dx + half)];
        }
    }
    return moment;
}

} // anonymous namespace

struct ZernikeEdgeCPU::Impl {
    ZernikeEdgeCPUParams params_;
    ZernikeTemplates templates_;
    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const ZernikeEdgeCPUParams& params)
        : params_(params)
        , templates_(ZernikeTemplates::create(params.templateSize))
    {
        params_.validate();
    }

    ~Impl() = default;

    ZernikeEdgeCPUResult Execute(const cv::Mat& srcImage)
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

        ZernikeEdgeCPUResult result;

        if (srcImage.empty()) {
            result.success = true;
            result.message = "Input image is empty";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        if (srcImage.type() != CV_8UC1) {
            result.success = false;
            result.message = "Input must be CV_8UC1 grayscale image";
            return result;
        }

        int minDim = std::min(srcImage.rows, srcImage.cols);
        if (minDim < params_.templateSize) {
            result.success = false;
            result.message = "Image too small for template size";
            return result;
        }

        cv::Mat blurred;
        cv::GaussianBlur(srcImage, blurred,
                          cv::Size(params_.gaussianKernelSize, params_.gaussianKernelSize),
                          params_.gaussianSigma);

        cv::Mat cannyEdges;
        cv::Canny(blurred, cannyEdges,
                  params_.cannyLowThreshold, params_.cannyHighThreshold,
                  params_.sobelApertureSize);

        result.cannyEdgeImage = cannyEdges.clone();

        std::vector<cv::Point> edgePixels;
        cv::findNonZero(cannyEdges, edgePixels);

        if (edgePixels.empty()) {
            result.success = true;
            result.message = "No edge pixels detected by Canny";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        result.edgePoints.reserve(edgePixels.size());
        int filteredCount = 0;

        for (const auto& pt : edgePixels) {
            double z11_re = computeZernikeMoment(blurred, pt.x, pt.y, templates_.T11_re, templates_.size);
            double z11_im = computeZernikeMoment(blurred, pt.x, pt.y, templates_.T11_im, templates_.size);
            double z20    = computeZernikeMoment(blurred, pt.x, pt.y, templates_.T20, templates_.size);

            double mag = std::sqrt(z11_re * z11_re + z11_im * z11_im);

            if (mag < params_.edgeStrengthThreshold) {
                ++filteredCount;
                continue;
            }

            double angle = std::atan2(z11_im, z11_re);
            double l = 0.0;
            if (mag > 1e-10) {
                l = z20 / mag;
            }

            double subX = static_cast<double>(pt.x) + l * std::cos(angle);
            double subY = static_cast<double>(pt.y) + l * std::sin(angle);

            EdgePoint ep;
            ep.x = subX;
            ep.y = subY;
            ep.angle = angle;
            ep.amplitude = mag;
            ep.pixelX = pt.x;
            ep.pixelY = pt.y;

            result.edgePoints.push_back(ep);
        }

        result.edgeCount = static_cast<int>(result.edgePoints.size());
        result.success = true;

        if (result.edgeCount == 0) {
            result.message = "All edge points filtered by strength threshold";
            result.qualityFlag = calib::QualityFlag::Warning;
        } else if (filteredCount > 0) {
            result.message = "Some edge points filtered by strength threshold";
            result.qualityFlag = calib::QualityFlag::Degraded;
        } else {
            result.message = "Sub-pixel edge extraction successful";
            result.qualityFlag = calib::QualityFlag::Normal;
        }

        CALIB_LOG_DEBUG("Execute() done: edgeCount={}, filtered={}",
                        result.edgeCount, filteredCount);

        return result;
    }

    void Warmup(int rows, int cols)
    {
        cv::Mat dummy = cv::Mat::zeros(rows, cols, CV_8UC1);
        cv::Mat blurred;
        cv::GaussianBlur(dummy, blurred, cv::Size(3, 3), 1.0);
        cv::Mat cannyOut;
        cv::Canny(blurred, cannyOut, 50, 150, 3);
        (void)cannyOut;

        warmup_rows_ = rows;
        warmup_cols_ = cols;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: rows={}, cols={}", rows, cols);
    }

    void SetParams(const ZernikeEdgeCPUParams& params)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() called while Execute() is running - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        templates_ = ZernikeTemplates::create(params.templateSize);
        warmed_up_ = false;
    }

    const ZernikeEdgeCPUParams& GetParams() const { return params_; }
};

ZernikeEdgeCPU::ZernikeEdgeCPU(const ZernikeEdgeCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("ZernikeEdgeCPU initialized: templateSize={}, canny=[{:.1f},{:.1f}]",
                   params.templateSize, params.cannyLowThreshold, params.cannyHighThreshold);
}

ZernikeEdgeCPU::~ZernikeEdgeCPU() = default;

void ZernikeEdgeCPU::Destroy() { }

ZernikeEdgeCPUResult ZernikeEdgeCPU::Execute(const cv::Mat& srcImage)
{
    CALIB_LOG_DEBUG("Execute() called: image size={}x{}", srcImage.cols, srcImage.rows);

    if (srcImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input image is empty");
        ZernikeEdgeCPUResult result;
        result.success = false;
        result.message = "Input image is empty";
        return result;
    }

    if (srcImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input must be CV_8UC1, got type {}", srcImage.type());
        ZernikeEdgeCPUResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 grayscale image";
        return result;
    }

    return pImpl_->Execute(srcImage);
}

void ZernikeEdgeCPU::Warmup(int rows, int cols)
{
    CALIB_LOG_INFO("warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->Warmup(rows, cols);
}

void ZernikeEdgeCPU::Warmup(const calib::WarmupConfig& config)
{
    CALIB_LOG_INFO("warmup(WarmupConfig) called: rows={}, cols={}", config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void ZernikeEdgeCPU::SetParams(const ZernikeEdgeCPUParams& params)
{
    CALIB_LOG_INFO("setParams() called: templateSize={}", params.templateSize);
    pImpl_->SetParams(params);
}

const ZernikeEdgeCPUParams& ZernikeEdgeCPU::GetParams() const
{
    return pImpl_->GetParams();
}
