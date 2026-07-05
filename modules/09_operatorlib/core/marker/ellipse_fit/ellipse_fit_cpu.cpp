/**
 * @file ellipse_fit_cpu.cpp
 * @brief 椭圆拟合和中心提取算子 - 实现文件
 *
 * 算法流程：
 *   1. 输入亚像素边缘点集
 *   2. RANSAC 采样拟合椭圆，用代数距离筛选内点
 *   3. 对内点集执行 fitEllipseAMS（或 fitEllipse 降级）进行最终拟合
 *   4. 输出椭圆中心和几何参数
 */

#include "ellipse_fit_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <cassert>
#include <cmath>

using namespace calib;

OperatorInfo getEllipseFitCPUInfo() {
    return OperatorInfo{"EllipseFitCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(07, EllipseFitCPU);

struct EllipseFitCPU::Impl {
    EllipseFitCPUParams params_;
    bool warmed_up_ = false;
    int warmup_maxPoints_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EllipseFitCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    ~Impl() = default;

    EllipseFitCPUResult ExecuteImpl(const std::vector<cv::Point2f>& subPixPoints) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        EllipseFitCPUResult result;
        result.totalPointCount = static_cast<int>(subPixPoints.size());

        if (subPixPoints.size() < 5) {
            result.success = false;
            result.message = "Too few points for ellipse fitting (need >= 5)";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        cv::RotatedRect bestModel;
        int maxInliers = 0;
        cv::RNG rng(12345);

        for (int i = 0; i < params_.ransacIterations; ++i) {
            std::vector<cv::Point2f> samplePts;
            samplePts.reserve(5);
            for (int j = 0; j < 5; ++j) {
                samplePts.push_back(subPixPoints[rng.uniform(0, static_cast<int>(subPixPoints.size()))]);
            }

            cv::Rect bbox = cv::boundingRect(samplePts);
            if (bbox.area() < 2)
                continue;

            cv::RotatedRect ell;
            try {
                ell = cv::fitEllipse(samplePts);
            } catch (...) {
                continue;
            }

            if (ell.size.width < params_.minEllipseAxis || ell.size.height < params_.minEllipseAxis)
                continue;

            float ratio = ell.size.width / ell.size.height;
            if (ratio > params_.maxAxisRatio || ratio < 1.0 / params_.maxAxisRatio)
                continue;

            float a = ell.size.width / 2.0f;
            float b = ell.size.height / 2.0f;
            float a2 = a * a;
            float b2 = b * b;
            float rad = ell.angle * static_cast<float>(CV_PI) / 180.0f;
            float cos_a = std::cos(rad);
            float sin_a = std::sin(rad);

            int inliersCount = 0;
            for (const auto& pt : subPixPoints) {
                float dx = pt.x - ell.center.x;
                float dy = pt.y - ell.center.y;
                float x_rot = dx * cos_a + dy * sin_a;
                float y_rot = -dx * sin_a + dy * cos_a;
                float dist = std::abs((x_rot * x_rot) / a2 + (y_rot * y_rot) / b2 - 1.0f);
                if (dist < static_cast<float>(params_.ransacThreshold))
                    inliersCount++;
            }

            if (inliersCount > maxInliers) {
                maxInliers = inliersCount;
                bestModel = ell;
                if (maxInliers > static_cast<int>(subPixPoints.size() * params_.earlyStopRatio))
                    break;
            }
        }

        if (maxInliers < params_.minInliers) {
            result.success = false;
            result.message = "Insufficient inliers after RANSAC";
            result.qualityFlag = calib::QualityFlag::Degraded;
            return result;
        }

        std::vector<cv::Point2f> finalInliers;
        finalInliers.reserve(maxInliers);

        float a = bestModel.size.width / 2.0f;
        float b = bestModel.size.height / 2.0f;
        float a2 = a * a;
        float b2 = b * b;
        float rad = bestModel.angle * static_cast<float>(CV_PI) / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);

        for (const auto& pt : subPixPoints) {
            float dx = pt.x - bestModel.center.x;
            float dy = pt.y - bestModel.center.y;
            float x_rot = dx * cos_a + dy * sin_a;
            float y_rot = -dx * sin_a + dy * cos_a;
            float dist = std::abs((x_rot * x_rot) / a2 + (y_rot * y_rot) / b2 - 1.0f);
            if (dist < static_cast<float>(params_.ransacThreshold)) {
                finalInliers.push_back(pt);
            }
        }

        if (finalInliers.size() < 5) {
            result.success = false;
            result.message = "Too few inliers for final fitting";
            result.qualityFlag = calib::QualityFlag::Degraded;
            return result;
        }

        cv::RotatedRect finalResult;
        try {
            if (params_.useAMS) {
                finalResult = cv::fitEllipseAMS(finalInliers);
            } else {
                finalResult = cv::fitEllipse(finalInliers);
            }
        } catch (...) {
            try {
                finalResult = cv::fitEllipse(finalInliers);
            } catch (...) {
                result.success = false;
                result.message = "Ellipse fitting failed";
                result.qualityFlag = calib::QualityFlag::Degraded;
                return result;
            }
        }

        result.success = true;
        result.message = "Ellipse fitting successful";
        result.qualityFlag = calib::QualityFlag::Normal;
        result.centerX = finalResult.center.x;
        result.centerY = finalResult.center.y;
        result.majorAxis = finalResult.size.width;
        result.minorAxis = finalResult.size.height;
        result.angle = finalResult.angle;
        result.inlierCount = static_cast<int>(finalInliers.size());
        result.inlierPoints = std::move(finalInliers);
        return result;
    }

    void Warmup(int maxPointCount) {
        warmup_maxPoints_ = maxPointCount;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxPointCount={}", maxPointCount);
    }

    void SetParams(const EllipseFitCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const EllipseFitCPUParams& GetParams() const { return params_; }
};

// ============================================================
// 构造 / 析构
// ============================================================
EllipseFitCPU::EllipseFitCPU(const EllipseFitCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EllipseFitCPU initialized (ransacIter={}, threshold={})",
                   params.ransacIterations, params.ransacThreshold);
}

EllipseFitCPU::~EllipseFitCPU() = default;

void EllipseFitCPU::Destroy() { }

// ============================================================
// fit(EdgePoint)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<EdgePoint>& edgePoints) {
    CALIB_LOG_DEBUG("fit(EdgePoint) called: pointCount={}", edgePoints.size());
    std::vector<cv::Point2f> pts;
    pts.reserve(edgePoints.size());
    for (const auto& ep : edgePoints)
        pts.emplace_back(static_cast<float>(ep.x), static_cast<float>(ep.y));
    return pImpl_->ExecuteImpl(pts);
}

// ============================================================
// fit(Point2f)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<cv::Point2f>& points) {
    CALIB_LOG_DEBUG("fit(Point2f) called: pointCount={}", points.size());
    return pImpl_->ExecuteImpl(points);
}

// ============================================================
// fit(Point2d)
// ============================================================
EllipseFitCPUResult EllipseFitCPU::Execute(const std::vector<cv::Point2d>& points) {
    CALIB_LOG_DEBUG("fit(Point2d) called: pointCount={}", points.size());
    std::vector<cv::Point2f> pts;
    pts.reserve(points.size());
    for (const auto& pt : points)
        pts.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    return pImpl_->ExecuteImpl(pts);
}

// ============================================================
// warmup
// ============================================================
void EllipseFitCPU::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
}

void EllipseFitCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================
void EllipseFitCPU::SetParams(const EllipseFitCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const EllipseFitCPUParams& EllipseFitCPU::GetParams() const {
    return pImpl_->GetParams();
}
