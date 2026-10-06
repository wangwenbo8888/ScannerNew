/**
 * @file epipolar_intersect_cpu.cpp
 * @brief 椭圆边界极线交点算子 - 实现文件
 *
 * 算法流程：
 *   1. 输入椭圆参数（中心、长轴、短轴、旋转角）
 *   2. 计算椭圆Y方向投影范围 [yMin, yMax]
 *   3. 以 epipolarStep 间距生成水平极线
 *   4. 对每条极线 y=const 代入椭圆方程，求解二次方程得到交点X
 *   5. 输出中心点 + 亚像素交点集
 */

#include "epipolar_intersect_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace calib;

OperatorInfo getEpipolarIntersectCPUInfo() {
    return OperatorInfo{"EpipolarIntersectCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(09, EpipolarIntersectCPU);

struct EpipolarIntersectCPU::Impl {
    EpipolarIntersectCPUParams params_;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EpipolarIntersectCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    ~Impl() = default;

    EllipseIntersectResult intersectImpl(
        double cx, double cy, double majorAxis, double minorAxis, double angleDeg)
    {
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

        EllipseIntersectResult result;
        result.centerX = cx;
        result.centerY = cy;
        result.majorAxis = majorAxis;
        result.minorAxis = minorAxis;
        result.angle = angleDeg;

        double a = majorAxis / 2.0;
        double b = minorAxis / 2.0;

        if (a <= 0.0 || b <= 0.0) {
            return result;
        }

        double rad = angleDeg * CV_PI / 180.0;
        double cosA = std::cos(rad);
        double sinA = std::sin(rad);

        double sinA2 = sinA * sinA;
        double cosA2 = cosA * cosA;
        double yHalfRange = std::sqrt(a * a * sinA2 + b * b * cosA2);
        double yMin = cy - yHalfRange;
        double yMax = cy + yHalfRange;

        double step = params_.epipolarStep;
        int idxStart = static_cast<int>(std::ceil(yMin / step));
        int idxEnd = static_cast<int>(std::floor(yMax / step));

        double a2 = a * a;
        double b2 = b * b;

        result.intersectPts.reserve(
            std::min(2 * (idxEnd - idxStart + 1), params_.maxIntersectionsPerEllipse));

        int totalPts = 0;

        for (int idx = idxStart; idx <= idxEnd; ++idx) {
            double yEpi = idx * step;
            double dy = yEpi - cy;

            double F = cosA2 / a2 + sinA2 / b2;
            double E = 2.0 * dy * sinA * cosA * (1.0 / a2 - 1.0 / b2);
            double D = dy * dy * (sinA2 / a2 + cosA2 / b2) - 1.0;

            double delta = E * E - 4.0 * F * D;

            if (delta < 0.0)
                continue;

            if (std::abs(delta) < 1e-12) {
                double x = -E / (2.0 * F);
                EpipolarIntersectPoint pt;
                pt.x = cx + x;
                pt.y = yEpi;
                pt.yEpipolar = yEpi;
                pt.epipolarIndex = idx;
                result.intersectPts.push_back(pt);
                ++totalPts;
            } else {
                double sqrtDelta = std::sqrt(delta);
                double x1 = (-E + sqrtDelta) / (2.0 * F);
                double x2 = (-E - sqrtDelta) / (2.0 * F);

                if (x1 > x2) std::swap(x1, x2);

                EpipolarIntersectPoint pt1;
                pt1.x = cx + x1;
                pt1.y = yEpi;
                pt1.yEpipolar = yEpi;
                pt1.epipolarIndex = idx;

                EpipolarIntersectPoint pt2;
                pt2.x = cx + x2;
                pt2.y = yEpi;
                pt2.yEpipolar = yEpi;
                pt2.epipolarIndex = idx;

                result.intersectPts.push_back(pt1);
                result.intersectPts.push_back(pt2);
                totalPts += 2;
            }

            if (totalPts >= params_.maxIntersectionsPerEllipse)
                break;
        }

        std::sort(result.intersectPts.begin(), result.intersectPts.end(),
            [](const EpipolarIntersectPoint& lhs, const EpipolarIntersectPoint& rhs) {
                if (lhs.epipolarIndex != rhs.epipolarIndex)
                    return lhs.epipolarIndex < rhs.epipolarIndex;
                return lhs.x < rhs.x;
            });

        return result;
    }

    void Warmup(int maxEllipseCount) {
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxEllipseCount={}", maxEllipseCount);
    }

    void SetParams(const EpipolarIntersectCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const EpipolarIntersectCPUParams& GetParams() const { return params_; }
};

// ============================================================
// 构造 / 析构
// ============================================================
EpipolarIntersectCPU::EpipolarIntersectCPU(const EpipolarIntersectCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EpipolarIntersectCPU initialized (epipolarStep={})",
                   params.epipolarStep);
}

EpipolarIntersectCPU::~EpipolarIntersectCPU() = default;

void EpipolarIntersectCPU::Destroy() { }

// ============================================================
// intersect(EllipseFitCPUResult)
// ============================================================
EllipseIntersectResult EpipolarIntersectCPU::Execute(const EllipseFitCPUResult& ellipseResult) {
    CALIB_LOG_DEBUG("intersect(EllipseFitCPUResult) called: center=({},{})",
                    ellipseResult.centerX, ellipseResult.centerY);
    return pImpl_->intersectImpl(
        ellipseResult.centerX, ellipseResult.centerY,
        ellipseResult.majorAxis, ellipseResult.minorAxis,
        ellipseResult.angle);
}

// ============================================================
// intersect(RotatedRect)
// ============================================================
EllipseIntersectResult EpipolarIntersectCPU::Execute(const cv::RotatedRect& ellipse) {
    CALIB_LOG_DEBUG("intersect(RotatedRect) called: center=({},{})",
                    ellipse.center.x, ellipse.center.y);
    return pImpl_->intersectImpl(
        static_cast<double>(ellipse.center.x),
        static_cast<double>(ellipse.center.y),
        static_cast<double>(ellipse.size.width),
        static_cast<double>(ellipse.size.height),
        static_cast<double>(ellipse.angle));
}

// ============================================================
// intersect(cx, cy, major, minor, angle)
// ============================================================
EllipseIntersectResult EpipolarIntersectCPU::Execute(
    double cx, double cy, double major, double minor, double angleDeg)
{
    CALIB_LOG_DEBUG("intersect(params) called: center=({},{})", cx, cy);
    return pImpl_->intersectImpl(cx, cy, major, minor, angleDeg);
}

// ============================================================
// intersectBatch(EllipseFitCPUResult)
// ============================================================
EpipolarIntersectCPUResult EpipolarIntersectCPU::Execute(
    const std::vector<EllipseFitCPUResult>& ellipseResults)
{
    CALIB_LOG_DEBUG("intersectBatch(EllipseFitCPUResult) called: count={}",
                    ellipseResults.size());

    EpipolarIntersectCPUResult result;
    result.ellipseResults.reserve(ellipseResults.size());

    for (const auto& er : ellipseResults) {
        result.ellipseResults.push_back(pImpl_->intersectImpl(
            er.centerX, er.centerY,
            er.majorAxis, er.minorAxis,
            er.angle));
    }

    result.success = true;
    result.message = "Batch intersection completed";
    result.qualityFlag = calib::QualityFlag::Normal;
    return result;
}

// ============================================================
// intersectBatch(RotatedRect)
// ============================================================
EpipolarIntersectCPUResult EpipolarIntersectCPU::Execute(
    const std::vector<cv::RotatedRect>& ellipses)
{
    CALIB_LOG_DEBUG("intersectBatch(RotatedRect) called: count={}", ellipses.size());

    EpipolarIntersectCPUResult result;
    result.ellipseResults.reserve(ellipses.size());

    for (const auto& er : ellipses) {
        result.ellipseResults.push_back(pImpl_->intersectImpl(
            static_cast<double>(er.center.x),
            static_cast<double>(er.center.y),
            static_cast<double>(er.size.width),
            static_cast<double>(er.size.height),
            static_cast<double>(er.angle)));
    }

    result.success = true;
    result.message = "Batch intersection completed";
    result.qualityFlag = calib::QualityFlag::Normal;
    return result;
}

// ============================================================
// warmup
// ============================================================
void EpipolarIntersectCPU::Warmup(int maxEllipseCount) {
    CALIB_LOG_INFO("warmup() called: maxEllipseCount={}", maxEllipseCount);
    pImpl_->Warmup(maxEllipseCount);
}

void EpipolarIntersectCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================
void EpipolarIntersectCPU::SetParams(const EpipolarIntersectCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const EpipolarIntersectCPUParams& EpipolarIntersectCPU::GetParams() const {
    return pImpl_->GetParams();
}
