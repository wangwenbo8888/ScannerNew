/**
 * @file undistort_points_cpu.cpp
 * @brief 双目立体去畸变+矫正算子 - 实现文件
 *
 * CPU 算子保留 Impl 结构与框架风格统一。
 * Stereo: cv::stereoRectify() 计算矫正矩阵, 再 cv::undistortPoints(R,P)
 */

#include "undistort_points_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/calib3d.hpp>
#include <atomic>
#include <cassert>
#include <algorithm>

using namespace calib;

OperatorInfo getMarkerUndistortCPUInfo() {
    return OperatorInfo{"MarkerUndistortCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(06, MarkerUndistortCPU);

struct MarkerUndistortCPU::Impl {
    MarkerUndistortCPUParams params_;
    cv::Mat cameraMatrix1_;
    cv::Mat distCoeffs1_;
    cv::Mat cameraMatrix2_;
    cv::Mat distCoeffs2_;
    cv::Mat R_;
    cv::Mat T_;

    cv::Mat extR1_, extR2_, extP1_, extP2_, extQ_;
    bool hasExternalMatrices_ = false;

    bool warmed_up_ = false;
    int warmup_maxPoints_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const MarkerUndistortCPUParams& params)
        : params_(params)
    {
        params_.validate();
        buildCameraParams();
    }

    ~Impl() = default;

    cv::Mat buildK(double fx, double fy, double cx, double cy) const {
        return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }

    cv::Mat buildD(double k1, double k2, double p1, double p2,
                   double k3, double k4, double k5, double k6) const {
        if (params_.distortionModel == "rational_polynomial") {
            return (cv::Mat_<double>(8, 1) << k1, k2, p1, p2, k3, k4, k5, k6);
        }
        return (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    }

    void buildCameraParams() {
        cameraMatrix1_ = buildK(params_.fx1, params_.fy1, params_.cx1, params_.cy1);
        distCoeffs1_ = buildD(params_.k1_1, params_.k2_1, params_.p1_1, params_.p2_1,
                              params_.k3_1, params_.k4_1, params_.k5_1, params_.k6_1);

        cameraMatrix2_ = buildK(params_.fx2, params_.fy2, params_.cx2, params_.cy2);
        distCoeffs2_ = buildD(params_.k1_2, params_.k2_2, params_.p1_2, params_.p2_2,
                              params_.k3_2, params_.k4_2, params_.k5_2, params_.k6_2);
        R_ = (cv::Mat_<double>(3, 3) <<
            params_.R[0], params_.R[1], params_.R[2],
            params_.R[3], params_.R[4], params_.R[5],
            params_.R[6], params_.R[7], params_.R[8]);
        T_ = (cv::Mat_<double>(3, 1) <<
            params_.T[0], params_.T[1], params_.T[2]);
    }

    cv::Mat toSrcMat(const std::vector<cv::Point2d>& points) const {
        int N = static_cast<int>(points.size());
        cv::Mat src(1, N, CV_64FC2);
        for (int i = 0; i < N; ++i)
            src.at<cv::Vec2d>(0, i) = cv::Vec2d(points[i].x, points[i].y);
        return src;
    }

    std::vector<cv::Point2d> fromDstMat(const cv::Mat& dst) const {
        int N = static_cast<int>(dst.total());
        cv::Mat flat = dst.reshape(2, 1);
        std::vector<cv::Point2d> pts(N);
        for (int i = 0; i < N; ++i)
            pts[i] = cv::Point2d(flat.at<cv::Vec2d>(0, i)[0], flat.at<cv::Vec2d>(0, i)[1]);
        return pts;
    }

    StereoUndistortResult ExecuteImpl(
        const std::vector<cv::Point2d>& points1,
        const std::vector<cv::Point2d>& points2,
        const std::vector<int>& groupIds1,
        const std::vector<int>& groupIds2)
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

        StereoUndistortResult result;

        if (points1.empty() || points2.empty()) {
            result.success = true;
            result.message = "One or both input point lists are empty";
            result.qualityFlag = calib::QualityFlag::Warning;
            return result;
        }

        cv::Size imgSize(params_.imageWidth, params_.imageHeight);
        cv::Mat R1, R2, P1, P2, Q;
        if (hasExternalMatrices_) {
            R1 = extR1_; R2 = extR2_; P1 = extP1_; P2 = extP2_; Q = extQ_;
        } else {
            cv::stereoRectify(cameraMatrix1_, distCoeffs1_,
                              cameraMatrix2_, distCoeffs2_,
                              imgSize, R_, T_, R1, R2, P1, P2, Q);
        }

        cv::Mat src1 = toSrcMat(points1);
        cv::Mat src2 = toSrcMat(points2);
        cv::Mat dst1, dst2;

        cv::undistortPoints(src1, dst1, cameraMatrix1_, distCoeffs1_, R1, P1);
        cv::undistortPoints(src2, dst2, cameraMatrix2_, distCoeffs2_, R2, P2);

        result.rectifiedPoints1 = fromDstMat(dst1);
        result.rectifiedPoints2 = fromDstMat(dst2);
        result.pointCount1 = static_cast<int>(result.rectifiedPoints1.size());
        result.pointCount2 = static_cast<int>(result.rectifiedPoints2.size());
        result.R1 = R1;
        result.R2 = R2;
        result.P1 = P1;
        result.P2 = P2;
        result.Q = Q;
        result.success = true;
        result.message = "Stereo undistort+rectify successful";
        result.qualityFlag = calib::QualityFlag::Normal;

        result.groupIds1 = groupIds1;
        result.groupIds2 = groupIds2;
        if (!groupIds1.empty()) {
            int gc1 = 0;
            for (int g : groupIds1) gc1 = (std::max)(gc1, g + 1);
            result.groupCount1 = gc1;
        }
        if (!groupIds2.empty()) {
            int gc2 = 0;
            for (int g : groupIds2) gc2 = (std::max)(gc2, g + 1);
            result.groupCount2 = gc2;
        }
        return result;
    }

    void Warmup(int maxPointCount) {
        warmup_maxPoints_ = maxPointCount;
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxPointCount={}", maxPointCount);
    }

    void SetParams(const MarkerUndistortCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        buildCameraParams();
        warmed_up_ = false;
    }

    const MarkerUndistortCPUParams& GetParams() const { return params_; }
};

MarkerUndistortCPU::MarkerUndistortCPU(const MarkerUndistortCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("MarkerUndistortCPU initialized (stereo mode)");
}

MarkerUndistortCPU::~MarkerUndistortCPU() = default;

void MarkerUndistortCPU::Destroy() { }

StereoUndistortResult MarkerUndistortCPU::Execute(
    const std::vector<EdgePoint>& edgePoints1,
    const std::vector<EdgePoint>& edgePoints2,
    const std::vector<int>& groupIds1,
    const std::vector<int>& groupIds2)
{
    CALIB_LOG_DEBUG("undistortRectifyStereo(EdgePoint) called: count1={}, count2={}",
                    edgePoints1.size(), edgePoints2.size());
    std::vector<cv::Point2d> pts1, pts2;
    pts1.reserve(edgePoints1.size());
    pts2.reserve(edgePoints2.size());
    for (const auto& ep : edgePoints1) pts1.emplace_back(ep.x, ep.y);
    for (const auto& ep : edgePoints2) pts2.emplace_back(ep.x, ep.y);
    return pImpl_->ExecuteImpl(pts1, pts2, groupIds1, groupIds2);
}

StereoUndistortResult MarkerUndistortCPU::Execute(
    const std::vector<cv::Point2d>& points1,
    const std::vector<cv::Point2d>& points2,
    const std::vector<int>& groupIds1,
    const std::vector<int>& groupIds2)
{
    CALIB_LOG_DEBUG("undistortRectifyStereo(Point2d) called: count1={}, count2={}",
                    points1.size(), points2.size());
    return pImpl_->ExecuteImpl(points1, points2, groupIds1, groupIds2);
}

void MarkerUndistortCPU::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
}

void MarkerUndistortCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

void MarkerUndistortCPU::SetRectifyMatrices(const cv::Mat& R1, const cv::Mat& R2,
                                            const cv::Mat& P1, const cv::Mat& P2,
                                            const cv::Mat& Q) {
    CALIB_LOG_INFO("setRectifyMatrices() called — external R1/R2/P1/P2/Q provided, stereoRectify will be skipped");
    pImpl_->extR1_ = R1.clone();
    pImpl_->extR2_ = R2.clone();
    pImpl_->extP1_ = P1.clone();
    pImpl_->extP2_ = P2.clone();
    pImpl_->extQ_ = Q.clone();
    pImpl_->hasExternalMatrices_ = true;
}

void MarkerUndistortCPU::ClearRectifyMatrices() {
    CALIB_LOG_INFO("clearRectifyMatrices() — reverting to internal stereoRectify");
    pImpl_->extR1_.release();
    pImpl_->extR2_.release();
    pImpl_->extP1_.release();
    pImpl_->extP2_.release();
    pImpl_->extQ_.release();
    pImpl_->hasExternalMatrices_ = false;
}

void MarkerUndistortCPU::SetParams(const MarkerUndistortCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const MarkerUndistortCPUParams& MarkerUndistortCPU::GetParams() const {
    return pImpl_->GetParams();
}
