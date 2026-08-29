#include "inverse_distort_cpu.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>

#include "common/calib_logging.h"
#include <nlohmann/json.hpp>

namespace calib {

CALIB_DEFINE_LOG_TAG(0, InverseDistort);

OperatorInfo getInverseDistortCPUInfo() {
    return OperatorInfo{"InverseDistortCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

namespace {

struct DistortionCoeffs {
    double k1, k2, p1, p2, k3;
};

DistortionCoeffs extractDistCoeffs(const cv::Mat& dist) {
    DistortionCoeffs c{};
    c.k1 = dist.at<double>(0, 0);
    c.k2 = dist.at<double>(0, 1);
    c.p1 = dist.at<double>(0, 2);
    c.p2 = dist.at<double>(0, 3);
    c.k3 = (dist.cols >= 5) ? dist.at<double>(0, 4) : 0.0;
    return c;
}

void applyDistortionModel(double x_u, double y_u,
                           const DistortionCoeffs& dc,
                           double& x_d, double& y_d)
{
    double r2 = x_u * x_u + y_u * y_u;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double radial = 1.0 + dc.k1 * r2 + dc.k2 * r4 + dc.k3 * r6;
    double dx_t = 2.0 * dc.p1 * x_u * y_u + dc.p2 * (r2 + 2.0 * x_u * x_u);
    double dy_t = dc.p1 * (r2 + 2.0 * y_u * y_u) + 2.0 * dc.p2 * x_u * y_u;
    x_d = x_u * radial + dx_t;
    y_d = y_u * radial + dy_t;
}

void undistortPointIter(
    double x_d, double y_d,
    const DistortionCoeffs& dc,
    int maxIter, double tol,
    double& x_u, double& y_u, int& iters)
{
    x_u = x_d;
    y_u = y_d;
    iters = 0;

    for (int i = 0; i < maxIter; ++i) {
        double xd_app, yd_app;
        applyDistortionModel(x_u, y_u, dc, xd_app, yd_app);

        double fx = xd_app - x_d;
        double fy = yd_app - y_d;

        Eigen::Vector2d fv(fx, fy);
        if (fv.norm() < tol) {
            iters = i + 1;
            return;
        }

        const double h = 1e-8;
        double xd_px, yd_px, xd_mx, yd_mx;
        double xd_py, yd_py, xd_my, yd_my;

        applyDistortionModel(x_u + h, y_u, dc, xd_px, yd_px);
        applyDistortionModel(x_u - h, y_u, dc, xd_mx, yd_mx);
        applyDistortionModel(x_u, y_u + h, dc, xd_py, yd_py);
        applyDistortionModel(x_u, y_u - h, dc, xd_my, yd_my);

        Eigen::Matrix2d J;
        J << (xd_px - xd_mx) / (2.0 * h), (xd_py - xd_my) / (2.0 * h),
             (yd_px - yd_mx) / (2.0 * h), (yd_py - yd_my) / (2.0 * h);

        Eigen::Vector2d delta = J.fullPivLu().solve(fv);

        x_u -= delta(0);
        y_u -= delta(1);
        iters = i + 1;
    }
}

void distortPointIter(
    double x_u, double y_u,
    const DistortionCoeffs& dc,
    int maxIter, double tol,
    double& x_d, double& y_d, int& iters)
{
    applyDistortionModel(x_u, y_u, dc, x_d, y_d);
    iters = 1;
    (void)maxIter;
    (void)tol;
}

}

void InverseDistortParams::validate() const {
    if (cameraMatrix.empty() || cameraMatrix.rows != 3 || cameraMatrix.cols != 3) {
        throw std::invalid_argument("InverseDistortParams: cameraMatrix must be 3x3");
    }
    if (distCoeffs.empty() || distCoeffs.rows != 1 || distCoeffs.cols < 5) {
        throw std::invalid_argument("InverseDistortParams: distCoeffs must be 1xN with N>=5");
    }
    if (R1.empty() || R1.rows != 3 || R1.cols != 3) {
        throw std::invalid_argument("InverseDistortParams: R1 must be 3x3");
    }
    if (P1.empty() || P1.rows != 3 || P1.cols != 4) {
        throw std::invalid_argument("InverseDistortParams: P1 must be 3x4");
    }
    if (maxIterations <= 0) {
        throw std::invalid_argument("InverseDistortParams: maxIterations must be > 0");
    }
    if (tolerance <= 0.0) {
        throw std::invalid_argument("InverseDistortParams: tolerance must be > 0");
    }
}

InverseDistortParams InverseDistortParams::fromJson(const std::string& json_path) {
    InverseDistortParams params;
    try {
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) {
            CALIB_LOG_WARN("Cannot open params file: " + json_path + ", using defaults");
            return params;
        }

        nlohmann::json j = nlohmann::json::parse(ifs);

        const std::string ctx = "InverseDistortParams::fromJson(" + json_path + ")";

        params.maxIterations = calib::getRequired<int>(j, "maxIterations", ctx);
        params.tolerance = calib::getRequired<double>(j, "tolerance", ctx);

        params.cameraMatrix = calib::jsonToMat(calib::getRequired<nlohmann::json>(j, "cameraMatrix", ctx), 3, 3);

        {
            auto jd = calib::getRequired<nlohmann::json>(j, "distCoeffs", ctx);
            int n = static_cast<int>(jd.size());
            params.distCoeffs = cv::Mat(1, n, CV_64F);
            for (int i = 0; i < n; ++i)
                params.distCoeffs.at<double>(0, i) = jd[i].get<double>();
        }

        params.R1 = calib::jsonToMat(calib::getRequired<nlohmann::json>(j, "R1", ctx), 3, 3);
        params.P1 = calib::jsonToMat(calib::getRequired<nlohmann::json>(j, "P1", ctx), 3, 4);

        CALIB_LOG_INFO(std::string("[01-InverseDistortCpu]") + " Loaded params from " + json_path);
    } catch (const std::exception& e) {
        CALIB_LOG_WARN("Failed to parse params file " + json_path + ": " + e.what());
    }
    return params;
}

class InverseDistortCPU::Impl {
public:
    explicit Impl(InverseDistortParams p) : params_(std::move(p)) {
        params_.validate();
        dc_ = extractDistCoeffs(params_.distCoeffs);
        fx_ = params_.cameraMatrix.at<double>(0, 0);
        fy_ = params_.cameraMatrix.at<double>(1, 1);
        cx_ = params_.cameraMatrix.at<double>(0, 2);
        cy_ = params_.cameraMatrix.at<double>(1, 2);

        fx_rect_ = params_.P1.at<double>(0, 0);
        fy_rect_ = params_.P1.at<double>(1, 1);
        cx_rect_ = params_.P1.at<double>(0, 2);
        cy_rect_ = params_.P1.at<double>(1, 2);

        R1_ = params_.R1.clone();
    }

    bool Execute(
        const std::vector<cv::Point2f>& rectifiedPoints,
        InverseDistortResult& result)
    {
        result.originalPoints.clear();
        result.iterationsPerPoint.clear();
        result.maxIterationsUsed = 0;
        result.success = false;

        if (rectifiedPoints.empty()) {
            result.message = "Empty input points";
            CALIB_LOG_WARN(std::string(kLogTag) + " Empty input points");
            return false;
        }

        result.originalPoints.reserve(rectifiedPoints.size());
        result.iterationsPerPoint.reserve(rectifiedPoints.size());

        double r00 = R1_.at<double>(0, 0), r01 = R1_.at<double>(0, 1), r02 = R1_.at<double>(0, 2);
        double r10 = R1_.at<double>(1, 0), r11 = R1_.at<double>(1, 1), r12 = R1_.at<double>(1, 2);
        double r20 = R1_.at<double>(2, 0), r21 = R1_.at<double>(2, 1), r22 = R1_.at<double>(2, 2);

        for (const auto& pt : rectifiedPoints) {
            double x_n = (pt.x - cx_rect_) / fx_rect_;
            double y_n = (pt.y - cy_rect_) / fy_rect_;

            double a0 = r00 - x_n * r20;
            double a1 = r01 - x_n * r21;
            double b0 = r10 - y_n * r20;
            double b1 = r11 - y_n * r21;
            double c0 = x_n * r22 - r02;
            double c1 = y_n * r22 - r12;

            double det = a0 * b1 - a1 * b0;
            double x_u, y_u;
            if (std::abs(det) > 1e-15) {
                x_u = (c0 * b1 - c1 * a1) / det;
                y_u = (a0 * c1 - b0 * c0) / det;
            } else {
                double rx = r00 * x_n + r01 * y_n + r02;
                double ry = r10 * x_n + r11 * y_n + r12;
                double rz = r20 * x_n + r21 * y_n + r22;
                x_u = rx / rz;
                y_u = ry / rz;
            }

            double x_d, y_d;
            int iters;
            distortPointIter(x_u, y_u, dc_, params_.maxIterations, params_.tolerance,
                           x_d, y_d, iters);

            double u_dist = fx_ * x_d + cx_;
            double v_dist = fy_ * y_d + cy_;

            result.originalPoints.emplace_back(static_cast<float>(u_dist), static_cast<float>(v_dist));
            result.iterationsPerPoint.push_back(iters);
            result.maxIterationsUsed = std::max(result.maxIterationsUsed, iters);
        }

        result.success = true;
        result.message = "OK";
        CALIB_LOG_INFO(std::string(kLogTag) + " Execute completed: "
                     + std::to_string(rectifiedPoints.size()) + " points, max iters="
                     + std::to_string(result.maxIterationsUsed));
        return true;
    }

    bool ApplyDistortion(
        const std::vector<cv::Point2f>& undistortedNormPoints,
        std::vector<cv::Point2f>& distortedPixelPoints)
    {
        if (undistortedNormPoints.empty()) {
            CALIB_LOG_WARN(std::string(kLogTag) + " applyDistortion: empty input");
            return false;
        }

        distortedPixelPoints.clear();
        distortedPixelPoints.reserve(undistortedNormPoints.size());

        for (const auto& pt : undistortedNormPoints) {
            double x_d, y_d;
            int dummy_iters;
            distortPointIter(static_cast<double>(pt.x), static_cast<double>(pt.y), dc_,
                           30, 1e-12, x_d, y_d, dummy_iters);

            double u = fx_ * x_d + cx_;
            double v = fy_ * y_d + cy_;
            distortedPixelPoints.emplace_back(static_cast<float>(u), static_cast<float>(v));
        }

        return true;
    }

    bool InverseRectify(
        const std::vector<cv::Point2f>& rectifiedPoints,
        std::vector<cv::Point2f>& unrectifiedNormPoints)
    {
        if (rectifiedPoints.empty()) {
            CALIB_LOG_WARN(std::string(kLogTag) + " inverseRectify: empty input");
            return false;
        }

        unrectifiedNormPoints.clear();
        unrectifiedNormPoints.reserve(rectifiedPoints.size());

        double r00 = R1_.at<double>(0, 0), r01 = R1_.at<double>(0, 1), r02 = R1_.at<double>(0, 2);
        double r10 = R1_.at<double>(1, 0), r11 = R1_.at<double>(1, 1), r12 = R1_.at<double>(1, 2);
        double r20 = R1_.at<double>(2, 0), r21 = R1_.at<double>(2, 1), r22 = R1_.at<double>(2, 2);

        for (const auto& pt : rectifiedPoints) {
            double x_n = (pt.x - cx_rect_) / fx_rect_;
            double y_n = (pt.y - cy_rect_) / fy_rect_;

            double a0 = r00 - x_n * r20;
            double a1 = r01 - x_n * r21;
            double b0 = r10 - y_n * r20;
            double b1 = r11 - y_n * r21;
            double c0 = x_n * r22 - r02;
            double c1 = y_n * r22 - r12;

            double det = a0 * b1 - a1 * b0;
            double x_u, y_u;
            if (std::abs(det) > 1e-15) {
                x_u = (c0 * b1 - c1 * a1) / det;
                y_u = (a0 * c1 - b0 * c0) / det;
            } else {
                double rx = r00 * x_n + r01 * y_n + r02;
                double ry = r10 * x_n + r11 * y_n + r12;
                double rz = r20 * x_n + r21 * y_n + r22;
                x_u = rx / rz;
                y_u = ry / rz;
            }

            unrectifiedNormPoints.emplace_back(static_cast<float>(x_u), static_cast<float>(y_u));
        }

        return true;
    }

    RoundTripVerifyResult VerifyRoundTrip(
        const std::vector<cv::Point2f>& originalDistortedPoints,
        double maxAllowableError)
    {
        RoundTripVerifyResult vr;
        vr.passed = false;
        vr.success = vr.passed;
        vr.maxError = 0.0;
        vr.meanError = 0.0;

        if (originalDistortedPoints.empty()) {
            vr.message = "Empty input points";
            return vr;
        }

        std::vector<cv::Point2f> undistorted;
        cv::Mat ptMat(originalDistortedPoints);
        ptMat.convertTo(ptMat, CV_64F);

        cv::undistortPoints(ptMat, ptMat, params_.cameraMatrix, params_.distCoeffs,
                           params_.R1, params_.P1);
        ptMat.convertTo(ptMat, CV_32F);
        std::vector<cv::Point2f> rectified(ptMat.begin<cv::Point2f>(), ptMat.end<cv::Point2f>());

        InverseDistortResult invResult;
        bool ok = Execute(rectified, invResult);
        if (!ok) {
            vr.message = "Inverse transform failed";
            return vr;
        }

        vr.perPointErrors.reserve(originalDistortedPoints.size());
        double sum = 0.0;
        for (size_t i = 0; i < originalDistortedPoints.size(); ++i) {
            double dx = static_cast<double>(originalDistortedPoints[i].x - invResult.originalPoints[i].x);
            double dy = static_cast<double>(originalDistortedPoints[i].y - invResult.originalPoints[i].y);
            double err = std::sqrt(dx * dx + dy * dy);
            vr.perPointErrors.push_back(err);
            vr.maxError = std::max(vr.maxError, err);
            sum += err;
        }
        vr.meanError = sum / static_cast<double>(originalDistortedPoints.size());
        vr.passed = (vr.maxError <= maxAllowableError);
        vr.success = vr.passed;
        vr.message = vr.passed ? "PASS" : "FAIL: max error " + std::to_string(vr.maxError)
                     + " > threshold " + std::to_string(maxAllowableError);

        CALIB_LOG_INFO(std::string(kLogTag) + " verifyRoundTrip: " + vr.message
                     + " mean=" + std::to_string(vr.meanError)
                     + " max=" + std::to_string(vr.maxError));
        return vr;
    }

    void SetParams(const InverseDistortParams& p) {
        params_ = p;
        params_.validate();
        dc_ = extractDistCoeffs(params_.distCoeffs);
        fx_ = params_.cameraMatrix.at<double>(0, 0);
        fy_ = params_.cameraMatrix.at<double>(1, 1);
        cx_ = params_.cameraMatrix.at<double>(0, 2);
        cy_ = params_.cameraMatrix.at<double>(1, 2);
        fx_rect_ = params_.P1.at<double>(0, 0);
        fy_rect_ = params_.P1.at<double>(1, 1);
        cx_rect_ = params_.P1.at<double>(0, 2);
        cy_rect_ = params_.P1.at<double>(1, 2);
        R1_ = params_.R1.clone();
    }
    const InverseDistortParams& GetParams() const noexcept { return params_; }

private:
    InverseDistortParams params_;
    DistortionCoeffs dc_;
    double fx_, fy_, cx_, cy_;
    double fx_rect_, fy_rect_, cx_rect_, cy_rect_;
    cv::Mat R1_;
};

InverseDistortCPU::InverseDistortCPU(const InverseDistortParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO(std::string(kLogTag) + " initialized");
}

InverseDistortCPU::~InverseDistortCPU() = default;

InverseDistortCPU::InverseDistortCPU(InverseDistortCPU&&) noexcept = default;
InverseDistortCPU& InverseDistortCPU::operator=(InverseDistortCPU&&) noexcept = default;

bool InverseDistortCPU::Execute(
    const std::vector<cv::Point2f>& rectifiedPoints,
    InverseDistortResult& result)
{
    return pImpl_->Execute(rectifiedPoints, result);
}

bool InverseDistortCPU::ApplyDistortion(
    const std::vector<cv::Point2f>& undistortedNormPoints,
    std::vector<cv::Point2f>& distortedPixelPoints)
{
    return pImpl_->ApplyDistortion(undistortedNormPoints, distortedPixelPoints);
}

bool InverseDistortCPU::InverseRectify(
    const std::vector<cv::Point2f>& rectifiedPoints,
    std::vector<cv::Point2f>& unrectifiedNormPoints)
{
    return pImpl_->InverseRectify(rectifiedPoints, unrectifiedNormPoints);
}

RoundTripVerifyResult InverseDistortCPU::VerifyRoundTrip(
    const std::vector<cv::Point2f>& originalDistortedPoints,
    double maxAllowableError)
{
    return pImpl_->VerifyRoundTrip(originalDistortedPoints, maxAllowableError);
}

void InverseDistortCPU::SetParams(const InverseDistortParams& params) {
    pImpl_->SetParams(params);
}

const InverseDistortParams& InverseDistortCPU::GetParams() const noexcept {
    return pImpl_->GetParams();
}

void InverseDistortCPU::Destroy() {
}

} // namespace calib
