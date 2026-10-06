#include <gtest/gtest.h>
#include "inverse_distort_cpu.h"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <random>

using namespace calib;

namespace {

cv::Mat makeCameraMatrix(double fx = 2500.0, double fy = 2500.0,
                          double cx = 1024.0, double cy = 768.0) {
    return (cv::Mat_<double>(3, 3) <<
        fx,  0,   cx,
        0,   fy,  cy,
        0,   0,   1);
}

cv::Mat makeDistCoeffs(double k1 = -0.1, double k2 = 0.05,
                        double p1 = 0.001, double p2 = -0.001, double k3 = 0.01) {
    return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

cv::Mat makeIdentityR1() {
    return cv::Mat::eye(3, 3, CV_64F);
}

cv::Mat makeP1(double fx = 2500.0, double fy = 2500.0,
               double cx = 1024.0, double cy = 768.0) {
    return (cv::Mat_<double>(3, 4) <<
        fx,  0,   cx,  0,
        0,   fy,  cy,  0,
        0,   0,   1,   0);
}

InverseDistortParams makeDefaultParams() {
    InverseDistortParams p;
    p.cameraMatrix = makeCameraMatrix();
    p.distCoeffs = makeDistCoeffs();
    p.R1 = makeIdentityR1();
    p.P1 = makeP1();
    p.maxIterations = 20;
    p.tolerance = 1e-10;
    return p;
}

std::vector<cv::Point2f> generateGridPoints(int cols, int rows,
                                             double x0, double y0,
                                             double step) {
    std::vector<cv::Point2f> pts;
    pts.reserve(static_cast<size_t>(cols) * rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pts.emplace_back(static_cast<float>(x0 + c * step),
                             static_cast<float>(y0 + r * step));
        }
    }
    return pts;
}

std::vector<cv::Point2f> forwardTransform(
    const std::vector<cv::Point2f>& distortedPoints,
    const cv::Mat& K, const cv::Mat& dist,
    const cv::Mat& R1, const cv::Mat& P1)
{
    cv::Mat pts(distortedPoints, true);
    pts.convertTo(pts, CV_64F);
    cv::undistortPoints(pts, pts, K, dist, R1, P1);
    pts.convertTo(pts, CV_32F);
    return std::vector<cv::Point2f>(pts.begin<cv::Point2f>(), pts.end<cv::Point2f>());
}

}

TEST(InverseDistortCpuTest, ParamsValidationValid) {
    auto p = makeDefaultParams();
    EXPECT_NO_THROW(p.validate());
}

TEST(InverseDistortCpuTest, ParamsValidationBadCameraMatrix) {
    auto p = makeDefaultParams();
    p.cameraMatrix = cv::Mat();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ParamsValidationBadDistCoeffs) {
    auto p = makeDefaultParams();
    p.distCoeffs = (cv::Mat_<double>(1, 3) << 0.1, 0.2, 0.3);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ParamsValidationBadR1) {
    auto p = makeDefaultParams();
    p.R1 = cv::Mat::eye(2, 2, CV_64F);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ParamsValidationBadP1) {
    auto p = makeDefaultParams();
    p.P1 = cv::Mat::eye(3, 3, CV_64F);
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ParamsValidationBadMaxIterations) {
    auto p = makeDefaultParams();
    p.maxIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ParamsValidationBadTolerance) {
    auto p = makeDefaultParams();
    p.tolerance = -1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(InverseDistortCpuTest, ConstructionAndParams) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);
    EXPECT_DOUBLE_EQ(cpu.GetParams().tolerance, 1e-10);
    EXPECT_EQ(cpu.GetParams().maxIterations, 20);
}

TEST(InverseDistortCpuTest, EmptyInputReturnsFalse) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);
    InverseDistortResult result;
    std::vector<cv::Point2f> empty;
    EXPECT_FALSE(cpu.Execute(empty, result));
}

TEST(InverseDistortCpuTest, RoundTripIdentityR1GridPoints) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    auto originals = generateGridPoints(10, 8, 200.0, 150.0, 80.0);

    auto vr = cpu.VerifyRoundTrip(originals, 0.001);
    EXPECT_TRUE(vr.passed) << vr.message;
    EXPECT_LT(vr.maxError, 0.001);
    EXPECT_LT(vr.meanError, 0.001);
}

TEST(InverseDistortCpuTest, RoundTripIdentityR1CenterPoints) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> originals = {
        {1024.0f, 768.0f},
        {1023.5f, 767.5f},
        {1024.5f, 768.5f}
    };

    auto vr = cpu.VerifyRoundTrip(originals, 0.001);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, RoundTripIdentityR1CornerPoints) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> originals = {
        {0.0f, 0.0f},
        {2047.0f, 0.0f},
        {0.0f, 1535.0f},
        {2047.0f, 1535.0f}
    };

    auto vr = cpu.VerifyRoundTrip(originals, 0.01);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, RoundTripWithRotationR1) {
    auto p = makeDefaultParams();

    double angle = 0.003;
    p.R1 = (cv::Mat_<double>(3, 3) <<
        std::cos(angle), -std::sin(angle), 0.0,
        std::sin(angle),  std::cos(angle), 0.0,
        0.0,              0.0,             1.0);

    double s = 0.9998;
    cv::Mat scale = (cv::Mat_<double>(3, 3) <<
        s, 0, 0, 0, s, 0, 0, 0, 1);
    p.R1 = scale * p.R1;

    p.P1 = makeP1(2500.0, 2500.0, 1024.0, 768.0);

    InverseDistortCPU cpu(p);
    auto originals = generateGridPoints(8, 6, 300.0, 200.0, 200.0);

    auto vr = cpu.VerifyRoundTrip(originals, 0.01);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, RoundTripZeroDistortion) {
    auto p = makeDefaultParams();
    p.distCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

    InverseDistortCPU cpu(p);
    auto originals = generateGridPoints(10, 8, 200.0, 150.0, 80.0);

    auto vr = cpu.VerifyRoundTrip(originals, 1e-6);
    EXPECT_TRUE(vr.passed) << vr.message;
    EXPECT_LT(vr.maxError, 1e-6);
}

TEST(InverseDistortCpuTest, RoundTripStrongDistortion) {
    auto p = makeDefaultParams();
    p.distCoeffs = (cv::Mat_<double>(1, 5) << -0.3, 0.15, 0.002, -0.002, 0.02);

    InverseDistortCPU cpu(p);

    auto originals = generateGridPoints(8, 6, 400.0, 300.0, 150.0);

    auto vr = cpu.VerifyRoundTrip(originals, 0.01);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, InverseRectifyAndDistortBasic) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> originals = {{500.0f, 400.0f}, {1500.0f, 1000.0f}};

    auto rectified = forwardTransform(originals, p.cameraMatrix, p.distCoeffs, p.R1, p.P1);

    InverseDistortResult result;
    bool ok = cpu.Execute(rectified, result);
    EXPECT_TRUE(ok);
    EXPECT_EQ(result.originalPoints.size(), originals.size());
    EXPECT_TRUE(result.success);

    for (size_t i = 0; i < originals.size(); ++i) {
        double dx = static_cast<double>(originals[i].x - result.originalPoints[i].x);
        double dy = static_cast<double>(originals[i].y - result.originalPoints[i].y);
        double err = std::sqrt(dx * dx + dy * dy);
        EXPECT_LT(err, 0.01) << "Point " << i << " error=" << err;
    }
}

TEST(InverseDistortCpuTest, InverseRectifyOnly) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> originals = {{500.0f, 400.0f}};

    cv::Mat pts(originals, true);
    pts.convertTo(pts, CV_64F);
    cv::undistortPoints(pts, pts, p.cameraMatrix, p.distCoeffs, p.R1, p.P1);

    std::vector<cv::Point2f> unrectNorm;
    bool ok = cpu.InverseRectify(originals, unrectNorm);
    EXPECT_TRUE(ok);
    EXPECT_FALSE(unrectNorm.empty());
}

TEST(InverseDistortCpuTest, ApplyDistortionOnly) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> undistNorm = {{0.0f, 0.0f}, {0.1f, -0.1f}};

    std::vector<cv::Point2f> distPixel;
    bool ok = cpu.ApplyDistortion(undistNorm, distPixel);
    EXPECT_TRUE(ok);
    EXPECT_EQ(distPixel.size(), undistNorm.size());

    EXPECT_NEAR(distPixel[0].x, 1024.0f, 1.0f);
    EXPECT_NEAR(distPixel[0].y, 768.0f, 1.0f);
}

TEST(InverseDistortCpuTest, ApplyDistortionEmptyInput) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> empty;
    std::vector<cv::Point2f> out;
    EXPECT_FALSE(cpu.ApplyDistortion(empty, out));
}

TEST(InverseDistortCpuTest, InverseRectifyEmptyInput) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> empty;
    std::vector<cv::Point2f> out;
    EXPECT_FALSE(cpu.InverseRectify(empty, out));
}

TEST(InverseDistortCpuTest, VerifyRoundTripEmptyInput) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> empty;
    auto vr = cpu.VerifyRoundTrip(empty);
    EXPECT_FALSE(vr.passed);
}

TEST(InverseDistortCpuTest, IterationCountWithinBounds) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    auto originals = generateGridPoints(10, 8, 200.0, 150.0, 80.0);
    auto rectified = forwardTransform(originals, p.cameraMatrix, p.distCoeffs, p.R1, p.P1);

    InverseDistortResult result;
    cpu.Execute(rectified, result);

    EXPECT_GT(result.maxIterationsUsed, 0);
    EXPECT_LE(result.maxIterationsUsed, p.maxIterations);

    for (int it : result.iterationsPerPoint) {
        EXPECT_GT(it, 0);
        EXPECT_LE(it, p.maxIterations);
    }
}

TEST(InverseDistortCpuTest, SinglePointRoundTrip) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    std::vector<cv::Point2f> originals = {{1024.0f, 768.0f}};

    auto vr = cpu.VerifyRoundTrip(originals, 0.001);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, ManyPointsRoundTrip) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu(p);

    auto originals = generateGridPoints(50, 40, 50.0, 50.0, 40.0);

    auto vr = cpu.VerifyRoundTrip(originals, 0.01);
    EXPECT_TRUE(vr.passed) << vr.message;
}

TEST(InverseDistortCpuTest, MoveSemantics) {
    auto p = makeDefaultParams();
    InverseDistortCPU cpu1(p);

    InverseDistortCPU cpu2(std::move(cpu1));

    std::vector<cv::Point2f> originals = {{500.0f, 400.0f}};
    auto vr = cpu2.VerifyRoundTrip(originals, 0.01);
    EXPECT_TRUE(vr.passed) << vr.message;
}
