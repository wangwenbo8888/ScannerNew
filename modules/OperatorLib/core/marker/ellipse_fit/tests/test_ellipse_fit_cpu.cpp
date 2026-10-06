/**
 * @file test_ellipse_fit_cpu.cpp
 * @brief 椭圆拟合和中心提取算子 - 单元测试
 *
 * 测试覆盖：
 * 参数校验 / JSON序列化 / Result结构体 / 空输入 / 完美椭圆 / 含噪声 /
 * 含离群点 / EdgePoint重载 / Point2d重载 / 边界 / setParams/getParams
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../ellipse_fit_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


namespace {

std::vector<cv::Point2f> generateEllipsePoints(
    float cx, float cy, float a, float b, float angleDeg, int n, float noiseStd = 0.0f)
{
    std::vector<cv::Point2f> pts;
    pts.reserve(n);
    float rad = angleDeg * static_cast<float>(CV_PI) / 180.0f;
    float cosA = std::cos(rad);
    float sinA = std::sin(rad);
    cv::RNG rng(42);

    for (int i = 0; i < n; ++i) {
        float theta = 2.0f * static_cast<float>(CV_PI) * i / n;
        float ex = a * std::cos(theta);
        float ey = b * std::sin(theta);
        float x = cx + ex * cosA - ey * sinA;
        float y = cy + ex * sinA + ey * cosA;
        if (noiseStd > 0.0f) {
            x += static_cast<float>(rng.gaussian(noiseStd));
            y += static_cast<float>(rng.gaussian(noiseStd));
        }
        pts.emplace_back(x, y);
    }
    return pts;
}

std::vector<cv::Point2f> addOutliers(
    const std::vector<cv::Point2f>& pts, int count, float cx, float cy, float spread)
{
    std::vector<cv::Point2f> result = pts;
    cv::RNG rng(123);
    for (int i = 0; i < count; ++i) {
        float x = cx + static_cast<float>(rng.uniform(-spread, spread));
        float y = cy + static_cast<float>(rng.uniform(-spread, spread));
        result.emplace_back(x, y);
    }
    return result;
}

} // anonymous namespace

// ============================================================
// 参数校验
// ============================================================
class EllipseFitCPUParamTest : public ::testing::Test {
protected:
    EllipseFitCPUParams defaultParams_;
};

TEST_F(EllipseFitCPUParamTest, DefaultParamsPassValidate) {
    EXPECT_NO_THROW(defaultParams_.validate());
}

TEST_F(EllipseFitCPUParamTest, ZeroRansacIterationsThrows) {
    auto p = defaultParams_; p.ransacIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, NegativeRansacIterationsThrows) {
    auto p = defaultParams_; p.ransacIterations = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, ZeroRansacThresholdThrows) {
    auto p = defaultParams_; p.ransacThreshold = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, ZeroMinEllipseAxisThrows) {
    auto p = defaultParams_; p.minEllipseAxis = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, MaxAxisRatioOneThrows) {
    auto p = defaultParams_; p.maxAxisRatio = 1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, MinInliersLessThanFiveThrows) {
    auto p = defaultParams_; p.minInliers = 4;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, EarlyStopRatioZeroThrows) {
    auto p = defaultParams_; p.earlyStopRatio = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EllipseFitCPUParamTest, EarlyStopRatioAboveOneThrows) {
    auto p = defaultParams_; p.earlyStopRatio = 1.1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化
// ============================================================
TEST_F(EllipseFitCPUParamTest, JsonRoundtrip) {
    auto j = defaultParams_.toJson();
    auto restored = EllipseFitCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_EQ(restored.ransacIterations, defaultParams_.ransacIterations);
    EXPECT_DOUBLE_EQ(restored.ransacThreshold, defaultParams_.ransacThreshold);
    EXPECT_DOUBLE_EQ(restored.minEllipseAxis, defaultParams_.minEllipseAxis);
    EXPECT_DOUBLE_EQ(restored.maxAxisRatio, defaultParams_.maxAxisRatio);
    EXPECT_EQ(restored.minInliers, defaultParams_.minInliers);
    EXPECT_DOUBLE_EQ(restored.earlyStopRatio, defaultParams_.earlyStopRatio);
    EXPECT_EQ(restored.useAMS, defaultParams_.useAMS);
}

TEST_F(EllipseFitCPUParamTest, JsonEmptyObjectGivesDefaults) {
    auto restored = EllipseFitCPUParams::fromJson(nlohmann::json{});
    EXPECT_EQ(restored.ransacIterations, 100);
    EXPECT_DOUBLE_EQ(restored.ransacThreshold, 0.5);
    EXPECT_TRUE(restored.useAMS);
}

TEST_F(EllipseFitCPUParamTest, JsonUnknownFieldsIgnored) {
    auto j = defaultParams_.toJson();
    j["unknownField"] = 42;
    auto restored = EllipseFitCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(restored.ransacThreshold, defaultParams_.ransacThreshold);
}

// ============================================================
// Result 结构体
// ============================================================
TEST_F(EllipseFitCPUParamTest, ResultDefaultValues) {
    EllipseFitCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_DOUBLE_EQ(result.centerX, 0.0);
    EXPECT_DOUBLE_EQ(result.centerY, 0.0);
    EXPECT_DOUBLE_EQ(result.majorAxis, 0.0);
    EXPECT_DOUBLE_EQ(result.minorAxis, 0.0);
    EXPECT_DOUBLE_EQ(result.angle, 0.0);
    EXPECT_EQ(result.inlierCount, 0);
    EXPECT_EQ(result.totalPointCount, 0);
    EXPECT_TRUE(result.inlierPoints.empty());
}

TEST_F(EllipseFitCPUParamTest, ResultMoveSemantics) {
    EllipseFitCPUResult r1;
    r1.success = true;
    r1.centerX = 100.0;
    r1.inlierCount = 50;
    r1.inlierPoints = {cv::Point2f(1, 2)};
    EllipseFitCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_DOUBLE_EQ(r2.centerX, 100.0);
    EXPECT_EQ(r2.inlierCount, 50);
}

// ============================================================
// 构造 / warmup
// ============================================================
TEST_F(EllipseFitCPUParamTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW(EllipseFitCPU op(defaultParams_));
}

TEST_F(EllipseFitCPUParamTest, WarmupWithMaxPointCount) {
    EllipseFitCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(500));
}

TEST_F(EllipseFitCPUParamTest, WarmupWithConfig) {
    EllipseFitCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(500)));
}

// ============================================================
// 拟合功能
// ============================================================
class EllipseFitCPUFitTest : public ::testing::Test {
protected:
    EllipseFitCPUParams params_;
    void SetUp() override {
        params_.ransacIterations = 200;
        params_.ransacThreshold = 0.5;
        params_.minInliers = 5;
    }
};

TEST_F(EllipseFitCPUFitTest, EmptyPointsReturnsFalse) {
    EllipseFitCPU op(params_);
    auto result = op.Execute(std::vector<cv::Point2f>{});
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.totalPointCount, 0);
}

TEST_F(EllipseFitCPUFitTest, TooFewPointsReturnsFalse) {
    EllipseFitCPU op(params_);
    std::vector<cv::Point2f> pts = {{0,0}, {1,1}, {2,2}, {3,3}};
    auto result = op.Execute(pts);
    EXPECT_FALSE(result.success);
}

TEST_F(EllipseFitCPUFitTest, PerfectCircle) {
    auto pts = generateEllipsePoints(100.0f, 100.0f, 15.0f, 15.0f, 0.0f, 60);
    EllipseFitCPU op(params_);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 100.0, 0.5);
    EXPECT_NEAR(result.centerY, 100.0, 0.5);
    EXPECT_NEAR(result.majorAxis, 30.0, 2.0);
    EXPECT_NEAR(result.minorAxis, 30.0, 2.0);
}

TEST_F(EllipseFitCPUFitTest, EllipseWithRotation) {
    auto pts = generateEllipsePoints(200.0f, 150.0f, 20.0f, 10.0f, 30.0f, 80);
    EllipseFitCPU op(params_);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 200.0, 1.0);
    EXPECT_NEAR(result.centerY, 150.0, 1.0);
}

TEST_F(EllipseFitCPUFitTest, NoisyEllipse) {
    auto pts = generateEllipsePoints(50.0f, 50.0f, 12.0f, 8.0f, 0.0f, 80, 0.3f);
    EllipseFitCPU op(params_);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 50.0, 1.0);
    EXPECT_NEAR(result.centerY, 50.0, 1.0);
}

TEST_F(EllipseFitCPUFitTest, EllipseWithOutliers) {
    auto pts = generateEllipsePoints(320.0f, 240.0f, 25.0f, 15.0f, 45.0f, 100);
    auto noisy = addOutliers(pts, 20, 320.0f, 240.0f, 60.0f);
    EllipseFitCPU op(params_);
    auto result = op.Execute(noisy);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 320.0, 2.0);
    EXPECT_NEAR(result.centerY, 240.0, 2.0);
    EXPECT_GT(result.inlierCount, 50);
}

TEST_F(EllipseFitCPUFitTest, EdgePointInput) {
    auto pts = generateEllipsePoints(100.0f, 100.0f, 15.0f, 15.0f, 0.0f, 60);
    std::vector<EdgePoint> edgePts;
    edgePts.reserve(pts.size());
    for (const auto& p : pts) {
        EdgePoint ep;
        ep.x = p.x; ep.y = p.y;
        ep.pixelX = static_cast<int>(p.x);
        ep.pixelY = static_cast<int>(p.y);
        edgePts.push_back(ep);
    }
    EllipseFitCPU op(params_);
    auto result = op.Execute(edgePts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 100.0, 0.5);
    EXPECT_NEAR(result.centerY, 100.0, 0.5);
}

TEST_F(EllipseFitCPUFitTest, Point2dInput) {
    auto ptsF = generateEllipsePoints(100.0f, 100.0f, 15.0f, 15.0f, 0.0f, 60);
    std::vector<cv::Point2d> ptsD;
    ptsD.reserve(ptsF.size());
    for (const auto& p : ptsF)
        ptsD.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y));
    EllipseFitCPU op(params_);
    auto result = op.Execute(ptsD);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 100.0, 0.5);
}

TEST_F(EllipseFitCPUFitTest, UseAMSFalse) {
    EllipseFitCPUParams p = params_;
    p.useAMS = false;
    auto pts = generateEllipsePoints(100.0f, 100.0f, 15.0f, 15.0f, 0.0f, 60);
    EllipseFitCPU op(p);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 100.0, 0.5);
}

TEST_F(EllipseFitCPUFitTest, LargeNumberOfPoints) {
    auto pts = generateEllipsePoints(500.0f, 500.0f, 100.0f, 80.0f, 0.0f, 5000);
    EllipseFitCPU op(params_);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 500.0, 1.0);
    EXPECT_NEAR(result.centerY, 500.0, 1.0);
    EXPECT_EQ(result.totalPointCount, 5000);
}

TEST_F(EllipseFitCPUFitTest, SmallEllipse) {
    auto pts = generateEllipsePoints(50.0f, 50.0f, 3.0f, 3.0f, 0.0f, 30);
    EllipseFitCPU op(params_);
    auto result = op.Execute(pts);
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.centerX, 50.0, 1.0);
    EXPECT_NEAR(result.centerY, 50.0, 1.0);
}

// ============================================================
// setParams / getParams
// ============================================================
class EllipseFitCPUSetParamsTest : public ::testing::Test {
protected:
    EllipseFitCPUParams params_;
};

TEST_F(EllipseFitCPUSetParamsTest, GetParamsReflectsConstructor) {
    EllipseFitCPU op(params_);
    EXPECT_EQ(op.GetParams().ransacIterations, params_.ransacIterations);
    EXPECT_DOUBLE_EQ(op.GetParams().ransacThreshold, params_.ransacThreshold);
}

TEST_F(EllipseFitCPUSetParamsTest, SetParamsUpdates) {
    EllipseFitCPU op(params_);
    EllipseFitCPUParams np;
    np.ransacIterations = 300;
    np.ransacThreshold = 0.3;
    op.SetParams(np);
    EXPECT_EQ(op.GetParams().ransacIterations, 300);
    EXPECT_DOUBLE_EQ(op.GetParams().ransacThreshold, 0.3);
}

TEST_F(EllipseFitCPUSetParamsTest, SetParamsInvalidThrows) {
    EllipseFitCPU op(params_);
    EllipseFitCPUParams bad;
    bad.ransacIterations = 0;
    EXPECT_THROW(op.SetParams(bad), std::invalid_argument);
}
