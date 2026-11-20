/**
 * @file test_epipolar_intersect_cpu.cpp
 * @brief 椭圆边界极线交点算子 - 单元测试
 *
 * 测试覆盖：
 * 参数校验 / JSON序列化 / Result结构体 / 圆形交点 / 椭圆交点 / 旋转椭圆 /
 * 批量计算 / RotatedRect输入 / 空椭圆 / 极线间距 / 排序验证 /
 * setParams/getParams / warmup / EllipseFitCPUResult输入
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

#include "../epipolar_intersect_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


namespace {

bool pointOnEllipse(double x, double y, double cx, double cy,
                    double majorAxis, double minorAxis, double angleDeg,
                    double tol = 1e-6)
{
    double a = majorAxis / 2.0;
    double b = minorAxis / 2.0;
    double rad = angleDeg * CV_PI / 180.0;
    double cosA = std::cos(rad);
    double sinA = std::sin(rad);
    double dx = x - cx;
    double dy = y - cy;
    double xr = dx * cosA + dy * sinA;
    double yr = -dx * sinA + dy * cosA;
    double val = (xr * xr) / (a * a) + (yr * yr) / (b * b);
    return std::abs(val - 1.0) < tol;
}

} // anonymous namespace

// ============================================================
// 参数校验
// ============================================================
class EpipolarParamTest : public ::testing::Test {
protected:
    EpipolarIntersectCPUParams defaultParams_;
};

TEST_F(EpipolarParamTest, DefaultParamsPassValidate) {
    EXPECT_NO_THROW(defaultParams_.validate());
}

TEST_F(EpipolarParamTest, ZeroStepThrows) {
    auto p = defaultParams_; p.epipolarStep = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EpipolarParamTest, NegativeStepThrows) {
    auto p = defaultParams_; p.epipolarStep = -0.5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EpipolarParamTest, ZeroMaxIntersectionsThrows) {
    auto p = defaultParams_; p.maxIntersectionsPerEllipse = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化
// ============================================================
TEST_F(EpipolarParamTest, JsonRoundtrip) {
    auto j = defaultParams_.toJson();
    auto restored = EpipolarIntersectCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_DOUBLE_EQ(restored.epipolarStep, defaultParams_.epipolarStep);
    EXPECT_EQ(restored.maxIntersectionsPerEllipse, defaultParams_.maxIntersectionsPerEllipse);
}

TEST_F(EpipolarParamTest, JsonEmptyObjectGivesDefaults) {
    auto restored = EpipolarIntersectCPUParams::fromJson(nlohmann::json{});
    EXPECT_DOUBLE_EQ(restored.epipolarStep, 0.5);
    EXPECT_EQ(restored.maxIntersectionsPerEllipse, 1000);
}

TEST_F(EpipolarParamTest, JsonUnknownFieldsIgnored) {
    auto j = defaultParams_.toJson();
    j["unknownField"] = 42;
    auto restored = EpipolarIntersectCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(restored.epipolarStep, defaultParams_.epipolarStep);
}

// ============================================================
// Result 结构体
// ============================================================
TEST_F(EpipolarParamTest, ResultDefaultValues) {
    EpipolarIntersectCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_TRUE(result.ellipseResults.empty());
}

TEST_F(EpipolarParamTest, EllipseResultDefaultValues) {
    EllipseIntersectResult r;
    EXPECT_DOUBLE_EQ(r.centerX, 0.0);
    EXPECT_DOUBLE_EQ(r.centerY, 0.0);
    EXPECT_TRUE(r.intersectPts.empty());
}

TEST_F(EpipolarParamTest, ResultMoveSemantics) {
    EpipolarIntersectCPUResult r1;
    r1.success = true;
    EpipolarIntersectCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
}

// ============================================================
// 构造 / warmup
// ============================================================
TEST_F(EpipolarParamTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW(EpipolarIntersectCPU op(defaultParams_));
}

TEST_F(EpipolarParamTest, WarmupWithMaxCount) {
    EpipolarIntersectCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(100));
}

TEST_F(EpipolarParamTest, WarmupWithConfig) {
    EpipolarIntersectCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(100)));
}

// ============================================================
// 核心功能测试
// ============================================================
class EpipolarIntersectTest : public ::testing::Test {
protected:
    EpipolarIntersectCPUParams params_;
    void SetUp() override {}
};

TEST_F(EpipolarIntersectTest, CircleCentered) {
    double cx = 100.0, cy = 100.0;
    double diameter = 30.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, diameter, diameter, 0.0);

    EXPECT_DOUBLE_EQ(result.centerX, cx);
    EXPECT_DOUBLE_EQ(result.centerY, cy);
    EXPECT_FALSE(result.intersectPts.empty());

    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, cx, cy, diameter, diameter, 0.0))
            << "Point (" << pt.x << "," << pt.y << ") not on ellipse";
    }
}

TEST_F(EpipolarIntersectTest, CircleCorrectCount) {
    double cx = 100.3, cy = 100.3;
    double diameter = 20.0;
    double radius = diameter / 2.0;
    EpipolarIntersectCPU op(params_);

    auto result = op.Execute(cx, cy, diameter, diameter, 0.0);

    int expectedLines = static_cast<int>(std::floor((cy + radius) / params_.epipolarStep))
                      - static_cast<int>(std::ceil((cy - radius) / params_.epipolarStep))
                      + 1;

    int expectedPts = expectedLines * 2;
    int actualPts = static_cast<int>(result.intersectPts.size());
    EXPECT_EQ(actualPts, expectedPts)
        << "Expected " << expectedPts << " points for " << expectedLines << " epipolar lines";
}

TEST_F(EpipolarIntersectTest, EllipseNoRotation) {
    double cx = 200.0, cy = 150.0;
    double major = 40.0, minor = 20.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, major, minor, 0.0);

    EXPECT_FALSE(result.intersectPts.empty());
    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, cx, cy, major, minor, 0.0, 1e-5));
    }
}

TEST_F(EpipolarIntersectTest, EllipseWithRotation) {
    double cx = 300.0, cy = 250.0;
    double major = 40.0, minor = 20.0;
    double angle = 45.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, major, minor, angle);

    EXPECT_FALSE(result.intersectPts.empty());
    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, cx, cy, major, minor, angle, 1e-5))
            << "Point (" << pt.x << "," << pt.y << ") not on rotated ellipse";
    }
}

TEST_F(EpipolarIntersectTest, AllPointsOnEpipolarLine) {
    double cx = 100.0, cy = 100.0;
    double major = 30.0, minor = 20.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, major, minor, 30.0);

    for (const auto& pt : result.intersectPts) {
        EXPECT_DOUBLE_EQ(pt.y, pt.yEpipolar);
        EXPECT_DOUBLE_EQ(pt.y, pt.epipolarIndex * params_.epipolarStep);
    }
}

TEST_F(EpipolarIntersectTest, PointsSortedByEpipolarIndexThenX) {
    double cx = 100.0, cy = 100.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, 30.0, 20.0, 0.0);

    for (size_t i = 1; i < result.intersectPts.size(); ++i) {
        const auto& prev = result.intersectPts[i - 1];
        const auto& curr = result.intersectPts[i];
        if (prev.epipolarIndex == curr.epipolarIndex) {
            EXPECT_LE(prev.x, curr.x) << "X not sorted within same epipolar line";
        } else {
            EXPECT_LT(prev.epipolarIndex, curr.epipolarIndex);
        }
    }
}

TEST_F(EpipolarIntersectTest, CircleAllPointsOnEllipseAndEvenCount) {
    double cx = 100.0, cy = 100.0;
    double diameter = 19.0;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(cx, cy, diameter, diameter, 0.0);

    int n = static_cast<int>(result.intersectPts.size());
    EXPECT_GT(n, 0);

    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, cx, cy, diameter, diameter, 0.0))
            << "Point (" << pt.x << "," << pt.y << ") not on circle";
    }

    int pairs = 0;
    int i = 0;
    while (i < n) {
        double yEpi = result.intersectPts[i].yEpipolar;
        int count = 0;
        while (i < n && result.intersectPts[i].yEpipolar == yEpi) {
            ++count;
            ++i;
        }
        EXPECT_TRUE(count == 1 || count == 2)
            << "Line y=" << yEpi << " has " << count << " intersections";
        if (count == 2) {
            double sumX = result.intersectPts[i-2].x + result.intersectPts[i-1].x;
            EXPECT_NEAR(sumX, 2.0 * cx, 1e-10)
                << "Pair on y=" << yEpi << " should be symmetric around cx";
            ++pairs;
        }
    }
    EXPECT_GT(pairs, 0) << "Should have at least one pair of intersections";
}

TEST_F(EpipolarIntersectTest, SmallStepMorePoints) {
    double cx = 100.0, cy = 100.0;
    double diameter = 20.0;

    EpipolarIntersectCPUParams coarse;
    coarse.epipolarStep = 1.0;
    EpipolarIntersectCPU opCoarse(coarse);
    auto resCoarse = opCoarse.Execute(cx, cy, diameter, diameter, 0.0);

    EpipolarIntersectCPUParams fine;
    fine.epipolarStep = 0.25;
    EpipolarIntersectCPU opFine(fine);
    auto resFine = opFine.Execute(cx, cy, diameter, diameter, 0.0);

    EXPECT_GT(resFine.intersectPts.size(), resCoarse.intersectPts.size());
}

TEST_F(EpipolarIntersectTest, ZeroAxisNoPoints) {
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(100.0, 100.0, 0.0, 0.0, 0.0);
    EXPECT_TRUE(result.intersectPts.empty());
}

TEST_F(EpipolarIntersectTest, RotatedRectInput) {
    cv::RotatedRect ellipse(cv::Point2f(100.0f, 100.0f), cv::Size2f(30.0f, 20.0f), 0.0f);
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(ellipse);
    EXPECT_FALSE(result.intersectPts.empty());
    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, 100.0, 100.0, 30.0, 20.0, 0.0, 1e-4));
    }
}

// ============================================================
// EllipseFitCPUResult 输入
// ============================================================
TEST_F(EpipolarIntersectTest, EllipseFitResultInput) {
    EllipseFitCPUResult fitResult;
    fitResult.success = true;
    fitResult.centerX = 100.0;
    fitResult.centerY = 100.0;
    fitResult.majorAxis = 30.0;
    fitResult.minorAxis = 20.0;
    fitResult.angle = 0.0;

    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(fitResult);

    EXPECT_DOUBLE_EQ(result.centerX, 100.0);
    EXPECT_DOUBLE_EQ(result.centerY, 100.0);
    EXPECT_FALSE(result.intersectPts.empty());
}

// ============================================================
// 批量测试
// ============================================================
TEST_F(EpipolarIntersectTest, BatchRotatedRect) {
    std::vector<cv::RotatedRect> ellipses = {
        cv::RotatedRect(cv::Point2f(100, 100), cv::Size2f(30, 20), 0),
        cv::RotatedRect(cv::Point2f(200, 200), cv::Size2f(40, 30), 45),
        cv::RotatedRect(cv::Point2f(300, 300), cv::Size2f(20, 20), 90)
    };

    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(ellipses);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.ellipseResults.size(), 3u);
    for (size_t i = 0; i < result.ellipseResults.size(); ++i) {
        EXPECT_FALSE(result.ellipseResults[i].intersectPts.empty())
            << "Ellipse " << i << " should have intersection points";
    }
}

TEST_F(EpipolarIntersectTest, BatchEllipseFitResult) {
    std::vector<EllipseFitCPUResult> fitResults(3);
    fitResults[0].centerX = 100.0; fitResults[0].centerY = 100.0;
    fitResults[0].majorAxis = 30.0; fitResults[0].minorAxis = 20.0; fitResults[0].angle = 0.0;

    fitResults[1].centerX = 200.0; fitResults[1].centerY = 200.0;
    fitResults[1].majorAxis = 40.0; fitResults[1].minorAxis = 20.0; fitResults[1].angle = 30.0;

    fitResults[2].centerX = 300.0; fitResults[2].centerY = 300.0;
    fitResults[2].majorAxis = 20.0; fitResults[2].minorAxis = 20.0; fitResults[2].angle = 0.0;

    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(fitResults);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.ellipseResults.size(), 3u);
}

TEST_F(EpipolarIntersectTest, BatchEmpty) {
    std::vector<cv::RotatedRect> ellipses;
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(ellipses);
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.ellipseResults.empty());
}

// ============================================================
// setParams / getParams
// ============================================================
class EpipolarSetParamsTest : public ::testing::Test {
protected:
    EpipolarIntersectCPUParams params_;
};

TEST_F(EpipolarSetParamsTest, GetParamsReflectsConstructor) {
    EpipolarIntersectCPU op(params_);
    EXPECT_DOUBLE_EQ(op.GetParams().epipolarStep, params_.epipolarStep);
}

TEST_F(EpipolarSetParamsTest, SetParamsUpdates) {
    EpipolarIntersectCPU op(params_);
    EpipolarIntersectCPUParams np;
    np.epipolarStep = 0.25;
    op.SetParams(np);
    EXPECT_DOUBLE_EQ(op.GetParams().epipolarStep, 0.25);
}

TEST_F(EpipolarSetParamsTest, SetParamsInvalidThrows) {
    EpipolarIntersectCPU op(params_);
    EpipolarIntersectCPUParams bad;
    bad.epipolarStep = 0.0;
    EXPECT_THROW(op.SetParams(bad), std::invalid_argument);
}

// ============================================================
// 大椭圆 / 边界
// ============================================================
TEST_F(EpipolarIntersectTest, LargeEllipse) {
    EpipolarIntersectCPU op(params_);
    auto result = op.Execute(500.0, 500.0, 200.0, 100.0, 0.0);
    EXPECT_FALSE(result.intersectPts.empty());
    for (const auto& pt : result.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, 500.0, 500.0, 200.0, 100.0, 0.0, 1e-4));
    }
}

TEST_F(EpipolarIntersectTest, Ellipse90DegreeRotation) {
    double cx = 100.0, cy = 100.0;
    double major = 40.0, minor = 20.0;
    EpipolarIntersectCPU op(params_);
    auto result0 = op.Execute(cx, cy, major, minor, 0.0);
    auto result90 = op.Execute(cx, cy, major, minor, 90.0);

    EXPECT_FALSE(result0.intersectPts.empty());
    EXPECT_FALSE(result90.intersectPts.empty());

    for (const auto& pt : result90.intersectPts) {
        EXPECT_TRUE(pointOnEllipse(pt.x, pt.y, cx, cy, major, minor, 90.0, 1e-4));
    }
}

TEST_F(EpipolarIntersectTest, VerySmallEllipse) {
    EpipolarIntersectCPUParams fineParams;
    fineParams.epipolarStep = 0.1;
    EpipolarIntersectCPU op(fineParams);
    auto result = op.Execute(50.0, 50.0, 6.0, 4.0, 0.0);
    EXPECT_FALSE(result.intersectPts.empty());
}
