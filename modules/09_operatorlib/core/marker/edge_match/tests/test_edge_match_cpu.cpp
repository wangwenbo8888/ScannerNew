/**
 * @file test_edge_match_cpu.cpp
 * @brief 椭圆边界边缘点匹配算子 - 单元测试
 *
 * 测试覆盖：
 * 参数校验 / JSON序列化 / Result结构体 / 构造warmup /
 * 圆形匹配 / 椭圆匹配 / 旋转椭圆 / 空输入 /
 * 多椭圆对匹配 / Point2f输入 / 统计信息 /
 * setParams/getParams / 鲁棒性
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>

#include "../edge_match_cpu.h"
#include "epipolar_intersect_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


namespace {

EllipseIntersectResult makeCircleIntersect(double cx, double cy, double diameter,
                                            double step = 0.5) {
    EpipolarIntersectCPUParams params;
    params.epipolarStep = step;
    EpipolarIntersectCPU op(params);
    return op.Execute(cx, cy, diameter, diameter, 0.0);
}

EllipseIntersectResult makeEllipseIntersect(double cx, double cy,
                                             double major, double minor,
                                             double angleDeg = 0.0,
                                             double step = 0.5) {
    EpipolarIntersectCPUParams params;
    params.epipolarStep = step;
    EpipolarIntersectCPU op(params);
    return op.Execute(cx, cy, major, minor, angleDeg);
}

void pushBack(std::vector<EllipseIntersectResult>& v, EllipseIntersectResult&& r) {
    v.push_back(std::move(r));
}

} // anonymous namespace

// ============================================================
// 参数校验
// ============================================================
class EdgeMatchParamTest : public ::testing::Test {
protected:
    EdgeMatchCPUParams defaultParams_;
};

TEST_F(EdgeMatchParamTest, DefaultParamsPassValidate) {
    EXPECT_NO_THROW(defaultParams_.validate());
}

TEST_F(EdgeMatchParamTest, ZeroYToleranceThrows) {
    auto p = defaultParams_; p.yTolerance = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EdgeMatchParamTest, NegativeYToleranceThrows) {
    auto p = defaultParams_; p.yTolerance = -0.1f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EdgeMatchParamTest, ZeroDisparityMaxDiffThrows) {
    auto p = defaultParams_; p.disparityMaxDiff = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(EdgeMatchParamTest, ZeroMaxMatchPairsThrows) {
    auto p = defaultParams_; p.maxMatchPairs = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化
// ============================================================
TEST_F(EdgeMatchParamTest, JsonRoundtrip) {
    auto j = defaultParams_.toJson();
    auto restored = EdgeMatchCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_FLOAT_EQ(restored.yTolerance, defaultParams_.yTolerance);
    EXPECT_FLOAT_EQ(restored.disparityMaxDiff, defaultParams_.disparityMaxDiff);
    EXPECT_EQ(restored.maxMatchPairs, defaultParams_.maxMatchPairs);
}

TEST_F(EdgeMatchParamTest, JsonEmptyObjectGivesDefaults) {
    auto restored = EdgeMatchCPUParams::fromJson(nlohmann::json{});
    EXPECT_FLOAT_EQ(restored.yTolerance, 0.2f);
    EXPECT_FLOAT_EQ(restored.disparityMaxDiff, 10.0f);
    EXPECT_EQ(restored.maxMatchPairs, 100000u);
}

TEST_F(EdgeMatchParamTest, JsonUnknownFieldsIgnored) {
    auto j = defaultParams_.toJson();
    j["unknownField"] = 42;
    auto restored = EdgeMatchCPUParams::fromJson(j);
    EXPECT_FLOAT_EQ(restored.yTolerance, defaultParams_.yTolerance);
}

// ============================================================
// Result 结构体
// ============================================================
TEST_F(EdgeMatchParamTest, ResultDefaultValues) {
    EdgeMatchCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_TRUE(result.ellipseResults.empty());
}

TEST_F(EdgeMatchParamTest, ResultMoveSemantics) {
    EdgeMatchCPUResult r1;
    r1.success = true;
    EdgeMatchCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
}

TEST_F(EdgeMatchParamTest, MatchPairDefaults) {
    EdgeMatchPair pair;
    EXPECT_DOUBLE_EQ(pair.leftX, 0.0);
    EXPECT_DOUBLE_EQ(pair.rightX, 0.0);
    EXPECT_FLOAT_EQ(pair.disparity, 0.0f);
    EXPECT_EQ(pair.epipolarIndex, 0);
    EXPECT_EQ(pair.leftEllipseIdx, -1);
    EXPECT_EQ(pair.rightEllipseIdx, -1);
}

TEST_F(EdgeMatchParamTest, StatsDefaults) {
    EdgeMatchStats stats;
    EXPECT_DOUBLE_EQ(stats.totalTimeMs, 0.0);
    EXPECT_EQ(stats.matchedPairs, 0u);
    EXPECT_FLOAT_EQ(stats.matchRate, 0.0f);
}

// ============================================================
// 构造 / warmup
// ============================================================
TEST_F(EdgeMatchParamTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW(EdgeMatchCPU op(defaultParams_));
}

TEST_F(EdgeMatchParamTest, WarmupWithMaxCount) {
    EdgeMatchCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(100));
}

TEST_F(EdgeMatchParamTest, WarmupWithConfig) {
    EdgeMatchCPU op(defaultParams_);
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(100)));
}

// ============================================================
// 核心匹配测试
// ============================================================
class EdgeMatchTest : public ::testing::Test {
protected:
    EdgeMatchCPUParams params_;
};

TEST_F(EdgeMatchTest, CirclePerfectMatch) {
    double cx = 100.0, cy = 100.0;
    double diameter = 30.0;
    double baseline = 5.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(cx, cy, diameter));
    pushBack(rightVec, makeCircleIntersect(cx - baseline, cy, diameter));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.ellipseResults.size(), 1u);
    EXPECT_FALSE(result.ellipseResults[0].matchedPairs.empty());

    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        EXPECT_NEAR(pair.disparity, baseline, 0.1);
        EXPECT_GT(pair.confidence, 0.0f);
    }
}

TEST_F(EdgeMatchTest, EllipsePerfectMatch) {
    double cx = 200.0, cy = 150.0;
    double major = 40.0, minor = 20.0;
    double baseline = 8.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeEllipseIntersect(cx, cy, major, minor, 0.0));
    pushBack(rightVec, makeEllipseIntersect(cx - baseline, cy, major, minor, 0.0));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.ellipseResults[0].matchedPairs.empty());

    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        EXPECT_NEAR(pair.disparity, baseline, 0.1);
    }
}

TEST_F(EdgeMatchTest, RotatedEllipseMatch) {
    double cx = 300.0, cy = 250.0;
    double major = 40.0, minor = 20.0;
    double angle = 45.0;
    double baseline = 6.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeEllipseIntersect(cx, cy, major, minor, angle));
    pushBack(rightVec, makeEllipseIntersect(cx - baseline, cy, major, minor, angle));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.ellipseResults[0].matchedPairs.empty());

    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        EXPECT_NEAR(pair.disparity, baseline, 0.5);
    }
}

TEST_F(EdgeMatchTest, EmptyCenterMatches) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    std::vector<int> centerMatches;

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.ellipseResults.empty());
}

TEST_F(EdgeMatchTest, NoMatchedCenters) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(100.0, 100.0, 30.0));
    pushBack(rightVec, makeCircleIntersect(95.0, 100.0, 30.0));
    std::vector<int> centerMatches = {-1};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.ellipseResults.empty());
}

TEST_F(EdgeMatchTest, EmptyIntersectionPoints) {
    EllipseIntersectResult emptyLeft;
    emptyLeft.centerX = 100.0;
    emptyLeft.centerY = 100.0;

    EllipseIntersectResult emptyRight;
    emptyRight.centerX = 95.0;
    emptyRight.centerY = 100.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, std::move(emptyLeft));
    pushBack(rightVec, std::move(emptyRight));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.ellipseResults.empty());
}

// ============================================================
// 多椭圆对匹配
// ============================================================
TEST_F(EdgeMatchTest, MultipleEllipsePairs) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    std::vector<int> centerMatches;

    double baselines[] = {5.0, 8.0, 6.0};
    for (int i = 0; i < 3; ++i) {
        double cx = 100.0 + i * 200.0;
        double cy = 100.0 + i * 150.0;
        pushBack(leftVec, makeCircleIntersect(cx, cy, 30.0));
        pushBack(rightVec, makeCircleIntersect(cx - baselines[i], cy, 30.0));
        centerMatches.push_back(i);
    }

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.ellipseResults.size(), 3u);

    for (size_t i = 0; i < result.ellipseResults.size(); ++i) {
        EXPECT_FALSE(result.ellipseResults[i].matchedPairs.empty());
        for (const auto& pair : result.ellipseResults[i].matchedPairs) {
            EXPECT_NEAR(pair.disparity, baselines[i], 0.1)
                << "Ellipse pair " << i << " disparity mismatch";
        }
    }
}

TEST_F(EdgeMatchTest, PartialMatchedCenters) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;

    pushBack(leftVec, makeCircleIntersect(100.0, 100.0, 30.0));
    pushBack(leftVec, makeCircleIntersect(300.0, 300.0, 30.0));
    pushBack(leftVec, makeCircleIntersect(500.0, 500.0, 30.0));

    pushBack(rightVec, makeCircleIntersect(95.0, 100.0, 30.0));
    pushBack(rightVec, makeCircleIntersect(292.0, 300.0, 30.0));

    std::vector<int> centerMatches = {0, 1, -1};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.ellipseResults.size(), 2u);
}

// ============================================================
// 左右侧匹配验证
// ============================================================
TEST_F(EdgeMatchTest, LeftRightSideCorrect) {
    double cx = 100.0, cy = 100.0;
    double diameter = 40.0;
    double baseline = 5.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(cx, cy, diameter));
    pushBack(rightVec, makeCircleIntersect(cx - baseline, cy, diameter));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    ASSERT_FALSE(result.ellipseResults.empty());

    int side0Count = 0;
    int side1Count = 0;
    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        if (pair.side == 0) side0Count++;
        if (pair.side == 1) side1Count++;
    }

    EXPECT_GT(side0Count, 0);
    EXPECT_GT(side1Count, 0);
    EXPECT_NEAR(side0Count, side1Count, 3);
}

// ============================================================
// epipolarIndex 一致性
// ============================================================
TEST_F(EdgeMatchTest, EpipolarIndexConsistent) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(100.0, 100.0, 30.0));
    pushBack(rightVec, makeCircleIntersect(95.0, 100.0, 30.0));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    ASSERT_FALSE(result.ellipseResults.empty());

    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        EXPECT_EQ(pair.leftEllipseIdx, 0);
        EXPECT_EQ(pair.rightEllipseIdx, 0);
        EXPECT_GT(pair.epipolarIndex, 0);
    }
}

// ============================================================
// 统计信息
// ============================================================
TEST_F(EdgeMatchTest, StatisticsPopulated) {
    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(100.0, 100.0, 30.0));
    pushBack(rightVec, makeCircleIntersect(95.0, 100.0, 30.0));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_GT(result.statistics.totalTimeMs, 0.0);
    EXPECT_GT(result.statistics.matchedPairs, 0u);
    EXPECT_GT(result.statistics.avgConfidence, 0.0f);
}

// ============================================================
// Point2f 输入
// ============================================================
TEST_F(EdgeMatchTest, Point2fInput) {
    std::vector<cv::Point2f> leftPoints = {
        {95.0f, 90.0f}, {105.0f, 90.0f},
        {94.5f, 91.0f}, {105.5f, 91.0f},
        {95.0f, 110.0f}, {105.0f, 110.0f}
    };
    std::vector<cv::Point2f> rightPoints = {
        {90.0f, 90.0f}, {100.0f, 90.0f},
        {89.5f, 91.0f}, {100.5f, 91.0f},
        {90.0f, 110.0f}, {100.0f, 110.0f}
    };

    std::vector<int> leftGroupIds = {0, 0, 0, 0, 0, 0};
    std::vector<int> rightGroupIds = {0, 0, 0, 0, 0, 0};
    std::vector<int> centerMatches = {0};
    std::vector<int> leftEpiIdx = {180, 180, 182, 182, 220, 220};
    std::vector<int> rightEpiIdx = {180, 180, 182, 182, 220, 220};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftPoints, rightPoints, leftGroupIds, rightGroupIds,
             centerMatches, leftEpiIdx, rightEpiIdx);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.ellipseResults.empty());
    EXPECT_EQ(result.ellipseResults[0].matchedPairs.size(), 6u);
}

// ============================================================
// setParams / getParams
// ============================================================
class EdgeMatchSetParamsTest : public ::testing::Test {
protected:
    EdgeMatchCPUParams params_;
};

TEST_F(EdgeMatchSetParamsTest, GetParamsReflectsConstructor) {
    EdgeMatchCPU op(params_);
    EXPECT_FLOAT_EQ(op.GetParams().yTolerance, params_.yTolerance);
}

TEST_F(EdgeMatchSetParamsTest, SetParamsUpdates) {
    EdgeMatchCPU op(params_);
    EdgeMatchCPUParams np;
    np.yTolerance = 0.1f;
    op.SetParams(np);
    EXPECT_FLOAT_EQ(op.GetParams().yTolerance, 0.1f);
}

TEST_F(EdgeMatchSetParamsTest, SetParamsInvalidThrows) {
    EdgeMatchCPU op(params_);
    EdgeMatchCPUParams bad;
    bad.yTolerance = 0.0f;
    EXPECT_THROW(op.SetParams(bad), std::invalid_argument);
}

TEST_F(EdgeMatchSetParamsTest, ResetStatistics) {
    EdgeMatchCPU op(params_);
    op.ResetStatistics();
    EXPECT_DOUBLE_EQ(op.GetStatistics().totalTimeMs, 0.0);
    EXPECT_EQ(op.GetStatistics().matchedPairs, 0u);
}

// ============================================================
// 大椭圆 / 边界
// ============================================================
TEST_F(EdgeMatchTest, LargeEllipse) {
    double cx = 500.0, cy = 500.0;
    double baseline = 10.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeEllipseIntersect(cx, cy, 200.0, 100.0, 0.0, 1.0));
    pushBack(rightVec, makeEllipseIntersect(cx - baseline, cy, 200.0, 100.0, 0.0, 1.0));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.ellipseResults[0].matchedPairs.size(), 0u);

    for (const auto& pair : result.ellipseResults[0].matchedPairs) {
        EXPECT_NEAR(pair.disparity, baseline, 1.0);
    }
}

TEST_F(EdgeMatchTest, SmallEllipse) {
    double cx = 50.0, cy = 50.0;
    double baseline = 2.0;

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, makeCircleIntersect(cx, cy, 6.0, 0.1));
    pushBack(rightVec, makeCircleIntersect(cx - baseline, cy, 6.0, 0.1));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.ellipseResults[0].matchedPairs.size(), 0u);
}

TEST_F(EdgeMatchTest, NonOverlappingEpipolarLines) {
    EllipseIntersectResult leftE;
    leftE.centerX = 100.0;
    leftE.centerY = 100.0;
    EpipolarIntersectPoint lp1;
    lp1.x = 90.0; lp1.y = 85.0; lp1.yEpipolar = 85.0; lp1.epipolarIndex = 170;
    EpipolarIntersectPoint lp2;
    lp2.x = 110.0; lp2.y = 85.0; lp2.yEpipolar = 85.0; lp2.epipolarIndex = 170;
    leftE.intersectPts.push_back(lp1);
    leftE.intersectPts.push_back(lp2);

    EllipseIntersectResult rightE;
    rightE.centerX = 95.0;
    rightE.centerY = 200.0;
    EpipolarIntersectPoint rp1;
    rp1.x = 85.0; rp1.y = 185.0; rp1.yEpipolar = 185.0; rp1.epipolarIndex = 370;
    EpipolarIntersectPoint rp2;
    rp2.x = 105.0; rp2.y = 185.0; rp2.yEpipolar = 185.0; rp2.epipolarIndex = 370;
    rightE.intersectPts.push_back(rp1);
    rightE.intersectPts.push_back(rp2);

    std::vector<EllipseIntersectResult> leftVec;
    std::vector<EllipseIntersectResult> rightVec;
    pushBack(leftVec, std::move(leftE));
    pushBack(rightVec, std::move(rightE));
    std::vector<int> centerMatches = {0};

    EdgeMatchCPU op(params_);
    auto result = op.Execute(leftVec, rightVec, centerMatches);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.ellipseResults.empty());
}
