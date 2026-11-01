/**
 * @file test_point_reconstruct_cpu.cpp
 * @brief 标记点法线和中心快速三维重建算子 - 单元测试
 *
 * 测试覆盖：
 * 参数校验 / JSON序列化 / Result结构体 / 构造warmup /
 * 平面拟合 / 圆拟合 / 端到端重建 / 空输入 /
 * 多标记点 / 统计信息 / setParams/getParams
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <numeric>

#include "../point_reconstruct_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;


namespace {

PointReconstructCPUParams makeValidParams() {
    PointReconstructCPUParams p;
    p.fxLeft = 2500.0;
    p.fyLeft = 2500.0;
    p.cxLeft = 640.0;
    p.cyLeft = 512.0;
    p.fxRight = 2500.0;
    p.fyRight = 2500.0;
    p.cxRight = 640.0;
    p.cyRight = 512.0;
    p.R = cv::Matx33d::eye();
    p.T = cv::Vec3d(120.0, 0.0, 0.0);
    p.minPointsForPlaneFit = 6;
    p.minPointsForCircleFit = 6;
    p.maxReprojError = 2.0;
    p.maxMarkerCount = 1000;
    return p;
}

struct Circle3D {
    cv::Vec3d center;
    cv::Vec3d normal;
    double radius;
};

void generateCirclePoints3D(const Circle3D& circle, int numPoints,
                            std::vector<double>& xs,
                            std::vector<double>& ys,
                            std::vector<double>& zs) {
    cv::Vec3d n = circle.normal;
    double len = std::sqrt(n(0)*n(0) + n(1)*n(1) + n(2)*n(2));
    n = n / len;

    cv::Vec3d ref(1.0, 0.0, 0.0);
    if (std::abs(n.dot(ref)) > 0.9) ref = cv::Vec3d(0.0, 1.0, 0.0);

    cv::Vec3d u = ref - n * n.dot(ref);
    u = u / std::sqrt(u.dot(u));
    cv::Vec3d v = n.cross(u);

    xs.clear(); ys.clear(); zs.clear();
    xs.reserve(numPoints);
    ys.reserve(numPoints);
    zs.reserve(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        double angle = 2.0 * CV_PI * i / numPoints;
        cv::Vec3d pt = circle.center + circle.radius * (std::cos(angle) * u + std::sin(angle) * v);
        xs.push_back(pt(0));
        ys.push_back(pt(1));
        zs.push_back(pt(2));
    }
}

void projectToCamera(double x, double y, double z,
                     double fx, double fy, double cx, double cy,
                     const cv::Matx33d& R, const cv::Vec3d& T,
                     double& u, double& v) {
    double rx = R(0,0)*x + R(0,1)*y + R(0,2)*z + T(0);
    double ry = R(1,0)*x + R(1,1)*y + R(1,2)*z + T(1);
    double rz = R(2,0)*x + R(2,1)*y + R(2,2)*z + T(2);
    u = fx * rx / rz + cx;
    v = fy * ry / rz + cy;
}

void generateSyntheticMatchData(const Circle3D& circle, int numPoints,
                                const PointReconstructCPUParams& params,
                                std::vector<cv::Point2f>& leftPts,
                                std::vector<cv::Point2f>& rightPts,
                                std::vector<int>& leftGroups,
                                std::vector<int>& rightGroups,
                                std::vector<int>& centerMatches) {
    std::vector<double> xs, ys, zs;
    generateCirclePoints3D(circle, numPoints, xs, ys, zs);

    leftPts.clear(); rightPts.clear();
    leftGroups.clear(); rightGroups.clear();
    centerMatches = {0};

    for (int i = 0; i < numPoints; ++i) {
        double uL, vL, uR, vR;
        projectToCamera(xs[i], ys[i], zs[i],
                        params.fxLeft, params.fyLeft, params.cxLeft, params.cyLeft,
                        cv::Matx33d::eye(), cv::Vec3d(0,0,0), uL, vL);
        projectToCamera(xs[i], ys[i], zs[i],
                        params.fxRight, params.fyRight, params.cxRight, params.cyRight,
                        params.R, params.T, uR, vR);
        leftPts.emplace_back(static_cast<float>(uL), static_cast<float>(vL));
        rightPts.emplace_back(static_cast<float>(uR), static_cast<float>(vR));
        leftGroups.push_back(0);
        rightGroups.push_back(0);
    }
}

}

// ============================================================
// 参数校验
// ============================================================
class PointReconstructParamTest : public ::testing::Test {
protected:
    PointReconstructCPUParams defaultParams_;
};

TEST_F(PointReconstructParamTest, ValidParamsPassValidate) {
    auto p = makeValidParams();
    EXPECT_NO_THROW(p.validate());
}

TEST_F(PointReconstructParamTest, ZeroFxLeftThrows) {
    auto p = makeValidParams(); p.fxLeft = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, NegativeFyLeftThrows) {
    auto p = makeValidParams(); p.fyLeft = -1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, ZeroFxRightThrows) {
    auto p = makeValidParams(); p.fxRight = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, ZeroFyRightThrows) {
    auto p = makeValidParams(); p.fyRight = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, MinPlaneFitLessThan3Throws) {
    auto p = makeValidParams(); p.minPointsForPlaneFit = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, MinCircleFitLessThan3Throws) {
    auto p = makeValidParams(); p.minPointsForCircleFit = 1;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, ZeroMaxReprojErrorThrows) {
    auto p = makeValidParams(); p.maxReprojError = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, ZeroMaxMarkerCountThrows) {
    auto p = makeValidParams(); p.maxMarkerCount = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST_F(PointReconstructParamTest, InvalidRotationMatrixThrows) {
    auto p = makeValidParams();
    p.R = cv::Matx33d::zeros();
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// JSON 序列化
// ============================================================
TEST_F(PointReconstructParamTest, JsonRoundtrip) {
    auto p = makeValidParams();
    auto j = p.toJson();
    auto restored = PointReconstructCPUParams::fromJson(j);
    EXPECT_NO_THROW(restored.validate());
    EXPECT_DOUBLE_EQ(restored.fxLeft, p.fxLeft);
    EXPECT_DOUBLE_EQ(restored.fyLeft, p.fyLeft);
    EXPECT_DOUBLE_EQ(restored.cxLeft, p.cxLeft);
    EXPECT_DOUBLE_EQ(restored.cyLeft, p.cyLeft);
    EXPECT_DOUBLE_EQ(restored.fxRight, p.fxRight);
    EXPECT_DOUBLE_EQ(restored.fyRight, p.fyRight);
    EXPECT_DOUBLE_EQ(restored.cxRight, p.cxRight);
    EXPECT_DOUBLE_EQ(restored.cyRight, p.cyRight);
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            EXPECT_DOUBLE_EQ(restored.R(i,k), p.R(i,k));
    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(restored.T(i), p.T(i));
    EXPECT_EQ(restored.minPointsForPlaneFit, p.minPointsForPlaneFit);
    EXPECT_EQ(restored.minPointsForCircleFit, p.minPointsForCircleFit);
    EXPECT_DOUBLE_EQ(restored.maxReprojError, p.maxReprojError);
    EXPECT_EQ(restored.maxMarkerCount, p.maxMarkerCount);
}

TEST_F(PointReconstructParamTest, JsonEmptyObjectGivesDefaults) {
    auto restored = PointReconstructCPUParams::fromJson(nlohmann::json{});
    EXPECT_DOUBLE_EQ(restored.fxLeft, 0.0);
    EXPECT_DOUBLE_EQ(restored.fxRight, 0.0);
    EXPECT_EQ(restored.minPointsForPlaneFit, 6);
    EXPECT_EQ(restored.minPointsForCircleFit, 6);
    EXPECT_DOUBLE_EQ(restored.maxReprojError, 2.0);
    EXPECT_EQ(restored.maxMarkerCount, 1000u);
}

TEST_F(PointReconstructParamTest, JsonUnknownFieldsIgnored) {
    auto p = makeValidParams();
    auto j = p.toJson();
    j["unknownField"] = 42;
    auto restored = PointReconstructCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(restored.fxLeft, p.fxLeft);
}

TEST_F(PointReconstructParamTest, JsonPartialFieldsGivesPartial) {
    nlohmann::json j;
    j["fxLeft"] = 3000.0;
    j["T"] = nlohmann::json::array({100.0, 0.0, 0.0});
    auto restored = PointReconstructCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(restored.fxLeft, 3000.0);
    EXPECT_DOUBLE_EQ(restored.T(0), 100.0);
    EXPECT_DOUBLE_EQ(restored.fyLeft, 0.0);
}

// ============================================================
// Result 结构体
// ============================================================
TEST_F(PointReconstructParamTest, ResultDefaultValues) {
    PointReconstructCPUResult result;
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Normal);
    EXPECT_TRUE(result.markerResults.empty());
}

TEST_F(PointReconstructParamTest, ResultMoveSemantics) {
    PointReconstructCPUResult r1;
    r1.success = true;
    r1.message = "test";
    PointReconstructCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.message, "test");
}

TEST_F(PointReconstructParamTest, MarkerResultDefaults) {
    MarkerReconstructResult mr;
    EXPECT_EQ(mr.leftEllipseIdx, -1);
    EXPECT_EQ(mr.rightEllipseIdx, -1);
    EXPECT_FALSE(mr.validPlane);
    EXPECT_FALSE(mr.validCircle);
    EXPECT_DOUBLE_EQ(mr.centerX, 0.0);
    EXPECT_DOUBLE_EQ(mr.normalX, 0.0);
    EXPECT_TRUE(mr.reconstructedPoints.empty());
}

TEST_F(PointReconstructParamTest, MarkerResultMoveSemantics) {
    MarkerReconstructResult r1;
    r1.centerX = 42.0;
    MarkerReconstructResult r2 = std::move(r1);
    EXPECT_DOUBLE_EQ(r2.centerX, 42.0);
}

TEST_F(PointReconstructParamTest, ReconstructedPointDefaults) {
    ReconstructedPoint3D pt;
    EXPECT_DOUBLE_EQ(pt.x, 0.0);
    EXPECT_DOUBLE_EQ(pt.reprojError, 0.0f);
}

TEST_F(PointReconstructParamTest, PlaneFitResultDefaults) {
    PlaneFitResult pf;
    EXPECT_DOUBLE_EQ(pf.nx, 0.0);
    EXPECT_DOUBLE_EQ(pf.fitError, 0.0);
    EXPECT_DOUBLE_EQ(pf.singularValues[0], 0.0);
}

TEST_F(PointReconstructParamTest, CircleFitResultDefaults) {
    CircleFitResult cf;
    EXPECT_DOUBLE_EQ(cf.radius, 0.0);
    EXPECT_DOUBLE_EQ(cf.fitError, 0.0);
}

TEST_F(PointReconstructParamTest, StatsDefaults) {
    PointReconstructStats stats;
    EXPECT_DOUBLE_EQ(stats.totalTimeMs, 0.0);
    EXPECT_EQ(stats.totalMarkerPairs, 0u);
    EXPECT_EQ(stats.validMarkerCount, 0u);
}

// ============================================================
// 构造 / warmup
// ============================================================
TEST_F(PointReconstructParamTest, ConstructWithDefaultParams) {
    EXPECT_NO_THROW(PointReconstructCPU op);
}

TEST_F(PointReconstructParamTest, ConstructWithValidParams) {
    auto p = makeValidParams();
    EXPECT_NO_THROW(PointReconstructCPU op(p));
}

TEST_F(PointReconstructParamTest, WarmupWithMaxCount) {
    PointReconstructCPU op;
    EXPECT_NO_THROW(op.Warmup(100));
}

TEST_F(PointReconstructParamTest, WarmupWithConfig) {
    PointReconstructCPU op;
    EXPECT_NO_THROW(op.Warmup(calib::WarmupConfig::forPointCloud(100)));
}

// ============================================================
// setParams / getParams
// ============================================================
class PointReconstructSetParamsTest : public ::testing::Test {
protected:
    PointReconstructCPUParams params_;
};

TEST_F(PointReconstructSetParamsTest, GetParamsReflectsDefault) {
    PointReconstructCPU op;
    EXPECT_DOUBLE_EQ(op.GetParams().fxLeft, 0.0);
}

TEST_F(PointReconstructSetParamsTest, SetParamsUpdates) {
    PointReconstructCPU op;
    auto p = makeValidParams();
    p.fxLeft = 3000.0;
    op.SetParams(p);
    EXPECT_DOUBLE_EQ(op.GetParams().fxLeft, 3000.0);
}

TEST_F(PointReconstructSetParamsTest, SetParamsInvalidThrows) {
    PointReconstructCPU op;
    PointReconstructCPUParams bad;
    bad.fxLeft = -1.0;
    EXPECT_THROW(op.SetParams(bad), std::invalid_argument);
}

TEST_F(PointReconstructSetParamsTest, ResetStatistics) {
    PointReconstructCPU op;
    op.ResetStatistics();
    EXPECT_DOUBLE_EQ(op.GetStatistics().totalTimeMs, 0.0);
    EXPECT_EQ(op.GetStatistics().validMarkerCount, 0u);
}

// ============================================================
// 空输入
// ============================================================
class PointReconstructTest : public ::testing::Test {
protected:
    PointReconstructCPUParams params_ = makeValidParams();
};

TEST_F(PointReconstructTest, EmptyEdgeMatchResult) {
    EdgeMatchCPUResult emptyMatch;
    emptyMatch.success = true;

    PointReconstructCPU op(params_);
    auto result = op.Execute(emptyMatch);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.markerResults.empty());
}

TEST_F(PointReconstructTest, EmptyPoint2fInput) {
    std::vector<cv::Point2f> left, right;
    std::vector<int> lg, rg, cm;

    PointReconstructCPU op(params_);
    auto result = op.Execute(left, right, lg, rg, cm);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.markerResults.empty());
}

TEST_F(PointReconstructTest, EmptyCenterMatches) {
    std::vector<cv::Point2f> left = {{100, 100}}, right = {{95, 100}};
    std::vector<int> lg = {0}, rg = {0};
    std::vector<int> cm;

    PointReconstructCPU op(params_);
    auto result = op.Execute(left, right, lg, rg, cm);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.markerResults.empty());
}

// ============================================================
// 端到端重建 - 正面圆 (法线朝Z轴)
// ============================================================
TEST_F(PointReconstructTest, FrontalCircleReconstruction) {
    Circle3D circle;
    circle.center = cv::Vec3d(0.0, 0.0, 500.0);
    circle.normal = cv::Vec3d(0.0, 0.0, 1.0);
    circle.radius = 10.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 36, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, leftGroups, rightGroups,
                             centerMatches);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.markerResults.size(), 1u);

    const auto& mr = result.markerResults[0];
    EXPECT_TRUE(mr.validPlane);
    EXPECT_TRUE(mr.validCircle);

    EXPECT_NEAR(mr.centerX, circle.center(0), 0.5);
    EXPECT_NEAR(mr.centerY, circle.center(1), 0.5);
    EXPECT_NEAR(mr.centerZ, circle.center(2), 1.0);

    cv::Vec3d reconstructedNormal(mr.normalX, mr.normalY, mr.normalZ);
    double nLen = std::sqrt(reconstructedNormal.dot(reconstructedNormal));
    EXPECT_NEAR(nLen, 1.0, 1e-6);

    cv::Vec3d expectedNormal = circle.normal / std::sqrt(circle.normal.dot(circle.normal));
    double dotProduct = std::abs(reconstructedNormal.dot(expectedNormal));
    EXPECT_NEAR(dotProduct, 1.0, 0.01);

    EXPECT_NEAR(mr.circleFit.radius, circle.radius, 0.5);
}

// ============================================================
// 端到端重建 - 倾斜圆
// ============================================================
TEST_F(PointReconstructTest, TiltedCircleReconstruction) {
    Circle3D circle;
    circle.center = cv::Vec3d(50.0, -30.0, 600.0);
    circle.normal = cv::Vec3d(0.3, -0.2, 1.0);
    circle.radius = 8.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 48, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, leftGroups, rightGroups,
                             centerMatches);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.markerResults.size(), 1u);

    const auto& mr = result.markerResults[0];
    EXPECT_TRUE(mr.validPlane);
    EXPECT_TRUE(mr.validCircle);

    EXPECT_NEAR(mr.centerX, circle.center(0), 0.5);
    EXPECT_NEAR(mr.centerY, circle.center(1), 0.5);
    EXPECT_NEAR(mr.centerZ, circle.center(2), 1.0);

    cv::Vec3d rn(mr.normalX, mr.normalY, mr.normalZ);
    cv::Vec3d en = circle.normal / std::sqrt(circle.normal.dot(circle.normal));
    double dp = std::abs(rn.dot(en));
    EXPECT_NEAR(dp, 1.0, 0.02);

    EXPECT_NEAR(mr.circleFit.radius, circle.radius, 0.5);
}

// ============================================================
// 多标记点
// ============================================================
TEST_F(PointReconstructTest, MultipleMarkers) {
    std::vector<cv::Point2f> allLeft, allRight;
    std::vector<int> allLeftGroups, allRightGroups, centerMatches;

    double offsets[] = {0.0, 100.0, -50.0};
    double radii[] = {10.0, 8.0, 12.0};

    for (int m = 0; m < 3; ++m) {
        Circle3D circle;
        circle.center = cv::Vec3d(offsets[m], offsets[m], 500.0);
        circle.normal = cv::Vec3d(0.0, 0.0, 1.0);
        circle.radius = radii[m];

        std::vector<cv::Point2f> leftPts, rightPts;
        std::vector<int> leftGroups, rightGroups, cm;
        generateSyntheticMatchData(circle, 36, params_, leftPts, rightPts,
                                   leftGroups, rightGroups, cm);

        for (auto& pt : leftPts) allLeft.push_back(pt);
        for (auto& pt : rightPts) allRight.push_back(pt);
        for (auto g : leftGroups) allLeftGroups.push_back(m);
        for (auto g : rightGroups) allRightGroups.push_back(m);
        centerMatches.push_back(m);
    }

    PointReconstructCPU op(params_);
    auto result = op.Execute(allLeft, allRight, allLeftGroups, allRightGroups,
                             centerMatches);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.markerResults.size(), 3u);

    for (int m = 0; m < 3; ++m) {
        const auto& mr = result.markerResults[m];
        EXPECT_TRUE(mr.validPlane);
        EXPECT_TRUE(mr.validCircle);
        EXPECT_NEAR(mr.centerX, offsets[m], 1.0);
        EXPECT_NEAR(mr.centerY, offsets[m], 1.0);
        EXPECT_NEAR(mr.circleFit.radius, radii[m], 0.5);
    }
}

// ============================================================
// 点数不足
// ============================================================
TEST_F(PointReconstructTest, TooFewPointsForPlaneFit) {
    std::vector<cv::Point2f> leftPts = {{100, 100}, {110, 100}, {105, 95}};
    std::vector<cv::Point2f> rightPts = {{95, 100}, {105, 100}, {100, 95}};
    std::vector<int> lg = {0, 0, 0}, rg = {0, 0, 0};
    std::vector<int> cm = {0};

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, lg, rg, cm);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.markerResults.size(), 1u);
    EXPECT_FALSE(result.markerResults[0].validPlane);
    EXPECT_FALSE(result.markerResults[0].validCircle);
}

// ============================================================
// 统计信息
// ============================================================
TEST_F(PointReconstructTest, StatisticsPopulated) {
    Circle3D circle;
    circle.center = cv::Vec3d(0.0, 0.0, 500.0);
    circle.normal = cv::Vec3d(0.0, 0.0, 1.0);
    circle.radius = 10.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 36, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, leftGroups, rightGroups,
                             centerMatches);

    EXPECT_GT(result.statistics.totalTimeMs, 0.0);
    EXPECT_EQ(result.statistics.totalMarkerPairs, 1u);
    EXPECT_EQ(result.statistics.validMarkerCount, 1u);
    EXPECT_GT(result.statistics.totalReconstructedPoints, 0u);
    EXPECT_GT(result.statistics.avgReprojError, 0.0f);
    EXPECT_GT(result.statistics.avgRadius, 0.0);
}

// ============================================================
// EdgeMatchCPUResult 输入
// ============================================================
TEST_F(PointReconstructTest, EdgeMatchCPUResultInput) {
    Circle3D circle;
    circle.center = cv::Vec3d(0.0, 0.0, 500.0);
    circle.normal = cv::Vec3d(0.0, 0.0, 1.0);
    circle.radius = 10.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 36, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    EllipseEdgeMatchResult eemr;
    eemr.leftEllipseIdx = 0;
    eemr.rightEllipseIdx = 0;
    for (size_t i = 0; i < leftPts.size(); ++i) {
        EdgeMatchPair pair;
        pair.leftX = leftPts[i].x;
        pair.leftY = leftPts[i].y;
        pair.rightX = rightPts[i].x;
        pair.rightY = rightPts[i].y;
        pair.disparity = leftPts[i].x - rightPts[i].x;
        pair.confidence = 1.0f;
        pair.leftEllipseIdx = 0;
        pair.rightEllipseIdx = 0;
        eemr.matchedPairs.push_back(pair);
    }

    EdgeMatchCPUResult matchResult;
    matchResult.success = true;
    matchResult.ellipseResults.push_back(std::move(eemr));

    PointReconstructCPU op(params_);
    auto result = op.Execute(matchResult);

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.markerResults.size(), 1u);
    EXPECT_TRUE(result.markerResults[0].validPlane);
    EXPECT_TRUE(result.markerResults[0].validCircle);
    EXPECT_NEAR(result.markerResults[0].centerX, circle.center(0), 0.5);
    EXPECT_NEAR(result.markerResults[0].centerZ, circle.center(2), 1.0);
}

// ============================================================
// 法线单位化
// ============================================================
TEST_F(PointReconstructTest, NormalIsNormalized) {
    Circle3D circle;
    circle.center = cv::Vec3d(0.0, 0.0, 500.0);
    circle.normal = cv::Vec3d(1.0, 2.0, 3.0);
    circle.radius = 10.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 36, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, leftGroups, rightGroups,
                             centerMatches);

    ASSERT_TRUE(result.success);
    const auto& mr = result.markerResults[0];
    EXPECT_TRUE(mr.validPlane);

    double nLen = std::sqrt(mr.normalX*mr.normalX +
                            mr.normalY*mr.normalY +
                            mr.normalZ*mr.normalZ);
    EXPECT_NEAR(nLen, 1.0, 1e-6);
}

// ============================================================
// 大圆
// ============================================================
TEST_F(PointReconstructTest, LargeRadius) {
    Circle3D circle;
    circle.center = cv::Vec3d(0.0, 0.0, 800.0);
    circle.normal = cv::Vec3d(0.0, 0.0, 1.0);
    circle.radius = 50.0;

    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftGroups, rightGroups, centerMatches;
    generateSyntheticMatchData(circle, 72, params_, leftPts, rightPts,
                               leftGroups, rightGroups, centerMatches);

    PointReconstructCPU op(params_);
    auto result = op.Execute(leftPts, rightPts, leftGroups, rightGroups,
                             centerMatches);

    ASSERT_TRUE(result.success);
    const auto& mr = result.markerResults[0];
    EXPECT_TRUE(mr.validCircle);
    EXPECT_NEAR(mr.circleFit.radius, circle.radius, 1.0);
}

// ============================================================
// 无匹配标记点（centerMatches全为-1）
// ============================================================
TEST_F(PointReconstructTest, NoMatchedCenters) {
    std::vector<cv::Point2f> left = {{100, 100}};
    std::vector<cv::Point2f> right = {{95, 100}};
    std::vector<int> lg = {0}, rg = {0};
    std::vector<int> cm = {-1};

    PointReconstructCPU op(params_);
    auto result = op.Execute(left, right, lg, rg, cm);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.markerResults.empty());
}
