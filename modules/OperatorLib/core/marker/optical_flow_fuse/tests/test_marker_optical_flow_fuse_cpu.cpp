#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <random>
#include "marker_optical_flow_fuse_cpu.h"

using namespace calib;

namespace {
cv::Matx33d makeRotation(double angleDeg, char axis) {
    double rad = angleDeg * CV_PI / 180.0;
    double c = std::cos(rad), s = std::sin(rad);
    if (axis == 'x') return cv::Matx33d(1,0,0, 0,c,-s, 0,s,c);
    if (axis == 'y') return cv::Matx33d(c,0,s, 0,1,0, -s,0,c);
    return cv::Matx33d(c,-s,0, s,c,0, 0,0,1);
}
}

TEST(MarkerOpticalFlowFuseCPU, ParamsValidate) {
    MarkerOpticalFlowFuseCPUParams p;
    EXPECT_NO_THROW(p.validate());
}

TEST(MarkerOpticalFlowFuseCPU, Construct) {
    MarkerOpticalFlowFuseCPU op;
    EXPECT_STREQ(MarkerOpticalFlowFuseCPU::kLogTag, "01-MarkerOpticalFlowFuseCPU");
}

TEST(MarkerOpticalFlowFuseCPU, ParamsValidationErrors) {
    MarkerOpticalFlowFuseCPUParams p;
    p.matchDistThresh = -1;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.matchDistThresh = 2.0;
    p.normalAngleThresh = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.normalAngleThresh = 15.0;
    p.minMatchedPoints = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);

    p.minMatchedPoints = 3;
    p.maxMarkerCount = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(MarkerOpticalFlowFuseCPU, ParamsJsonRoundTrip) {
    MarkerOpticalFlowFuseCPUParams p;
    p.matchDistThresh = 5.0;
    p.normalAngleThresh = 20.0;
    p.minMatchedPoints = 4;
    p.collectStatistics = false;
    p.maxMarkerCount = 500;

    nlohmann::json j = p.toJson();
    MarkerOpticalFlowFuseCPUParams p2 = MarkerOpticalFlowFuseCPUParams::fromJson(j);

    EXPECT_EQ(p.matchDistThresh, p2.matchDistThresh);
    EXPECT_EQ(p.normalAngleThresh, p2.normalAngleThresh);
    EXPECT_EQ(p.minMatchedPoints, p2.minMatchedPoints);
    EXPECT_EQ(p.collectStatistics, p2.collectStatistics);
    EXPECT_EQ(p.maxMarkerCount, p2.maxMarkerCount);
}

TEST(MarkerOpticalFlowFuseCPU, FirstFrameAutoInit) {
    MarkerOpticalFlowFuseCPU op;
    std::vector<cv::Point3d> pos = {{10, 20, 300}, {110, 220, 300}, {60, 120, 350}};
    std::vector<cv::Vec3d> norm = {{0,0,1}, {0,0,1}, {0,0,1}};
    PrevFrameState emptyPrev;

    auto result = op.Execute(pos, norm, emptyPrev);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.markers.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(result.markers[i].globalId, static_cast<int>(i));
        EXPECT_TRUE(result.markers[i].matched);
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(result.R(r,c), r==c?1.0:0.0, 1e-9);
}

TEST(MarkerOpticalFlowFuseCPU, KnownTransformRecovery) {
    std::vector<cv::Point3d> global = {
        {0, 0, 0}, {100, 0, 0}, {200, 0, 0}, {100, 100, 0}, {0, 100, 0}
    };
    std::vector<cv::Vec3d> globalNorm(5, {0, 0, 1});

    PrevFrameState prev;
    prev.rawPositions = global;
    prev.rawNormals = globalNorm;
    prev.globalIds = {0, 1, 2, 3, 4};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0, 0, 0);

    std::vector<cv::Point3d> cur = {
        {0.3, -0.2, 0.1}, {100.2, 0.4, -0.1}, {199.8, -0.3, 0.2},
        {100.1, 99.9, 0.3}, {-0.2, 100.1, -0.2}
    };
    std::vector<cv::Vec3d> curNorm(5, {0, 0, 1});

    MarkerOpticalFlowFuseCPU op;
    auto result = op.Execute(cur, curNorm, prev);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.getMatchedCount(), 5u);
    auto ids = result.getGlobalIds();
    for (int i = 0; i < 5; ++i) EXPECT_EQ(ids[static_cast<size_t>(i)], i);
    EXPECT_LT(result.statistics.rmse, 0.5);
}

TEST(MarkerOpticalFlowFuseCPU, UnmatchedPointTooFar) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}, {300,0,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {250,0,0}, {300.1,0,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    MarkerOpticalFlowFuseCPU op;
    auto result = op.Execute(cur, curNorm, prev);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.getMatchedCount(), 3u);
    EXPECT_FALSE(result.markers[2].matched);
    EXPECT_EQ(result.markers[2].globalId, -1);
}

TEST(MarkerOpticalFlowFuseCPU, NormalAngleFiltering) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}, {300,0,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {200.1,0,0}, {300.1,0,0}};
    std::vector<cv::Vec3d> curNorm = {{1,0,0}, {0,0,1}, {0,0,1}, {0,0,1}};

    MarkerOpticalFlowFuseCPUParams params;
    params.normalAngleThresh = 15.0;
    MarkerOpticalFlowFuseCPU op(params);

    auto result = op.Execute(cur, curNorm, prev);

    EXPECT_FALSE(result.markers[0].matched);
    EXPECT_TRUE(result.markers[1].matched);
    EXPECT_TRUE(result.markers[2].matched);
}

TEST(MarkerOpticalFlowFuseCPU, InsufficientMatchesFailure) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {500,0,0}, {600,0,0}};
    std::vector<cv::Vec3d> curNorm(3, {0,0,1});

    MarkerOpticalFlowFuseCPU op;
    auto result = op.Execute(cur, curNorm, prev);

    EXPECT_FALSE(result.success);
}

TEST(MarkerOpticalFlowFuseCPU, EmptyCurrentFrame) {
    MarkerOpticalFlowFuseCPU op;
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}};

    auto result = op.Execute({}, {}, prev);

    EXPECT_FALSE(result.success);
}

TEST(MarkerOpticalFlowFuseCPU, MultiFrameSequence) {
    std::vector<cv::Point3d> base = {{0,0,0}, {150,0,0}, {0,150,0}, {150,150,0}};
    std::vector<cv::Vec3d> baseNorm(4, {0,0,1});

    MarkerOpticalFlowFuseCPU op;

    PrevFrameState prev0;
    auto r0 = op.Execute(base, baseNorm, prev0);
    ASSERT_TRUE(r0.success);
    ASSERT_EQ(r0.getMatchedCount(), 4u);

    PrevFrameState prev1;
    prev1.rawPositions = r0.getRawPositions();
    prev1.rawNormals = baseNorm;
    prev1.globalIds = r0.getGlobalIds();
    prev1.R = r0.R;
    prev1.T = r0.T;

    std::vector<cv::Point3d> cur1 = {
        {0.3, -0.2, 0.1}, {150.4, 0.2, -0.1}, {0.1, 149.8, 0.2}, {150.2, 150.3, -0.2}
    };
    auto r1 = op.Execute(cur1, baseNorm, prev1);
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(r1.getMatchedCount(), 4u);

    PrevFrameState prev2;
    prev2.rawPositions = r1.getRawPositions();
    prev2.rawNormals = baseNorm;
    prev2.globalIds = r1.getGlobalIds();
    prev2.R = r1.R;
    prev2.T = r1.T;

    std::vector<cv::Point3d> cur2 = {
        {0.5, 0.1, -0.1}, {150.6, 0.4, 0.1}, {0.3, 150.1, -0.2}, {150.5, 150.5, 0.1}
    };
    auto r2 = op.Execute(cur2, baseNorm, prev2);
    ASSERT_TRUE(r2.success);
    EXPECT_EQ(r2.getMatchedCount(), 4u);
}

TEST(MarkerOpticalFlowFuseCPU, GlobalMarkerOverride) {
    std::vector<cv::Point3d> globalPos = {
        {1000, 2000, 3000}, {1100, 2000, 3000}, {1200, 2000, 3000}
    };
    std::vector<cv::Vec3d> globalNorm(3, {0, 0, 1});
    GlobalMarkerSet globals{globalPos, globalNorm};

    PrevFrameState prev;
    prev.rawPositions = {{1000.2, 2000.1, 3000}, {1100.1, 2000, 3000.1}, {1200, 2000.2, 2999.9}};
    prev.rawNormals = globalNorm;
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0, 0, 0);

    std::vector<cv::Point3d> cur = {{1000.5, 2000.3, 3000.1}, {1100.4, 2000.2, 3000.2}, {1200.1, 2000.4, 3000}};
    std::vector<cv::Vec3d> curNorm(3, {0, 0, 1});

    MarkerOpticalFlowFuseCPU op;
    auto result = op.Execute(cur, curNorm, prev, globals);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.getMatchedCount(), 3u);
    auto trans = result.getTransformedPositions();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(trans[i].x, globalPos[i].x, 0.5);
        EXPECT_NEAR(trans[i].y, globalPos[i].y, 0.5);
        EXPECT_NEAR(trans[i].z, globalPos[i].z, 0.5);
    }
}

TEST(MarkerOpticalFlowFuseCPU, PointReconstructResultInput) {
    PointReconstructCPUResult reconResult;
    reconResult.success = true;

    for (int i = 0; i < 3; ++i) {
        MarkerReconstructResult mr;
        mr.validCircle = true;
        mr.centerX = static_cast<double>(i) * 100.0;
        mr.centerY = 0.0;
        mr.centerZ = 300.0;
        mr.normalX = 0; mr.normalY = 0; mr.normalZ = 1;
        reconResult.markerResults.push_back(std::move(mr));
    }
    {
        MarkerReconstructResult mr;
        mr.validCircle = false;
        reconResult.markerResults.push_back(std::move(mr));
    }

    PrevFrameState prev;
    MarkerOpticalFlowFuseCPU op;
    auto result = op.Execute(reconResult, prev);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.markers.size(), 3u);
    EXPECT_EQ(result.markers[0].globalId, 0);
    EXPECT_EQ(result.markers[1].globalId, 1);
    EXPECT_EQ(result.markers[2].globalId, 2);
}
