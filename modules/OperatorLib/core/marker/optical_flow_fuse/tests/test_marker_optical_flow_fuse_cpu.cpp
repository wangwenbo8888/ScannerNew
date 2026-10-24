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

TEST(MarkerOpticalFlowFuseCPU, NearCollinearRejected) {
    // 2026-10-15 加固：旧秩阈值 1e-6 形同虚设——三点"几乎"排成线（偏 1mm/150mm 跨度）
    // 时解算病态（误差在远离标记点处放大至厘米级、rmse 盲区）。新门槛：第二大伸展量
    // 须 > max(2%×最大伸展, 0.5mm 绝对下限)。
    PrevFrameState prev;
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);
    std::vector<cv::Vec3d> curNorm(3, {0,0,1});
    MarkerOpticalFlowFuseCPU op;

    {   // 近共线：三点跨 150mm，弦中垂偏仅 ~0.1mm（旧阈 1e-6 会放行＝病态解）——必须拒
        prev.rawPositions = {{0,0,0}, {75,1.0,0}, {150,1.2,0}};
        std::vector<cv::Point3d> cur = {{0.05,0,0}, {75.05,1.0,0}, {150.05,1.2,0}};
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_FALSE(r.success);
    }
    {   // 健康：拱形（顶点离弦 8mm，伸展比≈5%>2%）——放行
        prev.rawPositions = {{0,0,0}, {75,8.0,0}, {150,0,0}};
        std::vector<cv::Point3d> cur = {{0.05,0,0}, {75.05,8.0,0}, {150.05,0,0}};
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_TRUE(r.success);
    }
}

TEST(MarkerOpticalFlowFuseCPU, ParamsCrossValidation) {
    // 2026-10-15 加固：跨参数守卫（防"单独合法、组合爆炸"的配置事故——
    // 20mm 门灾难的配置期闸门）
    {   // rmse 负值＝静默失效——必须拒（想关请明写 0）
        MarkerOpticalFlowFuseCPUParams p; p.rmseMaxGate = -3.0;
        EXPECT_THROW(p.validate(), std::invalid_argument);
    }
    {   // rmse 门过紧（< 门限/4）＝误杀正常帧
        MarkerOpticalFlowFuseCPUParams p; p.matchDistThresh = 2.0; p.rmseMaxGate = 0.3;
        EXPECT_THROW(p.validate(), std::invalid_argument);
    }
    {   // 法线门放宽到 30°+ 却不开软降级加权——硬门软级双撤
        MarkerOpticalFlowFuseCPUParams p; p.normalAngleThresh = 45.0;
        EXPECT_THROW(p.validate(), std::invalid_argument);
        p.useNormalWeight = true;
        EXPECT_NO_THROW(p.validate());
    }
    {   // 首帧搜索半径（门×倍率）> 100mm——达标记点间距量级，错锁自由发挥
        MarkerOpticalFlowFuseCPUParams p; p.matchDistThresh = 20.0;   // ×10=200mm
        EXPECT_THROW(p.validate(), std::invalid_argument);
        p.firstFrameSearchScale = 5.0;                                 // ×5=100mm 边界放行
        EXPECT_NO_THROW(p.validate());
    }
    {   // 倍率本身>25（防多打一个 0）
        MarkerOpticalFlowFuseCPUParams p; p.firstFrameSearchScale = 50.0;
        EXPECT_THROW(p.validate(), std::invalid_argument);
    }
    {   // 门限>50mm 属位姿跳变范畴，无意义
        MarkerOpticalFlowFuseCPUParams p; p.matchDistThresh = 80.0;
        EXPECT_THROW(p.validate(), std::invalid_argument);
    }
    {   // min>max 永不成配置
        MarkerOpticalFlowFuseCPUParams p; p.minMatchedPoints = 2000;
        EXPECT_THROW(p.validate(), std::invalid_argument);
    }
    {   // 当前两个工作点必须放行：2mm×10=20 半径 / 10mm×10=100 半径（踩线已实测）
        MarkerOpticalFlowFuseCPUParams p1;   // 默认 2mm
        EXPECT_NO_THROW(p1.validate());
        MarkerOpticalFlowFuseCPUParams p2; p2.matchDistThresh = 10.0; p2.rmseMaxGate = 3.0;
        EXPECT_NO_THROW(p2.validate());
    }
}

TEST(MarkerOpticalFlowFuseCPU, StatsCarriesEffectiveParams) {
    // 2026-10-15 加固：生效参数快照入 stats（事故复盘——config 值≠运行时值的对账依据）
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);
    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}, {100.2,100.1,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    MarkerOpticalFlowFuseCPUParams p;
    p.matchDistThresh = 10.0;
    p.rmseMaxGate = 3.0;
    MarkerOpticalFlowFuseCPU op(p);
    auto r = op.Execute(cur, curNorm, prev);
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(r.statistics.effMatchDistThresh, 10.0, 1e-12);
    EXPECT_NEAR(r.statistics.effGuardTolerance, 20.0, 1e-12);      // 守卫容差=2×门
    EXPECT_NEAR(r.statistics.effRejectFloor, 0.5, 1e-12);          // 剔坏对免死线（固定值）
    EXPECT_NEAR(r.statistics.effFirstFrameRadius, 100.0, 1e-12);   // 首帧半径=门×倍率
    EXPECT_NEAR(r.statistics.effRmseMaxGate, 3.0, 1e-12);
    EXPECT_FALSE(r.statistics.effUseNormalWeight);
}

TEST(MarkerOpticalFlowFuseCPU, CollectStatisticsGateEffective) {
    // 2026-10-15 清理①：统计开关此前是死参数（收不收统计它都收）——现生效：
    // false 时计时字段全为 0（不上报），功能统计（rmse/匹配数）不受影响
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);
    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}, {100.2,100.1,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    {
        MarkerOpticalFlowFuseCPUParams p;
        p.collectStatistics = false;
        MarkerOpticalFlowFuseCPU op(p);
        auto r = op.Execute(cur, curNorm, prev);
        ASSERT_TRUE(r.success);
        EXPECT_EQ(r.getMatchedCount(), 4u);              // 功能统计不受影响
        EXPECT_GT(r.statistics.rmse, 0.0);
        EXPECT_DOUBLE_EQ(r.statistics.totalTimeMs, 0.0); // 计时不上报
        EXPECT_DOUBLE_EQ(r.statistics.matchTimeMs, 0.0);
        EXPECT_DOUBLE_EQ(r.statistics.svdTimeMs, 0.0);
        EXPECT_DOUBLE_EQ(r.statistics.transformTimeMs, 0.0);
    }
    {   // 默认开：计时照报（值非负即可——快到 0 是合法的）
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, curNorm, prev);
        ASSERT_TRUE(r.success);
        EXPECT_GE(r.statistics.totalTimeMs, 0.0);
    }
}

TEST(MarkerOpticalFlowFuseCPU, ConstructorValidatesParams) {
    // 2026-10-15 清理④：构造函数此前不校验参数（坏参数能出生，跑到一半才爆）
    // ——现出生即体检（fromJson 早已有校验，补齐构造侧）
    MarkerOpticalFlowFuseCPUParams bad;
    bad.rmseMaxGate = -1.0;
    EXPECT_THROW({ MarkerOpticalFlowFuseCPU op(bad); }, std::invalid_argument);

    MarkerOpticalFlowFuseCPUParams good;                   // 默认参数必须可构造
    EXPECT_NO_THROW({ MarkerOpticalFlowFuseCPU op(good); });
}

TEST(MarkerOpticalFlowFuseCPU, FirstFrameAutoInitRequiresFourPoints) {
    // 2026-10-15 裁决：首帧建系强制 ≥4 点——首帧是全局基准，3 点零冗余（残差恒零
    // 质检失明），错首帧污染全会话
    MarkerOpticalFlowFuseCPU op;
    PrevFrameState emptyPrev;

    {   // 3 点：必须拒
        std::vector<cv::Point3d> pos3 = {{10, 20, 300}, {110, 220, 300}, {60, 120, 350}};
        std::vector<cv::Vec3d> norm3(3, {0,0,1});
        auto r = op.Execute(pos3, norm3, emptyPrev);
        EXPECT_FALSE(r.success);
    }
    {   // 4 点：放行
        std::vector<cv::Point3d> pos4 = {{10, 20, 300}, {110, 220, 300}, {60, 120, 350}, {200, 40, 310}};
        std::vector<cv::Vec3d> norm4(4, {0,0,1});
        auto r = op.Execute(pos4, norm4, emptyPrev);
        EXPECT_TRUE(r.success);
        EXPECT_EQ(r.markers.size(), 4u);
    }
}

TEST(MarkerOpticalFlowFuseCPU, ExactlyMinPairsFlaggedDegraded) {
    // 2026-10-15 裁决：恰好 minMatchedPoints(3) 对＝零冗余解（残差恒≈0 质检失明），
    // 成功也不得报 Normal——强制降级，下游知情
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    {   // 3 对成功 → qualityFlag 必须 ≤ Degraded
        std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}};
        std::vector<cv::Vec3d> curNorm(3, {0,0,1});
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, curNorm, prev);
        ASSERT_TRUE(r.success);
        EXPECT_NE(r.qualityFlag, calib::QualityFlag::Normal);
    }
    {   // 5 对成功（有冗余）→ 可报 Normal
        PrevFrameState prev5;
        prev5.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}, {50,50,30}};
        prev5.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
        prev5.globalIds = {0, 1, 2, 3, 4};
        prev5.R = cv::Matx33d::eye();
        prev5.T = cv::Vec3d(0,0,0);
        std::vector<cv::Point3d> cur5 = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}, {100.2,100.1,0}, {50.1,50.0,30.1}};
        std::vector<cv::Vec3d> curNorm5(5, {0,0,1});
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur5, curNorm5, prev5);
        ASSERT_TRUE(r.success);
        EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Normal);
    }
}

TEST(MarkerOpticalFlowFuseCPU, FirstFrameAutoInit) {
    // 注：2026-10-15 裁决首帧建系要求 ≥4 点（防 3 点零冗余盲区），本用例随之为 4 点
    MarkerOpticalFlowFuseCPU op;
    std::vector<cv::Point3d> pos = {{10, 20, 300}, {110, 220, 300}, {60, 120, 350}, {200, 40, 310}};
    std::vector<cv::Vec3d> norm(4, {0,0,1});
    PrevFrameState emptyPrev;

    auto result = op.Execute(pos, norm, emptyPrev);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.markers.size(), 4u);
    for (size_t i = 0; i < 4; ++i) {
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
    // 注：几何须非共线——共线匹配集会被退化检查正确拒绝（2026-10-14 行为变更）
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {200,200,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {250,0,0}, {200.1,200.1,0}};
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
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {300,300,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}, {300.1,300.1,0}};
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
    // 注：几何须非共线——共线匹配集会被退化检查正确拒绝（2026-10-14 行为变更）
    std::vector<cv::Point3d> globalPos = {
        {1000, 2000, 3000}, {1100, 2000, 3000}, {1000, 2100, 3000}
    };
    std::vector<cv::Vec3d> globalNorm(3, {0, 0, 1});
    GlobalMarkerSet globals{globalPos, globalNorm};

    PrevFrameState prev;
    prev.rawPositions = {{1000.2, 2000.1, 3000}, {1100.1, 2000, 3000.1}, {1000, 2100.2, 2999.9}};
    prev.rawNormals = globalNorm;
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0, 0, 0);

    std::vector<cv::Point3d> cur = {{1000.5, 2000.3, 3000.1}, {1100.4, 2000.2, 3000.2}, {1000.1, 2100.4, 3000}};
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
    // 注：2026-10-15 裁决首帧建系要求 ≥4 点，本用例 validCircle 点随之为 4 个
    PointReconstructCPUResult reconResult;
    reconResult.success = true;

    for (int i = 0; i < 4; ++i) {
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
    EXPECT_EQ(result.markers.size(), 4u);
    EXPECT_EQ(result.markers[0].globalId, 0);
    EXPECT_EQ(result.markers[1].globalId, 1);
    EXPECT_EQ(result.markers[2].globalId, 2);
    EXPECT_EQ(result.markers[3].globalId, 3);
}

// ============================================================================
// registration_optimize 专项（2026-10-14）：第一批护栏——输入校验/参数化/桥接
// ============================================================================

TEST(MarkerOpticalFlowFuseCPU, InputLengthValidation) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {200.1,0,0}};

    // 当前帧法线比位置少一个 → 必须安全失败，不得越界
    {
        std::vector<cv::Vec3d> shortNorm(2, {0,0,1});
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, shortNorm, prev);
        EXPECT_FALSE(r.success);
    }
    // 上帧法线比位置少一个 → 必须安全失败，不得越界
    {
        PrevFrameState badPrev;
        badPrev.rawPositions = prev.rawPositions;
        badPrev.rawNormals = {{0,0,1}, {0,0,1}};   // 少一个
        badPrev.globalIds = prev.globalIds;
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, std::vector<cv::Vec3d>(3, {0,0,1}), badPrev);
        EXPECT_FALSE(r.success);
    }
}

TEST(MarkerOpticalFlowFuseCPU, FirstFrameSearchScaleParam) {
    // 默认 10.0（保持旧口径）
    MarkerOpticalFlowFuseCPUParams p;
    EXPECT_DOUBLE_EQ(p.firstFrameSearchScale, 10.0);

    // JSON 往返
    p.firstFrameSearchScale = 5.0;
    nlohmann::json j = p.toJson();
    auto p2 = MarkerOpticalFlowFuseCPUParams::fromJson(j);
    EXPECT_DOUBLE_EQ(p2.firstFrameSearchScale, 5.0);

    // 非法值校验
    MarkerOpticalFlowFuseCPUParams bad;
    bad.firstFrameSearchScale = 0.0;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
}

TEST(MarkerOpticalFlowFuseCPU, NormalWeightSwitchDefaultOff) {
    // 2026-10-15 裁决：法线可信度加权默认关闭（等权解算）——默认 15° 朝向门下
    // 权重差仅 ~7%（近似等权、无实际效果）；开关保留作为"朝向门放宽时的软降级保险丝"。
    MarkerOpticalFlowFuseCPUParams p;
    EXPECT_FALSE(p.useNormalWeight);

    p.useNormalWeight = true;
    nlohmann::json j = p.toJson();
    auto p2 = MarkerOpticalFlowFuseCPUParams::fromJson(j);
    EXPECT_TRUE(p2.useNormalWeight);

    // 两种设置下同一场景均能正常解算（KnownTransform 数据）
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);
    std::vector<cv::Point3d> cur = {{0.2,-0.1,0}, {100.1,0.2,0}, {0.1,100.1,0}, {100.2,100.1,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    {
        MarkerOpticalFlowFuseCPU op;                       // 默认（关）
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_TRUE(r.success);
        EXPECT_EQ(r.getMatchedCount(), 4u);
        EXPECT_LT(r.statistics.rmse, 0.5);
    }
    {
        MarkerOpticalFlowFuseCPUParams pw;
        pw.useNormalWeight = true;                          // 开
        MarkerOpticalFlowFuseCPU op(pw);
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_TRUE(r.success);
        EXPECT_EQ(r.getMatchedCount(), 4u);
        EXPECT_LT(r.statistics.rmse, 0.5);
    }
}

TEST(MarkerOpticalFlowFuseCPU, ToPrevFrameStateBridgeCopiesPoints) {
    AtomicFrameState sched;
    sched.rawPoints = {
        {10, 20, 300, 0, 0, 1, 7},
        {110, 20, 300, 0, 0, 1, 8},
        {210, 20, 300, 0, 0, 1, 9}
    };
    sched.globalIds = {7, 8, 9};
    sched.R[0]=1; sched.R[4]=1; sched.R[8]=1;
    sched.T[0]=1; sched.T[1]=2; sched.T[2]=3;

    PrevFrameState of = toPrevFrameState(sched);

    // 桥接不得丢点位：这是配准查询的基准数据
    ASSERT_EQ(of.rawPositions.size(), 3u);
    ASSERT_EQ(of.rawNormals.size(), 3u);
    EXPECT_EQ(of.globalIds.size(), 3u);
    EXPECT_NEAR(of.rawPositions[0].x, 10.0, 1e-12);
    EXPECT_NEAR(of.rawPositions[2].z, 300.0, 1e-12);
    EXPECT_NEAR(of.rawNormals[1](2), 1.0, 1e-12);
    EXPECT_EQ(of.globalIds[0], 7);
    EXPECT_NEAR(of.T(0), 1.0, 1e-12);
}

// ============================================================================
// registration_optimize 专项（2026-10-14）：第二批核心关卡——一对一/退化/剔坏对/误差门/编号守卫
// ============================================================================

TEST(MarkerOpticalFlowFuseCPU, OneToOneMatching) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    // 两个当前点（0.2 与 1.5）都落在 prev[0]（原点）的 2mm 门内——只允许近者认领
    std::vector<cv::Point3d> cur = {{0.2,0,0}, {1.5,0,0}, {100.1,0,0}, {0.1,100,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    MarkerOpticalFlowFuseCPU op;
    auto r = op.Execute(cur, curNorm, prev);

    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.getMatchedCount(), 3u);          // 旧实现：4（重复认领）
    EXPECT_TRUE(r.markers[0].matched);           // 近者成立
    EXPECT_FALSE(r.markers[1].matched);          // 远者必须被挤掉
    EXPECT_EQ(r.markers[1].globalId, -1);
    EXPECT_LT(r.statistics.rmse, 0.5);           // 无重复对应污染
}

TEST(MarkerOpticalFlowFuseCPU, CollinearDegeneracyRejected) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}};   // 三点严格共线
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {199.9,0,0}};
    std::vector<cv::Vec3d> curNorm(3, {0,0,1});

    MarkerOpticalFlowFuseCPU op;
    auto r = op.Execute(cur, curNorm, prev);

    // 共线三点解刚体变换欠定——必须显式拒绝，不得"成功"一个任意姿态
    EXPECT_FALSE(r.success);
}

TEST(MarkerOpticalFlowFuseCPU, WorstPairResidualRejection) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    // 3 个好对（噪声 ≤0.3mm）＋1 个坏对（偏 1.9mm，贴近 2mm 匹配门边缘的典型错配）
    std::vector<cv::Point3d> cur = {{0.1,-0.1,0}, {100.2,0.1,0}, {0.1,100.2,0}, {101.9,100.2,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    MarkerOpticalFlowFuseCPU op;
    auto r = op.Execute(cur, curNorm, prev);

    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.getMatchedCount(), 3u);          // 坏对应被剔除重解
    EXPECT_FALSE(r.markers[3].matched);
    EXPECT_LT(r.statistics.rmse, 0.5);           // 旧实现 rmse≈0.7+（坏对拖累）
}

TEST(MarkerOpticalFlowFuseCPU, RmseGateOnSuccess) {
    PrevFrameState prev;
    prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {100,100,0}};
    prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
    prev.globalIds = {0, 1, 2, 3};
    prev.R = cv::Matx33d::eye();
    prev.T = cv::Vec3d(0,0,0);

    // 全体交错噪声（±1.2mm 交替，刚体变换无法吸收 → 解算残差必然高）
    std::vector<cv::Point3d> cur = {{1.2,0.8,0}, {98.8,0.9,0}, {1.1,99.0,0}, {99.0,101.1,0}};
    std::vector<cv::Vec3d> curNorm(4, {0,0,1});

    // 门=0（默认，历史口径）→ 仍成功
    {
        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_TRUE(r.success);
    }
    // 门=0.5 → 必须失败
    {
        MarkerOpticalFlowFuseCPUParams p;
        p.rmseMaxGate = 0.5;
        MarkerOpticalFlowFuseCPU op(p);
        auto r = op.Execute(cur, curNorm, prev);
        EXPECT_FALSE(r.success);
    }
}

TEST(MarkerOpticalFlowFuseCPU, GlobalIdMapConsistencyGuard) {
    // 场景 A：编号在地图范围内，但地图条目与上帧变换位置严重不符（编号语义串位）
    {
        std::vector<cv::Point3d> mapPos = {{5000,0,0}, {5100,0,0}, {5200,0,0}};
        std::vector<cv::Vec3d> mapNorm(3, {0,0,1});
        GlobalMarkerSet map{mapPos, mapNorm};

        PrevFrameState prev;
        prev.rawPositions = {{0,0,0}, {100,0,0}, {200,0,0}};
        prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}};
        prev.globalIds = {0, 1, 2};     // 与地图下标撞号但物理上无关
        prev.R = cv::Matx33d::eye();
        prev.T = cv::Vec3d(0,0,0);

        std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {199.9,0,0}};
        std::vector<cv::Vec3d> curNorm(3, {0,0,1});

        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, curNorm, prev, map);
        EXPECT_FALSE(r.success);        // 旧实现：拿 5000 外的地图点硬解出"成功"垃圾
    }
    // 场景 B：编号超出地图长度（帧内下标语义混入）→ 丢弃该对，不得回落旧坐标硬解
    {
        std::vector<cv::Point3d> mapPos = {{0,0,0}, {100,0,0}, {0,100,0}};
        std::vector<cv::Vec3d> mapNorm(3, {0,0,1});
        GlobalMarkerSet map{mapPos, mapNorm};

        PrevFrameState prev;
        prev.rawPositions = {{0,0,0}, {100,0,0}, {0,100,0}, {200,0,0}};
        prev.rawNormals = {{0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}};
        prev.globalIds = {0, 1, 2, 5};  // 5 超出地图长度
        prev.R = cv::Matx33d::eye();
        prev.T = cv::Vec3d(0,0,0);

        std::vector<cv::Point3d> cur = {{0.1,0,0}, {100.1,0,0}, {0.1,100.1,0}, {200.1,0,0}};
        std::vector<cv::Vec3d> curNorm(4, {0,0,1});

        MarkerOpticalFlowFuseCPU op;
        auto r = op.Execute(cur, curNorm, prev, map);
        ASSERT_TRUE(r.success);
        EXPECT_EQ(r.getMatchedCount(), 3u);   // 旧实现：4（超界编号回落旧坐标参与解算）
        EXPECT_FALSE(r.markers[3].matched);
    }
}
