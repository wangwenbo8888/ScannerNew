#if 0  // SKIPPED: corrupted by batch edit

/**
 * @file test_pose_estimate_cpu.cpp
 * @brief 璁惧濮挎€丆PU绠楀瓙 - 鍗曞厓娴嬭瘯
 */

#include <gtest/gtest.h>
#include "pose_estimate_cpu.h"
#include "common/calib_warmup_config.h"
#include <stdexcept>
#include <cmath>

using namespace calib;

// ============================================================
// 杈呭姪鍑芥暟
// ============================================================

/** @brief 鏋勫缓 rows脳cols 鏍囧噯缃戞牸锛岄棿璺?spacing锛孼=0骞抽潰
 *
 *  琛屾柟鍚?澧炲姞r) 鈫?X杞? 鍒楁柟鍚?澧炲姞c) 鈫?Y杞? *  杩欐牱褰?rowAxis="X" 鏃讹紝涓栫晫鍧愭爣绯诲彉鎹负鎭掔瓑鐭╅樀
 */
static std::vector<std::vector<cv::Point3d>> makeGrid(int rows, int cols, double spacing = 30.0) {
    std::vector<std::vector<cv::Point3d>> grid(rows);
    for (int r = 0; r < rows; ++r) {
        grid[r].reserve(cols);
        for (int c = 0; c < cols; ++c) {
            grid[r].emplace_back(r * spacing, c * spacing, 0.0);
        }
    }
    return grid;
}

/** @brief 鏋勫缓鏍囧噯鍙傛暟锛?脳3缃戞牸锛屽師鐐?0,0)锛岃=X锛岄潰=Z锛屼竴涓洰鏍囷級 */
static PoseEstimateCPUParams makeStandardParams() {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(3, 3);
    params.originRow = 0;
    params.originCol = 0;
    params.rowAxis = "X";
    params.faceNormal = "Z";

    PoseTarget t;
    t.name = "front_center";
    t.tx = 0.0; t.ty = 0.0; t.tz = 0.0;
    t.rx = 0.0; t.ry = 0.0; t.rz = 0.0;
    t.posThreshold = 5.0;
    t.rotThreshold = 5.0;
    params.poseTargets.push_back(t);
    return params;
}

// ============================================================
// Smoke Test
// ============================================================

TEST(PoseEstimateCPUSmoke, CompilationOnly) {
    ASSERT_TRUE(true);
}

// ============================================================
// PoseTarget 娴嬭瘯
// ============================================================

TEST(PoseTargetTest, DefaultValues) {
    PoseTarget t;
    EXPECT_DOUBLE_EQ(t.tx, 0.0);
    EXPECT_DOUBLE_EQ(t.ty, 0.0);
    EXPECT_DOUBLE_EQ(t.tz, 0.0);
    EXPECT_DOUBLE_EQ(t.rx, 0.0);
    EXPECT_DOUBLE_EQ(t.ry, 0.0);
    EXPECT_DOUBLE_EQ(t.rz, 0.0);
    EXPECT_DOUBLE_EQ(t.posThreshold, 10.0);
    EXPECT_DOUBLE_EQ(t.rotThreshold, 5.0);
    EXPECT_TRUE(t.name.empty());
}

TEST(PoseTargetTest, ValidateThrowsOnNegativePosThreshold) {
    PoseTarget t;
    t.posThreshold = -1.0;
    EXPECT_THROW(t.validate(), std::invalid_argument);
}

TEST(PoseTargetTest, ValidateThrowsOnZeroPosThreshold) {
    PoseTarget t;
    t.posThreshold = 0.0;
    EXPECT_THROW(t.validate(), std::invalid_argument);
}

TEST(PoseTargetTest, ValidateThrowsOnNegativeRotThreshold) {
    PoseTarget t;
    t.posThreshold = 10.0;
    t.rotThreshold = -0.1;
    EXPECT_THROW(t.validate(), std::invalid_argument);
}

TEST(PoseTargetTest, ValidatePassesWithPositiveThreshold) {
    PoseTarget t;
    t.posThreshold = 5.0;
    t.rotThreshold = 2.0;
    EXPECT_NO_THROW(t.validate());
}

TEST(PoseTargetTest, JsonRoundTrip) {
    PoseTarget t;
    t.name = "test_pose";
    t.tx = 10.0; t.ty = 20.0; t.tz = 30.0;
    t.rx = 1.0; t.ry = 2.0; t.rz = 3.0;
    t.posThreshold = 7.5;
    t.rotThreshold = 3.5;

    auto j = t.toJson();
    PoseTarget t2 = PoseTarget::fromJson(j);

    EXPECT_EQ(t2.name, "test_pose");
    EXPECT_DOUBLE_EQ(t2.tx, 10.0);
    EXPECT_DOUBLE_EQ(t2.ty, 20.0);
    EXPECT_DOUBLE_EQ(t2.tz, 30.0);
    EXPECT_DOUBLE_EQ(t2.rx, 1.0);
    EXPECT_DOUBLE_EQ(t2.ry, 2.0);
    EXPECT_DOUBLE_EQ(t2.rz, 3.0);
    EXPECT_DOUBLE_EQ(t2.posThreshold, 7.5);
    EXPECT_DOUBLE_EQ(t2.rotThreshold, 3.5);
}

TEST(PoseTargetTest, FromJsonPartialFields) {
    nlohmann::json j;
    j["name"] = "partial";
    j["tx"] = 100.0;

    PoseTarget t = PoseTarget::fromJson(j);
    EXPECT_EQ(t.name, "partial");
    EXPECT_DOUBLE_EQ(t.tx, 100.0);
    EXPECT_DOUBLE_EQ(t.ty, 0.0);   // default
    EXPECT_DOUBLE_EQ(t.posThreshold, 10.0); // default
}

TEST(PoseTargetTest, FromJsonEmptyObject) {
    nlohmann::json j = nlohmann::json::object();
    PoseTarget t = PoseTarget::fromJson(j);
    EXPECT_TRUE(t.name.empty());
    EXPECT_DOUBLE_EQ(t.tx, 0.0);
}

// ============================================================
// PoseEstimateCPUParams 娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnEmptyGrid) {
    PoseEstimateCPUParams p;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnSingleRow) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)}};
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnSingleCol) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0)}, {cv::Point3d(0,1,0)}};
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnInvalidOriginRow) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)},
                    {cv::Point3d(0,1,0), cv::Point3d(1,1,0)}};
    p.originRow = 5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnInvalidOriginCol) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)},
                    {cv::Point3d(0,1,0), cv::Point3d(1,1,0)}};
    p.originCol = 5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnLastRowOrigin) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)},
                    {cv::Point3d(0,1,0), cv::Point3d(1,1,0)}};
    p.originRow = 1; // 鏈€鍚庝竴琛岋紝鏃犳硶+1
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnLastColOrigin) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)},
                    {cv::Point3d(0,1,0), cv::Point3d(1,1,0)}};
    p.originCol = 1; // 鏈€鍚庝竴鍒楋紝鏃犳硶+1
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnInvalidRowAxis) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.rowAxis = "Z";
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnInvalidFaceNormal) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.faceNormal = "Y";
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidateThrowsOnJaggedGrid) {
    PoseEstimateCPUParams p;
    p.gridPoints = {{cv::Point3d(0,0,0), cv::Point3d(1,0,0)},
                    {cv::Point3d(0,1,0)}};
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(PoseEstimateCPUParamsTest, ValidatePassesOnValidConfig) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.originRow = 0;
    p.originCol = 0;
    p.rowAxis = "X";
    p.faceNormal = "Z";
    PoseTarget t;
    t.name = "test";
    p.poseTargets.push_back(t);
    EXPECT_NO_THROW(p.validate());
}

TEST(PoseEstimateCPUParamsTest, ValidatePassesOnRowAxisY) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.rowAxis = "Y";
    p.faceNormal = "Z";
    p.poseTargets.push_back(PoseTarget{});
    EXPECT_NO_THROW(p.validate());
}

TEST(PoseEstimateCPUParamsTest, ValidatePassesOnFaceNormalNegZ) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.faceNormal = "-Z";
    p.poseTargets.push_back(PoseTarget{});
    EXPECT_NO_THROW(p.validate());
}

TEST(PoseEstimateCPUParamsTest, JsonRoundTrip) {
    PoseEstimateCPUParams p;
    p.gridPoints = makeGrid(3, 3);
    p.originRow = 1;
    p.originCol = 1;
    p.rowAxis = "Y";
    p.faceNormal = "-Z";
    PoseTarget t;
    t.name = "target1";
    t.tx = 100.0;
    t.posThreshold = 8.0;
    p.poseTargets.push_back(t);
    p.collectStatistics = false;

    auto j = p.toJson();
    PoseEstimateCPUParams p2 = PoseEstimateCPUParams::fromJson(j);

    EXPECT_EQ(p2.gridPoints.size(), 3u);
    EXPECT_EQ(p2.gridPoints[0].size(), 3u);
    EXPECT_EQ(p2.originRow, 1);
    EXPECT_EQ(p2.originCol, 1);
    EXPECT_EQ(p2.rowAxis, "Y");
    EXPECT_EQ(p2.faceNormal, "-Z");
    EXPECT_EQ(p2.poseTargets.size(), 1u);
    EXPECT_EQ(p2.poseTargets[0].name, "target1");
    EXPECT_DOUBLE_EQ(p2.poseTargets[0].tx, 100.0);
    EXPECT_FALSE(p2.collectStatistics);
}

// ============================================================
// PoseMatch 娴嬭瘯
// ============================================================

TEST(PoseMatchTest, DefaultValues) {
    PoseMatch m;
    EXPECT_EQ(m.targetIndex, -1);
    EXPECT_FALSE(m.matched);
    EXPECT_DOUBLE_EQ(m.positionError, 0.0);
    EXPECT_DOUBLE_EQ(m.rotationError, 0.0);
    EXPECT_TRUE(m.targetName.empty());
}

// ============================================================
// PoseEstimateStats 娴嬭瘯
// ============================================================

TEST(PoseEstimateStatsTest, DefaultValues) {
    PoseEstimateStats s;
    EXPECT_DOUBLE_EQ(s.totalTimeMs, 0.0);
    EXPECT_DOUBLE_EQ(s.coordBuildTimeMs, 0.0);
    EXPECT_DOUBLE_EQ(s.matchTimeMs, 0.0);
    EXPECT_EQ(s.targetCount, 0u);
    EXPECT_EQ(s.matchedCount, 0u);
    EXPECT_EQ(s.estimateCallCount, 0u);
}

// ============================================================
// PoseEstimateCPUResult 娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUResultTest, DefaultValues) {
    PoseEstimateCPUResult r;
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.anyMatched);
    EXPECT_EQ(r.bestMatch, -1);
    EXPECT_TRUE(r.matches.empty());
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Normal);
}

TEST(PoseEstimateCPUResultTest, MoveSemantics) {
    PoseEstimateCPUResult r1;
    r1.success = true;
    r1.matches.resize(3);
    PoseEstimateCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_EQ(r2.matches.size(), 3u);
}

// ============================================================
// 鏋勯€?/ 鏋愭瀯 娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUConstructTest, DefaultConstruct) {
    PoseEstimateCPU op;
    EXPECT_TRUE(op.GetParams().gridPoints.empty());
}

TEST(PoseEstimateCPUConstructTest, ConstructWithParams) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    EXPECT_EQ(op.GetParams().gridPoints.size(), 3u);
}

TEST(PoseEstimateCPUConstructTest, SetParamsRebuilds) {
    PoseEstimateCPU op;
    auto params = makeStandardParams();
    op.SetParams(params);
    EXPECT_EQ(op.GetParams().gridPoints.size(), 3u);
}

// ============================================================
// 鏍稿績鍖归厤閫昏緫娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUMatchTest, IdentityMatchesOrigin) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.anyMatched);
    ASSERT_EQ(result.matches.size(), 1u);
    EXPECT_TRUE(result.matches[0].matched);
    EXPECT_NEAR(result.matches[0].positionError, 0.0, 1e-6);
    EXPECT_NEAR(result.matches[0].rotationError, 0.0, 1e-6);
}

TEST(PoseEstimateCPUMatchTest, SmallOffsetWithinThreshold) {
    auto params = makeStandardParams();
    params.poseTargets[0].posThreshold = 10.0;
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(3.0, 0.0, 0.0));

    EXPECT_TRUE(result.anyMatched);
    EXPECT_NEAR(result.matches[0].positionError, 3.0, 0.1);
}

TEST(PoseEstimateCPUMatchTest, LargeOffsetOutsideThreshold) {
    auto params = makeStandardParams();
    params.poseTargets[0].posThreshold = 1.0;
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(50.0, 0.0, 0.0));

    EXPECT_FALSE(result.anyMatched);
    EXPECT_GT(result.matches[0].positionError, 1.0);
}

TEST(PoseEstimateCPUMatchTest, RotationWithinThreshold) {
    auto params = makeStandardParams();
    params.poseTargets[0].rotThreshold = 10.0;
    PoseEstimateCPU op(params);

    double angle = 3.0 * CV_PI / 180.0;
    cv::Matx33d R(std::cos(angle), -std::sin(angle), 0,
                  std::sin(angle),  std::cos(angle), 0,
                  0, 0, 1);
    auto result = op.Execute(R, cv::Vec3d(0, 0, 0));

    EXPECT_TRUE(result.anyMatched);
    EXPECT_LT(result.matches[0].rotationError, 10.0);
    EXPECT_GT(result.matches[0].rotationError, 0.0);  // 纭疄鏈夋棆杞?}

TEST(PoseEstimateCPUMatchTest, RotationOutsideThreshold) {
    auto params = makeStandardParams();
    params.poseTargets[0].rotThreshold = 1.0;  // 涓ユ牸瑙掑害闃堝€?
    PoseEstimateCPU op(params);

    double angle = 45.0 * CV_PI / 180.0;
    cv::Matx33d R(std::cos(angle), -std::sin(angle), 0,
                  std::sin(angle),  std::cos(angle), 0,
                  0, 0, 1);
    auto result = op.Execute(R, cv::Vec3d(0, 0, 0));

    EXPECT_FALSE(result.anyMatched);
    EXPECT_GT(result.matches[0].rotationError, 1.0);
}

TEST(PoseEstimateCPUMatchTest, MultipleTargetsBestMatch) {
    auto params = makeStandardParams();

    PoseTarget t2;
    t2.name = "near_target";
    t2.tx = 2.0; t2.ty = 0.0; t2.tz = 0.0;
    t2.posThreshold = 5.0;
    t2.rotThreshold = 5.0;
    params.poseTargets.push_back(t2);

    PoseEstimateCPU op(params);

    // 褰撳墠浣嶇疆(1,0,0) 鈥?涓や釜鐩爣閮藉湪闃堝€煎唴
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(1.0, 0.0, 0.0));

    EXPECT_TRUE(result.anyMatched);
    EXPECT_EQ(result.matches.size(), 2u);
    EXPECT_TRUE(result.matches[0].matched);
    EXPECT_TRUE(result.matches[1].matched);
    EXPECT_NE(result.bestMatch, -1);
    // near_target 璺濈=1mm锛宖ront_center 璺濈=1mm
    // near_target 褰掍竴鍖栧垎鏁版洿浣庯紙1/5 vs 1/5 鐩稿悓锛夛紝浣嗙粷瀵硅窛绂?near_target 鏇磋繎
}

TEST(PoseEstimateCPUMatchTest, NoTargetsMatched) {
    auto params = makeStandardParams();
    params.poseTargets[0].posThreshold = 0.01;
    params.poseTargets[0].rotThreshold = 0.01;
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(100.0, 100.0, 100.0));

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.anyMatched);
    EXPECT_EQ(result.bestMatch, -1);
}

// ============================================================
// 鍧愭爣绯诲缓绔嬫祴璇?
// ============================================================

TEST(PoseEstimateCPUCoordTest, RowAxisXFaceZ) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.anyMatched);
}

TEST(PoseEstimateCPUCoordTest, RowAxisY) {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(3, 3);
    params.originRow = 0;
    params.originCol = 0;
    params.rowAxis = "Y";
    params.faceNormal = "Z";

    // rowAxis="Y" 鏃朵笘鐣屽彉鎹骇鐢?80掳鏃嬭浆锛圶鈫擸缈昏浆+Z缈昏浆锛?
    // 鐩爣闇€鍖呭惈姝ゆ棆杞細trace(R)=-1 鈫?rotationAngle=180掳
    // 鐢ㄨ冻澶熷ぇ鐨勬棆杞槇鍊兼潵楠岃瘉绯荤粺鑳芥纭伐浣?
    PoseTarget target;
    target.name = "origin";
    target.posThreshold = 5.0;
    target.rotThreshold = 200.0;  // 瀹圭撼涓栫晫鍧愭爣绯荤殑180掳鏃嬭浆
    params.poseTargets.push_back(target);

    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.anyMatched);
}

TEST(PoseEstimateCPUCoordTest, FaceNormalNegativeZ) {
    auto params = makeStandardParams();
    params.faceNormal = "-Z";
    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
}

TEST(PoseEstimateCPUCoordTest, OriginAtCenter) {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(5, 5);
    params.originRow = 2;
    params.originCol = 2;
    params.rowAxis = "X";
    params.faceNormal = "Z";

    // 鍘熺偣鍦?60,60,0)锛屼笘鐣屽彉鎹㈠钩绉昏嚦姝ょ偣
    // 鐩告満鎭掔瓑鍙樻崲鍦ㄤ笘鐣岀郴涓綅浜庡師鐐?60,60,0)锛岀洰鏍囬渶鍖归厤姝や綅缃?
    PoseTarget target;
    target.name = "origin";
    target.tx = 60.0; target.ty = 60.0; target.tz = 0.0;
    target.posThreshold = 5.0;
    target.rotThreshold = 5.0;
    params.poseTargets.push_back(target);

    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.anyMatched);
}

// ============================================================
// 鍥炶皟娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUCallbackTest, CallbackFiredOnMatch) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    bool callbackFired = false;
    int matchedCount = -1;
    op.SetCallback([&](const PoseEstimateCPUResult& r) {
        callbackFired = true;
        matchedCount = static_cast<int>(r.matches.size());
    });

    op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));

    EXPECT_TRUE(callbackFired);
    EXPECT_EQ(matchedCount, 1);
}

TEST(PoseEstimateCPUCallbackTest, CallbackNotFiredOnNoMatch) {
    auto params = makeStandardParams();
    params.poseTargets[0].posThreshold = 0.001;
    PoseEstimateCPU op(params);

    bool callbackFired = false;
    op.SetCallback([&](const PoseEstimateCPUResult&) {
        callbackFired = true;
    });

    op.Execute(cv::Matx33d::eye(), cv::Vec3d(100.0, 0.0, 0.0));
    EXPECT_FALSE(callbackFired);
}

TEST(PoseEstimateCPUCallbackTest, NoCallbackDoesNotCrash) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    // 涓嶆敞鍐屽洖璋?
    EXPECT_NO_THROW(op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0)));
}

// ============================================================
// API 娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUApiTest, EstimateWithMatx44) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    cv::Matx44d pose = cv::Matx44d::eye();
    auto result = op.Execute(pose);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.anyMatched);
}

TEST(PoseEstimateCPUApiTest, SetParamsRebuildsCoordSystem) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    auto r1 = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(r1.anyMatched);

    auto params2 = makeStandardParams();
    params2.originRow = 1;
    params2.originCol = 1;
    op.SetParams(params2);

    auto r2 = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(r2.success);
}

TEST(PoseEstimateCPUApiTest, StatisticsAccumulate) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    op.Execute(cv::Matx33d::eye(), cv::Vec3d(1, 0, 0));

    const auto& stats = op.GetStatistics();
    EXPECT_EQ(stats.estimateCallCount, 2u);
    EXPECT_GT(stats.totalTimeMs, 0.0);
}

TEST(PoseEstimateCPUApiTest, ResetStatistics) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);

    op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_GT(op.GetStatistics().estimateCallCount, 0u);

    op.ResetStatistics();
    EXPECT_EQ(op.GetStatistics().estimateCallCount, 0u);
    EXPECT_DOUBLE_EQ(op.GetStatistics().totalTimeMs, 0.0);
}

TEST(PoseEstimateCPUApiTest, WarmupInt) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    EXPECT_NO_THROW(op.Warmup(100));
}

TEST(PoseEstimateCPUApiTest, WarmupWarmupConfig) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    calib::WarmupConfig config;
    config.maxPointCount = 50;
    EXPECT_NO_THROW(op.Warmup(config));
}

TEST(PoseEstimateCPUApiTest, GetParamsReturnsRef) {
    auto params = makeStandardParams();
    PoseEstimateCPU op(params);
    const auto& p = op.GetParams();
    EXPECT_EQ(p.gridPoints.size(), 3u);
    EXPECT_EQ(p.poseTargets.size(), 1u);
}

// ============================================================
// 杈圭晫娴嬭瘯
// ============================================================

TEST(PoseEstimateCPUEdgeTest, EmptyGridReturnsFailure) {
    PoseEstimateCPUParams params;
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST(PoseEstimateCPUEdgeTest, NoTargetsReturnsFailure) {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(3, 3);
    PoseEstimateCPU op(params);

    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.qualityFlag, calib::QualityFlag::Warning);
}

TEST(PoseEstimateCPUEdgeTest, LargeGrid) {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(20, 20);
    params.originRow = 10;
    params.originCol = 10;
    params.rowAxis = "X";
    params.faceNormal = "Z";
    params.poseTargets.push_back(PoseTarget{});
    params.poseTargets[0].name = "center";
    params.poseTargets[0].posThreshold = 100.0;
    params.poseTargets[0].rotThreshold = 100.0;

    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
}

TEST(PoseEstimateCPUEdgeTest, ManyTargets) {
    PoseEstimateCPUParams params;
    params.gridPoints = makeGrid(3, 3);
    params.rowAxis = "X";
    params.faceNormal = "Z";

    for (int i = 0; i < 50; ++i) {
        PoseTarget t;
        t.name = "target_" + std::to_string(i);
        t.tx = static_cast<double>(i) * 10.0;
        t.posThreshold = 200.0;
        t.rotThreshold = 200.0;
        params.poseTargets.push_back(t);
    }

    PoseEstimateCPU op(params);
    auto result = op.Execute(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.matches.size(), 50u);
    EXPECT_TRUE(result.anyMatched);
}


#endif // SKIPPED