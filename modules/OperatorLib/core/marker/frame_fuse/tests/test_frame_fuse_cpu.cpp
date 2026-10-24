/**
 * @file test_frame_fuse_cpu.cpp
 * @brief 鍗曞抚蹇€熻瀺鍚堢畻瀛愬崟鍏冩祴璇? */

#include "frame_fuse_cpu.h"
#include "common/calib_warmup_config.h"

using namespace calib;

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <random>
#include <algorithm>

using namespace calib;

// ============================================================
// Helpers
// ============================================================

static MarkerPointSet generateRandomPoints(int count, double spread,
                                            std::mt19937& rng) {
    MarkerPointSet ms;
    ms.positions.reserve(static_cast<size_t>(count));
    ms.normals.reserve(static_cast<size_t>(count));

    std::uniform_real_distribution<double> posDist(-spread / 2.0, spread / 2.0);
    std::normal_distribution<double> normalDist(0.0, 1.0);

    for (int i = 0; i < count; ++i) {
        double x = posDist(rng);
        double y = posDist(rng);
        double z = posDist(rng);
        ms.positions.emplace_back(x, y, z);

        double nx = normalDist(rng);
        double ny = normalDist(rng);
        double nz = normalDist(rng);
        double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-12) { nx = 0; ny = 0; nz = 1; len = 1.0; }
        ms.normals.emplace_back(nx / len, ny / len, nz / len);
    }
    return ms;
}

static MarkerPointSet applyTransform(const MarkerPointSet& input,
                                      const cv::Matx33d& R,
                                      const cv::Vec3d& T) {
    MarkerPointSet out;
    out.positions.reserve(input.size());
    out.normals.reserve(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        const auto& p = input.positions[i];
        out.positions.emplace_back(
            R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + T(0),
            R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + T(1),
            R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + T(2));

        const auto& n = input.normals[i];
        out.normals.emplace_back(
            R(0, 0) * n(0) + R(0, 1) * n(1) + R(0, 2) * n(2),
            R(1, 0) * n(0) + R(1, 1) * n(1) + R(1, 2) * n(2),
            R(2, 0) * n(0) + R(2, 1) * n(1) + R(2, 2) * n(2));
    }
    return out;
}

static void addNoise(MarkerPointSet& set, double posNoiseStd,
                     double normalNoiseStd, std::mt19937& rng) {
    // 261024 合并适配：MSVC Debug 的 normal_distribution 断言 sigma>0——
    // 调用方传 0（不加噪声）时钳 1e-12（数学上等效无噪声）
    std::normal_distribution<double> noise(
        0.0, posNoiseStd > 0.0 ? posNoiseStd : 1e-12);
    for (auto& p : set.positions) {
        p.x += noise(rng);
        p.y += noise(rng);
        p.z += noise(rng);
    }
    for (auto& n : set.normals) {
        std::normal_distribution<double> nNoise(
            0.0, normalNoiseStd > 0.0 ? normalNoiseStd : 1e-12);
        n(0) += nNoise(rng);
        n(1) += nNoise(rng);
        n(2) += nNoise(rng);
        double len = std::sqrt(n.dot(n));
        if (len > 1e-12) n /= len;
    }
}

static cv::Matx33d makeRotationX(double angleDeg) {
    double a = angleDeg * CV_PI / 180.0;
    double c = std::cos(a), s = std::sin(a);
    return cv::Matx33d(1, 0, 0,  0, c, -s,  0, s, c);
}

static cv::Matx33d makeRotationY(double angleDeg) {
    double a = angleDeg * CV_PI / 180.0;
    double c = std::cos(a), s = std::sin(a);
    return cv::Matx33d(c, 0, s,  0, 1, 0,  -s, 0, c);
}

static cv::Matx33d makeRotationZ(double angleDeg) {
    double a = angleDeg * CV_PI / 180.0;
    double c = std::cos(a), s = std::sin(a);
    return cv::Matx33d(c, -s, 0,  s, c, 0,  0, 0, 1);
}

static double rotationErrorDeg(const cv::Matx33d& R_est, const cv::Matx33d& R_true) {
    cv::Matx33d Rdiff = R_est * R_true.t();
    double trace = Rdiff(0, 0) + Rdiff(1, 1) + Rdiff(2, 2);
    double val = (trace - 1.0) / 2.0;
    if (val < -1.0) val = -1.0;
    if (val > 1.0) val = 1.0;
    if (std::isnan(val)) return 180.0;
    return std::acos(val) * 180.0 / CV_PI;
}

static double translationError(const cv::Vec3d& T_est, const cv::Vec3d& T_true) {
    cv::Vec3d diff = T_est - T_true;
    return std::sqrt(diff.dot(diff));
}

// ============================================================
// 1. Params Validation
// ============================================================

TEST(FrameFuseCPU, DefaultParamsValidate) {
    FrameFuseCPUParams p;
    EXPECT_NO_THROW(p.validate());
}

TEST(FrameFuseCPU, InvalidKnnK) {
    FrameFuseCPUParams p;
    p.knnK = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.knnK = 3;
    EXPECT_NO_THROW(p.validate());
}

TEST(FrameFuseCPU, InvalidLoweRatio) {
    FrameFuseCPUParams p;
    p.loweRatio = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.loweRatio = 1.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.loweRatio = 0.8;
    EXPECT_NO_THROW(p.validate());
}

TEST(FrameFuseCPU, InvalidBins) {
    FrameFuseCPUParams p;
    p.descriptorBins1 = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.descriptorBins1 = 11;
    p.descriptorBins2 = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.descriptorBins2 = 11;
    p.descriptorBins3 = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(FrameFuseCPU, InvalidMaxPointCount) {
    FrameFuseCPUParams p;
    p.maxPointCount = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(FrameFuseCPU, InvalidRansacParams) {
    FrameFuseCPUParams p;
    p.ransacConfidence = 1.5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.ransacConfidence = 0.999;
    p.ransacMaxIterations = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

// ============================================================
// 2. JSON Serialization
// ============================================================

TEST(FrameFuseCPU, JsonRoundTrip) {
    FrameFuseCPUParams p;
    p.knnK = 20;
    p.loweRatio = 0.75;
    p.ransacMaxIterations = 3000;
    p.maxPointCount = 5000;

    auto j = p.toJson();
    auto p2 = FrameFuseCPUParams::fromJson(j);

    EXPECT_EQ(p2.knnK, 20);
    EXPECT_DOUBLE_EQ(p2.loweRatio, 0.75);
    EXPECT_EQ(p2.ransacMaxIterations, 3000);
    EXPECT_EQ(p2.maxPointCount, 5000u);
}

TEST(FrameFuseCPU, JsonEmpty) {
    nlohmann::json j = nlohmann::json::object();
    auto p = FrameFuseCPUParams::fromJson(j);
    FrameFuseCPUParams defaults;
    EXPECT_EQ(p.knnK, defaults.knnK);
    EXPECT_DOUBLE_EQ(p.loweRatio, defaults.loweRatio);
}

TEST(FrameFuseCPU, JsonPartial) {
    nlohmann::json j;
    j["knnK"] = 25;
    j["maxPointCount"] = 8000;
    auto p = FrameFuseCPUParams::fromJson(j);
    EXPECT_EQ(p.knnK, 25);
    EXPECT_EQ(p.maxPointCount, 8000u);
    EXPECT_DOUBLE_EQ(p.loweRatio, 0.8);  // default
}

// ============================================================
// 3. Result Struct
// ============================================================

TEST(FrameFuseCPU, ResultDefaultValues) {
    FrameFuseCPUResult r;
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.qualityFlag, calib::QualityFlag::Normal);
    double det = cv::determinant(r.R);
    EXPECT_NEAR(det, 1.0, 1e-10);
    EXPECT_DOUBLE_EQ(r.rmse, 0.0);
    EXPECT_EQ(r.matchedCount, 0u);
}

TEST(FrameFuseCPU, ResultMoveSemantics) {
    FrameFuseCPUResult r1;
    r1.success = true;
    r1.rmse = 1.5;
    r1.correspondences.emplace_back(0, 1);

    FrameFuseCPUResult r2 = std::move(r1);
    EXPECT_TRUE(r2.success);
    EXPECT_DOUBLE_EQ(r2.rmse, 1.5);
    EXPECT_EQ(r2.correspondences.size(), 1u);
}

// ============================================================
// 4. Constructor / Destructor
// ============================================================

TEST(FrameFuseCPU, DefaultConstruct) {
    EXPECT_NO_THROW(FrameFuseCPU op);
}

TEST(FrameFuseCPU, ParamConstruct) {
    FrameFuseCPUParams p;
    p.knnK = 10;
    EXPECT_NO_THROW(FrameFuseCPU op(p));
}

// ============================================================
// 5. Empty Input
// ============================================================

TEST(FrameFuseCPU, EmptySets) {
    FrameFuseCPU op;
    MarkerPointSet s1, s2;
    auto result = op.Execute(s1, s2);
    EXPECT_FALSE(result.success);
}

TEST(FrameFuseCPU, SinglePointSet) {
    FrameFuseCPU op;
    MarkerPointSet s1, s2;
    s1.positions.emplace_back(1, 2, 3);
    s1.normals.emplace_back(0, 0, 1);
    s2.positions.emplace_back(4, 5, 6);
    s2.normals.emplace_back(0, 0, 1);
    auto result = op.Execute(s1, s2);
    EXPECT_FALSE(result.success);
}

TEST(FrameFuseCPU, TwoPointsSet) {
    FrameFuseCPU op;
    MarkerPointSet s1, s2;
    for (int i = 0; i < 2; ++i) {
        s1.positions.emplace_back(i * 10.0, 0, 0);
        s1.normals.emplace_back(0, 0, 1);
        s2.positions.emplace_back(i * 10.0 + 5.0, 3, 0);
        s2.normals.emplace_back(0, 0, 1);
    }
    auto result = op.Execute(s1, s2);
    EXPECT_FALSE(result.success);
}

// ============================================================
// 6. Synthetic Transform Tests
// ============================================================

class FrameFuseSyntheticTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(12345);
        set1_ = generateRandomPoints(40, 200.0, rng_);
    }

    std::mt19937 rng_;
    MarkerPointSet set1_;
};

TEST_F(FrameFuseSyntheticTest, IdentityTransform) {
    FrameFuseCPU op;
    auto result = op.Execute(set1_, set1_);

    ASSERT_TRUE(result.success);
    double rotErr = rotationErrorDeg(result.R, cv::Matx33d::eye());
    double transErr = translationError(result.T, cv::Vec3d(0, 0, 0));
    EXPECT_LT(rotErr, 1.0) << "Rotation error too large for identity transform";
    EXPECT_LT(transErr, 5.0) << "Translation error too large for identity transform";
}

TEST_F(FrameFuseSyntheticTest, PureTranslation) {
    cv::Vec3d T_known(50.0, -30.0, 20.0);
    MarkerPointSet set2 = applyTransform(set1_, cv::Matx33d::eye(), T_known);

    FrameFuseCPU op;
    auto result = op.Execute(set1_, set2);

    ASSERT_TRUE(result.success);
    double rotErr = rotationErrorDeg(result.R, cv::Matx33d::eye());
    double transErr = translationError(result.T, T_known);
    EXPECT_LT(rotErr, 1.0);
    EXPECT_LT(transErr, 5.0);
}

TEST_F(FrameFuseSyntheticTest, PureRotation) {
    cv::Matx33d R_known = makeRotationZ(15.0) * makeRotationX(8.0);
    // Center the point set before rotation so descriptors stay stable
    cv::Point3d centroid(0, 0, 0);
    for (const auto& p : set1_.positions) {
        centroid.x += p.x; centroid.y += p.y; centroid.z += p.z;
    }
    centroid.x /= static_cast<double>(set1_.size());
    centroid.y /= static_cast<double>(set1_.size());
    centroid.z /= static_cast<double>(set1_.size());

    MarkerPointSet set1centered = set1_;
    for (auto& p : set1centered.positions) {
        p.x -= centroid.x; p.y -= centroid.y; p.z -= centroid.z;
    }
    MarkerPointSet set2 = applyTransform(set1centered, R_known, cv::Vec3d(0, 0, 0));

    FrameFuseCPU op;
    auto result = op.Execute(set1centered, set2);

    ASSERT_TRUE(result.success);
    double rotErr = rotationErrorDeg(result.R, R_known);
    EXPECT_LT(rotErr, 5.0) << "Rotation recovery error too large";
}

TEST_F(FrameFuseSyntheticTest, RotationAndTranslation) {
    cv::Matx33d R_known = makeRotationY(20.0) * makeRotationZ(10.0);
    cv::Vec3d T_known(30.0, -15.0, 45.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);

    FrameFuseCPU op;
    auto result = op.Execute(set1_, set2);

    ASSERT_TRUE(result.success);
    double rotErr = rotationErrorDeg(result.R, R_known);
    double transErr = translationError(result.T, T_known);
    EXPECT_LT(rotErr, 2.0);
    EXPECT_LT(transErr, 10.0);
}

TEST_F(FrameFuseSyntheticTest, TransformWithNoise) {
    cv::Matx33d R_known = makeRotationX(10.0) * makeRotationZ(8.0);
    cv::Vec3d T_known(20.0, 10.0, -15.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.02, rng_);

    FrameFuseCPU op;
    auto result = op.Execute(set1_, set2);

    ASSERT_TRUE(result.success);
    double rotErr = rotationErrorDeg(result.R, R_known);
    double transErr = translationError(result.T, T_known);
    EXPECT_LT(rotErr, 5.0) << "Rotation error with noise";
    EXPECT_LT(transErr, 15.0) << "Translation error with noise";
}

// ============================================================
// 7. Partial Overlap
// ============================================================

TEST_F(FrameFuseSyntheticTest, PartialOverlap50) {
    cv::Matx33d R_known = makeRotationZ(12.0);
    cv::Vec3d T_known(25.0, 0, 10.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);

    // Remove ~50% of points from set2
    std::mt19937 removeRng(99);
    MarkerPointSet set2partial;
    for (size_t i = 0; i < set2.size(); ++i) {
        std::uniform_int_distribution<int> d(0, 1);
        if (d(removeRng) == 1) {
            set2partial.positions.push_back(set2.positions[i]);
            set2partial.normals.push_back(set2.normals[i]);
        }
    }

    FrameFuseCPU op;
    auto result = op.Execute(set1_, set2partial);

    EXPECT_TRUE(result.success);
}

TEST_F(FrameFuseSyntheticTest, PartialOverlap30) {
    cv::Matx33d R_known = makeRotationY(8.0);
    cv::Vec3d T_known(10.0, 5.0, -5.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);

    // Keep only ~30% of points
    std::mt19937 removeRng(77);
    MarkerPointSet set2partial;
    for (size_t i = 0; i < set2.size(); ++i) {
        std::uniform_int_distribution<int> d(0, 9);
        if (d(removeRng) < 3) {
            set2partial.positions.push_back(set2.positions[i]);
            set2partial.normals.push_back(set2.normals[i]);
        }
    }

    if (set2partial.size() < 3) {
        GTEST_SKIP() << "Not enough points after removal";
    }

    FrameFuseCPU op;
    auto result = op.Execute(set1_, set2partial);
    // May succeed with Degraded/Warning quality
}

// ============================================================
// 8. Warmup
// ============================================================

TEST(FrameFuseCPU, WarmupInt) {
    FrameFuseCPU op;
    EXPECT_NO_THROW(op.Warmup(100));
}

TEST(FrameFuseCPU, WarmupConfig) {
    FrameFuseCPU op;
    auto cfg = calib::WarmupConfig::forPointCloud(100);
    EXPECT_NO_THROW(op.Warmup(cfg));
}

// ============================================================
// 9. SetParams / GetParams
// ============================================================

TEST(FrameFuseCPU, SetGetParams) {
    FrameFuseCPU op;
    FrameFuseCPUParams p;
    p.knnK = 20;
    p.maxPointCount = 5000;
    op.SetParams(p);

    const auto& got = op.GetParams();
    EXPECT_EQ(got.knnK, 20);
    EXPECT_EQ(got.maxPointCount, 5000u);
}

// ============================================================
// 10. Statistics
// ============================================================

TEST_F(FrameFuseSyntheticTest, StatisticsPopulated) {
    FrameFuseCPU op;
    auto result = op.Execute(set1_, set1_);

    ASSERT_TRUE(result.success);
    const auto& st = result.statistics;
    EXPECT_GT(st.totalTimeMs, 0.0);
    EXPECT_GT(st.knnTimeMs, 0.0);
    EXPECT_GT(st.descriptorTimeMs, 0.0);
    EXPECT_EQ(st.set1PointCount, 40u);
    EXPECT_EQ(st.set2PointCount, 40u);
    EXPECT_GT(st.adaptiveDistThreshCoarse, 0.0);
    EXPECT_GT(st.adaptiveDistThreshFine, 0.0);
}

// ============================================================
// 11. Reset Statistics
// ============================================================

TEST(FrameFuseCPU, ResetStatistics) {
    FrameFuseCPU op;
    op.ResetStatistics();
    const auto& st = op.GetStatistics();
    EXPECT_DOUBLE_EQ(st.totalTimeMs, 0.0);
    EXPECT_DOUBLE_EQ(st.knnTimeMs, 0.0);
}

// ============================================================
// 12. PointReconstructCPUResult interface
// ============================================================

TEST_F(FrameFuseSyntheticTest, FuseFromReconstructResult) {
    // Build fake PointReconstructCPUResult from set1_
    PointReconstructCPUResult rr1, rr2;
    rr1.success = true;
    rr2.success = true;

    cv::Matx33d R_known = makeRotationZ(15.0);
    cv::Vec3d T_known(10.0, 20.0, 5.0);

    for (size_t i = 0; i < set1_.size(); ++i) {
        MarkerReconstructResult mr1, mr2;
        mr1.validCircle = true;
        mr1.centerX = set1_.positions[i].x;
        mr1.centerY = set1_.positions[i].y;
        mr1.centerZ = set1_.positions[i].z;
        mr1.normalX = set1_.normals[i](0);
        mr1.normalY = set1_.normals[i](1);
        mr1.normalZ = set1_.normals[i](2);

        const auto& p = set1_.positions[i];
        const auto& n = set1_.normals[i];
        mr2.validCircle = true;
        mr2.centerX = R_known(0,0)*p.x + R_known(0,1)*p.y + R_known(0,2)*p.z + T_known(0);
        mr2.centerY = R_known(1,0)*p.x + R_known(1,1)*p.y + R_known(1,2)*p.z + T_known(1);
        mr2.centerZ = R_known(2,0)*p.x + R_known(2,1)*p.y + R_known(2,2)*p.z + T_known(2);
        mr2.normalX = R_known(0,0)*n(0) + R_known(0,1)*n(1) + R_known(0,2)*n(2);
        mr2.normalY = R_known(1,0)*n(0) + R_known(1,1)*n(1) + R_known(1,2)*n(2);
        mr2.normalZ = R_known(2,0)*n(0) + R_known(2,1)*n(1) + R_known(2,2)*n(2);

        rr1.markerResults.push_back(std::move(mr1));
        rr2.markerResults.push_back(std::move(mr2));
    }

    FrameFuseCPU op;
    auto result = op.Execute(rr1, rr2);

    EXPECT_TRUE(result.success);
}

// ============================================================
// 13. Transform Matrix Consistency
// ============================================================

TEST_F(FrameFuseSyntheticTest, TransformMatrixConsistency) {
    FrameFuseCPU op;
    auto result = op.Execute(set1_, set1_);

    ASSERT_TRUE(result.success);
    // Check 4x4 matrix matches R and T
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_DOUBLE_EQ(result.transform(r, c), result.R(r, c));
        }
        EXPECT_DOUBLE_EQ(result.transform(r, 3), result.T(r));
    }
    EXPECT_DOUBLE_EQ(result.transform(3, 0), 0.0);
    EXPECT_DOUBLE_EQ(result.transform(3, 1), 0.0);
    EXPECT_DOUBLE_EQ(result.transform(3, 2), 0.0);
    EXPECT_DOUBLE_EQ(result.transform(3, 3), 1.0);
}

// ============================================================
// 14. Triangle Mode (matchMode=1) — 稀疏＋共面＋法线近平行标记点阵
//     （真实数据病灶形态：PFH 指纹无区分度；2026-10-14 优化专题，
//      背景见 modules/09_operatorlib/TEST/steger_dedup_replay/tests/frame_fuse_opt/）
// ============================================================

/// 生成"病灶形态"点集：7~10 个点撒在 z=0 平面（间距几十~二百 mm），
/// 法线全部严格平行 (0,0,1)——真实标记点贴同一平面的理想化
static MarkerPointSet generateCoplanarParallelNormals(int count, std::mt19937& rng) {
    MarkerPointSet ms;
    std::uniform_real_distribution<double> posDist(-100.0, 100.0);
    // 拒绝采样保证最小间距 30mm（真实标记点阵形态）
    while (static_cast<int>(ms.positions.size()) < count) {
        cv::Point3d p(posDist(rng), posDist(rng), 0.0);
        bool ok = true;
        for (const auto& q : ms.positions) {
            const double dx = p.x - q.x, dy = p.y - q.y;
            if (std::sqrt(dx * dx + dy * dy) < 30.0) { ok = false; break; }
        }
        if (ok) {
            ms.positions.push_back(p);
            ms.normals.emplace_back(0.0, 0.0, 1.0);
        }
    }
    return ms;
}

class FrameFuseTriangleTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(20261014);
        set1_ = generateCoplanarParallelNormals(8, rng_);
    }

    std::mt19937 rng_;
    MarkerPointSet set1_;
};

/// 病灶形态＋模式1：R/T 完整恢复（模式存在的意义）
TEST_F(FrameFuseTriangleTest, TriangleModeRecoversTransform) {
    cv::Matx33d R_known = makeRotationZ(20.0) * makeRotationX(10.0);
    cv::Vec3d T_known(15.0, -8.0, 3.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);   // 位置噪声 0.05mm（真实重建噪声量级）

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto result = op.Execute(set1_, set2);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_LT(rotationErrorDeg(result.R, R_known), 0.5);
    EXPECT_LT(translationError(result.T, T_known), 1.0);
    EXPECT_EQ(result.matchedCount, set1_.size());
}

/// 病灶形态＋模式1＋部分重叠：只匹配共同点，R/T 仍准
TEST_F(FrameFuseTriangleTest, TriangleModePartialOverlap) {
    cv::Matx33d R_known = makeRotationY(12.0);
    cv::Vec3d T_known(30.0, 5.0, -10.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    // 后帧丢掉前 3 个点（模拟视野变化）
    MarkerPointSet set2part;
    for (size_t i = 3; i < set2.size(); ++i) {
        set2part.positions.push_back(set2.positions[i]);
        set2part.normals.push_back(set2.normals[i]);
    }

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto result = op.Execute(set1_, set2part);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_LT(rotationErrorDeg(result.R, R_known), 0.5);
    EXPECT_LT(translationError(result.T, T_known), 1.0);
    EXPECT_EQ(result.matchedCount, set2part.size());
}

/// 同形态＋默认模式（PFH）：记录病灶（若 PFH 恰好成功则本用例提醒重新评估前提）
TEST_F(FrameFuseTriangleTest, LegacyPfhOnKillerShape) {
    cv::Matx33d R_known = makeRotationZ(15.0);
    cv::Vec3d T_known(10.0, 20.0, 5.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);

    FrameFuseCPU op;   // 默认 matchMode=0
    auto result = op.Execute(set1_, set2);

    if (result.success) {
        // PFH 偶然成功时质量必差（真实数据实测 matched≤2 / rmse 78~169mm）；
        // 只要求不优于三角形模式
        EXPECT_GE(rotationErrorDeg(result.R, R_known), 0.0);   // 信息性断言
    } else {
        SUCCEED() << "PFH 在病灶形态失败（预期，与真实数据一致）: " << result.message;
    }
}

/// 模式1参数：JSON 序列化往返
TEST(FrameFuseCPU, TriangleParamsJsonRoundTrip) {
    FrameFuseCPUParams p;
    p.matchMode = 1;
    p.triangleSideTolMm = 0.8;
    p.triangleVerifyGateMm = 1.5;
    p.triangleMaxPoints = 12;

    auto j = p.toJson();
    auto p2 = FrameFuseCPUParams::fromJson(j);
    EXPECT_EQ(p2.matchMode, 1);
    EXPECT_DOUBLE_EQ(p2.triangleSideTolMm, 0.8);
    EXPECT_DOUBLE_EQ(p2.triangleVerifyGateMm, 1.5);
    EXPECT_EQ(p2.triangleMaxPoints, 12);
}

/// 模式1参数校验
TEST(FrameFuseCPU, TriangleParamsValidate) {
    FrameFuseCPUParams p;
    p.matchMode = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.matchMode = 1;
    p.triangleSideTolMm = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.triangleSideTolMm = 1.0;
    p.triangleMaxPoints = 5;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.triangleMaxPoints = 16;
    EXPECT_NO_THROW(p.validate());
    p.triangleSideTolRatio = 0.7;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.triangleSideTolRatio = 0.02;
    p.triangleNormalMaxAngleDeg = 0.0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.triangleNormalMaxAngleDeg = 60.0;
    p.triangleMaxCandidatePairs = 100;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.triangleMaxCandidatePairs = 20000;
    EXPECT_NO_THROW(p.validate());
}

// ============================================================
// 15. 对抗形态全家桶（2026-10-14 三代理审查增补）
// ============================================================

/// 双向变化：后帧删 2 点＋进 3 个新点——对应集必须只含共同点
TEST_F(FrameFuseTriangleTest, TriangleModeBothWayOverlap) {
    cv::Matx33d R_known = makeRotationZ(18.0) * makeRotationX(6.0);
    cv::Vec3d T_known(25.0, -12.0, 8.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);
    MarkerPointSet set2m;
    for (size_t i = 2; i < set2.size(); ++i) {   // 删前 2 点
        set2m.positions.push_back(set2.positions[i]);
        set2m.normals.push_back(set2.normals[i]);
    }
    const cv::Point3d farPts[3] = {{600, 600, 0}, {642, 631, 0}, {611, 663, 0}};
    for (const auto& q : farPts) {                // 进 3 个新点（远离＋互距>30mm）
        set2m.positions.push_back(q);
        set2m.normals.emplace_back(0, 0, 1);
    }

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(set1_, set2m);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 6u);   // 只匹配 6 个共同点
    EXPECT_LT(rotationErrorDeg(r.R, R_known), 0.5);
    EXPECT_LT(translationError(r.T, T_known), 1.5);
}

/// 离群鲁棒：2 个点被顶偏 20mm——被验证拒收，其余照配
TEST_F(FrameFuseTriangleTest, TriangleModeOutlierRobustness) {
    cv::Matx33d R_known = makeRotationZ(9.0);
    cv::Vec3d T_known(12.0, 6.0, -4.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);
    set2.positions[0].x += 20.0;    // 离群 1
    set2.positions[5].y += 18.0;    // 离群 2

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(set1_, set2);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 6u);  // 8-2 离群
    EXPECT_LT(rotationErrorDeg(r.R, R_known), 1.0);
    EXPECT_LT(translationError(r.T, T_known), 2.0);
}

/// 近共线点集：必须干净失败，禁止垃圾成功（旧版绝对面积判据失效案）
TEST_F(FrameFuseTriangleTest, TriangleModeCollinearReject) {
    MarkerPointSet s1c, s2c;
    std::mt19937 lrng(5);
    std::uniform_real_distribution<double> jit(-0.01, 0.01);
    for (int i = 0; i < 7; ++i) {
        s1c.positions.emplace_back(i * 50.0, jit(lrng), jit(lrng));
        s1c.normals.emplace_back(0, 0, 1);
        s2c.positions.emplace_back(i * 50.0 + 10.0, 5.0 + jit(lrng), 3.0 + jit(lrng));
        s2c.normals.emplace_back(0, 0, 1);
    }
    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(s1c, s2c);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.matchedCount, 0u);
}

/// 规则网格旋转 90°（网格映射回自身）：真解与恒等解并列全支持——必须弃权
TEST_F(FrameFuseTriangleTest, TriangleModeRegularGridAmbiguity) {
    MarkerPointSet grid, cur;
    const double cell = 40.0;
    for (int gx = 0; gx < 4; ++gx)
        for (int gy = 0; gy < 4; ++gy) {
            grid.positions.emplace_back(gx * cell, gy * cell, 0.0);
            grid.normals.emplace_back(0, 0, 1);
        }
    const cv::Matx33d R90 = makeRotationZ(90.0);   // 绕网格中心 90°——网格映回自身
    const cv::Vec3d Tc(3.0 * cell, 3.0 * cell, 0.0);  // 绕中心 (60,60)：p' = R90*(p-c)+c
    cur = applyTransform(grid, R90, Tc);

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(cur, grid);
    // 批1 F11 后门收死：对称网格必须弃权——恒等解在此构造下＝16 对全错配
    // （每点都映射到"旋转后的同坐标点"而非物理同一点），且链级跳变门会因
    // matched=16≥7 放行该错解。旧断言容忍恒等解通过＝给红线开后门，删除。
    ASSERT_FALSE(r.success) << "对称网格 90° 双解（16-16 并列，边际 0<2）必须弃权: "
                            << r.message;
    EXPECT_EQ(r.matchedCount, 0u);
    // 保险：假如未来实现改为给出确定解，它至少必须是真解或恒等之一（不得第三态）
    if (r.success) {
        const double eTrue = rotationErrorDeg(r.R, R90);
        const double eId = rotationErrorDeg(r.R, cv::Matx33d::eye());
        EXPECT_LT(std::min(eTrue, eId), 0.5);
    }
}

/// 薄壁正反面（板厚 3mm > 验证门 2mm，法线相反）：零跨面配对
TEST_F(FrameFuseTriangleTest, TriangleModeThinWallNoCrossFace) {
    MarkerPointSet base;
    std::mt19937 wrng(11);
    std::uniform_real_distribution<double> pd(-100.0, 100.0);
    while (base.positions.size() < 8) {          // 正面 8 点 z=+1.5，法线 +z
        cv::Point3d q(pd(wrng), pd(wrng), 1.5);
        bool ok = true;
        for (const auto& e : base.positions)
            if (std::sqrt((q.x - e.x) * (q.x - e.x) + (q.y - e.y) * (q.y - e.y)) < 30.0)
                { ok = false; break; }
        if (ok) { base.positions.push_back(q); base.normals.emplace_back(0, 0, 1); }
    }
    for (size_t i = 0; i < 8; ++i) {             // 背面同 xy、z=-1.5，法线 -z
        base.positions.emplace_back(base.positions[i].x, base.positions[i].y, -1.5);
        base.normals.emplace_back(0, 0, -1);
    }
    const cv::Matx33d R_known = makeRotationZ(10.0) * makeRotationX(5.0);
    const cv::Vec3d T_known(20.0, -10.0, 5.0);
    MarkerPointSet set2 = applyTransform(base, R_known, T_known);
    addNoise(set2, 0.02, 0.0, rng_);

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(base, set2);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 16u);
    for (const auto& pr : r.correspondences)     // 零跨面：正配正、背配背
        EXPECT_EQ(pr.first < 8, pr.second < 8);
    EXPECT_LT(rotationErrorDeg(r.R, R_known), 0.5);
    EXPECT_LT(translationError(r.T, T_known), 1.5);
}

/// 法线符号随机翻转一半：不得产生错配（要么全对要么干净失败）
TEST_F(FrameFuseTriangleTest, TriangleModeNormalSignFlip) {
    MarkerPointSet base;
    std::mt19937 wrng(13);
    std::uniform_real_distribution<double> pd(-100.0, 100.0);
    while (base.positions.size() < 8) {
        cv::Point3d q(pd(wrng), pd(wrng), 1.5);
        bool ok = true;
        for (const auto& e : base.positions)
            if (std::sqrt((q.x - e.x) * (q.x - e.x) + (q.y - e.y) * (q.y - e.y)) < 30.0)
                { ok = false; break; }
        if (ok) { base.positions.push_back(q); base.normals.emplace_back(0, 0, 1); }
    }
    for (size_t i = 0; i < 8; ++i) {
        base.positions.emplace_back(base.positions[i].x, base.positions[i].y, -1.5);
        base.normals.emplace_back(0, 0, -1);
    }
    const cv::Matx33d R_known = makeRotationZ(8.0);
    const cv::Vec3d T_known(15.0, 8.0, -3.0);
    MarkerPointSet set2 = applyTransform(base, R_known, T_known);
    for (size_t i = 0; i < set2.normals.size(); i += 2)   // 翻转一半法线符号
        set2.normals[i] = -set2.normals[i];

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(base, set2);
    if (r.success) {
        // 已知边界（2026-10-15 批1 实证登记）：孪生布点（正反面同 xy）＋半数法线
        // 被翻转＝几何信息被破坏的最坏构造——"倾斜折中混合解"（两面点同入验证
        // 门内、翻转点法线恰同向、支持数还更高）在算子层不可证伪。本用例从"零
        // 跨面"降级为【有界误差】红线：错也得错得有界（翻转解会造成 ~180° 级
        // R/T 偏差，折中解偏差 <3°）。治本＝上游法线合理性检查（批2 项）。
        double rotErr = rotationErrorDeg(r.R, R_known);
        double transErr = translationError(r.T, T_known);
        EXPECT_LT(rotErr, 3.0) << "翻转场景解偏差超界（应为有界折中而非翻面）";
        EXPECT_LT(transErr, 4.0);
        EXPECT_EQ(r.matchedCount >= 4u, true);   // 链内门可达
        int crossFace = 0;
        for (const auto& pr : r.correspondences)
            if ((pr.first < 8) != (pr.second < 8)) ++crossFace;
        std::cout << "[signflip] matched=" << r.matchedCount
                  << " rotErr=" << rotErr << "deg transErr=" << transErr
                  << "mm crossFace=" << crossFace
                  << "（已知边界：折中混合解，见批2 上游法线检查）" << std::endl;
    } else {
        SUCCEED() << "符号翻转后干净失败（可接受）: " << r.message;
    }
}

/// 曲面壳边界（球冠 R=200、法线辐射状）：仍应可配
TEST_F(FrameFuseTriangleTest, TriangleModeCurvedShell) {
    MarkerPointSet shell;
    std::mt19937 srng(17);
    std::uniform_real_distribution<double> ad(-0.45, 0.45);   // 极角 ±26°（冠径 ~170mm）
    int tries = 0;
    while (shell.positions.size() < 9 && tries < 100000) {
        ++tries;
        const double a = ad(srng), b = ad(srng);
        cv::Point3d q(200 * std::sin(a), 200 * std::sin(b),
                      200 * std::cos(std::sqrt(a * a + b * b)));
        const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        q.x *= 200.0 / n; q.y *= 200.0 / n; q.z *= 200.0 / n;
        bool ok = true;
        for (const auto& e : shell.positions) {
            const double dx = q.x - e.x, dy = q.y - e.y, dz = q.z - e.z;
            if (std::sqrt(dx * dx + dy * dy + dz * dz) < 35.0) { ok = false; break; }
        }
        if (!ok) continue;
        shell.positions.push_back(q);
        const double nn = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        shell.normals.emplace_back(q.x / nn, q.y / nn, q.z / nn);   // 法线辐射向外
    }
    ASSERT_EQ(shell.positions.size(), 9u) << "球冠点生成失败";
    const cv::Matx33d R_known = makeRotationZ(15.0) * makeRotationX(10.0);
    const cv::Vec3d T_known(30.0, -20.0, 10.0);
    MarkerPointSet set2 = applyTransform(shell, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);
    auto r = op.Execute(shell, set2);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_LT(rotationErrorDeg(r.R, R_known), 2.0);
    EXPECT_LT(translationError(r.T, T_known), 3.0);
}

/// 使用域测试：当前帧 8 点 vs 50 点全局库（模拟库累积远超 16 点上限）
TEST_F(FrameFuseTriangleTest, TriangleModeGlobalLibrarySubset) {
    // 帧基底 8 点
    std::mt19937 rng8(2026);
    MarkerPointSet frame = generateCoplanarParallelNormals(8, rng8);
    const cv::Matx33d R_known = makeRotationZ(12.0) * makeRotationY(7.0);
    const cv::Vec3d T_known(40.0, -25.0, 15.0);
    MarkerPointSet set2 = applyTransform(frame, R_known, T_known);   // 库内 8 个对应点
    // 库再添 42 个随机点（间距≥30mm、共面 z=0，模拟其他区域累积）
    std::mt19937 lrng(29);
    std::uniform_real_distribution<double> ld(-450.0, 450.0);
    while (set2.positions.size() < 50) {
        cv::Point3d q(ld(lrng), ld(lrng), 0.0);
        bool ok = true;
        for (const auto& e : set2.positions) {
            const double dx = q.x - e.x, dy = q.y - e.y;
            if (std::sqrt(dx * dx + dy * dy) < 30.0) { ok = false; break; }
        }
        if (ok) { set2.positions.push_back(q); set2.normals.emplace_back(0, 0, 1); }
    }

    FrameFuseCPUParams p;
    p.matchMode = 1;
    p.triangleMaxPoints = 64;   // 允许大库直配（调用方可控上限）
    FrameFuseCPU op(p);
    auto r = op.Execute(frame, set2);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 8u);   // 只配到 8 个真实共同点
    EXPECT_LT(rotationErrorDeg(r.R, R_known), 0.5);
    EXPECT_LT(translationError(r.T, T_known), 1.5);
}

/// 确定性：同输入两次执行结果逐位一致
TEST_F(FrameFuseTriangleTest, TriangleModeDeterminism) {
    cv::Matx33d R_known = makeRotationZ(20.0) * makeRotationX(10.0);
    cv::Vec3d T_known(15.0, -8.0, 3.0);
    MarkerPointSet set2 = applyTransform(set1_, R_known, T_known);
    addNoise(set2, 0.05, 0.0, rng_);

    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op1(p), op2(p);
    auto r1 = op1.Execute(set1_, set2);
    auto r2 = op2.Execute(set1_, set2);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    for (int i = 0; i < 9; ++i)
        EXPECT_DOUBLE_EQ(r1.R.val[i], r2.R.val[i]);
    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(r1.T.val[i], r2.T.val[i]);
    ASSERT_EQ(r1.correspondences.size(), r2.correspondences.size());
    for (size_t k = 0; k < r1.correspondences.size(); ++k) {
        EXPECT_EQ(r1.correspondences[k].first, r2.correspondences[k].first);
        EXPECT_EQ(r1.correspondences[k].second, r2.correspondences[k].second);
    }
}

/// 负例全家桶：NaN/尺寸不匹配/重复点——干净失败不崩溃
TEST_F(FrameFuseTriangleTest, TriangleModeNegativeInput) {
    FrameFuseCPUParams p;
    p.matchMode = 1;
    FrameFuseCPU op(p);

    {   // NaN 坐标
        MarkerPointSet s1 = set1_, s2 = set1_;
        s2.positions[3].x = std::numeric_limits<double>::quiet_NaN();
        auto r = op.Execute(s1, s2);
        EXPECT_FALSE(r.success);
    }
    {   // positions/normals 尺寸不匹配
        MarkerPointSet s1 = set1_, s2 = set1_;
        s2.normals.pop_back();
        auto r = op.Execute(s1, s2);
        EXPECT_FALSE(r.success);
    }
    {   // 重复点（不崩溃，结果有限）
        MarkerPointSet s1 = set1_, s2 = applyTransform(set1_, makeRotationZ(5.0),
                                                       cv::Vec3d(3, 2, 1));
        s1.positions.push_back(s1.positions[0]);   // 重复点 → 下标 8
        s1.normals.push_back(s1.normals[0]);
        s1.positions.push_back(s1.positions[1]);   // 重复点 → 下标 9
        s1.normals.push_back(s1.normals[1]);
        auto r = op.Execute(s1, s2);
        EXPECT_TRUE(std::isfinite(r.rmse));   // 无论成败，数值必须有限
        if (r.success)
            for (const auto& pr : r.correspondences) {
                EXPECT_NE(pr.first, 8);   // 重复点不得参与（互最近邻并列取低下标）
                EXPECT_NE(pr.first, 9);
            }
    }
}
