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
    std::normal_distribution<double> noise(0.0, posNoiseStd);
    for (auto& p : set.positions) {
        p.x += noise(rng);
        p.y += noise(rng);
        p.z += noise(rng);
    }
    std::normal_distribution<double> nNoise(0.0, normalNoiseStd);
    for (auto& n : set.normals) {
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
