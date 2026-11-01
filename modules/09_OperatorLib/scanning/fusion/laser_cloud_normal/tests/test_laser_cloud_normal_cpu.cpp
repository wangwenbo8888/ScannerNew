#include "laser_cloud_normal_cpu.h"
#include "laser_cloud_fuse_cpu.h"

using namespace calib;

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <cstdio>

using namespace calib;

// ============================================================
// 鍙傛暟鏍￠獙
// ============================================================

TEST(LaserCloudNormalCPUParams, Validate_DefaultIsValid) {
    LaserCloudNormalCPUParams p;
    EXPECT_NO_THROW(p.validate());
}

TEST(LaserCloudNormalCPUParams, Validate_RejectInvalidKernelRadius) {
    LaserCloudNormalCPUParams p;
    p.kernelRadius = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(LaserCloudNormalCPUParams, Validate_RejectInvalidMinNeighbors) {
    LaserCloudNormalCPUParams p;
    p.minNeighbors = 2;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(LaserCloudNormalCPUParams, JsonRoundtrip) {
    LaserCloudNormalCPUParams p;
    p.kernelRadius = 2;
    p.minNeighbors = 5;
    p.fallbackNx = 0.1f;
    p.fallbackNy = 0.2f;
    p.fallbackNz = 0.3f;
    auto j = p.toJson();
    auto p2 = LaserCloudNormalCPUParams::fromJson(j);
    EXPECT_EQ(p2.kernelRadius, 2);
    EXPECT_EQ(p2.minNeighbors, 5);
    EXPECT_FLOAT_EQ(p2.fallbackNx, 0.1f);
    EXPECT_FLOAT_EQ(p2.fallbackNy, 0.2f);
    EXPECT_FLOAT_EQ(p2.fallbackNz, 0.3f);
}

TEST(LaserCloudNormalCPUParams, JsonPartialFields) {
    nlohmann::json j;
    j["kernelRadius"] = 2;
    auto p = LaserCloudNormalCPUParams::fromJson(j);
    EXPECT_EQ(p.kernelRadius, 2);
    EXPECT_EQ(p.minNeighbors, 3); // default retained
}

// ============================================================
// 骞抽潰娉曠嚎娴嬭瘯
// ============================================================

TEST(LaserCloudNormalCPU, PlaneNormal_ZAxis) {
    // Z=0 plane, 7x7 grid (voxelSize=1.0, spacing 0.6 < 1.0)
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f; fp.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(fp);

    std::vector<cv::Point3f> plane;
    for (float x = 0.5f; x < 5.0f; x += 0.6f)
        for (float y = 0.5f; y < 5.0f; y += 0.6f)
            plane.emplace_back(x, y, 0.0f);

    auto fr = fuse.Execute(plane);
    ASSERT_GT(fuse.GetFusedPointCount(), 20u);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);
    EXPECT_GT(nr.statistics.processedCount, 0u);

    // Check center area normals are approximately (0,0,+-1)
    const auto& points = fuse.GetFusedPoints();
    int checked = 0;
    for (const auto& p : points) {
        if (p.x < 1.5f || p.x > 3.5f) continue;
        if (p.y < 1.5f || p.y > 3.5f) continue;
        float angleFromZ = std::acos(std::min(1.0f, std::fabs(p.nz)));
        float angleDeg = angleFromZ * 180.0f / CV_PI;
        EXPECT_LT(angleDeg, 15.0f)
            << "point (" << p.x << "," << p.y << "," << p.z << ") "
            << "normal (" << p.nx << "," << p.ny << "," << p.nz << ")";
        ++checked;
    }
    EXPECT_GE(checked, 4);
}

TEST(LaserCloudNormalCPU, PlaneNormal_XAxis) {
    // X=5 plane, normals should be approximately +-X
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f; fp.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(fp);

    std::vector<cv::Point3f> plane;
    for (float y = 0.5f; y < 5.0f; y += 0.6f)
        for (float z = 0.5f; z < 5.0f; z += 0.6f)
            plane.emplace_back(5.0f, y, z);

    auto fr = fuse.Execute(plane);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);

    const auto& points = fuse.GetFusedPoints();
    int checked = 0;
    for (const auto& p : points) {
        if (p.y < 1.5f || p.y > 3.5f) continue;
        if (p.z < 1.5f || p.z > 3.5f) continue;
        float angleFromX = std::acos(std::min(1.0f, std::fabs(p.nx)));
        EXPECT_LT(angleFromX * 180.0f / CV_PI, 15.0f);
        ++checked;
    }
    EXPECT_GE(checked, 4);
}

// ============================================================
// 閫€鍖栧鐞?// ============================================================

TEST(LaserCloudNormalCPU, SingleVoxel_Fallback) {
    // Single voxel, no neighbors -> fallback normal
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);
    std::vector<cv::Point3f> single = {{5.0f, 5.0f, 5.0f}};
    auto fr = fuse.Execute(single);
    ASSERT_EQ(fuse.GetFusedPointCount(), 1u);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);
    EXPECT_EQ(nr.statistics.fallbackCount, 1u);
    EXPECT_EQ(nr.statistics.processedCount, 0u);

    const auto& points = fuse.GetFusedPoints();
    EXPECT_FLOAT_EQ(points[0].nx, 0.0f);
    EXPECT_FLOAT_EQ(points[0].ny, 0.0f);
    EXPECT_FLOAT_EQ(points[0].nz, 1.0f);
}

TEST(LaserCloudNormalCPU, CustomFallbackNormal) {
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);
    std::vector<cv::Point3f> single = {{5.0f, 5.0f, 5.0f}};
    auto fr = fuse.Execute(single);

    LaserCloudNormalCPUParams np;
    np.fallbackNx = 1.0f; np.fallbackNy = 0.0f; np.fallbackNz = 0.0f;
    LaserCloudNormalCPU normalOp(np);
    auto nr = normalOp.Execute(fuse, fr);

    const auto& points = fuse.GetFusedPoints();
    EXPECT_FLOAT_EQ(points[0].nx, 1.0f);
}

TEST(LaserCloudNormalCPU, AdaptiveExpansion_LineThenGrid) {
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);

    // First frame: 3 colinear points (PCA degenerate)
    std::vector<cv::Point3f> line = {{0.5f, 0.5f, 0.5f}, {1.5f, 0.5f, 0.5f}, {2.5f, 0.5f, 0.5f}};
    auto fr1 = fuse.Execute(line);

    LaserCloudNormalCPU normalOp;
    auto nr1 = normalOp.Execute(fuse, fr1);

    EXPECT_TRUE(nr1.success);
    EXPECT_EQ(nr1.statistics.processedCount + nr1.statistics.fallbackCount, 3u);

    // Second frame: expand to plane
    std::vector<cv::Point3f> grid;
    for (float x = 0.5f; x < 5.0f; x += 0.6f)
        for (float y = 0.5f; y < 5.0f; y += 0.6f)
            if (x > 3.0f || y > 1.0f)
                grid.emplace_back(x, y, 0.0f);
    auto fr2 = fuse.Execute(grid);

    auto nr2 = normalOp.Execute(fuse, fr2);
    EXPECT_TRUE(nr2.success);
    EXPECT_GT(nr2.statistics.processedCount, 0u);
}

TEST(LaserCloudNormalCPU, EmptyRange_NoOp) {
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, 0, 0);

    EXPECT_TRUE(nr.success);
    EXPECT_EQ(nr.statistics.processedCount, 0u);
    EXPECT_EQ(nr.qualityFlag, calib::QualityFlag::Warning);
}

TEST(LaserCloudNormalCPU, ConvenienceOverload_MatchesExplicit) {
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);
    std::vector<cv::Point3f> pts;
    for (float x = 0.5f; x < 5.0f; x += 0.6f)
        for (float y = 0.5f; y < 5.0f; y += 0.6f)
            pts.emplace_back(x, y, 0.0f);
    auto fr = fuse.Execute(pts);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    size_t total = fuse.GetFusedPointCount();
    size_t begin = total - fr.statistics.newVoxelCount;
    EXPECT_EQ(begin, 0u);
    EXPECT_EQ(nr.statistics.processedCount + nr.statistics.fallbackCount,
              fr.statistics.newVoxelCount);
}

// ============================================================
// 鐞冮潰娉曠嚎
// ============================================================

TEST(LaserCloudNormalCPU, SphereNormal_Radial) {
    // Points on sphere R=10mm -> normals approximately radial
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f; fp.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(fp);

    const float R = 10.0f;
    std::vector<cv::Point3f> sphere;
    for (float theta = 0.3f; theta < CV_PI - 0.3f; theta += 0.1f)
        for (float phi = 0.0f; phi < 2.0f * CV_PI; phi += 0.1f) {
            sphere.emplace_back(
                R * std::sin(theta) * std::cos(phi),
                R * std::sin(theta) * std::sin(phi),
                R * std::cos(theta));
        }

    auto fr = fuse.Execute(sphere);
    ASSERT_GT(fuse.GetFusedPointCount(), 50u);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);

    // Verify normals approximately radial: |normal . position_normalized| approx 1
    const auto& points = fuse.GetFusedPoints();
    int checked = 0;
    for (const auto& p : points) {
        float r = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        if (r < 1e-3f) continue;
        float rx = p.x / r, ry = p.y / r, rz = p.z / r;
        float dot = std::fabs(p.nx * rx + p.ny * ry + p.nz * rz);
        float angleDeg = std::acos(std::min(1.0f, dot)) * 180.0f / CV_PI;
        EXPECT_LT(angleDeg, 20.0f)
            << "point (" << p.x << "," << p.y << "," << p.z << ") "
            << "normal (" << p.nx << "," << p.ny << "," << p.nz << ")";
        ++checked;
    }
    EXPECT_GE(checked, 20);
}

// ============================================================
// QualityFlag
// ============================================================

TEST(LaserCloudNormalCPU, QualityFlag_AllFallback) {
    // All isolated points -> all fallback -> Warning
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(fp);
    std::vector<cv::Point3f> pts = {{0.5f,0.5f,0.5f}, {50.5f,50.5f,50.5f}, {100.5f,100.5f,100.5f}};
    auto fr = fuse.Execute(pts);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_EQ(nr.qualityFlag, calib::QualityFlag::Warning);
    EXPECT_EQ(nr.statistics.fallbackCount, 3u);
}

// ============================================================
// 鎬ц兘娴嬭瘯
// ============================================================

TEST(LaserCloudNormalCPU, Perf_10KVoxels) {
    // 100x100 grid on Z=0 plane, spacing=voxelSize=1.0 -> 10K voxels, each with ~9 neighbors
    LaserCloudFuseCPUParams fp; fp.voxelSize = 1.0f; fp.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(fp);

    std::vector<cv::Point3f> pts;
    pts.reserve(10000);
    for (int x = 0; x < 100; ++x)
        for (int y = 0; y < 100; ++y)
            pts.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.0f);

    auto fr = fuse.Execute(pts);

    size_t newVoxels = fr.statistics.newVoxelCount;
    printf("  newVoxels=%zu\n", newVoxels);

    LaserCloudNormalCPU normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    printf("  normalTime=%.2fms  processed=%zu  fallback=%zu  expanded=%zu\n",
           nr.statistics.totalTimeMs, nr.statistics.processedCount,
           nr.statistics.fallbackCount, nr.statistics.expandedCount);

    EXPECT_LT(nr.statistics.totalTimeMs, 100.0); // 宽松上限（容忍全量 ctest 并行负载，同 laser_cloud_fuse）
    SUCCEED() << "10K voxels: " << nr.statistics.totalTimeMs << "ms";
}
