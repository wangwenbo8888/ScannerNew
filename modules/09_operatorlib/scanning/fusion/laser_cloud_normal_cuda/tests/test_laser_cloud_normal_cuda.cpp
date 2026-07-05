#include "laser_cloud_normal_cuda.h"
#include "laser_cloud_fuse_cuda.h"

using namespace calib;

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace calib;

// Helper: upload points to GpuMat
static cv::cuda::GpuMat makeGpuMat(const std::vector<cv::Point3f>& pts) {
    cv::Mat h(1, static_cast<int>(pts.size()), CV_32FC3, const_cast<cv::Point3f*>(pts.data()));
    cv::cuda::GpuMat d;
    d.upload(h);
    return d;
}

// Helper: download fused point data
static std::vector<float> downloadFusedXyz(const LaserCloudFuseDeviceContext& ctx, size_t count) {
    std::vector<float> h(count * 3);
    cudaMemcpy(h.data(), ctx.d_fusedXyz, count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static std::vector<float> downloadFusedNormal(const LaserCloudFuseDeviceContext& ctx, size_t count) {
    std::vector<float> h(count * 3);
    cudaMemcpy(h.data(), ctx.d_fusedNormal, count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// ============================================================
// 骞抽潰娉曠嚎娴嬭瘯
// ============================================================

TEST(LaserCloudNormalCuda, PlaneNormal_ZAxis) {
    LaserCloudFuseCUDAParams fp; fp.voxelSize = 1.0f; fp.saturationThreshold = 5;
    fp.reserveVoxelCount = 4096;
    LaserCloudFuseCuda fuse(fp);

    std::vector<cv::Point3f> plane;
    for (float x = 0.5f; x < 8.0f; x += 1.0f)
        for (float y = 0.5f; y < 8.0f; y += 1.0f)
            plane.emplace_back(x, y, 0.0f);

    auto d_mat = makeGpuMat(plane);
    auto fr = fuse.Execute(d_mat);
    ASSERT_GT(fuse.GetFusedPointCount(), 30u);

    LaserCloudNormalCuda normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);
    EXPECT_GT(nr.processedCount, 0u);

    auto ctx = fuse.GetDeviceContext();
    auto normals = downloadFusedNormal(ctx, ctx.fusedPointCount);
    auto xyz = downloadFusedXyz(ctx, ctx.fusedPointCount);

    // Check center area normals 鈮?(0,0,卤1)
    int checked = 0;
    for (size_t i = 0; i < ctx.fusedPointCount; i++) {
        if (xyz[i*3] < 2.5f || xyz[i*3] > 5.5f) continue;
        if (xyz[i*3+1] < 2.5f || xyz[i*3+1] > 5.5f) continue;
        float angleFromZ = std::acos(std::min(1.0f, std::fabs(normals[i*3+2])));
        float angleDeg = angleFromZ * 180.0f / CV_PI;
        EXPECT_LT(angleDeg, 15.0f);
        ++checked;
    }
    EXPECT_GE(checked, 4);
}

TEST(LaserCloudNormalCuda, SingleVoxel_Fallback) {
    LaserCloudFuseCUDAParams fp; fp.voxelSize = 1.0f;
    fp.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(fp);

    auto d_mat = makeGpuMat({{5.0f, 5.0f, 5.0f}});
    auto fr = fuse.Execute(d_mat);

    LaserCloudNormalCuda normalOp;
    auto nr = normalOp.Execute(fuse, fr);

    EXPECT_TRUE(nr.success);
    EXPECT_EQ(nr.fallbackCount, 1u);

    auto ctx = fuse.GetDeviceContext();
    auto normals = downloadFusedNormal(ctx, 1);
    EXPECT_FLOAT_EQ(normals[0], 0.0f);
    EXPECT_FLOAT_EQ(normals[1], 0.0f);
    EXPECT_FLOAT_EQ(normals[2], 1.0f);
}

TEST(LaserCloudNormalCuda, EmptyRange_NoOp) {
    LaserCloudFuseCuda fuse;
    LaserCloudNormalCuda normalOp;
    auto nr = normalOp.Execute(fuse, LaserCloudFuseCudaResult{});

    EXPECT_TRUE(nr.success);
    EXPECT_EQ(nr.processedCount, 0u);
    EXPECT_EQ(nr.qualityFlag, calib::QualityFlag::Warning);
}

TEST(LaserCloudNormalCuda, CrossFrame_NormalOnlyForNewVoxels) {
    LaserCloudFuseCUDAParams fp; fp.voxelSize = 1.0f;
    fp.reserveVoxelCount = 4096;
    LaserCloudFuseCuda fuse(fp);

    // Frame 1: 3脳3 grid
    std::vector<cv::Point3f> f1;
    for (float x = 0.5f; x < 3.0f; x += 1.0f)
        for (float y = 0.5f; y < 3.0f; y += 1.0f)
            f1.emplace_back(x, y, 0.0f);
    auto d1 = makeGpuMat(f1);
    auto fr1 = fuse.Execute(d1);

    LaserCloudNormalCuda normalOp;
    auto nr1 = normalOp.Execute(fuse, fr1);
    EXPECT_GT(nr1.processedCount, 0u);

    // Frame 2: add more points
    std::vector<cv::Point3f> f2;
    for (float x = 3.5f; x < 6.0f; x += 1.0f)
        for (float y = 0.5f; y < 3.0f; y += 1.0f)
            f2.emplace_back(x, y, 0.0f);
    auto d2 = makeGpuMat(f2);
    auto fr2 = fuse.Execute(d2);

    auto nr2 = normalOp.Execute(fuse, fr2);
    // Only new voxels from frame 2 should be processed
    EXPECT_EQ(nr2.processedCount + nr2.fallbackCount, fr2.newVoxelCount);
}
