#include "laser_cloud_fuse_cuda.h"

using namespace calib;

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <cmath>

using namespace calib;

// ============================================================
// 鍩烘湰鍔熻兘娴嬭瘯
// ============================================================

TEST(LaserCloudFuseCuda, SinglePoint_NewVoxel) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    std::vector<cv::Point3f> pts = {{1.5f, 2.5f, 3.5f}};
    cv::Mat h_mat(1, 1, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat);

    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.inputCount, 1);
    EXPECT_EQ(r.newVoxelCount, 1);
    EXPECT_EQ(r.survivingCount, 1);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);
}

TEST(LaserCloudFuseCuda, DuplicatePoints_SameVoxel) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    // 3 涓偣钀藉叆鍚屼竴浣撶礌
    std::vector<cv::Point3f> pts = {{0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f}, {0.3f, 0.3f, 0.3f}};
    cv::Mat h_mat(1, 3, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat);

    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.newVoxelCount, 1);   // 鍚屼竴浣撶礌
    EXPECT_EQ(r.survivingCount, 3);  // threshold=5, 全保留
    EXPECT_EQ(r.deletedCount, 0);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
}

TEST(LaserCloudFuseCuda, SaturationThreshold) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 2;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    // 4 个点落入同一体素，threshold=2 → 前2个保留，后2个丢弃
    std::vector<cv::Point3f> pts = {
        {0.1f, 0.1f, 0.1f}, {0.2f, 0.2f, 0.2f},
        {0.3f, 0.3f, 0.3f}, {0.4f, 0.4f, 0.4f}
    };
    cv::Mat h_mat(1, 4, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat);

    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.survivingCount, 2);
    EXPECT_EQ(r.deletedCount, 2);
}

TEST(LaserCloudFuseCuda, Grid_3x3) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    std::vector<cv::Point3f> pts;
    for (float x = 0.5f; x < 3.0f; x += 1.0f)
        for (float y = 0.5f; y < 3.0f; y += 1.0f)
            pts.emplace_back(x, y, 0.5f);

    cv::Mat h_mat(1, static_cast<int>(pts.size()), CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat);

    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.newVoxelCount, 9);
    EXPECT_EQ(fuse.GetFusedPointCount(), 9u);
}

TEST(LaserCloudFuseCuda, CrossFrame_Accumulation) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    // 绗竴甯э細3 涓偣
    std::vector<cv::Point3f> f1 = {{0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}, {2.5f, 2.5f, 2.5f}};
    cv::Mat h1(1, 3, CV_32FC3, f1.data());
    cv::cuda::GpuMat d1(h1);
    auto r1 = fuse.Execute(d1);
    EXPECT_EQ(fuse.GetFusedPointCount(), 3u);

    // 第二帧：2 个新点 + 1 个重复
    std::vector<cv::Point3f> f2 = {{0.5f, 0.5f, 0.5f}, {3.5f, 3.5f, 3.5f}, {4.5f, 4.5f, 4.5f}};
    cv::Mat h2(1, 3, CV_32FC3, f2.data());
    cv::cuda::GpuMat d2(h2);
    auto r2 = fuse.Execute(d2);
    EXPECT_EQ(r2.newVoxelCount, 2); // 3.5, 4.5 鏄柊浣撶礌
    EXPECT_EQ(fuse.GetFusedPointCount(), 5u);
}

TEST(LaserCloudFuseCuda, RT_Transform) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    // 杈撳叆 (0,0,0)锛孯=I, T=(10,20,30) 鈫?鍏ㄥ眬鍧愭爣 (10,20,30)
    std::vector<cv::Point3f> pts = {{0.0f, 0.0f, 0.0f}};
    cv::Mat h_mat(1, 1, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat, cv::Matx33d::eye(), cv::Vec3d(10, 20, 30));

    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.newVoxelCount, 1);

    // 楠岃瘉锛氱敤鐩稿悓 R/T 鍐嶈緭鍏?(0,0,0) 搴旇鍛戒腑宸叉湁浣撶礌
    auto r2 = fuse.Execute(d_mat, cv::Matx33d::eye(), cv::Vec3d(10, 20, 30));
    EXPECT_EQ(r2.newVoxelCount, 0); // 已存在
    EXPECT_EQ(r2.survivingCount, 1);
}

TEST(LaserCloudFuseCuda, Clear_Resets) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    std::vector<cv::Point3f> pts = {{1.5f, 2.5f, 3.5f}};
    cv::Mat h_mat(1, 1, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);

    auto r = fuse.Execute(d_mat);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);

    fuse.Clear();
    EXPECT_EQ(fuse.GetFusedPointCount(), 0u);

    // 娓呯┖鍚庡啀 fuse 搴旇鏄柊浣撶礌
    auto r2 = fuse.Execute(d_mat);
    EXPECT_EQ(r2.newVoxelCount, 1);
}

TEST(LaserCloudFuseCuda, DeviceContext_Valid) {
    LaserCloudFuseCUDAParams p; p.voxelSize = 0.5f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    LaserCloudFuseCuda fuse(p);

    std::vector<cv::Point3f> pts = {{1.0f, 2.0f, 3.0f}};
    cv::Mat h_mat(1, 1, CV_32FC3, pts.data());
    cv::cuda::GpuMat d_mat(h_mat);
    auto r = fuse.Execute(d_mat);

    auto ctx = fuse.GetDeviceContext();
    EXPECT_NE(ctx.d_keys, nullptr);
    EXPECT_NE(ctx.d_fusedIdx, nullptr);
    EXPECT_NE(ctx.d_fusedXyz, nullptr);
    EXPECT_NE(ctx.d_fusedNormal, nullptr);
    EXPECT_EQ(ctx.mask, 1023u); // 1024 slots 鈫?mask = 0x3FF
    EXPECT_FLOAT_EQ(ctx.voxelSize, 0.5f);
    EXPECT_EQ(ctx.fusedPointCount, 1u);
}

TEST(LaserCloudFuseCuda, EmptyFrame_Fails) {
    LaserCloudFuseCuda fuse;
    cv::cuda::GpuMat empty;
    auto r = fuse.Execute(empty);
    EXPECT_FALSE(r.success);
}
