// ============================================================================
// test_laser_cloud_fuse_remove.cpp — removePoints 编辑账本单测（实施计划 P2）
// 覆盖：子集移除/计数一致、越界整批原子、空幂等、重复去重、全删清空、
// 移除体素重建、再融合续积、DeviceContext 计数同步
// ============================================================================
#include "laser_cloud_fuse_cuda.h"

using namespace calib;

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

namespace {
LaserCloudFuseCUDAParams rmParams() {
    LaserCloudFuseCUDAParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    p.reserveVoxelCount = 1024;
    return p;
}
cv::cuda::GpuMat toGpu(const std::vector<cv::Point3f>& pts) {
    cv::Mat h(1, static_cast<int>(pts.size()), CV_32FC3,
              const_cast<cv::Point3f*>(pts.data()));
    return cv::cuda::GpuMat(h);
}
} // namespace

// —— 子集移除：计数与 DeviceContext 同步 ——
TEST(LaserCloudFuseRemove, SubsetRemoval) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}, {20.5f, 0, 0}})).success);
    ASSERT_EQ(fuse.GetFusedPointCount(), 3u);

    auto r = fuse.removePoints({1});
    ASSERT_TRUE(r.success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 2u);
    EXPECT_EQ(fuse.GetVoxelCount(), 2u);
    EXPECT_EQ(fuse.GetDeviceContext().fusedPointCount, 2u);   // 设备侧同步
}

// —— 越界整批原子（状态不动）——
TEST(LaserCloudFuseRemove, OutOfRangeAtomicFail) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}})).success);
    EXPECT_FALSE(fuse.removePoints({0, 5}).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 2u);
    EXPECT_EQ(fuse.GetVoxelCount(), 2u);
}

// —— 空幂等 / 重复去重 ——
TEST(LaserCloudFuseRemove, EmptyAndDuplicate) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}, {20.5f, 0, 0}})).success);
    EXPECT_TRUE(fuse.removePoints({}).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 3u);
    EXPECT_TRUE(fuse.removePoints({2, 2}).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 2u);
}

// —— 全删＝清空，之后可继续融合 ——
TEST(LaserCloudFuseRemove, RemoveAllClears) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}})).success);
    ASSERT_TRUE(fuse.removePoints({0, 1}).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 0u);
    EXPECT_EQ(fuse.GetVoxelCount(), 0u);
    EXPECT_TRUE(fuse.Execute(toGpu({{30.5f, 0, 0}})).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
}

// —— 移除体素重建：同位点再融合重新建格 ——
TEST(LaserCloudFuseRemove, RemovedVoxelRecreatedOnRefuse) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}})).success);
    ASSERT_TRUE(fuse.removePoints({0}).success);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);

    ASSERT_TRUE(fuse.Execute(toGpu({{1.6f, 0, 0}})).success);   // 同体素新点
    EXPECT_EQ(fuse.GetFusedPointCount(), 2u);
    EXPECT_EQ(fuse.GetVoxelCount(), 2u);
}

// —— 移除后继续融合：新点正常入账（哈希槽未被破坏）——
TEST(LaserCloudFuseRemove, ContinueFuseAfterRemoval) {
    LaserCloudFuseCuda fuse(rmParams());
    ASSERT_TRUE(fuse.Execute(toGpu({{1.5f, 0, 0}, {10.5f, 0, 0}, {20.5f, 0, 0}})).success);
    ASSERT_TRUE(fuse.removePoints({0}).success);                 // 摘 (1.5,0,0)

    auto r = fuse.Execute(toGpu({{30.5f, 0, 0}, {1.5f, 0, 0}})); // 新点＋被删位点
    ASSERT_TRUE(r.success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 4u);                    // 2 幸存＋2 新格
    EXPECT_EQ(fuse.GetVoxelCount(), 4u);
}
