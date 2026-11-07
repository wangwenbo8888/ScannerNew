// ============================================================================
// test_marker_cloud_fuse_remove.cpp — removePoints 编辑账本单测（实施计划 P2）
// 覆盖：子集移除/下标前移、越界整批原子、空幂等、重复去重、同位点重建、
// 幸存体素计数保留（饱和阈值再融合不重复触发）
// ============================================================================
#include <gtest/gtest.h>

#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"

#include <cmath>

using calib::MarkerCloudFuseCPU;
using calib::MarkerCloudFuseCPUParams;
using calib::MarkerCloudPoint;
using calib::MarkerFuseInput;

namespace {
MarkerFuseInput pt(float x, float y, float z) { return MarkerFuseInput{x, y, z}; }
MarkerCloudPoint cp(float x, float y, float z) { return MarkerCloudPoint{x, y, z}; }
bool nearPt(const MarkerCloudPoint& a, const MarkerCloudPoint& b) {
    return std::fabs(a.x - b.x) < 1e-5f && std::fabs(a.y - b.y) < 1e-5f &&
           std::fabs(a.z - b.z) < 1e-5f;
}
} // namespace

// —— 子集移除＋下标前移 ——
TEST(RemovePoints, SubsetRemovalCompacts) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.Execute({pt(0, 0, 0), pt(10, 0, 0), pt(20, 0, 0)}).success);
    ASSERT_EQ(f.GetFusedPointCount(), 3u);

    auto r = f.removePoints({1});                       // 移除中点
    ASSERT_TRUE(r.success);
    EXPECT_EQ(f.GetFusedPointCount(), 2u);
    EXPECT_EQ(f.GetVoxelCount(), 2u);
    const auto& pts = f.GetFusedPoints();
    ASSERT_EQ(pts.size(), 2u);
    EXPECT_TRUE(nearPt(pts[0], cp(0, 0, 0)));           // 幸存点下标前移
    EXPECT_TRUE(nearPt(pts[1], cp(20, 0, 0)));
}

// —— 越界整批原子（状态不动）——
TEST(RemovePoints, OutOfRangeAtomicFail) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.Execute({pt(0, 0, 0), pt(10, 0, 0)}).success);
    auto r = f.removePoints({0, 5});                    // 5 越界
    EXPECT_FALSE(r.success);
    EXPECT_EQ(f.GetFusedPointCount(), 2u);              // 整批不动
    EXPECT_EQ(f.GetVoxelCount(), 2u);
    EXPECT_TRUE(nearPt(f.GetFusedPoints()[0], cp(0, 0, 0)));
}

// —— 空幂等 / 重复去重 ——
TEST(RemovePoints, EmptyAndDuplicate) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.Execute({pt(0, 0, 0), pt(10, 0, 0), pt(20, 0, 0)}).success);
    EXPECT_TRUE(f.removePoints({}).success);
    EXPECT_EQ(f.GetFusedPointCount(), 3u);
    EXPECT_TRUE(f.removePoints({2, 2}).success);        // 重复算一次
    EXPECT_EQ(f.GetFusedPointCount(), 2u);
}

// —— 移除后同位点再融合＝重建（新体素、代表点可再落）——
TEST(RemovePoints, RemovedVoxelRecreatedOnRefuse) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.Execute({pt(0, 0, 0), pt(10, 0, 0)}).success);
    ASSERT_TRUE(f.removePoints({0}).success);           // 摘 (0,0,0) 体素
    EXPECT_EQ(f.GetVoxelCount(), 1u);

    auto r = f.Execute({pt(0.05f, 0, 0)});              // 同体素新点（voxel=0.5 默认）
    ASSERT_TRUE(r.success);
    EXPECT_EQ(f.GetFusedPointCount(), 2u);
    EXPECT_EQ(f.GetVoxelCount(), 2u);                   // 体素重建
    EXPECT_TRUE(nearPt(f.GetFusedPoints()[1], cp(0.05f, 0, 0)));
}

// —— 全清空后状态一致 ——
TEST(RemovePoints, RemoveAll) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.Execute({pt(0, 0, 0), pt(10, 0, 0)}).success);
    ASSERT_TRUE(f.removePoints({0, 1}).success);
    EXPECT_EQ(f.GetFusedPointCount(), 0u);
    EXPECT_EQ(f.GetVoxelCount(), 0u);
    EXPECT_TRUE(f.Execute({pt(30, 0, 0)}).success);     // 清后可继续融合
    EXPECT_EQ(f.GetFusedPointCount(), 1u);
}

// —— seed 点同样可移除（编辑会话对续扫基准生效）——
TEST(RemovePoints, SeedPointsRemovable) {
    MarkerCloudFuseCPU f;
    ASSERT_TRUE(f.seed({cp(0, 0, 0), cp(10, 0, 0), cp(20, 0, 0)}).success);
    ASSERT_TRUE(f.removePoints({0, 2}).success);
    EXPECT_EQ(f.GetFusedPointCount(), 1u);
    EXPECT_TRUE(nearPt(f.GetFusedPoints()[0], cp(10, 0, 0)));
}
