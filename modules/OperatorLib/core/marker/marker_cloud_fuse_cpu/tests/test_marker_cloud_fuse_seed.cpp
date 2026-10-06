/**
 * @file test_marker_cloud_fuse_seed.cpp
 * @brief marker_cloud_fuse_cpu::seed() 单元测试（existingMarkers 预填 / 先入代表语义）
 */

#include "marker_cloud_fuse_cpu.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <vector>

using namespace calib;

namespace {

MarkerCloudPoint mkSeed(float x, float y, float z, float radius = 2.0f) {
    MarkerCloudPoint p;
    p.x = x; p.y = y; p.z = z;
    p.nx = 0.0f; p.ny = 0.0f; p.nz = 1.0f;
    p.whiteRadius = radius;
    return p;
}

MarkerFuseInput mkInput(float x, float y, float z, float radius = 1.0f) {
    MarkerFuseInput in;
    in.x = x; in.y = y; in.z = z;
    in.nx = 0.0f; in.ny = 0.0f; in.nz = 1.0f;
    in.whiteRadius = radius;
    return in;
}

} // namespace

// ============================================================
// 1. seed 先于 fuse：seed 代表点不被后续扫描帧移动
// ============================================================

TEST(MarkerCloudFuseCPU, SeedThenFuseKeepsSeed) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    p.saturationThreshold = 3;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerCloudPoint> seeds = {
        mkSeed(0.5f,  0.5f, 0.5f),
        mkSeed(10.5f, 0.5f, 0.5f),
        mkSeed(20.5f, 0.5f, 0.5f),
    };
    ASSERT_TRUE(fuse.seed(seeds).success);

    // 多帧融合：seed 体素内近距离点 + 精确命中点 + 一个新体素点
    for (int f = 0; f < 5; ++f) {
        std::vector<MarkerFuseInput> frame = {
            mkInput(0.5f + 0.01f * f, 0.6f, 0.55f),   // 落入 seed[0] 体素
            mkInput(10.5f, 0.5f, 0.5f),               // 精确命中 seed[1] 体素
            mkInput(30.5f, 0.5f, 0.5f),               // 新体素
        };
        auto r = fuse.Execute(frame);
        ASSERT_TRUE(r.success) << r.message;
    }

    const auto& fused = fuse.GetFusedPoints();
    ASSERT_EQ(fused.size(), 4u);   // 3 seed 体素 + 1 新体素
    EXPECT_EQ(fuse.GetVoxelCount(), 4u);

    // seed 代表点位置/半径原样保留（首个落入者为代表，fuse 不移动已占体素代表）
    EXPECT_FLOAT_EQ(fused[0].x, 0.5f);
    EXPECT_FLOAT_EQ(fused[0].y, 0.5f);
    EXPECT_FLOAT_EQ(fused[0].z, 0.5f);
    EXPECT_FLOAT_EQ(fused[0].whiteRadius, 2.0f);
    EXPECT_FLOAT_EQ(fused[0].nz, 1.0f);
    EXPECT_FLOAT_EQ(fused[1].x, 10.5f);
    EXPECT_FLOAT_EQ(fused[1].whiteRadius, 2.0f);
    EXPECT_FLOAT_EQ(fused[2].x, 20.5f);
    EXPECT_FLOAT_EQ(fused[2].whiteRadius, 2.0f);
    // 新体素代表来自扫描帧
    EXPECT_FLOAT_EQ(fused[3].x, 30.5f);
}

// ============================================================
// 2. seed 立即可见（GetFusedPoints 立刻含 seed 点）；重复调用=追加
// ============================================================

TEST(MarkerCloudFuseCPU, SeedVisibleImmediately) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerCloudPoint> seeds = {
        mkSeed(0.5f, 0.5f, 0.5f),
        mkSeed(10.5f, 0.5f, 0.5f),
        mkSeed(20.5f, 0.5f, 0.5f),
    };
    ASSERT_TRUE(fuse.seed(seeds).success);

    EXPECT_EQ(fuse.GetFusedPointCount(), 3u);
    EXPECT_EQ(fuse.GetVoxelCount(), 3u);
    const auto& fused = fuse.GetFusedPoints();
    ASSERT_EQ(fused.size(), 3u);
    EXPECT_FLOAT_EQ(fused[0].x, 0.5f);
    EXPECT_FLOAT_EQ(fused[1].x, 10.5f);
    EXPECT_FLOAT_EQ(fused[2].x, 20.5f);

    // 重复调用 = 追加（不去重，调用方负责）
    std::vector<MarkerCloudPoint> more = {
        mkSeed(100.5f, 0.5f, 0.5f),
        mkSeed(200.5f, 0.5f, 0.5f),
    };
    ASSERT_TRUE(fuse.seed(more).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 5u);
    EXPECT_EQ(fuse.GetVoxelCount(), 5u);
}

// ============================================================
// 3. seed 内部两点同体素：取首个（后到者丢弃，代表不动）
// ============================================================

TEST(MarkerCloudFuseCPU, SeedSameVoxelFirstWins) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerCloudPoint> seeds = {
        mkSeed(0.5f, 0.5f, 0.5f, 1.0f),
        mkSeed(0.6f, 0.6f, 0.6f, 9.0f),   // 同体素 (0,0,0)
    };
    ASSERT_TRUE(fuse.seed(seeds).success);

    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
    const auto& fused = fuse.GetFusedPoints();
    ASSERT_EQ(fused.size(), 1u);
    EXPECT_FLOAT_EQ(fused[0].whiteRadius, 1.0f);   // 首个保留
    EXPECT_FLOAT_EQ(fused[0].x, 0.5f);
}

// ============================================================
// 3b. 越界 fail 全有或全无：任一点越界 → 整批拒绝，零写入
// ============================================================

TEST(MarkerCloudFuseCPU, SeedOutOfRangeAtomic) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerCloudPoint> ok = {
        mkSeed(0.5f, 0.5f, 0.5f),
        mkSeed(10.5f, 0.5f, 0.5f),
    };
    ASSERT_TRUE(fuse.seed(ok).success);
    ASSERT_EQ(fuse.GetFusedPointCount(), 2u);

    // 3 正常 + 1 越界（1e9 远超 21-bit 体素轴范围）→ 全有或全无
    std::vector<MarkerCloudPoint> batch = {
        mkSeed(20.5f, 0.5f, 0.5f),
        mkSeed(30.5f, 0.5f, 0.5f),
        mkSeed(40.5f, 0.5f, 0.5f),
        mkSeed(1e9f, 0.5f, 0.5f),
    };
    auto r = fuse.seed(batch);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 2u);   // 零写入：前 3 个正常点也未入表
    EXPECT_EQ(fuse.GetVoxelCount(), 2u);
}

// ============================================================
// 4. Clear 后 fusedPoints 为空；可重新 seed
// ============================================================

TEST(MarkerCloudFuseCPU, ClearThenSeedEmpty) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerCloudPoint> seeds = {
        mkSeed(0.5f, 0.5f, 0.5f),
        mkSeed(10.5f, 0.5f, 0.5f),
        mkSeed(20.5f, 0.5f, 0.5f),
    };
    ASSERT_TRUE(fuse.seed(seeds).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 3u);

    fuse.Clear();
    EXPECT_EQ(fuse.GetVoxelCount(), 0u);
    EXPECT_EQ(fuse.GetFusedPointCount(), 0u);
    EXPECT_TRUE(fuse.GetFusedPoints().empty());

    // Clear 后表可复用
    std::vector<MarkerCloudPoint> reseed = { mkSeed(5.5f, 5.5f, 5.5f) };
    ASSERT_TRUE(fuse.seed(reseed).success);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
    EXPECT_FLOAT_EQ(fuse.GetFusedPoints()[0].x, 5.5f);
}

// ============================================================
// 5. 顺序契约：seed 须先于任何 Execute()。
//    fuse 之后再 seed 的行为不作定义（同体素时后到 seed 点被丢弃），
//    仅验证该调用不崩溃。
// ============================================================

TEST(MarkerCloudFuseCPU, SeedBeforeFuseOrderDoc) {
    MarkerCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    MarkerCloudFuseCPU fuse(p);

    std::vector<MarkerFuseInput> frame = { mkInput(0.5f, 0.5f, 0.5f) };
    auto r = fuse.Execute(frame);
    ASSERT_TRUE(r.success);

    // 违反前置顺序（fuse 后 seed）：行为不定义，但不得崩溃
    std::vector<MarkerCloudPoint> late = { mkSeed(0.6f, 0.6f, 0.6f) };
    (void)fuse.seed(late);
    SUCCEED();
}
