// ============================================================================
// test_scan_types.cpp — C 扫描自有类型（ScanTypes 纯类型 + GpuPointCloudPool 池）
// 池测试走假分配器路径（返回默认构造空 GpuMat，无 GPU 设备依赖）。
// ============================================================================
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "pipelines/scan/ScanTypes.h"

using namespace Scanner::pipeline;

// ============================================================================
// 纯类型默认值（编译 + 字段断言）
// ============================================================================
TEST(ScanConfigDefaults, Fields) {
    ScanConfig cfg;
    EXPECT_TRUE(cfg.enableLaser);
    EXPECT_TRUE(cfg.enableFinalBA);
    EXPECT_EQ(cfg.laserCacheBudgetMB, 2048u);
    EXPECT_TRUE(cfg.existingMarkers.empty());
}

TEST(FrameResultFields, Defaults) {
    FrameResult fr;
    EXPECT_EQ(fr.frameId, 0u);
    EXPECT_EQ(fr.timestamp, 0u);
    EXPECT_DOUBLE_EQ(fr.temperature, 0.0);
    const double I[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(fr.R[i], I[i]);
    EXPECT_DOUBLE_EQ(fr.T[0], 0.0);
    EXPECT_DOUBLE_EQ(fr.T[1], 0.0);
    EXPECT_DOUBLE_EQ(fr.T[2], 0.0);
    EXPECT_TRUE(fr.markers.empty());
#ifdef JMW_BUILD_CUDA
    EXPECT_EQ(fr.laser, nullptr);                 // 空=无激光
#else
    EXPECT_EQ(fr.laser, 0);                       // CUDA 关闭占位
#endif
    EXPECT_EQ(fr.quality, Scanner::QualityFlag::Normal);
}

TEST(FrameObsDefaults, NoLaserSlot) {
    FrameObs fo;
    EXPECT_EQ(fo.frameId, 0u);
    const double I[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(fo.R_init[i], I[i]);
    EXPECT_DOUBLE_EQ(fo.t_init[0], 0.0);
    EXPECT_DOUBLE_EQ(fo.t_init[1], 0.0);
    EXPECT_DOUBLE_EQ(fo.t_init[2], 0.0);
    EXPECT_TRUE(fo.markerObs.empty());
    EXPECT_EQ(fo.laserCacheSlot, FrameObs::kNoLaserSlot);   // 默认=无激光槽
    EXPECT_EQ(FrameObs::kNoLaserSlot, static_cast<size_t>(-1));

    MarkerObs mo;
    EXPECT_DOUBLE_EQ(mo.xyz[0], 0.0);
    EXPECT_DOUBLE_EQ(mo.xyz[1], 0.0);
    EXPECT_DOUBLE_EQ(mo.xyz[2], 0.0);
    EXPECT_EQ(mo.globalId, -1);
    EXPECT_FALSE(mo.isHighPrecision);
}

// ============================================================================
// GpuPointCloudPool 测试段已随旧 lane 退役删除（261027）
