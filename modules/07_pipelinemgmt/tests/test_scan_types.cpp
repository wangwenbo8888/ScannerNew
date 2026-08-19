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
// GpuPointCloudPool（假分配器：默认构造 GpuMat，无 GPU）
// ============================================================================
#ifdef JMW_BUILD_CUDA
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ringbuffer_sink.h>

#include "pipelines/scan/GpuPointCloudPool.h"

static GpuPointCloudPool::AllocFn fakeAlloc() {
    return [](size_t) { return cv::cuda::GpuMat(); };
}

// 语义 1：取块→占用；析构→回池；再取得同块（slotId 相等，LIFO 复用）
TEST(PoolAcquireReleaseCycle, ReuseSameSlot) {
    GpuPointCloudPool pool(2, 1024, fakeAlloc());
    EXPECT_EQ(pool.available(), 2u);
    EXPECT_EQ(pool.inUse(), 0u);

    uint32_t slot = 0xFFFFFFFFu;
    {
        auto b = pool.acquire(std::chrono::milliseconds(100));
        ASSERT_TRUE(b.has_value());
        ASSERT_NE(*b, nullptr);
        slot = (*b)->slotId;
        (*b)->count = 7;                           // 用户填内容（复用可覆盖）
        EXPECT_EQ(pool.inUse(), 1u);
        EXPECT_EQ(pool.available(), 1u);
    }                                              // 块析构自动回池
    EXPECT_EQ(pool.inUse(), 0u);
    EXPECT_EQ(pool.available(), 2u);

    auto b2 = pool.acquire(std::chrono::milliseconds(100));
    ASSERT_TRUE(b2.has_value());
    EXPECT_EQ((*b2)->slotId, slot);                // 同块复用
    EXPECT_EQ((*b2)->count, 0);                    // T13 防御：acquire 出块 count 已清零
}

// 语义 2：取光后 acquire(50ms) 超时 → nullopt
TEST(PoolExhaustionTimeout, ExhaustedThenTimeout) {
    GpuPointCloudPool pool(2, 512, fakeAlloc());
    auto b1 = pool.acquire(std::chrono::milliseconds(100));
    auto b2 = pool.acquire(std::chrono::milliseconds(100));
    ASSERT_TRUE(b1.has_value() && b2.has_value());
    EXPECT_EQ(pool.inUse(), 2u);
    EXPECT_EQ(pool.available(), 0u);

    auto t0 = std::chrono::steady_clock::now();
    auto b3 = pool.acquire(std::chrono::milliseconds(50));
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    EXPECT_FALSE(b3.has_value());                  // 超时 nullopt
    EXPECT_GE(elapsedMs, 40);                      // 确实阻塞等待过
    EXPECT_EQ(pool.inUse(), 2u);                   // 占用不变
}

// 语义 3：池空阻塞中，另一线程归还 1 块 → 阻塞 acquire 被唤醒成功
TEST(PoolWakeupOnRelease, BlockedAcquireWoken) {
    GpuPointCloudPool pool(1, 256, fakeAlloc());
    auto b1 = pool.acquire(std::chrono::milliseconds(100));
    ASSERT_TRUE(b1.has_value());
    uint32_t slot = (*b1)->slotId;

    std::optional<std::shared_ptr<GpuPointCloudBlock>> woken;   // 存活越过 join
    std::thread waiter([&] {
        woken = pool.acquire(std::chrono::milliseconds(5000));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 让 waiter 入阻塞
    b1.reset();                                    // 归还 → 唤醒 waiter
    waiter.join();
    ASSERT_TRUE(woken.has_value());                // 阻塞 acquire 被唤醒成功
    EXPECT_EQ((*woken)->slotId, slot);             // 得到归还的同一块
    EXPECT_EQ(pool.inUse(), 1u);                   // woken 仍持有
    EXPECT_EQ(pool.available(), 0u);
}

// 语义 4（T13 防御）：池析构时仍有在飞块 → spdlog::warn（含 inUse 数字）。
// 在飞块故意泄漏不析构：其回池 deleter 会触碰已亡池（UB），泄漏交由进程退出回收。
TEST(PoolDestructorWithInFlight, WarnsInUseCount) {
    auto ring = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64);
    auto prev = spdlog::default_logger();
    spdlog::set_default_logger(std::make_shared<spdlog::logger>("pooltest", ring));
    {
        GpuPointCloudPool pool(2, 128, fakeAlloc());
        auto b = pool.acquire(std::chrono::milliseconds(100));
        ASSERT_TRUE(b.has_value());
        auto* leaked = new std::optional<std::shared_ptr<GpuPointCloudBlock>>(std::move(b));
        (void)leaked;                              // 故意不析构（防 UB）
        EXPECT_EQ(pool.inUse(), 1u);
    }                                              // 池先亡：inUse=1 → warn
    auto lines = ring->last_formatted(16);
    spdlog::set_default_logger(prev);
    bool found = false;
    for (const auto& s : lines)
        if (s.find("GpuPointCloudPool") != std::string::npos &&
            s.find("inUse=1") != std::string::npos)
            found = true;
    EXPECT_TRUE(found);
}
#endif // JMW_BUILD_CUDA
