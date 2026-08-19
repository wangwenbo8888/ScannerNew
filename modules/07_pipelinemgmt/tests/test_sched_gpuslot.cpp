// ============================================================================
// test_sched_gpuslot.cpp — GpuSlotService 单测（假工厂路径，无 GPU 设备依赖）
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include "sched/GpuSlotService.h"
using namespace Scanner::pipeline::sched;

using StreamH = GpuSlotService::StreamHandle;   // 头文件导出别名（BUILD_CUDA 无关）

static GpuSlotService::StreamFactory fakeFactory(std::atomic<int>& live) {
    return [&live](StreamH* s) { *s = reinterpret_cast<StreamH>(static_cast<uintptr_t>(++live)); return 0; };
}
static GpuSlotService::StreamDestroyer fakeDestroyer(std::atomic<int>& live) {
    return [&live](StreamH) { --live; };
}

TEST(GpuSlot, MutualExclusionAndRelease) {
    GpuSlotService svc;
    std::atomic<int> live{0};
    ASSERT_TRUE(svc.start(2, fakeFactory(live), fakeDestroyer(live)).success);
    auto g1 = svc.acquire(std::chrono::milliseconds(100));
    auto g2 = svc.acquire(std::chrono::milliseconds(100));
    ASSERT_TRUE(g1 && g2);
    EXPECT_NE(g1->stream, g2->stream);
    EXPECT_FALSE(svc.acquire(std::chrono::milliseconds(50)).has_value());   // 满→超时
    g1->reset();                                     // 显式提前归还
    EXPECT_TRUE(svc.acquire(std::chrono::milliseconds(100)).has_value());
    svc.shutdown();
}
TEST(GpuSlot, GuardDtorReleases) {
    GpuSlotService svc; std::atomic<int> live{0};
    svc.start(1, fakeFactory(live), fakeDestroyer(live));
    { auto g = svc.acquire(std::chrono::milliseconds(100)); ASSERT_TRUE(g); }
    EXPECT_TRUE(svc.acquire(std::chrono::milliseconds(100)).has_value());  // 析构已归还
    svc.shutdown();
}
TEST(GpuSlot, GuardResetIdempotent) {
    GpuSlotService svc; std::atomic<int> live{0};
    svc.start(1, fakeFactory(live), fakeDestroyer(live));
    auto g = svc.acquire(std::chrono::milliseconds(100));
    ASSERT_TRUE(g);
    g->reset(); g->reset();                          // 二次 reset 不重复归还
    EXPECT_TRUE(svc.acquire(std::chrono::milliseconds(100)).has_value());
    svc.shutdown();
}
TEST(GpuSlot, ShutdownDestroysStreams) {
    GpuSlotService svc; std::atomic<int> live{0};
    svc.start(2, fakeFactory(live), fakeDestroyer(live));
    svc.shutdown();
    EXPECT_EQ(live.load(), 0);                       // 创建/销毁计数对称
}
TEST(GpuSlot, AcquireBeforeStartFails) {
    GpuSlotService svc;
    EXPECT_FALSE(svc.acquire(std::chrono::milliseconds(10)).has_value());
}
