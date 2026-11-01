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

// M5 移植：shutdown 不销毁在飞 guard 的流；guard 归还即销毁恰好一次
TEST(GpuSlot, ShutdownWithInFlightGuard) {
    GpuSlotService svc;
    std::atomic<int> live{0};
    ASSERT_TRUE(svc.start(2, fakeFactory(live), fakeDestroyer(live)).success);
    {
        auto g1 = svc.acquire(std::chrono::milliseconds(100));
        auto g2 = svc.acquire(std::chrono::milliseconds(100));
        ASSERT_TRUE(g1 && g2);
        svc.shutdown();                           // 池空：在飞 2 流不被 shutdown 销毁
        EXPECT_FALSE(svc.isRunning());
        EXPECT_EQ(live.load(), 2);
        g1->reset();                              // 归还即销毁（不回池）
        EXPECT_EQ(live.load(), 1);
    }                                             // g2 析构归还
    EXPECT_EQ(live.load(), 0);                    // 恰好销毁一次（无泄漏/无二次销毁）
    svc.shutdown();                               // 幂等：不再动计数
    EXPECT_EQ(live.load(), 0);
}

// M5 移植：guard 移动链——移动构造/移动赋值后源析构不归还，新对象恰归还一次
TEST(GpuSlot, GuardMoveChain) {
    GpuSlotService svc;
    std::atomic<int> live{0};
    ASSERT_TRUE(svc.start(1, fakeFactory(live), fakeDestroyer(live)).success);
    GpuSlotService::SlotGuard mv2;
    {
        GpuSlotService::SlotGuard mv;
        {
            auto src = svc.acquire(std::chrono::milliseconds(100));
            ASSERT_TRUE(src);
            mv = GpuSlotService::SlotGuard(std::move(*src));  // 移动构造临时 + 移动赋值
        }                                         // src（moved-from）析构：不归还
        EXPECT_FALSE(svc.acquire(std::chrono::milliseconds(50)).has_value());  // 池仍空=源未归还
        mv2 = std::move(mv);                      // 移动赋值：mv 失效
    }                                             // mv（moved-from）析构：不归还
    EXPECT_FALSE(svc.acquire(std::chrono::milliseconds(50)).has_value());      // 池仍空
    mv2.reset();                                  // 新对象归还恰一次
    {
        auto back = svc.acquire(std::chrono::milliseconds(100));
        ASSERT_TRUE(back.has_value());            // 归还的槽可再取
        EXPECT_FALSE(svc.acquire(std::chrono::milliseconds(50)).has_value());  // 持有期间池空=无双重归还
    }
    svc.shutdown();
    EXPECT_EQ(live.load(), 0);                    // 创建/销毁对称（恰好一次）
}
