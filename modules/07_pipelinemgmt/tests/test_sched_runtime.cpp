// ============================================================================
// test_sched_runtime.cpp — SchedulerRuntime 端到端单测（假钩子 + 假 GPU 工厂）
// 组合底座件全链路：抓帧（两副面孔）→ GPU guard → gpuChain（ccl 就绪点
// frontReady 回调提交 pChain）→ eFinalize（收有效 future）→ 输出队列；
// 启停逆序 / 帧内并行 / 在飞帧排空 / 四计数统计 / restart。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "SlotRing.h"
#include "sched/FrameResultQueue.h"
#include "sched/GpuSlotService.h"
#include "sched/IFrameSource.h"
#include "sched/SchedConfig.h"
#include "sched/SchedulerRuntime.h"

using namespace Scanner::pipeline::sched;
using Scanner::Result;
using Scanner::data::SlotRing;

namespace {

struct Frame { uint64_t id; };
struct Front { uint64_t seen = 0; };
struct Out { uint64_t frameId = 0; };
using OutQueue = FrameResultQueue<Out>;
using StreamH = GpuSlotService::StreamHandle;

auto mkFrame = [](uint64_t id) { return std::make_shared<Frame>(Frame{id}); };

// 假 GPU 工厂（参照 test_sched_gpuslot.cpp：无 GPU 设备依赖，创建/销毁计数对称）
GpuSlotService::StreamFactory fakeFactory(std::atomic<int>& live) {
    return [&live](StreamH* s) {
        *s = reinterpret_cast<StreamH>(static_cast<uintptr_t>(++live));
        return 0;
    };
}
GpuSlotService::StreamDestroyer fakeDestroyer(std::atomic<int>& live) {
    return [&live](StreamH) { --live; };
}

SchedConfig baseCfg(int lanes) {
    SchedConfig c;
    c.lanes = lanes;
    c.gpuSlots = 1;
    c.queueCapacity = 32;
    c.dropThreshold = 64;   // 大阈值：用例内帧数不触发跳帧（用例 8 另行调小）
    return c;               // gpuAcquireTimeout 默认 2s：假钩子持槽极短，不会超时
}

void writeFrames(SlotRing<Frame>& ring, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) ring.write(mkFrame(i));
}

// 轮询等待 processed 达 n（10ms × 500 = 5s 上限）
bool waitProcessed(SchedulerRuntime& rt, uint64_t n) {
    for (int i = 0; i < 500; ++i) {
        if (rt.stats().processed >= n) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return rt.stats().processed >= n;
}

// 轮询等待谓词为真（10ms × 500 = 5s 上限）
template<typename Pred>
bool waitUntil(Pred pred) {
    for (int i = 0; i < 500; ++i) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

std::vector<Out> drainQueue(OutQueue& q) {
    std::vector<Out> out;
    while (auto o = q.pop(std::chrono::milliseconds(200))) out.push_back(*o);
    return out;
}

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// queue 实参占位：nullptr 无法推导模板 TResult，统一走显式转型
OutQueue* noQueue() { return static_cast<OutQueue*>(nullptr); }

} // namespace

// 用例 1：E2E — 20 帧全部走完 抓帧→GPU→P→eFinalize→队列，帧号 0..19 无缺无重
TEST(SchedRuntime, E2E_AllFramesProcessed) {
    SlotRing<Frame> ring(32, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 20);
    OutQueue q(32);
    std::atomic<int> live{0};
    std::mutex gpuSeenMu;
    std::vector<uint64_t> gpuSeen;                    // gpuChain 见过的帧号

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>& f, Front&,
                         std::function<void()>) {    // 不调 frontReady：走兜底提交
        { std::lock_guard<std::mutex> g(gpuSeenMu); gpuSeen.push_back(f->id); }
        return true;
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;                            // 填 result.frameId=帧号
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>& fut) {
        if (!fut.get().success) return Result::fail("pChain failed");  // 钩子消费 future
        q.push(r);
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(2), src, /*sequential=*/false, &q, hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 20));
    rt.requestStop();
    rt.drainAndShutdown();

    auto results = drainQueue(q);
    ASSERT_EQ(results.size(), 20u);                   // 无重：两 lane 分工恰 20 份
    std::set<uint64_t> ids, expect;
    for (auto& o : results) ids.insert(o.frameId);
    for (uint64_t i = 0; i < 20; ++i) expect.insert(i);
    EXPECT_EQ(ids, expect);                           // 无缺：帧号 0..19 全到
    EXPECT_EQ(rt.stats().processed, 20u);
    EXPECT_FALSE(rt.isRunning());                     // drain 后不在运行
    {   // gpuChain 也恰见 20 帧各一次
        std::lock_guard<std::mutex> g(gpuSeenMu);
        EXPECT_EQ(gpuSeen.size(), 20u);
    }
}

// 用例 2：gpuChain 偶数帧 false=帧销毁（未触发 frontReady→不提交）→ 队列仅奇数帧；
// false 只计 gpuRejects
TEST(SchedRuntime, GpuChainFalseDropsFrame) {
    SlotRing<Frame> ring(32, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 20);
    OutQueue q(32);
    std::atomic<int> live{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>& f, Front&,
                        std::function<void()>) {
        return (f->id % 2) == 1;                      // 偶数帧 false=帧销毁（不提交）
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>& fut) {
        if (!fut.get().success) return Result::fail("pChain failed");
        q.push(r);
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(2), src, /*sequential=*/false, &q, hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 10));
    rt.requestStop();
    rt.drainAndShutdown();

    auto results = drainQueue(q);
    ASSERT_EQ(results.size(), 10u);                   // 队列仅奇数帧（10 帧）
    for (auto& o : results) EXPECT_EQ(o.frameId % 2, 1u);
    const auto st = rt.stats();
    EXPECT_EQ(st.gpuRejects, 10u);                    // gpuChain false 只计 gpuRejects
    EXPECT_EQ(st.droppedSkips, 0u);                   // 与 droppedSkips 分开
    EXPECT_EQ(st.processed, 10u);
}

// 用例 3：ccl 就绪点提交 — gpuChain 前段记 T1、调 frontReady() 提交、sleep 50ms
// （模拟激光链仍在跑）、记 T3 返回；pChain 入口记 T2。
// T1 < T2 < T3：P 链在 GPU 链未完时已在跑（帧内并行）。偶发调度延迟 3 轮取满足者。
TEST(SchedRuntime, ParallelFrontAndP) {
    for (int round = 0; round < 3; ++round) {
        SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
        GrabLatestSource<Frame> src(ring, 64);
        ring.write(mkFrame(0));
        std::atomic<int> live{0};
        std::atomic<int64_t> t1{0}, t2{0}, t3{0};

        LaneHooks<Frame, Front, Out> hooks;
        hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                             std::function<void()> frontReady) {
            t1.store(nowNs());                        // GPU 前段完成（ccl 数据就绪）
            frontReady();                             // ccl 就绪点提交 pChain
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 激光链仍在跑
            t3.store(nowNs());
            return true;
        };
        hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out& r) {
            t2.store(nowNs());                        // P 核任务入口
            r.frameId = 0;
            return Result::ok();
        };
        hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) {
            return fut.get().success ? Result::ok() : Result::fail("pChain failed");
        };

        SchedulerRuntime rt;
        rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
        ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, noQueue(), hooks).success);
        ASSERT_TRUE(waitProcessed(rt, 1));
        rt.requestStop();
        rt.drainAndShutdown();

        if (t1.load() < t2.load() && t2.load() < t3.load()) {
            SUCCEED() << "round " << round << " 满足 T1(" << t1.load() << ") < T2(" << t2.load()
                      << ") < T3(" << t3.load() << ")";
            return;
        }
        ADD_FAILURE() << "round " << round << " 未满足: T1=" << t1.load() << " T2=" << t2.load()
                      << " T3=" << t3.load();
    }
    FAIL() << "3 轮均未观察到 T1 < T2 < T3";
}

// 用例 3b：gpuChain 不调 frontReady → 兜底路径：pChain 仍在 gpuChain 返回后才提交执行。
// gpuChain 内 sleep 20ms 保证若被错误提前提交则 T2 必落在 T3 之前 → 断言可检出。
TEST(SchedRuntime, FrontReadyNotCalledSubmitsAfterReturn) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    ring.write(mkFrame(0));
    std::atomic<int> live{0};
    std::atomic<int64_t> t2{0}, t3{0};
    std::atomic<int> pRan{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                         std::function<void()> frontReady) {
        (void)frontReady;                             // 不调：走兜底路径
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        t3.store(nowNs());                            // gpuChain 返回时刻
        return true;
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out& r) {
        t2.store(nowNs());                            // P 核任务入口
        ++pRan;
        r.frameId = 0;
        return Result::ok();
    };
    hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) {
        return fut.get().success ? Result::ok() : Result::fail("pChain failed");
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, noQueue(), hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 1));
    rt.requestStop();
    rt.drainAndShutdown();

    EXPECT_EQ(pRan.load(), 1);                        // 仍被提交执行
    EXPECT_GE(t2.load(), t3.load());                  // 提交发生在 gpuChain 返回之后
    EXPECT_EQ(rt.stats().processed, 1u);
}

// 用例 4：pChain 在飞时 requestStop+drain —— 在飞帧跑完并发布，不死锁（<5s）
TEST(SchedRuntime, StopDrainsInFlight) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 2);
    OutQueue q(8);
    std::atomic<int> live{0};
    std::atomic<bool> pChainEntered{false};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                        std::function<void()>) {      // 不调 frontReady：返回后兜底提交
        return true;
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        pChainEntered.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));  // P 核长活
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>& fut) {
        if (!fut.get().success) return Result::fail("pChain failed");
        q.push(r);
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, &q, hooks).success);
    ASSERT_TRUE(waitUntil([&] { return pChainEntered.load(); }));  // 帧 0 已进入 pChain 在飞
    const auto t0 = std::chrono::steady_clock::now();
    rt.requestStop();
    rt.drainAndShutdown();                            // join lane（在飞帧跑完）→ broker → gpu
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    EXPECT_GE(rt.stats().processed, 1u);              // 在飞帧不丢
    EXPECT_LT(elapsedMs, 5000);                       // 无死锁
    EXPECT_GE(drainQueue(q).size(), 1u);              // 已提交帧的结果仍发布
}

// 用例 5：不调 frontReady → 提交推迟到 gpuChain 返回后：pChain 入口必见 gpuDone
// （兜底路径保证数据依赖：gpuChain 完成后才提交）
TEST(SchedRuntime, DeferredSubmitSeesGpuDone) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    ring.write(mkFrame(0));
    std::atomic<int> live{0};
    std::atomic<bool> gpuDone{false};
    std::atomic<bool> pChainObservedAfterGpu{false};
    std::atomic<int> finalized{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                         std::function<void()>) {     // 不调 frontReady
        bool exp = false;
        gpuDone.compare_exchange_strong(exp, true);   // gpuChain 完成标记
        return true;
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out& r) {
        pChainObservedAfterGpu.store(gpuDone.load()); // 入口观察 gpuChain 是否已完成
        r.frameId = 0;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) {
        (void)fut.get();
        ++finalized;
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, noQueue(), hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 1));
    rt.requestStop();
    rt.drainAndShutdown();

    EXPECT_EQ(finalized.load(), 1);
    EXPECT_TRUE(pChainObservedAfterGpu.load());       // 在 eFinalize 之后核对：pChain 确见 gpuDone
}

// 用例 6：SequentialSource 顺序路径（Backpressure 环，lanes=1）—— 6 帧全处理，
// eFinalize 观察到的帧号单调递增
TEST(SchedRuntime, SequentialSourcePath) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Backpressure);
    SequentialSource<Frame> src(ring);                // 默认 100ms 超时
    writeFrames(ring, 6);
    std::atomic<int> live{0};
    std::mutex mu;
    std::vector<uint64_t> ids;                        // eFinalize 观察到的帧号序列

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                        std::function<void()>) {
        return true;
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>& f, Front&, Out&, std::future<Result>& fut) {
        (void)fut.get();
        { std::lock_guard<std::mutex> g(mu); ids.push_back(f->id); }
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/true, noQueue(), hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 6));
    rt.requestStop();
    rt.drainAndShutdown();

    std::lock_guard<std::mutex> g(mu);
    ASSERT_EQ(ids.size(), 6u);                        // 6 帧全处理
    for (size_t i = 1; i < ids.size(); ++i) {
        EXPECT_LT(ids[i - 1], ids[i]) << "帧号须单调递增";
    }
    EXPECT_EQ(rt.stats().processed, 6u);
}

// 用例 7：一次性落后 30 帧、dropThreshold=8 → 首抓跳最新：droppedSkips>0 且 processed>0
TEST(SchedRuntime, StatsSkipsAccumulate) {
    SlotRing<Frame> ring(32, SlotRing<Frame>::WriterMode::Overwrite);
    writeFrames(ring, 30);                            // 先写完再 start：首抓 lag=30 > 8
    GrabLatestSource<Frame> src(ring, 8);
    std::atomic<int> live{0};
    SchedConfig cfg = baseCfg(1);
    cfg.dropThreshold = 8;

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                        std::function<void()>) {
        return true;
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>&, Front&, Out&) { return Result::ok(); };
    hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) {
        (void)fut.get();
        return Result::ok();
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(cfg, src, /*sequential=*/false, noQueue(), hooks).success);
    ASSERT_TRUE(waitProcessed(rt, 1));
    rt.drainAndShutdown();                            // 未 requestStop 直接 drain（内部先置停）

    const auto st = rt.stats();
    EXPECT_GT(st.droppedSkips, 0u);                   // 期望 29（跳到最新丢中间帧）
    EXPECT_GT(st.processed, 0u);
}

// 用例 8：restart — start→drain 后同 runtime 再 start 全绿（gpu 服务重建、计数清零、
// running 守卫 setGpuStreamFactory、流创建/销毁对称）
TEST(SchedRuntime, RestartAfterDrain) {
    std::atomic<int> live{0};
    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));

    // 轮 1：2 帧处理 → drain
    SlotRing<Frame> ring1(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src1(ring1, 64);
    writeFrames(ring1, 2);
    LaneHooks<Frame, Front, Out> hooks1;
    hooks1.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                         std::function<void()>) { return true; };
    hooks1.pChain = [](const std::shared_ptr<const Frame>&, Front&, Out&) { return Result::ok(); };
    hooks1.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) {
        (void)fut.get();
        return Result::ok();
    };
    ASSERT_TRUE(rt.start(baseCfg(1), src1, /*sequential=*/false, noQueue(), hooks1).success);
    EXPECT_TRUE(rt.isRunning());
    ASSERT_TRUE(waitProcessed(rt, 2));
    rt.requestStop();
    rt.drainAndShutdown();
    EXPECT_FALSE(rt.isRunning());
    EXPECT_EQ(rt.stats().processed, 2u);

    // 轮 2：同 runtime 再 start（gpu 服务重建；计数清零）
    SlotRing<Frame> ring2(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src2(ring2, 64);
    writeFrames(ring2, 3);
    OutQueue q2(8);
    LaneHooks<Frame, Front, Out> hooks2;
    hooks2.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                         std::function<void()>) { return true; };
    hooks2.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;
        return Result::ok();
    };
    hooks2.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>& fut) {
        (void)fut.get();
        q2.push(r);
        return Result::ok();
    };
    ASSERT_TRUE(rt.start(baseCfg(1), src2, /*sequential=*/false, &q2, hooks2).success);
    EXPECT_TRUE(rt.isRunning());
    // running 守卫：运行中注入工厂须 fail
    EXPECT_FALSE(rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live)).success);
    ASSERT_TRUE(waitProcessed(rt, 3));
    rt.requestStop();
    rt.drainAndShutdown();
    EXPECT_FALSE(rt.isRunning());

    EXPECT_EQ(rt.stats().processed, 3u);              // 每轮清零后计数
    auto results = drainQueue(q2);
    ASSERT_EQ(results.size(), 3u);
    std::set<uint64_t> ids;
    for (auto& o : results) ids.insert(o.frameId);
    std::set<uint64_t> expect{0, 1, 2};
    EXPECT_EQ(ids, expect);                           // 轮 2 的 3 帧全到
    EXPECT_EQ(live.load(), 0);                        // 两轮流创建/销毁完全对称
}

// 用例 9（T10）：gpuChain 调 frontReady() 后返回 false —— pChain 成了孤儿但被
// fut.wait() 等待完成才进下一帧：帧被弃置（processed==0）但无孤儿残留、无
// TFront 并发写（等待在下一帧抓帧之前完成）、无崩溃，总时长 < 5s
TEST(SchedRuntime, OrphanWaitAfterFrontReadyThenFalse) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 2);
    std::atomic<int> live{0};
    std::atomic<int> pRan{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                        std::function<void()> frontReady) {
        frontReady();                                 // ccl 就绪点提交 pChain……
        return false;                                 // ……随后帧销毁 → pChain 成孤儿
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 孤儿在飞一会儿
        ++pRan;                                       // 正常执行完（写 TFront 分区）
        return Result::ok();
    };
    hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>&) {
        return Result::fail("frame discarded");       // 不应被调用（帧已弃置）
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, noQueue(), hooks).success);
    ASSERT_TRUE(waitUntil([&] { return pRan.load() >= 2 && rt.stats().gpuRejects >= 2u; }));
    rt.drainAndShutdown();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    EXPECT_EQ(rt.stats().processed, 0u);              // 帧全弃置
    EXPECT_EQ(pRan.load(), 2);                        // pChain 均执行过（孤儿被等完非弃置）
    EXPECT_EQ(rt.stats().gpuRejects, 2u);             // false 帧只计 gpuRejects
    EXPECT_EQ(live.load(), 0);                        // 流销毁对称：无泄漏/无二次销毁
    EXPECT_LT(elapsedMs, 5000);                       // 无孤儿卡死
}

// 用例 10（T10）：eFinalize 抛异常 → runtime 顶层捕获、异常即停：
// finalizeFails>=1，drain 不挂死（isRunning 变 false），processed 不计异常帧
TEST(SchedRuntime, HookExceptionStopsRuntime) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 3);
    std::atomic<int> live{0};
    std::atomic<int> finalized{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&,
                        std::function<void()>) { return true; };
    hooks.pChain = [](const std::shared_ptr<const Frame>&, Front&, Out&) { return Result::ok(); };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>& fut) -> Result {
        (void)fut.get();
        ++finalized;
        throw std::runtime_error("eFinalize boom");   // 钩子异常：顶层须捕获并停机
    };

    SchedulerRuntime rt;
    rt.setGpuStreamFactory(fakeFactory(live), fakeDestroyer(live));
    ASSERT_TRUE(rt.start(baseCfg(1), src, /*sequential=*/false, noQueue(), hooks).success);
    ASSERT_TRUE(waitUntil([&] { return finalized.load() >= 1; }));   // 异常已发生
    const auto t0 = std::chrono::steady_clock::now();
    rt.drainAndShutdown();                            // 异常已触发 requestStop：须不挂死
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    EXPECT_FALSE(rt.isRunning());                     // 停机
    EXPECT_GE(rt.stats().finalizeFails, 1u);          // 异常被顶层捕获计数
    EXPECT_EQ(rt.stats().processed, 0u);              // 异常帧不计成功
    EXPECT_LT(elapsedMs, 5000);                       // drain 不挂死
    EXPECT_EQ(live.load(), 0);                        // 流销毁对称（无崩溃）
}
