// ============================================================================
// test_sched_runtime.cpp — SchedulerRuntime 端到端单测（假钩子 + 假 GPU 工厂）
// 组合底座件全链路：抓帧（两副面孔）→ GPU guard → gpuChain → pChain（Broker）
// → eFinalize → 输出队列；启停逆序 / 帧内并行 / 在飞帧排空 / 三计数统计。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
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
    c.dropThreshold = 64;   // 大阈值：用例内帧数不触发跳帧（用例 7 另行调小）
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
    hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>& f, Front&) {
        { std::lock_guard<std::mutex> g(gpuSeenMu); gpuSeen.push_back(f->id); }
        return true;
    };
    hooks.onFrontReady = [](const std::shared_ptr<const Frame>&, Front&) {};
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;                            // 填 result.frameId=帧号
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>&) {
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

// 用例 2：gpuChain 偶数帧 false=帧销毁 → 队列仅奇数帧；false 只计 gpuRejects
TEST(SchedRuntime, GpuChainFalseDropsFrame) {
    SlotRing<Frame> ring(32, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 20);
    OutQueue q(32);
    std::atomic<int> live{0};

    LaneHooks<Frame, Front, Out> hooks;
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>& f, Front&) {
        return (f->id % 2) == 1;                      // 偶数帧 false=帧销毁
    };
    hooks.onFrontReady = [](const std::shared_ptr<const Frame>&, Front&) {};
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>&) {
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

// 用例 3：onFrontReady 非空 → 提交先于 gpuChain 返回（T1 前端就绪 < T2 P 链入口 < T3 GPU 链返回）。
// 偶发调度延迟允许 3 轮取满足者。
TEST(SchedRuntime, ParallelFrontAndP) {
    for (int round = 0; round < 3; ++round) {
        SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
        GrabLatestSource<Frame> src(ring, 64);
        ring.write(mkFrame(0));
        std::atomic<int> live{0};
        std::atomic<int64_t> t1{0}, t2{0}, t3{0};

        auto onFront = [&](const std::shared_ptr<const Frame>&, Front&) {
            int64_t exp = 0;
            t1.compare_exchange_strong(exp, nowNs()); // 首次触发记 T1（重复调用不覆盖）
        };
        LaneHooks<Frame, Front, Out> hooks;
        hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>& f, Front& fr) {
            onFront(f, fr);                           // 模拟 GPU 段中途触发前端就绪回调
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 模拟激光链仍在跑
            t3.store(nowNs());
            return true;
        };
        hooks.onFrontReady = onFront;
        hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out& r) {
            t2.store(nowNs());                        // P 核任务入口
            r.frameId = 0;
            return Result::ok();
        };
        hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>&) {
            return Result::ok();
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

// 用例 4：pChain 在飞时 requestStop+drain —— 在飞帧跑完并发布，不死锁（<5s）
TEST(SchedRuntime, StopDrainsInFlight) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    writeFrames(ring, 2);
    OutQueue q(8);
    std::atomic<int> live{0};
    std::atomic<bool> pChainEntered{false};

    LaneHooks<Frame, Front, Out> hooks;               // onFrontReady 留空：gpuChain 返回后才提交
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&) {
        return true;
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        pChainEntered.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));  // P 核长活
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out& r, std::future<Result>&) {
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

// 用例 5：onFrontReady 空 → 提交推迟到 gpuChain 返回后：pChain 入口必见 gpuDone
TEST(SchedRuntime, FrontReadyEmptyDefersSubmit) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 64);
    ring.write(mkFrame(0));
    std::atomic<int> live{0};
    std::atomic<bool> gpuDone{false};
    std::atomic<bool> pChainObservedAfterGpu{false};
    std::atomic<int> finalized{0};

    LaneHooks<Frame, Front, Out> hooks;               // onFrontReady 空
    hooks.gpuChain = [&](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&) {
        bool exp = false;
        gpuDone.compare_exchange_strong(exp, true);   // gpuChain 完成标记
        return true;
    };
    hooks.pChain = [&](const std::shared_ptr<const Frame>&, Front&, Out& r) {
        pChainObservedAfterGpu.store(gpuDone.load()); // 入口观察 gpuChain 是否已完成
        r.frameId = 0;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>&) {
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
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&) {
        return true;
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>& f, Front&, Out& r) {
        r.frameId = f->id;
        return Result::ok();
    };
    hooks.eFinalize = [&](const std::shared_ptr<const Frame>& f, Front&, Out&, std::future<Result>&) {
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
    hooks.gpuChain = [](GpuSlotService::SlotGuard&, const std::shared_ptr<const Frame>&, Front&) {
        return true;
    };
    hooks.pChain = [](const std::shared_ptr<const Frame>&, Front&, Out&) { return Result::ok(); };
    hooks.eFinalize = [](const std::shared_ptr<const Frame>&, Front&, Out&, std::future<Result>&) {
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
