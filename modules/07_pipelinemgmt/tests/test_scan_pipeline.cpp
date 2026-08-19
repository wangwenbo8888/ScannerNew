// ============================================================================
// test_scan_pipeline.cpp — ScanPipeline 对象总装（假链 + 假融合适配器 + 真 SlotRing）
// 真算子路径（ScanChains/融合算子）已由 T15/T16 覆盖——本测试经 attachTestHooks
// 注入直通假链、attachTestFuseAdapters 注入假融合，端到端验证生命周期编排：
// start/stop / seed 时序（先于任何 fuse）/ pause-resume 状态保留 / 空 markers 不
// seed / A 模式不碰 laserFuse（或无 CUDA 强制降级+一次性上报）/ 必备件缺失 fail。
// ============================================================================
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "base/EventBus.h"
#include "pipelines/scan/ScanPipeline.h"

using namespace Scanner::pipeline;
using Scanner::data::EnhancedFrame;
using Scanner::data::SlotRing;
using Scanner::Result;

namespace {

constexpr size_t kRingSlots = 16;

// —— 轮询等待谓词为真（10ms × 500 = 5s 上限）——
template<typename Pred>
bool waitUntil(Pred pred) {
    for (int i = 0; i < 500; ++i) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

// —— 等 obs 计数稳定（300ms 无变化；上限 5s）—— 返回稳定值
size_t waitStableObs(ScanPipeline& p) {
    size_t last = p.obs().frameCount();
    int stable = 0;
    for (int i = 0; i < 500 && stable < 30; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const size_t c = p.obs().frameCount();
        if (c == last) {
            ++stable;
        } else {
            stable = 0;
            last = c;
        }
    }
    return last;
}

std::shared_ptr<EnhancedFrame> mkFrame(uint64_t id) {
    auto f = std::make_shared<EnhancedFrame>();
    f->frameId = id;
    f->temperature = 25.0;
    return f;
}

void writeFrames(SlotRing<EnhancedFrame>& ring, uint64_t begin, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) ring.write(mkFrame(begin + i));
}

#ifdef JMW_BUILD_CUDA
struct FakeLaserFuse : ILaserFuse {
    std::atomic<int> calls{0};
    void fuse(const GpuPointCloudBlock&, const double*, const double*) override { ++calls; }
};
#endif

// —— 假 seedable 标记融合：全局时序计数（seed 与每次 fuse 各领一个递增序号）——
struct FakeSeedMarkerFuse : ISeedableMarkerFuse {
    std::atomic<int> seedCalls{0};
    std::atomic<uint64_t> seq{0};
    std::atomic<uint64_t> seedSeq{0};
    std::atomic<uint64_t> firstFuseSeq{0};
    std::atomic<uint64_t> fuseCalls{0};
    std::vector<calib::MarkerCloudPoint> fused;

    Result seed(const std::vector<calib::MarkerCloudPoint>&) override {
        ++seedCalls;
        seedSeq.store(++seq);
        return Result::ok();
    }
    void fuse(const std::vector<calib::MarkerPoint3D>&, const double*, const double*) override {
        ++fuseCalls;
        uint64_t s = ++seq;
        uint64_t expect = 0;
        firstFuseSeq.compare_exchange_strong(expect, s);   // 只留首 fuse 序号
    }
    const std::vector<calib::MarkerCloudPoint>& fusedPoints() const override { return fused; }
};

// —— 直通假链：gpuChain 提交即 true；pChain 产 3 个 globalId=0/1/2 + 1 个 99 标记点；
//    eFinalize 收 future 后 push 输出队列（T8 契约：eFinalize 自行 push）——
ScanPipeline::Hooks passthroughHooks(sched::FrameResultQueue<FrameResult>* q,
                                     bool withLaserBlock = false) {
    ScanPipeline::Hooks h;
    h.gpuChain = [](sched::GpuSlotService::SlotGuard&,
                    const std::shared_ptr<const EnhancedFrame>&, ScanFront&,
                    std::function<void()> frontReady) {
        frontReady();                                       // ccl 就绪点（直通）
        return true;
    };
    h.pChain = [withLaserBlock](const std::shared_ptr<const EnhancedFrame>& frame,
                                ScanFront&, FrameResult& r) {
        r.frameId = frame->frameId;
        r.temperature = frame->temperature;
        r.R[0] = 1.0; r.R[4] = 1.0; r.R[8] = 1.0;
        r.T[2] = 0.5 * static_cast<double>(frame->frameId);
        for (int i = 0; i < 3; ++i) {
            calib::MarkerPoint3D m{};                       // globalId=0/1/2（hp 集）
            m.x = 0.1 * i;
            m.y = 0.2 * i;
            m.z = 0.3 * i;
            m.nz = 1.0;
            m.globalId = i;
            r.markers.push_back(m);
        }
        calib::MarkerPoint3D m99{};                         // 非 hp（globalId=99）
        m99.x = 9.0;
        m99.nz = 1.0;
        m99.globalId = 99;
        r.markers.push_back(m99);
#ifdef JMW_BUILD_CUDA
        if (withLaserBlock) {                               // 空块（不触驱动）
            auto b = std::make_shared<GpuPointCloudBlock>();
            b->count = 1;
            b->frameId = frame->frameId;
            r.laser = std::move(b);
        }
#endif
        (void)withLaserBlock;
        return Result::ok();
    };
    h.eFinalize = [q](const std::shared_ptr<const EnhancedFrame>&, ScanFront&,
                      FrameResult& r, std::future<Result>& fut) {
        if (!fut.get().success) return Result::fail("p fail");
        q->push(std::move(r));
        return Result::ok();
    };
    return h;
}

ScanConfig baseCfg() {
    ScanConfig cfg;
    cfg.enableLaser = false;                                // 假链路径不测激光（用例 5 专测）
    return cfg;
}

} // namespace

// 1：生命周期——假链 + ring 写 10 帧 → start → obs 齐收 10 帧（真 CPU marker 融合适
//    配器路径不崩）→ stop → isRunning false，全程 < 5s
TEST(ScanPipelineTest, StartStopLifecycle) {
    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanPipeline pipe(baseCfg());
    pipe.attachRing(ring, 64);
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));

    PipelineDeps deps;                                      // eventBus/sceneFeed 均可空
    ASSERT_TRUE(pipe.configure(deps).success);
    EXPECT_FALSE(pipe.isRunning());
    writeFrames(ring, 0, 10);

    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(pipe.start().success);
    EXPECT_TRUE(pipe.isRunning());
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= 10; }));
    pipe.stop();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();

    EXPECT_FALSE(pipe.isRunning());
    EXPECT_LT(elapsedMs, 5000);                             // 无挂死

    auto snap = pipe.obs().snapshot();
    ASSERT_EQ(snap.obs.size(), 10u);
    std::set<uint64_t> ids;
    for (const auto& fo : snap.obs) ids.insert(fo.frameId);
    for (uint64_t i = 0; i < 10; ++i) EXPECT_EQ(ids.count(i), 1u);
}

// 2：seed 时序——existingMarkers 3 点 → start 内 seed 先于任何 fuse；
//    hpGlobalIds={0,1,2} 传对（obs 中 globalId∈{0,1,2} 置 isHighPrecision，99 不置）
TEST(ScanPipelineTest, SeedBeforeAnyFuse) {
    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanConfig cfg = baseCfg();
    for (int i = 0; i < 3; ++i) {
        calib::MarkerCloudPoint p{};
        p.x = 1.0 * i;
        p.y = 2.0 * i;
        p.z = 3.0 * i;
        cfg.existingMarkers.push_back(p);
    }
    ScanPipeline pipe(cfg);
    FakeSeedMarkerFuse mf;
    pipe.attachRing(ring, 64);
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
    pipe.attachTestFuseAdapters(&mf);
    ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);

    writeFrames(ring, 0, 2);
    ASSERT_TRUE(pipe.start().success);
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= 2; }));
    pipe.stop();

    ASSERT_EQ(mf.seedCalls.load(), 1);                      // seed 恰一次
    ASSERT_GE(mf.fuseCalls.load(), 1);
    EXPECT_GT(mf.firstFuseSeq.load(), 0u);
    EXPECT_LT(mf.seedSeq.load(), mf.firstFuseSeq.load());   // seed 先于首 fuse

    auto snap = pipe.obs().snapshot();
    ASSERT_FALSE(snap.obs.empty());
    for (const auto& fo : snap.obs) {
        ASSERT_EQ(fo.markerObs.size(), 4u);
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(fo.markerObs[k].globalId, k);
            EXPECT_TRUE(fo.markerObs[k].isHighPrecision);   // 0/1/2 → hp
        }
        EXPECT_EQ(fo.markerObs[3].globalId, 99);
        EXPECT_FALSE(fo.markerObs[3].isHighPrecision);      // 99 → 非 hp
    }
}

// 3：pause/resume——pause 后 ring 新帧不被处理；resume 续跑；obs/fusion 累积延续
TEST(ScanPipelineTest, PauseResumeKeepsState) {
    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanPipeline pipe(baseCfg());
    FakeSeedMarkerFuse mf;
    pipe.attachRing(ring, 2);                               // 小阈值：restart 后跳最新不重扫
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
    pipe.attachTestFuseAdapters(&mf);
    ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);
    ASSERT_TRUE(pipe.start().success);

    // 逐帧写-等消费（lag 恒小，不触发跳帧）
    for (uint64_t id = 0; id < 3; ++id) {
        writeFrames(ring, id, 1);
        const size_t want = static_cast<size_t>(id) + 1;
        ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= want; }));
    }

    pipe.pause();                                           // lane 停 + drain
    EXPECT_FALSE(pipe.isRunning());
    const size_t stable = waitStableObs(pipe);              // 在飞帧消费完
    ASSERT_EQ(stable, 3u);
    const uint64_t fuseBeforePause = mf.fuseCalls.load();

    writeFrames(ring, 10, 2);                               // pause 期间写入
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(pipe.obs().frameCount(), stable);             // 不被处理（runtime 已停）

    ASSERT_TRUE(pipe.resume().success);                     // restart：跳最新续跑
    EXPECT_TRUE(pipe.isRunning());
    writeFrames(ring, 20, 1);
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() > stable; }));

    auto snap = pipe.obs().snapshot();                      // 累积延续：旧观测保留
    std::set<uint64_t> ids;
    for (const auto& fo : snap.obs) ids.insert(fo.frameId);
    for (uint64_t i = 0; i < 3; ++i) EXPECT_EQ(ids.count(i), 1u);
    EXPECT_GT(mf.fuseCalls.load(), fuseBeforePause);        // 融合累积未清零

    pipe.stop();
    EXPECT_FALSE(pipe.isRunning());
}

// 4：existingMarkers 空 → 不调 seed（fuse 正常跑）
TEST(ScanPipelineTest, ExistingMarkersEmptyNoSeed) {
    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanPipeline pipe(baseCfg());                           // existingMarkers 空
    FakeSeedMarkerFuse mf;
    pipe.attachRing(ring, 64);
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
    pipe.attachTestFuseAdapters(&mf);
    ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);

    writeFrames(ring, 0, 2);
    ASSERT_TRUE(pipe.start().success);
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= 2; }));
    pipe.stop();

    EXPECT_EQ(mf.seedCalls.load(), 0);                      // 不调 seed
    EXPECT_GE(mf.fuseCalls.load(), 2);                      // fuse 正常
}

#ifdef JMW_BUILD_CUDA
// 5a（有 CUDA）：enableLaser=false（A 模式）——带激光块的帧也不碰 laserFuse
TEST(ScanPipelineTest, CudaOffDegradedToAMode) {
    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanConfig cfg = baseCfg();
    cfg.enableLaser = false;
    ScanPipeline pipe(cfg);
    FakeSeedMarkerFuse mf;
    FakeLaserFuse lf;
    pipe.attachRing(ring, 64);
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue(), /*withLaserBlock=*/true));
    pipe.attachTestFuseAdapters(&mf, &lf);
    ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);

    writeFrames(ring, 0, 2);
    ASSERT_TRUE(pipe.start().success);
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= 2; }));
    pipe.stop();

    EXPECT_EQ(lf.calls.load(), 0);                          // A 模式不碰 laserFuse
    EXPECT_GE(mf.fuseCalls.load(), 2);
}
#else
// 5b（无 CUDA）：enableLaser=true 被 configure 强制 false + 一次性 Warning(1603)
TEST(ScanPipelineTest, CudaOffDegradedToAMode) {
    Scanner::infra::EventBus bus;
    std::atomic<int> warns{0};
    bus.subscribe(Scanner::EventType::FaultOccurred, [&](const Scanner::Event& e) {
        if (e.param1 == 1603) ++warns;
    });

    SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
    ScanConfig cfg = baseCfg();
    cfg.enableLaser = true;                                 // 会被强制关
    ScanPipeline pipe(cfg);
    FakeSeedMarkerFuse mf;
    pipe.attachRing(ring, 64);
    pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
    pipe.attachTestFuseAdapters(&mf);

    PipelineDeps deps;
    deps.eventBus = &bus;
    ASSERT_TRUE(pipe.configure(deps).success);              // 降级不失败
    EXPECT_EQ(warns.load(), 1);                             // 一次性上报

    writeFrames(ring, 0, 2);
    ASSERT_TRUE(pipe.start().success);                      // A 模式照常跑
    ASSERT_TRUE(waitUntil([&] { return pipe.obs().frameCount() >= 2; }));
    pipe.stop();
}
#endif

// 6：必备件缺失——生产模式未 attachCalib → configure fail；未 configure → start
//    fail；测试模式未 attachRing → start fail；重复 configure → fail
TEST(ScanPipelineTest, ConfigureMissingDeps) {
    // a) 生产模式（未 attachTestHooks）未 attachCalib → configure fail
    {
        ScanPipeline pipe(baseCfg());
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
    // b) 未 configure 直接 start → fail
    {
        SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
        ScanPipeline pipe(baseCfg());
        pipe.attachRing(ring, 64);
        pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
        EXPECT_FALSE(pipe.start().success);
    }
    // c) 测试模式免标定 configure ok，但未 attachRing → start fail
    {
        ScanPipeline pipe(baseCfg());
        pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
        ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);
        EXPECT_FALSE(pipe.start().success);
    }
    // d) 重复 configure → fail
    {
        SlotRing<EnhancedFrame> ring(kRingSlots, SlotRing<EnhancedFrame>::WriterMode::Overwrite);
        ScanPipeline pipe(baseCfg());
        pipe.attachRing(ring, 64);
        pipe.attachTestHooks(passthroughHooks(&pipe.outputQueue()));
        ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
}
