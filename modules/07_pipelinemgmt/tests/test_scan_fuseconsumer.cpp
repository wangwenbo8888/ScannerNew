// ============================================================================
// test_scan_fuseconsumer.cpp — FuseConsumer（队列消费→双融合→渲染节流→攒观测）
// 全假注入：假 marker/laser 融合、假场景推送、假事件上报、假激光下载函数；
// 队列用真 FrameResultQueue 预填帧（消费源契约即队列契约）。
// 覆盖：按序消费+观测累积 / 激光路径开关 / 高精度映射 / 停止排空 / 渲染节流（含首帧）/
//       空转超时退出 / 激光缓存降级一次性上报 / 单帧异常兜底续跑 / 依赖缺失与重复启动失败
// ============================================================================
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pipelines/scan/FuseConsumer.h"

using namespace Scanner::pipeline;
using Scanner::pipeline::sched::FrameResultQueue;

namespace {

// —— 假标记点融合：记录每次 fuse 的 markers/R/T；fused 为稳定存储（渲染句柄目标）——
struct FakeMarkerFuse : IMarkerFuse {
    struct Call {
        std::vector<calib::MarkerPoint3D> markers;
        double R[9] = {};
        double T[3] = {};
    };
    std::vector<Call> calls;
    std::vector<calib::MarkerCloudPoint> fused;

    void fuse(const std::vector<calib::MarkerPoint3D>& m,
              const double R[9], const double T[3]) override {
        Call c;
        c.markers = m;
        for (int i = 0; i < 9; ++i) c.R[i] = R[i];
        for (int i = 0; i < 3; ++i) c.T[i] = T[i];
        calls.push_back(std::move(c));
    }
    const std::vector<calib::MarkerCloudPoint>& fusedPoints() const override { return fused; }
};

// —— 异常注入 marker 融合：第 throwOn 帧（1 基）fuse 抛异常，其余转发 inner ——
struct ThrowingMarkerFuse : IMarkerFuse {
    IMarkerFuse* inner = nullptr;             // 实际干活（可空）
    uint64_t throwOn = 0;                     // 0=不抛
    uint64_t seen = 0;

    void fuse(const std::vector<calib::MarkerPoint3D>& m,
              const double R[9], const double T[3]) override {
        ++seen;
        if (seen == throwOn) throw std::runtime_error("boom");
        if (inner) inner->fuse(m, R, T);
    }
    const std::vector<calib::MarkerCloudPoint>& fusedPoints() const override {
        static const std::vector<calib::MarkerCloudPoint> kEmpty;
        return inner ? inner->fusedPoints() : kEmpty;
    }
};

#ifdef JMW_BUILD_CUDA
// —— 假激光融合：只记录调用与块身份/R/T（不触 GPU）——
struct FakeLaserFuse : ILaserFuse {
    int calls = 0;
    const GpuPointCloudBlock* lastBlock = nullptr;
    double lastR[9] = {};
    double lastT[3] = {};

    void fuse(const GpuPointCloudBlock& b,
              const double R[9], const double T[3]) override {
        ++calls;
        lastBlock = &b;
        for (int i = 0; i < 9; ++i) lastR[i] = R[i];
        for (int i = 0; i < 3; ++i) lastT[i] = T[i];
    }
};
#endif

// —— 假场景推送：计数 + 记录 CloudViewHandle；mf 可选注入（push 时刻 fuse 已调
//    次数 = 帧序 1 基，fuse 与 push 同在消费线程 → 序确定，可断言"哪帧被推"）——
struct FakeSceneFeed : ISceneFeed {
    int cloudPushes = 0;
    std::vector<CloudViewHandle> handles;
    const FakeMarkerFuse* mf = nullptr;       // 注入后记录 push 时帧序
    std::vector<size_t> pushAtFrame;          // 每次 push 对应的帧序（1 基）

    void pushPostureView(const Scanner::Pose&, int,
                         const std::vector<uint8_t>&) override {}
    void pushCloudSnapshot(CloudViewHandle cloud) override {
        ++cloudPushes;
        handles.push_back(cloud);
        if (mf) pushAtFrame.push_back(mf->calls.size());
    }
    void notifyFreeze(bool) override {}
};

// —— 假事件上报：记录 quality/code ——
struct FakeSink : PipelineEventSink {
    std::vector<Scanner::QualityFlag> qualities;
    std::vector<int32_t> codes;

    void report(Scanner::QualityFlag q, int32_t code, const std::string&) override {
        qualities.push_back(q);
        codes.push_back(code);
    }
};

// 帧构造：frameId 可辨识的 R/T（其余 identity）+ n 个标记点
FrameResult makeFrame(uint64_t id, int nMarkers) {
    FrameResult fr;
    fr.frameId = id;
    fr.timestamp = 10 * id;
    fr.temperature = 25.0;
    fr.R[0] = 1.0 * id;
    fr.R[4] = 2.0 * id;
    fr.T[0] = -0.5 * id;
    fr.T[2] = 0.25 * id;
    for (int i = 0; i < nMarkers; ++i) {
        calib::MarkerPoint3D m{};
        m.x = 0.1 * id + i;
        m.y = 0.2 * id - i;
        m.z = 0.3 * id + 10 * i;
        m.nz = 1.0;
        m.globalId = static_cast<int>(100 * id + i);
        fr.markers.push_back(m);
    }
    return fr;
}

} // namespace

// 1：3 帧按序消费——markerFuse 收 3 次（帧序 markers/R/T 逐帧一致）、
//    obs.frameCount==3、markerObs 字段（xyz/globalId/isHighPrecision=false）一致
TEST(FuseConsumerTest, ConsumeInOrderAndObsAccum) {
    FrameResultQueue<FrameResult> q(8);
    FrameObsAccumulator obs(1 << 20);
    FakeMarkerFuse mf;
    const std::vector<FrameResult> frames = {makeFrame(1, 2), makeFrame(2, 1), makeFrame(3, 3)};
    for (const auto& f : frames) q.push(f);

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.obs = &obs;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.join();                                   // join 含置停（drain 后退出）

    EXPECT_EQ(c.consumed(), 3u);
    ASSERT_EQ(mf.calls.size(), 3u);
    ASSERT_EQ(obs.frameCount(), 3u);
    for (int i = 0; i < 3; ++i) {
        const auto& call = mf.calls[i];
        ASSERT_EQ(call.markers.size(), frames[i].markers.size());
        for (size_t k = 0; k < frames[i].markers.size(); ++k) {
            EXPECT_DOUBLE_EQ(call.markers[k].x, frames[i].markers[k].x);
            EXPECT_DOUBLE_EQ(call.markers[k].z, frames[i].markers[k].z);
            EXPECT_EQ(call.markers[k].globalId, frames[i].markers[k].globalId);
        }
        for (int k = 0; k < 9; ++k) EXPECT_DOUBLE_EQ(call.R[k], frames[i].R[k]);
        for (int k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(call.T[k], frames[i].T[k]);
    }

    auto snap = obs.snapshot();
    ASSERT_EQ(snap.obs.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        const auto& fo = snap.obs[i];
        EXPECT_EQ(fo.frameId, frames[i].frameId);
        for (int k = 0; k < 9; ++k) EXPECT_DOUBLE_EQ(fo.R_init[k], frames[i].R[k]);
        for (int k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(fo.t_init[k], frames[i].T[k]);
        ASSERT_EQ(fo.markerObs.size(), frames[i].markers.size());
        for (size_t k = 0; k < fo.markerObs.size(); ++k) {
            EXPECT_DOUBLE_EQ(fo.markerObs[k].xyz[0], frames[i].markers[k].x);
            EXPECT_DOUBLE_EQ(fo.markerObs[k].xyz[1], frames[i].markers[k].y);
            EXPECT_DOUBLE_EQ(fo.markerObs[k].xyz[2], frames[i].markers[k].z);
            EXPECT_EQ(fo.markerObs[k].globalId, frames[i].markers[k].globalId);
            EXPECT_FALSE(fo.markerObs[k].isHighPrecision);   // 未注入集合 → 恒 false
        }
    }
}

#ifdef JMW_BUILD_CUDA
// 2：B 模式带 laser 块 → laserFuse 被调（块/R/T 对）+ 假下载入缓存槽；
//    A 模式（laserFuse=null）带/不带块均不崩且 slot=kNoLaserSlot
TEST(FuseConsumerTest, LaserPathOnlyWhenEnabled) {
    // —— B 模式 ——
    FrameResultQueue<FrameResult> q1(8);
    FrameObsAccumulator obs1(1 << 20);
    FakeMarkerFuse mf1;
    FakeLaserFuse lf;
    const std::vector<float> fakePts = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    int downloadCalls = 0;
    auto dl = [&](const GpuPointCloudBlock& b) {
        ++downloadCalls;
        EXPECT_EQ(b.frameId, 5u);
        return fakePts;
    };
    auto fr = makeFrame(5, 1);
    auto blk = std::make_shared<GpuPointCloudBlock>();   // 空 GpuMat=假块（不触驱动）
    blk->count = 2;
    blk->frameId = 5;
    blk->slotId = 0;
    fr.laser = blk;
    q1.push(fr);

    FuseConsumer::Deps d;
    d.queue = &q1;
    d.markerFuse = &mf1;
    d.laserFuse = &lf;
    d.obs = &obs1;
    d.laserDownload = dl;
    FuseConsumer c1(d);
    ASSERT_TRUE(c1.start().success);
    c1.join();

    ASSERT_EQ(lf.calls, 1);
    EXPECT_EQ(lf.lastBlock, blk.get());
    for (int k = 0; k < 9; ++k) EXPECT_DOUBLE_EQ(lf.lastR[k], fr.R[k]);
    for (int k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(lf.lastT[k], fr.T[k]);
    EXPECT_EQ(downloadCalls, 1);
    auto snap = obs1.snapshot();
    ASSERT_EQ(snap.obs.size(), 1u);
    ASSERT_NE(snap.obs[0].laserCacheSlot, FrameObs::kNoLaserSlot);
    ASSERT_LT(snap.obs[0].laserCacheSlot, snap.laserFrames.size());
    EXPECT_EQ(snap.laserFrames[snap.obs[0].laserCacheSlot], fakePts);

    // —— A 模式：laserFuse=null ——
    FrameResultQueue<FrameResult> q2(8);
    FrameObsAccumulator obs2(1 << 20);
    FakeMarkerFuse mf2;
    q2.push(makeFrame(6, 1));                             // 无激光块
    auto fb = makeFrame(7, 1);
    fb.laser = std::make_shared<GpuPointCloudBlock>();    // 块在但融合关（防御）
    q2.push(fb);
    FuseConsumer::Deps d2;
    d2.queue = &q2;
    d2.markerFuse = &mf2;
    d2.obs = &obs2;
    FuseConsumer c2(d2);
    ASSERT_TRUE(c2.start().success);
    c2.join();                                             // 不崩
    EXPECT_EQ(c2.consumed(), 2u);
    auto snap2 = obs2.snapshot();
    ASSERT_EQ(snap2.obs.size(), 2u);
    EXPECT_EQ(snap2.obs[0].laserCacheSlot, FrameObs::kNoLaserSlot);
    EXPECT_EQ(snap2.obs[1].laserCacheSlot, FrameObs::kNoLaserSlot);
    EXPECT_TRUE(snap2.laserFrames.empty());
}
#endif

// 3：highPrecisionGlobalIds={2} → globalId==2 的 markerObs isHighPrecision=true 其余 false
TEST(FuseConsumerTest, HighPrecisionMapping) {
    FrameResultQueue<FrameResult> q(4);
    FrameObsAccumulator obs(1 << 20);
    FakeMarkerFuse mf;
    auto fr = makeFrame(1, 3);
    fr.markers[0].globalId = 1;
    fr.markers[1].globalId = 2;
    fr.markers[2].globalId = 3;
    q.push(fr);

    std::vector<int> hp = {2};
    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.obs = &obs;
    d.highPrecisionGlobalIds = &hp;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.join();

    auto snap = obs.snapshot();
    ASSERT_EQ(snap.obs.size(), 1u);
    ASSERT_EQ(snap.obs[0].markerObs.size(), 3u);
    EXPECT_FALSE(snap.obs[0].markerObs[0].isHighPrecision);
    EXPECT_TRUE(snap.obs[0].markerObs[1].isHighPrecision);
    EXPECT_FALSE(snap.obs[0].markerObs[2].isHighPrecision);
}

// 4：队列 10 帧，start 后立即 requestStop → join 后 consumed==10（排空再退）
TEST(FuseConsumerTest, DrainOnStop) {
    FrameResultQueue<FrameResult> q(16);
    FrameObsAccumulator obs(1 << 20);
    FakeMarkerFuse mf;
    for (uint64_t i = 1; i <= 10; ++i) q.push(makeFrame(i, 1));

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.obs = &obs;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.requestStop();                            // 立即置停：已有帧仍须消费完
    c.join();
    EXPECT_EQ(c.consumed(), 10u);
    EXPECT_EQ(obs.frameCount(), 10u);
}

// 5：throttle=3 → 6 帧后 sceneFeed 收到 2 次 pushCloudSnapshot（第 1、4 帧——
//    首帧即推，此后每 N 帧再推），hostMarker 指向 markerFuse->fusedPoints()、
//    deviceLaser=nullptr
TEST(FuseConsumerTest, RenderThrottle) {
    FrameResultQueue<FrameResult> q(16);
    FrameObsAccumulator obs(1 << 20);
    FakeMarkerFuse mf;
    mf.fused.push_back(calib::MarkerCloudPoint{});   // 非空云（句柄指向稳定存储）
    FakeSceneFeed sf;
    sf.mf = &mf;                                     // 记录 push 时刻帧序
    for (uint64_t i = 1; i <= 6; ++i) q.push(makeFrame(i, 0));

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.obs = &obs;
    d.sceneFeed = &sf;
    d.renderThrottleFrames = 3;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.join();

    EXPECT_EQ(c.consumed(), 6u);
    EXPECT_EQ(sf.cloudPushes, 2);
    ASSERT_EQ(sf.pushAtFrame.size(), 2u);
    EXPECT_EQ(sf.pushAtFrame[0], 1u);                // 首帧被推（前值 0 % N == 0）
    EXPECT_EQ(sf.pushAtFrame[1], 4u);                // 之后第 N+1=4 帧
    ASSERT_EQ(sf.handles.size(), 2u);
    EXPECT_EQ(sf.handles[0].hostMarker, static_cast<const void*>(&mf.fused));
    EXPECT_EQ(sf.handles[0].deviceLaser, nullptr);
    EXPECT_EQ(sf.handles[1].hostMarker, static_cast<const void*>(&mf.fused));
    EXPECT_EQ(sf.handles[1].deviceLaser, nullptr);
}

// 6：空队列 start→sleep 200ms→requestStop→join——不空转烧满（粗断言：consumed==0 且 join 返回）
TEST(FuseConsumerTest, PopTimeoutIdleNoSpin) {
    FrameResultQueue<FrameResult> q(4);
    FrameObsAccumulator obs(1024);
    FakeMarkerFuse mf;

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.obs = &obs;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // ≥2 个 pop 超时周期
    c.requestStop();
    c.join();                                   // 100ms 超时周期内返回
    EXPECT_EQ(c.consumed(), 0u);
    EXPECT_EQ(obs.frameCount(), 0u);
}

// 9：第 2 帧 markerFuse 抛异常 → 丢帧续跑：线程不崩（join 正常返回）、
//    consumed==3（出队即计数）、obs 只含第 1/3 帧、sink 一次性 Fault(1602)
TEST(FuseConsumerTest, FrameExceptionDroppedAndContinue) {
    FrameResultQueue<FrameResult> q(8);
    FrameObsAccumulator obs(1 << 20);
    FakeMarkerFuse mf;
    ThrowingMarkerFuse tf;
    tf.inner = &mf;
    tf.throwOn = 2;
    FakeSink sk;
    for (uint64_t i = 1; i <= 3; ++i) q.push(makeFrame(i, 1));

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &tf;
    d.obs = &obs;
    d.sink = &sk;
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.join();                                       // 不崩不退：join 正常返回

    EXPECT_EQ(c.consumed(), 3u);                    // 异常帧也计数（出队即消费）
    EXPECT_EQ(mf.calls.size(), 2u);                 // 第 2 帧未达 inner
    auto snap = obs.snapshot();
    ASSERT_EQ(snap.obs.size(), 2u);                 // 第 2 帧被丢
    EXPECT_EQ(snap.obs[0].frameId, 1u);
    EXPECT_EQ(snap.obs[1].frameId, 3u);             // 第 3 帧续跑
    ASSERT_EQ(sk.qualities.size(), 1u);
    EXPECT_EQ(sk.qualities[0], Scanner::QualityFlag::Fault);
    EXPECT_EQ(sk.codes[0], 1602);
}

#ifdef JMW_BUILD_CUDA
// 7：激光缓存预算 12 字节（恰容 1 帧 3 点）→ 第 2 帧超限降级 → sink 一次性 Degraded 上报
TEST(FuseConsumerTest, LaserCacheDegradedReportsOnce) {
    FrameResultQueue<FrameResult> q(8);
    FrameObsAccumulator obs(12);
    FakeMarkerFuse mf;
    FakeLaserFuse lf;
    FakeSink sk;
    auto mk = [](uint64_t id) {
        auto f = makeFrame(id, 0);
        auto b = std::make_shared<GpuPointCloudBlock>();
        b->count = 1;
        b->frameId = id;
        f.laser = b;
        return f;
    };
    q.push(mk(1));
    q.push(mk(2));
    q.push(mk(3));

    FuseConsumer::Deps d;
    d.queue = &q;
    d.markerFuse = &mf;
    d.laserFuse = &lf;
    d.obs = &obs;
    d.sink = &sk;
    d.laserDownload = [](const GpuPointCloudBlock&) { return std::vector<float>(3, 1.f); };
    FuseConsumer c(d);
    ASSERT_TRUE(c.start().success);
    c.join();

    EXPECT_EQ(c.consumed(), 3u);
    EXPECT_TRUE(obs.degradedLaser());
    ASSERT_EQ(sk.qualities.size(), 1u);         // 只报一次
    EXPECT_EQ(sk.qualities[0], Scanner::QualityFlag::Degraded);
}
#endif

// 8：依赖缺失 / 重复启动 → start 失败
TEST(FuseConsumerTest, StartFailsOnMissingDeps) {
    FrameResultQueue<FrameResult> q(4);
    FrameObsAccumulator obs(1024);
    FakeMarkerFuse mf;
    {
        FuseConsumer::Deps d;                   // queue 缺失
        d.markerFuse = &mf;
        d.obs = &obs;
        FuseConsumer c(d);
        EXPECT_FALSE(c.start().success);
    }
    {
        FuseConsumer::Deps d;                   // obs 缺失
        d.queue = &q;
        d.markerFuse = &mf;
        FuseConsumer c(d);
        EXPECT_FALSE(c.start().success);
    }
    {
        FuseConsumer::Deps d;                   // markerFuse 缺失
        d.queue = &q;
        d.obs = &obs;
        FuseConsumer c(d);
        EXPECT_FALSE(c.start().success);
    }
    {
        FuseConsumer::Deps d;                   // 依赖齐 → 启动成功；重复启动失败
        d.queue = &q;
        d.markerFuse = &mf;
        d.obs = &obs;
        FuseConsumer c(d);
        EXPECT_TRUE(c.start().success);
        EXPECT_FALSE(c.start().success);
        c.requestStop();
        c.join();
    }
}
