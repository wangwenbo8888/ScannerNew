// ============================================================================
// test_posture_pipeline.cpp — PosturePipeline 对象总装（A 姿态判断 · 假算子链）
// 真算子路径（GPU 前段/标记点链/配准）由 09 单测覆盖——本测试经 attachTestHooks
// 注入假链（eFinalize 假实现里调 table().report，给出 pose_estimate 等效输出），
// 端到端验证生命周期编排：顺序反压消费/激光线帧销毁整周期/集齐 25 自动收口
// （drain 先于 acquisition 停止→continueWithB）/中途 stop 不触发 completion/
// 必备件缺失 fail。
// ============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <future>
#include <initializer_list>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "CycleUnit.h"
#include "pipelines/ISceneFeed.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/posture/PosturePipeline.h"

using namespace Scanner::pipeline;
using Scanner::data::CycleUnit;
using Scanner::data::SlotRing;
using Scanner::Result;

namespace {

constexpr int kTargets = PostureSessionData::kTargetCount;   // 25
// 环槽 32：末块确认（最晚周期 374）后写侧余量 ≤1 必可写完——写侧 join 不挂
constexpr size_t kRingSlots = 32;
// 每目标 15 周期块（streak=3 确认 + 多 lane 乱序余量：块界换序至多吃掉若干
// streak 槽位，15 宽保证确认必落块内）；总周期 25×15=375（任务口径 25*5*3）
constexpr uint64_t kCyclesPerTarget = 15;
constexpr double kMaskRatioThreshold = 0.05;                  // 假链复刻 frame_filter 判定阈

// —— 轮询等待谓词为真（10ms × 1000 = 10s 上限）——
template<typename Pred>
bool waitUntil(Pred pred) {
    for (int i = 0; i < 1000; ++i) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

// —— 目标 i：单位旋转 + T=(i*10,0,0)（间隔 10mm > 5mm 阈值，互不串扰）——
void fillTargets(double t[kTargets][16]) {
    for (int i = 0; i < kTargets; ++i) {
        auto& m = t[i];
        std::fill(m, m + 16, 0.0);
        m[0] = 1.0; m[5] = 1.0; m[10] = 1.0;
        m[3] = i * 10.0;                                        // T.x（第 4 列）
        m[15] = 1.0;
    }
}

// —— 合成周期：isLaser=true → markerL 全黑（maskRatio=0 < 阈=激光线帧）——
std::shared_ptr<CycleUnit> mkCycle(uint64_t id, bool isLaser = false) {
    auto c = std::make_shared<CycleUnit>();
    c->id = id;
    c->temperature = 25.0;
    c->timestamp = id;
    c->markerL = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
    if (!isLaser) c->markerL(cv::Rect(8, 8, 8, 8)) = 255;      // 64/1024=6.25% ≥ 阈
    c->markerR = c->markerL.clone();
    c->laserFrames.resize(2);                                   // 激光管帧"乘客"
    return c;
}

// —— 采集控制假实现：计 stopAcquisition 次数 ——
struct FakeAcq : IAcquisitionControl {
    std::atomic<int> stops{0};
    void stopAcquisition() override { ++stops; }
};

// —— 渲染推送假实现：计 pushPostureView 与 confirmedCount 走向 ——
struct FakeSceneFeed : ISceneFeed {
    std::atomic<int> pushes{0};
    std::atomic<int> maxConfirmed{0};
    std::atomic<bool> sawZero{false};
    std::atomic<bool> sawMarkers{false};
    void pushPostureView(const Scanner::Pose&, int confirmedCount,
                         const std::vector<uint8_t>& markerDetected) override {
        ++pushes;
        int prev = maxConfirmed.load();
        while (confirmedCount > prev &&
               !maxConfirmed.compare_exchange_weak(prev, confirmedCount)) {}
        if (confirmedCount == 0) sawZero = true;
        if (!markerDetected.empty()) sawMarkers = true;
    }
    void pushCloudSnapshot(CloudViewHandle) override {}
    void notifyFreeze(bool) override {}
};

// —— 假链：gpuChain 复刻 frame_filter 判定（maskRatio<阈 → false 销毁整周期，
//    不调 frontReady——走 runtime 兜底提交路径，同 A 模式）；pChain 产占位标记点；
//    eFinalize 给 pose_estimate 等效输出（cycle k 报目标 (k/5)），report 后逐帧推送 ——
PosturePipeline::Hooks fakeHooks(PosturePipeline& pipe, FakeSceneFeed* feed,
                                 std::shared_ptr<std::set<uint64_t>> seen = nullptr) {
    PosturePipeline::Hooks h;
    h.gpuChain = [](sched::GpuSlotService::SlotGuard&,
                    const std::shared_ptr<const CycleUnit>& frame, PostureFront&,
                    std::function<void()>) -> bool {
        if (frame->markerL.empty()) return true;
        const double ratio = static_cast<double>(cv::countNonZero(frame->markerL)) /
                             static_cast<double>(frame->markerL.total());
        return ratio >= kMaskRatioThreshold;                    // false=激光线帧销毁
    };
    h.pChain = [](const std::shared_ptr<const CycleUnit>&, PostureFront&,
                  PostureFrameResult& r) {
        r.positions.assign(4, cv::Point3d(0, 0, 0));
        r.normals.assign(4, cv::Vec3d(0, 0, 1));
        r.ellipseCentersL.assign(4, cv::Point2f(1, 1));
        r.ellipseCentersR.assign(4, cv::Point2f(2, 2));
        return Result::ok();
    };
    h.eFinalize = [&pipe, feed, seen](const std::shared_ptr<const CycleUnit>& frame,
                                      PostureFront&, PostureFrameResult& r,
                                      std::future<Result>& fut) -> Result {
        if (!fut.get().success) return Result::fail("p fail");
        if (seen) seen->insert(frame->id);
        const int t = static_cast<int>((frame->id / kCyclesPerTarget) % kTargets);
        double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double T[3] = {t * 10.0, 0.0, 0.0};
        CycleUnit cycle = *frame;                               // const 共享帧 → 拷贝移入
        pipe.table().report(R, T, std::move(cycle), std::move(r.ellipseCentersL),
                            std::move(r.ellipseCentersR));
        if (feed) {
            Scanner::Pose p;
            std::copy(R, R + 9, p.R);
            std::copy(T, T + 3, p.t);
            p.frameId = frame->id;
            feed->pushPostureView(p, pipe.collectedCount(),
                                  std::vector<uint8_t>(r.positions.size(), 1));
        }
        return Result::ok();
    };
    return h;
}

// 显式逐元素构造（comma-initializer 出函数体即悬空引用，禁用）
static cv::Mat matOf(int rows, int cols, std::initializer_list<double> v) {
    cv::Mat_<double> m(rows, cols);
    int i = 0;
    for (double x : v) m(i / cols, i % cols) = x, ++i;
    return m;
}

PostureInitialParams mkParams() {
    PostureInitialParams p;
    p.K1 = matOf(3, 3, {800, 0, 320, 0, 800, 240, 0, 0, 1});
    p.K2 = matOf(3, 3, {810, 0, 320, 0, 810, 240, 0, 0, 1});
    p.D1 = matOf(5, 1, {0, 0, 0, 0, 0});
    p.D2 = matOf(5, 1, {0, 0, 0, 0, 0});
    p.R1 = cv::Mat::eye(3, 3, CV_64F);
    p.R2 = cv::Mat::eye(3, 3, CV_64F);
    p.P1 = matOf(3, 4, {800, 0, 320, 0, 0, 800, 240, 0, 0, 0, 1, 0});
    p.P2 = matOf(3, 4, {810, 0, 320, 0, 0, 810, 240, 0, 0, 0, 1, 0});
    p.Q = cv::Mat::eye(4, 4, CV_64F);
    p.imageWidth = 640;
    p.imageHeight = 480;
    p.maskRatioThreshold = kMaskRatioThreshold;
    return p;
}

} // namespace

// 1：集齐自动收口——25 目标表（T=(i*10,0,0)），合成 25×15 周期（块内第 3 次确认，
//    余量容忍多 lane 乱序）写入 Backpressure 环 → start → 等 completion（10s）→
//    断言：SessionData 25 条全确认、stopAcquisition 被调、continueWithB 收到数据、
//    isRunning false
TEST(PosturePipelineTest, CollectAllAndAutoClose) {
    double targets[kTargets][16];
    fillTargets(targets);
    SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);

    PosturePipeline pipe;
    pipe.attachRing(ring);
    pipe.attachTargets(targets, kTargets);
    auto seen = std::make_shared<std::set<uint64_t>>();
    FakeAcq acq;
    FakeSceneFeed feed;
    pipe.attachTestHooks(fakeHooks(pipe, &feed, seen));

    PipelineDeps deps;
    deps.acquisition = &acq;
    deps.sceneFeed = &feed;
    ASSERT_TRUE(pipe.configure(deps).success);

    std::promise<PostureSessionData> done;
    auto fut = done.get_future();
    pipe.setCompletionHook([&done](PostureSessionData&& s) {
        done.set_value(std::move(s));
    });

    ASSERT_TRUE(pipe.start().success);
    EXPECT_TRUE(pipe.isRunning());

    std::thread writer([&ring] {
        for (uint64_t i = 0; i < kTargets * kCyclesPerTarget; ++i)
            ring.write(mkCycle(i));
    });

    // 先取状态再 join 写侧：确认必落末块内（≤374 周期）→ 写侧必被消费完；
    // join 先于断言——失败路径下 lanes 未停仍会消费完全部 375 周期，join 不挂
    const auto status = fut.wait_for(std::chrono::seconds(10));
    writer.join();
    ASSERT_EQ(status, std::future_status::ready);

    auto s = fut.get();
    EXPECT_EQ(s.collectedCount, kTargets);
    for (int i = 0; i < kTargets; ++i) EXPECT_TRUE(s.collected[i]) << "pose " << i;
    EXPECT_GE(acq.stops.load(), 1);                 // 收口链：采集停止被调
    EXPECT_FALSE(pipe.isRunning());                 // 自动收口后自停
    EXPECT_GT(feed.pushes.load(), 0);               // 实时推送在跑

    // 确认时刻落在各自 5 周期块内（顺序反压逐周期；确认通常在第 3 次）
    for (int i = 0; i < kTargets; ++i) {
        EXPECT_GE(s.poses[i].cycleId, static_cast<uint64_t>(i * kCyclesPerTarget))
            << "pose " << i;
        EXPECT_LE(s.poses[i].cycleId,
                  static_cast<uint64_t>(i * kCyclesPerTarget + kCyclesPerTarget - 1))
            << "pose " << i;
        EXPECT_EQ(s.poses[i].cycle.id, s.poses[i].cycleId);
    }

    pipe.stop();                                    // 幂等收尾不挂死
    EXPECT_FALSE(pipe.isRunning());
}

// 2：激光线帧销毁整周期——markerL 全黑（maskRatio<阈）的周期 gpuChain 返 false
//    → 不入 eFinalize/ConfirmTable（collectedCount 不为其变化，streak 不受扰）
TEST(PosturePipelineTest, LaserFrameCycleDestroyed) {
    double targets[kTargets][16];
    fillTargets(targets);
    SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);

    PosturePipeline pipe;
    pipe.attachRing(ring);
    pipe.attachTargets(targets, kTargets);
    auto seen = std::make_shared<std::set<uint64_t>>();
    pipe.attachTestHooks(fakeHooks(pipe, nullptr, seen));
    ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);
    ASSERT_TRUE(pipe.start().success);

    // 2 次命中目标 0（streak=2）→ 激光线周期（id=2，必须被销毁）→ 第 3 次命中确认
    ring.write(mkCycle(0));
    ring.write(mkCycle(1));
    ring.write(mkCycle(2, /*isLaser=*/true));
    ring.write(mkCycle(3));

    ASSERT_TRUE(waitUntil([&] {
        return pipe.collectedCount() == 1 && seen->count(3) > 0;
    }));
    pipe.stop();

    EXPECT_EQ(seen->count(2), 0u);                  // 激光线周期未入 eFinalize
    EXPECT_EQ(seen->size(), 3u);                    // 恰 0/1/3 三周期入簿记
    EXPECT_EQ(pipe.collectedCount(), 1);            // 3 次有效命中 → 确认 1 姿态
    // 确认周期 ∈ {0,1,3}：多 lane 乱序下第 3 个处理完的有效命中做确认（2 恒被排除）
    const uint64_t cid = pipe.table().takeSessionData().poses[0].cycleId;
    EXPECT_TRUE(cid == 0u || cid == 1u || cid == 3u) << "cycleId=" << cid;
}

// 3：实时推送——sceneFeed 收到 pushPostureView（confirmedCount 递增 0→1、
//    标志点检出数随帧）
TEST(PosturePipelineTest, LiveViewPushed) {
    double targets[kTargets][16];
    fillTargets(targets);
    SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);

    PosturePipeline pipe;
    pipe.attachRing(ring);
    pipe.attachTargets(targets, kTargets);
    FakeSceneFeed feed;
    pipe.attachTestHooks(fakeHooks(pipe, &feed));

    PipelineDeps deps;
    deps.sceneFeed = &feed;
    ASSERT_TRUE(pipe.configure(deps).success);
    ASSERT_TRUE(pipe.start().success);

    for (uint64_t i = 0; i < 3; ++i) ring.write(mkCycle(i));   // 目标 0 ×3 → 确认
    ASSERT_TRUE(waitUntil([&] {
        return feed.pushes.load() >= 3 && pipe.collectedCount() == 1;
    }));
    pipe.stop();

    EXPECT_TRUE(feed.sawZero.load());               // 确认前推送计数 0
    EXPECT_EQ(feed.maxConfirmed.load(), 1);         // 确认后推送计数 1（递增）
    EXPECT_TRUE(feed.sawMarkers.load());            // 标志点检出数非空
}

// 4：中途 stop——未集齐时 stop() 同步停（不触发 completion、不停采集、不挂死）
TEST(PosturePipelineTest, StopBeforeComplete) {
    double targets[kTargets][16];
    fillTargets(targets);
    SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);

    PosturePipeline pipe;
    pipe.attachRing(ring);
    pipe.attachTargets(targets, kTargets);
    auto seen = std::make_shared<std::set<uint64_t>>();
    FakeAcq acq;
    pipe.attachTestHooks(fakeHooks(pipe, nullptr, seen));

    PipelineDeps deps;
    deps.acquisition = &acq;
    ASSERT_TRUE(pipe.configure(deps).success);
    std::atomic<bool> completed{false};
    pipe.setCompletionHook([&](PostureSessionData&&) { completed = true; });

    ASSERT_TRUE(pipe.start().success);
    ring.write(mkCycle(0));
    ring.write(mkCycle(1));                         // 目标 0 ×2（未确认）
    ASSERT_TRUE(waitUntil([&] { return seen->size() >= 2; }));

    const auto t0 = std::chrono::steady_clock::now();
    pipe.stop();                                    // 同步 stop
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    EXPECT_LT(elapsedMs, 5000);                     // 无挂死

    EXPECT_FALSE(pipe.isRunning());
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_FALSE(completed.load());                 // 不触发 completion
    EXPECT_EQ(acq.stops.load(), 0);                 // 用户 stop 不停采集（01 自管）
    EXPECT_EQ(pipe.collectedCount(), 0);            // 未确认任何姿态
}

// 5：必备件缺失——未 attachTargets / 生产模式未 attachInitialParams → configure
//    fail；未 configure / 未 attachRing → start fail；测试钩子与参数互斥
TEST(PosturePipelineTest, MissingAttachFail) {
    double targets[kTargets][16];
    fillTargets(targets);
    // a) 未 attachTargets → configure fail
    {
        SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);
        PosturePipeline pipe;
        pipe.attachRing(ring);
        pipe.attachTestHooks(fakeHooks(pipe, nullptr));
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
    // b) 生产模式（未 attachTestHooks）未 attachInitialParams → configure fail
    {
        SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);
        PosturePipeline pipe;
        pipe.attachRing(ring);
        pipe.attachTargets(targets, kTargets);
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
    // c) 未 configure → start fail
    {
        SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);
        PosturePipeline pipe;
        pipe.attachRing(ring);
        pipe.attachTargets(targets, kTargets);
        pipe.attachTestHooks(fakeHooks(pipe, nullptr));
        EXPECT_FALSE(pipe.start().success);
    }
    // d) configure ok 但未 attachRing → start fail
    {
        PosturePipeline pipe;
        pipe.attachTargets(targets, kTargets);
        pipe.attachTestHooks(fakeHooks(pipe, nullptr));
        ASSERT_TRUE(pipe.configure(PipelineDeps{}).success);
        EXPECT_FALSE(pipe.start().success);
    }
    // e) attachTestHooks 与 attachInitialParams 互斥 → configure fail
    {
        PosturePipeline pipe;
        pipe.attachTargets(targets, kTargets);
        pipe.attachInitialParams(mkParams());
        pipe.attachTestHooks(fakeHooks(pipe, nullptr));
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
    // f) maskRatioThreshold=0（默认占位=恒真不过滤，激光线帧永不销毁，破坏 D3）
    //    → configure fail；>=1.0 同理 fail
    {
        SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);
        PosturePipeline pipe;
        pipe.attachRing(ring);
        pipe.attachTargets(targets, kTargets);
        PostureInitialParams p0 = mkParams();
        p0.maskRatioThreshold = 0.0;
        pipe.attachInitialParams(p0);
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
    {
        SlotRing<CycleUnit> ring(kRingSlots, SlotRing<CycleUnit>::WriterMode::Backpressure);
        PosturePipeline pipe;
        pipe.attachRing(ring);
        pipe.attachTargets(targets, kTargets);
        PostureInitialParams p1 = mkParams();
        p1.maskRatioThreshold = 1.0;
        pipe.attachInitialParams(p1);
        EXPECT_FALSE(pipe.configure(PipelineDeps{}).success);
    }
}
