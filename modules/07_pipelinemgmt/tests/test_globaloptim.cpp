// ============================================================================
// test_globaloptim.cpp — D 全局优化对象（GBA+重融合成对 / 降级兜底 / 冻结推送）
//
// 合成小场景思路复用 calib_synthetic.h（真值位姿 + 共视 marker 观测 + 已知累积
// 漂移注入 R_init）。真算子路径真跑：GBA/Ceres（CPU）+ MarkerCloudFuseCPU（CPU）；
// 激光重放与 GBA 失败路径用注入假实现（ILaserReplayFuse 工厂 / GbaFn）——免 GPU
// 数据语义依赖、免故障真造（同 test_scan_fuseconsumer 注入式风格）。
// 事件断言走真 EventBus（EventBusEventSink 发布 FaultOccurred，param1=事件码）。
// 8 LaserReplayFailDegrades（I1）：假工厂部分帧 fuse 返回 false → quality=Degraded、
//   sink 收 1609 恰一次、run 返回 degraded 而非 fail。
// ============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

#include "base/EventBus.h"
#include "pipelines/globaloptim/GlobalOptimObject.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/ScanTypes.h"
#include "scanning/global_optim/global_ba_cpu.h"

using namespace Scanner::pipeline;
using Scanner::infra::EventBus;

namespace {

// ============================================================================
// 合成小场景：帧 0 恒等锚，帧 i 真值位姿 = 绕 z 轴 i×0.4° + 平移；
// 初值注入随帧累积漂移（额外旋转 i×driftDeg + 平移 i×driftMm）。
// GT 点手工布点：分量小数居体素中部（默认 0.5mm 栅格）、彼此远离 >5mm
// ——重融合去重计数对拍稳定（无噪观测收敛后同点同体素）。
// ============================================================================
struct ScanScene {
    std::vector<cv::Point3d> pts;          // GT 全局点（全帧共视前 ptsPerFrame 个）
    std::vector<cv::Matx33d> Rgt, Rinit;   // 真值位姿 / 带漂移初值
    std::vector<cv::Vec3d> tgt, tinit;
    int nFrames = 0, ptsPerFrame = 0;
};

cv::Matx33d rotZ(double deg) {
    const double a = deg * CV_PI / 180.0;
    return cv::Matx33d(std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1);
}

ScanScene makeScanScene(int nFrames, int ptsPerFrame, double driftDeg, double driftMm) {
    static const std::vector<cv::Point3d> kPool = {
        {10.3, -20.7, 35.1}, {45.9, 12.2, -8.4}, {-33.6, 55.8, 22.9}, {4.7, 4.1, 60.3},
        {-58.2, -41.3, -15.7}, {27.4, 68.6, 44.8}, {72.1, -63.5, 5.6}, {-12.8, 25.4, -42.9}};
    ScanScene s;
    s.nFrames = nFrames;
    s.ptsPerFrame = ptsPerFrame;
    s.pts.assign(kPool.begin(), kPool.begin() + ptsPerFrame);
    for (int i = 0; i < nFrames; ++i) {
        const cv::Matx33d R = rotZ(i * 0.4);                 // 真值：每帧 0.4°
        const cv::Vec3d t(i * 0.6, i * 0.2, 0.0);
        s.Rgt.push_back(R);
        s.tgt.push_back(t);
        s.Rinit.push_back(rotZ(i * driftDeg) * R);           // 累积漂移：i×driftDeg
        s.tinit.push_back(t + cv::Vec3d(i * driftMm, 0.1 * i, 0.0));
    }
    return s;
}

/// 观测压栈：markerObs = Rgtᵀ(P − tgt)（无噪真值投影）；R_init/t_init 注入漂移初值。
/// laserPerFrame 可选（下标=帧号）。
void pushObs(FrameObsAccumulator& acc, const ScanScene& s,
             const std::vector<std::vector<float>>& laserPerFrame = {}) {
    for (int i = 0; i < s.nFrames; ++i) {
        FrameObs fo;
        fo.frameId = static_cast<uint64_t>(i);
        const auto& Rg = s.Rgt[i];             // 观测生成：真值位姿
        const auto& tg = s.tgt[i];
        for (int r = 0; r < 3; ++r)            // 初值：带漂移
            for (int c = 0; c < 3; ++c) fo.R_init[3 * r + c] = s.Rinit[i](r, c);
        for (int k = 0; k < 3; ++k) fo.t_init[k] = s.tinit[i](k);
        for (int k = 0; k < s.ptsPerFrame; ++k) {
            const cv::Vec3d X(s.pts[k].x, s.pts[k].y, s.pts[k].z);
            const cv::Vec3d local = Rg.t() * (X - tg);
            MarkerObs mo;
            mo.xyz[0] = local(0); mo.xyz[1] = local(1); mo.xyz[2] = local(2);
            mo.globalId = k;
            fo.markerObs.push_back(mo);
        }
        std::vector<float> laser;
        if (i < static_cast<int>(laserPerFrame.size())) laser = laserPerFrame[i];
        acc.push(std::move(fo), laser);
    }
}

/// 对某 globalId 的全部观测 local x 加偏置（模拟该点观测系统性偏差）
void biasPointObs(FrameObsAccumulator& acc, int globalId, double bias) {
    auto snap = acc.snapshot();
    acc.clear();
    for (auto& fo : snap.obs)
        for (auto& mo : fo.markerObs)
            if (mo.globalId == globalId) mo.xyz[0] += bias;
    for (size_t i = 0; i < snap.obs.size(); ++i) {
        std::vector<float> laser;
        if (snap.obs[i].laserCacheSlot != FrameObs::kNoLaserSlot)
            laser = snap.laserFrames[snap.obs[i].laserCacheSlot];
        acc.push(std::move(snap.obs[i]), laser);
    }
}

double relAngleDeg(const cv::Matx33d& A, const cv::Matx33d& B) {   // 相对旋转角
    const cv::Matx33d D = A * B.t();
    double c = (D(0, 0) + D(1, 1) + D(2, 2) - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / CV_PI;
}

cv::Matx33d toMatx(const Scanner::Pose& p) {
    return cv::Matx33d(p.R[0], p.R[1], p.R[2], p.R[3], p.R[4], p.R[5], p.R[6], p.R[7], p.R[8]);
}

// —— 假场景推送：记录冻结序列与修正点云句柄 ——
struct FakeSceneFeed : ISceneFeed {
    std::vector<bool> freezes;
    int cloudPushes = 0;
    std::vector<CloudViewHandle> handles;

    void pushPostureView(const Scanner::Pose&, int, const std::vector<uint8_t>&) override {}
    void pushCloudSnapshot(CloudViewHandle cloud) override {
        ++cloudPushes;
        handles.push_back(cloud);
    }
    void notifyFreeze(bool frozen) override { freezes.push_back(frozen); }
};

// —— 假点云仓库：计数 + 记录 tag ——
struct FakeCloudRepo : ICloudRepoWriter {
    int writes = 0;
    std::string lastTag;
    bool write(const std::string& tag) override {
        ++writes;
        lastTag = tag;
        return true;
    }
};

#ifdef JMW_BUILD_CUDA
// —— 假激光重放：工厂每次新建实例、写进共享日志（记录 xyz/R/T，不触 GPU）；
//    failOn=第 N 次（1 基）fuse 返回 false（0=全成功）——I1 部分帧失败注入 ——
struct FakeLaserReplay : ILaserReplayFuse {
    struct Call {
        std::vector<float> xyz;
        double R[9] = {};
        double T[3] = {};
    };
    std::shared_ptr<std::vector<Call>> log;          // 多实例共享（工厂每 run 新建）
    uint64_t failOn = 0;
    uint64_t seen = 0;
    explicit FakeLaserReplay(std::shared_ptr<std::vector<Call>> l,
                             uint64_t fail = 0)
        : log(std::move(l)), failOn(fail) {}

    bool fuse(const std::vector<float>& xyz, const double R[9], const double T[3]) override {
        ++seen;
        Call c;
        c.xyz = xyz;
        for (int i = 0; i < 9; ++i) c.R[i] = R[i];
        for (int i = 0; i < 3; ++i) c.T[i] = T[i];
        log->push_back(std::move(c));
        return seen != failOn;
    }
};

std::unique_ptr<ILaserReplayFuse> makeFakeLaserReplay(
    const std::shared_ptr<std::vector<FakeLaserReplay::Call>>& log,
    uint64_t failOn = 0) {
    return std::unique_ptr<ILaserReplayFuse>(new FakeLaserReplay(log, failOn));
}
#endif

/// 事件采集：订阅 FaultOccurred，记录 param1（事件码）
struct EventCollector {
    EventBus bus;
    std::vector<int64_t> codes;
    Scanner::infra::SubscriberId sub = 0;

    EventCollector() {
        sub = bus.subscribe(Scanner::EventType::FaultOccurred, [this](const Scanner::Event& e) {
            codes.push_back(e.param1);
        });
    }
    ~EventCollector() { bus.unsubscribe(sub); }
    bool hasCode(int32_t code) const {
        return std::find(codes.begin(), codes.end(), static_cast<int64_t>(code)) != codes.end();
    }
    size_t countOf(int32_t code) const {
        return static_cast<size_t>(std::count(codes.begin(), codes.end(),
                                              static_cast<int64_t>(code)));
    }
};

} // namespace

// ============================================================================
// 1：核心数值——3 帧场景（真值位姿 + 全共视 4 marker + 累积漂移注入 R_init）→
//    run 后修正位姿角差 < 初值角差（对拍真值）；重融合 marker 数 == 去重后点数
// ============================================================================
TEST(GlobalOptimObjectTest, GbaCorrectsDriftAndRefuses) {
    const ScanScene scene = makeScanScene(3, 4, 1.5 /*driftDeg/帧*/, 0.8 /*driftMm/帧*/);
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene);

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);

    int lastPct = -1;
    ProgressCb cb = [&](int p, const std::string&) { lastPct = std::max(lastPct, p); };
    CancelToken cancel;
    auto res = obj.run(acc, cb, cancel);
    ASSERT_TRUE(res.success) << res.message;

    const auto& out = obj.output();
    EXPECT_TRUE(out.gbaSuccess);
    EXPECT_EQ(out.frameCount, 3u);
    EXPECT_EQ(lastPct, 100);                       // 进度走到终点
    ASSERT_EQ(out.poses.size(), 3u);
    for (int i = 1; i < 3; ++i) {                  // 帧 0 恒等锚（无漂移，跳过）
        const double initAng = relAngleDeg(scene.Rinit[i], scene.Rgt[i]);
        const double corrAng = relAngleDeg(toMatx(out.poses[i]), scene.Rgt[i]);
        EXPECT_GT(initAng, 1.0) << "漂移确已注入（帧 " << i << "）";
        EXPECT_LT(corrAng, initAng) << "修正应优于初值（帧 " << i << "）";
        EXPECT_LT(corrAng, 0.2) << "无噪场景应收敛到真值附近（帧 " << i << "）";
    }
    // GBA 输出点云重融合 marker 数 == 去重后 marker 数（4 点全共视 → 4）
    EXPECT_EQ(out.markerCloud.size(), 4u);
    EXPECT_EQ(out.quality, Scanner::QualityFlag::Normal);
}

// ============================================================================
// 2：软先验锚定（T24 语义端到端）——1 点高精度先验（真值位置）+ 该点观测一致
//    偏置 0.03mm → 有先验：优化后几乎不动（<0.01mm）；无先验对照：随偏置漂走
// ============================================================================
TEST(GlobalOptimObjectTest, ExistingPriorAnchors) {
    const ScanScene scene = makeScanScene(3, 5, 1.0, 0.5);

    // 有先验
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene);
    biasPointObs(acc, 2, 0.03);

    GlobalOptimObject withPrior;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    PipelineDeps deps;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(withPrior.configure(deps).success);
    withPrior.setExistingPrior({2}, {scene.pts[2].x, scene.pts[2].y, scene.pts[2].z},
                               {} /*σ 空=09 默认 0.001*/);
    CancelToken cancel;
    auto res = withPrior.run(acc, nullptr, cancel);
    ASSERT_TRUE(res.success) << res.message;

    double priorDist = -1.0;
    for (const auto& m : withPrior.output().gbaMarkers)
        if (m.globalId == 2)
            priorDist = cv::norm(m.X - scene.pts[2]);
    ASSERT_GE(priorDist, 0.0) << "先验点应在 GBA 输出中";
    EXPECT_LT(priorDist, 0.01) << "先验点几乎不动（距真值 " << priorDist << "mm）";

    // 无先验对照（同一偏置数据）：点随偏置漂走
    GlobalOptimObject noPrior;
    ASSERT_TRUE(noPrior.configure(deps).success);
    auto res2 = noPrior.run(acc, nullptr, cancel);
    ASSERT_TRUE(res2.success) << res2.message;
    double ctrlDist = -1.0;
    for (const auto& m : noPrior.output().gbaMarkers)
        if (m.globalId == 2)
            ctrlDist = cv::norm(m.X - scene.pts[2]);
    ASSERT_GE(ctrlDist, 0.0);
    EXPECT_GT(ctrlDist, 0.02) << "无先验时点应随偏置漂走（距真值 " << ctrlDist << "mm）";
    EXPECT_LT(priorDist, ctrlDist);
}

#ifdef JMW_BUILD_CUDA
// ============================================================================
// 3：激光未降级 → 全帧重放——2 帧激光 slot + 假 LaserFuse 工厂注入 →
//    重放调用次数 = 帧数、xyz 逐帧一致、位姿为修正后（≠初值且收敛）
// ============================================================================
TEST(GlobalOptimObjectTest, LaserNotDegradedReplay) {
    const ScanScene scene = makeScanScene(2, 4, 1.5, 0.8);
    const std::vector<std::vector<float>> laser = {{1, 2, 3, 4, 5, 6},
                                                   {7, 8, 9, 10, 11, 12}};
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene, laser);
    ASSERT_FALSE(acc.degradedLaser());

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    PipelineDeps deps;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);
    auto log = std::make_shared<std::vector<FakeLaserReplay::Call>>();
    obj.attachTestLaserFuseFactory([log] { return makeFakeLaserReplay(log); });

    CancelToken cancel;
    auto res = obj.run(acc, nullptr, cancel);
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_EQ(res.qualityFlag, Scanner::QualityFlag::Normal);

    const auto& out = obj.output();
    EXPECT_TRUE(out.laserReplayed);
    ASSERT_EQ(log->size(), 2u);                     // 重放调用次数 = 帧数
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ((*log)[i].xyz, laser[i]);         // host 激光逐帧一致
        for (int k = 0; k < 9; ++k)
            EXPECT_DOUBLE_EQ((*log)[i].R[k], out.poses[i].R[k]);       // 修正位姿
        for (int k = 0; k < 3; ++k)
            EXPECT_DOUBLE_EQ((*log)[i].T[k], out.poses[i].t[k]);
    }
    // "修正后"而非初值：帧 1 位姿与初值有实质差异，且对拍真值更近
    bool differs = false;
    for (int k = 0; k < 9; ++k)
        if (out.poses[1].R[k] != scene.Rinit[1](k / 3, k % 3)) differs = true;
    EXPECT_TRUE(differs);
    EXPECT_LT(relAngleDeg(toMatx(out.poses[1]), scene.Rgt[1]),
              relAngleDeg(scene.Rinit[1], scene.Rgt[1]));
}
#endif

#ifdef JMW_BUILD_CUDA
// ============================================================================
// 4：激光缓存降级 → 不重放激光——obs.degradedLaser()==true →
//    假工厂 0 次调用、quality=Degraded、run 返回 degraded
// ============================================================================
TEST(GlobalOptimObjectTest, LaserDegradedSkipsReplay) {
    const ScanScene scene = makeScanScene(2, 4, 1.0, 0.5);
    FrameObsAccumulator acc(12);                   // 预算 12B：仅容 1 帧 3 点激光
    pushObs(acc, scene, {{1, 2, 3}, {4, 5, 6}});   // 第 2 帧超限 → 降级
    ASSERT_TRUE(acc.degradedLaser());

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);
    auto log = std::make_shared<std::vector<FakeLaserReplay::Call>>();
    obj.attachTestLaserFuseFactory([log] { return makeFakeLaserReplay(log); });

    CancelToken cancel;
    auto res = obj.run(acc, nullptr, cancel);
    EXPECT_TRUE(res.success);                      // 降级非失败
    EXPECT_TRUE(res.isDegraded());

    const auto& out = obj.output();
    EXPECT_TRUE(out.laserDegraded);
    EXPECT_FALSE(out.laserReplayed);
    EXPECT_EQ(out.quality, Scanner::QualityFlag::Degraded);
    EXPECT_TRUE(log->empty());                     // 不重放激光
    EXPECT_TRUE(ev.hasCode(1608));                 // 激光未重放降级上报
}
#endif

// ============================================================================
// 5：GBA 失败兜底——注入失败 GbaFn → 沿初值位姿重融合、Fault(1605) 上报、
//    run 返回 degraded（非 fail）
// ============================================================================
TEST(GlobalOptimObjectTest, GbaFailFallback) {
    const ScanScene scene = makeScanScene(2, 4, 1.0, 0.5);
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene);

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);
    obj.attachTestGba([](const calib::GlobalBAInput&, const calib::GlobalBAParams&) {
        calib::GlobalBAResult r;
        r.success = false;
        r.message = "synthetic gba failure";
        return r;
    });

    CancelToken cancel;
    auto res = obj.run(acc, nullptr, cancel);
    EXPECT_TRUE(res.success) << "兜底路径应为 degraded 非 fail";
    EXPECT_TRUE(res.isDegraded());

    const auto& out = obj.output();
    EXPECT_FALSE(out.gbaSuccess);
    ASSERT_EQ(out.poses.size(), 2u);
    for (int i = 0; i < 2; ++i)                    // 位姿 = 初值（逐元素一致）
        for (int k = 0; k < 9; ++k)
            EXPECT_DOUBLE_EQ(out.poses[i].R[k], scene.Rinit[i](k / 3, k % 3));
    // 沿初值重融合非空——漂移初值下同点观测散落异体素（不去重属如实结果，
    // 仅断非空；GBA 成功路径的去重对拍见 GbaCorrectsDriftAndRefuses）
    EXPECT_GE(out.markerCloud.size(), 4u);
    EXPECT_EQ(out.quality, Scanner::QualityFlag::Degraded);
    EXPECT_TRUE(ev.hasCode(1605));                 // GBA 失败 Fault 上报
    EXPECT_EQ(sf.freezes, (std::vector<bool>{true, false}));   // 冻结/解冻成对
}

// ============================================================================
// 6：冻结序列与点云入库——notifyFreeze(true→false)；cloudRepo.write 恰一次
// ============================================================================
TEST(GlobalOptimObjectTest, FreezeAndRepoTouched) {
    const ScanScene scene = makeScanScene(2, 4, 1.0, 0.5);
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene);

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    PipelineDeps deps;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);

    CancelToken cancel;
    ASSERT_TRUE(obj.run(acc, nullptr, cancel).success);

    EXPECT_EQ(sf.freezes, (std::vector<bool>{true, false}));   // 冻结→解冻
    EXPECT_EQ(repo.writes, 1);                                 // 入库恰一次
    EXPECT_NE(repo.lastTag.find("globaloptim"), std::string::npos);
    ASSERT_EQ(sf.cloudPushes, 1);                              // 修正点云推送一次
    EXPECT_EQ(sf.handles[0].hostMarker,
              static_cast<const void*>(&obj.output().markerCloud));
}

// ============================================================================
// 7：取消尊重——GBA 完成后置取消 → 安全退出（不崩/不推送/不入库/解冻恢复）
// ============================================================================
TEST(GlobalOptimObjectTest, CancelRespected) {
    const ScanScene scene = makeScanScene(2, 4, 1.0, 0.5);
    const std::vector<std::vector<float>> laser = {{1, 2, 3, 4, 5, 6},
                                                   {7, 8, 9, 10, 11, 12}};
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene, laser);

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    PipelineDeps deps;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);
#ifdef JMW_BUILD_CUDA
    auto log = std::make_shared<std::vector<FakeLaserReplay::Call>>();
    obj.attachTestLaserFuseFactory([log] { return makeFakeLaserReplay(log); });
#endif
    CancelToken cancel;
    obj.attachTestGba([&](const calib::GlobalBAInput& in, const calib::GlobalBAParams& p) {
        calib::GlobalBundleAdjustmentCPU op(p);
        auto r = op.Execute(in);
        cancel.cancel();                           // GBA 完成后置取消（检查点触发）
        return r;
    });

    auto res = obj.run(acc, nullptr, cancel);
    EXPECT_FALSE(res.isFault());                   // 安全退出非崩非 Fault
    EXPECT_TRUE(res.isDegraded());

    const auto& out = obj.output();
    EXPECT_TRUE(out.cancelled);
    EXPECT_TRUE(out.markerCloud.empty());          // 未做重融合
    EXPECT_FALSE(out.laserReplayed);
#ifdef JMW_BUILD_CUDA
    EXPECT_TRUE(log->empty());                     // 激光未重放
#endif
    EXPECT_EQ(repo.writes, 0);                     // 未入库
    EXPECT_EQ(sf.cloudPushes, 0);                  // 未推送修正点云
    EXPECT_EQ(sf.freezes, (std::vector<bool>{true, false}));   // 解冻恢复
}

#ifdef JMW_BUILD_CUDA
// ============================================================================
// 8（I1）：激光重放融合失败聚合降级——假工厂第 2 帧返回 false（部分帧失败）→
//    quality=Degraded、sink 收 1609（恰一次）、run 返回 degraded 而非 fail
// ============================================================================
TEST(GlobalOptimObjectTest, LaserReplayFailDegrades) {
    const ScanScene scene = makeScanScene(2, 4, 1.0, 0.5);
    const std::vector<std::vector<float>> laser = {{1, 2, 3, 4, 5, 6},
                                                   {7, 8, 9, 10, 11, 12}};
    FrameObsAccumulator acc(1 << 20);
    pushObs(acc, scene, laser);
    ASSERT_FALSE(acc.degradedLaser());

    GlobalOptimObject obj;
    FakeSceneFeed sf;
    FakeCloudRepo repo;
    EventCollector ev;
    PipelineDeps deps;
    deps.eventBus = &ev.bus;
    deps.sceneFeed = &sf;
    deps.cloudRepo = &repo;
    ASSERT_TRUE(obj.configure(deps).success);
    auto log = std::make_shared<std::vector<FakeLaserReplay::Call>>();
    obj.attachTestLaserFuseFactory([log] { return makeFakeLaserReplay(log, 2); });   // 第 2 帧失败

    CancelToken cancel;
    auto res = obj.run(acc, nullptr, cancel);
    EXPECT_TRUE(res.success) << "部分帧失败应降级非 fail";
    EXPECT_TRUE(res.isDegraded());

    const auto& out = obj.output();
    EXPECT_EQ(out.quality, Scanner::QualityFlag::Degraded);
    EXPECT_TRUE(ev.hasCode(1609));                 // 激光重放失败降级上报
    EXPECT_EQ(ev.countOf(1609), 1u);               // 恰一次（非逐帧刷屏）
    ASSERT_EQ(log->size(), 2u);                    // 两帧都调用了（失败不中断重放）
    EXPECT_TRUE(out.laserReplayed);                // 重放整体已执行（部分失败如实聚合）
}
#endif
