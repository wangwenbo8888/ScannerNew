// ============================================================================
// test_scan_chains_build.cpp — ScanChains 编译级冒烟（真算子 + 空图，链路单线程直调）
// 不经 SchedulerRuntime：手动依次 gpuChain（真/假 guard）→ pChain → eFinalize。
// 空图=正常降级：不崩不挂、pChain ok、eFinalize 产 FrameResult（markers 空、
// quality Normal/Degraded）；enableLaser 两路（激光路空表→算子失败/异常被钩子捕获）。
// ============================================================================
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <opencv2/core.hpp>

#include "EnhancedFrame.h"
#include "pipelines/scan/ScanChains.h"
#include "pipelines/scan/ScanTypes.h"
#include "sched/FrameResultQueue.h"
#include "sched/GpuSlotService.h"

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include "pipelines/scan/GpuPointCloudPool.h"
#endif

using namespace Scanner::pipeline;
using Scanner::data::EnhancedFrame;
using Scanner::Result;
using Scanner::pipeline::sched::FrameResultQueue;
using Scanner::pipeline::sched::GpuSlotService;

namespace {

constexpr int kW = 64, kH = 64;

cv::Mat makeK(int w, int h) {
    return (cv::Mat_<double>(3, 3) << 100.0, 0.0, w / 2.0,
                                       0.0, 100.0, h / 2.0,
                                       0.0, 0.0, 1.0);
}
cv::Mat makeD() { return cv::Mat::zeros(5, 1, CV_64F); }

std::shared_ptr<EnhancedFrame> zeroFrame(uint64_t id) {
    auto f = std::make_shared<EnhancedFrame>();
    f->frameId = id;
    f->grayL = cv::Mat::zeros(kH, kW, CV_8UC1);
    f->grayR = cv::Mat::zeros(kH, kW, CV_8UC1);
    f->temperature = 25.0;                      // snapshot 保持零矩阵默认
    return f;
}

ScanChainDeps makeDeps(FrameResultQueue<FrameResult>& sink) {
    ScanChainDeps d;
    d.prevState = nullptr;                      // 原子锚初始空=首帧未初始化
    d.K1 = makeK(kW, kH);  d.D1 = makeD();
    d.K2 = makeK(kW, kH);  d.D2 = makeD();
    d.imageWidth = kW;     d.imageHeight = kH;
    d.sink = &sink;
    d.poolAcquireTimeout = std::chrono::milliseconds(100);
    return d;
}

// 槽服务启动：CUDA 构建有设备→默认真工厂（真 stream 走真算子）；
// 无设备 / 无 CUDA 构建→注入假工厂（gpuChain 内算子失败/异常被捕获，验"不崩"）
bool startSlotSvc(GpuSlotService& svc) {
#ifdef JMW_BUILD_CUDA
    int devices = 0;
    try { devices = cv::cuda::getCudaEnabledDeviceCount(); } catch (...) { devices = 0; }
    if (devices > 0) return svc.start(1).success;
#endif
    static std::atomic<int> seq{0};
    auto factory = [](GpuSlotService::StreamHandle* s) {
        *s = reinterpret_cast<GpuSlotService::StreamHandle>(
            static_cast<uintptr_t>(++seq));
        return 0;
    };
    auto destroyer = [](GpuSlotService::StreamHandle) {};
    return svc.start(1, factory, destroyer).success;
}

// 手动直调三钩子（单线程，frontReady 空实现——P 链由测试显式再调）
void runChainsManual(ScanChains::Hooks& hooks, GpuSlotService& svc,
                     const std::shared_ptr<EnhancedFrame>& frame,
                     ScanFront& front, FrameResult& result) {
    auto guard = svc.acquire(std::chrono::milliseconds(500));
    ASSERT_TRUE(guard.has_value());
    bool gpuOk = true;
    EXPECT_NO_THROW(gpuOk = hooks.gpuChain(*guard, frame, front, [] {}));
    (void)gpuOk;                                // 空图：true/false 均可（不崩不挂即可）

    Result pr;
    ASSERT_NO_THROW(pr = hooks.pChain(frame, front, result));
    EXPECT_TRUE(pr.success);                    // 空帧=正常降级非失败

    std::promise<Result> prom;
    prom.set_value(pr);
    auto fut = prom.get_future();
    Result er;
    ASSERT_NO_THROW(er = hooks.eFinalize(frame, front, result, fut));
    EXPECT_TRUE(er.success);

    EXPECT_EQ(result.frameId, frame->frameId);
    EXPECT_TRUE(result.markers.empty());        // 空图无标记点
    EXPECT_TRUE(result.quality == Scanner::QualityFlag::Normal ||
                result.quality == Scanner::QualityFlag::Degraded);
}

} // namespace

// 用例 1：enableLaser=false——GPU 前段（ccl 就绪）+ P 链 + E 终段，空图正常降级
TEST(ScanChainsBuild, EmptyFrame_LaserOff_NoCrash) {
    FrameResultQueue<FrameResult> sink(4);
    ScanConfig cfg;
    cfg.enableLaser = false;
    ScanChains chains(cfg, makeDeps(sink));
    auto hooks = chains.assemble();
    ASSERT_TRUE(hooks.gpuChain && hooks.pChain && hooks.eFinalize);

    GpuSlotService svc;
    ASSERT_TRUE(startSlotSvc(svc));

    auto frame = zeroFrame(1);
    ScanFront front;
    FrameResult result;
    ASSERT_NO_FATAL_FAILURE(runChainsManual(hooks, svc, frame, front, result));

    auto out = sink.pop(std::chrono::milliseconds(200));
    ASSERT_TRUE(out.has_value());               // eFinalize 自行 push（T8 契约）
    EXPECT_EQ(out->frameId, 1u);
    EXPECT_TRUE(out->markers.empty());
}

// 用例 2：enableLaser=true——激光链空图 + 空温度表：算子失败/抛异常被捕获，不崩；
// 真实联调（真数据+真表）留 P6。
TEST(ScanChainsBuild, EmptyFrame_LaserOn_NoCrash) {
    FrameResultQueue<FrameResult> sink(4);
    ScanConfig cfg;
    cfg.enableLaser = true;
    auto deps = makeDeps(sink);
    deps.laserTable =
        std::make_shared<const calib::LaserPlaneMapTempTable>();   // 空表（假温度表）
#ifdef JMW_BUILD_CUDA
    GpuPointCloudPool pool(1, 64, [](size_t) { return cv::cuda::GpuMat(); });
    deps.laserPool = &pool;                     // 假分配器池（无设备依赖）
#endif
    ScanChains chains(cfg, std::move(deps));
    auto hooks = chains.assemble();
    ASSERT_TRUE(hooks.gpuChain && hooks.pChain && hooks.eFinalize);

    GpuSlotService svc;
    ASSERT_TRUE(startSlotSvc(svc));

    auto frame = zeroFrame(7);
    ScanFront front;
    FrameResult result;
    ASSERT_NO_FATAL_FAILURE(runChainsManual(hooks, svc, frame, front, result));

    auto out = sink.pop(std::chrono::milliseconds(200));
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->frameId, 7u);
    EXPECT_TRUE(out->markers.empty());
}
