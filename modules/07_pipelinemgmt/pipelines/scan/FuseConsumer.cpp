// ============================================================================
// FuseConsumer.cpp — 融合消费线程实现（循环细节见 FuseConsumer.h 文件头）
// ============================================================================
#include "pipelines/scan/FuseConsumer.h"

#include <chrono>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#ifdef JMW_BUILD_CUDA
#include <opencv2/core.hpp>
#endif

namespace Scanner::pipeline {
namespace {

constexpr auto kPopTimeout = std::chrono::milliseconds(100);   // 空转周期（超时查 stop）
constexpr int32_t kEvtLaserCacheDegraded = 1601;               // 07 C 融合：激光帧缓存超预算
constexpr int32_t kEvtFuseFrameFault = 1602;                   // 07 C 融合：单帧处理异常（丢帧续跑）

#ifdef JMW_BUILD_CUDA
/// 生产默认下载：GpuMat(CV_32FC3 1×N) → host float xyz（空块/异常 → 空=本帧无激光观测）
/// 下载点数：count==0 或未设（0）按 cols 全量下载；count>0 取 min(count, cols)
std::vector<float> defaultLaserDownload(const GpuPointCloudBlock& b) {
    try {
        if (b.points.empty()) return {};
        const int cols = b.points.cols;
        const int n = (b.count > 0) ? (b.count < cols ? b.count : cols) : cols;
        if (n <= 0) return {};
        cv::Mat host;
        b.points.download(host);                       // 真 CUDA（默认流同步）
        if (host.empty() || !host.isContinuous()) return {};
        const float* p = host.ptr<float>();
        return std::vector<float>(p, p + static_cast<size_t>(n) * 3);
    } catch (...) {                                    // CUDA 异常（无设备/下载失败）
        return {};
    }
}
#endif

} // namespace

FuseConsumer::FuseConsumer(Deps d) : deps_(std::move(d)) {
    if (deps_.renderThrottleFrames <= 0) deps_.renderThrottleFrames = 1;
    if (deps_.highPrecisionGlobalIds) {
        hpIds_.insert(deps_.highPrecisionGlobalIds->begin(),
                      deps_.highPrecisionGlobalIds->end());
    }
}

FuseConsumer::~FuseConsumer() {
    requestStop();
    join();
}

Scanner::Result FuseConsumer::start() {
    if (thread_.joinable())
        return Scanner::Result::fail("FuseConsumer already running");
    if (!deps_.queue || !deps_.obs || !deps_.markerFuse)
        return Scanner::Result::fail("FuseConsumer deps: queue/obs/markerFuse 必填");

    stop_.store(false, std::memory_order_release);
    consumed_.store(0, std::memory_order_relaxed);
    laserDegradeReported_ = false;
    try {
        thread_ = std::thread([this] { loop(); });
    } catch (const std::system_error& e) {
        return Scanner::Result::fail(std::string("FuseConsumer 线程创建失败: ") + e.what());
    }
    return Scanner::Result::ok();
}

void FuseConsumer::requestStop() { stop_.store(true, std::memory_order_release); }

void FuseConsumer::join() {
    requestStop();                        // join 含置停；drain 由循环保证（排空后退出）
    if (thread_.joinable()) thread_.join();
}

uint64_t FuseConsumer::consumed() const {
    return consumed_.load(std::memory_order_relaxed);
}

void FuseConsumer::loop() {
    while (true) {
        auto item = deps_.queue->pop(kPopTimeout);
        if (!item) {
            if (stop_.load(std::memory_order_acquire)) return;   // 排空 + 停 → 退
            continue;                                            // 空转等待（不烧 CPU）
        }
        processOne(*item);
    }
}

void FuseConsumer::processOne(FrameResult& fr) {
    // 0) 出队即计数（异常帧也计入 consumed）；取 fetch_add 前值判渲染节流 → 首帧即推
    const uint64_t n = consumed_.fetch_add(1, std::memory_order_relaxed);
    try {
        // 1) 标记点融合（A/B 模式均跑；R/T 为配准结果，融合进全局云）
        deps_.markerFuse->fuse(fr.markers, fr.R, fr.T);

        // 2) 激光融合 + host 下载（A 模式 laserFuse 空 / 帧无块 → 跳过整段）
        std::vector<float> laserXyz;
#ifdef JMW_BUILD_CUDA
        if (deps_.laserFuse && fr.laser) {
            deps_.laserFuse->fuse(*fr.laser, fr.R, fr.T);
            laserXyz = deps_.laserDownload ? deps_.laserDownload(*fr.laser)
                                           : defaultLaserDownload(*fr.laser);
        }
#endif

        // 3) 渲染节流（先于 obs.push，序见文件头）：第 1、N+1…帧推送一次
        //    （激光句柄由 ILaserFuse 适配器持有 → nullptr）
        if (deps_.sceneFeed &&
            n % static_cast<uint64_t>(deps_.renderThrottleFrames) == 0) {
            CloudViewHandle h;
            h.hostMarker = &deps_.markerFuse->fusedPoints();
            h.deviceLaser = nullptr;
            deps_.sceneFeed->pushCloudSnapshot(h);
        }

        // 4) 攒观测：R_init/T_init 逐帧保留；markerObs 查高精度集；激光 host 拷贝入缓存
        FrameObs fo;
        fo.frameId = fr.frameId;
        for (int i = 0; i < 9; ++i) fo.R_init[i] = fr.R[i];
        for (int i = 0; i < 3; ++i) fo.t_init[i] = fr.T[i];
        fo.markerObs.reserve(fr.markers.size());
        for (const auto& m : fr.markers) {
            MarkerObs mo;
            mo.xyz[0] = m.x;
            mo.xyz[1] = m.y;
            mo.xyz[2] = m.z;
            mo.globalId = m.globalId;
            mo.isHighPrecision = hpIds_.count(m.globalId) > 0;
            fo.markerObs.push_back(mo);
        }
        deps_.obs->push(std::move(fo), laserXyz);

        if (deps_.sink && !laserDegradeReported_ && deps_.obs->degradedLaser()) {
            deps_.sink->report(Scanner::QualityFlag::Degraded, kEvtLaserCacheDegraded,
                               "激光帧缓存超预算，停累加（GBA 激光观测缺失）");
            laserDegradeReported_ = true;     // 一次性（只报首次降级）
        }
    } catch (...) {
        // 单帧兜底：融合消费属持续服务——丢该帧续跑，不崩不退线程
        JMW_LOG_ERROR("07-FuseConsumer", "[FuseConsumer] 单帧处理异常 frameId={}，丢弃该帧续跑", fr.frameId);
        if (deps_.sink) {
            deps_.sink->report(Scanner::QualityFlag::Fault, kEvtFuseFrameFault,
                               "融合消费单帧处理异常，丢帧续跑");
        }
    }
}

} // namespace Scanner::pipeline
