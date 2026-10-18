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
#include <opencv2/core/cuda.hpp>
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
        // 261018 流序下载退役默认流：原 download(host) 走 legacy 默认流（与
        // lane 阻塞流互斥＝跨 lane 闸）——thread_local 持有流＋本流同步
        //（块数据在入队前已经 lane 链尾同步落定，任意流读取皆安全）
        static thread_local cv::cuda::Stream s_dlStream;
        b.points.download(host, s_dlStream);
        s_dlStream.waitForCompletion();
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
    // 261018 消费段打点（节流 1/30）：10-18 真机实测消费仅 13fps（60fps 生产
    // →池干涸→47k 取块超时），补分段观测定位消费侧耗时归属
    using CLK = std::chrono::steady_clock;
    const auto tC0 = CLK::now();
    auto msSince = [&tC0] {
        return std::chrono::duration<double, std::milli>(CLK::now() - tC0).count();
    };
    double tMarker = 0, tLaser = 0, tRender = 0;
    try {
        // 1) 标记点融合（A/B 模式均跑；R/T 为配准结果，融合进全局云）
        deps_.markerFuse->fuse(fr.markers, fr.R, fr.T);
        tMarker = msSince();

        // 2) 激光融合 + host 下载（A 模式 laserFuse 空 / 帧无块 → 跳过整段）
        std::vector<float> laserXyz;
#ifdef JMW_BUILD_CUDA
        if (deps_.laserFuse && fr.laser) {
            deps_.laserFuse->fuse(*fr.laser, fr.R, fr.T);
            laserXyz = deps_.laserDownload ? deps_.laserDownload(*fr.laser)
                                           : defaultLaserDownload(*fr.laser);
        }
        // 261018 块即用即还：下载完成后立即归池——原持有到帧析构（渲染/obs
        // 段数十 ms 且随融合云增长），期间块被白占＝池干涸推手之二
        fr.laser.reset();
#endif
        tLaser = msSince() - tMarker;

        // 3) 渲染节流（先于 obs.push，序见文件头）：第 1、N+1…帧推送一次
        //    （激光句柄由 ILaserFuse 适配器持有 → nullptr）。
        //    ＋时间门控 ≥300ms（260919 卡顿修复：帧计数节流按 7fps 设计，
        //    60fps 时=12 次/秒全量推送（百万点 VBO 重传+色数组 new+仓库整包
        //    拷贝全在 UI 线程）=显示卡顿。时间门控保 2~3Hz 刷新）
        static thread_local std::chrono::steady_clock::time_point s_lastPush{};
        const auto nowTp = std::chrono::steady_clock::now();
        const bool timeOk =
            deps_.renderThrottleMinMs <= 0 ||
            s_lastPush.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(nowTp - s_lastPush).count() >=
                static_cast<std::chrono::milliseconds::rep>(deps_.renderThrottleMinMs);
        if (deps_.sceneFeed && timeOk &&
            n % static_cast<uint64_t>(deps_.renderThrottleFrames) == 0) {
            s_lastPush = nowTp;
            CloudViewHandle h;
            h.hostMarker = &deps_.markerFuse->fusedPoints();
            h.deviceLaser = nullptr;
            // 激光显示推融合云累计值（体素去重后单调不减——修复「点数忽
            // 大忽小」：旧版推当帧块，T/V 交替 79K/124K 跳变）
            std::vector<float> fusedXyz;
#ifdef JMW_BUILD_CUDA
            if (deps_.laserFuse) fusedXyz = deps_.laserFuse->downloadFusedXyz();
#endif
            // 点云仓库直写（260919 正式口）：融合云累计快照直入 06 PointCloudBuffer
            //（会话层替换语义——app UI「基线锁存+整包替换」补丁链由此退役）。
            // 仓库＝全量真值（计数/导出/后处理原料——不抽稀）
            if (deps_.cloudWarehouse && !fusedXyz.empty())
                deps_.cloudWarehouse->pushSessionCloud(fusedXyz);
            // 261018 UI 块抽稀 ≤200 万点：推送链上 adapter 值拷贝＋Qt queued 各
            // ~288MB（2400 万点×12B）＝render 段大头之二——UI 显示本来就抽 150
            // 万（MainWindow kDispMax），此处先抽省两跳大拷贝；仓库/日志计数仍
            // 全量真值
            std::vector<float> uiXyz;
            constexpr size_t kUiMaxPts = 2000000;
            const size_t totalPts = fusedXyz.size() / 3;
            if (totalPts > kUiMaxPts) {
                const size_t stride = (totalPts + kUiMaxPts - 1) / kUiMaxPts;
                uiXyz.reserve((totalPts + stride - 1) / stride * 3);
                for (size_t i = 0; i < totalPts; i += stride) {
                    uiXyz.push_back(fusedXyz[i * 3 + 0]);
                    uiXyz.push_back(fusedXyz[i * 3 + 1]);
                    uiXyz.push_back(fusedXyz[i * 3 + 2]);
                }
            }
            const std::vector<float>& laserOut = uiXyz.empty() ? fusedXyz : uiXyz;
            h.hostLaser = laserOut.empty() ? nullptr : laserOut.data();
            h.laserCount = laserOut.size() / 3;
            // 显示链观测（节流同频）：融合云点数（全量真值）——0=融合无产出；
            // >0=断在下游（SceneFeed/UI）
            JMW_LOG_INFO("07-ScanChains",
                "[FuseConsumer] 推送渲染: 融合云={} 点 激光融合云={} 点（UI 抽稀 {} 点，帧 {}）",
                deps_.markerFuse->fusedPoints().size(), totalPts, h.laserCount, n);
            deps_.sceneFeed->pushCloudSnapshot(h);
        }
        tRender = msSince() - tMarker - tLaser;

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

        // 消费段打点（节流 1/30）：marker=标记点融合 / laser=激光融合+下载
        // / render=节流推送（融合云全量下载+仓库整包） / obs=观测累加
        {
            static std::atomic<uint64_t> s_fcSeg{0};
            if (s_fcSeg.fetch_add(1, std::memory_order_relaxed) % 30 == 0)
                JMW_LOG_INFO("07-ScanChains",
                             "[FuseConsumer] 消费分段: marker={:.1f} laser={:.1f} render={:.1f} obs={:.1f} ms（帧 {}）",
                             tMarker, tLaser, tRender, msSince() - tMarker - tLaser - tRender, n);
        }

        if (deps_.sink && !laserDegradeReported_ && deps_.obs->degradedLaser()) {
            deps_.sink->report(Scanner::QualityFlag::Degraded, kEvtLaserCacheDegraded,
                               "激光帧缓存超预算，停累加（GBA 激光观测缺失）");
            laserDegradeReported_ = true;     // 一次性（只报首次降级）
        }
    } catch (...) {
        // 单帧兜底：融合消费属持续服务——丢该帧续跑，不崩不退线程。
        // 260927 严重度纠正：报 Warning 不报 Fault——「丢帧续跑」语义不停机；
        // 原 Fault→Error 级曾致故障桥 S7 全局停机，会话卡死连 finish_scan 都被
        // 门禁拒（真机日志 14:41/14:42 两帧异常→S7→关闭被拒实证）
        JMW_LOG_ERROR("07-FuseConsumer", "[FuseConsumer] 单帧处理异常 frameId={}，丢弃该帧续跑", fr.frameId);
        if (deps_.sink) {
            deps_.sink->report(Scanner::QualityFlag::Warning, kEvtFuseFrameFault,
                               "融合消费单帧处理异常，丢帧续跑");
        }
    }
}

} // namespace Scanner::pipeline
