// ============================================================================
// ScanPipeline.cpp — C 扫描处理流水线对象总装实现（生命周期与编排细节见头文件注）
//
// 实现要点：
//   - 真算子适配器在匿名命名空间：MarkerFuseAdapter 包 calib::MarkerCloudFuseCPU
//     （seed 透传 + MarkerPoint3D→MarkerFuseInput 映射 + fusedPoints 稳定存储）；
//     LaserFuseAdapter 包 LaserCloudFuseCuda + LaserCloudNormalCuda（融合后新体素
//     法线估计；仅 JMW_BUILD_CUDA，生产路径构造——测试注入假实现不触 GPU）
//   - 停止顺序（头文件注）：runtime.requestStop → drainAndShutdown →
//     consumer.requestStop+join——consumer 在 runtime 停后仍排空在飞帧输出，不丢
//   - pause 不动 consumer：融合/obs/pool 累积全保留；resume 走 runtime restart
// ============================================================================
#include "pipelines/scan/ScanPipeline.h"

#include <algorithm>
#include <utility>

#include <spdlog/spdlog.h>

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#include "scanning/fusion/laser_cloud_normal_cuda/laser_cloud_normal_cuda.h"
#endif

namespace Scanner::pipeline {
namespace {

constexpr int32_t kEvtLaserOffNoCuda = 1603;   // 无 CUDA：enableLaser 强制关（A 模式降级）
constexpr int32_t kEvtRuntimeDied = 1604;      // runtime 异常即停/lanes 全退（非用户 stop）
constexpr size_t kQueueCapacity = 64;          // 输出队列容量（consumer 持续排空，宽裕）
constexpr int kScanGpuSlots = 2;               // 扫描 GPU 槽（设计 §4.3）
constexpr int kRenderThrottleFrames = 5;       // 渲染节流（首帧起每 N 帧）
constexpr std::chrono::milliseconds kPoolAcquireTimeout{50};
#ifdef JMW_BUILD_CUDA
constexpr size_t kLaserPoolBlocks = 8;                 // 在飞激光块上限
constexpr size_t kLaserPoolPointsPerBlock = 1 << 21;   // 2M 点/块 ≈ 24MB 显存
#endif

cv::Matx33d matxFromArr9(const double* a) {
    return cv::Matx33d(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}
cv::Vec3d vec3FromArr3(const double* a) { return cv::Vec3d(a[0], a[1], a[2]); }

// ---------------------------------------------------------------------------
// MarkerFuseAdapter — 标记点 CPU 融合真算子适配器（ISeedableMarkerFuse 落地）
// ---------------------------------------------------------------------------
class MarkerFuseAdapter final : public ISeedableMarkerFuse {
public:
    MarkerFuseAdapter() = default;

    /// seed 透传（09 契约：须先于任何 Execute；整批原子，失败零写入）
    Scanner::Result seed(const std::vector<calib::MarkerCloudPoint>& pts) override {
        auto st = op_.seed(pts);
        if (!st.success) return Scanner::Result::fail(st.message);
        return Scanner::Result::ok();
    }

    void fuse(const std::vector<calib::MarkerPoint3D>& markers,
              const double R[9], const double T[3]) override {
        if (markers.empty()) return;
        std::vector<calib::MarkerFuseInput> in(markers.size());
        for (size_t i = 0; i < markers.size(); ++i) {
            in[i].x  = static_cast<float>(markers[i].x);
            in[i].y  = static_cast<float>(markers[i].y);
            in[i].z  = static_cast<float>(markers[i].z);
            in[i].nx = static_cast<float>(markers[i].nx);
            in[i].ny = static_cast<float>(markers[i].ny);
            in[i].nz = static_cast<float>(markers[i].nz);
            // whiteRadius：MarkerPoint3D 无半径字段，置 0（饱和判定降级，不影响位置融合）
        }
        auto r = op_.Execute(in, matxFromArr9(R), vec3FromArr3(T));
        if (!r.success) spdlog::warn("[ScanPipeline] marker 融合失败: {}", r.message);
    }

    /// 稳定存储：算子内部 vector 对象（地址跨调用稳定，渲染句柄安全）
    const std::vector<calib::MarkerCloudPoint>& fusedPoints() const override {
        return op_.GetFusedPoints();
    }

private:
    calib::MarkerCloudFuseCPU op_;
};

#ifdef JMW_BUILD_CUDA
// ---------------------------------------------------------------------------
// LaserFuseAdapter — 激光 CUDA 融合 + 法线估计真算子适配器（生产路径构造）
// ---------------------------------------------------------------------------
class LaserFuseAdapter final : public ILaserFuse {
public:
    LaserFuseAdapter() = default;

    void fuse(const GpuPointCloudBlock& block, const double R[9], const double T[3]) override {
        try {
            if (block.points.empty()) return;
            const int cols = block.points.cols;
            const int n = (block.count > 0) ? std::min(block.count, cols) : cols;
            if (n <= 0) return;
            cv::cuda::GpuMat view = block.points.colRange(0, n);   // 1×n CV_32FC3 视图
            auto fr = fuse_.Execute(view, matxFromArr9(R), vec3FromArr3(T));
            if (!fr.success) {
                spdlog::warn("[ScanPipeline] laser 融合失败: {}", fr.message);
                return;
            }
            auto nr = normal_.Execute(fuse_, fr);                  // 新体素法线估计
            if (!nr.success) spdlog::warn("[ScanPipeline] laser 法线估计失败: {}", nr.message);
        } catch (const std::exception& e) {
            spdlog::error("[ScanPipeline] laser 融合适配器异常: {}", e.what());
        }
    }

private:
    calib::LaserCloudFuseCuda fuse_;
    calib::LaserCloudNormalCuda normal_;
};
#endif // JMW_BUILD_CUDA

} // namespace

// ============================================================================
// 构造 / 装配
// ============================================================================
ScanPipeline::ScanPipeline(ScanConfig cfg)
    : cfg_(std::move(cfg)),
      queue_(kQueueCapacity),
      obs_(cfg_.laserCacheBudgetMB ? (cfg_.laserCacheBudgetMB << 20) : (1 << 20)) {
    // hpGlobalIds 映射简化：existingMarkers 下标 0..n-1 即高精度 globalId 集
    // （与 09 optical_flow_fuse 首帧 globalId=bestIdx 对接，见头文件注）
    hpGlobalIds_.reserve(cfg_.existingMarkers.size());
    for (size_t i = 0; i < cfg_.existingMarkers.size(); ++i)
        hpGlobalIds_.push_back(static_cast<int>(i));
}

ScanPipeline::~ScanPipeline() { stop(); }        // 安全网：幂等

void ScanPipeline::attachRing(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring,
                              size_t dropThreshold) {
    ring_ = &ring;
    dropThreshold_ = dropThreshold;
}

void ScanPipeline::attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                               const cv::Mat& D2, int imageWidth, int imageHeight,
                               std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable) {
    K1_ = K1;  D1_ = D1;  K2_ = K2;  D2_ = D2;
    imageWidth_ = imageWidth;  imageHeight_ = imageHeight;
    laserTable_ = std::move(laserTable);
}

sched::FrameResultQueue<FrameResult>& ScanPipeline::outputQueue() { return queue_; }
FrameObsAccumulator& ScanPipeline::obs() { return obs_; }

void ScanPipeline::attachTestHooks(Hooks hooks) {
    testHooksSet_ = true;
    testHooks_ = std::move(hooks);
}

void ScanPipeline::attachTestFuseAdapters(ISeedableMarkerFuse* markerFuse
#ifdef JMW_BUILD_CUDA
                                         ,
                                         ILaserFuse* laserFuse
#endif
) {
    testMarkerFuse_ = markerFuse;
    testMarkerFuseSet_ = true;
#ifdef JMW_BUILD_CUDA
    testLaserFuse_ = laserFuse;
    testLaserFuseSet_ = true;
#endif
}

std::unique_ptr<PipelineEventSink> ScanPipeline::makeSink(const PipelineDeps& deps) const {
    return std::make_unique<EventBusEventSink>(deps.eventBus);   // nullptr 安全（no-op）
}

void ScanPipeline::syncState() const {
    // Running 且 lanes 全退 = runtime 自灭（用户 stop/pause 路径会先转 Stopped/Paused，
    // 不会以 Running 态观测到全退）→ 惰性收敛 Faulted + Fault(1604) 一次性上报。
    // state_ 一经转移不会回到 Running → 上报天然幂等
    if (state_ == State::Running && runtime_.lanesExited()) {
        state_ = State::Faulted;
        if (sink_) {
            sink_->report(Scanner::QualityFlag::Fault, kEvtRuntimeDied,
                          "C 扫描流水线异常停（帧钩子异常/资源故障），累积数据保留待 stop 收尾");
        }
    }
}

sched::SchedConfig ScanPipeline::scanSchedConfig() const {
    sched::SchedConfig sc;
    sc.lanes = 0;                                // 自动探测（clamp(min(P-1,E),1,8)）
    sc.gpuSlots = kScanGpuSlots;
    sc.queueCapacity = kQueueCapacity;
    sc.dropThreshold = dropThreshold_;           // GrabLatestSource 由构造参携带，此处备档
    return sc;
}

// ============================================================================
// configure — 注入 sink/sceneFeed；装配融合适配器 + pool + ScanChains + consumer
// ============================================================================
Scanner::Result ScanPipeline::configure(const PipelineDeps& deps) {
    if (state_ != State::Idle)
        return Result::fail("ScanPipeline::configure: 当前态不可再配置（已配置/运行/暂停/"
                            "异常停/已停止，每对象恰一次）");
    if (testHooksSet_ && !K1_.empty())
        return Result::fail("ScanPipeline::configure: 测试钩子与 attachCalib 互斥");
    if (!testHooksSet_ && K1_.empty())
        return Result::fail("ScanPipeline::configure: 生产模式须先 attachCalib（K/D/W/H）");
    if (testMarkerFuseSet_ && !testHooksSet_)
        return Result::fail("ScanPipeline::configure: attachTestFuseAdapters 须配合 "
                            "attachTestHooks（无假链时假适配器无意义）");

    sink_ = makeSink(deps);

#ifndef JMW_BUILD_CUDA
    if (cfg_.enableLaser) {                      // A 模式降级：关激光 + 一次性上报
        cfg_.enableLaser = false;
        sink_->report(Scanner::QualityFlag::Warning, kEvtLaserOffNoCuda,
                      "无 CUDA 构建，enableLaser 强制关闭（降级 A 模式）");
    }
#endif

    // —— 标记点融合适配器（A/B 模式均跑；测试注入优先）——
    if (testMarkerFuseSet_) {
        markerFuse_ = testMarkerFuse_;
    } else {
        ownedMarkerFuse_ = std::make_unique<MarkerFuseAdapter>();
        markerFuse_ = ownedMarkerFuse_.get();
    }

#ifdef JMW_BUILD_CUDA
    // —— 激光融合适配器（enableLaser=false 时 null=A 模式）——
    if (cfg_.enableLaser) {
        if (testLaserFuseSet_) {
            laserFuse_ = testLaserFuse_;
        } else {
            ownedLaserFuse_ = std::make_unique<LaserFuseAdapter>();
            laserFuse_ = ownedLaserFuse_.get();
        }
    } else {
        laserFuse_ = nullptr;
    }

    // —— 激光显存块池（生产链 GPU 激光段写块用；真分配需 GPU 设备）——
    if (cfg_.enableLaser && !testHooksSet_) {
        try {
            laserPool_ = std::make_unique<GpuPointCloudPool>(kLaserPoolBlocks,
                                                             kLaserPoolPointsPerBlock);
        } catch (const std::exception& e) {
            return Result::fail(std::string("ScanPipeline::configure: 激光块池预分配失败: ") +
                                e.what());
        }
    }
#endif

    // —— 生产链（测试模式由假钩子替换）——
    if (!testHooksSet_) {
        ScanChainDeps sd;
        sd.laserTable = laserTable_;             // A 模式无表属正常配置
        sd.K1 = K1_;  sd.D1 = D1_;  sd.K2 = K2_;  sd.D2 = D2_;
        sd.imageWidth = imageWidth_;
        sd.imageHeight = imageHeight_;
        sd.sink = &queue_;
        sd.poolAcquireTimeout = kPoolAcquireTimeout;
#ifdef JMW_BUILD_CUDA
        sd.laserPool = laserPool_.get();         // enableLaser=false 时可空
#endif
        chains_ = std::make_unique<ScanChains>(cfg_, std::move(sd));
    }

    // —— 融合消费线程（不绑核默认调度；hp 集随 ctor 拷入）——
    FuseConsumer::Deps cd;
    cd.queue = &queue_;
    cd.markerFuse = markerFuse_;
#ifdef JMW_BUILD_CUDA
    cd.laserFuse = laserFuse_;
#endif
    cd.sceneFeed = deps.sceneFeed;
    cd.obs = &obs_;
    cd.sink = sink_.get();
    cd.renderThrottleFrames = kRenderThrottleFrames;
    cd.highPrecisionGlobalIds = hpGlobalIds_.empty() ? nullptr : &hpGlobalIds_;
    consumer_ = std::make_unique<FuseConsumer>(cd);

    state_ = State::Configured;
    return Result::ok();
}

// ============================================================================
// start — seed（若有）→ runtime.start（grabLatest 面孔）→ consumer.start
// ============================================================================
Scanner::Result ScanPipeline::start() {
    switch (state_) {
        case State::Idle:     return Result::fail("ScanPipeline::start: 须先 configure");
        case State::Running:  return Result::fail("ScanPipeline::start: 已在运行");
        case State::Paused:   return Result::fail("ScanPipeline::start: 暂停态请用 resume()");
        case State::Faulted:  return Result::fail("ScanPipeline::start: 已异常停（会话私有件，续扫建新对象）");
        case State::Stopped:  return Result::fail("ScanPipeline::start: 已停止（会话私有件，续扫建新对象）");
        case State::Configured: break;
    }
    if (!ring_) return Result::fail("ScanPipeline::start: 未 attachRing（输入源必备）");

    // 1) seed：时序保证——先于 runtime/consumer 起线程，即先于任何扫描帧 fuse；
    //    失败整批零写入（09 契约），start 可重试（seeded_ 守卫不重复 seed）
    if (!seeded_ && !cfg_.existingMarkers.empty()) {
        if (!markerFuse_) return Result::fail("ScanPipeline::start: marker 融合适配器缺失");
        auto sr = markerFuse_->seed(cfg_.existingMarkers);
        if (!sr.success)
            return Result::fail("ScanPipeline::start: seed 失败: " + sr.message);
        seeded_ = true;
    }

    // 2) runtime：X lane 抓最新 + GPU 槽 + 输出队列（eFinalize 自行 push）
    if (!source_) {
        source_ = std::make_unique<sched::GrabLatestSource<Scanner::data::EnhancedFrame>>(
            *ring_, dropThreshold_);
    }
    if (testHooksSet_) {
        // 测试模式：假流工厂（假链不触流；生产链用默认真 CUDA 工厂）
        const auto fr = runtime_.setGpuStreamFactory(
            [this](sched::GpuSlotService::StreamHandle* s) {
                *s = reinterpret_cast<sched::GpuSlotService::StreamHandle>(
                    static_cast<uintptr_t>(++fakeStreamSeq_));
                return 0;
            },
            [](sched::GpuSlotService::StreamHandle) {});
        if (!fr.success) return Result::fail("ScanPipeline::start: " + fr.message);
    }
    Hooks hooks = testHooksSet_ ? testHooks_ : chains_->assemble();
    auto rs = runtime_.start(scanSchedConfig(), *source_, /*sequential=*/false, &queue_, hooks);
    if (!rs.success)
        return Result::fail("ScanPipeline::start: runtime 启动失败: " + rs.message);

    // 3) consumer（runtime 后起；失败回退 runtime）
    auto cs = consumer_->start();
    if (!cs.success) {
        runtime_.requestStop();
        runtime_.drainAndShutdown();
        return Result::fail("ScanPipeline::start: FuseConsumer 启动失败: " + cs.message);
    }

    state_ = State::Running;
    return Result::ok();
}

// ============================================================================
// stop — 停止顺序见头文件：lane 停抓新帧 → 在飞排空 → consumer 排空队列后退出
// ============================================================================
void ScanPipeline::stop() {
    if (state_ == State::Idle) return;            // 未配置：无物可停（保持 Idle 可装配）
    syncState();                                  // Faulted 一次性上报后再收尾
    runtime_.requestStop();                       // 幂等；未 start 过也安全
    runtime_.drainAndShutdown();
    if (consumer_) {
        consumer_->requestStop();
        consumer_->join();                        // drain 语义：排空队列后返回
    }
    state_ = State::Stopped;
}

bool ScanPipeline::isRunning() const {
    syncState();                                  // 惰性收敛（异常停发现点）
    return state_ == State::Running;
}

// ============================================================================
// pause / resume — 扫描会话控制（⑥ 就绪态再按键回 ③④⑤）
// ============================================================================
void ScanPipeline::pause() {
    syncState();
    if (state_ != State::Running) {
        spdlog::warn("[ScanPipeline] pause: 非运行态（忽略）");
        return;
    }
    runtime_.requestStop();
    runtime_.drainAndShutdown();                 // lane 停抓新帧；在飞帧输出已入队列
    pauseCounter_ = runtime_.lastCounter();      // 记消费水位（resume 注入，防已消费帧重扫）
    state_ = State::Paused;
    // consumer 保持活着：fusion/obs/pool 与配准 prevState 锚（chains_ 内存续）全保留
}

Scanner::Result ScanPipeline::resume() {
    if (state_ != State::Paused) return Result::fail("ScanPipeline::resume: 非暂停态");
    // restart 语义：GpuService 每周期重建（流工厂已注入续用；T8 用例 8 验证）；
    // startCounter=暂停前水位——GrabLatest 语义"下一待读帧号"，跳过已消费帧
    Hooks hooks = testHooksSet_ ? testHooks_ : chains_->assemble();
    auto rs = runtime_.start(scanSchedConfig(), *source_, /*sequential=*/false, &queue_, hooks,
                             pauseCounter_);
    if (!rs.success)
        return Result::fail("ScanPipeline::resume: runtime 重启失败: " + rs.message);
    state_ = State::Running;
    return Result::ok();
}

} // namespace Scanner::pipeline
