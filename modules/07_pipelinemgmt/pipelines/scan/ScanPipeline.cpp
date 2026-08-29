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
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#include "scanning/fusion/laser_cloud_normal_cuda/laser_cloud_normal_cuda.h"
#endif

namespace Scanner::pipeline {
namespace {

constexpr int32_t kEvtLaserOffNoCuda = 1603;   // 无 CUDA：enableLaser 强制关（A 模式降级）
constexpr int32_t kEvtRuntimeDied = 1604;      // runtime 异常即停/lanes 全退（非用户 stop）
constexpr int32_t kEvtLaneHang = 1610;         // lane 心跳超时（看门狗检出；可恢复路径见 recover）
constexpr int32_t kEvtRecovered = 1611;        // Faulted 原地恢复成功（累积全保留）
constexpr int32_t kEvtCheckpointFail = 1612;   // 检查点写/读失败（Fault，不中断扫描）
constexpr size_t kQueueCapacity = 64;          // 输出队列容量（consumer 持续排空，宽裕）
constexpr int kScanGpuSlots = 2;               // 扫描 GPU 槽（设计 §4.3）
constexpr int kRenderThrottleFrames = 5;       // 渲染节流（首帧起每 N 帧）
constexpr std::chrono::milliseconds kPoolAcquireTimeout{50};
constexpr std::chrono::milliseconds kRecoverJoinTimeout{2000};  // recover 限时 drain（僵尸 detach 兜底）
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
        if (!r.success) JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] marker 融合失败: {}", r.message);
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
                JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] laser 融合失败: {}", fr.message);
                return;
            }
            auto nr = normal_.Execute(fuse_, fr);                  // 新体素法线估计
            if (!nr.success) JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] laser 法线估计失败: {}", nr.message);
        } catch (const std::exception& e) {
            JMW_LOG_ERROR("07-ScanPipeline", "[ScanPipeline] laser 融合适配器异常: {}", e.what());
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

uint64_t ScanPipeline::consumedFrames() const {          // P3 可观测（FuseConsumer 计数）
    return consumer_ ? consumer_->consumed() : 0;
}
uint64_t ScanPipeline::droppedFrames() const {           // P3 可观测（队列覆盖累计）
    return queue_.dropped();
}

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
    // state_ 一经转移不会回到 Running → 上报天然幂等。
    // 看门狗（1610）：hangDetected 即收敛——lane 可能未退（死循环算子），
    // 由 recover() 限时 drain 决定可否原地救回
    if (state_ == State::Running && runtime_.hangDetected()) {
        state_ = State::Faulted;
        if (sink_) {
            sink_->report(Scanner::QualityFlag::Fault, kEvtLaneHang,
                          "C 扫描 lane 心跳超时（看门狗）；可尝试 recover() 原地恢复");
        }
        return;
    }
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
    sc.hangTimeoutMs = cfg_.hangTimeoutMs;       // 看门狗（0=关）
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

    // 看门狗接线：runtime 检出 lane 心跳超时 → Fault(1610)（watchdog 线程调）。
    // runtime 自身已 requestStop；此处仅上报（syncState 的 hang 分支随后收敛 Faulted）
    runtime_.onHang = [this](int laneIdx, int64_t staleMs) {
        if (sink_) {
            sink_->report(Scanner::QualityFlag::Fault, kEvtLaneHang,
                          "C 扫描 lane " + std::to_string(laneIdx) + " 心跳静默 " +
                              std::to_string(staleMs) + "ms（看门狗检出）");
        }
    };

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
    // 会话参数快照（一次性——运行期排查"当时怎么配的"唯一依据）
    JMW_LOG_INFO("07-ScanPipeline",
        "[ScanPipeline] 会话启动: enableLaser={} budgetMB={} gpuSlots={} queueCap={} "
        "dropThreshold={} seedMarkers={} 水位={}",
        cfg_.enableLaser, cfg_.laserCacheBudgetMB, kScanGpuSlots, kQueueCapacity,
        dropThreshold_, cfg_.existingMarkers.size(), pauseCounter_);
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
    // 自动检查点（cfg.checkpointPath 非空时）：崩溃恢复链的常规落盘点。
    // 失败仅上报不改变 stop 语义（数据已在内存完成收尾，检查点属保险）
    if (!cfg_.checkpointPath.empty() && state_ != State::Stopped) {
        auto cs = saveCheckpoint(cfg_.checkpointPath);
        if (!cs.success)
            JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] stop 自动检查点失败: {}",
                         cs.message);
    }
    // 会话统计汇总（一次性——丢帧/异常排查的最终账本）
    if (state_ != State::Stopped) {               // 重复 stop 不重打
        const auto st = runtime_.stats();
        JMW_LOG_INFO("07-ScanPipeline",
            "[ScanPipeline] 会话结束: 态={} 已处理={} 跳帧={} GPU拒={} 钩子异常={} "
            "队列覆盖={} 已融合={} 水位={}",
            static_cast<int>(state_), st.processed, st.droppedSkips,
            st.gpuRejects, st.finalizeFails, queue_.dropped(),
            consumer_ ? consumer_->consumed() : 0, runtime_.lastCounter());
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
        JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] pause: 非运行态（忽略）");
        return;
    }
    runtime_.requestStop();
    runtime_.drainAndShutdown();                 // lane 停抓新帧；在飞帧输出已入队列
    pauseCounter_ = runtime_.lastCounter();      // 记消费水位（resume 注入，防已消费帧重扫）
    JMW_LOG_INFO("07-ScanPipeline", "[ScanPipeline] 暂停: 水位={} 已融合={}",
                 pauseCounter_, consumer_ ? consumer_->consumed() : 0);
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
    JMW_LOG_INFO("07-ScanPipeline", "[ScanPipeline] 恢复: 注入水位={}", pauseCounter_);
    return Result::ok();
}

// ============================================================================
// recover — Faulted 原地恢复（三件套之二；看门狗 1610/异常停 1604 均可达此路径）
// ============================================================================
Scanner::Result ScanPipeline::recover() {
    syncState();                                   // 惰性收敛先行（确保 Faulted 判据成立）
    if (state_ != State::Faulted)
        return Result::fail("ScanPipeline::recover: 仅 Faulted 态可恢复（当前非异常停）");
    if (recoverAttempts_ >= kMaxRecoverAttempts)
        return Result::fail("ScanPipeline::recover: 恢复次数用尽（" +
                            std::to_string(kMaxRecoverAttempts) +
                            "）——建议重建会话/进程重启");

    const uint64_t watermark = runtime_.lastCounter();
    runtime_.drainAndShutdown(kRecoverJoinTimeout);   // 限时：死循环算子→detach 僵尸兜底
    if (!runtime_.lanesExited()) {
        // 限时内 lane 未退尽（detached 僵尸在飞）：保守拒绝——僵尸仍引用
        // source/front/hooks，本对象继续重启有并发风险，升级交进程级处置
        if (sink_) {
            sink_->report(Scanner::QualityFlag::Fault, kEvtLaneHang,
                          "C 扫描 recover: lane 限时未退（僵尸在飞）——需进程重启收尾");
        }
        return Result::fail("ScanPipeline::recover: lane 限时未退，需进程重启");
    }

    // 与 resume 同款的 restart：水位注入防已消费帧重扫；consumer/obs/融合累积/
    // prevState 锚（chains_ 内存续）全保留——秒级满血续算
    Hooks hooks = testHooksSet_ ? testHooks_ : chains_->assemble();
    pauseCounter_ = watermark;
    auto rs = runtime_.start(scanSchedConfig(), *source_, /*sequential=*/false, &queue_,
                             hooks, pauseCounter_);
    if (!rs.success)
        return Result::fail("ScanPipeline::recover: runtime 重启失败: " + rs.message);
    state_ = State::Running;
    ++recoverAttempts_;
    JMW_LOG_INFO("07-ScanPipeline",
        "[ScanPipeline] 原地恢复: 第{}次 水位={} 已融合={}", recoverAttempts_, pauseCounter_,
        consumer_ ? consumer_->consumed() : 0);
    if (sink_) {
        sink_->report(Scanner::QualityFlag::Warning, kEvtRecovered,
                      "C 扫描 Faulted 原地恢复（累积保留，第 " +
                          std::to_string(recoverAttempts_) + " 次）");
    }
    return Result::ok();
}

// ============================================================================
// 检查点 — 三件套之三：崩溃恢复链（obs＋激光缓存＋配准锚＋水位）
// 双文件形态：<path>＝管道头（prevState/水位/统计）；<path>.obs＝观测累加器档
// ============================================================================
namespace {
constexpr char kCkMagic[8] = {'J', 'M', 'W', 'S', 'C', 'K', '1', '\0'};

template<typename T>
void ckWr(std::ostream& o, const T& v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template<typename T>
bool ckRd(std::istream& i, T& v) {
    return static_cast<bool>(i.read(reinterpret_cast<char*>(&v), sizeof(T)));
}
} // namespace

Scanner::Result ScanPipeline::saveCheckpoint(const std::string& path) {
    if (state_ == State::Idle)
        return Result::fail("ScanPipeline::saveCheckpoint: 须 configure 后");
    // 配准锚：生产链 chains_；测试模式（chains_ 不建）用 testPrevState_ 兜底
    AtomicFrameStatePtr& anchor = chains_ ? chains_->prevStateAnchor() : testPrevState_;
    try {
        // ① 管道头：prevState 配准锚（原子 load 快照）＋水位＋统计
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) throw std::runtime_error("打开失败: " + path);
        out.write(kCkMagic, sizeof(kCkMagic));
        const auto prev = calib::AtomicFrameState::load(anchor);
        const uint8_t hasPrev = prev ? 1 : 0;
        ckWr(out, hasPrev);
        if (prev) {
            ckWr(out, prev->frameId);
            out.write(reinterpret_cast<const char*>(prev->R), sizeof(prev->R));
            out.write(reinterpret_cast<const char*>(prev->T), sizeof(prev->T));
            ckWr(out, static_cast<uint64_t>(prev->rawPoints.size()));
            for (const auto& m : prev->rawPoints) {
                ckWr(out, m.x); ckWr(out, m.y); ckWr(out, m.z);
                ckWr(out, m.nx); ckWr(out, m.ny); ckWr(out, m.nz);
                ckWr(out, m.globalId);
            }
            ckWr(out, static_cast<uint64_t>(prev->normals.size()));
            if (!prev->normals.empty())
                out.write(reinterpret_cast<const char*>(prev->normals.data()),
                          prev->normals.size() * sizeof(double));
            ckWr(out, static_cast<uint64_t>(prev->globalIds.size()));
            if (!prev->globalIds.empty())
                out.write(reinterpret_cast<const char*>(prev->globalIds.data()),
                          prev->globalIds.size() * sizeof(int));
        }
        ckWr(out, runtime_.lastCounter());
        if (!out) throw std::runtime_error("管道头写入失败");
    } catch (const std::exception& e) {
        if (sink_)
            sink_->report(Scanner::QualityFlag::Fault, kEvtCheckpointFail,
                          std::string("检查点写失败: ") + e.what());
        return Result::fail(std::string("ScanPipeline::saveCheckpoint: ") + e.what());
    }
    // ② 观测档（obs＋激光缓存；FrameObsAccumulator 自带格式）
    auto ra = obs_.saveCheckpoint(path + ".obs");
    if (!ra.success) {
        if (sink_)
            sink_->report(Scanner::QualityFlag::Fault, kEvtCheckpointFail,
                          "检查点观测档写失败: " + ra.message);
        return Result::fail("ScanPipeline::saveCheckpoint: " + ra.message);
    }
    JMW_LOG_INFO("07-ScanPipeline",
        "[ScanPipeline] 检查点落盘: {} (观测帧={} 激光缓存B={} 水位={})", path,
        obs_.frameCount(), obs_.laserBytesUsed(), runtime_.lastCounter());
    return Result::ok();
}

Scanner::Result ScanPipeline::restoreCheckpoint(const std::string& path) {
    if (state_ == State::Running)
        return Result::fail("ScanPipeline::restoreCheckpoint: 运行中不可恢复（先 stop/pause）");
    if (state_ == State::Idle)
        return Result::fail("ScanPipeline::restoreCheckpoint: 须 configure 后");
    AtomicFrameStatePtr& anchor = chains_ ? chains_->prevStateAnchor() : testPrevState_;
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) throw std::runtime_error("打开失败: " + path);
        char magic[8];
        if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, kCkMagic, sizeof(kCkMagic)) != 0)
            throw std::runtime_error("魔术字不符(坏档)");
        uint8_t hasPrev = 0;
        if (!ckRd(in, hasPrev)) throw std::runtime_error("头截断");
        if (hasPrev) {
            auto st = std::make_shared<calib::AtomicFrameState>();
            if (!ckRd(in, st->frameId) ||
                !in.read(reinterpret_cast<char*>(st->R), sizeof(st->R)) ||
                !in.read(reinterpret_cast<char*>(st->T), sizeof(st->T)))
                throw std::runtime_error("prevState 记录截断");
            uint64_t n = 0;
            if (!ckRd(in, n)) throw std::runtime_error("rawPoints 段截断");
            st->rawPoints.resize(static_cast<size_t>(n));
            for (auto& m : st->rawPoints) {
                if (!ckRd(in, m.x) || !ckRd(in, m.y) || !ckRd(in, m.z) ||
                    !ckRd(in, m.nx) || !ckRd(in, m.ny) || !ckRd(in, m.nz) ||
                    !ckRd(in, m.globalId))
                    throw std::runtime_error("rawPoints 记录截断");
            }
            if (!ckRd(in, n)) throw std::runtime_error("normals 段截断");
            st->normals.resize(static_cast<size_t>(n));
            if (n && !in.read(reinterpret_cast<char*>(st->normals.data()),
                              static_cast<std::streamsize>(n * sizeof(double))))
                throw std::runtime_error("normals 数据截断");
            if (!ckRd(in, n)) throw std::runtime_error("globalIds 段截断");
            st->globalIds.resize(static_cast<size_t>(n));
            if (n && !in.read(reinterpret_cast<char*>(st->globalIds.data()),
                              static_cast<std::streamsize>(n * sizeof(int))))
                throw std::runtime_error("globalIds 数据截断");
            calib::AtomicFrameState::store(anchor, std::move(st));
        }
        // 水位读出但不注入 pauseCounter_：崩溃重启场景 ring 是新环，从头消费；
        // prevState 锚已恢复＝配准连续性已续（水位仅供人工核对）
        uint64_t watermark = 0;
        if (!ckRd(in, watermark)) watermark = 0;
    } catch (const std::exception& e) {
        if (sink_)
            sink_->report(Scanner::QualityFlag::Fault, kEvtCheckpointFail,
                          std::string("检查点读失败: ") + e.what());
        return Result::fail(std::string("ScanPipeline::restoreCheckpoint: ") + e.what());
    }
    auto ra = obs_.loadCheckpoint(path + ".obs");
    if (!ra.success)
        return Result::fail("ScanPipeline::restoreCheckpoint: " + ra.message);
    JMW_LOG_INFO("07-ScanPipeline",
        "[ScanPipeline] 检查点恢复: {} (观测帧={} 激光帧缓存={}档 降级={})", path,
        obs_.frameCount(),
        [&] { auto s = obs_.snapshot(); return s.laserFrames.size(); }(),
        obs_.degradedLaser());
    return Result::ok();
}

} // namespace Scanner::pipeline
