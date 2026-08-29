// ============================================================================
// GlobalOptimObject.cpp — D 全局优化对象实现
//
// 步骤编排（设计方案 §4.4，细节见 GlobalOptimObject.h 文件头）：
//   freeze → 装配 GBA 输入（软先验合并）→ GBA → marker/laser 重融合重放 →
//   入库 + 完成事件 → 解冻推送。GBA 失败=Fault 上报+初值兜底（Degraded）；
//   激光缓存降级=不重放（Degraded 近似）；取消=检查点安全退出。
// ============================================================================
#include "pipelines/globaloptim/GlobalOptimObject.h"

#include <algorithm>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"
#include "scanning/global_optim/global_ba_cpu.h"

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#endif

namespace Scanner::pipeline {
namespace {

// 事件码（07 D 全局优化族；16xx=C 扫描族内顺延）
constexpr int32_t kEvtGbaFail = 1605;        // GBA 失败（Fault；沿初值兜底）
constexpr int32_t kEvtDone = 1606;           // D 完成（Normal/Degraded 按 out_.quality）
constexpr int32_t kEvtRepoWriteFail = 1607;  // 点云入库失败（Fault）
constexpr int32_t kEvtLaserNoReplay = 1608;  // 激光缓存降级：未重放（Degraded 近似）
constexpr int32_t kEvtLaserReplayFail = 1609; // 激光重放融合部分帧失败（Degraded，每 run 一次）

cv::Matx33d matxFromArr9(const double* a) {
    return cv::Matx33d(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}
cv::Vec3d vec3FromArr3(const double* a) { return cv::Vec3d(a[0], a[1], a[2]); }

const char* qualityName(Scanner::QualityFlag q) {
    switch (q) {
        case Scanner::QualityFlag::Normal:   return "Normal";
        case Scanner::QualityFlag::Degraded: return "Degraded";
        case Scanner::QualityFlag::Warning:  return "Warning";
        case Scanner::QualityFlag::Fault:    return "Fault";
    }
    return "?";
}

#ifdef JMW_BUILD_CUDA
/// 生产激光重放适配器：host xyz → GpuMat 上传 → 新 LaserCloudFuseCuda 融合
/// （每次 run 新实例=重放语义；法线不随重放重估计——沿 C 线法线语义，如实注明）。
/// 返回 false=本帧失败/异常（空/畸形输入、上传异常、融合失败——调用方聚合降级）
class RealLaserReplayAdapter final : public ILaserReplayFuse {
public:
    bool fuse(const std::vector<float>& xyz, const double R[9], const double T[3]) override {
        if (xyz.empty() || xyz.size() % 3 != 0) return false;
        try {
            cv::Mat host(1, static_cast<int>(xyz.size() / 3), CV_32FC3,
                         const_cast<float*>(xyz.data()));   // 视图（upload 只读）
            cv::cuda::GpuMat dev;
            dev.upload(host);                               // host→device（默认流同步）
            auto r = fuse_.Execute(dev, matxFromArr9(R), vec3FromArr3(T));
            if (!r.success) {
                JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] 激光重放融合失败: {}", r.message);
                return false;
            }
            return true;
        } catch (const std::exception& e) {
            JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] 激光重放上传/融合异常: {}", e.what());
            return false;
        } catch (...) {
            JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] 激光重放上传/融合未知异常");
            return false;
        }
    }

    calib::LaserCloudFuseDeviceContext deviceContext() const override {
        return fuse_.GetDeviceContext();
    }

private:
    calib::LaserCloudFuseCuda fuse_;
};
#endif

} // namespace

// ============================================================================
// 构造 / 装配
// ============================================================================
GlobalOptimObject::GlobalOptimObject(Config cfg) : cfg_(cfg) {}

GlobalOptimObject::~GlobalOptimObject() { stop(); }

void GlobalOptimObject::setExistingPrior(std::vector<int> globalIds,
                                         std::vector<double> xyz3n,
                                         std::vector<double> sigma) {
    priorIds_ = std::move(globalIds);
    priorXyz_ = std::move(xyz3n);
    priorSigma_ = std::move(sigma);
}

Scanner::Result GlobalOptimObject::configure(const PipelineDeps& deps) {
    if (running_.load())
        return Scanner::Result::fail("GlobalOptimObject::configure: 运行中不可配置");
    cloudRepo_ = deps.cloudRepo;    // run 尾自动写（06 点云仓库适配由 02 接线）
    sceneFeed_ = deps.sceneFeed;    // 冻结/解冻 + 修正点云推送
    sink_ = std::make_unique<EventBusEventSink>(deps.eventBus);   // nullptr 安全
    configured_ = true;
    return Scanner::Result::ok("global optim configured");
}

void GlobalOptimObject::attachTestGba(GbaFn fn) { gbaFn_ = std::move(fn); }

#ifdef JMW_BUILD_CUDA
void GlobalOptimObject::attachTestLaserFuseFactory(LaserReplayFuseFactory f) {
    laserFactory_ = std::move(f);
}
#endif

calib::GlobalBAParams GlobalOptimObject::gbaParams() const {
    calib::GlobalBAParams p;        // 09 默认（软先验 σ 默认亦 09 默认 0.001）
    p.maxIterations = cfg_.maxIterations;
    p.tolerance = cfg_.tolerance;
    p.enablePoseGraphPreopt = cfg_.enablePoseGraphPreopt;
    p.useSoftPrior = cfg_.useSoftPrior;
    p.defaultPriorSigma = cfg_.defaultPriorSigma;
    p.sigmaObserved = cfg_.sigmaObserved;
    return p;
}

const GlobalOptimOutput& GlobalOptimObject::output() const { return out_; }

// ============================================================================
// run — 阻塞批算（步骤见文件头）
// ============================================================================
Scanner::Result GlobalOptimObject::run(FrameObsAccumulator& obs, const ProgressCb& cb,
                                       CancelToken& cancel) {
    if (!configured_)
        return Scanner::Result::fail("GlobalOptimObject::run: 须先 configure");
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Scanner::Result::fail("global optim already running");
    return runLocked(obs, cb, cancel);
}

Scanner::Result GlobalOptimObject::runLocked(FrameObsAccumulator& obsAcc,
                                             const ProgressCb& cb, CancelToken& cancel) {
    running_.store(true);
    struct RunningGuard {
        std::atomic<bool>& flag;
        ~RunningGuard() { flag.store(false); }
    } runningGuard{running_};

    out_ = GlobalOptimOutput{};
    if (cb) cb(0, "global optim start");

    // —— 观测快照（空观测快速失败，不动渲染冻结）——
    const auto snap = obsAcc.snapshot();
    out_.frameCount = snap.obs.size();
    if (snap.obs.empty())
        return Scanner::Result::fail("GlobalOptimObject::run: 逐帧观测为空（无扫描数据）");

    // 批算输入快照（一次性——重放/降级排查的输入依据）
    JMW_LOG_INFO("07-GlobalOptim",
        "[GlobalOptim] 启动: 观测帧={} 激光缓存帧={} 缓存字节={} 已降级={} 先验点={}",
        snap.obs.size(), snap.laserFrames.size(), obsAcc.laserBytesUsed(),
        obsAcc.degradedLaser(), priorIds_.size());

    // —— 步骤 1：冻结渲染（批处理期间 display 保持末帧）——
    if (sceneFeed_) sceneFeed_->notifyFreeze(true);
    struct UnfreezeGuard {                       // 异常/取消/失败路径保证解冻
        ISceneFeed* sf;
        bool dismissed = false;
        ~UnfreezeGuard() {
            if (sf && !dismissed) sf->notifyFreeze(false);
        }
    } unfreeze{sceneFeed_};

    Scanner::QualityFlag quality = Scanner::QualityFlag::Normal;

    // —— 步骤 2：装配 GlobalBAInput + 软先验合并 ——
    if (cb) cb(10, "装配 GBA 输入");
    calib::GlobalBAInput in;
    in.frames.reserve(snap.obs.size());
    std::unordered_set<int> obsHpIds;
    for (const auto& fo : snap.obs) {
        calib::GlobalBAFrame f;
        f.frameId = fo.frameId;
        f.R_init = matxFromArr9(fo.R_init);
        f.t_init = vec3FromArr3(fo.t_init);
        f.markerObs.reserve(fo.markerObs.size());
        for (const auto& mo : fo.markerObs) {
            f.markerObs.push_back({cv::Point3d(mo.xyz[0], mo.xyz[1], mo.xyz[2]), mo.globalId});
            if (mo.isHighPrecision) obsHpIds.insert(mo.globalId);
        }
        in.frames.push_back(std::move(f));
    }
    // 软先验：02 注入集（有位置/σ，σ 空=09 默认）为准；obs 高精度但缺先验位置的
    // id 无法加残差（无 X_existing）→ warn 跳过（09 侧对畸形 id 亦鲁棒忽略）
    in.highPrecisionGlobalIds = priorIds_;
    in.X_existing = priorXyz_;
    in.priorSigma = priorSigma_;
    for (int id : obsHpIds)
        if (std::find(priorIds_.begin(), priorIds_.end(), id) == priorIds_.end())
            JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] 高精度观测 id={} 无先验位置（setExistingPrior 未注入），"
                         "不加软先验", id);

    // —— 步骤 3：GBA 批算（位姿参数化在算子内：R_init/t_init 直喂，内部转四元数）——
    if (cb) cb(20, "GBA 批算");
    calib::GlobalBAResult gba;
    try {
        if (gbaFn_) {
            gba = gbaFn_(in, gbaParams());
        } else {
            calib::GlobalBundleAdjustmentCPU op(gbaParams());
            gba = op.Execute(in);
        }
    } catch (const std::exception& e) {
        gba.success = false;
        gba.message = std::string("GBA 异常: ") + e.what();
    } catch (...) {
        gba.success = false;
        gba.message = "GBA 未知异常";
    }
    out_.gbaSuccess = gba.success;
    out_.gbaMessage = gba.success ? std::string() : gba.message;
    out_.gbaStats = gba.statistics;
    out_.gbaMarkers = std::move(gba.optimizedMarkers);

    if (!gba.success) {
        // 设计 §4.4 降级路径：Fault 上报 + 沿初值位姿重融合兜底
        quality = Scanner::QualityFlag::Degraded;
        if (sink_)
            sink_->report(Scanner::QualityFlag::Fault, kEvtGbaFail,
                          "GBA 失败: " + gba.message + " —— 沿初值位姿重融合兜底");
    }

    // —— 取消检查点（GBA 后）：安全退出（解冻恢复、不重融合/不推送/不入库）——
    if (cancel.cancelled()) {
        out_.cancelled = true;
        out_.quality = Scanner::QualityFlag::Degraded;
        return Scanner::Result::degraded("GlobalOptim 已取消（GBA 后检查点），未做重融合");
    }

    // —— 位姿数组（GBA 修正 / 失败初值兜底；序=观测快照序，按 frameId 对齐）——
    std::unordered_map<uint64_t, const calib::FramePose*> poseById;
    if (gba.success)
        for (const auto& p : gba.optimizedPoses) poseById[p.frameId] = &p;
    out_.poses.resize(snap.obs.size());
    for (size_t i = 0; i < snap.obs.size(); ++i) {
        auto& pose = out_.poses[i];
        pose.frameId = snap.obs[i].frameId;
        const calib::FramePose* fp = nullptr;
        if (gba.success) {
            auto it = poseById.find(snap.obs[i].frameId);
            if (it != poseById.end()) fp = it->second;
        }
        if (fp) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) pose.R[3 * r + c] = fp->R(r, c);
            for (int k = 0; k < 3; ++k) pose.t[k] = fp->t(k);
        } else {                                   // 兜底：无修正可用=初值
            for (int k = 0; k < 9; ++k) pose.R[k] = snap.obs[i].R_init[k];
            for (int k = 0; k < 3; ++k) pose.t[k] = snap.obs[i].t_init[k];
        }
    }

    // —— 步骤 4a：marker 重融合（新实例按修正位姿重放全部观测）——
    if (cb) cb(50, "marker 重融合");
    size_t markerFuseFails = 0;
    {
        calib::MarkerCloudFuseCPU mFuse;           // 新实例=重放语义（与 C 线默认参数一致）
        for (size_t i = 0; i < snap.obs.size(); ++i) {
            if (cancel.cancelled()) break;         // 帧间检查点
            const auto& fo = snap.obs[i];
            if (fo.markerObs.empty()) continue;
            std::vector<calib::MarkerFuseInput> pts(fo.markerObs.size());
            for (size_t k = 0; k < fo.markerObs.size(); ++k) {
                pts[k].x = static_cast<float>(fo.markerObs[k].xyz[0]);
                pts[k].y = static_cast<float>(fo.markerObs[k].xyz[1]);
                pts[k].z = static_cast<float>(fo.markerObs[k].xyz[2]);
                // 法线缺省 (0,0,1)、半径 0：obs 未携带法线/半径（重融合仅位置，如实注明）
            }
            auto r = mFuse.Execute(pts, matxFromArr9(out_.poses[i].R),
                                   vec3FromArr3(out_.poses[i].t));
            if (!r.success) {
                ++markerFuseFails;
                JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] marker 重放融合失败 frameId={}: {}",
                             fo.frameId, r.message);
            }
        }
        out_.markerCloud = mFuse.GetFusedPoints();
    }
    if (markerFuseFails > 0) quality = Scanner::QualityFlag::Degraded;

    if (cancel.cancelled()) {
        out_.cancelled = true;
        out_.quality = Scanner::QualityFlag::Degraded;
        return Scanner::Result::degraded("GlobalOptim 已取消（marker 重融合帧间检查点），"
                                         "部分结果可用");
    }

    // —— 步骤 4b：激光重融合重放（未降级才重放；JMW_BUILD_CUDA 守卫）——
#ifdef JMW_BUILD_CUDA
    bool anyLaser = false;
    for (const auto& fo : snap.obs)
        if (fo.laserCacheSlot != FrameObs::kNoLaserSlot) { anyLaser = true; break; }
    if (anyLaser) {
        if (obsAcc.degradedLaser()) {
            // 近似路径：无激光修正重融合——沿 C 线初值融合结果不重算激光（简化，如实注明）
            out_.laserDegraded = true;
            quality = Scanner::QualityFlag::Degraded;
            if (sink_)
                sink_->report(Scanner::QualityFlag::Degraded, kEvtLaserNoReplay,
                              "激光帧缓存降级：无激光修正重融合（沿初值融合结果不重算）");
        } else {
            if (cb) cb(70, "laser 重融合重放");
            replayLaser_ = laserFactory_ ? laserFactory_()
                                         : std::make_unique<RealLaserReplayAdapter>();
            size_t laserFuseFails = 0;              // 融合失败/异常帧聚合（I1）
            for (size_t i = 0; i < snap.obs.size(); ++i) {
                if (cancel.cancelled()) break;     // 帧间检查点
                const size_t slot = snap.obs[i].laserCacheSlot;
                if (slot == FrameObs::kNoLaserSlot) continue;   // 本帧无激光（A 帧/空帧）
                if (slot >= snap.laserFrames.size()) continue;  // 防御（契约不会发生）
                if (!replayLaser_->fuse(snap.laserFrames[slot], out_.poses[i].R,
                                        out_.poses[i].t))
                    ++laserFuseFails;
            }
            if (!cancel.cancelled()) {
                out_.laserReplayed = true;
                out_.laserCtx = replayLaser_->deviceContext();
            }
            if (laserFuseFails > 0) {
                // 部分帧融合失败：修正后激光点云可能不全 → Degraded（每 run 上报一次，
                // 失败不中断重放——后续帧照常尝试）
                quality = Scanner::QualityFlag::Degraded;
                if (sink_)
                    sink_->report(Scanner::QualityFlag::Degraded, kEvtLaserReplayFail,
                                  "激光重放融合部分失败 " + std::to_string(laserFuseFails) +
                                      "/" + std::to_string(snap.obs.size()) +
                                      " 帧（修正后激光点云可能不全）");
            }
        }
    }
#endif

    if (cancel.cancelled()) {
        out_.cancelled = true;
        out_.quality = Scanner::QualityFlag::Degraded;
        return Scanner::Result::degraded("GlobalOptim 已取消（激光重放帧间检查点），"
                                         "部分结果可用");
    }

    // —— 步骤 5：点云入库（句柄 tag 形态交接）+ 完成事件 ——
    if (cb) cb(90, "点云入库");
    if (cloudRepo_) {
        const std::string tag = std::string("globaloptim|frames=") +
                                std::to_string(out_.frameCount) + "|markers=" +
                                std::to_string(out_.markerCloud.size()) + "|laserReplayed=" +
                                (out_.laserReplayed ? "1" : "0") + "|quality=" + qualityName(quality);
        bool wrote = false;
        std::string writeErr;
        try {
            wrote = cloudRepo_->write(tag);
        } catch (const std::exception& e) {
            writeErr = e.what();
        } catch (...) {
            writeErr = "unknown exception";
        }
        if (!wrote && sink_)
            sink_->report(Scanner::QualityFlag::Fault, kEvtRepoWriteFail,
                          "点云入库失败" + (writeErr.empty() ? std::string() : ": " + writeErr));
    }
    out_.quality = quality;
    if (sink_)
        sink_->reportCompletion(quality, kEvtDone,
                                std::string("scan final | GBA ") +
                                    (out_.gbaSuccess ? "ok" : "fail(fallback)") +
                                    " | frames=" + std::to_string(out_.frameCount) +
                                    " markers=" + std::to_string(out_.markerCloud.size()) +
                                    " laserReplayed=" + (out_.laserReplayed ? "1" : "0"));

    // —— 步骤 6：解冻 + 推送修正点云 ——
    if (sceneFeed_) {
        sceneFeed_->notifyFreeze(false);
        unfreeze.dismissed = true;
        CloudViewHandle h;
        h.hostMarker = &out_.markerCloud;   // 稳定存储（out_ 成员，存活至下次 run）
        h.deviceLaser = nullptr;            // 激光句柄经 output().laserCtx（T17 细化同源）
        sceneFeed_->pushCloudSnapshot(h);
    } else {
        unfreeze.dismissed = true;          // sceneFeed 空 → 从未冻结（guard 空指针安全）
    }

    // 批算结果汇总（一次性——质量定责的最终账本：GBA 成败/迭代/RMSE/重放/降级）
    JMW_LOG_INFO("07-GlobalOptim",
        "[GlobalOptim] 完成: GBA={} 迭代={} 初RMSE={:.4f} 末RMSE={:.4f} 帧数={} "
        "marker云={} 激光重放={} 激光降级={} 取消={} 质量={}",
        out_.gbaSuccess, out_.gbaStats.ceresIterations,
        out_.gbaStats.initialRMSE, out_.gbaStats.finalRMSE, out_.frameCount,
        out_.markerCloud.size(), out_.laserReplayed, out_.laserDegraded,
        out_.cancelled, qualityName(quality));

    if (cb) cb(100, "global optim done");
    return quality == Scanner::QualityFlag::Normal
               ? Scanner::Result::ok("global optim done")
               : Scanner::Result::degraded("global optim done (degraded)");
}

// ============================================================================
// IPipelineObject 适配：start=后台 run（attachObs 观测源）；stop=cancel+join
// ============================================================================
void GlobalOptimObject::attachObs(FrameObsAccumulator& obs) { attachedObs_ = &obs; }

Scanner::Result GlobalOptimObject::start() {
    std::lock_guard<std::mutex> lock(runMutex_);
    if (running_.load())
        return Scanner::Result::fail("global optim already running");
    if (!configured_)
        return Scanner::Result::fail("GlobalOptimObject::start: 须先 configure");
    if (!attachedObs_)
        return Scanner::Result::fail("GlobalOptimObject::start: 须先 attachObs(...)");
    if (worker_.joinable()) worker_.join();

    auto token = std::make_shared<CancelToken>();
    cancelToken_ = token;                    // 管道持有；worker 捕获共享保活
    FrameObsAccumulator* obs = attachedObs_;
    running_.store(true);                    // 先置位（runLocked 的 guard 负责清位）
    try {
        worker_ = std::thread([this, token, obs] { runLocked(*obs, nullptr, *token); });
    } catch (...) {                          // 线程构造失败：复位标志/令牌，不留卡死态
        running_.store(false);
        cancelToken_.reset();
        return Scanner::Result::fail("global optim failed to start worker thread");
    }
    return Scanner::Result::ok("global optim started (background run)");
}

void GlobalOptimObject::stop() {
    // 锁内取令牌共享引用再 cancel：与 start() 重建令牌互斥（消 UAF 窗口，同 B 模式）
    std::shared_ptr<CancelToken> token;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        token = cancelToken_;
    }
    if (token) token->cancel();
    std::thread local;
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        local = std::move(worker_);
    }
    if (local.joinable()) local.join();
}

bool GlobalOptimObject::isRunning() const { return running_.load(); }

} // namespace Scanner::pipeline
