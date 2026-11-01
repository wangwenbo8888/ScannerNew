// ============================================================================
// GlobalOptimObject.cpp — D 全局优化对象实现
//
// 步骤编排（设计方案 §4.4，细节见 GlobalOptimObject.h 文件头）：
//   freeze → 装配 GBA 输入（软先验合并）→ GBA → marker/laser 重融合重放 →
//   入库 + 完成事件 → 解冻推送。GBA 失败=Fault 上报+初值兜底（Degraded）；
//   激光缓存降级=不重放（Degraded 近似）；取消=检查点安全退出。
// ============================================================================
#include "pipelines/globaloptim/GlobalOptimObject.h"
#include <cuda_runtime.h>   // cudaMemcpy（终版激光 host 下载）

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
/// 261029 体素参数化：与 C 线会话融合同参（Config.laserVoxelSizeMm——02 注入）
class RealLaserReplayAdapter final : public ILaserReplayFuse {
public:
    explicit RealLaserReplayAdapter(float voxelSizeMm)
        : fuse_([voxelSizeMm] {
              calib::LaserCloudFuseCUDAParams fp;
              fp.voxelSize = voxelSizeMm;
              return fp;
          }()) {}
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

    // 终版融合云 host 下载（260920 完善流程：优化后激光入仓/显示——透传 09
    // 融合云 d_fusedXyz 设备下载，同 ScanPipeline::downloadFusedXyz 口径）
    std::vector<float> downloadFusedXyz() const override {
        const auto ctx = fuse_.GetDeviceContext();
        if (!ctx.d_fusedXyz || ctx.fusedPointCount == 0) return {};
        std::vector<float> host(ctx.fusedPointCount * 3);
        if (cudaMemcpy(host.data(), ctx.d_fusedXyz,
                       host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] laser 终版融合云下载失败");
            return {};
        }
        return host;
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
    p.maxSolveSecondsPerPhase = cfg_.maxSolveSecondsPerPhase;   // 261004 加固透传
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
    // 261030 关键帧子采样：GBA 在 >~6K 帧病态（Ceres numeric factorization
    // 失败——15025 帧会话 fail(fallback) 无重放实证）。超限只解关键帧（每
    // stride 帧取一，含首尾），非关键帧位姿在步骤 4 前由「关键帧修正 ∘
    // 初始相对位姿」传递（C_k·T_init_i——标准关键帧位姿图手法）
    const size_t kfCap = cfg_.gbaKeyframeCap > 0
        ? static_cast<size_t>(cfg_.gbaKeyframeCap) : SIZE_MAX;
    const size_t nObsAll = snap.obs.size();
    const size_t kfStride = nObsAll > kfCap ? (nObsAll + kfCap - 1) / kfCap : 1;
    if (kfStride > 1)
        JMW_LOG_WARN("07-GlobalOptim",
            "[GlobalOptim] 关键帧子采样: 观测 {} 帧 > 上限 {} → stride={}（解 {} 关键帧，"
            "非关键帧位姿按 C_k·T_init_i 传递）",
            nObsAll, kfCap, kfStride, (nObsAll + kfStride - 1) / kfStride);
    in.frames.reserve(nObsAll / kfStride + 1);
    std::unordered_set<int> obsHpIds;
    for (size_t i = 0; i < nObsAll; i += kfStride) {
        const auto& fo = snap.obs[i];
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
    // 261004：标点会话跳过批算（Config.skipBatchSolve——根因见字段注释：4888 帧
    // 稠密闭环边使 PGO 单迭代分钟级，时间上限只在迭代间生效拦不住）。跳过＝
    // 直接走 !gba.success 降级路径（初值位姿＋marker 重融合），不报 Fault 只
    // WARN（有意为之，非故障）；质量如实标 Degraded
    if (cfg_.skipBatchSolve) {
        gba.success = false;
        gba.message = "标点会话跳过批算（初值位姿重融合）";
        JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] {}", gba.message);
    } else
    try {
        // 261004 加固：把 run() 的 CancelToken 接进 Ceres 迭代回调——取消从
        // 「GBA 后检查点才可见」提前到「一次迭代内可见」（秒级打断）。此前
        // Solve() 黑盒不可打断：真机 2020 帧会话卡 5 分钟+，用户关程序后
        // AppContext 析构 join 收尾线程 → 整个进程挂死在优化器里被 Task 杀。
        // 打断后 Ceres 以 USER_FAILURE 终止 → 下方 gba.success=false → 沿初值
        // 兜底，随后的取消检查点（原 261004 前已有的「GBA 后检查点」）干净退出。
        calib::GlobalBAParams gp = gbaParams();
        gp.cancelRequested = [&cancel]() { return cancel.cancelled(); };
        // 261017 批算进度：算子逐迭代回调 [0,1] → 映射 20%~48%（此后 marker/激光
        // 重放/入库占 50-100%）——百秒级 Ceres 黑盒期进度条不再停在 20%
        gp.progressCb = [cb](double f) {
            if (cb) cb(20 + static_cast<int>(28.0 * f), "GBA 批算");
        };
        if (gbaFn_) {
            gba = gbaFn_(in, gp);
        } else {
            calib::GlobalBundleAdjustmentCPU op(gp);
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
        // 261017 开发期日志补全：失败原因第一时间落日志（原只进事件不上日志——
        // 真机「优化后点数不变」排障时全靠翻代码猜）
        JMW_LOG_WARN("07-GlobalOptim",
            "[GlobalOptim] GBA 批算失败（迭代={} 初RMSE={:.4f} 末RMSE={:.4f} 输入帧={}）: {}",
            gba.statistics.ceresIterations, gba.statistics.initialRMSE,
            gba.statistics.finalRMSE, snap.obs.size(), gba.message);
        // 设计 §4.4 降级路径：Fault 上报 + 沿初值位姿重融合兜底
        // （skipBatchSolve 有意跳过＝非故障，仅 WARN 不报 Fault——261004）
        quality = Scanner::QualityFlag::Degraded;
        if (sink_ && !cfg_.skipBatchSolve)
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
    // 261030 信任帧判定＋修正传递全覆盖：无标志点观测（<2 个）的帧不被
    // GBA 约束——解算器留其坏初值 → 位姿链 60-100mm 跳变簇＝终版 35% 稠密
    // 错位块根因（22:16 会话实测）。对不信任帧：用最近前置信任帧的修正
    // C_k=T_opt_k·T_init_k⁻¹ 传递（保段内运动初值，套全局修正）
    std::vector<int> trustedIdx;          // 信任帧索引（升序）
    trustedIdx.reserve(snap.obs.size());
    if (gba.success)
        for (size_t i = 0; i < snap.obs.size(); ++i) {
            int obsN = 0;
            for (const auto& mo : snap.obs[i].markerObs)
                if (mo.globalId >= 0) ++obsN;
            if (obsN >= 2) trustedIdx.push_back(static_cast<int>(i));
        }
    auto nearestTrusted = [&](size_t i) -> int {
        // 二分：≤i 的最大信任帧索引；无（i 在首信任帧前）→ -1
        int lo = 0, hi = static_cast<int>(trustedIdx.size()) - 1, ans = -1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            if (trustedIdx[static_cast<size_t>(mid)] <= static_cast<int>(i)) {
                ans = mid; lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans >= 0 ? trustedIdx[static_cast<size_t>(ans)] : -1;
    };
    out_.poses.resize(snap.obs.size());
    for (size_t i = 0; i < snap.obs.size(); ++i) {
        auto& pose = out_.poses[i];
        pose.frameId = snap.obs[i].frameId;
        const calib::FramePose* fp = nullptr;
        int obsN = 0;
        for (const auto& mo : snap.obs[i].markerObs)
            if (mo.globalId >= 0) ++obsN;
        const bool trusted = gba.success && obsN >= 2;
        if (trusted) {
            auto it = poseById.find(snap.obs[i].frameId);
            if (it != poseById.end()) fp = it->second;
        }
        if (fp) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) pose.R[3 * r + c] = fp->R(r, c);
            for (int k = 0; k < 3; ++k) pose.t[k] = fp->t(k);
        } else if (gba.success) {
            // 不信任帧（观测 <2 或缺解）：最近前置信任帧 k 的修正传递
            // pose_i = C_k · T_init_i；无前置信任帧＝初值兜底
            const int k = nearestTrusted(i);
            bool done = false;
            if (k >= 0) {
                auto itk = poseById.find(snap.obs[static_cast<size_t>(k)].frameId);
                if (itk != poseById.end()) {
                    const cv::Matx33d Rok(itk->second->R);
                    const cv::Vec3d tok(itk->second->t);
                    const cv::Matx33d Rik(snap.obs[static_cast<size_t>(k)].R_init);
                    const cv::Vec3d tik(snap.obs[static_cast<size_t>(k)].t_init);
                    const cv::Matx33d Rc = Rok * Rik.t();
                    const cv::Vec3d tc = tok - Rc * tik;
                    const cv::Matx33d Rii(snap.obs[i].R_init);
                    const cv::Vec3d tii(snap.obs[i].t_init);
                    const cv::Matx33d Ri = Rc * Rii;
                    const cv::Vec3d ti = Rc * tii + tc;
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) pose.R[3 * r + c] = Ri(r, c);
                    for (int q = 0; q < 3; ++q) pose.t[q] = ti(q);
                    done = true;
                }
            }
            if (!done) {                               // 无信任锚=初值兜底
                for (int q = 0; q < 9; ++q) pose.R[q] = snap.obs[i].R_init[q];
                for (int q = 0; q < 3; ++q) pose.t[q] = snap.obs[i].t_init[q];
            }
        } else {                                   // 兜底：无修正可用=初值
            for (int k = 0; k < 9; ++k) pose.R[k] = snap.obs[i].R_init[k];
            for (int k = 0; k < 3; ++k) pose.t[k] = snap.obs[i].t_init[k];
        }
    }

    // —— 步骤 4a：marker 终版云——GBA 成功直接采用 BA 最优解（261017：原一律体素
    // 重放——重放无 id 概念，同一标志点逐帧观测散布跨体素边界即分裂＋链断观测
    // （globalId<0）入云，真机 30 真实点重放出 52；optimizedMarkers 是逐 globalId
    // 的 BA 最优位置，直接就是答案）。仅 GBA 失败（或解空）兜底走 5mm 体素重放
    // （沿初值位姿，与 C 线同参）
    if (cb) cb(50, gba.success ? "marker BA 最优解" : "marker 重融合兜底");
    size_t markerFuseFails = 0;
    if (gba.success && !out_.gbaMarkers.empty()) {
        // 按共视数降序拷贝：链断重锚的重复 id 中观测多者（id 可信）先入列——
        // 后续近距合并保留先到者，即保留可信 id 的 BA 位置
        auto byCovis = out_.gbaMarkers;
        std::sort(byCovis.begin(), byCovis.end(),
                  [](const calib::GlobalMarker& a, const calib::GlobalMarker& b) {
                      return a.covisCount > b.covisCount;
                  });
        out_.markerCloud.clear();
        out_.markerCloud.reserve(byCovis.size());
        for (const auto& m : byCovis) {
            calib::MarkerCloudPoint p;
            p.x = static_cast<float>(m.X.x);
            p.y = static_cast<float>(m.X.y);
            p.z = static_cast<float>(m.X.z);
            out_.markerCloud.push_back(p);
        }
        JMW_LOG_INFO("07-GlobalOptim",
            "[GlobalOptim] marker 终版云=GBA 最优解 {} 点（免体素重放；id 数≠物理点数——"
            "链断重锚的重复 id 仍待近距合并）",
            out_.markerCloud.size());
    } else {
        if (gba.success)
            JMW_LOG_WARN("07-GlobalOptim",
                "[GlobalOptim] GBA 成功但 optimizedMarkers 空——回退体素重放");
    {
        calib::MarkerCloudFuseCPUParams mfp;
        mfp.voxelSize = 5.0f;   // 261017 与 C 线对齐（ScanPipeline 会话装配同值）——原默认
                                // 构造 0.5mm：重放时同一标志点的逐帧观测（亚毫米噪声）
                                // 各自成点 → 全局优化后 marker 数爆涨为观测量级
        calib::MarkerCloudFuseCPU mFuse{mfp};  // 新实例=重放语义（参数与 C 线一致）
        for (size_t i = 0; i < snap.obs.size(); ++i) {
            if (cancel.cancelled()) break;         // 帧间检查点
            if (cb && (i % 128) == 0)              // 261017 逐段进度（50→68%）
                cb(50 + static_cast<int>(18.0 * static_cast<double>(i) /
                                         static_cast<double>(snap.obs.size())),
                   "marker 重融合兜底");
            const auto& fo = snap.obs[i];
            if (fo.markerObs.empty()) continue;
            std::vector<calib::MarkerFuseInput> pts(fo.markerObs.size());
            for (size_t k = 0; k < fo.markerObs.size(); ++k) {
                pts[k].x = static_cast<float>(fo.markerObs[k].xyz[0]);
                pts[k].y = static_cast<float>(fo.markerObs[k].xyz[1]);
                pts[k].z = static_cast<float>(fo.markerObs[k].xyz[2]);
                // 法线缺省 (0,0,1)、半径 0：obs 未携带法线/半径（重融合仅位置，如实注明）
            }
            // 帧级兜异常（260927 真机实证：GBA 降级初值位姿下 marker 变换坐标
            // 可越体素 21 位界→Execute 抛异常炸掉整个终局遍——弹窗永卡 50%）：
            // 逐帧 catch 计入 markerFuseFails 降级续跑（与激光重放同口径）
            try {
                auto r = mFuse.Execute(pts, matxFromArr9(out_.poses[i].R),
                                       vec3FromArr3(out_.poses[i].t));
                if (!r.success) {
                    ++markerFuseFails;
                    JMW_LOG_WARN("07-GlobalOptim", "[GlobalOptim] marker 重放融合失败 frameId={}: {}",
                                 fo.frameId, r.message);
                }
            } catch (const std::exception& e) {
                ++markerFuseFails;
                JMW_LOG_WARN("07-GlobalOptim",
                    "[GlobalOptim] marker 重放融合异常 frameId={}: {}（计降级续跑）",
                    fo.frameId, e.what());
            } catch (...) {
                ++markerFuseFails;
                JMW_LOG_WARN("07-GlobalOptim",
                    "[GlobalOptim] marker 重放融合未知异常 frameId={}（计降级续跑）", fo.frameId);
            }
        }
        out_.markerCloud = mFuse.GetFusedPoints();
    }
    }   // ← else（GBA 失败兜底体素重放）结束
    if (markerFuseFails > 0) quality = Scanner::QualityFlag::Degraded;

    // —— 261017 终版云近距合并：链断重锚的重复 id（同一物理点两个 globalId，BA
    // 后位置几乎重合——RMSE 0.013mm 级）在此归一。阈值 10mm << 标志点间距 30mm
    // 安全；GBA 路径已按共视数降序＝可信 id 位置先入列被保留。真机对账：30 物理
    // 点此前显示 34（+4 幽灵 id）即此因。合并对数落日志（幽灵点/误检对账依据：
    // 若合并后仍 >物理点数，余量为误检假点——须上游检测质量解决）
    {
        constexpr float kMergeRadiusMm = 10.0f;
        const float r2 = kMergeRadiusMm * kMergeRadiusMm;
        std::vector<calib::MarkerCloudPoint> kept;
        kept.reserve(out_.markerCloud.size());
        size_t mergedCnt = 0;
        for (const auto& p : out_.markerCloud) {
            bool dup = false;
            for (const auto& q : kept) {
                const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
                if (dx * dx + dy * dy + dz * dz < r2) { dup = true; break; }
            }
            if (dup) ++mergedCnt; else kept.push_back(p);
        }
        if (mergedCnt > 0) {
            JMW_LOG_INFO("07-GlobalOptim",
                "[GlobalOptim] 终版云近距合并：{} 点并入邻点（阈值 {}mm）→ 保留 {} 点",
                mergedCnt, kMergeRadiusMm, kept.size());
            out_.markerCloud = std::move(kept);
        }
    }

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
            // 261017 开发期日志补全：降级跳过时把账面数字落日志（原只有事件码）
            JMW_LOG_WARN("07-GlobalOptim",
                "[GlobalOptim] 激光缓存已降级：缓存字节={} 缓存帧={} ——跳过激光修正重融合"
                "（沿初值融合结果不重算；预算见 ScanWorkflow laserCacheBudgetMB）",
                obsAcc.laserBytesUsed(), snap.laserFrames.size());
            out_.laserDegraded = true;
            quality = Scanner::QualityFlag::Degraded;
            if (sink_)
                sink_->report(Scanner::QualityFlag::Degraded, kEvtLaserNoReplay,
                              "激光帧缓存降级：无激光修正重融合（沿初值融合结果不重算）");
        } else {
            if (cb) cb(70, "laser 重融合重放");
            replayLaser_ = laserFactory_ ? laserFactory_()
                                         : std::make_unique<RealLaserReplayAdapter>(
                                               cfg_.laserVoxelSizeMm);
            size_t laserFuseFails = 0;              // 融合失败/异常帧聚合（I1）
            for (size_t i = 0; i < snap.obs.size(); ++i) {
                if (cancel.cancelled()) break;     // 帧间检查点
                if (cb && (i % 64) == 0)           // 261017 逐段进度（70→84%）
                    cb(70 + static_cast<int>(14.0 * static_cast<double>(i) /
                                             static_cast<double>(snap.obs.size())),
                       "laser 重融合重放");
                const size_t slot = snap.obs[i].laserCacheSlot;
                if (slot == FrameObs::kNoLaserSlot) continue;   // 本帧无激光（A 帧/空帧）
                if (slot >= snap.laserFrames.size()) continue;  // 防御（契约不会发生）
                // 帧级兜异常（同 marker 重放口径——260927：异常帧计失败降级，不炸终局遍）
                try {
                    if (!replayLaser_->fuse(snap.laserFrames[slot], out_.poses[i].R,
                                            out_.poses[i].t))
                        ++laserFuseFails;
                } catch (const std::exception& e) {
                    ++laserFuseFails;
                    JMW_LOG_WARN("07-GlobalOptim",
                        "[GlobalOptim] 激光重放融合异常 frameId={}（slot={}）: {}（计失败续跑）",
                        snap.obs[i].frameId, slot, e.what());
                } catch (...) {
                    ++laserFuseFails;
                    JMW_LOG_WARN("07-GlobalOptim",
                        "[GlobalOptim] 激光重放融合未知异常（slot={}，计失败续跑）", slot);
                }
            }
            if (!cancel.cancelled()) {
                out_.laserReplayed = true;
                out_.laserCtx = replayLaser_->deviceContext();
                // 终版激光融合云 host 下载（260920 完善流程：入仓/显示——
                // run 返回后 replayLaser_ 可能随下次 run 重建，host 拷贝长存）
                if (cb) cb(85, "终版激光下载");
                out_.laserXyzFinal = replayLaser_->downloadFusedXyz();
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
