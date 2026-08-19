// ============================================================================
// PosturePipeline.cpp — A 姿态判断流水线对象总装实现（生命周期与收口见头文件注）
//
// 实现要点（形态对齐 ScanChains/ScanPipeline 先例）：
//   - 算子集 PostureLaneOps 按 lane 惰性创建一次、跨帧复用（09 算子"每实例
//     非线程安全"）；gpuChain 首帧创建（先于 runtime 兜底提交，pChain 只读）；
//     立体矫正/投影矩阵用初始参数组在 makeOps 一次性 Set（A 无帧快照）
//   - gpuChain 不调 frontReady（A 无 ccl 就绪并行需求）：runtime 兜底提交，
//     GPU 前段与 P 链天然串行
//   - 配准在 eFinalize（E 核终段）：prevState 最新快照模型（首帧 null 初始化
//     → optical_flow_fuse → frame_fuse 兜底 → 沿用快照），同姿态调度方案 §5
//   - 收口序列由 watcher 线程执行（onComplete 在 lane 线程内仅置停+唤醒，
//     drain 需 join lanes——在 lane 内 join 自身=死锁）
// ============================================================================
#include "pipelines/posture/PosturePipeline.h"

#include <algorithm>
#include <utility>

#include <spdlog/spdlog.h>

#include "pipelines/ISceneFeed.h"

// ---- 09 GPU 前段算子（公开头 pImpl 隔离，不含 CUDA 类型）----
#include "core/vision/ccl/region_analyze_cuda.h"
#include "core/vision/frame_filter/frame_filter_cuda.h"
#include "core/vision/mask_extract/mask_extract_cuda.h"
// ---- 09 P 核标记点链算子 ----
#include "core/marker/edge_match/edge_match_cpu.h"
#include "core/marker/ellipse_fit/ellipse_fit_cpu.h"
#include "core/marker/epipolar_intersect/epipolar_intersect_cpu.h"
#include "core/marker/frame_fuse/frame_fuse_cpu.h"
#include "core/marker/image_merge/image_merge_cpu.h"
#include "core/marker/image_split/image_split_cpu.h"
#include "core/marker/marker_match/marker_match_cpu.h"
#include "core/marker/optical_flow_fuse/marker_optical_flow_fuse_cpu.h"
#include "core/marker/point_reconstruct/point_reconstruct_cpu.h"
#include "core/marker/undistort_cpu/undistort_points_cpu.h"
#include "core/marker/zernike_edge/zernike_edge_cpu.h"
// ---- 09 姿态估计（E 段）----
#include "calibration/posture/pose_estimate/pose_estimate_cpu.h"

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace Scanner::pipeline {
// ============================================================================
// PostureLaneOps — per-lane 算子集（同 lane 内串行复用，跨 lane 各自独占）
// ============================================================================
struct PostureLaneOps {
#ifdef JMW_BUILD_CUDA
    // GPU 前段（E 核线程独占；L/R 各持独立实例）
    std::unique_ptr<calib::MaskExtractCUDA> maskL, maskR;
    std::unique_ptr<calib::FrameFilterCUDA> filterL, filterR;
    std::unique_ptr<calib::RegionAnalyzerCUDA> cclL, cclR;
#endif
    // P 核标记点链（P worker 独占）
    std::unique_ptr<calib::ImageSplitCPU> split;
    std::unique_ptr<calib::ZernikeEdgeCPU> zernike;
    std::unique_ptr<calib::ImageMergeCPU> merge;
    std::unique_ptr<calib::MarkerUndistortCPU> undistCpu;
    std::unique_ptr<calib::EllipseFitCPU> ellipse;
    std::unique_ptr<calib::MarkerMatchCPU> match;
    std::unique_ptr<calib::EpipolarIntersectCPU> epiIntersect;
    std::unique_ptr<calib::EdgeMatchCPU> edgeMatch;
    std::unique_ptr<calib::PointReconstructCPU> reconstruct;
    // E 核终段（lane E 线程独占：eFinalize）
    std::unique_ptr<calib::MarkerOpticalFlowFuseCPU> flowFuse;
    std::unique_ptr<calib::FrameFuseCPU> frameFuse;
    std::unique_ptr<calib::PoseEstimateCPU> poseEstimate;
};

namespace {

constexpr int32_t kEvtRuntimeDied = 1701;        // runtime 异常即停/lanes 全退（非收口/用户 stop）
constexpr int kPostureGpuSlots = 1;              // 姿态 GPU 槽（设计 §1.2：GPU 活轻）

cv::Matx33d matxFromArr9(const double* a) {
    return cv::Matx33d(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}
cv::Vec3d vec3FromArr3(const double* a) { return cv::Vec3d(a[0], a[1], a[2]); }

// K/D Mat → 标量参数（越界补 0；D 顺序 [k1,k2,p1,p2,(k3)]）
double atOr(const cv::Mat& m, int idx, double def = 0.0) {
    if (m.empty() || static_cast<int>(m.total()) <= idx) return def;
    cv::Mat d;
    m.convertTo(d, CV_64F);
    return d.reshape(1, 1).at<double>(0, idx);
}

// ---------------------------------------------------------------------------
// 标记点链 9 算子（初始参数已随 makeOps Set 到算子上）。
// 返回 false=本周期无有效标记点（调用方按降级处理，不销毁周期）
// ---------------------------------------------------------------------------
bool runMarkerChain(const data::CycleUnit& cycle, PostureLaneOps& ops,
                    const PostureFront& front, PostureFrameResult& result) {
    auto runCamera = [&](const cv::Mat& gray, const std::vector<cv::Rect>& rois,
                         calib::ImageMergeCPUResult& merged) -> bool {
        auto sp = ops.split->Execute(gray, rois);
        if (!sp.success || sp.splitCount == 0) return false;
        std::vector<std::vector<calib::EdgePoint>> edges(
            static_cast<size_t>(sp.splitCount));
        for (int i = 0; i < sp.splitCount; ++i) {
            auto zr = ops.zernike->Execute(sp.splitImages[static_cast<size_t>(i)]);
            if (zr.success) edges[static_cast<size_t>(i)] = std::move(zr.edgePoints);
        }
        merged = ops.merge->Execute(edges, rois);
        return merged.success && merged.mergedEdgeCount > 0;
    };

    calib::ImageMergeCPUResult mL, mR;
    if (!runCamera(cycle.markerL, front.roisL, mL) ||
        !runCamera(cycle.markerR, front.roisR, mR))
        return false;

    auto ur = ops.undistCpu->Execute(mL.mergedEdgePoints, mR.mergedEdgePoints,
                                     mL.groupIds, mR.groupIds);
    if (!ur.success) return false;

    // ellipse_fit 逐组（矫正坐标系；中心即 2-7 输出）
    auto fitGroups = [&](const std::vector<std::vector<cv::Point2d>>& groups,
                         std::vector<calib::EllipseFitCPUResult>& out) {
        for (const auto& g : groups) {
            auto e = ops.ellipse->Execute(g);
            if (e.success) out.push_back(std::move(e));
        }
    };
    std::vector<calib::EllipseFitCPUResult> ellL, ellR;
    fitGroups(ur.splitRectifiedPoints1ByGroup(), ellL);
    fitGroups(ur.splitRectifiedPoints2ByGroup(), ellR);
    if (ellL.size() < 3 || ellR.size() < 3) return false;   // 重建最少点数

    std::vector<cv::Point2f> cL, cR;
    cL.reserve(ellL.size());
    cR.reserve(ellR.size());
    for (auto& e : ellL) cL.push_back(e.centerPoint2f());
    for (auto& e : ellR) cR.push_back(e.centerPoint2f());

    auto mm = ops.match->Execute(cL, cR);
    if (!mm.success) return false;

    auto eiL = ops.epiIntersect->Execute(ellL);
    auto eiR = ops.epiIntersect->Execute(ellR);
    if (!eiL.success || !eiR.success) return false;

    auto em = ops.edgeMatch->Execute(eiL.ellipseResults, eiR.ellipseResults,
                                     mm.centerMatches);
    if (!em.success) return false;

    auto pr = ops.reconstruct->Execute(em);
    if (!pr.success) return false;

    result.ellipseCentersL = std::move(cL);
    result.ellipseCentersR = std::move(cR);
    for (const auto& m : pr.markerResults) {
        if (!m.validPlane || !m.validCircle) continue;
        result.positions.emplace_back(m.centerX, m.centerY, m.centerZ);
        result.normals.emplace_back(m.normalX, m.normalY, m.normalZ);
    }
    return result.positions.size() >= 3;
}

} // namespace

// ============================================================================
// 构造 / 装配
// ============================================================================
PosturePipeline::PosturePipeline(PostureConfirmTable::Config confirmCfg)
    : confirmCfg_(confirmCfg) {}

PosturePipeline::~PosturePipeline() { stop(); }   // 安全网：幂等

void PosturePipeline::attachRing(Scanner::data::SlotRing<Scanner::data::CycleUnit>& ring) {
    ring_ = &ring;
}

void PosturePipeline::attachTargets(const double (*targets)[16], int count) {
    if (count <= 0 || count > PostureSessionData::kTargetCount) {
        spdlog::error("[PosturePipeline] attachTargets: count={} 非法（须 1..{}）",
                      count, PostureSessionData::kTargetCount);
        return;                                   // 保持未附目标态 → configure fail
    }
    for (int i = 0; i < count; ++i)
        std::copy(targets[i], targets[i] + 16, targets_[i]);
    targetCount_ = count;
    targetsAttached_ = true;
}

void PosturePipeline::attachInitialParams(const PostureInitialParams& params) {
    params_ = params;
    paramsAttached_ = true;
}

void PosturePipeline::setCompletionHook(
    std::function<void(PostureSessionData&&)> continueWithB) {
    continueWithB_ = std::move(continueWithB);
}

void PosturePipeline::attachTestHooks(Hooks hooks) {
    testHooksSet_ = true;
    testHooks_ = std::move(hooks);
}

int PosturePipeline::collectedCount() const {
    return table_ ? table_->collectedCount() : 0;
}

PostureConfirmTable& PosturePipeline::table() {
    return *table_;                               // 前置：已 configure（头注释契约）
}

std::unique_ptr<PipelineEventSink> PosturePipeline::makeSink(
    const PipelineDeps& deps) const {
    return std::make_unique<EventBusEventSink>(deps.eventBus);   // nullptr 安全（no-op）
}

// ============================================================================
// configure — 注入 sink/sceneFeed/acquisition + 建 25 目标确认表
// ============================================================================
Scanner::Result PosturePipeline::configure(const PipelineDeps& deps) {
    if (state_.load() != State::Idle)
        return Result::fail("PosturePipeline::configure: 当前态不可再配置（每对象恰一次）");
    if (testHooksSet_ && paramsAttached_)
        return Result::fail("PosturePipeline::configure: 测试钩子与 attachInitialParams 互斥");
    if (!targetsAttached_ || targetCount_ <= 0)
        return Result::fail("PosturePipeline::configure: 须先 attachTargets（25 目标表）");
    if (!testHooksSet_) {
        if (!paramsAttached_)
            return Result::fail("PosturePipeline::configure: 生产模式须先 "
                                "attachInitialParams（初始内参/畸变/R1/P1）");
        // 参数形状快检（深校验属算子构造期）
        if (params_.K1.empty() || params_.K1.size() != cv::Size(3, 3) ||
            params_.K2.empty() || params_.K2.size() != cv::Size(3, 3) ||
            params_.K1.at<double>(0, 0) <= 0 || params_.K2.at<double>(0, 0) <= 0)
            return Result::fail("PosturePipeline::configure: K1/K2 缺失或非法（须 3x3 且 fx>0）");
        if (params_.D1.empty() || params_.D1.total() < 4 ||
            params_.D2.empty() || params_.D2.total() < 4)
            return Result::fail("PosturePipeline::configure: D1/D2 缺失或元素数 <4");
        if (params_.R1.empty() || params_.R2.empty() || params_.P1.empty() ||
            params_.P2.empty() || params_.Q.empty())
            return Result::fail("PosturePipeline::configure: R1/R2/P1/P2/Q 须齐全");
        if (params_.imageWidth <= 0 || params_.imageHeight <= 0)
            return Result::fail("PosturePipeline::configure: imageWidth/imageHeight 须 >0");
    }

    sink_ = makeSink(deps);
    deps_ = deps;

    // —— 25 目标确认表（集齐回调：置停+唤醒收口 watcher；drain 由 watcher 做）——
    table_ = std::make_unique<PostureConfirmTable>(
        targets_, targetCount_, confirmCfg_,
        [](int poseIdx) {
            spdlog::info("[PosturePipeline] 姿态 {} 确认（渲染切下一目标）", poseIdx);
        },
        [this]() {
            runtime_.requestStop();               // 停抓新周期（原子置位，lane 线程内安全）
            {
                std::lock_guard<std::mutex> lk(watchMu_);
                completionRequested_ = true;
            }
            watchCv_.notify_all();                // 收口序列交 watcher 线程（防 lane 自 join）
        });

    state_.store(State::Configured);
    return Result::ok();
}

// ============================================================================
// start — runtime.start（GPU 1 槽 / sequential=true / queue=null）+ 收口 watcher
// ============================================================================
Scanner::Result PosturePipeline::start() {
    switch (state_.load()) {
        case State::Idle:      return Result::fail("PosturePipeline::start: 须先 configure");
        case State::Running:   return Result::fail("PosturePipeline::start: 已在运行");
        case State::Faulted:   return Result::fail("PosturePipeline::start: 已异常停（会话私有件）");
        case State::Stopped:   return Result::fail("PosturePipeline::start: 已停止（会话私有件，重启建新对象）");
        case State::Configured: break;
    }
    if (!ring_) return Result::fail("PosturePipeline::start: 未 attachRing（Backpressure 周期环必备）");

    source_ = std::make_unique<sched::SequentialSource<Scanner::data::CycleUnit>>(*ring_);
    if (testHooksSet_) {
        // 测试模式：假流工厂（假链不触流；生产链用默认真 CUDA 工厂）
        const auto fr = runtime_.setGpuStreamFactory(
            [this](sched::GpuSlotService::StreamHandle* s) {
                *s = reinterpret_cast<sched::GpuSlotService::StreamHandle>(
                    static_cast<uintptr_t>(++fakeStreamSeq_));
                return 0;
            },
            [](sched::GpuSlotService::StreamHandle) {});
        if (!fr.success) return Result::fail("PosturePipeline::start: " + fr.message);
    }

    sched::SchedConfig cfg;                       // lanes=0 自动探测；gpuSlots=1（GPU 活轻）
    cfg.gpuSlots = kPostureGpuSlots;
    Hooks hooks = testHooksSet_ ? testHooks_ : assembleChains();
    // queue=nullptr（A 无输出队列；eFinalize 自行消费）→ 显式模板实参免推导
    auto rs = runtime_.start<data::CycleUnit, PostureFront, PostureFrameResult>(
        cfg, *source_, /*sequential=*/true, /*queue=*/nullptr, hooks);
    if (!rs.success) {
        source_.reset();
        return Result::fail("PosturePipeline::start: runtime 启动失败: " + rs.message);
    }

    try {
        watcher_ = std::thread([this] { watcherLoop(); });
    } catch (const std::exception& e) {
        runtime_.requestStop();                   // 回退 runtime
        runtime_.drainAndShutdown();
        source_.reset();
        return Result::fail(std::string("PosturePipeline::start: 收口 watcher 线程创建失败: ") +
                            e.what());
    }

    state_.store(State::Running);
    return Result::ok();
}

// ============================================================================
// stop — 同步停：停 watcher（不触发 completion）→ lane 停抓新周期 → 在飞排空
// ============================================================================
void PosturePipeline::stop() {
    if (state_.load() == State::Idle) return;     // 未配置：无物可停
    syncState();                                  // Faulted 一次性上报后再收尾
    {
        std::lock_guard<std::mutex> lk(watchMu_);
        watcherStop_ = true;
    }
    watchCv_.notify_all();
    if (watcher_.joinable()) watcher_.join();     // watcher 在收口中则等其做完
    runtime_.requestStop();                       // 幂等；未 start 过也安全
    runtime_.drainAndShutdown();

    // 迟到补发守卫：stop 与集齐竞态——watcher 先被停信号唤醒退出而 25 已确认时，
    // 此处补发收口（drain 已完、数据完整，SessionData 不丢；采集停于排空之后）
    bool late = false;
    {
        std::lock_guard<std::mutex> lk(watchMu_);
        late = completionRequested_ && !completionFired_;
        if (late) completionFired_ = true;
    }
    if (late) {
        if (deps_.acquisition) deps_.acquisition->stopAcquisition();
        if (continueWithB_ && table_) continueWithB_(table_->takeSessionData());
    }

    state_.store(State::Stopped);
}

bool PosturePipeline::isRunning() const {
    syncState();                                  // 惰性收敛（异常停发现点）
    return state_.load() == State::Running;
}

void PosturePipeline::syncState() const {
    // Running 且 lanes 全退 = runtime 自灭（用户 stop/集齐收口路径均先转 Stopped，
    // 不会以 Running 态观测到全退）→ Faulted + Fault(1701) 一次性上报。
    if (state_.load() != State::Running || !runtime_.lanesExited()) return;
    std::lock_guard<std::mutex> lk(watchMu_);
    if (completionRequested_) return;             // 集齐收口中：watcher 将置 Stopped
    state_.store(State::Faulted);
    if (sink_) {
        sink_->report(Scanner::QualityFlag::Fault, kEvtRuntimeDied,
                      "A 姿态流水线异常停（帧钩子异常/资源故障）");
    }
}

// ============================================================================
// 收口 watcher — 集齐自动收口序列（设计 §4.1：在飞周期判完再收口）
// ============================================================================
void PosturePipeline::watcherLoop() {
    std::unique_lock<std::mutex> lk(watchMu_);
    watchCv_.wait(lk, [this] { return completionRequested_ || watcherStop_; });
    const bool complete = completionRequested_;   // 集齐优先（数据已全，必须收口）
    lk.unlock();
    if (!complete) return;                        // 用户 stop：不触发 completion

    runtime_.requestStop();                       // 幂等（onComplete 已置）
    runtime_.drainAndShutdown();                  // join lanes：在飞周期 eFinalize 判完
    if (deps_.acquisition) deps_.acquisition->stopAcquisition();   // 采集停于排空之后
    {
        std::lock_guard<std::mutex> lk2(watchMu_);
        completionFired_ = true;                  // 先标记（与 stop 迟到补发互斥）
    }
    state_.store(State::Stopped);                 // 先转停（completion 可见时 isRunning 已 false）
    if (continueWithB_ && table_) continueWithB_(table_->takeSessionData());
}

// ============================================================================
// 生产链装配
// ============================================================================
std::shared_ptr<PostureLaneOps> PosturePipeline::makeOps() const {
    auto ops = std::make_shared<PostureLaneOps>();
    try {
#ifdef JMW_BUILD_CUDA
        ops->maskL = std::make_unique<calib::MaskExtractCUDA>();
        ops->maskR = std::make_unique<calib::MaskExtractCUDA>();
        calib::FrameFilterParams ff;
        ff.maskRatioThreshold = params_.maskRatioThreshold;
        ops->filterL = std::make_unique<calib::FrameFilterCUDA>(ff);
        ops->filterR = std::make_unique<calib::FrameFilterCUDA>(ff);
        ops->cclL = std::make_unique<calib::RegionAnalyzerCUDA>();
        ops->cclR = std::make_unique<calib::RegionAnalyzerCUDA>();
#endif
        calib::ImageSplitCPUParams sp;
        sp.enableBoundaryCheck = true;            // ROI 越界安全裁剪
        ops->split = std::make_unique<calib::ImageSplitCPU>(sp);
        ops->zernike = std::make_unique<calib::ZernikeEdgeCPU>();
        ops->merge = std::make_unique<calib::ImageMergeCPU>();

        calib::MarkerUndistortCPUParams mp;
        mp.fx1 = params_.K1.at<double>(0, 0); mp.fy1 = params_.K1.at<double>(1, 1);
        mp.cx1 = params_.K1.at<double>(0, 2); mp.cy1 = params_.K1.at<double>(1, 2);
        mp.fx2 = params_.K2.at<double>(0, 0); mp.fy2 = params_.K2.at<double>(1, 1);
        mp.cx2 = params_.K2.at<double>(0, 2); mp.cy2 = params_.K2.at<double>(1, 2);
        mp.k1_1 = atOr(params_.D1, 0); mp.k2_1 = atOr(params_.D1, 1);
        mp.p1_1 = atOr(params_.D1, 2); mp.p2_1 = atOr(params_.D1, 3);
        mp.k3_1 = atOr(params_.D1, 4);
        mp.k1_2 = atOr(params_.D2, 0); mp.k2_2 = atOr(params_.D2, 1);
        mp.p1_2 = atOr(params_.D2, 2); mp.p2_2 = atOr(params_.D2, 3);
        mp.k3_2 = atOr(params_.D2, 4);
        mp.imageWidth = params_.imageWidth;
        mp.imageHeight = params_.imageHeight;
        ops->undistCpu = std::make_unique<calib::MarkerUndistortCPU>(mp);
        // 立体矫正矩阵：初始参数组一次性注入（A 无逐帧快照）
        ops->undistCpu->SetRectifyMatrices(params_.R1, params_.R2, params_.P1,
                                           params_.P2, params_.Q);

        ops->ellipse = std::make_unique<calib::EllipseFitCPU>();
        ops->match = std::make_unique<calib::MarkerMatchCPU>();
        ops->epiIntersect = std::make_unique<calib::EpipolarIntersectCPU>();
        ops->edgeMatch = std::make_unique<calib::EdgeMatchCPU>();

        calib::PointReconstructCPUParams pp;      // fx/cx 构造校验需要
        pp.fxLeft = mp.fx1;  pp.fyLeft = mp.fy1;  pp.cxLeft = mp.cx1;  pp.cyLeft = mp.cy1;
        pp.fxRight = mp.fx2; pp.fyRight = mp.fy2; pp.cxRight = mp.cx2; pp.cyRight = mp.cy2;
        ops->reconstruct = std::make_unique<calib::PointReconstructCPU>(pp);
        ops->reconstruct->SetProjectionMatrices(params_.P1, params_.P2, params_.Q);

        ops->flowFuse = std::make_unique<calib::MarkerOpticalFlowFuseCPU>();
        ops->frameFuse = std::make_unique<calib::FrameFuseCPU>();
        // pose_estimate：目标表匹配交给 ConfirmTable（poseTargets 空）；无网格
        // 参数时算子 Execute 优雅失败 → eFinalize 沿用配准 R/T
        ops->poseEstimate = std::make_unique<calib::PoseEstimateCPU>();
    } catch (const std::exception& e) {
        spdlog::error("[PosturePipeline] lane 算子集构造失败: {}", e.what());
        return nullptr;
    }
    return ops;
}

PosturePipeline::Hooks PosturePipeline::assembleChains() {
    Hooks hooks;

    // ------------------------------------------------------------------
    // gpuChain — GPU 轻前段（E 核线程，持槽）：L/R 各 mask_extract →
    // frame_filter（激光线帧→false 销毁整周期）→ ccl；不调 frontReady
    // ------------------------------------------------------------------
    hooks.gpuChain = [this](sched::GpuSlotService::SlotGuard& guard,
                            const std::shared_ptr<const data::CycleUnit>& frame,
                            PostureFront& front,
                            std::function<void()> frontReady) -> bool {
        (void)frontReady;                         // A 模式不用：runtime 兜底提交（天然串行）
#ifndef JMW_BUILD_CUDA
        (void)guard; (void)frame; (void)front;
        spdlog::error("[PosturePipeline] 无 CUDA 构建仅编译守卫，GPU 前段运行不支持");
        return false;
#else
        if (!front.ops) {                         // 每 lane 首周期惰性建（先于 pChain）
            front.ops = makeOps();
            if (!front.ops) return false;
        }
        PostureLaneOps& ops = *front.ops;
        auto stream = cv::cuda::StreamAccessor::wrapStream(guard.stream);

        // 单相机前段：mask_extract → frame_filter（判 marker 帧）→ ccl → 包围盒
        auto frontStage = [&](const cv::Mat& gray,
                              std::unique_ptr<calib::MaskExtractCUDA>& maskOp,
                              std::unique_ptr<calib::FrameFilterCUDA>& filterOp,
                              std::unique_ptr<calib::RegionAnalyzerCUDA>& cclOp,
                              std::vector<cv::Rect>& rois) -> bool {
            auto m = maskOp->Execute(gray, stream);
            if (!m.success) {
                spdlog::warn("[PosturePipeline] mask_extract 失败: {}", m.message);
                return false;
            }
            auto f = filterOp->Execute(m.d_cleanedMask, stream);
            if (!f.success) {
                spdlog::warn("[PosturePipeline] frame_filter 失败: {}", f.message);
                return false;
            }
            if (!f.isMarkerFrame) {               // maskRatio<阈=激光线帧：销毁整周期
                spdlog::info("[PosturePipeline] cycle {} 激光线帧（maskRatio={:.4f}），"
                             "整周期销毁", frame->id, f.maskRatio);
                return false;
            }
            auto c = cclOp->Execute(m.d_cleanedMask, stream);
            if (!c.success) {
                spdlog::warn("[PosturePipeline] ccl 失败: {}", c.message);
                return false;
            }
            rois = c.toRectList();                // host 数据（算子内已同步下载）
            return true;
        };

        try {
            if (!frontStage(frame->markerL, ops.maskL, ops.filterL, ops.cclL,
                            front.roisL))
                return false;
            if (!frontStage(frame->markerR, ops.maskR, ops.filterR, ops.cclR,
                            front.roisR))
                return false;
        } catch (const std::exception& e) {
            spdlog::error("[PosturePipeline] GPU 前段异常: {}", e.what());
            return false;
        }
        return true;
#endif
    };

    // ------------------------------------------------------------------
    // pChain — 标记点链 9 算子（初始参数；P 核 worker）。无配准（在 E 段）
    // ------------------------------------------------------------------
    hooks.pChain = [](const std::shared_ptr<const data::CycleUnit>& frame,
                      PostureFront& front, PostureFrameResult& result) -> Result {
        if (!front.ops)
            return Result::fail("PosturePipeline: lane 算子集未初始化（gpuChain 未先行）");
        try {
            // false=本周期无有效标记点：正常降级（E 段沿用快照），不销毁周期
            runMarkerChain(*frame, *front.ops, front, result);
        } catch (const std::exception& e) {
            spdlog::error("[PosturePipeline] pChain 异常: {}", e.what());
            return Result::fail(std::string("PosturePipeline pChain 异常: ") + e.what());
        }
        return Result::ok();
    };

    // ------------------------------------------------------------------
    // eFinalize — 配准（最新快照）→ pose_estimate → ConfirmTable.report →
    // 每帧 pushPostureView（E 核线程）
    // ------------------------------------------------------------------
    hooks.eFinalize = [this](const std::shared_ptr<const data::CycleUnit>& frame,
                             PostureFront& front, PostureFrameResult& result,
                             std::future<Scanner::Result>& fut) -> Result {
        Result pr;
        try {
            pr = fut.get();                       // pChain Result/异常均在此消费
        } catch (const std::exception& e) {
            spdlog::error("[PosturePipeline] pChain future 异常: {}", e.what());
            return Result::fail(std::string("pChain future 异常: ") + e.what());
        }
        if (!pr.success) return pr;               // P 链失败=周期丢弃（不入表）

        // 1) 配准三级降级（读/写 prevState 最新快照；首帧 null 初始化）
        double liveR[9], liveT[3];
        runRegistration(frame->id, *front.ops, result.positions, result.normals,
                        liveR, liveT);

        // 2) pose_estimate（目标表匹配交 ConfirmTable）：无网格参数时算子优雅
        //    失败 → 沿用配准 R/T（世界系换算待网格参数注入后生效）
        try {
            auto pe = front.ops->poseEstimate->Execute(matxFromArr9(liveR),
                                                       vec3FromArr3(liveT));
            if (pe.success) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) liveR[r * 3 + c] = pe.currentPose(r, c);
                    liveT[r] = pe.currentPose(r, 3);
                }
            }
        } catch (const std::exception& e) {
            spdlog::warn("[PosturePipeline] pose_estimate 异常，沿用配准 R/T: {}",
                         e.what());
        }

        // 3) 确认簿记（const 共享周期 → 拷贝移入；确认时整周期保存）
        Scanner::data::CycleUnit cycle = *frame;  // cv::Mat 引用计数浅拷贝（激光管帧随行）
        const int poseIdx = table_->report(
            liveR, liveT, std::move(cycle),
            std::move(result.ellipseCentersL), std::move(result.ellipseCentersR));
        if (poseIdx >= 0)
            spdlog::info("[PosturePipeline] cycle {} 确认姿态 {}/{}",
                         frame->id, poseIdx + 1, table_->collectedCount());

        // 4) 每帧实时推送（实时姿态+标志点检出数；新确认时 confirmedCount 已递增）
        if (deps_.sceneFeed) {
            Scanner::Pose p;
            std::copy(liveR, liveR + 9, p.R);
            std::copy(liveT, liveT + 3, p.t);
            p.frameId = frame->id;
            p.timestamp = frame->timestamp;
            deps_.sceneFeed->pushPostureView(p, table_->collectedCount(),
                                             std::vector<uint8_t>(result.positions.size(), 1));
        }
        return Result::ok();
    };

    return hooks;
}

// ============================================================================
// E 段配准 — 最新快照模型（同 ScanChains::runRegistration / 姿态调度 §5）：
// 首帧 null 初始化 → optical_flow_fuse → frame_fuse 兜底（不回写快照）→
// 再失败沿用快照 R/T
// ============================================================================
void PosturePipeline::runRegistration(uint64_t cycleId, PostureLaneOps& ops,
                                      const std::vector<cv::Point3d>& positions,
                                      const std::vector<cv::Vec3d>& normals,
                                      double liveR[9], double liveT[3]) {
    const auto n = positions.size();
    auto fill = [&](const cv::Matx33d& R, const cv::Vec3d& T) {
        for (int i = 0; i < 9; ++i) liveR[i] = R(i / 3, i % 3);
        liveT[0] = T(0);
        liveT[1] = T(1);
        liveT[2] = T(2);
    };

    auto prev = calib::AtomicFrameState::load(prevState_);

    // —— 首帧初始化分支（锚空 / 锚内无点）——
    if (!prev || prev->rawPoints.empty()) {
        if (n == 0) {                             // 无点且无锚：I 位姿
            fill(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
            return;
        }
        std::vector<int> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int>(i);
        auto st = std::make_shared<calib::AtomicFrameState>();
        st->rawPoints.reserve(n);
        st->normals.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            st->rawPoints.push_back(calib::MarkerPoint3D{positions[i].x, positions[i].y,
                                                         positions[i].z, normals[i](0),
                                                         normals[i](1), normals[i](2),
                                                         ids[i]});
            st->normals.push_back(normals[i](0));
            st->normals.push_back(normals[i](1));
            st->normals.push_back(normals[i](2));
        }
        st->globalIds = std::move(ids);
        // st->R/T 保持结构默认（I/0），与 live 一致
        st->frameId = cycleId;
        fill(cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
        calib::AtomicFrameState::store(prevState_, std::move(st));
        return;
    }

    // —— 有锚但本周期无点：沿用快照 R/T ——
    if (n == 0) {
        fill(matxFromArr9(prev->R), vec3FromArr3(prev->T));
        return;
    }

    // prev 快照 → 算子 PrevFrameState（桥接：rawPoints/normals/globalIds/R/T）
    calib::PrevFrameState pf;
    pf.rawPositions.reserve(prev->rawPoints.size());
    pf.rawNormals.reserve(prev->rawPoints.size());
    for (const auto& p : prev->rawPoints) {
        pf.rawPositions.emplace_back(p.x, p.y, p.z);
        pf.rawNormals.emplace_back(p.nx, p.ny, p.nz);
    }
    pf.globalIds = prev->globalIds;
    pf.R = matxFromArr9(prev->R);
    pf.T = vec3FromArr3(prev->T);

    // —— 配准-01 optical_flow_fuse（默认）——
    auto fr = ops.flowFuse->Execute(positions, normals, pf);
    if (fr.success) {
        fill(fr.R, fr.T);
        auto st = std::make_shared<calib::AtomicFrameState>();
        st->rawPoints.reserve(fr.markers.size());
        st->normals.reserve(fr.markers.size() * 3);
        st->globalIds.reserve(fr.markers.size());
        for (const auto& m : fr.markers) {        // 设备系坐标 + 链式 globalId
            st->rawPoints.push_back(calib::MarkerPoint3D{m.rawPosition.x, m.rawPosition.y,
                                                         m.rawPosition.z, m.rawNormal(0),
                                                         m.rawNormal(1), m.rawNormal(2),
                                                         m.globalId});
            st->normals.push_back(m.rawNormal(0));
            st->normals.push_back(m.rawNormal(1));
            st->normals.push_back(m.rawNormal(2));
            st->globalIds.push_back(m.globalId);
        }
        for (int i = 0; i < 9; ++i) st->R[i] = fr.R(i / 3, i % 3);
        st->T[0] = fr.T(0);
        st->T[1] = fr.T(1);
        st->T[2] = fr.T(2);
        st->frameId = cycleId;
        calib::AtomicFrameState::store(prevState_, std::move(st));
        return;
    }
    spdlog::warn("[PosturePipeline] optical_flow_fuse 失败（{}），转 frame_fuse 兜底",
                 fr.message);

    // —— 兜底 frame_fuse（本周期 vs 快照点集；不回写快照保 globalId 链完整）——
    calib::MarkerPointSet cur{positions, normals};
    calib::MarkerPointSet prevSet{std::move(pf.rawPositions), std::move(pf.rawNormals)};
    auto ff = ops.frameFuse->Execute(cur, prevSet);
    if (ff.success) {
        fill(ff.R, ff.T);
        return;
    }
    spdlog::warn("[PosturePipeline] frame_fuse 兜底亦失败（{}），沿用快照 R/T", ff.message);

    // —— 再失败：沿用快照 R/T ——
    fill(matxFromArr9(prev->R), vec3FromArr3(prev->T));
}

} // namespace Scanner::pipeline
