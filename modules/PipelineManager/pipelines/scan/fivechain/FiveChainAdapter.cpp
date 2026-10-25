// FiveChainAdapter.cpp — 适配层实现（自持算子宿主＋07 桥）＋FcHarnessCompat 方法实现
#include "FiveChainAdapter.h"
#include "harness.h"

#include "pipelines/ISceneFeed.h"
#include "ICloudWarehouse.h"
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#include "sched/IFrameSource.h"
#include "SlotRing.h"
#include "EnhancedFrame.h"
#include "fc_executor.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <chrono>
#include <array>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include "ccl_cpu.h"

namespace Scanner::pipeline {

FiveChainAdapter::FiveChainAdapter(const FiveChainConfig& cfg,
                                   FrameObsAccumulator& obs,
                                   Scanner::data::ICloudWarehouse* warehouse,
                                   ISceneFeed* sceneFeed)
    : cfg_(cfg), obs_(obs), warehouse_(warehouse), sceneFeed_(sceneFeed) {
    sepL_ = std::make_unique<calib::LaserMarkingSeparationCUDA>();
    sepR_ = std::make_unique<calib::LaserMarkingSeparationCUDA>();
    calib::LaserCloudFuseCUDAParams fp;
    fp.voxelSize = static_cast<float>(cfg_.voxelSizeMm);
    fp.reserveVoxelCount = static_cast<size_t>(cfg_.voxelReserve);
    fuse_ = std::make_unique<calib::LaserCloudFuseCuda>(fp);
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 构造: voxel={}mm reserve={}M graySlots={}",
        cfg_.voxelSizeMm, cfg_.voxelReserve / 1000000, cfg_.graySlots);
}

FiveChainAdapter::~FiveChainAdapter() {
    stopRealtime();
    stop();
}

Scanner::Result FiveChainAdapter::start() {
    if (active_.load()) return Scanner::Result::fail("五链适配器已激活");
    active_.store(true);
    return Scanner::Result::ok("fivechain adapter started");
}

void FiveChainAdapter::stop() {
    if (!active_.exchange(false)) return;
    JMW_LOG_INFO("07-FiveChain", "[FiveChainAdapter] 停止");
}

void FiveChainAdapter::ensureGrayLanes(int slots) {
    if (static_cast<int>(grayLanesL_.size()) >= slots) return;
    grayLanesL_.resize(static_cast<size_t>(slots));
    grayLanesR_.resize(static_cast<size_t>(slots));
}

void FiveChainAdapter::onFrameDone(uint64_t frameId, const double R[9], const double T[3],
                                   const std::vector<MarkerObs>& markers,
                                   const std::vector<float>& laserXyz) {
    FrameObs fo;
    fo.frameId = frameId;
    for (int i = 0; i < 9; ++i) fo.R_init[i] = R[i];
    for (int i = 0; i < 3; ++i) fo.t_init[i] = T[i];
    fo.markerObs = markers;
    obs_.push(std::move(fo), laserXyz);
}

// ============================================================================
// FcHarnessCompat 方法实现（从沙盒 harness.cpp 提取的算法体——逐段对应）
// ============================================================================

int FcHarnessCompat::submitLaserFront(LaserOps& L, scr::CycleProductsCompat& p,
                                        bool record, bool tubeOdd, int laneIdx,
                                        cv::cuda::Stream& ls) {
    (void)tubeOdd; (void)record;
    ensureGrayLanes(laneIdx + 1);
    auto& gL = d_grayLanesL_[static_cast<size_t>(laneIdx)];
    auto& gR = d_grayLanesR_[static_cast<size_t>(laneIdx)];
    if (!p.sepL.d_laserMask || p.sepL.d_laserMask->empty()) {
        fail("steger_fast", p, "d_laserMask 为空（本帧激光线未分离出）");
        return 0;
    }
    try {
        timeBegin();
        bool deferred = false;
        if (L.flatCenter && L.stegerFast) {
            // flat 路径（沙盒 config extractor=flat）：延迟提交——双眼连续入队
            // 不同步（stats 28B D2H 在飞，完成段排水）
            p.stegerL = L.flatCenter->ExecuteDeferred(gL, *p.sepL.d_laserMask, ls);
            p.stegerR = L.flatCenter->ExecuteDeferred(gR, *p.sepR.d_laserMask, ls);
            deferred = true;
            if (!p.stegerL.success || !p.stegerR.success) {
                fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
                return 0;
            }
        } else if (L.stegerFast) {
            // steger_fast legacy 路径
            L.stegerFast->SetCollectHostPoints(false);
            p.stegerL = L.stegerFast->Execute(gL, *p.sepL.d_laserMask, ls,
                                              calib::GroupMode::Flat);
            p.stegerR = L.stegerFast->Execute(gR, *p.sepR.d_laserMask, ls,
                                              calib::GroupMode::Flat);
            if (!p.stegerL.success || !p.stegerR.success) {
                fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
                return 0;
            }
            // 零点护栏（防 undistort 空解引用）
            if (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
                || p.stegerL.d_centerPoints->empty() || p.stegerR.d_centerPoints->empty()) {
                fail("undistort_cuda", p, "steger 0 中心点");
                return 0;
            }
        } else {
            fail("steger_fast", p, "LaserOps 未装配（flatCenter/stegerFast 均空）");
            return 0;
        }

        // undistort（延迟模式＝容量宽视图；legacy＝精确视图）
        // 261025 实时链：L/R 各按帧温快照 SetParams（SetParams 仅存宿主参数，
        // Execute 时消费——在飞安全；沙盒对拍路径 metaValid=false 用构造期参数）
        timeBegin();
        if (L.undistG) {
            if (p.metaValid) {
                calib::UndistortPointsParams ul, ur;
                ul.cameraMatrix = K1_;  ul.distCoeffs = D1_;
                ul.R = cv::Mat(p.snapR1); ul.P = cv::Mat(p.snapP1);
                ur.cameraMatrix = K2_;  ur.distCoeffs = D2_;
                ur.R = cv::Mat(p.snapR2);
                // 右目 P(0,3)=-B·f 平移项置零（沙盒 L387-391 口径：v3/recon
                // 消费"右目自身列坐标"约定——保留则 x 偏 -17.1 万像素 v3 全排除）
                ur.P = cv::Mat(p.snapP2).clone();
                if (ur.P.rows > 0 && ur.P.cols > 3)
                    ur.P.at<double>(0, 3) = 0.0;
                L.undistG->SetParams(ul);
                p.undistCudaL = L.undistG->Execute(*p.stegerL.d_centerPoints,
                                                    *p.stegerL.d_line_ids, ls);
                L.undistG->SetParams(ur);
                p.undistCudaR = L.undistG->Execute(*p.stegerR.d_centerPoints,
                                                    *p.stegerR.d_line_ids, ls);
            } else {
                p.undistCudaL = L.undistG->Execute(*p.stegerL.d_centerPoints,
                                                    *p.stegerL.d_line_ids, ls);
                p.undistCudaR = L.undistG->Execute(*p.stegerR.d_centerPoints,
                                                    *p.stegerR.d_line_ids, ls);
            }
        }
        if (!p.undistCudaL.success || !p.undistCudaR.success) {
            fail("undistort_cuda", p,
                 p.undistCudaL.message + " | " + p.undistCudaR.message);
            return deferred ? 1 : 0;
        }

        // interp（延迟模式＝设备计数驱动；legacy＝直接）
        timeBegin();
        if (deferred && L.flatCenter) {
            p.interpL = L.interpL_->ExecuteDeferred(
                *p.undistCudaL.d_rectifiedPoints, *p.undistCudaL.d_line_ids,
                L.flatCenter->GetDeviceCountPtr(), ls);
            p.interpR = L.interpR_->ExecuteDeferred(
                *p.undistCudaR.d_rectifiedPoints, *p.undistCudaR.d_line_ids,
                L.flatCenter->GetDeviceCountPtr(), ls);
            if (!p.interpL.success || !p.interpR.success) {
                fail("epipolar_interp_dual", p,
                     p.interpL.message + " | " + p.interpR.message);
                return 1;   // 已提交——排水
            }
        } else if (L.interpL_ && L.interpR_) {
            p.interpL = L.interpL_->Execute(*p.undistCudaL.d_rectifiedPoints,
                                            *p.undistCudaL.d_line_ids, ls);
            p.interpR = L.interpR_->Execute(*p.undistCudaR.d_rectifiedPoints,
                                            *p.undistCudaR.d_line_ids, ls);
            if (!p.interpL.success || !p.interpR.success) {
                fail("epipolar_interp_dual", p,
                     p.interpL.message + " | " + p.interpR.message);
                return 0;
            }
        }
        return deferred ? 1 : 2;
    } catch (const std::exception& e) {
        fail("steger_fast", p, e.what());
        return 0;
    }
}

void FcHarnessCompat::finishLaserFront(LaserOps& L, scr::CycleProductsCompat& p,
                                        bool record, int kind) {
    (void)record;
    if (kind == 0) return;
    const bool deferred = (kind == 1);
    if (deferred && L.flatCenter) {
        L.flatCenter->FinishDeferred(p.stegerL);
        L.flatCenter->FinishDeferred(p.stegerR);
        if (!p.stegerL.success || !p.stegerR.success) {
            fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
        }
    }
    // 零点检查
    if (p.stegerL.success && p.stegerR.success
        && (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
            || p.stegerL.d_centerPoints->empty() || p.stegerR.d_centerPoints->empty())) {
        fail("undistort_cuda", p, "steger 0 中心点（延迟段排水后）");
        return;
    }
    if (deferred && p.undistCudaL.success && p.undistCudaR.success
        && p.interpL.success && p.interpR.success && L.interpL_ && L.interpR_) {
        L.interpL_->FinishInterp(p.interpL);
        L.interpR_->FinishInterp(p.interpR);
        if (!p.interpL.success || !p.interpR.success) {
            fail("epipolar_interp_dual", p,
                 p.interpL.message + " | " + p.interpR.message);
        }
    }
}

void FcHarnessCompat::runMarkerChainRuns(scr::CycleProductsCompat& p, bool record,
                                          std::unique_ptr<MarkerOps>& ops,
                                          const std::vector<int>& runsL, int nL,
                                          const std::vector<int>& runsR, int nR,
                                          int cols, int rows) {
    // 沙盒 harness.cpp runMarkerChainRuns + runMarkerChainCore 全量（RLE 行程→
    // CPU CCL→标志点链→conic 一步重建）
    (void)record;
    if (!ops) return;
    try {
        p.cclL = scr::runCclCpuFromRuns(runsL, nL, cols, rows, 80, 20000, 120);
        p.cclR = scr::runCclCpuFromRuns(runsR, nR, cols, rows, 80, 20000, 120);
        if (!p.cclL.success || !p.cclR.success) {
            fail("ccl", p, "runCclCpuFromRuns 失败");
            return;
        }
    } catch (const std::exception& e) {
        fail("ccl", p, e.what());
        return;
    }
    runMarkerChainCore(p, ops);
}

void FcHarnessCompat::runMarkerChainCore(scr::CycleProductsCompat& p,
                                           std::unique_ptr<MarkerOps>& ops) {
    std::vector<cv::Rect> roisL, roisR;
    for (const auto& c : p.cclL.components) {
        roisL.emplace_back(c.boundingBoxX, c.boundingBoxY,
                           c.boundingBoxWidth, c.boundingBoxHeight);
    }
    for (const auto& c : p.cclR.components) {
        roisR.emplace_back(c.boundingBoxX, c.boundingBoxY,
                           c.boundingBoxWidth, c.boundingBoxHeight);
    }
    if (roisL.empty() || roisR.empty()) {
        fail("image_split", p, "ccl 无组件");
        return;
    }
    // 261025 实时链：帧温快照注入（生产口径——ScanChains L853-857/L333-346 同源；
    // 沙盒对拍路径 metaValid=false 用 harness 构造期标定）
    if (p.metaValid) {
        const cv::Mat R1(p.snapR1), R2(p.snapR2), P1(p.snapP1), P2(p.snapP2);
        if (ops->undist) ops->undist->SetRectifyMatrices(R1, R2, P1, P2, p.snapQ);
        if (ops->conic) {
            // 矫正系 K/C2 由标定档派生：K1=P1 左 3×3；K2=P2 左 3×3；C2=−K2⁻¹·t2
            cv::Matx33d cK1 = cv::Matx33d::eye(), cK2 = cv::Matx33d::eye();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    cK1(r, c) = p.snapP1(r, c);
                    cK2(r, c) = p.snapP2(r, c);
                }
            const cv::Vec3d cT2(p.snapP2(0, 3), p.snapP2(1, 3), p.snapP2(2, 3));
            const cv::Vec3d cC2 = -(cK2.inv() * cT2);
            ops->conic->SetProjectionMatrices(cK1, cK2, cC2);
        }
    }
    // 后续链：split→zernike→merge→undist→ellipse→match→conic（沙盒 runMarkerChainCore 全量）
    try {
        p.splitL = ops->split->Execute(p.frameGrayL, roisL);
        p.splitR = ops->split->Execute(p.frameGrayR, roisR);
        if (!p.splitL.success || !p.splitR.success) {
            fail("image_split", p, p.splitL.message + " | " + p.splitR.message);
            return;
        }
    } catch (const std::exception& e) {
        fail("image_split", p, e.what());
        return;
    }

    // zernike edge
    try {
        p.zernL.clear();
        for (const auto& img : p.splitL.splitImages) p.zernL.push_back(ops->zernike->Execute(img));
        p.zernR.clear();
        for (const auto& img : p.splitR.splitImages) p.zernR.push_back(ops->zernike->Execute(img));
        bool ok = !p.zernL.empty() && !p.zernR.empty();
        for (const auto& z : p.zernL) ok = ok && z.success;
        for (const auto& z : p.zernR) ok = ok && z.success;
        if (!ok) { fail("zernike_edge", p, "亚像素提取失败"); return; }
    } catch (const std::exception& e) {
        fail("zernike_edge", p, e.what());
        return;
    }

    // merge
    try {
        std::vector<std::vector<calib::EdgePoint>> epL, epR;
        epL.reserve(p.zernL.size()); epR.reserve(p.zernR.size());
        for (const auto& z : p.zernL) epL.push_back(z.edgePoints);
        for (const auto& z : p.zernR) epR.push_back(z.edgePoints);
        p.mergeL = ops->merge->Execute(epL, roisL);
        p.mergeR = ops->merge->Execute(epR, roisR);
        if (!p.mergeL.success || !p.mergeR.success) {
            fail("image_merge", p, p.mergeL.message + " | " + p.mergeR.message);
            return;
        }
    } catch (const std::exception& e) {
        fail("image_merge", p, e.what());
        return;
    }

    // undistort CPU
    try {
        p.undistCpu = ops->undist->Execute(p.mergeL.mergedEdgePoints,
                                            p.mergeR.mergedEdgePoints,
                                            p.mergeL.groupIds, p.mergeR.groupIds);
        if (!p.undistCpu.success) {
            fail("undistort_cpu", p, p.undistCpu.message);
            return;
        }
    } catch (const std::exception& e) {
        fail("undistort_cpu", p, e.what());
        return;
    }

    // ellipse fit（按组拟合）
    try {
        const auto g1 = p.undistCpu.splitRectifiedPoints1ByGroup();
        const auto g2 = p.undistCpu.splitRectifiedPoints2ByGroup();
        p.ellL.clear(); p.ellR.clear();
        for (size_t i = 0; i < g1.size(); ++i) {
            if (g1[i].size() < 5) continue;
            auto r = ops->ellipse->Execute(g1[i]);
            r.groupId = static_cast<int>(i);
            p.ellL.push_back(std::move(r));
        }
        for (size_t i = 0; i < g2.size(); ++i) {
            if (g2[i].size() < 5) continue;
            auto r = ops->ellipse->Execute(g2[i]);
            r.groupId = static_cast<int>(i);
            p.ellR.push_back(std::move(r));
        }
        int okL = 0, okR = 0;
        for (const auto& e : p.ellL) okL += e.success ? 1 : 0;
        for (const auto& e : p.ellR) okR += e.success ? 1 : 0;
        p.markersDetected = std::min(okL, okR);
        if (p.markersDetected == 0) {
            fail("ellipse_fit", p, "椭圆拟合成功数不足 L=" + std::to_string(okL) + " R=" + std::to_string(okR));
            return;
        }
    } catch (const std::exception& e) {
        fail("ellipse_fit", p, e.what());
        return;
    }

    // marker match
    try {
        p.mm = ops->match->Execute(calib::extractEllipseCenters(p.ellL),
                                   calib::extractEllipseCenters(p.ellR));
        if (!p.mm.success) {
            fail("marker_match", p, p.mm.message);
            return;
        }
    } catch (const std::exception& e) {
        fail("marker_match", p, e.what());
        return;
    }

    // conic reconstruct（一步重建）
    try {
        p.conic = ops->conic->Execute(p.ellL, p.ellR, p.mm.centerMatches);
        p.markersValid = static_cast<int>(p.conic.statistics.validCount);
    } catch (const std::exception& e) {
        fail("conic_reconstruct", p, e.what());
        return;
    }
}

void FcHarnessCompat::runRegistration(scr::CycleProductsCompat& p, bool record) {
    (void)record;
    // 基线：先置上帧 R/T
    if (!prevState_.empty()) {
        for (int i = 0; i < 9; ++i) p.R[i] = prevState_.R.val[i];
        for (int i = 0; i < 3; ++i) p.T[i] = prevState_.T.val[i];
    }
    if (!p.conic.success || p.conic.markers.empty()) {
        p.regSource = "skip";
        return;
    }
    if (!flowOp_) {
        p.regSource = "hold";
        return;
    }
    try {
        std::vector<cv::Point3d> centers;
        std::vector<cv::Vec3d> normals;
        for (const auto& m : p.conic.markers) {
            if (!m.validCircle) continue;
            centers.emplace_back(m.centerX, m.centerY, m.centerZ);
            normals.emplace_back(m.normalX, m.normalY, m.normalZ);
        }
        p.regFlow = flowOp_->Execute(centers, normals, prevState_);
        const bool flowOk = p.regFlow.success
                            && static_cast<int>(p.regFlow.getMatchedCount()) >= 3;
        if (flowOk) {
            p.regSource = prevState_.empty() ? "first" : "flow";
            for (int i = 0; i < 9; ++i) p.R[i] = p.regFlow.R.val[i];
            for (int i = 0; i < 3; ++i) p.T[i] = p.regFlow.T.val[i];
            prevState_.R = p.regFlow.R;
            prevState_.T = p.regFlow.T;
            prevState_.rawPositions.clear();
            prevState_.rawNormals.clear();
            prevState_.globalIds.clear();
            for (auto& m : p.regFlow.markers) {
                prevState_.rawPositions.push_back(m.rawPosition);
                prevState_.rawNormals.push_back(m.rawNormal);
                prevState_.globalIds.push_back(m.globalId);
            }
        } else {
            p.regSource = "hold";
        }
    } catch (const std::exception& e) {
        fail("registration", p, e.what());
        p.regSource = "hold";
    }
}

// ============================================================================
// 261025 实时链（Phase 2 收尾）——SlotRing→五链执行器→obs/仓库/渲染
// ============================================================================
namespace {

/// tubeOdd（＝生产口径 leftSkewFrame）：面片 T/V 交替（tvKnown 时间戳奇偶法，
/// 未知回退帧号奇偶）；精细/深孔单管恒 true（ScanChains L503-511 同源）
bool deriveTubeOdd(const Scanner::data::EnhancedFrame& f) {
    if (f.scanMode == Scanner::ScanMode::FineScan
        || f.scanMode == Scanner::ScanMode::DeepHoleScan)
        return true;
    return f.tvKnown ? f.tvLeftSkew : ((f.frameId % 2) == 0);
}

} // namespace

// ---------------------------------------------------------------------------
// RtBridge — SlotRing（Overwrite 抓最新面孔）→ fc FrameSourceCompat 桥
// ---------------------------------------------------------------------------
struct FiveChainAdapter::RtBridge {
    static constexpr size_t kRing = 128;   // > maxInflight(56)+通道深(<16)——帧宿主安全界

    explicit RtBridge(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& r,
                      size_t dropThreshold)
        : grab(r, dropThreshold) {}

    sched::GrabLatestSource<Scanner::data::EnhancedFrame> grab;
    std::atomic<bool> stopped{false};
    uint64_t counter = 0;                       // grabLatest 读游标（本桥单读者）
    std::atomic<uint64_t> skippedTotal{0};      // 过载丢帧累计（诊断）
    std::atomic<uint64_t> servedFrames{0};      // 实际服务帧数

    // 逐 fc-idx 元数据＋帧宿主（shared_ptr 保活 frameGray 浅拷贝——Overwrite 环
    // 槽位复用后老帧数据仍存活至在飞结束）
    struct Slot {
        std::atomic<int> seq{-1};               // fc idx（release 写/acquire 读）
        bool tubeOdd = true;
        Scanner::data::CalibSnapshot snap;      // Matx 定长——平凡可拷
        double temperature = 0.0;
        std::shared_ptr<const Scanner::data::EnhancedFrame> keep;   // 帧宿主
    };
    std::array<Slot, kRing> ring{};

    std::shared_ptr<const Scanner::data::EnhancedFrame> cur;   // L/R 对服务缓存
    cv::Mat cropL_, cropR_;                                    // 裁剪产物宿主（在飞帧消费）

    int algoW_ = 0, algoH_ = 0;        // 算法几何（0=直通不裁剪）
    std::atomic<uint64_t> cropFrames_{0};   // 裁剪帧计数（诊断）

    scr::FrameSourceCompat src;

    /// 居中裁剪到算法几何（与标定表几何一致——K/P/Q/激光表全 1688×1400）。
    /// 尺寸已符/裁剪关闭＝直通（浅拷）；不符＝居中 ROI 深拷（子矩阵连续性保算子安全）
    void cropToAlgo(const cv::Mat& src, cv::Mat& dst) {
        if (algoW_ <= 0 || algoH_ <= 0
            || (src.cols == algoW_ && src.rows == algoH_)) {
            dst = src;
            return;
        }
        if (src.cols < algoW_ || src.rows < algoH_) {
            dst = cv::Mat();       // 采集小于算法几何＝配置错——交上层零帧告警
            return;
        }
        const int x0 = (src.cols - algoW_) / 2;
        const int y0 = (src.rows - algoH_) / 2;
        dst = src(cv::Rect(x0, y0, algoW_, algoH_)).clone();
    }

    void wire() {
        src.cycles.fs = &src;
        src.cycleAt = [this](size_t i) -> scr::FrameCycleInfo {
            scr::FrameCycleInfo ci;
            ci.laserIdx = static_cast<int>(i);   // 桥语义：laserIdx 槽位复用透传 fc idx
            const Slot& s = ring[i % kRing];
            if (s.seq.load(std::memory_order_acquire) == static_cast<int>(i))
                ci.tubeOdd = s.tubeOdd;          // chain1/1b 读取时已填（通道序保证）
            return ci;
        };
        src.readFrame = [this](int idx, bool left, cv::Mat& out, std::string& err) -> bool {
            if (left) {
                std::shared_ptr<const Scanner::data::EnhancedFrame> f;
                for (;;) {
                    if (stopped.load(std::memory_order_acquire)) {
                        err = "rt bridge stopped";
                        return false;            // 墓碑：排水终止
                    }
                    size_t sk = 0;
                    f = grab.grabLatest(counter, sk);
                    if (f) {
                        skippedTotal.fetch_add(sk, std::memory_order_relaxed);
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                cur = std::move(f);
                // 元数据＋帧宿主落槽（release 序——cycleAt/chain 消费方 acquire）
                Slot& s = ring[static_cast<size_t>(idx) % kRing];
                s.keep = cur;
                s.tubeOdd = deriveTubeOdd(*cur);
                s.snap = cur->snapshot;
                s.temperature = cur->temperature;
                s.seq.store(idx, std::memory_order_release);
                servedFrames.fetch_add(1, std::memory_order_relaxed);
                // 居中裁剪到算法几何（标定表 1688×1400——采集 2048×1536 差异收口）
                cropToAlgo(cur->grayL, cropL_);
                cropToAlgo(cur->grayR, cropR_);
                if (cropL_.empty() || cropR_.empty()) {
                    err = "rt bridge: 采集尺寸小于算法几何 ("
                          + std::to_string(cur->grayL.cols) + "x"
                          + std::to_string(cur->grayL.rows) + " < "
                          + std::to_string(algoW_) + "x" + std::to_string(algoH_) + ")";
                    return false;
                }
                if (!(cropL_.data == cur->grayL.data))
                    cropFrames_.fetch_add(1, std::memory_order_relaxed);
            }
            if (!cur) { err = "rt bridge: R before L"; return false; }
            out = left ? cropL_ : cropR_;      // 裁剪后视图（宿主＝桥成员，在飞安全）
            return true;
        };
    }
};

// ---------------------------------------------------------------------------
// attachCalib — 标定注入（ScanPipeline configure 期）
// ---------------------------------------------------------------------------
void FiveChainAdapter::attachCalib(const cv::Mat& K1, const cv::Mat& D1,
                                   const cv::Mat& K2, const cv::Mat& D2,
                                   int imageWidth, int imageHeight,
                                   std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable) {
    rtK1_ = K1.clone(); rtD1_ = D1.clone();
    rtK2_ = K2.clone(); rtD2_ = D2.clone();
    rtW_ = imageWidth; rtH_ = imageHeight;
    rtLaserTable_ = std::move(laserTable);
}

// ---------------------------------------------------------------------------
// startRealtime — 桥＋harness 装配（激光链算子/温表/配准算子）＋执行器＋收割线程
// ---------------------------------------------------------------------------
Scanner::Result FiveChainAdapter::startRealtime(
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring, size_t dropThreshold) {
    if (rtActive_.load()) return Scanner::Result::fail("五链实时链已在运行");
    if (rtK1_.empty() || rtK2_.empty())
        return Scanner::Result::fail("五链实时链: 标定未注入（先 attachCalib）");
    if (!rtLaserTable_)
        return Scanner::Result::fail("五链实时链: 激光温表为空（v1 需激光链）");
    if (rtActive_.exchange(true)) return Scanner::Result::fail("五链实时链并发启动");

    try {
        // ① 桥（算法几何：默认标定表几何 1688×1400；env JMW_FC_W/H 覆盖，0=直通）
        bridge_ = std::make_unique<RtBridge>(ring, dropThreshold);
        {
            int aw = cfg_.algoWidth, ah = cfg_.algoHeight;
            if (const char* e = getenv("JMW_FC_W")) aw = atoi(e);
            if (const char* e = getenv("JMW_FC_H")) ah = atoi(e);
            bridge_->algoW_ = aw;
            bridge_->algoH_ = ah;
        }
        bridge_->wire();

        // ② harness（算子宿主）——产线口径装配
        harness_ = new FcHarnessCompat(*this, bridge_->src);
        harness_->cfg_.fivechain = cfg_;
        harness_->cfg.fivechain = cfg_;                    // 双名兼容
        harness_->cfg_.runtime.nominalTemperature = 25.0;  // v1 标称温（逐帧温 TODO）
        harness_->K1_ = rtK1_; harness_->D1_ = rtD1_;
        harness_->K2_ = rtK2_; harness_->D2_ = rtD2_;
        // 标定几何口径：K/P/Q/激光表全 1688×1400——undistCpu 边界检查等用算法
        // 几何（非采集几何；采集 2048×1536 已在桥居中裁剪）
        harness_->imageWidth_ = bridge_->algoW_ > 0 ? bridge_->algoW_ : rtW_;
        harness_->imageHeight_ = bridge_->algoH_ > 0 ? bridge_->algoH_ : rtH_;
        harness_->tableOdd_ = rtLaserTable_;               // 产线单表：奇偶共表
        harness_->tableEven_ = rtLaserTable_;
        harness_->sepLaserL_ = sepL_.get();
        harness_->sepLaserR_ = sepR_.get();
        harness_->voxelFuseOp_ = fuse_.get();
        harness_->prodProfile_ = true;                     // 产线：跳过沙盒记账
        harness_->onFrameMeta = [this](int idx, scr::CycleProductsCompat& p) {
            // reader 线程回调——cur 即本帧（L/R 已读，executor 读后即调）
            const auto& f = bridge_->cur;
            if (!f) return;
            p.metaFrameId = f->frameId;
            p.snapR1 = f->snapshot.R1; p.snapR2 = f->snapshot.R2;
            p.snapP1 = f->snapshot.P1; p.snapP2 = f->snapshot.P2;
            p.snapQ = cv::Mat(f->snapshot.Q);              // Mat(Matx) 深拷（16 double）
            p.temperature = f->temperature;
            p.metaValid = true;
            (void)idx;
        };

        // ③ 激光链前段算子（沙盒 makeLaserOps 产线版：默认参数＋强制 Scan 模式）
        auto& L = harness_->laserOps_;
        L.stegerFast = std::make_unique<calib::StegerExtractorFast>(
            calib::StegerParams::fromJson(harness_->cfg_.opParams("steger_fast")));
        L.flatCenter = std::make_unique<calib::FlatCenterExtractor>(
            calib::FlatCenterParams::fromJson(harness_->cfg_.opParams("flat_center")));
        L.undistG = std::make_unique<calib::UndistortPointsCuda>();   // 参数每帧 SetParams
        auto ip = calib::EpipolarInterpDualParams::fromJson(
            harness_->cfg_.opParams("epipolar_interp_dual"));
        ip.mode = calib::InterpMode::Scan;                 // 沙盒口径：强制扫描模式
        L.interpL_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
        L.interpR_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);

        // ④ 配准算子（生产调参口径——ScanChains L261-264 同源）
        calib::MarkerOpticalFlowFuseCPUParams ofp;
        ofp.normalAngleThresh = 45.0;
        ofp.useNormalWeight = true;
        harness_->flowOp_ = std::make_unique<calib::MarkerOpticalFlowFuseCPU>(ofp);

        // ⑤ 执行器（INT_MAX＝无界实时会话——reader 由桥停信号终止）
        executor_ = std::make_unique<scr::fc::FiveChainExecutor>(*harness_, 0, 0x7fffffff);

        // ⑥ 收割线程
        collector_ = std::thread([this] { collectorLoop(); });

        JMW_LOG_INFO("07-FiveChain",
            "[FiveChainAdapter] 实时链启动: voxel={}mm graySlots={} markerLanes={} "
            "maxInflight={} 温表=单表(奇偶共表) 算法几何={}x{}（裁剪={}）",
            cfg_.voxelSizeMm, cfg_.graySlots, cfg_.markerLanes,
            cfg_.maxInflight > 0 ? cfg_.maxInflight : 56,
            bridge_->algoW_ > 0 ? bridge_->algoW_ : rtW_,
            bridge_->algoH_ > 0 ? bridge_->algoH_ : rtH_,
            bridge_->algoW_ > 0 ? "居中裁剪" : "直通");
        return Scanner::Result::ok("fivechain realtime started");
    } catch (const std::exception& e) {
        rtActive_.store(false);
        if (collector_.joinable()) collector_.join();
        executor_.reset();
        delete harness_; harness_ = nullptr;
        bridge_.reset();
        return Scanner::Result::fail(std::string("五链实时链启动失败: ") + e.what());
    }
}

// ---------------------------------------------------------------------------
// collectorLoop — F 通道逐帧收割→obs 落账＋轻计数＋节流渲染推送
// ---------------------------------------------------------------------------
void FiveChainAdapter::collectorLoop() {
    int idx = 0;
    for (;;) {
        scr::CycleProductsCompat prod;
        std::string err;
        if (!executor_->collect(idx, prod, err)) break;    // F 关闭且空＝会话终
        const uint64_t fid = prod.metaValid ? prod.metaFrameId
                                            : static_cast<uint64_t>(idx);

        // ① 观测落账（配准 R/T＋globalId 标志点观测——GBA 链）
        FrameObs fo;
        fo.frameId = fid;
        for (int i = 0; i < 9; ++i) fo.R_init[i] = prod.R[i];
        for (int i = 0; i < 3; ++i) fo.t_init[i] = prod.T[i];
        if (prod.regFlow.success) {
            fo.markerObs.reserve(prod.regFlow.markers.size());
            for (const auto& m : prod.regFlow.markers) {
                if (m.globalId < 0) continue;
                MarkerObs mo;
                mo.xyz[0] = m.rawPosition.x;
                mo.xyz[1] = m.rawPosition.y;
                mo.xyz[2] = m.rawPosition.z;
                mo.globalId = m.globalId;
                fo.markerObs.push_back(mo);
            }
        }
        obs_.push(std::move(fo), {});                      // 激光已 GPU 侧融合——无逐帧 host 缓存

        // ② 轻计数上报（UI 统计亚秒刷新）
        if (sceneFeed_ && (idx % 8) == 0)
            sceneFeed_->notifyLaserCount(fuse_->GetFusedPointCount());

        // ③ 节流渲染推送（时间门 ≥3s——FuseConsumer 261018 口径）
        renderPushIfDue(/*forceFull=*/false);
        ++idx;
    }
    JMW_LOG_INFO("07-FiveChain", "[FiveChainAdapter] 收割线程退出: {} 帧处理", idx);
}

// ---------------------------------------------------------------------------
// 渲染推送（collector 调用；终局 stopRealtime forceFull=true）
// ---------------------------------------------------------------------------
void FiveChainAdapter::renderPushIfDue(bool forceFull) {
    const auto now = std::chrono::steady_clock::now();
    constexpr int64_t kMinMs = 3000;
    if (!forceFull) {
        if (lastRenderPush_.time_since_epoch().count() != 0
            && std::chrono::duration_cast<std::chrono::milliseconds>(
                   now - lastRenderPush_).count() < kMinMs)
            return;
    }
    if (!sceneFeed_ && !warehouse_) return;
    lastRenderPush_ = now;

    const size_t curPts = fuse_->GetFusedPointCount();
    const bool fullPush = forceFull || lastPushedLaserPts_ == 0
                          || lastPushedLaserPts_ > curPts || pushSeq_ % 10 == 0;

    // 融合云下载（设备上下文→dlStream_ 异拷＋同步——LaserFuseAdapter 口径）
    std::vector<float> fusedXyz;
    const auto ctx = fuse_->GetDeviceContext();
    if (ctx.d_fusedXyz && ctx.fusedPointCount > (fullPush ? 0 : lastPushedLaserPts_)) {
        const size_t from = fullPush ? 0 : lastPushedLaserPts_;
        const size_t n = ctx.fusedPointCount - from;
        fusedXyz.resize(n * 3);
        const cudaStream_t cs = cv::cuda::StreamAccessor::getStream(dlStream_);
        if (cudaMemcpyAsync(fusedXyz.data(), ctx.d_fusedXyz + from * 3,
                            fusedXyz.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs) != cudaSuccess
            || cudaStreamSynchronize(cs) != cudaSuccess) {
            JMW_LOG_WARN("07-FiveChain", "[FiveChainAdapter] 融合云下载失败");
            fusedXyz.clear();
        }
    }
    lastPushedLaserPts_ = curPts;
    ++pushSeq_;
    if (fusedXyz.empty()) return;

    // 仓库＝全量真值账（全量替换/尾部追加）
    if (warehouse_) {
        if (fullPush) warehouse_->pushSessionCloud(fusedXyz);
        else         warehouse_->appendSessionCloud(fusedXyz);
    }
    if (!sceneFeed_) return;

    // UI 块：全量抽稀 ≤200 万（FuseConsumer 同口径）
    std::vector<float> uiXyz;
    constexpr size_t kUiMaxPts = 2000000;
    if (fullPush && fusedXyz.size() / 3 > kUiMaxPts) {
        const size_t stride = (fusedXyz.size() / 3 + kUiMaxPts - 1) / kUiMaxPts;
        uiXyz.reserve((fusedXyz.size() / 3 + stride - 1) / stride * 3);
        for (size_t i = 0; i * 3 < fusedXyz.size(); i += stride) {
            uiXyz.push_back(fusedXyz[i * 3 + 0]);
            uiXyz.push_back(fusedXyz[i * 3 + 1]);
            uiXyz.push_back(fusedXyz[i * 3 + 2]);
        }
    }
    renderKeep_ = uiXyz.empty() ? std::move(fusedXyz) : std::move(uiXyz);

    CloudViewHandle h;
    h.hostMarker = nullptr;                       // v1：五链无 marker 融合账本（编辑不支持）
    h.deviceLaser = nullptr;
    h.hostLaser = renderKeep_.empty() ? nullptr : renderKeep_.data();
    h.laserCount = renderKeep_.size() / 3;
    h.laserAppend = !fullPush;
    h.laserTotal = curPts;
    sceneFeed_->pushCloudSnapshot(h);
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 推送渲染: 融合云={} 点（{}：UI {} 点）",
        curPts, fullPush ? "全量" : "增量", h.laserCount);
}

// ---------------------------------------------------------------------------
// stopRealtime — 桥停→执行器排水析构→收割 join→终局全量推送
// ---------------------------------------------------------------------------
void FiveChainAdapter::stopRealtime() {
    if (!rtActive_.exchange(false)) return;
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 实时链停止: 服务 {} 帧（过载跳帧 {}/裁剪 {}）/融合云 {} 点",
        bridge_ ? bridge_->servedFrames.load() : 0,
        bridge_ ? bridge_->skippedTotal.load() : 0,
        bridge_ ? bridge_->cropFrames_.load() : 0,
        fuse_ ? fuse_->GetFusedPointCount() : 0);    if (bridge_) bridge_->stopped.store(true);    // readFrame 速败→墓碑排空
    executor_.reset();                            // stopAll＋内部线程 join＋设备同步
    if (collector_.joinable()) collector_.join();
    renderPushIfDue(true);                        // 终局全量（仓库+UI 收口）
    delete harness_; harness_ = nullptr;
    bridge_.reset();
}

} // namespace Scanner::pipeline
