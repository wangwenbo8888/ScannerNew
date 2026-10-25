// FiveChainAdapter.cpp — 适配层实现（自持算子宿主＋07 桥）＋FcHarnessCompat 方法实现
#include "FiveChainAdapter.h"
#include "harness.h"

#include "pipelines/ISceneFeed.h"
#include "ICloudWarehouse.h"
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"

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

FiveChainAdapter::~FiveChainAdapter() { stop(); }

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
        timeBegin();
        if (L.undistG) {
            p.undistCudaL = L.undistG->Execute(*p.stegerL.d_centerPoints,
                                                *p.stegerL.d_line_ids, ls);
            p.undistCudaR = L.undistG->Execute(*p.stegerR.d_centerPoints,
                                                *p.stegerR.d_line_ids, ls);
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

} // namespace Scanner::pipeline
