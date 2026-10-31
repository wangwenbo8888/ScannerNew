// FiveChainAdapter.cpp — 适配层实现（自持算子宿主＋07 桥）＋FcHarnessCompat 方法实现
#include "FiveChainAdapter.h"
#include "harness.h"

#include "pipelines/ISceneFeed.h"
#include "ICloudWarehouse.h"
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"
#include "sched/IFrameSource.h"
#include "SlotRing.h"
#include "EnhancedFrame.h"
#include "fc_executor.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>    // cv::cuda::sum（激光掩膜内容检查）
#include <opencv2/imgcodecs.hpp>    // 诊断帧转储
#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <array>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include "ccl_cpu.h"

namespace Scanner::pipeline {

FiveChainAdapter::FiveChainAdapter(const FiveChainConfig& cfg,
                                   FrameObsAccumulator& obs,
                                   Scanner::data::ICloudWarehouse* warehouse,
                                   ISceneFeed* sceneFeed,
                                   const nlohmann::json* opJson)
    : cfg_(cfg), obs_(obs), warehouse_(warehouse), sceneFeed_(sceneFeed),
      opJson_(opJson) {
    // 算子参数：opJson 非空＝沙盒对拍口径（同参对拍）；空＝算子默认（产线）
    calib::LaserMarkingSeparationParams sepParams;
    if (opJson != nullptr && opJson->contains("mask_separation")) {
        try {
            sepParams = calib::LaserMarkingSeparationParams::fromJson(
                (*opJson)["mask_separation"]);
        } catch (const std::exception& e) {
            JMW_LOG_WARN("07-FiveChain",
                "[FiveChainAdapter] mask_separation 参数解析失败（用默认）: {}", e.what());
        }
    }
    // 261028 确定性开关：禁用 CUDA Graph（走 legacy 逐核路径）。
    // 【261029 复位 true：CONT4 实证 legacy 路径 FPS 289→250/CV 3.7%→16%——
    // 261028 conic 8→0 的真根因是捕获模式毒化（已由 flat ThreadLocal＋sep
    // Relaxed 根治），sep 图本身确定性无恙。JMW_FC_SEPNOGRAPH=1 可再禁（诊断）】
    sepParams.useCudaGraph = getenv("JMW_FC_SEPNOGRAPH") == nullptr;
    sepL_ = std::make_unique<calib::LaserMarkingSeparationCUDA>(sepParams);
    sepR_ = std::make_unique<calib::LaserMarkingSeparationCUDA>(sepParams);
    calib::LaserCloudFuseCUDAParams fp;
    fp.voxelSize = static_cast<float>(cfg_.voxelSizeMm);
    fp.reserveVoxelCount = static_cast<size_t>(cfg_.voxelReserve);
    fp.growSlotCount = static_cast<size_t>(cfg_.voxelGrowSlots);   // 沙盒 L533 同款
    fp.collectStatistics = true;
    fuse_ = std::make_unique<calib::LaserCloudFuseCuda>(fp);
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 构造: voxel={}mm reserve={}M graySlots={} opJson={}",
        cfg_.voxelSizeMm, cfg_.voxelReserve / 1000000, cfg_.graySlots,
        opJson != nullptr ? "沙盒对拍参" : "产线默认");
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
    // 261028 根修：标点模式激光掩膜非空但全零——在入口拦截（不做任何 GPU
    // 提交），彻底避免 flat/undistort/interp/v3 在垃圾数据上操作后崩溃。
    // cv::cuda::sum 是 GPU 归约（~0.1ms，标量回传）——成本可忽略
    {
        const cv::Scalar s = cv::cuda::sum(*p.sepL.d_laserMask);
        if (s[0] < 1.0) {
            fail("steger_fast", p, "laser mask 全零（标点模式/无激光帧）");
            return 0;
        }
    }
    try {
        timeBegin();
        bool deferred = false;
        if (L.flatL_ && L.flatR_) {
            // flat 路径（沙盒 extractor=flat）：延迟提交——双眼连续入队
            // 不同步（stats 28B D2H 在飞，完成段排水）。分眼实例（计数器私有）
            p.stegerL = L.flatL_->ExecuteDeferred(gL, *p.sepL.d_laserMask, ls);
            p.stegerR = L.flatR_->ExecuteDeferred(gR, *p.sepR.d_laserMask, ls);
            deferred = true;
            if (!p.stegerL.success || !p.stegerR.success) {
                fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
                return 0;
            }
            // 261027 零点护栏（延迟路径——标点模式激光掩膜空/近空时 flat 产
            // 0 点，d_line_ids 可能为 null → undistort 解引用崩溃真机实证）
            if (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
                || !p.stegerL.d_line_ids || !p.stegerR.d_line_ids) {
                fail("undistort_cuda", p, "steger 延迟路径 0 中心点（护护栏）");
                return deferred ? 1 : 0;   // 延迟：仍须排水（防在飞 GPU 工作悬挂）
            }
        } else if (L.stegerL_ && L.stegerR_) {
            // steger_fast legacy 路径（分眼实例）
            L.stegerL_->SetCollectHostPoints(false);
            L.stegerR_->SetCollectHostPoints(false);
            p.stegerL = L.stegerL_->Execute(gL, *p.sepL.d_laserMask, ls,
                                            calib::GroupMode::Flat);
            p.stegerR = L.stegerR_->Execute(gR, *p.sepR.d_laserMask, ls,
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
            fail("steger_fast", p, "LaserOps 未装配（flat/steger 均空）");
            return 0;
        }

        // undistort（延迟模式＝容量宽视图；legacy＝精确视图）
        // 双实例（沙盒 undistCudaL_/R_ 语义）：实时＝每帧温补快照 SetParams；
        // 离线对拍（metaValid=false）＝构造期沙盒标定参直用
        timeBegin();
        if (L.undistGL_ && L.undistGR_) {
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
                L.undistGL_->SetParams(ul);
                L.undistGR_->SetParams(ur);
            }
            p.undistCudaL = L.undistGL_->Execute(*p.stegerL.d_centerPoints,
                                                 *p.stegerL.d_line_ids, ls);
            p.undistCudaR = L.undistGR_->Execute(*p.stegerR.d_centerPoints,
                                                 *p.stegerR.d_line_ids, ls);
        }
        if (!p.undistCudaL.success || !p.undistCudaR.success) {
            fail("undistort_cuda", p,
                 p.undistCudaL.message + " | " + p.undistCudaR.message);
            return deferred ? 1 : 0;
        }

        // interp（延迟模式＝flat 分眼设备计数驱动；legacy＝直接）
        timeBegin();
        if (deferred && L.flatL_ && L.flatR_) {
            p.interpL = L.interpL_->ExecuteDeferred(
                *p.undistCudaL.d_rectifiedPoints, *p.undistCudaL.d_line_ids,
                L.flatL_->GetDeviceCountPtr(), ls);
            p.interpR = L.interpR_->ExecuteDeferred(
                *p.undistCudaR.d_rectifiedPoints, *p.undistCudaR.d_line_ids,
                L.flatR_->GetDeviceCountPtr(), ls);
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
    if (deferred && L.flatL_ && L.flatR_) {
        L.flatL_->FinishDeferred(p.stegerL);
        L.flatR_->FinishDeferred(p.stegerR);
        if (!p.stegerL.success || !p.stegerR.success) {
            fail("steger_fast", p, p.stegerL.message + " | " + p.stegerR.message);
        }
    }
    // 零点检查（261031 补 d_line_ids——棋盘格扫描崩溃实证：d_centerPoints 可
    // 非 null 但 d_line_ids null 时 interp/undistort 下游在 null+8 崩溃）
    if (p.stegerL.success && p.stegerR.success
        && (!p.stegerL.d_centerPoints || !p.stegerR.d_centerPoints
            || !p.stegerL.d_line_ids || !p.stegerR.d_line_ids
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
        // CCL 门限＝harness 成员（沙盒 config ccl 节默认 800/30000/0——对拍同参）
        p.cclL = scr::runCclCpuFromRuns(runsL, nL, cols, rows,
                                        cclMinArea_, cclMaxArea_, cclTopX_);
        p.cclR = scr::runCclCpuFromRuns(runsR, nR, cols, rows,
                                        cclMinArea_, cclMaxArea_, cclTopX_);
        if (!p.cclL.success || !p.cclR.success) {
            fail("ccl", p, "runCclCpuFromRuns 失败");
            diagCclFail_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        diagRoisL_.fetch_add(static_cast<long>(p.cclL.components.size()),
                             std::memory_order_relaxed);
    } catch (const std::exception& e) {
        fail("ccl", p, e.what());
        diagCclFail_.fetch_add(1, std::memory_order_relaxed);
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
        diagConicValid_.fetch_add(p.markersValid, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        fail("conic_reconstruct", p, e.what());
        return;
    }

    // ── 261031 Q 基准三角化（DUAL-Z 根治）：conic 锥面相交与激光 Q 三角化
    //    同源标定不同数学路径 → 系统性 ~2.3mm 深度偏差。将配对的椭圆中心
    //    用 Q 矩阵直接三角化（与激光 ReconstructInto 同路径），替换锥面相交
    //    的位置输出。validCircle/validPlane 质量门保留（conic 筛选仍有效），
    //    仅替换 centerX/Y/Z。法线保留 conic 值（用于配准/融合方向参考）。
    //    env JMW_FC_QBASIS=0 可回退锥面相交（对照用）。
    {
        static const bool sQBasis = [] {
            const char* e = getenv("JMW_FC_QBASIS");
            return e ? atoi(e) != 0 : true;   // 默认开
        }();
        if (sQBasis && p.markersValid > 0) {
            const cv::Matx44d& Qm = calib_.Q;
            int replaced = 0;
            int skipped = 0;
            for (auto& m : p.conic.markers) {
                if (!m.validCircle || !m.validPlane) { ++skipped; continue; }
                if (m.leftEllipseIdx < 0 || m.rightEllipseIdx < 0) { ++skipped; continue; }
                if (m.leftEllipseIdx >= static_cast<int>(p.ellL.size())
                    || m.rightEllipseIdx >= static_cast<int>(p.ellR.size())) { ++skipped; continue; }
                const auto& eL = p.ellL[static_cast<size_t>(m.leftEllipseIdx)];
                const auto& eR = p.ellR[static_cast<size_t>(m.rightEllipseIdx)];
                if (!eL.success || !eR.success) { ++skipped; continue; }
                // Q 三角化（矫正坐标系——与激光 ReconstructInto 同式）
                const double uL = eL.centerX, vL = eL.centerY;
                const double uR = eR.centerX;
                const double d = uL - uR;
                if (std::abs(d) < 1e-9) { ++skipped; continue; }
                const double X = Qm(0,0)*uL + Qm(0,1)*vL + Qm(0,2)*d + Qm(0,3);
                const double Y = Qm(1,0)*uL + Qm(1,1)*vL + Qm(1,2)*d + Qm(1,3);
                const double Z = Qm(2,0)*uL + Qm(2,1)*vL + Qm(2,2)*d + Qm(2,3);
                const double W = Qm(3,0)*uL + Qm(3,1)*vL + Qm(3,2)*d + Qm(3,3);
                if (std::abs(W) < 1e-12) { ++skipped; continue; }
                const double qX = X / W, qY = Y / W, qZ = Z / W;
                if (!std::isfinite(qX) || !std::isfinite(qY) || !std::isfinite(qZ)) { ++skipped; continue; }
                // 合理性门：Q 结果与锥面结果差 >30mm ＝异常——保留锥面值
                const double dx = qX - m.centerX, dy = qY - m.centerY, dz = qZ - m.centerZ;
                if (dx*dx + dy*dy + dz*dz > 900.0) { ++skipped; continue; }
                // 首帧诊断（Q vs 锥面）
                if (replaced == 0 && diagConicValid_.load() < 50) {
                    fprintf(stderr,
                        "[Q-BASIS] cone=(%.1f,%.1f,%.1f) Q=(%.1f,%.1f,%.1f) "
                        "uL=%.1f d=%.1f\n",
                        m.centerX, m.centerY, m.centerZ, qX, qY, qZ, uL, d);
                }
                m.centerX = qX;
                m.centerY = qY;
                m.centerZ = qZ;
                ++replaced;
            }
            if (diagConicValid_.load() < 50)
                fprintf(stderr, "[Q-BASIS] replaced=%d skipped=%d\n", replaced, skipped);
        }
    }
    // 逐级诊断计数（split/ellipse/match 成功帧）
    if (p.splitL.success && p.splitR.success)
        diagSplit_.fetch_add(1, std::memory_order_relaxed);
    if (p.markersDetected > 0)
        diagEllipse_.fetch_add(1, std::memory_order_relaxed);
    diagMatchPairs_.fetch_add(static_cast<long>(p.mm.centerMatches.size()),
                              std::memory_order_relaxed);
}

// ============================================================================
// 配准（三级降级：光流 → frame_fuse 兜底【matched≥门 且 rmse<门】→ 沿用）
// —— 沙盒 harness.cpp L1269-1548 全量对齐（261025 对拍补全）
// ============================================================================
namespace {
/// 规则网格板检测（沙盒 L276-302）：最近邻距离变异系数 < cvTol 判网格
/// （平移对称＝原理性不可分辨——裁决硬拒）
bool fcDetectRegularGrid(const std::vector<cv::Point3d>& pts, double cvTol = 0.012) {
    const size_t n = pts.size();
    if (n < 8) return false;
    std::vector<double> nn(n);
    for (size_t i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::max();
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double dx = pts[i].x - pts[j].x;
            const double dy = pts[i].y - pts[j].y;
            const double dz = pts[i].z - pts[j].z;
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < best) best = d;
        }
        if (best <= 0.0) return true;   // 重合点＝病态网格
        nn[i] = best;
    }
    double mean = 0.0;
    for (double v : nn) mean += v;
    mean /= static_cast<double>(n);
    if (mean <= 0.0) return false;
    double var = 0.0;
    for (double v : nn) var += (v - mean) * (v - mean);
    var /= static_cast<double>(n);
    const double cv = std::sqrt(var) / mean;
    return cv < cvTol;
}
} // namespace

void FcHarnessCompat::runRegistration(scr::CycleProductsCompat& p, bool record) {
    (void)record;
    // 基线：先置上帧 R/T（hold 路径即保留此值）
    if (!prevState_.empty()) {
        for (int i = 0; i < 9; ++i) p.R[i] = prevState_.R.val[i];
        for (int i = 0; i < 3; ++i) p.T[i] = prevState_.T.val[i];
    }

    if (!p.conic.success || p.markersValid == 0) {
        p.regSource = "skip";
        diagRegSkip_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // 规则网格板检测（硬拒锁存直至会话结束）
    if (!gridBoardRejected_) {
        std::vector<cv::Point3d> probe;
        for (const auto& m : p.conic.markers) {
            if (m.validCircle) probe.emplace_back(m.centerX, m.centerY, m.centerZ);
        }
        if (fcDetectRegularGrid(probe)) {
            gridBoardRejected_ = true;
            JMW_LOG_WARN("07-FiveChain",
                "[fivechain] 规则阵列标记点板：平移对称不可分辨，配准拒绝");
        }
    }
    if (gridBoardRejected_) {
        fail("registration", p, "规则阵列标记点板：配准拒绝");
        p.regSource = "hold";
        return;
    }

    try {
        // 有效圆（validCircle）center/normal → 光流（成功语义 matched≥3）
        std::vector<cv::Point3d> centers;
        std::vector<cv::Vec3d> normals;
        for (const auto& m : p.conic.markers) {
            if (!m.validCircle) continue;
            centers.emplace_back(m.centerX, m.centerY, m.centerZ);
            normals.emplace_back(m.normalX, m.normalY, m.normalZ);
        }
        p.regFlow = flowOp_->Execute(centers, normals, prevState_);
        // 261026 精度对拍探针（JMW_RTDBG=1）：R/T 17 位打印——与沙盒 harness
        // L1312-1323 同款同格式（regLoop 处理序＝两边同基準，逐行可比）
        if (getenv("JMW_RTDBG") && p.regFlow.success) {
            fprintf(stderr, "[RTDBG] matched=%d rmse=%.17g",
                    static_cast<int>(p.regFlow.getMatchedCount()),
                    p.regFlow.statistics.rmse);
            for (int i = 0; i < 9; ++i)
                fprintf(stderr, " R%d=%.17g", i, p.regFlow.R.val[i]);
            for (int i = 0; i < 3; ++i)
                fprintf(stderr, " T%d=%.17g", i, p.regFlow.T.val[i]);
            fprintf(stderr, "\n");
        }
        p.flowMatchedDiag = static_cast<int>(p.regFlow.getMatchedCount());
        p.flowRmseDiag = p.regFlow.statistics.rmse;
        p.flowMessageDiag = p.regFlow.message;
        // 误差趋势门：超"开局基线中位×倍率"判失败（专抓温漂/斜视角全体慢变）
        const double driftRatio = cfg_.runtime.flowRmseDriftRatio;
        bool driftRejected = false;
        if (driftRatio > 0.0 && p.regFlow.success) {
            if (flowRmseBaselineMed_ < 0.0) {
                flowRmseBaseline_.push_back(p.regFlow.statistics.rmse);
                if (flowRmseBaseline_.size() >= 50) {
                    auto s = flowRmseBaseline_;
                    std::sort(s.begin(), s.end());
                    flowRmseBaselineMed_ = s[s.size() / 2];
                }
            } else if (p.regFlow.statistics.rmse > driftRatio * flowRmseBaselineMed_) {
                driftRejected = true;
                p.flowMessageDiag += " [drift-gate: rmse>"
                                     + std::to_string(driftRatio * flowRmseBaselineMed_)
                                     + " = " + std::to_string(driftRatio) + "x baseline]";
            }
        }
        const bool flowOk = !driftRejected && p.regFlow.success
                            && static_cast<int>(p.regFlow.getMatchedCount()) >= 3;
        // 261030 运动合理性门（飞点-sheet 根治）：帧间平移/旋转跳变超物理界限
        // ＝光流劣解（真机位姿增量实测 P95=16.1mm/最大 28.9mm 每帧＝55fps 下
        // 1.6m/s 不可能——劣解整帧平面被甩离＝"成片飞点"）。拒采纳 → 走既有
        // frame_fuse 兜底/hold。阈值 env JMW_FC_JUMPMM（默认 15mm，0=关）＋
        // 旋转 8°（快扫保守界）
        bool flowOkGated = flowOk;
        if (flowOkGated && !prevState_.empty()) {
            static const double sJumpMm = [] {
                const char* e = getenv("JMW_FC_JUMPMM");
                return e ? atof(e) : 15.0;
            }();
            const cv::Vec3d dt(p.regFlow.T.val[0] - prevState_.T.val[0],
                               p.regFlow.T.val[1] - prevState_.T.val[1],
                               p.regFlow.T.val[2] - prevState_.T.val[2]);
            const double jumpMm = cv::norm(dt);
            // 旋转增量角：R_δ = R_cand·R_prevᵀ
            const cv::Matx33d Rc(p.regFlow.R.val);
            const cv::Matx33d Rp(prevState_.R.val);
            const cv::Matx33d Rd = Rc * Rp.t();
            const double tr = Rd(0, 0) + Rd(1, 1) + Rd(2, 2);
            const double angDeg = std::acos(std::min(1.0, std::max(-1.0,
                (tr - 1.0) / 2.0))) * 180.0 / CV_PI;
            if ((sJumpMm > 0 && jumpMm > sJumpMm) || angDeg > 8.0) {
                flowOkGated = false;
                p.flowMessageDiag += " [jump-gate: dT=" + std::to_string(jumpMm)
                                   + "mm ang=" + std::to_string(angDeg) + "deg]";
                diagRegHold_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (flowOkGated) {
            p.regSource = prevState_.empty() ? "first" : "flow";
            p.regMatched = static_cast<int>(p.regFlow.getMatchedCount());
            adoptRegistration(p.regFlow, p);
        } else if (!prevState_.empty() && !globalSet_.empty()) {
            // —— frame_fuse 兜底：快路径视野捞点 → 慢路径三角索引全库投票 ——
            calib::MarkerPointSet cur;
            for (const auto& m : p.conic.markers) {
                if (!m.validCircle) continue;
                cur.positions.emplace_back(m.centerX, m.centerY, m.centerZ);
                cur.normals.emplace_back(m.normalX, m.normalY, m.normalZ);
            }
            // 法线合理性门：双峰法线（正反混合）拒走兜底（折中解不如拒绝）
            lastNormalOpposeRatio_ = checkNormalPlausibility(cur.normals);
            if (lastNormalOpposeRatio_ > 0.25) {
                JMW_LOG_WARN("07-FiveChain",
                    "[fivechain] 法线双峰（反对占比 {:.1f}%>25%）——拒走兜底",
                    lastNormalOpposeRatio_ * 100.0);
                p.regSource = "hold";
            } else {
                p.usedFrameFuse = true;
                // 子集尺寸与算子上限解绑（防误配）
                const int libMax = std::max(6, std::min(64,
                    std::min(fuseParams_.triangleMaxPoints,
                             cfg_.runtime.fuseSubsetMax)));
                std::vector<int> sel;
                if (holdStreak_ <= 15) {
                    sel = cullByView(cur, libMax);
                    if (sel.size() >= 3)
                        p.regFuse = fuseExecute(cur, sel);
                }
                bool fuseOk = acceptFuse(p.regFuse);
                bool fromSlow = false;      // 本帧胜出解是否来自慢路径
                std::vector<int> sel2;      // 慢路径子集（胜出时译下标）
                const bool slowPathTurn = holdStreak_ <= 15 || (holdStreak_ % 3) == 0;
                if (!fuseOk && slowPathTurn) {
                    // 索引重建节流（≥5 帧才重建；旧索引仍可查）
                    if (libIndexVersion_ != libVersion_
                        && (lastIndexRebuildCycle_ < 0
                            || cycleCount_ - lastIndexRebuildCycle_ >= 5)) {
                        if (!libIndex_) libIndex_ = std::make_unique<scr::TriangleLibraryIndex>();
                        libIndex_->rebuild(globalSet_.positions, globalSet_.normals);
                        libIndexVersion_ = libVersion_;
                        lastIndexRebuildCycle_ = cycleCount_;
                    }
                    if (libIndex_ && !libIndex_->empty()) {
                        sel2 = libIndex_->query(cur.positions, libMax);
                        if (sel2.size() >= 3 && sel2 != sel) {
                            p.regFuse = fuseExecute(cur, sel2);
                            fromSlow = true;
                            fuseOk = acceptFuse(p.regFuse);
                        }
                    }
                }
                p.fuseSlowPath = fromSlow && fuseOk;
                // 跳变门：大跳且支持不足且未经慢路径复核 → 拒
                if (fuseOk) {
                    const cv::Matx33d dR = cv::Matx33d(p.regFuse.R) * cv::Matx33d(prevState_.R).t();
                    const double tr = dR(0, 0) + dR(1, 1) + dR(2, 2);
                    const double dAng = std::acos(std::min(1.0,
                        std::max(-1.0, (tr - 1.0) * 0.5))) * 180.0 / CV_PI;
                    const double dTmm = cv::norm(cv::Vec3d(p.regFuse.T) - cv::Vec3d(prevState_.T));
                    if ((dTmm > 500.0 || dAng > 60.0)
                        && p.regFuse.matchedCount < 7 && !fromSlow) {
                        fuseOk = false;
                        p.regFuse.message = "jump gate: 大跳且支持不足且未经慢路径复核，判 hold";
                    }
                }
                if (fuseOk) {
                    p.regSource = "fuse";
                    p.regMatched = static_cast<int>(p.regFuse.matchedCount);
                    for (int i = 0; i < 9; ++i) p.R[i] = p.regFuse.R.val[i];
                    for (int i = 0; i < 3; ++i) p.T[i] = p.regFuse.T.val[i];
                    // 子集局部下标 → 库下标（快/慢路径按胜出来源译码）
                    const bool slowWon = fromSlow && !sel2.empty();
                    const std::vector<int>& usedSel = slowWon ? sel2 : sel;
                    std::vector<int> libIdxOfPair(usedSel.size(), -1);
                    for (size_t j = 0; j < usedSel.size(); ++j)
                        libIdxOfPair[j] = usedSel[j];
                    std::vector<int> gidByIndex(globalSet_.positions.size(), -1);
                    for (const auto& kv : globalIndexById_)
                        if (kv.second < gidByIndex.size()) gidByIndex[kv.first] = kv.second;
                    prevState_.rawPositions.clear();
                    prevState_.rawNormals.clear();
                    prevState_.globalIds.clear();
                    prevState_.R = p.regFuse.R;
                    prevState_.T = p.regFuse.T;
                    constexpr double kMergeRadiusMm = 2.0;   // 与 verifyGate 同口径
                    int gi = 0;
                    for (const auto& m : p.conic.markers) {
                        if (!m.validCircle) continue;
                        int gid = -1;
                        for (const auto& co : p.regFuse.correspondences)
                            if (co.first == gi
                                && co.second >= 0
                                && co.second < static_cast<int>(libIdxOfPair.size())) {
                                const int libIdx = libIdxOfPair[static_cast<size_t>(co.second)];
                                if (libIdx >= 0 && libIdx < static_cast<int>(gidByIndex.size())
                                    && gidByIndex[libIdx] >= 0)
                                    gid = gidByIndex[libIdx];
                                break;
                            }
                        const cv::Point3d c(m.centerX, m.centerY, m.centerZ);
                        const cv::Vec3d n(m.normalX, m.normalY, m.normalZ);
                        const auto& Rf = p.regFuse.R;
                        const auto& Tf = p.regFuse.T;
                        const cv::Point3d gpos(Rf(0,0)*c.x + Rf(0,1)*c.y + Rf(0,2)*c.z + Tf(0),
                                               Rf(1,0)*c.x + Rf(1,1)*c.y + Rf(1,2)*c.z + Tf(1),
                                               Rf(2,0)*c.x + Rf(2,1)*c.y + Rf(2,2)*c.z + Tf(2));
                        const cv::Vec3d gnorm(Rf(0,0)*n(0) + Rf(0,1)*n(1) + Rf(0,2)*n(2),
                                              Rf(1,0)*n(0) + Rf(1,1)*n(1) + Rf(1,2)*n(2),
                                              Rf(2,0)*n(0) + Rf(2,1)*n(1) + Rf(2,2)*n(2));
                        if (gid < 0) {
                            // 未匹配点：合并半径内归并旧号（须过有向法线门），
                            // 否则会话计数器发新号入图
                            double bestD = 1e30;
                            int nearLib = -1;
                            for (size_t li = 0; li < globalSet_.positions.size(); ++li) {
                                const double dx = gpos.x - globalSet_.positions[li].x;
                                const double dy = gpos.y - globalSet_.positions[li].y;
                                const double dz = gpos.z - globalSet_.positions[li].z;
                                const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
                                if (d < bestD) { bestD = d; nearLib = static_cast<int>(li); }
                            }
                            const double mergeNormalMinCos = std::cos(
                                fuseParams_.triangleNormalMaxAngleDeg * CV_PI / 180.0);
                            if (nearLib >= 0 && bestD < kMergeRadiusMm
                                && gidByIndex[nearLib] >= 0
                                && gnorm.dot(globalSet_.normals[static_cast<size_t>(nearLib)])
                                       >= mergeNormalMinCos)
                                gid = gidByIndex[nearLib];
                            else
                                gid = nextGid_++;
                        }
                        // 写库体检：非有限值不入库（NaN 毒点致索引静默弃组）
                        const bool finiteOk = std::isfinite(gpos.x) && std::isfinite(gpos.y)
                            && std::isfinite(gpos.z) && std::isfinite(gnorm(0))
                            && std::isfinite(gnorm(1)) && std::isfinite(gnorm(2));
                        if (!finiteOk) {
                            JMW_LOG_WARN("07-FiveChain",
                                "[fivechain] frame_fuse: 写库拒绝非有限值点（gid={} 跳过）", gid);
                            prevState_.rawPositions.push_back(c);
                            prevState_.rawNormals.push_back(n);
                            prevState_.globalIds.push_back(gid);
                            ++gi;
                            continue;
                        }
                        const auto it = globalIndexById_.find(gid);
                        if (it == globalIndexById_.end()) {
                            globalIndexById_[gid] = globalSet_.positions.size();
                            globalSet_.positions.push_back(gpos);
                            globalSet_.normals.push_back(gnorm);
                        } else {
                            globalSet_.positions[it->second] = gpos;
                            globalSet_.normals[it->second] = gnorm;
                        }
                        prevState_.rawPositions.push_back(c);
                        prevState_.rawNormals.push_back(n);
                        prevState_.globalIds.push_back(gid);
                        ++gi;
                    }
                    ++libVersion_;   // 库已写 → 索引脏
                } else {
                    p.regSource = "hold";
                }
            }   // 法线合理性通过分支闭合
        } else {
            p.regSource = "hold";
        }
        // hold 连击记账（捞点半径自适应＋停手门）；非 hold 即清零
        if (p.regSource == "hold") ++holdStreak_;
        else holdStreak_ = 0;
        // 诊断计数
        if (p.regSource == "first") diagRegFirst_.fetch_add(1, std::memory_order_relaxed);
        else if (p.regSource == "flow") diagRegFlow_.fetch_add(1, std::memory_order_relaxed);
        else if (p.regSource == "fuse") diagRegFuse_.fetch_add(1, std::memory_order_relaxed);
        else if (p.regSource == "hold") diagRegHold_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        fail("registration", p, e.what());
        p.regSource = "hold";
        diagRegHold_.fetch_add(1, std::memory_order_relaxed);
    }
    // hold/skip：保留基线上帧 R/T（prevState 停在上一次成功帧）
}

// ---------------------------------------------------------------------------
// adoptRegistration — 收编光流成功结果：置 R/T、重建 prevState（含新点发号
// 入图/查重/候选池）、维护全局地图（沙盒 L1556-1666 对齐）
// ---------------------------------------------------------------------------
void FcHarnessCompat::adoptRegistration(calib::MarkerOpticalFlowFuseCPUResult& res,
                                        scr::CycleProductsCompat& p) {
    for (int i = 0; i < 9; ++i) p.R[i] = res.R.val[i];
    for (int i = 0; i < 3; ++i) p.T[i] = res.T.val[i];
    prevState_.rawPositions.clear();
    prevState_.rawNormals.clear();
    prevState_.globalIds.clear();
    prevState_.R = res.R;
    prevState_.T = res.T;
    // 发号查重用：地图下标 → 编号反查表
    std::vector<int> gidByIndex(globalSet_.positions.size(), -1);
    for (const auto& kv : globalIndexById_)
        if (kv.second < gidByIndex.size()) gidByIndex[kv.second] = kv.first;
    for (auto& m : res.markers) {
        // 新点发号前查重：全局系最近条目命中查重半径内沿用旧号（须过法线同向门）
        if (m.globalId < 0) {
            int nearGid = -1;
            double best = cfg_.runtime.mintDedupRadiusMm;
            for (size_t k = 0; k < globalSet_.positions.size(); ++k) {
                const double dx = m.transformedPosition.x - globalSet_.positions[k].x;
                const double dy = m.transformedPosition.y - globalSet_.positions[k].y;
                const double dz = m.transformedPosition.z - globalSet_.positions[k].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < best && gidByIndex[k] >= 0) {
                    if (!mergeNormalCompatible(m.transformedNormal,
                                               globalSet_.normals[k]))
                        continue;
                    best = d;
                    nearGid = gidByIndex[k];
                }
            }
            if (nearGid >= 0) {
                m.globalId = nearGid;
            } else {
                // 候选期：新点连续 3 次重观测一致才转正入图（防瞬态假点）
                bool promoted = false;
                for (auto& cand : candidates_) {
                    const double dx = m.transformedPosition.x - cand.position.x;
                    const double dy = m.transformedPosition.y - cand.position.y;
                    const double dz = m.transformedPosition.z - cand.position.z;
                    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d < kCandidateMergeMm) {
                        ++cand.count;
                        cand.position = m.transformedPosition;   // 最新坐标
                        cand.normal = m.transformedNormal;
                        if (cand.count >= kPromoteThreshold) {
                            m.globalId = nextGid_++;
                            promoted = true;
                        }
                        break;
                    }
                }
                if (!promoted && m.globalId < 0) {
                    candidates_.push_back({m.transformedPosition,
                                           m.transformedNormal, 1});
                    m.globalId = -1;   // 候选中——prevState 记录但入图
                }
            }
        } else {
            nextGid_ = std::max(nextGid_, m.globalId + 1);
        }
        prevState_.rawPositions.push_back(m.rawPosition);
        prevState_.rawNormals.push_back(m.rawNormal);
        prevState_.globalIds.push_back(m.globalId);
        // 写库体检：非有限值不入图
        const auto& tp = m.transformedPosition;
        const auto& tn = m.transformedNormal;
        const bool finiteOk = std::isfinite(tp.x) && std::isfinite(tp.y)
            && std::isfinite(tp.z) && std::isfinite(tn(0)) && std::isfinite(tn(1))
            && std::isfinite(tn(2));
        if (!finiteOk) {
            JMW_LOG_WARN("07-FiveChain",
                "[fivechain] adoptRegistration: 写库拒绝非有限值点（gid={} 跳过入图）",
                m.globalId);
            continue;
        }
        // 地图维护：新号追加、旧号最新覆盖；候选（-1）不入图
        if (m.globalId < 0) continue;
        const auto it = globalIndexById_.find(m.globalId);
        if (it == globalIndexById_.end()) {
            globalIndexById_[m.globalId] = globalSet_.positions.size();
            globalSet_.positions.push_back(m.transformedPosition);
            globalSet_.normals.push_back(m.transformedNormal);
        } else {
            globalSet_.positions[it->second] = m.transformedPosition;
            globalSet_.normals[it->second] = m.transformedNormal;
        }
    }
    ++libVersion_;   // 库已写 → 三角索引脏
    // 候选池过期清理：超 30 帧未转正（count<2）视为瞬态误检丢弃
    if (candidates_.size() > 4) {
        candidates_.erase(
            std::remove_if(candidates_.begin(), candidates_.end(),
                           [](const CandidateEntry& c) { return c.count < 2; }),
            candidates_.end());
    }
}

// ---------------------------------------------------------------------------
// frame_fuse 兜底三助手（沙盒 L1672-1726 对齐）
// ---------------------------------------------------------------------------
std::vector<int> FcHarnessCompat::cullByView(const calib::MarkerPointSet& cur,
                                             int maxCount) const {
    std::vector<int> out;
    if (cur.empty() || globalSet_.positions.empty() || maxCount < 3) return out;
    cv::Vec3d c(0, 0, 0);
    for (const auto& q : cur.positions)
        c += cv::Vec3d(q.x, q.y, q.z);
    c /= static_cast<double>(cur.size());
    // 库点经上帧 R/T 逆变换到当前帧系；半径自适应 150+30×hold 连击（封顶 600）
    const double radius = std::min(600.0, 150.0 + 30.0 * static_cast<double>(holdStreak_));
    std::vector<std::pair<double, int>> byDist;
    byDist.reserve(globalSet_.positions.size());
    const auto& Rp = prevState_.R;
    const auto& Tp = prevState_.T;
    for (size_t i = 0; i < globalSet_.positions.size(); ++i) {
        const auto& g = globalSet_.positions[i];
        const double gx = g.x - Tp(0), gy = g.y - Tp(1), gz = g.z - Tp(2);
        const cv::Vec3d pc(Rp(0, 0) * gx + Rp(1, 0) * gy + Rp(2, 0) * gz,
                           Rp(0, 1) * gx + Rp(1, 1) * gy + Rp(2, 1) * gz,
                           Rp(0, 2) * gx + Rp(1, 2) * gy + Rp(2, 2) * gz);
        byDist.emplace_back(cv::norm(pc - c), static_cast<int>(i));
    }
    std::sort(byDist.begin(), byDist.end());
    for (const auto& kv : byDist) {
        if (static_cast<int>(out.size()) >= maxCount) break;
        if (kv.first <= radius) out.push_back(kv.second);
    }
    if (out.size() < 3) {   // 半径内不足：退取全库最近（终审兜底）
        out.clear();
        for (size_t i = 0; i < byDist.size() && static_cast<int>(out.size()) < maxCount; ++i)
            out.push_back(byDist[i].second);
    }
    std::sort(out.begin(), out.end());   // 库下标序＝确定性
    return out;
}

calib::FrameFuseCPUResult FcHarnessCompat::fuseExecute(
    const calib::MarkerPointSet& cur, const std::vector<int>& libIdx) const {
    calib::MarkerPointSet gms;   // 子集装配（契约：绝不传全库）
    gms.positions.reserve(libIdx.size());
    gms.normals.reserve(libIdx.size());
    for (int li : libIdx) {
        gms.positions.push_back(globalSet_.positions[static_cast<size_t>(li)]);
        gms.normals.push_back(globalSet_.normals[static_cast<size_t>(li)]);
    }
    return fuseOp_->Execute(cur, gms);
}

bool FcHarnessCompat::acceptFuse(const calib::FrameFuseCPUResult& r) const {
    return r.success
           && static_cast<int>(r.matchedCount) >= cfg_.runtime.fuseMinMatched
           && r.rmse < cfg_.runtime.fuseMaxRmseMm;
}

double FcHarnessCompat::checkNormalPlausibility(
    const std::vector<cv::Vec3d>& normals) const {
    if (normals.size() < 4) return 0.0;
    cv::Vec3d median(0, 0, 0);
    for (const auto& n : normals) median += n;
    const double len = std::sqrt(median.dot(median));
    if (len < 1e-12) return 1.0;   // 全互相抵消＝极端双峰
    median /= len;
    int oppose = 0;
    for (const auto& n : normals)
        if (n.dot(median) < 0.0) ++oppose;
    return static_cast<double>(oppose) / static_cast<double>(normals.size());
}

bool FcHarnessCompat::mergeNormalCompatible(const cv::Vec3d& a,
                                            const cv::Vec3d& b) const {
    const double minCos = std::cos(
        fuseParams_.triangleNormalMaxAngleDeg * CV_PI / 180.0);
    return a.dot(b) >= minCos;
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
    // 分眼裁剪原点。优先级：分眼 env（JMW_FC_CROPL_X/Y、JMW_FC_CROPR_X/Y）
    // ＞双眼统一（JMW_FC_CROP_X/Y）＞本默认。
    // 261030 默认＝ConicCropSolve 实证值（conic 有效性为目标——激光目标解出的
    // L=(180,119)R=(180,68) 线上 conic 反归零＝目标脱节）：yL=yR=102 时 conic
    // 2→10、激光 2535→9058（3 帧累计）；x 用居中推定 (2048-1688)/2=180。
    // env -1 回居中；换机/重标定后须重解（JMW_FC_CONICROP 转储重扫）
    int cropXL_ = -1, cropYL_ = 102, cropXR_ = -1, cropYR_ = 102;
    std::atomic<uint64_t> cropFrames_{0};   // 裁剪帧计数（诊断）

    scr::FrameSourceCompat src;

    /// 诊断帧转储（env JMW_FC_DUMPFRAMES=N——前 N 帧 L/R 原图落盘
    /// diagnostics/fc_frames/，裁剪前 2048×1536——离线原点求解器原料）
    int dumpMax_ = 0;
    int dumpCount_ = 0;
    bool useSnapshotCalib_ = true;     // false＝名义温档口径（快照注入关闭）

    void maybeDump() {
        if (dumpCount_ >= dumpMax_ || !cur) return;
        try {
            namespace fsd = std::filesystem;
            const fsd::path dir = "diagnostics/fc_frames";
            fsd::create_directories(dir);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%06d", dumpCount_);
            cv::imwrite((dir / ("L_" + std::string(buf) + ".png")).string(), cur->grayL);
            cv::imwrite((dir / ("R_" + std::string(buf) + ".png")).string(), cur->grayR);
            ++dumpCount_;
        } catch (...) { /* 诊断件失败静默 */ }
    }

    /// 居中裁剪到算法几何（与标定表几何一致——K/P/Q/激光表全 1688×1400）。
    /// 尺寸已符/裁剪关闭＝直通（浅拷）；不符＝居中 ROI 深拷（子矩阵连续性保算子安全）
    void cropToAlgo(const cv::Mat& src, cv::Mat& dst, bool left) {
        if (algoW_ <= 0 || algoH_ <= 0
            || (src.cols == algoW_ && src.rows == algoH_)) {
            dst = src;
            return;
        }
        if (src.cols < algoW_ || src.rows < algoH_) {
            dst = cv::Mat();       // 采集小于算法几何＝配置错——交上层零帧告警
            return;
        }
        int x0 = (src.cols - algoW_) / 2, y0 = (src.rows - algoH_) / 2;
        const int fx = left ? cropXL_ : cropXR_;
        const int fy = left ? cropYL_ : cropYR_;
        if (fx >= 0) x0 = fx;
        if (fy >= 0) y0 = fy;
        dst = src(cv::Rect(x0, y0, algoW_, algoH_)).clone();
    }

    void wire() {
        src.cycles.fs = &src;
        if (const char* e = getenv("JMW_FC_DUMPFRAMES")) dumpMax_ = atoi(e);
        // 261031 config 转储开关（config/fc_dump.json {"count":N}——env 易丢；
        // 0/缺文件=关。N 帧 L/R 原图落 diagnostics/fc_frames/，硬件一致性
        // 验证/离线分析素材口）
        if (dumpMax_ == 0) {
            namespace fsd2 = std::filesystem;
            std::error_code dec;
            const fsd2::path p = fsd2::path("config") / "fc_dump.json";
            if (fsd2::exists(p, dec)) {
                try {
                    std::ifstream f(p);
                    nlohmann::json j;
                    f >> j;
                    dumpMax_ = j.value("count", 0);
                    if (dumpMax_ > 0)
                        JMW_LOG_WARN("07-FiveChain",
                            "[FiveChainAdapter] 帧转储开启: {} 帧 → diagnostics/fc_frames",
                            dumpMax_);
                } catch (...) { /* 开关件失败静默 */ }
            }
        }
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
                maybeDump();   // 诊断：裁剪前原图转储（env JMW_FC_DUMPFRAMES）
                // 元数据＋帧宿主落槽（release 序——cycleAt/chain 消费方 acquire）
                Slot& s = ring[static_cast<size_t>(idx) % kRing];
                s.keep = cur;
                s.tubeOdd = deriveTubeOdd(*cur);
                s.snap = cur->snapshot;
                s.temperature = cur->temperature;
                s.seq.store(idx, std::memory_order_release);
                servedFrames.fetch_add(1, std::memory_order_relaxed);
                // 居中裁剪到算法几何（标定表 1688×1400——采集 2048×1536 差异收口）
                cropToAlgo(cur->grayL, cropL_, /*left=*/true);
                cropToAlgo(cur->grayR, cropR_, /*left=*/false);
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
// loadTestData — 测试数据回放模式（<帧目录>——raw 优先/PNG 回退）
// 失败返回 fail（startRealtime 中止——拒绝静默回退在线数据）
// ---------------------------------------------------------------------------
Scanner::Result FiveChainAdapter::loadTestData(const std::string& dir) {
    namespace fsd = std::filesystem;
    std::vector<cv::Mat> framesL, framesR;
    std::vector<int> ids;
    const fsd::path base(dir);

    // ── 261029 优先读 raw 二进制（与离线回放完全一致——消除 PNG 解码差异）──
    const fsd::path metaPath = base / "meta.json";
    std::error_code ecRaw;
    if (fsd::exists(metaPath, ecRaw)) {
        std::ifstream mf(metaPath);
        if (mf) {
            nlohmann::json meta; mf >> meta;
            const int rw = meta.at("width").get<int>();
            const int rh = meta.at("height").get<int>();
            for (int id : meta.at("frames").get<std::vector<int>>()) ids.push_back(id);
            std::sort(ids.begin(), ids.end());
            if (!ids.empty() && rw > 0 && rh > 0) {
                const size_t fb = static_cast<size_t>(rw) * rh;
                const size_t total = fb * ids.size();
                // 限制 50 帧（内存）
                // 261031 帧上限改 env 可调（JMW_FC_TESTN，默认 399 全量——原硬编码 50）
    {
        int maxN = 399;
        if (const char* e = getenv("JMW_FC_TESTN")) maxN = atoi(e);
        if (maxN > 0 && ids.size() > static_cast<size_t>(maxN))
            ids.resize(static_cast<size_t>(maxN));
    }
                std::vector<unsigned char> bufL, bufR;
                {
                    std::ifstream fl(base / "left.bin", std::ios::binary);
                    std::ifstream fr(base / "right.bin", std::ios::binary);
                    if (fl && fr) {
                        bufL.resize(fb * ids.size());
                        bufR.resize(fb * ids.size());
                        fl.read(reinterpret_cast<char*>(bufL.data()), bufL.size());
                        fr.read(reinterpret_cast<char*>(bufR.data()), bufR.size());
                    }
                }
                if (!bufL.empty()) {
                    for (size_t i = 0; i < ids.size(); ++i) {
                        framesL.push_back(cv::Mat(rh, rw, CV_8UC1,
                            bufL.data() + i * fb).clone());
                        framesR.push_back(cv::Mat(rh, rw, CV_8UC1,
                            bufR.data() + i * fb).clone());
                    }
                    JMW_LOG_WARN("07-FiveChain",
                        "[FiveChainAdapter] 测试数据 raw 二进制模式: {} 帧 × {}×{}（离线一致口径）",
                        ids.size(), rw, rh);
                }
            }
        }
    }

    // PNG 回退（raw 不可用时）
    if (framesL.empty()) {
        const bool sameDirFmt = fsd::exists(base / "L_000000.png");
        for (int i = 0;; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%06d", i);
            fsd::path pl, pr;
            if (sameDirFmt) {
                pl = base / ("L_" + std::string(buf) + ".png");
                pr = base / ("R_" + std::string(buf) + ".png");
            } else {
                pl = base / "left" / (std::string(buf) + ".png");
                pr = base / "right" / (std::string(buf) + ".png");
            }
            std::error_code ec;
            if (!fsd::exists(pl, ec) || !fsd::exists(pr, ec)) break;
            cv::Mat l = cv::imread(pl.string(), cv::IMREAD_GRAYSCALE);
            cv::Mat r = cv::imread(pr.string(), cv::IMREAD_GRAYSCALE);
            if (l.empty() || r.empty()) break;
            framesL.push_back(std::move(l));
            framesR.push_back(std::move(r));
            ids.push_back(i);
            if (static_cast<int>(framesL.size()) >= 50) break;
        }
    }
    const int n = static_cast<int>(framesL.size());
    if (n == 0) {
        JMW_LOG_ERROR("07-FiveChain", "[FiveChainAdapter] 测试数据加载失败: {}", dir);
        return Scanner::Result::fail("测试数据加载失败（0 帧）: " + dir);
    }
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 测试数据回放: {} 帧 × {}×{}（{}）",
        n, framesL[0].cols, framesL[0].rows, dir);

    // （名义标定装载已前移至 startRealtime ①b ensureStereoNominalFromConfig——
    // 须先于 harness calib_/undistG 装配生效，此处不再重复加载）

    // 覆写桥的 readFrame——从内存数组直读，尺寸已符＝裁剪直通
    // 261029 像素校验（首帧 sum+首 4 像素值——与离线对拍定位差异）
    if (n > 0) {
        cv::Scalar sL = cv::sum(framesL[0]);
        cv::Scalar sR = cv::sum(framesR[0]);
        const uchar* pL = framesL[0].ptr<uchar>();
        const uchar* pR = framesR[0].ptr<uchar>();
        JMW_LOG_WARN("07-FiveChain",
            "[FC-PIXEL] L: sum={:.0f} [0]={} [1]={} [2]={} [3]={} {}×{} continuous={}",
            sL[0], (int)pL[0], (int)pL[1], (int)pL[2], (int)pL[3],
            framesL[0].cols, framesL[0].rows, framesL[0].isContinuous());
        JMW_LOG_WARN("07-FiveChain",
            "[FC-PIXEL] R: sum={:.0f} [0]={} [1]={} [2]={} [3]={}",
            sR[0], (int)pR[0], (int)pR[1], (int)pR[2], (int)pR[3]);
    }
    bridge_->algoW_ = framesL[0].cols;
    bridge_->algoH_ = framesL[0].rows;
    // 261029 单遍播放（游标 0..n-1 顺序；播完＝帧源尽——执行器自然收尾）。
    // 不循环：循环回放同一组帧多遍累积，配准链 R/T 每遍演化＋末帧→首帧跳变
    // 破坏光流，同一物理标志点带偏移重复入融合云→体素边界分裂成两两靠近的
    // 副本（真机会话帧不重复，无此形态）。重跑＝重新点模式键 arm（全新会话）
    auto cur = std::make_shared<int>(0);
    bridge_->src.readFrame = [this, framesL, framesR, n, cur](int idx, bool left,
                                                              cv::Mat& out,
                                                              std::string& e) -> bool {
        if (left) {
            // 261029 帧门（app 注入 isCapturing——与在线帧流同语义：模式键只
            // arm，M 键启停控制回放；门未开＝读帧阻塞等待（在线桥等环帧同
            // 款），停机令即退。门关再开＝本遍续播（同在线 M 暂停/续采语义）
            if (cfg_.testFrameGate) {
                while (!cfg_.testFrameGate()) {
                    if (bridge_->stopped.load(std::memory_order_acquire)) {
                        e = "test replay stopped (gate closed)";
                        return false;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
            if (*cur >= n) {
                JMW_LOG_WARN("07-FiveChain",
                    "[FiveChainAdapter] 测试回放单遍完成（{} 帧）——帧源已尽；"
                    "重跑＝重新点模式键 arm（全新会话）", n);
                return false;   // 帧源尽：执行器排水收尾（离线回放同款终态）
            }
            ++(*cur);
        }
        const int fi = *cur - 1;
        if (fi < 0) return false;                    // 防御：右先于左（不发生）
        out = (left ? framesL : framesR)[static_cast<size_t>(fi)];
        return true;
    };
    // 元数据：frameId=序号，tvKnown 交替（模拟 T/V 管）
    bridge_->src.cycleAt = [this, n](size_t i) {
        scr::FrameCycleInfo ci;
        ci.laserIdx = static_cast<int>(i % static_cast<size_t>(n));
        ci.tubeOdd = (i % 2) == 1;
        return ci;
    };
    // 快照注入关闭（测试数据用名义标定）
    bridge_->useSnapshotCalib_ = false;
    // 测试模式强制融合（跳过配准门控——验证激光链用）
    if (harness_) {
        harness_->cfg_.runtime.fuseOnHold = true;
        // 直打 stderr 确认生效
        fprintf(stderr, "[FC-TEST] fuseOnHold=%d (forced)\n",
                static_cast<int>(harness_->cfg_.runtime.fuseOnHold));
    }
    // onFrameMeta：填 frameId，metaValid=false
    if (harness_) {
        harness_->onFrameMeta = [this, n](int idx, scr::CycleProductsCompat& p) {
            p.metaFrameId = static_cast<uint64_t>(idx % n);
            p.metaValid = false;
        };
    }
    return Scanner::Result::ok("测试数据回放 " + std::to_string(n) + " 帧");
}

// ---------------------------------------------------------------------------
// startRealtime — 桥＋harness 装配（激光链算子/温表/配准算子）＋执行器＋收割线程
// ---------------------------------------------------------------------------
void FiveChainAdapter::attachStereoNominal(const cv::Mat& R1, const cv::Mat& R2,
                                           const cv::Mat& P1, const cv::Mat& P2,
                                           const cv::Mat& Q, const cv::Mat& K1c,
                                           const cv::Mat& K2c, const cv::Mat& D1,
                                           const cv::Mat& D2, const cv::Mat& Rext,
                                           const cv::Mat& Tcomp) {
    snR1_ = R1.clone(); snR2_ = R2.clone(); snP1_ = P1.clone(); snP2_ = P2.clone();
    snQ_ = Q.clone(); snK1c_ = K1c.clone(); snK2c_ = K2c.clone();
    snD1_ = D1.clone(); snD2_ = D2.clone();
    snRext_ = Rext.clone(); snTcomp_ = Tcomp.clone();
    hasStereoNominal_ = true;
}

// ---------------------------------------------------------------------------
// ensureStereoNominalFromConfig — config/camera_calib.json 就近 22.5℃ 温档
// → attachStereoNominal。测试数据模式前置件（快照注入关闭后名义标定为唯一
// 标定源）；任何失败返回 false（调用方 fail-fast）
// ---------------------------------------------------------------------------
bool FiveChainAdapter::ensureStereoNominalFromConfig() {
    namespace fsd = std::filesystem;
    const fsd::path calibPath = fsd::path("config") / "camera_calib.json";
    std::error_code cec;
    if (!fsd::exists(calibPath, cec)) {
        JMW_LOG_ERROR("07-FiveChain",
            "[FiveChainAdapter] 名义标定缺失: {}", calibPath.string());
        return false;
    }
    try {
        std::ifstream cf(calibPath.string());
        nlohmann::json j;
        cf >> j;
        const auto& table = j.at("stereoRectifyTempTable").at("table");
        const nlohmann::json* tier = nullptr;
        double bestT = 0;
        for (const auto& t : table) {
            if (!t.contains("temperature")) continue;
            const double tv = t.at("temperature").get<double>();
            if (tier == nullptr
                || std::abs(tv - 22.5) < std::abs(bestT - 22.5) - 1e-12) {
                tier = &t; bestT = tv;
            }
        }
        if (tier == nullptr) {
            JMW_LOG_ERROR("07-FiveChain",
                "[FiveChainAdapter] 名义标定无温档（stereoRectifyTempTable.table 空）");
            return false;
        }
        auto tierM = [tier](const char* field, int r, int c) {
            cv::Mat m(r, c, CV_64F);
            const auto& arr = tier->at(field);
            for (int i = 0; i < r; ++i)
                for (int k = 0; k < c; ++k)
                    m.at<double>(i, k) = arr.at(i).at(k).get<double>();
            return m;
        };
        auto mj3 = [&j](const char* section, const char* side,
                        const char* field, int r, int c) {
            cv::Mat m(r, c, CV_64F);
            const auto& arr = j.at(section).at(side).at(field);
            for (int i = 0; i < r; ++i)
                for (int k = 0; k < c; ++k)
                    m.at<double>(i, k) = arr.at(i).at(k).get<double>();
            return m;
        };
        auto mj2 = [&j](const char* section, const char* field, int r, int c) {
            cv::Mat m(r, c, CV_64F);
            const auto& arr = j.at(section).at(field);
            for (int i = 0; i < r; ++i)
                for (int k = 0; k < c; ++k)
                    m.at<double>(i, k) = arr.at(i).at(k).get<double>();
            return m;
        };
        attachStereoNominal(
            tierM("R1", 3, 3), tierM("R2", 3, 3),
            tierM("P1", 3, 4), tierM("P2", 3, 4), tierM("Q", 4, 4),
            tierM("compensatedCameraMatrixL", 3, 3),
            tierM("compensatedCameraMatrixR", 3, 3),
            mj3("intrinsic", "left", "dist_coeffs", 1, 5),
            mj3("intrinsic", "right", "dist_coeffs", 1, 5),
            mj2("extrinsic", "R", 3, 3), mj2("extrinsic", "T", 3, 1));
        JMW_LOG_WARN("07-FiveChain",
            "[FiveChainAdapter] 名义标定直接加载: tier={:.1f}℃ "
            "K1c.fx={:.1f}（确定性保证）",
            bestT, tierM("compensatedCameraMatrixL", 3, 3).at<double>(0, 0));
        return true;
    } catch (const std::exception& e) {
        JMW_LOG_ERROR("07-FiveChain",
            "[FiveChainAdapter] 名义标定加载异常: {}", e.what());
        return false;
    }
}

Scanner::Result FiveChainAdapter::startRealtime(
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring, size_t dropThreshold) {
    if (rtActive_.load()) return Scanner::Result::fail("五链实时链已在运行");
    if (rtK1_.empty() || rtK2_.empty())
        return Scanner::Result::fail("五链实时链: 标定未注入（先 attachCalib）");
    if (!rtLaserTable_)
        return Scanner::Result::fail("五链实时链: 激光温表为空（v1 需激光链）");
    if (rtActive_.exchange(true)) return Scanner::Result::fail("五链实时链并发启动");

    try {
        // ① 桥（算法几何：默认标定表几何 1688×1400；env JMW_FC_W/H 覆盖，0=直通；
        //    原点默认居中推定，JMW_FC_CROP_X/Y 定点——离线求解器 JMW_FC_SOLVE 定标）
        bridge_ = std::make_unique<RtBridge>(ring, dropThreshold);
        {
            int aw = cfg_.algoWidth, ah = cfg_.algoHeight;
            if (const char* e = getenv("JMW_FC_W")) aw = atoi(e);
            if (const char* e = getenv("JMW_FC_H")) ah = atoi(e);
            bridge_->algoW_ = aw;
            bridge_->algoH_ = ah;
            if (const char* e = getenv("JMW_FC_CROP_X")) { bridge_->cropXL_ = atoi(e); bridge_->cropXR_ = atoi(e); }
            if (const char* e = getenv("JMW_FC_CROP_Y")) { bridge_->cropYL_ = atoi(e); bridge_->cropYR_ = atoi(e); }
            if (const char* e = getenv("JMW_FC_CROPL_X")) bridge_->cropXL_ = atoi(e);
            if (const char* e = getenv("JMW_FC_CROPL_Y")) bridge_->cropYL_ = atoi(e);
            if (const char* e = getenv("JMW_FC_CROPR_X")) bridge_->cropXR_ = atoi(e);
            if (const char* e = getenv("JMW_FC_CROPR_Y")) bridge_->cropYR_ = atoi(e);
        }
        bridge_->wire();

        // ①b 测试数据回放解析（261029：config/fc_testdata.json 为正源，env
        // JMW_FC_TESTDATA 优先覆盖）——先于 ②③：测试模式需名义标定先行注入
        // （harness calib_/undistG 构造依赖 hasStereoNominal_）；帧源覆写在 ⑤b
        bool testMode = false;
        std::string tdDir, tdSrc;
        if (const char* e = getenv("JMW_FC_TESTDATA")) {
            tdDir = e;
            tdSrc = "env JMW_FC_TESTDATA";
        } else {
            namespace fsd = std::filesystem;
            std::error_code tec;
            const fsd::path tdCfgPath = fsd::path("config") / "fc_testdata.json";
            if (fsd::exists(tdCfgPath, tec)) {
                try {
                    std::ifstream tf(tdCfgPath.string());
                    nlohmann::json tj;
                    tf >> tj;
                    if (tj.value("enabled", false)) {
                        tdDir = tj.value("dir", std::string());
                        tdSrc = tdCfgPath.string();
                    }
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        std::string("fc_testdata.json 解析失败: ") + e.what());
                }
            }
        }
        if (!tdDir.empty()) {
            testMode = true;
            if (!hasStereoNominal_ && !ensureStereoNominalFromConfig())
                throw std::runtime_error(
                    "测试数据模式需名义温档标定（config/camera_calib.json）"
                    "——加载失败，拒绝静默回退在线数据");
            JMW_LOG_WARN("07-FiveChain",
                "[FiveChainAdapter] 测试数据回放模式（源={}）: {}", tdSrc, tdDir);
        } else if (!tdSrc.empty()) {
            throw std::runtime_error(
                "fc_testdata.json enabled=true 但 dir 为空——拒绝静默回退在线数据");
        }

        // ② harness（算子宿主）——产线口径装配
        harness_ = new FcHarnessCompat(*this, bridge_->src);
        harness_->cfg_.fivechain = cfg_;
        harness_->cfg.fivechain = cfg_;                    // 双名兼容
        if (opJson_ != nullptr)
            harness_->cfg_.operators = *opJson_;           // 算子参数源全链应用（v3/flow/fuse/标志点链）
        if (getenv("JMW_FC_FUSE_ON_HOLD") != nullptr)
            harness_->cfg_.runtime.fuseOnHold = true;   // 诊断：hold 帧强制融合（无标志点验激光链）
        harness_->cfg_.runtime.nominalTemperature =
            rtLaserTable_ ? rtLaserTable_->referenceTemperature : 25.0;   // 表参考温（离线 golden 同口径 22.5）
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
        if (hasStereoNominal_) {
            // 261027 名义温档标定（config/camera_calib.json——与产表/离线验证
            // 同源）：harness 构造期标定注入；逐帧快照注入关闭（metaValid=false
            // ——在线 21:28 实证快照路径 conic 0/4666 vs 名义口径 20-107/30 帧）
            harness_->calib_.R1 = snR1_;  harness_->calib_.R2 = snR2_;
            harness_->calib_.P1 = snP1_;  harness_->calib_.P2 = snP2_;
            harness_->calib_.Q = snQ_;
            harness_->calib_.K1c = snK1c_; harness_->calib_.K2c = snK2c_;
            harness_->calib_.D1 = snD1_;  harness_->calib_.D2 = snD2_;
            harness_->calib_.Rext = snRext_; harness_->calib_.Tcomp = snTcomp_;
            bridge_->useSnapshotCalib_ = false;
            JMW_LOG_INFO("07-FiveChain",
                "[FiveChainAdapter] 名义温档立体标定生效（快照注入关闭）");
        }
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
            // metaValid＝快照标定注入开关（名义温档口径下 false——undistG/
            // undistCpu/conic 全用 harness 构造期标定；frameId 照常透传）
            p.metaValid = bridge_->useSnapshotCalib_;
            (void)idx;
        };

        // ③ 激光链前段算子（沙盒 makeLaserOps 产线版：分眼实例＋默认参数＋
        // 强制 Scan 模式；undist 双实例——每帧 SetParams 温补快照）
        // 261028 标点模式：laserEnabled=false 时不创建任何激光算子——
        // submitLaserFront 走"LaserOps 未装配"分支直接返回 0（零 GPU 提交）
        auto& L = harness_->laserOps_;
        if (cfg_.laserEnabled) {
            const auto stg = calib::StegerParams::fromJson(
                harness_->cfg_.opParams("steger_fast"));
            L.stegerL_ = std::make_unique<calib::StegerExtractorFast>(stg);
            L.stegerR_ = std::make_unique<calib::StegerExtractorFast>(stg);
            const auto fp = calib::FlatCenterParams::fromJson(
                harness_->cfg_.opParams("flat_center"));
            L.flatL_ = std::make_unique<calib::FlatCenterExtractor>(fp);
            L.flatR_ = std::make_unique<calib::FlatCenterExtractor>(fp);
        try {
            calib::UndistortPointsParams ul, ur;
            if (hasStereoNominal_) {
                // 名义温档直用式（离线对拍同构——微测变体 a/c 验证过）
                ul.cameraMatrix = snK1c_; ul.distCoeffs = snD1_;
                ul.R = snR1_; ul.P = snP1_.clone();
                ur.cameraMatrix = snK2c_; ur.distCoeffs = snD2_;
                ur.R = snR2_; ur.P = snP2_.clone();
                ur.P.at<double>(0, 3) = 0.0;         // 右目 P 平移置零（沙盒口径）
            } else {
                // 快照口径占位（仅过校验——每帧 SetParams 覆写）
                ul.cameraMatrix = rtK1_; ul.distCoeffs = rtD1_;
                ul.R = cv::Mat::eye(3, 3, CV_64F);
                ul.P = cv::Mat::zeros(3, 4, CV_64F);
                rtK1_.copyTo(ul.P(cv::Rect(0, 0, 3, 3)));
                ur.cameraMatrix = rtK2_; ur.distCoeffs = rtD2_;
                ur.R = cv::Mat::eye(3, 3, CV_64F);
                ur.P = cv::Mat::zeros(3, 4, CV_64F);
                rtK2_.copyTo(ur.P(cv::Rect(0, 0, 3, 3)));
            }
            L.undistGL_ = std::make_unique<calib::UndistortPointsCuda>(ul);
        } catch (const std::exception& e) {
            JMW_LOG_ERROR("07-FiveChain", "[FiveChainAdapter] undistGL 构造失败: 失败: {}", e.what());
            throw;
        }
        try {
            calib::UndistortPointsParams ur;
            if (hasStereoNominal_) {
                ur.cameraMatrix = snK2c_; ur.distCoeffs = snD2_;
                ur.R = snR2_; ur.P = snP2_.clone();
                ur.P.at<double>(0, 3) = 0.0;         // 右目 P 平移置零（沙盒口径）
            } else {
                ur.cameraMatrix = rtK2_; ur.distCoeffs = rtD2_;
                ur.R = cv::Mat::eye(3, 3, CV_64F);
                ur.P = cv::Mat::zeros(3, 4, CV_64F);
                rtK2_.copyTo(ur.P(cv::Rect(0, 0, 3, 3)));
            }
            L.undistGR_ = std::make_unique<calib::UndistortPointsCuda>(ur);
        } catch (const std::exception& e) {
            JMW_LOG_ERROR("07-FiveChain", "[FiveChainAdapter] undistGR 构造失败: 失败: {}", e.what());
            throw;
        }
        calib::EpipolarInterpDualParams ip = calib::EpipolarInterpDualParams::fromJson(
            harness_->cfg_.opParams("epipolar_interp_dual"));
        ip.mode = calib::InterpMode::Scan;                 // 沙盒口径：强制扫描模式
        L.interpL_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);

        L.interpR_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
        } else {
            JMW_LOG_INFO("07-FiveChain",
                "[FiveChainAdapter] 标点模式：激光链算子未创建（零 GPU 提交防崩溃）");
        }

        // ④ 配准算子（生产调参口径——ScanChains L261-264 同源）＋兜底算子
        // （frame_fuse 默认参——断链自愈；operators json 有 frame_fuse 节时用之）
        calib::MarkerOpticalFlowFuseCPUParams ofp;
        ofp.normalAngleThresh = 45.0;
        ofp.useNormalWeight = true;
        harness_->flowOp_ = std::make_unique<calib::MarkerOpticalFlowFuseCPU>(ofp);
        harness_->fuseParams_ = calib::FrameFuseCPUParams::fromJson(
            harness_->cfg_.opParams("frame_fuse"));
        harness_->fuseOp_ = std::make_unique<calib::FrameFuseCPU>(
            harness_->fuseParams_);

        // ⑤b 测试数据回放模式（①b 已解析并确保标定）——覆写帧源（内存直读）
        if (testMode) {
            const Scanner::Result tdr = loadTestData(tdDir);
            if (!tdr.success) throw std::runtime_error(tdr.message);
        }

        // ⑤ 执行器（INT_MAX＝无界实时会话——reader 由桥停信号终止）
        //    前置预热（沙盒靠 serialHead 3 帧串行预热，07 直起执行器须显式）：
        //    a) sep Warmup——首 Execute CUDA Graph 捕获路径内部缓冲预分配
        //    b) 灰度环全槽预分配——捕获窗（Global 禁令）内上传线程懒分配会崩
        {
            const int fw = bridge_->algoW_ > 0 ? bridge_->algoW_ : rtW_;
            const int fh = bridge_->algoH_ > 0 ? bridge_->algoH_ : rtH_;
            sepL_->Warmup(fh, fw);
            sepR_->Warmup(fh, fw);
            harness_->preallocGrayLanes(cfg_.graySlots, fh, fw);
        }
        executor_ = std::make_unique<scr::fc::FiveChainExecutor>(*harness_, 0, 0x7fffffff);

        // 261028 修复：fuseOnHold 在执行器构造后强制设置（loadTestData 在执行器
        // 前设置但 fuseLoop 读到 false——引用/时序问题；此处直接改 harness 成员，
        // fuseLoop 每次迭代重新读取，保证生效）
        if (testMode) {
            harness_->cfg_.runtime.fuseOnHold = true;
            JMW_LOG_WARN("07-FiveChain",
                "[FiveChainAdapter] fuseOnHold 在执行器后强制=true（诊断确认修复）");
        }

        // ⑥ 收割线程
        // 261028 标志点融合算子（体素 5mm＋EMA——旧 lane MarkerFuseAdapter 同款）
        {
            calib::MarkerCloudFuseCPUParams mfp;
            mfp.voxelSize = 30.0f;  // 261031 15→30mm——399 帧长序列 R/T 漂移 ~20mm+
                              // 致同点跨 15mm 体素分裂（用户「同一物点很多像点」
                              // 实证）；30mm 容忍全程漂移，GBA 终版才出精细位
            mfp.saturationThreshold = 99;
            markerFuseOp_ = std::make_unique<calib::MarkerCloudFuseCPU>(mfp);
        }
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
        harness_->diagLaserPts_.fetch_add(prod.laserPoints3D,
                                          std::memory_order_relaxed);
        // 261028 标志点融合（体素 5mm＋EMA 精化——替代简单去重）
        if (prod.conic.success && prod.markersValid > 0) {
            std::vector<calib::MarkerFuseInput> mfi;
            mfi.reserve(prod.conic.markers.size());
            for (const auto& m : prod.conic.markers) {
                if (!m.validCircle) continue;
                calib::MarkerFuseInput in;
                in.x = static_cast<float>(m.centerX);
                in.y = static_cast<float>(m.centerY);
                in.z = static_cast<float>(m.centerZ);
                in.nx = static_cast<float>(m.normalX);
                in.ny = static_cast<float>(m.normalY);
                in.nz = static_cast<float>(m.normalZ);
                mfi.push_back(in);
            }
            if (!mfi.empty() && markerFuseOp_) {
                const cv::Matx33d Rm(prod.R);
                const cv::Vec3d Tv(prod.T[0], prod.T[1], prod.T[2]);
                markerFuseOp_->Execute(mfi, Rm, Tv);
            }
        }
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
        } else if (prod.conic.success && prod.markersValid > 0) {
            // 261028 配准失败时 conic 结果也落账（GBA 全局匹配不依赖逐帧
            // 光流——conic 有效圆 3D 位置就是观测，globalId=-1 待 GBA 关联）
            fo.markerObs.reserve(prod.conic.markers.size());
            for (const auto& m : prod.conic.markers) {
                if (!m.validCircle) continue;
                MarkerObs mo;
                mo.xyz[0] = m.centerX;
                mo.xyz[1] = m.centerY;
                mo.xyz[2] = m.centerZ;
                mo.globalId = -1;   // GBA 端全局关联
                fo.markerObs.push_back(mo);
            }
        }
        // 261029 D 线激光重放原料入缓存：fuseLoop 已自 E 块拷出本帧设备系激光
        // 3D（laserXyzHost）——obs 累加器按预算缓存（laserCacheSlot 槽号回填；
        // GlobalOptimObject anyLaser 判据即非 kNoLaserSlot，GBA 后按优化位姿重融合）
        obs_.push(std::move(fo), std::move(prod.laserXyzHost));

        // ② 轻计数上报（UI 统计亚秒刷新）
        if (sceneFeed_ && (idx % 8) == 0)
            sceneFeed_->notifyLaserCount(fuse_->GetFusedPointCount());

        // 261028 逐帧 conic 诊断（前 200 帧——定位 conic 率随时间衰减）
        if (idx < 200 && (idx % 10) == 0) {
            JMW_LOG_INFO("07-FiveChain",
                "[FC-CONIC] f={} conic={} valid={} det={} match={} regSrc={}",
                idx, prod.conic.statistics.validCount, prod.markersValid,
                prod.markersDetected, prod.mm.centerMatches.size(),
                prod.regSource);
            // 帧 0 详细诊断：逐配对输出 fail reason
            if (idx == 0) {
                for (size_t d = 0; d < prod.conic.diagnostics.size() && d < 8; ++d) {
                    const auto& dg = prod.conic.diagnostics[d];
                    JMW_LOG_WARN("07-FiveChain",
                        "[FC-CONIC-D] L{}-R{} reason={} radius={:.2f}mm resid={:.3f}px",
                        dg.leftIdx, dg.rightIdx,
                        calib::ConicFailReasonName(dg.failReason),
                        dg.reconstructedRadiusMm, dg.reprojResidualPx);
                }
            }
        }
        // 261031 双链深度探针（每 100 帧一次——标定一致性验收口）：同帧
        // conic 标志点 Z 均值 vs 激光 Z 中位——两链共用同一表面，系统偏差
        // ＝camera_calib（conic）与 CMTT（激光）标定口径错位（实测 ~3mm；
        // 重标定后此值应 <1mm）
        if ((idx % 100) == 0 && prod.conic.success && prod.markersValid > 0
            && !prod.laserXyzHost.empty()) {
            double mkZ = 0;
            int mkN = 0;
            for (const auto& m : prod.conic.markers) {
                if (!m.validCircle) continue;
                mkZ += m.centerZ;
                ++mkN;
            }
            if (mkN > 0) {
                std::vector<float> lz(prod.laserXyzHost.size() / 3);
                for (size_t k = 0; k < lz.size(); ++k)
                    lz[k] = prod.laserXyzHost[k * 3 + 2];
                std::sort(lz.begin(), lz.end());
                if (!lz.empty())
                    JMW_LOG_WARN("07-FiveChain",
                        "[DUAL-Z] f={} 标志点Z={:.1f}({}个) 激光Z={:.1f} "
                        "偏差={:.1f}mm（重标定后应<1mm）",
                        idx, mkZ / mkN, mkN, lz[lz.size() / 2],
                        mkZ / mkN - lz[lz.size() / 2]);
            }
        }

        // ③ 节流渲染推送（时间门 ≥3s——FuseConsumer 261018 口径）
        renderPushIfDue(/*forceFull=*/false);
        ++idx;
    }
    JMW_LOG_INFO("07-FiveChain", "[FiveChainAdapter] 收割线程退出: {} 帧处理", idx);
    // 261029 终局全量推送：帧源尽（回放单遍完）/停机时 F 通道已排空——节流
    // 推送可能停在半途（短会话 <500ms 节流窗只推过首段），不补全量则扫描中
    // 显示缺末段（空隙）直到关会话 GBA 后才由全量快照补上
    renderPushIfDue(/*forceFull=*/true);
}

// ---------------------------------------------------------------------------
// 渲染推送（collector 调用；终局 stopRealtime forceFull=true）
// ---------------------------------------------------------------------------
void FiveChainAdapter::renderPushIfDue(bool forceFull) {
    const auto now = std::chrono::steady_clock::now();
    constexpr int64_t kMinMs = 500;   // 261028 测试模式降节流（399 帧 ~3 秒需 ≥6 次推送）
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
    // 261029 游标对齐实际下载区间：原记 curPts（读 ctx 前的旧值）——fuseLoop
    // 并发追加时 fusedPointCount>curPts，多下载的段 [curPts, fusedPointCount)
    // 会被下一次增量重复推送→仓库重复累计（"GBA 后点变多"假象：五链未填
    // laserCacheSlot，GBA 激光重放本就跳过，仓库应恒等于 C 线融合云）
    if (!fusedXyz.empty()) {
        lastPushedLaserPts_ = (fullPush ? size_t(0) : lastPushedLaserPts_)
                              + fusedXyz.size() / 3;
    } else if (fullPush) {
        lastPushedLaserPts_ = 0;   // 空云全量：归零（下载失败下次全量重推）
    }
    ++pushSeq_;
    // 261028 激光显示诊断（stderr 直打——定位融合云是否为空）
    fprintf(stderr, "[FC-RENDER] fused=%zu pts laserUI=%zu markers=%zu full=%d\n",
            curPts, fusedXyz.size() / 3,
            markerFuseOp_ ? markerFuseOp_->GetFusedPoints().size() : 0,
            static_cast<int>(fullPush));
    // 261028 修复：标志点不依赖激光——即使激光云为空（标点模式）也推送
    const bool hasMarkers = markerFuseOp_ && !markerFuseOp_->GetFusedPoints().empty();
    if (fusedXyz.empty() && !hasMarkers && !forceFull) return;

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
    h.hostMarker = (markerFuseOp_ && !markerFuseOp_->GetFusedPoints().empty())
                       ? &markerFuseOp_->GetFusedPoints() : nullptr;
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
    // 逐级诊断汇总（真机排障：哪一级断链一眼可见）
    if (harness_) {
        JMW_LOG_WARN("07-FiveChain",
            "[FiveChainAdapter] 链路诊断: CCL失败帧={} CCL组件L累计={} split成功帧={} "
            "ellipse成功帧={} 匹配对累计={} conic有效圆累计={} | 配准 first/flow/fuse/hold/skip="
            "{}/{}/{}/{}/{} | 激光重建点累计={} 融合云={} 点 | 质量门弃帧={} "
            "飞线剪枝点={}(gate={}mm)",
            harness_->diagCclFail_.load(), harness_->diagRoisL_.load(),
            harness_->diagSplit_.load(), harness_->diagEllipse_.load(),
            harness_->diagMatchPairs_.load(), harness_->diagConicValid_.load(),
            harness_->diagRegFirst_.load(), harness_->diagRegFlow_.load(),
            harness_->diagRegFuse_.load(), harness_->diagRegHold_.load(),
            harness_->diagRegSkip_.load(), harness_->diagLaserPts_.load(),
            fuse_->GetFusedPointCount(),
            executor_ ? executor_->zGateDrops() : 0,
            executor_ ? executor_->zLinePruned() : 0,
            executor_ ? executor_->linePlaneGateMm() : 0);
    }
    JMW_LOG_INFO("07-FiveChain",
        "[FiveChainAdapter] 实时链停止: 服务 {} 帧（过载跳帧 {}/裁剪 {}）/融合云 {} 点",
        bridge_ ? bridge_->servedFrames.load() : 0,
        bridge_ ? bridge_->skippedTotal.load() : 0,
        bridge_ ? bridge_->cropFrames_.load() : 0,
        fuse_ ? fuse_->GetFusedPointCount() : 0);
    // 261030 崩溃修复（停止竞态——0xC0000005 读 0x748＝collect 在析构中对象上
    // 成员读）：原序「executor_.reset() 先于 collector_.join()」——收割线程可能
    // 正阻塞在 executor_->collect()（chF_ 等待），析构把通道/成员抽走＝UAF。
    // 正序：①桥停（readFrame 速败）②执行器软停（stop 置位＋通道全关——阻塞的
    // collect 解阻塞返回 false）③join 收割线程（含其终局全量推送）④再析构
    if (bridge_) bridge_->stopped.store(true);    // readFrame 速败→墓碑排空
    if (executor_) executor_->requestStop();      // 软停：解阻塞 collect/全部等待者
    if (collector_.joinable()) collector_.join(); // 收割线程先退干净（无人再用执行器）
    executor_.reset();                            // 此后析构安全（stopAll 幂等）
    if (markerFuseOp_) markerFuseOp_->Clear();
    renderPushIfDue(true);                        // 终局全量（仓库+UI 收口）
    delete harness_; harness_ = nullptr;
    bridge_.reset();
}

} // namespace Scanner::pipeline
