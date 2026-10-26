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
                                   ISceneFeed* sceneFeed,
                                   const nlohmann::json* opJson)
    : cfg_(cfg), obs_(obs), warehouse_(warehouse), sceneFeed_(sceneFeed) {
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
        // CCL 门限＝harness 成员（沙盒 config ccl 节默认 800/30000/0——对拍同参）
        p.cclL = scr::runCclCpuFromRuns(runsL, nL, cols, rows,
                                        cclMinArea_, cclMaxArea_, cclTopX_);
        p.cclR = scr::runCclCpuFromRuns(runsR, nR, cols, rows,
                                        cclMinArea_, cclMaxArea_, cclTopX_);
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
        if (flowOk) {
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
    } catch (const std::exception& e) {
        fail("registration", p, e.what());
        p.regSource = "hold";
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

        // ③ 激光链前段算子（沙盒 makeLaserOps 产线版：分眼实例＋默认参数＋
        // 强制 Scan 模式；undist 双实例——每帧 SetParams 温补快照）
        auto& L = harness_->laserOps_;
        {
            const auto stg = calib::StegerParams::fromJson(
                harness_->cfg_.opParams("steger_fast"));
            L.stegerL_ = std::make_unique<calib::StegerExtractorFast>(stg);
            L.stegerR_ = std::make_unique<calib::StegerExtractorFast>(stg);
            const auto fp = calib::FlatCenterParams::fromJson(
                harness_->cfg_.opParams("flat_center"));
            L.flatL_ = std::make_unique<calib::FlatCenterExtractor>(fp);
            L.flatR_ = std::make_unique<calib::FlatCenterExtractor>(fp);
        }
        L.undistGL_ = std::make_unique<calib::UndistortPointsCuda>();   // 参数每帧 SetParams
        L.undistGR_ = std::make_unique<calib::UndistortPointsCuda>();
        calib::EpipolarInterpDualParams ip = calib::EpipolarInterpDualParams::fromJson(
            harness_->cfg_.opParams("epipolar_interp_dual"));
        ip.mode = calib::InterpMode::Scan;                 // 沙盒口径：强制扫描模式
        L.interpL_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);
        L.interpR_ = std::make_unique<calib::EpipolarInterpDualCuda>(ip);

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
