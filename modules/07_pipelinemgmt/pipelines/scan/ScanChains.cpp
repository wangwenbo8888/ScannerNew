// ============================================================================
// ScanChains.cpp — C 扫描双链算子装配实现
//
// 实现要点：
//   - 算子集 ScanLaneOps 按 lane 惰性创建一次、跨帧复用（09 算子"每实例非线程
//     安全"，多 lane 并发下不可共享单实例——对 ScanWorkflow 单线程先例的必要调整）
//   - GPU 链算子仅被该 lane 的 E 核线程触碰；P 链算子仅被执行该帧 pChain 的
//     P worker 触碰；二者均存放于每 lane 一份的 front.ops（gpuChain 首帧先于
//     frontReady() 创建，pChain 只读）
//   - 每算子调用失败：spdlog 记录 + 钩子返回 false/fail；激光段失败已 frontReady
//     → 返回 false 交由 runtime 做 T8 孤儿等待
// ============================================================================
#include "pipelines/scan/ScanChains.h"

#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

// ---- 09 GPU 链算子（公开头 pImpl 隔离，不含 CUDA 类型）----
#include "core/vision/ccl/region_analyze_cuda.h"
#include "core/laser/epipolar_interp/epipolar_interp_cuda.h"
#include "core/laser/laser_reconstruct/laser_reconstruct_cuda.h"
#include "core/laser/steger/steger_extract_cuda.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "scanning/laser/laser_match_scan/laser_match_scan_cuda.h"
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
// ---- 09 P 核链算子 ----
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

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace Scanner::pipeline {

// ============================================================================
// ScanLaneOps — per-lane 算子集（同 lane 内串行复用，跨 lane 各自独占）
// ============================================================================
struct ScanLaneOps {
#ifdef JMW_BUILD_CUDA
    // GPU 链（E 核线程独占）
    std::unique_ptr<calib::LaserMarkingSeparationCUDA> sep;
    std::unique_ptr<calib::RegionAnalyzerCUDA> ccl;
    std::unique_ptr<calib::StegerExtractorCUDA> steger;
    std::unique_ptr<calib::UndistortPointsCuda> undistG;
    std::unique_ptr<calib::EpipolarInterpCuda> epipolar;
    std::unique_ptr<calib::LaserMatchScanCuda> matchScan;
    std::unique_ptr<calib::LaserReconstructCuda> recon;
    cv::cuda::GpuMat d_grayL, d_grayR;          // 灰度 device 副本（steger 输入）
#endif
    // P 核链（P worker 独占）
    std::unique_ptr<calib::ImageSplitCPU> split;
    std::unique_ptr<calib::ZernikeEdgeCPU> zernike;
    std::unique_ptr<calib::ImageMergeCPU> merge;
    std::unique_ptr<calib::MarkerUndistortCPU> undistCpu;
    std::unique_ptr<calib::EllipseFitCPU> ellipse;
    std::unique_ptr<calib::MarkerMatchCPU> match;
    std::unique_ptr<calib::EpipolarIntersectCPU> epiIntersect;
    std::unique_ptr<calib::EdgeMatchCPU> edgeMatch;
    std::unique_ptr<calib::PointReconstructCPU> reconstruct;
    std::unique_ptr<calib::MarkerOpticalFlowFuseCPU> flowFuse;
    std::unique_ptr<calib::FrameFuseCPU> frameFuse;
};

namespace {

Scanner::QualityFlag toScannerQuality(calib::QualityFlag q) {
    switch (q) {
        case calib::QualityFlag::Normal:   return Scanner::QualityFlag::Normal;
        case calib::QualityFlag::Degraded: return Scanner::QualityFlag::Degraded;
        default:                           return Scanner::QualityFlag::Warning;
    }
}

void fillRT(FrameResult& r, const cv::Matx33d& R, const cv::Vec3d& T) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.R[i * 3 + j] = R(i, j);
    r.T[0] = T(0); r.T[1] = T(1); r.T[2] = T(2);
}

cv::Matx33d matxFromArr9(const double* a) {
    return cv::Matx33d(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}
cv::Vec3d vec3FromArr3(const double* a) { return cv::Vec3d(a[0], a[1], a[2]); }

// K/D Mat → 标量参数（越界补 0；D 顺序 [k1,k2,p1,p2,k3,(k4,k5,k6)]）
double atOr(const cv::Mat& m, int idx, double def = 0.0) {
    if (m.empty() || static_cast<int>(m.total()) <= idx) return def;
    cv::Mat d;
    m.convertTo(d, CV_64F);
    return d.reshape(1, 1).at<double>(0, idx);
}

} // namespace

// ============================================================================
// 构造 / 装配
// ============================================================================
ScanChains::ScanChains(ScanConfig cfg, ScanChainDeps deps)
    : cfg_(std::move(cfg)), deps_(std::move(deps)) {
    try {
        // prevState 允许为空：null=首帧未初始化的正确初态（原子锚，存储即 deps_ 内
        // 的 shared_ptr 本体，atomic_store 直接写入；非空=热启动种子快照）
        if (deps_.K1.empty() || deps_.K1.size() != cv::Size(3, 3) ||
            deps_.K2.empty() || deps_.K2.size() != cv::Size(3, 3) ||
            deps_.K1.at<double>(0, 0) <= 0 || deps_.K2.at<double>(0, 0) <= 0) {
            initError_ = "双目内参 K1/K2 缺失或非法（须 3x3 且 fx>0）";
        } else if (deps_.D1.empty() || deps_.D1.total() < 4 ||
                   deps_.D2.empty() || deps_.D2.total() < 4) {
            initError_ = "畸变系数 D1/D2 缺失或元素数 <4";
        } else if (deps_.imageWidth <= 0 || deps_.imageHeight <= 0) {
            initError_ = "imageWidth/imageHeight 须 >0（undistort_cpu 构造校验）";
        }
#ifdef JMW_BUILD_CUDA
        else if (cfg_.enableLaser && !deps_.laserPool) {
            initError_ = "enableLaser=true 需要 laserPool";
        }
#endif
    } catch (const std::exception& e) {
        initError_ = std::string("依赖校验异常: ") + e.what();   // 如 K1 非 CV_64F
    }
    if (!initError_.empty()) {
        spdlog::error("[ScanChains] 装配依赖非法，三钩子将恒 fail: {}", initError_);
    }
}

ScanChains::~ScanChains() = default;

std::shared_ptr<ScanLaneOps> ScanChains::makeOps() const {
    auto ops = std::make_shared<ScanLaneOps>();
    try {
#ifdef JMW_BUILD_CUDA
        ops->sep      = std::make_unique<calib::LaserMarkingSeparationCUDA>();
        ops->ccl      = std::make_unique<calib::RegionAnalyzerCUDA>();
        ops->steger   = std::make_unique<calib::StegerExtractorCUDA>();
        ops->undistG  = std::make_unique<calib::UndistortPointsCuda>(
            calib::UndistortPointsParams{deps_.K1, deps_.D1, {}, {}});
        calib::EpipolarInterpParams epi;
        epi.lineIdCheck = false;                // 扫描模式：仅几何邻近判据
        ops->epipolar = std::make_unique<calib::EpipolarInterpCuda>(epi);
        ops->matchScan = std::make_unique<calib::LaserMatchScanCuda>();
        if (deps_.laserTable) {
            if (!ops->matchScan->SetTempTable(deps_.laserTable))
                spdlog::warn("[ScanChains] laser_match_scan 温度表注入失败（空表？）");
        } else if (cfg_.enableLaser) {
            spdlog::warn("[ScanChains] 未注入温度表，激光匹配将失败");
        }                                        // A 模式无表属正常配置，不告警
        ops->recon = std::make_unique<calib::LaserReconstructCuda>();
#endif
        calib::ImageSplitCPUParams sp;
        sp.enableBoundaryCheck = true;          // ROI 越界安全裁剪
        ops->split = std::make_unique<calib::ImageSplitCPU>(sp);
        ops->zernike = std::make_unique<calib::ZernikeEdgeCPU>();
        ops->merge  = std::make_unique<calib::ImageMergeCPU>();

        calib::MarkerUndistortCPUParams mp;
        mp.fx1 = deps_.K1.at<double>(0, 0); mp.fy1 = deps_.K1.at<double>(1, 1);
        mp.cx1 = deps_.K1.at<double>(0, 2); mp.cy1 = deps_.K1.at<double>(1, 2);
        mp.fx2 = deps_.K2.at<double>(0, 0); mp.fy2 = deps_.K2.at<double>(1, 1);
        mp.cx2 = deps_.K2.at<double>(0, 2); mp.cy2 = deps_.K2.at<double>(1, 2);
        mp.k1_1 = atOr(deps_.D1, 0); mp.k2_1 = atOr(deps_.D1, 1);
        mp.p1_1 = atOr(deps_.D1, 2); mp.p2_1 = atOr(deps_.D1, 3);
        mp.k3_1 = atOr(deps_.D1, 4);
        mp.k1_2 = atOr(deps_.D2, 0); mp.k2_2 = atOr(deps_.D2, 1);
        mp.p1_2 = atOr(deps_.D2, 2); mp.p2_2 = atOr(deps_.D2, 3);
        mp.k3_2 = atOr(deps_.D2, 4);
        mp.imageWidth  = deps_.imageWidth;      // 立体矫正矩阵走每帧 SetRectifyMatrices
        mp.imageHeight = deps_.imageHeight;
        ops->undistCpu = std::make_unique<calib::MarkerUndistortCPU>(mp);

        ops->ellipse     = std::make_unique<calib::EllipseFitCPU>();
        ops->match       = std::make_unique<calib::MarkerMatchCPU>();
        ops->epiIntersect = std::make_unique<calib::EpipolarIntersectCPU>();
        ops->edgeMatch   = std::make_unique<calib::EdgeMatchCPU>();

        calib::PointReconstructCPUParams pp;    // fx/cx 构造校验需要；P1/P2/Q 每帧注入
        pp.fxLeft = mp.fx1;  pp.fyLeft = mp.fy1;  pp.cxLeft = mp.cx1;  pp.cyLeft = mp.cy1;
        pp.fxRight = mp.fx2; pp.fyRight = mp.fy2; pp.cxRight = mp.cx2; pp.cyRight = mp.cy2;
        ops->reconstruct = std::make_unique<calib::PointReconstructCPU>(pp);

        ops->flowFuse  = std::make_unique<calib::MarkerOpticalFlowFuseCPU>();
        ops->frameFuse = std::make_unique<calib::FrameFuseCPU>();
    } catch (const std::exception& e) {
        spdlog::error("[ScanChains] lane 算子集构造失败: {}", e.what());
        return nullptr;
    }
    return ops;
}

ScanChains::Hooks ScanChains::assemble() {
    Hooks hooks;

    // ------------------------------------------------------------------
    // gpuChain — GPU 前段 + 激光链（E 核线程，持槽）
    // ------------------------------------------------------------------
    hooks.gpuChain = [this](sched::GpuSlotService::SlotGuard& guard,
                            const std::shared_ptr<const data::EnhancedFrame>& frame,
                            ScanFront& front,
                            std::function<void()> frontReady) -> bool {
        if (!initError_.empty()) {
            spdlog::error("[ScanChains] gpuChain: 装配错误 {}", initError_);
            return false;
        }
        if (!front.ops) {                       // 每 lane 首帧惰性建（先于 frontReady）
            front.ops = makeOps();
            if (!front.ops) return false;
        }
#ifndef JMW_BUILD_CUDA
        (void)guard; (void)frame; (void)frontReady;
        spdlog::error("[ScanChains] 无 CUDA 构建仅编译守卫，GPU 链运行不支持");
        return false;
#else
        ScanLaneOps& ops = *front.ops;
        auto stream = cv::cuda::StreamAccessor::wrapStream(guard.stream);
        front.laserBlock.reset();               // 防上帧残留（失败/池耗尽路径）
        front.laserTruncated = false;           // 同上（截断标志逐帧重置）

        // 1) mask_separation L/R（host Mat 入参，算子自上传）
        calib::LaserMarkingSeparationResult sepL, sepR;
        try {
            sepL = ops.sep->Execute(frame->grayL, stream);
            if (!sepL.success) {
                spdlog::warn("[ScanChains] mask_separation L 失败: {}", sepL.message);
                return false;
            }
            sepR = ops.sep->Execute(frame->grayR, stream);
            if (!sepR.success) {
                spdlog::warn("[ScanChains] mask_separation R 失败: {}", sepR.message);
                return false;
            }
        } catch (const std::exception& e) {
            spdlog::error("[ScanChains] mask_separation 异常: {}", e.what());
            return false;
        }

        // 2) ccl（吃 d_markingPointMask）L/R → 包围盒入 front 前段分区
        calib::RegionAnalysisResult cclL, cclR;
        try {
            cclL = ops.ccl->Execute(sepL.d_markingPointMask, stream);
            if (!cclL.success) {
                spdlog::warn("[ScanChains] ccl L 失败: {}", cclL.message);
                return false;
            }
            cclR = ops.ccl->Execute(sepR.d_markingPointMask, stream);
            if (!cclR.success) {
                spdlog::warn("[ScanChains] ccl R 失败: {}", cclR.message);
                return false;
            }
        } catch (const std::exception& e) {
            spdlog::error("[ScanChains] ccl 异常: {}", e.what());
            return false;
        }
        front.roisL = cclL.toRectList();        // host 数据（ccl 内部已同步下载）
        front.roisR = cclR.toRectList();

        // 3) ccl 就绪点：提交 P 链（此后激光段与 P 链帧内并行）
        frontReady();

        // 4) A 模式短路：无激光段，提前归还 GPU 槽
        if (!cfg_.enableLaser) {
            guard.reset();                      // 幂等，析构不再归还
            return true;
        }

        // 5) 激光链（同 stream 串行）。结果三态：
        //    1=激光块已产 / 0=无激光数据（空掩膜等，正常降级，帧仍有效）
        //    / -1=算子失败或异常（帧销毁，孤儿等待由 runtime 处理）。
        //    ⚠ 空数据须显式短路：部分 CUDA 算子对空 GpuMat 是 SEH 崩溃而非
        //    C++ 异常，try/catch 兜不住——逐级查空，空即跳出。
        auto laserStatus = [&]() -> int {
            try {
                ops.d_grayL.upload(frame->grayL, stream);
                ops.d_grayR.upload(frame->grayR, stream);

                auto hasPts = [](const std::shared_ptr<cv::cuda::GpuMat>& m) {
                    return m && !m->empty();
                };

                auto stL = ops.steger->Execute(ops.d_grayL, *sepL.d_laserMask, stream,
                                               calib::GroupMode::Flat);
                if (!stL.success) {
                    spdlog::warn("[ScanChains] steger L 失败: {}", stL.message);
                    return -1;
                }
                if (stL.totalPointCount == 0 || !hasPts(stL.d_centerPoints)) {
                    spdlog::info("[ScanChains] steger L 无激光点，本帧无激光（降级）");
                    return 0;
                }
                auto stR = ops.steger->Execute(ops.d_grayR, *sepR.d_laserMask, stream,
                                               calib::GroupMode::Flat);
                if (!stR.success) {
                    spdlog::warn("[ScanChains] steger R 失败: {}", stR.message);
                    return -1;
                }
                if (stR.totalPointCount == 0 || !hasPts(stR.d_centerPoints)) {
                    spdlog::info("[ScanChains] steger R 无激光点，本帧无激光（降级）");
                    return 0;
                }

                calib::UndistortPointsParams upL, upR;
                upL.cameraMatrix = deps_.K1; upL.distCoeffs = deps_.D1;
                upL.R = cv::Mat(frame->snapshot.R1); upL.P = cv::Mat(frame->snapshot.P1);
                upR.cameraMatrix = deps_.K2; upR.distCoeffs = deps_.D2;
                upR.R = cv::Mat(frame->snapshot.R2); upR.P = cv::Mat(frame->snapshot.P2);
                ops.undistG->SetParams(upL);
                auto unL = ops.undistG->Execute(*stL.d_centerPoints, *stL.d_line_ids,
                                                 stream);
                if (!unL.success) {
                    spdlog::warn("[ScanChains] undistort_cuda L 失败: {}", unL.message);
                    return -1;
                }
                if (!hasPts(unL.d_rectifiedPoints)) return 0;
                ops.undistG->SetParams(upR);
                auto unR = ops.undistG->Execute(*stR.d_centerPoints, *stR.d_line_ids,
                                                 stream);
                if (!unR.success) {
                    spdlog::warn("[ScanChains] undistort_cuda R 失败: {}", unR.message);
                    return -1;
                }
                if (!hasPts(unR.d_rectifiedPoints)) return 0;

                auto eiL = ops.epipolar->Execute(*unL.d_rectifiedPoints,
                                                 *unL.d_line_ids, stream);
                if (!eiL.success) {
                    spdlog::warn("[ScanChains] epipolar_interp L 失败: {}", eiL.message);
                    return -1;
                }
                if (eiL.interpCount == 0 || !hasPts(eiL.d_interpPoints)) return 0;
                auto eiR = ops.epipolar->Execute(*unR.d_rectifiedPoints,
                                                 *unR.d_line_ids, stream);
                if (!eiR.success) {
                    spdlog::warn("[ScanChains] epipolar_interp R 失败: {}", eiR.message);
                    return -1;
                }
                if (eiR.interpCount == 0 || !hasPts(eiR.d_interpPoints)) return 0;

                ops.matchScan->SetCurrentTemperature(frame->temperature);
                auto mr = ops.matchScan->Execute(*eiL.d_interpPoints,
                                                 *eiL.d_interp_line_ids,
                                                 *eiR.d_interpPoints,
                                                 *eiR.d_interp_line_ids, stream);
                if (!mr.success) {
                    spdlog::warn("[ScanChains] laser_match_scan 失败: {}", mr.message);
                    return -1;
                }
                if (mr.matchedCount == 0 || !hasPts(mr.d_matched_left) ||
                    !hasPts(mr.d_matched_right)) {
                    spdlog::info("[ScanChains] 激光匹配 0 点，本帧无激光（降级）");
                    return 0;
                }

                auto rc = ops.recon->Execute(*mr.d_matched_left, *mr.d_matched_right,
                                             *mr.d_matched_line_ids,
                                             cv::Mat(frame->snapshot.Q), stream);
                if (!rc.success) {
                    spdlog::warn("[ScanChains] laser_reconstruct 失败: {}", rc.message);
                    return -1;
                }
                if (rc.validCount == 0 || !hasPts(rc.d_points3d)) return 0;

                // 6) 激光块入池（池耗尽=降级无激光，帧仍有效）
                if (deps_.laserPool) {
                    auto blk = deps_.laserPool->acquire(deps_.poolAcquireTimeout);
                    if (blk) {
                        auto src = rc.d_points3d->reshape(3, 1);   // 统一 1×N CV_32FC3
                        int n = std::min<int>(src.cols,
                                              blk->get()->points.cols);  // 池容量裁剪
                        if (n < src.cols) {
                            spdlog::warn("[ScanChains] 激光点数 {} 超池块容量 {}，"
                                         "截断至 {}（降级）",
                                         src.cols, blk->get()->points.cols, n);
                            front.laserTruncated = true;
                        }
                        if (n > 0) {
                            cv::cuda::GpuMat dst = blk->get()->points.colRange(0, n);
                            src.colRange(0, n).copyTo(dst, stream);
                            blk->get()->count = n;
                            blk->get()->frameId = frame->frameId;
                            front.laserBlock = std::move(*blk);
                            return 1;
                        }
                    } else {
                        spdlog::warn("[ScanChains] 激光块池取块超时，本帧无激光（降级）");
                    }
                }
                return 0;
            } catch (const std::exception& e) {
                spdlog::error("[ScanChains] 激光链异常: {}", e.what());
                return -1;
            }
        }();
        if (laserStatus < 0) return false;

        stream.waitForCompletion();             // 块数据落定后方可交 eFinalize/FuseConsumer
        return true;
#endif
    };

    // ------------------------------------------------------------------
    // pChain — 标记点链 + 配准（P 核 worker）
    // ------------------------------------------------------------------
    hooks.pChain = [this](const std::shared_ptr<const data::EnhancedFrame>& frame,
                          ScanFront& front, FrameResult& result) -> Result {
        if (!initError_.empty()) return Result::fail("ScanChains: " + initError_);
        auto ops = front.ops;
        if (!ops) return Result::fail("ScanChains: lane 算子集未初始化（gpuChain 未先行）");
        try {
            std::vector<cv::Point3d> positions;
            std::vector<cv::Vec3d> normals;
            if (!runMarkerChain(*frame, *ops, front, positions, normals))
                result.quality = Scanner::QualityFlag::Degraded;   // 空帧=正常降级
            runRegistration(*frame, *ops, positions, normals, result);
        } catch (const std::exception& e) {
            spdlog::error("[ScanChains] pChain 异常: {}", e.what());
            return Result::fail(std::string("ScanChains pChain 异常: ") + e.what());
        }
        return Result::ok();
    };

    // ------------------------------------------------------------------
    // eFinalize — 汇合组装 + 发布（E 核线程）
    // ------------------------------------------------------------------
    hooks.eFinalize = [this](const std::shared_ptr<const data::EnhancedFrame>& frame,
                             ScanFront& front, FrameResult& result,
                             std::future<Scanner::Result>& fut) -> Result {
        Result pr;
        try {
            pr = fut.get();                     // pChain Result/异常均在此消费
        } catch (const std::exception& e) {
            spdlog::error("[ScanChains] pChain future 异常: {}", e.what());
            return Result::fail(std::string("pChain future 异常: ") + e.what());
        }
        if (!pr.success) return pr;             // P 链失败=帧丢弃（不入队列）

        result.frameId = frame->frameId;
        result.temperature = frame->temperature;
        result.timestamp = 0;                   // EnhancedFrame 暂无时间戳字段（06 契约待补）
#ifdef JMW_BUILD_CUDA
        result.laser = std::move(front.laserBlock);
        front.laserBlock.reset();
        // quality 单调降级：仅 Normal 可降为 Degraded（不把 Warning 升级/覆盖已有降级）
        if (result.quality == Scanner::QualityFlag::Normal) {
            if (cfg_.enableLaser && !result.laser)
                result.quality = Scanner::QualityFlag::Degraded;   // 有激光无块→降级
            else if (front.laserTruncated)
                result.quality = Scanner::QualityFlag::Degraded;   // 池容量截断→降级
        }
#endif
        if (deps_.sink) deps_.sink->push(result);                   // T8：eFinalize 自行 push
        return Result::ok();
    };

    return hooks;
}

// ============================================================================
// P 核链：标记点检测链（image_split → … → point_reconstruct）
// 返回 false=本帧无有效标记点（调用方按降级处理）
// ============================================================================
bool ScanChains::runMarkerChain(const data::EnhancedFrame& frame, ScanLaneOps& ops,
                                const ScanFront& front,
                                std::vector<cv::Point3d>& positions,
                                std::vector<cv::Vec3d>& normals) const {
    // 每相机：image_split(gray, rois) → zernike 逐子图 → image_merge
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
    if (!runCamera(frame.grayL, front.roisL, mL) ||
        !runCamera(frame.grayR, front.roisR, mR))
        return false;

    // undistort_cpu（矫正矩阵=帧 snapshot；R1/R2/P1/P2/Q 每帧注入）
    const cv::Mat R1(frame.snapshot.R1), R2(frame.snapshot.R2),
        P1(frame.snapshot.P1), P2(frame.snapshot.P2), Q(frame.snapshot.Q);
    ops.undistCpu->SetRectifyMatrices(R1, R2, P1, P2, Q);
    auto ur = ops.undistCpu->Execute(mL.mergedEdgePoints, mR.mergedEdgePoints,
                                     mL.groupIds, mR.groupIds);
    if (!ur.success) return false;

    // ellipse_fit 逐组（矫正坐标系）
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
    if (ellL.size() < 3 || ellR.size() < 3) return false;   // 重建/配准最少点数

    std::vector<cv::Point2f> cL, cR;
    cL.reserve(ellL.size()); cR.reserve(ellR.size());
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

    ops.reconstruct->SetProjectionMatrices(P1, P2, Q);
    auto pr = ops.reconstruct->Execute(em);
    if (!pr.success) return false;

    for (const auto& m : pr.markerResults) {
        if (!m.validPlane || !m.validCircle) continue;
        positions.emplace_back(m.centerX, m.centerY, m.centerZ);
        normals.emplace_back(m.normalX, m.normalY, m.normalZ);
    }
    return positions.size() >= 3;
}

// ============================================================================
// P 核链：配准（prevState 原子快照模型——最新快照，不严格等帧号 N-1）
// ============================================================================
void ScanChains::runRegistration(const data::EnhancedFrame& frame, ScanLaneOps& ops,
                                 const std::vector<cv::Point3d>& positions,
                                 const std::vector<cv::Vec3d>& normals,
                                 FrameResult& result) {  // 非常量：经 deps_.prevState 原子写全局快照
    const auto n = positions.size();

    // 当前帧 → MarkerPoint3D 形态（id 未知，-1 占位）
    auto toPoints = [&](const std::vector<int>& ids) {
        std::vector<calib::MarkerPoint3D> pts(n);
        for (size_t i = 0; i < n; ++i) {
            pts[i].x = positions[i].x; pts[i].y = positions[i].y; pts[i].z = positions[i].z;
            pts[i].nx = normals[i](0); pts[i].ny = normals[i](1); pts[i].nz = normals[i](2);
            pts[i].globalId = i < ids.size() ? ids[i] : -1;
        }
        return pts;
    };

    auto prev = calib::AtomicFrameState::load(deps_.prevState);

    // —— 首帧初始化分支（锚空 / 锚内无点）——
    if (!prev || prev->rawPoints.empty()) {
        if (n == 0) {                           // 无点且无锚：I 位姿，降级
            fillRT(result, cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
            result.quality = Scanner::QualityFlag::Degraded;
            return;
        }
        std::vector<int> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int>(i);
        auto st = std::make_shared<calib::AtomicFrameState>();
        st->rawPoints = toPoints(ids);
        st->normals.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            st->normals.push_back(normals[i](0));
            st->normals.push_back(normals[i](1));
            st->normals.push_back(normals[i](2));
        }
        st->globalIds = ids;
        fillRT(result, cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
        // st->R/T 保持结构默认（I/0），与结果一致
        st->frameId = frame.frameId;
        result.markers = st->rawPoints;
        calib::AtomicFrameState::store(deps_.prevState, std::move(st));
        return;
    }

    // —— 有锚但本帧无点：沿用快照 R/T，降级 ——
    if (n == 0) {
        fillRT(result, matxFromArr9(prev->R), vec3FromArr3(prev->T));
        result.quality = Scanner::QualityFlag::Degraded;
        return;
    }

    // prev 快照 → 算子 PrevFrameState（桥接：rawPoints/normals/globalIds/R/T 全量）
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
        fillRT(result, fr.R, fr.T);
        result.quality = toScannerQuality(fr.qualityFlag);
        std::vector<int> ids;
        ids.reserve(fr.markers.size());
        result.markers.reserve(fr.markers.size());
        for (const auto& m : fr.markers) {
            ids.push_back(m.globalId);
            calib::MarkerPoint3D mp{m.rawPosition.x, m.rawPosition.y, m.rawPosition.z,
                                    m.rawNormal(0), m.rawNormal(1), m.rawNormal(2),
                                    m.globalId};
            result.markers.push_back(mp);
        }
        auto st = std::make_shared<calib::AtomicFrameState>();
        st->rawPoints = result.markers;         // 设备系坐标 + 链式 globalId
        st->normals.reserve(result.markers.size() * 3);
        for (const auto& mp : st->rawPoints) {
            st->normals.push_back(mp.nx);
            st->normals.push_back(mp.ny);
            st->normals.push_back(mp.nz);
        }
        st->globalIds = std::move(ids);
        for (int i = 0; i < 9; ++i) st->R[i] = fr.R(i / 3, i % 3);
        st->T[0] = fr.T(0); st->T[1] = fr.T(1); st->T[2] = fr.T(2);
        st->frameId = frame.frameId;
        calib::AtomicFrameState::store(deps_.prevState, std::move(st));
        return;
    }
    spdlog::warn("[ScanChains] optical_flow_fuse 失败（{}），转 frame_fuse 兜底",
                 fr.message);

    // —— 兜底 frame_fuse（当前帧 vs 快照点集；不回写快照保 globalId 链完整）——
    calib::MarkerPointSet cur{positions, normals};
    calib::MarkerPointSet prevSet{std::move(pf.rawPositions), std::move(pf.rawNormals)};
    auto ff = ops.frameFuse->Execute(cur, prevSet);
    if (ff.success) {
        fillRT(result, ff.R, ff.T);
        result.quality = Scanner::QualityFlag::Degraded;    // 兜底成功=降级帧
        std::vector<int> ids(n, -1);                        // globalId 链断裂（-1）
        result.markers = toPoints(ids);
        return;
    }
    spdlog::warn("[ScanChains] frame_fuse 兜底亦失败（{}），沿用快照 R/T", ff.message);

    // —— 再失败：沿用快照 R/T，降级 ——
    fillRT(result, matxFromArr9(prev->R), vec3FromArr3(prev->T));
    result.quality = Scanner::QualityFlag::Degraded;
}

} // namespace Scanner::pipeline
