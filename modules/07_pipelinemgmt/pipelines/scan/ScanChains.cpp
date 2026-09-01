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
#include "jmw_logging.h"
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
#include <opencv2/cudaarithm.hpp>   // cv::cuda::threshold（A 模式亮斑二值化）
#include <opencv2/imgproc.hpp>      // cv::THRESH_BINARY
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
        JMW_LOG_ERROR("07-ScanChains", "[ScanChains] 装配依赖非法，三钩子将恒 fail: {}", initError_);
    }
}

ScanChains::~ScanChains() = default;

std::shared_ptr<ScanLaneOps> ScanChains::makeOps() const {
    auto ops = std::make_shared<ScanLaneOps>();
    try {
#ifdef JMW_BUILD_CUDA
        // 分离阈值按模式（2026-09-01 定版）：B 模式（含激光线）80——线碎片亮
        // 基线；A 模式（纯补光图）50——真标志点亮斑低于 80（B40 实测 ROI=2，
        // th50 纯点图曾出椭圆 20/13）
        calib::LaserMarkingSeparationParams sepParams;
        sepParams.threshold = cfg_.enableLaser ? 80 : 50;
        ops->sep      = std::make_unique<calib::LaserMarkingSeparationCUDA>(sepParams);
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
                JMW_LOG_WARN("07-ScanChains", "[ScanChains] laser_match_scan 温度表注入失败（空表？）");
        } else if (cfg_.enableLaser) {
            JMW_LOG_WARN("07-ScanChains", "[ScanChains] 未注入温度表，激光匹配将失败");
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
        // 扫描链极线 y 容差 0.15px→3.0px：亚像素中心误差 ±0.3~1px + 现场装配/
    // 温漂，真机实测左右中心 y 差 1~3px（1.0px 下匹配 0~3 对不稳定）。算子
    // validate 上限已同步放宽至 10（像素口径）——超限仍会令 lane 算子集构造
    // 抛异常、整链瘫，勿超
    calib::MarkerMatchCPUParams mmp;
    mmp.y_tolerance = 3.0f;
    mmp.max_points = 300;      // 100→300：亮灯场景 ROI 可达 137+，超上限算子
                               // 抛异常令 pChain 整帧报废（"无点"根因之一）
    ops->match       = std::make_unique<calib::MarkerMatchCPU>(mmp);
        ops->epiIntersect = std::make_unique<calib::EpipolarIntersectCPU>();
        ops->edgeMatch   = std::make_unique<calib::EdgeMatchCPU>();

        calib::PointReconstructCPUParams pp;    // fx/cx 构造校验需要；P1/P2/Q 每帧注入
        pp.fxLeft = mp.fx1;  pp.fyLeft = mp.fy1;  pp.cxLeft = mp.cx1;  pp.cyLeft = mp.cy1;
        pp.fxRight = mp.fx2; pp.fyRight = mp.fy2; pp.cxRight = mp.cx2; pp.cyRight = mp.cy2;
        ops->reconstruct = std::make_unique<calib::PointReconstructCPU>(pp);

        // 帧间配准法线角阈值 15°→45°：真机 y 差 1~3px 重建噪声下标志点法线角
    // 波动常超 15°（帧间匹配恒 0~1 对<3 → 配准失败 → markers 空 → 无点显示）
    calib::MarkerOpticalFlowFuseCPUParams ofp;
    ofp.normalAngleThresh = 45.0;
    ops->flowFuse  = std::make_unique<calib::MarkerOpticalFlowFuseCPU>(ofp);
        ops->frameFuse = std::make_unique<calib::FrameFuseCPU>();
    } catch (const std::exception& e) {
        JMW_LOG_ERROR("07-ScanChains", "[ScanChains] lane 算子集构造失败: {}", e.what());
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
            JMW_LOG_ERROR("07-ScanChains", "[ScanChains] gpuChain: 装配错误 {}", initError_);
            return false;
        }
        if (!front.ops) {                       // 每 lane 首帧惰性建（先于 frontReady）
            front.ops = makeOps();
            if (!front.ops) return false;
        }
#ifndef JMW_BUILD_CUDA
        (void)guard; (void)frame; (void)frontReady;
        JMW_LOG_ERROR("07-ScanChains", "[ScanChains] 无 CUDA 构建仅编译守卫，GPU 链运行不支持");
        return false;
#else
        ScanLaneOps& ops = *front.ops;
        auto stream = cv::cuda::StreamAccessor::wrapStream(guard.stream);
        front.laserBlock.reset();               // 防上帧残留（失败/池耗尽路径）
        front.laserTruncated = false;           // 同上（截断标志逐帧重置）

        // 1) 掩膜来源按模式（2026-09-01 定版）：
        //    B 模式（含激光线）→ mask_separation（线点共存形态学分离——成功基线）
        //    A 模式（纯补光图）→ 直接阈值化（分离算子在纯点图失效：实测 ROI 2-3；
        //      标准做法=固定阈值找亮斑——反光标志点亮斑远亮于漫反射背景）
        calib::LaserMarkingSeparationResult sepL, sepR;
        cv::cuda::GpuMat d_binL, d_binR;        // A 模式二值图（ccl 输入）
        std::shared_ptr<cv::cuda::GpuMat> markMaskL, markMaskR;
        try {
            if (cfg_.enableLaser) {
                sepL = ops.sep->Execute(frame->grayL, stream);
                if (!sepL.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] mask_separation L 失败: {}", sepL.message);
                    return false;
                }
                sepR = ops.sep->Execute(frame->grayR, stream);
                if (!sepR.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] mask_separation R 失败: {}", sepR.message);
                    return false;
                }
                markMaskL = sepL.d_markingPointMask;
                markMaskR = sepR.d_markingPointMask;
            } else {
                ops.d_grayL.upload(frame->grayL, stream);
                ops.d_grayR.upload(frame->grayR, stream);
                constexpr double kBinThresh = 100.0;   // A 模式亮斑阈值（B40 补光）
                cv::cuda::threshold(ops.d_grayL, d_binL, kBinThresh, 255.0, cv::THRESH_BINARY, stream);
                cv::cuda::threshold(ops.d_grayR, d_binR, kBinThresh, 255.0, cv::THRESH_BINARY, stream);
                markMaskL = std::make_shared<cv::cuda::GpuMat>(d_binL);
                markMaskR = std::make_shared<cv::cuda::GpuMat>(d_binR);
            }
        } catch (const std::exception& e) {
            JMW_LOG_ERROR("07-ScanChains", "[ScanChains] 掩膜来源异常: {}", e.what());
            return false;
        }

        // 2) ccl（吃标记点掩膜）L/R → 包围盒入 front 前段分区
        // 激光掩膜观测（节流 1/30 帧，仅 B 模式）：分离出的激光掩膜非零像素
        if (cfg_.enableLaser) {
            static std::atomic<uint64_t> s_maskFrames{0};
            if (s_maskFrames.fetch_add(1) % 30 == 0) {
                cv::Mat hMaskL, hMaskR;
                sepL.d_laserMask->download(hMaskL, stream);
                sepR.d_laserMask->download(hMaskR, stream);
                stream.waitForCompletion();
                const int nL = hMaskL.empty() ? 0 : cv::countNonZero(hMaskL);
                const int nR = hMaskR.empty() ? 0 : cv::countNonZero(hMaskR);
                JMW_LOG_INFO("07-ScanChains", "[ScanChains] 激光掩膜: L={}px R={}px（0=分离未出线）",
                             nL, nR);
            }
        }
        calib::RegionAnalysisResult cclL, cclR;
        try {
            cclL = ops.ccl->Execute(markMaskL, stream);
            if (!cclL.success) {
                JMW_LOG_WARN("07-ScanChains", "[ScanChains] ccl L 失败: {}", cclL.message);
                return false;
            }
            cclR = ops.ccl->Execute(markMaskR, stream);
            if (!cclR.success) {
                JMW_LOG_WARN("07-ScanChains", "[ScanChains] ccl R 失败: {}", cclR.message);
                return false;
            }
        } catch (const std::exception& e) {
            JMW_LOG_ERROR("07-ScanChains", "[ScanChains] ccl 异常: {}", e.what());
            return false;
        }
        front.roisL = cclL.toRectList();        // host 数据（ccl 内部已同步下载）
        front.roisR = cclR.toRectList();
        // ROI 尺寸过滤（2026-09-01 性能第一步）：真标志点 φ17px（距离浮动
        // 10~45px 带）；噪声碎片/大块全剔除——此前 50 个 ROI 仅 ~13 真点，
        // zernike/ellipse 按输入量耗时（28+45ms 大头），过滤后同比例降
        {
            auto passSize = [](const std::vector<cv::Rect>& in) {
                std::vector<cv::Rect> out;
                out.reserve(in.size());
                for (const auto& r : in)
                    if (r.width >= 10 && r.width <= 45 && r.height >= 10 && r.height <= 45)
                        out.push_back(r);
                return out;
            };
            front.roisL = passSize(front.roisL);
            front.roisR = passSize(front.roisR);
        }

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
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] steger L 失败: {}", stL.message);
                    return -1;
                }
                if (stL.totalPointCount == 0 || !hasPts(stL.d_centerPoints)) {
                    JMW_LOG_INFO("07-ScanChains", "[ScanChains] steger L 无激光点，本帧无激光（降级）");
                    return 0;
                }
                auto stR = ops.steger->Execute(ops.d_grayR, *sepR.d_laserMask, stream,
                                               calib::GroupMode::Flat);
                if (!stR.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] steger R 失败: {}", stR.message);
                    return -1;
                }
                if (stR.totalPointCount == 0 || !hasPts(stR.d_centerPoints)) {
                    JMW_LOG_INFO("07-ScanChains", "[ScanChains] steger R 无激光点，本帧无激光（降级）");
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
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] undistort_cuda L 失败: {}", unL.message);
                    return -1;
                }
                if (!hasPts(unL.d_rectifiedPoints)) return 0;
                ops.undistG->SetParams(upR);
                auto unR = ops.undistG->Execute(*stR.d_centerPoints, *stR.d_line_ids,
                                                 stream);
                if (!unR.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] undistort_cuda R 失败: {}", unR.message);
                    return -1;
                }
                if (!hasPts(unR.d_rectifiedPoints)) return 0;

                auto eiL = ops.epipolar->Execute(*unL.d_rectifiedPoints,
                                                 *unL.d_line_ids, stream);
                if (!eiL.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] epipolar_interp L 失败: {}", eiL.message);
                    return -1;
                }
                if (eiL.interpCount == 0 || !hasPts(eiL.d_interpPoints)) return 0;
                auto eiR = ops.epipolar->Execute(*unR.d_rectifiedPoints,
                                                 *unR.d_line_ids, stream);
                if (!eiR.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] epipolar_interp R 失败: {}", eiR.message);
                    return -1;
                }
                if (eiR.interpCount == 0 || !hasPts(eiR.d_interpPoints)) return 0;

                // —— 帧号奇偶→激光组别分派（用户口径 2026-08-31）：固件 T/V
                // 交替打灯，偶数帧=左斜线（T 组）→ 左斜激光标定参数
                // （left_skew）；奇数帧=右斜线（V 组）→ 右斜参数（未标定，
                // 暂用左斜参数重建——右斜到位后在此按组分派）
                const bool leftSkewFrame = (frame->frameId % 2) == 0;

                // —— 激光左右匹配：旁路 laser_match_scan（2026-08-31 用户口径：
                // 激光点显示——工厂档无 mapData，查表路线不可用）。标准立体
                // 方法：矫正后同行（y）配对——下载左右极线插值点 → y 桶最近
                // 配对（|dy|<0.75px、正视差 (0.5,150]px）→ 上传喂 laser_reconstruct
                std::vector<int> laserIds;
                cv::cuda::GpuMat d_mL, d_mR, d_mId;
                {
                    cv::Mat hL, hR, hLid;
                    eiL.d_interpPoints->download(hL, stream);
                    eiR.d_interpPoints->download(hR, stream);
                    eiL.d_interp_line_ids->download(hLid, stream);
                    const auto* pL = hL.ptr<cv::Point2f>();
                    const auto* pR = hR.ptr<cv::Point2f>();
                    const int* lids = hLid.ptr<int>();
                    const int nL = static_cast<int>(hL.total());
                    const int nR = static_cast<int>(hR.total());
                    // R 点按 y 分桶（2px 粒度）
                    std::unordered_map<int, std::vector<int>> rows;
                    for (int j = 0; j < nR; ++j)
                        rows[static_cast<int>(pR[j].y * 0.5f)].push_back(j);
                    std::vector<cv::Point2f> mL2, mR2;
                    for (int i = 0; i < nL; ++i) {
                        auto it = rows.find(static_cast<int>(pL[i].y * 0.5f));
                        if (it == rows.end()) continue;
                        int best = -1;
                        float bestDy = 1e9f;
                        for (int j : it->second) {
                            const float dy = std::abs(pL[i].y - pR[j].y);
                            const float disp = pL[i].x - pR[j].x;
                            if (dy < 0.75f && disp > 0.5f && disp < 150.0f && dy < bestDy) {
                                bestDy = dy;
                                best = j;
                            }
                        }
                        if (best >= 0) {
                            mL2.push_back(pL[i]);
                            mR2.push_back(pR[best]);
                            laserIds.push_back(lids[i]);
                        }
                    }
                    if (laserIds.size() < 8) {
                        JMW_LOG_INFO("07-ScanChains", "[ScanChains] 激光同行配对 {} 点（过少，本帧降级）",
                                     laserIds.size());
                        return 0;
                    }
                    JMW_LOG_INFO("07-ScanChains", "[ScanChains] 激光帧#{}（{}斜，{}斜参数）同行配对 {} 点",
                                 frame->frameId, leftSkewFrame ? "左" : "右",
                                 leftSkewFrame ? "左" : "左(暂)",
                                 laserIds.size());
                    const int n = static_cast<int>(laserIds.size());
                    d_mL.upload(cv::Mat(n, 1, CV_32FC2, mL2.data()), stream);
                    d_mR.upload(cv::Mat(n, 1, CV_32FC2, mR2.data()), stream);
                    d_mId.upload(cv::Mat(n, 1, CV_32SC1, laserIds.data()), stream);
                }

                auto rc = ops.recon->Execute(d_mL, d_mR, d_mId,
                                             cv::Mat(frame->snapshot.Q), stream);
                if (!rc.success) {
                    JMW_LOG_WARN("07-ScanChains", "[ScanChains] laser_reconstruct 失败: {}", rc.message);
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
                            JMW_LOG_WARN("07-ScanChains", "[ScanChains] 激光点数 {} 超池块容量 {}，"
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
                        JMW_LOG_WARN("07-ScanChains", "[ScanChains] 激光块池取块超时，本帧无激光（降级）");
                    }
                }
                return 0;
            } catch (const std::exception& e) {
                JMW_LOG_ERROR("07-ScanChains", "[ScanChains] 激光链异常: {}", e.what());
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
        // 调度观测：pChain 被调即打（节流 1/30+前 3）——区分"调度层没调"与
        // "链内没到 runMarkerChain"
        static std::atomic<uint64_t> s_pcall{0};
        const uint64_t cn = s_pcall.fetch_add(1);
        if (cn < 3 || cn % 30 == 0) {
            JMW_LOG_INFO("07-ScanChains", "[ScanChains] pChain调用#{}: ops={} roisL={}",
                         cn, front.ops ? 1 : 0, front.roisL.size());
        }
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
            JMW_LOG_ERROR("07-ScanChains", "[ScanChains] pChain 异常: {}", e.what());
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
            JMW_LOG_ERROR("07-ScanChains", "[ScanChains] pChain future 异常: {}", e.what());
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
    // 观测（前 3 帧强制全打 + 之后 1/30 节流）：P 链入口 ROI 数与各算子阶段
    // 卡点定位——0=上游分离/ccl 问题，>0=P 链内问题
    static std::atomic<uint64_t> s_pchainFrames{0};
    const uint64_t pchainSeq = s_pchainFrames.fetch_add(1);
    const bool verbose = (pchainSeq < 3) || (pchainSeq % 30 == 0);
    if (verbose) {
        JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: roisL={} roisR={}",
                     pchainSeq, front.roisL.size(), front.roisR.size());
    }
    // 分段计时（lane 慢定位 2026-08-31：实测整帧 ~220ms 只消化 4.5fps→丢帧
    // 90%；split/zernike/merge/undistort/ellipse/match/epi/edge/recon 各段）
    using CLK = std::chrono::steady_clock;
    const auto tAll0 = CLK::now();
    double tSplit = 0, tZern = 0, tMerge = 0;
    auto runCamera = [&](const cv::Mat& gray, const std::vector<cv::Rect>& rois,
                         calib::ImageMergeCPUResult& merged, const char* tag) -> bool {
        const auto s0 = CLK::now();
        auto sp = ops.split->Execute(gray, rois);
        tSplit += std::chrono::duration<double, std::milli>(CLK::now() - s0).count();
        if (!sp.success || sp.splitCount == 0) {
            if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: {} split 0/败（rois={}）",
                                      tag, rois.size());
            return false;
        }
        const auto s1 = CLK::now();
        std::vector<std::vector<calib::EdgePoint>> edges(
            static_cast<size_t>(sp.splitCount));
        for (int i = 0; i < sp.splitCount; ++i) {
            auto zr = ops.zernike->Execute(sp.splitImages[static_cast<size_t>(i)]);
            if (zr.success) edges[static_cast<size_t>(i)] = std::move(zr.edgePoints);
        }
        tZern += std::chrono::duration<double, std::milli>(CLK::now() - s1).count();
        const auto s2 = CLK::now();
        merged = ops.merge->Execute(edges, rois);
        tMerge += std::chrono::duration<double, std::milli>(CLK::now() - s2).count();
        if (!(merged.success && merged.mergedEdgeCount > 0)) {
            if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: {} merge 0/败（split={} 边缘0）",
                                      tag, sp.splitCount);
            return false;
        }
        return true;
    };

    calib::ImageMergeCPUResult mL, mR;
    if (!runCamera(frame.grayL, front.roisL, mL, "L") ||
        !runCamera(frame.grayR, front.roisR, mR, "R"))
        return false;

    // undistort_cpu（矫正矩阵=帧 snapshot；R1/R2/P1/P2/Q 每帧注入）
    const cv::Mat R1(frame.snapshot.R1), R2(frame.snapshot.R2),
        P1(frame.snapshot.P1), P2(frame.snapshot.P2), Q(frame.snapshot.Q);
    const auto tu0 = CLK::now();
    ops.undistCpu->SetRectifyMatrices(R1, R2, P1, P2, Q);
    auto ur = ops.undistCpu->Execute(mL.mergedEdgePoints, mR.mergedEdgePoints,
                                     mL.groupIds, mR.groupIds);
    const double tUndist = std::chrono::duration<double, std::milli>(CLK::now() - tu0).count();
    if (!ur.success) return false;

    // ellipse_fit 逐组（矫正坐标系）
    const auto te0 = CLK::now();
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
    const double tEllipse = std::chrono::duration<double, std::milli>(CLK::now() - te0).count();
    if (ellL.size() < 3 || ellR.size() < 3) {   // 重建/配准最少点数
        if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: 椭圆组不足 L={} R={}（需≥3）",
                                  ellL.size(), ellR.size());
        return false;
    }

    std::vector<cv::Point2f> cL, cR;
    cL.reserve(ellL.size()); cR.reserve(ellR.size());
    for (auto& e : ellL) cL.push_back(e.centerPoint2f());
    for (auto& e : ellR) cR.push_back(e.centerPoint2f());

    if (verbose) JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: →marker_match（L中心={} R中心={}）",
                              pchainSeq, cL.size(), cR.size());
    const auto tm0 = CLK::now();
    auto mm = ops.match->Execute(cL, cR);
    const double tMatch = std::chrono::duration<double, std::milli>(CLK::now() - tm0).count();
    if (!mm.success) {
        if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: marker_match 败（L中心={} R中心={}）",
                                  cL.size(), cR.size());
        return false;
    }
    if (verbose) JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: →epipolar_intersect", pchainSeq);

    const auto tp0 = CLK::now();
    auto eiL = ops.epiIntersect->Execute(ellL);
    auto eiR = ops.epiIntersect->Execute(ellR);
    const double tEpi = std::chrono::duration<double, std::milli>(CLK::now() - tp0).count();
    if (!eiL.success || !eiR.success) {
        if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: 极线交点败 L={} R={}",
                                  eiL.success, eiR.success);
        return false;
    }
    if (verbose) JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: →edge_match", pchainSeq);

    const auto tx0 = CLK::now();
    auto em = ops.edgeMatch->Execute(eiL.ellipseResults, eiR.ellipseResults,
                                     mm.centerMatches);
    const double tEdge = std::chrono::duration<double, std::milli>(CLK::now() - tx0).count();
    if (!em.success) {
        if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: edge_match 败（中心匹配对={}）",
                                  mm.centerMatches.size());
        return false;
    }
    if (verbose) JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: →reconstruct", pchainSeq);

    const auto tr0 = CLK::now();
    ops.reconstruct->SetProjectionMatrices(P1, P2, Q);
    auto pr = ops.reconstruct->Execute(em);
    const double tRecon = std::chrono::duration<double, std::milli>(CLK::now() - tr0).count();
    if (!pr.success) {
        if (verbose) JMW_LOG_WARN("07-ScanChains", "[ScanChains] P链: 三维重建败");
        return false;
    }
    if (verbose) JMW_LOG_INFO("07-ScanChains", "[ScanChains] P链观测#{}: reconstruct 完成", pchainSeq);

    for (const auto& m : pr.markerResults) {
        if (!m.validPlane || !m.validCircle) continue;
        // 无几何过滤（用户口径 2026-08-31：过滤掩盖真实问题——假匹配飞点
        // 须从源头治理：左右匹配质量/曝光/标定，不以范围判定藏污）
        positions.emplace_back(m.centerX, m.centerY, m.centerZ);
        normals.emplace_back(m.normalX, m.normalY, m.normalZ);
    }
    if (verbose) {
        const auto matchedCenters = static_cast<size_t>(std::count_if(
            mm.centerMatches.begin(), mm.centerMatches.end(),
            [](int v) { return v >= 0; }));
        // 耗时汇总（lane 慢定位：实测整帧 ~220ms/4.5fps→丢帧 90%）
        JMW_LOG_INFO("07-ScanChains",
                     "[ScanChains] P链耗时#{0}: split={1:.1f} zernike={2:.1f} merge={3:.1f} "
                     "undist={4:.1f} ellipse={5:.1f} match={6:.1f} epi={7:.1f} "
                     "edge={8:.1f} recon={9:.1f} | 总(含杂)={10:.1f}ms",
                     pchainSeq, tSplit, tZern, tMerge, tUndist, tEllipse, tMatch,
                     tEpi, tEdge, tRecon,
                     std::chrono::duration<double, std::milli>(CLK::now() - tAll0).count());
        // y 差诊断：成功对的左右中心 y 差分布（中位/最大）——判断立体校正
        // 质量：中位 <1.5px=矫正良好（容差安全）；中位 >3px=标定与装配不匹
        // 配（应重标定而非放容差）
        std::vector<float> ydiffs;
        for (size_t i = 0; i < mm.centerMatches.size() && i < cL.size(); ++i) {
            const int j = mm.centerMatches[i];
            if (j >= 0 && static_cast<size_t>(j) < cR.size()) {
                ydiffs.push_back(std::abs(cL[i].y - cR[static_cast<size_t>(j)].y));
            }
        }
        std::sort(ydiffs.begin(), ydiffs.end());
        const bool hasYd = !ydiffs.empty();
        JMW_LOG_INFO("07-ScanChains",
                     "[ScanChains] P链观测: 椭圆L={} R={} 中心匹配对={} y差(中位={:.2f} 最大={:.2f}px) 边匹配统计(总/成/跳={}/{}/{}) 重建结果={} 出口点={}",
                     ellL.size(), ellR.size(), matchedCenters,
                     hasYd ? ydiffs[ydiffs.size() / 2] : -1.0f,
                     hasYd ? ydiffs.back() : -1.0f,
                     em.statistics.totalEllipsePairs, em.statistics.matchedPairs,
                     em.statistics.skippedPairs, pr.markerResults.size(), positions.size());
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
    JMW_LOG_WARN("07-ScanChains", "[ScanChains] optical_flow_fuse 失败（{}），转 frame_fuse 兜底",
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
    JMW_LOG_WARN("07-ScanChains", "[ScanChains] frame_fuse 兜底亦失败（{}），沿用快照 R/T", ff.message);

    // —— 再失败：沿用快照 R/T，降级 ——（调试 2026-08-31：无 ID 也发点——
    // 配准失败帧此前 markers 空 → 融合无点 → 3D 恒无显示；先保显示，配准
    // 质量问题（法线/阈值/快照链）另行治理）
    fillRT(result, matxFromArr9(prev->R), vec3FromArr3(prev->T));
    result.quality = Scanner::QualityFlag::Degraded;
    result.markers = toPoints(std::vector<int>(n, -1));   // globalId=-1（链断）
}

} // namespace Scanner::pipeline
