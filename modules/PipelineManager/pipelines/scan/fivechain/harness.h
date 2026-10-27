// harness.h — fc_executor 兼容层（07 PipelineManager 五链集成 261025）
//
// 沙盒 fc_executor.h include "harness.h"——本文件提供等价的类型/算子/接口
// 声明，使 fc_executor.cpp 无需修改即可在 07 编译。实现由 FiveChainAdapter
// 提供（算子实例/配置/帧源/观测/融合——与沙盒 Harness 逐项对应）。
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <map>
#include <functional>
#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include "pipelines/scan/fivechain/FiveChainAdapter.h"
#include "pipelines/scan/ScanTypes.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/fivechain/triangle_index.h"

// 沙盒算子（Phase 1 已合入主库——同 namespace calib）
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"
#include "scanning/laser/laser_match_scan_v3_split/laser_match_scan_v3_split.h"
#include "core/laser/laser_reconstruct/laser_reconstruct_cpu.h"
#include "core/laser/steger_fast/steger_fast.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "core/laser/epipolar_interp_dual/epipolar_interp_dual_cuda.h"
#include "core/laser/flat_center/flat_center.h"
#include "core/vision/ccl/region_analyze_cuda.h"
// 标志点链算子（harness.h 的 MarkerOps 消费）
#include "core/marker/image_split/image_split_cpu.h"
#include "core/marker/zernike_edge/zernike_edge_cpu.h"
#include "core/marker/image_merge/image_merge_cpu.h"
#include "core/marker/undistort_cpu/undistort_points_cpu.h"
#include "core/marker/ellipse_fit/ellipse_fit_cpu.h"
#include "core/marker/marker_match/marker_match_cpu.h"
#include "core/marker/epipolar_intersect/epipolar_intersect_cpu.h"
#include "core/marker/edge_match/edge_match_cpu.h"
#include "core/marker/point_reconstruct/point_reconstruct_cpu.h"
#include "core/marker/conic_reconstruct/conic_reconstruct_cpu.h"
#include "core/marker/optical_flow_fuse/marker_optical_flow_fuse_cpu.h"
#include "core/marker/frame_fuse/frame_fuse_cpu.h"
#include "core/scheduler/prev_frame_state.h"

namespace scr {

/// 沙盒 ScrConfig 的 fivechain 节（fc_executor 只消费这些＋ opParams）
struct ScrConfigCompat {
    Scanner::pipeline::FiveChainConfig fivechain;
    struct Runtime {
        int warmupCycles = 0;
        int dumpFullFrames = 0;
        int maxCycles = 0;
        double nominalTemperature = 25.0;
        bool markerOnly = false;
        // 配准链 runtime 参数（沙盒 config runtime 节默认值——对拍同参必需）
        double flowRmseDriftRatio = 10.0;   // 光流误差趋势门倍率（0=关）
        double mintDedupRadiusMm = 5.0;     // 新点发号查重半径（累积漂移口径）
        int fuseSubsetMax = 16;             // 兜底子集装配上限（min 算子上限）
        int fuseMinMatched = 4;             // 兜底验收门 matched≥
        double fuseMaxRmseMm = 2.0;         // 兜底验收门 rmse<
        // 261027 诊断：hold 帧强制融合（无标志点场景验证激光链——R/T 为上帧
        // 沿用值，仅诊断用，非测量口径）。env JMW_FC_FUSE_ON_HOLD=1 开
        bool fuseOnHold = false;
    } runtime;
    nlohmann::json operators;   // 算子参数覆盖（opParams）

    nlohmann::json opParams(const std::string& name) const {
        if (operators.contains(name)) return operators[name];
        return nlohmann::json::object();
    }
};

/// 帧源周期信息（fc_executor readerLoop 消费——tubeOdd＝激光管奇偶标记）
struct FrameCycleInfo {
    int laserIdx = 0;
    bool tubeOdd = true;
};

struct FrameSourceCompat {
    /// 沙盒静态周期表（对拍路径预填）——实时路径走 cycleAt 动态求值
    std::vector<FrameCycleInfo> cyclesStatic;
    /// 07 实时桥：idx→周期信息动态求值（帧元数据在 readFrame 时才知）。
    /// fc_executor 三处消费（reader/chain1/chain1b 线程）——桥实现须线程安全
    std::function<FrameCycleInfo(size_t idx)> cycleAt;
    /// cycles[idx] 兼容代理（沙盒源码 h_.src_->cycles[idx] 直下标——双路径分发）
    struct CyclesProxy {
        FrameSourceCompat* fs = nullptr;
        FrameCycleInfo operator[](size_t i) const {
            if (fs == nullptr) return FrameCycleInfo{};
            if (fs->cycleAt) return fs->cycleAt(i);
            return i < fs->cyclesStatic.size() ? fs->cyclesStatic[i] : FrameCycleInfo{};
        }
    };
    CyclesProxy cycles;   // 构造后由拥有方置 cycles.fs = this
    /// readFrame：07 侧由外部逐帧注入（五链适配器桥接 SlotRing→GpuMat）
    std::function<bool(int idx, bool left, cv::Mat& out, std::string& err)> readFrame;
};

} // namespace scr

// 完整定义 CycleProducts（fc_executor 的 FcJob 按值持有——须完整定义）
namespace scr {
struct CycleProductsCompat {
    cv::Mat frameGrayL, frameGrayR;
    calib::LaserMarkingSeparationResult sepL, sepR;
    calib::StegerResult stegerL, stegerR;
    calib::UndistortPointsResult undistCudaL, undistCudaR;
    calib::EpipolarInterpResult interpL, interpR;
    calib::LaserMatchScanResultV3 matchV3;
    calib::LaserReconstructResult recon;
    calib::RegionAnalysisResult cclL, cclR;
    calib::ImageSplitCPUResult splitL, splitR;
    std::vector<calib::ZernikeEdgeCPUResult> zernL, zernR;
    calib::ImageMergeCPUResult mergeL, mergeR;
    calib::StereoUndistortResult undistCpu;
    std::vector<calib::EllipseFitCPUResult> ellL, ellR;
    calib::MarkerMatchCPUResult mm;
    calib::ConicReconstructCPUResult conic;
    calib::MarkerOpticalFlowFuseCPUResult regFlow;
    std::string regSource;
    double R[9] = {1,0,0,0,1,0,0,0,1};
    double T[3] = {0,0,0};
    // 配准兜底（沙盒 harness.h L91-93 同款）
    bool usedFrameFuse = false;
    bool fuseSlowPath = false;   // 配准兜底 5 帧胜出解来自慢路径（预测失效频率）
    calib::FrameFuseCPUResult regFuse;
    // 【E3】fivechain：1b/chain1 流已排空标记——RLE 帧宿主可见拷出免事件同步
    bool gpuDrained = false;
    // 统计/报告（沙盒 report.csv 字段——fc_executor 的 fuseLoop 写）
    int laserMatched = 0;
    int laserPoints3D = 0;
    double rowResidMedPx = 0.0;
    double dispMedPx = 0.0;
    int markersDetected = 0;
    int markersValid = 0;
    int regMatched = 0;
    // 步骤状态
    std::map<std::string, bool> stepSuccess_;
    std::map<std::string, std:: string> stepError_;
    // 配准诊断
    int flowMatchedDiag = -1;
    double flowRmseDiag = -1.0;
    std::string flowMessageDiag;

    // —— 261025 实时桥帧元数据（07 桥经 onFrameMeta 钩子填；沙盒对拍路径不填）——
    uint64_t metaFrameId = 0;                   // 真实帧号（fc idx≠ring 帧号）
    cv::Matx33d snapR1 = cv::Matx33d::eye(), snapR2 = cv::Matx33d::eye();
    cv::Matx34d snapP1 = cv::Matx34d::eye(), snapP2 = cv::Matx34d::eye();
    cv::Mat snapQ;                              // 4×4 CV_64F（chain2 ReconstructInto 消费）
    double temperature = 0.0;
    bool metaValid = false;                     // false=沙盒路径（算子用构造期标定）
};
} // namespace scr

using CycleProducts = scr::CycleProductsCompat;
namespace Scanner::pipeline { class FcHarnessCompat; }
using Harness = Scanner::pipeline::FcHarnessCompat;

namespace Scanner::pipeline {

/// 沙盒 Harness 的 07 等价体——fc_executor 经 h_ 成员访问的全部接口
/// （算子/帧源/配准/计时/融合/观测——由 FiveChainAdapter 持有实现）
class FcHarnessCompat {
public:
    FcHarnessCompat(FiveChainAdapter& adapter, scr::FrameSourceCompat& src)
        : src_{&src}, adapter_(&adapter) {
        cfg.fivechain = adapter.cfg();
    }

    // —— fc_executor 访问的 Harness 成员（h_.xxx → adapter_/cfg 桥接）——
    scr::ScrConfigCompat cfg_;           // fc_executor.cpp 用 h_.cfg_.fivechain.xxx
    scr::ScrConfigCompat cfg;            // 兼容双名（沙盒源码混用）
    scr::FrameSourceCompat* src_ = nullptr;

    /// 261025 实时桥：读帧成功后逐帧元数据注入钩子（readerLoop 调；沙盒路径空）
    std::function<void(int idx, scr::CycleProductsCompat&)> onFrameMeta;

    // 07 侧标定注入（适配器装配期填——undistG/undistCpu 每帧参数用）
    cv::Mat K1_, D1_, K2_, D2_;
    int imageWidth_ = 0, imageHeight_ = 0;

    // 算子（适配器自持实例）
    calib::LaserMarkingSeparationCUDA* sepLaserL_ = nullptr;
    calib::LaserMarkingSeparationCUDA* sepLaserR_ = nullptr;
    calib::LaserCloudFuseCuda* voxelFuseOp_ = nullptr;

    // 灰度显存环
    std::vector<cv::cuda::GpuMat> d_grayLanesL_, d_grayLanesR_;
    void ensureGrayLanes(int slots) {
        if (static_cast<int>(d_grayLanesL_.size()) >= slots) return;
        d_grayLanesL_.resize(static_cast<size_t>(slots));
        d_grayLanesR_.resize(static_cast<size_t>(slots));
    }
    /// 预分配全部灰度环槽（执行器线程启动前——sep 帧 0 走 CUDA Graph 全局
    /// 捕获，期间任何线程 cudaMalloc 非法（capture Global 禁令）；上传线程
    /// 懒分配后续槽会撞捕获窗——沙盒靠 serialHead 串行头避开，07 直起须显式）
    void preallocGrayLanes(int slots, int rows, int cols) {
        ensureGrayLanes(slots);
        for (int i = 0; i < slots; ++i) {
            if (d_grayLanesL_[static_cast<size_t>(i)].empty())
                d_grayLanesL_[static_cast<size_t>(i)].create(rows, cols, CV_8UC1);
            if (d_grayLanesR_[static_cast<size_t>(i)].empty())
                d_grayLanesR_[static_cast<size_t>(i)].create(rows, cols, CV_8UC1);
        }
    }

    // 标志点链 per-lane 算子容器（fc_executor 建 mkLanes_ 份）
    struct MarkerOps {
        std::unique_ptr<calib::ImageSplitCPU> split;
        std::unique_ptr<calib::ZernikeEdgeCPU> zernike;
        std::unique_ptr<calib::ImageMergeCPU> merge;
        std::unique_ptr<calib::MarkerUndistortCPU> undist;
        std::unique_ptr<calib::EllipseFitCPU> ellipse;
        std::unique_ptr<calib::MarkerMatchCPU> match;
        std::unique_ptr<calib::EpipolarIntersectCPU> epi;
        std::unique_ptr<calib::EdgeMatchCPU> edge;
        std::unique_ptr<calib::PointReconstructCPU> recon;
        std::unique_ptr<calib::ConicReconstructCPU> conic;
    };
    std::vector<std::unique_ptr<MarkerOps>> markerLanes_;
    std::unique_ptr<MarkerOps> makeMarkerOps() {
        auto ops = std::make_unique<MarkerOps>();
        // 双模式（261025 对拍口径）：operators json 非空＝沙盒配方（fromJson 全
        // 参＋构造期矫正系/投影注入——对拍同参）；空＝产线配方（K/D 构造校验＋
        // 真机调参 y_tolerance 6.0/max_points 300）
        const bool sj = !cfg_.operators.empty();
        if (sj) {
            ops->split = std::make_unique<calib::ImageSplitCPU>(
                calib::ImageSplitCPUParams::fromJson(cfg_.opParams("image_split")));
            ops->zernike = std::make_unique<calib::ZernikeEdgeCPU>(
                calib::ZernikeEdgeCPUParams::fromJson(cfg_.opParams("zernike_edge")));
            ops->merge = std::make_unique<calib::ImageMergeCPU>(
                calib::ImageMergeCPUParams::fromJson(cfg_.opParams("image_merge")));
        } else {
            calib::ImageSplitCPUParams sp;
            sp.enableBoundaryCheck = true;
            ops->split = std::make_unique<calib::ImageSplitCPU>(sp);
            ops->zernike = std::make_unique<calib::ZernikeEdgeCPU>();
            ops->merge = std::make_unique<calib::ImageMergeCPU>();
        }
        if (!calib_.K1c.empty() && !calib_.Rext.empty()) {
            // 沙盒配方：fillMarkerUndistortParams(K1c/D1/K2c/D2/Rext/Tcomp/W/H)
            // ＋构造期 SetRectifyMatrices（R1/R2/P1/P2/Q 名义温档）
            calib::MarkerUndistortCPUParams up;
            up.fx1 = calib_.K1c.at<double>(0, 0); up.fy1 = calib_.K1c.at<double>(1, 1);
            up.cx1 = calib_.K1c.at<double>(0, 2); up.cy1 = calib_.K1c.at<double>(1, 2);
            up.fx2 = calib_.K2c.at<double>(0, 0); up.fy2 = calib_.K2c.at<double>(1, 1);
            up.cx2 = calib_.K2c.at<double>(0, 2); up.cy2 = calib_.K2c.at<double>(1, 2);
            const double* d1v = calib_.D1.ptr<double>(0);
            const double* d2v = calib_.D2.ptr<double>(0);
            up.k1_1 = d1v[0]; up.k2_1 = d1v[1]; up.p1_1 = d1v[2]; up.p2_1 = d1v[3]; up.k3_1 = d1v[4];
            up.k1_2 = d2v[0]; up.k2_2 = d2v[1]; up.p1_2 = d2v[2]; up.p2_2 = d2v[3]; up.k3_2 = d2v[4];
            for (int i = 0; i < 9; ++i) up.R[i] = calib_.Rext.at<double>(i / 3, i % 3);
            for (int i = 0; i < 3; ++i) up.T[i] = calib_.Tcomp.at<double>(i, 0);
            up.imageWidth = imageWidth_;
            up.imageHeight = imageHeight_;
            ops->undist = std::make_unique<calib::MarkerUndistortCPU>(up);
            ops->undist->SetRectifyMatrices(calib_.R1, calib_.R2, calib_.P1, calib_.P2,
                                            calib_.Q);
        } else if (!K1_.empty() && !K2_.empty()) {
            // 产线配方：K/D 构造（矫正系每帧 SetRectifyMatrices 注入）
            calib::MarkerUndistortCPUParams mp;
            auto atOr = [](const cv::Mat& m, int i) {
                return m.total() > static_cast<size_t>(i)
                    ? m.at<double>(i) : 0.0;
            };
            mp.fx1 = K1_.at<double>(0, 0); mp.fy1 = K1_.at<double>(1, 1);
            mp.cx1 = K1_.at<double>(0, 2); mp.cy1 = K1_.at<double>(1, 2);
            mp.fx2 = K2_.at<double>(0, 0); mp.fy2 = K2_.at<double>(1, 1);
            mp.cx2 = K2_.at<double>(0, 2); mp.cy2 = K2_.at<double>(1, 2);
            mp.k1_1 = atOr(D1_, 0); mp.k2_1 = atOr(D1_, 1);
            mp.p1_1 = atOr(D1_, 2); mp.p2_1 = atOr(D1_, 3); mp.k3_1 = atOr(D1_, 4);
            mp.k1_2 = atOr(D2_, 0); mp.k2_2 = atOr(D2_, 1);
            mp.p1_2 = atOr(D2_, 2); mp.p2_2 = atOr(D2_, 3); mp.k3_2 = atOr(D2_, 4);
            mp.imageWidth = imageWidth_;
            mp.imageHeight = imageHeight_;
            ops->undist = std::make_unique<calib::MarkerUndistortCPU>(mp);
        } else {
            ops->undist = std::make_unique<calib::MarkerUndistortCPU>();
        }
        if (sj) {
            ops->ellipse = std::make_unique<calib::EllipseFitCPU>(
                calib::EllipseFitCPUParams::fromJson(cfg_.opParams("ellipse_fit")));
            ops->match = std::make_unique<calib::MarkerMatchCPU>(
                calib::MarkerMatchCPUParams::fromJson(cfg_.opParams("marker_match")));
        } else {
            ops->ellipse = std::make_unique<calib::EllipseFitCPU>();
            calib::MarkerMatchCPUParams mmp;
            mmp.y_tolerance = 6.0f;
            mmp.max_points = 300;
            ops->match = std::make_unique<calib::MarkerMatchCPU>(mmp);
        }
        ops->epi = std::make_unique<calib::EpipolarIntersectCPU>();
        ops->edge = std::make_unique<calib::EdgeMatchCPU>();
        ops->recon = std::make_unique<calib::PointReconstructCPU>();
        ops->conic = std::make_unique<calib::ConicReconstructCPU>(
            sj ? calib::ConicReconstructCPUParams::fromJson(
                     cfg_.opParams("conic_reconstruct"))
               : calib::ConicReconstructCPUParams{});
        if (!calib_.P1.empty() && !calib_.P2.empty()) {
            // 沙盒口径：构造期投影注入（K1=P1 左 3×3/K2=P2 左 3×3/C2=−K2⁻¹·t2）
            // ——实时路径每帧 SetProjectionMatrices 覆盖（metaValid 时）
            cv::Matx33d cK1 = cv::Matx33d::eye(), cK2 = cv::Matx33d::eye();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    cK1(r, c) = calib_.P1.at<double>(r, c);
                    cK2(r, c) = calib_.P2.at<double>(r, c);
                }
            const cv::Vec3d cT2(calib_.P2.at<double>(0, 3), calib_.P2.at<double>(1, 3),
                                calib_.P2.at<double>(2, 3));
            const cv::Vec3d cC2 = -(cK2.inv() * cT2);
            ops->conic->SetProjectionMatrices(cK1, cK2, cC2);
        }
        return ops;
    }

    // 激光链前段/v3
    std::unique_ptr<calib::LaserMatchScanSplitGpuV3> splitGpu_;
    std::shared_ptr<const calib::LaserPlaneMapTempTable> tableOdd_, tableEven_;

    // 融合流指针（fc_executor 写入——fuse 共用）
    cv::cuda::Stream* fcFuseStreamPtr_ = nullptr;

    // 计时（沙盒 timeBegin/timeEnd——CycleProducts 版本，07 侧 noop）
    void timeBegin() {}
    void timeEnd(const std::string&, scr::CycleProductsCompat&, bool, bool) {}
    void setStepOk(scr::CycleProductsCompat& p, const std::string& step, bool ok) {
        (void)p; (void)step; (void)ok;
    }
    bool stepOk(const std::string&, bool def = true) const { return def; }
    bool stepOk(scr::CycleProductsCompat& p, const char* def = nullptr) {
        (void)p; (void)def; return true;
    }
    void fail(const std::string& step, scr::CycleProductsCompat& p, const std::string& msg) {
        (void)p;
        JMW_LOG_WARN("07-FiveChain", "[fivechain] step fail: {} — {}", step, msg);
    }

    // 激光链算子集（fc_executor 直接访问 h_.laserOps_——沙盒 Harness 同名）
    // 分眼实例（沙盒 stegerL_/stegerR_/flatL_/flatR_ 语义——flat 延迟模式的
    // 设备计数器是实例私有，两眼共用会混计数→interp 计数虚高→v3 匹配崩塌）
    struct LaserOps {
        std::unique_ptr<calib::StegerExtractorFast> stegerL_;
        std::unique_ptr<calib::StegerExtractorFast> stegerR_;
        std::unique_ptr<calib::FlatCenterExtractor> flatL_;
        std::unique_ptr<calib::FlatCenterExtractor> flatR_;
        // 双实例（沙盒 undistCudaL_/undistCudaR_ 语义）：离线＝构造期 L/R 参；
        // 实时＝每帧 SetParams（metaValid 分支）
        std::unique_ptr<calib::UndistortPointsCuda> undistGL_;
        std::unique_ptr<calib::UndistortPointsCuda> undistGR_;
        std::unique_ptr<calib::EpipolarInterpDualCuda> interpL_;
        std::unique_ptr<calib::EpipolarInterpDualCuda> interpR_;
    };
    LaserOps laserOps_;

    // 沙盒激光链前段提交/收尾（v3 前段延迟执行；签名＝沙盒 6 参数全）
    int submitLaserFront(LaserOps& L, scr::CycleProductsCompat& p, bool record,
                          bool tubeOdd, int laneIdx, cv::cuda::Stream& ls);
    void finishLaserFront(LaserOps& L, scr::CycleProductsCompat& p, bool record,
                          int frontKind);

    // 标志点链 CPU 运行（沙盒 runMarkerChainRuns——RLE 行程驱动 CCL CPU 版）
    void runMarkerChainRuns(scr::CycleProductsCompat& p, bool record,
                           std::unique_ptr<MarkerOps>& ops,
                           const std::vector<int>& rleL, int nL,
                           const std::vector<int>& rleR, int nR,
                           int cols, int rows);
    void runMarkerChainCore(scr::CycleProductsCompat& p,
                            std::unique_ptr<MarkerOps>& ops);

    // 配准（沙盒 runRegistration——2 参签名）
    void runRegistration(scr::CycleProductsCompat& p, bool record);

    // prevState（光流链帧间状态——fc_executor/配准共用）
    calib::PrevFrameState prevState_;
    // 光流配准算子（runRegistration 消费——构造期创建）
    std::unique_ptr<calib::MarkerOpticalFlowFuseCPU> flowOp_;

    // ========================================================================
    // 261025 对拍补全：配准三级降级状态机（沙盒 harness.h L317-348 同款——
    // 光流 → frame_fuse 兜底【快路径视野捞点/慢路径三角索引全库投票】→ hold）
    // ========================================================================
    calib::GlobalMarkerSet globalSet_;          // 全局标志点地图（transform 累积，globalId 去重）
    std::map<int, size_t> globalIndexById_;     // gid → 库下标
    int nextGid_ = 0;                           // 会话新点发号计数（光流/兜底共用）
    bool gridBoardRejected_ = false;            // 规则网格板检测锁存（裁决硬拒）
    std::vector<double> flowRmseBaseline_;      // 光流误差趋势门基线（前 50 成功帧）
    double flowRmseBaselineMed_ = -1.0;
    int holdStreak_ = 0;                        // 连续 hold 帧数（捞点半径自适应＋停手门）
    size_t libVersion_ = 1;                     // 库写入递增版本（索引脏标记）
    std::unique_ptr<scr::TriangleLibraryIndex> libIndex_;   // 慢路径全库三角索引
    size_t libIndexVersion_ = 0;                // 索引对应库版本
    int lastIndexRebuildCycle_ = -1;            // 上次索引重建帧号
    double lastNormalOpposeRatio_ = 0.0;        // 上帧法线反对占比（诊断）
    // 候选池（新点先观察，连续 3 次重观测一致才转正入图——防瞬态假点）
    struct CandidateEntry {
        cv::Point3d position;
        cv::Vec3d normal;
        int count = 1;
    };
    std::vector<CandidateEntry> candidates_;
    static constexpr int kPromoteThreshold = 3;         // 转正发号计数 ≥3 次
    static constexpr double kCandidateMergeMm = 2.0;    // 候选匹配半径（同合并半径）
    // 兜底算子＋参数（frame_fuse——json 装配；生产实时亦建兜底＝断链自愈）
    calib::FrameFuseCPUParams fuseParams_;
    std::unique_ptr<calib::FrameFuseCPU> fuseOp_;
    // CCL 门限（沙盒 config ccl 节默认：800/30000/0——runMarkerChainRuns 消费）
    int cclMinArea_ = 800, cclMaxArea_ = 30000, cclTopX_ = 0;

    // —— 261027 真机诊断：标志点链/配准逐级计数（stopRealtime 汇总日志）——
    std::atomic<long> diagCclFail_{0};      // CCL 失败/无组件帧
    std::atomic<long> diagRoisL_{0};        // CCL 组件数累计（L）
    std::atomic<long> diagSplit_{0};        // split 成功帧
    std::atomic<long> diagEllipse_{0};     // ellipse 成功帧（detected>0）
    std::atomic<long> diagMatchPairs_{0};  // 双目匹配对累计
    std::atomic<long> diagConicValid_{0};  // conic 有效圆累计
    std::atomic<long> diagRegFirst_{0}, diagRegFlow_{0}, diagRegFuse_{0};
    std::atomic<long> diagRegHold_{0}, diagRegSkip_{0};
    std::atomic<long> diagLaserPts_{0};    // 激光重建点累计（chain2 产出）

    // 配准链方法（沙盒 harness.cpp L1269-1726 逐段对齐）
    void adoptRegistration(calib::MarkerOpticalFlowFuseCPUResult& res,
                           scr::CycleProductsCompat& p);
    std::vector<int> cullByView(const calib::MarkerPointSet& cur, int maxCount) const;
    calib::FrameFuseCPUResult fuseExecute(const calib::MarkerPointSet& cur,
                                          const std::vector<int>& libIdx) const;
    bool acceptFuse(const calib::FrameFuseCPUResult& r) const;
    double checkNormalPlausibility(const std::vector<cv::Vec3d>& normals) const;
    bool mergeNormalCompatible(const cv::Vec3d& a, const cv::Vec3d& b) const;

    // 轨迹/计数
    std::vector<std::array<double,3>> trajT_;
    std::atomic<uint64_t> cycleCount_{0};

    // 标定数据（沙盒 Harness.calib_——Q 供 CPU 重建消费；其余＝对拍补全的
    // 沙盒装配件：undistCpu 每帧 SetRectifyMatrices 基线＋Rext/Tcomp 构造参数）
    struct CalibDataCompat {
        cv::Mat Q;
        cv::Mat R1, R2, P1, P2;        // 名义温档矫正系（沙盒模式构造期注入）
        cv::Mat K1c, K2c, D1, D2;      // 补偿内参（沙盒模式 undistCpu 构造）
        cv::Mat Rext, Tcomp;           // 外参（沙盒 fillMarkerUndistortParams）
    };
    CalibDataCompat calib_;

    // 融合计数
    std::atomic<uint64_t> fuseFrameCount_{0};
    std::atomic<uint64_t> fuseInputTotal_{0};
    std::atomic<uint64_t> fuseOverloadCount_{0};
    std::atomic<uint64_t> fuseRehashCount_{0};
    std::atomic<bool> fcActive_{false};

    // 输出
    std::string outDir_;
    bool prodProfile_ = true;    // 07 产线模式（跳过沙盒调试记账）

private:
    FiveChainAdapter* adapter_ = nullptr;
};

} // namespace Scanner::pipeline
