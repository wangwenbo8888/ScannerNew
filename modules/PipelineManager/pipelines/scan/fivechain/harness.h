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
    std::vector<FrameCycleInfo> cycles;
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
    std::map<std::string, std::string> stepError_;
    // 配准诊断
    int flowMatchedDiag = -1;
    double flowRmseDiag = -1.0;
    std::string flowMessageDiag;
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
        ops->split = std::make_unique<calib::ImageSplitCPU>();
        ops->zernike = std::make_unique<calib::ZernikeEdgeCPU>();
        ops->merge = std::make_unique<calib::ImageMergeCPU>();
        ops->undist = std::make_unique<calib::MarkerUndistortCPU>();
        ops->ellipse = std::make_unique<calib::EllipseFitCPU>();
        ops->match = std::make_unique<calib::MarkerMatchCPU>();
        ops->epi = std::make_unique<calib::EpipolarIntersectCPU>();
        ops->edge = std::make_unique<calib::EdgeMatchCPU>();
        ops->recon = std::make_unique<calib::PointReconstructCPU>();
        ops->conic = std::make_unique<calib::ConicReconstructCPU>();
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
    struct LaserOps {
        std::unique_ptr<calib::StegerExtractorFast> stegerFast;
        std::unique_ptr<calib::FlatCenterExtractor> flatCenter;
        std::unique_ptr<calib::UndistortPointsCuda> undistG;
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

    // 轨迹/计数
    std::vector<std::array<double,3>> trajT_;
    std::atomic<uint64_t> cycleCount_{0};

    // 标定数据（沙盒 Harness.calib_——Q 矩阵供 CPU 重建消费）
    struct CalibDataCompat {
        cv::Mat Q;
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
