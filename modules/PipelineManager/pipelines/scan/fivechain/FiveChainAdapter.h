// FiveChainAdapter.h — 沙盒五链执行器 → 07 ScanPipeline 适配层
//
// 职责：为 fc_executor（沙盒五链执行器，本体不改）提供与沙盒 Harness 等价的
// 运行环境——自持算子实例/灰度显存环/融合算子/配置，不经 ScanChains 的
// per-lane 独占实例（五链执行器自己管生命周期）。
//
// 261025 Phase 2 产线集成
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include "pipelines/scan/ScanTypes.h"
#include "pipelines/scan/FrameObsAccumulator.h"

namespace Scanner::data { class ICloudWarehouse; }

namespace calib {
class LaserMarkingSeparationCUDA;
class LaserCloudFuseCuda;
class LaserMatchScanSplitGpuV3;
class LaserReconstructCPU;
}

namespace Scanner::pipeline {

class ISceneFeed;

/// 五链配置（沙盒 ScrFiveChain 的 07 映射——值域与沙盒 config_flat.json 一致）
struct FiveChainConfig {
    int graySlots = 8;
    int rleSlots = 8;
    int rleCap = 65536;
    int rleWindow = 4096;
    int markerLanes = 3;
    int bCap = 8, cCap = 8;
    int dQueueCap = 8, dPoolSlots = 4;
    int eQueueCap = 8, ePoolBlocks = 9, eMaxPoints = 34237;
    int fCap = 8, trCap = 8;
    int maxInflight = 0;
    int memCapMB = 512;
    long long voxelReserve = 12000000;
    double voxelSizeMm = 0.25;
    double voxelLoadFactor = 0.75;
    long long voxelGrowSlots = 2000000;
    std::string pinRoles = "1a,1b,marker,ch2";
    bool priorityBoost = true;
    bool splitChain1 = true;
    int lmaskSlots = 4;
    int holdSkip = 0;
    int slowChainMs = 0;
    std::string failInject;
    int kRleCapOverride = 0;
    int eBlockSizePct = 100;
    int regWakeP99UsMax = 1000;
    int dbgInterval = 32;
    double pushBudgetMs = 0.8;
};

/// 五链适配器——fc_executor 的算子宿主＋07 观测/融合/推送桥
class FiveChainAdapter {
public:
    FiveChainAdapter(const FiveChainConfig& cfg,
                    FrameObsAccumulator& obs,
                    Scanner::data::ICloudWarehouse* warehouse,
                    ISceneFeed* sceneFeed);
    ~FiveChainAdapter();

    Scanner::Result start();
    void stop();
    bool isActive() const { return active_.load(); }

    // —— fc_executor/harness 兼容口 ——
    calib::LaserMarkingSeparationCUDA& sepL() { return *sepL_; }
    calib::LaserMarkingSeparationCUDA& sepR() { return *sepR_; }
    calib::LaserCloudFuseCuda& fuse() { return *fuse_; }
    cv::cuda::GpuMat& grayL(int slot) { return grayLanesL_[static_cast<size_t>(slot)]; }
    cv::cuda::GpuMat& grayR(int slot) { return grayLanesR_[static_cast<size_t>(slot)]; }
    void ensureGrayLanes(int slots);
    const FiveChainConfig& cfg() const { return cfg_; }

    /// 观测落账（07 GBA 链）
    FrameObsAccumulator& obs() { return obs_; }

    /// 帧完成——obs 落账＋融合消费＋渲染推送
    void onFrameDone(uint64_t frameId, const double R[9], const double T[3],
                     const std::vector<MarkerObs>& markers,
                     const std::vector<float>& laserXyz);

private:
    FiveChainConfig cfg_;
    FrameObsAccumulator& obs_;
    Scanner::data::ICloudWarehouse* warehouse_;
    ISceneFeed* sceneFeed_;
    std::atomic<bool> active_{false};

    std::unique_ptr<calib::LaserMarkingSeparationCUDA> sepL_, sepR_;
    std::unique_ptr<calib::LaserCloudFuseCuda> fuse_;
    std::vector<cv::cuda::GpuMat> grayLanesL_, grayLanesR_;
};

} // namespace Scanner::pipeline
