// FiveChainAdapter.cpp — 适配层实现（自持算子宿主＋07 桥）＋FcHarnessCompat 方法实现
#include "FiveChainAdapter.h"
#include "harness.h"

#include "pipelines/ISceneFeed.h"
#include "ICloudWarehouse.h"
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

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
// FcHarnessCompat 方法实现（沙盒 harness.cpp 中 fc_executor 消费的段落——
// 从沙盒 harness.cpp 提取的算法体，仅适配 CycleProductsCompat 结构）
// ============================================================================

bool FcHarnessCompat::submitLaserFront(LaserOps& L, scr::CycleProductsCompat& p,
                                        bool record, bool tubeOdd, int slot,
                                        cv::cuda::Stream& s) {
    // 沙盒 harness.cpp submitLaserFront 等价——steger_fast/flat/undist/interp
    // + v3 前段。此处在 Phase 2 首版先返回 0（由 chain1b 的直接路径消费），
    // 后续五链全量接通时填完整实现
    (void)L; (void)p; (void)record; (void)tubeOdd; (void)slot; (void)s;
    return 0;
}

void FcHarnessCompat::finishLaserFront(LaserOps& L, scr::CycleProductsCompat& p,
                                        bool record, int frontKind) {
    (void)L; (void)p; (void)record; (void)frontKind;
}

void FcHarnessCompat::runMarkerChainRuns(scr::CycleProductsCompat& p, bool record,
                                          std::unique_ptr<MarkerOps>& ops,
                                          const std::vector<int>& rleL, int nL,
                                          const std::vector<int>& rleR, int nR,
                                          int cols, int rows) {
    // 沙盒 harness.cpp runMarkerChainRuns 等价——RLE 行程驱动 CPU 标志点链
    // Phase 2 首版：占位（五链执行器初始化即走，标志点链待后续接入）
    (void)p; (void)record; (void)ops; (void)rleL; (void)nL; (void)rleR; (void)nR;
    (void)cols; (void)rows;
}

void FcHarnessCompat::runRegistration(scr::CycleProductsCompat& p, bool record) {
    // 沙盒 harness.cpp runRegistration 等价——光流配准
    // Phase 2 首版：占位（配准链待后续接入）
    (void)p; (void)record;
    p.regSource = "hold";
}

} // namespace Scanner::pipeline
