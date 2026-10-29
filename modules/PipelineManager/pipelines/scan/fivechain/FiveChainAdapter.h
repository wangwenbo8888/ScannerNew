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
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <deque>

#include "pipelines/scan/ScanTypes.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "core/common/calib_result_types.h"   // calib::LaserPlaneMapTempTable（struct 真身——防 class 误前置致名修饰分裂）
#include <nlohmann/json.hpp>

namespace Scanner::data {
class ICloudWarehouse;
template<typename T> class SlotRing;
struct EnhancedFrame;
}

namespace calib {
class LaserMarkingSeparationCUDA;
class LaserCloudFuseCuda;
class LaserMatchScanSplitGpuV3;
class LaserReconstructCPU;
}

namespace scr { namespace fc { class FiveChainExecutor; } }

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
    // 261028 标点模式（A 模式）：false＝不创建激光链算子（submitLaserFront
    // 直接返回 0——从根源杜绝空激光数据在全链路上的级联崩溃）
    bool laserEnabled = true;
    // 261029 测试回放帧门（app 注入 dm->isCapturing()——与在线帧流同语义：
    // 模式键只 arm，设备 M 键启停控制回放起停；空=恒开（离线直跑/无设备））
    std::function<bool()> testFrameGate;
    // 261025 算法几何口径：全部标定表（K/D/P1/P2/Q/激光平面表/CMTT）为 1688×1400
    // 几何（laser-mark1688 标定链产物）；采集 2048×1536（camera.json——居中 ROI）。
    // 五链入口居中裁剪到算法几何。env JMW_FC_W/JMW_FC_H 可覆盖（=0 关闭裁剪直通）
    int algoWidth = 1688;
    int algoHeight = 1400;
};

/// 五链适配器——fc_executor 的算子宿主＋07 观测/融合/推送桥
class FiveChainAdapter {
public:
    /// opJson 非空＝沙盒对拍口径（算子参数从 config operators 节构造——
    /// mask_separation/fuse 等；空＝产线默认参）
    FiveChainAdapter(const FiveChainConfig& cfg,
                    FrameObsAccumulator& obs,
                    Scanner::data::ICloudWarehouse* warehouse,
                    ISceneFeed* sceneFeed,
                    const nlohmann::json* opJson = nullptr);
    ~FiveChainAdapter();   // 安全网：stopRealtime()

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

    // ========================================================================
    // 261025 实时链（Phase 2 收尾）：SlotRing→五链执行器→obs/仓库/渲染
    // ========================================================================
    /// 标定注入（ScanPipeline configure 期调用；温表=生产 laserTable 单表）
    void attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                     const cv::Mat& D2, int imageWidth, int imageHeight,
                     std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable);

    /// 261027 名义温档立体标定注入（config/camera_calib.json——与产表/离线
    /// 验证同源；启用后 harness 用构造期标定，逐帧快照注入关闭）
    void attachStereoNominal(const cv::Mat& R1, const cv::Mat& R2,
                             const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& Q,
                             const cv::Mat& K1c, const cv::Mat& K2c,
                             const cv::Mat& D1, const cv::Mat& D2,
                             const cv::Mat& Rext, const cv::Mat& Tcomp);

    /// 实时启动：桥接 SlotRing（Overwrite 抓最新面孔）→ 五链执行器（INT_MAX
    /// 无界哨兵）→ 收割线程（obs 落账＋节流渲染推送＋仓库直写）
    /// 261029 测试数据回放模式（config/fc_testdata.json enabled+dir 为正源；
    /// env JMW_FC_TESTDATA 优先覆盖）——绕过 SlotRing 直读帧文件走完整在线
    /// 管线。加载失败＝fail（拒绝静默回退在线数据）
    Scanner::Result loadTestData(const std::string& dir);

    Scanner::Result startRealtime(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring,
                                  size_t dropThreshold);

    /// 实时停止：桥停→执行器排水析构→收割线程 join→终局全量推送
    void stopRealtime();

    bool realtimeActive() const { return rtActive_.load(); }

private:
    /// 实时桥：fc readFrame/cycleAt ← SlotRing 抓最新；逐帧元数据环形表
    struct RtBridge;
    /// 收割线程：collect 逐帧→obs/notifyLaserCount/节流渲染
    void collectorLoop();
    /// 渲染推送（时间门 ≥3s；forceFull＝终局全量——仓库+UI 收口）
    void renderPushIfDue(bool forceFull);

    FiveChainConfig cfg_;
    FrameObsAccumulator& obs_;
    Scanner::data::ICloudWarehouse* warehouse_;
    ISceneFeed* sceneFeed_;
    std::atomic<bool> active_{false};

    std::unique_ptr<calib::LaserMarkingSeparationCUDA> sepL_, sepR_;
    std::unique_ptr<calib::LaserCloudFuseCuda> fuse_;
    std::vector<cv::cuda::GpuMat> grayLanesL_, grayLanesR_;

    // —— 实时链组件（startRealtime 建/stopRealtime 拆）——
    cv::Mat rtK1_, rtD1_, rtK2_, rtD2_;
    int rtW_ = 0, rtH_ = 0;
    std::shared_ptr<const calib::LaserPlaneMapTempTable> rtLaserTable_;
    bool hasStereoNominal_ = false;               // 名义温档标定已注入
    cv::Mat snR1_, snR2_, snP1_, snP2_, snQ_, snK1c_, snK2c_, snD1_, snD2_;
    cv::Mat snRext_, snTcomp_;
    // config/camera_calib.json 就近 22.5℃ 温档 → attachStereoNominal（测试
    // 数据模式前置——快照注入关闭后名义标定为唯一标定源）
    bool ensureStereoNominalFromConfig();
    std::unique_ptr<RtBridge> bridge_;
    class FcHarnessCompat* harness_ = nullptr;              // 五链 harness（自有）
    std::unique_ptr<scr::fc::FiveChainExecutor> executor_;  // 五链执行器（INT_MAX 无界）
    std::thread collector_;
    std::atomic<bool> rtActive_{false};
    const nlohmann::json* opJson_ = nullptr;                // 算子参数源（宿主保活——startRealtime 全链应用）

    // 渲染节流/推送状态（collector 线程私有——镜像 FuseConsumer 口径）
    std::chrono::steady_clock::time_point lastRenderPush_{};
    size_t lastPushedLaserPts_ = 0;
    uint64_t pushSeq_ = 0;
    std::vector<float> renderKeep_;                          // CloudViewHandle 指针宿主（活到下次推送）
    cv::cuda::Stream dlStream_;                              // 融合云 D2H 下载流
    // 261028 实时标志点云（扫描中边扫边出——conic 有效圆去重累积）
    std::vector<calib::MarkerCloudPoint> markerCloud_;       // hostMarker 指针宿主
    // 标志点融合算子（体素 5mm＋EMA 精化——旧 lane MarkerCloudFuseCPU 同款）
    std::unique_ptr<calib::MarkerCloudFuseCPU> markerFuseOp_;
};

} // namespace Scanner::pipeline
