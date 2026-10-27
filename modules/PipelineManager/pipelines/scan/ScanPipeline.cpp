// ScanPipeline.cpp — C 扫描流水线（261027 重构：五链独占——旧 lane 退役）
//
// 全链：06 SlotRing（相机帧）→ RtBridge（居中裁剪 2048×1536→1688×1400＋
// 帧温快照元数据）→ FiveChainExecutor（新算子五链）→ obs（GBA）＋仓库直写＋
// 点云显示（ISceneFeed 节流推送）。CMTT 曲线表与算子参数自 config/ 装载。
#include "pipelines/scan/ScanPipeline.h"

#include "pipelines/scan/fivechain/FiveChainAdapter.h"
#include "scanning/laser/laser_match_scan_v3/cmtt_table_loader.h"   // 五链 v3 CMTT 曲线表装载

#include <chrono>
#include <cstdlib>
#include <fstream>

#include "jmw_logging.h"

namespace Scanner::pipeline {

// ============================================================================
// 装配注入
// ============================================================================
ScanPipeline::ScanPipeline(ScanConfig cfg) : cfg_(std::move(cfg)) {}

ScanPipeline::~ScanPipeline() {
    stop();                                        // 安全网：幂等
    delete fcAdapter_;
    fcAdapter_ = nullptr;
}

void ScanPipeline::attachRing(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring,
                              size_t dropThreshold) {
    ring_ = &ring;
    dropThreshold_ = dropThreshold;
}

void ScanPipeline::attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                               const cv::Mat& D2, int imageWidth, int imageHeight,
                               std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable) {
    K1_ = K1.clone(); D1_ = D1.clone();
    K2_ = K2.clone(); D2_ = D2.clone();
    imageWidth_ = imageWidth; imageHeight_ = imageHeight;
    laserTable_ = std::move(laserTable);
}

void ScanPipeline::attachSimSource(SimScanSource* src) {
    // 退役桩（02 兼容）：五链模式无中段模拟注入
    if (src != nullptr)
        JMW_LOG_WARN("07-ScanPipeline", "[ScanPipeline] attachSimSource 已退役（五链模式忽略）");
}

FrameObsAccumulator& ScanPipeline::obs() { return obs_; }

// ============================================================================
// configure — 会话边界装配（仓库/显示口＋事件 sink）
// ============================================================================
Scanner::Result ScanPipeline::configure(const PipelineDeps& deps) {
    if (state_.load() != State::Idle)
        return Result::fail("ScanPipeline::configure: 会话私有件，须新对象");
    cloudWarehouse_ = deps.cloudWarehouse;        // 仓库直写口（可空）
    sceneFeed_ = deps.sceneFeed;                  // 点云显示口（可空）
    sink_ = std::make_unique<EventBusEventSink>(nullptr);   // 事件上报（no-op 底座）
    state_.store(State::Configured);
    return Result::ok();
}

// ============================================================================
// start — 五链独占：CMTT 曲线表＋算子参数装载 → 适配器 → 实时链
// ============================================================================
Scanner::Result ScanPipeline::start() {
    switch (state_.load()) {
        case State::Idle:     return Result::fail("ScanPipeline::start: 须先 configure");
        case State::Running:  return Result::fail("ScanPipeline::start: 已在运行");
        case State::Paused:   return Result::fail("ScanPipeline::start: 暂停态请用 resume()");
        case State::Faulted:  return Result::fail("ScanPipeline::start: 已异常停（会话私有件）");
        case State::Stopped:  return Result::fail("ScanPipeline::start: 已停止（会话私有件）");
        case State::Configured: break;
    }
    if (!ring_) return Result::fail("ScanPipeline::start: 未 attachRing（输入源必备）");

    if (!fcAdapter_) {
        // 惰性装配（首 start 建）
        FiveChainConfig fcCfg;              // 沙盒 config_flat.json 值域默认
        // 体素下限 0.25mm（五链验证口径；UI 更细档（如 0.01）爆显存/拖垮融合）
        if (cfg_.voxelSizeMm >= 0.25) {
            fcCfg.voxelSizeMm = cfg_.voxelSizeMm;
        } else if (cfg_.voxelSizeMm > 0) {
            JMW_LOG_WARN("07-ScanPipeline",
                "[ScanPipeline] 五链体素档 {}mm 低于下限——用 0.25mm（验证口径）",
                cfg_.voxelSizeMm);
        }
        // 算子参数源（离线验证同参）：config/fivechain_operators.json
        //（env JMW_FC_OPSJSON 可指定路径；缺文件用产线默认参并 WARN）
        if (!fcOpsLoaded_) {
            const std::string opsPath = getenv("JMW_FC_OPSJSON") != nullptr
                ? std::string(getenv("JMW_FC_OPSJSON"))
                : "config/fivechain_operators.json";
            std::ifstream f(opsPath);
            if (f) {
                try {
                    nlohmann::json j;
                    f >> j;
                    if (j.contains("operators")) {
                        fcOpsJson_ = j.at("operators");
                        fcOpsLoaded_ = true;
                        JMW_LOG_INFO("07-ScanPipeline",
                            "[ScanPipeline] 五链算子参数装载: {}（离线验证同参）", opsPath);
                    }
                } catch (const std::exception& e) {
                    JMW_LOG_WARN("07-ScanPipeline",
                        "[ScanPipeline] 五链算子参数解析失败（用默认）: {}", e.what());
                }
            } else {
                JMW_LOG_WARN("07-ScanPipeline",
                    "[ScanPipeline] 五链算子参数缺失（{}）——用产线默认参", opsPath);
            }
        }
        fcAdapter_ = new FiveChainAdapter(fcCfg, obs_, cloudWarehouse_, sceneFeed_,
                                          fcOpsLoaded_ ? &fcOpsJson_ : nullptr);
        // v3 真表匹配需 CMTT 曲线表（leftToRightMap）。生产 laser_calib.json
        // planeMapTempTable 无此数据——CMTT sidecar 部署于
        // config/curve_map_temp_table.bin（标定产物不入 git；env JMW_FC_CMTT
        // 可指定路径）。缺表回退生产表＝v3 无表可查（WARN）
        std::shared_ptr<const calib::LaserPlaneMapTempTable> fcTable;
        {
            const std::string cmttPath = getenv("JMW_FC_CMTT") != nullptr
                ? std::string(getenv("JMW_FC_CMTT"))
                : "config/curve_map_temp_table.bin";
            std::string lerr;
            auto t = calib::LoadTempTableFromCmtt(cmttPath, 22.5, lerr);
            if (t) {
                fcTable = t;
                JMW_LOG_INFO("07-ScanPipeline",
                    "[ScanPipeline] 五链 CMTT 曲线表装载: {}（referenceTemp={:.1f}℃）",
                    cmttPath, t->referenceTemperature);
            } else {
                JMW_LOG_WARN("07-ScanPipeline",
                    "[ScanPipeline] 五链 CMTT 曲线表缺失（{}: {}）——回退生产 "
                    "planeMap 表（无 leftToRightMap，v3 激光匹配将无表可查！）",
                    cmttPath, lerr);
            }
        }
        fcAdapter_->attachCalib(K1_, D1_, K2_, D2_, imageWidth_, imageHeight_,
                                fcTable ? fcTable : laserTable_);
    }
    // 仓库会话边界：既有内容折入基线前缀
    if (cloudWarehouse_) cloudWarehouse_->beginCloudSession();
    auto rs = fcAdapter_->startRealtime(*ring_, dropThreshold_);
    if (!rs.success) {
        delete fcAdapter_;
        fcAdapter_ = nullptr;
        return Result::fail("ScanPipeline::start: 五链实时链启动失败: " + rs.message);
    }
    fcActive_ = true;
    state_.store(State::Running);
    JMW_LOG_INFO("07-ScanPipeline",
        "[ScanPipeline] 五链会话启动: voxel={}mm dropThreshold={} seedMarkers={} "
        "（旧 lane 已退役；v1 限制：无 pause/检查点/编辑账本）",
        cfg_.voxelSizeMm >= 0.25 ? cfg_.voxelSizeMm : 0.25, dropThreshold_,
        cfg_.existingMarkers.size());
    return Result::ok();
}

// ============================================================================
// stop — 五链收尾：桥停→执行器排水→收割 join→终局全量推送→清环
// ============================================================================
void ScanPipeline::stop() {
    if (state_.load() == State::Idle) return;
    if (fcActive_) {
        if (fcAdapter_) fcAdapter_->stopRealtime();
        fcActive_ = false;
        state_.store(State::Stopped);
        if (ring_) ring_->resetForNewSession();   // 下会话从 0 抓
        return;
    }
    state_.store(State::Stopped);
    if (ring_) ring_->resetForNewSession();
}

bool ScanPipeline::isRunning() const {
    return state_.load() == State::Running;
}

// ============================================================================
// pause / resume / recover — 五链 v1 无暂停语义（pause 即停止）
// ============================================================================
void ScanPipeline::pause() {
    if (state_.load() != State::Running) return;
    JMW_LOG_WARN("07-ScanPipeline",
        "[ScanPipeline] pause: 五链无暂停态——转为停止（已融合数据保留，续扫请新会话）");
    stop();
}

Scanner::Result ScanPipeline::resume() {
    return Result::fail("ScanPipeline::resume: 五链无暂停态（续扫请新会话 start）");
}

Scanner::Result ScanPipeline::recover() {
    return Result::fail("ScanPipeline::recover: 原地恢复已随旧 lane 退役（重建会话）");
}

// —— 旧 lane 遗留接口桩（02 兼容）——
void ScanPipeline::setDepthOfField(int mode) {
    (void)mode;
    JMW_LOG_WARN("07-ScanPipeline",
        "[ScanPipeline] setDepthOfField 已退役（旧固定视差补偿件——五链 v3 真表匹配无需）");
}
void ScanPipeline::setDofRange(double nMin, double nMax, double fMin, double fMax) {
    (void)nMin; (void)nMax; (void)fMin; (void)fMax;   // 同上——no-op
}
uint64_t ScanPipeline::consumedFrames() const { return 0; }
uint64_t ScanPipeline::droppedFrames() const { return 0; }

} // namespace Scanner::pipeline
