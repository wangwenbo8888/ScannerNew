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

// —— 261027 名义温档标定装载（camera_calib.json——与产表/离线验证同源）——
namespace {
cv::Mat matFromJson22(const nlohmann::json& j, int rows, int cols) {
    cv::Mat m(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = j.at(static_cast<size_t>(r))
                                     .at(static_cast<size_t>(c)).get<double>();
    return m;
}
} // namespace

// ============================================================================
// 装配注入
// ============================================================================
ScanPipeline::ScanPipeline(ScanConfig cfg)
    : cfg_(std::move(cfg)),
      obs_(static_cast<size_t>(cfg_.laserCacheBudgetMB) * (size_t{1} << 20)) {}

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
            fcCfg.laserEnabled = cfg_.enableLaser;   // 标点模式=false：跳过激光链（防崩溃）
            fcCfg.testFrameGate = cfg_.testFrameGate; // 261029 测试回放帧门（app 注入；可空=恒开）
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
        // 名义温档立体标定（config/camera_calib.json——产表同源；env
        // JMW_FC_CAMJSON 可指定路径；缺文件回退逐帧快照注入并 WARN）
        {
            const std::string camPath = getenv("JMW_FC_CAMJSON") != nullptr
                ? std::string(getenv("JMW_FC_CAMJSON"))
                : "config/camera_calib.json";
            std::ifstream cf(camPath);
            if (cf) {
                try {
                    nlohmann::json j;
                    cf >> j;
                    const auto& table = j.at("stereoRectifyTempTable").at("table");
                    // 就近 22.5℃（标定参考温）——与离线验证同档
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
                    if (tier != nullptr) {
                        fcAdapter_->attachStereoNominal(
                            matFromJson22(tier->at("R1"), 3, 3),
                            matFromJson22(tier->at("R2"), 3, 3),
                            matFromJson22(tier->at("P1"), 3, 4),
                            matFromJson22(tier->at("P2"), 3, 4),
                            matFromJson22(tier->at("Q"), 4, 4),
                            matFromJson22(tier->at("compensatedCameraMatrixL"), 3, 3),
                            matFromJson22(tier->at("compensatedCameraMatrixR"), 3, 3),
                            matFromJson22(j.at("intrinsic").at("left").at("dist_coeffs"), 1, 5),
                            matFromJson22(j.at("intrinsic").at("right").at("dist_coeffs"), 1, 5),
                            matFromJson22(j.at("extrinsic").at("R"), 3, 3),
                            matFromJson22(j.at("extrinsic").at("T"), 3, 1));
                        JMW_LOG_INFO("07-ScanPipeline",
                            "[ScanPipeline] 五链名义温档标定装载: {}（tier={}℃）",
                            camPath, bestT);
                    }
                } catch (const std::exception& e) {
                    JMW_LOG_WARN("07-ScanPipeline",
                        "[ScanPipeline] 名义温档标定解析失败（回退快照注入）: {}", e.what());
                }
            } else {
                JMW_LOG_WARN("07-ScanPipeline",
                    "[ScanPipeline] camera_calib.json 缺失（{}）——回退逐帧快照注入",
                    camPath);
            }
        }
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
