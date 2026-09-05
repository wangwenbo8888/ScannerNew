// ============================================================================
// ScanWorkflow.cpp — 扫描工作流实现（编排/记账壳；帧处理移交 07 ScanPipeline）
//
// P6-T29a：旧 scanLoop 五 Stage（CPU 手搓掩码/标记点链/激光空壳/CPU 融合）
// 整体退役——单帧处理由 07 ScanPipeline 内部并行调度（E 核抓帧→GPU 链→
// P 核标记点链→融合消费线程）完成；本类仅做装配、生命周期与会话记账。
// ============================================================================

#include "ScanWorkflow.h"

#include "CalibrationRepository.h"
#include "PointCloudBuffer.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/scan/ScanPipeline.h"

#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include <chrono>
#include <utility>

namespace Scanner::workflow {

// ============================================================================
// ScanWorkflow
// ============================================================================
ScanWorkflow::ScanWorkflow(WorkflowContext* ctx) : ctx_(ctx) {}

ScanWorkflow::~ScanWorkflow() { stop(); }

void ScanWorkflow::setCalibration(const ScanCalibration& calib) {
    calib_ = calib;
}

Result ScanWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");
    JMW_LOG_INFO("02-ScanWorkflow", "[ScanWorkflow] 初始化 (模式={}, 标定={})",
                 scanMode_ == ScanMode::MarkerOnly ? "纯标记点" : "标记点+激光",
                 calib_.valid ? "已加载" : "未加载");
    // 出口查表（逐温档 K/D）经 session_.assemble 并行生效（EnhancedFrame.snapshot
    // 供算子）；attachCalib 静态 K/D 仍供 GPU 链初始化——两者数据源同为标定仓库
    return Result::ok();
}

Result ScanWorkflow::assemblePipeline() {
    namespace sp = Scanner::pipeline;

    // 出口查表装表（06）。A 模式（纯标记点）两表皆空会被此处挡下——裁决：app
    // pre 谓词已挡无参启动，能走到这里必有过参标定（两表至少一档），照 fail 返回
    const auto* repo = ctx_ ? ctx_->calibRepo() : nullptr;
    if (!repo) return Result::fail("出口查表装表失败：无标定仓库");
    if (const auto tblR = session_.assemble(*repo); !tblR.success)
        return Result::fail("出口查表装表失败: " + tblR.message);

    sp::ScanConfig cfg;
    cfg.enableLaser = (scanMode_ == ScanMode::MarkerPlusLaser);

    // 续扫基准注入：点云仓库快照 → 09 MarkerCloudPoint（globalId 语义在 obs 层，
    // 07 按下标 0..n-1 对接 hpGlobalIds——见 ScanPipeline.h seed 时序）
    uint64_t markerVer = 0;
    std::vector<Scanner::data::MarkerRecord> markerRecs;
    if (ctx_ && ctx_->pointCloudBuffer())
        ctx_->pointCloudBuffer()->snapshotMarkers(markerVer, markerRecs);
    if (!markerRecs.empty()) {
        cfg.existingMarkers.reserve(markerRecs.size());
        for (const auto& r : markerRecs)
            cfg.existingMarkers.push_back(
                calib::MarkerCloudPoint{r.pos.x, r.pos.y, r.pos.z,
                                        r.normal[0], r.normal[1], r.normal[2]});
        JMW_LOG_INFO("02-ScanWorkflow", "[ScanWorkflow] 续扫基准 {} 点", markerRecs.size());
    } else {
        JMW_LOG_WARN("02-ScanWorkflow", "[ScanWorkflow] 无续扫基准（新扫描或上次未留存）");
    }
    pipeline_ = std::make_unique<sp::ScanPipeline>(cfg);

    // 帧响应接线（08 采集回调→session 环）：app 双投递经 pushSessionFrame 注入
    pipeline_->attachRing(session_.ring(), /*dropThreshold=*/0);   // 0=自动（2*lanes）

    if (!calib_.valid || calib_.cameraMatrixL.empty() || calib_.cameraMatrixR.empty()) {
        pipeline_.reset();
        return Result::fail("无标定参数——07 生产链须 attachCalib");
    }

    // —— 激光温度表注入（§8-1 消费侧就位；生产者缺口见下）——
    // 消费契约：laser_match_scan 吃 mapData:[[xL,yL,uR,lineId],...]（LaserPlaneMap
    // 布局）。档表含 mapData 的档逐条装表；全档无 mapData（当前工厂档现状：仅
    // 温补相机参数）→ 表空＝B 模式降级 A（面片模式待算子侧补 mapData 生产者）
    std::shared_ptr<calib::LaserPlaneMapTempTable> laserTable;
    if (scanMode_ == ScanMode::MarkerPlusLaser) {
        laserTable = buildLaserTableFromRepo();
        if (!laserTable) {
            // 2026-08-31 起不降级：工厂档无 mapData（生产者缺口 §8-1）时
            // laser_match_scan 查表路线弃用——ScanChains 激光段走同行配对旁路
            // （标准立体方法，不依赖表），激光点照常重建/显示
            JMW_LOG_WARN("02-ScanWorkflow",
                "[ScanWorkflow] 激光温度表不可用（档表无 mapData）——查表匹配旁路为同行配对（标准立体）");
        }
    }
    pipeline_->attachCalib(calib_.cameraMatrixL, calib_.distCoeffsL,
                           calib_.cameraMatrixR, calib_.distCoeffsR,
                           calib_.imageSize.width, calib_.imageSize.height,
                           laserTable);           // 空=A 模式正常配置（ScanChains 不告警）

    sp::PipelineDeps deps;
    deps.eventBus = ctx_ ? ctx_->eventBus() : nullptr;
    deps.sceneFeed = ctx_ ? ctx_->sceneFeed() : nullptr;   // P2 渲染加固：app SceneFeedAdapter（可空）
    auto cr = pipeline_->configure(deps);
    if (!cr.success) {
        pipeline_.reset();
        return Result::fail("ScanPipeline 装配失败: " + cr.message);
    }
    return Result::ok();
}

Result ScanWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok("已在运行");

    auto ar = assemblePipeline();
    if (!ar.success) {
        state_ = WorkflowState::Error;
        JMW_LOG_ERROR("02-ScanWorkflow", "[ScanWorkflow] 装配失败: {}", ar.message);
        return ar;
    }

    auto sr = pipeline_->start();
    if (!sr.success) {
        pipeline_.reset();
        state_ = WorkflowState::Error;
        JMW_LOG_ERROR("02-ScanWorkflow", "[ScanWorkflow] ScanPipeline 启动失败: {}", sr.message);
        return Result::fail("ScanPipeline 启动失败: " + sr.message);
    }

    state_ = WorkflowState::Running;
    sessionStartTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sessionEndTime_ = 0;
    sessionFrames_ = 0;
    sessionFusedFrames_ = 0;
    ctx_->publishEvent(EventType::ScanStarted);
    JMW_LOG_INFO("02-ScanWorkflow", "[ScanWorkflow] 已启动（帧处理移交 07 ScanPipeline）");
    return Result::ok();
}

Result ScanWorkflow::pause() {
    if (state_ != WorkflowState::Running) return Result::fail("非运行状态");
    if (pipeline_) pipeline_->pause();
    state_ = WorkflowState::Paused;
    return Result::ok();
}

Result ScanWorkflow::resume() {
    if (state_ != WorkflowState::Paused) return Result::fail("非暂停状态");
    if (pipeline_) {
        auto rr = pipeline_->resume();
        if (!rr.success) return Result::fail("ScanPipeline 恢复失败: " + rr.message);
    }
    state_ = WorkflowState::Running;
    return Result::ok();
}

// —— P3 可观测薄转发（渲染加固计划：pipeline_ 会话私有，未装配=0）——
uint64_t ScanWorkflow::processedFrameCount() const {
    return pipeline_ ? pipeline_->consumedFrames() : 0;
}
uint64_t ScanWorkflow::droppedFrameCount() const {
    return pipeline_ ? pipeline_->droppedFrames() : 0;
}

// ============================================================================
// 帧响应接线——08 采集回调→会话环（app 双投递：预览 FrameBuffer＋此口）
// 实现 ScanWorkflow.cpp:74 原 TODO：ring 空转断链就此接通
// ============================================================================
void ScanWorkflow::pushSessionFrame(const cv::Mat& grayL, const cv::Mat& grayR,
                                    double temperatureC, uint64_t frameId) {
    // 会话期外丢弃（app 回调常驻注册：预览/标定期帧不进扫描环）
    const auto st = state_.load(std::memory_order_acquire);
    if (st != WorkflowState::Running && st != WorkflowState::Paused) return;
    // enrich 出口查表→Overwrite 环（fail 丢帧计数挂 session.droppedFrames）
    session_.pushFrame(grayL, grayR, temperatureC, frameId);
}

uint64_t ScanWorkflow::droppedSessionFrames() const {
    return session_.droppedFrames();
}

// —— 编辑账本窄出口（05 P4）——
Scanner::pipeline::IMarkerFuse* ScanWorkflow::markerFuse() {
    return pipeline_ ? pipeline_->markerFuse() : nullptr;
}
Scanner::pipeline::FrameObsAccumulator* ScanWorkflow::obsAccumulator() {
    return pipeline_ ? &pipeline_->obs() : nullptr;
}

// ============================================================================
// 激光温度表装载——06 仓库档表 JSON → 09 LaserPlaneMapTempTable
// 格式与 laser_match_scan::LoadTempTable 同源：{"table":[{temperature,mapData}]}
// 档缺 mapData（工厂档现状＝仅温补相机参数）跳过；全缺返 nullptr（调用方降级 A）
// ============================================================================
std::shared_ptr<calib::LaserPlaneMapTempTable>
ScanWorkflow::buildLaserTableFromRepo() const {
    const auto* repo = ctx_ ? ctx_->calibRepo() : nullptr;
    if (!repo) return nullptr;
    std::shared_ptr<calib::LaserPlaneMapTempTable> table;
    try {
        const nlohmann::json raw = repo->planeMapTempTableRaw();
        const auto* tab = raw.contains("table") ? &raw["table"] : nullptr;
        if (!tab || !tab->is_array() || tab->empty()) return nullptr;
        table = std::make_shared<calib::LaserPlaneMapTempTable>();
        size_t loaded = 0, skipped = 0;
        for (const auto& entry : *tab) {
            if (!entry.contains("temperature") || !entry.contains("mapData")) {
                ++skipped;
                continue;
            }
            const double temp = entry["temperature"].get<double>();
            const auto& mapData = entry["mapData"];
            const int n = static_cast<int>(mapData.size());
            if (n <= 0) { ++skipped; continue; }
            calib::LaserPlaneMap pm;
            pm.temperature = temp;
            pm.totalPairs = n;
            cv::Mat m(n, 4, CV_32FC1);            // 每行 [xL, yL, uR, lineId]
            for (int i = 0; i < n; ++i) {
                const auto& row = mapData[i];
                m.at<float>(i, 0) = row[0].get<float>();
                m.at<float>(i, 1) = row[1].get<float>();
                m.at<float>(i, 2) = row[2].get<float>();
                m.at<float>(i, 3) = row[3].get<float>();
            }
            pm.leftToRightMap = std::move(m);
            table->table[temp] = std::move(pm);
            ++loaded;
        }
        if (table->table.empty()) return nullptr;
        JMW_LOG_INFO("02-ScanWorkflow",
            "[ScanWorkflow] 激光温度表装载: {} 档（跳过无 mapData 档 {}）", loaded, skipped);
    } catch (const std::exception& e) {
        JMW_LOG_WARN("02-ScanWorkflow", "[ScanWorkflow] 激光温度表解析异常（降级 A）: {}",
                     e.what());
        return nullptr;
    }
    return table;
}

Scanner::pipeline::FrameObsAccumulator& ScanWorkflow::obs() {
    // pipeline_ 会话私有件——调方保证仅会话装配后访问（stop 后仍存活至析构）
    return pipeline_->obs();
}

Result ScanWorkflow::stop() {
    if (state_ == WorkflowState::Idle) return Result::ok();
    // 完成回报（P5-T15）只在「活跃会话终止」时恰一次：重复 stop / 停后析构 /
    // Error 态（start 同步失败已由 gate 回滚 S2）不重报——重报=gate 侧非法转换噪声
    const bool reportFinish =
        (state_ == WorkflowState::Running || state_ == WorkflowState::Paused);
    state_ = WorkflowState::Stopping;
    if (pipeline_) {
        pipeline_->stop();          // 停止顺序见 07：lane 停→在飞排空→consumer 排空
        pipeline_.reset();          // 会话私有件：stop 后不重启，续扫建新对象
    }
    state_ = WorkflowState::Completed;
    sessionEndTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (ctx_) {
        ctx_->publishEvent(EventType::ScanStopped);
    }
    JMW_LOG_INFO("02-ScanWorkflow", "[ScanWorkflow] 已停止（会话记账: 起={}ms 止={}ms 帧={}/融合={}）",
                 sessionStartTime_, sessionEndTime_, sessionFrames_, sessionFusedFrames_);
    // 合账钩子——app 侧注入调 gate->notifyCompleted 切 S2（§9 02-⑩）。
    // 02 不依赖 10：经 std::function 回调反向解耦（JMW_LOG 宏头在 10，
    // 以下按其展开格式直写 spdlog——输出与 JMW_LOG_INFO 等价，§8.2 生命周期）
    if (reportFinish && onFinished_) {
        // 会话终止账本（流程终点汇总——enrich 丢帧并入，与 07 会话结束账互补）
        JMW_LOG_INFO("02-ScanWorkflow",
            "[ScanWorkflow] 扫描会话终止: 已融合={} enrich丢帧={} 模式={}",
            processedFrameCount(), droppedSessionFrames(),
            scanMode_ == ScanMode::MarkerOnly ? "A(纯标记点)" : "B(标记点+激光)");
        onFinished_(true);
    }
    return Result::ok();
}

Result ScanWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    // TODO(接入期): 帧级处理进度/故障经 EventBus（07 PipelineEventSink →
    // FaultOccurred 族）消费侧接线后透传 UI（现无逐帧进度源）
    return Result::ok();
}

} // namespace Scanner::workflow
