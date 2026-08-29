// ============================================================================
// ScanWorkflow.cpp — 扫描工作流实现（编排/记账壳；帧处理移交 07 ScanPipeline）
//
// P6-T29a：旧 scanLoop 五 Stage（CPU 手搓掩码/标记点链/激光空壳/CPU 融合）
// 整体退役——单帧处理由 07 ScanPipeline 内部并行调度（E 核抓帧→GPU 链→
// P 核标记点链→融合消费线程）完成；本类仅做装配、生命周期与会话记账。
// ============================================================================

#include "ScanWorkflow.h"

#include "PointCloudBuffer.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/scan/ScanPipeline.h"

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

    // TODO(接入期): 08 采集回调 session_.pushFrame(grayL, grayR, tempC, frameId)
    // （enrich 出口查表→ring；fail 丢帧计数）；当前无真帧源——ring 空转，
    // 流水线各线程等帧（07 防御路径保留）
    pipeline_->attachRing(session_.ring(), /*dropThreshold=*/0);   // 0=自动（2*lanes）

    if (!calib_.valid || calib_.cameraMatrixL.empty() || calib_.cameraMatrixR.empty()) {
        pipeline_.reset();
        return Result::fail(
            "无标定参数——07 生产链须 attachCalib（TODO 接入期：06 出口查表供逐温档 K/D）");
    }
    // TODO(接入期): 激光网格数据归 §8-1 协调项，接线批注入；A 模式（纯标记点）空表属正常配置
    pipeline_->attachCalib(calib_.cameraMatrixL, calib_.distCoeffsL,
                           calib_.cameraMatrixR, calib_.distCoeffsR,
                           calib_.imageSize.width, calib_.imageSize.height,
                           /*laserTable=*/nullptr);

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
        JMW_LOG_INFO("02-ScanWorkflow", "[02-ScanWorkflow] 扫描会话终止 ok=true（⑩ 合账回报）");
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
