// ============================================================================
// ScanWorkflow.cpp — 扫描工作流实现（编排/记账壳；帧处理移交 07 ScanPipeline）
//
// P6-T29a：旧 scanLoop 五 Stage（CPU 手搓掩码/标记点链/激光空壳/CPU 融合）
// 整体退役——单帧处理由 07 ScanPipeline 内部并行调度（E 核抓帧→GPU 链→
// P 核标记点链→融合消费线程）完成；本类仅做装配、生命周期与会话记账。
// ============================================================================

#include "ScanWorkflow.h"

#include "pipelines/PipelineDeps.h"
#include "pipelines/scan/ScanPipeline.h"

#include <spdlog/spdlog.h>
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
    spdlog::info("[ScanWorkflow] 初始化 (模式={}, 标定={})",
                 scanMode_ == ScanMode::MarkerOnly ? "纯标记点" : "标记点+激光",
                 calib_.valid ? "已加载" : "未加载");
    // TODO(接入期): 06 出口查表接线后，此处改为从标定结果仓库装载逐温档
    // K/D + 激光温度表（现经 setCalibration 静态注入转接 attachCalib）
    return Result::ok();
}

Result ScanWorkflow::assemblePipeline() {
    namespace sp = Scanner::pipeline;

    sp::ScanConfig cfg;
    cfg.enableLaser = (scanMode_ == ScanMode::MarkerPlusLaser);
    // TODO(接入期): existingMarkers（app 点云仓库高精度先验）此处注入 cfg
    pipeline_ = std::make_unique<sp::ScanPipeline>(cfg);

    // TODO(接入期): 08 采集侧写 EnhancedFrame 进 ring_（FrameBuffer 扫描路径
    // 退役）；当前无真帧源——ring_ 空转，流水线各线程等帧（07 防御路径保留）
    pipeline_->attachRing(ring_, /*dropThreshold=*/0);   // 0=自动（2*lanes）

    if (!calib_.valid || calib_.cameraMatrixL.empty() || calib_.cameraMatrixR.empty()) {
        pipeline_.reset();
        return Result::fail(
            "无标定参数——07 生产链须 attachCalib（TODO 接入期：06 出口查表供逐温档 K/D）");
    }
    // TODO(接入期): 激光温度表（calib::LaserPlaneMapTempTable）经 06 标定结果
    // 仓库整表注入；A 模式（纯标记点）空表属正常配置
    pipeline_->attachCalib(calib_.cameraMatrixL, calib_.distCoeffsL,
                           calib_.cameraMatrixR, calib_.distCoeffsR,
                           calib_.imageSize.width, calib_.imageSize.height,
                           /*laserTable=*/nullptr);

    sp::PipelineDeps deps;
    deps.eventBus = ctx_ ? ctx_->eventBus() : nullptr;
    // TODO(接入期): 03 渲染 ISceneFeed 接线（pushCloudSnapshot→OSG 场景推送）
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
        spdlog::error("[ScanWorkflow] 装配失败: {}", ar.message);
        return ar;
    }

    auto sr = pipeline_->start();
    if (!sr.success) {
        pipeline_.reset();
        state_ = WorkflowState::Error;
        spdlog::error("[ScanWorkflow] ScanPipeline 启动失败: {}", sr.message);
        return Result::fail("ScanPipeline 启动失败: " + sr.message);
    }

    state_ = WorkflowState::Running;
    sessionStartTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sessionEndTime_ = 0;
    sessionFrames_ = 0;
    sessionFusedFrames_ = 0;
    ctx_->publishEvent(EventType::ScanStarted);
    spdlog::info("[ScanWorkflow] 已启动（帧处理移交 07 ScanPipeline）");
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

Result ScanWorkflow::stop() {
    if (state_ == WorkflowState::Idle) return Result::ok();
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
    spdlog::info("[ScanWorkflow] 已停止（会话记账: 起={}ms 止={}ms 帧={}/融合={}）",
                 sessionStartTime_, sessionEndTime_, sessionFrames_, sessionFusedFrames_);
    return Result::ok();
}

Result ScanWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    // TODO(接入期): 帧级处理进度/故障经 EventBus（07 PipelineEventSink →
    // FaultOccurred 族）消费侧接线后透传 UI（现无逐帧进度源）
    return Result::ok();
}

} // namespace Scanner::workflow
