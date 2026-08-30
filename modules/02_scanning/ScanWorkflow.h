#pragma once
// ============================================================================
// ScanWorkflow.h — 扫描工作流（编排/记账壳；帧处理已移交 07 ScanPipeline）
//
// P6-T29a（07 文档 §3 修改 1）：旧 scanLoop 单线程五 Stage
// （Capture/Preprocess/Marker/Laser/Fuse）退役——帧处理路径改为持有
// Scanner::pipeline::ScanPipeline（07 §1.1-C，会话私有件：每次扫描新建、
// stop 后不重启），本类只留 IWorkflow 生命周期编排 + 会话记账
// （D6：记账归工作流自身成员，SessionService 随 07_session 退役）+ EventBus/UI 回调。
//
// 装配序（start 内）：
//   session_.assemble（06 出口查表装表）→ ScanConfig（scanMode→enableLaser＋
//   existingMarkers 续扫基准注入）→ attachRing（session_.ring）→ attachCalib
//   （ScanCalibration 注入转接——静态 K/D 过渡契约）→ configure（EventBus 接线）
//   → start ⇄ pause/resume → stop。
//
// TODO(接入期接线，见 07 文档 §3/客户端扫描流水线.md）：
//   - 08 采集侧：采集回调 session_.pushFrame（enrich 出口查表→ring；FrameBuffer
//     扫描路径退役）；当前无真帧源写入——ring 空转，07 防御路径保留
//     （无 ring/无标定时 configure/start 返回 fail，不崩）。
//   - 激光网格数据（§8-1 协调项）：attachCalib 激光温度表整表注入（现空表=
//     A 模式可空；逐温档 K/D 经 session_ 已并行生效）。
//   - 03 渲染：PipelineDeps.sceneFeed（ISceneFeed→OSG 点云/姿态推送；现空）。
//   - 02-⑦ 收尾批算：GlobalOptimObject 消费 pipeline_->obs()（Q5 定案归 02）。
// ============================================================================

#include "IWorkflow.h"
#include "WorkflowContext.h"
#include "ScanSessionData.h"
#include "base/types.h"
#include <opencv2/core.hpp>
#include <functional>
#include <memory>
#include <atomic>

// 前向声明（07 流水线对象；定义见 modules/07_pipelinemgmt/pipelines/scan/）
namespace Scanner::pipeline {
class ScanPipeline;
class FrameObsAccumulator;
}
namespace calib { struct LaserPlaneMapTempTable; }   // 09 激光温度表（档→映射表）

namespace Scanner::workflow {

// ============================================================================
// ScanCalibration — 扫描所需的标定参数（由 CalibrationWorkflow 产出）
// ============================================================================
struct ScanCalibration {
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imageSize;
    bool valid = false;
};

// ============================================================================
// ScanWorkflow — 扫描工作流（实现 IWorkflow；帧处理移交 07）
// ============================================================================
class ScanWorkflow : public IWorkflow {
public:
    explicit ScanWorkflow(WorkflowContext* ctx);
    ~ScanWorkflow() override;

    void setScanMode(ScanMode mode) { scanMode_ = mode; }
    void setCalibration(const ScanCalibration& calib);

    // —— 完成回报钩子（P5-T15 门禁接线，同 01 T14 模式）——
    /// 会话终态回报（stop() 活跃会话终止处调用，bool=ok）。
    /// app 侧注入 lambda 调 CommandGate::notifyCompleted 合账切 S2——
    /// 依赖方向 app→02/10，02 不 include 10 头（经回调反向解耦）。
    void setOnFinished(std::function<void(bool)> cb) { onFinished_ = std::move(cb); }

    // P3 可观测（渲染加固计划）：app 装配喂 HardwareMonitor（会话外/未启动=0）
    uint64_t processedFrameCount() const;    // 已融合消费帧数（07 FuseConsumer）
    uint64_t droppedFrameCount() const;      // 输出队列覆盖丢帧累计（07 queue）

    // —— 帧响应接线（08 采集回调→会话环；app 帧回调双投递：预览 FrameBuffer＋此口）——
    /// 扫描会话帧入口：enrich 出口查表→Overwrite 环。仅 Running/Paused 态接收；
    /// 其余态静默丢弃（app 回调常驻注册，非扫描期帧不进环）。
    /// 线程＝相机回调线程（SlotRing 单生产者约定——只从这一个线程调）
    void pushSessionFrame(const cv::Mat& grayL, const cv::Mat& grayR,
                          double temperatureC, uint64_t frameId);
    /// enrich fail 丢帧累计（挂 10 故障桥的口径源）
    uint64_t droppedSessionFrames() const;

    /// 逐帧观测累加器出口（app 收尾 GBA 消费；pipeline_ 未装配/已停＝未定义——
    /// 调方保证仅在会话 stop 后、对象存活期内访问）
    Scanner::pipeline::FrameObsAccumulator& obs();

    // IWorkflow
    std::string getName() const override { return "ScanWorkflow"; }
    Result initialize() override;
    Result start() override;
    Result pause() override;
    Result resume() override;
    Result stop() override;
    WorkflowState getState() const override { return state_.load(); }
    Result setProgressCallback(WorkflowCallback cb) override;

private:
    /// 激光温度表装载：06 仓库档表 JSON → 09 LaserPlaneMapTempTable（mapData 格式，
    /// 与 laser_match_scan::LoadTempTable 同源）。全档无 mapData → nullptr（降级 A）
    std::shared_ptr<calib::LaserPlaneMapTempTable> buildLaserTableFromRepo() const;

private:
    WorkflowContext* ctx_;
    ScanMode scanMode_ = ScanMode::MarkerOnly;
    ScanCalibration calib_;
    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;
    std::function<void(bool)> onFinished_;         // 完成回报（app 注入；可空）

    // —— 会话记账（D6：归工作流自身；起止时间戳/帧计数，接入期由 07 流水线回调回填）——
    TimestampMs sessionStartTime_   = 0;
    TimestampMs sessionEndTime_     = 0;
    uint64_t    sessionFrames_      = 0;
    uint64_t    sessionFusedFrames_ = 0;

    // —— 06 会话件（装表＋ring＋pushFrame 唯一入口）——声明序铁律（§4.4）：必须在
    // pipeline_ 之前声明——C++ 逆序析构：pipeline_ 先死（停用 ring），ring 随 session_ 最后死
    Scanner::data::ScanSessionData session_;

    // —— 07 帧处理引擎（会话私有件：start 建、stop 收）——
    std::unique_ptr<Scanner::pipeline::ScanPipeline> pipeline_;

    /// 装配 07 ScanPipeline（attachRing/attachCalib/configure；失败自清理）
    Result assemblePipeline();
};

} // namespace Scanner::workflow
