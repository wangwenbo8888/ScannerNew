#pragma once
// ============================================================================
// ScanWorkflow.h — 扫描工作流（编排/记账壳；帧处理已移交 07 ScanPipeline）
//
// P6-T29a（07 文档 §3 修改 1）：旧 scanLoop 单线程五 Stage
// （Capture/Preprocess/Marker/Laser/Fuse）退役——帧处理路径改为持有
// Scanner::pipeline::ScanPipeline（07 §1.1-C，会话私有件：每次扫描新建、
// stop 后不重启），本类只留 IWorkflow 生命周期编排 + 会话记账
// （SessionService/EventBus/UI 回调）。
//
// 装配序（start 内）：
//   ScanConfig（scanMode→enableLaser）→ attachRing（本类持有的 06 SlotRing·
//   Overwrite 扫描面孔）→ attachCalib（ScanCalibration 注入转接——静态 K/D
//   过渡契约）→ configure（EventBus 接线）→ start ⇄ pause/resume → stop。
//
// TODO(接入期接线，见 07 文档 §3/客户端扫描流水线.md)：
//   - 08 采集侧：StereoFrame → EnhancedFrame 写入 ring_（FrameBuffer 扫描
//     路径退役）；当前无真帧源写入——ring_ 空转，07 防御路径保留
//     （无 ring/无标定时 configure/start 返回 fail，不崩）。
//   - 06 出口查表：attachCalib 的逐温档 K/D 与激光温度表整表注入
//     （现静态 K/D + 空表=A 模式可空）。
//   - 03 渲染：PipelineDeps.sceneFeed（ISceneFeed→OSG 点云/姿态推送；现空）。
//   - existingMarkers（app 点云仓库高精度先验）装配期注入 ScanConfig。
//   - 02-⑦ 收尾批算：GlobalOptimObject 消费 pipeline_->obs()（Q5 定案归 02）。
// ============================================================================

#include "IWorkflow.h"
#include "WorkflowContext.h"
#include "SlotRing.h"
#include "EnhancedFrame.h"
#include "base/types.h"
#include <memory>
#include <atomic>

// 前向声明（07 流水线对象；定义见 modules/07_pipelinemgmt/pipelines/scan/）
namespace Scanner::pipeline { class ScanPipeline; }

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
    WorkflowContext* ctx_;
    ScanMode scanMode_ = ScanMode::MarkerOnly;
    ScanCalibration calib_;
    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;

    // —— 07 帧处理引擎（会话私有件：start 建、stop 收）——
    std::unique_ptr<Scanner::pipeline::ScanPipeline> pipeline_;

    // —— 输入环（本类持有的 06 会话件；08 采集侧写入——TODO 接入期）——
    static constexpr size_t kRingSlots = 16;
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame> ring_{
        kRingSlots,
        Scanner::data::SlotRing<Scanner::data::EnhancedFrame>::WriterMode::Overwrite};

    /// 装配 07 ScanPipeline（attachRing/attachCalib/configure；失败自清理）
    Result assemblePipeline();
};

} // namespace Scanner::workflow
