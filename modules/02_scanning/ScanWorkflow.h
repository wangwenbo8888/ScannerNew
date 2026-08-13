#pragma once
// ============================================================================
// ScanWorkflow.h — 扫描工作流（双模式：纯标记点 / 标记点+激光）
//
// CPU 管线（BUILD_CUDA=OFF）:
//   Stage0: Capture    — 从 FrameBuffer 取帧
//   Stage1: Preprocess — OpenCV 阈值/形态学生成标记点掩码
//   Stage2: Marker     — 标记点链: CCL→split→zernike→ellipse→match→reconstruct
//   Stage3: Laser      — 跳过（CUDA only）
//   Stage4: Fuse       — laser_cloud_fuse_cpu → PointCloudBuffer
//
// GPU 管线（BUILD_CUDA=ON，待启用）:
//   Stage1: mask_separation_cuda
//   Stage3: steger→undistort→epipolar_interp→laser_match→laser_reconstruct
// ============================================================================

#include "IWorkflow.h"
#include "Pipeline.h"
#include "WorkflowContext.h"
#include "IFrameSink.h"
#include "base/types.h"
#include <opencv2/core.hpp>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

// 前向声明（全局命名空间）
namespace calib { class ZernikeEdgeCPU; class EllipseFitCPU; class MarkerMatchCPU; class LaserCloudFuseCPU; }

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
// ScanFrameResult — 单帧处理结果
// ============================================================================
struct ScanFrameResult {
    std::vector<cv::Point3f> markerPoints3d;
    std::vector<cv::Point3f> laserPoints3d;
    std::vector<cv::Point3f> markerNormals;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T{0, 0, 0};
    int markerCount = 0;
    bool success = false;
};

// ============================================================================
// CaptureStage — 从 FrameBuffer 消费帧
// ============================================================================
class CaptureStage : public Stage {
public:
    explicit CaptureStage(WorkflowContext* ctx);
    Result process() override;
    data::FrameData getLatestFrame() const;
private:
    WorkflowContext* ctx_;
    data::FrameData latestFrame_;
    mutable std::mutex frameMutex_;
};

// ============================================================================
// PreprocessStage — CPU 掩码生成（OpenCV 阈值 + 形态学）
// ============================================================================
class PreprocessStage : public Stage {
public:
    explicit PreprocessStage(WorkflowContext* ctx);
    Result process() override;
    void setInput(const data::FrameData& frame) { inputFrame_ = frame; hasInput_ = true; }

    cv::Mat leftMarkerMask;
    cv::Mat rightMarkerMask;

private:
    WorkflowContext* ctx_;
    data::FrameData inputFrame_;
    bool hasInput_ = false;

    cv::Mat createMarkerMask(const cv::Mat& gray);
};

//
// ============================================================================
// MarkerStage — 标记点链（CCL→split→zernike→ellipse→match→reconstruct）
// ============================================================================
class MarkerStage : public Stage {
public:
    explicit MarkerStage(WorkflowContext* ctx);
    ~MarkerStage() override;
    Result process() override;
    void setInput(const data::FrameData& frame,
                  const cv::Mat& leftMask, const cv::Mat& rightMask);
    void setCalibration(const ScanCalibration& calib) { calib_ = calib; }

    ScanFrameResult result;

private:
    WorkflowContext* ctx_;
    data::FrameData inputFrame_;
    cv::Mat leftMask_, rightMask_;
    ScanCalibration calib_;

    // 复用算子（避免每帧创建/销毁）
    std::unique_ptr<::calib::ZernikeEdgeCPU> zernikeOp_;
    std::unique_ptr<::calib::EllipseFitCPU> ellipseOp_;
    std::unique_ptr<::calib::MarkerMatchCPU> matchOp_;

    int processCounter_ = 0;  // 隔帧处理计数器

    std::vector<cv::Point2f> detectCenters(const cv::Mat& gray, const cv::Mat& mask);
};

// ============================================================================
// LaserStage — 激光链（CUDA only，CPU 模式跳过）
// ============================================================================
class LaserStage : public Stage {
public:
    explicit LaserStage(WorkflowContext* ctx, ScanMode mode);
    Result process() override;
    void setInput(const data::FrameData& frame) { inputFrame_ = frame; }
private:
    WorkflowContext* ctx_;
    ScanMode mode_;
    data::FrameData inputFrame_;
};

// ============================================================================
// FuseStage — 体素融合 → 写 PointCloudBuffer
// ============================================================================
class FuseStage : public Stage {
public:
    explicit FuseStage(WorkflowContext* ctx);
    ~FuseStage() override;
    Result process() override;
    void addPoints(const std::vector<cv::Point3f>& points,
                   const cv::Matx33d& R, const cv::Vec3d& T);
private:
    WorkflowContext* ctx_;
    std::vector<cv::Point3f> pendingPoints_;
    cv::Matx33d pendingR_ = cv::Matx33d::eye();
    cv::Vec3d pendingT_{0, 0, 0};
    mutable std::mutex pointsMutex_;
    std::unique_ptr<::calib::LaserCloudFuseCPU> fuseOp_;
};

// ============================================================================
// ScanWorkflow — 扫描工作流（实现 IWorkflow）
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

    std::unique_ptr<CaptureStage>    capture_;
    std::unique_ptr<PreprocessStage> preprocess_;
    std::unique_ptr<MarkerStage>     marker_;
    std::unique_ptr<LaserStage>      laser_;
    std::unique_ptr<FuseStage>       fuse_;

    std::thread scanThread_;
    std::atomic<bool> running_{false};

    void scanLoop();
    void notifyProgress(const std::string& stage, float progress);
};

} // namespace Scanner::workflow
