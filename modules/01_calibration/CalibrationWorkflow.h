#pragma once
// ============================================================================
// CalibrationWorkflow.h — 标定工作流
//
// 25姿态采集 → 标记点提取 → 相机标定 → 立体矫正 → 激光平面标定 → 温度补偿表
// ============================================================================

#include "workflow/IWorkflow.h"
#include "workflow/WorkflowContext.h"
#include <memory>
#include <atomic>
#include <thread>
#include <opencv2/core.hpp>

namespace Scanner::workflow {

struct CalibrationResult {
    bool success = false;
    double reprojErrorLeft = 0.0;
    double reprojErrorRight = 0.0;
    double stereoError = 0.0;
    std::string message;
};

class CalibrationWorkflow : public IWorkflow {
public:
    explicit CalibrationWorkflow(WorkflowContext* ctx);
    ~CalibrationWorkflow() override;

    void setNumPoses(int n) { numPoses_ = n; }
    void setChessboardSize(int cols, int rows) { boardCols_ = cols; boardRows_ = rows; }
    void setSquareSize(double mm) { squareSize_ = mm; }

    // IWorkflow
    std::string getName() const override { return "CalibrationWorkflow"; }
    Result initialize() override;
    Result start() override;
    Result pause() override { return Result::ok(); }
    Result resume() override { return Result::ok(); }
    Result stop() override;
    WorkflowState getState() const override { return state_.load(); }
    Result setProgressCallback(WorkflowCallback cb) override;

    CalibrationResult getResult() const { return result_; }

private:
    WorkflowContext* ctx_;
    int numPoses_ = 25;
    int boardCols_ = 11;
    int boardRows_ = 8;
    double squareSize_ = 5.0;  // mm

    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;
    CalibrationResult result_;

    std::thread calibThread_;
    std::atomic<bool> running_{false};

    void calibrationLoop();
    void notifyProgress(int current, int total, const std::string& stage);
};

} // namespace Scanner::workflow
