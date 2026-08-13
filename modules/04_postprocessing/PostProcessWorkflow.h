#pragma once
// ============================================================================
// PostProcessWorkflow.h — 后处理工作流（离线 batch）
//
// 全局标记点优化→重融合→法线→封装→补洞→光顺→边界优化→出STL
// ============================================================================

#include "IWorkflow.h"
#include "WorkflowContext.h"
#include <memory>
#include <atomic>
#include <thread>

namespace Scanner::workflow {

struct PostProcessResult {
    bool success = false;
    int meshTriangles = 0;
    double smoothTimeMs = 0.0;
    std::string outputPath;
    std::string message;
};

class PostProcessWorkflow : public IWorkflow {
public:
    explicit PostProcessWorkflow(WorkflowContext* ctx);
    ~PostProcessWorkflow() override;

    void setOutputPath(const std::string& path) { outputPath_ = path; }
    void setSmoothIterations(int n) { smoothIter_ = n; }

    // IWorkflow
    std::string getName() const override { return "PostProcessWorkflow"; }
    Result initialize() override;
    Result start() override;
    Result pause() override { return Result::ok(); }
    Result resume() override { return Result::ok(); }
    Result stop() override;
    WorkflowState getState() const override { return state_.load(); }
    Result setProgressCallback(WorkflowCallback cb) override;

    PostProcessResult getResult() const { return result_; }

private:
    WorkflowContext* ctx_;
    std::string outputPath_ = "output.stl";
    int smoothIter_ = 5;

    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;
    PostProcessResult result_;
    std::thread postThread_;
    std::atomic<bool> running_{false};

    void postProcessLoop();
    void notifyProgress(int stage, const std::string& name);
};

} // namespace Scanner::workflow
