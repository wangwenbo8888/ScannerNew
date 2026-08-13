#pragma once
// ============================================================================
// IWorkflow.h — 工作流接口（Workflow 层）
//
// 标定/扫描/后处理三条工作流均实现此接口。
// 生命周期由 Service（会话/状态机）编排。
// ============================================================================

#include "base/types.h"
#include <string>

namespace Scanner::workflow {

// ============================================================================
// 工作流状态
// ============================================================================
enum class WorkflowState : uint8_t {
    Idle,
    Running,
    Paused,
    Stopping,
    Completed,
    Error
};

// ============================================================================
// 工作流回调
// ============================================================================
struct WorkflowProgress {
    WorkflowState state = WorkflowState::Idle;
    int currentStage = 0;
    int totalStages = 0;
    std::string stageName;
    std::string message;
    float progress = 0.0f;  // 0.0 ~ 1.0
};

using WorkflowCallback = std::function<void(const WorkflowProgress&)>;

// ============================================================================
// IWorkflow — 工作流接口
// ============================================================================
class IWorkflow {
public:
    virtual ~IWorkflow() = default;

    virtual std::string getName() const = 0;

    /// 初始化（加载标定参数等）
    virtual Result initialize() = 0;

    /// 启动工作流
    virtual Result start() = 0;

    /// 暂停
    virtual Result pause() = 0;

    /// 恢复
    virtual Result resume() = 0;

    /// 停止
    virtual Result stop() = 0;

    /// 查询状态
    virtual WorkflowState getState() const = 0;

    /// 进度回调
    virtual Result setProgressCallback(WorkflowCallback cb) = 0;
};

} // namespace Scanner::workflow
