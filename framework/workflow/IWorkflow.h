#pragma once
namespace Scanner::workflow {
enum class WorkflowStatus { Idle, Running, Paused, Stopped, Faulted };
class IWorkflow {
public:
    virtual ~IWorkflow() = default;
    virtual void execute() = 0;
    virtual void stop() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual WorkflowStatus getStatus() const = 0;
};
// WorkflowContext：统一入口；ADR 7.7 窄角色接口（IScanContext/ICalibContext/...）待实现
class WorkflowContext {};
class Pipeline {};   // ADR 7.2 Stage 组合/有界队列/背压/丢帧
class Stage {};
}
