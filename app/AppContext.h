#pragma once
// ============================================================================
// AppContext.h — 应用层装配点
//
// 创建并拥有所有框架对象（Data/Service/HAL/Infra/Workflow），
// 通过 WorkflowContext 统一注入给 UI 和 Workflow。
// 运行期由 app/main 装配，生命周期 = 整个应用。
// ============================================================================

#include "base/types.h"
#include <memory>

namespace Scanner::data    { class FrameBuffer; class PointCloudBuffer; class DeviceStateCache; class CalibStore; }
namespace Scanner::service { class StateMachine; class ParameterManager; class FaultHandler; class CommandGate; class PerfMonitor; }
namespace Scanner::infra   { class EventBus; }
namespace Scanner::device  { class CameraControl; class MCUDriver; class HardwareMonitor; }
namespace Scanner::workflow{ class WorkflowContext; class ScanWorkflow; class CalibrationWorkflow; class PostProcessWorkflow; }

class AppContext {
public:
    AppContext();
    ~AppContext();

    // 初始化所有组件
    void initialize();
    void shutdown();

    // Data 层
    Scanner::data::FrameBuffer*      frameBuffer()     { return frameBuffer_.get(); }
    Scanner::data::PointCloudBuffer* pointCloudBuffer(){ return pointCloudBuffer_.get(); }
    Scanner::data::DeviceStateCache* deviceStateCache(){ return deviceStateCache_.get(); }
    Scanner::data::CalibStore*       calibStore()      { return calibStore_.get(); }

    // Service 层
    Scanner::service::StateMachine*    stateMachine()    { return stateMachine_.get(); }
    Scanner::service::ParameterManager*paramManager()    { return paramManager_.get(); }
    Scanner::service::FaultHandler*    faultHandler()    { return faultHandler_.get(); }
    Scanner::service::CommandGate*     commandGate()     { return commandGate_.get(); }
    Scanner::service::PerfMonitor*     perfMonitor()     { return perfMonitor_.get(); }

    // HAL 层
    Scanner::device::CameraControl*   camera()    { return camera_.get(); }
    Scanner::device::MCUDriver*       mcu()       { return mcu_.get(); }
    Scanner::device::HardwareMonitor* hwMonitor() { return hwMonitor_.get(); }

    // Infra
    Scanner::infra::EventBus* eventBus() { return eventBus_.get(); }

    // Workflow
    Scanner::workflow::WorkflowContext*     workflowCtx()    { return wfCtx_.get(); }
    Scanner::workflow::ScanWorkflow*        scanWorkflow()   { return scanWf_.get(); }
    Scanner::workflow::CalibrationWorkflow* calibWorkflow()  { return calibWf_.get(); }
    Scanner::workflow::PostProcessWorkflow* postWorkflow()   { return postWf_.get(); }

private:
    // Data
    std::unique_ptr<Scanner::data::FrameBuffer>       frameBuffer_;
    std::unique_ptr<Scanner::data::PointCloudBuffer>  pointCloudBuffer_;
    std::unique_ptr<Scanner::data::DeviceStateCache>  deviceStateCache_;
    std::unique_ptr<Scanner::data::CalibStore>        calibStore_;

    // Service（commandGate_/perfMonitor_ 声明于 faultHandler_/stateMachine_ 之后：
    // 析构逆序 → gate/perf 先亡——两者持 SM/FH 裸指针）
    std::unique_ptr<Scanner::service::StateMachine>    stateMachine_;
    std::unique_ptr<Scanner::service::ParameterManager> paramManager_;
    std::unique_ptr<Scanner::service::FaultHandler>     faultHandler_;
    std::unique_ptr<Scanner::service::CommandGate>      commandGate_;
    std::unique_ptr<Scanner::service::PerfMonitor>      perfMonitor_;
    int monitorSourceId_ = 0;

    // Infra
    std::unique_ptr<Scanner::infra::EventBus> eventBus_;

    // HAL
    std::unique_ptr<Scanner::device::CameraControl>   camera_;
    std::unique_ptr<Scanner::device::MCUDriver>       mcu_;
    std::unique_ptr<Scanner::device::HardwareMonitor> hwMonitor_;

    // Workflow
    std::unique_ptr<Scanner::workflow::WorkflowContext>     wfCtx_;
    std::unique_ptr<Scanner::workflow::ScanWorkflow>        scanWf_;
    std::unique_ptr<Scanner::workflow::CalibrationWorkflow> calibWf_;
    std::unique_ptr<Scanner::workflow::PostProcessWorkflow> postWf_;
};
