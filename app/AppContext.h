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
#include <mutex>
#include <string>

namespace Scanner::data    { class FrameBuffer; class PointCloudBuffer; class DeviceStateCache; class CalibrationRepository; }
namespace Scanner::service { class StateMachine; class ParameterManager; class FaultHandler; class CommandGate; class PerfMonitor; }
namespace Scanner::infra   { class EventBus; }
namespace Scanner::device  { class DeviceManager; class HardwareMonitor; class SelfCheckCollector; }
namespace Scanner::workflow{ class WorkflowContext; class ScanWorkflow; class CalibrationWorkflow; class PostProcessWorkflow; }

class AppContext {
public:
    AppContext();
    ~AppContext();

    // 初始化所有组件
    void initialize();
    void shutdown();

    // 自检清单（S1→S2 唯一出口经 system_ready，设计 §2.4）
    void notifySelfCheckItem(const std::string& item, bool ok);   // 自检项打勾/打叉（"camera"/"serialPort"/"license"）
    bool selfCheckAllPassed() const;

    // Data 层
    Scanner::data::FrameBuffer*      frameBuffer()     { return frameBuffer_.get(); }
    Scanner::data::PointCloudBuffer* pointCloudBuffer(){ return pointCloudBuffer_.get(); }
    Scanner::data::DeviceStateCache* deviceStateCache(){ return deviceStateCache_.get(); }
    Scanner::data::CalibrationRepository* calibRepo()      { return calibRepo_.get(); }

    // Service 层
    Scanner::service::StateMachine*    stateMachine()    { return stateMachine_.get(); }
    Scanner::service::ParameterManager*paramManager()    { return paramManager_.get(); }
    Scanner::service::FaultHandler*    faultHandler()    { return faultHandler_.get(); }
    Scanner::service::CommandGate*     commandGate()     { return commandGate_.get(); }
    Scanner::service::PerfMonitor*     perfMonitor()     { return perfMonitor_.get(); }

    // HAL 层（A-T17 三行收拢：相机+MCU 收进 DeviceManager 门面；HardwareMonitor
    // 为巡检件独立保留——门面不管它）
    Scanner::device::DeviceManager*    deviceManager() { return deviceManager_.get(); }
    Scanner::device::HardwareMonitor*  hwMonitor()     { return hwMonitor_.get(); }

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
    std::unique_ptr<Scanner::data::CalibrationRepository> calibRepo_;

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

    // HAL（deviceManager_ 声明于 selfCheckCollector_/hwMonitor_ 之前：析构逆序
    // → 巡检件先亡——hwMonitor 经裸指针/回调触达门面与自检采集器）
    std::unique_ptr<Scanner::device::DeviceManager>      deviceManager_;
    std::unique_ptr<Scanner::device::SelfCheckCollector> selfCheckCollector_;
    std::unique_ptr<Scanner::device::HardwareMonitor>    hwMonitor_;

    // Workflow
    std::unique_ptr<Scanner::workflow::WorkflowContext>     wfCtx_;
    std::unique_ptr<Scanner::workflow::ScanWorkflow>        scanWf_;
    std::unique_ptr<Scanner::workflow::CalibrationWorkflow> calibWf_;
    std::unique_ptr<Scanner::workflow::PostProcessWorkflow> postWf_;

    // 自检清单（ScannerWindow 线程回调与查询并发防护）
    struct SelfCheck {
        bool camera{false};        // 相机连接（ScannerWindow 设备连接事件置位）
        bool serialPort{false};    // 串口连接（同上）
        // 加密狗/授权：暂占位 true（TODO: 08/授权落地后接入实际检测）
        bool license{true};
    };
    SelfCheck selfCheck_;
    mutable std::mutex selfCheckMtx_;
};
