// ============================================================================
// AppContext.cpp — 应用层装配实现
// ============================================================================

#include "AppContext.h"
#include "FrameBuffer.h"
#include "PointCloudBuffer.h"
#include "DeviceStateCache.h"
#include "CalibStore.h"
#include "StateMachine.h"
#include "ParameterManager.h"
#include "FaultHandler.h"
#include "CommandGate.h"
#include "DefaultCommands.h"
#include "PerfMonitor.h"
#include "jmw_logging.h"
#include "base/EventBus.h"
#include "modules/08_devicemgmt/CameraControl.h"
#include "modules/08_devicemgmt/MCUDriver.h"
#include "modules/08_devicemgmt/HardwareMonitor.h"
#include "WorkflowContext.h"
#include "ScanWorkflow.h"
#include "CalibrationWorkflow.h"
#include "PostProcessWorkflow.h"
#include <spdlog/spdlog.h>

AppContext::AppContext() {}
AppContext::~AppContext() { shutdown(); }

void AppContext::initialize() {
    // === Infra ===
    eventBus_ = std::make_unique<Scanner::infra::EventBus>();

    // === Data ===
    frameBuffer_      = std::make_unique<Scanner::data::FrameBuffer>(60);
    pointCloudBuffer_ = std::make_unique<Scanner::data::PointCloudBuffer>();
    deviceStateCache_ = std::make_unique<Scanner::data::DeviceStateCache>();
    calibStore_       = std::make_unique<Scanner::data::CalibStore>();

    // === Service ===
    stateMachine_   = std::make_unique<Scanner::service::StateMachine>(eventBus_.get());
    paramManager_   = std::make_unique<Scanner::service::ParameterManager>();
    faultHandler_   = std::make_unique<Scanner::service::FaultHandler>(eventBus_.get());

    faultHandler_->setStateMachine(stateMachine_.get());
    faultHandler_->setSafeStopCallback([] {});  // TODO: 08 落地后接 DeviceManager::toIdle()
    faultHandler_->start();

    commandGate_ = std::make_unique<Scanner::service::CommandGate>(stateMachine_.get(), eventBus_.get());
    // P5-T14 01 标定接线：注册前逐个填 handler（DefaultCommands 返回时全空）
    auto specs = Scanner::service::makeDefaultCommandSpecs();
    for (auto& spec : specs) {
        if (spec.name == "start_calibration") {
            // 点火语义：initialize（参数校验）+ start 均毫秒级返回——start() 异步启
            // A 姿态采集（07 收口 watcher 线程），B 批算在专属 calibThread_，handler
            // 即返不阻塞；同步失败（缺参/A 启动失败）返回 fail 由 gate 回滚 S2（§3.3）。
            // finish_calibration（触发型）handler 留空：收尾由工作流跑完经 onFinished_
            // 回调 notifyCompleted 合账（下方 calibWf_ 装配处注入）
            spec.handler = [this]() {
                if (!calibWf_) return Scanner::Result::fail("标定工作流未装配");
                auto r = calibWf_->initialize();
                if (!r.success) return r;
                return calibWf_->start();
            };
        }
        commandGate_->registerCommand(std::move(spec));
    }

    monitorSourceId_ = faultHandler_->registerSource("Monitor");
    perfMonitor_ = std::make_unique<Scanner::service::PerfMonitor>(eventBus_.get(), faultHandler_.get(), monitorSourceId_);
    // 08 落地后注入 IHealthProvider 并由巡检线程/app 定时器调 poll()

    // === HAL ===
    Scanner::device::StereoPairConfig camCfg;
    camCfg.deviceIndexLeft = 0;
    camCfg.deviceIndexRight = 1;
    camCfg.rotateRight180 = true;
    camera_ = std::make_unique<Scanner::device::CameraControl>(camCfg);
    mcu_    = std::make_unique<Scanner::device::MCUDriver>(115200);

    hwMonitor_ = std::make_unique<Scanner::device::HardwareMonitor>();
    hwMonitor_->setDeviceStateCache(deviceStateCache_.get());
    hwMonitor_->setEventBus(eventBus_.get());
    hwMonitor_->setMCU(mcu_.get());
    hwMonitor_->setCamera(camera_.get());

    // === WorkflowContext 装配 ===
    wfCtx_ = std::make_unique<Scanner::workflow::WorkflowContext>();
    wfCtx_->setFrameBuffer(frameBuffer_.get());
    wfCtx_->setPointCloudBuffer(pointCloudBuffer_.get());
    wfCtx_->setDeviceStateCache(deviceStateCache_.get());
    wfCtx_->setCalibStore(calibStore_.get());
    wfCtx_->setStateMachine(stateMachine_.get());
    wfCtx_->setParameterManager(paramManager_.get());
    wfCtx_->setCamera(camera_.get());
    wfCtx_->setMCU(mcu_.get());
    wfCtx_->setEventBus(eventBus_.get());

    // === Workflow ===
    scanWf_  = std::make_unique<Scanner::workflow::ScanWorkflow>(wfCtx_.get());
    calibWf_ = std::make_unique<Scanner::workflow::CalibrationWorkflow>(wfCtx_.get());
    // P5-T14 完成回报注入（§9 01-⑨）：工作流 B 批算线程尾回调 → 合账切 S2；
    // app 是组合根，可同时触达 01 工作流与 10 门禁（01 自身不依赖 10）
    calibWf_->setOnFinished([this](bool ok) {
        commandGate_->notifyCompleted("start_calibration", ok);
    });
    postWf_  = std::make_unique<Scanner::workflow::PostProcessWorkflow>(wfCtx_.get());

    spdlog::info("[AppContext] 全部组件装配完成");

    // 启动 HardwareMonitor（始终运行，周期采集设备状态）
    hwMonitor_->start(1000);
    spdlog::info("[AppContext] HardwareMonitor 已启动");
}

void AppContext::shutdown() {
    if (hwMonitor_) hwMonitor_->stop();
    if (scanWf_)    scanWf_->stop();
    if (calibWf_)   calibWf_->stop();
    if (postWf_)    postWf_->stop();
    if (faultHandler_) faultHandler_->stop();
    if (camera_)    { camera_->stopAsyncCapture(); camera_->close(); }
    if (mcu_)       mcu_->close();
    spdlog::info("[AppContext] 全部组件已关闭");
}

void AppContext::notifySelfCheckItem(const std::string& item, bool ok) {
    {
        std::lock_guard<std::mutex> lock(selfCheckMtx_);
        if      (item == "camera")     selfCheck_.camera     = ok;
        else if (item == "serialPort") selfCheck_.serialPort = ok;
        else if (item == "license")    selfCheck_.license    = ok;
        else return;
    }
    JMW_LOG_INFO("app", "自检项 {}: {}", item, ok ? "通过" : "失败");
    // 全过且仍处 S1 → 经命令通道切 S2（恰一次：submit 成功即离 S1，重复调用幂等失败）
    if (selfCheckAllPassed() &&
        stateMachine_->getCurrentState() == Scanner::service::SystemState::Init) {
        commandGate_->submit("system_ready");
    }
}

bool AppContext::selfCheckAllPassed() const {
    std::lock_guard<std::mutex> lock(selfCheckMtx_);
    return selfCheck_.camera && selfCheck_.serialPort && selfCheck_.license;
}
