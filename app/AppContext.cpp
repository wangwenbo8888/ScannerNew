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
        if (spec.name == "start_scan") {
            // pre（§9 02-①）：参数就绪谓词——查 06 CalibStore（T14 后 01 已写入）。
            // TODO(06 差距): 出口查表（逐温档 K/D + 激光温度表）接入后由 02 侧升级判据
            spec.pre = [this]() {
                if (!wfCtx_ || !wfCtx_->calibStore() || !wfCtx_->calibStore()->hasData())
                    return Scanner::Result::fail("标定参数未就绪——请先完成标定");
                return Scanner::Result::ok();
            };
            // 点火语义：CalibStore→ScanCalibration 静态转接（app=组合根做注入，
            // CalibStore 本身注释「供 CalibrationWorkflow 写入、ScanWorkflow 读取」；
            // TODO 06 出口查表后逐温档 K/D/激光温度表归 02 侧自查）+ initialize +
            // start——帧处理归 07 内部线程，handler 毫秒级即返；同步失败（07 装配
            // 失败等）返回 fail 由 gate 回滚 S2（§3.3）。
            // ScanMode 不经 gate payload（handler 无参，payload 只喂 transition 的
            // S4/S5 判别）——UI 入口先 setScanMode 设进工作流（见 ScannerWindow）
            spec.handler = [this]() {
                if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
                auto* cs = wfCtx_ ? wfCtx_->calibStore() : nullptr;
                if (cs && cs->hasData()) {
                    Scanner::workflow::ScanCalibration c;
                    c.cameraMatrixL = cs->cameraMatrixL();
                    c.cameraMatrixR = cs->cameraMatrixR();
                    c.distCoeffsL   = cs->distCoeffsL();
                    c.distCoeffsR   = cs->distCoeffsR();
                    c.R1 = cs->R1();  c.R2 = cs->R2();
                    c.P1 = cs->P1();  c.P2 = cs->P2();  c.Q = cs->Q();
                    c.imageSize = cs->imageSize();
                    c.valid = true;
                    scanWf_->setCalibration(c);
                }
                auto r = scanWf_->initialize();
                if (!r.success) return r;
                return scanWf_->start();
            };
        }
        if (spec.name == "finish_scan") {
            // 触发型（§3.2 ⑦）：用户点「完成扫描/停止」仅点火收尾——S4/S5→S2 切态
            // 不在此（⑩ 合账后由 notifyCompleted 执行，02-D3/D4）。现状收尾=stop()
            // 回收合账；02-⑦ GBA 批算（GlobalOptimObject 消费 pipeline_->obs()）
            // TODO 接入期——enableFinalBA=false 时收尾语义不变（设计 §3.2 注）
            spec.handler = [this]() {
                if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
                return scanWf_->stop();
            };
        }
        if (spec.name == "start_postprocess") {
            // pre（§3.2 可选增强谓词「存在扫描产物」）：查 06 PointCloudBuffer
            // 原子点计数（getTotalPointCount——读写锁外的原子读，微秒级）。
            // 设计 §3.2 注：S6 进入前置=操作员全权，本谓词仅增强拦空跑；04
            // start() 内 makeCloudData 有同判据（空点云 fail），前置拦在切态前
            // 省一次 S2→S6→S2 失败往返
            spec.pre = [this]() {
                if (!wfCtx_ || !wfCtx_->pointCloudBuffer() ||
                    wfCtx_->pointCloudBuffer()->getTotalPointCount() <= 0)
                    return Scanner::Result::fail("无扫描产物——点云为空，请先完成扫描");
                return Scanner::Result::ok();
            };
            // 点火语义：initialize + start——start() 内快照点云/装配 07 E 后即启
            // postThread_ 阻塞批算，handler 毫秒级即返；同步失败（点云空/07 装配
            // 失败）返回 fail 由 gate 回滚 S2（§3.3）。完成回报经 onFinished_
            // 合账（下方 postWf_ 装配处注入）。
            // finish_postprocess（触发型）handler 留空：后处理为离线批，跑完
            // 自然回报切 S2；S6 内中途停止走 04 stop() → 线程尾合账
            spec.handler = [this]() {
                if (!postWf_) return Scanner::Result::fail("后处理工作流未装配");
                auto r = postWf_->initialize();
                if (!r.success) return r;
                return postWf_->start();
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
    mcu_    = std::make_unique<Scanner::device::MCUDriver>();   // 波特率/端口语义归 open(port)（S-T5）

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
    // P5-T15 完成回报注入（§9 02-⑩）：工作流 stop() 活跃会话终止回调 → 合账切 S2；
    // app 是组合根，可同时触达 02 工作流与 10 门禁（02 自身不依赖 10）
    scanWf_->setOnFinished([this](bool ok) {
        commandGate_->notifyCompleted("start_scan", ok);
    });
    calibWf_ = std::make_unique<Scanner::workflow::CalibrationWorkflow>(wfCtx_.get());
    // P5-T14 完成回报注入（§9 01-⑨）：工作流 B 批算线程尾回调 → 合账切 S2；
    // app 是组合根，可同时触达 01 工作流与 10 门禁（01 自身不依赖 10）
    calibWf_->setOnFinished([this](bool ok) {
        commandGate_->notifyCompleted("start_calibration", ok);
    });
    postWf_  = std::make_unique<Scanner::workflow::PostProcessWorkflow>(wfCtx_.get());
    // P5-T16 完成回报注入（§9 04 行）：工作流 postThread_ 批算线程尾回调 →
    // 合账切 S2；app 是组合根，可同时触达 04 工作流与 10 门禁（04 不依赖 10）。
    // 现无 UI 入口触发后处理（04 为离线批，入口待 04 工作流产品化时接）——
    // handler 已备好，05/04 UI 落地后 submit("start_postprocess") 即通
    postWf_->setOnFinished([this](bool ok) {
        commandGate_->notifyCompleted("start_postprocess", ok);
    });

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
