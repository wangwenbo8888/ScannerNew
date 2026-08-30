// ============================================================================
// AppContext.cpp — 应用层装配实现
// ============================================================================

#include "AppContext.h"
#include "SceneFeedAdapter.h"
#include "FrameBuffer.h"
#include "PointCloudBuffer.h"
#include "DeviceStateCache.h"
#include "CalibrationRepository.h"
#include "StateMachine.h"
#include "ParameterManager.h"
#include "FaultHandler.h"
#include "CommandGate.h"
#include "DefaultCommands.h"
#include "PerfMonitor.h"
#include "jmw_logging.h"
#include "base/EventBus.h"
#include "modules/08_devicemgmt/DeviceManager.h"
#include "modules/08_devicemgmt/CameraControl.h"   // 门面相机工厂构造体（注入式）
#include "modules/08_devicemgmt/HardwareMonitor.h"
#include "modules/08_devicemgmt/SelfCheckCollector.h"
#include "WorkflowContext.h"
#include "ScanWorkflow.h"
#include "CalibrationWorkflow.h"
#include "PostProcessWorkflow.h"
#include <spdlog/spdlog.h>

namespace {
// A-T17：08→10 健康桥（08 不链 10——app 组合根适配，10 设计 P3）。poll() 驱动
// 挂 MainWindow 既有 m_infoTimer（1s；AppContext 无 Qt 依赖不持定时器）
struct HealthAdapter final : Scanner::service::IHealthProvider {
    Scanner::device::HardwareMonitor* hw = nullptr;
    explicit HealthAdapter(Scanner::device::HardwareMonitor* h) : hw(h) {}
    Scanner::HealthMetrics snapshot() const override {
        return hw ? hw->snapshot() : Scanner::HealthMetrics{};
    }
};
} // namespace

AppContext::AppContext() {}
AppContext::~AppContext() { shutdown(); }

void AppContext::initialize() {
    // === Infra ===
    eventBus_ = std::make_unique<Scanner::infra::EventBus>();

    // === Data ===
    frameBuffer_      = std::make_unique<Scanner::data::FrameBuffer>(60);
    pointCloudBuffer_ = std::make_unique<Scanner::data::PointCloudBuffer>();
    deviceStateCache_ = std::make_unique<Scanner::data::DeviceStateCache>();
    calibRepo_ = std::make_unique<Scanner::data::CalibrationRepository>();
    if (const auto lr = calibRepo_->load("calibration.json"); !lr.success)
        JMW_LOG_INFO("app-AppContext", "[AppContext] 启动未装载标定仓库档（{}）——首次标定后生成", lr.message);

    // === Service ===
    stateMachine_   = std::make_unique<Scanner::service::StateMachine>(eventBus_.get());
    paramManager_   = std::make_unique<Scanner::service::ParameterManager>();
    faultHandler_   = std::make_unique<Scanner::service::FaultHandler>(eventBus_.get());

    faultHandler_->setStateMachine(stateMachine_.get());
    // A-T17：安全停回调兑现 08 门面（:46 TODO）——FaultHandler 档案触发时设备回空闲
    faultHandler_->setSafeStopCallback([this] {
        if (deviceManager_) deviceManager_->toIdle();
    });
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
            // pre（§9 02-①）：参数就绪谓词——查 06 标定仓库 readyForScan（缺哪样报哪样）。
            // TODO(06 差距): 出口查表（逐温档 K/D + 激光温度表）接入后由 02 侧升级判据
            spec.pre = [this]() {
                auto* repo = wfCtx_ ? wfCtx_->calibRepo() : nullptr;
                if (!repo)
                    return Scanner::Result::fail("标定参数未就绪——标定仓库未装配");
                Scanner::data::ReadyReport rr;
                repo->readyForScan(rr);
                if (!rr.ready) {
                    std::string miss;
                    for (const auto& m : rr.missing)
                        miss += miss.empty() ? m : "/" + m;
                    return Scanner::Result::fail("标定参数未就绪——缺: " + miss);
                }
                return Scanner::Result::ok();
            };
            // 点火语义：标定仓库→ScanCalibration 静态转接（app=组合根做注入：
            // repo->stereo() 一次取结构体逐字段赋——01 写入/02 读取同源；
            // TODO 06 出口查表后逐温档 K/D/激光温度表归 02 侧自查）+ initialize +
            // start——帧处理归 07 内部线程，handler 毫秒级即返；同步失败（07 装配
            // 失败等）返回 fail 由 gate 回滚 S2（§3.3）。
            // ScanMode 不经 gate payload（handler 无参，payload 只喂 transition 的
            // S4/S5 判别）——UI 入口先 setScanMode 设进工作流（见 ScannerWindow）
            spec.handler = [this]() {
                if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
                auto* repo = wfCtx_ ? wfCtx_->calibRepo() : nullptr;
                if (repo) {
                    const auto st = repo->stereo();
                    if (!st.cameraMatrixL.empty() && !st.cameraMatrixR.empty()) {
                        Scanner::workflow::ScanCalibration c;
                        c.cameraMatrixL = st.cameraMatrixL;
                        c.cameraMatrixR = st.cameraMatrixR;
                        c.distCoeffsL   = st.distCoeffsL;
                        c.distCoeffsR   = st.distCoeffsR;
                        c.R1 = st.R1;  c.R2 = st.R2;
                        c.P1 = st.P1;  c.P2 = st.P2;  c.Q = st.Q;
                        c.imageSize = st.imageSize;
                        c.valid = true;
                        scanWf_->setCalibration(c);
                    } else {
                        return Scanner::Result::fail("标定仓库数据为空——请先完成标定");
                    }
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
    // IHealthProvider 注入移至 HAL 段之后（adapter 持 hwMonitor 裸指针——需其先在）

    // === HAL ===（A-T17 三行门面：设备对象【相机+MCU】收进 DeviceManager；
    // HardwareMonitor 为巡检件留本层——门面不管它）
    Scanner::device::StereoPairConfig camCfg;
    camCfg.deviceIndexLeft = 0;
    camCfg.deviceIndexRight = 1;
    camCfg.rotateRight180 = true;
    Scanner::device::DeviceConfig devCfg;
    devCfg.serialPort = "auto";   // 串口自动搜（MCUDriver 逐口发 N12Z1 探测应答认定）；固定口填 "COMx"
    // protocol 默认 V3、baud 115200（DeviceConfig 缺省即产线口径）
    deviceManager_ = std::make_unique<Scanner::device::DeviceManager>(
        devCfg,
        [this](const std::string& op) -> Scanner::Result {
            // 08 门禁回调 → 10 状态机映射（app=组合根；08 不反链 10）。
            // 口径：enter_scan/enter_calibration 问 SM；其余 op 一律放行（记账）
            if (!stateMachine_) return Scanner::Result::ok();
            const bool allow = (op == "enter_scan")       ? stateMachine_->canOperate("scan")
                             : (op == "enter_calibration") ? stateMachine_->canOperate("calibrate")
                             : true;
            return allow ? Scanner::Result::ok()
                         : Scanner::Result::fail("状态门禁拒绝: " + op);
        },
        eventBus_.get(),
        [camCfg]() -> std::unique_ptr<Scanner::hal::IScannerCamera> {
            return std::make_unique<Scanner::device::CameraControl>(camCfg);
        });
    // 设备启动（open+自检）后台化：此处不再阻塞主窗口——main 在 window.show() 后
    // 调 startDevicesAsync()（相机枚举+自动搜口实测 ~5s，同步跑=白屏等）
    JMW_LOG_INFO("app-AppContext", "[AppContext] 组件装配完成（设备启动转后台 startDevicesAsync）");

    selfCheckCollector_ = std::make_unique<Scanner::device::SelfCheckCollector>();
    hwMonitor_ = std::make_unique<Scanner::device::HardwareMonitor>();
    hwMonitor_->setDeviceStateCache(deviceStateCache_.get());
    hwMonitor_->setEventBus(eventBus_.get());
    // MCU 温度改门面快照注入（H-T16 口径）；相机行注入口无法保留——DeviceManager
    // 铁规不漏相机指针（遗留：08 侧后续增相机状态快照口）
    hwMonitor_->setLastTemps([this]() {
        return deviceManager_ ? deviceManager_->getLastTemperatures()
                              : Scanner::device::serial::TempFrame{};
    });
    // setHeartbeatCheck 留空（A-T17 口径）：串口无声判定已在 DeviceManager logicTick
    // 巡检（0x0802）——巡检件不重复判定
    hwMonitor_->setSelfCheck(selfCheckCollector_.get());

    // 10-PerfMonitor 健康源接线（hwMonitor 就绪后；poll 驱动在 MainWindow m_infoTimer）
    perfMonitor_->setProvider(std::make_shared<HealthAdapter>(hwMonitor_.get()));

    // === WorkflowContext 装配 ===
    wfCtx_ = std::make_unique<Scanner::workflow::WorkflowContext>();
    wfCtx_->setFrameBuffer(frameBuffer_.get());
    wfCtx_->setPointCloudBuffer(pointCloudBuffer_.get());
    wfCtx_->setDeviceStateCache(deviceStateCache_.get());
    wfCtx_->setCalibRepo(calibRepo_.get());
    wfCtx_->setStateMachine(stateMachine_.get());
    wfCtx_->setParameterManager(paramManager_.get());
    wfCtx_->setEventBus(eventBus_.get());
    // P2 渲染加固：SceneFeedAdapter（ISceneFeed 首个实现——流水线→渲染跨线程 marshal）
    sceneFeed_ = std::make_unique<SceneFeedAdapter>();
    wfCtx_->setSceneFeed(sceneFeed_.get());

    // === Workflow ===
    scanWf_  = std::make_unique<Scanner::workflow::ScanWorkflow>(wfCtx_.get());
    // P5-T15 完成回报注入（§9 02-⑩）：工作流 stop() 活跃会话终止回调 → 合账切 S2；
    // app 是组合根，可同时触达 02 工作流与 10 门禁（02 自身不依赖 10）
    scanWf_->setOnFinished([this](bool ok) {
        // 标志点点云落 06 仓库（A 模式的产物出口）：渲染适配器末次快照 →
        // markers 通道（续扫基准 seed 同源＋exportMarkers 可导）。不放 points
        // 通道——A 模式"不进后处理网格链"口径（getTotalPointCount 仍 0，
        // start_postprocess 的 pre 继续拦稀疏点误入网格）
        if (ok && pointCloudBuffer_ && sceneFeed_) {
            auto markers = sceneFeed_->latestMarkers();
            if (!markers.empty()) {
                pointCloudBuffer_->setMarkers(markers);
                JMW_LOG_INFO("app-AppContext",
                    "[AppContext] 标志点点云落库: {} 点（续扫基准/导出就绪）", markers.size());
            } else {
                JMW_LOG_WARN("app-AppContext",
                    "[AppContext] 扫描合账：无标志点可落库（会话内 0 次有效推送）");
            }
        }
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

    JMW_LOG_INFO("app-AppContext", "[AppContext] 全部组件装配完成");
    // 灯态策略：启动/开门面不亮灯；startCapture（开始扫描）亮（N10 账本全参）、
    // stopCapture（停扫描）熄（B0/L0）。UI 滑条空闲仅记账，采集中改值随全参重发。

    // 启动 HardwareMonitor（始终运行，周期采集设备状态）
    hwMonitor_->start(1000);
    JMW_LOG_INFO("app-AppContext", "[AppContext] HardwareMonitor 已启动");
}

void AppContext::startDevicesAsync() {
    if (devStartThread_.joinable()) return;     // 已起（幂等）
    devStartThread_ = std::thread([this] {
        const auto devR = deviceManager_->open();   // 相机枚举→MCU 自动搜口→N12Z1→逻辑线程
        JMW_LOG_INFO("app-AppContext", "[AppContext] DeviceManager open: {}", devR.success ? "ok" : devR.message);

        notifySelfCheckItem("serialPort", devR.success);
        notifySelfCheckItem("license", true);   // 占位（狗到货接实检）

        if (devR.success) {
            // 启动自检序列：mcuLink/bgLight/laser 闪灯回环 + 相机收帧（无阻塞状态机，
            // 逻辑线程 tick 驱动）。report 回调直投 notifySelfCheckItem（mutex+submit 线程安全）
            deviceManager_->startupSelfCheck([this](const std::string& key, bool ok) {
                notifySelfCheckItem(key, ok);
                std::lock_guard<std::mutex> lock(selfCheckMtx_);
                if (selfCheck_.reportedCount >= 6) selfCheck_.done = true;
            });
        } else {
            // 设备没开成功：余项判失败收尾（S1 卡住原因状态栏可见）
            notifySelfCheckItem("mcuLink", false);
            notifySelfCheckItem("bgLight", false);
            notifySelfCheckItem("laser", false);
            notifySelfCheckItem("camera", false);
            std::lock_guard<std::mutex> lock(selfCheckMtx_);
            selfCheck_.done = true;
        }
    });
}

// ============================================================================
// 扫描会话点火——统一入口（工具栏标点/面片扫描＋ScannerWindow 共用同一条真链）
// ============================================================================
Scanner::Result AppContext::startScanSession(Scanner::ScanMode mode) {
    auto* dm = deviceManager_.get();
    if (!dm || !dm->isDeviceReady()) {
        JMW_LOG_WARN("app-AppContext", "[AppContext] 扫描点火被拦: 设备未就绪（门面空/相机或 MCU 未开）");
        return Scanner::Result::fail("设备未就绪——请等待自检完成（相机/串口）");
    }
    // 帧流双投递注册（单槽语义：后注册生效）＋采集启动（N10 账本全参→N11H1→开流）
    dm->startFrameStream([this, dm](const Scanner::hal::StereoFrame& frame) {
        // ① 预览链：06 FrameBuffer（ScannerWindow 10fps 消费）
        if (frameBuffer_) {
            Scanner::data::FrameData fd;
            fd.frameId = frame.frameId;
            fd.timestamp = frame.timestamp;
            fd.leftGray = frame.leftGray;
            fd.rightGray = frame.rightGray;
            frameBuffer_->pushFrame(fd);
        }
        // ② 扫描链：02 会话环（enrich 出口查表→SlotRing；非扫描期该口自弃）
        if (scanWf_) {
            const auto t = dm->getLastTemperatures();
            const double tempC = (t.channels & 0x01) ? t.celsius[0] : 25.0;
            scanWf_->pushSessionFrame(frame.leftGray, frame.rightGray, tempC, frame.frameId);
        }
    });
    // 灯型归采集组 N10（effectiveN10 带灯型覆写）——不预点亮：固件 H1"按上次
    // 采集参数重启"会重置灯态（2026-08-22 实测），预点亮＝闪一下→灭→组内 N10
    // 再亮（真机"先闪一下再持续"根因）；组序 H1→N10，灯以组内 N10 为准一次到位
    const bool laserOn = (mode != Scanner::ScanMode::MarkerOnly);
    dm->startCapture(laserOn);
    if (laserOn) {
        JMW_LOG_INFO("app-AppContext",
            "[AppContext] 面片扫描激光组: 左斜=T{} 右斜=V{}（交替归固件 H1 帧序）",
            static_cast<int>(dm->getParam("laserSelectA").value),
            static_cast<int>(dm->getParam("laserSelectB").value));
    }

    // 命令通道点火（门禁/前置/装配失败均带因返回；各"不走打印点"已落日志）
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    scanWf_->setScanMode(mode);
    const auto modeName = mode == Scanner::ScanMode::MarkerOnly ? "标点扫描(A)" : "面片扫描(B)";
    auto gr = commandGate_->submit("start_scan", static_cast<int64_t>(mode));
    if (!gr.success) {
        JMW_LOG_WARN("app-AppContext", "[AppContext] {} 点火被拒: {}", modeName, gr.message);
        return gr;
    }
    JMW_LOG_INFO("app-AppContext", "[AppContext] {} 已点火", modeName);
    return Scanner::Result::ok(modeName);
}

Scanner::Result AppContext::stopScanSession() {
    if (deviceManager_) {
        // 灯命令先行：相机停流实测阻塞 ~2s，若排在前会把灭灯压到 2s+ 后——
        // 先 lightsAllOff（N10 即刻落总线下页灯灭，体感即时）再停采集/流
        deviceManager_->lightsAllOff();
        deviceManager_->stopCapture();       // 设备侧采集停（幂等）
    }
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    return commandGate_->submit("finish_scan");          // 工作流合账（handler=stop）
}

bool AppContext::isScanSessionActive() const {
    if (!scanWf_) return false;
    using Scanner::workflow::WorkflowState;
    const auto st = scanWf_->getState();
    return st == WorkflowState::Running || st == WorkflowState::Paused;
}

void AppContext::shutdown() {
    // once 守卫：main 显式调用后，对象析构（及任何迟到路径）不再重复走关闭序列。
    // 实证（jmw_2026-08-29 日志）：无守卫时退出链跑了 3+ 轮 shutdown，末轮
    // DeviceManager::close 挂死致进程不退（cmd 窗口残留）——相机 SDK 的二次
    // 关闭路径不可依赖。首轮在 main 线程、时序确定，一轮即止
    if (shutdownDone_.exchange(true, std::memory_order_acq_rel)) return;
    if (devStartThread_.joinable()) devStartThread_.join();   // 设备启动收尾再关（防竞态）
    if (hwMonitor_) hwMonitor_->stop();
    if (scanWf_)    scanWf_->stop();
    if (calibWf_)   calibWf_->stop();
    if (postWf_)    postWf_->stop();
    if (faultHandler_) faultHandler_->stop();
    if (deviceManager_) deviceManager_->close();   // 相机+MCU 收口（门面倒序关）
    JMW_LOG_INFO("app-AppContext", "[AppContext] 全部组件已关闭");
}

void AppContext::notifySelfCheckItem(const std::string& item, bool ok) {
    {
        std::lock_guard<std::mutex> lock(selfCheckMtx_);
        if      (item == "camera")     selfCheck_.camera     = ok;
        else if (item == "serialPort") selfCheck_.serialPort = ok;
        else if (item == "license")    selfCheck_.license    = ok;
        else if (item == "mcuLink")    selfCheck_.mcuLink    = ok;
        else if (item == "bgLight")    selfCheck_.bgLight    = ok;
        else if (item == "laser")      selfCheck_.laser      = ok;
        else return;
        ++selfCheck_.reportedCount;
    }
    JMW_LOG_INFO("app", "自检项 {}: {}", item, ok ? "通过" : "失败");
    if (!ok) {
        // 流程不走点：自检失败升级 warn（S1 卡点根因要可在日志直接检索）
        JMW_LOG_WARN("app", "[自检] 项 '{}' 失败——该链路不通，后续依赖项将受阻", item);
    }
    // 全过且仍处 S1 → 经命令通道切 S2（恰一次：submit 成功即离 S1，重复调用幂等失败）
    if (selfCheckAllPassed() &&
        stateMachine_->getCurrentState() == Scanner::service::SystemState::Init) {
        commandGate_->submit("system_ready");
    }
}

bool AppContext::selfCheckAllPassed() const {
    std::lock_guard<std::mutex> lock(selfCheckMtx_);
    return selfCheck_.camera && selfCheck_.serialPort && selfCheck_.mcuLink &&
           selfCheck_.bgLight && selfCheck_.laser && selfCheck_.license;
}

std::vector<std::pair<std::string, bool>> AppContext::selfCheckSnapshot() const {
    std::lock_guard<std::mutex> lock(selfCheckMtx_);
    return {{"通讯",   selfCheck_.serialPort && selfCheck_.mcuLink},
            {"加密狗", selfCheck_.license},
            {"补光灯", selfCheck_.bgLight},
            {"激光器", selfCheck_.laser},
            {"相机",   selfCheck_.camera},
            {"完成",   selfCheck_.done}};
}

bool AppContext::selfCheckDone() const {
    std::lock_guard<std::mutex> lock(selfCheckMtx_);
    return selfCheck_.done;
}
