// ============================================================================
// AppContext.cpp — 应用层装配实现
// ============================================================================

#include "AppContext.h"
#include "SceneFeedAdapter.h"
#include "FrameBuffer.h"
#include "PointCloudBuffer.h"
#include "DeviceStateCache.h"

#include <algorithm>
#include <cstdlib>              // std::getenv——JMW_SIM_MARKERS_PLY 数据集可写
#include <fstream>
#include <QTimer>
#include <nlohmann/json.hpp>
#include "sched/CpuTopology.h"   // 自检起点 CPU 拓扑记录（2026-09-01）
#include "pipelines/scan/SimScanSource.h"   // 中段模拟提取源（UI 开关组装）
#include "file_io.h"             // fileio::importPLY（激光数据集文件装载）
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
#include "modules/08_devicemgmt/CameraFactory.h"   // 相机工厂契约（实现细节收在 08）
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
    // 标定档案归 config/（2026-09-06 口径：相机 calibration.json＋同目录
    // laser_calib.json 工厂档自动合并——CalibrationRepository 按父目录找）
    if (const auto lr = calibRepo_->load("config/calibration.json"); !lr.success)
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

    // 08 故障桥（P0-2，10 文档 §2.4 待办②）：外部 Error 级 FaultOccurred（08 相机/串口/
    // 温度等）→ 完整故障链 S7 安全停机。EventBus 同步分发持总线锁——锁内不得调
    // transition（StateChanged publish 重入死锁），故 QTimer 延后到主线程事件循环执行。
    // 档层记档仍由 FaultHandler 订阅完成；本桥只补「转态＋safeStop＋红灯」链段。
    faultBridgeSubId_ = eventBus_->subscribe(Scanner::EventType::FaultOccurred,
        [this](const Scanner::Event& evt) {
            if (evt.param1 < static_cast<int64_t>(Scanner::FaultSeverity::Error)) return;
            const auto cur = stateMachine_
                ? stateMachine_->getCurrentState() : Scanner::service::SystemState::Init;
            if (cur == Scanner::service::SystemState::PostProcessing ||
                cur == Scanner::service::SystemState::FaultSelfCheck)
                return;  // 与 handler 口径一致：S6 保活/S7 已停机不重转
            QTimer::singleShot(0, this, [this] {
                if (!stateMachine_) return;
                if (stateMachine_->getCurrentState() ==
                    Scanner::service::SystemState::FaultSelfCheck)
                    return;  // 已转 S7（防多帧风暴重复转）
                if (stateMachine_->transition(Scanner::EventType::FaultOccurred).success) {
                    if (deviceManager_) deviceManager_->toIdle();  // safeStop 语义（对齐直调口）
                    if (eventBus_) {
                        Scanner::Event led;
                        led.type = Scanner::EventType::LedControl;
                        led.param1 = 1;  // 红
                        eventBus_->publish(led);
                    }
                    JMW_LOG_WARN("app-AppContext",
                                 "[AppContext] 故障桥：外部 Error 级故障→S7 安全停机（led 红）");
                }
            });
        });

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
            // 点火语义（装配后台化）：标定转接＋02 initialize＋start 全套在后台
            // 线程执行——handler 毫秒级即返（§3.3 契约兑现：装配数百 ms 不再卡
            // 点击路径）。后台失败 → 设备收口（灯灭/采集停）＋notifyCompleted(false)
            // 合账回滚 S2 ＋ scanSessionEndedHandler_(false) 复原 UI 按钮态。
            // ScanMode 不经 gate payload——UI 入口先 setScanMode 设进工作流
            spec.handler = [this]() {
                if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
                ++scanActGen_;                   // 会话代递增：作废更早装配线程的回滚资格
                if (scanStartThread_.joinable())
                    return Scanner::Result::fail("扫描正在启动中（后台装配未完）");
                const uint64_t gen = scanActGen_.load();
                scanStartThread_ = std::thread([this, gen]() {
                    Scanner::Result r = Scanner::Result::fail("未知");
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
                            c.P1 = st.P1;  c.P2 = st.P2;
                            c.Q = st.Q;
                            c.imageSize = st.imageSize;
                            c.valid = true;
                            // 中段模拟提取（调试件）：UI「模拟数据」开关——置位则
                            // 本会话装配模拟源（真机前端照常，提取结果替换；配准/
                            // 融合/体素/渲染走生产代码）
                            if (simExtract_.load(std::memory_order_relaxed)) {
                                assembleSimSource();
                            } else {
                                simSource_.reset();
                            }
                            scanWf_->setSimSource(simSource_.get());
                            scanWf_->setCalibration(c);
                            r = scanWf_->initialize();
                            if (r.success) r = scanWf_->start();
                        } else {
                            r = Scanner::Result::fail("标定仓库数据为空——请先完成标定");
                        }
                    } else {
                        r = Scanner::Result::fail("无标定仓库");
                    }
                    if (!r.success) {
                        JMW_LOG_ERROR("app-AppContext",
                            "[AppContext] 扫描后台装配失败（已回滚待机）: {}", r.message);
                        // 代守卫（260911 停启竞态）：装配耗时数百 ms，期间用户可能
                        // 已停（finish_scan）再点新 start——无条件 stopCapture 会扑杀
                        // 新会话（预览冻结观感）。仅当代未变（本次装配仍是最新动作）
                        // 才收口硬件；过期=有更新动作接管，本线程不动手
                        if (scanActGen_.load() == gen) {
                            if (deviceManager_) {      // 设备收口（灯/采集已点，须收回）
                                deviceManager_->lightsAllOff();
                                deviceManager_->stopCapture();
                            }
                            commandGate_->notifyCompleted("start_scan", false);
                            if (scanSessionEndedHandler_) scanSessionEndedHandler_(false);
                        } else {
                            JMW_LOG_INFO("app-AppContext",
                                "[AppContext] 装配失败回滚跳过（会话代已前进——新会话已接管）");
                        }
                    } else {
                        JMW_LOG_INFO("app-AppContext", "[AppContext] 扫描后台装配完成{}",
                                     simSource_ ? "（中段模拟提取=开）" : "");
                    }
                });
                return Scanner::Result::ok("扫描启动中（后台装配）");
            };
        }
        if (spec.name == "finish_scan") {
            // 完成语义§3.2 高：用户点「扫描/停止」扫尾批S4/S5，S2 静态
            // 不在此。收尾链=stop()收尾链（02-D3/D4）：首次尾批=stop()
            // 收尾链（02-⑦ GBA 终局批（GlobalOptimObject 消费 pipeline_->obs()）。
            // 260920 后台化：GBA 千帧级 Ceres 分钟耗时——UI 线程同步执行曾
            // 鼠标转圈死等；改后台线程执行 stop()（含终局遍），完成后
            // notifyCompleted 合账（模式同 start_scan 装配后台化§3.3）
            spec.handler = [this]() {
                if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
                ++scanActGen_;                   // 会话代递增：作废更早线程的收口资格
                const uint64_t gen = scanActGen_.load();
                if (finishThread_.joinable()) finishThread_.join();   // 旧收尾应已完成（防御）
                scanWf_->setFinalBAProgress(finalBAProgress_);
                finishThread_ = std::thread([this, gen]() {
                    const bool ok = scanWf_->stop().success;
                    if (scanActGen_.load() != gen) {
                        JMW_LOG_INFO("app-AppContext",
                            "[AppContext] 完成收口跳过（会话代已前进——新动作已接管）");
                        return;
                    }
                    commandGate_->notifyCompleted("finish_scan", ok);
                });
                return Scanner::Result::ok("完成中（终局优化后台执行）");
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
    // HardwareMonitor 为巡检件留本层——门面不管它。相机经 08 工厂契约构造
    // ——实现类/配置细节不漏到装配根（跨层封装收口 2026-09-05）。
    // 装机口径自 config/camera.json（缺文件用内置默认＋WARN））
    struct CameraSetupCfg {
        int deviceIndexLeft = 0;
        int deviceIndexRight = 1;
        bool rotateRight180 = true;
        std::string triggerSource = "Line2";
        int previewFps = 10;
    };
    CameraSetupCfg camCfg;
    {
        std::ifstream f("config/camera.json");
        if (!f.is_open()) {
            JMW_LOG_WARN("app-AppContext",
                         "[AppContext] config/camera.json 不存在——相机装机口径用内置默认（exe 目录 config/ 下可覆盖）");
        } else {
            try {
                nlohmann::json j;
                f >> j;
                const auto& c = j.at("camera");
                camCfg.deviceIndexLeft = c.value("deviceIndexLeft", camCfg.deviceIndexLeft);
                camCfg.deviceIndexRight = c.value("deviceIndexRight", camCfg.deviceIndexRight);
                camCfg.rotateRight180 = c.value("rotateRight180", camCfg.rotateRight180);
                camCfg.triggerSource = c.value("triggerSource", camCfg.triggerSource);
                camCfg.previewFps = c.value("previewFps", camCfg.previewFps);
                JMW_LOG_INFO("app-AppContext",
                             "[AppContext] camera.json 已载：L={} R={} rot180={} trig={} previewFps={}",
                             camCfg.deviceIndexLeft, camCfg.deviceIndexRight,
                             camCfg.rotateRight180, camCfg.triggerSource, camCfg.previewFps);
            } catch (const std::exception& e) {
                JMW_LOG_WARN("app-AppContext",
                             "[AppContext] camera.json 解析失败（{}）——用内置默认", e.what());
            }
        }
    }
    cameraPreviewFps_ = std::max(1, camCfg.previewFps);
    Scanner::device::DeviceConfig devCfg;
    devCfg.serialPort = "auto";   // 串口自动搜（MCUDriver 逐口发 N12 T100 等 G 帧凭据认定）；固定口填 "COMx"
    // baud 115200 固定（260831 唯一口径：裸';' 分帧无版本号，DeviceConfig 缺省即产线口径）
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
            // 装机口径自 config/camera.json（捕获值构造）
            return Scanner::device::createGalaxyStereoCamera(
                camCfg.deviceIndexLeft, camCfg.deviceIndexRight,
                camCfg.rotateRight180, camCfg.triggerSource);
        });
    // 设备启动（open+自检）后台化：此处不再阻塞主窗口——main 在 window.show() 后
    // 调 startDevicesAsync()（相机枚举+自动搜口实测 ~5s，同步跑=白屏等）
    JMW_LOG_INFO("app-AppContext", "[AppContext] 组件装配完成（设备启动转后台 startDevicesAsync）");

    selfCheckCollector_ = std::make_unique<Scanner::device::SelfCheckCollector>();
    hwMonitor_ = std::make_unique<Scanner::device::HardwareMonitor>();
    hwMonitor_->setDeviceStateSink(deviceStateCache_.get());   // IDeviceStateSink*（隐式上转）
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
    // 灯态策略：启动/开门面不亮灯；startCapture（开始扫描）亮（N10 按模式四管
    // 掩码组帧）、stopCapture（停扫描）熄（N11 H0）。UI 滑条空闲仅记账，采集中
    // 改值随全参重发。

    // 启动 HardwareMonitor（始终运行，周期采集设备状态）
    hwMonitor_->start(1000);
    JMW_LOG_INFO("app-AppContext", "[AppContext] HardwareMonitor 已启动");
}

void AppContext::startDevicesAsync() {
    if (devStartThread_.joinable()) return;     // 已起（幂等）
    devStartThread_ = std::thread([this] {
        // 自检起点：记录 CPU 拓扑（用户口径 2026-09-01——并行度审计基线）
        {
            const auto topo = Scanner::pipeline::sched::CpuTopology::detect();
            const int lanes = Scanner::pipeline::sched::computeLanes(topo.pCores, topo.eCores, 0);
            JMW_LOG_INFO("app-AppContext",
                         "[自检] CPU: 物理核={} 逻辑核={} P核={} E核={} 混合架构={} → lanes={}",
                         topo.pCores + topo.eCores,
                         std::thread::hardware_concurrency(),
                         topo.pCores, topo.eCores, topo.hybrid, lanes);
        }
        const auto devR = deviceManager_->open();   // 相机枚举→MCU 自动搜口（N12 T100
        // 探测）→上行接线→参数装载→N12 定版→逻辑线程；不预亮灯（启采=N10 才亮）
        JMW_LOG_INFO("app-AppContext", "[AppContext] DeviceManager open: {}", devR.success ? "ok" : devR.message);

        notifySelfCheckItem("serialPort", devR.success);
        notifySelfCheckItem("license", true);   // 占位（狗到货接实检）

        if (devR.success) {
            // 设备状态落缓存（UI 连接状态显示源——2026-09-01：此前全工程无
            // 写入者，"未连接"恒定；app 桥接 08→06：相机/MCU open 成功即 Connected）
            if (deviceStateCache_) {
                const auto now = static_cast<Scanner::TimestampMs>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                Scanner::data::DeviceStateInfo cam;
                cam.deviceId = "Camera"; cam.deviceType = "ScannerCamera";
                cam.state = Scanner::DeviceState::Connected; cam.timestamp = now;
                deviceStateCache_->pushState(cam);
                Scanner::data::DeviceStateInfo mcu;
                mcu.deviceId = "MCU"; mcu.deviceType = "MCU";
                mcu.state = Scanner::DeviceState::Connected; mcu.timestamp = now;
                deviceStateCache_->pushState(mcu);
                JMW_LOG_INFO("app-AppContext", "[AppContext] 设备状态落缓存: Camera/MCU=Connected");
            }
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
    const auto t0 = std::chrono::steady_clock::now();   // 启停耗时打点（分段排查）
    {
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
            const double tempC = (t.ts > 0) ? t.celsius[0] : 25.0;   // 260831：G02 恒 4 路（ts=0=未收帧→25℃ 缺省档）
            scanWf_->pushSessionFrame(frame.leftGray, frame.rightGray, tempC, frame.frameId);
        }
        // ③ 调试分路：相机预览监视弹窗（相机 SDK 线程直调；订阅方切线程+节流自理）
        // 260912 终审证据：tap 空（监视窗未挂/已关）vs tap 心跳（活着）双路日志
        // ——帧在流而画面停时，一查日志即知死在哪一环（每 300 帧≈5s 一条）
        {
            std::lock_guard<std::mutex> lock(debugTapMtx_);
            if (debugFrameTap_) {
                debugFrameTap_(frame);
                if (++debugTapFrames_ % 300 == 0)
                    JMW_LOG_INFO("app-AppContext", "[预览] tap 心跳：累计 {} 帧",
                                 debugTapFrames_.load());
            } else if (++debugTapNullCnt_ % 300 == 0) {
                JMW_LOG_WARN("app-AppContext",
                             "[预览] 帧在流但 tap 空（监视窗未挂/已关闭）——累计 {} 帧",
                             debugTapNullCnt_.load());
            }
        }
    });
    // 灯型归采集组 N10（effectiveN10 按模式组装四管掩码）——不预点亮：固件 H1
    // "按上次采集参数重启"会重置灯态（2026-08-22 实测），组序内 N10 灯型一次到位。
    // 协议 260831 七参恢复（260919）：T/V/C/D=四管开关按 ScanMode 映射（A=纯补光；
    // B=T1V1 交叉；精细=D 管；深孔=C 管）——五参帧固件不解析=面片无激光线根因
    dm->startCapture(mode);
    JMW_LOG_INFO("app-AppContext",
        "[AppContext] 采集启动（N10 七参·四管掩码，mode={}）", static_cast<int>(mode));
    }   // ← 设备段结束（真机前端——中段模拟提取时设备照常采集，提取结果在 07 链内替换）

    // 命令通道点火（门禁/前置/装配失败均带因返回；各"不走打印点"已落日志）
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    scanWf_->setScanMode(mode);
    lastScanMode_ = mode;                       // 就绪态续采重启灯组依据（P3）
    const char* modeName = mode == Scanner::ScanMode::MarkerOnly      ? "标点扫描(A)"
                           : mode == Scanner::ScanMode::MarkerPlusLaser ? "面片扫描(B)"
                           : mode == Scanner::ScanMode::FineScan        ? "精细扫描(C)"
                                                                        : "深孔扫描(D)";
    auto gr = commandGate_->submit("start_scan", static_cast<int64_t>(mode));
    if (!gr.success) {
        JMW_LOG_WARN("app-AppContext", "[AppContext] {} 点火被拒: {}", modeName, gr.message);
        return gr;
    }
    {
        const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        JMW_LOG_INFO("app-AppContext", "[AppContext] ▶启动同步段完成 t+{}ms（灯命令已编队/装配转后台）", el);
    }
    return Scanner::Result::ok(modeName);
}

Scanner::Result AppContext::stopScanSession() {
    const auto t0 = std::chrono::steady_clock::now();   // 启停耗时打点（分段排查）
    JMW_LOG_INFO("app-AppContext", "[AppContext] ■停止点击 t+0ms");
    // 装配中防竞态：先等后台装配线程收尾，再走停止链——防 stop() 与 start()
    // 并发操作同一 pipeline_
    if (scanStartThread_.joinable()) scanStartThread_.join();
    if (deviceManager_) {
        // 停止只发单帧 N11 H0（灭灯，stopCapture 内）——不发 N10（用户口径
        // 2026-08-30：停止帧极简，lightsAllOff 的 N10 B0/L0 已删）
        deviceManager_->stopCapture();       // 设备侧采集停（幂等）
    }
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    auto r = commandGate_->submit("finish_scan");          // 工作流合账（handler=stop）
    if (simSource_) {           // 中段模拟提取源随会话终了弃（下会话按开关重组）
        simSource_.reset();
        scanWf_->setSimSource(nullptr);
    }
    {
        const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        JMW_LOG_INFO("app-AppContext", "[AppContext] ■停止同步段完成 t+{}ms（合账={}）", el,
                     r.success ? "ok" : r.message);
    }
    return r;
}

bool AppContext::isScanSessionActive() const {
    if (!scanWf_) return false;
    using Scanner::workflow::WorkflowState;
    const auto st = scanWf_->getState();
    return st == WorkflowState::Running || st == WorkflowState::Paused;
}

// ============================================================================
// 就绪态（05 D2/D8，实施计划 P3）——停采集保活会话：编辑会话的数据源基础
// ============================================================================
int AppContext::cameraMeasuredFps() const {
    return deviceManager_ ? deviceManager_->measuredCameraFps() : 0;
}

Scanner::Result AppContext::pauseScanSession() {
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    if (!isScanSessionActive()) return Scanner::Result::fail("无活跃扫描会话");
    if (isScanSessionPaused()) return Scanner::Result::ok("已处于就绪态");
    // 停采集保活（用户口径 2026-09-06）：N11 H0 停触发＋灭灯（协议正确口径）
    //——相机流保留（旧协议 stopCapture 不停流），管线暂停自丢帧
    if (deviceManager_) deviceManager_->stopCapture();
    const auto r = scanWf_->pause();
    JMW_LOG_INFO("app-AppContext", "[AppContext] 就绪态（N11H0 停触发灭灯·相机流保留）：{}（融合云/obs 账本保留）",
                 r.success ? "ok" : r.message);
    return r;
}

Scanner::Result AppContext::resumeScanSession() {
    if (!scanWf_) return Scanner::Result::fail("扫描工作流未装配");
    if (!isScanSessionPaused()) return Scanner::Result::fail("非就绪态（无暂停会话）");
    const auto r = scanWf_->resume();
    if (!r.success) return r;
    // 续采：N10 重启灯组参数＋N11 H1 重启触发（2026-09-06 实测：首次启动
    // N10 即够——MCU 从默认态进入触发；但 N11H0 停触发后仅 N10 不够，
    // MCU 停在「已停」态——串口无声根因；须补 N11H1 才恢复触发）
    if (deviceManager_) {
        deviceManager_->startCapture(lastScanMode_);   // N10→FLUSH（captureSeqSteps·模式掩码）
    }
    JMW_LOG_INFO("app-AppContext", "[AppContext] 续采（N10 参数＋N11H1 重启触发）：ok");
    return Scanner::Result::ok("续采中");
}

bool AppContext::isScanSessionPaused() const {
    if (!scanWf_) return false;
    return scanWf_->getState() == Scanner::workflow::WorkflowState::Paused;
}

bool AppContext::canEnterEditSession() const {
    // app 级门禁（10 状态机就绪态扩展另行）：就绪态＋标志点融合云非空
    if (!isScanSessionPaused()) return false;
    if (!sceneFeed_) return false;
    return !sceneFeed_->latestMarkers().empty();
}

void AppContext::shutdown() {
    // once 守卫：main 显式调用后，对象析构（及任何迟到路径）不再重复走关闭序列。
    // 实证（jmw_2026-08-29 日志）：无守卫时退出链跑了 3+ 轮 shutdown，末轮
    // DeviceManager::close 挂死致进程不退（cmd 窗口残留）——相机 SDK 的二次
    // 关闭路径不可依赖。首轮在 main 线程、时序确定，一轮即止
    if (shutdownDone_.exchange(true, std::memory_order_acq_rel)) return;
    if (scanStartThread_.joinable()) scanStartThread_.join();   // 装配线程收尾再关（防竞态 stop 误态）
    if (finishThread_.joinable()) finishThread_.join();         // 完成收尾线程（GBA 终局遍）先收
    if (devStartThread_.joinable()) devStartThread_.join();   // 设备启动收尾再关（防竞态）
    if (hwMonitor_) hwMonitor_->stop();
    if (faultBridgeSubId_ != 0 && eventBus_)
        eventBus_->unsubscribe(faultBridgeSubId_);
    faultBridgeSubId_ = 0;
    if (scanWf_)    scanWf_->stop();
    simSource_.reset();                                          // 模拟源随后者弃（lane 已 join）
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

// ============================================================================
// 中段模拟提取（调试件；主界面「模拟数据」开关——start_scan 时读取组装）
// ============================================================================

// ASCII PLY 限量读取（前 maxPts 点——pointcloud.ply 3100 万点大文件专用；
// 06 importPLY 对该量级文件实测失败回退，故装配层自持读取器）
namespace {
bool loadAsciiPlyLimited(const std::string& path, size_t maxPts,
                         std::vector<cv::Point3f>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    long long vertexCount = 0;
    bool headerEnd = false;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "end_header") { headerEnd = true; break; }
        if (line.compare(0, 14, "element vertex") == 0)
            vertexCount = std::stoll(line.substr(15));
    }
    if (!headerEnd || vertexCount <= 0) return false;
    const size_t n = std::min<size_t>(maxPts, static_cast<size_t>(vertexCount));
    out.clear();
    out.reserve(n);
    char buf[1 << 16];
    for (size_t i = 0; i < n && f.good(); ) {
        f.read(buf, sizeof buf);
        const std::streamsize got = f.gcount();
        if (got <= 0) break;
        std::streamsize pos = 0;
        while (pos < got && i < n) {
            const char* nl = static_cast<const char*>(
                memchr(buf + pos, '\n', static_cast<size_t>(got - pos)));
            if (!nl) break;                          // 半行留待下块（简化：丢弃头部残行）
            float x = 0, y = 0, z = 0;
            if (sscanf(buf + pos, "%f %f %f", &x, &y, &z) == 3) {
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
                    out.emplace_back(x, y, z);
                ++i;
            }
            pos = static_cast<std::streamsize>(nl - buf) + 1;
        }
    }
    return !out.empty();
}
} // namespace

// 模拟数据集·激光档主源读取（260919 用户口径）：「加载点云」按钮同源数据
// D:/pointcloud_100M.ply 的**前 3000 万点**——binary LE、15B/点（3×f32 xyz＋
// 3×u8 rgb，同 OSGWidget::loadTestDataFromPLY 的流式口径与截断上限）
namespace {
bool loadPointCloud100MLimited(const std::string& path, size_t maxPts,
                               std::vector<cv::Point3f>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::vector<char> sbuf(1 << 20, '\0');
    f.rdbuf()->pubsetbuf(sbuf.data(), static_cast<std::streamsize>(sbuf.size()));
    std::string line;
    long long vertexCount = 0;
    bool binaryLE = false, headerEnd = false;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "end_header") { headerEnd = true; break; }
        if (line.compare(0, 14, "element vertex") == 0)
            vertexCount = std::stoll(line.substr(15));
        if (line.find("binary_little_endian") != std::string::npos) binaryLE = true;
    }
    if (!headerEnd || vertexCount <= 0 || !binaryLE) return false;
    const size_t n = std::min<size_t>(maxPts, static_cast<size_t>(vertexCount));
    constexpr size_t kBytesPerPoint = 15;        // 3×f32 + 3×u8
    out.clear();
    out.reserve(n);
    std::vector<char> buf(1 << 20, '\0');
    size_t got = 0;
    while (got < n) {
        const size_t batch = std::min<size_t>(n - got, buf.size() / kBytesPerPoint);
        f.read(buf.data(), static_cast<std::streamsize>(batch * kBytesPerPoint));
        const size_t rd = static_cast<size_t>(f.gcount()) / kBytesPerPoint;
        if (rd == 0) break;
        for (size_t i = 0; i < rd; ++i) {
            float x, y, z;
            std::memcpy(&x, buf.data() + i * kBytesPerPoint, 4);
            std::memcpy(&y, buf.data() + i * kBytesPerPoint + 4, 4);
            std::memcpy(&z, buf.data() + i * kBytesPerPoint + 8, 4);
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
                out.emplace_back(x, y, z);
        }
        got += rd;
    }
    return !out.empty();
}
} // namespace
void AppContext::assembleSimSource() {
    // 场景组装：标志点←PLY（JMW_SIM_MARKERS_PLY 覆写，缺省 D:/markers_30.ply）；
    // 激光←「导入点云」stash；两者缺一 builtin fallback（保链路可用，warn 留痕）
    Scanner::pipeline::SimScene scene;
    const char* plyEnv = std::getenv("JMW_SIM_MARKERS_PLY");
    const std::string plyPath = plyEnv && *plyEnv ? std::string(plyEnv)
                                                  : std::string("D:/markers_30.ply");
    auto mr = Scanner::pipeline::SimScene::loadMarkersFromPly(plyPath, scene);
    if (!mr.success)
        JMW_LOG_WARN("app-AppContext", "[SimScan] 标志点 PLY 不可用（{}）——builtin 兜底: {}",
                     mr.message, plyPath);
    scene.laser = lastImportedCloud_;
    if (scene.laser.empty()) {
        // 模拟数据集·激光档（260919 用户口径）：主源＝「加载点云」按钮同源
        // D:/pointcloud_100M.ply 前 3000 万点；JMW_SIM_LASER_PLY 覆写走文本
        // PLY；均无才退 builtin
        std::vector<cv::Point3f> laserPts;
        std::string srcDesc;
        const char* laserEnv = std::getenv("JMW_SIM_LASER_PLY");
        if (laserEnv && *laserEnv) {
            if (Scanner::data::fileio::importPLY(laserEnv, laserPts) && !laserPts.empty())
                srcDesc = laserEnv;
        } else {
            // 主源＝pointcloud_100M.ply 前 3000 万点（260919 回退定版：该版本
            // 标志点/激光显示正常——终值 ~87 万为该窄带数据的 0.5mm 唯一格全量，
            // 属扫完非卡死。pointcloud.ply 米级场景版显示异常未解，仅经
            // JMW_SIM_LASER_PLY 覆写可用）
            if (loadPointCloud100MLimited("D:/pointcloud_100M.ply", 30000000,
                                          laserPts)) {
                srcDesc = "D:/pointcloud_100M.ply 前 3000 万点";
            }
        }
        if (!laserPts.empty()) {
            scene.laser = std::move(laserPts);
            JMW_LOG_INFO("app-AppContext", "[SimScan] 激光数据集文件装载: {} 点 ← {}",
                         scene.laser.size(), srcDesc);
        }
    }
    if (scene.laser.empty())
        JMW_LOG_WARN("app-AppContext",
            "[SimScan] 导入 stash 与数据集文件均空（先「文件管理→导入点云」或置 "
            "JMW_SIM_LASER_PLY）——builtin 兜底");
    if (scene.markers.empty()) {
        const auto fb = Scanner::pipeline::SimScene::builtin();
        scene.markers = fb.markers;
    }

    // 标志点包围盒（激光源组装/对齐基准）
    float mnx = scene.markers[0].x, mxx = mnx;
    float mny = scene.markers[0].y, mxy = mny;
    float mnz = scene.markers[0].z, mxz = mnz;
    for (const auto& m : scene.markers) {
        mnx = std::min(mnx, m.x); mxx = std::max(mxx, m.x);
        mny = std::min(mny, m.y); mxy = std::max(mxy, m.y);
        mnz = std::min(mnz, m.z); mxz = std::max(mxz, m.z);
    }

    if (scene.laser.empty()) {
        // builtin 兜底=覆盖标志点包围盒**全程**的大平面（生成于标志点区，无需
        // rebase）。验收口径「扫到哪看到哪」：激光源必须大于视场窗（400mm 深度
        // 处约 373×291mm）——旧 200×200 内置平面一帧全入窗=整片一次性出现
        //（260919 用户口径不符）。2.5mm 网格＋0.25 偏移钉 0.5mm 体素中心、微抖
        // ±0.02（对齐 SimScene::builtin 口径）；z 贴板前 2.25mm
        const float xc = (mnx + mxx) / 2.0f;
        const float halfW = (mxx - mnx) / 2.0f + 60.0f;
        const float yLo = mny - 60.0f, yHi = mxy + 60.0f;
        const float zc = (mnz + mxz) / 2.0f + 2.25f;
        std::mt19937 rng(2609);
        std::uniform_real_distribution<float> jit(-0.02f, 0.02f);
        for (float y = yLo + 1.25f; y < yHi; y += 2.5f)
            for (float x = xc - halfW + 1.25f; x < xc + halfW; x += 2.5f)
                scene.laser.emplace_back(x + jit(rng), y + jit(rng), zc + jit(rng));
        JMW_LOG_INFO("app-AppContext",
            "[SimScan] builtin 激光大平面: {} 点（{:.0f}×{:.0f}mm @({:.0f},{:.0f},{:.0f})）",
            scene.laser.size(), halfW * 2, yHi - yLo, xc, (mny + mxy) / 2.0f, zc);
    } else {
        // 导入点云 rebase：**中位数对齐**（260919 用户口径「同帧点云区域与标志点
        // 区域一致」：bbox 中心受远场离群点牵引——真扫描云含杂点，中心偏到空区
        // →标志点带内云点稀疏、激光线落在远处。中位数＝云主体中心，稳健；形状
        // 不变仅平移，z 贴板前 2.25mm）。单轴缓冲复用控内存
        std::vector<float> axis(scene.laser.size());
        auto medianAxis = [&](int c) {
            for (size_t i = 0; i < scene.laser.size(); ++i) {
                const auto& p = scene.laser[i];
                axis[i] = c == 0 ? p.x : (c == 1 ? p.y : p.z);
            }
            const size_t mid = axis.size() / 2;
            std::nth_element(axis.begin(), axis.begin() + static_cast<long>(mid), axis.end());
            return axis[mid];
        };
        const float lcx = medianAxis(0), lcy = medianAxis(1), lcz = medianAxis(2);
        const float dx = (mnx + mxx) / 2.0f - lcx;
        const float dy = (mny + mxy) / 2.0f - lcy;
        const float dz = (mnz + mxz) / 2.0f + 2.25f - lcz;
        for (auto& p : scene.laser) {
            p.x += dx;
            p.y += dy;
            p.z += dz;
        }
        JMW_LOG_INFO("app-AppContext",
            "[SimScan] 导入激光源中位数 rebase 至标志点区（{:+.1f},{:+.1f},{:+.1f}；{} 点）",
            dx, dy, dz, scene.laser.size());
    }

    // —— 标志点铺云（260919 终版口径：真实场景＝反光贴纸散布**整个工件表面**、
    //    扫描逐片覆盖工件——PLY 标志板布局与扫描云是两个独立场景，无论贴板/贴
    //    柱，区域都被标志板跨度框死（实测 ±120mm 带内仅 ~85 万点，采完即停滞，
    //    30M 云的其余部分永远扫不到）。改为从云上 **FPS 最远点采样 30 个铺展点**
    //    作标志点位置（数量保持 30、散布整个工件表面）：观测窗口沿 y 扫过哪些
    //    标志点、哪片表面逐帧长出来——与真实手持扫描一致。采样域=10 万随机
    //    子集（FPS 30×10 万=300 万次距离评估，后台秒级）
    if (!scene.laser.empty() && scene.laser.size() > 1000) {
        const size_t markerCount = scene.markers.empty() ? 30 : scene.markers.size();
        std::mt19937 rng(2609);
        // 深度护栏：只在工作距带（设备看向 -z，200~800mm）内采样——大跨度云
        // 的远端/背向点不当标志点位置
        std::vector<size_t> pick;
        pick.reserve(scene.laser.size());
        for (size_t i = 0; i < scene.laser.size(); ++i)
            if (scene.laser[i].z < -200.0f && scene.laser[i].z > -800.0f)
                pick.push_back(i);
        if (pick.size() < 1000) {                 // 带内过稀＝不铺（保留 PLY 布局）
            JMW_LOG_WARN("app-AppContext",
                "[SimScan] 工作距带内云点仅 {}——FPS 铺展跳过（保留原标志点布局）",
                pick.size());
        } else {
        // 鲁棒围栏（260919 破案：FPS 最远点采样偏爱离群点——采到 x=+81m 杂点，
        // 标志点包围球 86m→相机取景拉全远→真实场景缩成 1% 像素=「没有显示」。
        // 候选限各轴中位数 ±2500mm 内，主体场景内的远点才参与铺展）
        float pickMed[3] = {0, 0, 0};
        {
            const size_t PS = std::min<size_t>(50000, pick.size());
            std::vector<float> ax(PS);
            for (int c = 0; c < 3; ++c) {
                for (size_t i = 0; i < PS; ++i) {
                    const auto& p = scene.laser[pick[i]];
                    ax[i] = c == 0 ? p.x : (c == 1 ? p.y : p.z);
                }
                const size_t mid = PS / 2;
                std::nth_element(ax.begin(), ax.begin() + static_cast<long>(mid), ax.end());
                pickMed[c] = ax[mid];
            }
        }
        constexpr float kFenceXY = 2500.0f;       // mm（主体场景半径）
        constexpr float kFenceZ = 1500.0f;
        std::vector<size_t> fenced;
        fenced.reserve(pick.size());
        for (size_t idx : pick) {
            const auto& p = scene.laser[idx];
            if (std::abs(p.x - pickMed[0]) > kFenceXY ||
                std::abs(p.y - pickMed[1]) > kFenceXY ||
                std::abs(p.z - pickMed[2]) > kFenceZ)
                continue;
            fenced.push_back(idx);
        }
        if (fenced.size() >= 1000) pick.swap(fenced);
        const size_t S = std::min<size_t>(100000, pick.size());
        for (size_t i = 0; i < S; ++i) {          // 前 S 个洗牌＝随机子集（可复现）
            std::uniform_int_distribution<size_t> take(i, pick.size() - 1);
            std::swap(pick[i], pick[take(rng)]);
        }
        std::vector<float> bestD2(S, std::numeric_limits<float>::max());
        size_t cur = pick[0];                     // 起点＝随机子集首点
        std::vector<cv::Point3f> chosen;
        chosen.reserve(markerCount);
        for (size_t n = 0; n < markerCount; ++n) {
            const cv::Point3f& cp = scene.laser[cur];
            chosen.push_back(cp);
            size_t farIdx = 0;
            float farD2 = -1.0f;
            for (size_t i = 0; i < S; ++i) {
                const cv::Point3f& p = scene.laser[pick[i]];
                const float ddx = p.x - cp.x, ddy = p.y - cp.y, ddz = p.z - cp.z;
                const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (d2 < bestD2[i]) bestD2[i] = d2;
                if (bestD2[i] > farD2) { farD2 = bestD2[i]; farIdx = i; }
            }
            cur = pick[farIdx];                   // 下一锚＝离已选集最远点
        }
        scene.markers.assign(markerCount, calib::MarkerCloudPoint{});
        for (size_t i = 0; i < markerCount; ++i) {
            auto& m = scene.markers[i];
            m.x = chosen[i].x; m.y = chosen[i].y; m.z = chosen[i].z;
            m.nx = 0.0f; m.ny = 0.0f; m.nz = 1.0f;
            m.whiteRadius = 1.5f;
        }
        // 铺展后包围盒供日志（激光区域随观测窗口走，无需再 rebase——标志点
        // 已在云上）
        float snx = chosen[0].x, sxx = snx, sny = chosen[0].y, sxy = sny,
              snz = chosen[0].z, sxz = snz;
        for (const auto& c : chosen) {
            snx = std::min(snx, c.x); sxx = std::max(sxx, c.x);
            sny = std::min(sny, c.y); sxy = std::max(sxy, c.y);
            snz = std::min(snz, c.z); sxz = std::max(sxz, c.z);
        }
        JMW_LOG_INFO("app-AppContext",
            "[SimScan] 标志点铺云（FPS {} 点散布工件表面）: bbox X[{:.0f},{:.0f}] "
            "Y[{:.0f},{:.0f}] Z[{:.0f},{:.0f}]mm",
            markerCount, snx, sxx, sny, sxy, snz, sxz);
        }
    }

    // 模拟参数（260915 v3 部分观测）：视场窗 H50°/V40°＋深度窗＋纵向扫描行程
    //（视场窗沿 y 从场景底部扫到顶部）——标志点/激光逐帧进入视场，融合云渐进
    // 增长（复刻真实手持扫描「扫到哪看到哪」）；旋转/横移取小值保 sweep 主导
    Scanner::pipeline::SimTrajParams sp;
    // 递增观测调度（260919 用户定版「模拟真实扫描过程」）：帧0 观测 8 个标志点，
    // 每 15 帧一步：窗口前移 1（与上一步重合 ~W-1 个）＋每 2 步扩 1 → 8→9→10…
    // 逐帧增加；重合标志点驱动生产链光流配准逐帧链入全局锚。设备微动
    //（帧间 ~0.2mm << 匹配阈 2mm——旧 sweep 3.4mm/帧致配准逐帧全败，弃用）
    // 设备运动=0（260919 破案：7 路并行 lane 下 prevState 锚可落后当前帧 ~7 帧，
    // 0.25mm/帧 × 7 ≈ 1.75mm 卡在光流匹配阈 2mm 边缘→部分/全败→frame_fuse 兜底
    // 以 matched=0+RMSE130mm 垃圾"成功"→-1 标志点逐帧灌入融合云=乱序元凶。
    // 扫描推进由窗口调度承担，设备静态＋σ=0.05 噪声足够）
    sp.rotDegPerFrame = 0.0;
    sp.transMmPerFrame = 0.0;
    sp.markerNoiseSigmaMm = 0.05;
    sp.jitterSigmaMm = 0.05;
    sp.markerStepFrames = 5;           // 每 5 帧一步（260919：15 帧/步太慢——
                                       // 区域行程 ~4mm/帧，新点仅数百/帧。5 帧/步
                                       // ≈12~18mm/帧表面行程×行密度≈3~4 万新点/帧）
    sp.markerWindowStart = 8;          // 帧0 观测 8 个
    sp.markerGrowEvery = 1;            // 每步 +1（8→9→10…；到 30 后覆盖全云）
    sp.markerAdvancePerStep = 1;       // 与上一步重合 W-1 个
    sp.laserPerFrame = scene.laser.size() > 200000
                           ? 35000           // 导入大点云＝3~4 万点/帧（260919 用户
                                             // 口径；真机 25 线单帧量级同档）
                           : 4000;            // builtin：25 线/帧×~120 点/线（条带观测全量）
    sp.maxFrames = 2700;               // ~4.5min@10fps（260919：600 帧耗尽=点云
                                       // 冻结在 69.7 万点——延长扫描时长）
    sp.fovHalfHDeg = 25.0;             //（递增调度下 FOV 锥不再门控）
    sp.fovHalfVDeg = 20.0;
    sp.depthMinMm = 0.0;               // 深度窗放开（260919：250~800 真机工作距
    sp.depthMaxMm = 0.0;               // 把 30M 云的 ~97% 挡在域外=冻结 872,854
                                       // 的根因；模拟演示口径全域可达）
    sp.sweepY = false;                 // 递增调度取代几何扫掠
    simSource_ = std::make_unique<Scanner::pipeline::SimScanSource>(std::move(scene),
                                                                    std::move(sp));
    JMW_LOG_INFO("app-AppContext",
                 "[SimScan] 中段模拟提取源就绪（每帧观测替换提取结果，配准/融合走生产链）");
}
