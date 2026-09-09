// ============================================================================
// DeviceManager.cpp — 门面 + 逻辑线程实现（契约见 DeviceManager.h；协议 260831 批1-C）
// ============================================================================

#include "DeviceManager.h"

#include "KeySemantics.h"
#include "MCUDriver.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace Scanner::device {
namespace {

// 统一时基：system_clock 毫秒（与 MCUDriver 上行帧 ts 同域——Warmup tsMs/tick 共用）
int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
        .count();
}

constexpr size_t kPostQueueCap = 64;          // 编队队列容量（满丢新+warn——Critical #1）

// MarkerOnly（A 标点）灯型补光抬升基线：2026-08 真机标点检测成功配置（cpp 内散布
// 魔数归一于此；AppContext 日志文案/测试期望帧中的 40 为同源数值——注释指向本常量）
constexpr int kMarkerOnlyBg = 40;

// Fault 码表归头文件 DevFault；调用点统一经 code() 取值
constexpr int64_t code(DevFault f) { return static_cast<int64_t>(f); }

std::vector<ParamSpec> makeParamSpecs() {      // 参数字段定义归 08（红线）
    return {
        {"exposure", 10.0, 1.0, 100.0},        // 曝光 ms（相机直设）
        {"freqHz", 60.0, 1.0, 200.0},          // N10 H 拍照频率（协议域 1-200，批3 终态；
                                                //   UI 滑条量程已同步 1-200）
        {"bgLight", 10.0, 0.0, 100.0},         // N10 B 补光（默认 10——B 模式成功配置基线；
                                                //   MarkerOnly 灯型在 effectiveN10 抬升至 40）
        {"laserLevel", 40.0, 0.0, 100.0},      // N10 L 激光强度（默认 40；60 过亮→40 折中；
                                                //   旧档 >100 由 bootstrap 迁移钳 100）
        {"tempReportPeriodMs", 100.0, 5.0, 1000.0},  // N12 T 温度上报周期 ms（批3 入 specs；
                                                //   dispatch 走 N12，open 仍按 DeviceConfig 定版）
    };
}

// N10 生效参数（cpp 本地——不进头防 IMCU.h 类型泄漏）：freqHz 账本值；四激光管
// 按 ScanMode 掩码映射（D7 产线方案）——MarkerOnly T0V0C0D0 且 L=0、B 抬升 40
// （保留旧 A 模式补光抬升基线——2026-08 真机标点检测成功配置）；面片 T1V1C0D0；
// 精细（原点云扫描）T0V0C1D0；深孔 T0V0C0D1；其余模式 B/L 均账本值。
// ⑨b 周期模型待裁决：精细/深孔为单管周期（每帧仅 C 或 D），07 激光链「偶L奇R」
// 配对假设在单管下未验证——本函数仅做 08/UI 掩码映射就位。
// 调用点均在逻辑线程（捕获 this 直读成员）
hal::CaptureParams effectiveN10(const ParamStore& params, Scanner::ScanMode mode) {
    hal::CaptureParams p;
    p.freqHz = static_cast<int>(params.get("freqHz").value);
    p.bgLight = (mode == Scanner::ScanMode::MarkerOnly)
                    ? kMarkerOnlyBg                          // A 模式补光抬升基线（真机成功配置）
                    : static_cast<int>(params.get("bgLight").value);
    p.laserLevel = (mode == Scanner::ScanMode::MarkerOnly)
                       ? 0
                       : static_cast<int>(params.get("laserLevel").value);
    p.laserT = (mode == Scanner::ScanMode::MarkerPlusLaser) ? 1 : 0;
    p.laserV = (mode == Scanner::ScanMode::MarkerPlusLaser) ? 1 : 0;
    p.laserC = (mode == Scanner::ScanMode::FineScan) ? 1 : 0;
    p.laserD = (mode == Scanner::ScanMode::DeepHoleScan) ? 1 : 0;
    return p;
}

} // namespace

// ============================================================================
// 构造 / 析构
// ============================================================================

DeviceManager::DeviceManager(DeviceConfig cfg, GateQuery gate, infra::EventBus* bus,
                             CameraFactory camFactory, SerialWriteOverride serialWrite)
    : cfg_(std::move(cfg)),
      bus_(bus),
      camFactory_(std::move(camFactory)),
      writeOverride_(std::move(serialWrite)),
      mcu_(std::make_unique<MCUDriver>(writeOverride_)),
      menu_(std::make_unique<MenuLogic>()),
      mode_(std::make_unique<ModeController>(std::move(gate))),
      warmup_(std::make_unique<WarmupSequence>(cfg_.warmup)) {
    buildKeyActions();
    auto specs = makeParamSpecs();
    for (const auto& s : specs) paramKeys_.push_back(s.key);   // param1 索引线索
    params_ = std::make_unique<ParamStore>(
        std::move(specs),
        [this](const std::string& key, double v, ParamStore::Done done) {
            onParamDispatch(key, v, done);
        });
    params_->onParamChanged = [this](const std::string& key, const ParamEntry& e) {
        int64_t idx = -1;                       // 参数索引=specs 登记序号（Minor #9）
        for (size_t i = 0; i < paramKeys_.size(); ++i)
            if (paramKeys_[i] == key) idx = static_cast<int64_t>(i);
        publishEvent(EventType::UserDefined, idx, 0);   // base 暂无 ParamChanged——占位
        JMW_LOG_DEBUG("08-DeviceManager", "[DeviceManager] 参数改账 {}={:.3f}", key, e.value);
    };
    mode_->onChange = [this](DeviceMode oldM, DeviceMode newM) {
        publishEvent(EventType::StateChanged, static_cast<int64_t>(newM),
                     static_cast<int64_t>(oldM));
    };
    warmup_->onStable = [this] {
        auto cb = std::move(warmupDone_);
        if (cb) cb(true);
    };
    warmup_->onTimeout = [this] {
        publishFault(code(DevFault::WarmupTimeout),
                     "预热超时（只报不停加热——停止机制待协议方）");
        auto cb = std::move(warmupDone_);
        if (cb) cb(false);
    };
}

DeviceManager::~DeviceManager() { close(); }

// ============================================================================
// 编队队列（Critical #1：入队任意线程 / 消费逻辑线程）
// ============================================================================

void DeviceManager::post(std::function<void()> task) {
    if (!task) return;
    std::lock_guard<std::mutex> lock(postMutex_);
    if (postQueue_.size() >= kPostQueueCap) {   // 满丢新（调用方可重试）
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] 编队队列满(≥{})丢新", kPostQueueCap);
        return;
    }
    postQueue_.push_back(std::move(task));
}

void DeviceManager::drainPosts() {
    std::deque<std::function<void()>> tasks;
    {
        std::lock_guard<std::mutex> lock(postMutex_);
        tasks.swap(postQueue_);                 // 执行期间新入队归下一拍
    }
    for (auto& t : tasks) t();
}

// ============================================================================
// open 一条龙 / close 倒序
// ============================================================================

Result DeviceManager::open() {
    std::lock_guard<std::mutex> openLock(openMtx_);
    if (opened_) return Result::ok("设备已打开");
    const auto t0 = std::chrono::steady_clock::now();
    auto el = [t0]() { return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count(); };
    // 配置快照（一次性）——串口/周期/门限的运行依据；真机排障第一落脚点
    JMW_LOG_INFO("08-DeviceManager",
        "[DeviceManager] open 配置: 串口={} 波特率={} 温度周期={}ms "
        "心跳阈={}ms 温度上限={}℃ 乱跳(绝对差值)={}℃",
        cfg_.serialPort, cfg_.baud, cfg_.tempReportPeriodMs,
        cfg_.heartbeatTimeoutMs, cfg_.tempMaxC, cfg_.tempSpikeAbsC);
    // ① 相机（工厂缺省=不接相机）与 ② MCU 自动搜口并行——两链路无共享资源，
    //    串行白等（相机枚举 ~0.5s + 搜口 ~0.05s → 并行取大者）
    Result camR = Result::ok();
    std::thread camThread;
    if (camFactory_) {
        camThread = std::thread([this, &camR] {
            camera_ = camFactory_();
            const Result r = camera_->open();
            if (!r.success) {
                camera_.reset();
                camR = Result::fail("相机打开失败: " + r.message);
            } else {
                // 曝光保持相机自动（用户口径 2026-08-31：固定值试验 10/5/3ms
                // 均劣于自动——3ms+B30 时 ROI 暴增 137/76 触发 marker_match
                // maxPoints=100 上限异常，pChain 整帧报废）
            }
        });
    }
    // ② MCU（测试 writeOverride 模式=逻辑开，不开真串口不起 rx 线程；真机
    //    auto 搜口=发 N12 T100 等 300ms 收任意帧——探测命令面同温度周期）
    const Result rm = mcu_->open(cfg_.serialPort, cfg_.baud);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: MCU 链路完成 {}ms（{}）", el(), rm.message);
    if (camThread.joinable()) camThread.join();
    if (!camR.success) {
        publishFault(code(DevFault::CameraOpenFail), camR.message);
        return camR;
    }
    if (rm.success && camera_) camWasOpen_ = true;              // 0x0801 前置锚
    if (!rm.success) {
        if (camera_) {
            camera_->close();
            camera_.reset();
        }
        publishFault(code(DevFault::McuOpenFail), rm.message);
        return Result::fail("MCU 打开失败: " + rm.message);
    }
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: 相机+MCU 并行段 {}ms", el());
    // ③ 上行分流接线（onTemp 温度双警+记账+Warmup 喂入 / onGesture 手势暂存 /
    //    onShotCount 记账在 MCUDriver lastShot_——门面不转发）
    hal::McuUplink up;
    up.onTemp = [this](const serial::TempFrame& t) {
        checkTempFaults(t);                      // 0x0803 爆表 / 0x0804 乱跳（D-T13）
        lastTemps_ = t;
        warmup_->onTemperature(t.celsius[0], static_cast<int64_t>(t.ts));
    };
    up.onGesture = [this](const serial::GestureEvent& ev) {
        pendingGestures_.push_back(ev);          // logicTick ③ 统一派发（pump 回调内不发令）
    };
    up.onShotCount = [](const serial::ShotCountFrame&) {};   // G03 记账在 MCUDriver lastShot_
    mcu_->setUplink(up);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: uplink 接线 {}ms", el());
    // ⑤ 参数装载（Load 空档=全默认值；逐参数广播）+ 快照立即可读（单线程时刻）
    params_->bootstrap([] { return ""; });
    refreshParamSnapshot();
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: bootstrap+快照 {}ms", el());
    // ⑦ N12 温度周期下发（盲发无 ACK）：真机 auto 搜口探测已发 N12 T100 且命中
    //    口已生效——此处按配置定版（manual 口路径则为本会话首帧）；测试模式经
    //    writeOverride 记帧。写权交接口径：探测（open 线程经写线程出帧）后
    //    resetSerialWriteOwner 登记复位（R2-A1），此后下行全走 MCUDriver 写线程。
    //    批1 的 ≥50ms 临时钳已删（0x0804 改绝对差值判据后风暴根因消除——终态=
    //    specs 域 5-1000＋MCUDriver 钳，单一出口见 sendN12Clamped）
    mcu_->resetSerialWriteOwner();
    sendN12Clamped(cfg_.tempReportPeriodMs, nullptr);
    // D8 恢复路径（批4）：黑板非 Idle/采集中重开——按 lastCaptureMode_ 重发 N10
    //    参数重同步（正常 close→open 黑板已被 close 复位不触发；此为重开期间黑板
    //    被外部置非 Idle 的防御分支）。插在逻辑线程起前：params_/mode_ 尚单线程
    //    时刻无竞态；写权已归 MCUDriver 写线程
    if (mode_->mode() != DeviceMode::Idle || mode_->isCapturing()) {
        JMW_LOG_INFO("08-DeviceManager",
            "[DeviceManager] 重开重同步: 黑板 mode={} capturing={}——按 lastCaptureMode 重发 N10",
            static_cast<int>(mode_->mode()), mode_->isCapturing());
        mcu_->setCaptureParams(effectiveN10(*params_, lastCaptureMode_),
                               [this](bool ok, const std::string& p) {
            if (!ok) publishFault(code(DevFault::CmdNoAck), "N10 重开重同步 " + p);
        });
    }
    // 0x0802 武装（批4：open 成功即武装——全程无帧也纳入巡检；旧武装门 lastRx>0
    // 致 MCU 一步不出帧的「死链」永不告警）。lastRx>0 后按末帧计时（见 logicTick ⑥）
    serialArmed_ = true;
    serialArmedAtMs_ = nowMs();
    // ⑥ 逻辑线程（manualTick=true 测试跳过）。灯态策略：open 不发 N10（灯不亮）——
    // 灯只在 startCapture（N10 七参=启采）时亮、stopCapture（N11 H0）熄灯收口
    if (!cfg_.manualTick) {
        running_.store(true);
        logicThread_ = std::thread(&DeviceManager::logicLoop, this);
    }
    // ⑧ 广播
    publishEvent(EventType::DeviceConnected, 0, 0);
    opened_ = true;
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: 逻辑线程+广播+总计 {}ms", el());
    return Result::ok();
}

Result DeviceManager::close() {
    std::lock_guard<std::mutex> openLock(openMtx_);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] close: 设备关闭开始（串口={}）",
                 cfg_.serialPort);
    selfCheck_.stage = -1;                       // 自检状态机随设备关停作废
    if (logicThread_.joinable()) {
        running_.store(false);
        logicThread_.join();
    }
    {
        std::lock_guard<std::mutex> lock(postMutex_);
        postQueue_.clear();                     // 退出场景：余任务丢弃（不保送）
    }
    // —— 全量清理：停止采集熄灯 + 冲写队列 + 关串口 + 关相机 ——
    if (mcu_->isOpen()) {
        mcu_->stopScan(nullptr);                // N11 H0：停止/熄灯（260831 同一命令）
        mcu_->flushWrites(3000);                // 确保命令全部落线（3s 有界）
    }
    mcu_->close();                              // 关串口（停写线程+rx线程）
    if (camera_) {
        camera_->stopAsyncCapture();            // 停采集流
        camera_->close();                       // 关相机
    }
    // —— 状态复位（重开=全新会话）。D8 恢复路径（批4 补全）：ModeController 复位
    //    待机＋旧温清零（lastTemps_/tempSnap_ 同步——防 reopen 后首帧前
    //    getLastTemperatures 回上会话旧值，app 侧 ts>0 判定依赖）——
    (void)mode_->request(DeviceMode::Idle, "close_reset");   // 门禁拒也复位（退出场景）
    mode_->commit(DeviceMode::Idle);            // same-mode 不广播
    mode_->setCapturing(false);
    pendingGestures_.clear();
    lastTemps_ = serial::TempFrame{};           // 旧温清零（账本）
    {
        std::lock_guard<std::mutex> lock(tempSnapMtx_);
        tempSnap_ = serial::TempFrame{};        // 旧温清零（快照同步——读口同拍归零）
    }
    serialArmed_ = false;                       // 0x0802 武装随会话撤（reopen 再武装）
    serialArmedAtMs_ = 0;
    camFaultLatched_ = false;
    camWasOpen_ = false;
    serialSilentLatched_ = false;
    tempHotLatched_ = false;
    tempSpikeLatched_ = false;
    prevTempsValid_ = false;                    // 0x0804 差值锚清（reopen 首帧无前帧）
    opened_ = false;
    return Result::ok();
}

// ============================================================================
// 逻辑线程
// ============================================================================

void DeviceManager::logicLoop() {
    while (running_.load()) {
        logicTick();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DeviceManager::logicTick() {
    drainPosts();                               // ① 编队任务排空（首——本拍落地）
    mcu_->pump();                               // ② 上行 3 环排空（G01/G02/G03→uplink）
    for (const auto& ev : pendingGestures_) dispatchGesture(ev);   // ③ 手势派发（onGesture 入队）
    pendingGestures_.clear();
    const int64_t now = nowMs();
    warmup_->tick(now);                         // ④ 预热超时兜底
    // ⑤ 相机掉线巡检（#1，D-T13 扩为任意时刻：曾开→isOpen 翻 false 边沿；只报
    //    不停手；相机重开清锚允许再触发——防爆屏）
    if (camera_ && camera_->isOpen()) {
        camWasOpen_ = true;
        camFaultLatched_ = false;               // 恢复清锚
    } else if (camera_ && camWasOpen_ && !camFaultLatched_) {
        camFaultLatched_ = true;
        publishFault(code(DevFault::CameraLost), "相机掉线（isOpen 翻 false；只报不停手）");
    }
    // ⑥ 串口无声（#2≡#10 心跳丢失同源）：open 成功即武装（serialArmed_，批4——
    //    旧口径武装门 lastRx>0 导致「全程无帧」的死链永不告警）；距末帧（无帧则
    //    距武装时刻）超 heartbeatTimeoutMs → 边沿一次；再收到任何完整帧即恢复清锚。
    //    D9 自适应线索（登记不实现）：超时判据可选 max(10×N12 周期, 10000) 自适应
    if (serialArmed_) {
        const int64_t lastRx = static_cast<int64_t>(mcu_->lastRxTime());
        const int64_t ref = (lastRx > 0) ? lastRx : serialArmedAtMs_;
        if (now - ref > static_cast<int64_t>(cfg_.heartbeatTimeoutMs)) {
            if (!serialSilentLatched_) {
                serialSilentLatched_ = true;
                publishFault(code(DevFault::SerialSilent),
                             std::string(lastRx > 0 ? "串口无声(心跳丢失) 距末帧 "
                                                    : "串口无声(全程无帧) 距武装 ")
                                 + std::to_string(now - ref) + "ms");
            }
        } else {
            serialSilentLatched_ = false;       // 心跳恢复清锚
        }
    }
    // ⑦ 手势队列挤爆（#6）：G01 手势环满丢新计数增长即报（事件型——增量即边沿，
    //    无需恢复语义；计数单调累计）
    if (const uint64_t gd = mcu_->gestureRingDropped(); gd > lastKeyDrop_) {
        publishFault(code(DevFault::KeyRingOverflow),
                     "手势环满丢新 +" + std::to_string(gd - lastKeyDrop_) +
                         "（累计 " + std::to_string(gd) + "）");
        lastKeyDrop_ = gd;
    }
    // ⑧ 快照刷新（菜单/温度/参数——跨线程读口统一互斥快照；轻拷）
    refreshParamSnapshot();
    {
        std::lock_guard<std::mutex> lock(menuSnapMtx_);
        menuSnap_ = menu_->state();
    }
    {
        std::lock_guard<std::mutex> lock(tempSnapMtx_);
        tempSnap_ = lastTemps_;
    }
    // ⑨ 启动自检状态机推进（无阻塞；闲时 stage=-1 直返）
    selfCheckTick(now);
}

void DeviceManager::testInjectRaw(const std::string& frameBytes) {
    mcu_->testInjectRaw(frameBytes);
}

// ============================================================================
// 切模式（Critical #1：门禁调用方线程同步判；过→编队逻辑线程执行）
// ============================================================================

Result DeviceManager::enterScan() {
    const Result g = mode_->request(DeviceMode::Scanning, "enter_scan");
    if (!g.success) {
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] enterScan 门禁拒绝: {}", g.message);
        return g;                               // 拒→不入队，同步返回
    }
    post([this] { enterScanOnLogic(); });
    return g;
}

Result DeviceManager::enterCalibration() {
    const Result g = mode_->request(DeviceMode::Calibrating, "enter_calibration");
    if (!g.success) {
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] enterCalibration 门禁拒绝: {}", g.message);
        return g;
    }
    post([this] { enterCalibrationOnLogic(); });
    return g;
}

Result DeviceManager::toIdle() {
    const Result g = mode_->request(DeviceMode::Idle, "to_idle");
    if (!g.success) {
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] toIdle 门禁拒绝: {}", g.message);
        return g;
    }
    post([this] { toIdleOnLogic(); });
    return g;
}

void DeviceManager::sendSeq(std::vector<SeqStep> steps, std::function<void(bool)> onDone) {
    if (steps.empty()) {
        if (onDone) onDone(true);
        return;
    }
    SeqStep cur = std::move(steps.front());
    steps.erase(steps.begin());
    cur.send([this, desc = cur.desc, rest = std::move(steps), onDone = std::move(onDone)](
                 bool ok, const std::string& payload) mutable {
        if (!ok) {
            publishFault(code(DevFault::CmdNoAck),   // 发送失败（盲发口径 D8：写失败语义）
                         desc + " 命令组中段失败: " + payload);
            if (onDone) onDone(false);
            return;
        }
        sendSeq(std::move(rest), onDone);
    });
}

// 采集组步链（协议 260831）：单步 N10——启采=N10 本身（七参组装+灯控+启动，
// 不再有 N11 H1/FLUSH）。相机先启后发（帧号错位修正口径，见 startCaptureOnLogic）
std::vector<DeviceManager::SeqStep> DeviceManager::captureSeqSteps() {
    return {{"N10", [this](McuDone cb) {
                 mcu_->setCaptureParams(effectiveN10(*params_, lastCaptureMode_), std::move(cb));
             }}};
}

void DeviceManager::enterScanOnLogic() {
    sendSeq(captureSeqSteps(), [this](bool ok) {
        if (!ok) return;                         // Fault 已在 sendSeq 内报
        mode_->commit(DeviceMode::Scanning);     // 黑板「扫描中」
        mode_->setCapturing(true);               // + 采集开（02-D2：启停归 08）
        startStreamIfReady();
    });
}

void DeviceManager::enterCalibrationOnLogic() {
    // 260831：标定无专属下行命令（原 N16 已删）——纯软件落板；
    // 组链框架保留给采集用（captureSeqSteps）
    mode_->commit(DeviceMode::Calibrating);
}

void DeviceManager::toIdleOnLogic() {
    // 260831：待机无专属下行命令（原 N13 E1 已删）——纯软件落板
    mode_->commit(DeviceMode::Idle);
}

// ============================================================================
// 采集启停（启=N10（相机先启）/ 停=N11 H0；不切模式；幂等；编队执行）
// ============================================================================

void DeviceManager::startCapture(Scanner::ScanMode mode) {
    post([this, mode] {
        lastCaptureMode_ = mode;                  // 采集灯型（逻辑线程属主；N10 四管掩码组帧生效）
        startCaptureOnLogic();
    });
}

// —— 灯光直控（用户按钮直调；不启停采集——N10 灯字段即时生效，实测口径同
//    自检闪灯：固件收到 N10 即按新参数调灯）。激光管按 ScanMode 四管掩码
//（effectiveN10 同映射）；bgOn=false 压 B=0（可压过 MarkerOnly 的 40 抬升）——
void DeviceManager::setLights(bool bgOn, Scanner::ScanMode mode) {
    post([this, bgOn, mode] {
        if (!mcu_->isOpen()) return;
        hal::CaptureParams p = effectiveN10(*params_, mode);
        if (!bgOn) p.bgLight = 0;
        mcu_->setCaptureParams(p, nullptr);
    });
}

// —— 打光场景封装（灯型三态；组合语义入口，按钮/工作流直调）——
void DeviceManager::lightsBgOnly() {
    setLights(/*bgOn=*/true, Scanner::ScanMode::MarkerOnly);   // A 灯型：B=40 L=0 四管全关
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 打光场景: 只打补光灯（MarkerOnly 灯型 B=40 L=0 四管全关）");
}

void DeviceManager::lightsAllOff() {
    setLights(/*bgOn=*/false, Scanner::ScanMode::MarkerOnly);  // 全零 N10：B0 L0 T0V0C0D0
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 打光场景: 全灭（N11 H0 同语义）");
}

void DeviceManager::stopCapture() {
    post([this] { stopCaptureOnLogic(); });
}

void DeviceManager::startCaptureOnLogic() {
    if (mode_->isCapturing()) return;            // 幂等：黑板同值直返
    // —— 启动顺序（2026-09-06 帧号错位根因修正，260831 沿承）：**相机先启→再发
    //    N10** ——原序（N10→回调→startStream）在 N10 到达后 MCU 立即触发，而相机
    //    尚在配置中——左比右多吃 2~3 个触发脉冲，GetFrameID 偏移恒 2~3，严格配对
    //    全丢。改序后两台相机同时等触发，N10 到达时同步收第一个脉冲 → 偏移 ±1。
    startStreamIfReady();                        // ① 相机先就绪等触发
    sendSeq(captureSeqSteps(), [this](bool ok) { // ② N10=启采（盲发即回调）
        if (!ok) return;
        mode_->setCapturing(true);
    });
}

void DeviceManager::stopCaptureOnLogic() {
    if (!mode_->isCapturing()) return;
    mcu_->stopScan([this](bool ok, const std::string& p) {
        if (!ok) {
            publishFault(code(DevFault::CmdNoAck), "N11 H0 " + p);   // 写失败语义（盲发口径）
            return;
        }
        mode_->setCapturing(false);
        // 灯态收口：单帧 N11 H0 即停止熄灯（260831 同一命令）。停相机流前先冲
        // 队列——相机停流瞬间 USB 风暴会堵串口写（flush 有界 300ms）
        mcu_->flushWrites(300);
        if (camera_ && camera_->isOpen()) camera_->stopAsyncCapture();
    });
}

void DeviceManager::startStreamIfReady() {
    if (camera_ && camera_->isOpen() && frameCb_) camera_->startAsyncCapture(frameCb_);
}

// ============================================================================
// 启动自检（open 成功后由 app 调）——无阻塞状态机版：启动只置态，推进全由
// logicTick 每 10ms 的 selfCheckTick 分摊（单次 µs 级），逻辑线程不被 sleep 堵死
// ============================================================================
void DeviceManager::setWireTap(std::function<void(bool, const std::string&)> tap) {
    mcu_->setWireTap(std::move(tap));            // 装配期设置（open 前），无需编队
}

void DeviceManager::startupSelfCheck(std::function<void(const std::string&, bool)> report) {
    post([this, report = std::move(report)]() mutable {
        if (selfCheck_.stage != -1) return;      // 已在跑（幂等）
        selfCheck_.report = std::move(report);
        selfCheck_.frames.store(0, std::memory_order_relaxed);
        selfCheck_.frameValid.store(false, std::memory_order_relaxed);
        selfCheck_.item = 0;                     // 闪灯两段式：先 bgLight 项
        // mcuLink 凭据（260831 无回显）：lastRxTime>0——open 后 N12 周期上报的
        // G02 到达即活（manual 口首帧未到则 false，联调批4 复核凭据窗口）
        selfCheck_.report("mcuLink", mcu_->lastRxTime() > 0);
        selfCheck_.stage = 0;
        selfCheck_.stageStartMs = nowMs();
    });
}

void DeviceManager::selfCheckTick(int64_t nowMs_) {
    if (selfCheck_.stage < 0) return;
    switch (selfCheck_.stage) {
    case 0: {  // 发 N10 H5 短采（bgLight 项：B=账本值 L=0；laser 项：B=0 L=账本值 T1V1C0D0）
        hal::CaptureParams p{};
        p.freqHz = 5;                            // 短采低频（闪灯验证）
        if (selfCheck_.item == 0) {
            p.bgLight = static_cast<int>(params_->get("bgLight").value);
            p.laserLevel = 0;
        } else {
            p.bgLight = 0;
            p.laserLevel = static_cast<int>(params_->get("laserLevel").value);
            p.laserT = 1;
            p.laserV = 1;
        }
        mcu_->setCaptureParams(p, nullptr);      // 盲发（写失败仅写线程日志）
        selfCheck_.stage = 1;
        selfCheck_.stageStartMs = nowMs_;
        break;
    }
    case 1:  // 保持 800ms（闪亮窗口——不起相机流，相机 USB 流量会干扰串口收发）
        if (nowMs_ - selfCheck_.stageStartMs >= 800) {
            selfCheck_.stage = 2;
            selfCheck_.stageStartMs = nowMs();
        }
        break;
    case 2: {  // 收口：N11 H0（停止/熄灯）。盲发无回显——发毕即报该项通过
               //（批4 真机联调再定新凭据）；后进 laser 项或相机验证
        mcu_->stopScan(nullptr);                 // N11 H0
        mcu_->flushWrites(2000);
        selfCheck_.report(selfCheck_.item == 0 ? "bgLight" : "laser", true);
        if (selfCheck_.item == 0) {
            selfCheck_.item = 1;                 // 第二段：laser 项闪灯
            selfCheck_.stage = 0;
        } else {
            selfCheck_.stage = 3;
        }
        break;
    }
    case 3: {  // 相机验证（相机 open 状态判定——不起流验证，避免 USB 扰动串口）
        selfCheck_.report("camera", camera_ && camera_->isOpen());
        JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 启动自检完成（相机 open={}）",
                     camera_ && camera_->isOpen());
        selfCheck_.stage = -1;
        break;
    }
    }
}

// ============================================================================
// 预热（N13 S<目标 0-80，DeviceManager 与 MCUDriver 双层钳>；看火员只报稳/超，
// 不停加热；编队执行）
// ============================================================================

void DeviceManager::startWarmup(int targetC, std::function<void(bool stable)> done) {
    post([this, targetC, done = std::move(done)]() mutable {
        warmupDone_ = std::move(done);           // 重开=后值胜出（旧未触发作废）
        // N13 S 域 0-80：DeviceManager 先钳（99→80 / -5→0）——下发帧/看火目标/
        // detail 串三者同值；MCUDriver 层再钳一道（双层钳口径一致）
        if (targetC < 0) targetC = 0;
        if (targetC > 80) targetC = 80;
        warmup_->start(targetC);
        // 盲发——写失败经回调收口 0x0807（批2 直发版复核）
        mcu_->setHeatTarget(targetC, [this, targetC](bool ok, const std::string& p) {
            if (!ok) publishFault(code(DevFault::CmdNoAck),
                                  "N13 S" + std::to_string(targetC) + " " + p);
        });
    });
}

// ============================================================================
// 按键链接线（KeySemantics 11 出口；手势判定归 MCU G01——PC 只做语义映射）
// ============================================================================

void DeviceManager::buildKeyActions() {
    KeySemActions a;
    a.captureToggle = [this] {
        if (mode_->isCapturing()) stopCaptureOnLogic();
        else startCaptureOnLogic();              // 逻辑线程内直调本体（免编队绕行）
    };
    a.menuSelect = [this] {                      // 按 cursor 分叉（MCU 只给信号）
        const int cur = menu_->state().cursor;
        if (cur == 1 || cur == 2) {
            JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 菜单选中①/②：视野模式设定（modeCursor={} 账本为准）",
                         menu_->state().modeCursor);
        } else {
            // ③扫描完成/④开始后处理：派工作流归 app 订阅——门面只广播。
            // UserDefined+param1=3/4（Important #4：待 base 增专用事件类型，人工过会后改）
            publishEvent(EventType::UserDefined, cur, 0);
            JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] 菜单选中③/④：派发工作流事件 cursor={}", cur);
        }
    };
    a.cycleMode = [this] { menu_->apply(MenuOp::CycleMode); };
    a.enterMenu = [this] { menu_->apply(MenuOp::EnterMenu); };
    a.exitMenu = [this] { menu_->apply(MenuOp::ExitMenu); };
    a.cycleAdjustCtx = [this] { menu_->apply(MenuOp::CycleAdjustCtx); };
    a.cursorLeft = [this] {                      // 游标移动：步进余额清
        menu_->apply(MenuOp::CursorLeft);
        menu_->takeAdjustSteps();
    };
    a.cursorRight = [this] {
        menu_->apply(MenuOp::CursorRight);
        menu_->takeAdjustSteps();
    };
    a.adjustUp = [this] { applyAdjust(+1); };
    a.adjustDown = [this] { applyAdjust(-1); };
    a.dropped = [](const char* why) { JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 手势丢弃: {}", why); };
    semantics_ = std::make_unique<KeySemantics>(                  // M1：采集态菜单键门禁
        [this] { return !mode_->isCapturing(); }, std::move(a));
}

void DeviceManager::applyAdjust(int dir) {
    menu_->apply(dir > 0 ? MenuOp::AdjustUp : MenuOp::AdjustDown);
    const int steps = menu_->takeAdjustSteps();   // 净步数（本拍 ±1）
    const auto ctx = menu_->state().adjustCtx;
    if (ctx == MenuState::AdjustCtx::Brightness && steps != 0) {
        params_->setValue("exposure", params_->get("exposure").value + steps,
                          ParamEntry::Source::Key);
    } else {
        JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 调节步进 ctx={}（View 上下文暂仅记账）",
                     static_cast<int>(ctx));
    }
}

void DeviceManager::dispatchGesture(const serial::GestureEvent& ev) {
    semantics_->onGesture(ev, menu_->state());
}

// ============================================================================
// 参数下发（ParamStore Dispatch：exposure 相机直设 / N10 组参）+ 快照双口
// ============================================================================

// N12 T 单一下发出口（open 定版/param dispatch 改参共用——批3 审查发现 dispatch
// 裸发绕过 open 守门后收敛于此，防两路口径再不对称）：值域钳 5-1000 与 specs 域＋
// MCUDriver 层同域终态。批1 的 ≥50ms 临时钳已删（0x0804 改绝对差值判据后风暴
// 根因消除）
void DeviceManager::sendN12Clamped(int periodMs, McuDone cb) {
    if (periodMs < 5) periodMs = 5;
    if (periodMs > 1000) periodMs = 1000;
    mcu_->setTempReportPeriod(periodMs, std::move(cb));
}

void DeviceManager::onParamDispatch(const std::string& key, double v, ParamStore::Done done) {
    if (key == "exposure") {
        if (camera_ && camera_->isOpen()) {
            const Result r = camera_->setExposure(v);
            done(r.success);
        } else {
            done(true);                            // 无相机=纯记账（真机必接相机）
        }
        return;
    }
    if (key == "tempReportPeriodMs") {            // N12 T（批3 入 specs；值域 5-1000
        sendN12Clamped(static_cast<int>(v),       // 入口已在 setValue 钳）——单一出口同 open
                        [done](bool ok, const std::string&) { done(ok); });
        return;
    }
    // N10 组参：采集中任一变更即全参重发（N10=启采——参数即时生效）；空闲仅
    // 记账 done(true)（startCapture 时自账本组帧下发）
    if (mode_->isCapturing()) {
        mcu_->setCaptureParams(effectiveN10(*params_, lastCaptureMode_),
                               [done](bool ok, const std::string&) { done(ok); });
    } else {
        done(true);
    }
}

void DeviceManager::refreshParamSnapshot() {
    std::lock_guard<std::mutex> lock(snapshotMutex_);
    for (const auto& k : paramKeys_) paramSnapshot_[k] = params_->get(k);
}

ParamEntry DeviceManager::getParam(const std::string& key) const {
    std::lock_guard<std::mutex> lock(snapshotMutex_);
    const auto it = paramSnapshot_.find(key);
    return (it != paramSnapshot_.end()) ? it->second : ParamEntry{};   // 未登记→0/false
}

void DeviceManager::setParam(const std::string& key, double v, ParamEntry::Source src) {
    post([this, key, v, src] { params_->setValue(key, v, src); });
}

// ============================================================================
// 观测 / 相机薄转发（统一编队：返回值=前置检查）
// ============================================================================

bool DeviceManager::isDeviceReady() const {
    if (!mcu_->isOpen()) return false;
    if (!camFactory_) return true;                 // 未配相机=只看 MCU
    return camera_ && camera_->isOpen();
}

serial::TempFrame DeviceManager::getLastTemperatures() const {
    std::lock_guard<std::mutex> lock(tempSnapMtx_);
    return tempSnap_;                           // 快照（logicTick 末刷新——拍后读即最新）
}

bool DeviceManager::isCapturing() const { return mode_->isCapturing(); }
DeviceMode DeviceManager::mode() const { return mode_->mode(); }
MenuState DeviceManager::menuState() const {
    std::lock_guard<std::mutex> lock(menuSnapMtx_);
    return menuSnap_;                           // 快照（logicTick 末刷新——拍后读即最新）
}

bool DeviceManager::isCameraOpen() const { return camera_ && camera_->isOpen(); }

Result DeviceManager::setCameraExposure(double ms) {
    if (!camera_) return Result::fail("未配置相机");   // 前置检查同步；动作编队
    post([this, ms] {
        if (camera_ && camera_->isOpen()) camera_->setExposure(ms);
    });
    return Result::ok("已编队");
}

Result DeviceManager::startFrameStream(hal::FrameCallback cb) {
    if (!camera_ || !camera_->isOpen()) return Result::fail("相机未就绪");
    post([this, cb = std::move(cb)]() mutable {
        // 帧出口包一层计数（帧率测量归 08——封装性：计数在设备管理层，上层只读）
        frameCb_ = [this, userCb = std::move(cb)](const hal::StereoFrame& f) {
            const auto now = std::chrono::steady_clock::now();
            const uint64_t cnt = m_rxCnt_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (m_lastFpsTick_.time_since_epoch().count() == 0) {
                m_lastFpsTick_ = now;
            } else if (now - m_lastFpsTick_ >= std::chrono::seconds(1)) {
                const double sec =
                    std::chrono::duration<double>(now - m_lastFpsTick_).count();
                m_measuredFps.store(sec > 0 ? static_cast<int>(cnt / sec) : 0,
                                    std::memory_order_relaxed);
                m_rxCnt_.store(0, std::memory_order_relaxed);
                m_lastFpsTick_ = now;
            }
            userCb(f);      // 转发上层回调
        };
        if (camera_ && camera_->isOpen()) camera_->startAsyncCapture(frameCb_);
    });
    return Result::ok("已编队");
}

Result DeviceManager::stopFrameStream() {
    if (!camera_ || !camera_->isOpen()) return Result::fail("相机未就绪");
    post([this] {
        if (camera_ && camera_->isOpen()) camera_->stopAsyncCapture();
    });
    return Result::ok("已编队");
}

// ============================================================================
// 温度双警（D-T13 #3/#4；onTemp 回调内即逻辑线程，无跨线程）
// ============================================================================

void DeviceManager::checkTempFaults(const serial::TempFrame& t) {
    constexpr int n = 4;                         // G02 恒四路（260831）
    // #3 爆表：任一路 >tempMaxC → 边沿一次；全路回落 ≤ 限清锚
    bool over = false;
    for (int i = 0; i < n; ++i)
        if (t.celsius[i] > cfg_.tempMaxC) over = true;
    if (over && !tempHotLatched_) {
        tempHotLatched_ = true;
        std::string d;
        for (int i = 0; i < n; ++i)
            if (t.celsius[i] > cfg_.tempMaxC)
                d += " 路" + std::to_string(i) + "=" + std::to_string(t.celsius[i]) + "C";
        publishFault(code(DevFault::TempOverMax),
                     "温度爆表(>" + std::to_string(cfg_.tempMaxC) + "C):" + d);
    } else if (!over) {
        tempHotLatched_ = false;                // 全路回落清锚
    }
    // #4 乱跳（批4 绝对差值判据）：相邻 G02 帧同路 |Δ|>tempSpikeAbsC → 边沿一次；
    //    次帧差值回落清锚（原「次帧平稳清锚」语义平移——判据换差值口径）。旧速率
    //    判据 |Δ|/Δt>tempSpikeC 已删：上报周期敏感（G02@T=5ms 时 1LSB 抖动=20℃/s
    //    必风暴），绝对差值去周期依赖
    if (prevTempsValid_) {
        bool spiky = false;
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            const double delta = std::abs(t.celsius[i] - prevTemps_.celsius[i]);
            if (delta > cfg_.tempSpikeAbsC) {
                spiky = true;
                worst = std::max(worst, delta);
            }
        }
        if (spiky && !tempSpikeLatched_) {
            tempSpikeLatched_ = true;
            publishFault(code(DevFault::TempSpike),
                         "温度乱跳: 峰值差值 " + std::to_string(worst) + "C");
        } else if (!spiky) {
            tempSpikeLatched_ = false;          // 次帧差值回落清锚
        }
    }
    prevTemps_ = t;
    prevTempsValid_ = true;
}

// ============================================================================
// 事件出口
// ============================================================================

void DeviceManager::publishFault(int64_t code, const std::string& detail) {
    JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] Fault code={:#06x} {}", code, detail);
    if (!bus_) return;
    Event e;
    e.type = EventType::FaultOccurred;
    e.sourceId = 8;                                // 08 模块来源标识
    e.param1 = code;
    e.param2 = 0;
    e.timestamp = static_cast<TimestampMs>(nowMs());
    bus_->publish(e);
}

void DeviceManager::publishEvent(EventType t, int64_t p1, int64_t p2) {
    if (!bus_) return;
    Event e;
    e.type = t;
    e.sourceId = 8;
    e.param1 = p1;
    e.param2 = p2;
    e.timestamp = static_cast<TimestampMs>(nowMs());
    bus_->publish(e);
}

} // namespace Scanner::device
