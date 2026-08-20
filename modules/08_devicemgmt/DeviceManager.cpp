// ============================================================================
// DeviceManager.cpp — 门面 + 逻辑线程实现（契约见 DeviceManager.h / 设计方案 §4）
// ============================================================================

#include "DeviceManager.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <utility>

namespace Scanner::device {
namespace {

// 统一时基：system_clock 毫秒（与 MCUDriver 上行帧 ts 同域——Warmup tsMs/tick、
// KeyManager pcClock、v2 兜底判据共用）
int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

constexpr int64_t kV2TempFallbackMs = 1200;   // v2 温度断流兜底周期（§2.5）
constexpr int kV2TempQueryCh = 2;             // N15 查询通道（现状单路）

// Fault 码（0x08xx = 08 模块来源段；T13 故障表补全前先立占位）
constexpr int64_t kFaultCameraOpen = 0x0801;  // 相机打开失败
constexpr int64_t kFaultMcuOpen = 0x0802;     // MCU（串口）打开失败
constexpr int64_t kFaultCameraLost = 0x0803;  // 采集中相机掉线（只报不停手）
constexpr int64_t kFaultCmdFail = 0x0804;     // 单命令 ACK 3 败（N11 启停）
constexpr int64_t kFaultGroupFail = 0x0805;   // 命令组中段 3 败（切模式）
constexpr int64_t kFaultHeatCmd = 0x0806;     // 加热命令下发失败
constexpr int64_t kFaultSelfCheck = 0x0807;   // 开机自检命令失败

std::vector<ParamSpec> makeParamSpecs() {      // 参数字段定义归 08（红线）
    return {
        {"exposure", 10.0, 1.0, 100.0},        // 曝光 ms（相机直设）
        {"freqHz", 60.0, 20.0, 120.0},         // N10 H 拍照频率
        {"bgLight", 80.0, 0.0, 255.0},         // N10 B 补光
        {"laserLevel", 120.0, 0.0, 255.0},     // N10 L 激光强度
        {"laserSelectA", 1.0, 1.0, 6.0},       // N10 T 交叉激光选择 A
        {"laserSelectB", 1.0, 1.0, 6.0},       // N10 V 交叉激光选择 B
    };
}

} // namespace

// ============================================================================
// 构造 / 析构
// ============================================================================

DeviceManager::DeviceManager(DeviceConfig cfg, GateQuery gate, infra::EventBus* bus,
                             CameraFactory camFactory, MCUDriver::WriteOverride serialWrite)
    : cfg_(std::move(cfg)),
      bus_(bus),
      camFactory_(std::move(camFactory)),
      writeOverride_(std::move(serialWrite)),
      mcu_(std::make_unique<MCUDriver>(writeOverride_)),
      keyMgr_(std::make_unique<KeyManager>(cfg_.keys, [] { return nowMs(); })),
      menu_(std::make_unique<MenuLogic>()),
      mode_(std::make_unique<ModeController>(std::move(gate))),
      warmup_(std::make_unique<WarmupSequence>(cfg_.warmup)) {
    buildKeyActions();
    params_ = std::make_unique<ParamStore>(
        makeParamSpecs(),
        [this](const std::string& key, double v, ParamStore::Done done) {
            onParamDispatch(key, v, done);
        });
    params_->onParamChanged = [this](const std::string& key, const ParamEntry& e) {
        if (!bus_) return;
        Event ev;
        ev.type = EventType::UserDefined;      // base 暂无 ParamChanged——裁定占位
        ev.sourceId = 8;
        ev.param1 = 0;
        ev.param2 = 0;
        ev.timestamp = static_cast<TimestampMs>(nowMs());
        bus_->publish(ev);
        spdlog::debug("[DeviceManager] 参数改账 {}={:.3f} confirmed={}", key, e.value,
                      e.confirmed);
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
        auto cb = std::move(warmupDone_);
        if (cb) cb(false);
    };
}

DeviceManager::~DeviceManager() { close(); }

// ============================================================================
// open 一条龙 / close 倒序
// ============================================================================

Result DeviceManager::open() {
    if (opened_) return Result::ok("设备已打开");
    // ① 相机（工厂缺省=不接相机：isCameraOpen 恒 false 但不算失败）
    if (camFactory_) {
        camera_ = camFactory_();
        const Result r = camera_->open();
        if (!r.success) {
            camera_.reset();
            publishFault(kFaultCameraOpen, r.message);
            return Result::fail("相机打开失败: " + r.message);
        }
    }
    // ② MCU（测试 writeOverride 模式=逻辑开，不开真串口不起 rx 线程）
    mcu_->setProtocolVersion(cfg_.protocol);
    mcu_->setAckTimeoutMs(cfg_.ackTimeoutMs);
    const Result rm = mcu_->open(cfg_.serialPort);
    if (!rm.success) {
        if (camera_) {
            camera_->close();
            camera_.reset();
        }
        publishFault(kFaultMcuOpen, rm.message);
        return Result::fail("MCU 打开失败: " + rm.message);
    }
    // ③ 上行分流接线（onTemp 温度记账+Warmup 喂入 / onKey→KeyManager / onStatus 记账）
    hal::McuUplink up;
    up.onTemp = [this](const serial::TempFrame& t) {
        lastTemps_ = t;
        tempRxTime_ = t.ts;
        warmup_->onTemperature(t.celsius[0], static_cast<int64_t>(t.ts));
    };
    up.onKey = [this](const serial::RawKeyEvent& k) { keyMgr_->onRawEvent(k); };
    up.onStatus = [this](const serial::StatusFrame& s) {
        spdlog::warn("[DeviceManager] S 状态帧 code={:#x}（码表待协议 §8-8）", s.code);
    };
    mcu_->setUplink(up);
    // ⑤ 参数装载（Load 空档=全默认值；逐参数广播）
    params_->bootstrap([] { return ""; });
    // ⑥ 逻辑线程（manualTick=true 测试跳过）
    if (!cfg_.manualTick) {
        running_.store(true);
        logicThread_ = std::thread(&DeviceManager::logicLoop, this);
    }
    // ⑦ 开机自检 N12 Z1（S1 清单通讯项；ACK 失败异步 Fault——open 已返回）
    mcu_->enterSelfCheck([this](bool ok, const std::string& p) {
        if (!ok) publishFault(kFaultSelfCheck, "N12Z1 " + p);
    });
    // ⑧ 广播
    publishEvent(EventType::DeviceConnected, 0, 0);
    opened_ = true;
    return Result::ok();
}

Result DeviceManager::close() {
    if (logicThread_.joinable()) {
        running_.store(false);
        logicThread_.join();
    }
    mcu_->close();
    if (camera_) camera_->close();
    standbyActive_ = false;
    camFaultLatched_ = false;
    tempRxTime_ = 0;
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
    mcu_->pump();                                          // 上行环排空（含 onAck 回填）
    keyMgr_->tick(nowMs());                                // 手势判定 PC 域兜底
    for (const auto& g : keyMgr_->drain()) dispatchKeyGesture(g);
    mcu_->channelTick();                                   // 对账/重传/3 败收口
    const int64_t now = nowMs();
    warmup_->tick(now);                                    // 预热超时兜底
    // v2 温度断流兜底：首个 T 帧后武装，超 1.2s 无帧 → N15 查询（发后重臂）
    if (cfg_.protocol == serial::FrameCodec::Version::V2 && tempRxTime_ > 0 &&
        now - static_cast<int64_t>(tempRxTime_) > kV2TempFallbackMs) {
        mcu_->queryTemperature(kV2TempQueryCh);
        tempRxTime_ = static_cast<TimestampMs>(now);
    }
    // 采集中相机掉线巡检（只报不停手；边沿锁防复报，相机恢复即复位）
    if (mode_->isCapturing() && camera_ && !camera_->isOpen()) {
        if (!camFaultLatched_) {
            camFaultLatched_ = true;
            publishFault(kFaultCameraLost, "采集中相机掉线（不自主停采）");
        }
    } else if (!camera_ || camera_->isOpen()) {
        camFaultLatched_ = false;
    }
}

void DeviceManager::testInjectRaw(const std::string& frameBytes) {
    mcu_->testInjectRaw(frameBytes);
}

// ============================================================================
// 切模式命令组（§4-2：目标态非待机先退待机；组成功回调才擦板）
// ============================================================================

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
            publishFault(kFaultGroupFail, desc + " 命令组中段失败: " + payload);
            if (onDone) onDone(false);
            return;
        }
        sendSeq(std::move(rest), onDone);
    });
}

void DeviceManager::enterScan() {
    const Result g = mode_->request(DeviceMode::Scanning, "enter_scan");
    if (!g.success) {
        spdlog::warn("[DeviceManager] enterScan 门禁拒绝: {}", g.message);
        return;
    }
    std::vector<SeqStep> seq;
    if (standbyActive_) {
        seq.push_back({"N13E0", [this](McuDone cb) {
                           mcu_->exitStandby([this, cb](bool ok, const std::string& p) {
                               if (ok) standbyActive_ = false;   // ACK 确认退出待机
                               cb(ok, p);
                           });
                       }});
    }
    seq.push_back({"N10", [this](McuDone cb) {
                       mcu_->setCaptureParams(captureParamsFromAccount(), std::move(cb));
                   }});
    seq.push_back({"N11H1", [this](McuDone cb) { mcu_->startScan(std::move(cb)); }});
    sendSeq(std::move(seq), [this](bool ok) {
        if (!ok) return;                           // Fault 已在 sendSeq 内报
        standbyActive_ = false;
        mode_->commit(DeviceMode::Scanning);       // 黑板「扫描中」
        mode_->setCapturing(true);                 // + 采集开（02-D2：启停归 08）
        startStreamIfReady();
    });
}

void DeviceManager::enterCalibration() {
    const Result g = mode_->request(DeviceMode::Calibrating, "enter_calibration");
    if (!g.success) {
        spdlog::warn("[DeviceManager] enterCalibration 门禁拒绝: {}", g.message);
        return;
    }
    std::vector<SeqStep> seq;
    if (standbyActive_) {
        seq.push_back({"N13E0", [this](McuDone cb) {
                           mcu_->exitStandby([this, cb](bool ok, const std::string& p) {
                               if (ok) standbyActive_ = false;
                               cb(ok, p);
                           });
                       }});
    }
    seq.push_back({"N16B1", [this](McuDone cb) { mcu_->enterCalibration(std::move(cb)); }});
    sendSeq(std::move(seq), [this](bool ok) {
        if (!ok) return;
        standbyActive_ = false;
        mode_->commit(DeviceMode::Calibrating);
    });
}

void DeviceManager::toIdle() {
    const Result g = mode_->request(DeviceMode::Idle, "to_idle");
    if (!g.success) {
        spdlog::warn("[DeviceManager] toIdle 门禁拒绝: {}", g.message);
        return;
    }
    sendSeq({{"N13E1", [this](McuDone cb) { mcu_->enterStandby(std::move(cb)); }}},
            [this](bool ok) {
                if (!ok) return;
                standbyActive_ = true;             // MCU 侧已待机
                mode_->commit(DeviceMode::Idle);
            });
}

// ============================================================================
// 采集启停（N11 H1/H0 + 相机开停流；不切模式；幂等）
// ============================================================================

void DeviceManager::startCapture() {
    if (mode_->isCapturing()) return;              // 幂等：黑板同值直返
    mcu_->startScan([this](bool ok, const std::string& p) {
        if (!ok) {
            publishFault(kFaultCmdFail, "N11H1 " + p);
            return;
        }
        mode_->setCapturing(true);
        startStreamIfReady();
    });
}

void DeviceManager::stopCapture() {
    if (!mode_->isCapturing()) return;
    mcu_->stopScan([this](bool ok, const std::string& p) {
        if (!ok) {
            publishFault(kFaultCmdFail, "N11H0 " + p);
            return;
        }
        mode_->setCapturing(false);
        if (camera_ && camera_->isOpen()) camera_->stopAsyncCapture();
    });
}

void DeviceManager::startStreamIfReady() {
    if (camera_ && camera_->isOpen() && frameCb_) camera_->startAsyncCapture(frameCb_);
}

// ============================================================================
// 预热（§4-2：N14 T<目标>；看火员只报稳/超，不停加热）
// ============================================================================

void DeviceManager::startWarmup(int targetC, std::function<void(bool stable)> done) {
    warmupDone_ = std::move(done);                 // 重开=后值胜出（旧未触发作废）
    warmup_->start(targetC);
    mcu_->setHeatTarget(targetC, [this, targetC](bool ok, const std::string& p) {
        if (!ok) publishFault(kFaultHeatCmd, "N14T" + std::to_string(targetC) + " " + p);
    });
}

// ============================================================================
// 按键链接线（KeySemantics 11 出口）
// ============================================================================

void DeviceManager::buildKeyActions() {
    KeySemActions a;
    a.captureToggle = [this] {
        if (mode_->isCapturing()) stopCapture();
        else startCapture();
    };
    a.menuSelect = [this] {                        // 按 cursor 分叉（裁判只给信号）
        const int cur = menu_->state().cursor;
        if (cur == 1 || cur == 2) {
            spdlog::info("[DeviceManager] 菜单选中①/②：视野模式设定（modeCursor={} 账本为准）",
                         menu_->state().modeCursor);
        } else {
            // ③扫描完成/④开始后处理：派工作流归 app 订阅——门面只广播
            const EventType t = (cur == 3) ? EventType::ScanStopped
                                          : EventType::PostProcessStarted;
            publishEvent(t, cur, 0);
            spdlog::warn("[DeviceManager] 菜单选中③/④：派发工作流事件 cursor={}", cur);
        }
    };
    a.cycleMode = [this] { menu_->apply(MenuOp::CycleMode); };
    a.enterMenu = [this] { menu_->apply(MenuOp::EnterMenu); };
    a.exitMenu = [this] { menu_->apply(MenuOp::ExitMenu); };
    a.cycleAdjustCtx = [this] { menu_->apply(MenuOp::CycleAdjustCtx); };
    a.cursorLeft = [this] {                        // 游标移动：步进余额清
        menu_->apply(MenuOp::CursorLeft);
        menu_->takeAdjustSteps();
    };
    a.cursorRight = [this] {
        menu_->apply(MenuOp::CursorRight);
        menu_->takeAdjustSteps();
    };
    a.adjustUp = [this] { applyAdjust(+1); };
    a.adjustDown = [this] { applyAdjust(-1); };
    a.dropped = [](const char* why) { spdlog::info("[DeviceManager] 手势丢弃: {}", why); };
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
        spdlog::info("[DeviceManager] 调节步进 ctx={}（View 上下文暂仅记账）",
                     static_cast<int>(ctx));
    }
}

void DeviceManager::dispatchKeyGesture(const KeyGesture& g) {
    semantics_->onGesture(g, menu_->state());
}

// ============================================================================
// 参数下发（ParamStore Dispatch：exposure 相机直设 / N10 组参）
// ============================================================================

hal::CaptureParams DeviceManager::captureParamsFromAccount() const {
    hal::CaptureParams p;
    p.freqHz = static_cast<int>(params_->get("freqHz").value);
    p.bgLight = static_cast<int>(params_->get("bgLight").value);
    p.laserSelectA = static_cast<int>(params_->get("laserSelectA").value);
    p.laserSelectB = static_cast<int>(params_->get("laserSelectB").value);
    p.laserLevel = static_cast<int>(params_->get("laserLevel").value);
    return p;
}

void DeviceManager::onParamDispatch(const std::string& key, double v, ParamStore::Done done) {
    if (key == "exposure") {
        if (camera_ && camera_->isOpen()) {
            const Result r = camera_->setExposure(v);
            done(r.success, r.success);
        } else {
            done(true, true);                     // 无相机=纯记账（真机必接相机）
        }
        return;
    }
    // N10 组参：采集中任一变更即全参重发（协议表：更新参数需重设采集参数）；
    // 空闲仅记账（enterScan 时自账本组帧下发）。v2 通道发不等 → done(true,false)
    if (mode_->isCapturing()) {
        mcu_->setCaptureParams(captureParamsFromAccount(),
                               [done](bool ok, const std::string&) { done(ok, ok); });
    } else {
        done(true, false);
    }
}

// ============================================================================
// 观测 / 相机薄转发
// ============================================================================

bool DeviceManager::isDeviceReady() const {
    if (!mcu_->isOpen()) return false;
    if (!camFactory_) return true;                 // 未配相机=只看 MCU
    return camera_ && camera_->isOpen();
}

serial::TempFrame DeviceManager::getLastTemperatures() const { return lastTemps_; }

bool DeviceManager::isCapturing() const { return mode_->isCapturing(); }
DeviceMode DeviceManager::mode() const { return mode_->mode(); }
MenuState DeviceManager::menuState() const { return menu_->state(); }

bool DeviceManager::isCameraOpen() const { return camera_ && camera_->isOpen(); }

Result DeviceManager::setCameraExposure(double ms) {
    if (!camera_) return Result::fail("未配置相机");
    return camera_->setExposure(ms);
}

Result DeviceManager::startFrameStream(hal::FrameCallback cb) {
    frameCb_ = std::move(cb);
    if (!camera_ || !camera_->isOpen()) return Result::fail("相机未就绪");
    return camera_->startAsyncCapture(frameCb_);
}

Result DeviceManager::stopFrameStream() {
    if (!camera_ || !camera_->isOpen()) return Result::fail("相机未就绪");
    return camera_->stopAsyncCapture();
}

ParamStore& DeviceManager::params() { return *params_; }

// ============================================================================
// 事件出口
// ============================================================================

void DeviceManager::publishFault(int64_t code, const std::string& detail) {
    spdlog::warn("[DeviceManager] Fault code={:#06x} {}", code, detail);
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
