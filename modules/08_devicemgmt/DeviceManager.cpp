// ============================================================================
// DeviceManager.cpp — 门面 + 逻辑线程实现（契约见 DeviceManager.h / 设计方案 §4）
// ============================================================================

#include "DeviceManager.h"

#include "KeySemantics.h"
#include "MCUDriver.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
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
constexpr size_t kPostQueueCap = 64;          // 编队队列容量（满丢新+warn——Critical #1）

// Fault 码表归头文件 DevFault（D-T13 §6.2：十类 → 8 码 + 0x081x 开机段）；
// 调用点统一经 code() 取值
constexpr int64_t code(DevFault f) { return static_cast<int64_t>(f); }

std::vector<ParamSpec> makeParamSpecs() {      // 参数字段定义归 08（红线）
    return {
        {"exposure", 10.0, 1.0, 100.0},        // 曝光 ms（相机直设）
        {"freqHz", 50.0, 20.0, 120.0},         // N10 H 拍照频率（默认 50）
        {"bgLight", 60.0, 0.0, 100.0},         // N10 B 补光（默认 60；实测灯控量程 0-100）
        {"laserLevel", 60.0, 0.0, 100.0},      // N10 L 激光强度（默认 60；实测灯控量程 0-100）
        {"laserSelectA", 1.0, 1.0, 6.0},       // N10 T 交叉激光选择 A＝左斜组（1-6）
        {"laserSelectB", 2.0, 1.0, 6.0},       // N10 V 交叉激光选择 B＝右斜组（1-6；
                                               //   与 A 异组——同组则固件帧序交替失效，
                                               //   只打一族线；组号↔左/右映射随固件实测定）
    };
}

// N10 全参自 ParamStore 账本组帧（cpp 本地——不进头防 IMCU.h 类型泄漏）
hal::CaptureParams captureParamsFromAccount(const ParamStore& params) {
    hal::CaptureParams p;
    p.freqHz = static_cast<int>(params.get("freqHz").value);
    p.bgLight = static_cast<int>(params.get("bgLight").value);
    p.laserSelectA = static_cast<int>(params.get("laserSelectA").value);
    p.laserSelectB = static_cast<int>(params.get("laserSelectB").value);
    p.laserLevel = static_cast<int>(params.get("laserLevel").value);
    return p;
}

// N10 生效参数（cpp 本地）：账本全参 ＋ 采集灯型覆写——laserOn=false（标点扫描
// A 模式）激光量强制 L=0（只开补光）。调用点均在逻辑线程（捕获 this 直读成员）
hal::CaptureParams effectiveN10(const ParamStore& params, bool laserOn) {
    hal::CaptureParams p = captureParamsFromAccount(params);
    if (!laserOn) p.laserLevel = 0;
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
      keyMgr_(std::make_unique<KeyManager>(cfg_.keys, [] { return nowMs(); })),
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
        JMW_LOG_DEBUG("08-DeviceManager", "[DeviceManager] 参数改账 {}={:.3f} confirmed={}", key, e.value,
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
        publishFault(code(DevFault::WarmupTimeout),
                     "预热超时（只报不停加热——停止机制待协议 §8-13）");
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
    // 配置快照（一次性）——串口/协议/门限的运行依据；真机排障第一落脚点
    JMW_LOG_INFO("08-DeviceManager",
        "[DeviceManager] open 配置: 串口={} 波特率={} 协议={} ack超时={}ms "
        "心跳阈={}ms 温度上限={}℃ 乱跳={}℃/s seq警={}",
        cfg_.serialPort, cfg_.baud,
        cfg_.protocol == serial::FrameCodec::Version::V3 ? "v3" : "v2",
        cfg_.ackTimeoutMs, cfg_.heartbeatTimeoutMs, cfg_.tempMaxC,
        cfg_.tempSpikeC, cfg_.seqGapWarn);
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
            }
        });
    }
    // ② MCU（测试 writeOverride 模式=逻辑开，不开真串口不起 rx 线程）
    mcu_->setProtocolVersion(cfg_.protocol);
    mcu_->setAckTimeoutMs(cfg_.ackTimeoutMs);
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
    // ③ 上行分流接线（onTemp 温度双警+记账+Warmup 喂入 / onKey→KeyManager / onStatus 记账）
    hal::McuUplink up;
    up.onTemp = [this](const serial::TempFrame& t) {
        checkTempFaults(t);                      // 0x0803 爆表 / 0x0804 乱跳（D-T13）
        lastTemps_ = t;
        tempRxTime_ = t.ts;
        warmup_->onTemperature(t.celsius[0], static_cast<int64_t>(t.ts));
    };
    up.onKey = [this](const serial::RawKeyEvent& k) { keyMgr_->onRawEvent(k); };
    up.onStatus = [this](const serial::StatusFrame& s) {
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] S 状态帧 code={:#x}（码表待协议 §8-8）", s.code);
    };
    mcu_->setUplink(up);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: uplink 接线 {}ms", el());
    // ⑤ 参数装载（Load 空档=全默认值；逐参数广播）+ 快照立即可读（单线程时刻）
    params_->bootstrap([] { return ""; });
    refreshParamSnapshot();
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: bootstrap+快照 {}ms", el());
    // ⑦ （2026-08-30 去自检模式：不再发 N12 Z1——固件进自检态会强制亮灯且
    //    串口响应劣化，"自检完灯不灭"根因。探测改用 N10 回显凭据，见
    //    MCUDriver::probeAutoPort；灯控全走 N10 参数直控）
    // （写权交接已不需要：串口写者恒为 MCUDriver 写线程，open/逻辑线程只入队）
    // ⑥ 逻辑线程（manualTick=true 测试跳过）。灯态策略：open 不发 N10（灯不亮）——
    // 灯只在 startCapture（N10 账本全参+H1）时亮、stopCapture 熄灯收口
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
    // —— 全量清理：补光灯/激光器关 + 退自检 + 冲写队列 + 关串口 + 关相机 ——
    if (mcu_->isOpen()) {
        mcu_->stopScan(nullptr);                // N11 H0：全停（灯/电机/触发）
        mcu_->exitSelfCheck(nullptr);           // N12 Z0：退自检模式
        mcu_->flushWrites(3000);                // 确保命令全部落线（3s 有界）
    }
    mcu_->close();                              // 关串口（停写线程+rx线程）
    if (camera_) {
        camera_->stopAsyncCapture();            // 停采集流
        camera_->close();                       // 关相机
    }
    // —— 状态复位（重开=全新会话）——
    standbyActive_ = false;
    camFaultLatched_ = false;
    camWasOpen_ = false;
    serialSilentLatched_ = false;
    tempHotLatched_ = false;
    tempSpikeLatched_ = false;
    prevTempsValid_ = false;
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
    drainPosts();                               // ① 编队任务排空（首——本拍落地）
    mcu_->pump();                               // ② 上行环排空（含 onAck 回填）
    keyMgr_->tick(nowMs());                     // ③ 手势判定 PC 域兜底
    for (const auto& g : keyMgr_->drain()) dispatchKeyGesture(g);
    mcu_->channelTick();                        // ④ 对账/重传/3 败收口
    const int64_t now = nowMs();
    warmup_->tick(now);                         // ⑤ 预热超时兜底
    // ⑥ v2 温度断流兜底：首个 T 帧后武装，超 1.2s 无帧 → N15 查询（发后重臂）
    if (cfg_.protocol == serial::FrameCodec::Version::V2 && tempRxTime_ > 0 &&
        now - static_cast<int64_t>(tempRxTime_) > kV2TempFallbackMs) {
        mcu_->queryTemperature(kV2TempQueryCh);
        tempRxTime_ = static_cast<TimestampMs>(now);
    }
    // ⑦ 相机掉线巡检（#1，D-T13 扩为任意时刻：曾开→isOpen 翻 false 边沿；只报
    //    不停手；相机重开清锚允许再触发——防爆屏）
    if (camera_ && camera_->isOpen()) {
        camWasOpen_ = true;
        camFaultLatched_ = false;               // 恢复清锚
    } else if (camera_ && camWasOpen_ && !camFaultLatched_) {
        camFaultLatched_ = true;
        publishFault(code(DevFault::CameraLost), "相机掉线（isOpen 翻 false；只报不停手）");
    }
    // ⑧ 串口无声（#2≡#10 心跳丢失同源）：收到过帧（lastRx>0）后停更超
    //    heartbeatTimeoutMs → 边沿一次；再收到任何有效帧即恢复清锚
    if (const int64_t lastRx = static_cast<int64_t>(mcu_->lastRxTime()); lastRx > 0) {
        if (now - lastRx > static_cast<int64_t>(cfg_.heartbeatTimeoutMs)) {
            if (!serialSilentLatched_) {
                serialSilentLatched_ = true;
                publishFault(code(DevFault::SerialSilent),
                             "串口无声(心跳丢失) 距末帧 " + std::to_string(now - lastRx) + "ms");
            }
        } else {
            serialSilentLatched_ = false;       // 心跳恢复清锚
        }
    }
    // ⑨ 按键队列挤爆（#6）：K 事件环满丢新计数增长即报（事件型——增量即边沿，
    //    无需恢复语义；计数单调累计）
    if (const uint64_t kd = mcu_->keyDropCount(); kd > lastKeyDrop_) {
        publishFault(code(DevFault::KeyRingOverflow),
                     "K 事件环满丢新 +" + std::to_string(kd - lastKeyDrop_) +
                         "（累计 " + std::to_string(kd) + "）");
        lastKeyDrop_ = kd;
    }
    // ⑩ seq 跳变丢帧（#9）：T seq 对账计数每拍增量达 seqGapWarn 即报（事件型；
    //    v2 无 seq 对账恒 0 不触发）
    if (const uint64_t sg = mcu_->seqGapCount();
        sg - lastSeqGap_ >= static_cast<uint64_t>(std::max(1, cfg_.seqGapWarn))) {
        publishFault(code(DevFault::SeqGap),
                     "T 帧 seq 跳变 +" + std::to_string(sg - lastSeqGap_) +
                         "（累计 " + std::to_string(sg) + "）");
        lastSeqGap_ = sg;
    }
    // ⑪ 快照刷新（菜单/温度/参数——跨线程读口统一互斥快照；轻拷）
    refreshParamSnapshot();
    {
        std::lock_guard<std::mutex> lock(menuSnapMtx_);
        menuSnap_ = menu_->state();
    }
    {
        std::lock_guard<std::mutex> lock(tempSnapMtx_);
        tempSnap_ = lastTemps_;
    }
    // ⑫ 启动自检状态机推进（无阻塞；闲时 stage=-1 直返）
    selfCheckTick(now);
}

void DeviceManager::testInjectRaw(const std::string& frameBytes) {
    mcu_->testInjectRaw(frameBytes);
}

// ============================================================================
// 切模式（Critical #1：门禁调用方线程同步判；过→命令组编队逻辑线程执行）
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
            publishFault(code(DevFault::CmdNoAck),   // 组中段 3 败=无应答（#7≡#8）
                         desc + " 命令组中段失败: " + payload);
            if (onDone) onDone(false);
            return;
        }
        sendSeq(std::move(rest), onDone);
    });
}

// 采集组公共步链（2026-08-30 真机定版）：N10→FLUSH。
// 定版依据：① N11 H1 实测（jmw 17:11:14）发出后固件串口静默 10s+（0x0802、
// 回显全无），紧随的 N10 石沉大海——"点扫描灯不亮"根因；工厂软件"开始扫描
// 仪"只发 N10（ScannerControl.cpp 口径），参数帧即灯控+采集参数，照齐。
// ② N11 H0 停止收口保留（真机已验证灯灭）。③ flushWrites：串口命令全部落线
// 后才开相机流——相机开流 USB3 风暴会把并行串口写饿死 2~5s（实测 N10 卡
// 5279ms）；先冲净写队列再动相机
std::vector<DeviceManager::SeqStep> DeviceManager::captureSeqSteps() {
    return {{"N10", [this](McuDone cb) {
                 mcu_->setCaptureParams(effectiveN10(*params_, captureLaserOn_), std::move(cb));
             }},
            {"FLUSH", [this](McuDone cb) {
                 mcu_->flushWrites(300);        // 有界：残余慢写最多 300ms
                 cb(true, "");
            }}};
}

void DeviceManager::enterScanOnLogic() {
    std::vector<SeqStep> seq;
    if (standbyActive_) {
        seq.push_back({"N13E0", [this](McuDone cb) {
                           mcu_->exitStandby([this, cb](bool ok, const std::string& p) {
                               if (ok) standbyActive_ = false;   // ACK 确认退出待机
                               cb(ok, p);
                           });
                       }});
    }
    auto cap = captureSeqSteps();                              // 待机退出→采集组（复用）
    seq.insert(seq.end(), std::make_move_iterator(cap.begin()),
               std::make_move_iterator(cap.end()));
    sendSeq(std::move(seq), [this](bool ok) {
        if (!ok) return;                         // Fault 已在 sendSeq 内报
        standbyActive_ = false;
        mode_->commit(DeviceMode::Scanning);     // 黑板「扫描中」
        mode_->setCapturing(true);               // + 采集开（02-D2：启停归 08）
        startStreamIfReady();
    });
}

void DeviceManager::enterCalibrationOnLogic() {
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

void DeviceManager::toIdleOnLogic() {
    sendSeq({{"N13E1", [this](McuDone cb) { mcu_->enterStandby(std::move(cb)); }}},
            [this](bool ok) {
                if (!ok) return;
                standbyActive_ = true;           // MCU 侧已待机
                mode_->commit(DeviceMode::Idle);
            });
}

// ============================================================================
// 采集启停（N10(账本)→N11H1→开流 / N11H0；不切模式；幂等；编队执行）
// ============================================================================

void DeviceManager::startCapture(bool laserOn) {
    post([this, laserOn] {
        captureLaserOn_ = laserOn;               // 采集灯型（逻辑线程属主；N10 组帧生效）
        startCaptureOnLogic();
    });
}

// —— 灯光直控（用户按钮直调；不启停采集——N10 灯字段即时生效，实测口径同
//    自检闪灯：固件收到 N10 即按新参数调灯，无需 H1）——
// bgOn/laserOn：true=取账本值，false=0。标点扫描 A 模式＝(true,false) 只开补光
void DeviceManager::setLights(bool bgOn, bool laserOn) {
    post([this, bgOn, laserOn] {
        if (!mcu_->isOpen()) return;
        hal::CaptureParams p = captureParamsFromAccount(*params_);
        p.bgLight = bgOn ? p.bgLight : 0;
        p.laserLevel = laserOn ? p.laserLevel : 0;
        mcu_->setCaptureParams(p, nullptr);
    });
}

// —— 打光场景封装（灯型三态；组合语义入口，按钮/工作流直调）——
void DeviceManager::lightsBgOnly() {
    setLights(/*bgOn=*/true, /*laserOn=*/false);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 打光场景: 只打补光灯（L=0）");
}

void DeviceManager::lightsBgAndCrossLaser() {
    const int tSel = static_cast<int>(params_->get("laserSelectA").value);
    const int vSel = static_cast<int>(params_->get("laserSelectB").value);
    if (tSel == vSel) {
        JMW_LOG_WARN("08-DeviceManager",
            "[DeviceManager] 打光场景: 左斜组 T{} 与右斜组 V{} 同组——固件帧序交替"
            "将只打一族激光线（调 laserSelectA/B 参数异组）", tSel, vSel);
    }
    setLights(/*bgOn=*/true, /*laserOn=*/true);
    JMW_LOG_INFO("08-DeviceManager",
        "[DeviceManager] 打光场景: 补光＋左右斜激光（左斜=T{} 右斜=V{}，交替归固件帧序）",
        tSel, vSel);
}

void DeviceManager::lightsAllOff() {
    setLights(/*bgOn=*/false, /*laserOn=*/false);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 打光场景: 全灭");
}

void DeviceManager::stopCapture() {
    post([this] { stopCaptureOnLogic(); });
}

void DeviceManager::startCaptureOnLogic() {
    if (mode_->isCapturing()) return;            // 幂等：黑板同值直返
    // 命令组 [N10(账本全参)→N11H1]（A-T17 修复：原仅发 N11H1，N10 断链）——
    // 组成功回调才擦板+开流；任一步 3 败由 sendSeq 短路+Fault
    sendSeq(captureSeqSteps(), [this](bool ok) {
        if (!ok) return;
        mode_->setCapturing(true);
        startStreamIfReady();
    });
}

void DeviceManager::stopCaptureOnLogic() {
    if (!mode_->isCapturing()) return;
    mcu_->stopScan([this](bool ok, const std::string& p) {
        if (!ok) {
            publishFault(code(DevFault::CmdNoAck), "N11H0 " + p);   // 3 败=无应答（#7≡#8）
            return;
        }
        mode_->setCapturing(false);
        // 灯态收口：单帧 N11 H0 即灭灯（真机+工厂软件同款验证；原 lightsOff
        // 补发属重复帧，2026-08-30 删）。停相机流前先冲队列——相机停流瞬间
        // USB 风暴会堵串口写（flush 有界 300ms）
        mcu_->flushWrites(300);
        if (camera_ && camera_->isOpen()) camera_->stopAsyncCapture();
    });
}

void DeviceManager::startStreamIfReady() {
    if (camera_ && camera_->isOpen() && frameCb_) camera_->startAsyncCapture(frameCb_);
}

// ============================================================================
// 启动自检（open 成功后由 app 调）——无阻塞状态机版：启动只发帧+置态，等待全由
// logicTick 每 10ms 的 selfCheckTick 分摊（单次 µs 级），逻辑线程不再被 sleep 堵死
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
        selfCheck_.stage = 0;
        selfCheck_.stageStartMs = nowMs();
        selfCheck_.report("mcuLink", mcu_->isOpen());

        // 自检亮灯（用户定版流程）：N10 H50 B50 T1 V2 L50（auto 搜口已探测合一
        // 发过且回显命中——通常此处直接判过；manual 口/回显未达时补发兜底），
        // 停留 5 秒 → N11 H0 收口（停扫描，固件关灯）
        hal::CaptureParams on{};
        on.freqHz = 50; on.bgLight = 50; on.laserSelectA = 1; on.laserSelectB = 2; on.laserLevel = 50;
        selfCheck_.expectEcho = "N10 H50 B50 T1 V2 L50";
        if (mcu_->lastEchoPayload() != selfCheck_.expectEcho) {
            mcu_->setCaptureParams(on, nullptr);   // 兜底补发（manual 口路径）
        }
    });
}

void DeviceManager::selfCheckTick(int64_t nowMs_) {
    if (selfCheck_.stage < 0) return;
    switch (selfCheck_.stage) {
    case 0:  // 等 N10 回显——实测"相机配置后第一笔串口写"可被 USB 驱动卡（本机
        // 实测 5216ms，jmw 日志慢写行佐证；写线程内部消化，但回显延迟到）：
        // 窗口给足 8s；正常路径 <100ms 即过
        if (mcu_->lastEchoPayload() == selfCheck_.expectEcho) {
            selfCheck_.report("bgLight", true);
            selfCheck_.report("laser", true);
            selfCheck_.stage = 1;
            selfCheck_.stageStartMs = nowMs_;      // 闪亮窗口起
        } else if (nowMs_ - selfCheck_.stageStartMs > 8000) {
            selfCheck_.report("bgLight", false);
            selfCheck_.report("laser", false);
            selfCheck_.stage = 1;                  // 仍走 stage1（闪亮窗缩短+相机启动不跳过
            selfCheck_.stageStartMs = nowMs_;      // ——灯败与相机验证独立，原直跳 stage2
        }                                          //   会漏 startAsyncCapture→相机恒 0 帧）
        break;
    case 1:  // 补光灯/激光器亮 5 秒（停留窗口；不起相机流——相机 USB 流量会
             // 干扰 CH343 串口收发，2026-08-30 真机实测灭灯帧被丢的诱因）
        if (nowMs_ - selfCheck_.stageStartMs >= 5000) {
            selfCheck_.stage = 2;
            selfCheck_.stageStartMs = nowMs();
        }
        break;
    case 2: {  // 收口：N11 H0（停扫描——固件关灯），用户定版流程，不发 N10 灭灯。
        // camera 项以相机 open 状态判定（不起流验证——避免 USB 扰动串口）
        mcu_->stopScan(nullptr);              // N11 H0
        mcu_->flushWrites(2000);
        selfCheck_.report("camera", camera_ && camera_->isOpen());
        JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 启动自检完成（相机 open={}）",
                     camera_ && camera_->isOpen());
        selfCheck_.stage = -1;
        break;
    }
    }
}

// ============================================================================
// 预热（§4-2：N14 T<目标>；看火员只报稳/超，不停加热；编队执行）
// ============================================================================

void DeviceManager::startWarmup(int targetC, std::function<void(bool stable)> done) {
    post([this, targetC, done = std::move(done)]() mutable {
        warmupDone_ = std::move(done);           // 重开=后值胜出（旧未触发作废）
        warmup_->start(targetC);
        mcu_->setHeatTarget(targetC, [this, targetC](bool ok, const std::string& p) {
            if (!ok) publishFault(code(DevFault::CmdNoAck),           // 3 败=无应答（#7≡#8）
                                  "N14T" + std::to_string(targetC) + " " + p);
        });
    });
}

// ============================================================================
// 按键链接线（KeySemantics 11 出口）
// ============================================================================

void DeviceManager::buildKeyActions() {
    KeySemActions a;
    a.captureToggle = [this] {
        if (mode_->isCapturing()) stopCaptureOnLogic();
        else startCaptureOnLogic();              // 逻辑线程内直调本体（免编队绕行）
    };
    a.menuSelect = [this] {                      // 按 cursor 分叉（裁判只给信号）
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

void DeviceManager::dispatchKeyGesture(const KeyGesture& g) {
    semantics_->onGesture(g, menu_->state());
}

// ============================================================================
// 参数下发（ParamStore Dispatch：exposure 相机直设 / N10 组参）+ 快照双口
// ============================================================================

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
    // 空闲仅记账 done(true,false)（enterScan 时自账本组帧下发）。v2 通道发不等
    // → done(true,false)
    if (mode_->isCapturing()) {
        mcu_->setCaptureParams(effectiveN10(*params_, captureLaserOn_),
                               [done](bool ok, const std::string&) { done(ok, ok); });
    } else {
        done(true, false);
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
        frameCb_ = std::move(cb);                    // 帧出口记账归逻辑线程属主
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
    const int n = std::min<int>(t.channels, 4);
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
    // #4 乱跳：相邻 T 帧同路 |Δ|/Δt >tempSpikeC ℃/s → 边沿一次；次帧平稳清锚
    //（dt 下钳 1ms：同拍连注两帧按 1ms 算——测试回灌口径）
    if (prevTempsValid_) {
        const int pn = std::min<int>(prevTemps_.channels, 4);
        const int64_t dtMs =
            std::max<int64_t>(1, static_cast<int64_t>(t.ts) - static_cast<int64_t>(prevTemps_.ts));
        bool spiky = false;
        double worst = 0.0;
        for (int i = 0; i < n && i < pn; ++i) {
            const double rate =
                std::abs(t.celsius[i] - prevTemps_.celsius[i]) * 1000.0 / static_cast<double>(dtMs);
            if (rate > cfg_.tempSpikeC) {
                spiky = true;
                worst = std::max(worst, rate);
            }
        }
        if (spiky && !tempSpikeLatched_) {
            tempSpikeLatched_ = true;
            publishFault(code(DevFault::TempSpike),
                         "温度乱跳: 峰值速率 " + std::to_string(worst) + "C/s");
        } else if (!spiky) {
            tempSpikeLatched_ = false;          // 次帧平稳清锚
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
