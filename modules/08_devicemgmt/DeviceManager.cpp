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
// v2 兜底判据共用）
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
        {"freqHz", 120.0, 1.0, 200.0},        // N10 H 拍照频率（默认 120——260927 用户口径回调（60→120→30→120）；H120×双目全分辨率≈6Gbps 超 USB3 单控制器上限、帧号回退 ~2 次/秒（2b3155d 实测留痕，用户知情）；协议 260831 域 1-200）
        {"bgLight", 10.0, 0.0, 100.0},         // N10 B 补光（默认 10——B 模式成功配置基线；B30 过曝毁检测 ROI 跌至 4）
        {"laserLevel", 40.0, 0.0, 100.0},      // N10 L 激光强度（默认 40；60 过亮→40 折中）
    };
}

// N10 生效参数（cpp 本地——不进头防 IMCU.h 类型泄漏）：账本值 ＋ ScanMode 四管
// 掩码映射（261002 临时测试机协议——扫描仪损坏临时环境，管语义改：T=精细线、
// V=左斜线、C=右斜线、D=无对应线〔测试机未接，占位〕；原 260831/260910 口径
// 「面片 T1V1／精细 D／深孔 C」作废，回正式机须回退）：标志点 A（MarkerOnly）
// T0V0C0D0＋L=0；面片 B（MarkerPlusLaser）V1C1（左斜+右斜两管轮流）；精细 C
// （FineScan）T 管；深孔 D（DeepHoleScan）D 管（无对应线，仅协议占位）。
// 调用点均在逻辑线程
constexpr int kMarkerOnlyBg = 10;   // A 模式补光基线（260927 真机回调：B40 过曝——标志点
                                    // 检测只剩 1 点/光流配准 0<3 全败；B10=面片模式同场景
                                    // 36 点实证基线。原 B40「高补光代偿」口径作废，待定表）
hal::CaptureParams effectiveN10(const ParamStore& params, Scanner::ScanMode mode) {
    hal::CaptureParams p;
    p.freqHz = static_cast<int>(params.get("freqHz").value);
    // 标点模式帧率钳制 ≤30Hz（260927 真机实证）：H120×全分辨率≈6Gbps 超 USB3
    // 单控制器上限——扫描中相机链路反复重开（配对日志 L.fid=0 频现/R 检测 12↔6
    // 跳），标志点检测/立体匹配全线恶化。标点扫描静止对板无需高帧率；面片等
    // 激光模式维持账本值（用户 120 口径不变）
    if (mode == Scanner::ScanMode::MarkerOnly && p.freqHz > 30) p.freqHz = 30;
    p.bgLight = (mode == Scanner::ScanMode::MarkerOnly)
                    ? kMarkerOnlyBg
                    : static_cast<int>(params.get("bgLight").value);
    p.laserLevel = (mode == Scanner::ScanMode::MarkerOnly)
                       ? 0
                       : static_cast<int>(params.get("laserLevel").value);
    p.laserT = (mode == Scanner::ScanMode::FineScan) ? 1 : 0;          // T=精细线（261002 临时）
    p.laserV = (mode == Scanner::ScanMode::MarkerPlusLaser) ? 1 : 0;   // V=左斜线（面片交叉对）
    p.laserC = (mode == Scanner::ScanMode::MarkerPlusLaser) ? 1 : 0;   // C=右斜线（面片交叉对）
    p.laserD = (mode == Scanner::ScanMode::DeepHoleScan) ? 1 : 0;      // D=无对应线（占位）
    return p;
}

// 按键模式光标 → ScanMode（MenuLogic §4.2.3：1 精细/2 深孔/3 普通交叉，默认 3）。
// 260927 完善按键管理：cycleMode（中键双击）与菜单①②确认后经此落地
// lastCaptureMode_——按键切模式真正生效（原缺口：仅菜单记账，采集恒用 app
// 上次传入模式）。切模式在非采集态（M2/菜单键过 gate=!capturing 门禁），
// 落地值于下次 captureToggle 启采时组帧 N10 四管掩码
constexpr Scanner::ScanMode scanModeFromMenuCursor(int modeCursor) {
    return modeCursor == 1 ? Scanner::ScanMode::FineScan
         : modeCursor == 2 ? Scanner::ScanMode::DeepHoleScan
                           : Scanner::ScanMode::MarkerPlusLaser;
}

} // namespace

// ============================================================================
// 构造 / 析构
// ============================================================================

DeviceManager::DeviceManager(DeviceConfig cfg, GateQuery gate, infra::EventBus* bus,
                             CameraFactory camFactory, SerialWriteOverride serialWrite,
                             ParamIo paramIo)
    : cfg_(std::move(cfg)),
      paramIo_(std::move(paramIo)),
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
        // G7 落盘脏账（260927 补缺口）：改账即脏＋记时（bootstrap 装载不算脏），
        // logicTick 拍尾防抖 2s 冲刷、close 兜底
        if (!paramBootLoading_) {
            paramDirty_ = true;
            paramLastChangeMs_ = nowMs();
        }
        int64_t idx = -1;                       // 参数索引=specs 登记序号（Minor #9）
        for (size_t i = 0; i < paramKeys_.size(); ++i)
            if (paramKeys_[i] == key) idx = static_cast<int64_t>(i);
        publishEvent(EventType::UserDefined, 1000 + idx, 0);   // base 暂无 ParamChanged——占位
        //（260927 错峰：p1=1000+idx——原裸 idx(0..3) 与菜单③④/视点(1..4/100)在
        // UserDefined 上撞车，laserLevel 改账(p1=3)会被误当「扫描完成」消费）
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
            } else {
                // 曝光保持相机自动（用户口径 2026-08-31：固定值试验 10/5/3ms
                // 均劣于自动——3ms+B30 时 ROI 暴增 137/76 触发 marker_match
                // maxPoints=100 上限异常，pChain 整帧报废）
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
    // ②b 相机 ROI 应用：UI 预设分辨率（setCameraResolution 记账）→ 开相机后落地；
    //     0 值=未设过，用传感器缺省不动
    if (camera_) {
        const int pw = m_pendingResW_.load(std::memory_order_acquire);
        const int ph = m_pendingResH_.load(std::memory_order_acquire);
        if (pw > 0 && ph > 0) {
            const Result rr = camera_->setResolution(pw, ph);
            JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 应用相机分辨率 {}x{}: {}", pw, ph, rr.message);
        }
    }
    // ③ 上行分流接线（onTemp 温度双警+记账+Warmup 喂入 / onGesture→KeySemantics 直收
    //     MCU 已判手势 / onShotCount G03 触发计数记账）
    hal::McuUplink up;
    up.onTemp = [this](const serial::TempFrame& t) {
        checkTempFaults(t);                      // 0x0803 爆表 / 0x0804 乱跳（D-T13）
        lastTemps_ = t;
        tempRxTime_ = t.ts;
        warmup_->onTemperature(t.celsius[0], static_cast<int64_t>(t.ts));
    };
    up.onGesture = [this](const serial::GestureEvent& g) { dispatchGesture(g); };
    up.onShotCount = [this](const serial::ShotCountFrame& sc) {
        JMW_LOG_DEBUG("08-DeviceManager", "[DeviceManager] G03 触发计数累计 {}", sc.count);
    };
    mcu_->setUplink(up);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] open 计时: uplink 接线 {}ms", el());
    // ⑤ 参数装载（Load 空档=全默认值；逐参数广播）+ 快照立即可读（单线程时刻）
    paramBootLoading_ = true;       // 装载期改账不算脏（档值回写档=无意义写）
    params_->bootstrap([&] { return paramIo_.load ? paramIo_.load() : std::string(); });
    paramBootLoading_ = false;
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
    // G7 落盘 close 兜底：脏账无条件冲一次（防抖窗内退出的最后改动作不丢）
    if (paramIo_.persist && paramDirty_) {
        if (params_->persist(paramIo_.persist)) paramDirty_ = false;
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
    mcu_->pump();                               // ② 上行环排空（含 onAck 回填）；手势已由
                                                //    MCU 判定（G01），pump 直喂 semantics
    mcu_->channelTick();                        // ③ 对账/重传/3 败收口
    const int64_t now = nowMs();
    warmup_->tick(now);                         // ④ 预热超时兜底
    // ⑤ v2 温度断流兜底：首个 T 帧后武装，超 1.2s 无帧 → N15 查询（发后重臂）
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
    // ⑧ 按键队列挤爆（#6）：G01 手势环满丢新计数增长即报（事件型——增量即边沿，
    //    无需恢复语义；计数单调累计；260831 协议源改 G01 手势环）
    if (const uint64_t kd = mcu_->keyDropCount(); kd > lastKeyDrop_) {
        publishFault(code(DevFault::KeyRingOverflow),
                     "G01 手势环满丢新 +" + std::to_string(kd - lastKeyDrop_) +
                         "（累计 " + std::to_string(kd) + "）");
        lastKeyDrop_ = kd;
    }
    // ⑨ 帧计数对账丢帧（#9）：G03 触发计数跳变对账每拍增量达 seqGapWarn 即报
    //    （事件型；260831 协议改 G03 S 计数源——原 T-seq 跳变弃，v2 无 seq 恒不触发）
    if (const uint64_t sg = mcu_->seqGapCount();
        sg - lastSeqGap_ >= static_cast<uint64_t>(std::max(1, cfg_.seqGapWarn))) {
        publishFault(code(DevFault::SeqGap),
                     "G03 帧计数对账丢帧 +" + std::to_string(sg - lastSeqGap_) +
                         "（累计 " + std::to_string(sg) + "）");
        lastSeqGap_ = sg;
    }
    // ⑩ 快照刷新（菜单/温度/参数——跨线程读口统一互斥快照；轻拷）
    refreshParamSnapshot();
    {
        std::lock_guard<std::mutex> lock(menuSnapMtx_);
        menuSnap_ = menu_->state();
    }
    // 键控回显快照（260927）：档位梯状态随菜单快照同拍刷新（UI 状态栏常显）
    ladderIndexSnap_.store(ladder_.index(), std::memory_order_relaxed);
    ladderSizeSnap_.store(static_cast<int>(ladder_.ladder().size()), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(tempSnapMtx_);
        tempSnap_ = lastTemps_;
    }
    // ⑩b G7 参数落盘（拍尾冲刷）：脏账距末次改账 ≥2s 才落（防抖——连调滑条不
    // 逐拍写盘）；失败保脏下拍重试；close 另有无条件兜底
    persistParamsIfDue(now);
    // ⑪ 启动自检状态机推进（无阻塞；闲时 stage=-1 直返）
    selfCheckTick(now);
    //（⑫ 链路自动恢复已撤——260927 用户口径：改时间戳直配（CameraControl 方案B）
    //   零中断配对，不再「检测到丢帧→停扫重启对齐帧」；实现留档 git aaad29c）
}

void DeviceManager::testInjectRaw(const std::string& frameBytes) {
    mcu_->testInjectRaw(frameBytes);
}

void DeviceManager::testInjectTextLine(const std::string& line) {
    mcu_->testInjectTextLine(line);           // 透传：测试缝（v2 G01 手势帧文本行路径）
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

// 采集组公共步链（2026-08-30 三次定版·用户口径）：N10→FLUSH。
// 启动只发一帧 N10（参数+灯控）；不发 N11 H1。FLUSH：串口命令落线后才开
// 相机流（相机开流 USB3 风暴会饿死并行串口写 2~5s）。停止见 stopCaptureOnLogic
// （单帧 N11 H0，不发 N10）
std::vector<DeviceManager::SeqStep> DeviceManager::captureSeqSteps() {
    return {{"N10", [this](McuDone cb) {
                 mcu_->setCaptureParams(effectiveN10(*params_, lastCaptureMode_.load(std::memory_order_relaxed)), std::move(cb));
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

void DeviceManager::startCapture(Scanner::ScanMode mode) {
    post([this, mode] {
        lastCaptureMode_.store(mode, std::memory_order_relaxed);   // 采集灯型（逻辑线程写；N10 组帧生效）
        startCaptureOnLogic();
    });
}

// 只设模式不启采（260927 就绪流程）：UI 模式键→armScanSession→此口记账，
// 正式开扫由设备 M 键（captureToggle→startCaptureOnLogic）触发——四管掩码
// 组帧推迟到那一刻
void DeviceManager::setCaptureMode(Scanner::ScanMode mode) {
    post([this, mode] {
        lastCaptureMode_.store(mode, std::memory_order_relaxed);
        JMW_LOG_INFO("08-DeviceManager",
            "[DeviceManager] 采集模式就绪：ScanMode={}（等设备 M 键开扫）",
            static_cast<int>(mode));
    });
}

// G8 设备指示灯（260927 补缺口）：N14 S1-S4 状态回显——去重（同码不重发）后
// 编队下发；MCU 未开直返且**不记码**（open 后 app 补发当前态才能穿透去重落地）
void DeviceManager::setDeviceLed(int s1to4) {
    post([this, s1to4] {
        if (s1to4 < 1 || s1to4 > 4) return;
        if (!mcu_->isOpen()) return;              // 未开：不记码不发（补发口径见上）
        if (s1to4 == lastDeviceLed_) return;
        lastDeviceLed_ = s1to4;
        mcu_->setDeviceLed(s1to4);
        JMW_LOG_DEBUG("08-DeviceManager", "[DeviceManager] 设备指示灯 S{}", s1to4);
    });
}

// —— 灯光直控（用户按钮直调；不启停采集——N10 灯字段即时生效，实测口径同
//    自检闪灯：固件收到 N10 即按新参数调灯，无需 H1）——
// 基线=面片灯型（V1C1＋账本 B/L，261002 临时测试机管语义）；bgOn/laserOn：false 压 0。标点扫描 A
// 模式＝(true,false) 只开补光
void DeviceManager::setLights(bool bgOn, bool laserOn) {
    post([this, bgOn, laserOn] {
        if (!mcu_->isOpen()) return;
        hal::CaptureParams p = effectiveN10(*params_, Scanner::ScanMode::MarkerPlusLaser);
        if (!bgOn) p.bgLight = 0;
        if (!laserOn) {
            p.laserT = p.laserV = p.laserC = p.laserD = 0;
            p.laserLevel = 0;
        }
        mcu_->setCaptureParams(p, nullptr);
    });
}

// —— 打光场景封装（灯型三态；组合语义入口，按钮/工作流直调）——
void DeviceManager::lightsBgOnly() {
    setLights(/*bgOn=*/true, /*laserOn=*/false);
    JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 打光场景: 只打补光灯（激光管全关）");
}

void DeviceManager::lightsBgAndCrossLaser() {
    setLights(/*bgOn=*/true, /*laserOn=*/true);    // V1C1（面片交叉激光，交替归固件帧序）
    JMW_LOG_INFO("08-DeviceManager",
        "[DeviceManager] 打光场景: 补光＋左右斜激光（V1C1，交替归固件帧序）");
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
    // —— 启动顺序（2026-09-06 帧号错位根因修正）：**相机先启→再发 N10** ——
    // 原序（N10→回调→startStream）在 N10 到达后 MCU 立即触发，而相机尚在
    // 配置中（左 42ms/右 82ms 后才就绪）——左比右多吃 2~3 个触发脉冲，
    // GetFrameID 偏移恒 2~3，严格配对全丢（实测 L=92 R=89）。
    // 改序后两台相机同时等触发，N10 到达时同步收第一个脉冲 → 偏移 ±1。
    startStreamIfReady();                        // ① 相机先就绪等触发
    sendSeq(captureSeqSteps(), [this](bool ok) {
        if (!ok) return;
        mode_->setCapturing(true);
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

        // N12 温度回传兜底（260927 用户口径·条件发送）：N10 落线后 0.5s 内有 G02
        // 温度上行→不发 N12（真机 A/B 实证：伴随 N12 固件不点灯、N10 单发即亮）；
        // 未到→auto 搜口内补发 N12 T5（probeN12TSent 凭据）。此处（自检入口）恒
        // 不发——manual 口走 stage0 1s 活证观察，温度无源由固件方定位，不盲补
        //（260926 曾在此无条件发 N12 T5，260927 证伪撤销）

        // 自检亮灯（用户定版流程）：N10 H50 B50 T0 V1 C1 D0 L50（七参·261002
        // 临时测试机口径：V/C=左右斜线交叉点灯，探测帧同参）。
        // 只发一个：auto 搜口已发同参 N10（兼探测+点灯，probeN10Sent 凭据）——此处
        // 省略；manual 口（无探测帧）且回显未达才补发兜底。亮灯总窗 ~3s（stage0
        // 1s 上行活证过关＋stage1 停留 2s，2026-09-27 用户口径「灯亮三秒」；
        // 历程：5s→1s〔260906〕→~2s〔260926〕→~9s〔260927 午〕→~3s）
        // → N11 H0 收口（停扫描，固件关灯）
        hal::CaptureParams on{};
        on.freqHz = 50; on.bgLight = 50;
        on.laserT = 0; on.laserV = 1; on.laserC = 1; on.laserD = 0;
        on.laserLevel = 50;
        selfCheck_.expectEcho = "N10 H50 B50 T0 V1 C1 D0 L50";
        if (mcu_->lastEchoPayload() != selfCheck_.expectEcho && !mcu_->probeN10Sent()) {
            mcu_->setCaptureParams(on, nullptr);   // 兜底补发（仅 manual 口）
        }
    });
}

void DeviceManager::selfCheckTick(int64_t nowMs_) {
    if (selfCheck_.stage < 0) return;
    switch (selfCheck_.stage) {
    case 0:  // 等 N10 回显——实测"相机配置后第一笔串口写"可被 USB 驱动卡（本机
        // 实测 5216ms，jmw 日志慢写行佐证；写线程内部消化，但回显延迟到）：
        // 回显窗给足 8s；正常路径 <100ms 即过。无回显固件（260919 现行——只周期
        // 上行 G02 温度 ~300ms）不再恒走满 8s：1s 后凭上行活证提前过关（降级凭据
        // 同口径——3s 内收到过任一上行帧=链路+固件活，N10 已落线；回环线不产上行
        // 帧不误判）→ 亮灯总窗 ~3s（此处 1s＋stage1 停留 2s，2026-09-27 用户口径
        // 「灯亮三秒」；原恒走满 8s 灯亮 ~9s、260926 曾缩 ~2s）；上行亦静默才落到 8s 判败（判败 →
        // system_ready 永不触发 → 状态机卡 Init → start_scan 恒被拒，260919 真机实证）
        {
            const int64_t elapsed = nowMs_ - selfCheck_.stageStartMs;
            const bool linkAlive =
                mcu_->lastRxTime() > 0 &&
                nowMs_ - static_cast<int64_t>(mcu_->lastRxTime()) < 3000;
            if (mcu_->lastEchoPayload() == selfCheck_.expectEcho) {
                selfCheck_.report("bgLight", true);
                selfCheck_.report("laser", true);
                selfCheck_.stage = 1;
                selfCheck_.stageStartMs = nowMs_;  // 闪亮窗口起
            } else if ((elapsed >= 1000 && linkAlive) || elapsed > 8000) {
                // 活证提前过关 / 硬超时静默判败——同一出口（stage1 闪亮窗与相机启动
                // 不跳过——灯败与相机验证独立，原直跳 stage2 会漏 startAsyncCapture
                // → 相机恒 0 帧）
                selfCheck_.report("bgLight", linkAlive);
                selfCheck_.report("laser", linkAlive);
                JMW_LOG_WARN("08-DeviceManager",
                    "[DeviceManager] 自检回显未达（固件无回显）——{}",
                    linkAlive ? "1s 上行活证提前过关（降级凭据）" : "8s 上行亦静默，判败");
                selfCheck_.stage = 1;
                selfCheck_.stageStartMs = nowMs_;
            }
        }
        break;
    case 1:  // 补光灯/激光器停留窗口（不起相机流——相机 USB 流量会
             // 干扰 CH343 串口收发，2026-08-30 真机实测灭灯帧被丢的诱因。
             // 8s→2s：260927 用户口径「灯亮三秒」（总窗=stage0 1s＋此处 2s；
             // 历史 5s→1s〔260906〕→~2s〔260926〕→停留 8s〔260927 午〕→2s）
        if (nowMs_ - selfCheck_.stageStartMs >= 2000) {
            selfCheck_.stage = 2;
            selfCheck_.stageStartMs = nowMs();
        }
        break;
    case 2: {  // 收口：N11 H0（停扫描——固件关灯），用户定版流程，不发 N10 灭灯。
        // camera 项以相机 open 状态判定（不起流验证——避免 USB 扰动串口）
        mcu_->stopScan(nullptr);              // N11 H0
        mcu_->flushWrites(2000);
        // G02 回传周期定版 1s（260927 用户口径「下位机 1s 上传一次」）：自检收口
        //（灯已灭——避开「伴随 N12 固件不点灯」干扰窗）后发一次 N12 T1000。
        // 真机实证（260927）：T1000 生效——G02 周期 100ms→1s；T5 曾无效疑被固件
        // 钳到 100ms 下限。收口后发送点灯不受影响。probe 兜底已发过则不重发
        if (!mcu_->probeN12TSent()) {
            mcu_->setTempReportInterval(1000);
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] G02 回传周期定版 1s——自检收口后发 N12 T1000");
        }
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
            // 标点会话隔离（260927 用户口径）：标点扫描中按键只做启停——①② 的
            // 模式落地只对激光族（面片/精细/深孔）生效，标点会话不许经按键切走
            if (lastCaptureMode_.load(std::memory_order_relaxed) ==
                Scanner::ScanMode::MarkerOnly) {
                JMW_LOG_INFO("08-DeviceManager",
                    "[DeviceManager] 标点扫描会话中——菜单①②模式设定被拒（标点只启停）");
                return;
            }
            // 视野项①②：确认模式光标 → 采集模式落地（260927 完善——原仅日志，
            // 按键切的模式从不生效；现同步 lastCaptureMode_，下次启采按此组帧）
            lastCaptureMode_.store(scanModeFromMenuCursor(menu_->state().modeCursor), std::memory_order_relaxed);
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] 菜单选中①/②：扫描模式落地 modeCursor={} → ScanMode={}",
                menu_->state().modeCursor, static_cast<int>(lastCaptureMode_.load(std::memory_order_relaxed)));
        } else {
            // ③扫描完成/④开始后处理：派工作流归 app 订阅——门面只广播。
            // UserDefined+param1=3/4（Important #4：待 base 增专用事件类型，人工过会后改）
            publishEvent(EventType::UserDefined, cur, 0);
            JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] 菜单选中③/④：派发工作流事件 cursor={}", cur);
        }
    };
    a.cycleMode = [this] {                       // 中键双击切模式：记账＋即时落地
        // 标点会话隔离（260927 用户口径）：标点扫描中双击不切面片/精细/深孔——
        // 按键只启停标点；模式切换仅在激光族会话（面片/精细/深孔）中可用
        if (lastCaptureMode_.load(std::memory_order_relaxed) ==
            Scanner::ScanMode::MarkerOnly) {
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] 标点扫描会话中——切模式手势丢弃（标点只启停）");
            return;
        }
        menu_->apply(MenuOp::CycleMode);
        lastCaptureMode_.store(scanModeFromMenuCursor(menu_->state().modeCursor),
                               std::memory_order_relaxed);
        if (mode_->isCapturing()) {
            // 采集中切模式（260927 用户口径）：N10 全参实时重发——四管掩码即时
            // 换灯（D1 口径同滑条改参）；帧级激光种类由 scanMode 随帧分派接管
            mcu_->setCaptureParams(
                effectiveN10(*params_, lastCaptureMode_.load(std::memory_order_relaxed)),
                nullptr);
            JMW_LOG_WARN("08-DeviceManager",
                "[DeviceManager] 采集中切模式：N10 全参重发 modeCursor={} → ScanMode={}"
                "（激光掩码即时生效）",
                menu_->state().modeCursor,
                static_cast<int>(lastCaptureMode_.load(std::memory_order_relaxed)));
        } else {
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] 中键双击切模式：modeCursor={} → ScanMode={}（下次启采生效）",
                menu_->state().modeCursor,
                static_cast<int>(lastCaptureMode_.load(std::memory_order_relaxed)));
        }
    };
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
        // G6 档位梯（260927 补缺口）：Brightness 上下文左右键＝档位步进（曝光/
        // 激光/补光三参组合，内置 3 档钳制不环绕）——经 ParamStore 记账，下发/
        // 广播/落盘全走既有链（采集中自动 N10 全参重发 D1 口径）
        if (ladder_.step(steps > 0 ? +1 : -1)) {
            const auto& s = ladder_.current();
            params_->setValue("exposure",   s.exposureMs,  ParamEntry::Source::Key);
            params_->setValue("laserLevel", s.laserLevel,  ParamEntry::Source::Key);
            params_->setValue("bgLight",    s.bgLight,     ParamEntry::Source::Key);
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] 档位 {} → {}/{}（曝光{:.0f}ms 激光{:.0f} 补光{:.0f}）",
                ladder_.index(), steps > 0 ? "上调" : "下调",
                static_cast<int>(ladder_.ladder().size()),
                s.exposureMs, s.laserLevel, s.bgLight);
        } else {
            JMW_LOG_INFO("08-DeviceManager",
                "[DeviceManager] 档位已到顶/底（index={}）——步进无效", ladder_.index());
        }
    } else if (ctx == MenuState::AdjustCtx::View && steps != 0) {
        // View 上下文＝视点缩放（260927 按设计接线：协作文档「档位与视点调节」
        // ——事件口径 UserDefined p1=100，param2=+1 拉近/-1 拉远；app 订阅方驱动
        // OSGWidget::zoomView）
        publishEvent(EventType::UserDefined, 100, steps > 0 ? 1 : -1);
        JMW_LOG_INFO("08-DeviceManager",
            "[DeviceManager] 视点缩放：{}（View 上下文按键步进）", steps > 0 ? "拉近" : "拉远");
    } else {
        JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 调节步进 ctx={}（本上下文无动作）",
                     static_cast<int>(ctx));
    }
}

void DeviceManager::dispatchGesture(const serial::GestureEvent& g) {
    semantics_->onGesture(g, menu_->state());   // KeySemantics 直收 MCU 已判手势（KeyManager 退役）
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
        mcu_->setCaptureParams(effectiveN10(*params_, lastCaptureMode_.load(std::memory_order_relaxed)),
                               [done](bool ok, const std::string&) { done(ok, ok); });
    } else {
        done(true, false);
    }
}

void DeviceManager::refreshParamSnapshot() {
    std::lock_guard<std::mutex> lock(snapshotMutex_);
    for (const auto& k : paramKeys_) paramSnapshot_[k] = params_->get(k);
}

// G7 落盘拍尾冲刷（260927 补缺口）：脏账静默 ≥2s → persist；成功清脏，失败保脏
// 下拍重试（I/O 毛刺自愈）。无注入（测试）＝恒直返
void DeviceManager::persistParamsIfDue(int64_t nowMs_) {
    if (!paramIo_.persist || !paramDirty_) return;
    if (nowMs_ - paramLastChangeMs_ < 2000) return;   // 防抖窗内（连调不逐拍写盘）
    if (params_->persist(paramIo_.persist)) {
        paramDirty_ = false;
        JMW_LOG_INFO("08-DeviceManager", "[DeviceManager] 参数档已落盘（防抖 2s 冲刷）");
    } else {
        JMW_LOG_WARN("08-DeviceManager", "[DeviceManager] 参数档落盘失败——保脏下拍重试");
    }
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

Result DeviceManager::setCameraContrast(int leftValue, int rightValue) {
    if (!camera_) return Result::fail("未配置相机");
    post([this, leftValue, rightValue] {
        if (camera_ && camera_->isOpen()) camera_->setContrast(leftValue, rightValue);
    });
    return Result::ok("已编队");
}

Result DeviceManager::setCameraResolution(int width, int height) {
    if (!camera_) return Result::fail("未配置相机");
    if (width <= 0 || height <= 0) return Result::fail("分辨率非法");
    // 记账为待生效（open 重开应用）；采集中不变 ROI（抖动扰流，停采/重开生效）
    m_pendingResW_.store(width, std::memory_order_release);
    m_pendingResH_.store(height, std::memory_order_release);
    post([this, width, height] {
        if (camera_ && camera_->isOpen() && !isCapturing())
            camera_->setResolution(width, height);   // 未采集中立即生效
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
    // #4 乱跳：相邻 T 帧同路 |Δ| 超 tempSpikeAbsC（绝对钳大跳）或 速率超 tempSpikeC ℃/s
    //（双阈值——速率灵敏、绝对防量化；任一命中→边沿一次；次帧平稳清锚）
    //（dt 下钳 1ms：同拍连注两帧按 1ms 算——速率档测试回灌口径；绝对档不受 dt 影响）
    if (prevTempsValid_) {
        const int pn = std::min<int>(prevTemps_.channels, 4);
        const int64_t dtMs =
            std::max<int64_t>(1, static_cast<int64_t>(t.ts) - static_cast<int64_t>(prevTemps_.ts));
        bool spiky = false;
        double worst = 0.0;         // 峰值速率（绝对档仅置位，报文带差量）
        double worstAbs = 0.0;      // 峰值绝对差
        for (int i = 0; i < n && i < pn; ++i) {
            const double d = std::abs(t.celsius[i] - prevTemps_.celsius[i]);
            const double rate = d * 1000.0 / static_cast<double>(dtMs);
            if (rate > cfg_.tempSpikeC) {
                spiky = true;
                worst = std::max(worst, rate);
            }
            if (d > cfg_.tempSpikeAbsC) {
                spiky = true;
                worstAbs = std::max(worstAbs, d);
            }
        }
        if (spiky && !tempSpikeLatched_) {
            tempSpikeLatched_ = true;
            std::string msg = "温度乱跳: ";
            if (worst > 0.0) msg += "峰值速率 " + std::to_string(worst) + "C/s";
            if (worst > 0.0 && worstAbs > 0.0) msg += "，";
            if (worstAbs > 0.0) msg += "峰值差 " + std::to_string(worstAbs) + "C";
            publishFault(code(DevFault::TempSpike), msg);
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
    // FaultOccurred 统一契约：sourceId=模块源、param1=severity、param2=码
    // （08 故障均为 Error 级——对齐 10 FaultHandler 订阅语义 2026-09-20）
    e.param1 = static_cast<int64_t>(Scanner::FaultSeverity::Error);
    e.param2 = code;
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
