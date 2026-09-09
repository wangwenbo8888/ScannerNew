// ============================================================================
// MCUDriver.cpp — 三小层组合壳实现（契约见 MCUDriver.h；协议 260831 口径）
// ============================================================================

#include "MCUDriver.h"
#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include <chrono>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace Scanner::device {

namespace {

Scanner::TimestampMs systemNowMs() {   // 墙钟：上行帧时间戳/心跳（对齐 base TimestampMs 口径）
    return static_cast<Scanner::TimestampMs>(std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count());
}

} // namespace

namespace {
template <typename T, size_t N>
void drainRing(serial::SpscRing<T, N>& ring) { T v; while (ring.pop(v)) {} }
} // namespace

MCUDriver::MCUDriver(WriteOverride writeOverride)
    : writeOverride_(std::move(writeOverride)), channel_(makeDeps()) {}

MCUDriver::~MCUDriver() { close(); }

// ============================================================================
// 三小层装配
// ============================================================================
serial::CommandChannel::Deps MCUDriver::makeDeps() {
    serial::CommandChannel::Deps d;
    d.codec = &codec_;
    d.write = [this](const std::string& f) { return writeFrame(f); };
    return d;
}

bool MCUDriver::writeFrame(const std::string& frame) {
    JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] writeFrame: '{}' open={} override={}",
                 frame, open_.load(), writeOverride_ != nullptr);
    if (writeOverride_) return writeOverride_(frame);           // 测试模式
    if (!open_.load(std::memory_order_acquire)) return false;
    enqueueWrite(frame);                                        // 写线程异步送串口
    return true;                                                // 盲发即返回；写失败由写线程记
}

void MCUDriver::setWireTap(std::function<void(bool, const std::string&)> tap) {
    std::lock_guard<std::mutex> lock(tapMtx_);
    wireTap_ = std::move(tap);
}

void MCUDriver::notifyTap(bool tx, const std::string& data) {
    // 上下位机全帧留痕（用户口径 2026-09-05：联调期默认可见——TX/RX 全经此单点；
    // 协议稳定后可降 JMW_LOG_DEBUG 减噪）
    JMW_LOG_INFO("08-MCUDriver", "[{}] {}", tx ? "TX→MCU" : "RX←MCU", data);
    std::lock_guard<std::mutex> lock(tapMtx_);
    if (wireTap_) wireTap_(tx, data);
}

// —— 写线程：唯一串口写者（R2-A1 属主天然落此线程）——
void MCUDriver::enqueueWrite(const std::string& frame) {
    {
        std::lock_guard<std::mutex> lock(writeMtx_);
        writeQueue_.push_back(WriteItem{frame});
    }
    writeCv_.notify_one();
}

void MCUDriver::writeLoop() {
    // USB 保活：工厂软件对照（不发保活也顺畅——根因是 RX 占驱动而非 TX 空闲）。
    // 现改为 WaitCommEvent 读（空闲不占驱动），保活从 80ms 放宽到 500ms——
    // 仅防御 TX 真空闲挂起（低频不再加剧驱动负担）
    constexpr auto kKeepaliveInterval = std::chrono::milliseconds(500);
    while (writeRunning_.load(std::memory_order_acquire)) {
        std::string frame;
        {
            std::unique_lock<std::mutex> lock(writeMtx_);
            writeCv_.wait_for(lock, kKeepaliveInterval, [this] {
                return !writeQueue_.empty() || !writeRunning_.load(std::memory_order_acquire);
            });
            if (!writeRunning_.load(std::memory_order_acquire) && writeQueue_.empty()) return;
            if (!writeQueue_.empty()) {
                frame = std::move(writeQueue_.front().frame);
                writeQueue_.pop_front();
                JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] writeLoop 出队: '{}'", frame);
            }
        }
        if (frame.empty()) continue;               // 保活已删（卡写队列致灭灯命令出不去）
        notifyTap(true, frame);                    // 调试监视：TX 实际出队帧
        const auto r = serial_.write(frame);    // 阻塞只落在本线程（实测驱动可卡 2~2.5s）
        if (!r.success)
            JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 串口写失败: {}（帧 '{}'）", r.message, frame);
        {
            std::lock_guard<std::mutex> lock(writeMtx_);
            if (writeQueue_.empty()) drainCv_.notify_all();     // 排空通知（收口等待用）
        }
    }
}

void MCUDriver::waitWriteDrained(int timeoutMs) {
    std::unique_lock<std::mutex> lock(writeMtx_);
    drainCv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                      [this] { return writeQueue_.empty(); });
}

void MCUDriver::resetSerialWriteOwner() { serial_.resetWriteOwner(); }

// ============================================================================
// open/close（close 倒序：停 rx 线程 → 关串口）
// ============================================================================
Scanner::Result MCUDriver::open(const std::string& port) {
    return open(port, kDefaultBaud);
}

Scanner::Result MCUDriver::open(const std::string& port, int baud) {
    if (open_.load()) return Scanner::Result::ok("MCU已打开");
    codec_ = serial::FrameCodec{};                  // 复位半帧挂起缓冲（重开语义）
    channel_ = serial::CommandChannel(makeDeps());  // 重建依赖（重开会话干净起点）
    // reopen 复位：排空残留上行环 + 清观测计数/心跳——上一会话数据不串染
    drainRing(gestureRing_);
    drainRing(tempRing_);
    drainRing(shotRing_);
    lastShot_.store(0, std::memory_order_release);
    gSeen_.store(false, std::memory_order_release);
    lastRx_.store(0, std::memory_order_release);
    if (writeOverride_) {                 // 测试模式：不开真串口、不起 rx 线程
        open_.store(true);
        return Scanner::Result::ok();
    }
    std::string target = port;
    // 写线程先起（探测帧也走队列——WriteFile 实测可被 USB 驱动卡 2~2.5s，直写会
    // 把 open/逻辑线程一起堵死；写线程=唯一写者，R2-A1 属主天然落此）
    writeRunning_.store(true);
    writeThread_ = std::thread(&MCUDriver::writeLoop, this);
    if (target.empty() || target == "auto" || target == "AUTO") {
        target = probeAutoPort(baud);     // 自动搜口：逐口探测命中（含开串口+起 rx 线程）
        if (target.empty()) {
            stopWriteThread();
            return Scanner::Result::fail(-1, "MCU 自动搜口失败（无口应答 N12 T100 探测）");
        }
    } else {
        auto r = serial_.open(target, baud);
        if (!r.success) {
            stopWriteThread();
            return r;
        }
        open_.store(true);
        rxRunning_.store(true);
        rxThread_ = std::thread(&MCUDriver::rxLoop, this);
    }
    // 打开后不发任何额外命令（原 N12Z1 自检握手已废——N12 温度周期归 DeviceManager
    // 批1-C 调 setTempReportPeriod）
    JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] 串口已打开: {} @ {} baud", target, baud);
    return Scanner::Result::ok();
}

void MCUDriver::stopWriteThread() {
    if (writeThread_.joinable()) {
        waitWriteDrained(3000);           // 队列排空（灯熄帧收口；慢写上限兜底 3s）
        writeRunning_.store(false, std::memory_order_release);
        writeCv_.notify_all();
        writeThread_.join();
    }
    {
        std::lock_guard<std::mutex> lock(writeMtx_);
        writeQueue_.clear();
    }
}

// 自动搜口（open 前置，调用线程编排）：逐口 开→起 rx→发 N12 T100 探测→300ms 内
// 收到任意完整帧（G02 温度等）即认定 MCU。纯回环线不会误命中：TX 帧走写线程出
// 串口，不经 rx 线程回流（与旧回显探测同理）。探测帧经写队列由写线程出线——写
// 线程在 open 已起、贯穿整个搜口：未命中口只做 closePortOnly（停 rx＋关串口＋
// open_ 复位，写线程保活），此处若走整 close() 会 join 写线程，第 2 口起探测帧
// 入队无消费者永不出线（P1）。全败的写线程收尾由 open() 的 target.empty() 分支
// 统一做。命中口的 N12 T100 已生效（100ms 温度周期），open 流程不重发。
std::string MCUDriver::probeAutoPort(int baud) {
    const auto ports = serial::SerialPort::listPorts();
    if (ports.empty()) JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 自动搜口：本机未枚举到任何 COM 口");
    for (const auto& port : ports) {
        if (!serial_.open(port, baud).success) continue;
        open_.store(true);
        rxRunning_.store(true);
        rxThread_ = std::thread(&MCUDriver::rxLoop, this);
        lastRx_.store(0, std::memory_order_release);
        gSeen_.store(false, std::memory_order_release);
        channel_.sendFireAndForget("N12 T100");
        for (int waited = 0; waited < 300 && !gSeen_.load(std::memory_order_acquire); waited += 10)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (gSeen_.load(std::memory_order_acquire)) {
            JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] 自动搜口命中: {}（{} 口中，N12 T100 应答）",
                         port, ports.size());
            return port;
        }
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] 自动搜口: {} 无应答帧，试下一口", port);
        closePortOnly();                  // 停 rx＋关串口＋open_ 复位——写线程保活（下一口探测帧仍需它出线）
    }
    return {};
}

Scanner::Result MCUDriver::close() {
    if (rxThread_.joinable()) {
        rxRunning_.store(false);
        rxThread_.join();
    }
    stopWriteThread();                    // 写线程先收（队列排空后）——再关串口防截断
    serial_.close();                      // rx 靠 rxRunning_ 退出：read 受 COMMTIMEOUTS 50ms 解堵，join 有界
    if (open_.exchange(false)) JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] 串口已关闭");
    return Scanner::Result::ok();
}

// 探测逐口收尾（区别于 close：不动写线程——open 起的写线程贯穿整个搜口，命中口
// 直接续用、全败由 open() 的 target.empty() 分支收口）。写队列不在此清：探测期
// 队内只可能是同款探测帧，残帧落下一口为无害重复（N12 T100 幂等）。
void MCUDriver::closePortOnly() {
    if (rxThread_.joinable()) {
        rxRunning_.store(false);
        rxThread_.join();
    }
    serial_.close();
    open_.store(false);
}

bool MCUDriver::isOpen() const { return open_.load(); }

// ============================================================================
// rx 线程：零业务——read → feed → 分发；半帧超时推进（与 feed 同线程）
// ============================================================================
void MCUDriver::rxLoop() {
    ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);  // C7 优先级提升
    char buf[256];
    std::vector<serial::FrameCodec::Frame> frames;
    auto lastTick = std::chrono::steady_clock::now();
    while (rxRunning_.load()) {
        const int n = serial_.read(buf, static_cast<int>(sizeof buf));
        if (n > 0) {
            codec_.feed(std::string(buf, buf + n), frames);
            for (const auto& f : frames) dispatchFrame(f);
            frames.clear();
        } else if (n < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));  // 未开/错误防忙转
        }
        const auto now = std::chrono::steady_clock::now();
        const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTick);
        if (el.count() > 0) {
            codec_.advanceTimeout(el.count());
            lastTick = now;
        }
    }
}

void MCUDriver::dispatchFrame(const serial::FrameCodec::Frame& f) {
    const Scanner::TimestampMs now = systemNowMs();
    lastRx_.store(now, std::memory_order_release);   // 任何完整帧刷新心跳（§4-4）
    gSeen_.store(true, std::memory_order_release);   // 探测凭据：任意完整帧到达
    notifyTap(false, f.payload);                     // 调试监视：RX 完整帧载荷
    const std::string& p = f.payload;
    if (p.compare(0, 3, "G01") == 0) {
        serial::GestureEvent g;
        if (serial::parseGesturePayload(p, g)) {
            g.ts = lastRx_.load(std::memory_order_acquire);
            if (!gestureRing_.push(g))
                JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 手势环满丢新(计{})", gestureRing_.dropped());
        } else onParseFail(p);
        return;
    }
    if (p.compare(0, 3, "G02") == 0) {
        serial::TempFrame t;
        if (serial::parseTempPayload(p, t)) {
            t.ts = lastRx_.load(std::memory_order_acquire);
            if (!tempRing_.push(t))
                JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 温度环满丢新(计{})", tempRing_.dropped());
        } else onParseFail(p);
        return;
    }
    if (p.compare(0, 3, "G03") == 0) {
        serial::ShotCountFrame s;
        if (serial::parseShotCountPayload(p, s)) {
            s.ts = lastRx_.load(std::memory_order_acquire);
            lastShot_.store(s.count, std::memory_order_release);
            (void)shotRing_.push(s);             // 溢出不警（计数已有 lastShot_ 兜底）
        } else onParseFail(p);
        return;
    }
    onParseFail(p);                              // 未知帧（旧 T/K/S/A/E 全落此）
}

void MCUDriver::onParseFail(const std::string& payload) {
    const uint64_t n = parseFailCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 上行载荷丢弃(计{}): '{}'", n, payload);
}

// ============================================================================
// pump（逻辑线程）：排空 3 环 → uplink 分流（手势优先、计数次之、温度最后）
// ============================================================================
void MCUDriver::pump() {
    serial::GestureEvent g;
    while (gestureRing_.pop(g)) { if (uplink_.onGesture) uplink_.onGesture(g); }
    serial::ShotCountFrame s;
    while (shotRing_.pop(s))     { if (uplink_.onShotCount) uplink_.onShotCount(s); }
    serial::TempFrame t;
    while (tempRing_.pop(t))     { if (uplink_.onTemp) uplink_.onTemp(t); }
}

// ============================================================================
// typed 下行 4 条（协议 260831 §下行表；payload 拼装 → CommandChannel）
// ⚠ 参数与参数间空格（"N10 H30 B60 ..."）——无空格固件不解析
// ============================================================================
void MCUDriver::setCaptureParams(const hal::CaptureParams& p, DoneCb cb) {
    channel_.send("N10 H" + std::to_string(p.freqHz) +
                  " B" + std::to_string(p.bgLight) +
                  " T" + std::to_string(p.laserT) +
                  " V" + std::to_string(p.laserV) +
                  " C" + std::to_string(p.laserC) +
                  " D" + std::to_string(p.laserD) +
                  " L" + std::to_string(p.laserLevel), std::move(cb));
}
void MCUDriver::stopScan(DoneCb cb) { channel_.send("N11 H0", std::move(cb)); }

// 熄灯＝N11 H0（停止采集）——新协议停止与熄灯同一命令
void MCUDriver::lightsOff() {
    if (!open_.load() || writeOverride_) return;
    stopScan(nullptr);
}

void MCUDriver::flushWrites(int timeoutMs) {
    if (writeOverride_) return;                  // 测试模式无队列
    waitWriteDrained(timeoutMs);
}
void MCUDriver::setTempReportPeriod(int ms, DoneCb cb) {
    if (ms < 5) ms = 5;                          // D9：串口预算安全下限
    if (ms > 1000) ms = 1000;
    channel_.send("N12 T" + std::to_string(ms), std::move(cb));
}
void MCUDriver::setHeatTarget(int celsius, DoneCb cb) {
    if (celsius < 0) celsius = 0;                // 协议域 0-80 钳制
    if (celsius > 80) celsius = 80;
    channel_.send("N13 S" + std::to_string(celsius), std::move(cb));
}

// ============================================================================
// 上行/观测
// ============================================================================
void MCUDriver::setUplink(hal::McuUplink h) { uplink_ = std::move(h); }

Scanner::TimestampMs MCUDriver::lastRxTime() const {
    return lastRx_.load(std::memory_order_acquire);
}

uint64_t MCUDriver::gestureRingDropped() const {     // G01 手势环满丢新计数出口
    return gestureRing_.dropped();
}

uint64_t MCUDriver::parseFailCount() const {
    return parseFailCount_.load(std::memory_order_relaxed);
}

uint64_t MCUDriver::shotCount() const {              // 最近 G03 计数快照（D6 记账口径）
    return lastShot_.load(std::memory_order_acquire);
}

void MCUDriver::testInjectRaw(const std::string& frameBytes) {
    std::vector<serial::FrameCodec::Frame> out;
    codec_.feed(frameBytes, out);
    for (const auto& f : out) dispatchFrame(f);
}

} // namespace Scanner::device
