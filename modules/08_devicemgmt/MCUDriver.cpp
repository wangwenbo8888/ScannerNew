// ============================================================================
// MCUDriver.cpp — 三小层组合壳实现（契约见 MCUDriver.h / 设计方案 §2.2-§2.5）
// ============================================================================

#include "MCUDriver.h"
#include <spdlog/spdlog.h>
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

int64_t steadyNowMs() {   // 单调时钟：CommandChannel 对账/半帧超时用
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    d.nowMs = [] { return steadyNowMs(); };
    d.reliable = (version_ == serial::FrameCodec::Version::V3);   // v2 降级：发不等（§2.5）
    d.ackTimeoutMs = ackTimeoutMs_;                               // D-T12b：可注入（默认 100）
    return d;
}

void MCUDriver::applyVersion() {
    codec_ = serial::FrameCodec(version_);            // 复位半帧挂起缓冲（重开语义）
    channel_ = serial::CommandChannel(makeDeps());    // 重建依赖（reliable 随版本）
}

bool MCUDriver::writeFrame(const std::string& frame) {
    if (writeOverride_) return writeOverride_(frame);           // 测试模式
    if (!open_.load(std::memory_order_acquire)) return false;
    return serial_.write(frame).success;
}

// ============================================================================
// open/close（close 倒序：停 rx 线程 → 关串口）
// ============================================================================
Scanner::Result MCUDriver::open(const std::string& port) {
    return open(port, kDefaultBaud);
}

Scanner::Result MCUDriver::open(const std::string& port, int baud) {
    if (open_.load()) return Scanner::Result::ok("MCU已打开");
    applyVersion();
    // reopen 复位：排空残留上行环 + 清 seq 对账基线/心跳——上一会话数据不串染
    drainRing(keyRing_);
    drainRing(statusRing_);
    drainRing(ackRing_);
    drainRing(tempRing_);
    hasTSeq_ = false;
    lastTSeq_ = 0;
    lastRx_.store(0, std::memory_order_release);
    if (writeOverride_) {                 // 测试模式：不开真串口、不起 rx 线程
        open_.store(true);
        return Scanner::Result::ok();
    }
    auto r = serial_.open(port, baud);
    if (!r.success) return r;
    open_.store(true);
    rxRunning_.store(true);
    rxThread_ = std::thread(&MCUDriver::rxLoop, this);
    spdlog::info("[MCUDriver] 串口已打开: {} @ {} baud (v{})", port, baud,
                 version_ == serial::FrameCodec::Version::V3 ? 3 : 2);
    return Scanner::Result::ok();
}

Scanner::Result MCUDriver::close() {
    if (rxThread_.joinable()) {
        rxRunning_.store(false);
        rxThread_.join();
    }
    serial_.close();                      // rx 靠 rxRunning_ 退出：read 受 COMMTIMEOUTS 50ms 解堵，join 有界
    if (open_.exchange(false)) spdlog::info("[MCUDriver] 串口已关闭");
    return Scanner::Result::ok();
}

bool MCUDriver::isOpen() const { return open_.load(); }

// ============================================================================
// rx 线程：零业务——read → feed → 入环；半帧超时推进（与 feed 同线程）
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
    lastRx_.store(now, std::memory_order_release);   // 任何有效帧（分帧+CRC 过）刷新心跳（§4-4）
    if (f.payload.empty()) { onParseFail(f.payload); return; }
    switch (f.payload[0]) {
    case 'T': {
        serial::TempFrame t;
        if (serial::parseTempPayload(f.payload, t)) {
            t.seq = f.seq; t.ts = now;
            if (!tempRing_.push(t)) spdlog::debug("[MCUDriver] 遥测环满丢新(计{})", tempRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'K': {
        serial::RawKeyEvent k;
        if (serial::parseKeyPayload(f.payload, k)) {
            k.seq = f.seq; k.ts = now;
            if (!keyRing_.push(k)) spdlog::warn("[MCUDriver] 事件环满丢新(K,计{})", keyRing_.dropped());
        }
        else onParseFail(f.payload);   // v2 匿名 K1;/K0; 落此（§2.5 按键链停用）
        break;
    }
    case 'S': {
        serial::StatusFrame s;
        if (serial::parseStatusPayload(f.payload, s)) {
            s.seq = f.seq; s.ts = now;
            if (!statusRing_.push(s)) spdlog::debug("[MCUDriver] 事件环满丢新(S,计{})", statusRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'A': {
        serial::AckFrame a;
        if (serial::parseAckPayload(f.payload, a)) {
            a.seq = f.seq;
            if (!ackRing_.push(a)) spdlog::warn("[MCUDriver] 事件环满丢新(A,计{})", ackRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'E':
        onParseFail(f.payload);   // v2 旧 E1; 急停牌已删——忽略+warn（§2.2 删净）
        break;
    default:
        onParseFail(f.payload);
        break;
    }
}

void MCUDriver::onParseFail(const std::string& payload) {
    const uint64_t n = parseFailCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    spdlog::warn("[MCUDriver] 上行载荷丢弃(计{}): '{}'", n, payload);
}

// ============================================================================
// pump（逻辑线程）：排空 4 环 → Uplink 分流 / onAck 回填 / T seq 对账
// ============================================================================
void MCUDriver::pump() {
    serial::AckFrame a;
    while (ackRing_.pop(a)) channel_.onAck(a.ackedSeq);        // A 先行：ACK 最速销项
    serial::RawKeyEvent k;
    while (keyRing_.pop(k)) { if (uplink_.onKey) uplink_.onKey(k); }
    serial::StatusFrame s;
    while (statusRing_.pop(s)) { if (uplink_.onStatus) uplink_.onStatus(s); }
    serial::TempFrame t;
    while (tempRing_.pop(t)) {
        accountTempSeq(t.seq);
        if (uplink_.onTemp) uplink_.onTemp(t);
    }
}

void MCUDriver::accountTempSeq(uint16_t seq) {
    if (version_ != serial::FrameCodec::Version::V3) return;   // v2 seq 恒 0——不对账
    if (hasTSeq_ && static_cast<uint8_t>(lastTSeq_ + 1) != static_cast<uint8_t>(seq)) {
        seqGapCount_.fetch_add(1, std::memory_order_relaxed);
        spdlog::debug("[MCUDriver] T 帧 seq 跳变 {}→{}（丢帧对账计 {}）",
                      lastTSeq_, seq, seqGapCount_.load());
    }
    hasTSeq_ = true;
    lastTSeq_ = seq;
}

// ============================================================================
// typed N10–N16（协议表 §2.2；payload 拼装 → CommandChannel）
// ============================================================================
void MCUDriver::setCaptureParams(const hal::CaptureParams& p, DoneCb cb) {
    channel_.send("N10H" + std::to_string(p.freqHz) +
                  "B" + std::to_string(p.bgLight) +
                  "T" + std::to_string(p.laserSelectA) +
                  "V" + std::to_string(p.laserSelectB) +
                  "L" + std::to_string(p.laserLevel), std::move(cb));
}
void MCUDriver::startScan(DoneCb cb)        { channel_.send("N11H1", std::move(cb)); }
void MCUDriver::stopScan(DoneCb cb)         { channel_.send("N11H0", std::move(cb)); }
void MCUDriver::enterSelfCheck(DoneCb cb)   { channel_.send("N12Z1", std::move(cb)); }
void MCUDriver::exitSelfCheck(DoneCb cb)    { channel_.send("N12Z0", std::move(cb)); }
void MCUDriver::enterStandby(DoneCb cb)     { channel_.send("N13E1", std::move(cb)); }
void MCUDriver::exitStandby(DoneCb cb)      { channel_.send("N13E0", std::move(cb)); }
void MCUDriver::setHeatTarget(int celsius, DoneCb cb) {
    channel_.send("N14T" + std::to_string(celsius), std::move(cb));
}
void MCUDriver::queryTemperature(int v0to2) {
    channel_.sendFireAndForget("N15V" + std::to_string(v0to2));   // 查询类：发不等（§2.3）
}
void MCUDriver::enterCalibration(DoneCb cb) { channel_.send("N16B1", std::move(cb)); }
void MCUDriver::exitCalibration(DoneCb cb)  { channel_.send("N16B0", std::move(cb)); }

// ============================================================================
// 上行/配置/观测
// ============================================================================
void MCUDriver::setUplink(hal::McuUplink h) { uplink_ = std::move(h); }

void MCUDriver::setProtocolVersion(serial::FrameCodec::Version v) {
    if (open_.load()) {
        spdlog::warn("[MCUDriver] setProtocolVersion 开启中调用无效（重开串口才生效）");
        return;
    }
    version_ = v;
    applyVersion();
}

Scanner::TimestampMs MCUDriver::lastRxTime() const {
    return lastRx_.load(std::memory_order_acquire);
}

uint64_t MCUDriver::seqGapCount() const { return seqGapCount_.load(std::memory_order_relaxed); }

void MCUDriver::channelTick() { channel_.tick(); }   // D-T12b：对账出口（逻辑线程驱动）

uint64_t MCUDriver::keyDropCount() const {           // D-T13：K 环丢新计数出口
    return keyRing_.dropped();
}

void MCUDriver::setAckTimeoutMs(int ms) {            // D-T12b：open 前注入生效
    if (ms < 1) ms = 1;                              // 防 tick 活锁（与 Deps 构造防护同口径）
    ackTimeoutMs_ = ms;
    if (!open_.load()) applyVersion();               // 重建 channel 依赖
    else spdlog::warn("[MCUDriver] setAckTimeoutMs 开启中调用不重建（重开才生效）");
}

void MCUDriver::testInjectRaw(const std::string& frameBytes) {
    std::vector<serial::FrameCodec::Frame> out;
    codec_.feed(frameBytes, out);
    for (const auto& f : out) dispatchFrame(f);
}

} // namespace Scanner::device
