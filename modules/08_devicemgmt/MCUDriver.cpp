// ============================================================================
// MCUDriver.cpp — 三小层组合壳实现（契约见 MCUDriver.h / 设计方案 §2.2-§2.5）
// ============================================================================

#include "MCUDriver.h"
#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include <chrono>
#include <cstdlib>

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
    JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] writeFrame: '{}' open={} override={}",
                 frame, open_.load(), writeOverride_ != nullptr);
    if (writeOverride_) return writeOverride_(frame);           // 测试模式
    if (!open_.load(std::memory_order_acquire)) return false;
    enqueueWrite(frame);                                        // 写线程异步送串口
    return true;                                                // v2 发即返回；写失败由写线程记
}

void MCUDriver::setWireTap(std::function<void(bool, const std::string&)> tap) {
    std::lock_guard<std::mutex> lock(tapMtx_);
    wireTap_ = std::move(tap);
}

void MCUDriver::notifyTap(bool tx, const std::string& data) {
    // 上下位机全帧留痕（用户口径 2026-09-05：联调期默认可见——TX 三处/RX 三处
    // 全经此单点；协议稳定后可降 JMW_LOG_DEBUG 减噪）
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
    auto lastWrite = std::chrono::steady_clock::now();
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
        lastWrite = std::chrono::steady_clock::now();
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

// 回环验证（启动自检用，逻辑线程调）：发 payload→轮询回显（rx 线程记档 lastEcho_）
bool MCUDriver::sendEchoProbe(const std::string& payload, int timeoutMs) {
    if (!open_.load(std::memory_order_acquire) || writeOverride_) return false;
    {
        std::lock_guard<std::mutex> lock(echoMtx_);
        lastEcho_.clear();
    }
    channel_.sendFireAndForget(payload);
    for (int waited = 0; waited < timeoutMs; waited += 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(echoMtx_);
        if (lastEcho_ == payload) return true;
    }
    return false;
}

std::string MCUDriver::lastEchoPayload() const {
    std::lock_guard<std::mutex> lock(echoMtx_);
    return lastEcho_;
}

bool MCUDriver::probeDidSelfCheck() const { return probeN12Sent_.load(std::memory_order_acquire); }

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
    probeN12Sent_.store(false, std::memory_order_release);
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
            return Scanner::Result::fail(-1, "MCU 自动搜口失败（无口应答 N12Z1 探测）");
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
    JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] 串口已打开: {} @ {} baud (v{})", target, baud,
                 version_ == serial::FrameCodec::Version::V3 ? 3 : 2);
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

// 自动搜口（open 前置，调用线程执行）：逐口 开→发 N12Z1 探测→300ms 内收到
// 数值行（温度上报）即认定 MCU（v2 固件口径：回显是「载荷;」帧不算凭据——纯
// 回环线只会回显，不误命中）。半途收 rx 线程+关串口再试下一口。探测写在调用
// 线程——测毕 resetWriteOwner 归还写权给逻辑线程（R2-A1 单写者纪律）。
// 注：探测的 N12Z1 与 open 后 DeviceManager 的 enterSelfCheck 重复——Z1 幂等，无害。
std::string MCUDriver::probeAutoPort(int baud) {
    const auto ports = serial::SerialPort::listPorts();
    if (ports.empty()) JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 自动搜口：本机未枚举到任何 COM 口");
    for (const auto& port : ports) {
        if (!serial_.open(port, baud).success) continue;
        open_.store(true);
        rxRunning_.store(true);
        rxThread_ = std::thread(&MCUDriver::rxLoop, this);
        lastRx_.store(0, std::memory_order_release);
        probeHit_.store(false, std::memory_order_release);
        probeN12Sent_.store(false, std::memory_order_release);  // 不再发 N12Z1（自检模式已废弃）
        // 探测凭据=点灯参数帧（2026-08-30 合一定版）：探测与自检点灯共用一帧
        // N10 H50 B50 T1 V2 L50——回显即命中（v2 固件整帧回显），灯随之点亮；
        // 免去"探测(B0/L0)+点灯"两帧连发。回显=固件在线
        if (sendEchoProbe("N10 H50 B50 T1 V2 L50", 3000)) {
            JMW_LOG_INFO("08-MCUDriver", "[MCUDriver] 自动搜口命中: {}（{} 口中，N10 探测+点灯合一）", port, ports.size());
            return port;
        }
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] 自动搜口: {} 无回显，试下一口", port);
        close();                           // 停 rx 线程+关串口+open_ 复位（幂等）
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
            if (version_ == serial::FrameCodec::Version::V2) {
                // v2 调试口径（实测固件）：下行回显=「载荷;」帧；上行=裸文本行（\r\n 切行，
                // 数值行≈7Hz 温度上报）。行层先切：';' 段喂 codec 走帧路径，其余按文本行
                // 处理——防数值行字节污染 codec pending_ 缓冲串坏后续帧。
                for (int i = 0; i < n; ++i) {
                    const char ch = buf[i];
                    if (ch == ';') {
                        lineBuf_.push_back(';');
                        codec_.feed(lineBuf_, frames);
                        lineBuf_.clear();
                    } else if (ch == '\n') {
                        // 空行也喂：保活 "\r\n" 的回显=双向链路活证据（刷 lastRx_——
                        // 固件停发原生温度行后 0x0802 判据的唯一持续来源）
                        feedTextLine(lineBuf_);
                        lineBuf_.clear();
                    } else if (ch != '\r') {      // '\r' 丢弃
                        lineBuf_.push_back(ch);
                        if (lineBuf_.size() > 96) lineBuf_.clear();   // 无界行守卫
                    }
                }
            } else {
                codec_.feed(std::string(buf, buf + n), frames);
            }
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
    if (!f.payload.empty()) notifyTap(false, f.payload + ";");   // 调试监视：RX 完整帧（补分号还原协议原文）
    if (f.payload.empty()) { onParseFail(f.payload); return; }
    switch (f.payload[0]) {
    case 'T': {
        serial::TempFrame t;
        if (serial::parseTempPayload(f.payload, t)) {
            t.seq = f.seq; t.ts = now;
            if (!tempRing_.push(t)) JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] 遥测环满丢新(计{})", tempRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'K': {
        serial::RawKeyEvent k;
        if (serial::parseKeyPayload(f.payload, k)) {
            k.seq = f.seq; k.ts = now;
            if (!keyRing_.push(k)) JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 事件环满丢新(K,计{})", keyRing_.dropped());
        }
        else onParseFail(f.payload);   // v2 匿名 K1;/K0; 落此（§2.5 按键链停用）
        break;
    }
    case 'S': {
        serial::StatusFrame s;
        if (serial::parseStatusPayload(f.payload, s)) {
            s.seq = f.seq; s.ts = now;
            if (!statusRing_.push(s)) JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] 事件环满丢新(S,计{})", statusRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'A': {
        serial::AckFrame a;
        if (serial::parseAckPayload(f.payload, a)) {
            a.seq = f.seq;
            if (!ackRing_.push(a)) JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 事件环满丢新(A,计{})", ackRing_.dropped());
        }
        else onParseFail(f.payload);
        break;
    }
    case 'E':
        onParseFail(f.payload);   // v2 旧 E1; 急停牌已删——忽略+warn（§2.2 列净）
        break;
    default:
        // v2 固件对下行命令整帧回显（实测）——'N' 打头载荷即回显，记档供 sendEchoProbe
        if (!f.payload.empty() && f.payload[0] == 'N') {
            std::lock_guard<std::mutex> lock(echoMtx_);
            lastEcho_ = f.payload;
        }
        onParseFail(f.payload);
        break;
    }
}

void MCUDriver::onParseFail(const std::string& payload) {
    const uint64_t n = parseFailCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    // 分级：v2 下 MCU 回显命令帧（'N' 开头）是链路预期行为——非垃圾；降 debug
    // 防刷屏（真异常的 T/K/S/A 解析失败仍 warn）。回显帧已刷 lastRx（心跳口径）
    if (!payload.empty() && payload[0] == 'N') {
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] v2 命令回显(计{}): '{}'", n, payload);
        return;
    }
    JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] 上行载荷丢弃(计{}): '{}'", n, payload);
}

// v2 固件裸文本行（rx 线程）：数值行=温度上报（实测 ~133，单位待定——非合理 ℃，
// 暂不入 TempFrame：防 tempMaxC=60 的 0x0803 量纲误判 + Warmup 吃脏数据；单位
// 与固件方确认后再接）；其余为状态文本（如 "Self check begins"）。任何行=链路
// 活：刷 lastRx_（0x0802 心跳口径）。数值行兼作自动搜口命中凭据（回显/纯回环
// 线不产数值行，不会误命中）。
void MCUDriver::feedTextLine(const std::string& line) {
    notifyTap(false, line);                       // 调试监视：RX 裸文本行
    lastRx_.store(systemNowMs(), std::memory_order_release);
    char* endp = nullptr;
    const double v = std::strtod(line.c_str(), &endp);
    if (endp != line.c_str() && *endp == '\0') {
        probeHit_.store(true, std::memory_order_release);
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] v2 数值行: {}", v);
    } else {
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] v2 文本行: '{}'", line);
    }
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
        JMW_LOG_DEBUG("08-MCUDriver", "[MCUDriver] T 帧 seq 跳变 {}→{}（丢帧对账计 {}）",
                      lastTSeq_, seq, seqGapCount_.load());
    }
    hasTSeq_ = true;
    lastTSeq_ = seq;
}

// ============================================================================
// typed N10–N16（协议表 §2.2；payload 拼装 → CommandChannel）
// ⚠ 参数前必须有空格（对齐工厂 "N10 H60 B80 T1 V1 L120;"）——无空格固件不解析
// ============================================================================
void MCUDriver::setCaptureParams(const hal::CaptureParams& p, DoneCb cb) {
    channel_.send("N10 H" + std::to_string(p.freqHz) +
                  " B" + std::to_string(p.bgLight) +
                  " T" + std::to_string(p.laserSelectA) +
                  " V" + std::to_string(p.laserSelectB) +
                  " L" + std::to_string(p.laserLevel), std::move(cb));
}
void MCUDriver::startScan(DoneCb cb)        { channel_.send("N11 H1", std::move(cb)); }
void MCUDriver::stopScan(DoneCb cb)         { channel_.send("N11 H0", std::move(cb)); }

// 熄灯＝N11 H0（停止扫描）——工厂软件同款（实测 N10 B0/L0 不关灯，TX 日志确认）
void MCUDriver::lightsOff() {
    if (!open_.load() || writeOverride_) return;
    stopScan(nullptr);
}

void MCUDriver::flushWrites(int timeoutMs) {
    if (writeOverride_) return;                  // 测试模式无队列
    waitWriteDrained(timeoutMs);
}
void MCUDriver::enterSelfCheck(DoneCb cb)   { channel_.send("N12 Z1", std::move(cb)); }
void MCUDriver::exitSelfCheck(DoneCb cb)    { channel_.send("N12 Z0", std::move(cb)); }
void MCUDriver::enterStandby(DoneCb cb)     { channel_.send("N13 E1", std::move(cb)); }
void MCUDriver::exitStandby(DoneCb cb)      { channel_.send("N13 E0", std::move(cb)); }
void MCUDriver::setHeatTarget(int celsius, DoneCb cb) {
    channel_.send("N14 T" + std::to_string(celsius), std::move(cb));
}
void MCUDriver::queryTemperature(int v0to2) {
    channel_.sendFireAndForget("N15 V" + std::to_string(v0to2));
}
void MCUDriver::enterCalibration(DoneCb cb) { channel_.send("N16 B1", std::move(cb)); }
void MCUDriver::exitCalibration(DoneCb cb)  { channel_.send("N16 B0", std::move(cb)); }

// ============================================================================
// 上行/配置/观测
// ============================================================================
void MCUDriver::setUplink(hal::McuUplink h) { uplink_ = std::move(h); }

void MCUDriver::setProtocolVersion(serial::FrameCodec::Version v) {
    if (open_.load()) {
        JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] setProtocolVersion 开启中调用无效（重开串口才生效）");
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
    else JMW_LOG_WARN("08-MCUDriver", "[MCUDriver] setAckTimeoutMs 开启中调用不重建（重开才生效）");
}

void MCUDriver::testInjectRaw(const std::string& frameBytes) {
    std::vector<serial::FrameCodec::Frame> out;
    codec_.feed(frameBytes, out);
    for (const auto& f : out) dispatchFrame(f);
}

} // namespace Scanner::device
