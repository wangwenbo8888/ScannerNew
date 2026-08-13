// ============================================================================
// MCUDriver.cpp — 下位机 MCU 串口驱动实现（Windows）
// ============================================================================

#include "MCUDriver.h"
#include <spdlog/spdlog.h>
#include <cstring>
#include <chrono>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace Scanner::device {

MCUDriver::MCUDriver(int baudRate) : baudRate_(baudRate) {}

MCUDriver::~MCUDriver() {
    close();
}

// ============================================================================
// 打开/关闭
// ============================================================================
Result MCUDriver::open(const std::string& portOrDevice) {
    if (isOpen_) return Result::ok("MCU已打开");

    std::string dev = portOrDevice;
    if (dev.substr(0, 3) != "\\\\.") dev = "\\\\.\\" + dev;

    hSerial_ = CreateFileA(dev.c_str(), GENERIC_READ | GENERIC_WRITE,
                           0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hSerial_ == INVALID_HANDLE_VALUE) {
        return Result::fail(-1, "MCU串口打开失败: " + portOrDevice);
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(hSerial_, &dcb)) {
        CloseHandle(hSerial_); hSerial_ = nullptr;
        return Result::fail(-2, "GetCommState失败");
    }
    dcb.BaudRate = baudRate_;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    if (!SetCommState(hSerial_, &dcb)) {
        CloseHandle(hSerial_); hSerial_ = nullptr;
        return Result::fail(-3, "SetCommState失败");
    }

    COMMTIMEOUTS timeouts = {};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    SetCommTimeouts(hSerial_, &timeouts);

    isOpen_ = true;
    rxRunning_ = true;
    rxThread_ = std::thread(&MCUDriver::receiveLoop, this);

    spdlog::info("[MCUDriver] 串口已打开: {} @ {} baud", portOrDevice, baudRate_);
    return Result::ok();
}

Result MCUDriver::close() {
    if (!isOpen_) return Result::ok();
    rxRunning_ = false;
    if (rxThread_.joinable()) rxThread_.join();
    if (hSerial_) { CloseHandle(hSerial_); hSerial_ = nullptr; }
    isOpen_ = false;
    spdlog::info("[MCUDriver] 串口已关闭");
    return Result::ok();
}

bool MCUDriver::isOpen() const { return isOpen_; }

// ============================================================================
// 串口发送
// ============================================================================
void MCUDriver::sendCommand(const std::string& cmd) {
    if (!hSerial_ || !isOpen_) return;
    DWORD written = 0;
    WriteFile(hSerial_, cmd.c_str(), static_cast<DWORD>(cmd.size()), &written, nullptr);
    spdlog::debug("[MCUDriver] 发送 {} 字节: {}", written, cmd);
}

// ============================================================================
// 串口接收线程
// ============================================================================
void MCUDriver::receiveLoop() {
    char buf[256];
    std::string accum;
    while (rxRunning_ && hSerial_) {
        DWORD read = 0;
        if (ReadFile(hSerial_, buf, sizeof(buf) - 1, &read, nullptr) && read > 0) {
            accum.append(buf, read);
            size_t pos;
            while ((pos = accum.find(';')) != std::string::npos) {
                std::string msg = accum.substr(0, pos + 1);
                accum.erase(0, pos + 1);
                parseReceived(msg);
            }
        }
    }
}

void MCUDriver::parseReceived(const std::string& data) {
    spdlog::debug("[MCUDriver] 收到: {}", data);

    // 解析温度: "T25.3;"
    if (data.size() > 1 && data[0] == 'T') {
        try {
            double t = std::stod(data.substr(1));
            temperature_.store(t, std::memory_order_release);
            hal::McuEvent evt;
            evt.type = hal::McuEventType::Temperature;
            evt.param1 = static_cast<int64_t>(t * 100);
            evt.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::lock_guard lock(callbackMutex_);
            if (callback_) callback_(evt);
        } catch (...) {}
    }

    // 解析按键: "K1;" / "K0;"
    if (data.size() >= 2 && data[0] == 'K') {
        hal::McuEvent evt;
        evt.type = hal::McuEventType::KeyEvent;
        evt.param1 = (data[1] == '1') ? 1 : 0;
        std::lock_guard lock(callbackMutex_);
        if (callback_) callback_(evt);
    }

    // 急停: "E1;"
    if (data.size() >= 2 && data[0] == 'E' && data[1] == '1') {
        emergency_.store(true);
        hal::McuEvent evt;
        evt.type = hal::McuEventType::EmergencyStop;
        std::lock_guard lock(callbackMutex_);
        if (callback_) callback_(evt);
    }
}

// ============================================================================
// 触发
// ============================================================================
Result MCUDriver::sendSoftwareTrigger() {
    sendCommand("N12 T0;");
    return Result::ok();
}

Result MCUDriver::setHardwareTriggerMode(bool enabled) {
    hwTrigger_ = enabled;
    return Result::ok();
}

// ============================================================================
// 激光/光源
// ============================================================================
Result MCUDriver::setLaserOn(bool on) {
    laserOn_ = on;
    if (!on) sendCommand("N13 L0;");
    return Result::ok();
}

Result MCUDriver::setLaserPower(int level) {
    laserPower_ = level;
    sendCommand("N13 L" + std::to_string(level) + ";");
    return Result::ok();
}

Result MCUDriver::setLedOn(bool on) {
    ledOn_ = on;
    if (!on) sendCommand("N14 B0;");
    else sendCommand("N14 B" + std::to_string(fillLight_) + ";");
    return Result::ok();
}

// ============================================================================
// 急停
// ============================================================================
Result MCUDriver::emergencyStop() {
    emergency_.store(true);
    sendCommand("N15 E1;");
    spdlog::warn("[MCUDriver] 急停已发送");
    return Result::ok();
}

bool MCUDriver::isEmergencyStop() const {
    return emergency_.load();
}

// ============================================================================
// 温度
// ============================================================================
double MCUDriver::getTemperature() const {
    return temperature_.load(std::memory_order_acquire);
}

// ============================================================================
// 回调
// ============================================================================
Result MCUDriver::registerCallback(hal::McuEventCallback cb) {
    std::lock_guard lock(callbackMutex_);
    callback_ = std::move(cb);
    return Result::ok();
}

Result MCUDriver::unregisterCallback() {
    std::lock_guard lock(callbackMutex_);
    callback_ = nullptr;
    return Result::ok();
}

// ============================================================================
// 扫描控制
// ============================================================================
Result MCUDriver::startScan(int freq, int bgLight, int laserLight,
                            int trigger, int version) {
    fillLight_ = bgLight;
    laserPower_ = laserLight;
    laserOn_ = true;
    emergency_.store(false);

    char cmd[128];
    std::snprintf(cmd, sizeof(cmd),
        "N10 H%d B%d T%d V%d L%d;",
        freq, bgLight, trigger, version, laserLight);
    sendCommand(cmd);

    spdlog::info("[MCUDriver] 扫描启动: freq={} bg={} laser={}", freq, bgLight, laserLight);
    return Result::ok();
}

Result MCUDriver::stopScan() {
    laserOn_ = false;
    sendCommand("N11 H0;");
    spdlog::info("[MCUDriver] 扫描停止");
    return Result::ok();
}

} // namespace Scanner::device
