// ============================================================================
// SerialPort.cpp — 纯串口 IO 实现（Windows）
// ============================================================================

#include "SerialPort.h"
#include <spdlog/spdlog.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace Scanner::device::serial {

// 本类错误码段位：-100 ~ -108（统一负段，防与其它 HAL 错误码撞义；旧 -1~-5 已退役）
namespace {
constexpr int32_t kErrAlreadyOpen   = -100;  // open：句柄已存在
constexpr int32_t kErrCreateFile    = -101;  // open：CreateFile 失败
constexpr int32_t kErrGetCommState  = -102;
constexpr int32_t kErrSetCommState  = -103;
constexpr int32_t kErrSetCommTimeouts = -104;
constexpr int32_t kErrWriteNotOpen  = -105;  // write：未 open
constexpr int32_t kErrWriteOwner    = -106;  // write：并发写拒绝
constexpr int32_t kErrWriteFail     = -107;  // write：WriteFile 失败
constexpr int32_t kErrWritePartial  = -108;  // write：写不完整
}

SerialPort::~SerialPort() {
    close();
}

// ============================================================================
// 打开/关闭
// ============================================================================
Scanner::Result SerialPort::open(const std::string& port, int baud) {
    if (hSerial_) return Scanner::Result::fail(kErrAlreadyOpen, "串口已打开");

    std::string dev = port;
    if (dev.substr(0, 3) != "\\\\.") dev = "\\\\.\\" + dev;

    HANDLE h = CreateFileA(dev.c_str(), GENERIC_READ | GENERIC_WRITE,
                           0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return Scanner::Result::fail(kErrCreateFile, "串口打开失败: " + port);
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(h, &dcb)) {
        CloseHandle(h);
        return Scanner::Result::fail(kErrGetCommState, "GetCommState失败");
    }
    dcb.BaudRate = baud;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    if (!SetCommState(h, &dcb)) {
        CloseHandle(h);
        return Scanner::Result::fail(kErrSetCommState, "SetCommState失败");
    }

    COMMTIMEOUTS timeouts = {};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    if (!SetCommTimeouts(h, &timeouts)) {
        CloseHandle(h);
        return Scanner::Result::fail(kErrSetCommTimeouts, "SetCommTimeouts失败");
    }

    // 缓冲调大/清残留：非致命——失败仅 warn 继续，不 fail open（S-T5 前置清理）
    if (!SetupComm(h, 4096, 4096))
        spdlog::warn("[SerialPort] SetupComm 失败（继续使用默认缓冲）: {}", port);
    if (!PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR))
        spdlog::warn("[SerialPort] PurgeComm 失败（继续打开）: {}", port);

    hSerial_ = h;
    owner_.store(0, std::memory_order_release);
    return Scanner::Result::ok();
}

Scanner::Result SerialPort::close() {
    HANDLE h = static_cast<HANDLE>(hSerial_);
    hSerial_ = nullptr;
    owner_.store(0, std::memory_order_release);
    if (h) CloseHandle(h);
    return Scanner::Result::ok();
}

bool SerialPort::isOpen() const { return hSerial_ != nullptr; }

// ============================================================================
// 读（串口rx线程）
// ============================================================================
int SerialPort::read(char* buf, int cap) {
    HANDLE h = static_cast<HANDLE>(hSerial_);
    if (!h || !buf || cap <= 0) return -1;
    DWORD got = 0;
    if (!ReadFile(h, buf, static_cast<DWORD>(cap), &got, nullptr)) return -1;
    return static_cast<int>(got);
}

// ============================================================================
// 写（属主线程；R2-A1 并发写拒绝）
// ============================================================================
Scanner::Result SerialPort::write(const std::string& bytes) {
    HANDLE h = static_cast<HANDLE>(hSerial_);
    if (!h) return Scanner::Result::fail(kErrWriteNotOpen, "串口未打开");

    const uint64_t self = static_cast<uint64_t>(GetCurrentThreadId());
    uint64_t expected = 0;
    if (!owner_.compare_exchange_strong(expected, self,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)
        && expected != self) {
        return Scanner::Result::fail(kErrWriteOwner, "串口并发写拒绝");
    }

    DWORD written = 0;
    if (!WriteFile(h, bytes.data(), static_cast<DWORD>(bytes.size()),
                   &written, nullptr)) {
        return Scanner::Result::fail(kErrWriteFail, "串口写失败");
    }
    if (written != bytes.size()) {
        return Scanner::Result::fail(kErrWritePartial, "串口写不完整");
    }
    return Scanner::Result::ok();
}

} // namespace Scanner::device::serial
