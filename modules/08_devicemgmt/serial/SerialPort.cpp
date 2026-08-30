// ============================================================================
// SerialPort.cpp — 纯串口 IO 实现（Windows）
// ============================================================================

#include "SerialPort.h"
#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <utility>

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
    // 流控显式全关（工业串口标准动作）：GetCommState 保留的驱动默认值在
    // CH34X 某些版本为开（fOutxCtsFlow=TRUE）——CTS 未接线/未就绪时 WriteFile
    // 阻塞等"流控放开"，实测固定卡 ~2.6s 周期（其它软件显式关闭故快）。
    // DTR/RTS 置 ENABLE：维持线路电平（部分适配器借 DTR/RTS 供电或复位）
    dcb.fOutxCtsFlow = FALSE;              // CTS 硬件流控关
    dcb.fOutxDsrFlow = FALSE;              // DSR 硬件流控关
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutX = FALSE;                     // XON/XOFF 软件流控关
    dcb.fInX = FALSE;
    dcb.fTXContinueOnXoff = TRUE;
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

    // 清残留＋启用 EV_RXCHAR 事件掩码（read 的 WaitCommEvent 依赖此设置；
    // SetCommMask 失败则 read 退化为立即返回 0——不 fail open）
    if (!PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR))
        JMW_LOG_WARN("08-SerialPort", "[SerialPort] PurgeComm 失败（继续打开）: {}", port);
    if (!SetCommMask(h, EV_RXCHAR | EV_ERR))
        JMW_LOG_WARN("08-SerialPort", "[SerialPort] SetCommMask 失败（读将失效）: {}", port);

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

void SerialPort::resetWriteOwner() { owner_.store(0, std::memory_order_release); }

// ============================================================================
// 枚举 COM 口（QueryDosDevice 符号链接名，不开句柄；口号升序）
// ============================================================================
std::vector<std::string> SerialPort::listPorts() {
    std::vector<std::pair<int, std::string>> found;
    DWORD need = 8192;
    std::vector<char> buf(need);
    for (;;) {                                  // 缓冲不够按需翻倍重试（上限防炸）
        const DWORD n = QueryDosDeviceA(nullptr, buf.data(), need);
        if (n > 0) break;
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || need >= 1 << 20) return {};
        need *= 2;
        buf.assign(need, '\0');
    }
    for (const char* p = buf.data(); *p; p += std::strlen(p) + 1) {
        if (std::strncmp(p, "COM", 3) != 0) continue;
        const char* d = p + 3;
        if (!*d) continue;                      // 裸 "COM" 别名跳过
        bool digits = true;
        for (const char* q = d; *q; ++q)
            if (*q < '0' || *q > '9') { digits = false; break; }
        if (digits) found.emplace_back(std::atoi(d), p);
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> ports;
    ports.reserve(found.size());
    for (auto& f : found) ports.push_back(std::move(f.second));
    return ports;
}

// ============================================================================
// 读（串口rx线程）——WaitCommEvent 事件驱动：无数据时不占驱动。
// 工厂软件对照实证（QSerialPort 顺畅 vs 我们卡 2.6s）：同步句柄下 RX 线程
// 持续 ReadFile 超时循环会占住 CH343 驱动内部队列，写线程的 WriteFile 被排到
// 读间隙 → 秒级挂起（工厂不读串口故写独占、零延迟）。改为 EV_RXCHAR 事件
// 通知——有数据才 ReadFile，空闲期驱动上零挂起 IRP，写通道畅通
// ============================================================================
int SerialPort::read(char* buf, int cap) {
    HANDLE h = static_cast<HANDLE>(hSerial_);
    if (!h || !buf || cap <= 0) return -1;

    DWORD evMask = 0;
    if (!WaitCommEvent(h, &evMask, nullptr)) return -1;   // 阻塞等数据（rx 线程专属）
    if (!(evMask & EV_RXCHAR)) return 0;                  // 非数据事件：视作空读

    // 事件到＝缓冲有字节；一次性读完（紧跟的 ReadFile 立即返回已有数据）
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

    const auto t0 = std::chrono::steady_clock::now();   // 慢写探针：USB 串口驱动偶发
    DWORD written = 0;                                   // 长阻塞（流控/IRP 排队）——记档定位
    if (!WriteFile(h, bytes.data(), static_cast<DWORD>(bytes.size()),
                   &written, nullptr)) {
        return Scanner::Result::fail(kErrWriteFail, "串口写失败");
    }
    const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (el > 50) JMW_LOG_WARN("08-SerialPort", "[SerialPort] 慢写 {}ms（{} 字节）", el, bytes.size());
    if (written != bytes.size()) {
        return Scanner::Result::fail(kErrWritePartial, "串口写不完整");
    }
    return Scanner::Result::ok();
}

Scanner::Result SerialPort::writeKeepalive(const std::string& bytes) {
    HANDLE h = static_cast<HANDLE>(hSerial_);
    if (!h) return Scanner::Result::fail(kErrWriteNotOpen, "串口未打开");
    const auto t0 = std::chrono::steady_clock::now();   // 探针：保活写挂起现形（判
    DWORD written = 0;                                   // CH343 纯 TX 挂起 vs 总线竞争）
    if (!WriteFile(h, bytes.data(), static_cast<DWORD>(bytes.size()),
                   &written, nullptr)) {
        return Scanner::Result::fail(kErrWriteFail, "串口写失败");
    }
    const auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (el > 50)
        JMW_LOG_WARN("08-SerialPort", "[SerialPort] 保活慢写 {}ms（{} 字节）", el, bytes.size());
    if (written != bytes.size()) {
        return Scanner::Result::fail(kErrWritePartial, "串口写不完整");
    }
    return Scanner::Result::ok();
}

} // namespace Scanner::device::serial
