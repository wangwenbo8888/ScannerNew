#pragma once
// ============================================================================
// SerialPort.h — 纯串口 IO（Win32 句柄封装；无协议知识）
// R2-A1 铁律强制点：write 只许一个线程调（逻辑线程）——第二线程并发调用直接
// fail("串口并发写拒绝")，防同句柄并发 WriteFile 字节交错出乱帧。
// 线程契约：read=串口rx线程；write/open/close=属主（逻辑）线程。
// ============================================================================
#include "base/types.h"
#include <atomic>
#include <cstdint>
#include <string>

namespace Scanner::device::serial {

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort();
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    // port 如 "COM3"；open 内做：\\.\ 前缀补齐 + DCB 8N1 + COMMTIMEOUTS
    //（ReadIntervalTimeout=50/ReadTotalTimeoutConstant=50/ReadTotalTimeoutMultiplier=10，
    //  Write 同例）+ SetupComm(h, 4096, 4096)（C5 缓冲调大）+ PurgeComm 清残留
    Scanner::Result open(const std::string& port, int baud);
    // 幂等；不拥有线程（rx 线程归上层）——关 rx 靠 CloseHandle 令阻塞 ReadFile
    // 以 ERROR_OPERATION_ABORTED 退出
    Scanner::Result close();
    bool isOpen() const;

    // 串口rx线程调：返回实读字节数（0=超时，<0=错误）；非阻塞语义靠 COMMTIMEOUTS
    int read(char* buf, int cap);

    // 属主线程调；首次调用线程被记为属主（Win32 原生线程 id 比较，避开
    // std::thread::id 可原子性争议），他线程调用返回 fail("串口并发写拒绝")；
    // 未 open 调用返回 fail("串口未打开")
    Scanner::Result write(const std::string& bytes);

private:
    void* hSerial_ = nullptr;                 // HANDLE（Win32 不进头）
    std::atomic<uint64_t> owner_{0};          // write 属主线程 id（GetCurrentThreadId；0=未登记，close 复位）
};

} // namespace Scanner::device::serial
