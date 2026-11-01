#pragma once
// ============================================================================
// SerialPort.h — 纯串口 IO（QSerialPort 实现，工厂软件同款；无协议知识）
// 2026-08-30 定版：底层换 QSerialPort（Win32 CreateFile 版弃用）——与工厂
// 软件通信方式完全一致（DTR/RTS 默认不拉高等电气行为同厂），杜绝"工厂能停
// 灯我们不能"类差异。
// 线程模型：QSerialPort 住专属 QThread（事件循环驱动 readyRead/write）；
//           对上层保持同步阻塞接口——read=rx 线程阻塞取（cv 队列桥），
//           write=调用线程阻塞等 port 线程完成（cv 桥）。R2-A1 单写者纪律
//           保留（属主检查不变——虽然底层 invokeMethod 排队已天然串行）。
// ============================================================================
#include "base/types.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace Scanner::device::serial {

class SerialPort {
public:
    SerialPort();
    ~SerialPort();
    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    // 枚举本机 COM 口（QSerialPortInfo）；按口号升序，如 {"COM3","COM8"}
    static std::vector<std::string> listPorts();

    // port 如 "COM3"；open：起专属线程 + QSerialPort open（115200 等由 baud 传）
    Scanner::Result open(const std::string& port, int baud);
    // 幂等；port 线程持续存在（复用），close 只关 QSerialPort 本体
    Scanner::Result close();
    bool isOpen() const;

    // rx 线程调：阻塞至有数据或超时（50ms 语义同旧 COMMTIMEOUTS）；返回实读
    // 字节数（0=超时，<0=错误）
    int read(char* buf, int cap);

    // 属主线程调（单写者纪律）；阻塞至 QSerialPort 写完成（waitForBytesWritten
    // 3s 兜底）或桥超时。未 open 调用返回 fail
    Scanner::Result write(const std::string& bytes);

    // 与 write 同径（历史保活口径保留；无慢写日志区分）
    Scanner::Result writeKeepalive(const std::string& bytes);

    // 复位 write 属主登记（0=未登记）：open 后跨线程交接写权用
    void resetWriteOwner();

private:
    class Impl;
    Impl* impl_ = nullptr;                   // QSerialPort/QThread/桥（cpp 内定义）
    std::atomic<uint64_t> owner_{0};         // write 属主线程 id（0=未登记，close 复位）
};

} // namespace Scanner::device::serial
