// ============================================================================
// SerialPort.cpp — QSerialPort 实现（工厂软件同款；2026-08-30 定版）
//
// 架构：QSerialPort 住专属 QThread（事件循环）——QSerialPort 的 readyRead/
//       bytesWritten 依赖事件循环，此前裸 std::thread 调用会收不到数据（已踩坑）。
//       对上层保持同步阻塞接口：
//         read  = rx 线程阻塞取  ← cv 字节队列 ← port 线程 readyRead→readAll
//         write = 调用线程阻塞等 ← cv 结果     ← port 线程 write+waitForBytesWritten
// ============================================================================
#include "SerialPort.h"

#include <QMetaObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QThread>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <algorithm>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // GetCurrentThreadId（R2-A1 写属主登记）

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace Scanner::device::serial {

namespace {
// 本类错误码段位（沿用旧口径）
constexpr int32_t kErrAlreadyOpen = -100;
constexpr int32_t kErrOpenFail    = -101;
constexpr int32_t kErrNotOpen     = -105;
constexpr int32_t kErrOwner       = -106;
constexpr int32_t kErrWriteFail   = -107;
constexpr size_t  kRxBufCap = 64 * 1024;   // RX 桥队列上限（满丢新——rx 消费慢防爆）
} // namespace

class SerialPort::Impl {
public:
    QThread portThread;                      // QSerialPort 专属线程（事件循环）
    QSerialPort port;                        // 构造后 moveToThread；open/close/write 均经 invokeMethod
    std::atomic<bool> opened{false};

    // —— RX 桥：port 线程 → cv 队列 → 上层 rx 线程 ——
    std::mutex rxMtx;
    std::condition_variable rxCv;
    std::deque<char> rxBuf;
    bool rxHooked = false;                   // readyRead 是否已 connect（open 时一次）

    // —— TX 桥：上层调用线程 → port 线程（invokeMethod）→ cv 结果回传 ——
    std::mutex txMtx;
    std::condition_variable txCv;
    bool txPending = false;
    bool txOk = false;

    // —— open 桥结果 ——
    std::mutex openMtx;
    std::condition_variable openCv;
    bool openPending = false;
    bool openOk = false;
    QString openErr;
};

// ============================================================================
// 构造 / 析构
// ============================================================================
SerialPort::SerialPort() : impl_(new Impl()) {
    impl_->portThread.start();               // 线程常驻（open 复用）；析构 quit
    impl_->port.moveToThread(&impl_->portThread);
}

SerialPort::~SerialPort() {
    close();
    {
        std::lock_guard<std::mutex> lock(impl_->rxMtx);
        impl_->rxBuf.clear();
    }
    QMetaObject::invokeMethod(&impl_->port, [this]() { impl_->port.close(); },
                              Qt::BlockingQueuedConnection);
    impl_->portThread.quit();
    impl_->portThread.wait(1000);
    delete impl_;
}

// ============================================================================
// 枚举 COM 口
// ============================================================================
std::vector<std::string> SerialPort::listPorts() {
    std::vector<std::string> result;
    const auto ports = QSerialPortInfo::availablePorts();
    for (const auto& info : ports) {
        result.push_back(info.portName().toStdString());
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// 打开 / 关闭
// ============================================================================
Scanner::Result SerialPort::open(const std::string& port, int baud) {
    if (impl_->opened.load()) return Scanner::Result::fail(kErrAlreadyOpen, "串口已打开");
    {
        std::unique_lock<std::mutex> lock(impl_->openMtx);
        impl_->openPending = true;
    }
    const QString name = QString::fromStdString(port);
    QMetaObject::invokeMethod(&impl_->port, [this, name, baud]() {
        // 工厂同款序列：setPortName → open → 8N1/NoFlowControl（DTR/RTS 不设——
        // QSerialPort 默认不拉高，与工厂电气行为一致）
        impl_->port.setPortName(name);
        if (!impl_->port.open(QIODevice::ReadWrite)) {
            std::lock_guard<std::mutex> lock(impl_->openMtx);
            impl_->openOk = false;
            impl_->openErr = impl_->port.errorString();
            impl_->openPending = false;
            impl_->openCv.notify_all();
            return;
        }
        impl_->port.setBaudRate(baud);
        impl_->port.setDataBits(QSerialPort::Data8);
        impl_->port.setParity(QSerialPort::NoParity);
        impl_->port.setStopBits(QSerialPort::OneStop);
        impl_->port.setFlowControl(QSerialPort::NoFlowControl);
        if (!impl_->rxHooked) {
            impl_->rxHooked = true;
            QObject::connect(&impl_->port, &QSerialPort::readyRead, [this]() {
                const QByteArray data = impl_->port.readAll();
                std::lock_guard<std::mutex> lock(impl_->rxMtx);
                if (impl_->rxBuf.size() < kRxBufCap) {
                    impl_->rxBuf.insert(impl_->rxBuf.end(), data.begin(), data.end());
                }
                impl_->rxCv.notify_all();
            });
        }
        std::lock_guard<std::mutex> lock(impl_->openMtx);
        impl_->openOk = true;
        impl_->openPending = false;
        impl_->openCv.notify_all();
    }, Qt::QueuedConnection);

    {
        std::unique_lock<std::mutex> lock(impl_->openMtx);
        if (!impl_->openCv.wait_for(lock, std::chrono::seconds(3),
                                    [this] { return !impl_->openPending; })) {
            return Scanner::Result::fail(kErrOpenFail, "串口打开超时: " + port);
        }
    }
    if (!impl_->openOk) {
        return Scanner::Result::fail(kErrOpenFail,
            "串口打开失败: " + port + " (" + impl_->openErr.toStdString() + ")");
    }
    {
        std::lock_guard<std::mutex> lock(impl_->rxMtx);
        impl_->rxBuf.clear();
    }
    impl_->opened.store(true);
    return Scanner::Result::ok();
}

Scanner::Result SerialPort::close() {
    if (!impl_->opened.exchange(false)) return Scanner::Result::ok();
    QMetaObject::invokeMethod(&impl_->port, [this]() { impl_->port.close(); },
                              Qt::QueuedConnection);
    {
        std::lock_guard<std::mutex> lock(impl_->rxMtx);
        impl_->rxBuf.clear();
        impl_->rxCv.notify_all();            // 唤醒可能阻塞的 read（返回 0/-1）
    }
    JMW_LOG_INFO("08-SerialPort", "[SerialPort] 串口已关闭（QSerialPort）");
    return Scanner::Result::ok();
}

bool SerialPort::isOpen() const {
    return impl_->opened.load();
}

// ============================================================================
// 读（rx 线程；阻塞 50ms 语义） / 写（属主线程；阻塞至写完成）
// ============================================================================
int SerialPort::read(char* buf, int cap) {
    if (!impl_->opened.load() || !buf || cap <= 0) return -1;
    std::unique_lock<std::mutex> lock(impl_->rxMtx);
    if (impl_->rxBuf.empty()) {
        impl_->rxCv.wait_for(lock, std::chrono::milliseconds(50));
        if (impl_->rxBuf.empty()) return impl_->opened.load() ? 0 : -1;
    }
    const int n = static_cast<int>(std::min<size_t>(impl_->rxBuf.size(),
                                                    static_cast<size_t>(cap)));
    for (int i = 0; i < n; ++i) {
        buf[i] = impl_->rxBuf.front();
        impl_->rxBuf.pop_front();
    }
    return n;
}

Scanner::Result SerialPort::write(const std::string& bytes) {
    if (!impl_->opened.load()) return Scanner::Result::fail(kErrNotOpen, "串口未打开");
    // R2-A1 单写者纪律（口径保留：首次调用线程登记为属主）
    const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
    uint64_t expect = 0;
    if (!owner_.compare_exchange_strong(expect, tid) && expect != tid) {
        return Scanner::Result::fail(kErrOwner, "串口并发写拒绝");
    }
    {
        std::unique_lock<std::mutex> lock(impl_->txMtx);
        impl_->txPending = true;
    }
    const QByteArray data(bytes.data(), static_cast<int>(bytes.size()));
    QMetaObject::invokeMethod(&impl_->port, [this, data]() {
        const qint64 written = impl_->port.write(data);
        const bool flushed = (written == data.size()) &&
                             impl_->port.waitForBytesWritten(3000);
        std::lock_guard<std::mutex> lock(impl_->txMtx);
        impl_->txOk = flushed;
        impl_->txPending = false;
        impl_->txCv.notify_all();
    }, Qt::QueuedConnection);

    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lock(impl_->txMtx);
        if (!impl_->txCv.wait_for(lock, std::chrono::milliseconds(3500),
                                  [this] { return !impl_->txPending; })) {
            return Scanner::Result::fail(kErrWriteFail, "串口写桥超时");
        }
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    if (ms > 1000) {
        JMW_LOG_WARN("08-SerialPort", "[SerialPort] 慢写 {}ms（{} 字节）", ms, bytes.size());
    }
    if (!impl_->txOk) {
        return Scanner::Result::fail(kErrWriteFail,
            "串口写不完整: " + std::to_string(bytes.size()) + " 字节");
    }
    JMW_LOG_INFO("08-SerialPort", "[SerialPort] TX: '{}'", bytes);
    return Scanner::Result::ok();
}

Scanner::Result SerialPort::writeKeepalive(const std::string& bytes) {
    return write(bytes);
}

void SerialPort::resetWriteOwner() {
    owner_.store(0);
}

} // namespace Scanner::device::serial
