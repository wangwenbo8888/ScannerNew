#pragma once
// ============================================================================
// MCUDriver.h — 下位机 MCU 驱动 = 三小层组合壳（HAL 实现；协议 260831 口径）
//
// 组合：SerialPort(纯IO) + FrameCodec(裸';'成拆帧) + CommandChannel(下行出口)
//       + 3 个有界环（G01 手势环 64 / G03 计数环 64 + G02 温度环 8——满丢新
//       各自计数，D-T12a 口径）。
// 线程：rx 线程（open 起，零业务：read→feed→按 G01/G02/G03 前缀入环）；
//       逻辑线程 send*/setUplink/pump（排空环→Uplink 回调）。
// 下行 4 条 typed 命令（N10-N13）见 IMCU.h——无 ACK 无回显，盲发口径 D8。
// ============================================================================

#include "IMCU.h"
#include "serial/CommandChannel.h"
#include "serial/FrameCodec.h"
#include "serial/McuFrame.h"
#include "serial/SerialPort.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Scanner::device {

class MCUDriver : public hal::IMCU {
public:
    // 测试注入口：非空则下行帧不走 SerialPort，且 open 不开真串口/不起 rx 线程
    //（测试模式单线程，配合 testInjectRaw 回灌上行）；产线恒 nullptr
    using WriteOverride = std::function<bool(const std::string& frame)>;
    explicit MCUDriver(WriteOverride writeOverride = nullptr);
    ~MCUDriver() override;

    MCUDriver(const MCUDriver&) = delete;
    MCUDriver& operator=(const MCUDriver&) = delete;

    // —— IMCU：开闭（open 内起 rx 线程；close 倒序：停线程→关串口）——
    Scanner::Result open(const std::string& port) override;         // 默认波特率
    Scanner::Result open(const std::string& port, int baud);        // D-T12b：baud 接通
    Scanner::Result close() override;
    bool isOpen() const override;

    // —— IMCU：typed 下行 4 条（payload 拼装 → CommandChannel）——
    void setCaptureParams(const hal::CaptureParams& p, DoneCb cb) override;   // N10 七参（即启采）
    void stopScan(DoneCb cb) override;                                        // N11 H0（停止/熄灯）
    void setTempReportPeriod(int periodMs, DoneCb cb) override;               // N12 T<5-1000ms>
    void setHeatTarget(int celsius, DoneCb cb) override;                      // N13 S<0-80>

    // —— IMCU：上行/配置/观测 ——
    void setUplink(hal::McuUplink h) override;                    // open 时设，逻辑线程 pump 消费
    Scanner::TimestampMs lastRxTime() const override;
    void pump() override;    // 逻辑线程调：排空 3 环→uplink 回调（手势→计数→温度）

    // —— IMCU 观测口（调试/巡检）——
    uint64_t gestureRingDropped() const override;   // G01 手势环满丢新计数（0x0806）
    uint64_t parseFailCount() const override;       // 上行载荷解析失败计数
    uint64_t shotCount() const override;            // 最近 G03 计数值快照（记账口径 D6）

    // —— D-T12b 增补（门面驱动用）——
    void lightsOff();             // 熄灯收口＝stopScan（N11 H0——新协议停止与熄灯同一
                                  // 命令；close 场景调，写权已随逻辑线程 join 交接）
    void flushWrites(int timeoutMs = 300);  // 等写队列排空（有界）：停采集前用——熄灯帧
                                  // 落线后再触发相机停流（其 USB 风暴会把串口写堵 2~2.5s）
    void resetSerialWriteOwner(); // 写权交还：open 调用线程末次直写（搜口探测）后、
                                  // 逻辑线程首发前调（R2-A1 登记复位；否则逻辑线程首写被拒）
    // 串口收发监听（调试弹窗用）：TX=写线程实际出队帧（含分号），RX=上行完整帧载荷。
    // 回调可能来自 rx/写线程——订阅方自行切线程；open 前设置
    void setWireTap(std::function<void(bool tx, const std::string& data)> tap);

    // —— 测试缝（仅测试）：等价 rx 线程收到原始字节——喂 codec 并分发入环 ——
    void testInjectRaw(const std::string& frameBytes);

private:
    void rxLoop();                // rx 线程主体：read→feed→dispatch（含半帧超时推进）
    void dispatchFrame(const serial::FrameCodec::Frame& f);       // 单帧按 G01/G02/G03 分流（rx/测试共用）
    void onParseFail(const std::string& payload);                 // 载荷弃帧：warn+计数
    bool writeFrame(const std::string& frame);                    // CommandChannel 写出口
    serial::CommandChannel::Deps makeDeps();                      // 组装 channel 依赖（写口/时钟）
    std::string probeAutoPort(int baud);                          // 自动搜口：探测命中返回口名，全败返回空
    void closePortOnly();                                         // 探测逐口收尾：停 rx＋关串口＋open_ 复位（不
                                                                  // 动写线程——写线程贯穿整个搜口，全败由 open 收口）

    static constexpr int kDefaultBaud = 115200;   // 协议现状固定波特率（旧构造参已删）

    // —— 组合三小层（声明序即初始化序：codec_ 先于 channel_）——
    WriteOverride writeOverride_;                // 测试注入（空=走 SerialPort）
    serial::FrameCodec codec_;                   // 裸 ';' 成拆帧（协议 260831 唯一口径，无版本）
    serial::CommandChannel channel_;             // ctor/open 以 makeDeps() 重建
    serial::SerialPort serial_;

    // —— 上行 3 环（rx 生产 / pump 消费；满丢新各自计数——D-T12a 口径）——
    serial::SpscRing<serial::GestureEvent, 64> gestureRing_;
    serial::SpscRing<serial::TempFrame, 8> tempRing_;
    serial::SpscRing<serial::ShotCountFrame, 64> shotRing_;

    hal::McuUplink uplink_;                      // 逻辑线程设置/回调
    std::atomic<bool> gSeen_{false};             // 任意完整帧到达置位（自动搜口探测凭据）
    std::atomic<uint64_t> lastShot_{0};          // 最近 G03 计数快照（shotCount() 出口）

    // —— 串口写线程（实测 WriteFile 被 USB 驱动偶发阻塞 2~2.5s——相机 USB 流量挤占
    //    总线；直写会堵死 open/逻辑线程。全部下行帧入队即返，阻塞只落在写线程）——
    struct WriteItem { std::string frame; };
    std::deque<WriteItem> writeQueue_;
    std::mutex writeMtx_;
    std::condition_variable writeCv_;            // 入队唤醒 / 排空通知（lightsOff 收口用）
    std::condition_variable drainCv_;
    std::thread writeThread_;
    std::atomic<bool> writeRunning_{false};
    void writeLoop();                            // 写线程主体：出队→WriteFile（唯一写者）
    void enqueueWrite(const std::string& frame); // 入队（任何线程）
    void waitWriteDrained(int timeoutMs);        // 等队列排空（close/lightsOff 收口）
    void stopWriteThread();                      // 排空+停写线程+清队（open 失败/close 收口）
    std::atomic<bool> open_{false};
    std::atomic<bool> rxRunning_{false};
    std::thread rxThread_;

    std::atomic<Scanner::TimestampMs> lastRx_{0};   // 通讯心跳（任何完整上行帧刷新）
    std::atomic<uint64_t> parseFailCount_{0};       // 上行载荷解析失败计数
    std::function<void(bool, const std::string&)> wireTap_;  // 串口收发监听（调试）
    std::mutex tapMtx_;                             // wireTap_ 装卸互斥（回调热路径无锁快查）
    void notifyTap(bool tx, const std::string& data);
};

} // namespace Scanner::device
