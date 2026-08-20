#pragma once
// ============================================================================
// MCUDriver.h — 下位机 MCU 驱动 = 三小层组合壳（HAL 实现；设计方案 §2.2-§2.5）
//
// 组合：SerialPort(纯IO) + FrameCodec(v2/v3成拆帧) + CommandChannel(可靠下行)
//       + 4 个有界环（K/S/A 事件环各 64 + T 遥测环 8——实现口径：分开实例化，
//       比 McuFrame.h 原型「K/S/A 合一环」更简：无变体分发，丢最旧语义各自独立）。
// 线程：rx 线程（open 起，零业务：read→feed→按首字符入环）；
//       逻辑线程 send*/setUplink/pump（排空环→Uplink 回调 + onAck 回填 + seq 对账）。
// typed N10–N16 见 IMCU.h；旧错牌命令（软触发/急停/N13 L/N14 B 等）已删净（§2.2）。
// ============================================================================

#include "IMCU.h"
#include "serial/CommandChannel.h"
#include "serial/FrameCodec.h"
#include "serial/McuFrame.h"
#include "serial/SerialPort.h"

#include <atomic>
#include <functional>
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

    // —— IMCU：typed N10–N16（payload 拼装 → CommandChannel）——
    void setCaptureParams(const hal::CaptureParams& p, DoneCb cb) override;
    void startScan(DoneCb cb) override;
    void stopScan(DoneCb cb) override;
    void enterSelfCheck(DoneCb cb) override;
    void exitSelfCheck(DoneCb cb) override;
    void enterStandby(DoneCb cb) override;
    void exitStandby(DoneCb cb) override;
    void setHeatTarget(int celsius, DoneCb cb) override;
    void queryTemperature(int v0to2) override;
    void enterCalibration(DoneCb cb) override;
    void exitCalibration(DoneCb cb) override;

    // —— IMCU：上行/配置/观测 ——
    void setUplink(hal::McuUplink h) override;                    // 逻辑线程调
    void setProtocolVersion(serial::FrameCodec::Version v) override;  // open 前配；开启中调用无效直至重开
    Scanner::TimestampMs lastRxTime() const override;
    uint64_t seqGapCount() const override;
    void pump() override;                                         // 逻辑线程调（含 A→onAck/T→seq 对账）

    // —— D-T12b 增补（门面驱动用）——
    void channelTick();          // CommandChannel 对账 tick 出口（S-T5 口径：pump 不调
                                 // tick——重传/3 败收口归 DeviceManager 逻辑线程驱动）
    void setAckTimeoutMs(int ms);  // ACK 超时注入（open 前配；钳 ≥1；测试缩短重传周期用）

    // —— 测试缝（仅测试）：等价 rx 线程收到原始字节——喂 codec 并分发入环 ——
    void testInjectRaw(const std::string& frameBytes);

private:
    void rxLoop();                // rx 线程主体：read→feed→dispatch（含半帧超时推进）
    void dispatchFrame(const serial::FrameCodec::Frame& f);       // 单帧按首字符入环（rx/测试共用）
    void onParseFail(const std::string& payload);                 // 载荷弃帧：warn+计数
    void accountTempSeq(uint16_t seq);                            // T seq 跳变对账（pump 内；v2 不对账）
    bool writeFrame(const std::string& frame);                    // CommandChannel 写出口
    serial::CommandChannel::Deps makeDeps();                      // 组装 channel 依赖（写口/时钟/可靠开关）
    void applyVersion();                                          // version_ → codec_/channel_ 重建（须未 open）

    static constexpr int kDefaultBaud = 115200;   // 协议现状固定波特率（旧构造参已删）
    int ackTimeoutMs_ = 100;                      // CommandChannel Deps 默认口径（可注入）

    // —— 组合三小层（声明序即初始化序：version_ 先于 codec_/channel_）——
    WriteOverride writeOverride_;                // 测试注入（空=走 SerialPort）
    serial::FrameCodec::Version version_{serial::FrameCodec::Version::V3};  // v3 目标口径
    serial::FrameCodec codec_{serial::FrameCodec::Version::V3};
    serial::CommandChannel channel_;             // ctor/open/applyVersion 以 makeDeps() 重建
    serial::SerialPort serial_;

    // —— 上行 4 环（rx 生产 / pump 消费；满丢新各自计数——D-T12a 口径）——
    serial::SpscRing<serial::RawKeyEvent, 64> keyRing_;
    serial::SpscRing<serial::StatusFrame, 64> statusRing_;
    serial::SpscRing<serial::AckFrame, 64> ackRing_;
    serial::SpscRing<serial::TempFrame, 8> tempRing_;

    hal::McuUplink uplink_;                      // 逻辑线程设置/回调
    std::atomic<bool> open_{false};
    std::atomic<bool> rxRunning_{false};
    std::thread rxThread_;

    std::atomic<Scanner::TimestampMs> lastRx_{0};   // 通讯心跳（任何有效上行帧刷新）
    std::atomic<uint64_t> seqGapCount_{0};          // T seq 跳变丢帧计数
    std::atomic<uint64_t> parseFailCount_{0};       // 上行载荷解析失败计数（含 v2 匿名 K/旧 E）
    bool hasTSeq_ = false;                          // T seq 对账基线（仅逻辑线程）
    uint16_t lastTSeq_ = 0;
};

} // namespace Scanner::device
