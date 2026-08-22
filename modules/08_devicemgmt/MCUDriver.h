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
    void setUplink(hal::McuUplink h) override;                    // open 时设，逻辑线程 pump 消费
    void setProtocolVersion(serial::FrameCodec::Version v) override;  // open 前配；开启中调用无效直至重开
    Scanner::TimestampMs lastRxTime() const override;
    uint64_t seqGapCount() const override;
    void pump() override;                                         // 逻辑线程调（含 A→onAck/T→seq 对账）

    // —— D-T12b 增补（门面驱动用）——
    void lightsOff();             // 熄灯收口：N10 B0/L0 fire-and-forget（固件无独立灯控命令——
                                  // N10 参数即灯态；close 场景调，写权已随逻辑线程 join 交接）
    void flushWrites(int timeoutMs = 300);  // 等写队列排空（有界）：停采集前用——熄灯帧
                                  // 落线后再触发相机停流（其 USB 风暴会把串口写堵 2~2.5s）
    void resetSerialWriteOwner(); // 写权交还：open 调用线程末次直写（探测/N12Z1 自检）后、
                                  // 逻辑线程首发前调（R2-A1 登记复位；否则逻辑线程首写被拒）
    bool probeDidSelfCheck() const;  // auto 搜口是否已发过 N12Z1（幂等省略口径；open 查）
    // 回环验证（启动自检用）：发 payload → timeoutMs 内收到固件回显同载荷帧即 true。
    // v2 固件对每条命令整帧回显——链路级"MCU 真收到"凭据。调用线程=逻辑线程（已持写权）
    bool sendEchoProbe(const std::string& payload, int timeoutMs = 400);
    // 最近回显载荷快照（非阻塞轮询用——自检状态机逐 tick 查；线程安全）
    std::string lastEchoPayload() const;
    void channelTick();          // CommandChannel 对账 tick 出口（S-T5 口径：pump 不调
                                 // tick——重传/3 败收口归 DeviceManager 逻辑线程驱动）
    void setAckTimeoutMs(int ms);  // ACK 超时注入（open 前配；钳 ≥1；测试缩短重传周期用）
    uint64_t keyDropCount() const;  // D-T13：K 事件环满丢新计数出口（门面 0x0806 巡检用；
                                    // 单调累计不随 open 复位——与 parseFailCount 同口径）

    // —— 测试缝（仅测试）：等价 rx 线程收到原始字节——喂 codec 并分发入环 ——
    void testInjectRaw(const std::string& frameBytes);

private:
    void rxLoop();                // rx 线程主体：read→feed→dispatch（含半帧超时推进）
    void dispatchFrame(const serial::FrameCodec::Frame& f);       // 单帧按首字符入环（rx/测试共用）
    void feedTextLine(const std::string& line);                   // v2 固件裸文本行（数值=温度上报；任何行=链路活）
    void onParseFail(const std::string& payload);                 // 载荷弃帧：warn+计数
    void accountTempSeq(uint16_t seq);                            // T seq 跳变对账（pump 内；v2 不对账）
    bool writeFrame(const std::string& frame);                    // CommandChannel 写出口
    serial::CommandChannel::Deps makeDeps();                      // 组装 channel 依赖（写口/时钟/可靠开关）
    void applyVersion();                                          // version_ → codec_/channel_ 重建（须未 open）
    std::string probeAutoPort(int baud);                          // 自动搜口：探测命中返回口名，全败返回空

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
    std::string lineBuf_;                        // v2 行缓冲（rx 线程私有；';'→codec 帧路径，'\n'→文本行）
    std::atomic<bool> probeHit_{false};          // 自动搜口命中凭据：数值行（温度上报；回显/纯回环不算）
    std::atomic<bool> probeN12Sent_{false};      // 搜口探测已发 N12Z1（open 幂等省略口径）
    mutable std::mutex echoMtx_;                 // lastEcho_ 保护（rx 写 / lastEchoPayload const 读）
    std::string lastEcho_;                       // 最近一次回显载荷（v2 固件整帧回显下行命令）

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

    std::atomic<Scanner::TimestampMs> lastRx_{0};   // 通讯心跳（任何有效上行帧刷新）
    std::atomic<uint64_t> seqGapCount_{0};          // T seq 跳变丢帧计数
    std::atomic<uint64_t> parseFailCount_{0};       // 上行载荷解析失败计数（含 v2 匿名 K/旧 E）
    bool hasTSeq_ = false;                          // T seq 对账基线（仅逻辑线程）
    uint16_t lastTSeq_ = 0;
};

} // namespace Scanner::device
