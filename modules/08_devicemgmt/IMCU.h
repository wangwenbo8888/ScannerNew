#pragma once
// ============================================================================
// IMCU.h — 下位机 MCU 接口（HAL 层，2026-08-20 08 设计方案 §2.2/§2.4 重写）
//
// typed N10–N16 命令（经 CommandChannel 可靠通道）+ 上行结构化事件分流。
// 所有 send* 方法非阻塞：结果经 DoneCb 异步回告（v2 恒立即 ok「未确认」）。
// ============================================================================

#include "base/types.h"
#include "serial/CommandChannel.h"
#include "serial/McuFrame.h"
#include <functional>
#include <string>

namespace Scanner::hal {

struct CaptureParams {              // N10 五参（协议表：H20-120/B0-255/T1-6/V1-6/L0-255）
    int freqHz = 60; int bgLight = 80; int laserSelectA = 1; int laserSelectB = 1; int laserLevel = 120;
};

struct McuUplink {                  // 上行分流出口（DeviceManager 注册；均在逻辑线程回调）
    std::function<void(const Scanner::device::serial::TempFrame&)> onTemp;
    std::function<void(const Scanner::device::serial::RawKeyEvent&)> onKey;
    std::function<void(const Scanner::device::serial::StatusFrame&)> onStatus;
};

class IMCU {
public:
    virtual ~IMCU() = default;
    virtual Scanner::Result open(const std::string& port) = 0;   // 内起 rx 线程
    virtual Scanner::Result close() = 0;
    virtual bool isOpen() const = 0;

    using DoneCb = Scanner::device::serial::CommandChannel::DoneCb;
    virtual void setCaptureParams(const CaptureParams&, DoneCb) = 0;        // $N10H..B..T..V..L..
    virtual void startScan(DoneCb) = 0;   virtual void stopScan(DoneCb) = 0;         // N11 H1/H0
    virtual void enterSelfCheck(DoneCb) = 0; virtual void exitSelfCheck(DoneCb) = 0; // N12 Z1/Z0
    virtual void enterStandby(DoneCb) = 0; virtual void exitStandby(DoneCb) = 0;     // N13 E1/E0
    virtual void setHeatTarget(int celsius, DoneCb) = 0;                             // N14 T..
    virtual void queryTemperature(int v0to2) = 0;                                    // N15（发不等）
    virtual void enterCalibration(DoneCb) = 0; virtual void exitCalibration(DoneCb) = 0; // N16 B1/B0

    virtual void setUplink(McuUplink h) = 0;
    virtual void setProtocolVersion(Scanner::device::serial::FrameCodec::Version v) = 0; // open 前配
    virtual Scanner::TimestampMs lastRxTime() const = 0;    // 通讯心跳时间戳（设计方案 §4-4）
    virtual uint64_t seqGapCount() const = 0;               // seq 跳变丢帧计数（对账 §6.2-9）
    virtual void pump() = 0;    // 逻辑线程调：排空 4 环→分流回调+onAck；回调内不得再调 pump/send（短平快铁律）
};

} // namespace Scanner::hal
