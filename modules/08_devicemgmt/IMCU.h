#pragma once
// ============================================================================
// IMCU.h — 下位机 MCU 接口（HAL 层；协议 260831 口径）
//
// 下行 4 条（N10 采集/N11 停止/N12 温度周期/N13 加热）＋上行 G01/G02/G03 分流。
// 所有 send* 非阻塞：帧入写队列即返，完成回调同步回告写成败（无 ACK——盲发口径 D8）。
// ============================================================================
#include "base/types.h"
#include "serial/CommandChannel.h"
#include "serial/McuFrame.h"
#include <functional>
#include <string>

namespace Scanner::hal {

struct CaptureParams {              // N10 七参（协议表：H 1-200 / B 0-100 / T V C D 0|1 / L 0-100）
    int freqHz = 60;
    int bgLight = 10;
    int laserT = 1, laserV = 1, laserC = 0, laserD = 0;   // 四激光管开关（归 ScanMode 映射，批3）
    int laserLevel = 40;
};

struct McuUplink {                  // 上行分流出口（DeviceManager 注册；均在逻辑线程回调）
    std::function<void(const Scanner::device::serial::TempFrame&)> onTemp;
    std::function<void(const Scanner::device::serial::GestureEvent&)> onGesture;
    std::function<void(const Scanner::device::serial::ShotCountFrame&)> onShotCount;
};

class IMCU {
public:
    virtual ~IMCU() = default;
    virtual Scanner::Result open(const std::string& port) = 0;   // 内起 rx 线程
    virtual Scanner::Result close() = 0;
    virtual bool isOpen() const = 0;

    using DoneCb = Scanner::device::serial::CommandChannel::DoneCb;
    virtual void setCaptureParams(const CaptureParams&, DoneCb) = 0;   // "N10 H.. B.. T.. V.. C.. D.. L.."
    virtual void stopScan(DoneCb) = 0;                                 // "N11 H0"
    virtual void setTempReportPeriod(int periodMs, DoneCb) = 0;        // "N12 T<5-1000>"
    virtual void setHeatTarget(int celsius, DoneCb) = 0;               // "N13 S<0-80>"

    virtual void setUplink(McuUplink h) = 0;
    virtual Scanner::TimestampMs lastRxTime() const = 0;    // 通讯心跳时间戳
    virtual void pump() = 0;    // 逻辑线程调：排空 3 环→分流回调；回调内不得再调 pump/send

    // —— 观测（调试/巡检）——
    virtual uint64_t gestureRingDropped() const = 0;        // G01 手势环满丢新计数（0x0806）
    virtual uint64_t parseFailCount() const = 0;            // 上行载荷解析失败计数
    virtual uint64_t shotCount() const = 0;                 // 最近 G03 计数值快照（记账口径 D6）
};

} // namespace Scanner::hal
