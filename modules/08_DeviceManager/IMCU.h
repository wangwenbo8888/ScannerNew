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

struct CaptureParams {              // N10 八参（协议 260831 §下行表＋261105 增补 E；H 1-200 / B 0-100 / T V C D E 0|1 / L 0-100）
    int freqHz = 130;   // 261110 用户定版 130——相机问题现最大采集帧率 130（60→120→30→120→160→130；账本 makeParamSpecs 同源、默认即上限）
    int bgLight = 10;
    int laserT = 1, laserV = 1, laserC = 0, laserD = 0;   // 激光管开关 T/V/C/D（261105
                                     // 固件语义：T=左斜 V=右斜 C=深孔 D=精细；ScanMode 映射见 DeviceManager::effectiveN10；默认值从不直发——发送点均逐字段覆写）
    int laserE = 0;     // 第 8 参 E＝备用中竖25线激光管开关（261105 固件增补，帧序 D 后 L 前；
                        // 现无扫描模式启用恒 0——将来启用须真机验证灯型/轮序）
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
    virtual void setCaptureParams(const CaptureParams&, DoneCb) = 0;        // N10 H..B..T..V..C..D..E..L..
    virtual void startScan(DoneCb) = 0;   virtual void stopScan(DoneCb) = 0;         // N11 H1/H0
    virtual void enterSelfCheck(DoneCb) = 0; virtual void exitSelfCheck(DoneCb) = 0; // N12 Z1/Z0
    virtual void enterStandby(DoneCb) = 0; virtual void exitStandby(DoneCb) = 0;     // N13 E1/E0
    virtual void setHeatTarget(int celsius, DoneCb) = 0;                             // N14 T..
    virtual void setDeviceLed(int s1to4) = 0;   // N14 S1-S4 设备指示灯（协议(2) 260831：
                                                // S1 黄初始化/S2 红故障/S3 绿待机后处理/S4 蓝标定扫描；
                                                // 与 N14 T 加热参数字母并存——两版协议冲突待固件裁决，互不干扰）
    virtual void queryTemperature(int v0to2) = 0;                                    // N15（发不等）
    virtual void enterCalibration(DoneCb) = 0; virtual void exitCalibration(DoneCb) = 0; // N16 B1/B0

    virtual void setUplink(McuUplink h) = 0;
    virtual void setProtocolVersion(Scanner::device::serial::FrameCodec::Version v) = 0; // open 前配
    virtual Scanner::TimestampMs lastRxTime() const = 0;    // 通讯心跳时间戳（设计方案 §4-4）
    virtual uint64_t seqGapCount() const = 0;               // 帧计数对账丢帧计数（0x0808 源——
                                                            // 260831 协议改 G03 S 帧计数对账，T-seq 跳变弃）
    virtual void pump() = 0;    // 逻辑线程调：排空 4 环→分流回调+onAck；回调内不得再调 pump/send（短平快铁律）
};

} // namespace Scanner::hal
