#pragma once
// ============================================================================
// MCUDriver.h — 下位机 MCU 串口驱动（HAL 层实现）
//
// 实现 Scanner::hal::IMCU 接口。
// 串口协议: N10(启动) N11(停止) 温度/按键通过串口回传。
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "hal/IMCU.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace Scanner::device {

class MCUDriver : public hal::IMCU {
public:
    explicit MCUDriver(int baudRate = 115200);
    ~MCUDriver() override;

    MCUDriver(const MCUDriver&) = delete;
    MCUDriver& operator=(const MCUDriver&) = delete;

    // IMCU 接口
    Result open(const std::string& portOrDevice) override;
    Result close() override;
    bool isOpen() const override;

    Result sendSoftwareTrigger() override;
    Result setHardwareTriggerMode(bool enabled) override;

    Result setLaserOn(bool on) override;
    Result setLaserPower(int level) override;
    Result setLedOn(bool on) override;

    Result emergencyStop() override;
    bool isEmergencyStop() const override;

    double getTemperature() const override;

    Result registerCallback(hal::McuEventCallback cb) override;
    Result unregisterCallback() override;

    // 扫描控制（扩展）
    Result startScan(int freq, int bgLight, int laserLight,
                     int trigger = 1, int version = 2);
    Result stopScan();

    // 光源参数
    void setFillLight(int level) { fillLight_ = level; }
    int getFillLight() const { return fillLight_; }

private:
    void sendCommand(const std::string& cmd);
    void receiveLoop();

    int baudRate_;
    std::atomic<bool> isOpen_{false};
    std::atomic<bool> emergency_{false};
    std::atomic<double> temperature_{0.0};

    int laserPower_ = 60;
    int fillLight_ = 40;
    bool laserOn_ = false;
    bool ledOn_ = false;
    bool hwTrigger_ = false;

    // 串口句柄（Windows）
    void* hSerial_ = nullptr;  // HANDLE
    std::thread rxThread_;
    std::atomic<bool> rxRunning_{false};

    hal::McuEventCallback callback_;
    mutable std::mutex callbackMutex_;

    void parseReceived(const std::string& data);
};

} // namespace Scanner::device
