#pragma once
// ============================================================================
// IDeviceStateSink.h — 设备状态 Sink（Data 层拥有，注入 HAL）
// ============================================================================

#include "common/types.h"

namespace Scanner::data {

struct DeviceStateInfo {
    std::string deviceId;
    std::string deviceType;   // "ScannerCamera" / "MCU" / "Tracker"
    DeviceState state = DeviceState::Offline;
    double temperature = 0.0;
    double fps = 0.0;
    int droppedFrames = 0;
    TimestampMs timestamp = 0;
};

class IDeviceStateSink {
public:
    virtual ~IDeviceStateSink() = default;

    virtual Result pushState(const DeviceStateInfo& state) = 0;
    virtual DeviceStateInfo getState(const std::string& deviceId) const = 0;
};

} // namespace Scanner::data
