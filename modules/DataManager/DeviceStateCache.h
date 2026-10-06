#pragma once
// ============================================================================
// DeviceStateCache.h — 设备状态缓存（DataStore）
//
// 实现 IDeviceStateSink。原子存储各设备状态，供 UI/Service 轮询读取。
// ============================================================================

#include "IDeviceStateSink.h"
#include <mutex>
#include <unordered_map>

namespace Scanner::data {

class DeviceStateCache : public IDeviceStateSink {
public:
    DeviceStateCache();

    // IDeviceStateSink
    Result pushState(const DeviceStateInfo& state) override;
    DeviceStateInfo getState(const std::string& deviceId) const override;

    // 便捷查询
    double getTemperature(const std::string& deviceId) const;
    double getFps(const std::string& deviceId) const;
    int getTotalDroppedFrames() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, DeviceStateInfo> cache_;
};

} // namespace Scanner::data
