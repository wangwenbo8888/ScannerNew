#include "DeviceStateCache.h"

namespace Scanner::data {

DeviceStateCache::DeviceStateCache() {}

Result DeviceStateCache::pushState(const DeviceStateInfo& state) {
    std::lock_guard lock(mutex_);
    cache_[state.deviceId] = state;
    return Result::ok();
}

DeviceStateInfo DeviceStateCache::getState(const std::string& deviceId) const {
    std::lock_guard lock(mutex_);
    auto it = cache_.find(deviceId);
    if (it != cache_.end()) return it->second;
    return DeviceStateInfo{};
}

double DeviceStateCache::getTemperature(const std::string& deviceId) const {
    return getState(deviceId).temperature;
}

double DeviceStateCache::getFps(const std::string& deviceId) const {
    return getState(deviceId).fps;
}

int DeviceStateCache::getTotalDroppedFrames() const {
    std::lock_guard lock(mutex_);
    int total = 0;
    for (const auto& [id, info] : cache_) {
        total += info.droppedFrames;
    }
    return total;
}

} // namespace Scanner::data
