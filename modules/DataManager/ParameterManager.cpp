#include "ParameterManager.h"
#include <spdlog/spdlog.h>

namespace Scanner::service {

ParameterManager::ParameterManager() {
    setCameraDefaults();
    setScanDefaults();
}

void ParameterManager::set(const std::string& key, ParamValue value) {
    std::lock_guard lock(mutex_);
    params_[key] = std::move(value);
}

std::optional<ParamValue> ParameterManager::get(const std::string& key) const {
    std::lock_guard lock(mutex_);
    auto it = params_.find(key);
    if (it != params_.end()) return it->second;
    return std::nullopt;
}

int ParameterManager::getInt(const std::string& key, int defaultVal) const {
    auto v = get(key);
    if (!v) return defaultVal;
    if (auto p = std::get_if<int>(&*v)) return *p;
    if (auto p = std::get_if<double>(&*v)) return static_cast<int>(*p);
    return defaultVal;
}

double ParameterManager::getDouble(const std::string& key, double defaultVal) const {
    auto v = get(key);
    if (!v) return defaultVal;
    if (auto p = std::get_if<double>(&*v)) return *p;
    if (auto p = std::get_if<int>(&*v)) return static_cast<double>(*p);
    return defaultVal;
}

bool ParameterManager::getBool(const std::string& key, bool defaultVal) const {
    auto v = get(key);
    if (!v) return defaultVal;
    if (auto p = std::get_if<bool>(&*v)) return *p;
    return defaultVal;
}

std::string ParameterManager::getString(const std::string& key, const std::string& defaultVal) const {
    auto v = get(key);
    if (!v) return defaultVal;
    if (auto p = std::get_if<std::string>(&*v)) return *p;
    return defaultVal;
}

bool ParameterManager::has(const std::string& key) const {
    std::lock_guard lock(mutex_);
    return params_.count(key) > 0;
}

void ParameterManager::remove(const std::string& key) {
    std::lock_guard lock(mutex_);
    params_.erase(key);
}

void ParameterManager::clear() {
    std::lock_guard lock(mutex_);
    params_.clear();
}

void ParameterManager::setCameraDefaults() {
    std::lock_guard lock(mutex_);
    params_["camera.exposure_ms"] = 10.0;
    params_["camera.gain_db"] = 0.0;
    params_["camera.auto_exposure"] = false;
    params_["camera.width"] = 2048;
    params_["camera.height"] = 1536;
}

void ParameterManager::setScanDefaults() {
    std::lock_guard lock(mutex_);
    params_["scan.mode"] = static_cast<int>(0);  // MarkerOnly
    params_["scan.laser_freq"] = 50;
    params_["scan.laser_power"] = 60;
    params_["scan.fill_light"] = 40;
    params_["scan.trigger_mode"] = 1;
    params_["scan.target_fps"] = 180;
    params_["scan voxel_size_mm"] = 0.5;
    params_["postprocess.smooth_iterations"] = 5;
}

} // namespace Scanner::service
