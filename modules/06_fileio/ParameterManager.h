#pragma once
// ============================================================================
// ParameterManager.h — 参数集中管理（Service 层）
// ============================================================================

#include "base/types.h"
#include <mutex>
#include <unordered_map>
#include <string>
#include <variant>
#include <optional>

namespace Scanner::service {

using ParamValue = std::variant<int, double, bool, std::string>;

class ParameterManager {
public:
    ParameterManager();

    void set(const std::string& key, ParamValue value);
    std::optional<ParamValue> get(const std::string& key) const;

    int getInt(const std::string& key, int defaultVal = 0) const;
    double getDouble(const std::string& key, double defaultVal = 0.0) const;
    bool getBool(const std::string& key, bool defaultVal = false) const;
    std::string getString(const std::string& key, const std::string& defaultVal = "") const;

    bool has(const std::string& key) const;
    void remove(const std::string& key);
    void clear();

    // 预设参数组
    void setCameraDefaults();
    void setScanDefaults();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ParamValue> params_;
};

} // namespace Scanner::service
