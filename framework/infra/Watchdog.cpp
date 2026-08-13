#include "Watchdog.h"
#include <spdlog/spdlog.h>

namespace Scanner::infra {

Watchdog::Watchdog(int checkIntervalMs) : checkIntervalMs_(checkIntervalMs) {}

Watchdog::~Watchdog() { stop(); }

void Watchdog::registerStage(const std::string& name, int timeoutMs) {
    std::lock_guard lock(mutex_);
    StageInfo info;
    info.timeoutMs = timeoutMs;
    info.lastHeartbeat = std::chrono::steady_clock::now();
    stages_[name] = info;
    spdlog::debug("[Watchdog] 注册 Stage: {} (超时 {}ms)", name, timeoutMs);
}

void Watchdog::heartbeat(const std::string& name) {
    std::lock_guard lock(mutex_);
    auto it = stages_.find(name);
    if (it != stages_.end()) {
        it->second.lastHeartbeat = std::chrono::steady_clock::now();
    }
}

void Watchdog::unregisterStage(const std::string& name) {
    std::lock_guard lock(mutex_);
    stages_.erase(name);
}

void Watchdog::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread(&Watchdog::monitorLoop, this);
    spdlog::info("[Watchdog] 监控启动, 间隔 {}ms", checkIntervalMs_);
}

void Watchdog::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    spdlog::info("[Watchdog] 监控停止");
}

void Watchdog::monitorLoop() {
    while (running_) {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(mutex_);
            for (auto& [name, info] : stages_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - info.lastHeartbeat).count();
                if (elapsed > info.timeoutMs) {
                    spdlog::warn("[Watchdog] Stage '{}' 超时: {}ms > {}ms",
                                 name, elapsed, info.timeoutMs);
                    if (callback_) callback_(name);
                    // 重置心跳避免重复告警
                    info.lastHeartbeat = now;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(checkIntervalMs_));
    }
}

} // namespace Scanner::infra
