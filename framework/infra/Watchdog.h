#pragma once
// ============================================================================
// Watchdog.h — Stage 心跳监控（infra 层）
//
// 独立于被监控对象。Workflow 注册 Stage 心跳，Watchdog 周期检测超时。
// ============================================================================

#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>

namespace Scanner::infra {

class Watchdog {
public:
    using TimeoutCallback = std::function<void(const std::string& stageName)>;

    explicit Watchdog(int checkIntervalMs = 1000);
    ~Watchdog();

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    // 注册 Stage（名称 + 超时阈值）
    void registerStage(const std::string& name, int timeoutMs);

    // Stage 心跳（每次处理完一帧调用）
    void heartbeat(const std::string& name);

    // 注销 Stage
    void unregisterStage(const std::string& name);

    // 超时回调
    void onTimeout(TimeoutCallback cb) { callback_ = std::move(cb); }

    // 启动/停止
    void start();
    void stop();
    bool isRunning() const { return running_.load(); }

private:
    void monitorLoop();

    struct StageInfo {
        int timeoutMs = 0;
        std::chrono::steady_clock::time_point lastHeartbeat;
    };

    int checkIntervalMs_;
    std::unordered_map<std::string, StageInfo> stages_;
    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    TimeoutCallback callback_;
};

} // namespace Scanner::infra
