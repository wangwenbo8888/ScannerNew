// ============================================================================
// HardwareMonitor.cpp — 硬件状态监控实现
// ============================================================================

#include "HardwareMonitor.h"
#include "DeviceStateCache.h"
#include "base/EventBus.h"
#include "IScannerCamera.h"
#include "SelfCheckCollector.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <string>

namespace Scanner::device {

namespace {
// steady 域毫秒（与 SelfCheckCollector::steadyNowMs 同域——collect 时基）
int64_t steadyNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

HardwareMonitor::HardwareMonitor() {}

HardwareMonitor::~HardwareMonitor() {
    stop();
}

void HardwareMonitor::start(int intervalMs) {
    if (running_) return;
    intervalMs_ = (intervalMs < 100) ? 100 : intervalMs;
    running_ = true;
    thread_ = std::thread(&HardwareMonitor::monitorLoop, this);
    spdlog::info("[HardwareMonitor] 监控启动, 间隔 {}ms", intervalMs_);
}

void HardwareMonitor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    spdlog::info("[HardwareMonitor] 监控停止");
}

void HardwareMonitor::monitorLoop() {
    while (running_) {
        auto tick = std::chrono::steady_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // 1. 心跳检查（注入回调；超时判定与发 Fault 归 DeviceManager——巡检只调）
        if (heartbeatCheck_) heartbeatCheck_();

        // 2. MCU 温度行（改注入：有帧即在线；channels 按位写 MCU_T0..T3，首路
        //    同步写 MCU 行兼容 MainWindow 读法；注入源空不写——垫片行为延续）
        if (getLastTemps_ && stateCache_) {
            serial::TempFrame tf = getLastTemps_();
            const int n = std::min<int>(tf.channels, 4);
            for (int i = 0; i < n; ++i) {
                data::DeviceStateInfo info;
                info.deviceId = "MCU_T" + std::to_string(i);
                info.deviceType = "MCU";
                info.state = DeviceState::Connected;
                info.temperature = tf.celsius[i];
                info.timestamp = ts;
                stateCache_->pushState(info);
            }
            if (n > 0) {
                data::DeviceStateInfo first;
                first.deviceId = "MCU";
                first.deviceType = "MCU";
                first.state = DeviceState::Connected;
                first.temperature = tf.celsius[0];
                first.timestamp = ts;
                stateCache_->pushState(first);
            }
        }

        // 3. 相机行照旧（isOpen/state/温度/fps）+ droppedFrames 累计
        if (camera_ && stateCache_) {
            data::DeviceStateInfo info;
            info.deviceId = "Camera";
            info.deviceType = "ScannerCamera";
            info.state = camera_->isOpen() ? DeviceState::Connected : DeviceState::Offline;
            info.temperature = camera_->getTemperature();
            info.fps = getFps_ ? getFps_() : 0;
            info.droppedFrames = static_cast<int>(getDropCount_ ? getDropCount_() : 0.0);
            info.timestamp = ts;
            stateCache_->pushState(info);
        }

        // 4. HealthMetrics 快照组装（自检结果 + 三件套回填；供 app 喂 10-PerfMonitor）
        {
            HealthMetrics m;
            if (selfCheck_) m = selfCheck_->collect(steadyNowMs());
            m.captureFps = getFps_        ? static_cast<double>(getFps_()) : -1.0;
            m.processFps = getProcessFps_ ? getProcessFps_()               : -1.0;
            m.dropRate   = getDropCount_  ? getDropCount_()                : -1.0;
            m.timestampMs = static_cast<int64_t>(ts);
            std::lock_guard<std::mutex> lock(snapMutex_);
            lastMetrics_ = m;
        }

        // 5. 等待下一个周期
        auto elapsed = std::chrono::steady_clock::now() - tick;
        auto sleep = std::chrono::milliseconds(intervalMs_) -
                     std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        if (sleep.count() > 0) {
            std::this_thread::sleep_for(sleep);
        }
    }
}

HealthMetrics HardwareMonitor::snapshot() const {
    std::lock_guard<std::mutex> lock(snapMutex_);
    return lastMetrics_;
}

} // namespace Scanner::device
