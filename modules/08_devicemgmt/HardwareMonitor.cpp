// ============================================================================
// HardwareMonitor.cpp — 硬件状态监控实现
// ============================================================================

#include "HardwareMonitor.h"
#include "DeviceStateCache.h"
#include "base/EventBus.h"
#include "IMCU.h"
#include "IScannerCamera.h"
#include <spdlog/spdlog.h>
#include <chrono>

namespace Scanner::device {

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
        auto start = std::chrono::steady_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // 1. MCU（TODO(T16): 接新 IMCU 后恢复 MCU 行（心跳/状态/温度）——垫片期
        //    不推 MCU 状态行：无真实数据源，推 0 温/借相机判态属错误状态；
        //    读方（DeviceStateCache.getState）对未写设备返回默认 Offline/0.0，兼容）

        // 2. 相机温度 + 帧率
        if (camera_) {
            double camTemp = camera_->getTemperature();
            double fps = getFps_ ? getFps_() : 0;

            if (stateCache_) {
                data::DeviceStateInfo info;
                info.deviceId = "Camera";
                info.deviceType = "ScannerCamera";
                info.state = camera_->isOpen() ? DeviceState::Connected : DeviceState::Offline;
                info.temperature = camTemp;
                info.fps = fps;
                info.timestamp = ts;
                stateCache_->pushState(info);
            }
        }

        // 3. 等待下一个周期
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep = std::chrono::milliseconds(intervalMs_) -
                     std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        if (sleep.count() > 0) {
            std::this_thread::sleep_for(sleep);
        }
    }
}

} // namespace Scanner::device
