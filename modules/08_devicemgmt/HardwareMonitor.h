#pragma once
// ============================================================================
// HardwareMonitor.h — 硬件状态监控（HAL 层）
//
// 周期性采集温度/帧率，推入 DeviceStateCache + 发布 EventBus 事件。
// 纯读取，不做业务判断（如温度阈值）。
// ============================================================================

#include "base/types.h"
#include <atomic>
#include <thread>
#include <functional>

namespace Scanner::data { class DeviceStateCache; }
namespace Scanner::infra { class EventBus; }
namespace Scanner::hal  { class IMCU; class IScannerCamera; }

namespace Scanner::device {

class HardwareMonitor {
public:
    HardwareMonitor();
    ~HardwareMonitor();

    HardwareMonitor(const HardwareMonitor&) = delete;
    HardwareMonitor& operator=(const HardwareMonitor&) = delete;

    // 注入依赖
    void setDeviceStateCache(data::DeviceStateCache* cache) { stateCache_ = cache; }
    void setEventBus(infra::EventBus* bus) { eventBus_ = bus; }
    void setMCU(hal::IMCU* mcu) { mcu_ = mcu; }
    void setCamera(hal::IScannerCamera* cam) { camera_ = cam; }

    // 设置帧率统计回调（由 Pipeline 注入）
    void setFrameCounter(std::function<int()> getFps) { getFps_ = std::move(getFps); }

    // 启动/停止监控
    void start(int intervalMs = 1000);
    void stop();

    bool isRunning() const { return running_; }

private:
    void monitorLoop();

    std::atomic<bool> running_{false};
    std::thread thread_;
    int intervalMs_ = 1000;

    data::DeviceStateCache* stateCache_ = nullptr;
    infra::EventBus* eventBus_ = nullptr;
    hal::IMCU* mcu_ = nullptr;
    hal::IScannerCamera* camera_ = nullptr;
    std::function<int()> getFps_;
};

} // namespace Scanner::device
