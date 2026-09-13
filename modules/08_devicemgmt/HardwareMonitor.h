#pragma once
// ============================================================================
// HardwareMonitor.h — 硬件状态监控（HAL 层；H-T16 升级版）
//
// 周期巡检纯读取，不做业务判断（温度阈值等判定归 10-PerfMonitor）：
//   - 相机行照旧：isOpen/state/温度/fps（现状行为保持）+ droppedFrames 累计
//   - MCU 温度改注入（H-T16 删 IMCU 轮询）：DeviceManager 互斥快照
//     getLastTemperatures 直传，不再持有 IMCU 指针、不发任何命令
//   - 心跳经注入回调上报（R2-A1 不发命令）：超时判定与发 Fault 归注入方
//   - HealthMetrics 快照出口：自检结果 + 帧率三件套回填（供 app 适配器喂
//     10-PerfMonitor——08 不链 10，IHealthProvider 适配在 app 侧）
// 依赖全注入零直连：指针（Cache/Bus/相机/SelfCheck，可空）+ std::function。
// ============================================================================

#include "base/types.h"
#include "serial/McuFrame.h"   // serial::TempFrame（setLastTemps 注入口类型）
#include "IDeviceStateSink.h"  // 解 06 实现类耦合（挂账清账 2026-09-01）：
                               // 只用 pushState（虚接口），不再 include DeviceStateCache

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

namespace Scanner::infra { class EventBus; }
namespace Scanner::hal  { class IScannerCamera; }

namespace Scanner::device {

class SelfCheckCollector;

class HardwareMonitor {
public:
    HardwareMonitor();
    ~HardwareMonitor();

    HardwareMonitor(const HardwareMonitor&) = delete;
    HardwareMonitor& operator=(const HardwareMonitor&) = delete;

    // —— 注入依赖（装配期调用；巡检线程起后不再改）——
    // 状态落地口换契约接口（DeviceStateCache* 隐式上转 IDeviceStateSink*，
    // 调用方零改动）；虚调用免链接符号汇聚
    void setDeviceStateSink(data::IDeviceStateSink* sink) { stateSink_ = sink; }
    void setEventBus(infra::EventBus* bus) { eventBus_ = bus; }
    void setCamera(hal::IScannerCamera* cam) { camera_ = cam; }

    // MCU 温度快照注入（删 IMCU 轮询的替代口）：DeviceManager::getLastTemperatures
    // 直传。口径：有帧（channels>0）即在线——按位写 MCU_T0..T3 行 + 首路同步写
    // MCU 行（MainWindow.cpp 现读法兼容）；注入源空则不写任何 MCU 行（垫片行为延续）
    void setLastTemps(std::function<serial::TempFrame()> get) { getLastTemps_ = std::move(get); }

    // 心跳检查回调（R2-A1 巡检不发命令）：DeviceManager 注入（内部判超时发
    // Fault）；巡检线程每周期只调一次，不关心结果
    void setHeartbeatCheck(std::function<void()> check) { heartbeatCheck_ = std::move(check); }

    // —— 帧率三件套（Pipeline 注入）——
    // 采集 fps（保留旧签名）：写 Camera 行 fps（现状行为保持）+ 快照 captureFps
    void setFrameCounter(std::function<int()> getFps) { getFps_ = std::move(getFps); }
    // 口径：三件套只进 HealthMetrics 快照（processFps/dropRate）；丢帧值同步
    // 累计进 Camera 行 droppedFrames（注入方给累计计数，监控透传不换算）
    void setProcessCounter(std::function<double()> getFps) { getProcessFps_ = std::move(getFps); }
    void setDropCounter(std::function<double()> getCount) { getDropCount_ = std::move(getCount); }

    // 上位机自检采集（cpu/mem/disk…；可空——空则快照 cpu 侧字段保持 -1 缺省）
    void setSelfCheck(SelfCheckCollector* sc) { selfCheck_ = sc; }

    // 启动/停止监控
    void start(int intervalMs = 1000);
    void stop();

    bool isRunning() const { return running_; }

    // HealthMetrics 快照（mutex 保护小缓存，巡检每周期刷新）：自检结果 + 帧率
    // 三件套回填——供 app 适配器喂 10-PerfMonitor
    Scanner::HealthMetrics snapshot() const;

private:
    void monitorLoop();

    std::atomic<bool> running_{false};
    std::thread thread_;
    int intervalMs_ = 1000;

    data::IDeviceStateSink* stateSink_ = nullptr;
    infra::EventBus* eventBus_ = nullptr;
    hal::IScannerCamera* camera_ = nullptr;
    SelfCheckCollector* selfCheck_ = nullptr;

    std::function<serial::TempFrame()> getLastTemps_;
    std::function<void()> heartbeatCheck_;
    std::function<int()> getFps_;
    std::function<double()> getProcessFps_;
    std::function<double()> getDropCount_;

    mutable std::mutex snapMutex_;              // lastMetrics_ 巡检写 / snapshot() 读
    Scanner::HealthMetrics lastMetrics_;
};

} // namespace Scanner::device
