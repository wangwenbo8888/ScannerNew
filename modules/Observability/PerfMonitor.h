#pragma once
// ============================================================================
// PerfMonitor.h — 健康指标消费端（poll 拉取式；采集归 08，设计方案 §6）
//
// 与 §6.1 的偏差（控制器已批准）：不订阅 EventBus ping——EventBus 同步分发
// 持总线锁，订阅回调内 publish HealthReport 会重入死锁；改为显式 poll()
// 消费端（拉取+判定+发布），08 巡检线程/app 定时器定期调用。
//
//   poll() = provider 拉快照 → 阈值判定 → 双级告警：
//     · 预警级（CPU 过热/内存持续/磁盘预警/帧率持续）→ HealthReport(param1=1)
//     · 故障级（磁盘停写保护）→ FaultHandler reportFault(Error) 完整故障链
//   指标 -1=未知跳过；阈值 -1=禁用该规则（§6.2）；持续类规则连续越限满
//   N 秒才降级，恢复清锚；降级进入/全绿恢复按边沿发事件（防刷屏）。
// ============================================================================
#include "base/types.h"
#include <functional>
#include <memory>

namespace Scanner::infra { class EventBus; }
namespace Scanner::service { class FaultHandler; }

namespace Scanner::service {

struct IHealthProvider {                       // 08 侧（或测试假源）实现并注入
    virtual ~IHealthProvider() = default;
    virtual HealthMetrics snapshot() const = 0;
};

struct PerfThresholds {                        // -1=禁用该规则（设计方案 §6.2）
    double cpuTempWarnC   = 90.0;
    double memPercentWarn = 85.0;  int memSustainSec  = 10;
    double diskFreeWarnGB = 10.0;  double diskFreeStopGB = 1.0;
    double fpsTarget      = 100.0; double fpsRatioWarn = 0.8; int fpsSustainSec = 3;
};

class PerfMonitor {
public:
    PerfMonitor(infra::EventBus* bus, FaultHandler* faults, int monitorSourceId);

    void setProvider(std::shared_ptr<IHealthProvider> p);
    void setThresholds(const PerfThresholds& t);
    void setClock(std::function<int64_t()> nowMs);      // 测试假钟
    void poll();                                         // 拉取+判定+发布（08 巡检线程/app 定时器调）
    bool degraded() const;                               // 最近一轮是否处于降级

private:
    struct Sustained { int64_t sinceMs = -1; };          // 持续类规则的计时锚

    // 持续类规则公共判定：越限记锚（首次），满 sustainSec 秒返回真；不越限清锚
    static bool sustainedViolated(bool violating, Sustained& s, int sustainSec, int64_t now);
    void publishHealth(int64_t now, int64_t code);       // HealthReport(param1=code)

    infra::EventBus* bus_ = nullptr;
    FaultHandler* faults_ = nullptr;
    int monitorSourceId_ = 0;
    PerfThresholds thresholds_{};
    std::shared_ptr<IHealthProvider> provider_;
    std::function<int64_t()> nowFn_;                     // 默认 steady_clock 毫秒

    Sustained mem_;                                      // 内存持续越限锚
    Sustained fps_;                                      // 帧率持续越限锚
    bool degraded_ = false;                              // 边沿触发基准（上轮状态）
};

} // namespace Scanner::service
