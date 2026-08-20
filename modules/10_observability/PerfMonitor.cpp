#include "PerfMonitor.h"
#include "FaultHandler.h"
#include "base/EventBus.h"
#include <chrono>
#include <spdlog/spdlog.h>

namespace Scanner::service {

namespace {
constexpr int64_t kMsPerSec = 1000;
} // namespace

PerfMonitor::PerfMonitor(infra::EventBus* bus, FaultHandler* faults, int monitorSourceId)
    : bus_(bus), faults_(faults), monitorSourceId_(monitorSourceId),
      nowFn_([] {
          return std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
      }) {}

void PerfMonitor::setProvider(std::shared_ptr<IHealthProvider> p) {
    provider_ = std::move(p);
}

void PerfMonitor::setThresholds(const PerfThresholds& t) {
    thresholds_ = t;
}

void PerfMonitor::setClock(std::function<int64_t()> nowMs) {
    nowFn_ = std::move(nowMs);
}

bool PerfMonitor::degraded() const {
    return degraded_;
}

bool PerfMonitor::sustainedViolated(bool violating, Sustained& s, int sustainSec, int64_t now) {
    if (violating) {
        if (s.sinceMs < 0) s.sinceMs = now;              // 首次越限记锚
        return now - s.sinceMs >= static_cast<int64_t>(sustainSec) * kMsPerSec;
    }
    s.sinceMs = -1;                                      // 恢复清锚（持续=连续越限）
    return false;
}

void PerfMonitor::publishHealth(int64_t now, int64_t code) {
    if (!bus_) return;
    Event evt;
    evt.type = EventType::HealthReport;
    evt.sourceId = static_cast<uint32_t>(monitorSourceId_);
    evt.timestamp = static_cast<TimestampMs>(now);
    evt.param1 = code;                                   // 0=正常 1=降级摘要码
    bus_->publish(evt);                                  // poll 内发布，非总线回调——无重入
}

void PerfMonitor::poll() {
    if (!provider_) return;
    const HealthMetrics m = provider_->snapshot();
    const int64_t now = nowFn_ ? nowFn_() : 0;
    const PerfThresholds& th = thresholds_;

    bool warn = false;        // 预警级越限（→ HealthReport param1=1）
    bool stopFault = false;   // 停写保护（→ FaultHandler Error 完整故障链）

    // 规则1：CPU 过热预警（瞬时；-1=禁用，指标<0=未知跳过）
    if (th.cpuTempWarnC >= 0 && m.cpuTempC >= 0 && m.cpuTempC > th.cpuTempWarnC) {
        warn = true;
    }

    // 规则2：内存水位 >85% 持续 10s（持续类）
    const bool memOver = th.memPercentWarn >= 0 && m.memPercent >= 0 &&
                         m.memPercent > th.memPercentWarn;
    if (sustainedViolated(memOver, mem_, th.memSustainSec, now)) {
        warn = true;
    }

    // 规则3：磁盘剩余两档——<1GB 停写保护（故障级）；<10GB 预警（降级级）
    if (m.diskFreeGB >= 0) {
        if (th.diskFreeStopGB >= 0 && m.diskFreeGB < th.diskFreeStopGB) {
            stopFault = true;
        } else if (th.diskFreeWarnGB >= 0 && m.diskFreeGB < th.diskFreeWarnGB) {
            warn = true;
        }
    }

    // 规则4：采集帧率 < 目标×比例 持续 3s（持续类）
    const bool fpsOver = th.fpsTarget >= 0 && th.fpsRatioWarn >= 0 && m.captureFps >= 0 &&
                         m.captureFps < th.fpsTarget * th.fpsRatioWarn;
    if (sustainedViolated(fpsOver, fps_, th.fpsSustainSec, now)) {
        warn = true;
    }

    const bool nowDegraded = warn || stopFault;

    // 停写保护 → reportFault 直调口（完整故障链；1s 聚合防风暴，poll 周期直调安全）
    if (stopFault && faults_) {
        faults_->reportFault(monitorSourceId_, FaultSeverity::Error,
                             "disk free " + std::to_string(m.diskFreeGB) +
                                 "GB below stop threshold " +
                                 std::to_string(th.diskFreeStopGB) + "GB (write-stop)");
    }

    // HealthReport 边沿触发：进入降级（含预警级越限）发 1；全绿恢复发 0
    if (warn && !degraded_) {
        spdlog::warn("[PerfMonitor] 性能降级：cpuTemp={}C mem={}%(anchor {}ms) disk={}GB captureFps={}",
                     m.cpuTempC, m.memPercent,
                     mem_.sinceMs >= 0 ? now - mem_.sinceMs : -1,
                     m.diskFreeGB, m.captureFps);
        publishHealth(now, 1);
    } else if (!nowDegraded && degraded_) {
        spdlog::info("[PerfMonitor] 性能恢复");
        publishHealth(now, 0);
    }

    degraded_ = nowDegraded;
}

} // namespace Scanner::service
