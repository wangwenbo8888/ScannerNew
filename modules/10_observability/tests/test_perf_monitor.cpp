// ============================================================================
// test_perf_monitor.cpp — PerfMonitor 健康指标消费端单测（P4-T11，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §6
//   · 数据通路偏差（控制器已批准）：§6.1「订阅 HealthReport ping + 拉取」简化为
//     显式 poll() 模型——EventBus 同步分发持总线锁，订阅回调内 publish
//     HealthReport 会重入死锁；08 本就每秒巡检，由其（或 app 定时器）调 poll()
//   · poll() = 拉 provider 快照 → 判阈值 → 发 EventBus HealthReport(param1:
//     0=正常 1=降级) / FaultHandler reportFault（停写保护 Error 级）
//   · 阈值规则（§6.2，-1=禁用该规则；指标 -1=未知跳过）：
//     CPU>90℃ 瞬时预警 / 内存>85% 持续 10s / 磁盘<10GB 预警、<1GB 停写保护 /
//     采集帧率<目标×80% 持续 3s
//   · 持续类规则：越限记锚点，连续越限满 N 秒才降级；恢复清锚
//   · 降级进入发 param1=1，全绿恢复发 param1=0（边沿触发防刷屏）
//
// 用真 EventBus + 真 FaultHandler（不挂 SM——FaultHandler 容忍空 SM，
// 见 FaultHandler.cpp 故障链 stateMachine_ 判空）+ FakeProvider 假源 +
// 构造注入假钟（std::function<int64_t()>）。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include "PerfMonitor.h"
#include "FaultHandler.h"
#include "base/EventBus.h"

using Scanner::service::PerfMonitor;
using Scanner::service::IHealthProvider;
using Scanner::service::PerfThresholds;
using Scanner::service::FaultHandler;
using Scanner::infra::EventBus;
using Scanner::EventType;
using Scanner::Event;
using Scanner::FaultSeverity;
using Scanner::HealthMetrics;

namespace {

// 事件计数小助手：订阅指定类型，计数并记录最近一次 param1/param2
struct EventSpy {
    std::atomic<int> count{0};
    std::atomic<int64_t> param1{-1};
    std::atomic<int64_t> param2{-1};
    void subscribe(EventBus& bus, EventType type) {
        bus.subscribe(type, [this](const Event& e) {
            param1.store(e.param1);
            param2.store(e.param2);
            count.fetch_add(1);
        });
    }
};

// 假源：快照可任意设置（08 侧 IHealthProvider 的测试替身）
class FakeProvider : public IHealthProvider {
public:
    HealthMetrics metrics;
    HealthMetrics snapshot() const override { return metrics; }
};

// 全绿基准快照（其余字段保持 -1=未知）
HealthMetrics greenMetrics() {
    HealthMetrics m;
    m.cpuTempC = 50;
    m.memPercent = 40;
    m.diskFreeGB = 100;
    m.captureFps = 95;
    return m;
}

// 装置：真 EventBus + 真 FaultHandler（不挂 SM）+ 假源 + 假钟
// 成员声明序即构造序：bus → faults → monitorId → pm（依赖前置）
struct PerfFixture {
    EventBus bus;
    FaultHandler faults;
    int monitorId;
    std::shared_ptr<FakeProvider> provider = std::make_shared<FakeProvider>();
    int64_t now = 0;                       // 假钟（毫秒，测试直接拨）
    PerfMonitor pm;
    EventSpy health;
    EventSpy faultOccurred;

    PerfFixture()
        : faults(&bus),
          monitorId(faults.registerSource("Monitor")),
          pm(&bus, &faults, monitorId) {
        health.subscribe(bus, EventType::HealthReport);
        faultOccurred.subscribe(bus, EventType::FaultOccurred);
        pm.setProvider(provider);
        pm.setClock([this] { return now; });
    }
};

} // namespace

TEST(PM, CpuTempWarnDegraded) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.cpuTempC = 92;          // > 90 预警线
    f.pm.poll();

    EXPECT_EQ(f.health.count.load(), 1);        // HealthReport(param1=1)
    EXPECT_EQ(f.health.param1.load(), 1);
    EXPECT_TRUE(f.pm.degraded());
    EXPECT_TRUE(f.faults.activeFaults().empty());  // 预警级不进故障档案
}

TEST(PM, MemSustain10s) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.memPercent = 90;        // > 85 持续 10s 规则两侧

    for (int64_t t = 0; t <= 9500; t += 500) {  // 连续越限 9.5s：不报
        f.now = t;
        f.pm.poll();
    }
    EXPECT_EQ(f.health.count.load(), 0);
    EXPECT_FALSE(f.pm.degraded());

    f.now = 11000;                              // 满 10s：报降级
    f.pm.poll();
    EXPECT_EQ(f.health.count.load(), 1);
    EXPECT_EQ(f.health.param1.load(), 1);
    EXPECT_TRUE(f.pm.degraded());
}

TEST(PM, DiskStopGoesFault) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.diskFreeGB = 0.5;       // < 1.0 停写线
    f.pm.poll();

    const auto faults = f.faults.activeFaults();
    ASSERT_EQ(faults.size(), 1u);               // 经 reportFault 进档案
    EXPECT_EQ(faults[0].severity, FaultSeverity::Error);
    EXPECT_EQ(faults[0].sourceId, f.monitorId);
    EXPECT_EQ(f.faultOccurred.count.load(), 1); // FaultOccurred 事件照发
    EXPECT_TRUE(f.pm.degraded());
}

TEST(PM, DiskWarnOnlyDegraded) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.diskFreeGB = 5;         // <10 预警线、>1 停写线
    f.pm.poll();

    EXPECT_EQ(f.health.count.load(), 1);        // 只降级事件
    EXPECT_EQ(f.health.param1.load(), 1);
    EXPECT_TRUE(f.faults.activeFaults().empty());  // 无 Fault
    EXPECT_TRUE(f.pm.degraded());
}

TEST(PM, CaptureFpsRatio) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.captureFps = 70;        // < 100×0.8=80 持续 3s 两侧

    f.now = 0;
    f.pm.poll();                                // 首次越限记锚
    f.now = 2500;
    f.pm.poll();                                // 2.5s：不报
    EXPECT_EQ(f.health.count.load(), 0);
    EXPECT_FALSE(f.pm.degraded());

    f.now = 3500;
    f.pm.poll();                                // 3.5s ≥ 3s：报降级
    EXPECT_EQ(f.health.count.load(), 1);
    EXPECT_EQ(f.health.param1.load(), 1);
    EXPECT_TRUE(f.pm.degraded());
}

TEST(PM, RecoveryReportsNormal) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.cpuTempC = 92;
    f.pm.poll();
    ASSERT_EQ(f.health.count.load(), 1);
    ASSERT_EQ(f.health.param1.load(), 1);

    f.provider->metrics.cpuTempC = 50;          // 恢复
    f.pm.poll();
    EXPECT_EQ(f.health.count.load(), 2);        // 恢复通知
    EXPECT_EQ(f.health.param1.load(), 0);
    EXPECT_FALSE(f.pm.degraded());
}

TEST(PM, UnknownMetricsIgnored) {
    PerfFixture f;
    f.provider->metrics = HealthMetrics{};      // 全 -1（未知）
    f.pm.poll();

    EXPECT_EQ(f.health.count.load(), 0);
    EXPECT_EQ(f.faultOccurred.count.load(), 0);
    EXPECT_TRUE(f.faults.activeFaults().empty());
    EXPECT_FALSE(f.pm.degraded());
}

TEST(PM, SustainAnchorResetsOnRecovery) {
    PerfFixture f;
    f.provider->metrics = greenMetrics();
    f.provider->metrics.memPercent = 90;

    f.now = 0;                                  // 记锚 t=0
    f.pm.poll();
    f.now = 5000;
    f.pm.poll();                                // 越限 5s 未满

    f.provider->metrics.memPercent = 40;        // 恢复：锚点应清
    f.now = 6000;
    f.pm.poll();

    f.provider->metrics.memPercent = 90;        // 再次越限：锚点重记
    f.now = 6100;
    f.pm.poll();
    f.now = 12000;                              // 新锚仅 5.9s：不报（旧锚不清则此处误报）
    f.pm.poll();
    EXPECT_EQ(f.health.count.load(), 0);

    f.now = 17000;                              // 新锚满 10.9s：报
    f.pm.poll();
    EXPECT_EQ(f.health.count.load(), 1);
    EXPECT_EQ(f.health.param1.load(), 1);
}

TEST(PM, DisabledRuleIgnored) {
    PerfFixture f;
    PerfThresholds th;                          // -1=禁用该规则（§6.2）
    th.cpuTempWarnC = -1;
    f.pm.setThresholds(th);

    f.provider->metrics = greenMetrics();
    f.provider->metrics.cpuTempC = 92;          // 规则已禁用：越限值不触发
    f.pm.poll();

    EXPECT_EQ(f.health.count.load(), 0);
    EXPECT_FALSE(f.pm.degraded());
}
