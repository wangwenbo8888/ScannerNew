// ============================================================================
// test_fault_recovery.cpp — S7 恢复边界用例（P2-T8）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §4.4/§4.5
//   · 恢复续作语义：S7 出口唯 SelfCheckPassed→lastHealthy；恢复目标=进 S7 前
//     的健康态（含 S1 边界：自检中出故障，恢复回 S1 重走初始化）
//   · S7 内 DeviceDisconnected 无边保持（自检含重连）；全 7 态断连矩阵：
//     S2–S5→S1 ok，S1（本就是 S1）/S6（免疫）/S7（保持）fail
//   · FaultHandler 聚合窗口过期：同源同级 1s 内只计数不发事件；窗口过后
//     第三次上报重置 count=1 并再发事件（真实 steady_clock，sleep 1.1s）
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include "FaultHandler.h"
#include "StateMachine.h"
#include "base/EventBus.h"

using Scanner::service::FaultHandler;
using Scanner::service::StateMachine;
using Scanner::service::SystemState;
using Scanner::infra::EventBus;
using Scanner::EventType;
using Scanner::Event;
using Scanner::FaultSeverity;
using Scanner::Result;

namespace {

// 全部 7 态（S1–S7）
const SystemState kAllStates[] = {
    SystemState::Init,           SystemState::Standby,       SystemState::Calibrating,
    SystemState::ScanMarker,     SystemState::ScanMarkerLaser,
    SystemState::PostProcessing, SystemState::FaultSelfCheck};

// 驱动新状态机到目标态（仅走矩阵内合法路径；S7 固定自 S2 进入 → lastHealthy=S2）
void driveTo(StateMachine& sm, SystemState target) {
    ASSERT_EQ(sm.getCurrentState(), SystemState::Init);
    if (target == SystemState::Init) return;
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    switch (target) {
        case SystemState::Standby:         return;
        case SystemState::Calibrating:
            ASSERT_TRUE(sm.transition(EventType::CalibStarted).success); break;
        case SystemState::ScanMarker:
            ASSERT_TRUE(sm.transition(EventType::ScanStarted, 0).success); break;
        case SystemState::ScanMarkerLaser:
            ASSERT_TRUE(sm.transition(EventType::ScanStarted, 1).success); break;
        case SystemState::PostProcessing:
            ASSERT_TRUE(sm.transition(EventType::PostProcessStarted).success); break;
        case SystemState::FaultSelfCheck:
            ASSERT_TRUE(sm.transition(EventType::FaultOccurred).success); break;
        default: FAIL() << "unreachable"; return;
    }
    ASSERT_EQ(sm.getCurrentState(), target);
}

// 事件计数小助手：订阅指定类型并计数
struct EventSpy {
    std::atomic<int> count{0};
    void subscribe(EventBus& bus, EventType type) {
        bus.subscribe(type, [this](const Event&) { count.fetch_add(1); });
    }
};

} // namespace

TEST(RC, S1FaultRecoversToS1) {
    // 恢复目标=S1 边界：S1（自检中）出故障 → S7（lastHealthy=S1）→ 恢复回 S1
    StateMachine sm;
    ASSERT_EQ(sm.getCurrentState(), SystemState::Init);
    ASSERT_TRUE(sm.transition(EventType::FaultOccurred).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
    ASSERT_TRUE(sm.transition(EventType::SelfCheckPassed).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
}

TEST(RC, S4FaultRecoversToS4) {
    // 扫描中故障恢复续作：S4 → S7（lastHealthy=S4）→ SelfCheckPassed 回 S4
    StateMachine sm;
    driveTo(sm, SystemState::ScanMarker);
    ASSERT_TRUE(sm.transition(EventType::FaultOccurred).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
    ASSERT_TRUE(sm.transition(EventType::SelfCheckPassed).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
}

TEST(RC, S7DisconnectStaysS7) {
    // §4.5：S7 内 DeviceDisconnected 无边 fail 保持——自检清单含重连，断连不丢 S7
    StateMachine sm;
    driveTo(sm, SystemState::FaultSelfCheck);
    EXPECT_FALSE(sm.transition(EventType::DeviceDisconnected).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
}

TEST(RC, DisconnectFromAllStates) {
    // 全 7 态逐一断连：S2–S5→S1 ok；S1（无边）/S6（免疫）/S7（保持）fail 且状态不变
    for (SystemState s : kAllStates) {
        StateMachine sm;
        driveTo(sm, s);
        Result r = sm.transition(EventType::DeviceDisconnected);
        const bool toInit = (s >= SystemState::Standby && s <= SystemState::ScanMarkerLaser);
        EXPECT_EQ(r.success, toInit) << "state=S" << static_cast<int>(s);
        EXPECT_EQ(sm.getCurrentState(), toInit ? SystemState::Init : s)
            << "state=S" << static_cast<int>(s);
    }
}

TEST(RC, AggregateWindowExpiry) {
    // FaultHandler 聚合窗口过期重置：同源同级 1s 窗口内 ++count 抑制（只 1 次事件）；
    // 窗口过期后第三次上报 → count 重置 1、再发事件（真实 steady_clock，sleep 1.1s）
    EventBus bus;
    FaultHandler fh(&bus);  // 不挂状态机：聚焦聚合档案与事件层
    const int cam = fh.registerSource("Camera");
    EventSpy spy;
    spy.subscribe(bus, EventType::FaultOccurred);

    fh.reportFault(cam, FaultSeverity::Warning, "temp high");
    fh.reportFault(cam, FaultSeverity::Warning, "temp high again");  // 窗口内抑制

    auto faults = fh.activeFaults();
    ASSERT_EQ(faults.size(), 1u);
    EXPECT_EQ(faults[0].count, 2u);      // 窗口内聚合计数
    EXPECT_EQ(spy.count.load(), 1);      // 只发 1 次事件

    std::this_thread::sleep_for(std::chrono::milliseconds(1100));  // 过 1s 聚合窗口

    fh.reportFault(cam, FaultSeverity::Warning, "temp high 3rd");  // 窗口过期

    faults = fh.activeFaults();
    ASSERT_EQ(faults.size(), 1u);        // 沿用档位不新增
    EXPECT_EQ(faults[0].count, 1u);      // 计数重置
    EXPECT_EQ(faults[0].firstTimeMs > 0, true);
    EXPECT_EQ(spy.count.load(), 2);      // 再发一次事件
}
