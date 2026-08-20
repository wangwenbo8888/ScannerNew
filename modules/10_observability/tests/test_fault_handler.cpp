// ============================================================================
// test_fault_handler.cpp — FaultHandler 故障档案表改造单测（P2-T7，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §4
//   · 故障档案表：{id, sourceId, severity, message, firstTimeMs, count}；
//     同 source+severity 1s 窗口聚合计数（防风暴：不发事件不转态，§4.1）
//   · 来源标识（修「事件丢 source」缺陷）：registerSource 稳定 id（重名同 id）；
//     FaultOccurred(param1=severity, param2=sourceId)（§4.2）
//   · S7 故障链（§4.4）：Error 级以上且当前态非 S6（免疫）/S7（不重转）→
//     transition(FaultOccurred) 成功 → safeStopCb_() + LedControl(param1=1 红)
//     ——红灯不再复用 EmergencyStop 通道（§4.3，T8 删值前置）
//   · 免疫矩阵：S6 只记档+事件；Warning/Info 只记档+事件不转态不停机
//   · 恢复：clearFault 只清档+FaultCleared 事件（矩阵无 FaultCleared 边，
//     不得报错——恢复唯 SelfCheckPassed）；selfCheckPassed → lastHealthy
//     + LedControl(param1=2 绿)
//   · bus 订阅口（start 后外部源直接 publish）仅记档：EventBus 同步分发持
//     总线锁，锁内转态会经 StateChanged publish 重入死锁——故障链动作归
//     reportFault 直调口（§4.6：08 落地接注入口）
//
// 用真 StateMachine + 真 EventBus（模块内已有，不 mock）；safeStop 用
// lambda 计数注入（setSafeStopCallback）。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
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

// 驱动 SM 到 S2（各用例公共起点；故障链用例均自 S2 展开）
void driveToStandby(StateMachine& sm) {
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Standby);
}

} // namespace

TEST(FH, SourceRegistrationStableId) {
    FaultHandler fh;
    const int cam1 = fh.registerSource("Camera");
    const int cam2 = fh.registerSource("Camera");
    const int mcu  = fh.registerSource("MCU");
    EXPECT_GT(cam1, 0);
    EXPECT_EQ(cam1, cam2) << "重复名应返回同 id";
    EXPECT_NE(cam1, mcu);
    EXPECT_EQ(fh.registerSource("MCU"), mcu) << "稳定性跨调用保持";
}

TEST(FH, ReportFaultEventCarriesSource) {
    EventBus bus;
    FaultHandler fh(&bus);
    const int cam = fh.registerSource("Camera");
    EventSpy spy;
    spy.subscribe(bus, EventType::FaultOccurred);

    fh.reportFault(cam, FaultSeverity::Warning, "capture timeout");

    EXPECT_EQ(spy.count.load(), 1);
    EXPECT_EQ(spy.param1.load(), static_cast<int64_t>(FaultSeverity::Warning));
    EXPECT_EQ(spy.param2.load(), cam);  // 修「丢 source」：param2 携带来源 id
}

TEST(FH, AggregateWithinWindow) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    const int cam = fh.registerSource("Camera");
    EventSpy spy;
    spy.subscribe(bus, EventType::FaultOccurred);
    driveToStandby(sm);

    fh.reportFault(cam, FaultSeverity::Error, "frame drop");
    fh.reportFault(cam, FaultSeverity::Error, "frame drop again");  // 同源同级 1s 内

    const auto faults = fh.activeFaults();
    ASSERT_EQ(faults.size(), 1u);
    EXPECT_EQ(faults[0].count, 2u);      // 档案聚合计数
    EXPECT_EQ(spy.count.load(), 1);      // 只发 1 次事件
    EXPECT_EQ(stops, 1);                 // 只 1 次转态链（safeStop 随转态成功）
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
}

TEST(FH, ErrorInS2TransitionsToS7) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    EventSpy led;
    led.subscribe(bus, EventType::LedControl);
    const int cam = fh.registerSource("Camera");
    driveToStandby(sm);

    fh.reportFault(cam, FaultSeverity::Error, "device lost");

    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
    EXPECT_EQ(stops, 1);
    ASSERT_EQ(led.count.load(), 1);
    EXPECT_EQ(led.param1.load(), 1);  // 红灯：LedControl(param1=1)
}

TEST(FH, WarningDoesNotTransition) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    EventSpy fault, led;
    fault.subscribe(bus, EventType::FaultOccurred);
    led.subscribe(bus, EventType::LedControl);
    const int cam = fh.registerSource("Camera");
    driveToStandby(sm);

    fh.reportFault(cam, FaultSeverity::Warning, "temp high");

    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);  // 不转态
    EXPECT_EQ(stops, 0);                                    // 不安全停机
    EXPECT_EQ(led.count.load(), 0);                         // 不发灯事件
    EXPECT_EQ(fault.count.load(), 1);                       // 事件照发
    ASSERT_EQ(fh.activeFaults().size(), 1u);                // 记档照记
    EXPECT_EQ(fh.activeFaults()[0].severity, FaultSeverity::Warning);
}

TEST(FH, S6ImmuneRecordOnly) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    EventSpy fault, led;
    fault.subscribe(bus, EventType::FaultOccurred);
    led.subscribe(bus, EventType::LedControl);
    const int cam = fh.registerSource("Camera");
    driveToStandby(sm);
    ASSERT_TRUE(sm.transition(EventType::PostProcessStarted).success);  // S6

    fh.reportFault(cam, FaultSeverity::Error, "gpu oom");

    EXPECT_EQ(sm.getCurrentState(), SystemState::PostProcessing);  // 免疫不转态
    EXPECT_EQ(stops, 0);
    EXPECT_EQ(led.count.load(), 0);
    EXPECT_EQ(fault.count.load(), 1);  // 只记档+事件
    ASSERT_EQ(fh.activeFaults().size(), 1u);
    EXPECT_EQ(fh.activeFaults()[0].count, 1u);
}

TEST(FH, S7StaysOnRepeatFault) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    EventSpy fault, led;
    fault.subscribe(bus, EventType::FaultOccurred);
    led.subscribe(bus, EventType::LedControl);
    const int cam = fh.registerSource("Camera");
    const int mcu = fh.registerSource("MCU");
    driveToStandby(sm);

    fh.reportFault(cam, FaultSeverity::Error, "camera lost");  // → S7
    fh.reportFault(mcu, FaultSeverity::Error, "mcu offline");  // S7 内异源新档

    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);  // 不重转
    EXPECT_EQ(stops, 1);                 // 故障链仅首次成功转态触发
    EXPECT_EQ(led.count.load(), 1);
    EXPECT_EQ(fault.count.load(), 2);    // 异源非聚合，事件照发
    ASSERT_EQ(fh.activeFaults().size(), 2u);  // 记档两条（源隔离）
}

TEST(FH, ClearFaultPublishesAndArchives) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    const int cam = fh.registerSource("Camera");
    EventSpy cleared;
    cleared.subscribe(bus, EventType::FaultCleared);
    driveToStandby(sm);
    fh.reportFault(cam, FaultSeverity::Error, "camera lost");
    ASSERT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
    ASSERT_EQ(fh.activeFaults().size(), 1u);

    fh.clearFault(fh.activeFaults()[0].id);  // 矩阵无 FaultCleared 边——不得报错

    EXPECT_EQ(fh.activeFaults().size(), 0u);  // 档案清空
    EXPECT_EQ(cleared.count.load(), 1);       // FaultCleared 事件
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);  // 恢复不走此口
}

TEST(FH, SelfCheckPassedRecoversAndGreen) {
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    EventSpy led;
    led.subscribe(bus, EventType::LedControl);
    const int cam = fh.registerSource("Camera");
    driveToStandby(sm);
    fh.reportFault(cam, FaultSeverity::Error, "camera lost");
    ASSERT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);

    fh.selfCheckPassed();  // S7 → lastHealthy(=S2) 续作恢复

    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
    ASSERT_EQ(led.count.load(), 2);    // 红（故障）+ 绿（恢复）
    EXPECT_EQ(led.param1.load(), 2);   // LedControl(param1=2 绿)
}

TEST(FH, ExternalBusPublishEntersArchive) {
    // bus 订阅口仅记档：EventBus 同步分发持总线锁，锁内转态会经 StateChanged
    // publish 重入死锁——故障链动作归 reportFault 直调口（§4.6 注入口）
    EventBus bus;
    StateMachine sm;
    FaultHandler fh(&bus);
    fh.setStateMachine(&sm);
    int stops = 0;
    fh.setSafeStopCallback([&] { ++stops; });
    const int cam = fh.registerSource("Camera");
    fh.start();
    driveToStandby(sm);

    Event evt;
    evt.type = EventType::FaultOccurred;
    evt.param1 = static_cast<int64_t>(FaultSeverity::Error);
    evt.param2 = cam;
    bus.publish(evt);  // 外部源（如 08）直接 publish 也能进档案

    const auto faults = fh.activeFaults();
    ASSERT_EQ(faults.size(), 1u);
    EXPECT_EQ(faults[0].sourceId, cam);
    EXPECT_EQ(faults[0].severity, FaultSeverity::Error);
    EXPECT_EQ(faults[0].count, 1u);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);  // 不转态（见用例头注释）
    EXPECT_EQ(stops, 0);
}
