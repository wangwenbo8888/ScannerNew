// ============================================================================
// test_command_gate.cpp — CommandGate 双口命令通道单测（P1-T5，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §3
//   · submit 三关（注册/门禁/pre）→ 即切态（防装配期闯入）→ handler 仅点火（§3.4）
//   · 触发型命令（finish_scan 型）startedEvent=0：submit 只点火不切态，
//     收尾切态唯 notifyCompleted（§3.1 双口，⑦≠转态时点）
//   · handler 同步失败：回滚 finishedEvent + ERROR 日志 + CommandRejected（§3.3）
//
// 事件契约：CommandRejected（任一关挂）与 startedEvent（如 CalibStarted，
//   param1=from param2=to）经 EventBus 同步分发；StateChanged 由状态机自发，不重复验。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include "CommandGate.h"
#include "StateMachine.h"
#include "base/EventBus.h"

using Scanner::service::CommandGate;
using Scanner::service::StateMachine;
using Scanner::service::SystemState;
using Scanner::EventType;
using Scanner::Result;
using Scanner::infra::EventBus;

namespace {

// 标准标定命令 Spec（S2→S3，CalibStarted/CalibFinished 成对）
CommandGate::Spec calibSpec() {
    CommandGate::Spec s;
    s.name = "start_calibration";
    s.gateOp = "calibrate";
    s.startedEvent = EventType::CalibStarted;
    s.startedTo = SystemState::Calibrating; // 调试/文档用
    s.finishedEvent = EventType::CalibFinished;
    return s;
}

// 驱动新状态机到 S2（Standby，门禁主战场）
void driveToStandby(StateMachine& sm) {
    ASSERT_EQ(sm.getCurrentState(), SystemState::Init);
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Standby);
}

// 事件计数器（EventBus 同步分发，单线程发布下 param 读写安全）
struct EventCounter {
    std::atomic<int> count{0};
    int64_t p1 = -1, p2 = -1;
    void sub(EventBus& bus, EventType t) {
        bus.subscribe(t, [this, t](const Scanner::Event& e) {
            if (e.type != t) return;
            p1 = e.param1;
            p2 = e.param2;
            count.fetch_add(1);
        });
    }
};

} // namespace

TEST(Gate, SubmitUnregistered_Rejected) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    EventCounter rej;
    rej.sub(bus, EventType::CommandRejected);
    Result r = gate.submit("nope");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
    EXPECT_EQ(rej.count.load(), 1);
}

TEST(Gate, SubmitGateDenied_Rejected) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    ASSERT_TRUE(gate.registerCommand(calibSpec()).success);
    EventCounter rej;
    rej.sub(bus, EventType::CommandRejected);
    // S1（Init）下 canOperate("calibrate")=false → 门禁拒绝，态不变
    Result r = gate.submit("start_calibration");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
    EXPECT_EQ(rej.count.load(), 1);
}

TEST(Gate, PreconditionDenied_Rejected) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    auto spec = calibSpec();
    spec.pre = [] { return Result::fail("参数未就绪"); };
    ASSERT_TRUE(gate.registerCommand(spec).success);
    driveToStandby(sm);
    EventCounter rej;
    rej.sub(bus, EventType::CommandRejected);
    Result r = gate.submit("start_calibration");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.message, "参数未就绪"); // pre 的 fail message 透传
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby); // 拒绝后态不变
    EXPECT_EQ(rej.count.load(), 1);
}

TEST(Gate, StartCommandSwitchesStateImmediately) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    auto spec = calibSpec();
    std::atomic<bool> sawCalibrating{false};
    spec.handler = [&] {
        // 先切态后点火：handler 执行时状态应已是 S3（§3.1 防装配期闯入）
        sawCalibrating.store(sm.getCurrentState() == SystemState::Calibrating);
        return Result::ok();
    };
    ASSERT_TRUE(gate.registerCommand(spec).success);
    driveToStandby(sm);
    EventCounter started;
    started.sub(bus, EventType::CalibStarted);
    Result r = gate.submit("start_calibration");
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(sawCalibrating.load());
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
    EXPECT_EQ(started.count.load(), 1);
    EXPECT_EQ(started.p1, static_cast<int64_t>(SystemState::Standby));
    EXPECT_EQ(started.p2, static_cast<int64_t>(SystemState::Calibrating));
}

TEST(Gate, HandlerSyncFailureRollsBack) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    auto spec = calibSpec();
    spec.handler = [] { return Result::fail("设备打不开"); };
    ASSERT_TRUE(gate.registerCommand(spec).success);
    driveToStandby(sm);
    EventCounter rej;
    rej.sub(bus, EventType::CommandRejected);
    EventCounter started;
    started.sub(bus, EventType::CalibStarted);
    Result r = gate.submit("start_calibration");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby); // 回滚 S2（§3.3）
    EXPECT_EQ(rej.count.load(), 1);
    EXPECT_EQ(started.count.load(), 0);    // 失败路径不发 CalibStarted
}

TEST(Gate, NotifyCompletedSwitchesBack) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    ASSERT_TRUE(gate.registerCommand(calibSpec()).success);
    driveToStandby(sm);
    ASSERT_TRUE(gate.submit("start_calibration").success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Calibrating);
    Result r = gate.notifyCompleted("start_calibration", true);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(Gate, ConcurrentSubmitExactlyOneWins) {
    StateMachine sm; // 不接 bus：并发下 publish 计数无断言意义
    CommandGate gate(&sm);
    auto spec = calibSpec();
    spec.handler = [] { return Result::ok(); };
    ASSERT_TRUE(gate.registerCommand(spec).success);
    driveToStandby(sm);
    constexpr int kThreads = 8;
    std::atomic<int> okCount{0};
    std::atomic<bool> go{false};
    auto racer = [&] {
        while (!go.load(std::memory_order_acquire)) {}
        if (gate.submit("start_calibration").success) okCount.fetch_add(1);
    };
    std::thread threads[kThreads];
    for (auto& t : threads) t = std::thread(racer);
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    EXPECT_EQ(okCount.load(), 1); // CAS 转态唯一仲裁：败者门禁或切态关被拒
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
}

TEST(Gate, FinishScanOnlyCompletesOnNotify) {
    EventBus bus;
    StateMachine sm(&bus);
    CommandGate gate(&sm, &bus);
    // finish_scan：触发型（startedEvent=0 不切态），finishedEvent=ScanStopped
    CommandGate::Spec fin;
    fin.name = "finish_scan";
    fin.gateOp = ""; // 触发型在 S4 内提交，不走 S2 门禁
    fin.startedEvent = static_cast<EventType>(0);
    fin.finishedEvent = EventType::ScanStopped;
    std::atomic<int> fired{0};
    fin.handler = [&] {
        fired.fetch_add(1);
        return Result::ok();
    };
    ASSERT_TRUE(gate.registerCommand(fin).success);
    driveToStandby(sm);
    ASSERT_TRUE(sm.transition(EventType::ScanStarted, 0).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
    // ⑦ submit 只触发（点火），不切态
    EXPECT_TRUE(gate.submit("finish_scan").success);
    EXPECT_EQ(fired.load(), 1);
    EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
    // ⑩ notifyCompleted 才收尾切态 S4→S2
    EXPECT_TRUE(gate.notifyCompleted("finish_scan", true).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

// ---- 酌情补充（目录/收尾口边界）----

TEST(Gate, DuplicateRegistrationRejected) {
    StateMachine sm;
    CommandGate gate(&sm);
    ASSERT_TRUE(gate.registerCommand(calibSpec()).success);
    EXPECT_FALSE(gate.registerCommand(calibSpec()).success); // 重名注册 fail
}

TEST(Gate, NotifyCompletedNoFinishEventFails) {
    StateMachine sm;
    CommandGate gate(&sm);
    // system_ready 型：finishedEvent=0 → notifyCompleted 无转态可切，fail
    CommandGate::Spec sys;
    sys.name = "system_ready";
    sys.gateOp = "";
    sys.startedEvent = EventType::SystemReady;
    sys.finishedEvent = static_cast<EventType>(0);
    ASSERT_TRUE(gate.registerCommand(sys).success);
    EXPECT_FALSE(gate.notifyCompleted("system_ready", true).success);
    EXPECT_FALSE(gate.notifyCompleted("unregistered", true).success); // 未注册同 fail
}
