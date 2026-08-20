// ============================================================================
// test_state_machine.cpp — 状态机 7 态表驱动重写单测（P1-T3，TDD 先行）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §2/§4.4/§4.5
//   · 7 态 S1–S7；S3/S4/S5/S6 互斥必经 S2；S4/S5 由 ScanStarted param 定
//   · S6 免疫故障与断连（矩阵无边）；S7 断连保持（自检含重连）
//   · 恢复唯一出口 SelfCheckPassed→lastHealthy（§4.4 续作语义，永不回 S6，兜底 S2）
//
// 与任务示意转换表的一处差异（设计语义修表，此处声明）：
//   {S7, DeviceConnected→S2} 不设边——设备连接只是自检清单一项（§2.4），
//   且该边会直跳 S2 丢失「回前态续作」语义（§4.4）；S7 出口唯 SelfCheckPassed。
//
// 约定补充：
//   · ScanStarted param：0→S4 / 1→S5，其余值拒绝（base ScanMode 契约仅 0/1）
//   · S7 内再次 FaultOccurred / S1 内 DeviceDisconnected：无→边 fail 且状态不变
//     （「保持」由非法转换不改状态保证；防风暴聚合计数归 T7 FaultHandler）
//   · FaultCleared 入 FullMatrix 事件集以固化「旧故障恢复通道已死」（T7 前仅运行时 fail）
//
// FullMatrix 事件集（12 个）：SystemReady/CalibStarted/CalibFinished/ScanStarted/
//   ScanStopped/PostProcessStarted/PostProcessFinished/FaultOccurred/FaultCleared/
//   SelfCheckPassed/DeviceConnected/DeviceDisconnected。
//   不入集：CommandRejected/LedControl/HealthReport（通知类，与转态无关）、
//   ScanPaused/EmergencyStop/SessionStarted（待清理旧值）、其余帧/温度类事件。
//   期望表 kLegal 独立于实现 kEdges 抄自设计文档——两表分叉即测试红。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include "StateMachine.h"
#include "base/EventBus.h"

using Scanner::service::StateMachine;
using Scanner::service::SystemState;
using Scanner::EventType;
using Scanner::Result;
namespace ss = Scanner::service;

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

} // namespace

TEST(SM, InitialIsInit) {
    StateMachine sm;
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
    EXPECT_EQ(sm.getStateName(), "Init");
    EXPECT_FALSE(sm.canOperate("scan"));
}

TEST(SM, StandbyGateRules) {
    // S2：calibrate/scan/postprocess/edit 全放行（edit=05 交互会话前置，§2.6）
    {
        StateMachine sm;
        driveTo(sm, SystemState::Standby);
        for (const char* op : {"calibrate", "scan", "postprocess", "edit"}) {
            EXPECT_TRUE(sm.canOperate(op)) << op;
        }
        EXPECT_FALSE(sm.canOperate("bogus"));
    }
    // 非 S2：全拒绝（含未知操作名）
    for (SystemState s : kAllStates) {
        if (s == SystemState::Standby) continue;
        StateMachine sm;
        driveTo(sm, s);
        for (const char* op : {"calibrate", "scan", "postprocess", "edit", "bogus"}) {
            EXPECT_FALSE(sm.canOperate(op)) << "state=" << static_cast<int>(s) << " op=" << op;
        }
    }
}

TEST(SM, DeviceConnectedDoesNotTransition) {
    // §2.4：设备连接只是 S1 自检清单一项，不单独切态（废 ScannerWindow 直切旧法）
    StateMachine sm;
    EXPECT_FALSE(sm.transition(EventType::DeviceConnected).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
}

TEST(SM, MutuallyExclusiveViaS2) {
    // §2.2：S3/S4/S5/S6 互斥必经 S2——S3 内直接开扫/后处理必须拒绝
    StateMachine sm;
    driveTo(sm, SystemState::Calibrating);
    EXPECT_FALSE(sm.transition(EventType::ScanStarted).success);
    EXPECT_FALSE(sm.transition(EventType::PostProcessStarted).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
}

TEST(SM, ScanModeParamPicksS4S5) {
    {
        StateMachine sm;
        driveTo(sm, SystemState::Standby);
        ASSERT_TRUE(sm.transition(EventType::ScanStarted, 0).success);
        EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
        EXPECT_EQ(sm.getStateName(), "ScanMarker");
    }
    {
        StateMachine sm;
        driveTo(sm, SystemState::Standby);
        ASSERT_TRUE(sm.transition(EventType::ScanStarted, 1).success);
        EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarkerLaser);
        EXPECT_EQ(sm.getStateName(), "ScanMarkerLaser");
    }
    // ScanMode 契约仅 0/1：非法模式参数拒绝、保持 S2
    {
        StateMachine sm;
        driveTo(sm, SystemState::Standby);
        EXPECT_FALSE(sm.transition(EventType::ScanStarted, 2).success);
        EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
    }
}

TEST(SM, FaultFromS3ToS7AndRecoverBack) {
    // §4.4：进 S7 记 lastHealthy=S3；SelfCheckPassed 续作回 S3
    StateMachine sm;
    driveTo(sm, SystemState::Calibrating);
    ASSERT_TRUE(sm.transition(EventType::FaultOccurred).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
    ASSERT_TRUE(sm.transition(EventType::SelfCheckPassed).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
}

TEST(SM, S7NeverRecoversToS6) {
    // 不变量：lastHealthy 不可能为 S6（S6 免疫进不了 S7）——对每个可达前态
    // 验证恢复回原态（含 S1：自检中出故障场景）且永不为 S6。实现内 resolveNext
    // 的「S6→S2」兜底分支在当前矩阵下不可达，属防御代码，由此不变量间接锁定。
    for (SystemState from : kAllStates) {
        StateMachine sm;
        driveTo(sm, from);
        Result r = sm.transition(EventType::FaultOccurred);
        if (from == SystemState::PostProcessing) {
            EXPECT_FALSE(r.success) << "S6 应免疫故障转态";
            EXPECT_EQ(sm.getCurrentState(), SystemState::PostProcessing);
            continue;
        }
        if (from == SystemState::FaultSelfCheck) {
            EXPECT_FALSE(r.success) << "S7 内重复故障应无边保持";
            EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
            continue;
        }
        ASSERT_TRUE(r.success) << "from=S" << static_cast<int>(from);
        ASSERT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
        ASSERT_TRUE(sm.transition(EventType::SelfCheckPassed).success);
        EXPECT_EQ(sm.getCurrentState(), from);
        EXPECT_NE(sm.getCurrentState(), SystemState::PostProcessing);
    }
}

TEST(SM, S6ImmuneToFault) {
    // §4.5：S6 免疫故障转态（GPU/算法失败走 Result 人工处置，不进 S7）
    StateMachine sm;
    driveTo(sm, SystemState::PostProcessing);
    EXPECT_FALSE(sm.transition(EventType::FaultOccurred).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::PostProcessing);
}

TEST(SM, DisconnectGoesS1ExceptS6S7) {
    // §2.2/§4.5：断连 S2–S5→S1；S6 免疫；S7 保持（自检含重连）；S1 无边 fail
    for (SystemState s : kAllStates) {
        StateMachine sm;
        driveTo(sm, s);
        Result r = sm.transition(EventType::DeviceDisconnected);
        const bool toInit = (s >= SystemState::Standby && s <= SystemState::ScanMarkerLaser);
        EXPECT_EQ(r.success, toInit) << "state=" << static_cast<int>(s);
        EXPECT_EQ(sm.getCurrentState(), toInit ? SystemState::Init : s)
            << "state=" << static_cast<int>(s);
    }
}

TEST(SM, PublishesStateChangedOnBus) {
    // EventBus 同步分发（publish 在锁内联调 handler）；param1=from param2=to
    Scanner::infra::EventBus bus;
    StateMachine sm(&bus);
    std::atomic<int> got{0};
    int64_t p1 = -1, p2 = -1;
    bus.subscribe(EventType::StateChanged, [&](const Scanner::Event& e) {
        if (e.type == EventType::StateChanged) {
            p1 = e.param1;
            p2 = e.param2;
            got++;
        }
    });
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    EXPECT_EQ(got.load(), 1);
    EXPECT_EQ(p1, static_cast<int64_t>(SystemState::Init));
    EXPECT_EQ(p2, static_cast<int64_t>(SystemState::Standby));
}

TEST(SM, OnStateChangeCallbackFires) {
    // onStateChange 回调保留（AppContext/UI 注入口），CAS 成功后无锁环境触发
    StateMachine sm;
    int calls = 0;
    SystemState o = SystemState::Init, n = SystemState::Init;
    sm.onStateChange([&](SystemState oldS, SystemState newS) {
        o = oldS;
        n = newS;
        ++calls;
    });
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(o, SystemState::Init);
    EXPECT_EQ(n, SystemState::Standby);
}

TEST(SM, ConcurrentFaultExactlyOneWins) {
    // §2.3/§10.1：CAS 并发转态——两线程自 S2 同发 FaultOccurred，恰好一次成功
    // 入 S7；败者重判后无 {S7,FaultOccurred} 边 → fail（保持 S7，不丢转态不非法覆盖）
    StateMachine sm;
    driveTo(sm, SystemState::Standby);
    std::atomic<int> okCount{0};
    std::atomic<bool> go{false};
    auto racer = [&] {
        while (!go.load(std::memory_order_acquire)) {}
        if (sm.transition(EventType::FaultOccurred).success) okCount.fetch_add(1);
    };
    std::thread t1(racer), t2(racer);
    go.store(true, std::memory_order_release);
    t1.join();
    t2.join();
    EXPECT_EQ(okCount.load(), 1);
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
}

TEST(SM, FullMatrix) {
    // 期望合法边集合——独立于实现 kEdges 抄自设计 §2.2/§4.4/§4.5。
    // S7 行的 SelfCheckPassed 目标 = lastHealthy（本表 S7 固定自 S2 进入 → 期望 S2）；
    // 其他 lastHealthy 值由 FaultFromS3ToS7AndRecoverBack / S7NeverRecoversToS6 覆盖。
    struct Legal {
        SystemState from;
        EventType ev;
        int64_t param;
        SystemState to;
    };
    const Legal kLegal[] = {
        {SystemState::Init,            EventType::SystemReady,         0, SystemState::Standby},
        {SystemState::Init,            EventType::FaultOccurred,       0, SystemState::FaultSelfCheck},
        {SystemState::Standby,         EventType::CalibStarted,        0, SystemState::Calibrating},
        {SystemState::Standby,         EventType::ScanStarted,         0, SystemState::ScanMarker},
        {SystemState::Standby,         EventType::ScanStarted,         1, SystemState::ScanMarkerLaser},
        {SystemState::Standby,         EventType::PostProcessStarted,  0, SystemState::PostProcessing},
        {SystemState::Standby,         EventType::FaultOccurred,       0, SystemState::FaultSelfCheck},
        {SystemState::Standby,         EventType::DeviceDisconnected,  0, SystemState::Init},
        {SystemState::Calibrating,     EventType::CalibFinished,       0, SystemState::Standby},
        {SystemState::Calibrating,     EventType::FaultOccurred,       0, SystemState::FaultSelfCheck},
        {SystemState::Calibrating,     EventType::DeviceDisconnected,  0, SystemState::Init},
        {SystemState::ScanMarker,      EventType::ScanStopped,         0, SystemState::Standby},
        {SystemState::ScanMarker,      EventType::FaultOccurred,       0, SystemState::FaultSelfCheck},
        {SystemState::ScanMarker,      EventType::DeviceDisconnected,  0, SystemState::Init},
        {SystemState::ScanMarkerLaser, EventType::ScanStopped,         0, SystemState::Standby},
        {SystemState::ScanMarkerLaser, EventType::FaultOccurred,       0, SystemState::FaultSelfCheck},
        {SystemState::ScanMarkerLaser, EventType::DeviceDisconnected,  0, SystemState::Init},
        {SystemState::PostProcessing,  EventType::PostProcessFinished, 0, SystemState::Standby},
        {SystemState::FaultSelfCheck,  EventType::SelfCheckPassed,     0, SystemState::Standby},
    };
    const EventType kEvents[] = {
        EventType::SystemReady,         EventType::CalibStarted,
        EventType::CalibFinished,       EventType::ScanStarted,
        EventType::ScanStopped,         EventType::PostProcessStarted,
        EventType::PostProcessFinished, EventType::FaultOccurred,
        EventType::FaultCleared,        EventType::SelfCheckPassed,
        EventType::DeviceConnected,     EventType::DeviceDisconnected};

    auto findLegal = [&](SystemState from, EventType ev, int64_t param) -> const Legal* {
        for (const auto& l : kLegal) {
            if (l.from == from && l.ev == ev && l.param == param) return &l;
        }
        return nullptr;
    };

    // 两层 for 全组合：7 态 × 12 事件（ScanStarted 加验 param 0/1/2，其余事件
    // param 无语义仅验 0）；正例断言 ok+目标态，非法组合断言 fail+状态不变
    for (SystemState s : kAllStates) {
        for (EventType ev : kEvents) {
            const int64_t params[3] = {0, 1, 2};
            for (int64_t p : params) {
                if (ev != EventType::ScanStarted && p != 0) continue;
                StateMachine sm;
                driveTo(sm, s);
                const Legal* hit = findLegal(s, ev, p);
                Result r = sm.transition(ev, p);
                EXPECT_EQ(r.success, hit != nullptr)
                    << "from=S" << static_cast<int>(s) << " ev=" << static_cast<int>(ev)
                    << " param=" << p;
                EXPECT_EQ(sm.getCurrentState(), hit ? hit->to : s)
                    << "from=S" << static_cast<int>(s) << " ev=" << static_cast<int>(ev)
                    << " param=" << p;
            }
        }
    }
}
