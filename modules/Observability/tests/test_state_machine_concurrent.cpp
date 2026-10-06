// ============================================================================
// test_state_machine_concurrent.cpp — CAS 并发转态竞争压力测试（P1-T4）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §2.3/§10.1
//   · transition 内层为「resolve → CAS → 败者按最新态重判」循环：并发同发
//     事件时仅首个 CAS 胜者完成转态，败者重查转换表无边即 fail——
//     「不丢转态、不非法覆盖」由成功计数 + 终态合法集共同锁定
//   · 断言基于 CAS 语义（恰一成功 / 终态 ∈ 合法集）而非时序，天然不 flaky
//
// 与 test_state_machine.cpp 的 ConcurrentFaultExactlyOneWins（2 线程 × 1 次）
// 互补：本文件做 8 线程 × 200 次放量压力 + 双事件混合风暴。
// ============================================================================
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include "StateMachine.h"

using Scanner::service::StateMachine;
using Scanner::service::SystemState;
using Scanner::EventType;

namespace {

constexpr int kThreads = 8;      // TEST 2 内 4+4 分组
constexpr int kPerThread = 200;

} // namespace

TEST(SMConcurrent, FaultStormExactlyOneWins) {
    // 8 线程 × 200 次自 S2 同发 FaultOccurred：首个 CAS 胜者入 S7；
    // 其余 1599 次重判 {S7,FaultOccurred} 无边 → fail 且保持 S7
    StateMachine sm;
    ASSERT_TRUE(sm.transition(EventType::SystemReady).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Standby);

    std::atomic<int> okCount{0};
    std::atomic<bool> go{false};
    auto racer = [&] {
        while (!go.load(std::memory_order_acquire)) {}
        int localOk = 0;
        for (int i = 0; i < kPerThread; ++i) {
            if (sm.transition(EventType::FaultOccurred).success) ++localOk;
        }
        okCount.fetch_add(localOk);
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < kThreads; ++i) pool.emplace_back(racer);
    go.store(true, std::memory_order_release);
    for (auto& t : pool) t.join();  // 全员 join 到达即「无崩溃」

    EXPECT_EQ(okCount.load(), 1);
    EXPECT_EQ(sm.getCurrentState(), SystemState::FaultSelfCheck);
}

TEST(SMConcurrent, MixedStormTerminalInLegalSet) {
    // 4 线程 × 200 FaultOccurred + 4 线程 × 200 SystemReady，自 S1 混合竞争：
    //   · 首胜者若是 FaultOccurred：S1 有故障边（S1 设备故障）→ 直入 S7
    //   · 首胜者若是 SystemReady：S1→S2 后首个 Fault 胜者 S2→S7
    // 两条路径终态 ∈ {Standby, FaultSelfCheck}（矩阵推演实际必收敛 S7，
    // 此处按任务口径以合法集断言）；成功总数 ≥ 1（S1 起首次尝试必有边）
    StateMachine sm;
    ASSERT_EQ(sm.getCurrentState(), SystemState::Init);

    std::atomic<int> okCount{0};
    std::atomic<bool> go{false};
    auto racer = [&](EventType ev) {
        while (!go.load(std::memory_order_acquire)) {}
        int localOk = 0;
        for (int i = 0; i < kPerThread; ++i) {
            if (sm.transition(ev).success) ++localOk;
        }
        okCount.fetch_add(localOk);
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < kThreads / 2; ++i) pool.emplace_back(racer, EventType::FaultOccurred);
    for (int i = 0; i < kThreads / 2; ++i) pool.emplace_back(racer, EventType::SystemReady);
    go.store(true, std::memory_order_release);
    for (auto& t : pool) t.join();  // 全员 join 到达即「无崩溃」

    EXPECT_GE(okCount.load(), 1);
    const SystemState finalState = sm.getCurrentState();
    EXPECT_TRUE(finalState == SystemState::Standby || finalState == SystemState::FaultSelfCheck)
        << "终态非法: " << StateMachine::stateToString(finalState);
}
