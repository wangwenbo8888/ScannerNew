// ============================================================================
// test_warmup_sequence.cpp — WarmupSequence 预热看火员单测（D-T11）
//
// 契约钉死（2026-08-18 设计 §2.2 + D-T11 双点锚点法口径）：
//   - 状态机 Idle→Heating→Done（Done=已回调稳或超时；start() 复位重入
//     Heating——锚点/超时基准全清重开）；
//   - 锚点=进入 Heating 后第一个喂入；窗口内任一喂入偏出锚点 >stableDeltaC
//     → 锚点重置为该喂入重新计时；「当前-锚点 ≥ stableWindowMs」且波动
//     ≤stableDeltaC 且离目标 ≤nearTargetC → onStable 恰一次；
//   - 未达目标（离目标 >nearTargetC）→ 即使稳定也不判稳（继续等——时间照走，
//     超时兜底）；
//   - 超时 = now-基准 ≥ timeoutMs（基准=进入 Heating 后首个观测：喂温或 tick
//     ——温度停更也能超时）→ onTimeout 恰一次；只报不停（类无停止加热出口，
//     以「无第三回调」代之断言）；
//   - 时基：tsMs/nowMs 调用方保证同源单调（统一 PC 接收时刻）；
//   - Idle 喂温：lastTemp 更新（报警快照）但不锚定不判定；Done 后喂温/tick
//     全静默（lastTemp 照常更新）。
// 用例 = 实施计划 Task 11 九条 + 复位重开一条补钉。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/WarmupSequence.h"

using namespace Scanner::device;

namespace {

// 回调计数器（onStable/onTimeout 各记次数——「恰一次」断言用）
struct Counters {
    int stable = 0;
    int timeout = 0;
    void bind(WarmupSequence& w) {
        w.onStable = [this] { ++stable; };
        w.onTimeout = [this] { ++timeout; };
    }
};

} // namespace

// —— 1. DefaultIdle：默认 Idle；start 前喂温/tick 全无效（无回调、状态不变）——
TEST(WarmupSequence, DefaultIdle) {
    WarmupSequence w;
    Counters cnt;
    cnt.bind(w);
    EXPECT_EQ(w.state(), WarmupSequence::State::Idle);
    w.onTemperature(25.0, 0);
    w.tick(600000);                      // Idle 中时钟推进也不超时（无会话）
    EXPECT_EQ(w.state(), WarmupSequence::State::Idle);
    EXPECT_EQ(cnt.stable, 0);
    EXPECT_EQ(cnt.timeout, 0);
}

// —— 2. RiseThenStable：升温跳变重锚 → 窗满 10s 且离目标 ≤2 → @12s 判稳恰一次 ——
TEST(WarmupSequence, RiseThenStable) {
    WarmupSequence w;                    // 默认窗口 10s / 波动 0.1 / 近目标 2.0
    Counters cnt;
    cnt.bind(w);
    w.start(42.0);
    EXPECT_EQ(w.state(), WarmupSequence::State::Heating);
    w.onTemperature(30.0, 0);            // 锚点=30@0
    w.onTemperature(40.0, 2000);         // 跳变 10>0.1 → 锚点重置 40@2s
    for (int t = 3000; t <= 11000; t += 1000) {
        w.onTemperature(40.05, t);       // 波动 0.05≤0.1 不重锚；离目标 1.95≤2
        EXPECT_EQ(cnt.stable, 0) << "t=" << t;   // 窗未满（≤9s）不判稳
    }
    w.onTemperature(40.05, 12000);       // 窗满 10s（12s-2s）→ 判稳
    EXPECT_EQ(cnt.stable, 1);
    EXPECT_EQ(cnt.timeout, 0);
    EXPECT_EQ(w.state(), WarmupSequence::State::Done);
}

// —— 3. TimeoutNeverStable：恒 30 稳但离目标 20 → tick 推过 timeoutMs 只超时 ——
TEST(WarmupSequence, TimeoutNeverStable) {
    WarmupConfig cfg;
    cfg.timeoutMs = 1000;                // 测试缩短超时
    WarmupSequence w(cfg);
    Counters cnt;
    cnt.bind(w);
    w.start(50.0);
    for (int t = 0; t <= 750; t += 250) w.onTemperature(30.0, t);  // 稳但差 20℃
    EXPECT_EQ(cnt.stable, 0);
    EXPECT_EQ(cnt.timeout, 0);           // 未到 1000ms 不超时
    w.tick(1000);                        // 推到 timeoutMs（≥ 即超）
    EXPECT_EQ(cnt.timeout, 1);
    EXPECT_EQ(cnt.stable, 0);
    EXPECT_EQ(w.state(), WarmupSequence::State::Done);
}

// —— 4. AnchorResetOnJump：@5s 跳变重锚 → 判稳在 @15s 不在 @10s ——
TEST(WarmupSequence, AnchorResetOnJump) {
    WarmupSequence w;                    // 默认窗口 10s
    Counters cnt;
    cnt.bind(w);
    w.start(47.0);                       // 离 45 恰 2.0（≤nearTargetC 边界含）
    w.onTemperature(35.0, 0);            // 锚点=35@0
    for (int t = 1000; t <= 4000; t += 1000) w.onTemperature(35.0, t);
    w.onTemperature(45.0, 5000);         // 跳变 10>0.1 → 锚点重置 45@5s
    for (int t = 6000; t <= 14000; t += 1000) {
        w.onTemperature(45.0, t);
        EXPECT_EQ(cnt.stable, 0) << "t=" << t;   // @10s 窗仅 5s——重锚生效不早判
    }
    w.onTemperature(45.0, 15000);        // 窗满 10s（15s-5s）
    EXPECT_EQ(cnt.stable, 1);
    EXPECT_EQ(w.state(), WarmupSequence::State::Done);
}

// —— 5. StableButFarNoJudge：稳定 10s+ 但离目标 10℃ → 不判稳，超时兜底 ——
TEST(WarmupSequence, StableButFarNoJudge) {
    WarmupConfig cfg;
    cfg.timeoutMs = 20000;               // 比稳定演示窗长——超时另走 tick
    WarmupSequence w(cfg);
    Counters cnt;
    cnt.bind(w);
    w.start(50.0);
    for (int t = 0; t <= 12000; t += 2000) w.onTemperature(40.0, t);  // 稳但差 10℃
    EXPECT_EQ(cnt.stable, 0);            // 窗满 12s 也不判稳
    EXPECT_EQ(cnt.timeout, 0);
    w.tick(20000);                       // 时间照走——超时兜底
    EXPECT_EQ(cnt.timeout, 1);
    EXPECT_EQ(cnt.stable, 0);
}

// —— 6. DoneNoMoreCallbacks：判稳/超时后再喂温/tick → 无第二次回调 ——
TEST(WarmupSequence, DoneNoMoreCallbacks) {
    {   // 稳了分支
        WarmupSequence w;
        Counters cnt;
        cnt.bind(w);
        w.start(42.0);
        for (int t = 0; t <= 10000; t += 1000) w.onTemperature(40.05, t);  // @10s 判稳
        ASSERT_EQ(cnt.stable, 1);
        for (int t = 11000; t <= 13000; t += 1000) w.onTemperature(41.0, t);
        w.tick(15 * 60 * 1000);          // 远超默认超时——Done 不再超时
        EXPECT_EQ(cnt.stable, 1);
        EXPECT_EQ(cnt.timeout, 0);
        EXPECT_EQ(w.state(), WarmupSequence::State::Done);
        EXPECT_EQ(w.lastTemp(), 41.0);   // 报警快照照常更新
    }
    {   // 超时分支
        WarmupConfig cfg;
        cfg.timeoutMs = 100;
        WarmupSequence w(cfg);
        Counters cnt;
        cnt.bind(w);
        w.start(50.0);
        w.onTemperature(30.0, 0);
        w.tick(100);                     // 超时
        ASSERT_EQ(cnt.timeout, 1);
        w.onTemperature(30.0, 150);
        w.tick(300);
        EXPECT_EQ(cnt.timeout, 1);
        EXPECT_EQ(cnt.stable, 0);
    }
}

// —— 7. TimeoutNoStopHeating：超时只报不停——超时后即使稳+达标也无第三回调 ——
TEST(WarmupSequence, TimeoutNoStopHeating) {
    WarmupConfig cfg;
    cfg.timeoutMs = 1000;
    WarmupSequence w(cfg);
    Counters cnt;
    cnt.bind(w);
    w.start(50.0);
    for (int t = 0; t <= 750; t += 250) w.onTemperature(30.0, t);
    w.tick(1000);
    ASSERT_EQ(cnt.timeout, 1);           // 超时回调一次
    for (int t = 5000; t <= 20000; t += 1000) {
        w.onTemperature(48.0, t);        // 若仍 Heating 本会判稳（稳+离目标恰 2）
        w.tick(t);
    }
    EXPECT_EQ(cnt.timeout, 1);           // 无第二次超时
    EXPECT_EQ(cnt.stable, 0);            // 也无「稳了」补发——Done 全静默
    EXPECT_EQ(w.state(), WarmupSequence::State::Done);
}

// —— 8. IdleIgnoresTemp：未 start 喂温 → lastTemp 更新但不锚定不判定 ——
TEST(WarmupSequence, IdleIgnoresTemp) {
    WarmupSequence w;
    Counters cnt;
    cnt.bind(w);
    w.onTemperature(40.05, 0);           // Idle 喂温：只记 lastTemp（报警快照）
    EXPECT_EQ(w.lastTemp(), 40.05);
    EXPECT_EQ(w.state(), WarmupSequence::State::Idle);
    EXPECT_EQ(cnt.stable, 0);
    w.start(42.0);                       // 锚点=此后的首个喂入（@5s，非 Idle 期 @0）
    for (int t = 5000; t <= 10000; t += 1000) {
        w.onTemperature(40.05, t);
        EXPECT_EQ(cnt.stable, 0) << "t=" << t;   // 若 @0 被锚定，@10s 已判稳
    }
    w.onTemperature(40.05, 15000);       // 窗满 10s（15s-5s）
    EXPECT_EQ(cnt.stable, 1);
}

// —— 9. TickTimeoutWithoutTemp：Heating 中温度停更 → 纯 tick 也能超时 ——
TEST(WarmupSequence, TickTimeoutWithoutTemp) {
    WarmupConfig cfg;
    cfg.timeoutMs = 1000;
    WarmupSequence w(cfg);
    Counters cnt;
    cnt.bind(w);
    w.start(50.0);
    w.tick(0);                           // 首个 tick 立超时基准
    w.tick(500);
    EXPECT_EQ(cnt.timeout, 0);
    w.tick(1000);                        // 无任何温度喂入也超时
    EXPECT_EQ(cnt.timeout, 1);
    EXPECT_EQ(cnt.stable, 0);
    EXPECT_EQ(w.lastTemp(), 0.0);        // 从未喂温
}

// —— 10. RestartAfterDone：Done 后 start() 复位重入 Heating——锚点/基准全重开 ——
TEST(WarmupSequence, RestartAfterDone) {
    WarmupSequence w;
    Counters cnt;
    cnt.bind(w);
    w.start(42.0);
    for (int t = 0; t <= 10000; t += 1000) w.onTemperature(40.05, t);   // @10s 判稳
    ASSERT_EQ(cnt.stable, 1);
    w.start(42.0);                       // 复位重开：回 Heating、锚点全清
    EXPECT_EQ(w.state(), WarmupSequence::State::Heating);
    w.onTemperature(40.05, 20000);       // 旧锚若未清此处即判稳（窗 20s）
    EXPECT_EQ(cnt.stable, 1);            // 仍 1——锚点已重开为 @20s
    for (int t = 21000; t <= 29000; t += 1000) w.onTemperature(40.05, t);
    EXPECT_EQ(cnt.stable, 1);
    w.onTemperature(40.05, 30000);       // 新窗满 10s（30s-20s）
    EXPECT_EQ(cnt.stable, 2);
    EXPECT_EQ(cnt.timeout, 0);
}
