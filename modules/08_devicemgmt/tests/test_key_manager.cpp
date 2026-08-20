// ============================================================================
// test_key_manager.cpp — KeyManager 手势判定单测（K-T6）
//
// 确定论：注入假 PC 时钟（FakePc.now 由用例拨动，与 tick 入参同域）；MCU 时刻
// 由事件自带。判定驱动双口径都测：事件优先（任一事件以 MCU 时刻全局扫）+
// tick 兜底（PC 域静默窗/达阈）。
// 用例 = 实施计划 Task 6 十二条 + 口径钉死五条（中长按压无手势/二次按压抖动
// 回退/二次按压长按/Hold 后新按压/他键事件驱动 S 到期）。口径全文见 KeyManager.h。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/KeyManager.h"

#include <cstdint>
#include <vector>

using namespace Scanner::device;
using Scanner::device::serial::KeyId;
using Scanner::device::serial::RawKeyEvent;
using G = KeyGesture::G;

namespace {

RawKeyEvent ev(KeyId k, bool pressed, uint32_t mcuMs) {
    RawKeyEvent e{};
    e.key = k; e.pressed = pressed; e.mcuMs = mcuMs;
    return e;
}

// 假 PC 时钟：now 由用例拨动
struct FakePc {
    int64_t now = 0;
    KeyManager::PcClock clock() { return [this] { return now; }; }
};

bool hasGesture(const std::vector<KeyGesture>& gs, KeyId k, G g, uint32_t mcuMs) {
    for (const auto& x : gs)
        if (x.key == k && x.gesture == g && x.mcuMs == mcuMs) return true;
    return false;
}

} // namespace

// —— 1. ShortTap：按下@0 松开@300 → tick 过静默窗 → S 锚=松开时刻 ——
TEST(KeyManager, ShortTap) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 300));   // pc=0 锚定
    EXPECT_TRUE(km.drain().empty());                // 静默窗未满不出手
    pc.now = 401; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 300u);
}

// —— 2. ShortNotPremature：松开@300 后 tick 仅推进 200ms（及 399ms 边界）→ 静默窗未满 ——
TEST(KeyManager, ShortNotPremature) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 300));   // pc=0 锚定
    pc.now = 200; km.tick(pc.now);
    EXPECT_TRUE(km.drain().empty());
    pc.now = 399; km.tick(pc.now);                  // 399<400 边界仍不满
    EXPECT_TRUE(km.drain().empty());
}

// —— 3. DoubleClick：窗口内二次按压 → 第二次松开即判 D（纯事件驱动），全程无 S ——
TEST(KeyManager, DoubleClick) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));   // 短按候选 → ShortPending@100
    km.onRawEvent(ev(KeyId::Middle, true, 350));    // 350-100=250 ∈ [20,400] 窗内二次按压
    EXPECT_TRUE(km.drain().empty());                // 第二次未松开前无产出
    km.onRawEvent(ev(KeyId::Middle, false, 420));
    auto gs = km.drain();                           // 无 tick
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Double);
    EXPECT_EQ(gs[0].mcuMs, 420u);
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 4. Hold：达阈即判（tick 兜底路径，799/800 边界）；其后松开不再出事件 ——
TEST(KeyManager, HoldViaTick) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));      // pc=0 锚定
    pc.now = 799; km.tick(pc.now);
    EXPECT_TRUE(km.drain().empty());                // 799<800 未达阈
    pc.now = 800; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Hold);
    EXPECT_EQ(gs[0].mcuMs, 800u);                   // H 锚=按下+holdMs（达阈时刻）
    km.onRawEvent(ev(KeyId::Middle, false, 1200));  // 迟到松开被吞
    EXPECT_TRUE(km.drain().empty());
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 5. HoldJudgeOnEvent：事件优先——松开永不来，他键/本键事件@900 直接判 H ——
TEST(KeyManager, HoldViaEvent) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Up, true, 900));        // 他键事件驱动全局扫（M 0+800 达阈）
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Hold);
    EXPECT_EQ(gs[0].mcuMs, 800u);

    // 变体：本键迟到的松开@900——先按时间扫判 H@800，松开本身被吞
    KeyManager km2(GestureThresholds{}, pc.clock());
    km2.onRawEvent(ev(KeyId::Middle, true, 0));
    km2.onRawEvent(ev(KeyId::Middle, false, 900));
    gs = km2.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Hold);
    EXPECT_EQ(gs[0].mcuMs, 800u);
    EXPECT_EQ(km2.droppedCount(), 0u);
}

// —— 6. DebouncePressRelease：按→松 <20ms 抖动作废（无手势不计丢弃）；后续按压不受扰 ——
TEST(KeyManager, DebouncePressRelease) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 15));    // <20 本次按压作废
    pc.now = 1000; km.tick(pc.now);
    EXPECT_TRUE(km.drain().empty());
    EXPECT_EQ(km.droppedCount(), 0u);               // 抖动≠丢弃（不作对账告警）

    km.onRawEvent(ev(KeyId::Middle, true, 100));    // 100-15=85 ≥20 新按压
    km.onRawEvent(ev(KeyId::Middle, false, 400));   // pc=1000 锚定
    pc.now = 1401; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 400u);
}

// —— 7. DebounceBetweenPresses：松开→同键再按下 <20ms=抖动延续（不开新按压、
//      不判双击，与窗内二次按压的区分见用例 3）；其间散落松开按「无对应按下」计数 ——
TEST(KeyManager, DebounceBetweenPresses) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));   // ShortPending@100（pc=0）
    km.onRawEvent(ev(KeyId::Middle, true, 110));    // 10<20 同键抖动 → 忽略
    km.onRawEvent(ev(KeyId::Middle, false, 115));   // 无对应接受按压 → dropped
    EXPECT_EQ(km.droppedCount(), 1u);
    pc.now = 501; km.tick(pc.now);                  // 静默窗自首击松开@100 起
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Short);             // 判 S 不判 D
    EXPECT_EQ(gs[0].mcuMs, 100u);
    EXPECT_EQ(km.droppedCount(), 1u);               // 不重复计
}

// —— 8. MutualExclusion：双击全程仅 1 手势；事后 tick 兜底亦不追补 ——
TEST(KeyManager, MutualExclusion) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));
    km.onRawEvent(ev(KeyId::Middle, true, 350));
    km.onRawEvent(ev(KeyId::Middle, false, 420));
    pc.now = 5000; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Double);
    EXPECT_EQ(gs[0].mcuMs, 420u);
}

// —— 9. OutOfOrderDropped：倒序按下（早于该键已处理时刻）→ 丢弃且不污染状态 ——
TEST(KeyManager, OutOfOrderDropped) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 300));   // lastMcu=300
    pc.now = 401; km.tick(pc.now);
    ASSERT_EQ(km.drain().size(), 1u);               // S@300 已出
    km.onRawEvent(ev(KeyId::Middle, true, 200));    // 倒序（200<300）
    EXPECT_EQ(km.droppedCount(), 1u);
    pc.now = 2000; km.tick(pc.now);                 // 若被污染将追补 H——必须为空
    EXPECT_TRUE(km.drain().empty());
}

// —— 10. ReleaseWithoutPress：凭空松开 → 丢弃计数，无手势 ——
TEST(KeyManager, ReleaseWithoutPress) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, false, 500));
    EXPECT_EQ(km.droppedCount(), 1u);
    pc.now = 5000; km.tick(pc.now);
    EXPECT_TRUE(km.drain().empty());
}

// —— 11. FourKeysIndependent：四键各自状态机互不干扰（U 长按中 M 短按完成，同一拍双出）——
TEST(KeyManager, FourKeysIndependent) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Up, true, 0));          // U 按下（将长按）
    km.onRawEvent(ev(KeyId::Middle, true, 10));
    km.onRawEvent(ev(KeyId::Middle, false, 300));   // M 短按完成
    pc.now = 900; km.tick(pc.now);                  // U 达阈 + M 静默窗满
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 2u);
    EXPECT_TRUE(hasGesture(gs, KeyId::Up, G::Hold, 800u));
    EXPECT_TRUE(hasGesture(gs, KeyId::Middle, G::Short, 300u));
    km.onRawEvent(ev(KeyId::Up, false, 1200));      // U 迟到松开被吞
    EXPECT_TRUE(km.drain().empty());
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 12. DifferentKeysNoDebounceCross：消抖仅同键——异键 <20ms 各自正常判定 ——
TEST(KeyManager, DifferentKeysNoDebounceCross) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Up, true, 0));
    km.onRawEvent(ev(KeyId::Up, false, 100));       // U ShortPending@100（pc=0）
    km.onRawEvent(ev(KeyId::Middle, true, 110));    // M 按下距 U 松开仅 10ms——不受扰
    km.onRawEvent(ev(KeyId::Middle, false, 200));   // M ShortPending@200（pc=0）
    pc.now = 600; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 2u);
    EXPECT_TRUE(hasGesture(gs, KeyId::Up, G::Short, 100u));
    EXPECT_TRUE(hasGesture(gs, KeyId::Middle, G::Short, 200u));
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 13. MediumPressNoGesture（口径）：400<按压<800 既非 S 候选也够不到 H →
//        无手势不计数；其后新按压照常 ——
TEST(KeyManager, MediumPressNoGesture) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 600));
    pc.now = 10000; km.tick(pc.now);
    EXPECT_TRUE(km.drain().empty());
    EXPECT_EQ(km.droppedCount(), 0u);
    km.onRawEvent(ev(KeyId::Middle, true, 700));    // 700-600=100 ≥20 新按压
    km.onRawEvent(ev(KeyId::Middle, false, 800));   // pc=10000 锚定
    pc.now = 10500; km.tick(pc.now);
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 800u);
}

// —— 14. SecondPressBounceAnnulsDouble（口径）：双击的二次按压 <20ms=抖动 →
//        回退首击 ShortPending（锚不丢），最终判 S 不判 D，不计丢弃 ——
TEST(KeyManager, SecondPressBounceAnnulsDouble) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));
    km.onRawEvent(ev(KeyId::Middle, true, 350));
    km.onRawEvent(ev(KeyId::Middle, false, 358));   // 二次按压 8ms 抖动作废
    pc.now = 501; km.tick(pc.now);                  // 窗自首击松开@100 起（pc=0）
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 100u);
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 15. SecondPressHold（口径）：双击的第二次按压长按达阈 → 判 H（首击静默
//        取消，不出 S/D）；其后松开被吞 ——
TEST(KeyManager, SecondPressHold) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));
    km.onRawEvent(ev(KeyId::Middle, true, 350));    // 二次按压 Down@350
    km.onRawEvent(ev(KeyId::Up, true, 1150));       // 他键事件驱动：350+800 达阈
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Hold);
    EXPECT_EQ(gs[0].mcuMs, 1150u);
    km.onRawEvent(ev(KeyId::Middle, false, 1300));  // 迟到松开被吞
    EXPECT_TRUE(km.drain().empty());
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 16. HoldThenNewTap（口径）：H 后 PostHold 只吞一次迟到松开，其后新按压=
//        全新操作 ——
TEST(KeyManager, HoldThenNewTap) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 900));   // 事件驱动 H@800（松开本身被吞）
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Hold);
    EXPECT_EQ(gs[0].mcuMs, 800u);
    km.onRawEvent(ev(KeyId::Middle, true, 1300));   // 新按压（1300-900=400 ≥20）
    km.onRawEvent(ev(KeyId::Middle, false, 1400));  // pc=0 锚定
    pc.now = 401; km.tick(pc.now);                  // 静默窗自@1400 起
    gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 1400u);
    EXPECT_EQ(km.droppedCount(), 0u);
}

// —— 17. ShortExpiryViaOtherKeyEvent（事件优先）：静默窗由他键事件时间戳到期 ——
TEST(KeyManager, ShortExpiryViaOtherKeyEvent) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 300));
    km.onRawEvent(ev(KeyId::Up, true, 800));        // 800-300=500 ≥400 → M 的 S 到期
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].key, KeyId::Middle);
    EXPECT_EQ(gs[0].gesture, G::Short);
    EXPECT_EQ(gs[0].mcuMs, 300u);
}

// —— 18. DuplicatePressWhileDown（口径）：已接受按压后的再次按下=协议异常 →
//        dropped++（不得误按「松开后抖动」吞掉）；双击链不受扰 ——
TEST(KeyManager, DuplicatePressWhileDown) {
    FakePc pc;
    KeyManager km(GestureThresholds{}, pc.clock());
    km.onRawEvent(ev(KeyId::Middle, true, 0));
    km.onRawEvent(ev(KeyId::Middle, false, 100));   // ShortPending@100
    km.onRawEvent(ev(KeyId::Middle, true, 350));    // 窗内二次按压（已接受）
    km.onRawEvent(ev(KeyId::Middle, true, 360));    // 距前次按下仅 10ms——但非松开后抖动
    EXPECT_EQ(km.droppedCount(), 1u);               // → 重复按下计数，不静默吞
    km.onRawEvent(ev(KeyId::Middle, false, 420));   // 第二次按压（@350）正常收尾
    auto gs = km.drain();
    ASSERT_EQ(gs.size(), 1u);
    EXPECT_EQ(gs[0].gesture, G::Double);
    EXPECT_EQ(gs[0].mcuMs, 420u);
}
