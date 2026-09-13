// ============================================================================
// test_key_semantics.cpp — KeySemantics 按键裁判单测（K-T8）
//
// 转移表逐条钉死（08 文档 §4.2.3 弃奇偶版 + 实施计划 T8 表格）：
//   - M/S：主层=启停同信号（capturing 真假不影响裁判——黑板态 DeviceManager
//     在回调里查）；layer=2=选中当前项；
//   - M/D 任意态切模式；U/S 按层开关菜单；U/D 任意态换调节上下文；
//   - 菜单优先于调节（口径钉死）：L/S、R/S 在 layer=2 一律游标，即使 ctx≠None；
//   - 门禁口径（M1 分类）：注入 gate=「菜单类键生效吗」——M/S 主层启停不问
//     gate；其余（含 M/S layer=2 的 menuSelect）gate=false 全丢弃；
//   - 预留手势（M/H、U/H、L/D、R/D、L/H、R/H）与主层无上下文左右短按 → 丢弃。
// 全 Mock lambda 计数；capturing 不入裁判输入（测试 1/3 仅钉「启停同信号」）。
// ============================================================================

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include "modules/08_devicemgmt/KeySemantics.h"

using namespace Scanner::device;
using serial::KeyId;
using Gest = KeyGesture::G;
using Ctx = MenuState::AdjustCtx;

namespace {

// 动作出口计数账（全 Mock lambda）
struct Counters {
    int captureToggle = 0, menuSelect = 0, cycleMode = 0;
    int enterMenu = 0, exitMenu = 0, cycleAdjustCtx = 0;
    int adjustUp = 0, adjustDown = 0, cursorLeft = 0, cursorRight = 0;
    std::vector<std::string> drops;   // 丢弃原因序列

    int fired() const {               // 有效动作总数（不含 drops）
        return captureToggle + menuSelect + cycleMode + enterMenu + exitMenu
             + cycleAdjustCtx + adjustUp + adjustDown + cursorLeft + cursorRight;
    }
};

KeyGesture kg(KeyId k, Gest g) { return KeyGesture{k, g, 0}; }   // mcuMs 裁判不读

MenuState ms(int layer, Ctx ctx = Ctx::None) {
    MenuState m;
    m.layer = layer;
    m.adjustCtx = ctx;
    return m;
}

KeySemantics referee(Counters& c, std::function<bool()> gate = [] { return true; }) {
    return KeySemantics(std::move(gate), KeySemActions{
        [&] { ++c.captureToggle; },
        [&] { ++c.menuSelect; },
        [&] { ++c.cycleMode; },
        [&] { ++c.enterMenu; },
        [&] { ++c.exitMenu; },
        [&] { ++c.cycleAdjustCtx; },
        [&] { ++c.adjustUp; },
        [&] { ++c.adjustDown; },
        [&] { ++c.cursorLeft; },
        [&] { ++c.cursorRight; },
        [&](const char* why) { c.drops.emplace_back(why); },
    });
}

} // namespace

// —— 1. MidShortCapturingToggle：采集中（capturing=true）主层启停 → captureToggle 恰 1 ——
TEST(KeySemantics, MidShortCapturingToggle) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(1));   // capturing 在黑板，裁判同信号
    EXPECT_EQ(c.captureToggle, 1);
    EXPECT_EQ(c.fired(), 1);
    EXPECT_TRUE(c.drops.empty());
}

// —— 2. MidShortInMenuSelects：菜单层中键 → menuSelect（不 captureToggle）——
TEST(KeySemantics, MidShortInMenuSelects) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(2));
    EXPECT_EQ(c.menuSelect, 1);
    EXPECT_EQ(c.captureToggle, 0);
    EXPECT_EQ(c.fired(), 1);
}

// —— 3. MidShortReadyLayer1：就绪（capturing=false）主层启停 → captureToggle（启停同信号）——
TEST(KeySemantics, MidShortReadyLayer1) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(1));
    EXPECT_EQ(c.captureToggle, 1);                        // 与测试 1 同一信号——黑白不分
    EXPECT_EQ(c.fired(), 1);
}

// —— 4. MidDoubleCyclesMode：中键双击任意态 → cycleMode ——
TEST(KeySemantics, MidDoubleCyclesMode) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(1));
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(2, Ctx::View));
    EXPECT_EQ(c.cycleMode, 2);
    EXPECT_EQ(c.fired(), 2);
}

// —— 5. MidHoldDropped：中键长按 → dropped("预留")，无任何动作 ——
TEST(KeySemantics, MidHoldDropped) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Middle, Gest::Hold), ms(2));
    EXPECT_EQ(c.fired(), 0);
    ASSERT_EQ(c.drops.size(), 2u);
    EXPECT_EQ(c.drops[0], "预留");
    EXPECT_EQ(c.drops[1], "预留");
}

// —— 6. UpShortEnterMenu：上键短按 layer=1 → enterMenu ——
TEST(KeySemantics, UpShortEnterMenu) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(1));
    EXPECT_EQ(c.enterMenu, 1);
    EXPECT_EQ(c.exitMenu, 0);
    EXPECT_EQ(c.fired(), 1);
}

// —— 7. UpShortExitMenu：上键短按 layer=2 → exitMenu ——
TEST(KeySemantics, UpShortExitMenu) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(2));
    EXPECT_EQ(c.exitMenu, 1);
    EXPECT_EQ(c.enterMenu, 0);
    EXPECT_EQ(c.fired(), 1);
}

// —— 8. UpDoubleCycleAdjust：上键双击任意态 → cycleAdjustCtx ——
TEST(KeySemantics, UpDoubleCycleAdjust) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(1));
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(2));
    EXPECT_EQ(c.cycleAdjustCtx, 2);
    EXPECT_EQ(c.fired(), 2);
}

// —— 9. LeftShortCursor：左键短按 layer=2 → cursorLeft（ctx≠None 也游标——菜单优先）——
TEST(KeySemantics, LeftShortCursor) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(2));
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(2, Ctx::Brightness));  // 优先级钉死
    EXPECT_EQ(c.cursorLeft, 2);
    EXPECT_EQ(c.adjustDown, 0);
    EXPECT_EQ(c.fired(), 2);
}

// —— 10. RightShortCursor：右键短按 layer=2 → cursorRight（同上菜单优先）——
TEST(KeySemantics, RightShortCursor) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(2));
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(2, Ctx::View));       // 优先级钉死
    EXPECT_EQ(c.cursorRight, 2);
    EXPECT_EQ(c.adjustUp, 0);
    EXPECT_EQ(c.fired(), 2);
}

// —— 11. LeftShortAdjustDown：主层 ctx=Brightness 左短按 → adjustDown ——
TEST(KeySemantics, LeftShortAdjustDown) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(1, Ctx::Brightness));
    EXPECT_EQ(c.adjustDown, 1);
    EXPECT_EQ(c.cursorLeft, 0);
    EXPECT_EQ(c.fired(), 1);
}

// —— 12. RightShortAdjustUp：主层 ctx=View 右短按 → adjustUp ——
TEST(KeySemantics, RightShortAdjustUp) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(1, Ctx::View));
    EXPECT_EQ(c.adjustUp, 1);
    EXPECT_EQ(c.cursorRight, 0);
    EXPECT_EQ(c.fired(), 1);
}

// —— 13. LeftRightNoCtxDropped：主层 ctx=None 左/右短按 → dropped("无效") ——
TEST(KeySemantics, LeftRightNoCtxDropped) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(1));
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(1));
    EXPECT_EQ(c.fired(), 0);
    ASSERT_EQ(c.drops.size(), 2u);
    EXPECT_EQ(c.drops[0], "无效");
    EXPECT_EQ(c.drops[1], "无效");
}

// —— 14. GateBlocksMenuKeys：gate=false → M/S 主层启停仍放行，其余全 dropped("门禁") ——
TEST(KeySemantics, GateBlocksMenuKeys) {
    Counters c;
    auto k = referee(c, [] { return false; });           // 菜单类键全关
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(1));  // 启停不问门禁 → 放行
    EXPECT_EQ(c.captureToggle, 1);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(2));           // menuSelect 菜单类
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(1));               // enterMenu
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(2));               // exitMenu
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(1));          // cycleMode
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(1));              // cycleAdjustCtx
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(2));             // cursorLeft
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(2));            // cursorRight
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(1, Ctx::View));  // adjustDown
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(1, Ctx::View)); // adjustUp
    EXPECT_EQ(c.fired(), 1);                             // 全程仅启停放行
    ASSERT_EQ(c.drops.size(), 9u);
    for (const auto& d : c.drops) EXPECT_EQ(d, "门禁");
}

// —— 15. ReservedGestures：全部 Hold + 左右双击 → dropped("预留")，无动作 ——
TEST(KeySemantics, ReservedGestures) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Up, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Left, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Right, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Left, Gest::Double), ms(1));
    k.onGesture(kg(KeyId::Right, Gest::Double), ms(1));
    EXPECT_EQ(c.fired(), 0);
    ASSERT_EQ(c.drops.size(), 6u);
    for (const auto& d : c.drops) EXPECT_EQ(d, "预留");
}
