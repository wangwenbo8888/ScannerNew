// ============================================================================
// test_menu_logic.cpp — MenuLogic 菜单记账本单测（K-T7）
//
// 纯记账钉死：转移表 = 08 文档 §4.2.3 弃奇偶版（显式状态，2026-08-18 设计
// §2.3 二次修订 + 08 设计 §2.4 状态源表）：
//   - 进菜单 cursor 复位 ①；ExitMenu 连带清调节上下文（口径钉死）；
//   - 游标 ①~④ 环绕（Left 1→4；Right 4→1）；
//   - modeCursor 默认=3 普通交叉（08 文档 §4.2.3 v2.0 原文），1→2→3→1 循环，
//     独立于菜单层；adjustCtx None→View→Brightness→None 循环；
//   - 步进只累计净步数（take 取走清零；Reset 连带清零——会话复位语义），
//     实际参数值归 ParamStore（后续任务），账本仅转发步进事件。
// 用例 = 实施计划 Task 7 十三条 + 口径钉死一条（ExitMenu 清调节上下文）。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/MenuLogic.h"

using namespace Scanner::device;
using Ctx = MenuState::AdjustCtx;

namespace {

// 快照逐字段断言（契约未给 operator==，按字段钉）
void expectState(const MenuLogic& m, int layer, int cursor, Ctx ctx, int modeCursor) {
    const MenuState s = m.state();
    EXPECT_EQ(s.layer, layer);
    EXPECT_EQ(s.cursor, cursor);
    EXPECT_EQ(s.adjustCtx, ctx);
    EXPECT_EQ(s.modeCursor, modeCursor);
}

} // namespace

// —— 1. DefaultState：默认 {layer=1, cursor=①, ctx=None, modeCursor=3} ——
TEST(MenuLogic, DefaultState) {
    MenuLogic m;
    expectState(m, 1, 1, Ctx::None, 3);
    EXPECT_EQ(m.adjustSteps(), 0);
}

// —— 2. EnterExitMenu：Enter→L2/cursor 复位 ①；Exit→L1（复位口径：再进必回 ①）——
TEST(MenuLogic, EnterExitMenu) {
    MenuLogic m;
    EXPECT_TRUE(m.apply(MenuOp::EnterMenu));       // L1→L2，cursor 复位 ①
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ①→②（制造非 ① 初值）
    EXPECT_TRUE(m.apply(MenuOp::ExitMenu));        // L2→L1（cursor 保留 ②——表中不动）
    expectState(m, 1, 2, Ctx::None, 3);
    EXPECT_TRUE(m.apply(MenuOp::EnterMenu));       // 再进：cursor 复位 ①（口径）
    expectState(m, 2, 1, Ctx::None, 3);
}

// —— 3. EnterMenuOnlyFromL1：L2 再 Enter → false 状态不变 ——
TEST(MenuLogic, EnterMenuOnlyFromL1) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    EXPECT_FALSE(m.apply(MenuOp::EnterMenu));
    expectState(m, 2, 1, Ctx::None, 3);
}

// —— 4. ExitMenuOnlyFromL2：L1 Exit → false ——
TEST(MenuLogic, ExitMenuOnlyFromL2) {
    MenuLogic m;
    EXPECT_FALSE(m.apply(MenuOp::ExitMenu));
    expectState(m, 1, 1, Ctx::None, 3);
}

// —— 5. CursorWrapLeft：L2 ① Left→④ 环绕；④ Left→③ ——
TEST(MenuLogic, CursorWrapLeft) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);                    // cursor=①
    EXPECT_TRUE(m.apply(MenuOp::CursorLeft));      // ①←④ 环绕：1→4
    EXPECT_EQ(m.state().cursor, 4);
    EXPECT_TRUE(m.apply(MenuOp::CursorLeft));      // 4→3
    EXPECT_EQ(m.state().cursor, 3);
}

// —— 6. CursorWrapRight：③ Right→④；④ Right→① 环绕（全环 ①→②→③→④→①）——
TEST(MenuLogic, CursorWrapRight) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ①→②
    EXPECT_EQ(m.state().cursor, 2);
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ②→③
    EXPECT_EQ(m.state().cursor, 3);
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ③→④
    EXPECT_EQ(m.state().cursor, 4);
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ④→① 环绕
    EXPECT_EQ(m.state().cursor, 1);
}

// —— 7. CursorOnlyInMenu：L1 Left/Right → false 不变（游标只在菜单层）——
TEST(MenuLogic, CursorOnlyInMenu) {
    MenuLogic m;
    EXPECT_FALSE(m.apply(MenuOp::CursorLeft));
    EXPECT_FALSE(m.apply(MenuOp::CursorRight));
    expectState(m, 1, 1, Ctx::None, 3);
}

// —— 8. CycleMode3Way：3→1→2→3→1（默认 3 起步，回环钉死）——
TEST(MenuLogic, CycleMode3Way) {
    MenuLogic m;                                   // modeCursor=3（默认普通交叉）
    EXPECT_TRUE(m.apply(MenuOp::CycleMode));       // 3→1
    EXPECT_EQ(m.state().modeCursor, 1);
    EXPECT_TRUE(m.apply(MenuOp::CycleMode));       // 1→2
    EXPECT_EQ(m.state().modeCursor, 2);
    EXPECT_TRUE(m.apply(MenuOp::CycleMode));       // 2→3
    EXPECT_EQ(m.state().modeCursor, 3);
    EXPECT_TRUE(m.apply(MenuOp::CycleMode));       // 3→1 回环
    EXPECT_EQ(m.state().modeCursor, 1);
}

// —— 9. CycleModeAnyLayer：L2 也生效（模式光标独立于菜单层，账本内无门禁）——
TEST(MenuLogic, CycleModeAnyLayer) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);                    // L2
    EXPECT_TRUE(m.apply(MenuOp::CycleMode));       // 3→1 无条件生效
    expectState(m, 2, 1, Ctx::None, 1);            // 菜单态不受扰
}

// —— 10. AdjustCtxCycle：None→View→Brightness→None ——
TEST(MenuLogic, AdjustCtxCycle) {
    MenuLogic m;
    EXPECT_TRUE(m.apply(MenuOp::CycleAdjustCtx));  // None→View
    EXPECT_EQ(m.state().adjustCtx, Ctx::View);
    EXPECT_TRUE(m.apply(MenuOp::CycleAdjustCtx));  // View→Brightness
    EXPECT_EQ(m.state().adjustCtx, Ctx::Brightness);
    EXPECT_TRUE(m.apply(MenuOp::CycleAdjustCtx));  // Brightness→None
    EXPECT_EQ(m.state().adjustCtx, Ctx::None);
}

// —— 11. AdjustStepsAccumulate：ctx=Brightness 后 Up,Up,Down → 净 +1；take 清零 ——
TEST(MenuLogic, AdjustStepsAccumulate) {
    MenuLogic m;
    m.apply(MenuOp::CycleAdjustCtx);               // View
    m.apply(MenuOp::CycleAdjustCtx);               // Brightness
    EXPECT_TRUE(m.apply(MenuOp::AdjustUp));        // +1
    EXPECT_TRUE(m.apply(MenuOp::AdjustUp));        // +1
    EXPECT_TRUE(m.apply(MenuOp::AdjustDown));      // −1
    EXPECT_EQ(m.adjustSteps(), 1);                 // 净步数 +1（未 take 可重复读）
    EXPECT_EQ(m.takeAdjustSteps(), 1);             // 取走
    EXPECT_EQ(m.adjustSteps(), 0);
    EXPECT_EQ(m.takeAdjustSteps(), 0);             // 已清零
}

// —— 12. AdjustOnlyInCtx：ctx=None 时 Up/Down → false 且步数不累计 ——
TEST(MenuLogic, AdjustOnlyInCtx) {
    MenuLogic m;                                   // ctx=None
    EXPECT_FALSE(m.apply(MenuOp::AdjustUp));
    EXPECT_FALSE(m.apply(MenuOp::AdjustDown));
    EXPECT_EQ(m.adjustSteps(), 0);
}

// —— 13. ResetAll：改乱（层/游标/模式/上下文/步进）后 Reset → 全默认 ——
TEST(MenuLogic, ResetAll) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::CursorRight);                  // cursor=②
    m.apply(MenuOp::CycleMode);                    // mode=1
    m.apply(MenuOp::CycleAdjustCtx);               // ctx=View
    m.apply(MenuOp::AdjustUp);                     // steps=+1
    EXPECT_TRUE(m.apply(MenuOp::Reset));
    expectState(m, 1, 1, Ctx::None, 3);
    EXPECT_EQ(m.takeAdjustSteps(), 0);             // 会话复位：未取走步进一并清
}

// —— 14. ExitMenuClearsAdjustCtx（口径钉死）：Exit 连带 adjustCtx=None ——
TEST(MenuLogic, ExitMenuClearsAdjustCtx) {
    MenuLogic m;
    m.apply(MenuOp::CycleAdjustCtx);               // ctx=View
    m.apply(MenuOp::EnterMenu);                    // L2（Enter 不动 ctx）
    m.apply(MenuOp::ExitMenu);                     // L2→L1 且 ctx 清 None
    expectState(m, 1, 1, Ctx::None, 3);
}
