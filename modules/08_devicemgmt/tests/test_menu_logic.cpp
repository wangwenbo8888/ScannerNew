// ============================================================================
// test_menu_logic.cpp — MenuLogic 菜单记账本单测（261002 按键交互域定稿版）
//
// 纯记账钉死（设计 §3.3.3 五项重定义）：
//   - 进菜单 cursor 复位 ①/子态清；游标 ①~⑤ 环绕（Left 1→5；Right 5→1）
//   - ①⑤子态：EnterAdjustSubstate（游标①）/EnterConfirmSubstate（游标⑤）/
//     CancelSubstate（回浏览态不退菜单）/浏览态游标在子态内无效
//   - ②③④选中＝执行并退菜单的记账归 DeviceManager（本类只 ExitMenu）
//   - 调节对象二选一（亮度默认↔显示远近）；主界面/①子态步进记账，浏览态无效
//   - BackToMain：清层/游标/子态，调节对象回默认亮度；modeCursor 不动（业务态）
//   - Reset 全默认；takeAdjustSteps 取走清零
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/MenuLogic.h"

using namespace Scanner::device;
using Ctx = MenuState::AdjustCtx;
using Sub = MenuState::Substate;

namespace {

// 快照逐字段断言（契约未给 operator==，按字段钉）
void expectState(const MenuLogic& m, int layer, int cursor, Sub sub, Ctx ctx, int modeCursor) {
    const MenuState s = m.state();
    EXPECT_EQ(s.layer, layer);
    EXPECT_EQ(s.cursor, cursor);
    EXPECT_EQ(s.substate, sub);
    EXPECT_EQ(s.adjustCtx, ctx);
    EXPECT_EQ(s.modeCursor, modeCursor);
}

} // namespace

// —— 1. DefaultState：默认 {layer=1, cursor=①, 子态=无, ctx=Brightness, modeCursor=3} ——
TEST(MenuLogic, DefaultState) {
    MenuLogic m;
    expectState(m, 1, 1, Sub::None, Ctx::Brightness, 3);
    EXPECT_EQ(m.adjustSteps(), 0);
}

// —— 2. EnterExitMenu：Enter→L2/cursor 复位 ①/子态清；Exit→L1 ——
TEST(MenuLogic, EnterExitMenu) {
    MenuLogic m;
    EXPECT_TRUE(m.apply(MenuOp::EnterMenu));
    EXPECT_TRUE(m.apply(MenuOp::CursorRight));     // ①→②（制造非 ① 初值）
    EXPECT_TRUE(m.apply(MenuOp::ExitMenu));        // L2→L1（cursor 保留——表中不动）
    expectState(m, 1, 2, Sub::None, Ctx::Brightness, 3);
    EXPECT_TRUE(m.apply(MenuOp::EnterMenu));       // 再进：cursor 复位 ①（口径）
    expectState(m, 2, 1, Sub::None, Ctx::Brightness, 3);
}

// —— 3. CursorWrapFive：游标 ①~⑤ 环绕（Left 1→5；Right 5→1）——
TEST(MenuLogic, CursorWrapFive) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    EXPECT_TRUE(m.apply(MenuOp::CursorLeft));      // ①→⑤（左环绕）
    EXPECT_EQ(m.state().cursor, 5);
    for (int i = 0; i < 4; ++i) m.apply(MenuOp::CursorRight);   // ⑤→①→②→③→④
    EXPECT_EQ(m.state().cursor, 4);
    m.apply(MenuOp::CursorRight);                  // ④→⑤
    m.apply(MenuOp::CursorRight);                  // ⑤→①（右环绕）
    EXPECT_EQ(m.state().cursor, 1);
}

// —— 4. CursorBrowseOnly：子态内游标无效（取消走 CancelSubstate）——
TEST(MenuLogic, CursorBrowseOnly) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::EnterAdjustSubstate);          // ①子态（游标停①）
    EXPECT_FALSE(m.apply(MenuOp::CursorLeft));     // 子态内游标不动
    EXPECT_FALSE(m.apply(MenuOp::CursorRight));
    EXPECT_EQ(m.state().cursor, 1);
    EXPECT_TRUE(m.apply(MenuOp::CancelSubstate));  // 取消→浏览态（不退菜单）
    expectState(m, 2, 1, Sub::None, Ctx::Brightness, 3);
}

// —— 5. SubstatesEnter：①选中→调节子态；⑤选中→确认子态（游标前提钉死）——
TEST(MenuLogic, SubstatesEnter) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    EXPECT_TRUE(m.apply(MenuOp::EnterAdjustSubstate));    // 游标①：进调节子态
    EXPECT_EQ(m.state().substate, Sub::AdjustVoxel);
    EXPECT_TRUE(m.apply(MenuOp::CancelSubstate));
    EXPECT_FALSE(m.apply(MenuOp::EnterConfirmSubstate));  // 游标①≠⑤：拒
    for (int i = 0; i < 4; ++i) m.apply(MenuOp::CursorRight);   // ①→⑤
    EXPECT_TRUE(m.apply(MenuOp::EnterConfirmSubstate));   // 游标⑤：进确认子态
    EXPECT_EQ(m.state().substate, Sub::ConfirmReset);
}

// —— 6. SwitchAdjustCtxBinary：调节对象 亮度↔显示远近 二选一切换（无 None）——
TEST(MenuLogic, SwitchAdjustCtxBinary) {
    MenuLogic m;
    EXPECT_EQ(m.state().adjustCtx, Ctx::Brightness);      // 默认亮度
    m.apply(MenuOp::SwitchAdjustCtx);
    EXPECT_EQ(m.state().adjustCtx, Ctx::DisplayDistance);
    m.apply(MenuOp::SwitchAdjustCtx);
    EXPECT_EQ(m.state().adjustCtx, Ctx::Brightness);      // 来回切不出现第三态
}

// —— 7. AdjustStepsMainAndSubstate：主界面与①子态步进记账；浏览态无效 ——
TEST(MenuLogic, AdjustStepsMainAndSubstate) {
    MenuLogic m;
    EXPECT_TRUE(m.apply(MenuOp::AdjustUp));               // 主界面：恒有调节对象
    EXPECT_TRUE(m.apply(MenuOp::AdjustDown));
    EXPECT_EQ(m.adjustSteps(), 0);                        // +1-1 净零
    m.apply(MenuOp::EnterMenu);
    EXPECT_FALSE(m.apply(MenuOp::AdjustUp));              // 菜单浏览态：无效
    m.apply(MenuOp::EnterAdjustSubstate);
    EXPECT_TRUE(m.apply(MenuOp::AdjustUp));               // ①子态：体素密度步进
    EXPECT_EQ(m.adjustSteps(), 1);
    EXPECT_EQ(m.takeAdjustSteps(), 1);                    // 取走清零
    EXPECT_EQ(m.adjustSteps(), 0);
}

// —— 8. CycleMode：1→2→3→1，独立于层/子态（业务状态）——
TEST(MenuLogic, CycleMode) {
    MenuLogic m;
    m.apply(MenuOp::CycleMode);                           // 3→1
    m.apply(MenuOp::CycleMode);                           // 1→2
    EXPECT_EQ(m.state().modeCursor, 2);
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::CycleMode);                           // 2→3（菜单内也转——落地
    EXPECT_EQ(m.state().modeCursor, 3);                   //  归 KeySemantics 收紧管）
}

// —— 9. BackToMain：清层/游标/子态，调节对象回默认亮度；modeCursor 不动 ——
TEST(MenuLogic, BackToMain) {
    MenuLogic m;
    m.apply(MenuOp::CycleMode);                           // modeCursor 3→1
    m.apply(MenuOp::SwitchAdjustCtx);                     // 对象→显示远近
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::CursorRight);                         // 游标②
    m.apply(MenuOp::CursorRight);
    m.apply(MenuOp::CursorRight);
    m.apply(MenuOp::CursorRight);                         // 游标⑤
    m.apply(MenuOp::EnterConfirmSubstate);                // ⑤确认子态
    EXPECT_TRUE(m.apply(MenuOp::BackToMain));
    expectState(m, 1, 1, Sub::None, Ctx::Brightness, 1);  // 层/游标/子态/对象清，
    EXPECT_EQ(m.state().modeCursor, 1);                   //  模式不动（业务态保留）
}

// —— 10. Reset：全默认（含步进余额）——
TEST(MenuLogic, Reset) {
    MenuLogic m;
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::EnterAdjustSubstate);
    m.apply(MenuOp::AdjustUp);
    EXPECT_TRUE(m.apply(MenuOp::Reset));
    expectState(m, 1, 1, Sub::None, Ctx::Brightness, 3);
    EXPECT_EQ(m.adjustSteps(), 0);
}

// —— 11. ExitMenuKeepsCtx：退菜单不清调节对象（那是回主界面的活——口径钉死）——
TEST(MenuLogic, ExitMenuKeepsCtx) {
    MenuLogic m;
    m.apply(MenuOp::SwitchAdjustCtx);                     // 对象→显示远近
    m.apply(MenuOp::EnterMenu);
    m.apply(MenuOp::ExitMenu);
    EXPECT_EQ(m.state().adjustCtx, Ctx::DisplayDistance); // 保留（仅 BackToMain 回默认）
}
