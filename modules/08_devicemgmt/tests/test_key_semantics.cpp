// ============================================================================
// test_key_semantics.cpp — KeySemantics 按键裁判单测（261002 按键交互域定稿版）
//
// 手势总表 12 槽逐格钉死（设计 §3.2.2）＋三条规则＋子态分流＋类别门禁：
//   - M/S：主界面=启停同信号；菜单层=选中（menuSelect）
//   - M/D：仅主界面切模式（菜单内双击=丢弃收紧）；M/H=急停（全域免门禁）
//   - U/S：按层进/退菜单；U/D：仅主界面景深直切；U/H=回主界面（全域免门禁）
//   - L/D：仅主界面换调节对象；R/D 预留；L/R/S：菜单=游标（优先）/主界面=调档
//     （调节对象二选一恒有——无「无上下文无效」路径）
//   - 子态：①体素密度（L/R 步进、M 确认、其余取消）；⑤重置确认（M 执行、
//     其余取消）——逃生类优先（子态内 M/H、U/H 照常执行）
//   - 门禁 gate(类别)：四类各问；逃生类不问（gate=false 也执行）
// 全 Mock lambda 计数；capturing 不入裁判输入（启停同信号）。
// ============================================================================

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <vector>

#include "modules/08_devicemgmt/KeySemantics.h"

using namespace Scanner::device;
using serial::KeyId;
using Gest = serial::GestureEvent::Gesture;
using Ctx = MenuState::AdjustCtx;
using Sub = MenuState::Substate;

namespace {

// 动作出口计数账（全 Mock lambda）
struct Counters {
    int captureToggle = 0, menuSelect = 0, cycleMode = 0;
    int emergencyStop = 0, backToMain = 0;
    int enterMenu = 0, exitMenu = 0, toggleDepthOfField = 0, switchAdjustCtx = 0;
    int adjustUp = 0, adjustDown = 0, cursorLeft = 0, cursorRight = 0;
    int substateCancelled = 0;
    std::vector<std::string> drops;   // 丢弃原因序列
    std::vector<KeyCategory> asked;   // gate 收到的类别序列

    int fired() const {               // 有效动作总数（不含 drops/cancelled）
        return captureToggle + menuSelect + cycleMode + emergencyStop + backToMain
             + enterMenu + exitMenu + toggleDepthOfField + switchAdjustCtx
             + adjustUp + adjustDown + cursorLeft + cursorRight;
    }
};

serial::GestureEvent kg(KeyId k, Gest g) { return serial::GestureEvent{k, g, 0}; }   // 手势直喂

MenuState ms(int layer, Ctx ctx = Ctx::Brightness) {
    MenuState m;
    m.layer = layer;
    m.adjustCtx = ctx;
    return m;
}

MenuState msSub(Sub s) {              // 菜单层子态（游标随子态：①=1 ⑤=5）
    MenuState m;
    m.layer = 2;
    m.substate = s;
    m.cursor = s == Sub::AdjustVoxel ? 1 : 5;
    return m;
}

KeySemantics referee(Counters& c,
                     std::function<bool(KeyCategory)> gate =
                         [](KeyCategory) { return true; }) {
    KeySemActions a;
    a.captureToggle = [&] { ++c.captureToggle; };
    a.menuSelect = [&] { ++c.menuSelect; };
    a.cycleMode = [&] { ++c.cycleMode; };
    a.emergencyStop = [&] { ++c.emergencyStop; };
    a.backToMain = [&] { ++c.backToMain; };
    a.enterMenu = [&] { ++c.enterMenu; };
    a.exitMenu = [&] { ++c.exitMenu; };
    a.toggleDepthOfField = [&] { ++c.toggleDepthOfField; };
    a.switchAdjustCtx = [&] { ++c.switchAdjustCtx; };
    a.adjustUp = [&] { ++c.adjustUp; };
    a.adjustDown = [&] { ++c.adjustDown; };
    a.cursorLeft = [&] { ++c.cursorLeft; };
    a.cursorRight = [&] { ++c.cursorRight; };
    a.substateCancelled = [&] { ++c.substateCancelled; };
    a.dropped = [&](const char* why) { c.drops.emplace_back(why); };
    return KeySemantics(
        [&c, gate](KeyCategory cat) {
            c.asked.push_back(cat);
            return gate(cat);
        },
        std::move(a));
}

} // namespace

// —— 1. MidShortMainToggle：主界面中键短按 → captureToggle（启停同信号）——
TEST(KeySemantics, MidShortMainToggle) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(1));
    EXPECT_EQ(c.captureToggle, 1);
    EXPECT_EQ(c.fired(), 1);
    EXPECT_TRUE(c.drops.empty());
    ASSERT_EQ(c.asked.size(), 1u);
    EXPECT_EQ(c.asked[0], KeyCategory::StartStop);        // 类别定档钉死
}

// —— 2. MidShortInMenuSelects：菜单层中键短按 → menuSelect ——
TEST(KeySemantics, MidShortInMenuSelects) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(2));
    EXPECT_EQ(c.menuSelect, 1);
    EXPECT_EQ(c.captureToggle, 0);
    ASSERT_EQ(c.asked.size(), 1u);
    EXPECT_EQ(c.asked[0], KeyCategory::Menu);
}

// —— 3. MidDoubleMainCyclesMode：中键双击主界面 → cycleMode；菜单内 → 丢弃收紧 ——
TEST(KeySemantics, MidDoubleMainCyclesMode) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(1));
    EXPECT_EQ(c.cycleMode, 1);
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(2));  // 菜单内双击＝丢弃（规则2）
    EXPECT_EQ(c.cycleMode, 1);
    ASSERT_EQ(c.drops.size(), 1u);
    EXPECT_EQ(c.drops[0], "菜单内双击");
    ASSERT_EQ(c.asked.size(), 1u);                        // 收紧在手禁前＝菜单内不问门禁
    EXPECT_EQ(c.asked[0], KeyCategory::ModeSwitch);
}

// —— 4. MidHoldEmergencyStop：中键长按＝急停——261004 用户指令暂时停用：
//      两长按手势按「预留」丢弃留痕（不执行、不问门禁）；恢复见 KeySemantics 注释 ——
TEST(KeySemantics, MidHoldEmergencyStopDisabled) {
    Counters c;
    auto k = referee(c, [](KeyCategory) { return false; });   // 门禁全关
    k.onGesture(kg(KeyId::Middle, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Middle, Gest::Hold), ms(2));
    EXPECT_EQ(c.emergencyStop, 0);                          // 停用：不执行
    EXPECT_TRUE(c.asked.empty());                           // 仍不问门禁（丢弃在门禁前）
    ASSERT_EQ(c.drops.size(), 2u);                          // 丢弃留痕恰两条
    EXPECT_NE(c.drops[0].find("逃生类已停用"), std::string::npos);
}

// —— 5. UpShortEnterExit：上键短按 按层进/退菜单 ——
TEST(KeySemantics, UpShortEnterExit) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(1));
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(2));
    EXPECT_EQ(c.enterMenu, 1);
    EXPECT_EQ(c.exitMenu, 1);
    EXPECT_EQ(c.fired(), 2);
}

// —— 6. UpDoubleDepthToggle：上键双击主界面 → 景深直切；菜单内 → 收紧丢弃 ——
TEST(KeySemantics, UpDoubleDepthToggle) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(1));
    EXPECT_EQ(c.toggleDepthOfField, 1);
    ASSERT_EQ(c.asked.size(), 1u);
    EXPECT_EQ(c.asked[0], KeyCategory::Adjust);           // 景深归调节全家
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(2));
    EXPECT_EQ(c.toggleDepthOfField, 1);
    ASSERT_EQ(c.drops.size(), 1u);
    EXPECT_EQ(c.drops[0], "菜单内双击");
}

// —— 7. UpHoldBackToMain：上键长按＝回主界面——261004 暂时停用（丢弃留痕）——
TEST(KeySemantics, UpHoldBackToMainDisabled) {
    Counters c;
    auto k = referee(c, [](KeyCategory) { return false; });
    k.onGesture(kg(KeyId::Up, Gest::Hold), ms(2, Ctx::DisplayDistance));
    k.onGesture(kg(KeyId::Up, Gest::Hold), msSub(Sub::ConfirmReset));
    EXPECT_EQ(c.backToMain, 0);                            // 停用：不执行
    EXPECT_TRUE(c.asked.empty());
    ASSERT_EQ(c.drops.size(), 2u);
    EXPECT_NE(c.drops[1].find("逃生类已停用"), std::string::npos);
}

// —— 8. LeftDoubleSwitchCtx：左键双击主界面 → 换调节对象；菜单内收紧；右双击预留 ——
TEST(KeySemantics, LeftDoubleSwitchCtx) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Double), ms(1));
    EXPECT_EQ(c.switchAdjustCtx, 1);
    k.onGesture(kg(KeyId::Left, Gest::Double), ms(2));
    EXPECT_EQ(c.switchAdjustCtx, 1);
    k.onGesture(kg(KeyId::Right, Gest::Double), ms(1));
    EXPECT_EQ(c.fired(), 1);
    ASSERT_EQ(c.drops.size(), 2u);
    EXPECT_EQ(c.drops[0], "菜单内双击");
    EXPECT_EQ(c.drops[1], "预留");
}

// —— 9. LeftRightShortMainAdjust：主界面左右短按 → 调档（对象二选一恒有——无 None 无效路径）——
TEST(KeySemantics, LeftRightShortMainAdjust) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(1, Ctx::Brightness));
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(1, Ctx::DisplayDistance));
    EXPECT_EQ(c.adjustDown, 1);
    EXPECT_EQ(c.adjustUp, 1);
    EXPECT_EQ(c.fired(), 2);
    EXPECT_TRUE(c.drops.empty());
}

// —— 10. LeftRightShortMenuCursor：菜单浏览态左右短按 → 游标（菜单优先于调节）——
TEST(KeySemantics, LeftRightShortMenuCursor) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(2));
    k.onGesture(kg(KeyId::Right, Gest::Short), ms(2));
    EXPECT_EQ(c.cursorLeft, 1);
    EXPECT_EQ(c.cursorRight, 1);
    EXPECT_EQ(c.adjustDown + c.adjustUp, 0);
}

// —— 11. AdjustSubstateVoxel：①子态——左右步进/中键确认/其余取消 ——
TEST(KeySemantics, AdjustSubstateVoxel) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Short), msSub(Sub::AdjustVoxel));
    k.onGesture(kg(KeyId::Right, Gest::Short), msSub(Sub::AdjustVoxel));
    EXPECT_EQ(c.adjustDown, 1);
    EXPECT_EQ(c.adjustUp, 1);
    k.onGesture(kg(KeyId::Middle, Gest::Short), msSub(Sub::AdjustVoxel));
    EXPECT_EQ(c.menuSelect, 1);                           // 确认退出
    k.onGesture(kg(KeyId::Up, Gest::Short), msSub(Sub::AdjustVoxel));
    k.onGesture(kg(KeyId::Middle, Gest::Double), msSub(Sub::AdjustVoxel));
    EXPECT_EQ(c.substateCancelled, 2);                    // 其余任意键＝取消
    EXPECT_TRUE(c.drops.empty());                         // 取消不是丢弃
}

// —— 12. ConfirmSubstateReset：⑤子态——中键确认执行/其余取消 ——
TEST(KeySemantics, ConfirmSubstateReset) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Middle, Gest::Short), msSub(Sub::ConfirmReset));
    EXPECT_EQ(c.menuSelect, 1);
    k.onGesture(kg(KeyId::Left, Gest::Short), msSub(Sub::ConfirmReset));
    k.onGesture(kg(KeyId::Up, Gest::Short), msSub(Sub::ConfirmReset));
    EXPECT_EQ(c.substateCancelled, 2);
    EXPECT_EQ(c.adjustDown, 0);                           // ⑤子态左右＝取消非步进
}

// —— 13. GateCategoryBlocks：gate(类别)=false → 对应手势拦下；类别各归其位 ——
TEST(KeySemantics, GateCategoryBlocks) {
    Counters c;
    auto k = referee(c, [](KeyCategory cat) { return cat != KeyCategory::Menu; });
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(1));   // StartStop 放行
    EXPECT_EQ(c.captureToggle, 1);
    k.onGesture(kg(KeyId::Middle, Gest::Double), ms(1));  // ModeSwitch 放行
    EXPECT_EQ(c.cycleMode, 1);
    k.onGesture(kg(KeyId::Up, Gest::Double), ms(1));      // Adjust 放行
    EXPECT_EQ(c.toggleDepthOfField, 1);
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(1));     // Adjust 放行
    EXPECT_EQ(c.adjustDown, 1);
    k.onGesture(kg(KeyId::Middle, Gest::Short), ms(2));   // Menu 拦
    k.onGesture(kg(KeyId::Up, Gest::Short), ms(1));       // Menu 拦
    k.onGesture(kg(KeyId::Left, Gest::Short), ms(2));     // Menu 拦（游标）
    EXPECT_EQ(c.fired(), 4);
    ASSERT_EQ(c.drops.size(), 3u);
    for (const auto& d : c.drops) EXPECT_EQ(d, "门禁");
}

// —— 14. ReservedGestures：左右长按 → dropped("预留")，无动作 ——
TEST(KeySemantics, ReservedGestures) {
    Counters c;
    auto k = referee(c);
    k.onGesture(kg(KeyId::Left, Gest::Hold), ms(1));
    k.onGesture(kg(KeyId::Right, Gest::Hold), ms(1));
    EXPECT_EQ(c.fired(), 0);
    ASSERT_EQ(c.drops.size(), 2u);
    for (const auto& d : c.drops) EXPECT_EQ(d, "预留");
}
