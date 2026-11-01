// ============================================================================
// MenuLogic.cpp — 菜单记账本实现（261002 按键交互域定稿版）；转移表见 MenuLogic.h
// ============================================================================

#include "MenuLogic.h"

namespace Scanner::device {

bool MenuLogic::apply(MenuOp op) {
    switch (op) {
    case MenuOp::EnterMenu:                     // 主界面→菜单，cursor 复位 ①（口径）
        if (st_.layer != 1) return false;
        st_.layer = 2;
        st_.cursor = 1;
        st_.substate = MenuState::Substate::None;
        return true;
    case MenuOp::ExitMenu:                      // 菜单→主界面（子态一并清；调节对象
        if (st_.layer != 2) return false;       // 二选一不回默认——那是「回主界面」的活）
        st_.layer = 1;
        st_.substate = MenuState::Substate::None;
        return true;
    case MenuOp::CursorLeft:                    // 仅菜单浏览态：①←⑤ 环绕
        if (st_.layer != 2 || st_.substate != MenuState::Substate::None) return false;
        st_.cursor = (st_.cursor == 1) ? 5 : st_.cursor - 1;
        return true;
    case MenuOp::CursorRight:                   // 仅菜单浏览态：⑤→① 环绕
        if (st_.layer != 2 || st_.substate != MenuState::Substate::None) return false;
        st_.cursor = (st_.cursor == 5) ? 1 : st_.cursor + 1;
        return true;
    case MenuOp::CycleMode:                     // 无条件：1→2→3→1（业务状态，子态外也转）
        st_.modeCursor = (st_.modeCursor % 3) + 1;
        return true;
    case MenuOp::SwitchAdjustCtx:               // 调节对象 亮度↔显示远近 二选一切换
        st_.adjustCtx = (st_.adjustCtx == MenuState::AdjustCtx::Brightness)
                            ? MenuState::AdjustCtx::DisplayDistance
                            : MenuState::AdjustCtx::Brightness;
        return true;
    case MenuOp::AdjustUp:                      // 主界面（调当前对象）或①子态（体素密度）
        if (st_.layer == 1) { ++steps_; return true; }
        if (st_.substate == MenuState::Substate::AdjustVoxel) { ++steps_; return true; }
        return false;                           // 菜单浏览态/⑤子态：左右＝游标/取消，不步进
    case MenuOp::AdjustDown:
        if (st_.layer == 1) { --steps_; return true; }
        if (st_.substate == MenuState::Substate::AdjustVoxel) { --steps_; return true; }
        return false;
    case MenuOp::EnterAdjustSubstate:           // ①选中→体素密度调节子态（游标停①）
        if (st_.layer != 2 || st_.cursor != 1 ||
            st_.substate != MenuState::Substate::None) return false;
        st_.substate = MenuState::Substate::AdjustVoxel;
        return true;
    case MenuOp::EnterConfirmSubstate:          // ⑤选中→重置确认子态（游标停⑤）
        if (st_.layer != 2 || st_.cursor != 5 ||
            st_.substate != MenuState::Substate::None) return false;
        st_.substate = MenuState::Substate::ConfirmReset;
        return true;
    case MenuOp::CancelSubstate:                // ①⑤子态→菜单浏览态（不退菜单）
        if (st_.substate == MenuState::Substate::None) return false;
        st_.substate = MenuState::Substate::None;
        return true;
    case MenuOp::BackToMain:                    // 一键回主界面：清层/游标/子态，调节
        st_.layer = 1;                          // 对象回默认亮度；模式光标不动（业务态）
        st_.cursor = 1;
        st_.substate = MenuState::Substate::None;
        st_.adjustCtx = MenuState::AdjustCtx::Brightness;
        return true;
    case MenuOp::Reset:                         // 会话复位：全默认（未取走步进一并清）
        st_ = MenuState{};
        steps_ = 0;
        return true;
    }
    return false;                               // 不可达（枚举全覆盖）
}

MenuState MenuLogic::state() const { return st_; }

int MenuLogic::adjustSteps() const { return steps_; }

int MenuLogic::takeAdjustSteps() {
    const int n = steps_;
    steps_ = 0;
    return n;
}

} // namespace Scanner::device
