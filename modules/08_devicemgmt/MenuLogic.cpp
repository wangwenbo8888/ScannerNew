// ============================================================================
// MenuLogic.cpp — 菜单记账本实现（K-T7）；转移表与口径见 MenuLogic.h
// ============================================================================

#include "MenuLogic.h"

namespace Scanner::device {

bool MenuLogic::apply(MenuOp op) {
    switch (op) {
    case MenuOp::EnterMenu:                     // L1→L2，cursor 复位 ①（口径）
        if (st_.layer != 1) return false;
        st_.layer = 2;
        st_.cursor = 1;
        return true;
    case MenuOp::ExitMenu:                      // L2→L1，调节上下文一并清
        if (st_.layer != 2) return false;
        st_.layer = 1;
        st_.adjustCtx = MenuState::AdjustCtx::None;
        return true;
    case MenuOp::CursorLeft:                    // 仅菜单层：①←④ 环绕
        if (st_.layer != 2) return false;
        st_.cursor = (st_.cursor == 1) ? 4 : st_.cursor - 1;
        return true;
    case MenuOp::CursorRight:                   // 仅菜单层：④→① 环绕
        if (st_.layer != 2) return false;
        st_.cursor = (st_.cursor == 4) ? 1 : st_.cursor + 1;
        return true;
    case MenuOp::CycleMode:                     // 无条件：1→2→3→1
        st_.modeCursor = (st_.modeCursor % 3) + 1;
        return true;
    case MenuOp::CycleAdjustCtx:                // 无条件：None→View→Brightness→None
        switch (st_.adjustCtx) {
        case MenuState::AdjustCtx::None:       st_.adjustCtx = MenuState::AdjustCtx::View; break;
        case MenuState::AdjustCtx::View:       st_.adjustCtx = MenuState::AdjustCtx::Brightness; break;
        case MenuState::AdjustCtx::Brightness: st_.adjustCtx = MenuState::AdjustCtx::None; break;
        }
        return true;
    case MenuOp::AdjustUp:                      // 仅 ctx≠None：只记步进，不碰参数值
        if (st_.adjustCtx == MenuState::AdjustCtx::None) return false;
        ++steps_;
        return true;
    case MenuOp::AdjustDown:
        if (st_.adjustCtx == MenuState::AdjustCtx::None) return false;
        --steps_;
        return true;
    case MenuOp::Reset:                         // 全回默认（未取走步进一并清）
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
