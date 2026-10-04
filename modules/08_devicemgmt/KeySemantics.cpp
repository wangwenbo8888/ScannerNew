// ============================================================================
// KeySemantics.cpp — 按键裁判实现（261002 按键交互域定稿 §3.2.2 手势总表；
// 12 槽逐格钉死见 test_key_semantics.cpp）
// ============================================================================
#include "KeySemantics.h"

namespace Scanner::device {

KeySemantics::KeySemantics(std::function<bool(KeyCategory)> gate, KeySemActions actions)
    : gate_(std::move(gate)), actions_(std::move(actions)) {}

void KeySemantics::onGesture(const serial::GestureEvent& g, const MenuState& menu) {
    using K = serial::KeyId;
    using G = serial::GestureEvent::Gesture;
    using Sub = MenuState::Substate;

    // ── 0. 逃生类优先（免门禁、全域一致、子态内照常执行）──
    if (g.key == K::Middle && g.gesture == G::Hold) {
        actions_.emergencyStop();                    // 急停：子态解散归 DeviceManager 收尾
        return;
    }
    if (g.key == K::Up && g.gesture == G::Hold) {
        actions_.backToMain();                       // 回主界面：含子态/调节全清
        return;
    }

    const bool inMenu = (menu.layer == 2);
    const bool inSubstate = inMenu && menu.substate != Sub::None;

    // ── 1. 子态分流（①体素密度调节 / ⑤重置确认；§3.3.3 子态取消语义）──
    if (inSubstate) {
        if (menu.substate == Sub::AdjustVoxel) {
            if (g.gesture == G::Short && g.key == K::Left)  { actions_.adjustDown(); return; }
            if (g.gesture == G::Short && g.key == K::Right) { actions_.adjustUp();   return; }
            if (g.gesture == G::Short && g.key == K::Middle) { actions_.menuSelect(); return; } // 确认退出
            actions_.substateCancelled();            // 其余任意键＝取消子态回浏览态
            return;
        }
        if (menu.substate == Sub::ConfirmReset) {
            if (g.gesture == G::Short && g.key == K::Middle) { actions_.menuSelect(); return; } // 确认执行
            actions_.substateCancelled();
            return;
        }
    }

    // ── 2. 类别定档（按手势＋界面）→ 门禁 → ──
    const auto askGate = [this](KeyCategory c) { return gate_ ? gate_(c) : true; };

    switch (g.key) {
    case K::Middle:
        if (g.gesture == G::Short) {
            if (inMenu) {
                if (!askGate(KeyCategory::Menu)) { actions_.dropped("门禁"); return; }
                actions_.menuSelect();               // 浏览态选中当前项
            } else {
                if (!askGate(KeyCategory::StartStop)) { actions_.dropped("门禁"); return; }
                actions_.captureToggle();            // 主界面启停（同信号）
            }
        } else if (g.gesture == G::Double) {
            if (inMenu) { actions_.dropped("菜单内双击"); return; }   // 双击收紧（规则2）
            if (!askGate(KeyCategory::ModeSwitch)) { actions_.dropped("门禁"); return; }
            actions_.cycleMode();
        } else {
            actions_.dropped("预留");                // M/H 已被逃生类接管（不可达兜底）
        }
        break;

    case K::Up:
        if (g.gesture == G::Short) {
            if (!askGate(KeyCategory::Menu)) { actions_.dropped("门禁"); return; }
            if (inMenu) actions_.exitMenu();
            else actions_.enterMenu();
        } else if (g.gesture == G::Double) {
            if (inMenu) { actions_.dropped("菜单内双击"); return; }   // 双击收紧
            if (!askGate(KeyCategory::Adjust)) { actions_.dropped("门禁"); return; }
            actions_.toggleDepthOfField();           // 景深 近↔远 直切
        } else {
            actions_.dropped("预留");                // U/H 已被逃生类接管（不可达兜底）
        }
        break;

    case K::Left:
    case K::Right:
        if (g.gesture == G::Double) {
            if (g.key == K::Right) { actions_.dropped("预留"); return; }   // 右键双击预留
            if (inMenu) { actions_.dropped("菜单内双击"); return; }        // 双击收紧
            if (!askGate(KeyCategory::Adjust)) { actions_.dropped("门禁"); return; }
            actions_.switchAdjustCtx();              // 调节对象 亮度↔显示远近
            return;
        }
        if (g.gesture != G::Short) {
            actions_.dropped("预留");                // L/H、R/H 预留
            return;
        }
        if (inMenu) {                                // 菜单浏览态：游标（菜单优先于调节）
            if (!askGate(KeyCategory::Menu)) { actions_.dropped("门禁"); return; }
            // 261004 竖排方向对齐：菜单 UI 五项竖排（①顶→⑤底），而 L/R 键在主
            // 界面的语义是「调档↓/↑」——原映射 L＝cursorLeft（向①＝视觉上）致
            // 「按↓游标往上走」方向拧了（261004 用户实测报）。对齐为：L(↓)＝向⑤
            // （视觉下），R(↑)＝向①（视觉上）。MenuLogic 算子名 Left/Right 为
            // 历史命名（Left=-1 向① / Right=+1 向⑤），此处按视觉方向接线。
            if (g.key == K::Left) actions_.cursorRight();   // L(调档↓)＝游标向⑤（下）
            else actions_.cursorLeft();                     // R(调档↑)＝游标向①（上）
        } else {                                     // 主界面：调当前对象（二选一恒有对象）
            if (!askGate(KeyCategory::Adjust)) { actions_.dropped("门禁"); return; }
            if (g.key == K::Left) actions_.adjustDown();
            else actions_.adjustUp();
        }
        break;
    }
}

} // namespace Scanner::device
