// ============================================================================
// KeySemantics.cpp — 按键裁判实现（K-T8；转移表=08 文档 §4.2.3 弃奇偶版，
// 测试 test_key_semantics.cpp 逐条钉死）
// ============================================================================
#include "KeySemantics.h"

namespace Scanner::device {

KeySemantics::KeySemantics(std::function<bool()> gate, KeySemActions actions)
    : gate_(std::move(gate)), actions_(std::move(actions)) {}

void KeySemantics::onGesture(const serial::GestureEvent& g, const MenuState& menu) {
    using K = serial::KeyId;
    using G = serial::GestureEvent::Gesture;

    // 一问门禁（M1 分类＋260927 两轮增补）：启停（M/S 主层）不问——采集门由
    // capturing 状态自身表达；切模式（M/D）不问——采集中实时切模式（N10 四管
    // 掩码重发）；调节键（U/D 切上下文＋L/R/S 步进）不问——采集态实时调档同
    // UI 滑条口径（260927 用户口径；ParamStore→N10 全参重发/相机直设链路已在。
    // 采集态 layer 恒=1——菜单键被门禁挡住进不去，L/R/S 无游标歧义）。
    // 其余菜单键（含 M/S layer=2 的 menuSelect、U/S 进退菜单、游标）gate 关一律丢弃。
    const bool startStop = g.key == K::Middle && g.gesture == G::Short && menu.layer == 1;
    const bool modeSwitch = g.key == K::Middle && g.gesture == G::Double;
    const bool adjustKey = (g.key == K::Up && g.gesture == G::Double) ||
                           ((g.key == K::Left || g.key == K::Right) && g.gesture == G::Short);
    if (!startStop && !modeSwitch && !adjustKey && !gate_()) {
        actions_.dropped("门禁");
        return;
    }

    // 二查状态源 → 转移表（弃奇偶版；左右键口径：菜单优先于调节）
    switch (g.key) {
    case K::Middle:
        if (g.gesture == G::Short) {
            if (menu.layer == 2) actions_.menuSelect();      // 菜单层选中当前项
            else actions_.captureToggle();                   // 主层启停（同信号不分启/停）
        } else if (g.gesture == G::Double) {
            actions_.cycleMode();                            // 任意态切模式
        } else {
            actions_.dropped("预留");                        // M/H
        }
        break;

    case K::Up:
        if (g.gesture == G::Short) {
            if (menu.layer == 1) actions_.enterMenu();
            else actions_.exitMenu();
        } else if (g.gesture == G::Double) {
            actions_.cycleAdjustCtx();                       // 任意态换调节上下文
        } else {
            actions_.dropped("预留");                        // U/H
        }
        break;

    case K::Left:
    case K::Right:
        if (g.gesture != G::Short) {
            actions_.dropped("预留");                        // L/D、R/D、L/H、R/H
            break;
        }
        if (menu.layer == 2) {                               // 菜单优先于调节（口径钉死）
            if (g.key == K::Left) actions_.cursorLeft();
            else actions_.cursorRight();
        } else if (menu.adjustCtx != MenuState::AdjustCtx::None) {
            if (g.key == K::Left) actions_.adjustDown();
            else actions_.adjustUp();
        } else {
            actions_.dropped("无效");                        // 主层无上下文：左右无义
        }
        break;
    }
}

} // namespace Scanner::device
