#pragma once
// ============================================================================
// KeySemantics.h — 按键裁判（2026-08-18 设计 §2.4 弃奇偶版；2026-08-20 08 设计 §4；
// 260831 批1-C：输入换 serial::GestureEvent）
//
// 一条规则：手势+当前状态 → 状态转移表 → 执行动作 + 状态变为何。
// 手势判定归 MCU（G01 帧 S/D/H 位）——本类只做语义映射，无 PC 侧判定
// （KeyManager 已退役）；纯函数式映射：无内部计数、不碰硬件、发令统一经
// DeviceManager。门禁由注入 gate() 判——不让按→丢弃记日志。依赖方向：
// 裁判→账本/黑板（单向只读）。
// 丢弃口径：「预留」= 协议可达、产品未定义（G01 手势位可达但转移表未定义动作）。
// ============================================================================
#include "MenuLogic.h"
#include "serial/McuFrame.h"

#include <functional>

namespace Scanner::device {

struct KeySemActions {        // 判出的动作出口（DeviceManager 接）
    std::function<void()> captureToggle;   // 中键短按启停（→DeviceManager 查黑板发 N10/N11 H0）
                                            //  capturing 真假裁判不辨（启停同信号），回调里查黑板定值
    std::function<void()> menuSelect;      // 中键短按 layer=2 选中当前项（①②=视野项→设模式
                                            //  ③=扫描完成→派工作流 ④=开始后处理→派工作流；
                                            //  DeviceManager 按 cursor 分叉，此处只给信号）
    std::function<void()> cycleMode;       // 中键双击（→MenuLogic.CycleMode）
    std::function<void()> enterMenu;       // 上键短按 layer=1（→EnterMenu）
    std::function<void()> exitMenu;        // 上键短按 layer=2（→ExitMenu）
    std::function<void()> cycleAdjustCtx;  // 上键双击（→CycleAdjustCtx）
    std::function<void()> adjustUp;        // 右键短按 ctx≠None（→MenuLogic.AdjustUp + ParamStore）
    std::function<void()> adjustDown;      // 左键短按 ctx≠None（→MenuLogic.AdjustUp/Down）
    std::function<void()> cursorLeft;      // 左键短按 layer=2（→CursorLeft）
    std::function<void()> cursorRight;     // 右键短按 layer=2（→CursorRight）
    std::function<void(const char* why)> dropped;  // 门禁拒/协议可达未定义手势丢弃（记日志）
};

class KeySemantics {
public:
    // gate：这类键现在让按吗（DeviceManager 注入——采集态门控 M1 规矩）。
    // 门控分类（M1 定案，测试钉死）：启停键=M/S 主层不问 gate（采集态门控由
    // capturing 状态自身表达，DeviceManager 保证只有可采集态才可能收到）；
    // 菜单/模式/调节键（含 M/S layer=2 的 menuSelect）=gate 关一律丢弃。
    KeySemantics(std::function<bool()> gate, KeySemActions actions);

    // 处理一个手势（MCU G01 已判 S/D/H）：查门禁→查状态源→转移表→触发动作
    // （动作内做不做账本变更由 DeviceManager 在动作回调里调 MenuLogic.apply——
    // 本类不直接碰账本）。左右键短按口径：菜单优先于调节——layer=2 一律游标
    // （即使 ctx≠None；进菜单时 ctx 被 Exit 清，理论不并存，转移表定死优先级）。
    void onGesture(const serial::GestureEvent& ev, const MenuState& menu);

private:
    std::function<bool()> gate_;   // 菜单类键门禁（每手势现问——DeviceManager 按当前态给）
    KeySemActions actions_;        // 动作出口
};

} // namespace Scanner::device
