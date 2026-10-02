#pragma once
// ============================================================================
// KeySemantics.h — 按键裁判（261002 按键交互域定稿 §3.2 版）
//
// 一条规则：手势＋MenuState → 类别定档 → 门禁 gate(类别) → 转移表 → 动作出口。
// 纯函数式映射：不碰硬件、发令统一经 DeviceManager 回调。
//
// 门禁（§3.2.3）：gate 升级为按类别问话——逃生类（急停/回主界面）真免门禁
// 不问；启停/切模式/调节/菜单四类都问，app 注入闭包按「类别＋全局态」答话
// （08 内部态谓词在装配时组合进闭包；KeySemantics 保持纯函数，不认识全局态）。
//
// 双击收紧（§3.2.2 规则2）：双击一律仅主界面生效（切模式/景深直切/换调节
// 对象）；菜单里双击＝丢弃记日志（效果被菜单盖住看不见）。
//
// 长按全域一致（§3.2.2 规则3）：中键长按＝急停、上键长按＝回主界面，任何
// 界面/子态都执行；左右长按预留。
//
// 子态（§3.3.3）：①体素密度调节/⑤重置确认内——功能键照常（①:左右步进中键
// 确认；⑤:中键确认），其余任意键＝取消子态回菜单浏览态（逃生类优先照常执行）。
// ============================================================================
#include "serial/McuFrame.h"
#include "MenuLogic.h"

#include <functional>

namespace Scanner::device {

// 门禁类别（逃生类不进枚举——免门禁）：KeySemantics 按手势＋界面定类别，
// gate 闭包按「类别＋全局态」答话（true=放行交功能口细分 / false=拦下丢弃）
enum class KeyCategory : uint8_t {
    StartStop = 0,   // 启停（中键短按·主界面）
    ModeSwitch = 1,  // 切扫描模式（中键双击·主界面）
    Adjust = 2,      // 调节全家：档位步进/换调节对象（左双击）/景深直切（上双击）
    Menu = 3,        // 菜单类：进退/游标/选中
};

struct KeySemActions {        // 判出的动作出口（DeviceManager 接；功能口拒在其内做）
    std::function<void()> captureToggle;      // 中键短按·主界面（启停同信号——回调查黑板定值）
    std::function<void()> menuSelect;         // 中键短按·菜单层（浏览态=选中当前项；
                                              //  ①⑤子态=确认——DeviceManager 按子态分叉）
    std::function<void()> cycleMode;          // 中键双击·主界面（→MenuLogic.CycleMode）
    std::function<void()> emergencyStop;      // 中键长按＝急停（§3.3.1：停扫描＋全灭灯，
                                              //  纯设备动作不碰会话/全局态/参数盘；重复按不出错）
    std::function<void()> backToMain;         // 上键长按＝回主界面（§3.3.1：菜单/调节状态
                                              //  一键全退；模式与景深选择不动）
    std::function<void()> enterMenu;          // 上键短按·主界面（→EnterMenu）
    std::function<void()> exitMenu;           // 上键短按·菜单浏览态（→ExitMenu）
    std::function<void()> toggleDepthOfField;// 上键双击·主界面（景深 近↔远 直切，纯计算侧）
    std::function<void()> switchAdjustCtx;    // 左键双击·主界面（调节对象 亮度↔显示远近）
    std::function<void()> adjustUp;           // 右键短按·主界面（调当前对象）或①子态（体素密度）
    std::function<void()> adjustDown;         // 左键短按（同上）
    std::function<void()> cursorLeft;         // 左键短按·菜单浏览态（游标环移）
    std::function<void()> cursorRight;        // 右键短按·菜单浏览态
    std::function<void()> substateCancelled;  // ①⑤子态内任意键取消→回菜单浏览态
    std::function<void(const char* why)> dropped;  // 门禁拦/收紧丢弃/预留手势（记日志）
};

class KeySemantics {
public:
    // gate(类别)：这类键现在让按吗（DeviceManager 注入——按「类别＋全局态」
    // 组合的闭包；KeySemantics 不认识全局态）。capturing 裁判不看（启停同信号）。
    KeySemantics(std::function<bool(KeyCategory)> gate, KeySemActions actions);

    // 处理一个手势（直收 MCU 已判的 G01，1短/2双/3长）：
    // 逃生类优先（免门禁）→ 子态分流 → 类别定档问门禁 → 转移表 → 动作。
    void onGesture(const serial::GestureEvent& g, const MenuState& menu);

private:
    std::function<bool(KeyCategory)> gate_;   // 分类门禁（每手势现问）
    KeySemActions actions_;                   // 动作出口
};

} // namespace Scanner::device
