#pragma once
// ============================================================================
// MenuLogic.h — 菜单记账本（261002 按键交互域定稿 §3.3.3 五项重定义版）
//
// 纯记账：不判该不该（门控在 KeySemantics＋功能口在 DeviceManager）、不发命令；
// 只经 apply 变状态。显式状态（UI 常显）：层/游标/子态/调节对象/模式光标。
//
// 菜单五项（①分辨率设置/体素密度 ②进入就绪 ③扫描完成 ④后处理 ⑤重置）：
//   - 游标 ①~⑤ 环绕（Left 1→5,n→n-1；Right 5→1,n→n+1）
//   - ①⑤选中先进子态（①=体素密度调节：左右切档中键确认退出；⑤=重置确认：
//     再按中键执行，其他键取消）；②③④选中＝执行并自动退菜单（执行归
//     DeviceManager 派事件，本类只做退菜单记账）
//   - 子态取消语义：①⑤子态内除各自功能键与逃生类外，任意键＝取消子态、
//     回菜单浏览态（不直接退菜单）
//   - 回主界面（上键长按）：layer=1/cursor=①/子态清/调节对象回默认亮度；
//     扫描模式与景深档不动（业务状态不归界面）
//   - 调节对象二选一：亮度↔显示远近（无「不调节」态；默认亮度）——仅主界面
//     生效（菜单内左右键＝游标）
//   - CycleMode：无条件 modeCursor 1→2→3→1（1 精细/2 深孔/3 普通交叉）
//   - 会话级不持久化（重启回默认）
// ============================================================================
#include <cstdint>

namespace Scanner::device {

enum class MenuOp {           // KeySemantics 判定后投给账本的唯一入口类型
    EnterMenu, ExitMenu,      // 上键短按（进/退菜单）
    CursorLeft, CursorRight,  // 左右键短按（layer=2 游标环移 ①~⑤）
    CycleMode,                // 中键双击（扫描模式光标 1→2→3→1）
    SwitchAdjustCtx,          // 左键双击（调节对象 亮度↔显示远近 二选一切换）
    AdjustUp, AdjustDown,     // 档位步进 ±1（只记净步数；实际档值归梯子/ParamStore）
    EnterAdjustSubstate,      // 菜单①选中→子态（游标停①，左右切体素密度档）
    EnterConfirmSubstate,     // 菜单⑤选中→确认子态（游标停⑤）
    CancelSubstate,           // 子态内任意键取消→回菜单浏览态
    BackToMain,               // 上键长按＝一键回主界面（清层/游标/子态/调节对象回默认）
    Reset,                    // 会话复位（开机/异常恢复——全默认）
};

struct MenuState {            // 快照（UI 读）
    int layer = 1;            // 1=主界面 / 2=菜单
    int cursor = 1;           // 菜单游标 ①~⑤（1 分辨率设置/2 进入就绪/3 扫描完成/
                              //  4 后处理/5 重置）
    enum class Substate : uint8_t {
        None = 0,             // 菜单浏览态（或主界面）
        AdjustVoxel = 1,      // ①子态：体素密度调节中（左右切档/中键确认退出）
        ConfirmReset = 2,     // ⑤子态：重置二次确认中（再按中键执行/其他键取消）
    } substate = Substate::None;
    enum class AdjustCtx : uint8_t {
        Brightness = 0,       // 亮度梯（默认）
        DisplayDistance = 1,  // 显示远近梯
    } adjustCtx = AdjustCtx::Brightness;   // 二选一（无 None——设计 §3.3.2）
    int modeCursor = 3;       // 扫描模式 1 精细/2 深孔/3 普通交叉（默认 3）
};

// 单线程契约：仅逻辑线程调用（KeySemantics 判定后 DeviceManager 逻辑线程 apply）
class MenuLogic {
public:
    // apply 一个操作 → 返回是否生效（门禁外的操作 KeySemantics 已拦，这里恒 true；
    // 无效操作（如 layer=1 CursorLeft）返回 false 不变状态）
    bool apply(MenuOp op);
    MenuState state() const;          // 快照
    // AdjustUp/Down 步进事件出口（DeviceManager 接去梯子；只在 apply 返回 true 时累计）
    int adjustSteps() const;          // 自上次 takeAdjustSteps 以来的净步数（+上/−下）
    int takeAdjustSteps();            // 取走并清零

private:
    MenuState st_;                    // 显式状态（唯一真相源）
    int steps_ = 0;                   // 净步数累计（+上/−下；take/Reset 清零）
};

} // namespace Scanner::device
