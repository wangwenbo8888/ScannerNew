#pragma once
// ============================================================================
// MenuLogic.h — 菜单记账本（K-T7；2026-08-18 设计 §2.3 二次修订 + 2026-08-20
// 08 设计 §2.4 状态源表；转移表 = 08 文档 §4.2.3 弃奇偶全显式状态版）
//
// 纯记账：不判该不该（门控在 KeySemantics）、不发命令；只经 apply 变状态。
// 显式状态（UI 常显）：层/游标/调节上下文/模式光标（模式光标=唯一真相源，
// 02 不存副本）；会话级不持久化（重启回默认）。
//
// 转移表（测试 test_menu_logic.cpp 逐条钉死）：
//   - EnterMenu：L1→L2 且 cursor 复位 ①（口径：进菜单必回 ①）；仅 L1 生效
//   - ExitMenu：L2→L1 且 adjustCtx=None；仅 L2 生效
//   - CursorLeft/Right：仅 L2——①←④环绕（Left 1→4,n→n-1；Right 4→1,n→n+1）
//   - CycleMode：无条件 modeCursor 1→2→3→1（默认 3 普通交叉，§4.2.3 v2.0 原文）
//   - CycleAdjustCtx：无条件 None→View→Brightness→None
//   - AdjustUp/Down：仅 adjustCtx≠None 生效（步进 ±1，只记计数不碰参数值）
//   - Reset：全回默认（layer=1/cursor=1/adjustCtx=None/modeCursor=3，
//     未取走步进一并清——会话复位语义，防异常恢复后转发陈旧步进）
// ============================================================================
#include <cstdint>

namespace Scanner::device {

enum class MenuOp {           // KeySemantics 判定后投给账本的唯一入口类型
    EnterMenu, ExitMenu,      // 上键短按（进/退菜单）
    CursorLeft, CursorRight,  // 左右键短按（layer=2 游标环移）
    CycleMode,                // 中键双击（扫描模式光标 1→2→3→1）
    CycleAdjustCtx,           // 上键双击（调节上下文 None→View→Brightness→None）
    AdjustUp, AdjustDown,     // 左右键短按（adjustCtx≠None 时加/减——只记步进计数，
                              //  实际参数值归 ParamStore，账本仅转发步进事件）
    Reset,                    // 会话复位（开机/异常恢复）
};

struct MenuState {            // 快照（UI 读）
    int layer = 1;            // 1=主层（扫描/就绪）/ 2=菜单层
    int cursor = 1;           // 菜单游标 ①~④（1 正常视野扫描/2 大视野扫描/3 扫描完成/4 开始后处理）
    enum class AdjustCtx { None, View, Brightness } adjustCtx = AdjustCtx::None;
    int modeCursor = 3;       // 扫描模式 1 精细/2 深孔/3 普通交叉（默认 3——口径：
                              //  08 文档 §4.2.3 v2.0 原文 modeCursor 默认=3 普通交叉）
};

class MenuLogic {
public:
    // apply 一个操作 → 返回是否生效（门禁外的操作 KeySemantics 已拦，这里恒 true；
    // 无效操作（如 layer=1 CursorLeft）返回 false 不变状态）
    bool apply(MenuOp op);
    MenuState state() const;          // 快照
    // AdjustUp/Down 步进事件出口（DeviceManager 接去 ParamStore；只在 apply 返回 true 时累计）
    int adjustSteps() const;          // 自上次 takeAdjustSteps 以来的净步数（+上/−下）
    int takeAdjustSteps();            // 取走并清零

private:
    MenuState st_;                    // 显式状态（唯一真相源）
    int steps_ = 0;                   // 净步数累计（+上/−下；take/Reset 清零）
};

} // namespace Scanner::device
