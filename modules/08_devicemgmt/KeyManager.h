#pragma once
// ============================================================================
// KeyManager.h — 按键手势判定（PC 侧；2026-08-20 08 设计方案 §3）
//
// 输入 RawKeyEvent（键号+按下/松开+MCU 时刻 ms），判定用 MCU 时刻（排队延迟不扰
// 手感）；逻辑线程 10ms tick 兜底（S 静默窗到期/长按达阈）。零业务状态：
// 不门控、不计数、不映射——判完即出 KeyGesture，语义归 KeySemantics。
// 一次物理操作只出 S/D/H 之一（互斥）；乱序/超期事件丢弃+warn 计数。
//
// 口径裁定（§3 未尽边界，测试 test_key_manager.cpp 逐条钉死）：
//   - 判定驱动=事件优先：任一 onRawEvent 以「已见最大 MCU 时刻」全局扫一遍
//     （他键事件同样推动判定），且先扫后处理本事件；tick 仅 PC 域兜底
//     （事件断流时保活，nowMs 须与构造时钟同域）。
//   - 手势锚点：S=松开时刻 / D=第二次松开 / H=按下+holdMs（名义达阈时刻，
//     与触发驱动无关）。
//   - 400ms<按压<800ms：非 S 候选也够不到 H → 无手势不计数，静默回 Idle。
//   - 双击第二次按压：时长不设上限（≥holdMs 判 H，首击静默取消）；
//     <debounceMs 视为抖动 → 回退首击 ShortPending（最终判 S 不判 D）。
//   - 消抖仅同键（异键间隔互不干扰）；抖动作废不计 droppedCount。
//   - H 判定后 PostHold：迟到松开静默吞一次；其后的新按压=全新操作。
//   - 倒序（早于该键已处理时刻）/松开无对应按下（含被抖动吞掉的按压后的散落
//     松开）/Down 中重复按下 → droppedCount++（对账）。
// ============================================================================

#include "serial/McuFrame.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace Scanner::device {

struct GestureThresholds {
    int debounceMs = 20;      // 消抖（MCU 侧保留，PC 双保险）
    int shortMs = 400;        // 松开≤此值=短按候选
    int doubleWindowMs = 400; // 双击窗口（第二次按下须在窗口内）
    int holdMs = 800;         // 持续≥此值=长按（达阈即判，不等松开）
};

struct KeyGesture {
    serial::KeyId key;
    enum class G { Short, Double, Hold } gesture;
    uint32_t mcuMs;           // 判定所锚的 MCU 时刻（S=松开时刻/D=第二次松开/H=达阈时刻）
};

class KeyManager {
public:
    // PC 时钟注入（默认 steady_clock 毫秒）：仅用于事件到达时刻锚定（tick 兜底
    // 的比较基准），tick 入参须与之同域；测试注入假钟取确定论
    using PcClock = std::function<int64_t()>;
    explicit KeyManager(GestureThresholds t = {}, PcClock pcClock = {});

    void onRawEvent(const serial::RawKeyEvent& ev);   // 逻辑线程喂（MCUDriver pump 分流来）
    void tick(int64_t nowMs);                          // 逻辑线程 10ms：nowMs=PC 时钟（仅兜底比较用）
    std::vector<KeyGesture> drain();                   // 取走自上次以来判定的手势
    uint64_t droppedCount() const;                     // 乱序/超期丢弃计数（对账）

private:
    // 每键独立状态机（无锁：单逻辑线程属主）：
    //   Idle ──按下──▶ Down ──松开≤shortMs──▶ ShortPending ──静默窗到期──▶ Idle（出 S）
    //     ▲              │ │      ▲  ▲              │窗内再按下(≥debounceMs)
    //     │              │ │      │  └──二次按压抖动(<debounceMs) 作废回退──┘
    //     │              │ │      └──────────(第二次按压)──▶ Down(second) ──松开─▶ Idle（出 D）
    //     │              │ │                                  └持续≥holdMs─▶ PostHold（出 H，首击取消）
    //     │              │ └─shortMs<持续<holdMs 松开──▶ Idle（无手势）
    //     │              └──持续≥holdMs（事件扫/tick）──▶ PostHold（出 H）
    //     └─PostHold：迟到松开吞一次；再按下=全新 Down
    //   消抖：按下→松开 <debounceMs 本次按压作废（二次按压则回退 ShortPending）；
    //         松开→同键再按下 <debounceMs 视为抖动延续（不开新按压）。仅同键。
    enum class St { Idle, Down, ShortPending, PostHold };

    struct KeyState {
        St st = St::Idle;
        bool hasLast = false;        // lastMcu 有效位
        uint32_t lastMcu = 0;        // 该键已处理最新 MCU 时刻（倒序判据；丢弃事件不推进）
        uint32_t prevMcu = 0;        // 前一接受事件时刻（「松开→再按下」消抖间隔基准）
        bool lastWasRelease = false; // 末次接受事件为松开（「松开→再按下」消抖判据）
        uint32_t anchorMcu = 0;      // Down=按下时刻 / ShortPending=松开时刻（MCU 域锚）
        int64_t anchorPc = 0;        // 同上事件的 PC 到达时刻（tick 兜底比较用）
        bool second = false;         // Down 期间=双击第二次按压
        uint32_t pendMcu = 0;        // 二次按压期间保留的首击松开锚（抖动回退用）
        int64_t pendPc = 0;
    };

    static constexpr size_t kKeyCount = 4;   // 与 serial::KeyId（Up/Left/Middle/Right）同步
    static_assert(static_cast<uint8_t>(serial::KeyId::Right) + 1 == kKeyCount,
                  "KeyId 与 kKeyCount 基数同步");

    KeyState& stOf(serial::KeyId k) { return keys_[static_cast<size_t>(k)]; }
    void processEvent(serial::KeyId k, bool pressed, uint32_t mcuMs, int64_t pcMs);
    void sweepMcu(uint32_t nowMcu);          // 事件优先：MCU 域全局扫（H 达阈 / S 窗到期）
    void sweepPc(int64_t nowPc);             // tick 兜底：PC 域全局扫（同一套判定）
    // 注：命名避开 Qt 的 emit 宏（scan_demo 内 Qt 头先入即冲突——P2 渲染加固时暴露）
    void emitGesture(serial::KeyId k, KeyGesture::G g, uint32_t mcuMs);

    GestureThresholds th_;
    PcClock pcNow_;                          // 空=steady 毫秒（ctor 内包装兜底）
    std::array<KeyState, kKeyCount> keys_;   // U/L/M/R 各自状态机
    std::vector<KeyGesture> pending_;        // 待 drain 手势
    uint32_t mcuNow_ = 0;                    // 已见最大 MCU 时刻（全局扫基准）
    uint64_t dropped_ = 0;
};

} // namespace Scanner::device
