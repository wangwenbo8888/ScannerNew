#pragma once
// ============================================================================
// WarmupSequence.h — 预热看火员（2026-08-18 设计 §2.2；只管预热不管危险判定）
// 状态机：待命→加热→(稳了|超时)。火候判定：连续 stableWindowMs 内温度几乎不动
// (≤stableDeltaC) 且 离目标 ≤nearTargetC → 稳了回调恰一次；等太久(超时)→超时
// 回调一次——自己不停加热（报告上去听人的）。温度异常危险判定不归它——归故障链。
// 温度由外部喂（DeviceManager 逻辑线程把 T 帧 tick 给它——不轮询）。
//
// 火候判定算法（D-T11 钉死：双点锚点法）：
//   - 锚点 = 进入 Heating 后第一个喂入；窗口内任一喂入偏出锚点 >stableDeltaC
//     → 锚点重置为该喂入重新计时（「连续 stableWindowMs 几乎不动」）；
//   - 窗满（当前时刻-锚点时刻 ≥ stableWindowMs）且未重锚 且 离目标 ≤nearTargetC
//     → 判稳；未达目标（离目标 >nearTargetC）→ 即使稳定也不判稳（继续等——
//     时间照走，超时兜底）；
//   - 超时 = now-基准 ≥ timeoutMs（基准 = 进入 Heating 后首个观测：喂温或
//     tick——温度停更也能超时）；判稳优先于超时（同刻双满足先报稳）。
//
// 时基（D-T11 口径裁定）：onTemperature 的 tsMs 与 tick 的 nowMs 必须同源——
// 统一用 PC 接收时刻（DeviceManager 喂时给 now），mcu 帧时刻不进本类；
// 调用方保证单调统一。
//
// 单线程属主：逻辑线程（DeviceManager 直调）——本类不加锁。
// ============================================================================
#include <cstdint>
#include <functional>

namespace Scanner::device {

struct WarmupConfig {
    int stableWindowMs = 10000;   // 稳定判定窗口（默认 10s）
    double stableDeltaC = 0.1;    // 窗口内波动 ≤ 此值=「几乎不动」
    double nearTargetC = 2.0;     // 离目标 ≤ 此值
    int timeoutMs = 15 * 60 * 1000;  // 超时（默认 15 分钟）
};

class WarmupSequence {
public:
    enum class State { Idle, Heating, Done };   // Done=已回调（稳或超时），复位回 Idle
    explicit WarmupSequence(WarmupConfig cfg = {});

    // 开始预热（目标温度℃）——发加热命令归 DeviceManager（本类只记账状态）；
    // Heating/Done 中再调 = 复位重开（锚点/超时基准全清）
    void start(double targetC);
    // 温度事件喂入（tsMs=PC 接收时刻）——State==Heating 才参与判定；
    // lastTemp 任何状态都更新（报警快照用）
    void onTemperature(double celsius, int64_t tsMs);
    // 时钟推进（超时判定兜底——温度停更也能超时；nowMs 与 tsMs 同源）
    void tick(int64_t nowMs);
    State state() const;
    double lastTemp() const;      // 最近一次喂入（报警内容用；未喂=0.0）

    std::function<void()> onStable;     // 稳了（恰一次）
    std::function<void()> onTimeout;    // 超时（恰一次；不停加热）

private:
    void checkTimeout(int64_t nowMs);   // Heating 内超时判定（喂温/tick 共用）

    WarmupConfig cfg_;
    State state_ = State::Idle;
    double targetC_ = 0.0;
    double lastTemp_ = 0.0;
    bool hasRef_ = false;         // 超时基准已立？（进入 Heating 后首个观测）
    int64_t refMs_ = 0;
    bool hasAnchor_ = false;      // 锚点已立？（进入 Heating 后首个喂入）
    double anchorC_ = 0.0;
    int64_t anchorMs_ = 0;
};

} // namespace Scanner::device
