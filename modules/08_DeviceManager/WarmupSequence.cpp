// ============================================================================
// WarmupSequence.cpp — 预热看火员实现（D-T11；测试 test_warmup_sequence.cpp 钉死）
// ============================================================================
#include "WarmupSequence.h"

#include <cmath>

namespace Scanner::device {

WarmupSequence::WarmupSequence(WarmupConfig cfg) : cfg_(cfg) {}

void WarmupSequence::start(double targetC) {
    targetC_ = targetC;
    state_ = State::Heating;
    hasRef_ = false;             // 复位重开：锚点/超时基准全清
    refMs_ = 0;
    hasAnchor_ = false;
    anchorC_ = 0.0;
    anchorMs_ = 0;
}

void WarmupSequence::onTemperature(double celsius, int64_t tsMs) {
    lastTemp_ = celsius;                          // 报警快照：任何状态都记
    if (state_ != State::Heating) return;         // Idle/Done 不参与判定
    if (!hasRef_) { hasRef_ = true; refMs_ = tsMs; }   // 超时基准=Heating 后首个观测
    if (!hasAnchor_) {
        anchorC_ = celsius;                       // 锚点=进入 Heating 后第一个喂入
        anchorMs_ = tsMs;
        hasAnchor_ = true;
    } else if (std::fabs(celsius - anchorC_) > cfg_.stableDeltaC) {
        anchorC_ = celsius;                       // 跳变 → 重锚重新计时
        anchorMs_ = tsMs;
    } else if (tsMs - anchorMs_ >= cfg_.stableWindowMs
               && std::fabs(targetC_ - celsius) <= cfg_.nearTargetC) {
        state_ = State::Done;                     // 火候到了——只报不停
        if (onStable) onStable();
        return;                                   // 判稳优先于超时（同刻先报稳）
    }
    checkTimeout(tsMs);
}

void WarmupSequence::tick(int64_t nowMs) {
    if (state_ != State::Heating) return;
    if (!hasRef_) { hasRef_ = true; refMs_ = nowMs; }   // 温度停更：tick 也可立基准
    checkTimeout(nowMs);
}

void WarmupSequence::checkTimeout(int64_t nowMs) {
    if (state_ != State::Heating || !hasRef_) return;
    if (nowMs - refMs_ >= cfg_.timeoutMs) {
        state_ = State::Done;                     // 只报不停（无停止加热出口）
        if (onTimeout) onTimeout();
    }
}

WarmupSequence::State WarmupSequence::state() const { return state_; }
double WarmupSequence::lastTemp() const { return lastTemp_; }

} // namespace Scanner::device
