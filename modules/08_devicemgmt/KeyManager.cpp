// ============================================================================
// KeyManager.cpp — 按键手势判定实现（K-T6）；状态机与口径见 KeyManager.h
// ============================================================================

#include "KeyManager.h"

#include <chrono>
#include <utility>

namespace Scanner::device {

namespace {

int64_t steadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// MCU 时刻差（uint32 环回安全：真实事件间隔远小于 2^31ms）
int32_t diffMs(uint32_t a, uint32_t b) { return static_cast<int32_t>(a - b); }

} // namespace

KeyManager::KeyManager(GestureThresholds t, PcClock pcClock)
    : th_(t), pcNow_(pcClock ? std::move(pcClock) : PcClock([] { return steadyMs(); })) {}

void KeyManager::onRawEvent(const serial::RawKeyEvent& ev) {
    KeyState& ks = stOf(ev.key);
    if (ks.hasLast && diffMs(ev.mcuMs, ks.lastMcu) < 0) {   // 倒序：丢弃，不推进已处理时刻
        ++dropped_;
        return;
    }
    ks.prevMcu = ks.lastMcu;                                // 前一事件时刻（供消抖间隔比较）
    ks.hasLast = true;
    ks.lastMcu = ev.mcuMs;
    if (diffMs(ev.mcuMs, mcuNow_) > 0) mcuNow_ = ev.mcuMs;  // 全局时刻单调推进

    sweepMcu(mcuNow_);                                      // 事件优先：先按时间收尾（过期 S/达阈 H）
    processEvent(ev.key, ev.pressed, ev.mcuMs, pcNow_());   // 再处理本事件（PostHold 吞/新按压）
}

void KeyManager::tick(int64_t nowMs) { sweepPc(nowMs); }

std::vector<KeyGesture> KeyManager::drain() {
    std::vector<KeyGesture> out;
    out.swap(pending_);
    return out;
}

uint64_t KeyManager::droppedCount() const { return dropped_; }

void KeyManager::processEvent(serial::KeyId k, bool pressed, uint32_t t, int64_t pc) {
    KeyState& ks = stOf(k);
    if (pressed) {
        // 「松开→同键再按下 <debounceMs」=抖动延续：不开新按压（仅同键；异键无此
        // 判据）。间隔=本事件与前一事件（prevMcu 此时即松开时刻）之差
        if (ks.lastWasRelease && diffMs(t, ks.prevMcu) < th_.debounceMs) return;
        if (ks.st == St::ShortPending) {                    // 双击窗内二次按压（sweep 已保证在窗内）
            ks.pendMcu = ks.anchorMcu;                      // 保留首击松开锚（二次抖动回退用）
            ks.pendPc = ks.anchorPc;
            ks.st = St::Down;
            ks.second = true;
            ks.anchorMcu = t;
            ks.anchorPc = pc;
            ks.lastWasRelease = false;                      // 已接受按压：消抖判据只看「松开后」
            return;
        }
        if (ks.st == St::Down) {                            // 重复按下=协议异常（松开丢了）→ 对账
            ++dropped_;
            return;
        }
        ks.st = St::Down;                                   // Idle / PostHold：全新操作
        ks.second = false;
        ks.anchorMcu = t;
        ks.anchorPc = pc;
        ks.lastWasRelease = false;
        return;
    }

    // —— 松开 ——
    if (ks.st == St::PostHold) {                            // H 后迟到松开：静默吞一次
        ks.st = St::Idle;
        ks.lastWasRelease = true;
        return;
    }
    if (ks.st != St::Down) {                                // 松开无对应接受按压 → 对账
        ++dropped_;
        return;
    }
    const int32_t d = diffMs(t, ks.anchorMcu);
    if (d < th_.debounceMs) {                               // 按压抖动作废（按压+松开一起作废）
        if (ks.second) {                                    // 二次按压抖动 → 回退首击 ShortPending
            ks.st = St::ShortPending;
            ks.anchorMcu = ks.pendMcu;
            ks.anchorPc = ks.pendPc;
        } else {
            ks.st = St::Idle;
        }
        ks.lastWasRelease = true;
        return;
    }
    ks.lastWasRelease = true;
    if (ks.second) {                                        // 第二次松开即判 D（时长不设上限，
        emit(k, KeyGesture::G::Double, t);                  //   ≥holdMs 已被 sweep 判 H）
        ks.st = St::Idle;
        return;
    }
    if (d <= th_.shortMs) {                                 // 短按候选：静默等双击窗
        ks.st = St::ShortPending;
        ks.anchorMcu = t;
        ks.anchorPc = pc;
        return;
    }
    ks.st = St::Idle;                                       // shortMs<d<holdMs：无手势（口径）
}

void KeyManager::sweepMcu(uint32_t nowMcu) {
    for (size_t i = 0; i < kKeyCount; ++i) {
        KeyState& ks = keys_[i];
        if (ks.st == St::Down && diffMs(nowMcu, ks.anchorMcu) >= th_.holdMs) {
            emit(static_cast<serial::KeyId>(i), KeyGesture::G::Hold,
                 ks.anchorMcu + static_cast<uint32_t>(th_.holdMs));
            ks.st = St::PostHold;
        } else if (ks.st == St::ShortPending && diffMs(nowMcu, ks.anchorMcu) >= th_.doubleWindowMs) {
            emit(static_cast<serial::KeyId>(i), KeyGesture::G::Short, ks.anchorMcu);
            ks.st = St::Idle;
        }
    }
}

void KeyManager::sweepPc(int64_t nowPc) {
    for (size_t i = 0; i < kKeyCount; ++i) {
        KeyState& ks = keys_[i];
        if (ks.st == St::Down && nowPc - ks.anchorPc >= th_.holdMs) {
            emit(static_cast<serial::KeyId>(i), KeyGesture::G::Hold,
                 ks.anchorMcu + static_cast<uint32_t>(th_.holdMs));
            ks.st = St::PostHold;
        } else if (ks.st == St::ShortPending && nowPc - ks.anchorPc >= th_.doubleWindowMs) {
            emit(static_cast<serial::KeyId>(i), KeyGesture::G::Short, ks.anchorMcu);
            ks.st = St::Idle;
        }
    }
}

void KeyManager::emit(serial::KeyId k, KeyGesture::G g, uint32_t mcuMs) {
    pending_.push_back(KeyGesture{k, g, mcuMs});
}

} // namespace Scanner::device
