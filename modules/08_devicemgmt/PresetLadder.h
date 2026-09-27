#pragma once
// ============================================================================
// PresetLadder.h — 参数档位梯（G6 缺口补齐·260927；协作文档 v1.1 §4.2.4 原设计）
//
// 亮度档＝曝光＋激光亮度＋补光三参数组合步进（内置 3 档，可整体注入替换）；
// 钳制不环绕（到顶/底再按无效）。纯账本件：只算档位与参数组，写账归
// DeviceManager（ParamStore.setValue Source::Key——下发/广播/落盘全走既有链）。
// 档值内置为产线基线（对齐 UI 预设族），真机定表后经 setLadder 整体替换。
//
// 单线程属主：逻辑线程（与 MenuLogic/ParamStore 同口径——本类不加锁）。
// ============================================================================
#include <string>
#include <vector>

namespace Scanner::device {

struct LadderStep {              // 一档＝三参组合（单位同 ParamStore specs）
    double exposureMs = 3.0;     // 曝光 ms（spec 1-100；UI 口径实用 1-5）
    double laserLevel = 40.0;    // 激光亮度 0-100
    double bgLight = 10.0;       // 补光 0-100
};

class PresetLadder {
public:
    PresetLadder() = default;

    // 内置 3 档（低/中/高——产线基线，真机定表后 setLadder 替换）
    static std::vector<LadderStep> builtinLadder() {
        return {
            {1.0, 40.0, 10.0},    // 档1 低：短曝光（面片预设族 B10/L40/曝光3 的暗档）
            {3.0, 70.0, 40.0},    // 档2 中：面片推荐基线附近
            {5.0, 100.0, 80.0},   // 档3 高：强光（标点/暗目标）
        };
    }

    void setLadder(std::vector<LadderStep> ladder) {   // 整体替换（档数≥1；空忽略）
        if (!ladder.empty()) ladder_ = std::move(ladder);
    }
    const std::vector<LadderStep>& ladder() const { return ladder_; }
    int index() const { return index_; }               // 当前档（1 起；0=未启用）

    // 步进 ±1（钳制不环绕）：返回 true=档位变化（调用方据此写三参账），
    // false=已到顶/底（无效步，静默）
    bool step(int dir) {
        const int next = index_ + (dir > 0 ? 1 : -1);
        if (next < 1 || next > static_cast<int>(ladder_.size())) return false;
        index_ = next;
        return true;
    }
    void reset() { index_ = 1; }                       // 会话复位（口径同 MenuLogic.Reset）
    const LadderStep& current() const { return ladder_[static_cast<size_t>(index_ - 1)]; }

private:
    std::vector<LadderStep> ladder_ = builtinLadder();
    int index_ = 1;
};

} // namespace Scanner::device
