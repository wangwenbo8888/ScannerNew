// ============================================================================
// ParamStore.cpp — 参数账本实现（D-T10；测试 test_param_store.cpp 钉死）
// ============================================================================
#include "ParamStore.h"

#include <cmath>

namespace Scanner::device {
namespace {

// 入口即钳：账内值永不越 spec 范围（setValue/直写/读档同口径）
double clampToSpec(const ParamSpec& spec, double v) {
    if (v < spec.min) return spec.min;
    if (v > spec.max) return spec.max;
    return v;
}

// 解析 "key=value;key=value;" 手写极简格式——坏段（无 '='、空值、非数值、
// 数值越界抛出）一律忽略该段，不影响其余段
void parseLedger(const std::string& text, std::map<std::string, double>& out) {
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t semi = text.find(';', pos);
        const std::string token = text.substr(pos, (semi == std::string::npos ? text.size() : semi) - pos);
        pos = (semi == std::string::npos) ? text.size() : semi + 1;
        const size_t eq = token.find('=');
        if (eq == std::string::npos || eq == 0) continue;      // 无 '=' 或空 key → 坏段
        const std::string key = token.substr(0, eq);
        const std::string val = token.substr(eq + 1);
        if (val.empty()) continue;                             // 空值 → 坏段
        try {
            size_t used = 0;
            const double v = std::stod(val, &used);
            if (used != val.size()) continue;                  // 尾部有渣（如 "abc"）→ 坏段
            out[key] = v;
        } catch (...) {
            continue;                                          // 非数值/越界 → 坏段
        }
    }
}

} // namespace

ParamStore::ParamStore(std::vector<ParamSpec> specs, Dispatch dispatch)
    : dispatch_(std::move(dispatch)) {
    for (auto& s : specs) {
        specs_[s.key] = s;
        entries_[s.key] = ParamEntry{s.def, false, ParamEntry::Source::Boot};
    }
}

void ParamStore::bootstrap(Load load) {
    inflight_.clear();   // 会话重启语义：在途全废（gen 单调递增，迟到回调必失配作废）
    std::map<std::string, double> file;
    if (load) parseLedger(load(), file);
    for (const auto& [key, spec] : specs_) {
        const auto it = file.find(key);                        // 未知 key 天然忽略
        const double v = (it != file.end()) ? clampToSpec(spec, it->second) : spec.def;
        const ParamEntry e{v, false, ParamEntry::Source::Boot};
        entries_[key] = e;
        if (onParamChanged) onParamChanged(key, e);            // 逐参数广播（含默认值项）
    }
}

void ParamStore::setValue(const std::string& key, double v, ParamEntry::Source src) {
    const auto specIt = specs_.find(key);
    if (specIt == specs_.end()) return;                        // 未登记：不下发不广播
    const double clamped = clampToSpec(specIt->second, v);     // 入口即钳
    const uint64_t gen = ++genSeq_;
    inflight_[key] = InFlight{clamped, gen};                   // 同 key 后值胜出（覆盖旧在途）
    if (!dispatch_) return;                                    // 无下发通道（测试/装配前）保在途记账
    dispatch_(key, clamped, [this, key, clamped, src, gen](bool ok, bool confirmed) {
        const auto fl = inflight_.find(key);
        if (fl == inflight_.end() || fl->second.gen != gen) return;   // 已被后值/bootstrap 取代
        inflight_.erase(fl);                                   // 出队（成败皆决）
        if (ok) {
            const ParamEntry e{clamped, confirmed, src};       // v3: confirmed=ok；v2: 恒 false
            entries_[key] = e;
            if (onParamChanged) onParamChanged(key, e);        // 改账后广播
        } else if (onReject) {
            onReject(key, entries_[key].value);                // 保旧值 + 弹回（旧值随行）
        }
    });
}

bool ParamStore::persist(Persist p) const {
    if (!p) return false;
    std::string text;
    for (const auto& [key, entry] : entries_) {                // map 序确定（档可复现）
        text += key + "=" + std::to_string(entry.value) + ";"; // 6 位小数——UI 档位值绰绰有余
    }
    return p(text);
}

ParamEntry ParamStore::get(const std::string& key) const {
    const auto it = entries_.find(key);
    return (it != entries_.end()) ? it->second : ParamEntry{}; // 未登记 → 0/false/Boot
}

bool ParamStore::has(const std::string& key) const { return specs_.count(key) > 0; }

bool ParamStore::pending(const std::string& key) const { return inflight_.count(key) > 0; }

void ParamStore::setEntryDirect(const std::string& key, double v, bool confirmed, ParamEntry::Source src) {
    const auto it = entries_.find(key);
    if (it == entries_.end()) return;                          // 未登记不动账
    it->second = ParamEntry{clampToSpec(specs_.at(key), v), confirmed, src};  // 静默写：不广播
}

} // namespace Scanner::device
