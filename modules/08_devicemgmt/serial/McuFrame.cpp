// ============================================================================
// McuFrame.cpp — G01/G02/G03 载荷解析（协议 260831；契约见 McuFrame.h）
// ============================================================================
#include "McuFrame.h"

#include <charconv>
#include <cmath>

namespace Scanner::device::serial {

namespace {

// 整串消费式浮点（from_chars：locale 无关；溢出/残留/非有限全拒）
bool parseDoubleFull(const std::string& s, double& out) {
    if (s.empty()) return false;
    double v = 0.0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// 纯十进制数字串 → uint64
bool parseU64Full(const std::string& s, uint64_t& out) {
    if (s.empty()) return false;
    uint64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v, 10);
    if (r.ec != std::errc{} || r.ptr != s.data() + s.size()) return false;
    out = v;
    return true;
}

} // namespace

// "G01 U1" —— 指令字＋空格＋键字母＋手势位（1=单击 2=双击 3=长按）；
// 索引钉死：[0..3]="G01 " 前缀、[4]=键字母、[5]=手势位（批1-A 原版误检 [3]
// （空格）恒拒——批1-C 修正）
bool parseGesturePayload(const std::string& payload, GestureEvent& out) {
    out = GestureEvent{};
    if (payload.size() != 6 || payload.compare(0, 4, "G01 ") != 0) return false;
    switch (payload[4]) {
        case 'U': out.key = KeyId::Up; break;
        case 'L': out.key = KeyId::Left; break;
        case 'M': out.key = KeyId::Middle; break;
        case 'R': out.key = KeyId::Right; break;
        default: return false;
    }
    if (payload[5] < '1' || payload[5] > '3') return false;
    out.gesture = static_cast<Gesture>(payload[5] - '0');
    return true;
}

// "G02 A25.3 B25.4 C26.0 D24.5" —— 四路必齐（协议恒 4 路），A/B/C/D 序逐项消费；
// 缺路/键错/非法值一律 false。
bool parseTempPayload(const std::string& payload, TempFrame& out) {
    out = TempFrame{};
    if (payload.size() < 4 || payload.compare(0, 4, "G02 ") != 0) return false;
    const char kKeys[4] = {'A', 'B', 'C', 'D'};
    size_t pos = 4;
    for (int i = 0; i < 4; ++i) {
        if (pos + 1 >= payload.size() || payload[pos] != kKeys[i]) return false;
        size_t end = payload.find(' ', pos + 1);
        if (end == std::string::npos) end = payload.size();
        double v = 0.0;
        if (!parseDoubleFull(payload.substr(pos + 1, end - pos - 1), v)) return false;
        out.celsius[i] = v;
        pos = (end == payload.size()) ? end : end + 1;
    }
    return pos == payload.size();
}

// "G03 S500" —— 指令字＋空格＋S＋十进制计数
bool parseShotCountPayload(const std::string& payload, ShotCountFrame& out) {
    out = ShotCountFrame{};
    if (payload.size() < 6 || payload.compare(0, 4, "G03 ") != 0 || payload[4] != 'S') return false;
    return parseU64Full(payload.substr(5), out.count);
}

} // namespace Scanner::device::serial
