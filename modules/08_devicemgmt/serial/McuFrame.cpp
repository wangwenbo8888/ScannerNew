// ============================================================================
// McuFrame.cpp — 上行帧载荷解析实现（v3 默认口径，契约见 McuFrame.h）
//
// 入参载荷已由 FrameCodec 剥去 '$'/seq/crc/';'（如完整帧 "$T25.3,24.80AF1;"
// 的载荷是 "T25.3,24.8"）。凡前缀/长度/字符集/数值非法一律 false，out 先归零。
// ============================================================================

#include "McuFrame.h"

#include <charconv>
#include <cmath>

namespace Scanner::device::serial {

namespace {

// 整串消费式浮点（from_chars：locale 无关、拒 '+'/前导空白；溢出/残留/非有限全拒）
bool parseDoubleFull(const std::string& s, double& out) {
    if (s.empty()) return false;
    const char* b = s.data();
    const char* e = b + s.size();
    double v = 0.0;
    const auto r = std::from_chars(b, e, v);
    if (r.ec != std::errc{} || r.ptr != e || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// 纯 hex 字符串 → uint32（拒空串/非 hex 字符/溢出；限长由调用方保证）
bool parseHexFull(const std::string& s, uint32_t& out) {
    if (s.empty()) return false;
    const char* b = s.data();
    const char* e = b + s.size();
    uint32_t v = 0;
    const auto r = std::from_chars(b, e, v, 16);
    if (r.ec != std::errc{} || r.ptr != e) return false;
    out = v;
    return true;
}

} // namespace

bool parseTempPayload(const std::string& payload, TempFrame& out) {
    out = TempFrame{};
    if (payload.size() < 2 || payload[0] != 'T') return false;
    size_t n = 0, start = 1;
    for (size_t i = 1; i <= payload.size(); ++i) {
        if (i == payload.size() || payload[i] == ',') {
            if (i == start || n >= 4) return false;   // 空字段 / 超 4 通道
            double v = 0.0;
            if (!parseDoubleFull(payload.substr(start, i - start), v)) return false;
            out.celsius[n++] = v;
            start = i + 1;
        }
    }
    out.channels = static_cast<uint8_t>(n);
    return true;
}

bool parseG01Payload(const std::string& payload, GestureEvent& out) {
    out = GestureEvent{};
    // 格式：G01 <键><手势位> —— 键 U/L/M/R，手势位 1短/2双/3长（MCU 已判）。
    // 容忍 G01 与键位之间可选空白（协议样例 "G01 U1"，真机文本行实测带空格）。
    if (payload.size() < 5 || payload[0] != 'G' || payload[1] != '0' || payload[2] != '1')
        return false;
    auto isW = [](char c) { return c == ' ' || c == '\t'; };
    size_t i = 3;
    while (i < payload.size() && isW(payload[i])) ++i;
    if (i + 2 > payload.size()) return false;
    switch (payload[i]) {
        case 'U': out.key = KeyId::Up; break;
        case 'L': out.key = KeyId::Left; break;
        case 'M': out.key = KeyId::Middle; break;
        case 'R': out.key = KeyId::Right; break;
        default: return false;
    }
    switch (payload[i + 1]) {
        case '1': out.gesture = GestureEvent::Gesture::Short; break;
        case '2': out.gesture = GestureEvent::Gesture::Double; break;
        case '3': out.gesture = GestureEvent::Gesture::Hold; break;
        default: return false;
    }
    for (size_t j = i + 2; j < payload.size(); ++j)   // 键位后仅允尾随空白/回车（CRLF 行尾）
        if (!isW(payload[j]) && payload[j] != '\r') return false;
    return true;
}

bool parseStatusPayload(const std::string& payload, StatusFrame& out) {
    out = StatusFrame{};
    if (payload.size() < 2 || payload.size() > 3 || payload[0] != 'S') return false;
    uint32_t v = 0;
    if (!parseHexFull(payload.substr(1), v)) return false;
    out.code = static_cast<uint8_t>(v);
    return true;
}

bool parseAckPayload(const std::string& payload, AckFrame& out) {
    out = AckFrame{};
    if (payload.size() < 2 || payload.size() > 3 || payload[0] != 'A') return false;
    uint32_t v = 0;
    if (!parseHexFull(payload.substr(1), v)) return false;
    out.ackedSeq = static_cast<uint16_t>(v);
    return true;
}

} // namespace Scanner::device::serial
