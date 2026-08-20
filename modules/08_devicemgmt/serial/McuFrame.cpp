// ============================================================================
// McuFrame.cpp — 上行帧载荷解析实现（v3 默认口径，契约见 McuFrame.h）
//
// 入参载荷已由 FrameCodec 剥去 '$'/seq/crc/';'（如完整帧 "$T25.3,24.80AF1;"
// 的载荷是 "T25.3,24.8"）。凡前缀/长度/字符集/数值非法一律 false，out 先归零。
// ============================================================================

#include "McuFrame.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace Scanner::device::serial {

namespace {

// 整串消费式浮点：拒绝空串/残留字符/溢出/非有限值（"25.x"/""/"1e999" 等）
bool parseDoubleFull(const std::string& s, double& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// 纯十进制数字串 → uint32（拒绝空串/非数字/溢出）
bool parseU32Full(const std::string& s, uint32_t& out) {
    if (s.empty()) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE || v > 0xFFFFFFFFull) return false;
    out = static_cast<uint32_t>(v);
    return true;
}

// 纯 hex 字符串 → uint32（拒绝空串/非 hex 字符/溢出；限长由调用方保证）
bool parseHexFull(const std::string& s, uint32_t& out) {
    if (s.empty()) return false;
    for (char c : s) {
        bool hex = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
        if (!hex) return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 16);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE) return false;
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

bool parseKeyPayload(const std::string& payload, RawKeyEvent& out) {
    out = RawKeyEvent{};
    if (payload.size() < 5 || payload[0] != 'K' || payload[3] != ',') return false;
    switch (payload[1]) {
        case 'U': out.key = KeyId::Up; break;
        case 'L': out.key = KeyId::Left; break;
        case 'M': out.key = KeyId::Middle; break;
        case 'R': out.key = KeyId::Right; break;
        default: return false;
    }
    if (payload[2] != '0' && payload[2] != '1') return false;
    out.pressed = (payload[2] == '1');
    return parseU32Full(payload.substr(4), out.mcuMs);
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
