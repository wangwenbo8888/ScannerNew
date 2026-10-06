// ============================================================================
// FrameCodec.cpp — v2/v3 成拆帧实现（契约见 FrameCodec.h）
//
// CRC16-CCITT 查表 256 项（poly 0x1021，constexpr 表）；v3 状态机：
// 扫 '$' 起始（挂起中再见 '$' 即重新同步）→ 累积至 ';' → 总长≤32 →
// 剥尾 seq2hex+crc4hex → 重算 CRC(覆盖 "$…seq") 比对 → Frame。
// ============================================================================

#include "FrameCodec.h"

#include <array>
#include <utility>

namespace Scanner::device::serial {

namespace {

constexpr size_t kMaxFrame = 32;   // 帧总长上限（'$'…';' 含首尾）
constexpr size_t kTailLen = 6;     // 尾 = seq 2 hex + crc 4 hex
constexpr int64_t kHalfTimeoutMs = 50;

constexpr uint16_t crcEntry(uint8_t index) {
    uint16_t crc = static_cast<uint16_t>(index << 8);
    for (int b = 0; b < 8; ++b)
        crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                             : static_cast<uint16_t>(crc << 1);
    return crc;
}

template <size_t... I>
constexpr std::array<uint16_t, 256> crcTableImpl(std::index_sequence<I...>) {
    return {{crcEntry(static_cast<uint8_t>(I))...}};
}

constexpr auto kCrcTable = crcTableImpl(std::make_index_sequence<256>{});

// 定长大写 hex 串 → 值（非 hex / 超 max 拒绝）
bool parseHexFixed(const std::string& s, uint32_t max, uint32_t& out) {
    uint32_t v = 0;
    for (char c : s) {
        uint32_t d;
        if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
        else if (c >= 'A' && c <= 'F') d = static_cast<uint32_t>(c - 'A' + 10);
        else if (c >= 'a' && c <= 'f') d = static_cast<uint32_t>(c - 'a' + 10);
        else return false;
        v = v * 16 + d;
        if (v > max) return false;
    }
    out = v;
    return true;
}

std::string toHex(uint32_t v, size_t width) {
    static const char kDigits[] = "0123456789ABCDEF";
    std::string s(width, '0');
    for (size_t i = width; i-- > 0; v >>= 4)
        s[i] = kDigits[v & 0xF];
    return s;
}

} // namespace

FrameCodec::FrameCodec(Version v) : v_(v) {}

uint16_t FrameCodec::crc16ccitt(const std::string& bytes, uint16_t init) {
    uint16_t crc = init;
    for (char c : bytes)
        crc = static_cast<uint16_t>((crc << 8) ^ kCrcTable[(crc >> 8) ^ static_cast<uint8_t>(c)]);
    return crc;
}

void FrameCodec::feed(const std::string& bytes, std::vector<Frame>& out) {
    for (char ch : bytes) {
        if (v_ == Version::V3) {
            if (ch == '$') {                        // 帧起始 / 半帧中重新同步
                pending_.assign(1, '$');
                pendingAgeMs_ = 0;
                continue;
            }
            if (pending_.empty()) continue;         // 非 '$' 前导丢弃（含 ';'）
            if (ch == ';') { parsePending(out); continue; }
            pending_.push_back(ch);
            if (pending_.size() >= kMaxFrame) {  // 无尾超长流早弃：挂起缓冲封顶（可观测行为不变）
                pending_.clear();
                pendingAgeMs_ = 0;
            }
        } else {
            if (ch == ';') {
                if (!pending_.empty()) out.push_back(Frame{std::move(pending_), 0});
                pending_.clear();
                pendingAgeMs_ = 0;
            } else {
                if (pending_.empty()) pendingAgeMs_ = 0;
                pending_.push_back(ch);
            }
        }
    }
}

bool FrameCodec::advanceTimeout(int64_t ms) {
    if (pending_.empty()) return false;
    pendingAgeMs_ += ms;
    if (pendingAgeMs_ < kHalfTimeoutMs) return false;
    pending_.clear();
    pendingAgeMs_ = 0;
    return true;
}

std::string FrameCodec::encode(const std::string& payload, uint16_t seq) const {
    if (v_ == Version::V2) return payload + ";";
    for (char c : payload)
        if (c == '$' || c == ';') return "";
    std::string f = "$" + payload + toHex(seq & 0xFF, 2);  // seq 0~255 循环
    f += toHex(crc16ccitt(f, 0xFFFF), 4);
    f += ";";
    return f;
}

void FrameCodec::parsePending(std::vector<Frame>& out) {
    std::string f = std::move(pending_);
    pending_.clear();
    pendingAgeMs_ = 0;
    if (f.size() < 1 + kTailLen || f.size() + 1 > kMaxFrame) return;  // 尾过短 / 总长超限
    uint32_t seq = 0, crc = 0;
    if (!parseHexFixed(f.substr(f.size() - 6, 2), 0xFF, seq)) return;
    if (!parseHexFixed(f.substr(f.size() - 4), 0xFFFF, crc)) return;
    if (crc16ccitt(f.substr(0, f.size() - 4), 0xFFFF) != crc) return;
    out.push_back(Frame{f.substr(1, f.size() - 1 - kTailLen), static_cast<uint16_t>(seq)});
}

} // namespace Scanner::device::serial
