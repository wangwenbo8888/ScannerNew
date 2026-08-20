#pragma once
// ============================================================================
// McuFrame.h — MCU 上行帧类型 + 分流双有界环（设计方案 §2.4）
//
// 事件环（K/S/A，容量64，满丢最旧+计数）；遥测环（T，容量8，覆盖旧）。
// SPSC：串口rx线程生产 / 逻辑线程消费。协议默认口径见 §8（未定稿，改只改这里）。
// ============================================================================

#include "base/types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace Scanner::device::serial {

enum class KeyId : uint8_t { Up, Left, Middle, Right };   // U/L/M/R
enum class FrameKind : uint8_t { Temperature, Key, Status, Ack };

struct TempFrame   { double celsius[4]; uint8_t channels; uint16_t seq; TimestampMs ts; };
struct RawKeyEvent { KeyId key; bool pressed; uint32_t mcuMs; uint16_t seq; TimestampMs ts; };
struct StatusFrame { uint8_t code; uint16_t seq; TimestampMs ts; };  // 码表待协议方（§8-8 占位）
struct AckFrame    { uint16_t ackedSeq; uint16_t seq; };

// —— 载荷解析（v3 默认口径；v2 由 MCUDriver 兜底路径绕过）——
// T: "$T25.3,24.8<seq><crc>;"  K: "$KM1,1234<seq><crc>;"  S: "$S0A<seq><crc>;"  A: "$A0B<seq><crc>;"
bool parseTempPayload(const std::string& payload, TempFrame& out);
bool parseKeyPayload(const std::string& payload, RawKeyEvent& out);
bool parseStatusPayload(const std::string& payload, StatusFrame& out);
bool parseAckPayload(const std::string& payload, AckFrame& out);

// —— 丢最旧有界环（复用语义同 06 RingBuffer，但 08 自带轻量实现——分层铁律 C2）——
template <typename T, size_t N>
class SpscRing {
public:
    bool push(T v) {               // 生产者：满则丢最旧（恒 true——丢最旧语义，保留 bool 兼容既有调用）
        if (full()) { dropCount_.fetch_add(1, std::memory_order_relaxed); head_.store((head_+1)%N); }
        buf_[tail_] = std::move(v); tail_.store((tail_+1)%N); return true; }
    bool pop(T& out) {             // 消费者
        if (empty()) return false; out = std::move(buf_[head_]); head_.store((head_+1)%N); return true; }
    bool empty() const { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }
    uint64_t dropped() const { return dropCount_.load(std::memory_order_relaxed); }
private:
    bool full() const { return (tail_+1)%N == head_.load(std::memory_order_acquire); }
    std::array<T, N> buf_{};
    std::atomic<size_t> head_{0}, tail_{0};
    std::atomic<uint64_t> dropCount_{0};
};

} // namespace Scanner::device::serial
