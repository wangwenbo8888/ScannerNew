#pragma once
// ============================================================================
// McuFrame.h — MCU 上行帧类型（设计方案 §2.4；260831 协议对齐）
//
// 上行三帧：G01 手势（文本行）/ G02 温度（文本行）/ G03 帧计数+状态（文本行，
// 待固件升后启用）＋ v2 codec 帧 T/S/A。事件环（G/S/A，容量64，满丢新+计数）；
// 遥测环（T/G02，容量8，满丢新）。SPSC：串口rx线程生产 / 逻辑线程消费。
// K 原始按键链已停用（KeyManager 退役）——G01 手势判定在 MCU，PC 直收。
// ============================================================================

#include "base/types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace Scanner::device::serial {

enum class KeyId : uint8_t { Up, Left, Middle, Right };   // U/L/M/R

struct TempFrame   { double celsius[4]; uint8_t channels; uint16_t seq; TimestampMs ts; };
// 260831 协议：G01 帧=键+手势位（1短/2双/3长），手势判定在 MCU——KeyManager 退役后
// KeySemantics 直收此型（08 设计 §1.4/§2.9；v2 K 帧原始按键链已停用）
struct GestureEvent {
    KeyId key;
    enum class Gesture : uint8_t { Short = 1, Double = 2, Hold = 3 } gesture;
    TimestampMs ts = 0;          // 文本行无 seq，ts 由 MCUDriver 到达时刻填
};
struct StatusFrame { uint8_t code; uint16_t seq; TimestampMs ts; };  // 码表待协议方（§8-8 占位）
struct AckFrame    { uint16_t ackedSeq; uint16_t seq; };

// —— 载荷解析（v3 默认口径；v2 由 MCUDriver 兜底路径绕过）——
// T: "$T25.3,24.8<seq><crc>;"  G01: 文本行 "G01 U1"  S: "$S0A<seq><crc>;"  A: "$A0B<seq><crc>;"
bool parseTempPayload(const std::string& payload, TempFrame& out);
bool parseG01Payload(const std::string& payload, GestureEvent& out);
bool parseStatusPayload(const std::string& payload, StatusFrame& out);
bool parseAckPayload(const std::string& payload, AckFrame& out);

// —— 满环丢新有界环（08 自带轻量实现——分层铁律 C2）——
// 口径（D-T12a）：满环时生产者放弃本帧（不写槽不推 head_），dropCount_++，
// push 返回 false。理由：① 生产者永不碰 head_——旧版满丢最旧需生产者推 head_，
// 与消费者 pop 的 head_.store 竞争、还可能覆写消费者正在读的槽（竞态根除）；
// ② 事件环 64 深下丢新=用户这一下没反应可重按（安全），丢旧=丢老事件。
template <typename T, size_t N>
class SpscRing {
public:
    bool push(T v) {               // 生产者：满则丢新（返回 false，未入队）
        if (full()) { dropCount_.fetch_add(1, std::memory_order_relaxed); return false; }
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
