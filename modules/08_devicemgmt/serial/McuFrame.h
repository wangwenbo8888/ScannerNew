#pragma once
// ============================================================================
// McuFrame.h — MCU 上行帧类型 + 分流双有界环（协议 260831 口径）
//
// 上行三类：G01 手势 / G02 四路温度 / G03 触发计数（协议说明.md §三）。
// 环：手势/计数事件环容量 64，温度遥测环容量 8（满丢新+计数）。
// SPSC：串口 rx 线程生产 / 逻辑线程消费。
// ============================================================================
#include "base/types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace Scanner::device::serial {

enum class KeyId : uint8_t { Up, Left, Middle, Right };   // G01 键 U/L/M/R
enum class Gesture : uint8_t { Short = 1, Double = 2, Hold = 3 };  // G01 手势位

struct TempFrame      { double celsius[4]; TimestampMs ts; };      // G02 恒 4 路
struct GestureEvent   { KeyId key; Gesture gesture; TimestampMs ts; };  // G01（MCU 已判手势）
struct ShotCountFrame { uint64_t count; TimestampMs ts; };         // G03 硬件触发计数

// —— 载荷解析（入参已由 FrameCodec 剥去 ';'；非法一律 false，out 先归零）——
// G01: "G01 U1"   G02: "G02 A25.3 B25.4 C26.0 D24.5"   G03: "G03 S500"
bool parseGesturePayload(const std::string& payload, GestureEvent& out);
bool parseTempPayload(const std::string& payload, TempFrame& out);
bool parseShotCountPayload(const std::string& payload, ShotCountFrame& out);

// —— 满环丢新有界环（同旧实现，口径 D-T12a）——
template <typename T, size_t N>
class SpscRing {
public:
    bool push(T v) {
        if (full()) { dropCount_.fetch_add(1, std::memory_order_relaxed); return false; }
        buf_[tail_] = std::move(v); tail_.store((tail_+1)%N); return true; }
    bool pop(T& out) {
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
