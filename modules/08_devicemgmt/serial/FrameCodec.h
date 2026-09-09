#pragma once
// ============================================================================
// FrameCodec.h — 裸文本成拆帧器（协议 260831 定稿口径）
// 帧 = "载荷;"——ASCII、';' 结尾分帧、无校验无 seq（协议说明.md §一）。
// 载荷允许空格（"N10 H30 B60 ..." / "G02 A25.3 B25.4 ..."）。
// 半帧超时 50ms（advanceTimeout 推进——防御截断滞留，rx 线程驱动）。
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace Scanner::device::serial {

class FrameCodec {
public:
    FrameCodec() = default;

    struct Frame { std::string payload; };        // 拆帧产物

    // 喂字节流（串口rx线程调）：内部累积+分帧，完整帧追加到 out
    void feed(const std::string& bytes, std::vector<Frame>& out);
    // 逻辑时钟推进（半帧超时）：超时丢弃挂起缓冲并复位（返回是否发生丢弃）
    // 线程契约：须与 feed 同线程调用（挂起缓冲无锁）
    bool advanceTimeout(int64_t ms);
    // 组帧：payload + ";"
    std::string encode(const std::string& payload) const;

private:
    std::string pending_;     // 挂起缓冲（未收到 ';'）
    int64_t pendingAgeMs_ = 0;
};

} // namespace Scanner::device::serial
