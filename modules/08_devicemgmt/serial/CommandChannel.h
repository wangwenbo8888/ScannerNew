#pragma once
// ============================================================================
// CommandChannel.h — 下行通道（协议 260831：无 ACK——盲发口径 D8）
// 职责：encode ＋ 写出口，同步回告写成败。sendGroup＝依序直发，
// 任一步写失败整组短路。单线程属主：逻辑线程。
// ============================================================================
#include "serial/FrameCodec.h"
#include <functional>
#include <string>
#include <vector>

namespace Scanner::device::serial {

class CommandChannel {
public:
    using DoneCb = std::function<void(bool ok, const std::string& payload)>;

    struct Deps {
        FrameCodec* codec = nullptr;                          // 组帧（encode）
        std::function<bool(const std::string& frame)> write;  // 写帧（返回是否成功）
    };

    explicit CommandChannel(Deps d);

    void send(const std::string& payload, DoneCb onDone);     // 组帧+写，立即回告写成败
    void sendFireAndForget(const std::string& payload);       // 组帧+写（无回调）
    void sendGroup(std::vector<std::string> payloads, DoneCb onGroupDone);  // 依序直发，写败短路

private:
    bool writeFrame(const std::string& payload);
    Deps deps_;
};

} // namespace Scanner::device::serial
