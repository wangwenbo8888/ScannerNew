// ============================================================================
// CommandChannel.cpp — 实现见头文件契约（协议 260831：盲发口径 D8）
// ============================================================================
#include "serial/CommandChannel.h"

namespace Scanner::device::serial {

CommandChannel::CommandChannel(Deps d) : deps_(std::move(d)) {
    if (!deps_.write) deps_.write = [](const std::string&) { return false; };  // 空写=恒失败
    // codec 空保留：裸载荷直发（现口径延续）
}

bool CommandChannel::writeFrame(const std::string& payload) {
    const std::string frame = deps_.codec ? deps_.codec->encode(payload) : payload;
    return deps_.write ? deps_.write(frame) : false;
}

void CommandChannel::send(const std::string& payload, DoneCb onDone) {
    const bool ok = writeFrame(payload);
    if (onDone) onDone(ok, ok ? "未确认" : payload);   // 写成败同步回告（写队列即返）
}

void CommandChannel::sendFireAndForget(const std::string& payload) {
    writeFrame(payload);
}

void CommandChannel::sendGroup(std::vector<std::string> payloads, DoneCb onGroupDone) {
    if (payloads.empty()) {
        if (onGroupDone) onGroupDone(true, "");
        return;
    }
    std::string last;
    for (size_t i = 0; i < payloads.size(); ++i) {   // 索引循环：回调内可再 send（无迭代器失效）
        if (!writeFrame(payloads[i])) {
            if (onGroupDone) onGroupDone(false, payloads[i]);   // 写败短路：后续不发
            return;
        }
        last = payloads[i];
    }
    if (onGroupDone) onGroupDone(true, last);        // 组成功载荷=末步载荷
}

} // namespace Scanner::device::serial
