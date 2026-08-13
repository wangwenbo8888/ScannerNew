#include "FrameBuffer.h"

namespace Scanner::data {

FrameBuffer::FrameBuffer(size_t capacity, OverflowPolicy policy)
    : ringBuffer_(capacity, policy) {}

Result FrameBuffer::pushFrame(const FrameData& frame) {
    bool ok = ringBuffer_.push(frame);
    if (!ok) return Result::warning("FrameBuffer: 帧被丢弃(DropNewest)");
    return Result::ok();
}

int FrameBuffer::getBufferLevel() const {
    return static_cast<int>(ringBuffer_.size());
}

Result FrameBuffer::reset() {
    ringBuffer_.clear();
    return Result::ok();
}

std::optional<FrameData> FrameBuffer::popFrame(std::chrono::milliseconds timeout) {
    return ringBuffer_.pop(timeout);
}

} // namespace Scanner::data
