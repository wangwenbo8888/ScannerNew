#pragma once
// ============================================================================
// FrameBuffer.h — 帧缓冲池（DataPlane）
//
// 实现 IFrameSink，包装 RingBuffer<FrameData>。
// HAL 采集 → pushFrame() → Workflow Stage0 popFrame() 消费。
// 溢出策略：DropOldest（实时扫描优先最新帧）。
// ============================================================================

#include "IFrameSink.h"
#include "RingBuffer.h"
#include <optional>
#include <chrono>

namespace Scanner::data {

class FrameBuffer : public IFrameSink {
public:
    explicit FrameBuffer(size_t capacity = 60,
                         OverflowPolicy policy = OverflowPolicy::DropOldest);

    // IFrameSink
    Result pushFrame(const FrameData& frame) override;
    int getBufferLevel() const override;
    Result reset() override;

    // Workflow Stage0 消费
    std::optional<FrameData> popFrame(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(100));

private:
    RingBuffer<FrameData> ringBuffer_;
};

} // namespace Scanner::data
