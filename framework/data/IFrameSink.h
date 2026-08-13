#pragma once
// ============================================================================
// IFrameSink.h — 帧数据 Sink（Data 层拥有，注入 HAL）
//
// 依赖倒置：HAL 产出帧数据推给 Sink，不反向依赖 Data 实现。
// ============================================================================

#include "common/types.h"
#include <opencv2/core.hpp>

namespace Scanner::data {

// ============================================================================
// 帧数据（双目，由 HAL 推送）
// ============================================================================
struct FrameData {
    FrameId frameId = 0;
    TimestampMs timestamp = 0;
    cv::Mat leftGray;
    cv::Mat rightGray;
};

// ============================================================================
// IFrameSink — 帧数据接收接口
// ============================================================================
class IFrameSink {
public:
    virtual ~IFrameSink() = default;

    /// HAL 调用：推送一帧双目数据
    virtual Result pushFrame(const FrameData& frame) = 0;

    /// 查询当前缓冲水位
    virtual int getBufferLevel() const = 0;

    /// 重置缓冲区
    virtual Result reset() = 0;
};

} // namespace Scanner::data
