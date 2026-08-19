// ISceneFeed.h — 渲染推送窄接口（07 定义，03/app 场景内容侧实现）
// 兑现 PipelineDeps.h 中 class ISceneFeed 的前向声明（同名空间，无需改动该文件）
#pragma once

#include "base/types.h"
#include <cstdint>
#include <vector>

namespace Scanner::pipeline {

struct CloudViewHandle { const void* hostMarker = nullptr; const void* deviceLaser = nullptr; };  // 句柄式，实施期细化

class ISceneFeed {
public:
    virtual ~ISceneFeed() = default;
    virtual void pushPostureView(const Scanner::Pose& live, int confirmedCount,
                                 const std::vector<uint8_t>& markerDetected) = 0;
    virtual void pushCloudSnapshot(CloudViewHandle cloud) = 0;
    virtual void notifyFreeze(bool frozen) = 0;
};

} // namespace Scanner::pipeline
