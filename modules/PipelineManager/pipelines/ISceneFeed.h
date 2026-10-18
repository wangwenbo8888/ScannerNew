// ISceneFeed.h — 渲染推送窄接口（07 定义，03/app 场景内容侧实现）
// 兑现 PipelineDeps.h 中 class ISceneFeed 的前向声明（同名空间，无需改动该文件）
#pragma once

#include "base/types.h"
#include <cstdint>
#include <vector>

namespace Scanner::pipeline {

struct CloudViewHandle {
    const void* hostMarker = nullptr;   // const std::vector<calib::MarkerCloudPoint>*
    const void* deviceLaser = nullptr;  // 预留 GPU 句柄（未实施）
    const float* hostLaser = nullptr;   // 激光点 host xyz（n×3 平铺；FuseConsumer
                                        // 帧块下载直推——2026-08-31 激光显示）
    size_t laserCount = 0;              // hostLaser 点数
    bool laserAppend = false;           // 261018 增量推送：hostLaser＝自上次推送后
                                        // 的新增尾部（融合云追加语义——显示端只
                                        // 追加不重建；false＝全量替换口径）
    size_t laserTotal = 0;              // 增量时＝融合云累计全量点数（计数对账）
};  // 句柄式，实施期细化

class ISceneFeed {
public:
    virtual ~ISceneFeed() = default;
    virtual void pushPostureView(const Scanner::Pose& live, int confirmedCount,
                                 const std::vector<uint8_t>& markerDetected) = 0;
    virtual void pushCloudSnapshot(CloudViewHandle cloud) = 0;
    virtual void notifyFreeze(bool frozen) = 0;
    /// 261018 轻计数通道：会话激光融合累计点数（host 计数器，逐帧零成本）——
    /// 与重推送（3s 节流）解耦，UI 统计列表亚秒刷新用。默认 no-op 兼容测试
    virtual void notifyLaserCount(size_t sessionPts) { (void)sessionPts; }
};

} // namespace Scanner::pipeline
