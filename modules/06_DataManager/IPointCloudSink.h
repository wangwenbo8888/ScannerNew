#pragma once
// ============================================================================
// IPointCloudSink.h — 点云 Sink（Data 层拥有）
// ============================================================================

#include "base/types.h"
#include <opencv2/core.hpp>
#include <vector>

namespace Scanner::data {

struct PointCloudFrame {
    FrameId frameId = 0;
    Pose pose;                          // 本帧位姿（世界系）
    std::vector<cv::Point3f> points;    // 三维点
    std::vector<cv::Vec3b> colors;      // 颜色（可选）
    int pointCount = 0;
};

class IPointCloudSink {
public:
    virtual ~IPointCloudSink() = default;

    /// 融合线程调用：写入一帧点云
    virtual Result pushPointCloud(const PointCloudFrame& cloud) = 0;

    /// 渲染线程调用：获取只读快照（版本号，无锁）
    virtual Result getSnapshot(uint64_t& version,
                               std::vector<cv::Point3f>& points,
                               std::vector<cv::Vec3b>& colors) const = 0;

    /// 查询点云总数
    virtual int getTotalPointCount() const = 0;

    /// 重置
    virtual Result clear() = 0;
};

} // namespace Scanner::data
