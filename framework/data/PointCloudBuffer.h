#pragma once
// ============================================================================
// PointCloudBuffer.h — 全局点云缓冲（DataPlane）
//
// 实现 IPointCloudSink。读写锁保护，融合线程写、渲染线程并发只读。
// 版本号机制：每次写入递增 version，UI 可检测变化后再取快照。
// ============================================================================

#include "IPointCloudSink.h"
#include <atomic>
#include <shared_mutex>

namespace Scanner::data {

class PointCloudBuffer : public IPointCloudSink {
public:
    PointCloudBuffer();

    // IPointCloudSink
    Result pushPointCloud(const PointCloudFrame& cloud) override;
    Result getSnapshot(uint64_t& version,
                       std::vector<cv::Point3f>& points,
                       std::vector<cv::Vec3b>& colors) const override;
    int getTotalPointCount() const override;
    Result clear() override;

private:
    mutable std::shared_mutex rwlock_;
    std::vector<cv::Point3f> allPoints_;
    std::vector<cv::Vec3b>   allColors_;
    std::atomic<uint64_t>    version_{0};
    std::atomic<int>         totalPoints_{0};
};

} // namespace Scanner::data
