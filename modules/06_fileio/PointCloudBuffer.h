#pragma once
// ============================================================================
// PointCloudBuffer.h — 全局点云缓冲（DataPlane）
//
// 实现 IPointCloudSink。读写锁保护，融合线程写、渲染线程并发只读。
// 版本号机制：每次写入递增 version，UI 可检测变化后再取快照。
// ============================================================================

#include "IPointCloudSink.h"
#include <atomic>
#include <cstdint>
#include <shared_mutex>

namespace Scanner::data {

// 续扫基准标志点（会话尾产物）：globalId 保真走内存通道，落盘格式只有坐标
struct MarkerRecord {
    uint32_t globalId = 0;
    cv::Point3f pos;
    cv::Vec3f normal;
};

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

    // marker 通道：独立锁＋独立版本号（与点云版本分家）
    void setMarkers(const std::vector<MarkerRecord>& markers);
    void snapshotMarkers(uint64_t& version, std::vector<MarkerRecord>& out) const;

private:
    mutable std::shared_mutex rwlock_;
    std::vector<cv::Point3f> allPoints_;
    std::vector<cv::Vec3b>   allColors_;
    std::atomic<uint64_t>    version_{0};
    std::atomic<int>         totalPoints_{0};

    mutable std::shared_mutex markerRwlock_;
    std::vector<MarkerRecord> markers_;
    std::atomic<uint64_t>     markerVersion_{0};
};

} // namespace Scanner::data
