#include "PointCloudBuffer.h"

#include "file_io.h"

#include <algorithm>   // replacePointCloud: min

namespace Scanner::data {

PointCloudBuffer::PointCloudBuffer() {}

Result PointCloudBuffer::pushPointCloud(const PointCloudFrame& cloud) {
    {
        std::unique_lock lock(rwlock_);
        allPoints_.insert(allPoints_.end(),
                          cloud.points.begin(),
                          cloud.points.begin() + cloud.pointCount);
        if (!cloud.colors.empty()) {
            allColors_.insert(allColors_.end(),
                              cloud.colors.begin(),
                              cloud.colors.begin() + cloud.pointCount);
        }
    }
    version_.fetch_add(1, std::memory_order_release);
    totalPoints_.store(static_cast<int>(allPoints_.size()), std::memory_order_release);
    return Result::ok();
}

Result PointCloudBuffer::getSnapshot(uint64_t& version,
                                     std::vector<cv::Point3f>& points,
                                     std::vector<cv::Vec3b>& colors) const {
    {
        std::shared_lock lock(rwlock_);
        points = allPoints_;
        colors = allColors_;
        version = version_.load(std::memory_order_acquire);
    }
    return Result::ok();
}

Result PointCloudBuffer::replacePointCloud(const PointCloudFrame& cloud) {
    const size_t n = std::min(cloud.points.size(), static_cast<size_t>(cloud.pointCount));
    {
        std::unique_lock lock(rwlock_);
        allPoints_.assign(cloud.points.begin(), cloud.points.begin() + n);
        allColors_.clear();
        if (cloud.colors.size() >= n)
            allColors_.assign(cloud.colors.begin(), cloud.colors.begin() + n);
    }
    version_.fetch_add(1, std::memory_order_release);
    totalPoints_.store(static_cast<int>(n), std::memory_order_release);
    return Result::ok();
}

int PointCloudBuffer::getTotalPointCount() const {
    return totalPoints_.load(std::memory_order_acquire);
}

Result PointCloudBuffer::clear() {
    {
        std::unique_lock lock(rwlock_);
        allPoints_.clear();
        allColors_.clear();
        cloudBaseCount_ = 0;
    }
    version_.fetch_add(1, std::memory_order_release);
    totalPoints_.store(0, std::memory_order_release);
    return Result::ok();
}

// ============================================================================
// ICloudWarehouse —— 07 融合线程直写正式口（260919；语义见 ICloudWarehouse.h）
// ============================================================================

void PointCloudBuffer::beginCloudSession() {
    std::unique_lock lock(rwlock_);
    cloudBaseCount_ = allPoints_.size();   // 既有内容（含上一会话成果）折入基线
    version_.fetch_add(1, std::memory_order_release);
}

void PointCloudBuffer::pushSessionCloud(const std::vector<float>& xyzInterleaved) {
    const size_t n = xyzInterleaved.size() / 3;
    {
        std::unique_lock lock(rwlock_);
        // 会话层替换（累计快照语义）：截回基线前缀再重建会话段——跨会话只增
        // 不减、会话内恒等当前融合云（导出/计数皆真）
        allPoints_.resize(cloudBaseCount_);
        allPoints_.reserve(cloudBaseCount_ + n);
        for (size_t i = 0; i < n; ++i)
            allPoints_.emplace_back(xyzInterleaved[i * 3],
                                    xyzInterleaved[i * 3 + 1],
                                    xyzInterleaved[i * 3 + 2]);
        if (!allColors_.empty()) allColors_.clear();   // 会话层无色（显示侧自配色）
    }
    version_.fetch_add(1, std::memory_order_release);
    totalPoints_.store(static_cast<int>(cloudBaseCount_ + n), std::memory_order_release);
}

void PointCloudBuffer::setMarkers(const std::vector<MarkerRecord>& markers) {
    {
        std::unique_lock lock(markerRwlock_);
        markers_ = markers;
    }
    markerVersion_.fetch_add(1, std::memory_order_release);
}

void PointCloudBuffer::snapshotMarkers(uint64_t& version,
                                       std::vector<MarkerRecord>& out) const {
    std::shared_lock lock(markerRwlock_);
    out = markers_;
    version = markerVersion_.load(std::memory_order_acquire);
}

bool PointCloudBuffer::exportCloud(const std::string& path) {
    std::vector<cv::Point3f> points;
    {
        std::shared_lock lock(rwlock_);
        points = allPoints_;
    }
    return fileio::exportPointCloud(path, points);
}

bool PointCloudBuffer::exportMarkers(const std::string& path) {
    std::vector<MarkerRecord> markers;
    {
        std::shared_lock lock(markerRwlock_);
        markers = markers_;
    }
    std::vector<cv::Point3f> pts;
    pts.reserve(markers.size());
    for (const auto& m : markers) pts.push_back(m.pos);
    return fileio::exportMarkers(path, pts);
}

} // namespace Scanner::data
