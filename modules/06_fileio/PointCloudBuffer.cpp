#include "PointCloudBuffer.h"

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

int PointCloudBuffer::getTotalPointCount() const {
    return totalPoints_.load(std::memory_order_acquire);
}

Result PointCloudBuffer::clear() {
    {
        std::unique_lock lock(rwlock_);
        allPoints_.clear();
        allColors_.clear();
    }
    version_.fetch_add(1, std::memory_order_release);
    totalPoints_.store(0, std::memory_order_release);
    return Result::ok();
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

} // namespace Scanner::data
