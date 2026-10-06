// ============================================================================
// GpuPointCloudPool.cpp — GPU 点云显存块池实现
// ============================================================================
#include <spdlog/spdlog.h>
#include "jmw_logging.h"

#include "pipelines/scan/GpuPointCloudPool.h"

#ifdef JMW_BUILD_CUDA

namespace Scanner::pipeline {

GpuPointCloudPool::GpuPointCloudPool(size_t blocks, size_t pointsPerBlock, AllocFn alloc) {
    blocks_.reserve(blocks);
    free_.reserve(blocks);
    for (size_t i = 0; i < blocks; ++i) {
        auto b = std::make_unique<GpuPointCloudBlock>();
        if (alloc) {
            b->points = alloc(pointsPerBlock);                  // 注入分配器（测试假分配器）
        } else {
            b->points.create(1, static_cast<int>(pointsPerBlock), CV_32FC3);  // 真分配（需 GPU）
        }
        b->slotId = static_cast<uint32_t>(i);
        free_.push_back(b.get());
        blocks_.push_back(std::move(b));
    }
}

GpuPointCloudPool::~GpuPointCloudPool() {
    if (size_t n = inUse(); n > 0) {
        JMW_LOG_WARN("07-GpuPointCloudPool", "GpuPointCloudPool: 析构时 inUse={}（生命周期契约违反：池须长于所有在飞块，"
                     "在飞块回池将触碰已亡池）", n);
    }
}

std::optional<std::shared_ptr<GpuPointCloudBlock>> GpuPointCloudPool::acquire(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!blockAvailable_.wait_for(lock, timeout, [this] { return !free_.empty(); }))
        return std::nullopt;                                     // 超时

    GpuPointCloudBlock* raw = free_.back();
    free_.pop_back();
    raw->count = 0;                                               // 防御：复用块顺手清 count
    // 块析构自动回池（LIFO 复用，内容可覆盖）；池生命周期须长于所有在飞块
    return std::shared_ptr<GpuPointCloudBlock>(raw, [this](GpuPointCloudBlock* b) { release(b); });
}

size_t GpuPointCloudPool::inUse() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size() - free_.size();
}

size_t GpuPointCloudPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_.size();
}

void GpuPointCloudPool::release(GpuPointCloudBlock* b) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(b);
    }
    blockAvailable_.notify_one();
}

} // namespace Scanner::pipeline

#endif // JMW_BUILD_CUDA
