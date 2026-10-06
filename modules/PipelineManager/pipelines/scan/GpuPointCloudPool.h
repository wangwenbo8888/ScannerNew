#pragma once
// ============================================================================
// GpuPointCloudPool.h — GPU 点云显存块池（C 扫描流水线）
//
// 构造时预分配 blocks 块 cv::cuda::GpuMat（CV_32FC3 1×pointsPerBlock，真分配）；
// acquire 返 shared_ptr<GpuPointCloudBlock>，块析构经 custom deleter 自动回池
// （LIFO 复用，内容可覆盖——持有者必须重填 count/frameId）。
//
// 分配器注入：alloc 为空 → 真 GpuMat::create（需 GPU 设备）；
// 测试注入假分配器返回默认构造空 GpuMat（无 GPU 合法，字段不触驱动）。
//
// ⚠ 生命周期契约：Pool 须长于所有在飞块（块析构回池时解引用池内部 mutex/cv）。
// 线程安全：acquire/inUse/available 可多线程并发调用。
// ============================================================================
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "pipelines/scan/ScanTypes.h"

#ifdef JMW_BUILD_CUDA

namespace Scanner::pipeline {

class GpuPointCloudPool {
public:
    using AllocFn = std::function<cv::cuda::GpuMat(size_t n)>;   // 分配器注入（测试用假分配器）

    /// 预分配 blocks 块；alloc 为空时真 GpuMat::create（需 GPU）
    GpuPointCloudPool(size_t blocks, size_t pointsPerBlock, AllocFn alloc = {});
    ~GpuPointCloudPool();
    GpuPointCloudPool(const GpuPointCloudPool&) = delete;
    GpuPointCloudPool& operator=(const GpuPointCloudPool&) = delete;

    /// 取块（空则阻塞 cv 等待归还或超时 nullopt）
    std::optional<std::shared_ptr<GpuPointCloudBlock>> acquire(std::chrono::milliseconds timeout);

    size_t inUse() const;       // 在飞块数
    size_t available() const;   // 空闲块数

private:
    void release(GpuPointCloudBlock* b);

    mutable std::mutex mutex_;
    std::condition_variable blockAvailable_;
    std::vector<std::unique_ptr<GpuPointCloudBlock>> blocks_;  // 池拥有全部块
    std::vector<GpuPointCloudBlock*> free_;                    // 空闲栈（LIFO；size() 恒等承担计数）
};

} // namespace Scanner::pipeline

#endif // JMW_BUILD_CUDA
