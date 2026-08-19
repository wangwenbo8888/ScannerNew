// ============================================================================
// FrameObsAccumulator.cpp — 逐帧观测累加器 + 激光帧缓存实现
// ============================================================================
#include "pipelines/scan/FrameObsAccumulator.h"

#include <utility>

namespace Scanner::pipeline {

FrameObsAccumulator::FrameObsAccumulator(size_t laserBudgetBytes)
    : budgetBytes_(laserBudgetBytes) {}

void FrameObsAccumulator::push(FrameObs obs, const std::vector<float>& laserXyz) {
    obs.laserCacheSlot = FrameObs::kNoLaserSlot;    // 无激光/超限均无槽
    std::lock_guard<std::mutex> lock(mu_);
    if (!laserXyz.empty()) {
        const size_t bytes = laserXyz.size() * sizeof(float);
        if (!degradedLaser_.load(std::memory_order_relaxed) &&
            usedBytes_ + bytes <= budgetBytes_) {   // 预算内（<= 判定）
            laserFrames_.push_back(laserXyz);
            obs.laserCacheSlot = laserFrames_.size() - 1;
            usedBytes_ += bytes;
        } else {
            degradedLaser_.store(true, std::memory_order_relaxed);  // 超限停累加
        }
    }
    obsList_.push_back(std::move(obs));             // markerObs 照常累加，整体不失败
}

bool FrameObsAccumulator::degradedLaser() const {
    return degradedLaser_.load(std::memory_order_relaxed);
}

size_t FrameObsAccumulator::laserBytesUsed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return usedBytes_;
}

size_t FrameObsAccumulator::frameCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return obsList_.size();
}

FrameObsAccumulator::Snapshot FrameObsAccumulator::snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    Snapshot s;
    s.obs = obsList_;
    s.laserFrames = laserFrames_;
    return s;
}

void FrameObsAccumulator::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    obsList_.clear();
    laserFrames_.clear();
    usedBytes_ = 0;
    degradedLaser_.store(false, std::memory_order_relaxed);
}

} // namespace Scanner::pipeline
