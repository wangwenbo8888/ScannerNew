// ============================================================================
// FrameObsAccumulator.cpp — 逐帧观测累加器 + 激光帧缓存实现
// ============================================================================
#include "pipelines/scan/FrameObsAccumulator.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <utility>

namespace Scanner::pipeline {

namespace {
// 检查点格式：魔术字｜版本｜obs 段｜激光段｜降级标志（小端原生布局，同机恢复）
constexpr char kMagic[8] = {'J', 'M', 'W', 'F', 'O', 'B', 'S', '1'};
constexpr uint32_t kVersion = 1;

template<typename T>
void wr(std::ostream& o, const T& v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template<typename T>
bool rd(std::istream& i, T& v) {
    return static_cast<bool>(i.read(reinterpret_cast<char*>(&v), sizeof(T)));
}
} // namespace

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

Scanner::Result FrameObsAccumulator::saveCheckpoint(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return Scanner::Result::fail("检查点打开失败: " + path);
    out.write(kMagic, sizeof(kMagic));
    wr(out, kVersion);
    wr(out, static_cast<uint64_t>(obsList_.size()));
    for (const auto& fo : obsList_) {
        wr(out, fo.frameId);
        out.write(reinterpret_cast<const char*>(fo.R_init), sizeof(fo.R_init));
        out.write(reinterpret_cast<const char*>(fo.t_init), sizeof(fo.t_init));
        wr(out, static_cast<uint64_t>(fo.markerObs.size()));
        for (const auto& mo : fo.markerObs) {
            out.write(reinterpret_cast<const char*>(mo.xyz), sizeof(mo.xyz));
            wr(out, mo.globalId);
            wr(out, static_cast<uint8_t>(mo.isHighPrecision ? 1 : 0));
        }
    }
    wr(out, static_cast<uint64_t>(laserFrames_.size()));
    for (const auto& lf : laserFrames_) {
        wr(out, static_cast<uint64_t>(lf.size()));
        if (!lf.empty())
            out.write(reinterpret_cast<const char*>(lf.data()), lf.size() * sizeof(float));
    }
    wr(out, static_cast<uint8_t>(degradedLaser_.load(std::memory_order_relaxed) ? 1 : 0));
    out.flush();
    if (!out) return Scanner::Result::fail("检查点写入失败: " + path);
    return Scanner::Result::ok();
}

Scanner::Result FrameObsAccumulator::loadCheckpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return Scanner::Result::fail("检查点打开失败: " + path);
    char magic[8];
    if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        return Scanner::Result::fail("检查点魔术字不符(坏档): " + path);
    uint32_t ver = 0;
    if (!rd(in, ver) || ver != kVersion)
        return Scanner::Result::fail("检查点版本不符: " + path);

    Snapshot snap;
    uint8_t degraded = 0;
    try {
        uint64_t obsCount = 0;
        if (!rd(in, obsCount)) return Scanner::Result::fail("检查点 obs 段截断");
        snap.obs.reserve(static_cast<size_t>(obsCount));
        for (uint64_t k = 0; k < obsCount; ++k) {
            FrameObs fo;
            if (!rd(in, fo.frameId) ||
                !in.read(reinterpret_cast<char*>(fo.R_init), sizeof(fo.R_init)) ||
                !in.read(reinterpret_cast<char*>(fo.t_init), sizeof(fo.t_init)))
                return Scanner::Result::fail("检查点 obs 记录截断");
            uint64_t mc = 0;
            if (!rd(in, mc)) return Scanner::Result::fail("检查点 marker 段截断");
            fo.markerObs.resize(static_cast<size_t>(mc));
            for (auto& mo : fo.markerObs) {
                uint8_t hp = 0;
                if (!in.read(reinterpret_cast<char*>(mo.xyz), sizeof(mo.xyz)) ||
                    !rd(in, mo.globalId) || !rd(in, hp))
                    return Scanner::Result::fail("检查点 marker 记录截断");
                mo.isHighPrecision = hp != 0;
            }
            snap.obs.push_back(std::move(fo));
        }
        uint64_t lfCount = 0;
        if (!rd(in, lfCount)) return Scanner::Result::fail("检查点激光段截断");
        snap.laserFrames.reserve(static_cast<size_t>(lfCount));
        for (uint64_t k = 0; k < lfCount; ++k) {
            uint64_t n = 0;
            if (!rd(in, n)) return Scanner::Result::fail("检查点激光帧截断");
            std::vector<float> lf(static_cast<size_t>(n));
            if (n && !in.read(reinterpret_cast<char*>(lf.data()),
                              static_cast<std::streamsize>(n * sizeof(float))))
                return Scanner::Result::fail("检查点激光数据截断");
            snap.laserFrames.push_back(std::move(lf));
        }
        if (!rd(in, degraded)) degraded = 0;
    } catch (const std::exception& e) {
        return Scanner::Result::fail(std::string("检查点读异常: ") + e.what());
    }
    replace(std::move(snap), degraded != 0);
    return Scanner::Result::ok();
}

void FrameObsAccumulator::replace(Snapshot&& snap, bool degraded) {
    size_t bytes = 0;
    for (const auto& lf : snap.laserFrames) bytes += lf.size() * sizeof(float);
    std::lock_guard<std::mutex> lock(mu_);
    obsList_ = std::move(snap.obs);
    laserFrames_ = std::move(snap.laserFrames);
    usedBytes_ = bytes;
    degradedLaser_.store(degraded, std::memory_order_relaxed);
}

} // namespace Scanner::pipeline
