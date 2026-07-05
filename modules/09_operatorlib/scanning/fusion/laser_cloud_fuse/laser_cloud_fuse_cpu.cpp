#include "laser_cloud_fuse_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <atomic>
#include <cassert>

using namespace calib;

OperatorInfo getLaserCloudFuseCPUInfo() {
    return OperatorInfo{"LaserCloudFuseCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(02, LaserCloudFuseCPU);

// ============================================================
// Anonymous namespace helpers
// ============================================================

namespace {

constexpr int64_t kVoxelBias = (int64_t)1 << 20;
constexpr uint64_t kAxisBits = 21;
constexpr uint64_t kAxisMask = ((uint64_t)1 << kAxisBits) - 1;
constexpr double kLoadFactor = 0.7;

inline uint64_t hash64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// hash 高 8 位 → tag（0 保留给空槽，映射为 1）
inline uint8_t tagFromHash(uint64_t h) {
    uint8_t t = static_cast<uint8_t>(h >> 56);
    return t != 0 ? t : 1;
}

inline uint64_t packVoxelKey(int ix, int iy, int iz) {
    uint64_t bx = (uint64_t)((int64_t)ix + kVoxelBias);
    uint64_t by = (uint64_t)((int64_t)iy + kVoxelBias);
    uint64_t bz = (uint64_t)((int64_t)iz + kVoxelBias);
    if ((bx | by | bz) > kAxisMask) {
        throw std::overflow_error("Voxel coordinate out of 21-bit range");
    }
    return bx | (by << kAxisBits) | (bz << (kAxisBits + kAxisBits));
}

inline size_t nextPow2(size_t x) {
    if (x < 2) return 2;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    if (sizeof(size_t) > 4) x |= x >> 32;
    return x + 1;
}

} // anonymous namespace

// ============================================================
// Params
// ============================================================

void LaserCloudFuseCPUParams::validate() const {
    if (voxelSize <= 0.0f)
        throw std::invalid_argument("voxelSize must be > 0");
    if (saturationThreshold < 1)
        throw std::invalid_argument("saturationThreshold must be >= 1");
    if (reserveVoxelCount < 64)
        throw std::invalid_argument("reserveVoxelCount must be >= 64");
}

nlohmann::json LaserCloudFuseCPUParams::toJson() const {
    nlohmann::json j;
    j["voxelSize"] = voxelSize;
    j["saturationThreshold"] = saturationThreshold;
    j["reserveVoxelCount"] = reserveVoxelCount;
    j["collectStatistics"] = collectStatistics;
    return j;
}

LaserCloudFuseCPUParams LaserCloudFuseCPUParams::fromJson(const nlohmann::json& j) {
    LaserCloudFuseCPUParams p;
    if (j.contains("voxelSize")) p.voxelSize = j.at("voxelSize").get<float>();
    if (j.contains("saturationThreshold")) p.saturationThreshold = j.at("saturationThreshold").get<int>();
    if (j.contains("reserveVoxelCount")) p.reserveVoxelCount = j.at("reserveVoxelCount").get<size_t>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    return p;
}

// ============================================================
// Impl — SoA flat hash table（方案1）+ vector 首点缓冲（方案2）
// ============================================================

struct LaserCloudFuseCPU::Impl {
    LaserCloudFuseCPUParams params_;
    LaserCloudFuseCPUStats stats_;
    bool warmed_up_ = false;

    // SoA 哈希表：4 个并行数组，tag 预过滤减少 cache miss
    std::vector<uint8_t>  tags_;       // 0=空槽, 1-255=hash高8位
    std::vector<uint64_t> keys_;       // 体素键（仅 tag 匹配时读取）
    std::vector<uint32_t> fusedIdx_;   // → fusedPoints_ 下标（4B 替代 8B 指针）
    std::vector<uint32_t> counts_;     // 命中计数（仅命中时读取）

    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;

    std::vector<CloudPoint> fusedPoints_;  // 连续存储，reserve 预分配

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserCloudFuseCPUParams& p) : params_(p) {
        params_.validate();
        rehashTo(params_.reserveVoxelCount);
        fusedPoints_.reserve(params_.reserveVoxelCount);
    }

    void rehashTo(size_t desired) {
        size_t newCap = nextPow2(std::max<size_t>(desired, 64));
        if (newCap <= capacity_) return;

        auto oldTags = std::move(tags_);
        auto oldKeys = std::move(keys_);
        auto oldFIdx = std::move(fusedIdx_);
        auto oldCnt  = std::move(counts_);

        tags_.assign(newCap, 0);
        keys_.assign(newCap, 0);
        fusedIdx_.assign(newCap, 0);
        counts_.assign(newCap, 0);

        capacity_ = newCap;
        mask_ = newCap - 1;
        size_ = 0;

        for (size_t i = 0; i < oldTags.size(); ++i) {
            if (oldTags[i] == 0) continue;
            placeSlot(oldKeys[i], oldFIdx[i], oldCnt[i]);
        }
    }

    void placeSlot(uint64_t key, uint32_t fi, uint32_t cnt) {
        uint64_t h = hash64(key);
        size_t idx = h & mask_;
        while (tags_[idx] != 0) {
            idx = (idx + 1) & mask_;
        }
        tags_[idx]     = tagFromHash(h);
        keys_[idx]     = key;
        fusedIdx_[idx] = fi;
        counts_[idx]   = cnt;
        ++size_;
    }

    bool ensureCapacityForOneMore() {
        if ((size_ + 1) > capacity_ * kLoadFactor) {
            rehashTo(capacity_ * 2);
            return true;
        }
        return false;
    }

    LaserCloudFuseCPUResult fuseImpl(const cv::Point3f* pts, size_t n,
                  const cv::Matx33d& R, const cv::Vec3d& T) {
        LaserCloudFuseCPUResult result;
        LaserCloudFuseCPUStats stats;

        auto totalStart = std::chrono::high_resolution_clock::now();

        if (n == 0) {
            result.success = false;
            result.message = "Empty frame";
            stats.totalVoxelCount = size_;
            if (params_.collectStatistics) result.statistics = stats;
            return result;
        }

        const uint32_t threshold = static_cast<uint32_t>(params_.saturationThreshold);
        const float inv = 1.0f / params_.voxelSize;

        std::vector<cv::Point3f>& surviving = result.survivingPoints;
        surviving.reserve(n);

        size_t survivingCount = 0, deletedCount = 0, newVoxelCount = 0;

        auto hashStart = std::chrono::high_resolution_clock::now();

        // 缓存裸指针：减少 vector::operator[] 的边界检查开销
        uint8_t*  tagsData  = tags_.data();
        uint64_t* keysData  = keys_.data();
        uint32_t* fidxData  = fusedIdx_.data();
        uint32_t* cntData   = counts_.data();

        for (size_t i = 0; i < n; ++i) {
            const cv::Point3f& raw = pts[i];
            cv::Point3f p(
                static_cast<float>(R(0,0)*raw.x + R(0,1)*raw.y + R(0,2)*raw.z + T(0)),
                static_cast<float>(R(1,0)*raw.x + R(1,1)*raw.y + R(1,2)*raw.z + T(1)),
                static_cast<float>(R(2,0)*raw.x + R(2,1)*raw.y + R(2,2)*raw.z + T(2)));
            int ix = static_cast<int>(std::floor(p.x * inv));
            int iy = static_cast<int>(std::floor(p.y * inv));
            int iz = static_cast<int>(std::floor(p.z * inv));
            uint64_t key = packVoxelKey(ix, iy, iz);

            uint64_t h = hash64(key);
            uint8_t tag = tagFromHash(h);

            size_t idx = h & mask_;
            while (tagsData[idx] != 0) {
                if (tagsData[idx] == tag && keysData[idx] == key) break;
                idx = (idx + 1) & mask_;
            }

            if (tagsData[idx] != 0) {
                if (cntData[idx] >= threshold) {
                    ++deletedCount;
                } else {
                    ++cntData[idx];
                    surviving.push_back(p);
                    ++survivingCount;
                }
            } else {
                if (ensureCapacityForOneMore()) {
                    tagsData = tags_.data();
                    keysData = keys_.data();
                    fidxData = fusedIdx_.data();
                    cntData  = counts_.data();
                    idx = h & mask_;
                    while (tagsData[idx] != 0) {
                        if (tagsData[idx] == tag && keysData[idx] == key) break;
                        idx = (idx + 1) & mask_;
                    }
                }
                uint32_t fi = static_cast<uint32_t>(fusedPoints_.size());
                fusedPoints_.push_back(CloudPoint{p.x, p.y, p.z, 0.0f, 0.0f, 0.0f});
                tagsData[idx] = tag;
                keysData[idx] = key;
                fidxData[idx] = fi;
                cntData[idx]  = 1;
                ++size_;
                ++newVoxelCount;
                surviving.push_back(p);
                ++survivingCount;
            }
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats.hashTimeMs = std::chrono::duration<double, std::milli>(totalEnd - hashStart).count();
        stats.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
        stats.inputCount = n;
        stats.survivingCount = survivingCount;
        stats.deletedCount = deletedCount;
        stats.newVoxelCount = newVoxelCount;
        stats.totalVoxelCount = size_;

        result.success = true;
        result.message = "Frame fusion completed";
        double keepRatio = static_cast<double>(survivingCount) / static_cast<double>(n);
        result.qualityFlag = keepRatio >= 0.5 ? calib::QualityFlag::Normal :
                             (keepRatio >= 0.1 ? calib::QualityFlag::Degraded :
                                                calib::QualityFlag::Warning);
        if (params_.collectStatistics) result.statistics = stats;
        stats_ = stats;

        CALIB_LOG_DEBUG("fuse: in={} kept={} del={} new={} voxels={} t={:.3f}ms",
                        n, survivingCount, deletedCount, newVoxelCount, size_, stats.totalTimeMs);

        return result;
    }

    size_t GatherVoxelNeighbors(const cv::Point3f& worldPos, int kernelRadius,
                                std::vector<const CloudPoint*>& out) const {
        const float inv = 1.0f / params_.voxelSize;
        int ix = static_cast<int>(std::floor(worldPos.x * inv));
        int iy = static_cast<int>(std::floor(worldPos.y * inv));
        int iz = static_cast<int>(std::floor(worldPos.z * inv));

        out.clear();

        for (int dz = -kernelRadius; dz <= kernelRadius; ++dz)
        for (int dy = -kernelRadius; dy <= kernelRadius; ++dy)
        for (int dx = -kernelRadius; dx <= kernelRadius; ++dx) {
            uint64_t key;
            try {
                key = packVoxelKey(ix + dx, iy + dy, iz + dz);
            } catch (const std::overflow_error&) {
                continue;  // coordinate out of 21-bit range, skip
            }
            uint64_t h = hash64(key);
            uint8_t  tag = tagFromHash(h);
            size_t idx = h & mask_;
            while (tags_[idx] != 0) {
                if (tags_[idx] == tag && keys_[idx] == key) {
                    out.push_back(&fusedPoints_[fusedIdx_[idx]]);
                    break;
                }
                idx = (idx + 1) & mask_;
            }
        }
        return out.size();
    }
};

// ============================================================
// Constructor / Destructor
// ============================================================

LaserCloudFuseCPU::LaserCloudFuseCPU(const LaserCloudFuseCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserCloudFuseCPU initialized (voxelSize={}, threshold={}, reserve={})",
                   params.voxelSize, params.saturationThreshold, params.reserveVoxelCount);
}

LaserCloudFuseCPU::~LaserCloudFuseCPU() = default;

// ============================================================
// Execute
// ============================================================

LaserCloudFuseCPUResult LaserCloudFuseCPU::Execute(const std::vector<cv::Point3f>& frame) {
    return pImpl_->fuseImpl(frame.data(), frame.size(),
                             cv::Matx33d::eye(), cv::Vec3d(0,0,0));
}

LaserCloudFuseCPUResult LaserCloudFuseCPU::Execute(const cv::Point3f* pts, size_t count) {
    return pImpl_->fuseImpl(pts, count,
                             cv::Matx33d::eye(), cv::Vec3d(0,0,0));
}

LaserCloudFuseCPUResult LaserCloudFuseCPU::Execute(const std::vector<cv::Point3f>& frame,
                             const cv::Matx33d& R, const cv::Vec3d& T) {
    return pImpl_->fuseImpl(frame.data(), frame.size(), R, T);
}

LaserCloudFuseCPUResult LaserCloudFuseCPU::Execute(const cv::Point3f* pts, size_t count,
                             const cv::Matx33d& R, const cv::Vec3d& T) {
    return pImpl_->fuseImpl(pts, count, R, T);
}

// ============================================================
// Destroy
// ============================================================

void LaserCloudFuseCPU::Destroy() {
}

// ============================================================
// capacity / Warmup / clear
// ============================================================

void LaserCloudFuseCPU::Reserve(size_t voxelCount) {
    size_t want = nextPow2(std::max<size_t>(voxelCount, 64));
    if (want > pImpl_->capacity_) pImpl_->rehashTo(want);
}

void LaserCloudFuseCPU::Warmup(int maxPointCount) {
    pImpl_->warmed_up_ = true;
    CALIB_LOG_INFO("Warmup: maxPointCount={}", maxPointCount);
}

void LaserCloudFuseCPU::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.maxPointCount);
}

void LaserCloudFuseCPU::Clear() noexcept {
    std::fill(pImpl_->tags_.begin(), pImpl_->tags_.end(), static_cast<uint8_t>(0));
    pImpl_->size_ = 0;
    pImpl_->fusedPoints_.clear();
    pImpl_->stats_ = LaserCloudFuseCPUStats();
}

// ============================================================
// accessors
// ============================================================

size_t LaserCloudFuseCPU::GetVoxelCount() const noexcept {
    return pImpl_->size_;
}

size_t LaserCloudFuseCPU::GetFusedPointCount() const noexcept {
    return pImpl_->fusedPoints_.size();
}

CloudPoint* LaserCloudFuseCPU::FusedPointPtr(size_t index) {
    return &pImpl_->fusedPoints_[index];
}

const std::vector<CloudPoint>& LaserCloudFuseCPU::GetFusedPoints() const {
    return pImpl_->fusedPoints_;
}

void LaserCloudFuseCPU::SnapshotVoxels(std::vector<VoxelInfo>& out) const {
    out.clear();
    out.reserve(pImpl_->size_);
    for (size_t i = 0; i < pImpl_->capacity_; ++i) {
        if (pImpl_->tags_[i] != 0) {
            out.push_back(VoxelInfo{
                pImpl_->keys_[i],
                &pImpl_->fusedPoints_[pImpl_->fusedIdx_[i]],
                pImpl_->counts_[i]
            });
        }
    }
}

void LaserCloudFuseCPU::SetParams(const LaserCloudFuseCPUParams& params) {
#ifndef NDEBUG
    assert(!pImpl_->inProcess_.load() && "setParams() while processing");
#endif
    params.validate();
    pImpl_->params_ = params;
    pImpl_->warmed_up_ = false;
}

const LaserCloudFuseCPUParams& LaserCloudFuseCPU::GetParams() const {
    return pImpl_->params_;
}

const LaserCloudFuseCPUStats& LaserCloudFuseCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}

void LaserCloudFuseCPU::ResetStatistics() noexcept {
    pImpl_->stats_ = LaserCloudFuseCPUStats();
}

// ============================================================
// gatherVoxelNeighbors
// ============================================================

size_t LaserCloudFuseCPU::GatherVoxelNeighbors(
    const cv::Point3f& worldPos, int kernelRadius,
    std::vector<const CloudPoint*>& outNeighbors) const {
    return pImpl_->GatherVoxelNeighbors(worldPos, kernelRadius, outNeighbors);
}
