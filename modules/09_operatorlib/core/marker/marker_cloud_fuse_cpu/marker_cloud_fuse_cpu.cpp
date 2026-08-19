#include "marker_cloud_fuse_cpu.h"
#include "common/calib_logging.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cassert>

using namespace calib;

OperatorInfo getMarkerCloudFuseCPUInfo() {
    return OperatorInfo{"MarkerCloudFuseCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(02M, MarkerCloudFuseCPU);

// ============================================================
// Anonymous namespace helpers（与 laser_cloud_fuse_cpu 一致）
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

void MarkerCloudFuseCPUParams::validate() const {
    if (voxelSize <= 0.0f)
        throw std::invalid_argument("voxelSize must be > 0");
    if (saturationThreshold < 1)
        throw std::invalid_argument("saturationThreshold must be >= 1");
    if (reserveVoxelCount < 64)
        throw std::invalid_argument("reserveVoxelCount must be >= 64");
}

nlohmann::json MarkerCloudFuseCPUParams::toJson() const {
    nlohmann::json j;
    j["voxelSize"] = voxelSize;
    j["saturationThreshold"] = saturationThreshold;
    j["reserveVoxelCount"] = reserveVoxelCount;
    j["collectStatistics"] = collectStatistics;
    return j;
}

MarkerCloudFuseCPUParams MarkerCloudFuseCPUParams::fromJson(const nlohmann::json& j) {
    MarkerCloudFuseCPUParams p;
    if (j.contains("voxelSize")) p.voxelSize = j.at("voxelSize").get<float>();
    if (j.contains("saturationThreshold")) p.saturationThreshold = j.at("saturationThreshold").get<int>();
    if (j.contains("reserveVoxelCount")) p.reserveVoxelCount = j.at("reserveVoxelCount").get<size_t>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    return p;
}

// ============================================================
// Impl — SoA flat hash table（与 laser_cloud_fuse_cpu 相同结构）
// ============================================================

struct MarkerCloudFuseCPU::Impl {
    MarkerCloudFuseCPUParams params_;
    MarkerCloudFuseCPUStats stats_;

    // SoA 哈希表：4 个并行数组，tag 预过滤减少 cache miss
    std::vector<uint8_t>  tags_;
    std::vector<uint64_t> keys_;
    std::vector<uint32_t> fusedIdx_;
    std::vector<uint32_t> counts_;

    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;

    std::vector<MarkerCloudPoint> fusedPoints_;

    explicit Impl(const MarkerCloudFuseCPUParams& p) : params_(p) {
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

    MarkerCloudFuseCPUResult ExecuteImpl(const MarkerFuseInput* pts, size_t n,
                  const cv::Matx33d& R, const cv::Vec3d& T) {
        MarkerCloudFuseCPUResult result;
        MarkerCloudFuseCPUStats stats;

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

        size_t survivingCount = 0, deletedCount = 0, newVoxelCount = 0;

        auto hashStart = std::chrono::high_resolution_clock::now();

        std::vector<MarkerCloudPoint>& surviving = result.survivingPoints;
        surviving.reserve(n);

        uint8_t*  tagsData  = tags_.data();
        uint64_t* keysData  = keys_.data();
        uint32_t* fidxData  = fusedIdx_.data();
        uint32_t* cntData   = counts_.data();

        for (size_t i = 0; i < n; ++i) {
            const MarkerFuseInput& raw = pts[i];

            // R/T 变换位置
            float gx = static_cast<float>(R(0,0)*raw.x + R(0,1)*raw.y + R(0,2)*raw.z + T(0));
            float gy = static_cast<float>(R(1,0)*raw.x + R(1,1)*raw.y + R(1,2)*raw.z + T(1));
            float gz = static_cast<float>(R(2,0)*raw.x + R(2,1)*raw.y + R(2,2)*raw.z + T(2));

            // R 变换法线（无平移）
            float gnx = static_cast<float>(R(0,0)*raw.nx + R(0,1)*raw.ny + R(0,2)*raw.nz);
            float gny = static_cast<float>(R(1,0)*raw.nx + R(1,1)*raw.ny + R(1,2)*raw.nz);
            float gnz = static_cast<float>(R(2,0)*raw.nx + R(2,1)*raw.ny + R(2,2)*raw.nz);

            int ix = static_cast<int>(std::floor(gx * inv));
            int iy = static_cast<int>(std::floor(gy * inv));
            int iz = static_cast<int>(std::floor(gz * inv));
            uint64_t key = packVoxelKey(ix, iy, iz);

            uint64_t h = hash64(key);
            uint8_t tag = tagFromHash(h);

            size_t idx = h & mask_;
            while (tagsData[idx] != 0) {
                if (tagsData[idx] == tag && keysData[idx] == key) break;
                idx = (idx + 1) & mask_;
            }

            if (tagsData[idx] != 0) {
                // 已有体素
                if (cntData[idx] >= threshold) {
                    ++deletedCount;
                } else {
                    ++cntData[idx];
                    surviving.push_back(MarkerCloudPoint{
                        gx, gy, gz, gnx, gny, gnz, raw.whiteRadius});
                    ++survivingCount;
                }
            } else {
                // 新体素
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
                fusedPoints_.push_back(MarkerCloudPoint{
                    gx, gy, gz,
                    gnx, gny, gnz,
                    raw.whiteRadius
                });
                tagsData[idx] = tag;
                keysData[idx] = key;
                fidxData[idx] = fi;
                cntData[idx]  = 1;
                ++size_;
                ++newVoxelCount;
                surviving.push_back(MarkerCloudPoint{
                    gx, gy, gz, gnx, gny, gnz, raw.whiteRadius});
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
        result.message = "Marker frame fusion completed";
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

    // seed：与 ExecuteImpl 共用同一插入路径（首个落入者为代表）——seed 点
    // 只写入空体素作为代表；体素已占（含 seed 内部冲突）则丢弃后到者，
    // 既不移动代表也不改计数。后续 fuse 命中已占体素仅递增 counts_，
    // 不触碰 fusedPoints_，故 seed 代表点位置永不改变。
    ResultStatus SeedImpl(const std::vector<MarkerCloudPoint>& pts) {
        const float inv = 1.0f / params_.voxelSize;
        try {
            for (const auto& pt : pts) {
                int ix = static_cast<int>(std::floor(pt.x * inv));
                int iy = static_cast<int>(std::floor(pt.y * inv));
                int iz = static_cast<int>(std::floor(pt.z * inv));
                uint64_t key = packVoxelKey(ix, iy, iz);

                uint64_t h = hash64(key);
                uint8_t tag = tagFromHash(h);

                size_t idx = h & mask_;
                while (tags_[idx] != 0) {
                    if (tags_[idx] == tag && keys_[idx] == key) break;
                    idx = (idx + 1) & mask_;
                }
                if (tags_[idx] != 0) continue;   // 体素已占：首个为代表，后到 seed 点丢弃

                ensureCapacityForOneMore();      // 可能 rehash → 重新探测
                idx = h & mask_;
                while (tags_[idx] != 0) idx = (idx + 1) & mask_;

                uint32_t fi = static_cast<uint32_t>(fusedPoints_.size());
                fusedPoints_.push_back(pt);
                tags_[idx]     = tag;
                keys_[idx]     = key;
                fusedIdx_[idx] = fi;
                counts_[idx]   = 1;              // 与 fuse 首落语义一致
                ++size_;
            }
        } catch (const std::exception& e) {
            return ResultStatus::fail(std::string("seed: ") + e.what());
        }
        return ResultStatus::ok();
    }
};

// ============================================================
// Constructor / Destructor
// ============================================================

MarkerCloudFuseCPU::MarkerCloudFuseCPU(const MarkerCloudFuseCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("MarkerCloudFuseCPU initialized (voxelSize={}, threshold={}, reserve={})",
                   params.voxelSize, params.saturationThreshold, params.reserveVoxelCount);
}

MarkerCloudFuseCPU::~MarkerCloudFuseCPU() = default;

void MarkerCloudFuseCPU::Destroy() { }

// ============================================================
// fuse
// ============================================================

MarkerCloudFuseCPUResult MarkerCloudFuseCPU::Execute(const MarkerFuseInput* pts, size_t count,
                               const cv::Matx33d& R, const cv::Vec3d& T) {
    return pImpl_->ExecuteImpl(pts, count, R, T);
}

MarkerCloudFuseCPUResult MarkerCloudFuseCPU::Execute(const std::vector<MarkerFuseInput>& frame,
                               const cv::Matx33d& R, const cv::Vec3d& T) {
    return pImpl_->ExecuteImpl(frame.data(), frame.size(), R, T);
}

MarkerCloudFuseCPUResult MarkerCloudFuseCPU::Execute(const std::vector<MarkerFuseInput>& frame) {
    return pImpl_->ExecuteImpl(frame.data(), frame.size(),
                     cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
}

// ============================================================
// seed（existingMarkers 预填）
// ============================================================

ResultStatus MarkerCloudFuseCPU::seed(const std::vector<MarkerCloudPoint>& pts) {
    return pImpl_->SeedImpl(pts);
}

// ============================================================
// capacity / clear
// ============================================================

void MarkerCloudFuseCPU::Reserve(size_t voxelCount) {
    size_t want = nextPow2(std::max<size_t>(voxelCount, 64));
    if (want > pImpl_->capacity_) pImpl_->rehashTo(want);
}

void MarkerCloudFuseCPU::Clear() noexcept {
    std::fill(pImpl_->tags_.begin(), pImpl_->tags_.end(), static_cast<uint8_t>(0));
    pImpl_->size_ = 0;
    pImpl_->fusedPoints_.clear();
    pImpl_->stats_ = MarkerCloudFuseCPUStats();
}

// ============================================================
// accessors
// ============================================================

size_t MarkerCloudFuseCPU::GetVoxelCount() const noexcept {
    return pImpl_->size_;
}

size_t MarkerCloudFuseCPU::GetFusedPointCount() const noexcept {
    return pImpl_->fusedPoints_.size();
}

const std::vector<MarkerCloudPoint>& MarkerCloudFuseCPU::GetFusedPoints() const {
    return pImpl_->fusedPoints_;
}

void MarkerCloudFuseCPU::SetParams(const MarkerCloudFuseCPUParams& params) {
    params.validate();
    pImpl_->params_ = params;
}

const MarkerCloudFuseCPUParams& MarkerCloudFuseCPU::GetParams() const {
    return pImpl_->params_;
}

const MarkerCloudFuseCPUStats& MarkerCloudFuseCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}
