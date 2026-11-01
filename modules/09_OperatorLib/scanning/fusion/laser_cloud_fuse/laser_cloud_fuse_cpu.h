#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// ============================================================
// 体素代表点（每个体素记录第一个落入的点；法线后续计算写入�?// ============================================================
struct CloudPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;
};

// ============================================================
// 体素快照条目（snapshotVoxels 输出用）
// ============================================================
struct VoxelInfo {
    uint64_t key = 0;
    CloudPoint* firstPoint = nullptr;
    uint32_t count = 0;
};

// ============================================================
// 参数
// ============================================================
struct LaserCloudFuseCPUParams {
    float  voxelSize          = 0.5f;
    int    saturationThreshold = 5;
    size_t reserveVoxelCount  = static_cast<size_t>(1) << 16;
    bool   collectStatistics  = true;

    void validate() const;
    nlohmann::json toJson() const;
    static LaserCloudFuseCPUParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct LaserCloudFuseCPUStats {
    double totalTimeMs   = 0.0;
    double hashTimeMs    = 0.0;
    size_t inputCount    = 0;
    size_t survivingCount = 0;
    size_t deletedCount  = 0;
    size_t newVoxelCount = 0;
    size_t totalVoxelCount = 0;
};

// ============================================================
// 结果（仅移动�?// ============================================================
struct LaserCloudFuseCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<cv::Point3f> survivingPoints;
    LaserCloudFuseCPUStats statistics;

    LaserCloudFuseCPUResult() = default;
    ~LaserCloudFuseCPUResult() = default;

    LaserCloudFuseCPUResult(LaserCloudFuseCPUResult&&) = default;
    LaserCloudFuseCPUResult& operator=(LaserCloudFuseCPUResult&&) = default;

    LaserCloudFuseCPUResult(const LaserCloudFuseCPUResult&) = delete;
    LaserCloudFuseCPUResult& operator=(const LaserCloudFuseCPUResult&) = delete;
};

// ============================================================
// 激光点云体素哈希融�?CPU 算子（单线程�?// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 自持久化
// 说明: 实例内部维护体素哈希累积器，跨 Execute 调用累积激光点并影响结果，通过 Clear() 重置。
// 重置接口: Clear()
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserCloudFuseCPU {
public:
    static constexpr const char* kLogTag = "02-LaserCloudFuseCPU";

    explicit LaserCloudFuseCPU(const LaserCloudFuseCPUParams& params = {});
    ~LaserCloudFuseCPU();

    LaserCloudFuseCPU(const LaserCloudFuseCPU&) = delete;
    LaserCloudFuseCPU& operator=(const LaserCloudFuseCPU&) = delete;

    LaserCloudFuseCPUResult Execute(const std::vector<cv::Point3f>& frame);
    LaserCloudFuseCPUResult Execute(const cv::Point3f* pts, size_t count);

    LaserCloudFuseCPUResult Execute(const std::vector<cv::Point3f>& frame,
                 const cv::Matx33d& R, const cv::Vec3d& T);
    LaserCloudFuseCPUResult Execute(const cv::Point3f* pts, size_t count,
                 const cv::Matx33d& R, const cv::Vec3d& T);

    void Destroy();

    void Reserve(size_t voxelCount);
    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);
    void Clear() noexcept;

    size_t GetVoxelCount() const noexcept;
    size_t GetFusedPointCount() const noexcept;
    CloudPoint* FusedPointPtr(size_t index);
    const std::vector<CloudPoint>& GetFusedPoints() const;
    void SnapshotVoxels(std::vector<VoxelInfo>& out) const;

    /// Collect representative points from voxels within kernel radius of worldPos (read-only).
    /// kernelRadius=1 -> 3x3x3 (27 voxels), kernelRadius=2 -> 5x5x5 (125 voxels).
    /// Includes the center voxel itself. Empty voxels are skipped.
    /// @return number of neighbors found
    size_t GatherVoxelNeighbors(const cv::Point3f& worldPos, int kernelRadius,
                                std::vector<const CloudPoint*>& outNeighbors) const;

    void SetParams(const LaserCloudFuseCPUParams& params);
    const LaserCloudFuseCPUParams& GetParams() const;
    const LaserCloudFuseCPUStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserCloudFuseCPUInfo();

} // namespace calib