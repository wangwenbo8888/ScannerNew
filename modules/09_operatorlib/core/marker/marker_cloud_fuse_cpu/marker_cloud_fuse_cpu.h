#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "common/calib_types.h"
#include "common/calib_warmup_config.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ============================================================
// 输入：单帧标记点（相机坐标系，含法线+半径�?// ============================================================
struct MarkerFuseInput {
    float x  = 0.0f;
    float y  = 0.0f;
    float z  = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 1.0f;
    float whiteRadius = 0.0f;
};

// ============================================================
// 体素代表点（全局坐标系，首个落入该体素的点）
// ============================================================
struct MarkerCloudPoint {
    float x  = 0.0f;
    float y  = 0.0f;
    float z  = 0.0f;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 1.0f;
    float whiteRadius = 0.0f;
};

// ============================================================
// 参数
// ============================================================
struct MarkerCloudFuseCPUParams {
    float  voxelSize          = 0.5f;
    int    saturationThreshold = 99;
    size_t reserveVoxelCount  = 1024;
    bool   collectStatistics  = true;

    void validate() const;
    nlohmann::json toJson() const;
    static MarkerCloudFuseCPUParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct MarkerCloudFuseCPUStats {
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
struct MarkerCloudFuseCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    MarkerCloudFuseCPUStats statistics;
    std::vector<MarkerCloudPoint> survivingPoints;

    MarkerCloudFuseCPUResult() = default;
    ~MarkerCloudFuseCPUResult() = default;

    MarkerCloudFuseCPUResult(MarkerCloudFuseCPUResult&&) = default;
    MarkerCloudFuseCPUResult& operator=(MarkerCloudFuseCPUResult&&) = default;

    MarkerCloudFuseCPUResult(const MarkerCloudFuseCPUResult&) = delete;
    MarkerCloudFuseCPUResult& operator=(const MarkerCloudFuseCPUResult&) = delete;
};

// ============================================================
// 标记点点云体素哈希融�?CPU 算子（单线程�?//
// 算法�?laser_cloud_fuse_cpu 相同（tag + open-addressing），
// 点结构增�?whiteRadius，法线由输入直接携带（无需后续估计）�?// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 自持久化
// 说明: 实例内部维护体素哈希累积器，跨 Execute 调用累积标记点并影响结果，通过 clear() 重置。
// 重置接口: clear()
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API MarkerCloudFuseCPU {
public:
    static constexpr const char* kLogTag = "02M-MarkerCloudFuseCPU";

    explicit MarkerCloudFuseCPU(const MarkerCloudFuseCPUParams& params = {});
    ~MarkerCloudFuseCPU();

    MarkerCloudFuseCPU(const MarkerCloudFuseCPU&) = delete;
    MarkerCloudFuseCPU& operator=(const MarkerCloudFuseCPU&) = delete;

    /// 融合一帧标记点（指针 + R/T 变换）
    MarkerCloudFuseCPUResult Execute(const MarkerFuseInput* pts, size_t count,
              const cv::Matx33d& R, const cv::Vec3d& T);

    /// 融合一帧标记点（vector + R/T 变换）
    MarkerCloudFuseCPUResult Execute(const std::vector<MarkerFuseInput>& frame,
              const cv::Matx33d& R, const cv::Vec3d& T);

    /// 融合一帧标记点（无变换，已在全球坐标系）
    MarkerCloudFuseCPUResult Execute(const std::vector<MarkerFuseInput>& frame);

    void Clear() noexcept;
    void Reserve(size_t voxelCount);

    size_t GetVoxelCount() const noexcept;
    size_t GetFusedPointCount() const noexcept;
    const std::vector<MarkerCloudPoint>& GetFusedPoints() const;

    void Warmup(int maxPointCount) { }
    void Warmup(const WarmupConfig& config) { (void)config; }

    void SetParams(const MarkerCloudFuseCPUParams& params);
    const MarkerCloudFuseCPUParams& GetParams() const;
    const MarkerCloudFuseCPUStats& GetStatistics() const noexcept;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMarkerCloudFuseCPUInfo();

} // namespace calib