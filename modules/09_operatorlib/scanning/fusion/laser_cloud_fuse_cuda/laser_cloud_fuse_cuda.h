#pragma once


#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
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
// 参数
// ============================================================
struct LaserCloudFuseCUDAParams {
    float  voxelSize          = 0.5f;
    int    saturationThreshold = 5;
    size_t reserveVoxelCount  = static_cast<size_t>(1) << 21; // 2M slots
    bool   collectStatistics  = true;

    void validate() const;
    nlohmann::json toJson() const;
    static LaserCloudFuseCUDAParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 结果（仅移动�?// ============================================================
struct LaserCloudFuseCudaResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    int inputCount      = 0;
    int survivingCount  = 0;
    int deletedCount    = 0;
    int newVoxelCount   = 0;
    int totalVoxelCount = 0;
    double totalTimeMs  = 0.0;

    LaserCloudFuseCudaResult() = default;
    ~LaserCloudFuseCudaResult() = default;
    LaserCloudFuseCudaResult(LaserCloudFuseCudaResult&&) = default;
    LaserCloudFuseCudaResult& operator=(LaserCloudFuseCudaResult&&) = default;
    LaserCloudFuseCudaResult(const LaserCloudFuseCudaResult&) = delete;
    LaserCloudFuseCudaResult& operator=(const LaserCloudFuseCudaResult&) = delete;
};

// ============================================================
// 设备上下�?�?供法线算子访问哈希表和点缓冲
// ============================================================
struct LaserCloudFuseDeviceContext {
    const unsigned long long* d_keys        = nullptr;
    const unsigned int*       d_fusedIdx    = nullptr;
    const float*              d_fusedXyz    = nullptr;
    float*                    d_fusedNormal = nullptr;
    unsigned long long        mask          = 0;
    float                     voxelSize     = 0.5f;
    float                     invVoxelSize  = 2.0f;
    size_t                    fusedPointCount = 0;
};

// ============================================================
// 激光点云体素哈希融�?CUDA 算子
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 自持久化
// 说明: 实例内部维护体素哈希累积器，跨 Execute 调用累积激光点并影响结果，通过 Clear() 重置。
// 重置接口: Clear()
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserCloudFuseCuda {
public:
    static constexpr const char* kLogTag = "02C-LaserCloudFuseCuda";

    explicit LaserCloudFuseCuda(const LaserCloudFuseCUDAParams& params = {});
    ~LaserCloudFuseCuda();

    LaserCloudFuseCuda(const LaserCloudFuseCuda&) = delete;
    LaserCloudFuseCuda& operator=(const LaserCloudFuseCuda&) = delete;

    LaserCloudFuseCudaResult Execute(const cv::cuda::GpuMat& d_points3d,
                 cv::cuda::Stream& stream);

    LaserCloudFuseCudaResult Execute(const cv::cuda::GpuMat& d_points3d);

    LaserCloudFuseCudaResult Execute(const cv::cuda::GpuMat& d_points3d,
                 const cv::Matx33d& R, const cv::Vec3d& T,
                 cv::cuda::Stream& stream);

    LaserCloudFuseCudaResult Execute(const cv::cuda::GpuMat& d_points3d,
                 const cv::Matx33d& R, const cv::Vec3d& T);

    void Destroy();

    void Clear() noexcept;
    void Reserve(size_t voxelCount);
    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);

    size_t GetVoxelCount() const noexcept;
    size_t GetFusedPointCount() const noexcept;
    LaserCloudFuseDeviceContext GetDeviceContext() const;

    void SetParams(const LaserCloudFuseCUDAParams& params);
    const LaserCloudFuseCUDAParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserCloudFuseCudaInfo();

} // namespace calib