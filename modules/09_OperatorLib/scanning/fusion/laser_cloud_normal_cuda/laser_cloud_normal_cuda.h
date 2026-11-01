#pragma once


#include "laser_cloud_fuse_cuda.h"
#include <string>
#include <memory>
#include <cstdint>
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// ============================================================
// 参数
// ============================================================
struct LaserCloudNormalCUDAParams {
    int   minNeighbors  = 3;
    float fallbackNx    = 0.0f;
    float fallbackNy    = 0.0f;
    float fallbackNz    = 1.0f;

    void validate() const;
    nlohmann::json toJson() const;
    static LaserCloudNormalCUDAParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 结果
// ============================================================
struct LaserCloudNormalCudaResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    size_t processedCount = 0;
    size_t fallbackCount  = 0;
    double totalTimeMs    = 0.0;

    LaserCloudNormalCudaResult() = default;
    LaserCloudNormalCudaResult(LaserCloudNormalCudaResult&&) = default;
    LaserCloudNormalCudaResult& operator=(LaserCloudNormalCudaResult&&) = default;
    LaserCloudNormalCudaResult(const LaserCloudNormalCudaResult&) = delete;
    LaserCloudNormalCudaResult& operator=(const LaserCloudNormalCudaResult&) = delete;
};

// ============================================================
// 激光点云法线估�?CUDA 算子
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 跨调用的体素状态由调用方持有的 LaserCloudFuseCuda/LaserCloudFuseDeviceContext 承载并按调用传入，本算子仅读写其法线，实例本身不持有累积状态。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserCloudNormalCuda {
public:
    static constexpr const char* kLogTag = "03C-LaserCloudNormalCuda";

    explicit LaserCloudNormalCuda(const LaserCloudNormalCUDAParams& params = {});
    ~LaserCloudNormalCuda();

    LaserCloudNormalCuda(const LaserCloudNormalCuda&) = delete;
    LaserCloudNormalCuda& operator=(const LaserCloudNormalCuda&) = delete;

    LaserCloudNormalCudaResult Execute(const LaserCloudFuseDeviceContext& ctx,
                 size_t beginIdx, size_t endIdx,
                 cv::cuda::Stream& stream);

    LaserCloudNormalCudaResult Execute(const LaserCloudFuseDeviceContext& ctx,
                 size_t beginIdx, size_t endIdx);

    LaserCloudNormalCudaResult Execute(LaserCloudFuseCuda& fuse,
                 const LaserCloudFuseCudaResult& fuseResult,
                 cv::cuda::Stream& stream);

    LaserCloudNormalCudaResult Execute(LaserCloudFuseCuda& fuse,
                 const LaserCloudFuseCudaResult& fuseResult);

    void Destroy();

    void Warmup(int, int) { }
    void Warmup(const WarmupConfig&) { }

    void SetParams(const LaserCloudNormalCUDAParams& params);
    const LaserCloudNormalCUDAParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserCloudNormalCudaInfo();

} // namespace calib