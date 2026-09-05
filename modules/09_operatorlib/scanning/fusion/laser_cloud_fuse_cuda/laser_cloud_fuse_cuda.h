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
#include "common/result.h"
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

    /// 编辑账本移除（05 D4 双账本·激光云侧，实施计划 P2）：按下标移除融合点
    /// ——设备端压缩（点/法线同映射保全关联、幸存体素饱和计数保留、体素随
    /// 代表点整体摘除）。indices 为融合点下标（GetDeviceContext()
    /// .fusedPointCount 界内；编辑会话期下标稳定）。空=幂等 ok；越界=fail
    /// 整批不动；重复去重；全删=等价 Clear。移除后幸存点按原序前移（下标
    /// 重排，调用方重新取 DeviceContext/快照）。
    ResultStatus removePoints(const std::vector<uint32_t>& indices,
                              cv::cuda::Stream& stream);
    ResultStatus removePoints(const std::vector<uint32_t>& indices);

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