#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "laser_cloud_fuse_cpu.h"
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ============================================================
// 参数
// ============================================================
struct LaserCloudNormalCPUParams {
    int   kernelRadius  = 1;       // 邻域半径（1=3x3x3, 2=5x5x5）
    int   minNeighbors  = 3;       // PCA 最小邻居数（含中心体素）
    float fallbackNx    = 0.0f;    // 退化法向量X
    float fallbackNy    = 0.0f;
    float fallbackNz    = 1.0f;

    void validate() const;
    nlohmann::json toJson() const;
    static LaserCloudNormalCPUParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct LaserCloudNormalCPUStats {
    double totalTimeMs    = 0.0;
    size_t processedCount = 0;  // 成功计算法线的体素数
    size_t expandedCount  = 0;  // 自适应扩大 kernelRadius 的体素数
    size_t fallbackCount  = 0;
};

// ============================================================
// 结果（仅移动�?// ============================================================
struct LaserCloudNormalCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    LaserCloudNormalCPUStats statistics;

    LaserCloudNormalCPUResult() = default;
    ~LaserCloudNormalCPUResult() = default;
    LaserCloudNormalCPUResult(LaserCloudNormalCPUResult&&) = default;
    LaserCloudNormalCPUResult& operator=(LaserCloudNormalCPUResult&&) = default;
    LaserCloudNormalCPUResult(const LaserCloudNormalCPUResult&) = delete;
    LaserCloudNormalCPUResult& operator=(const LaserCloudNormalCPUResult&) = delete;
};

// ============================================================
// 激光点云法线估�?CPU 算子（单线程�?// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 跨调用的体素状态由调用方持有的 LaserCloudFuseCPU 实例承载并按调用传入，本算子仅读写其法线，实例本身不持有累积状态；另有统计遥测但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserCloudNormalCPU {
public:
    static constexpr const char* kLogTag = "03-LaserCloudNormalCPU";

    explicit LaserCloudNormalCPU(const LaserCloudNormalCPUParams& params = {});
    ~LaserCloudNormalCPU();

    LaserCloudNormalCPU(const LaserCloudNormalCPU&) = delete;
    LaserCloudNormalCPU& operator=(const LaserCloudNormalCPU&) = delete;

    /// �?[beginIdx, endIdx) 范围内的体素计算法线，写�?fuse.FusedPointPtr(i)->{nx,ny,nz}
    LaserCloudNormalCPUResult Execute(LaserCloudFuseCPU& fuse,
                 size_t beginIdx, size_t endIdx);

    /// 便捷重载：从融合结果推导新建体素范围
    LaserCloudNormalCPUResult Execute(LaserCloudFuseCPU& fuse,
                 const LaserCloudFuseCPUResult& fuseResult);

    void Destroy();

    void Warmup(int, int) { }
    void Warmup(const WarmupConfig&) { }

    void SetParams(const LaserCloudNormalCPUParams& params);
    const LaserCloudNormalCPUParams& GetParams() const;
    const LaserCloudNormalCPUStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserCloudNormalCPUInfo();

} // namespace calib