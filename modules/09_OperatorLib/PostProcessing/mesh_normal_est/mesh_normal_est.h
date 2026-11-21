#pragma once
// ============================================================================
// mesh_normal_est.h — 网格法线估计（定向＋清理，后处理阶段 0）
//
// 来源：PCMP 沙盒 NormalEstimation（GPU PCA + BFS 全局定向 + 赶野点 + 拔钉贴回
//       + 法线质量门禁）移植；参数默认值＝沙盒 chain_config 实测口径。
// 职责：对 IndexedMesh.xyz 重算法线并全局定向（inward==0），产出唯一法线
//       权威源（R2）；赶野/拔钉就地收缩 xyz（点数变化入 Stats）。
// 状态：M0 骨架（Execute 恒 degraded「operator pending」，M1 落地实现）。
// ============================================================================
#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <memory>
#include <cstdint>
#include <cstddef>

#include "common/calib_types.h"
#include "common/calib_warmup_config.h"
#include "common/scanner_api.h"
#include "common/version.h"
#include "mesh_stage_types.h"

namespace calib {

// ============================================================
// 参数（默认值＝PCMP 沙盒 50W 实测口径）
// ============================================================
struct MeshNormalEstParams {
    // —— 法线估计 ——
    float radiusMultiplier = 3.0f;   // 邻域半径 = spacing × 此值
    int   minNeighbors    = 10;      // PCA 最小邻居数
    float knownSpacingMm  = 0.0f;    // 0 = 自动估点距

    // —— 赶野点（离群浮点，kNN 平均距离判决）——
    bool  removeOutliers  = true;
    int   outlierK        = 20;
    float outlierFactor   = 2.0f;    // 距离 > factor × 均值 → 删

    // —— 拔钉（贴表面凸起，邻域平面判据 + 投影贴回）——
    bool  removeSpikes        = true;
    bool  spikeReportOnly     = false;
    bool  spikeIncludeOutliers= true;
    float spikeRatioThreshold = 4.0f;
    float spikeMinAbsMm       = 0.10f;
    float spikeNoiseFloorMm   = 0.01f;
    int   spikeMaxRounds      = 5;

    // —— 法线质量门禁（坏法线占比）——
    float gateWarnPct = 5.0f;        // >warn% 降级
    float gateFailPct = 15.0f;       // >fail% 拒绝

    // —— 并行（D2 口径：批算子 OpenMP 运行时开关，0 = 算子自定）——
    int   numThreads = 0;

    void validate() const;
    nlohmann::json toJson() const;
    static MeshNormalEstParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct MeshNormalEstStats {
    double totalTimeMs     = 0.0;
    float  spacingMm       = 0.0f;  // 实采点距（自动/人工）
    size_t pointsIn        = 0;   // 入口点数
    size_t pointsOut       = 0;   // 出口点数（与入口一致：赶野/拔钉不改点数）
    size_t outliersRemoved = 0;   // 赶野命中（法线清零，点保留）
    size_t spikesRemoved   = 0;   // 拔钉命中（坐标贴回，点保留）
    int    spikeRounds     = 0;
    size_t orientFlips     = 0;   // BFS 全局定向翻转数
    size_t fillMissing     = 0;   // 邻域不足补缺失数
    double badNormalPct    = 0.0; // 门禁实测值
};

// ============================================================
// 结果（move-only）
// ============================================================
struct MeshNormalEstResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    MeshNormalEstStats statistics;

    MeshNormalEstResult() = default;
    ~MeshNormalEstResult() = default;
    MeshNormalEstResult(MeshNormalEstResult&&) = default;
    MeshNormalEstResult& operator=(MeshNormalEstResult&&) = default;
    MeshNormalEstResult(const MeshNormalEstResult&) = delete;
    MeshNormalEstResult& operator=(const MeshNormalEstResult&) = delete;
};

// ============================================================
// 算子：网格法线估计（GPU+CPU 混合）
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 实例不保留跨调用数据（统计遥测除外，不影响计算结果）；GPU 工作区
//       随调用创建释放（批算子，Warmup 空实现口径）。
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占；无 static 可变状态。
// ==============================
class SCANNER_API MeshNormalEst {
public:
    static constexpr const char* kLogTag = "PP-MeshNormalEst";

    explicit MeshNormalEst(const MeshNormalEstParams& params = {});
    ~MeshNormalEst();

    MeshNormalEst(const MeshNormalEst&) = delete;
    MeshNormalEst& operator=(const MeshNormalEst&) = delete;

    /// 就地重算并定向 io.normals；赶野/拔钉就地收缩 io.xyz（点数变化入 Stats）
    MeshNormalEstResult Execute(IndexedMesh& io);

    void Destroy();                       // idempotent
    void Warmup(int, int) { }             // 批算子空实现（§2.3）
    void Warmup(const WarmupConfig&) { }

    void SetParams(const MeshNormalEstParams& params);
    const MeshNormalEstParams& GetParams() const;
    const MeshNormalEstStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

    /// 取消探针（返回 true = 已取消；适配层注入，算子主循环粗粒度检查）
    void SetCancelProbe(const std::function<bool()>& probe);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMeshNormalEstInfo();

} // namespace calib
