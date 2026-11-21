#pragma once
// ============================================================================
// mesh_smooth_sr.h — 网格光顺（后处理阶段 3，SR 三遍投票光顺）
//
// 来源：PCMP 沙盒 smooth_reconstruction 移植（定向 orient + 三遍扫描投票
//       修位置 + 边翻转调线六关卡；零第三方依赖形态保持）。
//       ⚠ 合规前置门：参数标定值出处须经法务书面裁定后方可随产品发布
//       （移植计划 §6-C2；未决前不入发布分支）。
// 状态：M0 骨架（Execute 恒 degraded「operator pending」，M3 落地实现）。
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
// 参数（默认值＝PCMP 沙盒 50W 实测口径；单位 mm/度）
// ============================================================
struct MeshSmoothSrParams {
    // —— 修位置（三遍扫描＋投票取平均）——
    int   ring        = 2;         // 邻域跳数
    float angleDeg    = 12.5f;     // 方向门槛（挡跨棱邻居）
    float angle2Deg   = 4.1667f;   // 第二门槛
    float toleranceMm = 0.02f;     // 位移容差
    int   outer       = 2;         // 外层迭代

    // —— 定向（嵌入法线 + 均值投票 + 锚面传播）——
    bool enableOrient = true;

    // —— 边翻转调线（六道关卡）——
    bool  enableFlip    = true;
    float coplanarDeg   = 1.0f;    // 共面跳过门
    bool  flipNmGuard   = true;    // 非流形防护

    // —— 并行（D2 口径）——
    int   numThreads = 0;

    void validate() const;
    nlohmann::json toJson() const;
    static MeshSmoothSrParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct MeshSmoothSrStats {
    double totalTimeMs        = 0.0;
    size_t trisIn             = 0;
    size_t trisOut            = 0;
    size_t orientEmbeddedFaces= 0;  // 嵌入法线覆盖面数
    size_t orientFlips        = 0;  // 定向翻转修正数
    int    edgeFlipPasses     = 0;
    size_t edgeFlips          = 0;  // 调线翻转总数
    float  maxDeltaVmm        = 0.0f; // max|Δv|（≤10×tol 门）
    double totalMsOrient      = 0.0;
    double totalMsSmooth      = 0.0;
    double totalMsFlip        = 0.0;
};

// ============================================================
// 结果（move-only）
// ============================================================
struct MeshSmoothSrResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    MeshSmoothSrStats statistics;

    MeshSmoothSrResult() = default;
    ~MeshSmoothSrResult() = default;
    MeshSmoothSrResult(MeshSmoothSrResult&&) = default;
    MeshSmoothSrResult& operator=(MeshSmoothSrResult&&) = default;
    MeshSmoothSrResult(const MeshSmoothSrResult&) = delete;
    MeshSmoothSrResult& operator=(const MeshSmoothSrResult&) = delete;
};

// ============================================================
// 算子：网格光顺（CPU，OpenMP 运行时开关）
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 实例不保留跨调用数据（统计遥测除外）。
// 并发策略: 每实例非线程安全（§1.4）；无 static 可变状态。
// ==============================
class SCANNER_API MeshSmoothSr {
public:
    static constexpr const char* kLogTag = "PP-MeshSmoothSr";

    explicit MeshSmoothSr(const MeshSmoothSrParams& params = {});
    ~MeshSmoothSr();

    MeshSmoothSr(const MeshSmoothSr&) = delete;
    MeshSmoothSr& operator=(const MeshSmoothSr&) = delete;

    /// 就地光顺（读 io.normals 为定向参考，修位置 + 调线）
    MeshSmoothSrResult Execute(IndexedMesh& io);

    void Destroy();                       // idempotent
    void Warmup(int, int) { }             // 批算子空实现（§2.3）
    void Warmup(const WarmupConfig&) { }

    void SetParams(const MeshSmoothSrParams& params);
    const MeshSmoothSrParams& GetParams() const;
    const MeshSmoothSrStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

    void SetCancelProbe(const std::function<bool()>& probe);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMeshSmoothSrInfo();

} // namespace calib
