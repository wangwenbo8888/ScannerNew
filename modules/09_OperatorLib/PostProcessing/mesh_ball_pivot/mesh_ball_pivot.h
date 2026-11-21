#pragma once
// ============================================================================
// mesh_ball_pivot.h — 滚球网格封装（后处理阶段 1）
//
// 来源：PCMP 沙盒 HybridBallPivotV2 移植（GPU/CPU 分块滚球 + halo 拼接 +
//       三角形去重 + 折缝剔除器 seam_remove + mesh_validator 哨兵）。
// 职责：消费 IndexedMesh 的定向点云（xyz+normals），就地产出 io.triangles
//       （顶点集不变，R1 统一索引网格）。
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
struct MeshBallPivotParams {
    // —— 重建半径（radii = spacing × base × 2^level）——
    float radiusBase   = 1.5f;
    int   radiusLevels = 2;

    // —— GPU/CPU 分块（CPU-only M1：恒 K=1 语义＝A/B 两块＋halo；
    //    gpuBlocks>1 暂按 K=1 处理，GPU 引擎落地后恢复多块路径）——
    float gpuFraction  = 0.6f;    // 点数比例：A 块（原 GPU owner）占比
    int   gpuBlocks    = 1;
    int   cpuBlocks    = 12;      // 链 K=1 路径实际生效值（沙盒源码硬编码 12）
    int   cpuThreads   = 0;       // CPU 块线程（0 = 算子自定）
    float halo         = 2.0f;    // 分块重叠带宽（× maxRadius）

    // —— 折缝剔除（朝向翻转边，路线图快赢 2 成果）——
    bool  seamRemove     = true;
    bool  seamReportOnly = false;
    int   seamMaxRemove  = 0;     // 剔除上限（0 = max(16, tris/400)）
    float seamMinDot     = 0.10f; // 参考法线点积下限

    // —— 点距 ——
    float manualSpacingMm = 0.0f; // 0 = 自动估（nn∈(0.2,1.0)×vol 双口径切换须日志消歧）

    void validate() const;
    nlohmann::json toJson() const;
    static MeshBallPivotParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct MeshBallPivotStats {
    double totalTimeMs   = 0.0;
    float  spacingMm     = 0.0f;   // 实采点距（含口径标记入 message）
    size_t trisGpu       = 0;
    size_t trisCpu       = 0;
    size_t trisStitched  = 0;      // 去重后总三角形
    size_t dedupRemoved  = 0;
    size_t seamCandidates= 0;      // 折缝检出数
    size_t seamRemoved   = 0;      // 折缝剔除数
    size_t validatorRejects = 0;   // mesh_validator 哨兵拒绝数
};

// ============================================================
// 结果（move-only）
// ============================================================
struct MeshBallPivotResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    MeshBallPivotStats statistics;

    MeshBallPivotResult() = default;
    ~MeshBallPivotResult() = default;
    MeshBallPivotResult(MeshBallPivotResult&&) = default;
    MeshBallPivotResult& operator=(MeshBallPivotResult&&) = default;
    MeshBallPivotResult(const MeshBallPivotResult&) = delete;
    MeshBallPivotResult& operator=(const MeshBallPivotResult&) = delete;
};

// ============================================================
// 算子：滚球网格封装（GPU+CPU 混合）
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 实例不保留跨调用数据（统计遥测除外）；GPU 工作区随调用创建释放。
// 并发策略: 每实例非线程安全（§1.4）；无 static 可变状态。
// ==============================
class SCANNER_API MeshBallPivot {
public:
    static constexpr const char* kLogTag = "PP-MeshBallPivot";

    explicit MeshBallPivot(const MeshBallPivotParams& params = {});
    ~MeshBallPivot();

    MeshBallPivot(const MeshBallPivot&) = delete;
    MeshBallPivot& operator=(const MeshBallPivot&) = delete;

    /// 消费定向点云（io.xyz + io.normals），就地填 io.triangles（顶点集不变）
    MeshBallPivotResult Execute(IndexedMesh& io);

    void Destroy();                       // idempotent
    void Warmup(int, int) { }             // 批算子空实现（§2.3）
    void Warmup(const WarmupConfig&) { }

    void SetParams(const MeshBallPivotParams& params);
    const MeshBallPivotParams& GetParams() const;
    const MeshBallPivotStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

    void SetCancelProbe(const std::function<bool()>& probe);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMeshBallPivotInfo();

} // namespace calib
