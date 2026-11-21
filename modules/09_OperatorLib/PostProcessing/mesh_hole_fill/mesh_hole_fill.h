#pragma once
// ============================================================================
// mesh_hole_fill.h — 网格补洞（后处理阶段 2）
//
// 来源：PCMP 沙盒 MeshHoleFillerPro 移植（进门清扫判据族 + MWT/耳切路由 +
//       密度重网格 + 三调和光顺 + 水密门禁 + R1 重试 + 标志点切割回填）。
//       Open3D 四步清理内化为自研（entry_cleaner 已内化一半，剥离清单 §5）。
// 职责：消费封装后网格，补洞并维持法线权威源（补片顶点借法线，索引对齐）。
// 标志点：markers 由调用方从 marker_cloud_fuse_cpu 输出直取（文件路径参数废除）。
// 状态：M0 骨架（Execute 恒 degraded「operator pending」，M2 落地实现）。
// ============================================================================
#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstddef>

#include "common/calib_types.h"
#include "common/calib_warmup_config.h"
#include "common/scanner_api.h"
#include "common/version.h"
#include "mesh_stage_types.h"

namespace calib {

// ============================================================
// 参数（默认值＝PCMP 沙盒 50W 实测口径；单位 mm）
// ============================================================
struct MeshHoleFillParams {
    // —— 进门清扫（entry clean）——
    bool  entryClean            = true;
    int   entrySmallCcMinFaces  = 50;     // 孤立碎片：面数下限
    float entrySmallCcAreaRatio = 0.01f;  // 面积比下限
    int   entryBowtieMaxIters   = 3;      // 蝴蝶形修复轮数
    bool  entryFlapPeel         = true;   // 翘皮撕除
    bool  entrySpikeRemove      = false;  // 网格毛刺（默认关）
    float entrySpikeRatio       = 10.0f;
    int   entrySpikeRing        = 2;
    bool  entryVeilRemove       = true;   // 面纱薄片
    float entryVeilEdgeMult     = 8.0f;   // 边长 > mult × 均值 → 删

    // —— 补洞主体 ——
    int   neighborhoodRings   = 2;
    int   densityRemeshPasses = 4;
    float splitFactor         = 2.0f;
    float collapseFactor      = 0.1f;
    int   fairingOrder        = 3;        // 三调和 C²
    int   maxHoleEdges        = 300;      // 边界点数门
    float maxHolePerimeterMm  = 0.0f;     // 0 = 不限周长
    int   trimRings           = 0;        // 铲边环数（默认关，路线图裁定）

    // —— 出口审计 ——
    bool exitNmAudit          = true;
    bool exitNmPrune          = true;
    bool componentExemptClosed= true;

    // —— 缝合与焊接 ——
    bool arcStitch            = true;
    bool s1Hash               = true;     // 重合顶点（内化的四步清理 S1a）
    bool nrmHash              = true;     // 法线哈希去重
    bool edgeUnify            = true;     // 边统一

    // —— 表面引导填充（大洞）——
    bool  fillSurfaceGuided    = true;
    float fillGuidePerimeterMinMm = 25.0f;
    float fillGuideLambdaAlpha = 50.0f;
    float fillGuideBandWidthMm = 3.0f;

    // —— 前沿填充 ——
    bool  fillFront           = true;
    bool  fillFrontFlip       = true;
    float fillFrontPerimeterMinMm = 20.0f;

    // —— 重网格与洞缘 ——
    bool  remeshFlip          = true;
    bool  rimSoftAnchor       = true;
    bool  rimGradient         = true;
    float rimGradientRings    = 4.0f;
    float rimGradientRatio    = 1.0f;
    bool  remeshCollapse      = true;
    bool  rimLift             = true;

    // —— 标志点切割回填 ——
    bool  markerCut        = true;
    float markerDiameterMm= 10.0f;
    float markerRadiusScale = 1.2f;     // 切割半径 = radius × scale

    // —— 并行（D2 口径）——
    int   numThreads = 0;

    void validate() const;
    nlohmann::json toJson() const;
    static MeshHoleFillParams fromJson(const nlohmann::json& j);
};

// ============================================================
// 统计
// ============================================================
struct MeshHoleFillStats {
    double totalTimeMs   = 0.0;
    int    loopsDetected = 0;    // 边界环检出数（watertight pre 口径）
    int    holesFilled   = 0;
    int    holesSkipped  = 0;
    int    openArcs      = 0;    // 开放弧（策略保留）
    int    rejected      = 0;    // 拒绝（超门）
    int    watertightPre = 0;    // 进口水密计数（门禁差分）
    int    watertightPost= 0;    // 出口残留
    int    markersCut    = 0;    // 标志点切割回填数
    int    retryR1       = 0;    // R1 重试次数
    double cleanMs       = 0.0;  // 进门清扫耗时
    double fillMs        = 0.0;  // 补洞主体耗时
};

// ============================================================
// 结果（move-only）
// ============================================================
struct MeshHoleFillResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    MeshHoleFillStats statistics;

    MeshHoleFillResult() = default;
    ~MeshHoleFillResult() = default;
    MeshHoleFillResult(MeshHoleFillResult&&) = default;
    MeshHoleFillResult& operator=(MeshHoleFillResult&&) = default;
    MeshHoleFillResult(const MeshHoleFillResult&) = delete;
    MeshHoleFillResult& operator=(const MeshHoleFillResult&) = delete;
};

// ============================================================
// 算子：网格补洞（CPU，OpenMP 运行时开关）
// ============================================================
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 实例不保留跨调用数据（统计遥测除外）；无跨洞缓存（沙盒的
//       mean_edge_length/v2t 全局预计算随调用生命周期）。
// 并发策略: 每实例非线程安全（§1.4）；无 static 可变状态。
// ==============================
class SCANNER_API MeshHoleFill {
public:
    static constexpr const char* kLogTag = "PP-MeshHoleFill";

    explicit MeshHoleFill(const MeshHoleFillParams& params = {});
    ~MeshHoleFill();

    MeshHoleFill(const MeshHoleFill&) = delete;
    MeshHoleFill& operator=(const MeshHoleFill&) = delete;

    /// 就地补洞；markers 为空且 markerCut=true 时跳过标志点段（message 注记）
    MeshHoleFillResult Execute(IndexedMesh& io,
                               const std::vector<MarkerHoleRef>& markers);

    void Destroy();                       // idempotent
    void Warmup(int, int) { }             // 批算子空实现（§2.3）
    void Warmup(const WarmupConfig&) { }

    void SetParams(const MeshHoleFillParams& params);
    const MeshHoleFillParams& GetParams() const;
    const MeshHoleFillStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

    void SetCancelProbe(const std::function<bool()>& probe);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMeshHoleFillInfo();

} // namespace calib
