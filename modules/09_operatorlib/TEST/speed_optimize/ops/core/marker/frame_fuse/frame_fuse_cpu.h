/**
 * @file frame_fuse_cpu.h
 * @brief 单帧快速融合算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑫（共用�? * 平台：CPU
 *
 * 功能：给定两帧标记点的三维空间坐标和法线，在未知对应关系下，
 *       通过�?法线信息实现快速粗匹配建立对应关系�? *       再经法线加权SVD精配准，计算刚体变换（旋转R + 平移T），
 *       将集�?坐标系映射到集合2坐标系�? *
 * 输入：来�?point_reconstruct_cpu (11) 的标记点集合（位�?法线�? * 输出：刚体变�?R, T, 4x4 齐次变换矩阵
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <utility>
#include "common/calib_types.h"
#include "common/zitai_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

// ============================================================
// Parameters
// ============================================================

struct FrameFuseCPUParams {
    int knnK = 15;
    int descriptorBins1 = 11;
    int descriptorBins2 = 11;
    int descriptorBins3 = 15;
    double loweRatio = 0.8;
    double normalPreFilterAngleDeg = 15.0;
    double ransacConfidence = 0.999;
    int ransacMaxIterations = 5000;
    int minInlierCount = 3;
    int refineIterations = 3;
    double refineConvergeRatio = 1e-4;
    bool collectStatistics = true;
    size_t maxPointCount = 10000;

    // —— 匹配模式（2026-10-14 新增，tests/frame_fuse_opt 优化专题）——
    // 0=PFH 指纹（默认；适用法线分布有各向性的表面点集）
    // 1=三角形边长匹配（稀疏＋共面＋法线近平行的标记点阵——PFH 在此形态先天
    //   无区分度；三角形穷举 C(n,3)² 组合，边长一致→三点闭式解→全体验证）
    int matchMode = 0;
    double triangleSideTolMm = 1.0;    // 模式1：三边长度差容差（绝对项 mm）
    double triangleSideTolRatio = 0.02; // 模式1：边长容差相对项（容差=max(绝对, 相对×边长)）
    double triangleVerifyGateMm = 1.0; // 模式1：验证门基线（近距口径；实际按 Z 深度自适应
                                       // 放大——见 triangleVerifyGateRatio）。2026-10-15 批1 定标
                                       // 2.0→1.0：金样真值残差 ≤0.17mm（6 倍余量）；跨面风险带
                                       // 缩半。已知边界：孪生布点＋法线翻转时折中混合解几何
                                       // 不可证伪——批2 法线合理性门从上游拦截
    double triangleVerifyGateRatio = 0.002; // 模式1：验证门 Z 深度自适应系数
                                            // 有效门 = max(基线, min(2.0, ratio × Z 中位))
                                            // Z=330→1.0（不变）；Z=1000→2.0（远距噪声放大）
                                            // 2026-10-15 批3 落地（预审批1 F10 定参版）
    double triangleNormalMaxAngleDeg = 60.0; // 模式1：终审法线同向门（有向夹角上限；镜像解杀手）
    int triangleMaxPoints = 16;        // 模式1：单侧点数上限（超限回退 PFH——复杂度守门）
    int triangleMaxCandidatePairs = 20000; // 模式1：SVD+验证工作量封顶（次数制，非墙钟——保确定性）

    void validate() const {
        if (knnK < 3)
            throw std::invalid_argument("knnK must be >= 3");
        if (descriptorBins1 < 3)
            throw std::invalid_argument("descriptorBins1 must be >= 3");
        if (descriptorBins2 < 3)
            throw std::invalid_argument("descriptorBins2 must be >= 3");
        if (descriptorBins3 < 3)
            throw std::invalid_argument("descriptorBins3 must be >= 3");
        if (loweRatio <= 0.0 || loweRatio >= 1.0)
            throw std::invalid_argument("loweRatio must be in (0, 1)");
        if (normalPreFilterAngleDeg <= 0.0 || normalPreFilterAngleDeg >= 90.0)
            throw std::invalid_argument("normalPreFilterAngleDeg must be in (0, 90)");
        if (ransacConfidence <= 0.0 || ransacConfidence > 1.0)
            throw std::invalid_argument("ransacConfidence must be in (0, 1]");
        if (ransacMaxIterations < 1)
            throw std::invalid_argument("ransacMaxIterations must be >= 1");
        if (minInlierCount < 3)
            throw std::invalid_argument("minInlierCount must be >= 3");
        if (refineIterations < 1)
            throw std::invalid_argument("refineIterations must be >= 1");
        if (refineConvergeRatio <= 0.0)
            throw std::invalid_argument("refineConvergeRatio must be > 0");
        if (maxPointCount == 0)
            throw std::invalid_argument("maxPointCount must be > 0");
        if (matchMode != 0 && matchMode != 1)
            throw std::invalid_argument("matchMode must be 0 (PFH) or 1 (triangle)");
        if (triangleSideTolMm <= 0.0)
            throw std::invalid_argument("triangleSideTolMm must be > 0");
        if (triangleSideTolRatio < 0.0 || triangleSideTolRatio >= 0.5)
            throw std::invalid_argument("triangleSideTolRatio must be in [0, 0.5)");
        if (triangleVerifyGateMm <= 0.0)
            throw std::invalid_argument("triangleVerifyGateMm must be > 0");
        if (triangleNormalMaxAngleDeg <= 0.0 || triangleNormalMaxAngleDeg >= 180.0)
            throw std::invalid_argument("triangleNormalMaxAngleDeg must be in (0, 180)");
        if (triangleMaxPoints < 6)
            throw std::invalid_argument("triangleMaxPoints must be >= 6");
        if (triangleMaxCandidatePairs < 1000)
            throw std::invalid_argument("triangleMaxCandidatePairs must be >= 1000");
    }

    nlohmann::json toJson() const {
        nlohmann::json j;
        j["knnK"] = knnK;
        j["descriptorBins1"] = descriptorBins1;
        j["descriptorBins2"] = descriptorBins2;
        j["descriptorBins3"] = descriptorBins3;
        j["loweRatio"] = loweRatio;
        j["normalPreFilterAngleDeg"] = normalPreFilterAngleDeg;
        j["ransacConfidence"] = ransacConfidence;
        j["ransacMaxIterations"] = ransacMaxIterations;
        j["minInlierCount"] = minInlierCount;
        j["refineIterations"] = refineIterations;
        j["refineConvergeRatio"] = refineConvergeRatio;
        j["collectStatistics"] = collectStatistics;
        j["maxPointCount"] = maxPointCount;
        j["matchMode"] = matchMode;
        j["triangleSideTolMm"] = triangleSideTolMm;
        j["triangleSideTolRatio"] = triangleSideTolRatio;
        j["triangleVerifyGateMm"] = triangleVerifyGateMm;
        j["triangleVerifyGateRatio"] = triangleVerifyGateRatio;
        j["triangleNormalMaxAngleDeg"] = triangleNormalMaxAngleDeg;
        j["triangleMaxPoints"] = triangleMaxPoints;
        j["triangleMaxCandidatePairs"] = triangleMaxCandidatePairs;
        return j;
    }

    static FrameFuseCPUParams fromJson(const nlohmann::json& j) {
        FrameFuseCPUParams p;
        if (j.contains("knnK")) p.knnK = j.at("knnK").get<int>();
        if (j.contains("descriptorBins1")) p.descriptorBins1 = j.at("descriptorBins1").get<int>();
        if (j.contains("descriptorBins2")) p.descriptorBins2 = j.at("descriptorBins2").get<int>();
        if (j.contains("descriptorBins3")) p.descriptorBins3 = j.at("descriptorBins3").get<int>();
        if (j.contains("loweRatio")) p.loweRatio = j.at("loweRatio").get<double>();
        if (j.contains("normalPreFilterAngleDeg")) p.normalPreFilterAngleDeg = j.at("normalPreFilterAngleDeg").get<double>();
        if (j.contains("ransacConfidence")) p.ransacConfidence = j.at("ransacConfidence").get<double>();
        if (j.contains("ransacMaxIterations")) p.ransacMaxIterations = j.at("ransacMaxIterations").get<int>();
        if (j.contains("minInlierCount")) p.minInlierCount = j.at("minInlierCount").get<int>();
        if (j.contains("refineIterations")) p.refineIterations = j.at("refineIterations").get<int>();
        if (j.contains("refineConvergeRatio")) p.refineConvergeRatio = j.at("refineConvergeRatio").get<double>();
        if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
        if (j.contains("maxPointCount")) p.maxPointCount = j.at("maxPointCount").get<size_t>();
        if (j.contains("matchMode")) p.matchMode = j.at("matchMode").get<int>();
        if (j.contains("triangleSideTolMm")) p.triangleSideTolMm = j.at("triangleSideTolMm").get<double>();
        if (j.contains("triangleSideTolRatio")) p.triangleSideTolRatio = j.at("triangleSideTolRatio").get<double>();
        if (j.contains("triangleVerifyGateMm")) p.triangleVerifyGateMm = j.at("triangleVerifyGateMm").get<double>();
        if (j.contains("triangleVerifyGateRatio")) p.triangleVerifyGateRatio = j.at("triangleVerifyGateRatio").get<double>();
        if (j.contains("triangleNormalMaxAngleDeg")) p.triangleNormalMaxAngleDeg = j.at("triangleNormalMaxAngleDeg").get<double>();
        if (j.contains("triangleMaxPoints")) p.triangleMaxPoints = j.at("triangleMaxPoints").get<int>();
        if (j.contains("triangleMaxCandidatePairs")) p.triangleMaxCandidatePairs = j.at("triangleMaxCandidatePairs").get<int>();
        return p;
    }
};

// ============================================================
// Input Data
// ============================================================

struct MarkerPointSet {
    std::vector<cv::Point3d> positions;
    std::vector<cv::Vec3d> normals;

    size_t size() const noexcept { return positions.size(); }
    bool empty() const noexcept { return positions.empty(); }
    void clear() noexcept { positions.clear(); normals.clear(); }
};

// ============================================================
// Statistics
// ============================================================

struct FrameFuseStats {
    double totalTimeMs = 0.0;
    double knnTimeMs = 0.0;
    double descriptorTimeMs = 0.0;
    double matchingTimeMs = 0.0;
    double ransacTimeMs = 0.0;
    double refineTimeMs = 0.0;
    size_t set1PointCount = 0;
    size_t set2PointCount = 0;
    size_t descriptorCorrespondences = 0;
    size_t normalFilteredCorrespondences = 0;
    size_t ransacInliers = 0;
    int ransacIterations = 0;
    int refineIterationsActual = 0;
    double adaptiveDistThreshCoarse = 0.0;
    double adaptiveDistThreshFine = 0.0;
    double initialRMSE = 0.0;
    double finalRMSE = 0.0;
    // —— 三角形模式诊断（2026-10-15 批1 埋点）——
    int solutionMargin = -1;             // 胜出解 vs 最强竞争解边际（独占对数差；<2 已弃权）
    bool triangleCapped = false;         // 工作量封顶触发（结果可能不完整）
    int zeroNormalCount = 0;             // 零法线静默替换计数（上游质量信号）
    double winnerPairDistMaxMm = -1.0;   // 胜出解最大配对距离（残差分布锚点）
    // —— 三角形模式分项计时（2026-10-15 提速步①：定位热点，单位 ms）——
    double triCollectMs = 0.0;           // 三角形枚举预处理（两侧 C(n,3)+退化剔除）
    double triScanMs = 0.0;              // 边长门扫描（ta×tb 组合比较，扣除解算后的净扫描）
    double triEvalMs = 0.0;              // 候选解算（SVD 三点解＋全点复核）
    double triSvdMs = 0.0;               // 解算内部：三点闭式解 svd3Point
    double triVerifyMs = 0.0;            // 解算内部：全点复核 triVerify
    double triJudgeMs = 0.0;             // 裁决（解账本扫描＋独占边际）
    long long triPairChecks = 0;         // 组合比较次数（ta×tb 迭代数）
    long long triEvalCount = 0;          // 过门组合数
    long long triSvdCount = 0;           // 实际 SVD+复核调用数（含排列展开）
};

// ============================================================
// Result
// ============================================================

struct FrameFuseCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Matx44d transform = cv::Matx44d::eye();

    double rmse = 0.0;
    double normalRMSE = 0.0;
    size_t matchedCount = 0;
    size_t totalCorrespondences = 0;
    double overlapRatio = 0.0;
    std::vector<std::pair<int, int>> correspondences;

    FrameFuseStats statistics;

    FrameFuseCPUResult() = default;
    ~FrameFuseCPUResult() = default;

    FrameFuseCPUResult(FrameFuseCPUResult&&) = default;
    FrameFuseCPUResult& operator=(FrameFuseCPUResult&&) = default;

    FrameFuseCPUResult(const FrameFuseCPUResult&) = delete;
    FrameFuseCPUResult& operator=(const FrameFuseCPUResult&) = delete;
};

// ============================================================
// Operator Class
// ============================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 两帧点集均按调用传入，实例不持有跨调用状态；SetParams 缓存 RANSAC/描述子等只读配置；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API FrameFuseCPU {
public:
    static constexpr const char* kLogTag = "12-FrameFuseCPU";

    explicit FrameFuseCPU(const FrameFuseCPUParams& params = {});
    ~FrameFuseCPU();

    FrameFuseCPU(const FrameFuseCPU&) = delete;
    FrameFuseCPU& operator=(const FrameFuseCPU&) = delete;

    FrameFuseCPUResult Execute(const MarkerPointSet& set1,
              const MarkerPointSet& set2);

    FrameFuseCPUResult Execute(const PointReconstructCPUResult& reconstruct1,
              const PointReconstructCPUResult& reconstruct2);

    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const FrameFuseCPUParams& params);
    const FrameFuseCPUParams& GetParams() const;

    void Destroy();

    const FrameFuseStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getFrameFuseCPUInfo();

} // namespace calib