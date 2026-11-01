#pragma once
/**
 * @file conic_reconstruct_cpu.h
 * @brief 双目圆锥相交重建算子 - 公开头（纯 C++，CPU 实现）
 *
 * 所属流程：扫描/标定标记点链（marker_match 之后）
 * 功能：左右椭圆整体参数 → 空间圆（圆心/单位法向/半径），方案 A 结构化闭式。
 * 输入：EllipseFitCPUResult（矫正系）×2 ＋ centerMatches 种子对 ＋ 注入 K1/K2/C2
 * 输出：MarkerReconstructResult 兼容（centerX/Y/Z, normalX/Y/Z, circleFit.radius）
 * 精度容差档次：档次②（依赖输入椭圆亚像素质量）
 */
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "common/calib_types.h"
#include "common/zitai_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

struct WarmupConfig;

struct ConicReconstructCPUParams {
    double nominalRadiusMm = 3.0;
    double radiusToleranceRatio = 0.15;
    double maxReprojResidualPx = 1.5;
    double consistencyMax = 0.05;
    double circleAxisRatioMax = 0.03;
    double degeneracyGateAngleDeg = 8.0;
    double depthMinMm = 100.0;
    double depthMaxMm = 2000.0;
    double minPlaneDistMm = 10.0;
    double yPreGatePx = 60.0;
    double rowOffsetPx = 0.0;   // 右目椭圆中心行偏置预修正：vR'=vR−rowOffsetPx。本数据
                                // 标定残差代偿（conic_offline --row-offset scan 定档，重标定后重定）
    int ringPointsForViz = 36;
    bool collectStatistics = true;

    void validate() const;
    nlohmann::json toJson() const;
    static ConicReconstructCPUParams fromJson(const nlohmann::json& j);
};

enum class ConicFailReason {
    None, Numerical, Coincident, NoRealRoot, Inconsistent, VirtualPlane,
    PlaneDegenerate, NotCircle, NegativeR, Cheirality, Radius, Residual,
    Degeneracy, Ambiguous, InputClass
};

// 失败原因计数口径（failReasonHistogram 数组维度＝枚举值个数）
static constexpr size_t kFailReasonCount = static_cast<size_t>(ConicFailReason::InputClass) + 1;
static_assert(kFailReasonCount == 15, "ConicFailReason enum changed - update kFailReasonCount users");

// 拒因名单源名表（顺序＝枚举值序；harness/conic_offline/单测共用，禁本地复制——
// 重排枚举时 static_assert 强制同步）
inline const char* ConicFailReasonName(ConicFailReason r) {
    static const char* const kNames[] = { "None", "Numerical", "Coincident", "NoRealRoot",
        "Inconsistent", "VirtualPlane", "PlaneDegenerate", "NotCircle", "NegativeR",
        "Cheirality", "Radius", "Residual", "Degeneracy", "Ambiguous", "InputClass" };
    static_assert(std::size(kNames) == kFailReasonCount, "new enum needs a name");
    return kNames[static_cast<size_t>(r)];
}

struct ConicPairDiagnostic {
    int leftIdx = -1;
    int rightIdx = -1;
    ConicFailReason failReason = ConicFailReason::None;
    double reconstructedRadiusMm = 0.0;
    double reprojResidualPx = -1.0;   // 双目 max-median：成功/Ambiguous=幸存者残差；全灭=末次被拒候选值；-1=未达残差闸
    double sectionAxisRatio = -1.0;   // 截面非圆度（轴比），-1=未达截面提取（幸存者/末次评估候选口径，同 reprojResidualPx）
    double etaBaseline = 0.0;   // |n·b̂|
    double etaView = 0.0;       // |(L−C)·n̂|/|C−L|
    int solutionCount = 0;
    bool droppedByCompetition = false;   // 同左竞争落败者标记——failReason 仍为 None
};

struct ConicReconstructStats {
    size_t totalPairs = 0;       // 种子对数
    size_t candidatePairs = 0;   // 实际参与 solvePair 的候选数（种子＋预筛，cap 截断后）
    size_t validCount = 0;
    size_t rejectedCount = 0;
    std::array<size_t, kFailReasonCount> failReasonHistogram{};   // 按 ConicFailReason 计数（None=过闸对计数，含竞争落败者，见 droppedByCompetition）
};

struct ConicReconstructCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::vector<MarkerReconstructResult> markers;
    std::vector<ConicPairDiagnostic> diagnostics;
    ConicReconstructStats statistics;

    ConicReconstructCPUResult() = default;
    ~ConicReconstructCPUResult() = default;
    ConicReconstructCPUResult(ConicReconstructCPUResult&&) = default;
    ConicReconstructCPUResult& operator=(ConicReconstructCPUResult&&) = default;
    ConicReconstructCPUResult(const ConicReconstructCPUResult&) = delete;
    ConicReconstructCPUResult& operator=(const ConicReconstructCPUResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 标定矩阵由调用方经 SetProjectionMatrices 注入（矫正系 K1/K2/右光心）；
//       Impl 仅持有每调用重置的暂存缓冲，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API ConicReconstructCPU {
public:
    static constexpr const char* kLogTag = "13-ConicReconstructCPU";

    explicit ConicReconstructCPU(const ConicReconstructCPUParams& params = {});
    ~ConicReconstructCPU();
    ConicReconstructCPU(const ConicReconstructCPU&) = delete;
    ConicReconstructCPU& operator=(const ConicReconstructCPU&) = delete;

    ConicReconstructCPUResult Execute(const std::vector<EllipseFitCPUResult>& left,
                                      const std::vector<EllipseFitCPUResult>& right,
                                      const std::vector<int>& centerMatches);

    void SetProjectionMatrices(const cv::Matx33d& K1, const cv::Matx33d& K2,
                               const cv::Vec3d& rightCenter);
    void Warmup(int maxPairCount);
    void Warmup(const WarmupConfig& config);
    void SetParams(const ConicReconstructCPUParams& params);
    const ConicReconstructCPUParams& GetParams() const;
    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getConicReconstructCPUInfo();

} // namespace calib
