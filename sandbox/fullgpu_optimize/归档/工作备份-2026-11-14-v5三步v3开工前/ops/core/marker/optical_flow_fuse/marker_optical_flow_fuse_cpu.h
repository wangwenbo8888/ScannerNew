#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "common/calib_types.h"
#include "common/zitai_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"
#include "scheduler/prev_frame_state.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

struct MarkerOpticalFlowFuseCPUParams {
    double matchDistThresh = 2.0;
    double normalAngleThresh = 15.0;
    int minMatchedPoints = 3;
    bool collectStatistics = true;
    size_t maxMarkerCount = 1000;
    /// 首帧挂全局地图分支的搜索半径倍率（×matchDistThresh）。首帧是整条轨迹基准，
    /// 历史口径放宽 10 倍兜住"当前帧与地图间无位姿先验"的初始不确定性——参数化备调。
    double firstFrameSearchScale = 10.0;
    /// 成败判定的误差门（mm，rmse）。0=关闭（历史口径：只数匹配对数不看误差）；
    /// >0 时解算 rmse 超门即判失败——堵"数量够但解歪了"的静默错误。
    double rmseMaxGate = 0.0;
    /// 法线可信度加权（cos²）——【2026-10-15 裁决：默认关闭（等权解算）】。
    /// 理由：默认 15° 朝向门下过门配对权重差仅 ~7%，近似等权、无实际区分作用；
    /// 保留开关作为"朝向门放宽时的软降级保险丝"（门=硬淘汰，权重=软降级）——
    /// 若将来把 normalAngleThresh 调大到 30°+，应同步置 true 抑制低质量配对。
    /// 真正的质量加权方向是按残差迭代降权（见 registration_optimize 后续方案）。
    bool useNormalWeight = false;

    void validate() const;
    nlohmann::json toJson() const;
    static MarkerOpticalFlowFuseCPUParams fromJson(const nlohmann::json& j);
};

struct PrevFrameState {
    std::vector<cv::Point3d> rawPositions;
    std::vector<cv::Vec3d> rawNormals;
    std::vector<int> globalIds;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T = cv::Vec3d(0.0, 0.0, 0.0);

    bool empty() const noexcept { return rawPositions.empty(); }
    size_t size() const noexcept { return rawPositions.size(); }
};

// 调度层 AtomicFrameState → 本算子 PrevFrameState 的桥接转换（算子规范 §3.1：
// 集中一处，消除扫描/标定两条 pipeline 各抄一份）。位于目标类型 PrevFrameState 旁。
// ⚠ 必须带走点位（rawPositions/rawNormals）——它们是后续帧配准查询的基准；
//   只拷 R/T/globalIds 会产出恒空状态，使算子每帧都走首帧建系（2026-10-14 修复）。
inline PrevFrameState toPrevFrameState(const AtomicFrameState& sched) {
    PrevFrameState of;
    of.rawPositions.reserve(sched.rawPoints.size());
    of.rawNormals.reserve(sched.rawPoints.size());
    for (const auto& p : sched.rawPoints) {
        of.rawPositions.emplace_back(p.x, p.y, p.z);
        of.rawNormals.emplace_back(p.nx, p.ny, p.nz);
    }
    if (sched.globalIds.size() == sched.rawPoints.size()) {
        of.globalIds = sched.globalIds;
    } else {
        of.globalIds.reserve(sched.rawPoints.size());
        for (const auto& p : sched.rawPoints) of.globalIds.push_back(p.globalId);
    }
    of.R = cv::Matx33d(
        sched.R[0], sched.R[1], sched.R[2],
        sched.R[3], sched.R[4], sched.R[5],
        sched.R[6], sched.R[7], sched.R[8]);
    of.T = cv::Vec3d(sched.T[0], sched.T[1], sched.T[2]);
    return of;
}

struct GlobalMarkerSet {
    std::vector<cv::Point3d> positions;
    std::vector<cv::Vec3d> normals;

    bool empty() const noexcept { return positions.empty(); }
    size_t size() const noexcept { return positions.size(); }
};

struct TransformedMarker {
    cv::Point3d rawPosition;
    cv::Vec3d rawNormal;
    cv::Point3d transformedPosition;
    cv::Vec3d transformedNormal;
    int globalId = -1;
    bool matched = false;
    double matchDistance = 0.0;
};

struct MarkerOpticalFlowFuseStats {
    double totalTimeMs = 0.0;
    double matchTimeMs = 0.0;
    double svdTimeMs = 0.0;
    double transformTimeMs = 0.0;
    size_t currentFrameCount = 0;
    size_t prevFrameCount = 0;
    size_t matchedCount = 0;
    size_t unmatchedCount = 0;
    double rmse = 0.0;
    // —— 生效参数快照（2026-10-15 加固：事故复盘对账——config 值≠运行时值，
    //    且守卫容差/免死线/首帧半径是门的隐藏派生量，改门时四值同变）——
    double effMatchDistThresh = -1.0;   // 匹配门
    double effGuardTolerance = -1.0;    // 编号-地图守卫容差 = 2×门
    double effRejectFloor = -1.0;       // 剔坏对绝对下限（固定 0.5mm，与门解耦）
    double effFirstFrameRadius = -1.0;  // 首帧/挂图搜索半径 = 门×firstFrameSearchScale
    double effRmseMaxGate = -1.0;       // 误差门（0=关）
    bool effUseNormalWeight = false;    // 法线加权开关
};

struct MarkerOpticalFlowFuseCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Matx44d transform = cv::Matx44d::eye();

    std::vector<TransformedMarker> markers;
    MarkerOpticalFlowFuseStats statistics;

    MarkerOpticalFlowFuseCPUResult() = default;
    ~MarkerOpticalFlowFuseCPUResult() = default;

    MarkerOpticalFlowFuseCPUResult(MarkerOpticalFlowFuseCPUResult&&) = default;
    MarkerOpticalFlowFuseCPUResult& operator=(MarkerOpticalFlowFuseCPUResult&&) = default;

    MarkerOpticalFlowFuseCPUResult(const MarkerOpticalFlowFuseCPUResult&) = delete;
    MarkerOpticalFlowFuseCPUResult& operator=(const MarkerOpticalFlowFuseCPUResult&) = delete;

    std::vector<cv::Point3d> getTransformedPositions() const;
    std::vector<cv::Point3d> getRawPositions() const;
    std::vector<int> getGlobalIds() const;
    size_t getMatchedCount() const;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 跨帧状态(PrevFrameState)由调用方按调用传入，实例本身不持有跨调用功能性状态；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API MarkerOpticalFlowFuseCPU {
public:
    static constexpr const char* kLogTag = "01-MarkerOpticalFlowFuseCPU";

    explicit MarkerOpticalFlowFuseCPU(const MarkerOpticalFlowFuseCPUParams& params = {});
    ~MarkerOpticalFlowFuseCPU();

    MarkerOpticalFlowFuseCPU(const MarkerOpticalFlowFuseCPU&) = delete;
    MarkerOpticalFlowFuseCPU& operator=(const MarkerOpticalFlowFuseCPU&) = delete;

    MarkerOpticalFlowFuseCPUResult Execute(const std::vector<cv::Point3d>& currentPositions,
              const std::vector<cv::Vec3d>& currentNormals,
              const PrevFrameState& prevState);

    MarkerOpticalFlowFuseCPUResult Execute(const PointReconstructCPUResult& currentFrame,
              const PrevFrameState& prevState);

    MarkerOpticalFlowFuseCPUResult Execute(const std::vector<cv::Point3d>& currentPositions,
              const std::vector<cv::Vec3d>& currentNormals,
              const PrevFrameState& prevState,
              const GlobalMarkerSet& globalMarkers);

    void Warmup(int maxMarkerCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const MarkerOpticalFlowFuseCPUParams& params);
    const MarkerOpticalFlowFuseCPUParams& GetParams() const;

    void Destroy();

    const MarkerOpticalFlowFuseStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMarkerOpticalFlowFuseCPUInfo();

} // namespace calib