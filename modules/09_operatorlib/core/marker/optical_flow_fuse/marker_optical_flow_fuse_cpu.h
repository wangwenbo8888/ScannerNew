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
inline PrevFrameState toPrevFrameState(const AtomicFrameState& sched) {
    PrevFrameState of;
    of.R = cv::Matx33d(
        sched.R[0], sched.R[1], sched.R[2],
        sched.R[3], sched.R[4], sched.R[5],
        sched.R[6], sched.R[7], sched.R[8]);
    of.T = cv::Vec3d(sched.T[0], sched.T[1], sched.T[2]);
    of.globalIds = sched.globalIds;
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