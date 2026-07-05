#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <utility>
#include "common/calib_types.h"
#include "common/quality_flag.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct WarmupConfig;

// 单帧观测:本机系3D点 + 全局ID(globalId 由前置全局ID模块赋予)
struct GlobalBAMarkerObs {
    cv::Point3d local;     // point_reconstruct 输出的本机系坐标(= 观测 z_ij)
    int         globalId = -1;  // 同物理标记点 = 同 globalId
};

struct GlobalBAFrame {
    uint64_t frameId = 0;
    cv::Matx33d R_init = cv::Matx33d::eye();   // 绝对位姿初值(相邻帧R/T链乘累积)
    cv::Vec3d  t_init = cv::Vec3d::zeros();
    std::vector<GlobalBAMarkerObs> markerObs;
};

struct GlobalBAInput {
    std::vector<GlobalBAFrame> frames;
};

struct GlobalBAParams {
    bool   enablePoseGraphPreopt = true;
    double sigmaObserved         = 0.01;   // mm
    double tukeyC                = 0.03;   // =3σ
    double chiSquareGate         = 11.34;  // 3-DoF 99%
    double tolerance             = 1e-10;
    int    maxIterations         = 200;
    int    minCovisForLoopEdge   = 5;
    int    loopFrameGap          = 30;
    int    minPointsPerFrame     = 3;
    bool   centerOrigin          = true;

    void validate() const;
    nlohmann::json toJson() const;
    static GlobalBAParams fromJson(const nlohmann::json& j);
};

struct GlobalBAStats {
    bool   loopDetected     = false;
    int    ceresIterations  = 0;
    double initialRMSE      = 0.0;   // mm
    double finalRMSE        = 0.0;   // mm
    std::vector<int> outlierObsIds;
    double loopClosureResidual = 0.0;
};

struct FramePose {
    uint64_t frameId = 0;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d  t = cv::Vec3d::zeros();
};

struct GlobalMarker {
    cv::Point3d X;
    int         globalId = -1;
    int         covisCount = 0;
};

struct GlobalBAResult {
    bool   success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<FramePose>    optimizedPoses;
    std::vector<GlobalMarker> optimizedMarkers;
    GlobalBAStats statistics;

    GlobalBAResult() = default;
    ~GlobalBAResult() = default;
    GlobalBAResult(GlobalBAResult&&) = default;
    GlobalBAResult& operator=(GlobalBAResult&&) = default;
    GlobalBAResult(const GlobalBAResult&) = delete;
    GlobalBAResult& operator=(const GlobalBAResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 全量数据按调用传入,实例不持有跨调用累积状态;SetParams 缓存只读配置。
// 重置接口: N/A
// 并发策略: 每实例非线程安全,多实例并行各自独占
// ==============================
class SCANNER_API GlobalBundleAdjustmentCPU {
public:
    static constexpr const char* kLogTag = "13-GlobalBundleAdjustmentCPU";

    explicit GlobalBundleAdjustmentCPU(const GlobalBAParams& params = {});
    ~GlobalBundleAdjustmentCPU();

    GlobalBundleAdjustmentCPU(const GlobalBundleAdjustmentCPU&) = delete;
    GlobalBundleAdjustmentCPU& operator=(const GlobalBundleAdjustmentCPU&) = delete;

    GlobalBAResult Execute(const GlobalBAInput& input);

    void SetParams(const GlobalBAParams& params);
    const GlobalBAParams& GetParams() const;
    const GlobalBAStats& GetStatistics() const noexcept;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getGlobalBundleAdjustmentCPUInfo();

} // namespace calib
