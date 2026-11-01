#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>
#include "common/json_utils.h"
#include "common/quality_flag.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

struct InverseDistortParams {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat R1;
    cv::Mat P1;
    int maxIterations = 20;
    double tolerance = 1e-10;

    static InverseDistortParams fromJson(const std::string& json_path);
    // validate() is called automatically in the InverseDistortCPU constructor
    void validate() const;
};

struct InverseDistortResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<cv::Point2f> originalPoints;
    std::vector<int> iterationsPerPoint;
    int maxIterationsUsed = 0;

    InverseDistortResult() = default;
    // move-only (spec §3.3)
    InverseDistortResult(const InverseDistortResult&) = delete;
    InverseDistortResult& operator=(const InverseDistortResult&) = delete;
    InverseDistortResult(InverseDistortResult&&) noexcept = default;
    InverseDistortResult& operator=(InverseDistortResult&&) noexcept = default;
};

struct RoundTripVerifyResult {
    bool success = false;
    bool passed = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    double maxError = 0.0;
    double meanError = 0.0;
    std::vector<double> perPointErrors;

    RoundTripVerifyResult() = default;
    // move-only (spec §3.3)
    RoundTripVerifyResult(const RoundTripVerifyResult&) = delete;
    RoundTripVerifyResult& operator=(const RoundTripVerifyResult&) = delete;
    RoundTripVerifyResult(RoundTripVerifyResult&&) noexcept = default;
    RoundTripVerifyResult& operator=(RoundTripVerifyResult&&) noexcept = default;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: 待去畸变点集按调用传入；SetParams 缓存内参/畸变/矫正矩阵等只读配置（可改为按调用传入），无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API InverseDistortCPU {
public:
    inline static const char* kLogTag = "[01-InverseDistortCpu]";

    explicit InverseDistortCPU(const InverseDistortParams& params);
    ~InverseDistortCPU();

    InverseDistortCPU(const InverseDistortCPU&) = delete;
    InverseDistortCPU& operator=(const InverseDistortCPU&) = delete;
    InverseDistortCPU(InverseDistortCPU&&) noexcept;
    InverseDistortCPU& operator=(InverseDistortCPU&&) noexcept;

    bool Execute(
        const std::vector<cv::Point2f>& rectifiedPoints,
        InverseDistortResult& result);

    bool ApplyDistortion(
        const std::vector<cv::Point2f>& undistortedNormPoints,
        std::vector<cv::Point2f>& distortedPixelPoints);

    bool InverseRectify(
        const std::vector<cv::Point2f>& rectifiedPoints,
        std::vector<cv::Point2f>& unrectifiedNormPoints);

    RoundTripVerifyResult VerifyRoundTrip(
        const std::vector<cv::Point2f>& originalDistortedPoints,
        double maxAllowableError = 0.001);

    void Warmup() { }

    void Destroy();

    void SetParams(const InverseDistortParams& params);
    const InverseDistortParams& GetParams() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getInverseDistortCPUInfo();

} // namespace calib
