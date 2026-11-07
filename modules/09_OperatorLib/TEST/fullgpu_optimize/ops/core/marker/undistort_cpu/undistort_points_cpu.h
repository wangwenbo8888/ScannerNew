/**
 * @file undistort_points_cpu.h
 * @brief 双目立体去畸�?矫正算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑥（共用�? * 平台：CPU
 *
 * 功能：对双目稀疏特征点执行去畸�?+ 立体矫正（一步完成）�? *       核心算法委托 cv::stereoRectify() + cv::undistortPoints()�? *       默认所有标定参数已给定�? *
 * 推荐调用位置：阶段B最前，07 椭圆拟合之前�? *   05 输出 EdgePoint（含畸变）→ 本算子去畸变+立体矫正 �?07 在无畸变坐标上拟合椭�?�?08 极线约束匹配�? *   先去畸变再拟合椭圆可避免非线性畸变导致的椭圆参数系统误差�? *   立体矫正后左右对应点 Y 坐标对齐，是 08 正确工作的前提�? *
 * 精度容差档次：档次①（亚像素级，~0.01 pixel�? */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

struct MarkerUndistortCPUParams {
    // ---- Camera 1 (左相�? ----
    double fx1 = 0.0;
    double fy1 = 0.0;
    double cx1 = 0.0;
    double cy1 = 0.0;
    double k1_1 = 0.0;
    double k2_1 = 0.0;
    double p1_1 = 0.0;
    double p2_1 = 0.0;
    double k3_1 = 0.0;
    double k4_1 = 0.0;
    double k5_1 = 0.0;
    double k6_1 = 0.0;

    // ---- Camera 2 (右相�? ----
    double fx2 = 0.0;
    double fy2 = 0.0;
    double cx2 = 0.0;
    double cy2 = 0.0;
    double k1_2 = 0.0;
    double k2_2 = 0.0;
    double p1_2 = 0.0;
    double p2_2 = 0.0;
    double k3_2 = 0.0;
    double k4_2 = 0.0;
    double k5_2 = 0.0;
    double k6_2 = 0.0;

    // ---- 立体外参 (R: 3x3 行优�? T: 3x1) ----
    std::array<double, 9> R = {};
    std::array<double, 3> T = {};

    // ---- 图像尺寸 (stereoRectify 需�? ----
    int imageWidth = 0;
    int imageHeight = 0;

    // ---- 畸变模型 ----
    std::string distortionModel = "brown_conrady";

    void validate() const {
        if (distortionModel != "brown_conrady" && distortionModel != "rational_polynomial")
            throw std::invalid_argument("distortionModel must be 'brown_conrady' or 'rational_polynomial'");

        validateCamera1();
        validateCamera2();
    }

    void validateCamera1() const {
        if (fx1 <= 0) throw std::invalid_argument("fx1 must be > 0");
        if (fy1 <= 0) throw std::invalid_argument("fy1 must be > 0");
    }

    void validateCamera2() const {
        if (fx2 <= 0) throw std::invalid_argument("fx2 must be > 0");
        if (fy2 <= 0) throw std::invalid_argument("fy2 must be > 0");
        if (imageWidth <= 0) throw std::invalid_argument("imageWidth must be > 0");
        if (imageHeight <= 0) throw std::invalid_argument("imageHeight must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"fx1", fx1}, {"fy1", fy1}, {"cx1", cx1}, {"cy1", cy1},
            {"k1_1", k1_1}, {"k2_1", k2_1}, {"p1_1", p1_1}, {"p2_1", p2_1},
            {"k3_1", k3_1}, {"k4_1", k4_1}, {"k5_1", k5_1}, {"k6_1", k6_1},
            {"fx2", fx2}, {"fy2", fy2}, {"cx2", cx2}, {"cy2", cy2},
            {"k1_2", k1_2}, {"k2_2", k2_2}, {"p1_2", p1_2}, {"p2_2", p2_2},
            {"k3_2", k3_2}, {"k4_2", k4_2}, {"k5_2", k5_2}, {"k6_2", k6_2},
            {"R", R}, {"T", T},
            {"imageWidth", imageWidth}, {"imageHeight", imageHeight},
            {"distortionModel", distortionModel}
        };
    }

    static MarkerUndistortCPUParams fromJson(const nlohmann::json& j) {
        MarkerUndistortCPUParams p;
        if (j.contains("fx1")) p.fx1 = j.at("fx1").get<double>();
        if (j.contains("fy1")) p.fy1 = j.at("fy1").get<double>();
        if (j.contains("cx1")) p.cx1 = j.at("cx1").get<double>();
        if (j.contains("cy1")) p.cy1 = j.at("cy1").get<double>();
        if (j.contains("k1_1")) p.k1_1 = j.at("k1_1").get<double>();
        if (j.contains("k2_1")) p.k2_1 = j.at("k2_1").get<double>();
        if (j.contains("p1_1")) p.p1_1 = j.at("p1_1").get<double>();
        if (j.contains("p2_1")) p.p2_1 = j.at("p2_1").get<double>();
        if (j.contains("k3_1")) p.k3_1 = j.at("k3_1").get<double>();
        if (j.contains("k4_1")) p.k4_1 = j.at("k4_1").get<double>();
        if (j.contains("k5_1")) p.k5_1 = j.at("k5_1").get<double>();
        if (j.contains("k6_1")) p.k6_1 = j.at("k6_1").get<double>();
        if (j.contains("fx2")) p.fx2 = j.at("fx2").get<double>();
        if (j.contains("fy2")) p.fy2 = j.at("fy2").get<double>();
        if (j.contains("cx2")) p.cx2 = j.at("cx2").get<double>();
        if (j.contains("cy2")) p.cy2 = j.at("cy2").get<double>();
        if (j.contains("k1_2")) p.k1_2 = j.at("k1_2").get<double>();
        if (j.contains("k2_2")) p.k2_2 = j.at("k2_2").get<double>();
        if (j.contains("p1_2")) p.p1_2 = j.at("p1_2").get<double>();
        if (j.contains("p2_2")) p.p2_2 = j.at("p2_2").get<double>();
        if (j.contains("k3_2")) p.k3_2 = j.at("k3_2").get<double>();
        if (j.contains("k4_2")) p.k4_2 = j.at("k4_2").get<double>();
        if (j.contains("k5_2")) p.k5_2 = j.at("k5_2").get<double>();
        if (j.contains("k6_2")) p.k6_2 = j.at("k6_2").get<double>();
        if (j.contains("R")) p.R = j.at("R").get<std::array<double, 9>>();
        if (j.contains("T")) p.T = j.at("T").get<std::array<double, 3>>();
        if (j.contains("imageWidth")) p.imageWidth = j.at("imageWidth").get<int>();
        if (j.contains("imageHeight")) p.imageHeight = j.at("imageHeight").get<int>();
        if (j.contains("distortionModel")) p.distortionModel = j.at("distortionModel").get<std::string>();
        return p;
    }
};

struct StereoUndistortResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<cv::Point2d> rectifiedPoints1;
    std::vector<cv::Point2d> rectifiedPoints2;
    int pointCount1 = 0;
    int pointCount2 = 0;

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;

    std::vector<int> groupIds1;
    std::vector<int> groupIds2;
    int groupCount1 = 0;
    int groupCount2 = 0;

    std::vector<std::vector<cv::Point2d>> splitRectifiedPoints1ByGroup() const {
        if (groupCount1 == 0 || groupIds1.empty())
            return {};
        std::vector<std::vector<cv::Point2d>> groups(groupCount1);
        for (size_t i = 0; i < rectifiedPoints1.size() && i < groupIds1.size(); ++i)
            groups[groupIds1[i]].push_back(rectifiedPoints1[i]);
        return groups;
    }

    std::vector<std::vector<cv::Point2d>> splitRectifiedPoints2ByGroup() const {
        if (groupCount2 == 0 || groupIds2.empty())
            return {};
        std::vector<std::vector<cv::Point2d>> groups(groupCount2);
        for (size_t i = 0; i < rectifiedPoints2.size() && i < groupIds2.size(); ++i)
            groups[groupIds2[i]].push_back(rectifiedPoints2[i]);
        return groups;
    }

    StereoUndistortResult() = default;
    ~StereoUndistortResult() = default;

    StereoUndistortResult(StereoUndistortResult&&) = default;
    StereoUndistortResult& operator=(StereoUndistortResult&&) = default;

    StereoUndistortResult(const StereoUndistortResult&) = delete;
    StereoUndistortResult& operator=(const StereoUndistortResult&) = delete;
};

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 跨调用所需的 R1/R2/P1/P2/Q 矫正矩阵由调用方经 setRectifyMatrices 注入（可改为按调用传入），实例本身不持有其他功能性状态。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API MarkerUndistortCPU {
public:
    static constexpr const char* kLogTag = "06-MarkerUndistortCPU";

    explicit MarkerUndistortCPU(const MarkerUndistortCPUParams& params = {});

    ~MarkerUndistortCPU();

    MarkerUndistortCPU(const MarkerUndistortCPU&) = delete;
    MarkerUndistortCPU& operator=(const MarkerUndistortCPU&) = delete;

    StereoUndistortResult Execute(
        const std::vector<EdgePoint>& edgePoints1,
        const std::vector<EdgePoint>& edgePoints2,
        const std::vector<int>& groupIds1 = {},
        const std::vector<int>& groupIds2 = {});

    StereoUndistortResult Execute(
        const std::vector<cv::Point2d>& points1,
        const std::vector<cv::Point2d>& points2,
        const std::vector<int>& groupIds1 = {},
        const std::vector<int>& groupIds2 = {});

    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);

    void SetRectifyMatrices(const cv::Mat& R1, const cv::Mat& R2,
                            const cv::Mat& P1, const cv::Mat& P2,
                            const cv::Mat& Q);
    void ClearRectifyMatrices();

    void SetParams(const MarkerUndistortCPUParams& params);
    const MarkerUndistortCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMarkerUndistortCPUInfo();

} // namespace calib