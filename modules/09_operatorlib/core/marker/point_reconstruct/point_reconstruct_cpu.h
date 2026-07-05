/**
 * @file point_reconstruct_cpu.h
 * @brief 标记点法线和中心快速三维重建算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑪（共用�? * 平台：CPU
 *
 * 功能：基于匹配的椭圆亚像素边缘点对，三角测量重建三维点，
 *       拟合平面，投影到平面上拟合圆，提取圆心三维坐标和法线方向�? *
 * 输入：来�?edge_match_cpu (10) 的匹配边缘点�?+ 相机内外�? * 输出：每个标记点的圆心三维坐标、法线方向、平�?圆拟合参�? *
 * 精度容差档次：档次②（亚像素级，~0.05 pixel 输入精度�? */

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

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

struct PointReconstructCPUParams {
    double fxLeft = 0.0;
    double fyLeft = 0.0;
    double cxLeft = 0.0;
    double cyLeft = 0.0;
    double fxRight = 0.0;
    double fyRight = 0.0;
    double cxRight = 0.0;
    double cyRight = 0.0;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T = cv::Vec3d(0.0, 0.0, 0.0);
    int minPointsForPlaneFit = 6;
    int minPointsForCircleFit = 6;
    double maxReprojError = 2.0;
    bool collectStatistics = true;
    size_t maxMarkerCount = 1000;

    void validate() const {
        if (fxLeft <= 0.0) throw std::invalid_argument("fxLeft must be > 0");
        if (fyLeft <= 0.0) throw std::invalid_argument("fyLeft must be > 0");
        if (fxRight <= 0.0) throw std::invalid_argument("fxRight must be > 0");
        if (fyRight <= 0.0) throw std::invalid_argument("fyRight must be > 0");
        if (minPointsForPlaneFit < 3)
            throw std::invalid_argument("minPointsForPlaneFit must be >= 3");
        if (minPointsForCircleFit < 3)
            throw std::invalid_argument("minPointsForCircleFit must be >= 3");
        if (maxReprojError <= 0.0)
            throw std::invalid_argument("maxReprojError must be > 0");
        if (maxMarkerCount == 0)
            throw std::invalid_argument("maxMarkerCount must be > 0");
        double det = cv::determinant(R);
        if (std::abs(std::abs(det) - 1.0) > 1e-3)
            throw std::invalid_argument("R must be a valid rotation matrix (|det| ~ 1)");
    }

    nlohmann::json toJson() const {
        nlohmann::json j;
        j["fxLeft"] = fxLeft;
        j["fyLeft"] = fyLeft;
        j["cxLeft"] = cxLeft;
        j["cyLeft"] = cyLeft;
        j["fxRight"] = fxRight;
        j["fyRight"] = fyRight;
        j["cxRight"] = cxRight;
        j["cyRight"] = cyRight;
        nlohmann::json rArr = nlohmann::json::array();
        for (int i = 0; i < 3; ++i) {
            nlohmann::json row = nlohmann::json::array();
            for (int k = 0; k < 3; ++k) row.push_back(R(i, k));
            rArr.push_back(row);
        }
        j["R"] = rArr;
        j["T"] = nlohmann::json::array({T(0), T(1), T(2)});
        j["minPointsForPlaneFit"] = minPointsForPlaneFit;
        j["minPointsForCircleFit"] = minPointsForCircleFit;
        j["maxReprojError"] = maxReprojError;
        j["collectStatistics"] = collectStatistics;
        j["maxMarkerCount"] = maxMarkerCount;
        return j;
    }

    static PointReconstructCPUParams fromJson(const nlohmann::json& j) {
        PointReconstructCPUParams p;
        if (j.contains("fxLeft")) p.fxLeft = j.at("fxLeft").get<double>();
        if (j.contains("fyLeft")) p.fyLeft = j.at("fyLeft").get<double>();
        if (j.contains("cxLeft")) p.cxLeft = j.at("cxLeft").get<double>();
        if (j.contains("cyLeft")) p.cyLeft = j.at("cyLeft").get<double>();
        if (j.contains("fxRight")) p.fxRight = j.at("fxRight").get<double>();
        if (j.contains("fyRight")) p.fyRight = j.at("fyRight").get<double>();
        if (j.contains("cxRight")) p.cxRight = j.at("cxRight").get<double>();
        if (j.contains("cyRight")) p.cyRight = j.at("cyRight").get<double>();
        if (j.contains("R") && j.at("R").is_array()) {
            auto ra = j.at("R");
            for (int i = 0; i < 3 && i < static_cast<int>(ra.size()); ++i) {
                auto row = ra[i];
                for (int k = 0; k < 3 && k < static_cast<int>(row.size()); ++k) {
                    p.R(i, k) = row[k].get<double>();
                }
            }
        }
        if (j.contains("T") && j.at("T").is_array()) {
            auto ta = j.at("T");
            for (int i = 0; i < 3 && i < static_cast<int>(ta.size()); ++i) {
                p.T(i) = ta[i].get<double>();
            }
        }
        if (j.contains("minPointsForPlaneFit"))
            p.minPointsForPlaneFit = j.at("minPointsForPlaneFit").get<int>();
        if (j.contains("minPointsForCircleFit"))
            p.minPointsForCircleFit = j.at("minPointsForCircleFit").get<int>();
        if (j.contains("maxReprojError"))
            p.maxReprojError = j.at("maxReprojError").get<double>();
        if (j.contains("collectStatistics"))
            p.collectStatistics = j.at("collectStatistics").get<bool>();
        if (j.contains("maxMarkerCount"))
            p.maxMarkerCount = j.at("maxMarkerCount").get<size_t>();
        return p;
    }
};

/**
 * @brief 标记点法线和中心快速三维重建算子（CPU 实现�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 投影矩阵 P1/P2/Q 由调用方经 setProjectionMatrices 注入（可改为按调用传入）；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API PointReconstructCPU {
public:
    static constexpr const char* kLogTag = "11-PointReconstructCPU";

    explicit PointReconstructCPU(const PointReconstructCPUParams& params = {});
    ~PointReconstructCPU();

    PointReconstructCPU(const PointReconstructCPU&) = delete;
    PointReconstructCPU& operator=(const PointReconstructCPU&) = delete;

    PointReconstructCPUResult Execute(const EdgeMatchCPUResult& edgeMatchResult);

    PointReconstructCPUResult Execute(const std::vector<cv::Point2f>& leftPoints,
                     const std::vector<cv::Point2f>& rightPoints,
                     const std::vector<int>& leftGroupIds,
                     const std::vector<int>& rightGroupIds,
                     const std::vector<int>& centerMatches);

    void Warmup(int maxMarkerCount);
    void Warmup(const WarmupConfig& config);

    void SetProjectionMatrices(const cv::Mat& P1, const cv::Mat& P2,
                               const cv::Mat& Q = cv::Mat());
    void ClearProjectionMatrices();

    void SetParams(const PointReconstructCPUParams& params);
    const PointReconstructCPUParams& GetParams() const;

    void Destroy();

    const PointReconstructStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getPointReconstructCPUInfo();

} // namespace calib