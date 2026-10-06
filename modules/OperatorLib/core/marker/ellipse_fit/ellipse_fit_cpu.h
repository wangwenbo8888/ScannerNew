/**
 * @file ellipse_fit_cpu.h
 * @brief 椭圆拟合和中心提取算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑦（共用�? * 平台：CPU
 *
 * 功能：对输入的椭圆边缘亚像素点集执行 RANSAC 离群点剔除，
 *       再用 fitEllipseAMS 进行鲁棒椭圆拟合，提取椭圆中心坐标�? *
 * 输入：去畸变矫正后的点集（推荐来�?06 MarkerUndistortCPU �?Point2d 输出�? *       也兼容直接来�?04 �?EdgePoint 亚像素点集）
 * 输出：椭圆中�?(cx, cy)、长�?短轴、旋转角
 *
 * 精度容差档次：档次①（亚像素级，~0.01 pixel�? */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
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

/**
 * @brief 椭圆拟合参数
 *
 * 异常行为�? * - fromJson() 未知字段 -> 忽略（不抛异常），保证前向兼�? */
struct EllipseFitCPUParams {
    int ransacIterations = 100;
    double ransacThreshold = 0.5;
    double minEllipseAxis = 2.0;
    double maxAxisRatio = 100.0;
    int minInliers = 5;
    double earlyStopRatio = 0.8;
    bool useAMS = true;

    void validate() const {
        if (ransacIterations <= 0)
            throw std::invalid_argument("ransacIterations must be > 0");
        if (ransacThreshold <= 0.0)
            throw std::invalid_argument("ransacThreshold must be > 0");
        if (minEllipseAxis <= 0.0)
            throw std::invalid_argument("minEllipseAxis must be > 0");
        if (maxAxisRatio <= 1.0)
            throw std::invalid_argument("maxAxisRatio must be > 1");
        if (minInliers < 5)
            throw std::invalid_argument("minInliers must be >= 5");
        if (earlyStopRatio <= 0.0 || earlyStopRatio > 1.0)
            throw std::invalid_argument("earlyStopRatio must be in (0, 1]");
    }

    nlohmann::json toJson() const {
        return {
            {"ransacIterations", ransacIterations},
            {"ransacThreshold", ransacThreshold},
            {"minEllipseAxis", minEllipseAxis},
            {"maxAxisRatio", maxAxisRatio},
            {"minInliers", minInliers},
            {"earlyStopRatio", earlyStopRatio},
            {"useAMS", useAMS}
        };
    }

    static EllipseFitCPUParams fromJson(const nlohmann::json& j) {
        EllipseFitCPUParams p;
        if (j.contains("ransacIterations")) p.ransacIterations = j.at("ransacIterations").get<int>();
        if (j.contains("ransacThreshold")) p.ransacThreshold = j.at("ransacThreshold").get<double>();
        if (j.contains("minEllipseAxis")) p.minEllipseAxis = j.at("minEllipseAxis").get<double>();
        if (j.contains("maxAxisRatio")) p.maxAxisRatio = j.at("maxAxisRatio").get<double>();
        if (j.contains("minInliers")) p.minInliers = j.at("minInliers").get<int>();
        if (j.contains("earlyStopRatio")) p.earlyStopRatio = j.at("earlyStopRatio").get<double>();
        if (j.contains("useAMS")) p.useAMS = j.at("useAMS").get<bool>();
        return p;
    }
};

/**
 * @brief 椭圆拟合和中心提取算子（CPU 实现�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 单线程流水线中应在帧间隙调用 setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 * - Debug 模式下维�?inProcess_ 原子变量进行断言检�? * - Release 模式下并发调用为未定义行为（数据竞争�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；SetParams 缓存 RANSAC/拟合阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EllipseFitCPU {
public:
    static constexpr const char* kLogTag = "07-EllipseFitCPU";

    explicit EllipseFitCPU(const EllipseFitCPUParams& params = {});

    ~EllipseFitCPU();

    EllipseFitCPU(const EllipseFitCPU&) = delete;
    EllipseFitCPU& operator=(const EllipseFitCPU&) = delete;

    /**
     * @brief 椭圆拟合 - EdgePoint 列表
     */
    EllipseFitCPUResult Execute(const std::vector<EdgePoint>& edgePoints);

    /**
     * @brief 椭圆拟合 - cv::Point2f 列表
     */
    EllipseFitCPUResult Execute(const std::vector<cv::Point2f>& points);

    /**
     * @brief 椭圆拟合 - cv::Point2d 列表
     */
    EllipseFitCPUResult Execute(const std::vector<cv::Point2d>& points);

    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const EllipseFitCPUParams& params);
    const EllipseFitCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEllipseFitCPUInfo();

} // namespace calib