/**
 * @file epipolar_intersect_cpu.h
 * @brief 椭圆边界极线交点算子 - 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑨（共用�? * 平台：CPU
 *
 * 功能：基于已拟合的椭圆参数和中心点，定义极线（水平线 y=const），
 *       计算极线与椭圆边界的亚像素交点集，将中心点和交点集标记为一个集合�? *
 * 输入：来�?ellipse_fit_cpu (07) 的椭圆参数（中心、长轴、短轴、旋转角�? * 输出：每个椭圆的中心�?+ 极线交点亚像素坐标集
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
 * @brief 极线交点算子参数
 *
 * 异常行为�? * - fromJson() 未知字段 -> 忽略（不抛异常），保证前向兼�? */
struct EpipolarIntersectCPUParams {
    double epipolarStep = 0.5;
    int maxIntersectionsPerEllipse = 1000;

    void validate() const {
        if (epipolarStep <= 0.0)
            throw std::invalid_argument("epipolarStep must be > 0");
        if (maxIntersectionsPerEllipse <= 0)
            throw std::invalid_argument("maxIntersectionsPerEllipse must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"epipolarStep", epipolarStep},
            {"maxIntersectionsPerEllipse", maxIntersectionsPerEllipse}
        };
    }

    static EpipolarIntersectCPUParams fromJson(const nlohmann::json& j) {
        EpipolarIntersectCPUParams p;
        if (j.contains("epipolarStep")) p.epipolarStep = j.at("epipolarStep").get<double>();
        if (j.contains("maxIntersectionsPerEllipse")) p.maxIntersectionsPerEllipse = j.at("maxIntersectionsPerEllipse").get<int>();
        return p;
    }
};

/**
 * @brief 极线交点算子完整结果
 */
struct EpipolarIntersectCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<EllipseIntersectResult> ellipseResults;

    EpipolarIntersectCPUResult() = default;
    ~EpipolarIntersectCPUResult() = default;

    EpipolarIntersectCPUResult(EpipolarIntersectCPUResult&&) = default;
    EpipolarIntersectCPUResult& operator=(EpipolarIntersectCPUResult&&) = default;

    EpipolarIntersectCPUResult(const EpipolarIntersectCPUResult&) = delete;
    EpipolarIntersectCPUResult& operator=(const EpipolarIntersectCPUResult&) = delete;
};

/**
 * @brief 椭圆边界极线交点算子（CPU 实现�? *
 * 算法原理�? *   1. 根据椭圆旋转参数计算Y方向边界范围
 *   2. �?epipolarStep 间距生成水平极线 y = y0 + k*step
 *   3. 对每条极线，代入椭圆方程求解二次方程得到交点X坐标
 *   4. 将中心点与交点集组合输出
 *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；SetParams 缓存极线步距等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EpipolarIntersectCPU {
public:
    static constexpr const char* kLogTag = "09-EpipolarIntersectCPU";

    explicit EpipolarIntersectCPU(const EpipolarIntersectCPUParams& params = {});

    ~EpipolarIntersectCPU();

    EpipolarIntersectCPU(const EpipolarIntersectCPU&) = delete;
    EpipolarIntersectCPU& operator=(const EpipolarIntersectCPU&) = delete;

    /**
     * @brief 单椭圆极线交点计�?- EllipseFitCPUResult 输入
     */
    EllipseIntersectResult Execute(const EllipseFitCPUResult& ellipseResult);

    /**
     * @brief 单椭圆极线交点计�?- RotatedRect 输入
     */
    EllipseIntersectResult Execute(const cv::RotatedRect& ellipse);

    /**
     * @brief 单椭圆极线交点计�?- 参数输入
     */
    EllipseIntersectResult Execute(double cx, double cy, double major, double minor, double angleDeg);

    /**
     * @brief 批量多椭圆极线交点计�?     */
    EpipolarIntersectCPUResult Execute(const std::vector<EllipseFitCPUResult>& ellipseResults);

    /**
     * @brief 批量多椭圆极线交点计�?- RotatedRect
     */
    EpipolarIntersectCPUResult Execute(const std::vector<cv::RotatedRect>& ellipses);

    void Warmup(int maxEllipseCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const EpipolarIntersectCPUParams& params);
    const EpipolarIntersectCPUParams& GetParams() const;

    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getEpipolarIntersectCPUInfo();

} // namespace calib