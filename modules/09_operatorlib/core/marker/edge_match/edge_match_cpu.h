/**
 * @file edge_match_cpu.h
 * @brief 椭圆边界边缘点匹配算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑩（共用�? * 平台：CPU
 *
 * 功能：基于已匹配的椭圆中心点对和极线交点集，对左右相机椭圆边界上�? *       亚像素交点进行双目立体匹配，输出匹配点对及视差�? *
 * 输入：来�?epipolar_intersect_cpu (09) 的左右极线交点集
 *       + 来自 marker_match_cpu (08) 的中心点匹配关系
 * 输出：匹配点对坐标、视差、置信度、质量统�? *
 * 精度容差档次：档次②（亚像素级，~0.05 pixel 输入精度�? */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/zitai_result_types.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

/**
 * @brief 边缘匹配算子参数
 *
 * 异常行为�? * - fromJson() 未知字段 -> 忽略（不抛异常），保证前向兼�? */
struct EdgeMatchCPUParams {
    float yTolerance = 0.2f;
    float disparityMaxDiff = 10.0f;
    bool collectStatistics = true;
    size_t maxMatchPairs = 100000;

    void validate() const {
        if (yTolerance <= 0.0f)
            throw std::invalid_argument("yTolerance must be > 0");
        if (disparityMaxDiff <= 0.0f)
            throw std::invalid_argument("disparityMaxDiff must be > 0");
        if (maxMatchPairs == 0)
            throw std::invalid_argument("maxMatchPairs must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"yTolerance", yTolerance},
            {"disparityMaxDiff", disparityMaxDiff},
            {"collectStatistics", collectStatistics},
            {"maxMatchPairs", maxMatchPairs}
        };
    }

    static EdgeMatchCPUParams fromJson(const nlohmann::json& j) {
        EdgeMatchCPUParams p;
        if (j.contains("yTolerance")) p.yTolerance = j.at("yTolerance").get<float>();
        if (j.contains("disparityMaxDiff")) p.disparityMaxDiff = j.at("disparityMaxDiff").get<float>();
        if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
        if (j.contains("maxMatchPairs")) p.maxMatchPairs = j.at("maxMatchPairs").get<size_t>();
        return p;
    }
};

/**
 * @brief 椭圆边界边缘点匹配算子（CPU 实现�? *
 * 算法原理�? *   1. 利用已匹配的中心点对关系，确定左右椭圆的对应关系
 *   2. 对每组椭圆对，按 epipolarIndex 建立左右交点索引
 *   3. 对共有极线索引，�?X 坐标排序后顺序配�? *   4. 计算视差和置信度
 *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的暂存缓冲；SetParams 缓存容差等只读配置；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API EdgeMatchCPU {
public:
    static constexpr const char* kLogTag = "10-EdgeMatchCPU";

    explicit EdgeMatchCPU(const EdgeMatchCPUParams& params = {});

    ~EdgeMatchCPU();

    EdgeMatchCPU(const EdgeMatchCPU&) = delete;
    EdgeMatchCPU& operator=(const EdgeMatchCPU&) = delete;

    /**
     * @brief 执行边缘点匹配（EllipseIntersectResult 输入）
     *
    * @param leftIntersect   左相机极线交点集（来�?09epipolar_intersect_cpu�? 
    * @param rightIntersect  右相机极线交点集（来�?09epipolar_intersect_cpu�? 
    * @param centerMatches   中心匹配映射：centerMatches[leftIdx] = rightIdx, -1 = 未匹�? 
    * @param result          匹配结果
     */
    EdgeMatchCPUResult Execute(const std::vector<EllipseIntersectResult>& leftIntersect,
               const std::vector<EllipseIntersectResult>& rightIntersect,
               const std::vector<int>& centerMatches);

    /**
     * @brief 执行边缘点匹配（Point2f + 分组输入）
     *
    * @param leftPoints      左相机边缘点�? 
    * @param rightPoints     右相机边缘点�? 
    * @param leftGroupIds    左点所属椭圆编�? 
    * @param rightGroupIds   右点所属椭圆编�? 
    * @param centerMatches   中心匹配映射
     * @param epipolarIndices 每个点的极线索引
     * @param result          匹配结果
     */
    EdgeMatchCPUResult Execute(const std::vector<cv::Point2f>& leftPoints,
               const std::vector<cv::Point2f>& rightPoints,
               const std::vector<int>& leftGroupIds,
               const std::vector<int>& rightGroupIds,
               const std::vector<int>& centerMatches,
               const std::vector<int>& leftEpipolarIndices,
               const std::vector<int>& rightEpipolarIndices);

    void Warmup(int maxEllipsePairs);
    void Warmup(const WarmupConfig& config);

    void SetParams(const EdgeMatchCPUParams& params);
    const EdgeMatchCPUParams& GetParams() const;

    void Destroy();

    const EdgeMatchStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib