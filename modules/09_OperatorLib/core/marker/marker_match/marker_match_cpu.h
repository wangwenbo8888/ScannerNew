/**
 * @file marker_match_cpu.h
 * @brief 标记点匹配算�?- 公开头文件（�?C++，CPU 实现�? *
 * 所属流程：标定流程1�?/ 标定流程2⑧（共用�? * 平台：CPU
 *
 * 功能：对立交矫正后的亚像素稀疏标记点执行双目立体匹配�? *       基于极线约束（Y坐标相同），在自适应容差窗口内进行唯一性匹配，
 *       输出视差、匹配置信度和质量统计�? *
 * 输入：来�?ellipse_fit_cpu (07) 的左右相机椭圆中心点�? *       �?7 �?06 去畸变矫正后的无畸变坐标上拟合，中心天然处于矫正坐标系中�? * 输出：匹配点对视差、有效标志、置信度
 *
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
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

/**
 * @brief 标记点匹配参�? *
 * 异常行为�? * - fromJson() 未知字段 -> 忽略（不抛异常），保证前向兼�? */
struct MarkerMatchCPUParams {
    float y_tolerance = 0.15f;
    bool enable_parallel = false;
    int num_threads = 0;
    bool collect_statistics = true;
    size_t prealloc_buffer_size = 128;
    size_t parallel_threshold = 50;
    size_t max_points = 100;
    size_t max_buffer_size = 10000;

    float density_threshold_high = 2.0f;
    float density_threshold_low = 0.5f;
    float dense_tolerance_scale = 0.8f;
    float sparse_tolerance_scale = 1.2f;

    void validate() const;

    nlohmann::json toJson() const {
        return {
            {"y_tolerance", y_tolerance},
            {"enable_parallel", enable_parallel},
            {"num_threads", num_threads},
            {"collect_statistics", collect_statistics},
            {"prealloc_buffer_size", prealloc_buffer_size},
            {"parallel_threshold", parallel_threshold},
            {"max_points", max_points},
            {"max_buffer_size", max_buffer_size},
            {"density_threshold_high", density_threshold_high},
            {"density_threshold_low", density_threshold_low},
            {"dense_tolerance_scale", dense_tolerance_scale},
            {"sparse_tolerance_scale", sparse_tolerance_scale}
        };
    }

    static MarkerMatchCPUParams fromJson(const nlohmann::json& j) {
        MarkerMatchCPUParams p;
        if (j.contains("y_tolerance")) p.y_tolerance = j.at("y_tolerance").get<float>();
        if (j.contains("enable_parallel")) p.enable_parallel = j.at("enable_parallel").get<bool>();
        if (j.contains("num_threads")) p.num_threads = j.at("num_threads").get<int>();
        if (j.contains("collect_statistics")) p.collect_statistics = j.at("collect_statistics").get<bool>();
        if (j.contains("prealloc_buffer_size")) p.prealloc_buffer_size = j.at("prealloc_buffer_size").get<size_t>();
        if (j.contains("parallel_threshold")) p.parallel_threshold = j.at("parallel_threshold").get<size_t>();
        if (j.contains("max_points")) p.max_points = j.at("max_points").get<size_t>();
        if (j.contains("max_buffer_size")) p.max_buffer_size = j.at("max_buffer_size").get<size_t>();
        if (j.contains("density_threshold_high")) p.density_threshold_high = j.at("density_threshold_high").get<float>();
        if (j.contains("density_threshold_low")) p.density_threshold_low = j.at("density_threshold_low").get<float>();
        if (j.contains("dense_tolerance_scale")) p.dense_tolerance_scale = j.at("dense_tolerance_scale").get<float>();
        if (j.contains("sparse_tolerance_scale")) p.sparse_tolerance_scale = j.at("sparse_tolerance_scale").get<float>();
        return p;
    }
};

/**
 * @brief 单点匹配结果
 */
struct PointMatch {
    float disparity = std::numeric_limits<float>::quiet_NaN();
    bool valid = false;
    float confidence = 0.0f;
};

/**
 * @brief 标记点匹配统�? */
struct MarkerMatchStats {
    double total_time_ms = 0.0;
    double sort_time_ms = 0.0;
    double match_time_ms = 0.0;
    size_t total_points = 0;
    size_t matched_points = 0;
    size_t ambiguous_points = 0;
    size_t unvisited_points = 0;
    float match_rate = 0.0f;
    float avg_disparity = 0.0f;
    float disparity_std = 0.0f;
    float avg_confidence = 0.0f;
    float disparity_consistency = 0.0f;
    int num_threads_used = 1;
    bool used_precomputed = false;
};

/**
 * @brief 标记点匹配算子完整结�? */
struct MarkerMatchCPUResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::vector<float> disparities;
    std::vector<uint8_t> valid_flags;
    std::vector<float> confidence;
    std::vector<int> centerMatches;
    MarkerMatchStats statistics;

    MarkerMatchCPUResult() = default;
    ~MarkerMatchCPUResult() = default;

    MarkerMatchCPUResult(MarkerMatchCPUResult&&) = default;
    MarkerMatchCPUResult& operator=(MarkerMatchCPUResult&&) = default;

    MarkerMatchCPUResult(const MarkerMatchCPUResult&) = delete;
    MarkerMatchCPUResult& operator=(const MarkerMatchCPUResult&) = delete;
};

/**
 * @brief 标记点匹配算子（CPU 实现�? *
 * 算法原理�? *   1. 极线约束：左右图像对应点应具有相�?Y 坐标（已立体矫正�? *   2. 唯一性检测：窗口内左点数=1 且右点数=1 时判定为匹配
 *   3. 标记机制：查询过的点立即标记，避免重复处�? *   4. 自适应容差：根据点集密度动态调整搜索窗�? *   5. 置信度计算：基于 Y 距离和视差一致�? *
 * 线程安全约束�? * - 非线程安全：不同线程不得同时调用 Execute() �?setParams()
 * - 多实例并发场景各实例独立持有参数无需加锁
 */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 右相机参考点由调用方经 setReferencePoints 注入并预处理（可改为按调用传入），matchWithReference 依赖之；另有统计遥测(getStatistics/resetStatistics)但不影响计算结果。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API MarkerMatchCPU {
public:
    static constexpr const char* kLogTag = "08-MarkerMatchCPU";

    explicit MarkerMatchCPU(const MarkerMatchCPUParams& params = {});

    ~MarkerMatchCPU();

    MarkerMatchCPU(const MarkerMatchCPU&) = delete;
    MarkerMatchCPU& operator=(const MarkerMatchCPU&) = delete;

    /**
     * @brief 执行双目标记点匹配
     *
    * @param left_points  左相机亚像素中心点集（已立体矫正�? 
    * @param right_points 右相机亚像素中心点集（已立体矫正�? 
    * @param result       匹配结果
     */
    MarkerMatchCPUResult Execute(const std::vector<cv::Point2f>& left_points,
               const std::vector<cv::Point2f>& right_points);

    /**
     * @brief 预计算右相机参考点（重复匹配场景优化）
     */
    void SetReferencePoints(const std::vector<cv::Point2f>& right_points);

    /**
     * @brief 使用预计算右参考点执行匹配
     */
    MarkerMatchCPUResult MatchWithReference(const std::vector<cv::Point2f>& left_points);

    /**
     * @brief 清除预计算参考点
     */
    void ClearReferencePoints() noexcept;

    bool HasReferencePoints() const noexcept;

    void Warmup(int maxPointCount);
    void Warmup(const WarmupConfig& config);

    void SetParams(const MarkerMatchCPUParams& params);
    const MarkerMatchCPUParams& GetParams() const;

    void Destroy();

    const MarkerMatchStats& GetStatistics() const noexcept;
    void ResetStatistics() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getMarkerMatchCPUInfo();

} // namespace calib