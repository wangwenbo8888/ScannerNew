/**
 * @file laser_markingpoint_mask_separation_cuda.h
 * @brief 激光线与标记点掩膜分离算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 功能：从灰度图像中分离激光线掩膜和标记点掩膜
 *
 * 算法流程�? 步）�? * 1. 全局高斯模糊 + 二值化 �?基础掩膜
 * 2. 去除小噪点：腐蚀→膨胀（Open 操作），去除比提取物小的噪点
 * 3. 去除大噪点：�?step2 掩膜上腐蚀→膨胀（更大核），去除提取物，
 *    然后 step2 掩膜减去 step3 掩膜 �?两个提取物的掩膜
 * 4. 获取大提取物（激光线）：�?step3 掩膜上腐蚀→膨胀（更大核），
 *    去除小提取物 �?大提取物掩膜
 * 5. 获取小提取物（标记点）：step3 掩膜减去 step4 掩膜 �?小提取物掩膜
 * 6. 小提取物膨胀：膨胀标记点掩膜，恢复边缘渐变区域
 *
 * 前提：两个提取物（激光线和标记点）宽度有明显差异
 */

#pragma once


#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {


struct WarmupConfig;

/**
 * @brief 激光线与标记点掩膜分离参数
 */
struct LaserMarkingSeparationParams {
    int gaussianSize = 5;
    int threshold = 80;
    int spotThreshold = 0;            ///< 纯点图模式阈值（0=关闭走线点分离形态学；
                                      ///<  >0=A 模式纯补光图直接阈值化出标记点掩膜
                                      ///<  ——激光掩膜空。2026-09-01 还账收编）

    int step2_erodeSize = 3;       ///< �?步腐蚀核大小（奇数，去小噪点）
    int step2_dilateSize = 3;      ///< �?步膨胀核大小（奇数，同腐蚀核）

    int step3_erodeSize = 5;
    int step3_dilateSize = 7;
    int step4_erodeSize = 5;
    int step4_dilateSize = 9;

    int step6_dilateSize = 5;      ///< �?步小提取物膨胀核大小（奇数，恢复边缘）

    /**
    * @brief 参数合法性校�? 
    * @throws std::invalid_argument 参数不合�?     */
    void validate() const {
        auto checkOdd = [](int v, const char* name) {
            if (v < 1 || v % 2 == 0)
                throw std::invalid_argument(
                    std::string("LaserMarkingSeparationParams::") + name +
                    " must be positive odd, got " + std::to_string(v));
        };
        checkOdd(gaussianSize, "gaussianSize");
        if (threshold < 0 || threshold > 255)
            throw std::invalid_argument("threshold must be [0, 255]");
        checkOdd(step2_erodeSize, "step2_erodeSize");
        checkOdd(step2_dilateSize, "step2_dilateSize");
        checkOdd(step3_erodeSize, "step3_erodeSize");
        checkOdd(step3_dilateSize, "step3_dilateSize");
        checkOdd(step4_erodeSize, "step4_erodeSize");
        checkOdd(step4_dilateSize, "step4_dilateSize");
        checkOdd(step6_dilateSize, "step6_dilateSize");
        if (step3_erodeSize <= step2_erodeSize)
            throw std::invalid_argument(
                "step3_erodeSize must be > step2_erodeSize for larger noise removal");
        if (step4_dilateSize <= step3_dilateSize)
            throw std::invalid_argument(
                "step4_dilateSize must be > step3_dilateSize to include edge gradients");
    }

    nlohmann::json toJson() const {
        return {
            {"gaussianSize", gaussianSize},
            {"threshold", threshold},
            {"step2_erodeSize", step2_erodeSize},
            {"step2_dilateSize", step2_dilateSize},
            {"step3_erodeSize", step3_erodeSize},
            {"step3_dilateSize", step3_dilateSize},
            {"step4_erodeSize", step4_erodeSize},
            {"step4_dilateSize", step4_dilateSize},
            {"step6_dilateSize", step6_dilateSize}
        };
    }

    static LaserMarkingSeparationParams fromJson(const nlohmann::json& j) {
        LaserMarkingSeparationParams p;
        if (j.contains("gaussianSize"))    p.gaussianSize    = j.at("gaussianSize").get<int>();
        if (j.contains("threshold"))       p.threshold       = j.at("threshold").get<int>();
        if (j.contains("step2_erodeSize")) p.step2_erodeSize = j.at("step2_erodeSize").get<int>();
        if (j.contains("step2_dilateSize"))p.step2_dilateSize= j.at("step2_dilateSize").get<int>();
        if (j.contains("step3_erodeSize")) p.step3_erodeSize = j.at("step3_erodeSize").get<int>();
        if (j.contains("step3_dilateSize"))p.step3_dilateSize= j.at("step3_dilateSize").get<int>();
        if (j.contains("step4_erodeSize")) p.step4_erodeSize = j.at("step4_erodeSize").get<int>();
        if (j.contains("step4_dilateSize"))p.step4_dilateSize= j.at("step4_dilateSize").get<int>();
        if (j.contains("step6_dilateSize"))p.step6_dilateSize= j.at("step6_dilateSize").get<int>();
        p.validate();
        return p;
    }
};

/**
 * @brief 激光线与标记点掩膜分离结果
 */
struct MaskSeparationTimings {
    double step1_gaussian_threshold_ms = 0.0;
    double step2_remove_small_noise_ms = 0.0;
    double step3_remove_large_noise_ms = 0.0;
    double step4_extract_laser_ms = 0.0;
    double step5_extract_marking_ms = 0.0;
    double step6_dilate_marking_ms = 0.0;
    double total_pipeline_ms = 0.0;
    double upload_ms = 0.0;
};

struct LaserMarkingSeparationResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::shared_ptr<cv::cuda::GpuMat> d_laserMask;       ///< 激光线（大提取物）掩膜
    std::shared_ptr<cv::cuda::GpuMat> d_markingPointMask; ///< 标记点（小提取物）掩膜
    std::shared_ptr<cv::cuda::GpuMat> d_combinedMask;     ///< 两者合并掩膜（step3 输出）
    MaskSeparationTimings timings;

    LaserMarkingSeparationResult() = default;
    ~LaserMarkingSeparationResult() = default;

    LaserMarkingSeparationResult(LaserMarkingSeparationResult&&) = default;
    LaserMarkingSeparationResult& operator=(LaserMarkingSeparationResult&&) = default;

    LaserMarkingSeparationResult(const LaserMarkingSeparationResult&) = delete;
    LaserMarkingSeparationResult& operator=(const LaserMarkingSeparationResult&) = delete;
};

/**
 * @brief 激光线与标记点掩膜分离算子（CUDA 实现�? *
 * 使用 pImpl 模式隔离 CUDA 类型�? * 线程安全约束�?MaskExtractCUDA：非线程安全，不得并发调用�? */
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态
// 说明: Impl 仅持有每调用重置的 GPU 暂存缓冲；SetParams 缓存形态学核大小/阈值等只读配置，无跨调用累积。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================
class SCANNER_API LaserMarkingSeparationCUDA {
public:
    static constexpr const char* kLogTag = "LaserMarkingSeparationCUDA";

    explicit LaserMarkingSeparationCUDA(const LaserMarkingSeparationParams& params = {});
    ~LaserMarkingSeparationCUDA();

    LaserMarkingSeparationCUDA(const LaserMarkingSeparationCUDA&) = delete;
    LaserMarkingSeparationCUDA& operator=(const LaserMarkingSeparationCUDA&) = delete;

    /**
    * @brief 分离激光线和标记点掩膜（主接口�? 
    * @param grayImage Host 端灰度图像（CV_8UC1�? 
    * @param stream CUDA �? 
    * @return LaserMarkingSeparationResult
     */
    LaserMarkingSeparationResult Execute(const cv::Mat& grayImage,
                                          cv::cuda::Stream& stream);

    LaserMarkingSeparationResult Execute(const cv::Mat& grayImage);

    void Destroy();

    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);

    void SetParams(const LaserMarkingSeparationParams& params);
    const LaserMarkingSeparationParams& GetParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMarkingSeparationCUDAInfo();

} // namespace calib