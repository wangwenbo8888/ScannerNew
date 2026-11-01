/**
 * @file laser_markingpoint_mask_separation_cuda.h
 * @brief 激光线与标记点掩膜分离算子 - 公开头文件（�?C++，不�?CUDA 类型�? *
 * 功能：同帧掩膜分离——激光线（细，真实宽 4-7px）与标记点（宽，真实 ~35px）
 * 按宽度域分离，两路二值掩膜输出。
 *
 * 算法流程（细激光形态 v2，2026-10-04 重设计）：
 * 1. 固定 5x5 高斯＋二值化（threshold）
 * 2. 腐蚀(noiseErodeSize)：去小噪点，得全量结构图 X（各结构 -1px/边）
 * 3. 腐蚀(laserKillSize) → 融合[膨胀(kill+2)+减法]：
 *    laserRaw = X − 还原幸存者（激光线死透落入）；survivor = 标记点+中/大噪点
 *    （还原核=kill+2 恰好补平第 2 步的 -2px，第 4 步阈值语义 = 真实宽度）
 * 4. 腐蚀(markerKillSize) → 融合[膨胀(kill+2)+减法]：
 *    markingRaw = survivor − 还原幸存者（标记点死透落入）；大噪点幸存 → 丢弃
 * 5. 边缘保留膨胀：laser = dilate(laserEdgeDilateSize)、
 *    marking = dilate(markingEdgeDilateSize)（供下游亚像素边缘提取）
 *
 * 减法 = top-hat（掩膜 − 膨胀(腐蚀(掩膜))）：还原膨胀是必要组成，
 * 纯腐蚀相减会残留幸存者边缘环污染输出。
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
    int threshold = 80;              ///< 二值化阈值 [0,255]（高斯平滑后灰度标度）

    int noiseErodeSize = 3;          ///< 去小噪点腐蚀核（杀真宽 < 本值）
    int laserKillSize = 9;           ///< 杀激光线腐蚀核（激光真实 4-7px）
    int markerKillSize = 37;         ///< 杀标记点腐蚀核（标记点真实 ~35px，余量 2px；
                                     ///<  >=37px 结构视为大噪点丢弃）

    int laserEdgeDilateSize = 9;     ///< 激光掩膜边缘保留膨胀（净外扩 ~3px/边）
    int markingEdgeDilateSize = 7;   ///< 标记点掩膜边缘保留膨胀（净外扩 ~3px/边）

    int kernelShape = 1;             ///< 0=方形核 1=圆形磁盘核（默认圆形——圆斑宽度
                                     ///<  语义精确：杀/留判定＝结构宽 vs 核宽；方形核对
                                     ///<  圆斑实际杀宽＝核宽×√2，分档语义畸变）

    /// 调试开关：结果额外携带分步中间掩膜（GPU 克隆；生产保持 false）
    bool dumpAllSteps = false;

    /// 261028 确定性开关：禁用 CUDA Graph（走 legacy 逐核路径）。
    /// CUDA Graph 在多会话场景下捕获不确定（conic 8/帧 vs 0-1/帧）。
    bool useCudaGraph = true;

    void validate() const {
        if (threshold < 0 || threshold > 255)
            throw std::invalid_argument(
                "LaserMarkingSeparationParams::threshold must be [0, 255], got "
                + std::to_string(threshold));
        constexpr int kMaxKernel = 63;   // CUDA kernel 模板上限 KHALF=31
        auto checkKernel = [&](int v, const char* name) {
            if (v < 1 || v % 2 == 0 || v > kMaxKernel)
                throw std::invalid_argument(
                    std::string("LaserMarkingSeparationParams::") + name
                    + " must be positive odd <= 63, got " + std::to_string(v));
        };
        checkKernel(noiseErodeSize, "noiseErodeSize");
        // laserKill/markerKill 内部还原核 = +2，不得超模板上限 → 上限 61
        auto checkKill = [&](int v, const char* name) {
            if (v < 1 || v % 2 == 0 || v > 61)
                throw std::invalid_argument(
                    std::string("LaserMarkingSeparationParams::") + name
                    + " must be positive odd <= 61 (restore kernel = +2 vs "
                      "template limit 63), got " + std::to_string(v));
        };
        checkKill(laserKillSize, "laserKillSize");
        checkKill(markerKillSize, "markerKillSize");
        checkKernel(laserEdgeDilateSize, "laserEdgeDilateSize");
        checkKernel(markingEdgeDilateSize, "markingEdgeDilateSize");
        if (kernelShape != 0 && kernelShape != 1)
            throw std::invalid_argument(
                "LaserMarkingSeparationParams::kernelShape must be 0 or 1, got "
                + std::to_string(kernelShape));
        // 注：laserKill 允许 <= noiseErode（laserKill 作用在 ② 压过的域上，
        // 原始宽度激光带 = [noiseErode, laserKill + noiseErode - 1)——下限抬升式
        // 编码合法，如 ②=7/③=5 ＝ 原始带宽 [7,11)）
        if (markerKillSize <= laserKillSize)
            throw std::invalid_argument(
                "LaserMarkingSeparationParams::markerKillSize must be > "
                "laserKillSize");
    }

    nlohmann::json toJson() const {
        return {
            {"threshold", threshold},
            {"noiseErodeSize", noiseErodeSize},
            {"laserKillSize", laserKillSize},
            {"markerKillSize", markerKillSize},
            {"laserEdgeDilateSize", laserEdgeDilateSize},
            {"markingEdgeDilateSize", markingEdgeDilateSize},
            {"kernelShape", kernelShape},
            {"dumpAllSteps", dumpAllSteps}
        };
    }

    static LaserMarkingSeparationParams fromJson(const nlohmann::json& j) {
        LaserMarkingSeparationParams p;
        if (j.contains("threshold"))              p.threshold              = j.at("threshold").get<int>();
        if (j.contains("noiseErodeSize"))         p.noiseErodeSize         = j.at("noiseErodeSize").get<int>();
        if (j.contains("laserKillSize"))          p.laserKillSize          = j.at("laserKillSize").get<int>();
        if (j.contains("markerKillSize"))         p.markerKillSize         = j.at("markerKillSize").get<int>();
        if (j.contains("laserEdgeDilateSize"))    p.laserEdgeDilateSize    = j.at("laserEdgeDilateSize").get<int>();
        if (j.contains("markingEdgeDilateSize"))  p.markingEdgeDilateSize  = j.at("markingEdgeDilateSize").get<int>();
        if (j.contains("kernelShape"))            p.kernelShape            = j.at("kernelShape").get<int>();
        if (j.contains("dumpAllSteps"))           p.dumpAllSteps           = j.at("dumpAllSteps").get<bool>();
        p.validate();
        return p;
    }
};

/**
 * @brief 激光线与标记点掩膜分离结果
 */
struct MaskSeparationTimings {
    double step1_gaussian_threshold_ms = 0.0;
    double step2_noise_erode_ms = 0.0;
    double step3_laser_extract_ms = 0.0;    // 腐蚀 kill + 融合膨胀还原+减法
    double step4_marking_extract_ms = 0.0;  // 同上（标记点级）
    double step5_edge_dilate_ms = 0.0;      // 双掩膜边缘保留膨胀
    double total_pipeline_ms = 0.0;
    double upload_ms = 0.0;
};

struct LaserMarkingSeparationResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    std::shared_ptr<cv::cuda::GpuMat> d_laserMask;       ///< 激光线掩膜
    std::shared_ptr<cv::cuda::GpuMat> d_markingPointMask; ///< 标记点掩膜
    std::shared_ptr<cv::cuda::GpuMat> d_combinedMask;     ///< 全量结构图（步骤②输出 X，去小噪点后全部结构）
    // —— 分步中间掩膜：d_step1_binary 常驻（harness 产物消费）；其余 4 张仅
    //    params.dumpAllSteps=true 时填充（GPU 克隆；生产保持 false）——
    std::shared_ptr<cv::cuda::GpuMat> d_step1_binary;    ///< ① 高斯+阈值二值图（未分割）
    std::shared_ptr<cv::cuda::GpuMat> d_structuresMask;  ///< ② 全量结构图 X（=d_combinedMask 同源）
    std::shared_ptr<cv::cuda::GpuMat> d_laserRaw;        ///< ③ 激光原始掩膜（膨胀前）
    std::shared_ptr<cv::cuda::GpuMat> d_survivorMask;    ///< ③ 幸存支（标记点+中/大噪点）
    std::shared_ptr<cv::cuda::GpuMat> d_markingRaw;      ///< ④ 标记点原始掩膜（膨胀前）
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

    /// 【速度优化】GpuMat 直输：输入帧已在显存（与 steger 同源）——免每帧二次
    /// H2D 上传；逐内核序列与 Host 版一致 → 结果逐位一致
    LaserMarkingSeparationResult Execute(const cv::cuda::GpuMat& d_grayImage,
                                         cv::cuda::Stream& stream);

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