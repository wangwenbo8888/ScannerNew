/**
 * @file epipolar_types.h
 * @brief 极线插值算子族共享类型（EpipolarInterpResult）唯一驻点
 *
 * 2026-09-05 自 epipolar_interp/epipolar_interp_cuda.h 原文抽出（定义逐字未动）：
 * epipolar_interp_dual 复用原版头取 Result 的传递依赖就此解除——dual 仅依赖本头，
 * retired/epipolar_interp 目录整体缺席亦可独立编译；原版算子头同样改 include 本头，
 * 两版类型仍是同一份定义（无重定义风险）。
 */

#pragma once

#include <string>
#include <memory>
#include "common/calib_types.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

// ============================================================================
// EpipolarInterpResult
// ============================================================================

struct EpipolarInterpResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;
    std::shared_ptr<cv::cuda::GpuMat> d_interpPoints;
    std::shared_ptr<cv::cuda::GpuMat> d_interp_line_ids;
    int interpCount = 0;
    // [dual·Scan 专用诊断] 段链接相位（行间簇连通计算：dualScanLinkParent + pointer
    // doubling）的 GPU 耗时（ms）。仅 EpipolarInterpDualCuda Scan 模式且 scan_link_dx>0
    // 时非零；Labeled 路径与其余使用者恒为 0（默认值，行为无变化）。
    double scan_link_ms = 0.0;

    EpipolarInterpResult() = default;
    ~EpipolarInterpResult() = default;

    EpipolarInterpResult(EpipolarInterpResult&&) = default;
    EpipolarInterpResult& operator=(EpipolarInterpResult&&) = default;

    EpipolarInterpResult(const EpipolarInterpResult&) = delete;
    EpipolarInterpResult& operator=(const EpipolarInterpResult&) = delete;
};

} // namespace calib
