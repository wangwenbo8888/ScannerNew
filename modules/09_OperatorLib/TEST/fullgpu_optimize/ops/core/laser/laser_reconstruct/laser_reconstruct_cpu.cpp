/**
 * @file laser_reconstruct_cpu.cpp
 * @brief 激光线三维重建 CPU 算子 - 实现（CUDA 版逐行机械移植）
 *
 * 移植基准：laser_reconstruct_cuda_impl.cu
 *   kernelComputeDisparity（:41-50）＋ kernelReconstruct3D（:52-89）＋
 *   CUB DeviceSelect::Flagged 紧凑输出（顺序保持）。
 * 全 float 运算、同运算序（MSVC 默认不 FMA 收缩，与 CUDA 核存在 ULP 级微差，
 * 对拍口径 |Δ|≤1e-3×max(1,|z|) 见测试）。
 */

#include "laser_reconstruct_cpu.h"
#include "common/calib_logging.h"

namespace calib {

LaserReconstructCpu::LaserReconstructCpu(const LaserReconstructParams& params)
    : params_(params) {
    params_.validate();
}

int LaserReconstructCpu::ReconstructInto(const cv::Point2f* left, const cv::Point2f* right,
                                         const int* lineIds, int count, const cv::Mat& Q,
                                         float minDepth, float maxDepth,
                                         float* outXyz, int* outIds, int cap) {
    float qf[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            qf[i * 4 + j] = static_cast<float>(Q.at<double>(i, j));

    int n = 0;
    for (int idx = 0; idx < count; ++idx) {
        const float x = left[idx].x;
        const float y = left[idx].y;
        const float d = left[idx].x - right[idx].x;   // kernelComputeDisparity

        const float W = qf[14] * d + qf[15];
        if (W <= 0.0f) continue;

        const float X = (qf[0] * x + qf[3]) / W;
        const float Y = (qf[5] * y + qf[7]) / W;
        const float Z = qf[11] / W;
        if (Z < minDepth || Z > maxDepth) continue;

        if (n >= cap) break;   // 定容截断（调用方记账）
        outXyz[3 * n + 0] = X;
        outXyz[3 * n + 1] = Y;
        outXyz[3 * n + 2] = Z;
        outIds[n] = lineIds[idx];
        ++n;
    }
    return n;
}

LaserReconstructCpuResult LaserReconstructCpu::Execute(
    const std::vector<cv::Point2f>& matchedLeft,
    const std::vector<cv::Point2f>& matchedRight,
    const std::vector<int>& matchedLineIds,
    const cv::Mat& Q) {

    LaserReconstructCpuResult result;

    const int count = static_cast<int>(matchedLeft.size());
    if (matchedRight.size() != matchedLeft.size() || matchedLineIds.size() != matchedLeft.size()) {
        result.success = false;
        result.message = "Input size mismatch (left/right/lineIds)";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
        return result;
    }
    if (Q.empty() || Q.rows != 4 || Q.cols != 4 || Q.type() != CV_64F) {
        result.success = false;
        result.message = "Q must be 4x4 CV_64F";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
        return result;
    }

    if (count == 0) {
        result.success = true;
        result.message = "Empty input, no reconstruction";
        result.validCount = 0;
        result.totalInput = 0;
        return result;
    }

    try {
        result.points3d.reserve(static_cast<size_t>(count));
        result.validLineIds.reserve(static_cast<size_t>(count));

        float qf[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                qf[i * 4 + j] = static_cast<float>(Q.at<double>(i, j));

        for (int idx = 0; idx < count; ++idx) {
            const float x = matchedLeft[idx].x;
            const float y = matchedLeft[idx].y;
            const float d = matchedLeft[idx].x - matchedRight[idx].x;

            const float W = qf[14] * d + qf[15];
            if (W <= 0.0f) continue;

            const float X = (qf[0] * x + qf[3]) / W;
            const float Y = (qf[5] * y + qf[7]) / W;
            const float Z = qf[11] / W;
            if (Z < params_.minDepth || Z > params_.maxDepth) continue;

            result.points3d.emplace_back(X, Y, Z);
            result.validLineIds.push_back(matchedLineIds[idx]);
        }

        result.success = true;
        result.message = "Success";
        result.validCount = static_cast<int>(result.points3d.size());
        result.totalInput = count;
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        result.points3d.clear();
        result.validLineIds.clear();
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    }

    return result;
}

void LaserReconstructCpu::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }
    warmedUp_ = true;
}

void LaserReconstructCpu::SetParams(const LaserReconstructParams& params) {
    params.validate();
    params_ = params;
    warmedUp_ = false;
}

void LaserReconstructCpu::Destroy() {
    warmedUp_ = false;
}

} // namespace calib
