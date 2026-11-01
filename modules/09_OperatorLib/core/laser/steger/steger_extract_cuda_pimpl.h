/**
 * @file steger_extract_cuda_pimpl.h
 * @brief Steger激光中心亚像素提取算子 - 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>   // 261018：float2 成员类型需无条件可见（原仅 NDEBUG 下引入）
#include <memory>
#include <atomic>
#include <vector>
#include <cstddef>

#include "steger_extract_cuda.h"

namespace calib {

struct SubpixelPoint {
    int label;
    float px;
    float py;
};

struct StegerExtractorCUDA::Impl {
    StegerParams params_;
    int actualKernelSize_ = 0;

    thrust::device_vector<float> d_g_kernel_;
    thrust::device_vector<float> d_gx_kernel_;
    thrust::device_vector<float> d_gxx_kernel_;

    cv::cuda::GpuMat d_ix_;
    cv::cuda::GpuMat d_iy_;
    cv::cuda::GpuMat d_ixx_;
    cv::cuda::GpuMat d_iyy_;
    cv::cuda::GpuMat d_ixy_;

    cv::cuda::GpuMat d_temp_;
    cv::cuda::GpuMat d_float_;
    cv::cuda::GpuMat d_int_mask_;

    thrust::device_vector<SubpixelPoint> d_points_;
    thrust::device_vector<int> d_point_count_;

    // 261018 逐帧分配退役：持久 device 缓冲（resize 仅增长，容量跨帧复用；
    // Execute 尾部 Step 6 同步保证复用安全）
    thrust::device_vector<SubpixelPoint> d_valid_points_;
    thrust::device_vector<float2> d_coords_;
    thrust::device_vector<int> d_labels_buf_;

    // 261018 pinned host 回读缓冲（pageable cudaMemcpyAsync 退化同步暂存拷贝）
    int* h_pinned_count_ = nullptr;
    SubpixelPoint* h_pinned_points_ = nullptr;
    size_t h_pinned_points_cap_ = 0;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;
    int old_device_id_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const StegerParams& params);
    ~Impl() = default;

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         cv::cuda::Stream& stream);
    StegerResult extractFlat(const cv::cuda::GpuMat& d_grayImage,
                             const cv::cuda::GpuMat& d_binaryMask,
                             cv::cuda::Stream& stream);
    void Destroy();
    void Warmup(int rows, int cols);
    void SetParams(const StegerParams& params);
    const StegerParams& GetParams() const { return params_; }

    void buildGaussianKernels();
    int computeKernelSize() const;
};

} // namespace calib
