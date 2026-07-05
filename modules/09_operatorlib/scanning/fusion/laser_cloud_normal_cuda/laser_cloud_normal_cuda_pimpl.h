#pragma once

#include "laser_cloud_normal_cuda.h"
#include <cuda_runtime.h>

namespace calib {

struct LaserCloudNormalCuda::Impl {
    LaserCloudNormalCUDAParams params_;

    int* d_stats_ = nullptr;       // [2]: processed, fallback
    int  h_stats_[2] = {0, 0};

    explicit Impl(const LaserCloudNormalCUDAParams& p) : params_(p) {
        params_.validate();
        cudaMalloc(&d_stats_, 2 * sizeof(int));
    }
    ~Impl() { if (d_stats_) cudaFree(d_stats_); }

    LaserCloudNormalCudaResult computeImpl(const LaserCloudFuseDeviceContext& ctx,
                     size_t beginIdx, size_t endIdx,
                     cv::cuda::Stream& stream);
};

} // namespace calib
