#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <atomic>
#include <cassert>

#include "laser_markingpoint_mask_separation_cuda.h"

namespace calib {

struct LaserMarkingSeparationCUDA::Impl {
    LaserMarkingSeparationParams params_;

    cv::cuda::GpuMat d_inputBuffer;
    cv::cuda::GpuMat d_binary;
    cv::cuda::GpuMat d_temp;
    cv::cuda::GpuMat d_step2_mask;
    cv::cuda::GpuMat d_combined;
    cv::cuda::GpuMat d_laser_mask;
    cv::cuda::GpuMat d_marking_raw;
    cv::cuda::GpuMat d_marking_final;

    cudaEvent_t event_start_;
    cudaEvent_t event_upload_done_;
    cudaEvent_t event_step1_done_;
    cudaEvent_t event_step2_done_;
    cudaEvent_t event_step3_done_;
    cudaEvent_t event_step4_done_;
    cudaEvent_t event_step5_done_;
    cudaEvent_t event_step6_done_;
    bool events_created_ = false;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMarkingSeparationParams& params);
    ~Impl();

    void createEvents();
    void destroyEvents();

    void executePipeline(cv::cuda::Stream& stream, MaskSeparationTimings& timings);

    LaserMarkingSeparationResult separate(const cv::Mat& grayImage,
                                          cv::cuda::Stream& stream);

    void setParams(const LaserMarkingSeparationParams& params);
    void warmup(int rows, int cols);

    const LaserMarkingSeparationParams& getParams() const { return params_; }
};

} // namespace calib
