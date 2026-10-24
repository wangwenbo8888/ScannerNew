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
    cv::cuda::GpuMat d_binary;       // ① 二值图
    cv::cuda::GpuMat d_structures;   // ② 全量结构图 X（去小噪点）
    cv::cuda::GpuMat d_temp;         // 腐蚀暂存（③④ 复用）
    cv::cuda::GpuMat d_survivor;     // ③ 幸存支（膨胀还原后）
    cv::cuda::GpuMat d_laserRaw;     // ③ 激光原始掩膜（减法输出）
    cv::cuda::GpuMat d_markingRaw;   // ④ 标记点原始掩膜（减法输出）
    cv::cuda::GpuMat d_discard;      // ④ 大噪点还原结果（丢弃）
    cv::cuda::GpuMat d_laserFinal;   // ⑤ 激光掩膜（边缘膨胀后）
    cv::cuda::GpuMat d_markingFinal; // ⑤ 标记点掩膜（边缘膨胀后）
    cv::cuda::GpuMat d_prevzero;     // 【速度优化】大核腐蚀 prevZero 行扫描中转（w*h uint16）
    cv::cuda::GpuMat d_prevone;      // 【速度优化】大核膨胀 prevOne 行扫描中转（w*h uint16）

    cudaEvent_t event_start_;
    cudaEvent_t event_upload_done_;
    cudaEvent_t event_step1_done_;
    cudaEvent_t event_step2_done_;
    cudaEvent_t event_step3_done_;
    cudaEvent_t event_step4_done_;
    cudaEvent_t event_step5_done_;
    cudaEvent_t ev_dbgA_ = nullptr, ev_dbgB_ = nullptr;   // 【诊断】s3/s4 内部拆分
    bool events_created_ = false;

    bool warmed_up_ = false;
    int warmup_rows_ = 0;
    int warmup_cols_ = 0;

    // 【速度优化·二轮】pinned 灰度暂存（pageable 上传有驱动侧同步与staging 开销）
    uint8_t* h_grayPin_ = nullptr;
    size_t grayPinCap_ = 0;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMarkingSeparationParams& params);
    ~Impl();

    void createEvents();
    void destroyEvents();

    // 【速度优化·二轮】CUDA Graph 整管线一次提交（18 个内核/帧 → 1 次 graphLaunch，
    // 消除逐核驱动提交开销）。JMW_SEPTIME 诊断模式走原路径（事件不可捕获）。
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    // 【速度优化】设备输入路径专用图（无 H2D 上传节点——输入已在显存，
    // 免除每帧重复上传；D2D 入 d_inputBuffer 在图外发射，源指针可变）
    cudaGraph_t graphDev_ = nullptr;
    cudaGraphExec_t graphExecDev_ = nullptr;

    LaserMarkingSeparationResult separate(const cv::Mat& grayImage,
                                          cv::cuda::Stream& stream);

    // 【速度优化】GpuMat 直输（生产：帧已上传供 steger 消费——免二次 H2D）
    LaserMarkingSeparationResult separateDev(const cv::cuda::GpuMat& d_grayImage,
                                             cv::cuda::Stream& stream);

    void setParams(const LaserMarkingSeparationParams& params);
    void warmup(int rows, int cols);

    void executePipeline(cv::cuda::Stream& stream, MaskSeparationTimings& timings,
                         bool recordEvents = true);

    const LaserMarkingSeparationParams& getParams() const { return params_; }
};

} // namespace calib
