/**
 * @file flat_center_pimpl.h
 * @brief 平顶激光中心提取算子（flat_center）- 内部桥接头文件
 *
 * 声明 struct FlatCenterExtractor::Impl 完整结构；仅供
 * flat_center.cpp / flat_center_impl.cu 使用（照 steger_fast_pimpl.h 惯例）。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <atomic>

#include "flat_center.h"

namespace calib {

struct FlatCenterExtractor::Impl {
    static constexpr int kLabelBins = 4096;   // K2 hist bins（照 steger 硬编码常量；
                                              // FlatCenterParams.maxLabels 仅契约兼容）

    FlatCenterParams params_;

    // —— 设备缓冲（ensureBuffers 按 capacity 分配，grow-only）——
    float2* d_coords_ = nullptr;   // 源序出点 (x, cy)
    int*     d_ids_ = nullptr;     // 源序线编号
    int capacity_ = 0;             // = rows×cols（硬上界 ~786k 段/帧 ≪ 3.1M）

    int* d_stats_ = nullptr;       // 连续 7 槽：[0]=count、[1..6]=六门 drop 计数
                                   // （minRun/contrast/bright/edge/gate/width）——同批一次 D2H
    int* d_hist_ = nullptr;        // ByLabel：kLabelBins bins
    int* d_base_ = nullptr;        // ByLabel：逐 label 基址
    int* d_cursor_ = nullptr;      // ByLabel：scatter 游标
    float2* d_gcoords_ = nullptr;  // ByLabel：分组后输出（scatter 目的，独立于源序缓冲）
    int* d_gids_ = nullptr;
    float* d_gsmooth_ = nullptr;   // 【flat 优化 C】预平滑灰度（row-major float，
                                   // 逐位同 ReadGS(smooth3)；K1 段内多趟重读免
                                   // 3 抽头放大——JMW_FLATPRE=0 关闭走原路径 A/B）

    // —— 宿主镜像 ——
    int* h_stats_ = nullptr;       // pinned 7 槽（照 steger cudaHostAlloc 惯例）
    int h_hist_[kLabelBins] = {0};
    int h_base_[kLabelBins] = {0};
    float2* h_coords_ = nullptr;   // collectHostPoints_ / ByLabel host 构建缓冲（cap 宽）
    int* h_ids_ = nullptr;

    // —— bench（JMW_FLATTIME=1 时记录；零点早退帧清零防残留）——
    cudaEvent_t ev_start_ = nullptr, ev_scan_ = nullptr, ev_end_ = nullptr;
    bool events_ready_ = false;
    float bench_ms_[3] = {0.f, 0.f, 0.f};   // 0=Total 1=Scan 2=Group/Out

    // 【flat 优化·宏打包】[PreSmooth+K1] 免改参数图（全参数跨帧稳定：灰度/掩膜
    // 为每 lane 固定缓冲指针、模板/规模常量）——每帧 2 发射→1 图发射。
    // JMW_FLATGRAPH=0 关闭（对照）；SetParams 触发重建
    cudaGraph_t flatGraph_ = nullptr;
    cudaGraphExec_t flatGraphExec_ = nullptr;

    bool bench_ = false;
    bool collectHostPoints_ = false;   // 默认 false（设计 §0：默认只出 GPU 视图）
    bool warmed_up_ = false;
    int warmup_rows_ = 0, warmup_cols_ = 0;
    int old_device_id_ = 0;

    // Flat 快路输出视图：包 d_coords_/d_ids_ 原始设备内存（capacity 宽），每帧取
    // [0,N) ROI 返回（帧内消费契约同 steger d_out_cp_/d_out_pl_ 注记）。
    // ByLabel 用 d_gcoords_/d_gids_ 同款视图（d_out_gc_/d_out_gl_）。
    cv::cuda::GpuMat d_out_cp_, d_out_pl_, d_out_gc_, d_out_gl_;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const FlatCenterParams& params);
    ~Impl();

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         bool maskIs8U,
                         cv::cuda::Stream& stream);          // ByLabel 主路径
    StegerResult extractFlat(const cv::cuda::GpuMat& d_grayImage,
                              const cv::cuda::GpuMat& d_binaryMask,
                              cv::cuda::Stream& stream);       // Flat 快路
    /// 【flat 优化·合并唤醒】提交段（不同步；容量宽临时 ROI）＋收尾段（调用方
    /// 同步流后读 stats 重建 [0,N) 视图）——extractFlat＝两段直连
    StegerResult extractFlatSubmit(const cv::cuda::GpuMat& d_grayImage,
                                    const cv::cuda::GpuMat& d_binaryMask,
                                    cv::cuda::Stream& stream);
    void finishFlat(StegerResult& result);
    void Destroy();
    void Warmup(int rows, int cols);
    void SetParams(const FlatCenterParams& params);
    const FlatCenterParams& GetParams() const { return params_; }

    void ensureBuffers(int rows, int cols);
    void createEvents();
    void freeAll();
};

} // namespace calib
