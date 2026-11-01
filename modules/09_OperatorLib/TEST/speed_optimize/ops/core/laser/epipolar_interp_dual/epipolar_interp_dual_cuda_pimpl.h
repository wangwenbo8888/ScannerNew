/**
 * @file epipolar_interp_dual_cuda_pimpl.h
 * @brief 激光中心点极线插值CUDA算子·双模式版 - 内部桥接头文件
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <string>

#include <cuda_runtime.h>

#include "epipolar_interp_dual_cuda.h"

namespace calib {

struct EpipolarInterpDualCuda::Impl {
    static constexpr int SCAN_MAX_ROWS = 8192;     // 扫描模式固定行网格（同 match 算子约定）
    static constexpr int SCAN_ROW_PTS_CAP = 256;   // 每行窗口内点收集上限（smem 2KB）
    static constexpr int SCAN_SLOTS = 32;          // 每行输出簇上限

    EpipolarInterpDualParams params_;

    // ── 排序（Labeled: 64bit (fid,y)；Scan: 32bit y）──
    cv::cuda::GpuMat d_keys_in_, d_keys_out_;      // CV_64FC1
    cv::cuda::GpuMat d_ykeys_in_, d_ykeys_out_;    // CV_32SC1
    cv::cuda::GpuMat d_idx_in_, d_idx_out_;
    cv::cuda::GpuMat d_sorted_pts_, d_sorted_fids_;
    void* d_radix_temp_ = nullptr; size_t radix_temp_size_ = 0; int radix_cap_ = 0;

    // ── Labeled: 线号表 ──
    cv::cuda::GpuMat d_rl_flags_, d_rl_ranks_;
    cv::cuda::GpuMat d_line_begin_, d_line_ids_map_;
    void* d_scan_temp_ = nullptr; size_t scan_temp_size_ = 0; int scan_cap_ = 0;

    // ── 元信息 {num_lines/umin/umax 或 id_min/id_max} ──
    cv::cuda::GpuMat d_meta_;
    int* h_meta_pinned_ = nullptr;

    // ── Labeled 插值输出（temp+compact）──
    cv::cuda::GpuMat d_flags_, d_temp_interp_, d_temp_fids_;
    cv::cuda::GpuMat d_output_, d_output_fids_, d_output_count_;
    int lines_cap_ = 0;

    // ── Scan 插值输出（[row*SCAN_SLOTS + slot] 布局 + compact）──
    cv::cuda::GpuMat d_scan_flags_, d_scan_out_pts_, d_scan_out_fids_;
    cv::cuda::GpuMat d_scan_cpts_, d_scan_cfids_, d_scan_ccount_;

    // ── Scan 段链接（行间簇合并 → 段 ID，pointer doubling）──
    cv::cuda::GpuMat d_link_parent_, d_link_label_;

    // 【刀二C】Scan 行桶族缓冲（hist 自清洁）
    cv::cuda::GpuMat d_rowb_hist_, d_rowb_start_, d_rowb_cursor_;
    cv::cuda::GpuMat d_rowb_cnt_, d_rowb_cbase_;

    void* d_cub_temp_ = nullptr; size_t cub_temp_size_ = 0; int cub_cap_ = 0;

    int capacity_ = 0;
    bool warmed_up_ = false;
    int warmup_count_ = 0;
    int old_device_id_ = 0;

    // 【测速工具】最近一次 Execute 的阶段耗时串（gpu_stress 聚合用）
    std::string last_breakdown_;
    // 【速度优化】段链接计时事件持久化（原每圈 create/destroy）
    cudaEvent_t ev_link_a_ = nullptr, ev_link_b_ = nullptr;
    // 【E4】ExecuteDeferred 记录本帧流——FinishInterp 的链接内核用（同帧同流）
    cudaStream_t lastCs_ = nullptr;
    // 【WDDM 滴流修复·2026-10-23】延迟路径链接内核改提交段按容量发射（藏进排水
    // 等待）——FinishInterp 只补视图；true＝本帧链接已在飞（消费即清）
    bool lastLinkAtCap_ = false;

    // 【G1·石头二】ExecuteDeferred 整段 CUDA Graph（JMW_1BGRAPH=1 开，默认关）：
    // 全定容后参数/指针跨帧稳定→一次性捕获（ThreadLocal＋捕获前排空本流——mask_sep
    // 配方）→每帧 1 次 graphLaunch 替代 ~44 次 WDDM 发射（实测税：中位 32.8μs/
    // p90 152μs/次）。任一环失败→graphDisabled_ 永久回退逐发。
    cudaGraph_t deferGraph_ = nullptr;
    cudaGraphExec_t deferGraphExec_ = nullptr;
    bool graphDisabled_ = false;
    int graphWarmCalls_ = 0;   // ≥2 才尝试捕获（首帧摊平 radix/cub temp 惰性分配）

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EpipolarInterpDualParams& params);
    ~Impl();

    EpipolarInterpResult Execute(const cv::cuda::GpuMat& d_points,
                                 const cv::cuda::GpuMat& d_line_ids,
                                 cv::cuda::Stream& stream);

    // 【E4】Scan 延迟完成对（h_meta_pinned_[3]＝Scan 计数专用槽，Labeled 用 [0..2]）
    EpipolarInterpResult ExecuteDeferred(const cv::cuda::GpuMat& d_points,
                                         const cv::cuda::GpuMat& d_line_ids,
                                         const int* d_deviceCount,
                                         cv::cuda::Stream& stream);
    void FinishInterp(EpipolarInterpResult& result);
    // Scan 计数后共尾（精确 N 已知）：视图＋段链接＋result 填真值——两路共用
    void finishScanResult(EpipolarInterpResult& result, int h_count, cudaStream_t cs);
    // 【G1】Scan 提交序列（逐发/图捕获同源体）——返回 false＝分配/发射失败
    bool submitScanSequence(cudaStream_t cs, const float2* dPts,
                            const int* d_deviceCount, int cap, int gridN);
    // 【前段提前提交】延迟期访问口（供 v3 前段容量提交；浅引用同帧有效）
    cv::cuda::GpuMat GetCapacityPointsView() const { return d_scan_cpts_; }
    cv::cuda::GpuMat GetCapacityIdsView() const {
        return (params_.scan_link_dx > 0.0f) ? d_link_label_ : d_scan_cfids_;
    }
    const int* GetOutputCountPtr() const { return d_scan_ccount_.ptr<int>(); }
    // 【WDDM 滴流修复】链接相位按容量提交（网格/轮数用 capacity_，尾部线程按
    // 设备输出计数 d_scan_ccount_ 直退）——有效区 [0,count) 输出与精确 N 版
    // 逐位一致（链内核仅写自身槽位、valid 区父链索引恒 < 自身 < count，不读
    // 尾部）；false＝发射失败（调用方回退精确版）
    bool submitLinkAtCapacity(cudaStream_t cs);

    void Warmup(int pointCount);
    void SetParams(const EpipolarInterpDualParams& params);
    const EpipolarInterpDualParams& GetParams() const { return params_; }

    void ensureCapacity(int pointCount);
    void Destroy();
};

} // namespace calib
