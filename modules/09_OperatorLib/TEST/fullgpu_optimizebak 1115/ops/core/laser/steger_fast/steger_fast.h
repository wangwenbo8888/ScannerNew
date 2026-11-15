/**
 * @file steger_fast.h
 * @brief Steger激光中心亚像素提取算子（性能优化版）- 公开头文件
 *
 * 与原版 StegerExtractorCUDA（retired/steger/）同契约、同精度
 * （复用 StegerParams / StegerResult / GroupMode——2026-09-04 起三类型唯一驻点为
 * common/steger_types.h，本头不再 include 原版头，可独立使用；导数与点坐标逐位一致）。
 *
 * 优化点（对照原版，见 steger_fast_impl.cu 文件头）：
 *   - 行级 3 输出扇出 kernel（uchar 直读，5 次行卷积 → 1 次 3 输出）
 *   - 列级 5 输出扇出 kernel（5 次列卷积 → 1 次）
 *   - 计数直写分组替代 stable_sort（直方图+前缀和+scatter）
 *   - 单次 D2H + D2D 组装 CPU/GPU 双输出（替代 D2H→H2D 往返）
 *   - 缓冲/事件预分配复用，全部操作挂调用方 stream（消除默认流与每帧 cudaMalloc）
 */

#pragma once

#include <memory>
#include "common/steger_types.h"   // GroupMode/StegerParams/StegerResult（不再依赖原版头）
#include "common/scanner_api.h"    // SCANNER_API（原经原版头传递，解耦后自带）
#include "common/version.h"        // OperatorInfo（同上）

namespace calib {

struct WarmupConfig;

class SCANNER_API StegerExtractorFast {
public:
    static constexpr const char* kLogTag = "10-StegerExtractorFast";

    explicit StegerExtractorFast(const StegerParams& params = {});
    ~StegerExtractorFast();

    StegerExtractorFast(const StegerExtractorFast&) = delete;
    StegerExtractorFast& operator=(const StegerExtractorFast&) = delete;

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask,
                         cv::cuda::Stream& stream);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_labeledMask);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         cv::cuda::Stream& stream,
                         GroupMode groupMode);

    StegerResult Execute(const cv::cuda::GpuMat& d_grayImage,
                         const cv::cuda::GpuMat& d_mask,
                         GroupMode groupMode);

    void Destroy();
    void Warmup(int rows, int cols);
    void Warmup(const WarmupConfig& config);
    void SetParams(const StegerParams& params);
    const StegerParams& GetParams() const;

    /// 【速度优化】控制是否收集 host 侧 centerPoints（h_pts_ D2H＋CPU 拷贝）。
    /// false=跳过下载与同步，下游只消费 GPU 视图（d_centerPoints/d_line_ids）。
    /// 默认 true（向后兼容）；dump/诊断帧须开。
    void SetCollectHostPoints(bool collect);

    /// 【测速工具】最近一次 Execute 的分步耗时（毫秒，共 7 项：
    /// 0=Total 1=Row3 2=Col5 3=Hessian 4=Dedup 5=Group 6=Out）。
    /// 需构造前置 JMW_STEGTIME=1（bench 模式）；否则全零
    void LastBenchMs(float out[7]) const;

    /// 【批0·测速工具】最近一次 Execute 的去重链细分耗时（毫秒，共 5 项：
    /// 0=D1排序+gather 1=D2+D3查找 2=D4传播 3=D5排序2 4=D6段归约）。
    /// 需 JMW_STEGTIME=1；未跑去重时全零。注意 Group 段已改净口径（不含 Dedup）
    void LastDedupMs(float out[5]) const;

    private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getStegerFastInfo();

} // namespace calib
