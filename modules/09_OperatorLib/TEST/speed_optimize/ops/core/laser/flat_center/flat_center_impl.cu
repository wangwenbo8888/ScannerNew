/**
 * @file flat_center_impl.cu
 * @brief 平顶激光中心提取算子（flat_center）CUDA 实现 — Task 2 桩版
 *
 * ⚠️ 桩版：Impl 全方法最小定义（解链接）；Task 3 替换 Execute/extractFlat 真体
 * （Flat 路径）＋Task 5 填 ByLabel。设计 §3/§4。
 */

#include "flat_center_pimpl.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>

using namespace calib;

FlatCenterExtractor::Impl::Impl(const FlatCenterParams& params)
    : params_(params)
{
    params_.validate();
    bench_ = (getenv("JMW_FLATTIME") != nullptr);
    if (bench_) createEvents();
}

FlatCenterExtractor::Impl::~Impl() {
    freeAll();
}

void FlatCenterExtractor::Impl::createEvents() {
    if (events_ready_) return;
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_scan_);
    cudaEventCreate(&ev_end_);
    events_ready_ = true;
}

void FlatCenterExtractor::Impl::freeAll() {
    if (d_coords_)  { cudaFree(d_coords_); d_coords_ = nullptr; }
    if (d_ids_)     { cudaFree(d_ids_); d_ids_ = nullptr; }
    if (d_stats_)   { cudaFree(d_stats_); d_stats_ = nullptr; }
    if (d_hist_)    { cudaFree(d_hist_); d_hist_ = nullptr; }
    if (d_base_)    { cudaFree(d_base_); d_base_ = nullptr; }
    if (d_cursor_)  { cudaFree(d_cursor_); d_cursor_ = nullptr; }
    if (d_gcoords_) { cudaFree(d_gcoords_); d_gcoords_ = nullptr; }
    if (d_gids_)    { cudaFree(d_gids_); d_gids_ = nullptr; }
    if (d_gsmooth_) { cudaFree(d_gsmooth_); d_gsmooth_ = nullptr; }   // 【flat 优化 C】
    if (flatGraphExec_) { cudaGraphExecDestroy(flatGraphExec_); flatGraphExec_ = nullptr; }   // 【flat 优化·宏打包】
    if (flatGraph_)     { cudaGraphDestroy(flatGraph_); flatGraph_ = nullptr; }
    if (h_stats_)   { cudaFreeHost(h_stats_); h_stats_ = nullptr; }
    if (h_coords_)  { cudaFreeHost(h_coords_); h_coords_ = nullptr; }
    if (h_ids_)     { cudaFreeHost(h_ids_); h_ids_ = nullptr; }
    if (events_ready_) {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_scan_);
        cudaEventDestroy(ev_end_);
        events_ready_ = false;
    }
    capacity_ = 0;
}

void FlatCenterExtractor::Impl::ensureBuffers(int rows, int cols) {
    const int cap = rows * cols;
    if (cap <= capacity_) return;
    freeAll();
    capacity_ = cap;
    cudaMalloc(&d_coords_, capacity_ * sizeof(float2));
    cudaMalloc(&d_ids_, capacity_ * sizeof(int));
    cudaMalloc(&d_stats_, 7 * sizeof(int));
    cudaMalloc(&d_hist_, kLabelBins * sizeof(int));
    cudaMalloc(&d_base_, kLabelBins * sizeof(int));
    cudaMalloc(&d_cursor_, kLabelBins * sizeof(int));
    cudaMalloc(&d_gcoords_, capacity_ * sizeof(float2));
    cudaMalloc(&d_gids_, capacity_ * sizeof(int));
    cudaMalloc(&d_gsmooth_, (size_t)capacity_ * sizeof(float));   // 【flat 优化 C】预平滑
    cudaMallocHost(&h_stats_, 7 * sizeof(int));
    cudaMallocHost(&h_coords_, capacity_ * sizeof(float2));
    cudaMallocHost(&h_ids_, capacity_ * sizeof(int));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        CALIB_LOG_ERROR("ensureBuffers failed: {}", cudaGetErrorString(err));
        throw std::runtime_error(std::string("[10-FlatCenterExtractor] buffer alloc failed: ")
                                 + cudaGetErrorString(err));
    }
}

// ============================================================================
// CUDA Kernel（设计 §4：单 kernel、1 线程/列、逐段两次行走＋括号重心第三小遍）
// ============================================================================

// —— 平滑灰度读（段端读段外原始灰度＋图像边界钳位；统计遍/找沿遍/括号重心同款）——
__device__ __forceinline__ float ReadGS(const unsigned char* g, size_t step,
                                        int rows, int x, int y, bool s3) {
    const unsigned char* col = g + x;
    if (!s3) return static_cast<float>(col[(size_t)y * step]);
    const int y0 = max(y - 1, 0), y2 = min(y + 1, rows - 1);
    return (col[(size_t)y0 * step] + col[(size_t)y * step] + col[(size_t)y2 * step])
           * (1.f / 3.f);
}

// 【flat 优化 C+D】全图预平滑（逐位同 ReadGS(smooth3)：三抽头＋边界钳位，
// (a+b+c)*(1.f/3.f) 同式逐位一致）＋(0,0,0) 线程顺带清 d_stats_ 7 槽（吸收
// 独立 memsetAsync——本内核无人碰 stats，K1 流序在后见零初值）
__global__ void FlatPreSmoothKernel(const unsigned char* __restrict__ g,
                                    size_t step, int rows, int cols,
                                    float* __restrict__ out,
                                    int* __restrict__ stats7) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 7; ++i) stats7[i] = 0;
    }
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    const unsigned char* col = g + x;
    const int y0 = max(y - 1, 0), y2 = min(y + 1, rows - 1);
    out[(size_t)y * cols + x] =
        (col[(size_t)y0 * step] + col[(size_t)y * step] + col[(size_t)y2 * step])
        * (1.f / 3.f);
}

template <bool MASK8>
__global__ void FlatCenterK1(
    const unsigned char* __restrict__ d_gray, const void* __restrict__ d_mask,
    size_t gray_step, size_t mask_step, int rows, int cols,
    float low_thresh, float half_ratio, float plateau_ratio, int plateau_min_w,
    float min_contrast, float outlier_gate, float max_line_width, int min_run,
    bool smooth3,
    const float* __restrict__ d_gs,   /* 【flat 优化 C】预平滑缓冲（空＝现算） */
    float2* __restrict__ d_coords, int* __restrict__ d_ids,
    int* __restrict__ d_stats /*[0]=count [1..6]=六门 drop*/, int max_points)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols) return;
    // 【速度·列带切分（设计 §4 备选，规则钉死）】1 线程/列在 1400 行图像上仅
    // ~106 warp 全核延迟饥饿（实测 7.8ms/眼）；gridDim.y＝带数（8）：
    //   ① 段归段首所在带；② owner 向下走到段尾（越带无上界）；
    //   ③ 带首行若为上方延续段（maskAt(top−1)==label）跳到段尾不处理
    const int bands = gridDim.y;
    const int bh = (rows + bands - 1) / bands;
    const int top = blockIdx.y * bh;
    if (top >= rows) return;
    const int bot = min(rows, top + bh);
    int* d_count = d_stats;
    int* d_drop  = d_stats + 1;   // [0]minRun [1]contrast [2]bright [3]edge [4]gate [5]width
    // 列内段扫描（Flat：连续 mask>0；ByLabel：连续同 label，label 变化即断段）
    auto maskAt = [&](int y) -> int {
        if (MASK8)
            return reinterpret_cast<const unsigned char*>(
                reinterpret_cast<const char*>(d_mask) + (size_t)y * mask_step)[x] > 0 ? 1 : 0;
        return reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(d_mask) + (size_t)y * mask_step)[x];
    };
    // 【flat 优化 C】统一读口：预平滑缓冲优先（逐位同 ReadGS）；空＝原路径现算
    auto RD = [&](int yy) -> float {
        if (d_gs) return d_gs[(size_t)yy * cols + x];
        return ReadGS(d_gray, gray_step, rows, x, yy, smooth3);
    };
    int y = top;
    while (y < bot) {
        const int label = maskAt(y);
        if (label <= 0) { ++y; continue; }
        if (y == top && top > 0 && maskAt(top - 1) == label) {
            // 规则③：上方延续段（段首在上面的带）——跳到段尾，由 owner 处理
            while (y < rows && maskAt(y) == label) ++y;
            continue;
        }
        const int y0 = y;
        while (y < rows && maskAt(y) == label) ++y;          // 段=[y0, y)，段尾可越带
        const int y1 = y, len = y1 - y0;
        // —— 门1：minRun ——
        if (len < min_run) { atomicAdd(&d_drop[0], 1); continue; }
        // —— 统计遍第一小趟：vmax/yPeak（自上而下严格 > 首个）/vmin ——
        float vmax = -1.f, vmin = 1e9f; int yPeak = y0;
        for (int k = y0; k < y1; ++k) {
            const float v = RD(k);
            if (v > vmax) { vmax = v; yPeak = k; }
            if (v < vmin) vmin = v;
        }
        // —— 门2/3：bright / contrast（后者兼保 Σw>0 防除零）——
        if (vmax < low_thresh) { atomicAdd(&d_drop[2], 1); continue; }
        if (vmax - vmin < min_contrast) { atomicAdd(&d_drop[1], 1); continue; }
        // —— 统计遍第二小趟：pw 对最终 vmax ——
        const float thr_p = plateau_ratio * vmax;
        int pw = 0;
        for (int k = y0; k < y1; ++k)
            if (RD(k) >= thr_p) ++pw;
        float cy;
        float yL = 0.f, yR = 0.f;
        if (pw >= plateau_min_w) {
            // —— 平顶：半高中点。沿搜索界＝图像界 ∩ [yPeak−32, yPeak+33)（不钳段界：
            //    掩膜细缝/段端处越段读原始灰度找沿；±32 窗保证不跨入相邻线）——
            const int lo = max(0, yPeak - 32), hi = min(rows, yPeak + 33);
            // 【链级对拍修正 2026-10-19】半高＝相对背景（vmin＋halfRatio×对比度）——
            // 真实帧背景 ≈110-120（非 0），绝对半高 0.5×vmax 只比背景高 ~12 灰度级，
            // 线稍暗/背景稍亮即找不到沿（edge 弃 30%）；背景基准后合成图（bg=0）
            // 数值不变，现有用例全部兼容。
            const float half = vmin + half_ratio * (vmax - vmin);
            int k = yPeak;
            while (k >= lo && RD(k) >= half) --k;
            if (k < lo) { atomicAdd(&d_drop[3], 1); continue; }                  // edge
            const float g1 = RD(k + 1);
            const float g0 = RD(k);
            yL = (k + 1) - (g1 - half) / (g1 - g0);  // crossing ∈ (k, k+1]
            int j = yPeak;
            while (j < hi && RD(j) >= half) ++j;
            if (j >= hi) { atomicAdd(&d_drop[3], 1); continue; }
            const float h1 = RD(j - 1);
            const float h0 = RD(j);
            yR = (j - 1) + (h1 - half) / (h1 - h0); // crossing ∈ [j-1, j)
            // —— 宽度门（v4.4）：半高全宽超限 → 弃点（0=off 不拦）——
            if (max_line_width > 0.f && (yR - yL) > max_line_width) {
                atomicAdd(&d_drop[5], 1); continue;
            }
            cy = 0.5f * (yL + yR);
        } else {
            // —— 尖峰：去背景重心（vmin≈段内背景＝天然背景扣除，勿"修"）——
            float sw = 0.f, sy = 0.f;
            for (int k = y0; k < y1; ++k) {
                const float w = RD(k) - vmin;
                if (w > 0.f) { sw += w; sy += k * w; }
            }
            if (sw < 1e-6f) { atomicAdd(&d_drop[1], 1); continue; }
            cy = sy / sw;
        }
        // —— 统一自检（v4.3）：平顶＝**括号区间**重心（越缝括号覆盖全线→重心=线心，
        //    细缝不误杀；段内全剖面重心因半剖面偏移会误杀 ⑦）；尖峰＝改比锚点
        //    |yPeak−cy|（cy=段内重心，与重心比恒 0；兼捕侧翼亮斑拉锚构型）——
        float ref;
        if (pw >= plateau_min_w) {
            float sw = 0.f, sy = 0.f;
            const int b0 = max(0, (int)yL), b1 = min(rows - 1, (int)yR + 1);
            for (int k = b0; k <= b1; ++k) {
                const float w = RD(k) - vmin;
                if (w > 0.f) { sw += w; sy += k * w; }
            }
            ref = (sw >= 1e-6f) ? sy / sw : cy;
        } else {
            ref = static_cast<float>(yPeak);
        }
        if (fabsf(cy - ref) > outlier_gate) { atomicAdd(&d_drop[4], 1); continue; } // gate
        // 防御（v4.3）：cy=两沿中点必在括号内，段界防御废除；留图像界＋NaN 防护
        if (!(cy == cy) || cy < -1.f || cy > rows) { atomicAdd(&d_drop[4], 1); continue; }
        // —— 出点 (x, cy)（整数格点口径，同 steger kernel px=x+t·nx）——
        const int idx = atomicAdd(d_count, 1);
        if (idx < max_points) {
            d_coords[idx] = make_float2(static_cast<float>(x), cy);
            d_ids[idx] = label;
        }
    }
}

// ============================================================================
// Flat 快路（设计 §4 extractFlat 主流程）
// ============================================================================

// 【合并唤醒】提交段：不同步（stats D2H 已入队）——extractFlat＝Submit＋
// 同步＋finishFlat；流水线双眼可两次 Submit 共用一次流同步
StegerResult FlatCenterExtractor::Impl::extractFlatSubmit(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    StegerResult result;
    const int rows = d_grayImage.rows, cols = d_grayImage.cols;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    ensureBuffers(rows, cols);
    if (bench_ && !events_ready_) createEvents();
    if (bench_ && events_ready_) cudaEventRecord(ev_start_, cs);

    // 【flat 优化 C+D】smooth3 开启时：全图预平滑一次（段内 3-4 趟重读免 3 抽头
    // 放大；逐位同式）＋顺带清 stats；JMW_FLATPRE=0 关闭走原路径（A/B）。
    // 【flat 优化·宏打包】usePre 且未禁用时 [PreSmooth+K1] 免改参数图一次发射
    //（全参数跨帧稳定；首次捕获前排空本流——WDDM 规则）
    const bool usePre = params_.smooth3
                        && !(getenv("JMW_FLATPRE") && getenv("JMW_FLATPRE")[0] == '0');
    // 【flat 优化·宏打包·裁定默认关】串行口径图化逐位一致且稳定，但多收尾
    // lane 流水线下实测必崩（0xC0000005 ×3/3，与 JMW_V3GRAPH 1/8 崩溃同类
    // ——WDDM 对 lane 流上图敏感，根因未归因）。默认关闭；JMW_FLATGRAPH=1
    // 显式开启（串行/诊断用），崩溃压力测试归因后另行裁决
    static const bool graphOn_ = (getenv("JMW_FLATGRAPH")
                                  && getenv("JMW_FLATGRAPH")[0] == '1');
    const bool useGraph = usePre && graphOn_;
    if (usePre) {
        const dim3 pb(128, 8);
        const dim3 pg(static_cast<unsigned>((cols + 127) / 128),
                      static_cast<unsigned>((rows + 7) / 8));
        int nbandsG = 48;
        if (const char* e = getenv("JMW_FLATBANDS"))
            nbandsG = std::max(1, std::min(64, atoi(e)));
        const dim3 block(256),
            grid((cols + 255) / 256, static_cast<unsigned>(nbandsG));
        if (useGraph) {
            cudaError_t gerr = cudaSuccess;
            if (!flatGraphExec_) {
                stream.waitForCompletion();   // 捕获前必须排空本流（WDDM 驱动崩溃防线）
                gerr = cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
                if (gerr == cudaSuccess) {
                    FlatPreSmoothKernel<<<pg, pb, 0, cs>>>(
                        d_grayImage.ptr<unsigned char>(), d_grayImage.step, rows, cols,
                        d_gsmooth_, d_stats_);
                    FlatCenterK1<true><<<grid, block, 0, cs>>>(
                        d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                        d_grayImage.step, d_binaryMask.step, rows, cols,
                        params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
                        params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
                        params_.maxLineWidth, params_.minRunLength, params_.smooth3,
                        d_gsmooth_, d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】出点容量截断
                    cudaGraph_t g = nullptr;
                    gerr = cudaStreamEndCapture(cs, &g);
                    if (gerr == cudaSuccess)
                        gerr = cudaGraphInstantiate(&flatGraphExec_, g, nullptr, nullptr, 0);
                    if (gerr == cudaSuccess && g) {
                        flatGraph_ = g;
                    } else {
                        if (g) cudaGraphDestroy(g);
                        flatGraphExec_ = nullptr;   // 捕获/实例化失败→回退逐发路径
                    }
                }
            }
            if (flatGraphExec_) {
                gerr = cudaGraphLaunch(flatGraphExec_, cs);
                if (gerr != cudaSuccess) {
                    CALIB_LOG_WARN("[flat] graphLaunch 失败回退逐发: {}",
                                   cudaGetErrorString(gerr));
                    flatGraphExec_ = nullptr;
                }
            }
            if (!flatGraphExec_) {
                // 回退：逐发原路径
                FlatPreSmoothKernel<<<pg, pb, 0, cs>>>(
                    d_grayImage.ptr<unsigned char>(), d_grayImage.step, rows, cols,
                    d_gsmooth_, d_stats_);
                FlatCenterK1<true><<<grid, block, 0, cs>>>(
                    d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                    d_grayImage.step, d_binaryMask.step, rows, cols,
                    params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
                    params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
                    params_.maxLineWidth, params_.minRunLength, params_.smooth3,
                    d_gsmooth_, d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】出点容量截断
            }
        } else {
            FlatPreSmoothKernel<<<pg, pb, 0, cs>>>(
                d_grayImage.ptr<unsigned char>(), d_grayImage.step, rows, cols,
                d_gsmooth_, d_stats_);
            FlatCenterK1<true><<<grid, block, 0, cs>>>(
                d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
                d_grayImage.step, d_binaryMask.step, rows, cols,
                params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
                params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
                params_.maxLineWidth, params_.minRunLength, params_.smooth3,
                d_gsmooth_, d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】出点容量截断
        }
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    } else {
        // 【flat 优化 D】无预平滑路径：清零独立发射（stats 清零）
        cudaMemsetAsync(d_stats_, 0, 7 * sizeof(int), cs);
        int nbands = 48;
        if (const char* e = getenv("JMW_FLATBANDS"))
            nbands = std::max(1, std::min(64, atoi(e)));
        const dim3 block(256),
            grid((cols + 255) / 256, static_cast<unsigned>(nbands));
        FlatCenterK1<true><<<grid, block, 0, cs>>>(
            d_grayImage.ptr<unsigned char>(), d_binaryMask.ptr<unsigned char>(),
            d_grayImage.step, d_binaryMask.step, rows, cols,
            params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
            params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
            params_.maxLineWidth, params_.minRunLength, params_.smooth3,
            static_cast<const float*>(nullptr),
            d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_scan_, cs);

    // d_stats_ 7 槽一次 D2H（pinned）——【合并唤醒】提交段到此为止（不同步）
    cudaMemcpyAsync(h_stats_, d_stats_, 7 * sizeof(int), cudaMemcpyDeviceToHost, cs);
    // 容量宽临时 ROI（N 未定——FinishDeferred 重建 [0,N)）
    d_out_cp_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32FC2, d_coords_);   // 【E5】容量宽视图
    d_out_pl_ = cv::cuda::GpuMat(1, FlatCenterExtractor::kOutCap, CV_32SC1, d_ids_);
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
        d_out_cp_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
        d_out_pl_, cv::Rect(0, 0, FlatCenterExtractor::kOutCap, 1));
    result.totalPointCount = -1;   // 占位——FinishDeferred 填真值
    result.lineCount = 1;
    result.mergedCount = 0;
    result.success = true;   // 【修复】提交级成功（harness submitLaserFront 818 检查
                              // 要求提交期 success=true；N/零点在 FinishDeferred 定）
    return result;
}

void FlatCenterExtractor::Impl::finishFlat(StegerResult& result) {
    result.success = true;   // 【修复】原尾段置位随拆分遗漏——成功路径两分叉通用
    // 【E5】出点容量截断口径：K1 槽位≥kOutCap 不写但计数照加——min(N,kOutCap)
    // 与写入侧一致（超出容量点静默弃——实测 N max≈34k，2×余量下不触发）
    const int N = std::min(h_stats_[0], FlatCenterExtractor::kOutCap);   // 钳位（照 steger:1635）
    int dropSum = 0;
    for (int i = 1; i <= 6; ++i) dropSum += h_stats_[i];

    // —— 零点分叉（设计 §2）：全零掩膜→Warning；全被门弃→Degraded ——
    if (N == 0) {
        result.d_centerPoints.reset();
        result.d_line_ids.reset();
        result.success = true;
        result.totalPointCount = 0;
        result.lineCount = 0;
        result.mergedCount = 0;
        if (dropSum == 0) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "No laser points (mask all zero?)";
        } else {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "flat_center: all runs dropped";
        }
        bench_ms_[0] = bench_ms_[1] = bench_ms_[2] = 0.f;   // 早退帧清零防残留
        return;
    }

    // —— Flat 直通快路：[0,N) ROI 视图（帧内消费契约同 steger d_out_cp_/d_out_pl_）——
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
        d_out_cp_, cv::Rect(0, 0, N, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
        d_out_pl_, cv::Rect(0, 0, N, 1));
    result.totalPointCount = N;
    result.lineCount = 1;         // Flat 单标签
    result.mergedCount = 0;

    if (dropSum > 0) {
        result.qualityFlag = QualityFlag::Degraded;
        result.message = "flat_center: dropped " + std::to_string(dropSum)
            + " runs, reasons=minRun:" + std::to_string(h_stats_[1])
            + " contrast:" + std::to_string(h_stats_[2])
            + " bright:" + std::to_string(h_stats_[3])
            + " edge:" + std::to_string(h_stats_[4])
            + " gate:" + std::to_string(h_stats_[5])
            + " width:" + std::to_string(h_stats_[6]);
        CALIB_LOG_DEBUG("{}", result.message);
    } else {
        result.qualityFlag = QualityFlag::Normal;
        result.message = "Subpixel extraction successful";
    }
}

StegerResult FlatCenterExtractor::Impl::extractFlat(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    StegerResult result = extractFlatSubmit(d_grayImage, d_binaryMask, stream);
    if (bench_ && events_ready_) cudaEventRecord(ev_end_, cs);
    stream.waitForCompletion();
    finishFlat(result);
    if (result.totalPointCount == 0) {
        result.success = true;
        return result;   // 零点早退（原口径：跳过 host 输出与 bench）
    }

    const int N = result.totalPointCount;
    // —— 可选 host 输出（collectHostPoints_=true）——
    if (collectHostPoints_) {
        cudaMemcpyAsync(h_coords_, d_coords_, N * sizeof(float2), cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(h_ids_, d_ids_, N * sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();
        std::vector<cv::Point2f> v(N);
        for (int i = 0; i < N; ++i) v[i] = cv::Point2f(h_coords_[i].x, h_coords_[i].y);
        result.centerPoints.emplace(1, std::move(v));   // Flat 单标签
    }

    result.success = true;

    // —— bench 汇总（JMW_FLATTIME=1 时）——
    if (bench_ && events_ready_) {
        cudaEventSynchronize(ev_end_);
        float t = 0;
        cudaEventElapsedTime(&t, ev_start_, ev_end_);  bench_ms_[0] = t;
        cudaEventElapsedTime(&t, ev_start_, ev_scan_); bench_ms_[1] = t;
        cudaEventElapsedTime(&t, ev_scan_, ev_end_);   bench_ms_[2] = t;
        cudaGetLastError();   // 吞事件非致命错防粘性（照 steger:1972 惯例）
        CALIB_LOG_DEBUG("[BENCH_FLAT] N={}", N);
    }
    return result;
}

// —— ByLabel 分组 kernel（照抄 steger LabelHist/LabelScatter 模式，键改 ids 直读）——
__global__ void FlatLabelHistKernel(const int* __restrict__ ids, int count, int bins,
                                    int* __restrict__ hist)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = ids[i];
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    atomicAdd(&hist[lb], 1);
}
__global__ void FlatLabelScatterKernel(
    const float2* __restrict__ coords, const int* __restrict__ ids, int count, int bins,
    const int* __restrict__ base, int* __restrict__ cursor,
    float2* __restrict__ outCoords, int* __restrict__ outIds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = ids[i];
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    const int slot = atomicAdd(&cursor[lb], 1);
    const int idx = base[lb] + slot;
    outCoords[idx] = coords[i];
    outIds[idx] = lb;
}

// ============================================================================
// ByLabel 主路径（设计 §4：K1<false> 源序出点 → hist/前缀和/scatter 分组）
// ============================================================================

StegerResult FlatCenterExtractor::Impl::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_labeledMask,
    bool maskIs8U,
    cv::cuda::Stream& stream)
{
    (void)maskIs8U;   // 本实现恒走 32S 路径（桥接层已保证类型）
    StegerResult result;
    const int rows = d_grayImage.rows, cols = d_grayImage.cols;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    ensureBuffers(rows, cols);
    if (bench_ && !events_ready_) createEvents();
    cudaMemsetAsync(d_stats_, 0, 7 * sizeof(int), cs);
    if (bench_ && events_ready_) cudaEventRecord(ev_start_, cs);

    {
        const dim3 block(256), grid((cols + 255) / 256, 8);   // 列带×8（设计 §4 备选）
        FlatCenterK1<false><<<grid, block, 0, cs>>>(
            d_grayImage.ptr<unsigned char>(), d_labeledMask.ptr<int>(),
            d_grayImage.step, d_labeledMask.step, rows, cols,
            params_.lowThreshold, params_.halfRatio, params_.plateauRatio,
            params_.plateauMinWidth, params_.minContrast, params_.outlierGatePx,
            params_.maxLineWidth, params_.minRunLength, params_.smooth3,
            static_cast<const float*>(nullptr),   /* 【flat 优化 C】ByLabel 不预平滑 */
            d_coords_, d_ids_, d_stats_, FlatCenterExtractor::kOutCap);   // 【E5】
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("[10-FlatCenterExtractor] kernel launch failed: ")
                             + cudaGetErrorString(err);
            CALIB_LOG_ERROR("{}", result.message);
            return result;
        }
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_scan_, cs);

    cudaMemcpyAsync(h_stats_, d_stats_, 7 * sizeof(int), cudaMemcpyDeviceToHost, cs);
    stream.waitForCompletion();

    const int N = std::min(h_stats_[0], capacity_);
    int dropSum = 0;
    for (int i = 1; i <= 6; ++i) dropSum += h_stats_[i];

    if (N == 0) {
        result.success = true;
        result.totalPointCount = 0;
        result.lineCount = 0;
        result.mergedCount = 0;
        if (dropSum == 0) {
            result.qualityFlag = QualityFlag::Warning;
            result.message = "No laser points (mask all zero?)";
        } else {
            result.qualityFlag = QualityFlag::Degraded;
            result.message = "flat_center: all runs dropped";
        }
        bench_ms_[0] = bench_ms_[1] = bench_ms_[2] = 0.f;
        return result;
    }

    // —— 分组：hist → CPU 前缀和 → scatter（照 steger ByLabel 组装段模式）——
    cudaMemsetAsync(d_hist_, 0, kLabelBins * sizeof(int), cs);
    {
        const int threads = 256, blocks = (N + threads - 1) / threads;
        FlatLabelHistKernel<<<blocks, threads, 0, cs>>>(d_ids_, N, kLabelBins, d_hist_);
    }
    cudaMemcpyAsync(h_hist_, d_hist_, kLabelBins * sizeof(int), cudaMemcpyDeviceToHost, cs);
    stream.waitForCompletion();

    int running = 0, lineCount = 0;
    for (int lb = 0; lb < kLabelBins; ++lb) {
        h_base_[lb] = running;
        running += h_hist_[lb];
        if (h_hist_[lb] > 0) ++lineCount;
    }

    cudaMemcpyAsync(d_base_, h_base_, kLabelBins * sizeof(int), cudaMemcpyHostToDevice, cs);
    cudaMemsetAsync(d_cursor_, 0, kLabelBins * sizeof(int), cs);
    {
        const int threads = 256, blocks = (N + threads - 1) / threads;
        FlatLabelScatterKernel<<<blocks, threads, 0, cs>>>(
            d_coords_, d_ids_, N, kLabelBins, d_base_, d_cursor_, d_gcoords_, d_gids_);
    }
    if (bench_ && events_ready_) cudaEventRecord(ev_end_, cs);

    d_out_gc_ = cv::cuda::GpuMat(1, capacity_, CV_32FC2, d_gcoords_);
    d_out_gl_ = cv::cuda::GpuMat(1, capacity_, CV_32SC1, d_gids_);
    result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(d_out_gc_, cv::Rect(0, 0, N, 1));
    result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(d_out_gl_, cv::Rect(0, 0, N, 1));
    result.totalPointCount = N;
    result.lineCount = lineCount;               // hist 非零桶数（与 collect 开关无关）
    result.mergedCount = 0;

    if (dropSum > 0) {
        result.qualityFlag = QualityFlag::Degraded;
        result.message = "flat_center: dropped " + std::to_string(dropSum)
            + " runs, reasons=minRun:" + std::to_string(h_stats_[1])
            + " contrast:" + std::to_string(h_stats_[2])
            + " bright:" + std::to_string(h_stats_[3])
            + " edge:" + std::to_string(h_stats_[4])
            + " gate:" + std::to_string(h_stats_[5])
            + " width:" + std::to_string(h_stats_[6]);
        CALIB_LOG_DEBUG("{}", result.message);
    } else {
        result.qualityFlag = QualityFlag::Normal;
        result.message = "Subpixel extraction successful";
    }

    if (collectHostPoints_) {
        cudaMemcpyAsync(h_coords_, d_gcoords_, N * sizeof(float2), cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(h_ids_, d_gids_, N * sizeof(int), cudaMemcpyDeviceToHost, cs);
        stream.waitForCompletion();
        for (int lb = 0; lb < kLabelBins; ++lb) {
            const int n = h_hist_[lb];
            if (n <= 0) continue;
            std::vector<cv::Point2f> v(n);
            for (int i = 0; i < n; ++i) {
                const float2& p = h_coords_[h_base_[lb] + i];
                v[i] = cv::Point2f(p.x, p.y);
            }
            result.centerPoints.emplace(lb, std::move(v));
        }
    }

    result.success = true;

    if (bench_ && events_ready_) {
        cudaEventSynchronize(ev_end_);
        float t = 0;
        cudaEventElapsedTime(&t, ev_start_, ev_end_);  bench_ms_[0] = t;
        cudaEventElapsedTime(&t, ev_start_, ev_scan_); bench_ms_[1] = t;
        cudaEventElapsedTime(&t, ev_scan_, ev_end_);   bench_ms_[2] = t;
        cudaGetLastError();
        CALIB_LOG_DEBUG("[BENCH_FLAT] {}x{} | Total={:.2f}ms | Scan={:.2f}ms | Group/Out={:.2f}ms | N={} lines={}",
                        cols, rows, bench_ms_[0], bench_ms_[1], bench_ms_[2], N, lineCount);
    }
    return result;
}

void FlatCenterExtractor::Impl::Destroy() {
    freeAll();
}

void FlatCenterExtractor::Impl::Warmup(int rows, int cols) {
    ensureBuffers(rows, cols);
    warmed_up_ = true;
    warmup_rows_ = rows;
    warmup_cols_ = cols;
#ifndef NDEBUG
    // Debug 下 OOM 校验惯例（照 steger_fast_impl.cu:1338-1351）
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        CALIB_LOG_ERROR("Warmup device sync failed: {}", cudaGetErrorString(err));
    }
#endif
}

void FlatCenterExtractor::Impl::SetParams(const FlatCenterParams& params) {
    params.validate();
    if (params.deviceId != params_.deviceId) {
        old_device_id_ = params_.deviceId;
        // 设备切换：缓冲随设备走，重置后下次 Execute 重建
        freeAll();
        warmed_up_ = false;
    }
    // 【flat 优化·宏打包】参数变更→捕获图失效（内核实参已变）——销毁待重捕
    if (flatGraphExec_) { cudaGraphExecDestroy(flatGraphExec_); flatGraphExec_ = nullptr; }
    if (flatGraph_)     { cudaGraphDestroy(flatGraph_); flatGraph_ = nullptr; }
    params_ = params;
}
