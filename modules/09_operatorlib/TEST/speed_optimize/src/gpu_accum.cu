// gpu_accum.cu — 全局激光点云 GPU 累积（全链速度优化：需求 §5）
// 逐点 R/T 变换＋追加在 GPU 完成（double 运算——与原 CPU 逐位一致后转 float，
// PLY 字节级不变）；host 侧零下载零同步，容量 grow-only。
#include "gpu_accum.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace scr {

__global__ void transformAppendKernel(
    const float3* __restrict__ pts, int n,
    const double* __restrict__ RT,   // [R00..R22, T0..T2]
    float3* __restrict__ out, int base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 q = pts[i];
    // 与原 CPU 逐位一致：__dmul_rn/__dadd_rn 显式 IEEE 运算（禁 FMA 收缩），
    // 左结合树 ((R0*x + R1*y) + R2*z) + T 与 CPU 求值序相同
    const double x = q.x, y = q.y, z = q.z;
    out[base + i] = make_float3(
        static_cast<float>(__dadd_rn(__dadd_rn(__dadd_rn(
            __dmul_rn(RT[0], x), __dmul_rn(RT[1], y)),
            __dmul_rn(RT[2], z)), RT[9])),
        static_cast<float>(__dadd_rn(__dadd_rn(__dadd_rn(
            __dmul_rn(RT[3], x), __dmul_rn(RT[4], y)),
            __dmul_rn(RT[5], z)), RT[10])),
        static_cast<float>(__dadd_rn(__dadd_rn(__dadd_rn(
            __dmul_rn(RT[6], x), __dmul_rn(RT[7], y)),
            __dmul_rn(RT[8], z)), RT[11])));
}

void gpuTransformAppend(const cv::cuda::GpuMat& pts3f,
                         const double R[9], const double T[3],
                         cv::cuda::GpuMat& cloud, size_t& count,
                         cv::cuda::Stream& stream)
{
    const int n = pts3f.rows * pts3f.cols;
    if (n <= 0) return;
    // ROI 视图（上游持久缓冲的有效区）先拷成连续（同原 CPU copyTo 口径）
    static thread_local cv::cuda::GpuMat contiguous;
    if (!pts3f.isContinuous()) {
        pts3f.copyTo(contiguous, stream);
    } else {
        contiguous = pts3f;   // 浅引用（连续时直接用）
    }
    if (cloud.cols < static_cast<int>(count + n)) {
        int cap = std::max(cloud.cols * 2, static_cast<int>(count + n));
        cv::cuda::GpuMat grown(1, cap, CV_32FC3);
        if (count > 0)
            cloud.copyTo(grown(cv::Rect(0, 0, cloud.cols, 1)), stream);  // ROI 视图收数——防 copyTo 降采样回源尺寸
        cloud = grown;
    }
    static thread_local cv::cuda::GpuMat d_rt;
    static thread_local double* h_rtPin = nullptr;   // pinned 常驻（12 double）——
    if (!h_rtPin) cudaMallocHost(&h_rtPin, 12 * sizeof(double));  // 免 pageable 上传的隐式流同步
    {
        for (int i = 0; i < 9; ++i) h_rtPin[i] = R[i];
        for (int i = 0; i < 3; ++i) h_rtPin[9 + i] = T[i];
        cv::Mat h_rt(1, 12, CV_64FC1, h_rtPin);      // 包装 pinned（零拷贝构造）
        d_rt.upload(h_rt, stream);
    }
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    const int grid = (n + 255) / 256;
    transformAppendKernel<<<grid, 256, 0, cs>>>(
        contiguous.ptr<float3>(), n, d_rt.ptr<double>(),
        cloud.ptr<float3>(), static_cast<int>(count));
    count += static_cast<size_t>(n);
}

// ============================================================================
// 【150FPS】GPU-RLE 掩膜压缩：每行一块两遍法（行内 x 升序确定性输出）
//   Kernel1 rleCountKernel：逐行数行程
//   Kernel2 rleScanKernel  ：单块行偏移前缀和＋总计数
//   Kernel3 rleEmitKernel  ：逐行两遍（本地计数→块内独占前缀→按序写出）
// ============================================================================
__global__ void rleCountKernel(const unsigned char* __restrict__ mask, int step,
                               int W, int* __restrict__ rowCounts) {
    const int y = blockIdx.x;
    const unsigned char* row = mask + (size_t)y * step;
    int cnt = 0;
    for (int x = threadIdx.x; x < W; x += blockDim.x) {
        const bool cur = row[x] != 0;
        const bool prev = x > 0 ? row[x - 1] != 0 : false;
        if (cur && !prev) ++cnt;   // 行程起点
    }
    // 块内规约（thread-local 计数之和）
    __shared__ int sSum;
    if (threadIdx.x == 0) sSum = 0;
    __syncthreads();
    atomicAdd(&sSum, cnt);
    __syncthreads();
    if (threadIdx.x == 0) rowCounts[y] = sSum;
}

__global__ void rleScanKernel(int* __restrict__ rowCounts, int* __restrict__ rowOffsets,
                              int H, int* __restrict__ total) {
    if (threadIdx.x == 0) {   // H≤4096：单线程串行前缀（μs 级）
        int acc = 0;
        for (int y = 0; y < H; ++y) {
            rowOffsets[y] = acc;
            acc += rowCounts[y];
        }
        *total = acc;
    }
}

__global__ void rleEmitKernel(const unsigned char* __restrict__ mask, int step, int W,
                              const int* __restrict__ rowOffsets,
                              int* __restrict__ out /* 3*cap */, int cap) {
    const int y = blockIdx.x;
    const unsigned char* row = mask + (size_t)y * step;
    __shared__ int sThreadCnt[256];
    __shared__ int sBase;
    if (threadIdx.x == 0) sBase = rowOffsets[y];
    const int tid = threadIdx.x;
    // 第一遍：各线程数自己的行程起点
    int cnt = 0;
    for (int x = tid; x < W; x += blockDim.x) {
        const bool cur = row[x] != 0;
        const bool prev = x > 0 ? row[x - 1] != 0 : false;
        if (cur && !prev) ++cnt;
    }
    sThreadCnt[tid] = cnt;
    __syncthreads();
    // 块内独占前缀（Hillis-Steele on shared）
    for (int d = 1; d < blockDim.x; d <<= 1) {
        int add = 0;
        if (tid >= d) add = sThreadCnt[tid - d];
        __syncthreads();
        sThreadCnt[tid] += add;
        __syncthreads();
    }
    const int myOff = sThreadCnt[tid] - cnt;   // 独占前缀
    // 第二遍：按序写出（保持与计数遍历完全相同的线程步进——禁止跳位，
    // 否则写槽数与各线程预扫描计数不一致会错位/越槽）
    int k = 0;
    for (int x = tid; x < W; x += blockDim.x) {
        const bool cur = row[x] != 0;
        const bool prev = x > 0 ? row[x - 1] != 0 : false;
        if (cur && !prev) {
            int e = x;
            while (e + 1 < W && row[e + 1] != 0) ++e;   // 行程末端（稀疏→短扫）
            const int slot = sBase + myOff + k;
            if (slot < cap) {                            // 溢出钳位（host 侧按 total 检测回退）
                int* o = out + (size_t)slot * 3;
                o[0] = y; o[1] = x; o[2] = e;
            }
            ++k;
        }
    }
}

// 【E7】公共段：两眼 RLE 内核发射＋counts D2H（整包版/窗口版共用；调用方
// 线程常驻暂存——countsL/R、offsL/R、total）
static void rleEnqueueKernelsAndCounts(const cv::cuda::GpuMat& maskL,
                                       const cv::cuda::GpuMat& maskR,
                                       cv::cuda::GpuMat& runBufL,
                                       cv::cuda::GpuMat& runBufR,
                                       int capRuns, int* countsPinned,
                                       cudaStream_t cs) {
    const int H = maskL.rows, W = maskL.cols;
    // 临时区：rowCounts/rowOffsets ×2 眼＋total ×2（调用方线程常驻）
    static thread_local int* tRows = nullptr;     // 4H ints（counts+offs ×2 眼）
    static thread_local int* tTotal = nullptr;    // 2 ints（device）
    if (tRows == nullptr) {
        cudaMalloc(&tRows, sizeof(int) * 4 * H);
        cudaMalloc(&tTotal, 2 * sizeof(int));
    }
    int* countsL = tRows;
    int* offsL = tRows + H;
    int* countsR = tRows + 2 * H;
    int* offsR = tRows + 3 * H;
    rleCountKernel<<<H, 256, 0, cs>>>(maskL.ptr<unsigned char>(),
                                      static_cast<int>(maskL.step), W, countsL);
    rleCountKernel<<<H, 256, 0, cs>>>(maskR.ptr<unsigned char>(),
                                      static_cast<int>(maskR.step), W, countsR);
    rleScanKernel<<<1, 32, 0, cs>>>(countsL, offsL, H, tTotal + 0);
    rleScanKernel<<<1, 32, 0, cs>>>(countsR, offsR, H, tTotal + 1);
    rleEmitKernel<<<H, 256, 0, cs>>>(maskL.ptr<unsigned char>(),
                                     static_cast<int>(maskL.step), W, offsL,
                                     runBufL.ptr<int>(), capRuns);
    rleEmitKernel<<<H, 256, 0, cs>>>(maskR.ptr<unsigned char>(),
                                     static_cast<int>(maskR.step), W, offsR,
                                     runBufR.ptr<int>(), capRuns);
    cudaMemcpyAsync(countsPinned, tTotal, 2 * sizeof(int), cudaMemcpyDeviceToHost, cs);
}

bool gpuMaskRleBoth(const cv::cuda::GpuMat& maskL, const cv::cuda::GpuMat& maskR,
                    cv::cuda::GpuMat& runBufL, cv::cuda::GpuMat& runBufR,
                    int capRuns, int* countsPinned, int* runsPinned,
                    cv::cuda::Stream& stream) {
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    rleEnqueueKernelsAndCounts(maskL, maskR, runBufL, runBufR, capRuns, countsPinned, cs);
    // 一次同步口径：total 与 runs（cap 定长）都排进流后调用方同步一次即全就绪
    cudaMemcpyAsync(runsPinned, runBufL.ptr<int>(),
                    sizeof(int) * 3 * capRuns, cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(runsPinned + 3 * capRuns, runBufR.ptr<int>(),
                    sizeof(int) * 3 * capRuns, cudaMemcpyDeviceToHost, cs);
    return true;
}

bool gpuMaskRleBothWindowed(const cv::cuda::GpuMat& maskL, const cv::cuda::GpuMat& maskR,
                            cv::cuda::GpuMat& runBufL, cv::cuda::GpuMat& runBufR,
                            int capRuns, int window,
                            int* countsPinned, int* runsPinned,
                            cv::cuda::Stream& stream) {
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    rleEnqueueKernelsAndCounts(maskL, maskR, runBufL, runBufR, capRuns, countsPinned, cs);
    const int w = std::max(0, std::min(window, capRuns));
    // 常态窗口 D2H：双眼各 3*w ints（w=4096 → ~49KB vs 整包 1.57MB，实测行程
    // ~1.1k/帧＝窗口 3.7× 余量；[w, n) 段由 gpuRleTailCopy 罕见路径补齐）
    cudaMemcpyAsync(runsPinned, runBufL.ptr<int>(),
                    sizeof(int) * 3 * w, cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(runsPinned + 3 * capRuns, runBufR.ptr<int>(),
                    sizeof(int) * 3 * w, cudaMemcpyDeviceToHost, cs);
    return true;
}

void gpuRleTailCopy(const cv::cuda::GpuMat& runBufL, const cv::cuda::GpuMat& runBufR,
                    int capRuns, int window, int nL, int nR,
                    int* runsPinned, cudaEvent_t evDone) {
    if (nL <= window && nR <= window) return;
    // 私有流（调用线程常驻；勿借调用方业务流——流归属纪律）
    static thread_local cudaStream_t s = nullptr;
    if (s == nullptr) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    // evDone＝1a 在 RLE D2H 之后记录的事件 → 等它即覆盖 emit 内核完成（同流 FIFO）
    if (evDone) cudaStreamWaitEvent(s, evDone, 0);
    if (nL > window)
        cudaMemcpyAsync(runsPinned + 3 * window, runBufL.ptr<int>() + 3 * window,
                        sizeof(int) * 3 * (nL - window), cudaMemcpyDeviceToHost, s);
    if (nR > window)
        cudaMemcpyAsync(runsPinned + 3 * static_cast<size_t>(capRuns) + 3 * window,
                        runBufR.ptr<int>() + 3 * window,
                        sizeof(int) * 3 * (nR - window), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);   // E7③：调用方 store（槽释放）前同步完成
}

} // namespace scr
