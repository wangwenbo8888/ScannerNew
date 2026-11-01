#pragma once
// gpu_accum.h — 全局激光点云 GPU 累积（全链速度优化：需求 §5）
#include <cstddef>
#include <cuda_runtime_api.h>   // cudaEvent_t（E7 尾部补拷参数）
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace scr {

/// pts3f（CV_32FC3 激光 3D 点）经 R/T（double，与原 CPU 同序同精度）变换后
/// 追加到 cloud（CV_32FC3 容量缓冲，grow-only），count 累加。零同步零下载。
void gpuTransformAppend(const cv::cuda::GpuMat& pts3f,
                        const double R[9], const double T[3],
                        cv::cuda::GpuMat& cloud, size_t& count,
                        cv::cuda::Stream& stream);

// ============================================================================
// 【150FPS】GPU-RLE 掩膜压缩下载：标记点掩膜（CV_8UC1 稀疏）在 GPU 上行程
// 编码为 [row, x0, x1] 三元组（行主序、行内 x 升序——确定性两遍块扫描），
// 仅下载 KB 级行程（原 2.36MB/眼 D2H）。ccl_cpu 的 runCclCpuFromRuns 直接消费。
/// 双眼一次接口（全部异步排入 stream；调用方随后同步一次）：
///   countsPinned（2 int 锁页）← [nL, nR]；runsPinned（6*capRuns int 锁页）←
///   [L 行程…, R 行程…]（cap 定长下载，溢出行程被钳位丢弃——同步后按 nL/nR
///   >capRuns 判溢出走像素回退）。runBufL/R 为设备缓冲（1×3*capRuns CV_32SC1）。
bool gpuMaskRleBoth(const cv::cuda::GpuMat& maskL, const cv::cuda::GpuMat& maskR,
                    cv::cuda::GpuMat& runBufL, cv::cuda::GpuMat& runBufR,
                    int capRuns, int* countsPinned, int* runsPinned,
                    cv::cuda::Stream& stream);

// ============================================================================
// 【E7·RLE 按实拷】窗口版（2026-10-23 落地）：实测标记掩膜行程 ~1.1k/帧 vs
// 整包 65536×3×4B×2 眼＝1.57MB（过拷 118×）。常态只拷 counts＋双眼各前
// window 条行程（~49KB）；超窗罕见帧由 gpuRleTailCopy 私有流补拷尾部。
// 三不变量（R1-B）：①双侧 clamp cap（溢出仍由 n>capRuns 判墓碑，语义不变）
// ②补拷仅 window<n≤capRuns ③调用方 rleCopied_.store 须在本函数同步完成后。
/// 窗口版：runs 仅下载 [0, window) 段（counts 照常 8B；窗口钳 [0, capRuns]）
bool gpuMaskRleBothWindowed(const cv::cuda::GpuMat& maskL, const cv::cuda::GpuMat& maskR,
                            cv::cuda::GpuMat& runBufL, cv::cuda::GpuMat& runBufR,
                            int capRuns, int window,
                            int* countsPinned, int* runsPinned,
                            cv::cuda::Stream& stream);
/// 尾部补拷（同步返回；n≤window 直通）：私有流先等 evDone（1a 的 RLE 发射序
/// ——emit 内核完成的充分条件），再拷 [window, n) 段入锁页环同一槽尾部，
/// 布局与整包版逐位同构（下游零改动）。勿借 gS_/gS2_（流归属纪律）。
void gpuRleTailCopy(const cv::cuda::GpuMat& runBufL, const cv::cuda::GpuMat& runBufR,
                    int capRuns, int window, int nL, int nR,
                    int* runsPinned, cudaEvent_t evDone);

} // namespace scr
