/**
 * @file steger_fast_impl.cu
 * @brief Steger激光中心亚像素提取算子（优化版）CUDA 实现
 *
 * 步骤（对照原版 ../steger/steger_extract_cuda_impl.cu 的 6 步）：
 *   Step 1+2a: RowFan3Kernel — uchar 直读一次算出行级 g/gx/gxx 三个中间图
 *              （原版 5 次行卷积、其中 g 与 g' 各被重复计算一次；本版读 1 遍写 3 遍）
 *   Step 2b:   ColFan5Kernel — 一次算出 Ix/Iy/Ixx/Iyy/Ixy 五个导数（原版 5 次列卷积）
 *   Step 3+4:  HessianEigenAndTaylorKernel — 与原版逐位一致（Flat 模式模板化 8U 掩膜）
 *   Step 5:    LabelHist/LabelScatter — 计数直写分组替代 thrust stable_sort
 *   Step 6:    单次 D2H + D2D 组装 CPU/GPU 双输出，替代原版 D2H→cv::Mat→H2D 往返
 *
 * 数值等价性：导数累加保持与原版相同的 k 升序与 renormalize（w_sum 除法）语义，
 * 输入 uchar→float 转换无损，因此导数与点坐标与原版逐位一致；
 * 分组后 label 内点序与原版一样由原子时序决定（本就非确定），坐标值不变。
 */

#include "steger_fast_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

using namespace calib;

CALIB_DEFINE_LOG_TAG(10, StegerExtractorFast);

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void RowFan3Kernel(
    const unsigned char* __restrict__ d_in,
    float* __restrict__ d_out_g, float* __restrict__ d_out_gx, float* __restrict__ d_out_gxx,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* __restrict__ kg, const float* __restrict__ kgx, const float* __restrict__ kgxx,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    const unsigned char* in_row = d_in + y * in_step;

    float s_g = 0.0f, ws_g = 0.0f, s_gx = 0.0f, s_gxx = 0.0f;
    for (int k = -half; k <= half; ++k) {
        int xx = x + k;
        if (xx >= 0 && xx < cols) {
            float v = static_cast<float>(in_row[xx]);
            float wg = kg[k + half];
            s_g += wg * v;
            ws_g += wg;
            s_gx += kgx[k + half] * v;
            s_gxx += kgxx[k + half] * v;
        }
    }

    char* og = reinterpret_cast<char*>(d_out_g) + y * out_step;
    char* ogx = reinterpret_cast<char*>(d_out_gx) + y * out_step;
    char* ogxx = reinterpret_cast<char*>(d_out_gxx) + y * out_step;
    reinterpret_cast<float*>(og)[x] = s_g / ws_g;
    reinterpret_cast<float*>(ogx)[x] = s_gx;
    reinterpret_cast<float*>(ogxx)[x] = s_gxx;
}

__global__ void ColFan5Kernel(
    const float* __restrict__ d_tg, const float* __restrict__ d_tgx, const float* __restrict__ d_tgxx,
    float* __restrict__ d_ix, float* __restrict__ d_iy,
    float* __restrict__ d_ixx, float* __restrict__ d_iyy, float* __restrict__ d_ixy,
    int rows, int cols, size_t in_step, size_t out_step,
    const float* __restrict__ kg, const float* __restrict__ kgx, const float* __restrict__ kgxx,
    int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int half = kernel_size / 2;
    float s_ix = 0.0f, ws_ix = 0.0f;
    float s_iy = 0.0f;
    float s_ixx = 0.0f, ws_ixx = 0.0f;
    float s_iyy = 0.0f;
    float s_ixy = 0.0f;

    for (int k = -half; k <= half; ++k) {
        int yy = y + k;
        if (yy >= 0 && yy < rows) {
            const float* rg = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tg) + yy * in_step);
            const float* rgx = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tgx) + yy * in_step);
            const float* rgxx = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(d_tgxx) + yy * in_step);
            float vg = rg[x], vgx = rgx[x], vgxx = rgxx[x];
            float wg = kg[k + half], wgx = kgx[k + half], wgxx = kgxx[k + half];
            s_ix += wg * vgx;  ws_ix += wg;
            s_iy += wgx * vg;
            s_ixx += wg * vgxx;  ws_ixx += wg;
            s_iyy += wgxx * vg;
            s_ixy += wgx * vgx;
        }
    }

    char* oix = reinterpret_cast<char*>(d_ix) + y * out_step;
    char* oiy = reinterpret_cast<char*>(d_iy) + y * out_step;
    char* oixx = reinterpret_cast<char*>(d_ixx) + y * out_step;
    char* oiyy = reinterpret_cast<char*>(d_iyy) + y * out_step;
    char* oixy = reinterpret_cast<char*>(d_ixy) + y * out_step;
    reinterpret_cast<float*>(oix)[x] = s_ix / ws_ix;
    reinterpret_cast<float*>(oiy)[x] = s_iy;
    reinterpret_cast<float*>(oixx)[x] = s_ixx / ws_ixx;
    reinterpret_cast<float*>(oiyy)[x] = s_iyy;
    reinterpret_cast<float*>(oixy)[x] = s_ixy;
}

template <bool MASK8>
__global__ void HessianEigenAndTaylorKernel(
    const float* d_ix,  const float* d_iy,
    const float* d_ixx, const float* d_iyy, const float* d_ixy,
    const void* d_mask, size_t mask_step,
    const float* d_tg, size_t tg_step,
    int rows, int cols,
    size_t ix_step, size_t ixx_step,
    float low_thresh, float high_thresh,
    float max_line_width,
    FastPoint* d_out_points, int* d_out_count, int* d_reject_count,
    int max_points)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int label;
    if (MASK8) {
        const unsigned char* mask_row =
            reinterpret_cast<const unsigned char*>(d_mask) + y * mask_step;
        label = mask_row[x] > 0 ? 1 : 0;
    } else {
        const int* mask_row = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(d_mask) + y * mask_step);
        label = mask_row[x];
    }
    if (label <= 0) return;

    const float* ix_row  = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ix) + y * ix_step);
    const float* iy_row  = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_iy) + y * ix_step);
    const float* ixx_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ixx) + y * ixx_step);
    const float* iyy_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_iyy) + y * ixx_step);
    const float* ixy_row = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(d_ixy) + y * ixx_step);

    float ix_val  = ix_row[x];
    float iy_val  = iy_row[x];
    float ixx_val = ixx_row[x];
    float iyy_val = iyy_row[x];
    float ixy_val = ixy_row[x];

    float trace = ixx_val + iyy_val;
    float det = ixx_val * iyy_val - ixy_val * ixy_val;
    float disc = trace * trace - 4.0f * det;

    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float lambda1 = 0.5f * (trace + sqrt_disc);
    float lambda2 = 0.5f * (trace - sqrt_disc);

    float lambda;
    float nx, ny;
    if (fabsf(lambda1) >= fabsf(lambda2)) {
        lambda = lambda1;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 1.0f;
            ny = 0.0f;
        } else {
            nx = lambda1 - iyy_val;
            ny = ixy_val;
        }
    } else {
        lambda = lambda2;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 0.0f;
            ny = 1.0f;
        } else {
            nx = lambda2 - iyy_val;
            ny = ixy_val;
        }
    }

    if (fabsf(lambda) < low_thresh) return;
    if (high_thresh > 0.0f && fabsf(lambda) > high_thresh) return;

    // 逐点宽度门禁(maxLineWidth>0 启用; off 时分支恒 false → 与原版逐位一致):
    // 平滑后口径 w² = g/|λ|（g=行平滑幅度, 近似法向峰幅; λ 已过 low_thresh 下限无除零）。
    // 过曝局部变宽/平顶剖面 → w² 显著大于正常线 → 弃点（防亚像素中心偏移噪点）。
    // g<0 数值噪声 → w²<0 → 比较为 false → 放行（等同 off, NaN 语义同）。
    if (max_line_width > 0.0f) {
        const float* tg_row = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(d_tg) + y * tg_step);
        const float w2 = tg_row[x] / fabsf(lambda);
        if (w2 > max_line_width * max_line_width) {
            atomicAdd(d_reject_count, 1);
            return;
        }
    }

    float norm = sqrtf(nx * nx + ny * ny);
    if (norm < 1e-10f) return;
    nx /= norm;
    ny /= norm;

    float denom2 = ixx_val * nx * nx + 2.0f * ixy_val * nx * ny + iyy_val * ny * ny;
    if (fabsf(denom2) < 1e-10f) return;

    float t = -(ix_val * nx + iy_val * ny) / denom2;

    if (fabsf(t) >= 0.5f) return;

    float px = static_cast<float>(x) + t * nx;
    float py = static_cast<float>(y) + t * ny;

    if (px < 0.0f || px >= static_cast<float>(cols) ||
        py < 0.0f || py >= static_cast<float>(rows)) return;

    int idx = atomicAdd(d_out_count, 1);
    if (idx < max_points) {
        d_out_points[idx].label = label;
        d_out_points[idx].px = px;
        d_out_points[idx].py = py;
    }
}

// ============================================================================
// 全融合卷积 kernel（P1）：Row3＋Col5＋Hessian/Taylor/宽度门三合一。
// tile 32×16 输出/CTA（512 线程）——灰度带双向 halo 进 smem，三中间平面
// （tg/tgx/tgxx）只在 smem 存在（不落显存），每输出像素的 5 导数在寄存器
// 内直出后接 Hessian。数值逐位对齐三分kernel版：
//   - 行卷积/列卷积 k 严格升序＋同款越界 guard（全局坐标判定）＋renormalize
//     除法语义逐字保留；smem float＝global float（均 fp32 物化）
//   - Hessian/Taylor/宽度门代码自 HessianEigenAndTaylorKernel 原样内联
// tile 完全自足（tg 不再全局可见）→ 空 tile（核心区掩膜全零）直接早退，
// 不影响任何其它 tile 的输出（跳块谓词无需 halo 膨胀）。
// ============================================================================
constexpr int kFuseTileX = 32;                        // 输出 tile 宽
constexpr int kFuseTileY = 16;                        // 输出 tile 高
constexpr int kFuseHalf = 15;                         // kMaxKernelSize(31)/2——
                                                     // smem 尺寸按最大核备
constexpr int kFuseSmemCols = kFuseTileX + 2 * kFuseHalf;   // 62
constexpr int kFuseSmemRows = kFuseTileY + 2 * kFuseHalf;   // 46

template <bool MASK8>
__global__ void FusedConvHessianKernel(
    const unsigned char* __restrict__ d_gray, size_t gray_step,
    const void* d_mask, size_t mask_step,
    int rows, int cols,
    const float* __restrict__ kg, const float* __restrict__ kgx,
    const float* __restrict__ kgxx, int kernel_size,
    float low_thresh, float high_thresh,
    float max_line_width,
    FastPoint* d_out_points, int* d_out_count, int* d_reject_count,
    int max_points)
{
    __shared__ unsigned char s_gray[kFuseSmemRows][kFuseSmemCols];
    __shared__ float s_tg[kFuseSmemRows][kFuseTileX];
    __shared__ float s_tgx[kFuseSmemRows][kFuseTileX];
    __shared__ float s_tgxx[kFuseSmemRows][kFuseTileX];
    __shared__ int s_any;

    const int half = kernel_size / 2;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * kFuseTileX + tx;
    const int x0 = blockIdx.x * kFuseTileX;
    const int y0 = blockIdx.y * kFuseTileY;

    // —— Stage A0: 空_tile 早退（核心区掩膜判空；tile 自足 ⇒ 无 halo 膨胀需求）——
    if (tid == 0) s_any = 0;
    __syncthreads();
    {
        const int x = x0 + tx, y = y0 + ty;
        int lab = 0;
        if (x < cols && y < rows) {
            if (MASK8) {
                const unsigned char* mask_row =
                    reinterpret_cast<const unsigned char*>(d_mask) + y * mask_step;
                lab = mask_row[x] > 0 ? 1 : 0;
            } else {
                const int* mask_row = reinterpret_cast<const int*>(
                    reinterpret_cast<const char*>(d_mask) + y * mask_step);
                lab = mask_row[x];
            }
        }
        if (lab > 0) s_any = 1;
        __syncthreads();
        if (!s_any) return;
    }

    // —— Stage A: 协作载灰度 tile（含双向 halo；越界置 0——不会被读，读处有
    //    与原版同款的全局坐标 guard）——
    for (int idx = tid; idx < kFuseSmemRows * kFuseSmemCols;
         idx += kFuseTileX * kFuseTileY) {
        const int c = idx % kFuseSmemCols, r = idx / kFuseSmemCols;
        const int gx = x0 - kFuseHalf + c, gy = y0 - kFuseHalf + r;
        s_gray[r][c] = (gx >= 0 && gx < cols && gy >= 0 && gy < rows)
            ? *reinterpret_cast<const unsigned char*>(
                  reinterpret_cast<const char*>(d_gray) + gy * gray_step + gx)
            : 0;
    }
    __syncthreads();

    // —— Stage B: 行卷积 → smem 三平面（列卷积所需的 (2·half+16) 行 × 32 列；
    //    每线程 ~2.5 位置。行越界不写（ColFan5 消费侧同款行 guard，永不读）——
    for (int idx = tid; idx < kFuseTileX * (2 * half + kFuseTileY);
         idx += kFuseTileX * kFuseTileY) {
        const int lx = idx % kFuseTileX;
        const int li = idx / kFuseTileX;                 // band 内行号 [0, 2·half+16)
        const int r = li + (kFuseHalf - half);           // smem 行索引
        const int gy = y0 - half + li;
        const int gx = x0 + lx;
        float s_g = 0.0f, ws_g = 0.0f, s_gx = 0.0f, s_gxx = 0.0f;
        if (gy >= 0 && gy < rows) {
            const unsigned char* grow = s_gray[r];
            for (int k = -half; k <= half; ++k) {
                int xx = gx + k;
                if (xx >= 0 && xx < cols) {
                    float v = static_cast<float>(grow[xx - (x0 - kFuseHalf)]);
                    float wg = kg[k + half];
                    s_g += wg * v;
                    ws_g += wg;
                    s_gx += kgx[k + half] * v;
                    s_gxx += kgxx[k + half] * v;
                }
            }
        }
        s_tg[r][lx] = s_g / ws_g;
        s_tgx[r][lx] = s_gx;
        s_tgxx[r][lx] = s_gxx;
    }
    __syncthreads();

    // —— Stage C: 每线程 1 输出像素——列卷积（读 smem 三平面，k 升序＋行 guard
    //    ＋renormalize 语义同 ColFan5Kernel）＋Hessian/Taylor/宽度门（原样内联）——
    const int x = x0 + tx, y = y0 + ty;
    if (x >= cols || y >= rows) return;

    int label;
    if (MASK8) {
        const unsigned char* mask_row =
            reinterpret_cast<const unsigned char*>(d_mask) + y * mask_step;
        label = mask_row[x] > 0 ? 1 : 0;
    } else {
        const int* mask_row = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(d_mask) + y * mask_step);
        label = mask_row[x];
    }
    if (label <= 0) return;

    float s_ix = 0.0f, ws_ix = 0.0f;
    float s_iy = 0.0f;
    float s_ixx = 0.0f, ws_ixx = 0.0f;
    float s_iyy = 0.0f;
    float s_ixy = 0.0f;
    for (int k = -half; k <= half; ++k) {
        int yy = y + k;
        if (yy >= 0 && yy < rows) {
            const int r = yy - (y0 - kFuseHalf);
            float vg = s_tg[r][tx], vgx = s_tgx[r][tx], vgxx = s_tgxx[r][tx];
            float wg = kg[k + half], wgx = kgx[k + half], wgxx = kgxx[k + half];
            s_ix += wg * vgx;  ws_ix += wg;
            s_iy += wgx * vg;
            s_ixx += wg * vgxx;  ws_ixx += wg;
            s_iyy += wgxx * vg;
            s_ixy += wgx * vgx;
        }
    }
    const float ix_val = s_ix / ws_ix;
    const float iy_val = s_iy;
    const float ixx_val = s_ixx / ws_ixx;
    const float iyy_val = s_iyy;
    const float ixy_val = s_ixy;

    float trace = ixx_val + iyy_val;
    float det = ixx_val * iyy_val - ixy_val * ixy_val;
    float disc = trace * trace - 4.0f * det;

    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float lambda1 = 0.5f * (trace + sqrt_disc);
    float lambda2 = 0.5f * (trace - sqrt_disc);

    float lambda;
    float nx, ny;
    if (fabsf(lambda1) >= fabsf(lambda2)) {
        lambda = lambda1;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 1.0f;
            ny = 0.0f;
        } else {
            nx = lambda1 - iyy_val;
            ny = ixy_val;
        }
    } else {
        lambda = lambda2;
        float denom = ixy_val;
        if (fabsf(denom) < 1e-10f) {
            nx = 0.0f;
            ny = 1.0f;
        } else {
            nx = lambda2 - iyy_val;
            ny = ixy_val;
        }
    }

    if (fabsf(lambda) < low_thresh) return;
    if (high_thresh > 0.0f && fabsf(lambda) > high_thresh) return;

    if (max_line_width > 0.0f) {
        const float tg_own = s_tg[y - (y0 - kFuseHalf)][tx];
        const float w2 = tg_own / fabsf(lambda);
        if (w2 > max_line_width * max_line_width) {
            atomicAdd(d_reject_count, 1);
            return;
        }
    }

    float norm = sqrtf(nx * nx + ny * ny);
    if (norm < 1e-10f) return;
    nx /= norm;
    ny /= norm;

    float denom2 = ixx_val * nx * nx + 2.0f * ixy_val * nx * ny + iyy_val * ny * ny;
    if (fabsf(denom2) < 1e-10f) return;

    float t = -(ix_val * nx + iy_val * ny) / denom2;

    if (fabsf(t) >= 0.5f) return;

    float px = static_cast<float>(x) + t * nx;
    float py = static_cast<float>(y) + t * ny;

    if (px < 0.0f || px >= static_cast<float>(cols) ||
        py < 0.0f || py >= static_cast<float>(rows)) return;

    int idx = atomicAdd(d_out_count, 1);
    if (idx < max_points) {
        d_out_points[idx].label = label;
        d_out_points[idx].px = px;
        d_out_points[idx].py = py;
    }
}

__global__ void LabelHistKernel(
    const FastPoint* __restrict__ pts, int count, int bins,
    int* __restrict__ hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = pts[i].label;
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    atomicAdd(&hist[lb], 1);
}

__global__ void LabelScatterKernel(
    const FastPoint* __restrict__ pts, int count, int bins,
    const int* __restrict__ base, int* __restrict__ cursor,
    FastPoint* __restrict__ out,
    float2* __restrict__ coords, int* __restrict__ ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int lb = pts[i].label;
    if (lb < 1) return;
    if (lb >= bins) lb = bins - 1;
    int slot = atomicAdd(&cursor[lb], 1);
    int idx = base[lb] + slot;
    FastPoint p = pts[i];
    p.label = lb;
    out[idx] = p;
    coords[idx] = make_float2(p.px, p.py);
    ids[idx] = lb;
}

// Flat 快路分组直通（mask8 下 label 恒 1）：hist/CPU 前缀和/scatter 全免——
// 点序＝源序（旧 scatter 为原子序，点集多重集等价，golden 按排序比对口径）
__global__ void FlatCopyOutKernel(const FastPoint* __restrict__ pts, int n,
                                  float2* __restrict__ coords, int* __restrict__ ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    coords[i] = make_float2(pts[i].px, pts[i].py);
    ids[i] = 1;
}

// ============================================================================
// dedup v2（dedupMode=1）：平顶双点位移式合并——网格排序 + 邻域 PCA 几何切向
// + 位移门配对（定长邻接行）+ pull/shortcut 标签传播 + (簇键,rrid) 排序段归约。
// 判据常量与 CPU 验证版（TEST/steger_dedup_replay/src/steger_dedup_merge.cpp）逐字同值；
// 确定性：排序键拼 rrid（round 恢复有半 ulp 撞号洞——键重复时稳定排序保输入序，
// 见 D1 注）；PCA top-10 按 (d²,rrid) 平局裁决＋定序累加；传播双缓冲纯 pull
// 无原子（rootkey 的 atomicMin 结果序无关合法）。
// ============================================================================
constexpr float kDedupPairDist   = 1.6f;   // 配对欧氏距上限
constexpr float kDedupAlongGate  = 0.5f;   // 同簇：沿线偏移 ≤0.5px
constexpr float kDedupNormalGate = 1.5f;   //        法向偏移 ≤1.5px
constexpr float kDedupTangentDot = 0.5f;   // 切向一致门 |t̂ᵢ·t̂ⱼ|≥0.5
constexpr float kDedupNbrRadius  = 2.5f;   // PCA 邻域半径
constexpr int   kDedupNbrMax     = 10;     // PCA 邻居上限
constexpr int   kDedupClusterMax = 8;      // 簇点数上限（超则整段直通）
constexpr float kDedupSpanMax    = 2.5f;   // 簇法向跨度上限（超则直通）
constexpr int   kDedupAdjStride  = 24;     // 邻接行宽：几何上界 8（半径 1.6 圆盘含自己
                                           // ≤9 格点、每像素≤1 点），3 倍余量；溢出→flag 直通
constexpr int   kDedupCellBits   = 12;     // cell 坐标位宽（4096 守卫保证域）
constexpr int   kDedupRidxBits   = 22;     // rrid 位宽（4M 守卫保证域）
// 【计数设备侧化】去重链实用上限：定容发射容量（真实 N 本数据 ≤4.4 万；
// 超限帧由 flags[3] 捕获→宿主以真 N 重跑同链保逐位正确）。CUB 排序按此 N
// 发射（真 N→上限 N 的主要新增 GPU 成本 ~+0.3ms/眼）
constexpr int   kPracDedupCap    = 65536;

// D1 打包：cell=(floor(py)<<12|floor(px))，rrid=round 像素恢复 y*cols+x。
// ⚠ round 恢复并非绝对不变量：|px-x|=|t·nx|<0.5 严格，但 t 处于距 0.5 半个 ulp
// 内时 round 可舍到相邻像素 → rrid 撞号（角例）。因此 rrid 只作排序/平局裁决键，
// 不承担寻址唯一性（切向等中间数组按 D1 排序位索引，见 D2/D3/D6）。
// 撞号残余（文档化接受）：D1/D5 键重复时 CUB 稳定排序保 atomicAdd 输入序；
// 撞号两点坐标差 ~ulp 量级，仅影响中位并列时的极微取舍；D3 对门方规则用 >=
// 使等 rrid 对双侧一致弃权（直通，无害且确定）。
// 排序载荷用 u32 下标（16B DedupPt 载荷经 gather 物化——u32 载荷排序更省）
__global__ void DedupCellPackKernel(const FastPoint* pts, const int* np, int cap,
                                    int cols,
                                    unsigned long long* keys, unsigned* idxs,
                                    DedupPt* vals, int* rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap) return;
    // 【计数设备侧化】n 从显存读（宿主不回读）——网格按实用上限定容发射
    const int n = min(*np, cap);
    if (i >= n) {
        // 哨兵垫：排序 pad 位定值（46bit 全 1 → 排后垫尾；载荷=自身下标，确定）
        keys[i] = 0xFFFFFFFFFFFFULL;
        idxs[i] = (unsigned)i;
        return;
    }
    rowlen[i] = 0;   // 顺带清邻接行长度（原独立 memset [0,n) 吸收进 kernel）
    float px = pts[i].px, py = pts[i].py;
    int cell = ((int)floorf(py) << kDedupCellBits) | (int)floorf(px);
    int rrid = __float2int_rn(py) * cols + __float2int_rn(px);
    keys[i] = ((unsigned long long)(unsigned)cell << kDedupRidxBits)
            | (unsigned long long)(unsigned)rrid;
    idxs[i] = (unsigned)i;
    DedupPt v;
    v.rrid = rrid; v.label = pts[i].label; v.px = px; v.py = py;
    vals[i] = v;
}

// D1 gather：按排序后下标物化 DedupPt 载荷
__global__ void DedupGatherKernel(const unsigned* __restrict__ idxs,
                                  const DedupPt* __restrict__ vals_in,
                                  DedupPt* __restrict__ vals_out,
                                  const int* np, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);
    if (i >= n) return;
    vals_out[i] = vals_in[idxs[i]];
}

// ============================================================================
// 【批1·D1 行桶计数排序】替代 CUB RadixSort#1——等价性：CUB 稳定排序按键
// (cy,cx,rrid,i) 全序；行桶版桶间按 cy 升序、桶内按 (cx,rrid,i) 插入排序
// ——产出与 CUB 逐元素相同（构造性证明）。
// ============================================================================

__global__ void DedupRowCountKernel(
    const unsigned long long* __restrict__ ckeys, int n,
    int* __restrict__ rowcnt)                       // [4097] 已清零
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned cy = (unsigned)(ckeys[i] >> 34);  // cell 高 12bit = cy
    atomicAdd(&rowcnt[cy + 1], 1);
}

__global__ void DedupRowScanKernel(
    int* __restrict__ rowcnt, int n_rows)            // [4097] 入=计数 出=rowbase
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int run = 0;
    rowcnt[0] = 0;
    for (int i = 1; i <= n_rows; ++i) {
        run += rowcnt[i];
        rowcnt[i] = run;
    }
}

__global__ void DedupRowScatterKernel(
    const unsigned long long* __restrict__ ckeys,
    const unsigned* __restrict__ cidx_a,
    const int* __restrict__ rowbase,
    int n,
    int* __restrict__ rowcursor,                     // [4096] 已清零
    unsigned* __restrict__ rowslot)                  // [n] 行内槽位（存原始 i）
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned cy = (unsigned)(ckeys[i] >> 34);
    rowslot[rowbase[cy] + atomicAdd(&rowcursor[cy], 1)] = i;
}

__global__ void DedupRowSortKernel(
    const unsigned long long* __restrict__ ckeys,
    const unsigned* __restrict__ cidx_a,
    const int* __restrict__ rowbase,
    const unsigned* __restrict__ rowslot,
    int n, int cols,
    unsigned long long* __restrict__ okey,
    unsigned* __restrict__ oidx,
    int* __restrict__ flags)                         // [3]（[2]=行溢出标志）
{
    const int gy = blockIdx.x;
    const int beg = rowbase[gy];
    const int cnt = rowbase[gy + 1] - beg;
    if (cnt == 0) return;
    if (cnt > cols) { atomicOr(&flags[2], 1); return; }

    __shared__ unsigned s_i[2048];
    for (int k = threadIdx.x; k < cnt && k < 2048; k += blockDim.x)
        s_i[k] = rowslot[beg + k];
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int a = 1; a < cnt; ++a) {
            const unsigned v = s_i[a];
            const unsigned long long vk = ckeys[v];
            int b = a - 1;
            while (b >= 0) {
                const unsigned long long uk = ckeys[s_i[b]];
                // (cx,rrid,i) 全序：先比低 34bit（cx+rrid），等则比原始 i
                const unsigned long long ku = uk & 0x3FFFFFFFFULL;   // bits[0,33]=cx|rrid
                const unsigned long long kv2 = vk & 0x3FFFFFFFFULL;
                if (ku > kv2 || (ku == kv2 && s_i[b] > v)) {
                    s_i[b + 1] = s_i[b]; --b;
                } else break;
            }
            s_i[b + 1] = v;
        }
        for (int k = 0; k < cnt; ++k) {
            const unsigned src = s_i[k];
            okey[beg + k] = ckeys[src];
            oidx[beg + k] = cidx_a[src];
        }
    }
}

// 排序键数组上按 cell 下界二分：首个 ≥ (cell<<22) 的位置（~22 步）
__device__ __forceinline__ int DedupCellLowerBound(
    const unsigned long long* __restrict__ keys, int n, unsigned long long target) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 行带扫描定位（D2/D3 共用）：对每个 gy 行带 [gx_lo,gx_hi] 一次下界二分＋线性扫到
// 带尾（cell=1px² 且每像素 ≤1 点 → 带内 ≤ 带宽个点）——替代逐 cell 49/25 次二分。
// 行带尾 cell=((gy<<12)|(gx_hi+1))：gx_hi=4095 时恰为下一行首，break 条件仍正确
__device__ __forceinline__ void DedupBandRange(const unsigned long long* __restrict__ ckeys,
                                               int n, int gy, int gx_lo, int gx_hi,
                                               int& k_beg, unsigned& band_end_cell) {
    const unsigned long long bandLo =
        ((unsigned long long)((((unsigned)gy) << kDedupCellBits) | (unsigned)gx_lo)
         << kDedupRidxBits);
    band_end_cell = (((unsigned)gy) << kDedupCellBits) | (unsigned)(gx_hi + 1);
    k_beg = DedupCellLowerBound(ckeys, n, bandLo);
}

// D2 邻域 PCA 切向：±3 格（7×7）窗收 d²≤2.5² 候选，在线维护按 (d²,rrid) 全序的
// top-10（插入排序——劣于当前第 10 的候选可安全丢弃，无需全量收集），
// 定序累加 2×2 协方差 → 解析特征 → 各向异性门（CPU 严格式）→ 符号规约切向。
// 切向/valid 按 D1 排序位索引（非 rrid——round 恢复有半 ulp 撞号洞，见 D1 注）；
// 自跳过按排序位 k==i（撞号对方不再被误丢，更忠实 CPU kNN 语义）。
// 候选几何上界 20（半径 2.5 圆盘含自己 ≤21 格点、每像素≤1 点），
// top-10 在线法天然免上界约束
__global__ void DedupTangentKernel(const unsigned long long* __restrict__ ckeys,
                                    const DedupPt* __restrict__ cvals,
                                    const int* np, int cap,
                                    float2* __restrict__ tan_, char* __restrict__ tan_valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) return;
    const DedupPt me = cvals[i];
    const int cx = (int)floorf(me.px), cy = (int)floorf(me.py);
    const float r2 = kDedupNbrRadius * kDedupNbrRadius;

    float top_d2[kDedupNbrMax], top_dx[kDedupNbrMax], top_dy[kDedupNbrMax];
    int top_rid[kDedupNbrMax];
    int cnt = 0;
    const int gx_lo = max(cx - 3, 0), gx_hi = min(cx + 3, (1 << kDedupCellBits) - 1);
    const int gy_lo = max(cy - 3, 0), gy_hi = min(cy + 3, (1 << kDedupCellBits) - 1);
    for (int gy = gy_lo; gy <= gy_hi; ++gy) {
        int k0; unsigned bandEnd;
        DedupBandRange(ckeys, n, gy, gx_lo, gx_hi, k0, bandEnd);
        for (int k = k0; k < n; ++k) {
            if ((unsigned)(ckeys[k] >> kDedupRidxBits) >= bandEnd) break;
            int rid = (int)(ckeys[k] & ((1u << kDedupRidxBits) - 1));
            if (k == i) continue;   // 自跳过按排序位（rrid 撞号对方不再误丢）
            float dx = cvals[k].px - me.px, dy = cvals[k].py - me.py;
            float d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            if (cnt == kDedupNbrMax &&
                (d2 > top_d2[kDedupNbrMax - 1] ||
                 (d2 == top_d2[kDedupNbrMax - 1] && rid >= top_rid[kDedupNbrMax - 1])))
                continue;   // 劣于当前第 10 → 丢弃（top-10 只会更好，安全）
            int p = cnt < kDedupNbrMax ? cnt++ : kDedupNbrMax - 1;
            while (p > 0 && (top_d2[p - 1] > d2 ||
                             (top_d2[p - 1] == d2 && top_rid[p - 1] > rid))) {
                top_d2[p] = top_d2[p - 1]; top_rid[p] = top_rid[p - 1];
                top_dx[p] = top_dx[p - 1]; top_dy[p] = top_dy[p - 1];
                --p;
            }
            top_d2[p] = d2; top_rid[p] = rid; top_dx[p] = dx; top_dy[p] = dy;
        }
    }

    if (cnt < 2) { tan_valid[i] = 0; return; }
    const int take = cnt;
    float sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int k = 0; k < take; ++k) {   // 定序累加（top 数组已按 (d²,rrid) 全序）
        sx += top_dx[k]; sy += top_dy[k];
        sxx += top_dx[k] * top_dx[k];
        syy += top_dy[k] * top_dy[k];
        sxy += top_dx[k] * top_dy[k];
    }
    const float mx = sx / take, my = sy / take;
    const float vxx = sxx / take - mx * mx, vyy = syy / take - my * my;
    const float vxy = sxy / take - mx * my;
    const float tr = vxx + vyy, det = vxx * vyy - vxy * vxy;
    const float disc = tr * tr * 0.25f - det;
    const float sq = sqrtf(fmaxf(0.f, disc));
    const float l1 = tr * 0.5f + sq, l2 = tr * 0.5f - sq;
    if (l1 <= 1e-6f || l1 < 4.f * fmaxf(l2, 1e-9f)) { tan_valid[i] = 0; return; }
    const float ang = 0.5f * atan2f(2.f * vxy, vxx - vyy);
    float tx = cosf(ang), ty = sinf(ang);
    if (tx < 0.f || (tx == 0.f && ty < 0.f)) { tx = -tx; ty = -ty; }
    tan_[i] = make_float2(tx, ty);
    tan_valid[i] = 1;
}

// D3 位移门配对（对称边）：±2 格（5×5）窗内，仅 rrid 小者（规范化对门方——CPU
// j>i 单侧求值口径的确定性规范化，设计 §3 良性偏差）以自己的切向求值；过门 →
// 边对称写入双方邻接行（atomicAdd 槽位，行内序不定——传播取 min 序无关，确定性
// 保持）。行满 → flags[0] 溢出（几何上界 8：半径 1.6 圆盘含自己 ≤9 格点、
// 每像素≤1 点；取 3 倍余量=24，防御门）。
// 切向/valid 按 D1 排序位索引；自跳过 k==i；对门方规则用 >=——等 rrid 病态对
// 双侧一致弃权（键重复时该对直通，无害且确定）。
// ⚠ 不可"各写各的"单侧行：pull 传播只沿行持有方拉取，单向边会把本应同簇的点
// 留在不同分量（CPU 并查集 union 天然对称——GPU 必须显式双边写）
__global__ void DedupPairEdgeKernel(const unsigned long long* __restrict__ ckeys,
                                     const DedupPt* __restrict__ cvals,
                                     const float2* __restrict__ tan_,
                                     const char* __restrict__ tan_valid,
                                     const int* np, int cap,
                                     int* __restrict__ adj, int* __restrict__ rowlen,
                                     int* __restrict__ flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) return;
    const DedupPt me = cvals[i];
    if (!tan_valid[i]) return;
    const int cx = (int)floorf(me.px), cy = (int)floorf(me.py);
    const float2 t = tan_[i];
    const float pd2 = kDedupPairDist * kDedupPairDist;
    const int gx_lo = max(cx - 2, 0), gx_hi = min(cx + 2, (1 << kDedupCellBits) - 1);
    const int gy_lo = max(cy - 2, 0), gy_hi = min(cy + 2, (1 << kDedupCellBits) - 1);
    for (int gy = gy_lo; gy <= gy_hi; ++gy) {
        int k0; unsigned bandEnd;
        DedupBandRange(ckeys, n, gy, gx_lo, gx_hi, k0, bandEnd);
        for (int k = k0; k < n; ++k) {
            if ((unsigned)(ckeys[k] >> kDedupRidxBits) >= bandEnd) break;
            int rid = (int)(ckeys[k] & ((1u << kDedupRidxBits) - 1));
            if (k == i || !tan_valid[k]) continue;   // 自跳过按排序位＋valid 按排序位
            if (me.rrid >= rid) continue;   // 对门方=rrid 小者；等 rrid 双侧弃权（确定）
            float dx = cvals[k].px - me.px, dy = cvals[k].py - me.py;
            if (dx * dx + dy * dy > pd2) continue;
            if (fabsf(dx * t.x + dy * t.y) > kDedupAlongGate) continue;
            if (fabsf(-dx * t.y + dy * t.x) > kDedupNormalGate) continue;
            const float2 tj = tan_[k];
            if (fabsf(t.x * tj.x + t.y * tj.y) < kDedupTangentDot) continue;
            int si = atomicAdd(&rowlen[i], 1);
            if (si < kDedupAdjStride) adj[(size_t)i * kDedupAdjStride + si] = k;
            else atomicOr(&flags[0], 1);
            int sj = atomicAdd(&rowlen[k], 1);
            if (sj < kDedupAdjStride) adj[(size_t)k * kDedupAdjStride + sj] = i;
            else atomicOr(&flags[0], 1);
        }
    }
}

// D4 初始化（吸收 3 次 memset）：lab[i]=i（下标域——cur[cur[i]] 索引跳变合法性
// 来源）；rootkey[i]=+∞ 哨兵（0x7F7F7F7F 与原 0x7F 字节型 memset 同值）；
// 线程 0 清 flags[0..1]/dedup_count（消费均在流后）＋置 flags[3]＝实用上限
// 溢出标志（真 N > cap 时宿主帧尾回退真 N 重跑——保结果逐位正确）
__global__ void DedupInitKernel(int* lab, int* rootkey, int* flags,
                                int* dedup_count, const int* np, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i == 0) {
        flags[0] = 0; flags[1] = 0; *dedup_count = 0;
        flags[3] = (*np > cap) ? 1 : 0;
    }
    if (i >= n) return;
    lab[i] = i;
    rootkey[i] = 0x7F7F7F7F;
}

// D4 传播（pull+shortcut，纯 pull 无原子——min 幂等、序无关；双缓冲 ping-pong）：
// next[i]=min(cur[i], cur[cur[i]], min_{j∈adj行(i)} cur[j])——每轮可达距离 ≥ 翻倍。
// 邻接行以 rowlen[i] 为界（D3 对称写、atomic 槽位，行内序不定但 min 序无关）
__global__ void DedupPropagateKernel(const int* __restrict__ cur, int* __restrict__ next,
                                      const int* __restrict__ adj, const int* __restrict__ rowlen,
                                      const int* np, int cap,
                                      int* __restrict__ flags /*非空=收敛检查轮*/) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) return;
    int m = cur[i];
    const int c = cur[m];              // shortcut（值为下标，合法）
    if (c < m) m = c;
    const int len = min(rowlen[i], kDedupAdjStride);
    const int* row = adj + (size_t)i * kDedupAdjStride;
    for (int k = 0; k < len; ++k) {
        const int j = row[k];
        if (cur[j] < m) m = cur[j];
    }
    next[i] = m;
    if (flags != nullptr && m != cur[i]) atomicOr(&flags[1], 1);
}

// 收敛后：root=cur[i]；簇键=分量最小 rrid（atomicMin 结果序无关、唯一——跨运行规范化）
__global__ void DedupRootKeyKernel(const int* __restrict__ cur,
                                    const DedupPt* __restrict__ cvals,
                                    const int* np, int cap, int* __restrict__ rootkey) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) return;
    atomicMin(&rootkey[cur[i]], cvals[i].rrid);
}

// D5 簇排序打包：key=(clusterKey<<22)|rrid（clusterKey=rootkey[cur[i]]，Relabel
// 语义内联）；载荷=u32 下标（D6 经 lidx 间接读 cval_b——省 16B 载荷排序）
__global__ void DedupClusterPackKernel(const DedupPt* __restrict__ cvals,
                                        const int* __restrict__ cur,
                                        const int* __restrict__ rootkey,
                                        const int* np, int cap,
                                        unsigned long long* keys, unsigned* idxs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap) return;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) {
        // 哨兵垫（44bit 全 1 → 排后垫尾；载荷=自身下标，确定）
        keys[i] = 0xFFFFFFFFFFFULL;
        idxs[i] = (unsigned)i;
        return;
    }
    keys[i] = ((unsigned long long)(unsigned)rootkey[cur[i]] << kDedupRidxBits)
            | (unsigned long long)(unsigned)cvals[i].rrid;
    idxs[i] = (unsigned)i;
}

// D6 段边界标志：flag[i]=1 表示 i 是其所在段末位（i 与 i+1 分属不同簇）。
// ExclusiveSum 后 seg[i]=i 之前的边界数=段号（0 基）——⚠ 不可把标志打在"段首"：
// 排他求和不含自身，段首会拿到前一段的段号（v1 隐患：段错位在合成小图上恰好
// 观测不出，真实大 N 下段与簇完全错配）
__global__ void DedupSegBoundaryKernel(const unsigned long long* __restrict__ keys,
                                        const int* np, int cap,
                                        int* seg_id, int* __restrict__ seg_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap) return;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n) {
        // pad 定值（排他扫描输入确定性；SegCount/SegEmit 均按 n 守卫不读 pad）
        seg_id[i] = 0;
        seg_cnt[i] = 0;
        return;
    }
    seg_id[i] = (i == n - 1 || (keys[i] >> kDedupRidxBits) != (keys[i + 1] >> kDedupRidxBits))
                ? 1 : 0;
    seg_cnt[i] = 0;   // 顺带清段计数（原独立 memset 吸收——SegCount 只写段首）
}

// 段首判据（kernel 内复用）：i==0 或段号变化
__device__ __forceinline__ bool DedupIsSegStart(const int* seg, int i) {
    return i == 0 || seg[i] != seg[i - 1];
}
// 段尾+1（段首线程向前走）
__device__ __forceinline__ int DedupSegEnd(const int* seg, int i, int n) {
    int e = i + 1;
    while (e < n && seg[e] == seg[i]) ++e;
    return e;
}

// 段成员读取（D6）：排序后位置 k 的点 = cvals[lidx[k]]（间接一层）
__device__ __forceinline__ DedupPt DedupMember(const DedupPt* __restrict__ cvals,
                                               const unsigned* __restrict__ lidx, int k) {
    return cvals[lidx[k]];
}

// 段合并资格：2≤size≤8 ∧ label 全一致 ∧ 全员切向 valid ∧ 法向跨度≤2.5
// （切向取段首=簇内最小 rrid 成员——CPU mem[0] 口径的确定性规范化；
//   切向按 D1 排序位索引——段首成员的排序位即 lidx[i]）
__device__ __forceinline__ bool DedupMergeEligible(const DedupPt* v,
                                                   const unsigned* __restrict__ lidx,
                                                   const float2* __restrict__ tan_,
                                                   const char* __restrict__ tan_valid,
                                                   int i, int end) {
    int size = end - i;
    if (size < 2 || size > kDedupClusterMax) return false;
    const DedupPt h = DedupMember(v, lidx, i);
    const int lb = h.label;
    if (!tan_valid[lidx[i]]) return false;
    const float2 t = tan_[lidx[i]];
    float nmin = 1e9f, nmax = -1e9f;
    for (int k = i; k < end; ++k) {
        const DedupPt m = DedupMember(v, lidx, k);
        if (m.label != lb || !tan_valid[lidx[k]]) return false;
        const float pr = -m.px * t.y + m.py * t.x;
        nmin = fminf(nmin, pr);
        nmax = fmaxf(nmax, pr);
    }
    return (nmax - nmin) <= kDedupSpanMax;
}

__global__ void DedupSegCountKernel(const DedupPt* v, const unsigned* __restrict__ lidx,
                                     const float2* __restrict__ tan_,
                                     const char* __restrict__ tan_valid,
                                     const int* seg, const int* np, int cap, int* seg_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n || !DedupIsSegStart(seg, i)) return;
    int end = DedupSegEnd(seg, i, n);
    seg_cnt[seg[i]] = DedupMergeEligible(v, lidx, tan_, tan_valid, i, end) ? 1 : (end - i);
}

__global__ void DedupSegEmitKernel(const DedupPt* v, const unsigned* __restrict__ lidx,
                                    const float2* __restrict__ tan_, const int* seg,
                                    const int* np, int cap,
                                    const int* seg_cnt, const int* seg_base,
                                    FastPoint* out, int* out_count,
                                    float2* __restrict__ coords, int* __restrict__ ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = min(*np, cap);   // 【计数设备侧化】
    if (i >= n || !DedupIsSegStart(seg, i)) return;
    const int sid = seg[i];
    const int end = DedupSegEnd(seg, i, n);
    const int size = end - i;
    const int base = seg_base[sid];
    if (seg_cnt[sid] == 1 && size >= 2) {      // 合并段：1 点
        const DedupPt h = DedupMember(v, lidx, i);
        const float2 t = tan_[lidx[i]];        // 段首切向（按 D1 排序位索引）
        float px, py;
        if (size == 2) {
            const DedupPt g = DedupMember(v, lidx, i + 1);
            px = 0.5f * (h.px + g.px);
            py = 0.5f * (h.py + g.py);
        } else {                                // ≥3：按法向投影插入排序取中位
            float ns[kDedupClusterMax], xs[kDedupClusterMax], ys[kDedupClusterMax];
            for (int k = 0; k < size; ++k) {
                const DedupPt m = DedupMember(v, lidx, i + k);
                ns[k] = -m.px * t.y + m.py * t.x;
                xs[k] = m.px; ys[k] = m.py;
            }
            for (int a = 1; a < size; ++a) {    // size≤8，寄存器插入排序
                float kn = ns[a], kx = xs[a], ky = ys[a]; int b = a - 1;
                while (b >= 0 && ns[b] > kn) {
                    ns[b + 1] = ns[b]; xs[b + 1] = xs[b]; ys[b + 1] = ys[b]; --b;
                }
                ns[b + 1] = kn; xs[b + 1] = kx; ys[b + 1] = ky;
            }
            const int h2 = size / 2;
            if (size & 1) { px = xs[h2]; py = ys[h2]; }
            else { px = 0.5f * (xs[h2 - 1] + xs[h2]); py = 0.5f * (ys[h2 - 1] + ys[h2]); }
        }
        FastPoint p; p.label = h.label; p.px = px; p.py = py;
        out[base] = p;
        if (coords) { coords[base] = make_float2(px, py); ids[base] = h.label; }
        atomicAdd(out_count, 1);
    } else {                                    // 直通段：原样逐点
        for (int k = 0; k < size; ++k) {
            const DedupPt m = DedupMember(v, lidx, i + k);
            FastPoint p; p.label = m.label; p.px = m.px; p.py = m.py;
            out[base + k] = p;
            if (coords) { coords[base + k] = make_float2(m.px, m.py); ids[base + k] = m.label; }
            atomicAdd(out_count, 1);
        }
    }
}

// ============================================================================
// Impl
// ============================================================================

StegerExtractorFast::Impl::Impl(const StegerParams& params)
    : params_(params), old_device_id_(params.deviceId)
{
    params_.validate();

    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA-capable GPU found");
    }

    if (params_.deviceId >= device_count) {
        throw std::invalid_argument(
            "StegerParams::deviceId=" + std::to_string(params_.deviceId)
            + " exceeds available device count=" + std::to_string(device_count));
    }
    cv::cuda::setDevice(params_.deviceId);

    buildGaussianKernels();
    createEvents();
    bench_ = (std::getenv("JMW_STEGTIME") != nullptr);
    {   // JMW_STEG_ROWSORT=1 → 开 D1 行桶排序（A/B；默认关——锁频公平对比收益仅
        //    0.03ms/帧，gtest 合成图边界未过。基础设施保留供后续优化）
        const char* e = std::getenv("JMW_STEG_ROWSORT");
        rowsort_ = (e != nullptr && e[0] == '1');
    }
    {   // JMW_STEG_FUSED=0 → 关全融合 kernel（A/B 对拍开关；默认开）
        const char* e = std::getenv("JMW_STEG_FUSED");
        fused_ = !(e != nullptr && e[0] == '0');
    }
    {   // JMW_STEG_FLATFAST=0 → 关 Flat 直通快路（A/B 对拍开关；默认开）
        const char* e = std::getenv("JMW_STEG_FLATFAST");
        flatfast_ = !(e != nullptr && e[0] == '0');
    }
}

StegerExtractorFast::Impl::~Impl() {
    Destroy();
}

int StegerExtractorFast::Impl::computeKernelSize() const {
    if (params_.kernelSize != 0) return params_.kernelSize;
    int ks = static_cast<int>(std::ceil(params_.sigma * 3.0f)) * 2 + 1;
    return std::max(3, std::min(ks, 31));
}

void StegerExtractorFast::Impl::buildGaussianKernels() {
    actualKernelSize_ = computeKernelSize();
    int half = actualKernelSize_ / 2;
    float sigma2 = params_.sigma * params_.sigma;

    std::vector<float> h_g(actualKernelSize_);
    std::vector<float> h_gx(actualKernelSize_);
    std::vector<float> h_gxx(actualKernelSize_);

    float g_sum = 0.0f;
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_g[i] = std::exp(-x * x / (2.0f * sigma2));
        g_sum += h_g[i];
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        h_g[i] /= g_sum;
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_gx[i] = x / sigma2 * h_g[i];
    }
    for (int i = 0; i < actualKernelSize_; ++i) {
        float x = static_cast<float>(i - half);
        h_gxx[i] = (x * x / sigma2 - 1.0f) / sigma2 * h_g[i];
    }

    if (!d_kg_) {
        cudaMalloc(&d_kg_, kMaxKernelSize * sizeof(float));
        cudaMalloc(&d_kgx_, kMaxKernelSize * sizeof(float));
        cudaMalloc(&d_kgxx_, kMaxKernelSize * sizeof(float));
    }
    cudaMemcpy(d_kg_, h_g.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kgx_, h_gx.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kgxx_, h_gxx.data(), actualKernelSize_ * sizeof(float), cudaMemcpyHostToDevice);

    CALIB_LOG_INFO("Gaussian kernels built: sigma={}, kernelSize={}", params_.sigma, actualKernelSize_);
}

void StegerExtractorFast::Impl::createEvents() {
    if (events_ready_) return;
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_row_);
    cudaEventCreate(&ev_col_);
    cudaEventCreate(&ev_hess_);
    cudaEventCreate(&ev_dedup_);
    cudaEventCreate(&ev_grp_);
    cudaEventCreate(&ev_out_);
    cudaEventCreate(&ev_end_);
    cudaEventCreate(&ev_d1_);      // 【批0】去重细分
    cudaEventCreate(&ev_d23_);
    cudaEventCreate(&ev_d4_);
    cudaEventCreate(&ev_d5_);
    events_ready_ = true;
}

void StegerExtractorFast::Impl::ensureBuffers(int rows, int cols) {
    if (d_tg_.empty() || d_tg_.cols != cols || d_tg_.rows != rows) {
        d_tg_.create(rows, cols, CV_32FC1);
        d_tgx_.create(rows, cols, CV_32FC1);
        d_tgxx_.create(rows, cols, CV_32FC1);
        d_ix_.create(rows, cols, CV_32FC1);
        d_iy_.create(rows, cols, CV_32FC1);
        d_ixx_.create(rows, cols, CV_32FC1);
        d_iyy_.create(rows, cols, CV_32FC1);
        d_ixy_.create(rows, cols, CV_32FC1);
    }

    int64_t cap = static_cast<int64_t>(rows) * cols;
    if (capacity_ < cap) {
        if (d_points_) cudaFree(d_points_);
        if (d_sorted_) cudaFree(d_sorted_);
        if (d_coords_) cudaFree(d_coords_);
        if (d_ids_) cudaFree(d_ids_);
        if (h_pts_) cudaFreeHost(h_pts_);
        if (d_ckey_a_) cudaFree(d_ckey_a_);
        if (d_ckey_b_) cudaFree(d_ckey_b_);
        if (d_cidx_a_) cudaFree(d_cidx_a_);
        if (d_cidx_b_) cudaFree(d_cidx_b_);
        if (d_cval_a_) cudaFree(d_cval_a_);
        if (d_cval_b_) cudaFree(d_cval_b_);
        if (d_tan_) cudaFree(d_tan_);
        if (d_tan_valid_) cudaFree(d_tan_valid_);
        if (d_adj_) cudaFree(d_adj_);
        if (d_rowlen_) cudaFree(d_rowlen_);
    if (d_rowcnt_) cudaFree(d_rowcnt_);
    if (d_rowcursor_) cudaFree(d_rowcursor_);
    if (d_rowslot_) cudaFree(d_rowslot_);
    rowslot_cap_ = 0;
        if (d_lab_a_) cudaFree(d_lab_a_);
        if (d_lab_b_) cudaFree(d_lab_b_);
        if (d_rootkey_) cudaFree(d_rootkey_);
        if (d_lkey_a_) cudaFree(d_lkey_a_);
        if (d_lkey_b_) cudaFree(d_lkey_b_);
        if (d_lidx_a_) cudaFree(d_lidx_a_);
        if (d_lidx_b_) cudaFree(d_lidx_b_);
        if (d_seg_id_) cudaFree(d_seg_id_);
        if (d_seg_scan_) cudaFree(d_seg_scan_);
        if (d_seg_cnt_) cudaFree(d_seg_cnt_);
        if (d_seg_base_) cudaFree(d_seg_base_);
        if (d_dedup_points_) cudaFree(d_dedup_points_);
        if (d_cub_temp_) { cudaFree(d_cub_temp_); d_cub_temp_ = nullptr; cub_temp_bytes_ = 0; }

        cudaError_t err = cudaMalloc(&d_points_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_points_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_points_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_sorted_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_sorted_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_sorted_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_coords_, cap * sizeof(float2));
        if (err != cudaSuccess) {
            d_coords_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_coords_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_ids_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_ids_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ids_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaHostAlloc(&h_pts_, cap * sizeof(FastPoint), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            h_pts_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaHostAlloc h_pts_ failed: ")
                                     + cudaGetErrorString(err));
        }
        // —— dedup v2 缓冲（按 capacity 一并分配）——
        err = cudaMalloc(&d_ckey_a_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_ckey_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ckey_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_ckey_b_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_ckey_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_ckey_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cidx_a_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_cidx_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cidx_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cidx_b_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_cidx_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cidx_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cval_a_, cap * sizeof(DedupPt));
        if (err != cudaSuccess) {
            d_cval_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cval_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_cval_b_, cap * sizeof(DedupPt));
        if (err != cudaSuccess) {
            d_cval_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cval_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_tan_, cap * sizeof(float2));
        if (err != cudaSuccess) {
            d_tan_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_tan_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_tan_valid_, cap * sizeof(char));
        if (err != cudaSuccess) {
            d_tan_valid_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_tan_valid_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_adj_, cap * static_cast<size_t>(kDedupAdjStride) * sizeof(int));
        if (err != cudaSuccess) {
            d_adj_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_adj_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_rowlen_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_rowlen_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_rowlen_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lab_a_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_lab_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lab_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lab_b_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_lab_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lab_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_rootkey_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_rootkey_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_rootkey_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lkey_a_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_lkey_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lkey_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lkey_b_, cap * sizeof(unsigned long long));
        if (err != cudaSuccess) {
            d_lkey_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lkey_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lidx_a_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_lidx_a_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lidx_a_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_lidx_b_, cap * sizeof(unsigned));
        if (err != cudaSuccess) {
            d_lidx_b_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_lidx_b_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_id_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_id_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_id_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_scan_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_scan_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_scan_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_cnt_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_cnt_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_cnt_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_seg_base_, cap * sizeof(int));
        if (err != cudaSuccess) {
            d_seg_base_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_seg_base_ failed: ")
                                     + cudaGetErrorString(err));
        }
        err = cudaMalloc(&d_dedup_points_, cap * sizeof(FastPoint));
        if (err != cudaSuccess) {
            d_dedup_points_ = nullptr;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_dedup_points_ failed: ")
                                     + cudaGetErrorString(err));
        }
        // CUB 临时空间：全部调用（两次 SortPairs＋一次 ExclusiveSum）stream 串行
        // 复用同一块，按 capacity 各空查询一次取 max
        size_t sort1_tmp = 0, sort2_tmp = 0, scan_tmp = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort1_tmp,
            (unsigned long long*)nullptr, (unsigned long long*)nullptr,
            (unsigned*)nullptr, (unsigned*)nullptr,
            static_cast<int>(cap), 0, 64, 0);
        cub::DeviceRadixSort::SortPairs(nullptr, sort2_tmp,
            (unsigned long long*)nullptr, (unsigned long long*)nullptr,
            (unsigned*)nullptr, (unsigned*)nullptr,
            static_cast<int>(cap), 0, 64, 0);
        cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp,
            (int*)nullptr, (int*)nullptr, static_cast<int>(cap), 0);
        cub_temp_bytes_ = std::max(std::max(sort1_tmp, sort2_tmp), scan_tmp);
        err = cudaMalloc(&d_cub_temp_, cub_temp_bytes_);
        if (err != cudaSuccess) {
            d_cub_temp_ = nullptr;
            cub_temp_bytes_ = 0;
            throw std::runtime_error(std::string("[10-StegerExtractorFast] cudaMalloc d_cub_temp_ failed: ")
                                     + cudaGetErrorString(err));
        }
        capacity_ = static_cast<int>(cap);
    }

    if (!d_count_) {
        cudaMalloc(&d_count_, sizeof(int));
        cudaMalloc(&d_reject_, sizeof(int));
        cudaMalloc(&d_hist_, kLabelBins * sizeof(int));
        cudaMalloc(&d_base_, kLabelBins * sizeof(int));
        cudaMalloc(&d_cursor_, kLabelBins * sizeof(int));
        cudaMalloc(&d_dedup_count_, sizeof(int));
        cudaMalloc(&d_flags_, 4 * sizeof(int));   // 【批1】[2]=行桶排序行溢出
        cudaHostAlloc(&h_count_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_reject_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_hist_, kLabelBins * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_base_, kLabelBins * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_dedup_count_, sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_flags_, 4 * sizeof(int), cudaHostAllocDefault);
    }

    // Flat 快路输出视图（capacity 变更后重建；非持有——包原始设备指针）
    if (d_out_cp_.empty() || d_out_cp_.cols != capacity_
        || static_cast<void*>(d_out_cp_.data) != static_cast<void*>(d_coords_)) {
        d_out_cp_ = cv::cuda::GpuMat(1, capacity_, CV_32FC2,
                                     static_cast<void*>(d_coords_));
        d_out_pl_ = cv::cuda::GpuMat(1, capacity_, CV_32SC1,
                                     static_cast<void*>(d_ids_));
    }
}

void StegerExtractorFast::Impl::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() pre-allocating: {}x{}", rows, cols);
    ensureBuffers(rows, cols);

    warmup_rows_ = rows;
    warmup_cols_ = cols;

#ifndef NDEBUG
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorFast] GPU OOM: Warmup validation failed: ")
            + cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[10-StegerExtractorFast] Warmup() CUDA error: ")
            + cudaGetErrorString(err));
    }
#endif

    warmed_up_ = true;
    CALIB_LOG_INFO("Warmup() completed successfully");
}

void StegerExtractorFast::Impl::SetParams(const StegerParams& params) {
#ifndef NDEBUG
    assert(!inProcess_.load() && "SetParams() called while Execute() is running - NOT thread-safe!");
#endif

    params_ = params;
    params_.validate();

    if (params_.deviceId != old_device_id_) {
        int device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (params_.deviceId >= device_count) {
            throw std::invalid_argument(
                "StegerParams::deviceId=" + std::to_string(params_.deviceId)
                + " exceeds available device count=" + std::to_string(device_count));
        }
        cv::cuda::setDevice(params_.deviceId);
        CALIB_LOG_INFO("SetParams() device switched: {} -> {}", old_device_id_, params_.deviceId);
        old_device_id_ = params_.deviceId;
    }

    buildGaussianKernels();

    CALIB_LOG_INFO("SetParams() updated: sigma={}, kernelSize={}, lowThreshold={}, highThreshold={}, maxLabels={}, deviceId={}",
                   params_.sigma, params_.kernelSize, params_.lowThreshold, params_.highThreshold,
                   params_.maxLabels, params_.deviceId);
}

void StegerExtractorFast::Impl::Destroy() {
    if (d_kg_) { cudaFree(d_kg_); d_kg_ = nullptr; }
    if (d_kgx_) { cudaFree(d_kgx_); d_kgx_ = nullptr; }
    if (d_kgxx_) { cudaFree(d_kgxx_); d_kgxx_ = nullptr; }
    if (d_points_) { cudaFree(d_points_); d_points_ = nullptr; }
    if (d_sorted_) { cudaFree(d_sorted_); d_sorted_ = nullptr; }
    if (d_coords_) { cudaFree(d_coords_); d_coords_ = nullptr; }
    if (d_ids_) { cudaFree(d_ids_); d_ids_ = nullptr; }
    if (d_count_) { cudaFree(d_count_); d_count_ = nullptr; }
    if (d_reject_) { cudaFree(d_reject_); d_reject_ = nullptr; }
    if (d_hist_) { cudaFree(d_hist_); d_hist_ = nullptr; }
    if (d_base_) { cudaFree(d_base_); d_base_ = nullptr; }
    if (d_cursor_) { cudaFree(d_cursor_); d_cursor_ = nullptr; }
    if (h_count_) { cudaFreeHost(h_count_); h_count_ = nullptr; }
    if (h_reject_) { cudaFreeHost(h_reject_); h_reject_ = nullptr; }
    if (h_hist_) { cudaFreeHost(h_hist_); h_hist_ = nullptr; }
    if (h_base_) { cudaFreeHost(h_base_); h_base_ = nullptr; }
    if (h_pts_) { cudaFreeHost(h_pts_); h_pts_ = nullptr; }
    if (d_ckey_a_) { cudaFree(d_ckey_a_); d_ckey_a_ = nullptr; }
    if (d_ckey_b_) { cudaFree(d_ckey_b_); d_ckey_b_ = nullptr; }
    if (d_cidx_a_) { cudaFree(d_cidx_a_); d_cidx_a_ = nullptr; }
    if (d_cidx_b_) { cudaFree(d_cidx_b_); d_cidx_b_ = nullptr; }
    if (d_cval_a_) { cudaFree(d_cval_a_); d_cval_a_ = nullptr; }
    if (d_cval_b_) { cudaFree(d_cval_b_); d_cval_b_ = nullptr; }
    if (d_tan_) { cudaFree(d_tan_); d_tan_ = nullptr; }
    if (d_tan_valid_) { cudaFree(d_tan_valid_); d_tan_valid_ = nullptr; }
    if (d_adj_) { cudaFree(d_adj_); d_adj_ = nullptr; }
    if (d_rowlen_) { cudaFree(d_rowlen_); d_rowlen_ = nullptr; }
    if (d_rowcnt_) { cudaFree(d_rowcnt_); d_rowcnt_ = nullptr; }
    if (d_rowcursor_) { cudaFree(d_rowcursor_); d_rowcursor_ = nullptr; }
    if (d_rowslot_) { cudaFree(d_rowslot_); d_rowslot_ = nullptr; }
    rowslot_cap_ = 0;
    if (d_lab_a_) { cudaFree(d_lab_a_); d_lab_a_ = nullptr; }
    if (d_lab_b_) { cudaFree(d_lab_b_); d_lab_b_ = nullptr; }
    if (d_rootkey_) { cudaFree(d_rootkey_); d_rootkey_ = nullptr; }
    if (d_lkey_a_) { cudaFree(d_lkey_a_); d_lkey_a_ = nullptr; }
    if (d_lkey_b_) { cudaFree(d_lkey_b_); d_lkey_b_ = nullptr; }
    if (d_lidx_a_) { cudaFree(d_lidx_a_); d_lidx_a_ = nullptr; }
    if (d_lidx_b_) { cudaFree(d_lidx_b_); d_lidx_b_ = nullptr; }
    if (d_seg_id_) { cudaFree(d_seg_id_); d_seg_id_ = nullptr; }
    if (d_seg_scan_) { cudaFree(d_seg_scan_); d_seg_scan_ = nullptr; }
    if (d_seg_cnt_) { cudaFree(d_seg_cnt_); d_seg_cnt_ = nullptr; }
    if (d_seg_base_) { cudaFree(d_seg_base_); d_seg_base_ = nullptr; }
    if (d_dedup_points_) { cudaFree(d_dedup_points_); d_dedup_points_ = nullptr; }
    if (d_dedup_count_) { cudaFree(d_dedup_count_); d_dedup_count_ = nullptr; }
    if (h_dedup_count_) { cudaFreeHost(h_dedup_count_); h_dedup_count_ = nullptr; }
    if (d_flags_) { cudaFree(d_flags_); d_flags_ = nullptr; }
    if (h_flags_) { cudaFreeHost(h_flags_); h_flags_ = nullptr; }
    if (cub_temp_bytes_ > 0 && d_cub_temp_) { cudaFree(d_cub_temp_); d_cub_temp_ = nullptr; }
    cub_temp_bytes_ = 0;
    d_out_cp_.release();
    d_out_pl_.release();
    d_tg_.release();
    d_tgx_.release();
    d_tgxx_.release();
    d_ix_.release();
    d_iy_.release();
    d_ixx_.release();
    d_iyy_.release();
    d_ixy_.release();
    if (events_ready_) {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_row_);
        cudaEventDestroy(ev_col_);
        cudaEventDestroy(ev_hess_);
        cudaEventDestroy(ev_dedup_);        cudaEventDestroy(ev_grp_);
        cudaEventDestroy(ev_out_);
        cudaEventDestroy(ev_end_);
        cudaEventDestroy(ev_d1_);           // 【批0】去重细分
        cudaEventDestroy(ev_d23_);
        cudaEventDestroy(ev_d4_);
        cudaEventDestroy(ev_d5_);
        events_ready_ = false;
    }
    capacity_ = 0;
    warmed_up_ = false;
    warmup_rows_ = 0;
    warmup_cols_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}

StegerResult StegerExtractorFast::Impl::extractFlat(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_binaryMask,
    cv::cuda::Stream& stream)
{
    return Execute(d_grayImage, d_binaryMask, true, stream);
}

StegerResult StegerExtractorFast::Impl::Execute(
    const cv::cuda::GpuMat& d_grayImage,
    const cv::cuda::GpuMat& d_mask,
    bool maskIs8U,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    assert(!inProcess_.load() && "Concurrent Execute() calls detected - NOT thread-safe!");

    struct ScopedFlag {
        std::atomic<bool>* flag;
        ScopedFlag(std::atomic<bool>* f) : flag(f) {}
        ~ScopedFlag() { flag->store(false); }
    };

    ScopedFlag guard(&inProcess_);
    inProcess_.store(true);
#endif

    StegerResult result;

    // 【150FPS·发射拆解】sub＝两同步点之间的提交耗时；wait＝同步等待（含内核
    // 执行）。JMW_STEGSUB=1 时逐 Execute 打印 stderr——验证墙钟-内核差额是否
    // 为 WDDM 变尺寸发射开销（v3 已证实 2.9ms 同病，此处同法取证）
    const bool subProbe_ = getenv("JMW_STEGSUB") != nullptr;
    double subMs_ = 0.0, waitMs_ = 0.0, propMs_ = 0.0; int propRounds_ = 0;
    double subA_ = 0.0, subB_ = 0.0;   // 【审核】段1（卷积+Hessian）/ 段2（去重链）分开记账
    double aMsetMs_ = 0.0, aLaunchMs_ = 0.0, aErrMs_ = 0.0;   // 【追凶】段A 四调用分账
    int subSeg_ = 0;
    auto subT0_ = std::chrono::steady_clock::now();
    auto syncMark_ = [&]() {
        // 【150FPS·仪表修正】先掐表后同步——原版先同步后掐表把"等 GPU 执行"
        // 全记进 sub 桶（wait 恒 0），造成"纯发射成本"伪影。修正后：
        // sub＝真提交；wait＝同步等待（含内核执行）
        if (subProbe_) {
            const auto a_ = std::chrono::steady_clock::now();
            const double seg = std::chrono::duration<double, std::milli>(a_ - subT0_).count();
            subMs_ += seg;
            if (subSeg_ == 0) subA_ += seg; else subB_ += seg;
            ++subSeg_;
            stream.waitForCompletion();
            const auto b_ = std::chrono::steady_clock::now();
            waitMs_ += std::chrono::duration<double, std::milli>(b_ - a_).count();
            subT0_ = b_;
        } else {
            stream.waitForCompletion();
        }
    };

    try {
        int rows = d_grayImage.rows;
        int cols = d_grayImage.cols;
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

        ensureBuffers(rows, cols);

        if (bench_) cudaEventRecord(ev_start_, cs);

        cudaError_t op_err = cudaSuccess;
        const void* mask_ptr = maskIs8U
            ? static_cast<const void*>(d_mask.ptr<uchar>())
            : static_cast<const void*>(d_mask.ptr<int>());

        if (fused_) {
            // === Step 1-4 全融合（P1）：Row3＋Col5＋Hessian/Taylor/宽度门一 kernel
            // ——8 张全图中间平面显存流量（~118MB/眼）归零＋空 tile 掩膜早退；
            // 数值逐位对齐三分 kernel 版（k 升序＋同款 guard＋renormalize 语义）===
            // 【150FPS·追凶】段 A 四调用逐个掐表：mset/launch/errck/cpyq
            const auto fxA_ = std::chrono::steady_clock::now();
            cudaMemsetAsync(d_count_, 0, sizeof(int), cs);
            cudaMemsetAsync(d_reject_, 0, sizeof(int), cs);
            const auto fxB_ = std::chrono::steady_clock::now();
            dim3 fused_block(kFuseTileX, kFuseTileY);
            dim3 fused_grid((cols + kFuseTileX - 1) / kFuseTileX,
                            (rows + kFuseTileY - 1) / kFuseTileY);
            if (maskIs8U) {
                FusedConvHessianKernel<true><<<fused_grid, fused_block, 0, cs>>>(
                    d_grayImage.ptr<uchar>(), d_grayImage.step,
                    mask_ptr, d_mask.step,
                    rows, cols,
                    d_kg_, d_kgx_, d_kgxx_, actualKernelSize_,
                    params_.lowThreshold, params_.highThreshold,
                    params_.maxLineWidth,
                    d_points_, d_count_, d_reject_, capacity_);
            } else {
                FusedConvHessianKernel<false><<<fused_grid, fused_block, 0, cs>>>(
                    d_grayImage.ptr<uchar>(), d_grayImage.step,
                    mask_ptr, d_mask.step,
                    rows, cols,
                    d_kg_, d_kgx_, d_kgxx_, actualKernelSize_,
                    params_.lowThreshold, params_.highThreshold,
                    params_.maxLineWidth,
                    d_points_, d_count_, d_reject_, capacity_);
            }
            const auto fxC_ = std::chrono::steady_clock::now();
            if (bench_) cudaEventRecord(ev_hess_, cs);
            op_err = cudaGetLastError();
            const auto fxD_ = std::chrono::steady_clock::now();
            if (subProbe_) {
                aMsetMs_ += std::chrono::duration<double, std::milli>(fxB_ - fxA_).count();
                aLaunchMs_ += std::chrono::duration<double, std::milli>(fxC_ - fxB_).count();
                aErrMs_ += std::chrono::duration<double, std::milli>(fxD_ - fxC_).count();
            }
            if (op_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUDA error in fused conv/Hessian kernel: ")
                                 + cudaGetErrorString(op_err);
                return result;
            }
        } else {
            dim3 conv_block(16, 16);
            dim3 conv_grid((cols + conv_block.x - 1) / conv_block.x,
                           (rows + conv_block.y - 1) / conv_block.y);

            // === Step 1+2a: uchar 直读 → 行级 3 输出 ===
            RowFan3Kernel<<<conv_grid, conv_block, 0, cs>>>(
                d_grayImage.ptr<uchar>(),
                d_tg_.ptr<float>(), d_tgx_.ptr<float>(), d_tgxx_.ptr<float>(),
                rows, cols, d_grayImage.step, d_tg_.step,
                d_kg_, d_kgx_, d_kgxx_, actualKernelSize_);

            if (bench_) cudaEventRecord(ev_row_, cs);

            // === Step 2b: 列级 5 输出导数 ===
            ColFan5Kernel<<<conv_grid, conv_block, 0, cs>>>(
                d_tg_.ptr<float>(), d_tgx_.ptr<float>(), d_tgxx_.ptr<float>(),
                d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                rows, cols, d_tg_.step, d_ix_.step,
                d_kg_, d_kgx_, d_kgxx_, actualKernelSize_);

            if (bench_) cudaEventRecord(ev_col_, cs);

            op_err = cudaGetLastError();
            if (op_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUDA error in convolution: ") + cudaGetErrorString(op_err);
                return result;
            }

            // === Step 3+4: Hessian + Taylor ===
            cudaMemsetAsync(d_count_, 0, sizeof(int), cs);
            cudaMemsetAsync(d_reject_, 0, sizeof(int), cs);

            if (maskIs8U) {
                HessianEigenAndTaylorKernel<true><<<conv_grid, conv_block, 0, cs>>>(
                    d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                    d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                    mask_ptr, d_mask.step,
                    d_tg_.ptr<float>(), d_tg_.step,
                    rows, cols,
                    d_ix_.step, d_ixx_.step,
                    params_.lowThreshold, params_.highThreshold,
                    params_.maxLineWidth,
                    d_points_, d_count_, d_reject_, capacity_);
            } else {
                HessianEigenAndTaylorKernel<false><<<conv_grid, conv_block, 0, cs>>>(
                    d_ix_.ptr<float>(), d_iy_.ptr<float>(),
                    d_ixx_.ptr<float>(), d_iyy_.ptr<float>(), d_ixy_.ptr<float>(),
                    mask_ptr, d_mask.step,
                    d_tg_.ptr<float>(), d_tg_.step,
                    rows, cols,
                    d_ix_.step, d_ixx_.step,
                    params_.lowThreshold, params_.highThreshold,
                    params_.maxLineWidth,
                    d_points_, d_count_, d_reject_, capacity_);
            }

            if (bench_) cudaEventRecord(ev_hess_, cs);

            op_err = cudaGetLastError();
            if (op_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUDA error in Hessian/Taylor kernel: ") + cudaGetErrorString(op_err);
                return result;
            }
        }

        // === Step 5+4.5 【计数设备侧化】真 N 不再中途回读——去重链按实用上限
        //     定容发射：内核从显存读计数自裁剪（哨兵垫两处排序尾、pad 定值保
        //     扫描确定性、传播按上限定轮——幂等收敛与轮次无关），帧尾一次合并
        //     回读＝本算子唯一同步点。真 N > 上限（本数据 ≤4.4 万 << 65536，
        //     未观测）由 flags[3] 捕获 → 宿主以真 N 重跑同一链（cap 参数化＝
        //     原逐 N 路径，逐位还原）。CUB 排序从真 N 变上限 N 为主要新增 GPU
        //     成本（~+0.3ms/眼）——换来消去同步排空，利用率口径净赚 ===
        FastPoint* src_points = d_points_;
        int point_count = 0;
        int point_count_pre = 0;
        bool dedup_ran = false;
        bool dedup_fallback = false;
        bool dedup_overflow = false;
        bool coords_ready = false;   // Flat：dedup 成功路径 SegEmit 已直出 coords/ids
        const int capLaunch = static_cast<int>(
            std::min<int64_t>(kPracDedupCap, capacity_));

        // 去重链整链提交（capN＝发射容量；正常帧＝capLaunch，溢出回退帧＝真 N）
        auto submitDedupChain_ = [&](int capN) -> cudaError_t {
            const int threads = 256;
            const int blocks = (capN + threads - 1) / threads;
            cudaError_t e = cudaSuccess;
            // D1 网格排序：(cell|rrid) 唯一键 → CUB RadixSort（稳定——valid 区
            // 相对序与逐 N 排序一致；哨兵垫排后垫尾）
            DedupCellPackKernel<<<blocks, threads, 0, cs>>>(
                d_points_, d_count_, capN, cols,
                d_ckey_a_, d_cidx_a_, d_cval_a_, d_rowlen_);
            size_t tmp = cub_temp_bytes_;
            e = cub::DeviceRadixSort::SortPairs(d_cub_temp_, tmp,
                d_ckey_a_, d_ckey_b_, d_cidx_a_, d_cidx_b_, capN, 0, 46, cs);
            if (e != cudaSuccess) return e;
            DedupGatherKernel<<<blocks, threads, 0, cs>>>(
                d_cidx_b_, d_cval_a_, d_cval_b_, d_count_, capN);
            if (bench_) cudaEventRecord(ev_d1_, cs);   // 【批0】D1 止
            // D2 邻域 PCA 几何切向（带域搜索按 n 截断——哨兵键 > 带界自然断）
            DedupTangentKernel<<<blocks, threads, 0, cs>>>(
                d_ckey_b_, d_cval_b_, d_count_, capN, d_tan_, d_tan_valid_);
            // D3 位移门配对 → 对称邻接行（行满置 flags[0]）
            DedupPairEdgeKernel<<<blocks, threads, 0, cs>>>(
                d_ckey_b_, d_cval_b_, d_tan_, d_tan_valid_, d_count_, capN,
                d_adj_, d_rowlen_, d_flags_);
            if (bench_) cudaEventRecord(ev_d23_, cs);  // 【批0】D2+D3 止
            // D4 初始化＋标签传播（定轮按 capN——上限轮数 ≥ 真 N 轮数，传播幂等
            // 收敛于不动点，输出与轮次无关；邻接溢出 flags[0] 帧尾合并回读）
            DedupInitKernel<<<blocks, threads, 0, cs>>>(
                d_lab_a_, d_rootkey_, d_flags_, d_dedup_count_, d_count_, capN);
            int* lab_cur = d_lab_a_;
            int* lab_nxt = d_lab_b_;
            int rounds = 1;
            while ((1 << rounds) < capN) ++rounds;
            ++rounds;   // +1 余量
            const auto propT0_ = std::chrono::steady_clock::now();
            for (int r = 0; r < rounds; ++r) {
                DedupPropagateKernel<<<blocks, threads, 0, cs>>>(
                    lab_cur, lab_nxt, d_adj_, d_rowlen_, d_count_, capN, nullptr);
                int* sw = lab_cur; lab_cur = lab_nxt; lab_nxt = sw;
            }
            if (subProbe_)
                propMs_ += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - propT0_).count();
            if (bench_) cudaEventRecord(ev_d4_, cs);   // 【批0】D4 止
            // 簇键=分量最小 rrid；D5 簇排序（44bit＋哨兵垫）
            DedupRootKeyKernel<<<blocks, threads, 0, cs>>>(
                lab_cur, d_cval_b_, d_count_, capN, d_rootkey_);
            DedupClusterPackKernel<<<blocks, threads, 0, cs>>>(
                d_cval_b_, lab_cur, d_rootkey_, d_count_, capN, d_lkey_a_, d_lidx_a_);
            size_t tmp2 = cub_temp_bytes_;
            e = cub::DeviceRadixSort::SortPairs(d_cub_temp_, tmp2,
                d_lkey_a_, d_lkey_b_, d_lidx_a_, d_lidx_b_, capN, 0, 44, cs);
            if (e != cudaSuccess) return e;
            if (bench_) cudaEventRecord(ev_d5_, cs);   // 【批0】D5 止
            // D6 段归约（ExclusiveSum 输入/输出不得别名——pad 段定值 0 保确定性）
            DedupSegBoundaryKernel<<<blocks, threads, 0, cs>>>(
                d_lkey_b_, d_count_, capN, d_seg_id_, d_seg_cnt_);
            size_t tmp3 = cub_temp_bytes_;
            e = cub::DeviceScan::ExclusiveSum(d_cub_temp_, tmp3,
                d_seg_id_, d_seg_scan_, capN, cs);
            if (e != cudaSuccess) return e;
            DedupSegCountKernel<<<blocks, threads, 0, cs>>>(
                d_cval_b_, d_lidx_b_, d_tan_, d_tan_valid_, d_seg_scan_,
                d_count_, capN, d_seg_cnt_);
            size_t tmp4 = cub_temp_bytes_;
            e = cub::DeviceScan::ExclusiveSum(d_cub_temp_, tmp4,
                d_seg_cnt_, d_seg_base_, capN, cs);
            if (e != cudaSuccess) return e;
            DedupSegEmitKernel<<<blocks, threads, 0, cs>>>(
                d_cval_b_, d_lidx_b_, d_tan_, d_seg_scan_, d_count_, capN,
                d_seg_cnt_, d_seg_base_, d_dedup_points_, d_dedup_count_,
                (flatfast_ && maskIs8U) ? d_coords_ : static_cast<float2*>(nullptr),
                (flatfast_ && maskIs8U) ? d_ids_ : static_cast<int*>(nullptr));
            return cudaGetLastError();
        };

        if (params_.dedupMode != 0) {
            if (static_cast<int64_t>(rows) * cols >= (1 << 22)
                || rows > 4096 || cols > 4096) {
                result.success = false;
                result.message = "dedup: image exceeds key-packing limits "
                                 "(dims<=4096, pixels<4M)";
                return result;
            }
            // rowsort 行桶路径已退役（260831 裁决默认关；上限发射与行桶宿主 N
            // 参数互斥——诊断走 JMW_STEG_ROWSORT 不再生效，内核保留存档）
            op_err = submitDedupChain_(capLaunch);
            if (op_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUDA error in dedup chain: ")
                                 + cudaGetErrorString(op_err);
                return result;
            }
            if (bench_) cudaEventRecord(ev_dedup_, cs);
            dedup_ran = true;
            // 帧尾一次合并回读（真 N＋N'＋flags×4）——本算子唯一同步点
            cudaMemcpyAsync(h_count_, d_count_, sizeof(int), cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(h_dedup_count_, d_dedup_count_, sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(h_flags_, d_flags_, 4 * sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            if (bench_)
                cudaMemcpyAsync(h_reject_, d_reject_, sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
            syncMark_();
            if (h_flags_[3]) {
                // 实用上限溢出（未观测）——以真 N 重跑同一链（cap 参数化＝原逐 N
                // 语义，逐位还原）后再回读
                const int trueN = std::min(*h_count_, capacity_);
                op_err = submitDedupChain_(trueN);
                if (op_err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUDA error in dedup chain: ")
                                     + cudaGetErrorString(op_err);
                    return result;
                }
                cudaMemcpyAsync(h_dedup_count_, d_dedup_count_, sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
                cudaMemcpyAsync(h_flags_, d_flags_, 4 * sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
                syncMark_();
            }
            point_count_pre = std::min(*h_count_, capacity_);
            if (bench_) CALIB_LOG_INFO("Extracted {} subpixel points{}", point_count_pre,
                           params_.maxLineWidth > 0.0f
                               ? std::string(" (widthGate rejected ")
                                     + std::to_string(*h_reject_) + ")"
                               : std::string());
            dedup_overflow = (h_flags_[0] != 0);
            if (dedup_overflow) {
                // 邻接溢出 → 本帧保守直通（坐标零改动、mergedCount=0）
                dedup_fallback = true;
                src_points = d_points_;
                point_count = point_count_pre;
                CALIB_LOG_WARN("dedup fallback (adjacency overflow): "
                               "frame passed through unmerged");
            } else {
                src_points = d_dedup_points_;
                point_count = *h_dedup_count_;
                result.mergedCount = point_count_pre - point_count;
                coords_ready = flatfast_ && maskIs8U;
                CALIB_LOG_DEBUG("Extracted {} subpixel points after dedup (merged {})",
                                point_count, result.mergedCount);
                if (point_count == 0) {
                    result.success = true;
                    result.message = point_count_pre == 0
                        ? "No subpixel points extracted"
                        : "No subpixel points after dedup";
                    result.qualityFlag = QualityFlag::Warning;
                    return result;
                }
            }
        } else {
            // dedupMode==0（外挂 CPU oracle 对照路径）：需 host 计数——保留原回读
            cudaMemcpyAsync(h_count_, d_count_, sizeof(int), cudaMemcpyDeviceToHost, cs);
            if (bench_)
                cudaMemcpyAsync(h_reject_, d_reject_, sizeof(int),
                                cudaMemcpyDeviceToHost, cs);
            syncMark_();
            point_count = std::min(*h_count_, capacity_);
            point_count_pre = point_count;
            if (bench_) CALIB_LOG_INFO("Extracted {} subpixel points{}", point_count,
                           params_.maxLineWidth > 0.0f
                               ? std::string(" (widthGate rejected ")
                                     + std::to_string(*h_reject_) + ")"
                               : std::string());
            if (point_count == 0) {
                result.success = true;
                result.message = "No subpixel points extracted";
                result.qualityFlag = calib::QualityFlag::Warning;
                return result;
            }
        }

        // === Step 6: 分组＋双输出组装 ===
        // Flat 直通快路（flatfast_ && mask8：label 恒 1）——hist/CPU 前缀和/scatter/
        // 每帧 cudaMalloc＋D2D 全免，输出＝Impl 常驻缓冲的 [0,N') ROI 视图。
        // 点序＝源序（基线 scatter 为原子序）——点集多重集逐位等价；整帧翻转类
        // 抖动经双版本对照实验证实为链条固有（与点序无关，见 flatfast_ 注）。
        // ByLabel（32S 标号掩膜）恒走原直方图分组路径。
        if (flatfast_ && maskIs8U) {
            if (!coords_ready) {   // dedup 成功路径 SegEmit 已直出 coords/ids——免拷
                int threads = 256;
                int blocks = (point_count + threads - 1) / threads;
                FlatCopyOutKernel<<<blocks, threads, 0, cs>>>(
                    src_points, point_count, d_coords_, d_ids_);
            }
            result.d_centerPoints = std::make_shared<cv::cuda::GpuMat>(
                d_out_cp_, cv::Rect(0, 0, point_count, 1));
            result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(
                d_out_pl_, cv::Rect(0, 0, point_count, 1));

            // 【速度优化】非导出帧跳过 h_pts_ D2H＋同步（下游只吃 GPU 视图）
            if (collectHostPoints_) {
                cudaMemcpyAsync(h_pts_, src_points, point_count * sizeof(FastPoint),
                                cudaMemcpyDeviceToHost, cs);
            }
            if (bench_) cudaEventRecord(ev_grp_, cs);
            if (collectHostPoints_) {
                stream.waitForCompletion();
            }
        } else {
            cudaMemsetAsync(d_hist_, 0, kLabelBins * sizeof(int), cs);
            {
                int threads = 256;
                int blocks = (point_count + threads - 1) / threads;
                LabelHistKernel<<<blocks, threads, 0, cs>>>(src_points, point_count, kLabelBins, d_hist_);
            }
            cudaMemcpyAsync(h_hist_, d_hist_, kLabelBins * sizeof(int), cudaMemcpyDeviceToHost, cs);
            stream.waitForCompletion();

            int running = 0;
            for (int lb = 0; lb < kLabelBins; ++lb) {
                h_base_[lb] = running;
                running += h_hist_[lb];
            }

            cudaMemcpyAsync(d_base_, h_base_, kLabelBins * sizeof(int), cudaMemcpyHostToDevice, cs);
            cudaMemsetAsync(d_cursor_, 0, kLabelBins * sizeof(int), cs);
            {
                int threads = 256;
                int blocks = (point_count + threads - 1) / threads;
                LabelScatterKernel<<<blocks, threads, 0, cs>>>(
                    src_points, point_count, kLabelBins, d_base_, d_cursor_,
                    d_sorted_, d_coords_, d_ids_);
            }
            if (bench_) cudaEventRecord(ev_grp_, cs);

            // === Step 6（ByLabel）: 单次 D2H + D2D 组装 CPU/GPU 双输出 ===
            auto d_cp = std::make_shared<cv::cuda::GpuMat>();
            d_cp->create(1, point_count, CV_32FC2);
            cudaMemcpyAsync(d_cp->ptr<float2>(), d_coords_, point_count * sizeof(float2),
                            cudaMemcpyDeviceToDevice, cs);

            auto d_pl = std::make_shared<cv::cuda::GpuMat>();
            d_pl->create(1, point_count, CV_32SC1);
            cudaMemcpyAsync(d_pl->ptr<int>(), d_ids_, point_count * sizeof(int),
                            cudaMemcpyDeviceToDevice, cs);

            cudaMemcpyAsync(h_pts_, d_sorted_, point_count * sizeof(FastPoint),
                            cudaMemcpyDeviceToHost, cs);
            if (bench_) cudaEventRecord(ev_out_, cs);
            if (collectHostPoints_) {
                stream.waitForCompletion();
            }

            result.d_centerPoints = d_cp;
            result.d_line_ids = d_pl;
        }

        // 【速度优化】非导出帧跳过 centerPoints 构建（省 CPU 拷贝 ~0.2ms/眼）
        if (collectHostPoints_) {
            std::map<int, std::vector<cv::Point2f>> centerPoints;
            if (flatfast_ && maskIs8U) {
                std::vector<cv::Point2f> v(point_count);
                for (int i = 0; i < point_count; ++i) {
                    v[i] = cv::Point2f(h_pts_[i].px, h_pts_[i].py);
                }
                centerPoints.emplace(1, std::move(v));
            } else {
                for (int lb = 1; lb < kLabelBins; ++lb) {
                    int n = h_hist_[lb];
                    if (n <= 0) continue;
                    const FastPoint* src = h_pts_ + h_base_[lb];
                    std::vector<cv::Point2f> v(n);
                    for (int i = 0; i < n; ++i) {
                        v[i] = cv::Point2f(src[i].px, src[i].py);
                    }
                    centerPoints.emplace(lb, std::move(v));
                }
            }
            result.centerPoints = std::move(centerPoints);
            result.lineCount = static_cast<int>(result.centerPoints.size());
        } else {
            result.lineCount = (point_count > 0) ? 1 : 0;   // Flat 路径单标签
        }

        if (bench_) cudaEventRecord(ev_end_, cs);

        if (bench_) {
            cudaEventSynchronize(ev_end_);   // bench 事件就绪后再取时长（免读零值）
            float ms_total = 0, ms_row = 0, ms_col = 0, ms_hess = 0, ms_grp = 0, ms_out = 0;
            float ms_dedup = 0;
            cudaEventElapsedTime(&ms_total, ev_start_, ev_end_);
            if (fused_) {
                // 融合核路径：Row3/Col5/Hessian 三步合一（ev_row_/ev_col_ 未记录，
                // 不可取时长——否则 cudaErrorInvalidResourceHandle 粘性污染下游）
                cudaEventElapsedTime(&ms_hess, ev_start_, ev_hess_);
            } else {
                cudaEventElapsedTime(&ms_row, ev_start_, ev_row_);
                cudaEventElapsedTime(&ms_col, ev_row_, ev_col_);
                cudaEventElapsedTime(&ms_hess, ev_col_, ev_hess_);
            }
            if (dedup_ran) cudaEventElapsedTime(&ms_dedup, ev_hess_, ev_dedup_);
            // 【批0·双计修复】Group 段原取 [hess,grp]——区间内含 Dedup 整段（报表
            // 把 Dedup 重复计入）。改取 [dedup,grp]（未跑去重时回退 [hess,grp]）
            if (dedup_ran) cudaEventElapsedTime(&ms_grp, ev_dedup_, ev_grp_);
            else            cudaEventElapsedTime(&ms_grp, ev_hess_, ev_grp_);
            cudaEventElapsedTime(&ms_out, ev_grp_, ev_end_);
            bench_ms_[0] = ms_total; bench_ms_[1] = ms_row; bench_ms_[2] = ms_col;
            bench_ms_[3] = ms_hess;  bench_ms_[4] = ms_dedup; bench_ms_[5] = ms_grp;
            bench_ms_[6] = ms_out;   // 【测速工具】暴露给 gpu_stress 聚合
            // 【批0】去重链细分：D1/D2+D3/D4/D5/D6（事件未记录时 ElapsedTime 会报
            // 粘性错——先判 dedup_ran）
            if (dedup_ran) {
                float m1 = 0, m23 = 0, m4 = 0, m5 = 0, m6 = 0;
                cudaEventElapsedTime(&m1,  ev_hess_, ev_d1_);
                cudaEventElapsedTime(&m23, ev_d1_,   ev_d23_);
                cudaEventElapsedTime(&m4,  ev_d23_,  ev_d4_);
                cudaEventElapsedTime(&m5,  ev_d4_,   ev_d5_);
                cudaEventElapsedTime(&m6,  ev_d5_,   ev_dedup_);
                dedup_ms_[0] = m1; dedup_ms_[1] = m23; dedup_ms_[2] = m4;
                dedup_ms_[3] = m5; dedup_ms_[4] = m6;
            }
            cudaGetLastError();      // 吞掉 bench 自身的非致命错误（如事件未就绪），防粘性污染

            CALIB_LOG_DEBUG("[BENCH_FAST] {}x{} | Total={:.2f}ms | Row3={:.2f}ms | Col5={:.2f}ms | Hessian={:.2f}ms | Dedup={:.2f}ms | Group={:.2f}ms | Out={:.2f}ms | N={} | merged={}",
                           cols, rows, ms_total, ms_row, ms_col, ms_hess, ms_dedup,
                           ms_grp, ms_out, point_count, result.mergedCount);
        }

        // centerPoints/lineCount 已在 if/else 分支内赋值
        result.totalPointCount = point_count;
        result.success = true;

        int64_t total_pixels = static_cast<int64_t>(rows) * cols;
        if (point_count > total_pixels / 2) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Too many points extracted (>50% of pixels), possible noise";
        } else if (collectHostPoints_ && !result.centerPoints.empty()) {
            bool has_degraded = false;
            for (const auto& [label, pts] : result.centerPoints) {
                if (static_cast<int>(pts.size()) < 10) {
                    has_degraded = true;
                    break;
                }
            }
            if (has_degraded) {
                result.qualityFlag = calib::QualityFlag::Degraded;
                result.message = "Some lines have fewer than 10 points";
            } else {
                result.qualityFlag = calib::QualityFlag::Normal;
                result.message = "Subpixel extraction successful";
            }
        }

        if (dedup_fallback) {
            result.qualityFlag = QualityFlag::Warning;
            result.message += std::string(" | dedup fallback: adjacency overflow")
                + " — frame passed through unmerged";
        }

        CALIB_LOG_DEBUG("Execute() completed: {} points, {} lines, qualityFlag={}",
                        result.totalPointCount, result.lineCount,
                        static_cast<int>(result.qualityFlag));

    } catch (const cv::Exception& e) {
        result.success = false;
        result.message = std::string("OpenCV error: ") + e.what();
        CALIB_LOG_ERROR("Execute() OpenCV exception: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Error: ") + e.what();
        CALIB_LOG_ERROR("Execute() exception: {}", e.what());
    }

    // 【150FPS·发射拆解】帧尾汇报（两次 Execute＝双眼）
    if (subProbe_)
        fprintf(stderr,
                "[steg-sub] pts=%d sub=%.2f(A%.2f/B%.2f) wait=%.2f prop=%.2f | A: mset=%.2f launch=%.2f err=%.2f\n",
                result.totalPointCount, subMs_, subA_, subB_, waitMs_, propMs_,
                aMsetMs_, aLaunchMs_, aErrMs_);

    return result;
}


