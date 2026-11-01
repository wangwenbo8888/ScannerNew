/**
 * @file epipolar_interp_dual_cuda_impl.cu
 * @brief 激光中心点极线插值CUDA算子·双模式版实现
 *
 * Labeled 路径：与 epipolar_interp_opt 逐位一致（kernel 复刻, 仅符号加 dual 前缀）。
 * Scan 路径（新增, 无线号）：
 *   Step 1: 按 y 排序（32bit 有序位型键, CUB SortPairs）
 *   Step 2: 逐极线行 kernel（固定 grid=SCAN_MAX_ROWS）:
 *     a) 窗口 [y_k±window] 内点收集到 smem（≤SCAN_ROW_PTS_CAP）
 *     b) 按 x 插入排序 + 间隙聚类（间隙 > scan_x_gap 分簇, 每簇≈一条线穿过该行）
 *     c) 簇内就近原则: 精确命中取最近；否则最近上窗点+最近下窗点配对（|Δx|≤max_x_diff）插值；
 *        单侧缺失兜底: 仅上窗或仅下窗有点时, 取该侧最近点垂直投影 (x 不变, y=y_k)
 *     d) 每簇输出一点, fid=0
 *   Step 3: CUB DeviceSelect::Flagged 压缩
 */

#include "epipolar_interp_dual_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace calib;

CALIB_DEFINE_LOG_TAG(06, EpipolarInterpDualCuda);

static constexpr int BLOCK_SIZE = 256;
static constexpr float EPSILON = 1e-4f;      // 精确命中判定（与原版一致）
static constexpr float DEGEN_EPS = 1e-6f;    // 除零保护

// ============================================================================
// kernels（dual 前缀, 避免与原版/opt 的 .cu 符号冲突）
// ============================================================================
// ── Labeled 路径（与 opt 逐位一致）──

__global__ void dualBuildSortKeys(
    const float2* __restrict__ d_pts,
    const int* __restrict__ d_fids,
    unsigned long long* __restrict__ d_keys,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float y = d_pts[idx].y;
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    unsigned long long key =
        ((unsigned long long)(unsigned int)d_fids[idx] << 32) | (unsigned int)yi;
    d_keys[idx] = key;
}

__global__ void dualFillIndices(int* __restrict__ d_idx, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_idx[idx] = idx;
}

__global__ void dualGatherSorted(
    const float2* __restrict__ d_pts_src,
    const int* __restrict__ d_fids_src,
    const int* __restrict__ d_indices,
    float2* __restrict__ d_pts_dst,
    int* __restrict__ d_fids_dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int src = d_indices[i];
    d_pts_dst[i] = d_pts_src[src];
    if (d_fids_dst) d_fids_dst[i] = d_fids_src[src];
}

__global__ void dualBoundaryFlags(
    const int* __restrict__ d_fids,
    int* __restrict__ d_flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_flags[i] = (i == 0 || d_fids[i] != d_fids[i - 1]) ? 1 : 0;
}

__global__ void dualInitMeta(unsigned int* __restrict__ d_meta)
{
    if (threadIdx.x == 0) {
        d_meta[0] = 0u;
        d_meta[1] = 0xffffffffu;
        d_meta[2] = 0u;
    }
}

__global__ void dualScatterLineTable(
    const int* __restrict__ d_fids,
    const int* __restrict__ d_flags,
    const int* __restrict__ d_ranks,
    int n,
    int* __restrict__ d_map,
    int* __restrict__ d_begin,
    unsigned int* __restrict__ d_meta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_flags[i]) {
        int r = d_ranks[i];
        d_map[r] = d_fids[i];
        d_begin[r] = i;
    }
    if (i == n - 1) {
        int num_runs = d_ranks[i] + d_flags[i];
        d_begin[num_runs] = n;
        d_meta[0] = (unsigned int)num_runs;
    }
}

__global__ void dualYRange(
    const float2* __restrict__ d_pts,
    int n,
    unsigned int* __restrict__ d_meta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = (unsigned int)__float_as_int(d_pts[i].y);
    atomicMin(&d_meta[1], u);
    atomicMax(&d_meta[2], u);
}

__global__ void __launch_bounds__(256, 4) dualEpipolarInterp(
    const float2* __restrict__ d_pts_sorted,
    const int* __restrict__ d_fids_sorted,
    const int* __restrict__ d_line_begin,
    const int* __restrict__ d_line_ids_map,
    int num_lines,
    int num_rows,
    int row_begin,
    float row_step,
    float window,
    float max_x_diff,
    int* __restrict__ d_flags,
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= num_lines * num_rows) return;

    int line_idx = linear / num_rows;
    int row_idx = linear - line_idx * num_rows;

    int fid = d_line_ids_map[line_idx];
    int begin = d_line_begin[line_idx];
    int end = d_line_begin[line_idx + 1];
    int cnt = end - begin;
    if (cnt <= 0) { d_flags[linear] = 0; return; }

    float y_k = (float)(row_begin + row_idx) * row_step;

    int lo = begin, hi = end;
    float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }
    int hit_idx = -1;
    float hit_dy = 1e30f;
    int up_idx = -1;
    float up_dy = 1e30f;
    int down_idx = -1;
    float down_dy = 1e30f;

    for (int i = lo; i < end; ++i) {
        float y = d_pts_sorted[i].y;
        if (y > y_k + window) break;
        float dy = y - y_k;
        float ady = fabsf(dy);
        if (ady < EPSILON) {
            if (ady < hit_dy) { hit_dy = ady; hit_idx = i; }
        } else if (dy < 0.0f) {
            if (ady < up_dy) { up_dy = ady; up_idx = i; }
        } else {
            if (ady < down_dy) { down_dy = ady; down_idx = i; }
        }
    }

    float2 out;
    if (hit_idx >= 0) {
        out = d_pts_sorted[hit_idx];
        out.y = y_k;
    } else {
        if (up_idx < 0 || down_idx < 0) { d_flags[linear] = 0; return; }
        float2 pu = d_pts_sorted[up_idx];
        float2 pd = d_pts_sorted[down_idx];
        if (fabsf(pu.x - pd.x) > max_x_diff) { d_flags[linear] = 0; return; }
        float denom = pd.y - pu.y;
        if (fabsf(denom) < DEGEN_EPS) { d_flags[linear] = 0; return; }
        float t = (y_k - pu.y) / denom;
        out.x = pu.x + t * (pd.x - pu.x);
        out.y = y_k;
    }

    d_out_pts[linear] = out;
    d_out_fids[linear] = fid;
    d_flags[linear] = 1;
}

// ── 模式自动判别：line_ids 最小/最大值 ──

__global__ void dualInitIdRange(int* __restrict__ d_meta2)
{
    if (threadIdx.x == 0) {
        d_meta2[0] = INT_MAX;   // min
        d_meta2[1] = INT_MIN;   // max
    }
}

__global__ void dualIdRange(
    const int* __restrict__ d_ids,
    int n,
    int* __restrict__ d_meta2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicMin(&d_meta2[0], d_ids[i]);
    atomicMax(&d_meta2[1], d_ids[i]);
}

// ── Scan 路径（新增）──

// 32bit 有序 y 键 + 顺带 idx 序列
// 【E4】d_dcnt 非空＝容量模式：i≥*dcnt 的槽戳 0xFFFFFFFF（float 有序键全域最大
// 0x7FFFFFFF——尾键严格排最后，容量排序后有效前缀 [0,*dcnt) 与精确排序逐位一致）；
// d_idx[i]=i 恒写（尾区 idx 合法，gather 后不被消费）
__global__ void dualBuildYKeys(
    const float2* __restrict__ d_pts,
    int n,
    unsigned int* __restrict__ d_keys,
    int* __restrict__ d_idx,
    const int* __restrict__ d_dcnt = nullptr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_dcnt != nullptr) {
        // 【R3-中2】计数钳位：flat 计数未钳位可超容量——i≥min(count,n) 一律尾戳
        const int dn = *d_dcnt;
        if (dn >= n || i >= dn) {
            d_keys[i] = 0xFFFFFFFFu;
            d_idx[i] = i;
            return;
        }
    }
    float y = d_pts[i].y;
    unsigned int yi = (unsigned int)(__float_as_int(y) & 0x7fffffff);
    if (__float_as_int(y) < 0) yi = ~yi;
    d_keys[i] = yi;
    d_idx[i] = i;
}

// 逐极线行扫描插值：收集→x 排序→聚类→簇内就近配对（thread0 串行, 块=32）
// 【E4】d_nptr 非空＝容量模式：搜索界 n 改读设备计数（防二分/收集闯入尾区垃圾）；
// 【R3-中2】设备计数钳位到容量（flat 计数未钳位可超 kOutCap——防越界读）
__global__ void dualScanInterp(
    const float2* __restrict__ d_pts_sorted,   // y 升序
    int n,
    float step,
    float window,
    float max_x_diff,
    float x_gap,
    int* __restrict__ d_flags,                 // [row*SCAN_SLOTS + slot]
    float2* __restrict__ d_out_pts,
    int* __restrict__ d_out_fids,
    const int* __restrict__ d_nptr = nullptr)
{
    if (d_nptr != nullptr) {
        const int dn = *d_nptr;
        if (dn < n) n = dn;   // 钳位：计数＞容量时按容量界（写入侧已截断）
    }
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;

    __shared__ float sx[256];
    __shared__ float sy[256];

    const float y_k = row * step;

    // 二分: 第一个 y >= y_k - window
    int lo = 0, hi = n;
    const float ylw = y_k - window;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_pts_sorted[mid].y < ylw) lo = mid + 1;
        else hi = mid;
    }

    int cnt = 0;
    for (int i = lo; i < n; ++i) {
        float y = d_pts_sorted[i].y;
        if (y > y_k + window) break;
        if (cnt >= 256) break;   // CAP
        sx[cnt] = d_pts_sorted[i].x;
        sy[cnt] = y;
        ++cnt;
    }
    if (cnt == 0) return;        // flags 已由 host memset 清零

    // 按 x 插入排序（窗口内点规模 ~10-60）
    for (int i = 1; i < cnt; ++i) {
        float x = sx[i], y = sy[i];
        int j = i - 1;
        while (j >= 0 && sx[j] > x) {
            sx[j + 1] = sx[j];
            sy[j + 1] = sy[j];
            --j;
        }
        sx[j + 1] = x;
        sy[j + 1] = y;
    }

    // 间隙聚类 + 簇内就近插值
    int slot = 0;
    int start = 0;
    for (int i = 1; i <= cnt; ++i) {
        const bool boundary = (i == cnt) || (sx[i] - sx[i - 1] > x_gap);
        if (!boundary) continue;

        int hit = -1;  float hit_dy = 1e30f;
        int up = -1;   float up_dy = 1e30f;
        int dn = -1;   float dn_dy = 1e30f;
        for (int k = start; k < i; ++k) {
            const float dy = sy[k] - y_k;
            const float ady = fabsf(dy);
            if (ady < EPSILON) {
                if (ady < hit_dy) { hit_dy = ady; hit = k; }
            } else if (dy < 0.0f) {
                if (ady < up_dy) { up_dy = ady; up = k; }
            } else {
                if (ady < dn_dy) { dn_dy = ady; dn = k; }
            }
        }

        float ox = 0.f;
        bool ok = false;
        if (hit >= 0) {
            ox = sx[hit];
            ok = true;
        } else if (up >= 0 && dn >= 0) {
            if (fabsf(sx[up] - sx[dn]) <= max_x_diff) {
                const float denom = sy[dn] - sy[up];
                if (fabsf(denom) >= DEGEN_EPS) {
                    const float t = (y_k - sy[up]) / denom;
                    ox = sx[up] + t * (sx[dn] - sx[up]);
                    ok = true;
                }
            }
        } else if (up >= 0) {   // 单侧兜底: 仅上窗有点 → 垂直投影
            ox = sx[up];
            ok = true;
        } else if (dn >= 0) {   // 单侧兜底: 仅下窗有点 → 垂直投影
            ox = sx[dn];
            ok = true;
        }

        if (ok && slot < 32) {
            const int at = row * 32 + slot;
            d_out_pts[at] = make_float2(ox, y_k);
            d_out_fids[at] = at;    // 扫描模式: 簇 ID（row*SCAN_SLOTS+slot 编码）
            d_flags[at] = 1;
            ++slot;
        }
        start = i;
    }
}

// ============================================================================
// 【刀二C】Scan 行桶族（替代 buildYKeys→radixSort→gather→memset flags→
// dualScanInterp→Select×2 共 13 内核＋1MB memset → 6 内核＋32KB memset）
// 输出序：行主序＋行内 slot 升序＝Select 稳定紧缩序（逐位一致）；
// 窗内收集集合＝原二分窗集合（同 y 过滤式）；插入排序归一化桶内到达序。
// ============================================================================
constexpr int kInterpRows = 8192;   // 同 Impl::SCAN_MAX_ROWS（类成员在命名空间作用域不可见）

// 1) 行计数（E4 容量模式：i≥*dcnt 尾区不入桶——尾戳机制随之消亡）
__global__ void dualRowHistV2(
    const float2* __restrict__ d_pts,
    int cap,
    float step,
    int* __restrict__ hist,          // [8193]
    const int* __restrict__ d_dcnt)
{
    const int n = min(*d_dcnt, cap);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = __float2int_rn(d_pts[i].y / step);
    if (row < 0) row = 0;
    if (row > kInterpRows) row = kInterpRows;   // 死桶（物理不现——y 有界）
    atomicAdd(&hist[row], 1);
}

// 2) 行扫描：rowStart[8193] 排他前缀（[8192]=total）＋cursor 副本＋total→计数；
//    顺带清零 hist（自清洁）；死桶弃
__global__ void dualRowScanV2(
    int* __restrict__ hist,          // [8193] 入＝计数/出＝全零
    int* __restrict__ row_start,     // [8193] 出（[r]=行 r 起点；[8192]=total）
    int* __restrict__ cursor,        // [8192] 出
    int* __restrict__ total_out)     // d_scan_ccount_
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int chunk = (kInterpRows + bdim - 1) / bdim;
    const int begin = tid * chunk;
    const int end = min(begin + chunk, kInterpRows);
    int localCnt = 0;
    for (int b = begin; b < end; ++b) localCnt += hist[b];
    __shared__ int scan_[1024];
    scan_[tid] = localCnt;
    __syncthreads();
    for (int off = 1; off < bdim; off <<= 1) {
        int v = (tid >= off) ? scan_[tid - off] : 0;
        __syncthreads();
        scan_[tid] += v;
        __syncthreads();
    }
    const int prefix = (tid == 0) ? 0 : scan_[tid - 1];
    const int total = scan_[bdim - 1];
    int running = prefix;
    for (int b = begin; b < end; ++b) {
        row_start[b] = running;
        cursor[b] = running;
        running += hist[b];
        hist[b] = 0;
    }
    if (tid == 0) {
        row_start[kInterpRows] = total;
        hist[kInterpRows] = 0;
        *total_out = total;
    }
}

// 3) 散射直写 d_sorted_pts_（行主序；行内到达序——插入排序归一化）
__global__ void dualRowScatterPtsV2(
    const float2* __restrict__ d_pts,
    int cap,
    float step,
    int* __restrict__ cursor,        // [8192]
    float2* __restrict__ d_out,
    const int* __restrict__ d_dcnt)
{
    const int n = min(*d_dcnt, cap);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = __float2int_rn(d_pts[i].y / step);
    if (row < 0) row = 0;
    if (row >= kInterpRows) return;   // 死桶不入槽
    const int slot = atomicAdd(&cursor[row], 1);
    d_out[slot] = d_pts[i];
}

// 4) 行窗版逐行扫描插值（thread0 串行/行；行窗取代二分；rowCnt 恒写——免 flags memset）
__global__ void dualScanInterpRowed(
    const float2* __restrict__ d_pts_sorted,   // 行主序（行内任意序）
    const int* __restrict__ d_row_start,       // [8193]
    float step,
    float window,
    float max_x_diff,
    float x_gap,
    float2* __restrict__ d_out_pts,            // [row*32+slot]
    int* __restrict__ d_out_fids,
    int* __restrict__ d_row_cnt)               // [8192]（恒写——含 0）
{
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;

    __shared__ float sx[256];
    __shared__ float sy[256];

    const float y_k = row * step;

    // 行窗：y∈[y_k-window, y_k+window] 的点必在行 [row-wRows, row+wRows]
    //（round 差半 step＋1 行余量）
    const int wRows = static_cast<int>(ceilf(window / step)) + 1;
    const int rLo = max(0, row - wRows);
    const int rHi = min(kInterpRows, row + wRows + 1);
    const int lo = d_row_start[rLo];
    const int hi = d_row_start[rHi];

    int cnt = 0;
    const float ylow = y_k - window;
    const float yhigh = y_k + window;
    for (int i = lo; i < hi; ++i) {
        const float y = d_pts_sorted[i].y;
        if (y < ylow || y > yhigh) continue;   // 显式双侧过滤（原二分＋break 等价集）
        if (cnt >= 256) break;                 // CAP（窗内 >256＝物理不现）
        sx[cnt] = d_pts_sorted[i].x;
        sy[cnt] = y;
        ++cnt;
    }
    if (cnt == 0) {
        d_row_cnt[row] = 0;
        return;
    }

    // 按 x 插入排序（与原版同款——归一化桶内到达序）
    for (int i = 1; i < cnt; ++i) {
        float x = sx[i], y = sy[i];
        int j = i - 1;
        while (j >= 0 && sx[j] > x) {
            sx[j + 1] = sx[j];
            sy[j + 1] = sy[j];
            --j;
        }
        sx[j + 1] = x;
        sy[j + 1] = y;
    }

    // 间隙聚类 + 簇内就近插值（与原版逐字同）
    int slot = 0;
    int start = 0;
    for (int i = 1; i <= cnt; ++i) {
        const bool boundary = (i == cnt) || (sx[i] - sx[i - 1] > x_gap);
        if (!boundary) continue;

        int hit = -1;  float hit_dy = 1e30f;
        int up = -1;   float up_dy = 1e30f;
        int dn = -1;   float dn_dy = 1e30f;
        for (int k = start; k < i; ++k) {
            const float dy = sy[k] - y_k;
            const float ady = fabsf(dy);
            if (ady < EPSILON) {
                if (ady < hit_dy) { hit_dy = ady; hit = k; }
            } else if (dy < 0.0f) {
                if (ady < up_dy) { up_dy = ady; up = k; }
            } else {
                if (ady < dn_dy) { dn_dy = ady; dn = k; }
            }
        }

        float ox = 0.f;
        bool ok = false;
        if (hit >= 0) {
            ox = sx[hit];
            ok = true;
        } else if (up >= 0 && dn >= 0) {
            if (fabsf(sx[up] - sx[dn]) <= max_x_diff) {
                const float denom = sy[dn] - sy[up];
                if (fabsf(denom) >= DEGEN_EPS) {
                    const float t = (y_k - sy[up]) / denom;
                    ox = sx[up] + t * (sx[dn] - sx[up]);
                    ok = true;
                }
            }
        } else if (up >= 0) {   // 单侧兜底: 仅上窗有点 → 垂直投影
            ox = sx[up];
            ok = true;
        } else if (dn >= 0) {   // 单侧兜底: 仅下窗有点 → 垂直投影
            ox = sx[dn];
            ok = true;
        }

        if (ok && slot < 32) {
            const int at = row * 32 + slot;
            d_out_pts[at] = make_float2(ox, y_k);
            d_out_fids[at] = at;    // 扫描模式: 簇 ID（row*SCAN_SLOTS+slot 编码）
            ++slot;
        }
        start = i;
    }
    d_row_cnt[row] = slot;
}

// 5) 行计数扫描：紧缩基址（排他前缀）＋输出总数→cntOut
//   【修复】原实现漏写输出总数——d_scan_ccount_ 残留输入桶总数（dualRowScanV2
//   写入），finishScanResult 按输入数截断 ROI（点数暴减）。legacy Select 路径
//   的 num_selected 即输出数——本处对齐。
__global__ void dualRowCntScanV2(
    const int* __restrict__ row_cnt,     // [8192]
    int* __restrict__ cbase,             // [8192]
    int* __restrict__ total_out)         // d_scan_ccount_
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int chunk = (kInterpRows + bdim - 1) / bdim;
    const int begin = tid * chunk;
    const int end = min(begin + chunk, kInterpRows);
    int localCnt = 0;
    for (int b = begin; b < end; ++b) localCnt += row_cnt[b];
    __shared__ int scan_[1024];
    scan_[tid] = localCnt;
    __syncthreads();
    for (int off = 1; off < bdim; off <<= 1) {
        int v = (tid >= off) ? scan_[tid - off] : 0;
        __syncthreads();
        scan_[tid] += v;
        __syncthreads();
    }
    int running = (tid == 0) ? 0 : scan_[tid - 1];
    for (int b = begin; b < end; ++b) {
        cbase[b] = running;
        running += row_cnt[b];
    }
    if (tid == bdim - 1) *total_out = scan_[bdim - 1];   // 输出总数（紧缩后）
}

// 6) 紧缩（线程/行；序＝行主序＋slot 升序＝Select 稳定序，逐位一致）
__global__ void dualCompactV2(
    const int* __restrict__ row_cnt,
    const int* __restrict__ cbase,
    const float2* __restrict__ in_pts,    // [row*32+slot]
    const int* __restrict__ in_fids,
    float2* __restrict__ out_pts,         // [total]
    int* __restrict__ out_fids)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= kInterpRows) return;
    const int cnt = row_cnt[row];
    const int base = cbase[row];
    const int rowBase = row * 32;
    for (int s = 0; s < cnt; ++s) {
        out_pts[base + s] = in_pts[rowBase + s];
        out_fids[base + s] = in_fids[rowBase + s];
    }
}

// ── Scan 段链接：行间簇合并（每簇向近 gap 行内 |Δx|≤thr 的最近簇建父指针）──
__global__ void dualScanLinkParent(
    const float2* __restrict__ d_cpts,   // 压缩后（行序升序，行内 x 序）
    int n,
    float step,
    float dxThr,
    int gapRows,
    int* __restrict__ d_parent,
    int* __restrict__ d_label = nullptr,  // 【刀三微】可选双写（建链初值）
    const int* __restrict__ d_count = nullptr)  // 【WDDM 滴流修复】容量发射时的
                                                // 设备侧有效界：尾部槽位是未初始化
                                                // 垃圾（行号乱值→下方 break 永不触
                                                // 发→O(cap) 反向全扫＝性能病灶），
                                                // i≥*d_count 线程直退（读一次显存，
                                                // 旧路径传 nullptr 行为逐字不变）
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_count != nullptr && i >= *d_count) return;
    const float2 p = d_cpts[i];
    const int row = __float2int_rn(p.y / step);
    int best = -1;
    int bestRowDiff = gapRows + 1;
    float bestDx = dxThr + 1.0f;
    for (int j = i - 1; j >= 0; --j) {
        const float2 q = d_cpts[j];
        const int rj = __float2int_rn(q.y / step);
        const int rd = row - rj;
        if (rd <= 0) continue;          // 同行不链（行内点连续位于前方）
        if (rd > gapRows) break;        // 出窗（行序保证后续更远）
        const float dx = fabsf(p.x - q.x);
        if (dx > dxThr) continue;
        if (rd < bestRowDiff || (rd == bestRowDiff && dx < bestDx)) {
            bestRowDiff = rd;
            bestDx = dx;
            best = j;
        }
    }
    const int par = best >= 0 ? best : i;
    d_parent[i] = par;
    if (d_label != nullptr) d_label[i] = par;   // 【刀三微】双写建链初值（免 D2D 拷贝）
}

// pointer doubling 一轮（森林求根）
__global__ void dualScanLinkJump(
    int* __restrict__ d_label,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_label[i] = d_label[d_label[i]];
}

// 【刀三微】LinkParent 双写 parent＋label 同值——建链后免一笔 D2D 拷贝

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class DualScopedFlag {
public:
    explicit DualScopedFlag(std::atomic<bool>* flag_) : flag_(flag_) {
        flag_->store(true);
    }
    ~DualScopedFlag() { flag_->store(false); }
    DualScopedFlag(const DualScopedFlag&) = delete;
    DualScopedFlag& operator=(const DualScopedFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// Impl
// ============================================================================

EpipolarInterpDualCuda::Impl::Impl(const EpipolarInterpDualParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[06-EpipolarInterpDualCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[06-EpipolarInterpDualCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    if (cudaMallocHost(&h_meta_pinned_, 4 * sizeof(int)) != cudaSuccess) {
        h_meta_pinned_ = nullptr;
        throw std::runtime_error("[06-EpipolarInterpDualCuda] pinned host alloc failed");
    }

    d_meta_.create(1, 4, CV_32SC1);
}

EpipolarInterpDualCuda::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
    if (h_meta_pinned_) {
        cudaFreeHost(h_meta_pinned_);
        h_meta_pinned_ = nullptr;
    }
}

template<typename T>
static inline void safeCudaFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            CALIB_LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

void EpipolarInterpDualCuda::Impl::ensureCapacity(int pointCount) {
    if (pointCount <= capacity_) return;

    d_keys_in_.create(1, pointCount, CV_64FC1);
    d_keys_out_.create(1, pointCount, CV_64FC1);
    d_ykeys_in_.create(1, pointCount, CV_32SC1);
    d_ykeys_out_.create(1, pointCount, CV_32SC1);
    d_idx_in_.create(1, pointCount, CV_32SC1);
    d_idx_out_.create(1, pointCount, CV_32SC1);
    d_sorted_pts_.create(1, pointCount, CV_32FC2);
    d_sorted_fids_.create(1, pointCount, CV_32SC1);
    d_rl_flags_.create(1, pointCount, CV_32SC1);
    d_rl_ranks_.create(1, pointCount, CV_32SC1);
    d_line_begin_.create(1, pointCount + 1, CV_32SC1);
    d_line_ids_map_.create(1, pointCount, CV_32SC1);
    capacity_ = pointCount;
}

// ============================================================================
// 【E4】Scan 延迟完成：ExecuteDeferred（容量提交·零宿主同步）
// 输入＝容量宽视图＋设备计数指针；键尾戳 0xFFFFFFFF 保容量排序前缀＝精确排序；
// dualScanInterp 搜索界读设备计数；计数 D2H 入 pinned 槽[3]（不同步）。
// ============================================================================
EpipolarInterpResult EpipolarInterpDualCuda::Impl::ExecuteDeferred(
    const cv::cuda::GpuMat& d_points, const cv::cuda::GpuMat& d_line_ids,
    const int* d_deviceCount, cv::cuda::Stream& stream) {
    (void)d_line_ids;
    EpipolarInterpResult result;
    lastLinkAtCap_ = false;   // 【WDDM 滴流修复】每帧重置（消费即清＋入口双保险）
    if (params_.mode != InterpMode::Scan) {
        result.success = false;
        result.message = "ExecuteDeferred 仅支持 Scan 模式(Auto 判别在容量视图下不可判定)";
        CALIB_LOG_ERROR("ExecuteDeferred(): {}", result.message);
        return result;
    }
    if (d_points.empty() || d_deviceCount == nullptr) {
        result.success = false;
        result.message = "ExecuteDeferred: empty input or null device count";
        CALIB_LOG_ERROR("ExecuteDeferred(): {}", result.message);
        return result;
    }
    const int cap = d_points.rows * d_points.cols;
    if (cap < 2) {
        result.success = true;
        result.message = "Less than 2 points, no interpolation";
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>();
        result.interpCount = 0;
        lastCs_ = nullptr;   // 【R3-低6】无待完成段——防 FinishInterp 消费陈旧上下文
        return result;
    }
    ensureCapacity(cap);
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    lastCs_ = cs;
    const int gridN = (cap + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float2* dPts = d_points.ptr<float2>();

    // ══ 【G1·石头二】图路径（JMW_1BGRAPH=1；全定容→参数跨帧稳定→一次性捕获）══
    static const bool gOn_ = (getenv("JMW_1BGRAPH") != nullptr
                              && getenv("JMW_1BGRAPH")[0] != '0');
    if (gOn_ && !graphDisabled_ && ++graphWarmCalls_ >= 2 && cap == capacity_
        && radix_cap_ >= cap && d_radix_temp_ != nullptr && d_cub_temp_ != nullptr) {
        // 前置：缓冲已预热（graphWarmCalls_≥2 且无待分配分支）——捕获区内禁 cudaMalloc
        if (deferGraphExec_ == nullptr) {
            // 捕获（一次性）：ThreadLocal＋先排空本流（mask_sep 铁规：在飞前序与
            // BeginCapture 交叠＝驱动崩溃案在案）。排空含本帧 flat/undist——一次性代价。
            cudaStreamSynchronize(cs);
            cudaError_t cerr = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
            if (cerr == cudaSuccess)
                cerr = submitScanSequence(cs, dPts, d_deviceCount, cap, gridN)
                           ? cudaSuccess : cudaErrorUnknown;
            cudaGraph_t g = nullptr;
            if (cerr == cudaSuccess) cerr = cudaStreamEndCapture(cs, &g);
            if (cerr == cudaSuccess && g != nullptr)
                cerr = cudaGraphInstantiate(&deferGraphExec_, g, nullptr, nullptr, 0);
            if (cerr == cudaSuccess && deferGraphExec_ != nullptr) {
                deferGraph_ = g;
                CALIB_LOG_INFO("ExecuteDeferred graph captured (nodes≈44, cap={})", cap);
            } else {
                if (g != nullptr) cudaGraphDestroy(g);
                deferGraphExec_ = nullptr;
                graphDisabled_ = true;   // 永久回退（fg_.disabled 先例）
                cudaGetLastError();      // 捕获探测残留清账
                CALIB_LOG_WARN("ExecuteDeferred graph capture failed ({}), fallback to per-launch",
                               cudaGetErrorString(cerr));
            }
        }
        if (deferGraphExec_ != nullptr) {
            cudaError_t lerr = cudaGraphLaunch(deferGraphExec_, cs);
            if (lerr == cudaSuccess) {
                // 【WDDM 滴流修复】链接相位不在图内——图发射后按容量补提交
                //（与逐发路径同藏进排水等待；失败则 FinishInterp 回退精确版）
                lastLinkAtCap_ = submitLinkAtCapacity(cs);
                result.success = true;
                result.message = "Success (scan mode, deferred graph)";
                result.interpCount = -1;
                return result;
            }
            // 发射失败（罕见）：毁图回退，本帧走逐发
            cudaGraphExecDestroy(deferGraphExec_);
            deferGraphExec_ = nullptr;
            graphDisabled_ = true;
            cudaGetLastError();
            CALIB_LOG_WARN("ExecuteDeferred graph launch failed, permanent fallback");
        }
    }

    // ══ 逐发路径（默认/回退/首帧预热）══
    try {
        if (!submitScanSequence(cs, dPts, d_deviceCount, cap, gridN)) {
            result.success = false;
            result.message = "ExecuteDeferred submit failed (alloc/launch)";
            CALIB_LOG_ERROR("ExecuteDeferred(): {}", result.message);
            return result;
        }
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("ExecuteDeferred kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("ExecuteDeferred(): {}", result.message);
            return result;
        }
        // 【WDDM 滴流修复】链接相位随提交段按容量发射（28 个 2µs 内核藏在
        // 排水等待里——原 FinishInterp 排水后逐个放行 ~90µs/个＝2.5ms 滴流）
        lastLinkAtCap_ = submitLinkAtCapacity(cs);
        result.success = true;          // 提交级成功；interpCount/视图在 FinishInterp 填
        result.message = "Success (scan mode, deferred)";
        result.interpCount = -1;
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("ExecuteDeferred exception: ") + e.what();
        CALIB_LOG_ERROR("ExecuteDeferred(): {}", result.message);
        return result;
    }
}

// 【刀二C】Scan 提交序列（行桶族，图捕获体＝逐发体同源）：
// hist→rowScan→scatter→rowedInterp→cntScan→compact→count D2H。
// 替代 buildYKeys→SortPairs→gather→memset(1MB)→scanInterp→Select×2
//（13 内核＋1 memset→6 内核＋32KB memset）；输出序与 Select 稳定序逐位一致。
bool EpipolarInterpDualCuda::Impl::submitScanSequence(
    cudaStream_t cs, const float2* dPts, const int* d_deviceCount, int cap, int gridN) {
    // 【刀二C 回退开关】JMW_K2C_LEGACY=1 走旧排序路径（诊断拆分：A/B 用 JMW_K2_LEGACY）
    static const bool k2Legacy_ = (getenv("JMW_K2C_LEGACY") != nullptr
                                   && getenv("JMW_K2C_LEGACY")[0] == '1');
    if (k2Legacy_) {
        // —— 旧路径：键构造（尾戳）＋容量排序＋gather＋memset＋scanInterp＋Select×2 ——
        dualBuildYKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
            dPts, cap,
            reinterpret_cast<unsigned int*>(d_ykeys_in_.ptr<int>()),
            d_idx_in_.ptr<int>(), d_deviceCount);
        if (radix_cap_ < cap || d_radix_temp_ == nullptr) {
            size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(
                nullptr, bytes,
                (const unsigned int*)nullptr, (unsigned int*)nullptr,
                (const int*)nullptr, (int*)nullptr,
                cap, 0, sizeof(unsigned int) * 8, cs);
            safeCudaFree(d_radix_temp_);
            cudaError_t err = cudaMalloc(&d_radix_temp_, bytes);
            if (err != cudaSuccess) return false;
            radix_temp_size_ = bytes;
            radix_cap_ = cap;
        }
        {
            cudaError_t err = cub::DeviceRadixSort::SortPairs(
                d_radix_temp_, radix_temp_size_,
                reinterpret_cast<const unsigned int*>(d_ykeys_in_.ptr<int>()),
                reinterpret_cast<unsigned int*>(d_ykeys_out_.ptr<int>()),
                d_idx_in_.ptr<int>(), d_idx_out_.ptr<int>(),
                cap, 0, sizeof(unsigned int) * 8, cs);
            if (err != cudaSuccess) return false;
        }
        dualGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
            dPts, nullptr, d_idx_out_.ptr<int>(),
            d_sorted_pts_.ptr<float2>(), nullptr, cap);
        const int totalSlots = SCAN_MAX_ROWS * SCAN_SLOTS;
        d_scan_flags_.create(1, totalSlots, CV_32SC1);
        d_scan_out_pts_.create(1, totalSlots, CV_32FC2);
        d_scan_out_fids_.create(1, totalSlots, CV_32SC1);
        d_scan_cpts_.create(1, totalSlots, CV_32FC2);
        d_scan_cfids_.create(1, totalSlots, CV_32SC1);
        d_scan_ccount_.create(1, 1, CV_32SC1);
        cudaMemsetAsync(d_scan_flags_.ptr<int>(), 0, totalSlots * sizeof(int), cs);
        dualScanInterp<<<SCAN_MAX_ROWS, 32, 0, cs>>>(
            d_sorted_pts_.ptr<float2>(), cap,
            params_.epipolar_row_step, params_.effectiveWindow(),
            params_.max_x_diff, params_.scan_x_gap,
            d_scan_flags_.ptr<int>(),
            d_scan_out_pts_.ptr<float2>(), d_scan_out_fids_.ptr<int>(),
            d_deviceCount);
        if (cub_cap_ < totalSlots || d_cub_temp_ == nullptr) {
            size_t t1 = 0, t2 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                (const float2*)nullptr, (const int*)nullptr,
                (float2*)nullptr, (int*)nullptr, totalSlots, cs);
            cub::DeviceSelect::Flagged(nullptr, t2,
                (const int*)nullptr, (const int*)nullptr,
                (int*)nullptr, (int*)nullptr, totalSlots, cs);
            const size_t need = t1 > t2 ? t1 : t2;
            safeCudaFree(d_cub_temp_);
            cudaError_t err = cudaMalloc(&d_cub_temp_, need);
            if (err != cudaSuccess) return false;
            cub_temp_size_ = need;
            cub_cap_ = totalSlots;
        }
        {
            cudaError_t err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const float2*)d_scan_out_pts_.ptr<float2>(),
                (const int*)d_scan_flags_.ptr<int>(),
                d_scan_cpts_.ptr<float2>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
            if (err != cudaSuccess) return false;
            err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const int*)d_scan_out_fids_.ptr<int>(),
                (const int*)d_scan_flags_.ptr<int>(),
                d_scan_cfids_.ptr<int>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
            if (err != cudaSuccess) return false;
        }
        cudaMemcpyAsync(h_meta_pinned_ + 3, d_scan_ccount_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        return true;
    }

    // —— 新路径：行桶族 ——
    (void)gridN;   // 兼容旧签名（新序列网格自定）
    // —— 行桶缓冲（8193/8192 定容——首建后 no-op）——
    d_rowb_hist_.create(1, kInterpRows + 1, CV_32SC1);
    d_rowb_start_.create(1, kInterpRows + 1, CV_32SC1);
    d_rowb_cursor_.create(1, kInterpRows, CV_32SC1);
    d_rowb_cnt_.create(1, kInterpRows, CV_32SC1);
    d_rowb_cbase_.create(1, kInterpRows, CV_32SC1);
    d_scan_out_pts_.create(1, SCAN_MAX_ROWS * SCAN_SLOTS, CV_32FC2);
    d_scan_out_fids_.create(1, SCAN_MAX_ROWS * SCAN_SLOTS, CV_32SC1);
    d_scan_cpts_.create(1, SCAN_MAX_ROWS * SCAN_SLOTS, CV_32FC2);
    d_scan_cfids_.create(1, SCAN_MAX_ROWS * SCAN_SLOTS, CV_32SC1);
    d_scan_ccount_.create(1, 1, CV_32SC1);
    d_sorted_pts_.create(1, cap, CV_32FC2);

    cudaMemsetAsync(d_rowb_hist_.ptr<int>(), 0,
                    (kInterpRows + 1) * sizeof(int), cs);
    const int gridCap = (cap + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dualRowHistV2<<<gridCap, BLOCK_SIZE, 0, cs>>>(
        dPts, cap, params_.epipolar_row_step,
        d_rowb_hist_.ptr<int>(), d_deviceCount);
    dualRowScanV2<<<1, 1024, 0, cs>>>(
        d_rowb_hist_.ptr<int>(), d_rowb_start_.ptr<int>(),
        d_rowb_cursor_.ptr<int>(), d_scan_ccount_.ptr<int>());
    dualRowScatterPtsV2<<<gridCap, BLOCK_SIZE, 0, cs>>>(
        dPts, cap, params_.epipolar_row_step,
        d_rowb_cursor_.ptr<int>(), d_sorted_pts_.ptr<float2>(), d_deviceCount);
    dualScanInterpRowed<<<SCAN_MAX_ROWS, 32, 0, cs>>>(
        d_sorted_pts_.ptr<float2>(), d_rowb_start_.ptr<int>(),
        params_.epipolar_row_step, params_.effectiveWindow(),
        params_.max_x_diff, params_.scan_x_gap,
        d_scan_out_pts_.ptr<float2>(), d_scan_out_fids_.ptr<int>(),
        d_rowb_cnt_.ptr<int>());
    dualRowCntScanV2<<<1, 1024, 0, cs>>>(
        d_rowb_cnt_.ptr<int>(), d_rowb_cbase_.ptr<int>(),
        d_scan_ccount_.ptr<int>());
    dualCompactV2<<<(SCAN_MAX_ROWS + 255) / 256, 256, 0, cs>>>(
        d_rowb_cnt_.ptr<int>(), d_rowb_cbase_.ptr<int>(),
        d_scan_out_pts_.ptr<float2>(), d_scan_out_fids_.ptr<int>(),
        d_scan_cpts_.ptr<float2>(), d_scan_cfids_.ptr<int>());

    // 计数→pinned 槽[3]（Scan 专用；Labeled 用 [0..2]——两模式互斥）
    cudaMemcpyAsync(h_meta_pinned_ + 3, d_scan_ccount_.ptr<int>(), sizeof(int),
                    cudaMemcpyDeviceToHost, cs);
    return true;
}

void EpipolarInterpDualCuda::Impl::FinishInterp(EpipolarInterpResult& result) {
    if (h_meta_pinned_ == nullptr || lastCs_ == nullptr) {
        result.success = false;
        result.message = "FinishInterp: no deferred context (ExecuteDeferred 未调用或已析构)";
        CALIB_LOG_ERROR("FinishInterp(): {}", result.message);
        return;
    }
    finishScanResult(result, h_meta_pinned_[3], lastCs_);
}

// 【WDDM 滴流修复·2026-10-23】链接相位按容量提交：网格/轮数用 capacity_
// （grow-only 缓冲提到容量）。正确性＝有效区 [0,count) 输出与精确 N 版逐位
// 一致：dualScanLinkParent 仅写 d_parent[i]/d_label[i] 自身槽位、父候选
// j<i 恒落在 valid 区（行序升序前提），尾部线程读写不越过自身；跳转轮
// pointer-doublling 链深 ≤行数 8192→13 轮封顶与精确版同界。尾部 [count,cap)
// 为陈旧数据，其标签演化无害——下游只消费 colRange(0,count)。
bool EpipolarInterpDualCuda::Impl::submitLinkAtCapacity(cudaStream_t cs) {
    const int cap = capacity_;
    if (params_.scan_link_dx <= 0.0f || cap < 2) return false;
    if (d_link_parent_.cols < cap) {
        d_link_parent_.create(1, cap, CV_32SC1);
        d_link_label_.create(1, cap, CV_32SC1);
    }
    const int gridL = (cap + 255) / 256;
    // 有效界＝扫描序列的设备输出计数（d_scan_ccount_——插值会补点，输出数可
    // 大于输入数，勿用 d_deviceCount！同流 FIFO 保证 parent 执行时计数已写好）
    dualScanLinkParent<<<gridL, 256, 0, cs>>>(
        d_scan_cpts_.ptr<float2>(), cap,
        params_.epipolar_row_step, params_.scan_link_dx,
        params_.scan_link_gap_rows, d_link_parent_.ptr<int>(),
        d_link_label_.ptr<int>(), d_scan_ccount_.ptr<int>());
    if (cudaGetLastError() != cudaSuccess) return false;
    int rounds = 1;
    while ((1 << rounds) < cap) ++rounds;
    if (rounds > 13) rounds = 13;   // 与精确版同封顶（链深 ≤ 行数 8192）
    for (int r = 0; r < rounds; ++r)
        dualScanLinkJump<<<gridL, 256, 0, cs>>>(d_link_label_.ptr<int>(), cap);
    return cudaGetLastError() == cudaSuccess;
}

// Scan 计数后共尾（精确 N 已知）：视图＋段链接＋result 填真值——Execute（legacy）
// 与 FinishInterp（延迟）两路共用；链接网格按精确 N（内核零改动）
void EpipolarInterpDualCuda::Impl::finishScanResult(EpipolarInterpResult& result,
                                                    int h_count, cudaStream_t cs) {
    result.success = true;
    result.message = "Success (scan mode)";
    result.interpCount = h_count;
    // 【速度优化】持久缓冲 ROI 视图（v3 契约）——原 clone() 每圈 cudaMalloc＋D2D
    result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>(
        d_scan_cpts_.colRange(0, h_count));

    // 段链接：行间簇合并 → 段 ID 写入 d_interp_line_ids
    // （禁用时输出簇 ID —— 同段行间不重号，仅断口不合并）
    bool linked = false;
    if (params_.scan_link_dx > 0.0f && h_count > 1) {
        // 【WDDM 滴流修复】延迟路径：链接内核已在提交段按容量发射（在飞），
        // 此处只建视图——d_link_label_[0,h_count) 与精确 N 版逐位一致（见
        // submitLinkAtCapacity 正确性注）。发射失败回退：lastLinkAtCap_=false
        // 走下方精确版逐发（部分在飞容量版内核被精确版覆写，无害）。
        const bool linkInFlight = lastLinkAtCap_;
        lastLinkAtCap_ = false;   // 消费即清（含 h_count≤1 的空耗路径）
        if (linkInFlight) {
            linked = true;
        } else {
        // 【速度优化】链接缓冲 grow-only；计时事件持久化（原每圈 create/destroy）
        if (d_link_parent_.cols < h_count) {
            d_link_parent_.create(1, h_count, CV_32SC1);
            d_link_label_.create(1, h_count, CV_32SC1);
        }
        const int gridL = (h_count + 255) / 256;
        // [诊断计时·门控] 段链接 GPU 耗时 → result.scan_link_ms——仅
        // JMW_LINKTIME=1 才付事件同步（生产路径零等待，免每眼一次 sync）
        const bool linkTime_ = getenv("JMW_LINKTIME") != nullptr;
        if (linkTime_) {
            if (!ev_link_a_) { cudaEventCreate(&ev_link_a_); cudaEventCreate(&ev_link_b_); }
            cudaEventRecord(ev_link_a_, cs);
        }
        dualScanLinkParent<<<gridL, 256, 0, cs>>>(
            d_scan_cpts_.ptr<float2>(), h_count,
            params_.epipolar_row_step, params_.scan_link_dx,
            params_.scan_link_gap_rows, d_link_parent_.ptr<int>(),
            d_link_label_.ptr<int>());   // 【刀三微】双写——建链后 D2D 拷贝消灭
        cudaError_t link_err = cudaGetLastError();
        if (link_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("dualScanLinkParent failed: ") + cudaGetErrorString(link_err);
            CALIB_LOG_ERROR("finishScanResult(): {}", result.message);
            return;
        }
        // 【刀三微】跳转轮数封顶：父链每步必跨行（rd>0 强制）→ 链深 ≤ 行数 8192
        // → 2^13 轮 doubling 结构性收敛（原 ceil(log2(h_count)) 可到 18）
        int rounds = 1;
        while ((1 << rounds) < h_count) ++rounds;
        if (rounds > 13) rounds = 13;
        for (int r = 0; r < rounds; ++r)
            dualScanLinkJump<<<gridL, 256, 0, cs>>>(d_link_label_.ptr<int>(), h_count);
        link_err = cudaGetLastError();
        if (link_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("dualScanLinkJump failed: ") + cudaGetErrorString(link_err);
            CALIB_LOG_ERROR("finishScanResult(): {}", result.message);
            return;
        }
        if (linkTime_) {
            cudaEventRecord(ev_link_b_, cs);
            cudaEventSynchronize(ev_link_b_);
            float link_ms = 0.f;
            cudaEventElapsedTime(&link_ms, ev_link_a_, ev_link_b_);
            cudaGetLastError();   // 【R3-低7】探针错误码自清账（防毒害后续消费者）
            result.scan_link_ms = static_cast<double>(link_ms);
        }
        }   // 【WDDM 滴流修复】else（精确 N 版逐发）结束
        linked = true;
    }
    // 【速度优化】持久缓冲 ROI 视图（v3 契约）——原 clone() 每圈 cudaMalloc＋D2D
    if (linked)
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(
            d_link_label_.colRange(0, h_count));
    else
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(
            d_scan_cfids_.colRange(0, h_count));
}

void EpipolarInterpDualCuda::Impl::Warmup(int pointCount) {
    if (pointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid pointCount={}, skipping", pointCount);
        return;
    }
    if (warmed_up_ && warmup_count_ == pointCount) return;

    ensureCapacity(pointCount);
    warmed_up_ = true;
    warmup_count_ = pointCount;
    CALIB_LOG_INFO("Warmup(): ensured GPU capacity for {} points", pointCount);
}

EpipolarInterpResult EpipolarInterpDualCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    DualScopedFlag guard(&inProcess_);
#endif

    EpipolarInterpResult result;
    lastLinkAtCap_ = false;   // 【WDDM 滴流修复】防御：legacy 路径不吃延迟路径残留旗

    // 【测速工具】阶段计时（host 口径；gpu_stress 聚合用）
    const auto tT0_ = std::chrono::steady_clock::now();
    auto tMark_ = tT0_;
    auto tSince_ = [&tMark_]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tMark_).count();
        tMark_ = now;
        return ms;
    };
    std::string tBreakdown_;

    try {
        int pointCount = d_points.rows * d_points.cols;

        if (pointCount < 2) {
            result.success = true;
            result.message = "Less than 2 points, no interpolation";
            result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>();
            result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.interpCount = 0;
            return result;
        }

        ensureCapacity(pointCount);
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
        const int gridN = (pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // ================= 模式判别 =================
        bool scanMode;
        if (params_.mode == InterpMode::Labeled) {
            scanMode = false;
        } else if (params_.mode == InterpMode::Scan) {
            scanMode = true;
        } else {
            // Auto: 线号全部相同（含全 0）→ 扫描；存在多线号 → 标定
            dualInitIdRange<<<1, 32, 0, cs>>>(d_meta_.ptr<int>());
            dualIdRange<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_line_ids.ptr<int>(), pointCount, d_meta_.ptr<int>());
            int h_range[2] = {0, 0};
            cudaMemcpyAsync(h_range, d_meta_.ptr<int>(), 2 * sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            scanMode = (h_range[0] == h_range[1]);
            CALIB_LOG_DEBUG("mode auto-detect: id[min={},max={}] -> {}",
                            h_range[0], h_range[1], scanMode ? "scan" : "labeled");
        }

        // ================= 排序 =================
        cudaError_t kernel_err;
        if (scanMode) {
            // ── Scan: 32bit y 键 ──
            dualBuildYKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), pointCount,
                reinterpret_cast<unsigned int*>(d_ykeys_in_.ptr<int>()),
                d_idx_in_.ptr<int>());

            if (radix_cap_ < pointCount || d_radix_temp_ == nullptr) {
                size_t bytes = 0;
                cub::DeviceRadixSort::SortPairs(
                    nullptr, bytes,
                    (const unsigned int*)nullptr, (unsigned int*)nullptr,
                    (const int*)nullptr, (int*)nullptr,
                    pointCount, 0, sizeof(unsigned int) * 8, cs);
                safeCudaFree(d_radix_temp_);
                cudaError_t err = cudaMalloc(&d_radix_temp_, bytes);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix temp alloc failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                radix_temp_size_ = bytes;
                radix_cap_ = pointCount;
            }
            {
                cudaError_t err = cub::DeviceRadixSort::SortPairs(
                    d_radix_temp_, radix_temp_size_,
                    reinterpret_cast<const unsigned int*>(d_ykeys_in_.ptr<int>()),
                    reinterpret_cast<unsigned int*>(d_ykeys_out_.ptr<int>()),
                    d_idx_in_.ptr<int>(), d_idx_out_.ptr<int>(),
                    pointCount, 0, sizeof(unsigned int) * 8, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix sort failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }
            dualGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), nullptr,
                d_idx_out_.ptr<int>(),
                d_sorted_pts_.ptr<float2>(), nullptr, pointCount);
        } else {
            // ── Labeled: 64bit (fid,y) 键（与 opt 一致）──
            dualBuildSortKeys<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                reinterpret_cast<unsigned long long*>(d_keys_in_.ptr<double>()), pointCount);
            dualFillIndices<<<gridN, BLOCK_SIZE, 0, cs>>>(d_idx_in_.ptr<int>(), pointCount);

            if (radix_cap_ < pointCount || d_radix_temp_ == nullptr) {
                size_t bytes = 0;
                cub::DeviceRadixSort::SortPairs(
                    nullptr, bytes,
                    (const unsigned long long*)nullptr, (unsigned long long*)nullptr,
                    (const int*)nullptr, (int*)nullptr,
                    pointCount, 0, sizeof(unsigned long long) * 8, cs);
                safeCudaFree(d_radix_temp_);
                cudaError_t err = cudaMalloc(&d_radix_temp_, bytes);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix temp alloc failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                radix_temp_size_ = bytes;
                radix_cap_ = pointCount;
            }
            {
                cudaError_t err = cub::DeviceRadixSort::SortPairs(
                    d_radix_temp_, radix_temp_size_,
                    reinterpret_cast<const unsigned long long*>(d_keys_in_.ptr<double>()),
                    reinterpret_cast<unsigned long long*>(d_keys_out_.ptr<double>()),
                    d_idx_in_.ptr<int>(), d_idx_out_.ptr<int>(),
                    pointCount, 0, sizeof(unsigned long long) * 8, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("radix sort failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }
            dualGatherSorted<<<gridN, BLOCK_SIZE, 0, cs>>>(
                d_points.ptr<float2>(), d_line_ids.ptr<int>(),
                d_idx_out_.ptr<int>(),
                d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(), pointCount);
        }

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Sort kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        if (scanMode) {
            // ================= Scan: 逐行聚类插值 + 压缩 =================
            const int totalSlots = SCAN_MAX_ROWS * SCAN_SLOTS;
            d_scan_flags_.create(1, totalSlots, CV_32SC1);
            d_scan_out_pts_.create(1, totalSlots, CV_32FC2);
            d_scan_out_fids_.create(1, totalSlots, CV_32SC1);
            d_scan_cpts_.create(1, totalSlots, CV_32FC2);
            d_scan_cfids_.create(1, totalSlots, CV_32SC1);
            d_scan_ccount_.create(1, 1, CV_32SC1);

            cudaMemsetAsync(d_scan_flags_.ptr<int>(), 0, totalSlots * sizeof(int), cs);

            dualScanInterp<<<SCAN_MAX_ROWS, 32, 0, cs>>>(
                d_sorted_pts_.ptr<float2>(), pointCount,
                params_.epipolar_row_step, params_.effectiveWindow(),
                params_.max_x_diff, params_.scan_x_gap,
                d_scan_flags_.ptr<int>(),
                d_scan_out_pts_.ptr<float2>(), d_scan_out_fids_.ptr<int>());

            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                result.success = false;
                result.message = std::string("dualScanInterp failed: ") + cudaGetErrorString(kernel_err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }

            if (cub_cap_ < totalSlots || d_cub_temp_ == nullptr) {
                size_t t1 = 0, t2 = 0;
                cub::DeviceSelect::Flagged(nullptr, t1,
                    (const float2*)nullptr, (const int*)nullptr,
                    (float2*)nullptr, (int*)nullptr, totalSlots, cs);
                cub::DeviceSelect::Flagged(nullptr, t2,
                    (const int*)nullptr, (const int*)nullptr,
                    (int*)nullptr, (int*)nullptr, totalSlots, cs);
                const size_t need = t1 > t2 ? t1 : t2;
                safeCudaFree(d_cub_temp_);
                cudaError_t err = cudaMalloc(&d_cub_temp_, need);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                cub_temp_size_ = need;
                cub_cap_ = totalSlots;
            }
            {
                cudaError_t err = cub::DeviceSelect::Flagged(
                    d_cub_temp_, cub_temp_size_,
                    (const float2*)d_scan_out_pts_.ptr<float2>(),
                    (const int*)d_scan_flags_.ptr<int>(),
                    d_scan_cpts_.ptr<float2>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
                err = cub::DeviceSelect::Flagged(
                    d_cub_temp_, cub_temp_size_,
                    (const int*)d_scan_out_fids_.ptr<int>(),
                    (const int*)d_scan_flags_.ptr<int>(),
                    d_scan_cfids_.ptr<int>(), d_scan_ccount_.ptr<int>(), totalSlots, cs);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                    CALIB_LOG_ERROR("Execute(): {}", result.message);
                    return result;
                }
            }

            int h_count = 0;
            cudaMemcpyAsync(&h_count, d_scan_ccount_.ptr<int>(), sizeof(int),
                            cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            // 【仪表税开关】JMW_NOTBD=1：跳过分相计时串拼装（纯工程口径）
            static const bool noBd_ = (getenv("JMW_NOTBD") != nullptr);
            if (!noBd_) tBreakdown_ += fmt::format(" sortInterp={:.2f}", tSince_());

            finishScanResult(result, h_count, cs);
            if (!noBd_)
                tBreakdown_ += fmt::format(" linkOut={:.2f} total={:.2f}", tSince_(),
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tT0_).count());
            last_breakdown_ = tBreakdown_;
            return result;
        }

        // ================= Labeled: 线号表 + y 范围（与 opt 一致） =================
        dualBoundaryFlags<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(), pointCount);

        if (scan_cap_ < pointCount || d_scan_temp_ == nullptr) {
            size_t bytes = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr, bytes, (const int*)nullptr, (int*)nullptr, pointCount, cs);
            safeCudaFree(d_scan_temp_);
            cudaError_t err = cudaMalloc(&d_scan_temp_, bytes);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("scan temp alloc failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            scan_temp_size_ = bytes;
            scan_cap_ = pointCount;
        }
        {
            cudaError_t err = cub::DeviceScan::ExclusiveSum(
                d_scan_temp_, scan_temp_size_,
                (const int*)d_rl_flags_.ptr<int>(), d_rl_ranks_.ptr<int>(), pointCount, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("exclusive sum failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        dualInitMeta<<<1, 32, 0, cs>>>(reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));
        dualScatterLineTable<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_fids_.ptr<int>(), d_rl_flags_.ptr<int>(),
            d_rl_ranks_.ptr<int>(), pointCount,
            d_line_ids_map_.ptr<int>(), d_line_begin_.ptr<int>(),
            reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));
        dualYRange<<<gridN, BLOCK_SIZE, 0, cs>>>(
            d_sorted_pts_.ptr<float2>(), pointCount,
            reinterpret_cast<unsigned int*>(d_meta_.ptr<int>()));

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Line table kernels failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        cudaMemcpyAsync(h_meta_pinned_, d_meta_.ptr<int>(), 3 * sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        static const bool noBdL_ = (getenv("JMW_NOTBD") != nullptr);   // 【仪表税开关】
        if (!noBdL_) tBreakdown_ += fmt::format(" lineTable={:.2f}", tSince_());

        const int num_lines = h_meta_pinned_[0];
        float y_min = 0.f, y_max = 0.f;
        {
            unsigned int umin = (unsigned int)h_meta_pinned_[1];
            unsigned int umax = (unsigned int)h_meta_pinned_[2];
            std::memcpy(&y_min, &umin, sizeof(float));
            std::memcpy(&y_max, &umax, sizeof(float));
        }

        const float step = params_.epipolar_row_step;
        const int row_begin = (int)std::floor(y_min / step);
        const int row_end = (int)std::ceil(y_max / step);
        const int num_rows = row_end - row_begin + 1;
        const int total_lines = num_lines * num_rows;

        if (total_lines > lines_cap_) {
            d_flags_.create(1, total_lines, CV_32SC1);
            d_temp_interp_.create(1, total_lines, CV_32FC2);
            d_temp_fids_.create(1, total_lines, CV_32SC1);
            d_output_.create(1, total_lines, CV_32FC2);
            d_output_fids_.create(1, total_lines, CV_32SC1);
            d_output_count_.create(1, 1, CV_32SC1);
            lines_cap_ = total_lines;
        }

        {
            const int gridL = (total_lines + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dualEpipolarInterp<<<gridL, BLOCK_SIZE, 0, cs>>>(
                d_sorted_pts_.ptr<float2>(), d_sorted_fids_.ptr<int>(),
                d_line_begin_.ptr<int>(), d_line_ids_map_.ptr<int>(),
                num_lines, num_rows, row_begin,
                step, params_.effectiveWindow(), params_.max_x_diff,
                d_flags_.ptr<int>(), d_temp_interp_.ptr<float2>(),
                d_temp_fids_.ptr<int>());
        }

        kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

        if (cub_cap_ < total_lines || d_cub_temp_ == nullptr) {
            size_t t1 = 0, t2 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                (const float2*)nullptr, (const int*)nullptr,
                (float2*)nullptr, (int*)nullptr, total_lines, cs);
            cub::DeviceSelect::Flagged(nullptr, t2,
                (const int*)nullptr, (const int*)nullptr,
                (int*)nullptr, (int*)nullptr, total_lines, cs);
            const size_t need = t1 > t2 ? t1 : t2;
            safeCudaFree(d_cub_temp_);
            cudaError_t err = cudaMalloc(&d_cub_temp_, need);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB temp storage allocation failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            cub_temp_size_ = need;
            cub_cap_ = total_lines;
        }

        {
            cudaError_t err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const float2*)d_temp_interp_.ptr<float2>(), (const int*)d_flags_.ptr<int>(),
                d_output_.ptr<float2>(), d_output_count_.ptr<int>(), total_lines, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
            err = cub::DeviceSelect::Flagged(
                d_cub_temp_, cub_temp_size_,
                (const int*)d_temp_fids_.ptr<int>(), (const int*)d_flags_.ptr<int>(),
                d_output_fids_.ptr<int>(), d_output_count_.ptr<int>(), total_lines, cs);
            if (err != cudaSuccess) {
                result.success = false;
                result.message = std::string("CUB::Flagged execution failed: ") + cudaGetErrorString(err);
                CALIB_LOG_ERROR("Execute(): {}", result.message);
                return result;
            }
        }

        int h_count = 0;
        cudaMemcpyAsync(&h_count, d_output_count_.ptr<int>(), sizeof(int),
                        cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        if (!noBdL_) tBreakdown_ += fmt::format(" compact={:.2f}", tSince_());

        result.success = true;
        result.message = "Success (labeled mode)";
        result.interpCount = h_count;
        // 【速度优化】持久缓冲 ROI 视图（v3 契约）——原 clone() 每圈 cudaMalloc＋D2D
        result.d_interpPoints = std::make_shared<cv::cuda::GpuMat>(
            d_output_.colRange(0, h_count));
        result.d_interp_line_ids = std::make_shared<cv::cuda::GpuMat>(
            d_output_fids_.colRange(0, h_count));
        if (!noBdL_)
            tBreakdown_ += fmt::format(" total={:.2f}",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tT0_).count());
        last_breakdown_ = tBreakdown_;

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    }

    return result;
}

void EpipolarInterpDualCuda::Impl::SetParams(const EpipolarInterpDualParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("SetParams(): called while Execute() is running");
        throw std::runtime_error("[06-EpipolarInterpDualCuda] SetParams() called during Execute()");
    }
#endif

    params.validate();

    bool deviceChanged = (params.deviceId != params_.deviceId);

    params_ = params;

    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }

    warmed_up_ = false;
    warmup_count_ = 0;

    CALIB_LOG_INFO("SetParams(): params updated, warmup reset");
}

void EpipolarInterpDualCuda::Impl::Destroy() {
    // 【G1】图对象先行销毁（引用内部缓冲指针——先于缓冲释放）
    if (deferGraphExec_) { cudaGraphExecDestroy(deferGraphExec_); deferGraphExec_ = nullptr; }
    if (deferGraph_) { cudaGraphDestroy(deferGraph_); deferGraph_ = nullptr; }

    safeCudaFree(d_radix_temp_);
    safeCudaFree(d_scan_temp_);
    safeCudaFree(d_cub_temp_);
    radix_temp_size_ = scan_temp_size_ = cub_temp_size_ = 0;
    radix_cap_ = scan_cap_ = cub_cap_ = 0;
    if (ev_link_a_) { cudaEventDestroy(ev_link_a_); ev_link_a_ = nullptr; }
    if (ev_link_b_) { cudaEventDestroy(ev_link_b_); ev_link_b_ = nullptr; }
    if (h_meta_pinned_) {
        cudaFreeHost(h_meta_pinned_);
        h_meta_pinned_ = nullptr;
    }
    d_keys_in_.release();
    d_keys_out_.release();
    d_ykeys_in_.release();
    d_ykeys_out_.release();
    d_idx_in_.release();
    d_idx_out_.release();
    d_sorted_pts_.release();
    d_sorted_fids_.release();
    d_rl_flags_.release();
    d_rl_ranks_.release();
    d_line_begin_.release();
    d_line_ids_map_.release();
    d_meta_.release();
    d_flags_.release();
    d_temp_interp_.release();
    d_temp_fids_.release();
    d_output_.release();
    d_output_fids_.release();
    d_output_count_.release();
    d_scan_flags_.release();
    d_scan_out_pts_.release();
    d_scan_out_fids_.release();
    d_scan_cpts_.release();
    d_scan_cfids_.release();
    d_scan_ccount_.release();
    capacity_ = 0;
    lines_cap_ = 0;
    warmed_up_ = false;
    warmup_count_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}
