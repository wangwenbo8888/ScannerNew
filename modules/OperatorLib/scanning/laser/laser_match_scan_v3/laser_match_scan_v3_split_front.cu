/**
 * @file laser_match_scan_v3_split_front.cu
 * @brief 激光线匹配扫描 V3 流水线拆分 — GPU 前段实现（Step0-6＋紧缩＋D2H＋同步）
 *
 * 2026-10-21 流水线拆分（详 laser_match_v3_split_设计方案.md）：
 *   - Impl::executeFront 只跑 process() 的前半段（右排序/CSR＋发卡＋GPU 段表），
 *     发射参数从 process() 普通发射路径逐字拷贝（不走 CUDA Graph）
 *   - 卡表紧缩：CUB ExclusiveSum(card_count)→偏移＋kernelTightenCardsSplitV3
 *     写三路 SoA 紧缩数组（lid/hitRx/hitRidx），下载量 ~1-2MB（stride-32 全表的 1/10）
 *   - 两段式同步（与原版结构一致）：sync#1 下 counts/diag/cardOff/cardCnt（得段数
 *     与总卡数）→ sync#2 下主批次（紧缩卡＋segmap＋leftStatus＋左点/track）
 *   - 成功末尾 status_gen_++（与 Execute 同代际契约：视图有效期至下一次
 *     Execute/ExecuteFront/Destroy/setParams）
 *
 * 原算子 process() 与全部既有内核零改动；本文件经 extern 原型跨 TU 发射
 * 原算子内核（全局命名空间、外部链接——CUDA 标准用法，失配链接期暴露）。
 */

#include "laser_match_scan_v3_cuda_pimpl.h"
#include "../laser_match_scan_v3_split/laser_match_scan_v3_split.h"
#include "common/calib_logging.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cub/cub.cuh>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace calib;

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCudaV3);

#if BUILD_CUDA

// ============================================================================
// 常量（同值拷贝自 laser_match_scan_v3_cuda_impl.cu 文件级 static——勿漂移）
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int V3_MAX_CARDS = 32;   // per-point card cap (nearest card per lid)

// ============================================================================
// 既有内核 extern 原型（逐字拷贝自 laser_match_scan_v3_cuda_impl.cu 定义；
// 勿改签名（含 __restrict__）；勿加 static（内部链接会 LNK2001））
// ============================================================================

__global__ void kernelGenKeysIdxV3(
    const float2* __restrict__ d_points,
    int count,
    float epipolar_row_step,
    unsigned int* __restrict__ d_keys,
    int* __restrict__ d_idx_in);

__global__ void kernelScatterSortedV3(
    const float2* __restrict__ d_src_pts,
    const int*    __restrict__ d_src_lids,
    const int*    __restrict__ d_sorted_idx,
    int count,
    float2* __restrict__ d_dst_pts,
    int*    __restrict__ d_dst_lids);

__global__ void kernelBuildCSRV3(
    const unsigned int* __restrict__ d_sort_keys,
    int count,
    int numRows,
    int* __restrict__ d_row_start,
    int* __restrict__ d_row_count);

__global__ void kernelIssueCardsV3(
    const float2* __restrict__ d_left_pts,
    int leftCount,
    const float4* __restrict__ d_map_byrow,
    const int*    __restrict__ d_map_row_start,
    const int*    __restrict__ d_map_row_count,
    const float2* __restrict__ d_right_sorted_pts,
    const int*    __restrict__ d_right_row_start,
    const int*    __restrict__ d_right_row_count,
    float epipolar_row_step,
    float vL_tolerance,
    float match_window,
    int*   __restrict__ d_card_lid,
    float* __restrict__ d_card_hitRx,
    float* __restrict__ d_card_res,
    int*   __restrict__ d_card_hitRidx,
    int*   __restrict__ d_card_count,
    const float* __restrict__ d_ur_bias,
    const float4* __restrict__ d_bias_params,
    int*   __restrict__ d_left_status,
    int*   __restrict__ d_max_out,
    const int* __restrict__ d_bucket_pool,
    const int* __restrict__ d_bucket_row_off,
    float bucket_min_x, float bucket_inv_w, int bucket_k, int use_buckets,
    const int* __restrict__ d_n = nullptr);   //【前段提前提交】设备计数尾闸（与定义同步）

__global__ void kernelCopySeq(
    const int* __restrict__ src,
    int count,
    int* __restrict__ dstKeys,
    int* __restrict__ dstIdx);

__global__ void kernelRunHeadsV3(
    const int* __restrict__ sortedTracks,
    int count,
    int* __restrict__ flags);

__global__ void kernelSegMapBackV3(
    const int* __restrict__ runRank,
    const int* __restrict__ flags,
    const int* __restrict__ sortedIdx,
    int count,
    int* __restrict__ segMap,
    int* __restrict__ segCountOut);

// ============================================================================
// 本文件新增内核
// ============================================================================

// 【刀二A】标签桶分类三件套（替代 CopySeq→radixSort→RunHeads→ExclSum→SegMapBack
// 五步 11 次发射；段号按升标签序分配＝radix 游程序，segMap 逐位一致——下游
// split_cpu 按原序迭代，组内序不离开 GPU）
constexpr int kV3LabelDomain = 8192 * 32;   // interp 簇 label 编码上界（row*32+slot）
constexpr int kV3ListCap = 8192;            // 【B2】非空桶清单容量（物理 ~72，113× 余量；超出走 Rank 兜底径）

// 【B2 两步法·核 A】Hist 首增者收桶号（探针 probe_v3.cu 78/78 逐位一致＋产线
// d_n 尾闸）。hist 拼槽布局：[0,262144)=计数（262144=守卫桶）；[262145]=listCnt
//（帧首 memset 承重——快径不自清 hist）。首增者恰每桶一次→清单＝非空桶集合。
__global__ void kernelLabelHistCollectV3(
    const int* __restrict__ labels,
    int count,
    int* __restrict__ hist,        // [kV3LabelDomain+2]
    int* __restrict__ list,        // [kV3ListCap]
    const int* __restrict__ d_n = nullptr)  // 【前段提前提交】设备计数尾闸：
                                            // 容量发射时尾部陈旧数据不得入桶/清单
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (d_n != nullptr && i >= *d_n) return;
    int lb = labels[i];
    if (lb < 0 || lb >= kV3LabelDomain) lb = kV3LabelDomain;   // 物理不出现——守卫
    int old = atomicAdd(&hist[lb], 1);
    if (old == 0) {
        int p = atomicAdd(&hist[kV3LabelDomain + 1], 1);   // listCnt＝hist 尾槽
        if (p < kV3ListCap) list[p] = lb;
    }
}

// 【B2 两步法·核 B】计数定秩 O(n²)（segId[b]＝|{list[j]<b}|＝升桶序发号＝radix
// 游程序，探针 78/78 逐位一致）；n>kV3ListCap 时核内回落原 BaseScan 全域三相
//（源体 :143-180（B2 前）逐字内嵌；n 为设备值——图捕获/延迟路安全）。
// ⚠ 块宽必须 ≤256：满占用形态（4×256/SM 驻留）下 1024 线程单块饿死（探针 Bd2：
// BaseScan P50 535→1134µs；Rank@128 同形态 6.1µs）——1024 会复活本批要杀的病灶。
__global__ void kernelLabelRankV3(
    int* __restrict__ hist,        // [kV3LabelDomain+2]（尾槽＝listCnt）
    const int* __restrict__ list,  // [kV3ListCap]
    int* __restrict__ segId,       // [kV3LabelDomain+1] 桶→段号
    int* __restrict__ cntOut)      // &counts[2]
{
    __shared__ int sh_n;
    if (threadIdx.x == 0) sh_n = hist[kV3LabelDomain + 1];
    __syncthreads();
    const int n = sh_n;
    if (n <= kV3ListCap) {
        // ── 快径：segId[b] = #{list[j] < b} ──
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int b = list[i];
            int r = 0;
            for (int j = 0; j < n; ++j)
                if (list[j] < b) ++r;
            segId[b] = r;
        }
        if (threadIdx.x == 0) {
            cntOut[0] = n;                        // counts[2]=segCount（raw，同 BaseScan）
            cntOut[1] = (n > 4096) ? 1 : 0;       // counts[3]=overflow（kernelSegMapBack 旧口径）
        }
    } else {
        // ── 兜底径：原 kernelLabelBaseScanV3 逐字（源体 bdim 动态通用；scan[1024]≥块宽）──
        const int tid = threadIdx.x;
        const int bdim = blockDim.x;
        const int domain = kV3LabelDomain + 1;
        const int chunk = (domain + bdim - 1) / bdim;
        const int begin = tid * chunk;
        const int end = min(begin + chunk, domain);
        int localCnt = 0;
        for (int b = begin; b < end; ++b)
            if (hist[b] > 0) ++localCnt;
        __shared__ int scan[1024];
        scan[tid] = localCnt;
        __syncthreads();
        for (int off = 1; off < bdim; off <<= 1) {
            int v = (tid >= off) ? scan[tid - off] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const int prefix = (tid == 0) ? 0 : scan[tid - 1];
        const int total = scan[bdim - 1];
        int running = prefix;
        for (int b = begin; b < end; ++b) {
            if (hist[b] > 0) segId[b] = running++;
            hist[b] = 0;   // 兜底径自清 hist[0..262144]
        }
        if (tid == 0) {
            cntOut[0] = total;
            cntOut[1] = (total > 4096) ? 1 : 0;
        }
    }
    // listCnt 尾槽复位：两径统一（快径稳态零；兜底径防陈值）——帧首 memset(domain+2) 双保险
    if (threadIdx.x == 0) hist[kV3LabelDomain + 1] = 0;
}

// 单块两相扫描：非空桶按升序发段号（=radix 游程序）＋顺带清零 hist（自清洁）
// ＋末尾写 counts[2]=segCount、counts[3]=overflow（对齐 SegMapBack 口径）
__global__ void kernelLabelBaseScanV3(
    int* __restrict__ hist,
    int* __restrict__ segId,        // [domain+1] 出
    int* __restrict__ cntOut)       // &counts[2]
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;    // 1024
    const int domain = kV3LabelDomain + 1;
    const int chunk = (domain + bdim - 1) / bdim;
    const int begin = tid * chunk;
    const int end = min(begin + chunk, domain);
    // 相 1：本线程连续段内非空计数
    int localCnt = 0;
    for (int b = begin; b < end; ++b)
        if (hist[b] > 0) ++localCnt;
    // 相 2：块内排他扫描（Hillis-Steele；线程序＝升桶序）
    __shared__ int scan[1024];
    scan[tid] = localCnt;
    __syncthreads();
    for (int off = 1; off < bdim; off <<= 1) {
        int v = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    const int prefix = (tid == 0) ? 0 : scan[tid - 1];
    const int total = scan[bdim - 1];
    // 相 3：重走本段发段号＋清零 hist
    int running = prefix;
    for (int b = begin; b < end; ++b) {
        if (hist[b] > 0) segId[b] = running++;
        hist[b] = 0;
    }
    if (tid == 0) {
        cntOut[0] = total;                       // counts[2]=segCount
        cntOut[1] = (total > 4096) ? 1 : 0;      // counts[3]=overflow（对齐旧口径）
    }
}

__global__ void kernelSegMapWriteV3(
    const int* __restrict__ labels,
    const int* __restrict__ segId,
    int count,
    int* __restrict__ segMap,
    const int* __restrict__ d_n = nullptr)  // 【前段提前提交】设备计数尾闸
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (d_n != nullptr && i >= *d_n) return;
    int lb = labels[i];
    if (lb < 0 || lb >= kV3LabelDomain) lb = kV3LabelDomain;
    segMap[i] = segId[lb];
}

// ============================================================================
// 【刀二B】右路行桶四件套（替代 GenKeys→radixSort→Scatter→memset×2→BuildCSR
// 共 11 次 API；行内 (xq,origIdx) 严格全序＝radix 稳定序 → sorted_pts/lids 与
// CSR 逐位一致——IssueCards 的行窗二分契约不变）
// ============================================================================
constexpr int kV3RightRows = 8192;     // MAX_EPIPOLAR_ROWS
constexpr unsigned int kV3XqOne = 256u;
constexpr unsigned int kV3XqMask = (1u << 19) - 1u;

// 与 kernelGenKeysIdxV3 同式：row<0 钳 0（并入 0 行）；row≥8192 → 死桶 8192
//（radix 中为 KEY_SENTINEL 沉尾、CSR 丢弃——桶版直接不进有效槽，尾部不再有
//  哨兵残点；IssueCards 只在 rCnt>0 窗内读，语义等价）
__device__ __forceinline__ int v3RightRowOf(float y, float step) {
    int row = static_cast<int>(roundf(y / step));
    if (row < 0) row = 0;
    if (row > kV3RightRows) row = kV3RightRows;   // 8192＝死桶
    return row;
}

__global__ void kernelRightRowHistV3(
    const float2* __restrict__ pts,
    int count,
    float step,
    int* __restrict__ hist,   // [8193]（8192＝死桶；扫描核自清洁）
    const int* __restrict__ d_n = nullptr)  // 【前段提前提交】设备计数尾闸
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (d_n != nullptr && i >= *d_n) return;
    atomicAdd(&hist[v3RightRowOf(pts[i].y, step)], 1);
}

// 单块扫描：row_start[r]=排他前缀（仅有效行 0..8191）、row_count[r]=hist[r]、
// cursor=副本；顺带清零 hist（自清洁）；死桶计数弃（不进 CSR）
__global__ void kernelRightRowScanV3(
    int* __restrict__ hist,          // [8193] 入＝计数/出＝全零
    int* __restrict__ row_start,     // [8192] 出
    int* __restrict__ row_count,     // [8192] 出
    int* __restrict__ cursor)        // [8192] 出（=row_start 副本）
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;    // 1024
    const int domain = kV3RightRows;   // 仅有效行（死桶只清不进 CSR）
    const int chunk = (domain + bdim - 1) / bdim;
    const int begin = tid * chunk;
    const int end = min(begin + chunk, domain);
    int localCnt = 0;
    for (int b = begin; b < end; ++b) localCnt += hist[b];
    __shared__ int scan[1024];
    scan[tid] = localCnt;
    __syncthreads();
    for (int off = 1; off < bdim; off <<= 1) {
        int v = (tid >= off) ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    const int prefix = (tid == 0) ? 0 : scan[tid - 1];
    int running = prefix;
    for (int b = begin; b < end; ++b) {
        const int c = hist[b];
        row_start[b] = running;
        row_count[b] = c;
        cursor[b] = running;
        running += c;
        hist[b] = 0;
    }
    if (tid == bdim - 1) hist[kV3RightRows] = 0;   // 死桶清零（不进 CSR）
}

__global__ void kernelRightMemberScatterV3(
    const float2* __restrict__ pts,
    int count,
    float step,
    int* __restrict__ cursor,       // [8192]
    int* __restrict__ members,      // [validCount]（origIdx）
    const int* __restrict__ d_n = nullptr)  // 【前段提前提交】设备计数尾闸
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (d_n != nullptr && i >= *d_n) return;
    const int row = v3RightRowOf(pts[i].y, step);
    if (row >= kV3RightRows) return;   // 死桶——不入槽
    const int slot = atomicAdd(&cursor[row], 1);
    members[slot] = i;
}

// 行内排序＋直写 sorted_pts/lids：键＝(xq<<32)|origIdx 严格全序＝radix 稳定序。
// 块/行；smem 位序（≤2048）；超界行（物理不现）退线程 0 插入序保正确
__device__ __forceinline__ unsigned long long v3RightSortKey(const float2& p, int idx) {
    float x = (p.x >= 0.0f) ? p.x : 0.0f;
    int xq = __float2int_rn(x * static_cast<float>(kV3XqOne));
    if (xq < 0) xq = 0;
    if (xq > static_cast<int>(kV3XqMask)) xq = static_cast<int>(kV3XqMask);
    return (static_cast<unsigned long long>(xq) << 32)
           | static_cast<unsigned long long>(static_cast<unsigned int>(idx));
}

__global__ void kernelRightRowSortScatterV3(
    const float2* __restrict__ pts,
    const int* __restrict__ lids,
    const int* __restrict__ row_start,
    const int* __restrict__ row_count,
    int* __restrict__ members,      // 就地排序（[start,start+cnt) 段内）
    float2* __restrict__ out_pts,
    int* __restrict__ out_lids)
{
    const int row = blockIdx.x;
    const int start = row_start[row];
    const int cnt = row_count[row];
    if (cnt == 0) return;
    const int tid = threadIdx.x;
    if (cnt <= 2048) {
        __shared__ unsigned long long key[2048];
        for (int k = tid; k < cnt; k += blockDim.x) {
            const int mi = members[start + k];
            key[k] = v3RightSortKey(pts[mi], mi);
        }
        __syncthreads();
        int np = 1;
        while (np < cnt) np <<= 1;
        for (int k = cnt + tid; k < np; k += blockDim.x)
            key[k] = 0xFFFFFFFFFFFFFFFFull;   // pad 沉底
        __syncthreads();
        for (int size = 2; size <= np; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int k = tid; k < np; k += blockDim.x) {
                    const int j = k ^ stride;
                    if (j > k) {
                        const bool up = ((k & size) == 0);
                        const unsigned long long a = key[k], b = key[j];
                        if ((up && a > b) || (!up && a < b)) {
                            key[k] = b;
                            key[j] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int k = tid; k < cnt; k += blockDim.x) {
            const int mi = static_cast<int>(
                static_cast<unsigned int>(key[k] & 0xFFFFFFFFu));
            out_pts[start + k] = pts[mi];
            out_lids[start + k] = lids[mi];
        }
    } else {
        // 兜底（行点数>2048——物理不现）：线程 0 插入序（同全序，保正确）
        if (tid == 0) {
            for (int k = 1; k < cnt; ++k) {
                const int mi = members[start + k];
                const unsigned long long kv = v3RightSortKey(pts[mi], mi);
                int j = k - 1;
                while (j >= 0 && v3RightSortKey(pts[members[start + j]],
                                                members[start + j]) > kv) {
                    members[start + j + 1] = members[start + j];
                    --j;
                }
                members[start + j + 1] = mi;
            }
            for (int k = 0; k < cnt; ++k) {
                const int mi = members[start + k];
                out_pts[start + k] = pts[mi];
                out_lids[start + k] = lids[mi];
            }
        }
    }
}

// 卡表紧缩（SoA 三路）：stride-32 布局 → 按偏移紧凑排布。
// 每点卡内顺序保持 stride 表顺序（对下游判定无影响：每 lid 至多一卡，
// 票计数/winner 卡查找/P6b hasW-hasS 均序无关——见设计方案 §9 算法评审 B 项）。
__global__ void kernelTightenCardsSplitV3(
    const int*   __restrict__ d_card_lid,       // [i*32 + c]
    const float* __restrict__ d_card_hitRx,
    const int*   __restrict__ d_card_hitRidx,
    const int*   __restrict__ d_card_count,     // N
    const int*   __restrict__ d_card_off,       // N（ExclusiveSum 前置）
    int leftCount,
    int*   __restrict__ d_tight_lid,            // totalCards
    float* __restrict__ d_tight_rx,
    int*   __restrict__ d_tight_ridx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int nc = d_card_count[i];
    if (nc <= 0) return;
    const int off = d_card_off[i];
    const int base = i * V3_MAX_CARDS;
    for (int c = 0; c < nc; ++c) {
        d_tight_lid[off + c] = d_card_lid[base + c];
        d_tight_rx[off + c] = d_card_hitRx[base + c];
        d_tight_ridx[off + c] = d_card_hitRidx[base + c];
    }
}

// ============================================================================
// 文件级小工具（同值拷贝自 impl.cu static——勿漂移）
// ============================================================================

static void splitFrontGrowOnly(cv::cuda::GpuMat& m, int cols, int type) {
    if (m.cols < cols || m.type() != type) m.create(1, cols, type);
}

#ifndef NDEBUG
// 重入守卫（同 impl.cu ScopedFlagV2）
class SplitFrontGuard {
public:
    explicit SplitFrontGuard(std::atomic<bool>* flag) : flag_(flag) {
        bool expect = false;
        if (!flag_->compare_exchange_strong(expect, true))
            throw std::runtime_error("LaserMatchScanCudaV3::executeFront() reentry");
    }
    ~SplitFrontGuard() { flag_->store(false); }
    SplitFrontGuard(const SplitFrontGuard&) = delete;
    SplitFrontGuard& operator=(const SplitFrontGuard&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// executeFront — 前半段流水线
// ============================================================================

void LaserMatchScanCudaV3::Impl::executeFront(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
#ifndef NDEBUG
    SplitFrontGuard guard(&inProcess_);
#endif

    const auto t0 = std::chrono::steady_clock::now();
    auto tMark = t0;
    auto tSince = [&tMark]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tMark).count();
        tMark = now;
        return ms;
    };
    std::string bd;

    out = LaserMatchSplitFrontV3{};   // 复位（含失败缺省）

    try {
        if (!tempSet_) {
            out.success = false;
            out.message = "Temperature not set: call setCurrentTemperature() first";
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }

        const int leftCount = d_left_points.rows * d_left_points.cols;
        const int rightCount = d_right_points.rows * d_right_points.cols;
        out.leftCount = leftCount;
        out.rightCount = rightCount;

        if (leftCount == 0 || rightCount == 0) {
            out.success = true;
            out.message = "No points to match";
            return;
        }

        if (d_left_points.type() != CV_32FC2 || d_right_points.type() != CV_32FC2 ||
            d_left_line_ids.type() != CV_32SC1 || d_right_line_ids.type() != CV_32SC1) {
            out.success = false;
            out.message = "Input type mismatch (expect 32FC2 points + 32SC1 ids)";
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }

        if (!allocateBuffers(leftCount, rightCount)) {
            out.success = false;
            out.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
        cudaError_t kernel_err;
        const int gen = status_gen_ & 1;

        // ── Step 0: 计数/诊断清零（counts[2]=segCount、[3]=overflow 由 SegMapBack 写）──
        cudaMemsetAsync(d_counts_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_diag_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);

        // ---- Step 1-4【刀二B：右路行桶＋行内排序——11 次 API→5；
        //      (xq,origIdx) 严格全序＝radix 稳定序 → sorted_pts/lids/CSR 逐位一致；
        //      死桶点不进有效槽（radix 中沉尾且 CSR 丢弃——IssueCards 只读 rCnt>0
        //      行窗，语义等价）。JMW_K2_LEGACY=1 回退旧 radix 路径（A/B 专用）】----
        static const bool k2Legacy_ = (getenv("JMW_K2_LEGACY") != nullptr
                                       && getenv("JMW_K2_LEGACY")[0] == '1');
        if (k2Legacy_) {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenKeysIdxV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(), rightCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
                d_right_idx_in_.ptr<int>());
            cub::DeviceRadixSort::SortPairs(
                d_cub_temp_, cub_temp_size_,
                reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
                reinterpret_cast<unsigned int*>(d_right_keys_out_.ptr<int>()),
                d_right_idx_in_.ptr<int>(), d_right_sorted_idx_.ptr<int>(),
                rightCount, 0, sizeof(unsigned int) * 8, cuda_stream);
            const int gridR = grid;
            kernelScatterSortedV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(), d_right_line_ids.ptr<int>(),
                d_right_sorted_idx_.ptr<int>(), rightCount,
                d_right_sorted_pts_.ptr<float2>(), d_right_sorted_lids_.ptr<int>());
            cudaMemsetAsync(d_right_row_start_.ptr<int>(), 0x7F,
                            MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
            cudaMemsetAsync(d_right_row_count_.ptr<int>(), 0,
                            MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
            kernelBuildCSRV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const unsigned int*>(d_right_keys_out_.ptr<int>()),
                rightCount, MAX_EPIPOLAR_ROWS,
                d_right_row_start_.ptr<int>(), d_right_row_count_.ptr<int>());
        } else {
            splitFrontGrowOnly(d_right_hist_, kV3RightRows + 1, CV_32SC1);
            splitFrontGrowOnly(d_right_cursor_, kV3RightRows, CV_32SC1);
            splitFrontGrowOnly(d_right_members_, rightCount, CV_32SC1);
            const int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemsetAsync(d_right_hist_.ptr<int>(), 0,
                            (kV3RightRows + 1) * sizeof(int), cuda_stream);
            kernelRightRowHistV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(), rightCount,
                params_.epipolar_row_step, d_right_hist_.ptr<int>());
            kernelRightRowScanV3<<<1, 1024, 0, cuda_stream>>>(
                d_right_hist_.ptr<int>(),
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>(),
                d_right_cursor_.ptr<int>());
            kernelRightMemberScatterV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(), rightCount,
                params_.epipolar_row_step, d_right_cursor_.ptr<int>(),
                d_right_members_.ptr<int>());
            kernelRightRowSortScatterV3<<<kV3RightRows, 256, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                d_right_line_ids.ptr<int>(),
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>(),
                d_right_members_.ptr<int>(),
                d_right_sorted_pts_.ptr<float2>(),
                d_right_sorted_lids_.ptr<int>());
        }
        cudaMemsetAsync(d_right_status_[gen].ptr<int>(), 0,
                        rightCount * sizeof(int), cuda_stream);

        // ---- Step 5: 发卡（【B3】TP2 网格＋useB 三态统一助手）----
        {
            const int gridL = issueGrid_(leftCount);
            kernelIssueCardsV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                leftCount,
                d_map_byrow_.ptr<float4>(),
                d_map_row_start_.ptr<int>(),
                d_map_row_count_.ptr<int>(),
                d_right_sorted_pts_.ptr<float2>(),
                d_right_row_start_.ptr<int>(),
                d_right_row_count_.ptr<int>(),
                params_.epipolar_row_step,
                params_.vL_tolerance,
                params_.match_threshold,
                d_card_lid_.ptr<int>(),
                d_card_hitRx_.ptr<float>(),
                d_card_res_.ptr<float>(),
                d_card_hitRidx_.ptr<int>(),
                d_card_count_.ptr<int>(),
                urBiasActive_ ? d_ur_bias_.ptr<float>() : nullptr,
                urBiasModelActive_ ? d_bias_params_.ptr<float4>() : nullptr,
                d_left_status_[gen].ptr<int>(),
                d_diag_.ptr<int>() + 1,
                bucketK_ > 0 && !d_bucket_pool_.empty()
                    ? d_bucket_pool_.ptr<int>() : nullptr,
                bucketK_ > 0 && !d_bucket_row_off_.empty()
                    ? d_bucket_row_off_.ptr<int>() : nullptr,
                bucketMinX_, bucketInvW_, bucketK_,
                issueUseBuckets_());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            out.success = false;
            out.message = std::string("front sort/CSR/issue failed: ")
                          + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }
        bd += fmt::format(" step1-5={:.2f}", tSince());

        // ---- Step 6: GPU 段表【刀二A：标签桶分类——11 次发射→3＋1 memset；
        //      段号分配序＝radix 升标签游程序 → segMap 逐位一致（静态论证）。
        //      JMW_K2_LEGACY=1 回退旧 radix 路径（A/B 专用）】----
        {
            splitFrontGrowOnly(d_seg_map_, leftCount, CV_32SC1);
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // 诊断拆分：JMW_K2A_LEGACY 仅回退段表路径（B 用 JMW_K2_LEGACY）
            static const bool k2aLegacy_ = (getenv("JMW_K2A_LEGACY") != nullptr
                                            && getenv("JMW_K2A_LEGACY")[0] == '1');
            if (k2aLegacy_) {
                splitFrontGrowOnly(d_seg_flags_, leftCount, CV_32SC1);
                splitFrontGrowOnly(d_seg_runrank_, leftCount, CV_32SC1);
                kernelCopySeq<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_left_line_ids.ptr<int>(), leftCount,
                    reinterpret_cast<int*>(d_left_keys_in_.ptr<int>()),
                    d_left_idx_in_.ptr<int>());
                cub::DeviceRadixSort::SortPairs(
                    d_cub_temp_, cub_temp_size_,
                    reinterpret_cast<unsigned int*>(d_left_keys_in_.ptr<int>()),
                    reinterpret_cast<unsigned int*>(d_left_keys_out_.ptr<int>()),
                    d_left_idx_in_.ptr<int>(), d_left_sorted_idx_.ptr<int>(),
                    leftCount, 0, sizeof(unsigned int) * 8, cuda_stream);
                kernelRunHeadsV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    reinterpret_cast<const int*>(d_left_keys_out_.ptr<int>()),
                    leftCount, d_seg_flags_.ptr<int>());
                size_t scanTemp = 0;
                cub::DeviceScan::ExclusiveSum(nullptr, scanTemp,
                    (const int*)nullptr, (int*)nullptr, leftCount, cuda_stream);
                if (scan_temp_size_ < scanTemp || d_scan_temp_ == nullptr) {
                    if (d_scan_temp_) cudaFree(d_scan_temp_);
                    cudaMalloc(&d_scan_temp_, scanTemp);
                    scan_temp_size_ = scanTemp;
                }
                cub::DeviceScan::ExclusiveSum(d_scan_temp_, scan_temp_size_,
                    d_seg_flags_.ptr<int>(), d_seg_runrank_.ptr<int>(),
                    leftCount, cuda_stream);
                kernelSegMapBackV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_seg_runrank_.ptr<int>(), d_seg_flags_.ptr<int>(),
                    d_left_sorted_idx_.ptr<int>(), leftCount,
                    d_seg_map_.ptr<int>(), d_counts_.ptr<int>() + 2);
            } else {
                // 【B2 两步法】Collect(首增者收桶号)→Rank(计数定秩/兜底全域扫)→SegMapWrite
                // 不变量：counts[2]=segCount、counts[3]=(>4096)、segId 只写非空桶、
                // segmap 逐位一致；hist 帧首 memset 承重（Rank 快径不自清洁）勿删
                splitFrontGrowOnly(d_label_hist_, kV3LabelDomain + 2, CV_32SC1);
                splitFrontGrowOnly(d_label_seg_, kV3LabelDomain + 1, CV_32SC1);
                splitFrontGrowOnly(d_label_list_, kV3ListCap, CV_32SC1);
                cudaMemsetAsync(d_label_hist_.ptr<int>(), 0,
                                (kV3LabelDomain + 2) * sizeof(int), cuda_stream);
                kernelLabelHistCollectV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_left_line_ids.ptr<int>(), leftCount,
                    d_label_hist_.ptr<int>(), d_label_list_.ptr<int>());
                kernelLabelRankV3<<<1, 256, 0, cuda_stream>>>(
                    d_label_hist_.ptr<int>(), d_label_list_.ptr<int>(),
                    d_label_seg_.ptr<int>(), d_counts_.ptr<int>() + 2);
                kernelSegMapWriteV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_left_line_ids.ptr<int>(), d_label_seg_.ptr<int>(), leftCount,
                    d_seg_map_.ptr<int>());
            }
        }

        // ---- 卡表偏移：ExclusiveSum(card_count)（同流复用 scan temp）----
        // 【修复】scan temp 分配原本只在 Step6 legacy 分支内——刀二A 新路径跳过
        // 该分支导致 temp 恒为 nullptr，ExclusiveSum 陷入查询模式不执行扫描，
        // d_card_off_ 永不初始化。此处无条件确保分配。
        {
            splitFrontGrowOnly(d_card_off_, leftCount, CV_32SC1);
            size_t scanNeed = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scanNeed,
                (const int*)nullptr, (int*)nullptr, leftCount, cuda_stream);
            if (scan_temp_size_ < scanNeed || d_scan_temp_ == nullptr) {
                if (d_scan_temp_) cudaFree(d_scan_temp_);
                cudaMalloc(&d_scan_temp_, scanNeed);
                scan_temp_size_ = scanNeed;
            }
            cub::DeviceScan::ExclusiveSum(d_scan_temp_, scan_temp_size_,
                d_card_count_.ptr<int>(), d_card_off_.ptr<int>(),
                leftCount, cuda_stream);
        }
        bd += fmt::format(" step6={:.2f}", tSince());

        // ── sync#1：小批量（counts/diag/cardOff/cardCnt）→ 段数/总卡数 ──
        int* h_cnt = hpin_counts_.ensure(4);
        int* h_dia = hpin_diag_.ensure(4);
        int* h_off = hpin_cardoff_.ensure(static_cast<size_t>(leftCount));
        int* h_ccn = hpin_cardcnt_.ensure(static_cast<size_t>(leftCount));
        if (!h_cnt || !h_dia || !h_off || !h_ccn) {
            out.success = false;
            out.message = "pinned host buffer alloc failed (sync1)";
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }
        cudaMemcpyAsync(h_cnt, d_counts_.ptr<int>(), 4 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_dia, d_diag_.ptr<int>(), 4 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_off, d_card_off_.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_ccn, d_card_count_.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        bd += fmt::format(" sync1={:.2f}", tSince());

        out.segCount = h_cnt[2] > 0 ? h_cnt[2] : 0;
        out.segOverflow = h_cnt[3];
        out.totalCards = h_off[leftCount - 1] + h_ccn[leftCount - 1];
        const int totalCards = out.totalCards;

        // ── 主批次：紧缩卡（totalCards==0 时跳过——空 GpuMat ptr() 在 Debug 下断言）＋
        //    segmap＋leftStatus＋左点/track ──
        int* h_tlid = nullptr;
        float* h_trx = nullptr;
        int* h_tridx = nullptr;
        if (totalCards > 0) {
            splitFrontGrowOnly(d_tight_lid_, totalCards, CV_32SC1);
            splitFrontGrowOnly(d_tight_rx_, totalCards, CV_32FC1);
            splitFrontGrowOnly(d_tight_ridx_, totalCards, CV_32SC1);
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelTightenCardsSplitV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_card_lid_.ptr<int>(), d_card_hitRx_.ptr<float>(),
                d_card_hitRidx_.ptr<int>(), d_card_count_.ptr<int>(),
                d_card_off_.ptr<int>(), leftCount,
                d_tight_lid_.ptr<int>(), d_tight_rx_.ptr<float>(),
                d_tight_ridx_.ptr<int>());
            h_tlid = hpin_tight_lid_.ensure(static_cast<size_t>(totalCards));
            h_trx = hpin_tight_rx_.ensure(static_cast<size_t>(totalCards));
            h_tridx = hpin_tight_ridx_.ensure(static_cast<size_t>(totalCards));
            if (!h_tlid || !h_trx || !h_tridx) {
                out.success = false;
                out.message = "pinned host buffer alloc failed (tight)";
                CALIB_LOG_ERROR("executeFront(): {}", out.message);
                return;
            }
            cudaMemcpyAsync(h_tlid, d_tight_lid_.ptr<int>(),
                            static_cast<size_t>(totalCards) * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_trx, d_tight_rx_.ptr<float>(),
                            static_cast<size_t>(totalCards) * sizeof(float),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_tridx, d_tight_ridx_.ptr<int>(),
                            static_cast<size_t>(totalCards) * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
        }
        int* h_segmap = hpin_segmap_.ensure(static_cast<size_t>(leftCount));
        int* h_lstat = hpin_leftstat_.ensure(static_cast<size_t>(leftCount));
        float2* h_left = hpin_left_.ensure(static_cast<size_t>(leftCount));
        int* h_ltrack = hpin_lefttrack_.ensure(static_cast<size_t>(leftCount));
        if (!h_segmap || !h_lstat || !h_left || !h_ltrack) {
            out.success = false;
            out.message = "pinned host buffer alloc failed (main)";
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }
        cudaMemcpyAsync(h_segmap, d_seg_map_.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_lstat, d_left_status_[gen].ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_left, d_left_points.ptr<float2>(),
                        static_cast<size_t>(leftCount) * sizeof(float2),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_ltrack, d_left_line_ids.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        bd += fmt::format(" dl2+sync2={:.2f}", tSince());

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            out.success = false;
            out.message = std::string("front stage failed: ")
                          + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("executeFront(): {}", out.message);
            return;
        }

        // ── 值语义拷出（脱离生命周期契约；~2MB）──
        out.leftPts.resize(static_cast<size_t>(leftCount));
        std::memcpy(out.leftPts.data(), h_left,
                    sizeof(cv::Point2f) * static_cast<size_t>(leftCount));
        out.leftTracks.assign(h_ltrack, h_ltrack + leftCount);
        out.segmap.assign(h_segmap, h_segmap + leftCount);
        out.leftStatus.assign(h_lstat, h_lstat + leftCount);
        out.cardPtr.assign(h_off, h_off + leftCount);
        out.cardPtr.push_back(totalCards);
        if (totalCards > 0) {
            out.cardLid.assign(h_tlid, h_tlid + totalCards);
            out.cardRx.assign(h_trx, h_trx + totalCards);
            out.cardRidx.assign(h_tridx, h_tridx + totalCards);
        } else {
            out.cardLid.clear();
            out.cardRx.clear();
            out.cardRidx.clear();
        }
        bd += fmt::format(" copyout={:.2f} total={:.2f}", tSince(),
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());

        status_gen_++;   // 代际推进（与 Execute 同契约）

        out.success = true;
        out.message = fmt::format(
            "OK front: N={} M={} segs={} cards={} overflow={}",
            leftCount, rightCount, out.segCount, totalCards, out.segOverflow);
        out.timing = bd;
        CALIB_LOG_DEBUG("executeFront(): {}", out.message);
    } catch (const std::exception& e) {
        out.success = false;
        out.message = std::string("Exception: ") + e.what();
        out.timing = bd;
        CALIB_LOG_ERROR("executeFront(): {}", out.message);
    } catch (...) {
        out.success = false;
        out.message = "Unknown exception";
        out.timing = bd;
        CALIB_LOG_ERROR("executeFront(): {}", out.message);
    }
}

// ============================================================================
// 【前段提前提交·WDDM 滴流根治】延迟对实现：提交段（排水前，容量定尺）＋
// 完成段（排水后，精确 N 收尾）。正确性三支柱：①聚合内核设备计数尾闸
//（hist/发卡/段表——陈尾不入结构）②ExclusiveSum 前缀只受左侧影响（容量界
// 下 [0,count) 前缀与精确版逐位一致）③紧缩/主批按宿主精确 N（排水后知），
// 与旧路径同式。
// ============================================================================

void LaserMatchScanCudaV3::Impl::executeFrontDeferred(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    deferActive_ = false;
    try {
        if (!tempSet_) {
            out.success = false;
            out.message = "Temperature not set: call setCurrentTemperature() first";
            CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
            return;
        }
        const int capL = d_left_points_cap.rows * d_left_points_cap.cols;
        const int capR = d_right_points_cap.rows * d_right_points_cap.cols;
        if (capL < 2 || capR < 2 || d_leftCount == nullptr || d_rightCount == nullptr) {
            out.success = false;
            out.message = "deferred: empty capacity views or null device counts";
            return;   // 执行器回退旧 executeFront
        }
        if (d_left_points_cap.type() != CV_32FC2
            || d_right_points_cap.type() != CV_32FC2
            || d_left_line_ids_cap.type() != CV_32SC1
            || d_right_line_ids_cap.type() != CV_32SC1) {
            out.success = false;
            out.message = "deferred: input type mismatch";
            CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
            return;
        }
        if (!allocateBuffers(capL, capR)) {
            out.success = false;
            out.message = "GPU buffer allocation failed (deferred)";
            CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
            return;
        }
        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
        const int gen = status_gen_ & 1;

        // ── Step 0：计数/诊断清零 ──
        cudaMemsetAsync(d_counts_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_diag_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);

        // ── Step 1-4：右路行桶（容量网格＋右计数尾闸；SortScatter 行表驱动天然安全）──
        splitFrontGrowOnly(d_right_hist_, kV3RightRows + 1, CV_32SC1);
        splitFrontGrowOnly(d_right_cursor_, kV3RightRows, CV_32SC1);
        splitFrontGrowOnly(d_right_members_, capR, CV_32SC1);
        const int gridR = (capR + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemsetAsync(d_right_hist_.ptr<int>(), 0,
                        (kV3RightRows + 1) * sizeof(int), cuda_stream);
        kernelRightRowHistV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_points_cap.ptr<float2>(), capR,
            params_.epipolar_row_step, d_right_hist_.ptr<int>(), d_rightCount);
        kernelRightRowScanV3<<<1, 1024, 0, cuda_stream>>>(
            d_right_hist_.ptr<int>(),
            d_right_row_start_.ptr<int>(),
            d_right_row_count_.ptr<int>(),
            d_right_cursor_.ptr<int>());
        kernelRightMemberScatterV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
            d_right_points_cap.ptr<float2>(), capR,
            params_.epipolar_row_step, d_right_cursor_.ptr<int>(),
            d_right_members_.ptr<int>(), d_rightCount);
        kernelRightRowSortScatterV3<<<kV3RightRows, 256, 0, cuda_stream>>>(
            d_right_points_cap.ptr<float2>(),
            d_right_line_ids_cap.ptr<int>(),
            d_right_row_start_.ptr<int>(),
            d_right_row_count_.ptr<int>(),
            d_right_members_.ptr<int>(),
            d_right_sorted_pts_.ptr<float2>(),
            d_right_sorted_lids_.ptr<int>());
        cudaMemsetAsync(d_right_status_[gen].ptr<int>(), 0,
                        static_cast<size_t>(capR) * sizeof(int), cuda_stream);

        // ── Step 5：发卡（容量网格＋左计数尾闸）──
        const int gridL = issueGrid_(capL);
        kernelIssueCardsV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_points_cap.ptr<float2>(), capL,
            d_map_byrow_.ptr<float4>(),
            d_map_row_start_.ptr<int>(),
            d_map_row_count_.ptr<int>(),
            d_right_sorted_pts_.ptr<float2>(),
            d_right_row_start_.ptr<int>(),
            d_right_row_count_.ptr<int>(),
            params_.epipolar_row_step,
            params_.vL_tolerance,
            params_.match_threshold,
            d_card_lid_.ptr<int>(),
            d_card_hitRx_.ptr<float>(),
            d_card_res_.ptr<float>(),
            d_card_hitRidx_.ptr<int>(),
            d_card_count_.ptr<int>(),
            urBiasActive_ ? d_ur_bias_.ptr<float>() : nullptr,
            urBiasModelActive_ ? d_bias_params_.ptr<float4>() : nullptr,
            d_left_status_[gen].ptr<int>(),
            d_diag_.ptr<int>() + 1,
            bucketK_ > 0 && !d_bucket_pool_.empty()
                ? d_bucket_pool_.ptr<int>() : nullptr,
            bucketK_ > 0 && !d_bucket_row_off_.empty()
                ? d_bucket_row_off_.ptr<int>() : nullptr,
            bucketMinX_, bucketInvW_, bucketK_,
            issueUseBuckets_(),
            d_leftCount);

        // ── Step 6：GPU 段表【B2 两步法】（容量网格＋左计数尾闸；Rank 全域安全）──
        splitFrontGrowOnly(d_seg_map_, capL, CV_32SC1);
        splitFrontGrowOnly(d_label_hist_, kV3LabelDomain + 2, CV_32SC1);
        splitFrontGrowOnly(d_label_seg_, kV3LabelDomain + 1, CV_32SC1);
        splitFrontGrowOnly(d_label_list_, kV3ListCap, CV_32SC1);
        cudaMemsetAsync(d_label_hist_.ptr<int>(), 0,
                        (kV3LabelDomain + 2) * sizeof(int), cuda_stream);
        kernelLabelHistCollectV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_line_ids_cap.ptr<int>(), capL,
            d_label_hist_.ptr<int>(), d_label_list_.ptr<int>(), d_leftCount);
        kernelLabelRankV3<<<1, 256, 0, cuda_stream>>>(
            d_label_hist_.ptr<int>(), d_label_list_.ptr<int>(),
            d_label_seg_.ptr<int>(), d_counts_.ptr<int>() + 2);
        kernelSegMapWriteV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
            d_left_line_ids_cap.ptr<int>(), d_label_seg_.ptr<int>(), capL,
            d_seg_map_.ptr<int>(), d_leftCount);

        // ── 卡表偏移前缀（容量界——[0,count) 前缀与精确版逐位一致）──
        splitFrontGrowOnly(d_card_off_, capL, CV_32SC1);
        size_t scanNeed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scanNeed,
            (const int*)nullptr, (int*)nullptr, capL, cuda_stream);
        if (scan_temp_size_ < scanNeed || d_scan_temp_ == nullptr) {
            if (d_scan_temp_) cudaFree(d_scan_temp_);
            cudaMalloc(&d_scan_temp_, scanNeed);
            scan_temp_size_ = scanNeed;
        }
        cub::DeviceScan::ExclusiveSum(d_scan_temp_, scan_temp_size_,
            d_card_count_.ptr<int>(), d_card_off_.ptr<int>(), capL, cuda_stream);

        // ── sync#1 小批 D2H（容量尺寸；宿主只读 [0,count)）──
        int* h_cnt = hpin_counts_.ensure(4);
        int* h_dia = hpin_diag_.ensure(4);
        int* h_off = hpin_cardoff_.ensure(static_cast<size_t>(capL));
        int* h_ccn = hpin_cardcnt_.ensure(static_cast<size_t>(capL));
        if (!h_cnt || !h_dia || !h_off || !h_ccn) {
            out.success = false;
            out.message = "pinned host buffer alloc failed (deferred sync1)";
            return;
        }
        cudaMemcpyAsync(h_cnt, d_counts_.ptr<int>(), 4 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_dia, d_diag_.ptr<int>(), 4 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_off, d_card_off_.ptr<int>(),
                        static_cast<size_t>(capL) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_ccn, d_card_count_.ptr<int>(),
                        static_cast<size_t>(capL) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        if (cudaGetLastError() != cudaSuccess) {
            out.success = false;
            out.message = "deferred submit failed (launch)";
            CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
            return;
        }
        deferActive_ = true;
        deferCapL_ = capL;
        deferCapR_ = capR;
        deferGen_ = gen;
        d_defer_ptsL_ = d_left_points_cap;   // 浅引用视图（完成段主批 D2H 源）
        d_defer_idsL_ = d_left_line_ids_cap;
        out.success = true;            // 提交级；真值在 finishFront 填
        out.leftCount = -1;            // 占位（finishFront 填精确 N）
        out.rightCount = -1;
        out.message = "OK front deferred submit";
    } catch (const std::exception& e) {
        out.success = false;
        out.message = std::string("deferred exception: ") + e.what();
        CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
    } catch (...) {
        out.success = false;
        out.message = "deferred unknown exception";
        CALIB_LOG_ERROR("executeFrontDeferred(): {}", out.message);
    }
}

// ============================================================================
// 【B4·前段提交图化】executeFrontGraph——JMW_V3FRONTGRAPH=1 显式开（默认关）。
// 机理：容量视图下提交段全参数跨帧稳定→单图发射消灭 ~19 笔 WDDM 逐内核放行
//（前段窗 1.9ms 空隙主源）。发射相位＝排水后（fcV3Front else 分支——不是已证伪
// 的 H2 defer 相位）。安全形态对齐 mask_sep 生产图：捕获前排空本流/零逐帧
// SetParams/签名兜底/失败永久回退。回退链：env 关→disabled→逐发提交段（现路径）。
// ============================================================================
void LaserMatchScanCudaV3::Impl::executeFrontGraph(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    static const bool fgFrontOn_ = [] {
        const char* e = getenv("JMW_V3FRONTGRAPH");
        return e != nullptr && e[0] == '1';
    }();
    if (!fgFrontOn_ || fgFrontDisabled_) {
        executeFrontDeferred(d_left_points_cap, d_left_line_ids_cap, d_leftCount,
                             d_right_points_cap, d_right_line_ids_cap, d_rightCount,
                             stream, out);
        return;
    }
    // 前置检查（与 deferred 同式；失败→回退逐发路径自带报错）
    const int capL = d_left_points_cap.rows * d_left_points_cap.cols;
    const int capR = d_right_points_cap.rows * d_right_points_cap.cols;
    if (!tempSet_ || capL < 2 || capR < 2
        || d_leftCount == nullptr || d_rightCount == nullptr
        || !allocateBuffers(capL, capR)) {
        executeFrontDeferred(d_left_points_cap, d_left_line_ids_cap, d_leftCount,
                             d_right_points_cap, d_right_line_ids_cap, d_rightCount,
                             stream, out);
        return;
    }
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
    const int gen = status_gen_ & 1;
    FrontGraph& F = fgs_[gen];

    // —— 签名取当前值（提交段全部触达指针＋容量＋值参数＋表 epoch）——
    FrontGraphSig cur{};
    cur.ptsL = d_left_points_cap.ptr();   cur.idsL = d_left_line_ids_cap.ptr();
    cur.ptsR = d_right_points_cap.ptr();  cur.idsR = d_right_line_ids_cap.ptr();
    cur.cntL = d_leftCount;               cur.cntR = d_rightCount;
    cur.capL = capL;                      cur.capR = capR;
    cur.rHist = d_right_hist_.ptr();      cur.rCursor = d_right_cursor_.ptr();
    cur.rMembers = d_right_members_.ptr();
    cur.rRowStart = d_right_row_start_.ptr(); cur.rRowCount = d_right_row_count_.ptr();
    cur.rSortedPts = d_right_sorted_pts_.ptr(); cur.rSortedLids = d_right_sorted_lids_.ptr();
    cur.rStatusSlot = d_right_status_[gen].ptr();   // ★gen 双缓冲（审核 R1 高危项）
    cur.lStatusSlot = d_left_status_[gen].ptr();
    cur.counts = d_counts_.ptr();         cur.diag = d_diag_.ptr();
    cur.segMap = d_seg_map_.ptr();        cur.labelHist = d_label_hist_.ptr();
    cur.labelSeg = d_label_seg_.ptr();    cur.labelList = d_label_list_.ptr();
    cur.cardOff = d_card_off_.ptr();
    cur.cardLid = d_card_lid_.ptr();      cur.cardHitRx = d_card_hitRx_.ptr();
    cur.cardRes = d_card_res_.ptr();      cur.cardHitRidx = d_card_hitRidx_.ptr();
    cur.cardCount = d_card_count_.ptr();
    cur.mapByrow = d_map_byrow_.ptr();    cur.mapRowStart = d_map_row_start_.ptr();
    cur.mapRowCount = d_map_row_count_.ptr();
    cur.bucketPool = (bucketK_ > 0 && !d_bucket_pool_.empty())
                         ? d_bucket_pool_.ptr() : nullptr;
    cur.bucketRowOff = (bucketK_ > 0 && !d_bucket_row_off_.empty())
                           ? d_bucket_row_off_.ptr() : nullptr;
    cur.urBias = urBiasActive_ ? d_ur_bias_.ptr() : nullptr;
    cur.biasParams = urBiasModelActive_ ? d_bias_params_.ptr() : nullptr;
    cur.scanTemp = d_scan_temp_;
    cur.hCnt = hpin_counts_.ensure(4);   cur.hDia = hpin_diag_.ensure(4);
    cur.hOff = hpin_cardoff_.ensure(static_cast<size_t>(capL));
    cur.hCcn = hpin_cardcnt_.ensure(static_cast<size_t>(capL));
    cur.rowStep = params_.epipolar_row_step;
    cur.vLtol = params_.vL_tolerance;    cur.matchWin = params_.match_threshold;
    cur.bMinX = bucketMinX_;             cur.bInvW = bucketInvW_;
    cur.bK = bucketK_;                   cur.useB = issueUseBuckets_();
    cur.epoch = mapEpoch_;

    auto setCtxAndOk = [&]() {
        deferActive_ = true;             // FinishFront 收尾（与 deferred 同协议）
        deferCapL_ = capL;               deferCapR_ = capR;   deferGen_ = gen;
        d_defer_ptsL_ = d_left_points_cap;
        d_defer_idsL_ = d_left_line_ids_cap;
        out.success = true;
        out.leftCount = -1;              // 哨兵＝提交级成功（FinishFront 填真值）
        out.rightCount = -1;
        out.message = "OK front graph";
    };
    auto failSlot = [&]() {
        if (F.exec) { cudaGraphExecDestroy(F.exec); F.exec = nullptr; }
        if (F.graph) { cudaGraphDestroy(F.graph); F.graph = nullptr; }
        F.sigValid = false;
        if (++F.attempts > 8) fgFrontDisabled_ = true;   // 永久回退（fg_ 先例）
    };

    // —— 稳态：签名相符→单发 ——
    if (F.sigValid) {
        if (memcmp(&cur, &F.sig, sizeof(cur)) == 0) {
            if (cudaGraphLaunch(F.exec, cs) == cudaSuccess) {
                setCtxAndOk();
                return;
            }
            cudaGetLastError();
            failSlot();
            fgFrontDisabled_ = true;     // 发射失败＝永久回退（1b 先例）
        } else {
            failSlot();                  // 失配→毁图重捕（下一轮走捕获）
        }
    }

    // —— 捕获（暖机 ≥2：摊平 growOnly/scan_temp/pinned 惰性分配——捕获区禁分配）——
    if (!fgFrontDisabled_ && ++fgFrontWarm_ >= 2) {
        cudaStreamSynchronize(cs);       // mask_sep 铁规：捕获前排空本流
        cudaError_t cerr = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
        LaserMatchSplitFrontV3 capOut{};
        if (cerr == cudaSuccess) {
            executeFrontDeferred(d_left_points_cap, d_left_line_ids_cap, d_leftCount,
                                 d_right_points_cap, d_right_line_ids_cap, d_rightCount,
                                 stream, capOut);
            cerr = capOut.success ? cudaSuccess : cudaErrorUnknown;
        }
        cudaGraph_t g = nullptr;
        if (cerr == cudaSuccess) cerr = cudaStreamEndCapture(cs, &g);
        cudaGraphExec_t ge = nullptr;
        if (cerr == cudaSuccess && g != nullptr)
            cerr = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        if (cerr == cudaSuccess && ge != nullptr) {
            if (F.exec) cudaGraphExecDestroy(F.exec);
            if (F.graph) cudaGraphDestroy(F.graph);
            F.graph = g; F.exec = ge; F.sig = cur; F.sigValid = true;
            size_t nn = 0; cudaGraphGetNodes(g, nullptr, &nn);
            CALIB_LOG_INFO("frontGraph captured gen={} nodes={} cap=({},{})",
                           gen, nn, capL, capR);
            if (cudaGraphLaunch(F.exec, cs) == cudaSuccess) {
                setCtxAndOk();           // 捕获不执行——本帧补真跑（fg_/1b 先例）
                return;
            }
            cudaGetLastError();
            failSlot();
            fgFrontDisabled_ = true;
        } else {
            if (g != nullptr) cudaGraphDestroy(g);
            cudaGetLastError();          // 捕获探测残留清账
            // 捕获失败＝未执行任何节点；deferred 已置 deferActive_——重跑兜底幂等
            CALIB_LOG_WARN("frontGraph capture failed gen={} ({})", gen,
                           cudaGetErrorString(cerr));
            ++F.attempts;
            if (F.attempts > 8) fgFrontDisabled_ = true;
        }
    }

    // —— 回退：逐发提交段（图关/disabled/暖机/捕获失败——现路径，幂等）——
    executeFrontDeferred(d_left_points_cap, d_left_line_ids_cap, d_leftCount,
                         d_right_points_cap, d_right_line_ids_cap, d_rightCount,
                         stream, out);
}

void LaserMatchScanCudaV3::Impl::finishFront(int leftCount, int rightCount,
                                             cv::cuda::Stream& stream,
                                             LaserMatchSplitFrontV3& out)
{
    if (!deferActive_) {
        out = LaserMatchSplitFrontV3{};
        out.success = false;
        out.message = "finishFront: no deferred context";
        CALIB_LOG_ERROR("finishFront(): {}", out.message);
        return;
    }
    deferActive_ = false;
    cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
    const int gen = deferGen_;
    // 提交段产物（含 sync#1 小批 D2H）由调用方排水保证就绪——防御性同步
    //（已 drained 时即时返回）
    cudaStreamSynchronize(cuda_stream);
    if (leftCount <= 0 || rightCount <= 0) {
        out = LaserMatchSplitFrontV3{};
        out.success = true;
        out.message = "No points to match";
        status_gen_++;
        return;
    }
    try {
        const int* h_cnt = hpin_counts_.ensure(4);
        const int* h_off = hpin_cardoff_.ensure(
            static_cast<size_t>(std::max(leftCount, deferCapL_)));
        const int* h_ccn = hpin_cardcnt_.ensure(
            static_cast<size_t>(std::max(leftCount, deferCapL_)));
        out.leftCount = leftCount;
        out.rightCount = rightCount;
        out.segCount = h_cnt[2] > 0 ? h_cnt[2] : 0;
        out.segOverflow = h_cnt[3];
        const int totalCards = h_off[leftCount - 1] + h_ccn[leftCount - 1];
        out.totalCards = totalCards;

        // ── 主批次：紧缩（精确网格）＋tight/main D2H＋sync#2（与旧路径同式）──
        int* h_tlid = nullptr;
        float* h_trx = nullptr;
        int* h_tridx = nullptr;
        if (totalCards > 0) {
            splitFrontGrowOnly(d_tight_lid_, totalCards, CV_32SC1);
            splitFrontGrowOnly(d_tight_rx_, totalCards, CV_32FC1);
            splitFrontGrowOnly(d_tight_ridx_, totalCards, CV_32SC1);
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelTightenCardsSplitV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_card_lid_.ptr<int>(), d_card_hitRx_.ptr<float>(),
                d_card_hitRidx_.ptr<int>(), d_card_count_.ptr<int>(),
                d_card_off_.ptr<int>(), leftCount,
                d_tight_lid_.ptr<int>(), d_tight_rx_.ptr<float>(),
                d_tight_ridx_.ptr<int>());
            h_tlid = hpin_tight_lid_.ensure(static_cast<size_t>(totalCards));
            h_trx = hpin_tight_rx_.ensure(static_cast<size_t>(totalCards));
            h_tridx = hpin_tight_ridx_.ensure(static_cast<size_t>(totalCards));
            if (!h_tlid || !h_trx || !h_tridx) {
                out.success = false;
                out.message = "pinned host buffer alloc failed (tight)";
                return;
            }
            cudaMemcpyAsync(h_tlid, d_tight_lid_.ptr<int>(),
                            static_cast<size_t>(totalCards) * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_trx, d_tight_rx_.ptr<float>(),
                            static_cast<size_t>(totalCards) * sizeof(float),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_tridx, d_tight_ridx_.ptr<int>(),
                            static_cast<size_t>(totalCards) * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
        }
        int* h_segmap = hpin_segmap_.ensure(static_cast<size_t>(leftCount));
        int* h_lstat = hpin_leftstat_.ensure(static_cast<size_t>(leftCount));
        float2* h_left = hpin_left_.ensure(static_cast<size_t>(leftCount));
        int* h_ltrack = hpin_lefttrack_.ensure(static_cast<size_t>(leftCount));
        if (!h_segmap || !h_lstat || !h_left || !h_ltrack) {
            out.success = false;
            out.message = "pinned host buffer alloc failed (main)";
            return;
        }
        // 主批 4 笔按精确 N（此刻宿主已知）；源＝容量视图基址（前 count 元素
        // 与精确视图同址——colRange 语义）
        cudaMemcpyAsync(h_segmap, d_seg_map_.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_lstat, d_left_status_[gen].ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        // 左点/track 从提交段保存的容量视图取——deferCtx 视图由桥接层持有
        cudaMemcpyAsync(h_ltrack, d_defer_idsL_.ptr<int>(),
                        static_cast<size_t>(leftCount) * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_left, d_defer_ptsL_.ptr<float2>(),
                        static_cast<size_t>(leftCount) * sizeof(float2),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        cudaError_t kernel_err;
        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            out.success = false;
            out.message = std::string("finishFront stage failed: ")
                          + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("finishFront(): {}", out.message);
            return;
        }

        // ── 值语义拷出（与旧路径同式）──
        out.leftPts.resize(static_cast<size_t>(leftCount));
        std::memcpy(out.leftPts.data(), h_left,
                    sizeof(cv::Point2f) * static_cast<size_t>(leftCount));
        out.leftTracks.assign(h_ltrack, h_ltrack + leftCount);
        out.segmap.assign(h_segmap, h_segmap + leftCount);
        out.leftStatus.assign(h_lstat, h_lstat + leftCount);
        out.cardPtr.assign(h_off, h_off + leftCount);
        out.cardPtr.push_back(totalCards);
        if (totalCards > 0) {
            out.cardLid.assign(h_tlid, h_tlid + totalCards);
            out.cardRx.assign(h_trx, h_trx + totalCards);
            out.cardRidx.assign(h_tridx, h_tridx + totalCards);
        } else {
            out.cardLid.clear();
            out.cardRx.clear();
            out.cardRidx.clear();
        }
        status_gen_++;   // 代际推进（与 Execute 同契约）
        out.success = true;
        out.message = fmt::format(
            "OK front(deferred): N={} M={} segs={} cards={} overflow={}",
            leftCount, rightCount, out.segCount, totalCards, out.segOverflow);
        CALIB_LOG_DEBUG("finishFront(): {}", out.message);
    } catch (const std::exception& e) {
        out.success = false;
        out.message = std::string("finishFront exception: ") + e.what();
        CALIB_LOG_ERROR("finishFront(): {}", out.message);
    } catch (...) {
        out.success = false;
        out.message = "finishFront unknown exception";
        CALIB_LOG_ERROR("finishFront(): {}", out.message);
    }
}



void LaserMatchScanCudaV3::ExecuteFront(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
    CALIB_LOG_DEBUG("ExecuteFront() called: left={}x{} right={}x{}",
                    d_left_points.cols, d_left_points.rows,
                    d_right_points.cols, d_right_points.rows);
    // 注意：空输入交由 Impl::executeFront 处理（先查 tempSet_ 再查空——与 process()
    // 的检查顺序一致；桥接处不提前短路，避免「未注温＋空帧」行为与原算子分歧）
    if (!pImpl_) {
        out = LaserMatchSplitFrontV3{};
        out.success = false;
        out.message = "ExecuteFront() after Destroy()";
        return;
    }
    pImpl_->executeFront(d_left_points, d_left_line_ids,
                         d_right_points, d_right_line_ids, stream, out);
}

void LaserMatchScanCudaV3::ExecuteFrontDeferred(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
    if (!pImpl_) {
        out = LaserMatchSplitFrontV3{};
        out.success = false;
        out.message = "ExecuteFrontDeferred() after Destroy()";
        return;
    }
    pImpl_->executeFrontDeferred(d_left_points_cap, d_left_line_ids_cap,
                                 d_leftCount, d_right_points_cap,
                                 d_right_line_ids_cap, d_rightCount, stream, out);
}

void LaserMatchScanCudaV3::ExecuteFrontGraph(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& out)
{
    if (!pImpl_) {
        out = LaserMatchSplitFrontV3{};
        out.success = false;
        out.message = "ExecuteFrontGraph() after Destroy()";
        return;
    }
    pImpl_->executeFrontGraph(d_left_points_cap, d_left_line_ids_cap,
                              d_leftCount, d_right_points_cap,
                              d_right_line_ids_cap, d_rightCount, stream, out);
}

void LaserMatchScanCudaV3::FinishFront(int leftCount, int rightCount,
                                       cv::cuda::Stream& stream,
                                       LaserMatchSplitFrontV3& out)
{
    if (!pImpl_) {
        out = LaserMatchSplitFrontV3{};
        out.success = false;
        out.message = "FinishFront() after Destroy()";
        return;
    }
    pImpl_->finishFront(leftCount, rightCount, stream, out);
}

#else // BUILD_CUDA

void LaserMatchScanCudaV3::ExecuteFront(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&,
    cv::cuda::Stream&, LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    out.success = false;
    out.message = "CUDA not available (BUILD_CUDA=OFF)";
}

void LaserMatchScanCudaV3::ExecuteFrontDeferred(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const int*,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const int*,
    cv::cuda::Stream&, LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    out.success = false;
    out.message = "CUDA not available (BUILD_CUDA=OFF)";
}

void LaserMatchScanCudaV3::FinishFront(
    int, int, cv::cuda::Stream&, LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    out.success = false;
    out.message = "CUDA not available (BUILD_CUDA=OFF)";
}

void LaserMatchScanCudaV3::ExecuteFrontGraph(
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const int*,
    const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const int*,
    cv::cuda::Stream&, LaserMatchSplitFrontV3& out)
{
    out = LaserMatchSplitFrontV3{};
    out.success = false;
    out.message = "CUDA not available (BUILD_CUDA=OFF)";
}

#endif // BUILD_CUDA
