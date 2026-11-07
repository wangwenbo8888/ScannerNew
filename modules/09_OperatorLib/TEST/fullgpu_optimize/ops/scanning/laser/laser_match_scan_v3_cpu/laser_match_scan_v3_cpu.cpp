/**
 * @file laser_match_scan_v3_cpu.cpp
 * @brief 激光线匹配扫描 V3 — 全 CPU 单线程备用算子实现（沙箱专用）
 *
 * 逐级移植自 laser_match_scan_v3_cuda_impl.cu（GPU 基准），移植要点：
 *   - 右行排序复刻 GPU 量化键稳定序（row<0 钳位 0；仅 row≥8192 丢弃；
 *     xq=nearbyintf(x*256) clamp＋origIdx 稳定 tie）
 *   - 发卡查表走桶索引锚点（kBucketK=256，与 GPU 默认路径逐字一致，含 8 步跳走）
 *   - 段映射 sort/unique/lower_bound ＝ GPU CUB 游程 rank（track id 非负前提）
 *   - 默认路径冲突双弃净语义：同一 (row,lid) 被 ≥2 段主张 → 全部主张段弃
 *     （已弃段仍参与占胞，对齐 kernelConflictMarkV3 不检查 drop 即 CAS 的行为）
 *   - P6b（segment_split）为 conflict_reelect 前置 modifier；路径分支键＝
 *     conflict_reelect（SetParams 不做 cascade→reelect 隐含归一，照抄 GPU）
 *   - P5/P6b/P7 宿主代码机械移植（判据/加法序列原样保留）
 *
 * 设计/评审记录：TEST/speed_optimize/laser_match_v3_cpu_设计方案.md
 */

#include "laser_match_scan_v3_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>   // _mm_prefetch / __cpuidex
#include <immintrin.h>
#endif

using namespace calib;

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCpuV3);

OperatorInfo getLaserMatchScanCpuV3Info() {
    return OperatorInfo{"LaserMatchScanCpuV3", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

// ============================================================================
// 常量（与 GPU 实现同值）
// ============================================================================

namespace {

constexpr int MAX_EPIPOLAR_ROWS = 8192;
constexpr int V3_MAX_CARDS = 32;      // 每点卡上限（每 lid 就近单卡）
constexpr int V3_MAX_LIDS = 32;       // lid 桶距（lid 1..32）
constexpr int kRowWords = MAX_EPIPOLAR_ROWS / 64;   // 128（段行位图字数）
constexpr unsigned int XQ_ONE = 256u;              // x 量化 1/256 px
constexpr unsigned int XQ_MASK = (1u << 19) - 1u;
constexpr int kBucketK = 256;                      // GPU 同款桶数（参照基准）
constexpr int kCpuBucketK = 1024;                   // CPU 专用桶数（更细桶→更短跳走；
                                                    // 锚点＝min(lower_bound,行末) 与 K 无关）

// MSVC 的 roundf/nearbyintf 为 CRT 调用（~25cyc）；行值域 <2^22 下按位等价的内联实现：
//  - froundAwayI：roundf 半值远离零语义（(int)(x±0.5f) 截断，|x|<2^22 精确）
//  - froundEvenI：__float2int_rn 就近偶（cvtsi2ss 的标量转换，MXCSR 默认就近偶）
inline int froundAwayI(float x) { return static_cast<int>(x >= 0.0f ? x + 0.5f : x - 0.5f); }
inline int froundEvenI(float x) { return _mm_cvt_ss2si(_mm_set_ss(x)); }

// uR 4 参数位置相关偏差模型常数（与 GPU 版逐字同值）
constexpr float URB_F  = 1231.139f;
constexpr float URB_B  = 133.311f;
constexpr float URB_CX = 830.523f;
constexpr float URB_CY = 685.0f;
constexpr float URB_TX = 39.847f;
constexpr float URB_TY = 8.087f;
constexpr float URB_TZ = 10.626f;

inline int clampRow(int row) { return row < 0 ? 0 : (row >= MAX_EPIPOLAR_ROWS ? MAX_EPIPOLAR_ROWS - 1 : row); }

// 发卡紧凑记录（CSR 元素：lid＋右排序系索引＋命中 x）
struct LmCard { int lid; int ridx; float hitRx; };

// 温表行内条目（12B AoS：锚点跳走/展开扫卡/uR 取值同一条缓存行流）
struct MapEnt { float x, uR; int lid; };

// 单点发卡（锚点→展开→HIT 验卡；noinline 独立函数体＝干净寄存器分配——
// 巨型循环体内 ymm/局部量混驻导致实测 4x 寄存器溢出税）
// 返回 {有效卡数, 展开卡数}；卡写入 cardOut（容量 ≥V3_MAX_CARDS）
__declspec(noinline) static std::pair<int, int> issueOnePoint(
    const cv::Point2f& p, int row,
    const MapEnt* ent, const int* mapRowStart, const int* mapRowCount,
    const int* bucketPool, const int* bucketRowOff,
    float bMinX, float bInvW, int bK,
    float vLtol, float matchWin,
    const float* rxSorted, const int* rRowStart, const int* rRowCount,
    bool biasModel, bool biasTable, const float* urBias, const float* biasParams,
    bool avx2, LmCard* cardOut)
{
    const float xL = p.x;
    const int mapCnt = mapRowCount[row];
    const int start = mapRowStart[row];

    // 首个 x≥xL 锚点（桶直达＋8 步跳走＋线性收尾；结果＝min(lower_bound,行末)，
    // 与 GPU 默认路径逐位一致；bK==0 退化表回退二分）
    int lo;
    if (bK > 0) {
        int b = static_cast<int>((xL - bMinX) * bInvW);
        if (b < 0) b = 0;
        if (b > bK - 1) b = bK - 1;
        const int poolBase = bucketRowOff[row];
        int j = start + bucketPool[static_cast<size_t>(poolBase + b)];
        const int jEnd = start + bucketPool[static_cast<size_t>(poolBase + b + 1)];
        while (j + 8 <= jEnd && ent[j + 7].x < xL) j += 8;
        while (j < jEnd && ent[j].x < xL) ++j;
        if (j >= start + mapCnt) j = start + mapCnt - 1;   // 行内无 x≥xL（对齐收尾）
        lo = j;
    } else {
        lo = start;
        int hi = start + mapCnt - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ent[mid].x < xL) lo = mid + 1; else hi = mid;
        }
    }

    // 双向展开：每 lid 就近单卡（cap 32）——lid→槽直达映射 O(1) 去重
    int nc = 0;
    int locLid[V3_MAX_CARDS];
    float locU[V3_MAX_CARDS];
    float locDx[V3_MAX_CARDS];
    unsigned char lidSlot[64];
    std::memset(lidSlot, 0xFF, sizeof(lidSlot));
    for (int dir = 0; dir < 2; ++dir) {
        int q = (dir == 0) ? lo : lo - 1;
        const int step = (dir == 0) ? 1 : -1;
        while (q >= start && q < start + mapCnt) {
            const float dx = ent[q].x - xL;
            if (std::fabs(dx) > vLtol) break;
            const int lid = ent[q].lid;
            const float uR = ent[q].uR;
            const float adx = std::fabs(dx);
            if (static_cast<unsigned>(lid - 1) < 32u) {
                const unsigned char cs = lidSlot[lid];
                if (cs == 0xFF) {
                    if (nc < V3_MAX_CARDS) {
                        locLid[nc] = lid; locU[nc] = uR; locDx[nc] = adx;
                        lidSlot[lid] = static_cast<unsigned char>(nc + 1);
                        ++nc;
                    }
                } else {
                    const int c = cs - 1;
                    if (adx < locDx[c]) {
                        locDx[c] = adx;
                        locU[c] = uR;
                    }
                }
            } else {
                // 越界 lid 防御（GPU 无守卫；线性回退保持同语义）
                int found = -1;
                for (int c = 0; c < nc; ++c)
                    if (locLid[c] == lid) { found = c; break; }
                if (found < 0) {
                    if (nc < V3_MAX_CARDS) {
                        locLid[nc] = lid; locU[nc] = uR; locDx[nc] = adx;
                        ++nc;
                    }
                } else if (adx < locDx[found]) {
                    locDx[found] = adx;
                    locU[found] = uR;
                }
            }
            q += step;
        }
    }

    // HIT 验卡（±match_window 取最左）＋ uR 补偿（互斥两模式）
    // 行向量按点预载：点的各卡共享同一右行（4×AVX2 载入复用于全部卡，≤32 元素）
    const int rStart = rRowStart[row];
    const int rCnt = rRowCount[row];
    const bool avx2Hit = avx2 && rCnt > 0 && rStart >= 0 && rCnt <= 32;
    __m256 rv0, rv1, rv2, rv3;
    unsigned rCntMask = 0;
    if (avx2Hit) {
        rv0 = _mm256_loadu_ps(rxSorted + rStart);
        rv1 = _mm256_loadu_ps(rxSorted + rStart + 8);
        rv2 = _mm256_loadu_ps(rxSorted + rStart + 16);
        rv3 = _mm256_loadu_ps(rxSorted + rStart + 24);
        rCntMask = (rCnt >= 32) ? 0xFFFFFFFFu : ((1u << rCnt) - 1u);
    }
    int out = 0;
    for (int c = 0; c < nc; ++c) {
        float bias = 0.0f;
        if (biasModel) {
            const float wxx = biasParams[0], wyy = biasParams[1], wzz = biasParams[2];
            const float Z = URB_F * URB_B / std::fmaxf(xL - locU[c], 1.0f);
            const float X = (xL - URB_CX) * Z / URB_F;
            const float Y = (p.y - URB_CY) * Z / URB_F;
            const float dx = X - URB_TX, dy = Y - URB_TY, dz = Z - URB_TZ;
            const float ddy = wzz * dx - wxx * dz;   // ω×d 一阶旋转（仅 v 向链进深度）
            const float ddz = wxx * dy - wyy * dx;
            const float dv  = URB_F * (ddy * dz - dy * ddz) / (dz * dz);
            const float dZ  = dv * Z / dz;
            bias = URB_F * URB_B * dZ / (Z * Z);
        } else if (biasTable && locLid[c] >= 1 && locLid[c] <= V3_MAX_LIDS) {
            bias = urBias[locLid[c] - 1];
        }
        const float uR = locU[c] + bias;
        float hitRx = -1.0f;
        int hitRidx = -1;
        if (rCnt > 0 && rStart >= 0) {
            // lower_bound（首 x ≥ uR−win）＝小于 bound 的元素数（有序数组）。
            // 掩码按 rCnt 截断防串行（越界读无害）；大行标量无分支二分
            const float bound = uR - matchWin;
            int rlo;
            if (avx2Hit) {
                const __m256 vb = _mm256_set1_ps(bound);
                unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(rv0, vb, _CMP_LT_OQ)))
                    | (static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(rv1, vb, _CMP_LT_OQ))) << 8)
                    | (static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(rv2, vb, _CMP_LT_OQ))) << 16)
                    | (static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(rv3, vb, _CMP_LT_OQ))) << 24);
                rlo = rStart + __popcnt(mask & rCntMask);
            } else if (rCnt <= 32) {
                const __m128 vb4 = _mm_set1_ps(bound);
                unsigned mask = 0;
                for (int v = 0; v < 8; ++v) {
                    const __m128 vv = _mm_loadu_ps(rxSorted + rStart + v * 4);
                    mask |= static_cast<unsigned>(_mm_movemask_ps(_mm_cmplt_ps(vv, vb4))) << (v * 4);
                }
                mask &= (1u << rCnt) - 1u;
                rlo = rStart + __popcnt(mask);
            } else {
                int rl = rStart;
                int n = rCnt;
                while (n > 1) {
                    const int half = n >> 1;
                    rl += (rxSorted[rl + half] < bound) ? half : 0;
                    n -= half;
                }
                if (rxSorted[rl] < bound) ++rl;
                rlo = rl;
            }
            if (rlo < rStart + rCnt &&
                std::fabs(rxSorted[rlo] - uR) <= matchWin) {
                hitRx = rxSorted[rlo];
                hitRidx = rlo;
            }
        }
        if (hitRidx < 0) continue;                // card fails HIT
        cardOut[out] = {locLid[c], hitRidx, hitRx};
        ++out;
    }
    return {out, nc};
}

} // namespace

// ============================================================================
// P5/P6b/P7 宿主求解（自 GPU 版机械移植；匿名 namespace 防链接冲突）
// ============================================================================

namespace {

int electWinnerArgmax(const int* votes26) {          // 原 kernelSegWinnerV3/K3 语义：平票取小 lid
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        if (votes26[l] > best) { best = votes26[l]; bestLid = l + 1; }
    }
    return bestLid;
}

// 剔除 excludeLid 后的最佳次优；门槛：票数 ≥ max(2, 30%×winnerVotes)
int reelectSecond(const int* votes26, int winnerLid, int winnerVotes) {
    const int threshold = std::max(2, (winnerVotes * 3) / 10);
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        if (l + 1 == winnerLid) continue;
        if (votes26[l] > best) { best = votes26[l]; bestLid = l + 1; }
    }
    return (best >= threshold) ? bestLid : -1;
}

using BitRows = std::vector<unsigned long long>;    // kRowWords 位图

// P5 v2：位置定序重选（同行内线号随 x 单调的结构不变量）
int reelectByPosition(int loserSeg, int keeperSeg, int contestedLid,
                      const std::vector<int>& h_segmap,
                      const cv::Point2f* h_left, int leftCount, float rowStep,
                      const std::vector<BitRows>& segRows,
                      const std::vector<std::vector<int>>& segVotes)
{
    std::vector<unsigned long long> shared(kRowWords, 0ULL);
    for (int w = 0; w < kRowWords; ++w)
        shared[w] = segRows[loserSeg][w] & segRows[keeperSeg][w];
    double sumL = 0, sumK = 0;
    long long nL = 0, nK = 0;
    for (int i = 0; i < leftCount; ++i) {
        const int seg = h_segmap[static_cast<size_t>(i)];
        if (seg != loserSeg && seg != keeperSeg) continue;
        const int row = static_cast<int>(roundf(h_left[static_cast<size_t>(i)].y / rowStep));
        if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
        if (!((shared[static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL)) continue;
        if (seg == loserSeg) { sumL += h_left[static_cast<size_t>(i)].x; ++nL; }
        else { sumK += h_left[static_cast<size_t>(i)].x; ++nK; }
    }
    const int contestedVotes = segVotes[loserSeg][contestedLid - 1];
    const int geoThreshold = std::max(2, (contestedVotes * 15) / 100);
    auto votesOk = [&](int lid) {
        return lid >= 1 && lid <= V3_MAX_LIDS && segVotes[loserSeg][lid - 1] >= geoThreshold;
    };
    if (nL > 0 && nK > 0) {
        const double dx = sumL / nL - sumK / nK;    // 输方相对 keeper 的横向位置
        if (dx > 2.0 && votesOk(contestedLid + 1)) return contestedLid + 1;
        if (dx < -2.0 && votesOk(contestedLid - 1)) return contestedLid - 1;
    }
    return reelectSecond(segVotes[loserSeg].data(), contestedLid, contestedVotes);
}

// P5：单轮冲突消解（cellA/cellB 占用＋v3b 逐行序投票＋位置重选回退）
bool resolveConflictsOnce(int segCount,
                          const std::vector<int>& h_segmap,
                          const cv::Point2f* h_left, int leftCount, float rowStep,
                          const std::vector<std::vector<int>>& segVotes,
                          const std::vector<BitRows>& segRows,
                          std::vector<int>& winner,
                          std::vector<char>& dropped,
                          int* reelectedCnt,
                          int* forcedDropCnt)
{
    std::vector<int> cellA(static_cast<size_t>(MAX_EPIPOLAR_ROWS) * V3_MAX_LIDS, -1);
    std::vector<int> cellB(static_cast<size_t>(MAX_EPIPOLAR_ROWS) * V3_MAX_LIDS, -1);
    for (int s = 0; s < segCount; ++s) {
        if (dropped[s] || winner[s] < 1) continue;
        const int lidIdx = winner[s] - 1;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = segRows[s][w];
            for (int row = w * 64; bits; bits >>= 1, ++row) {
                if (!(bits & 1ULL)) continue;
                const size_t cell = static_cast<size_t>(row) * V3_MAX_LIDS + lidIdx;
                if (cellA[cell] < 0) cellA[cell] = s;
                else if (cellB[cell] < 0 && cellA[cell] != s) cellB[cell] = s;
            }
        }
    }
    for (size_t cell = 0; cell < cellA.size(); ++cell) {
        if (cellB[cell] < 0) continue;
        const int sa = cellA[cell], sb = cellB[cell];
        const int contestedLid = static_cast<int>(cell % V3_MAX_LIDS) + 1;
        const int va = segVotes[sa][contestedLid - 1];
        const int vb = segVotes[sb][contestedLid - 1];
        // —— v3b：逐行序投票（段均值在合并大段上失义；逐行比较两段行内均值，多数票定左右）——
        long long rowsALeft = 0, rowsBLeft = 0;
        {
            std::vector<double> sumA(MAX_EPIPOLAR_ROWS, 0.0), sumB(MAX_EPIPOLAR_ROWS, 0.0);
            std::vector<int> cntA(MAX_EPIPOLAR_ROWS, 0), cntB(MAX_EPIPOLAR_ROWS, 0);
            for (int i = 0; i < leftCount; ++i) {
                const int seg = h_segmap[static_cast<size_t>(i)];
                if (seg != sa && seg != sb) continue;
                const int row = static_cast<int>(roundf(h_left[static_cast<size_t>(i)].y / rowStep));
                if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
                if (!((segRows[sa][static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL)
                    || !((segRows[sb][static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL))
                    continue;
                if (seg == sa) { sumA[row] += h_left[static_cast<size_t>(i)].x; ++cntA[row]; }
                else { sumB[row] += h_left[static_cast<size_t>(i)].x; ++cntB[row]; }
            }
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                if (cntA[r] == 0 || cntB[r] == 0) continue;
                const double d = sumA[r] / cntA[r] - sumB[r] / cntB[r];
                if (d > 2.0) ++rowsBLeft;            // A 行内均值大 → A 在右
                else if (d < -2.0) ++rowsALeft;
            }
        }
        const bool orderable = (rowsALeft + rowsBLeft >= 5)
                               && (rowsALeft > rowsBLeft * 2 || rowsBLeft > rowsALeft * 2);
        if (orderable) {
            const int leftSeg = (rowsALeft > rowsBLeft) ? sa : sb;
            const int rightSeg = (leftSeg == sa) ? sb : sa;
            const int upLid = contestedLid + 1;
            const auto votesOk = [&](int s, int lid) {
                return lid >= 1 && lid <= V3_MAX_LIDS
                       && segVotes[s][lid - 1] >= std::max(2, (segVotes[s][contestedLid - 1] * 15) / 100);
            };
            CALIB_LOG_INFO("P5 conflict: segs {}/{} lid={} votesA={} votesB={} rowVote {}/{} "
                           "left={} vLeftLid={} vRightUp={}",
                           sa, sb, contestedLid, va, vb, rowsALeft, rowsBLeft, leftSeg,
                           segVotes[leftSeg][contestedLid - 1], segVotes[rightSeg][upLid - 1]);
            if (votesOk(rightSeg, upLid) && votesOk(leftSeg, contestedLid)) {
                winner[leftSeg] = contestedLid;
                winner[rightSeg] = upLid;
                *reelectedCnt += 2;
                return true;
            }
        } else {
            CALIB_LOG_INFO("P5 conflict (no order): segs {}/{} lid={} votesA={} votesB={} "
                           "rowVote {}/{}",
                           sa, sb, contestedLid, va, vb, rowsALeft, rowsBLeft);
        }
        // —— 回退 v1/v2 ——
        const int loser = (va >= vb) ? sb : sa;
        const int keeper = (loser == sa) ? sb : sa;
        const int second = reelectByPosition(loser, keeper, contestedLid,
                                             h_segmap, h_left, leftCount, rowStep,
                                             segRows, segVotes);
        if (second >= 1) {
            winner[loser] = second;
            ++*reelectedCnt;
        } else {
            dropped[loser] = 1;
            ++*forcedDropCnt;
        }
        return true;
    }
    return false;
}

// P6b：段分裂（双峰门槛＋卡独占/行序分割＋行块连续三门槛；原地更新全部段结构）
// 卡表为 CPU 常驻 CSR（cardDat_+cardPtr_，ensureCards 等价 no-op）
int splitBimodalSegments(std::vector<int>& h_segmap,
                         const cv::Point2f* h_left, int leftCount, float /*rowStep*/,
                         const LmCard* h_card_dat, const int* h_card_ptr, const int* h_card_count,
                         const int* h_exclCnt,
                         int& segCount,
                         std::vector<std::vector<int>>& segVotes,
                         std::vector<BitRows>& segRows,
                         std::vector<int>& winner,
                         std::vector<char>& dropped,
                         const std::vector<int>& ptRow,
                         std::vector<std::vector<int>>& segPts)
{
    const int segCountIn = segCount;
    int splits = 0;
    std::vector<int> rowCnt(MAX_EPIPOLAR_ROWS + 1, 0);
    std::vector<int> rowOff(MAX_EPIPOLAR_ROWS + 1, 0);
    std::vector<int> flatPts;
    std::vector<char> movedFlag(leftCount, 0);
    for (int s = 0; s < segCountIn; ++s) {
        if (dropped[s] || winner[s] < 1) continue;
        const int W = winner[s];
        const int* v = segVotes[s].data();
        // ① 双峰：次线票 ≥ max(30, 30%×winner 票)
        int S = -1, sVotes = 0;
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            if (l + 1 != W && v[l] > sVotes) { sVotes = v[l]; S = l + 1; }
        if (S < 1 || sVotes < std::max(30, (v[W - 1] * 3) / 10)) continue;
        // ② 点可分：优先卡独占（持 S 不持 W）；不足量 → 行序分割兜底
        std::vector<int> exclPts;
        std::vector<unsigned long long> exclRows(kRowWords, 0ULL);
        int exclSz = -1;
        if (h_exclCnt != nullptr) exclSz = h_exclCnt[s];
        bool needCardScan = (exclSz < 0) || (exclSz >= (sVotes * 8) / 10);
        if (needCardScan) {
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int nc = h_card_count[i];
                const LmCard* cards = h_card_dat + h_card_ptr[i];
                bool hasW = false, hasS = false;
                for (int c = 0; c < nc; ++c) {
                    const int lid = cards[c].lid;
                    if (lid == W) hasW = true;
                    else if (lid == S) hasS = true;
                }
                if (!hasS || hasW) continue;
                exclPts.push_back(i);
                const int row = ptRow[static_cast<size_t>(i)];
                if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                    exclRows[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            }
            if (exclSz < 0) exclSz = static_cast<int>(exclPts.size());   // 旧路径自算
        }
        bool byRowOrder = false;
        if (exclSz < (sVotes * 8) / 10) {
            // —— 行序分割：该段每行按 x 排序，S<W 时左点归 S、右点归 W（S>W 反向）——
            byRowOrder = true;
            exclPts.clear();
            std::fill(exclRows.begin(), exclRows.end(), 0ULL);
            int minR = MAX_EPIPOLAR_ROWS, maxR = -1;
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int row = ptRow[static_cast<size_t>(i)];
                if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
                if (row < minR) minR = row;
                if (row > maxR) maxR = row;
            }
            if (maxR < 0) {   // 无有效行点
                CALIB_LOG_INFO("P6b reject(seg {}): rowOrder qty=0/{} out of band (W={} S={})",
                               s, sVotes, W, S);
                continue;
            }
            for (int r = minR; r <= maxR; ++r) rowCnt[static_cast<size_t>(r)] = 0;
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int row = ptRow[static_cast<size_t>(i)];
                if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
                ++rowCnt[static_cast<size_t>(row)];
            }
            // 量门预检：行序分割的量＝「≥2 点的行数」
            {
                int qtyPre = 0;
                for (int r = minR; r <= maxR; ++r)
                    if (rowCnt[static_cast<size_t>(r)] >= 2) ++qtyPre;
                const int lo2 = sVotes * 6 / 10, hi2 = sVotes * 14 / 10;
                if (qtyPre < lo2 || qtyPre > hi2) {
                    CALIB_LOG_INFO("P6b reject(seg {}): rowOrder qty={}/{} out of band (W={} S={})",
                                   s, qtyPre, sVotes, W, S);
                    continue;
                }
            }
            rowOff[static_cast<size_t>(minR)] = 0;
            for (int r = minR; r < maxR; ++r)
                rowOff[static_cast<size_t>(r) + 1] = rowOff[static_cast<size_t>(r)] + rowCnt[static_cast<size_t>(r)];
            const int total = rowOff[static_cast<size_t>(maxR)] + rowCnt[static_cast<size_t>(maxR)];
            flatPts.assign(static_cast<size_t>(total), -1);
            {
                std::vector<int> fill(rowOff.begin() + minR,
                                      rowOff.begin() + static_cast<long long>(maxR) + 1);
                for (int i : segPts[static_cast<size_t>(s)]) {
                    const int row = ptRow[static_cast<size_t>(i)];
                    if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
                    flatPts[static_cast<size_t>(fill[static_cast<size_t>(row) - minR]++)] = i;
                }
            }
            for (int r = minR; r <= maxR; ++r) {
                const int beg = rowOff[static_cast<size_t>(r)];
                const int cnt = rowCnt[static_cast<size_t>(r)];
                if (cnt < 2) continue;               // 单点行不判（保守归 W）
                int* ps = flatPts.data() + beg;
                std::sort(ps, ps + cnt, [&](int a, int b) {
                    return h_left[static_cast<size_t>(a)].x < h_left[static_cast<size_t>(b)].x;
                });
                const int takeIdx = (S < W) ? ps[0] : ps[cnt - 1];   // S 是左线取最左，反之最右
                exclPts.push_back(takeIdx);
                exclRows[static_cast<size_t>(r) / 64] |= 1ULL << (r % 64);
            }
            // 行序分割的量必须与 sVotes 大体吻合（±40%），否则放弃
            const int lo = sVotes * 6 / 10, hi = sVotes * 14 / 10;
            if (static_cast<int>(exclPts.size()) < lo ||
                static_cast<int>(exclPts.size()) > hi) {
                CALIB_LOG_DEBUG("P6b reject(seg {}): rowOrder qty={}/{} out of band (W={} S={})",
                                s, exclPts.size(), sVotes, W, S);
                continue;
            }
        }
        // ③ 行块连续：行集合排序后最长近连续段（缺口≤25 行）覆盖 ≥70% 且 ≥50 行
        std::vector<int> rows;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = exclRows[w];
            for (int row = w * 64; bits; bits >>= 1, ++row)
                if (bits & 1ULL) rows.push_back(row);
        }
        if (rows.size() < 50) {
            CALIB_LOG_DEBUG("P6b reject(seg {}): gate3a rows={} < 50 (W={} S={})",
                            s, rows.size(), W, S);
            continue;
        }
        int bestRun = 1, run = 1;
        for (size_t k = 1; k < rows.size(); ++k) {
            run = (rows[k] - rows[k - 1] <= 25) ? run + 1 : 1;
            if (run > bestRun) bestRun = run;
        }
        if (bestRun < 50 || bestRun * 10 < static_cast<int>(rows.size()) * 7) {
            CALIB_LOG_DEBUG("P6b reject(seg {}): gate3b bestRun={}/{} (W={} S={})",
                            s, bestRun, rows.size(), W, S);
            continue;
        }
        // —— 执行分裂：独占点移入新段 ——
        const int newSeg = segCount++;
        for (int idx : exclPts) {
            h_segmap[static_cast<size_t>(idx)] = newSeg;
            movedFlag[static_cast<size_t>(idx)] = 1;
        }
        segPts.resize(static_cast<size_t>(segCount));
        segPts[static_cast<size_t>(newSeg)] = exclPts;
        std::vector<int> remain;
        remain.reserve(segPts[static_cast<size_t>(s)].size() - exclPts.size());
        for (int i : segPts[static_cast<size_t>(s)])
            if (!movedFlag[static_cast<size_t>(i)]) remain.push_back(i);
        segPts[static_cast<size_t>(s)] = std::move(remain);
        for (int idx : exclPts) movedFlag[static_cast<size_t>(idx)] = 0;
        // 重建两段的票直方图与行位图（votes 整数累加＋位图 OR——序无关）
        std::vector<int> vA(V3_MAX_LIDS, 0), vB(V3_MAX_LIDS, 0);
        std::vector<unsigned long long> rA(kRowWords, 0ULL), rB(kRowWords, 0ULL);
        for (int i : segPts[static_cast<size_t>(s)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                rA[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            const int nc = h_card_count[i];
            const LmCard* cards = h_card_dat + h_card_ptr[i];
            for (int c = 0; c < nc; ++c) {
                const int lid = cards[c].lid;
                if (lid >= 1 && lid <= V3_MAX_LIDS) ++vA[lid - 1];
            }
        }
        for (int i : segPts[static_cast<size_t>(newSeg)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                rB[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            const int nc = h_card_count[i];
            const LmCard* cards = h_card_dat + h_card_ptr[i];
            for (int c = 0; c < nc; ++c) {
                const int lid = cards[c].lid;
                if (lid >= 1 && lid <= V3_MAX_LIDS) ++vB[lid - 1];
            }
        }
        segVotes[s] = vA;
        segRows[s] = rA;
        segVotes.push_back(vB);
        segRows.push_back(rB);
        winner[s] = electWinnerArgmax(vA.data());
        winner.push_back(S);
        dropped.push_back(0);
        ++splits;
        CALIB_LOG_DEBUG("P6b split(seg {}): winner {} -> keep lid{} + new seg lid{} "
                        "({} pts, mode={}, sVotes={})",
                        s, W, winner[s], S, exclPts.size(),
                        byRowOrder ? "rowOrder" : "cardExcl", sVotes);
    }
    return splits;
}

// ============================================================================
// P7：帧内级联判定（友军合并 → 门槛级联消元 → 撞号/间距安全检查禁赛重跑）
// ============================================================================

struct CascadeCfg {
    int gateH; float gateShare, gateMargin;
    float friendXpx; int friendMinRows;
    float spacingPx, spacingTol;
};

struct CascadeUnit {
    std::vector<int> members;
    std::vector<long long> origVotes;
    std::vector<long long> votes;
    std::vector<unsigned long long> rows;              // kRowWords 位图
    std::vector<int> rowIdx;                           // 升序在场行（稀疏）
    std::vector<double> rowXSum;                       // 对齐 rowIdx（稀疏）
    std::vector<int> rowXCnt;
    bool determined = false, banned = false;
    int detLid = -1;
};

int cascadeSolve(int segCount,
                 const std::vector<int>& h_segmap,
                 const cv::Point2f* h_left, int leftCount, float rowStep,
                 const std::vector<int>& ptRow,
                 const std::vector<std::vector<int>>& segPts,
                 const std::vector<std::vector<int>>& segVotes,
                 const std::vector<BitRows>& segRows,
                 std::vector<int>& winner,
                 std::vector<char>& dropped,
                 const CascadeCfg& cfg,
                 int* outBanned, int* outMerged)
{
    // —— 每段稀疏行数据（升序行＋xSum/xCnt；行值来自 ptRow 无 clamp）——
    // 构建序：段内点升序 i 累加 → 与 GPU 宿主路径同一加法序列
    thread_local std::vector<std::vector<int>> segRowIdxP;
    thread_local std::vector<std::vector<double>> segXSumP;
    thread_local std::vector<std::vector<int>> segXCntP;
    thread_local std::vector<double> stampSumP;
    thread_local std::vector<int> stampCntP;
    auto& segRowIdx = segRowIdxP;
    auto& segXSum = segXSumP;
    auto& segXCnt = segXCntP;
    auto& stampSum = stampSumP;
    auto& stampCnt = stampCntP;
    segRowIdx.resize(static_cast<size_t>(segCount));
    segXSum.resize(static_cast<size_t>(segCount));
    segXCnt.resize(static_cast<size_t>(segCount));
    for (int s = 0; s < segCount; ++s) {
        segRowIdx[static_cast<size_t>(s)].clear();
        segXSum[static_cast<size_t>(s)].clear();
        segXCnt[static_cast<size_t>(s)].clear();
    }
    stampSum.assign(MAX_EPIPOLAR_ROWS, 0.0);
    stampCnt.assign(MAX_EPIPOLAR_ROWS, 0);
    for (int s = 0; s < segCount; ++s) {
        for (int i : segPts[static_cast<size_t>(s)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
            stampSum[static_cast<size_t>(row)] += h_left[static_cast<size_t>(i)].x;
            ++stampCnt[static_cast<size_t>(row)];
        }
        const auto& bits_ = segRows[static_cast<size_t>(s)];
        size_t n = 0;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long b = bits_[static_cast<size_t>(w)];
            while (b) { n += (b & 1ULL); b >>= 1ULL; }
        }
        if (n == 0) continue;
        segRowIdx[static_cast<size_t>(s)].resize(n);
        segXSum[static_cast<size_t>(s)].resize(n);
        segXCnt[static_cast<size_t>(s)].resize(n);
        size_t k = 0;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long b = bits_[static_cast<size_t>(w)];
            for (int row = w * 64; b; b >>= 1ULL, ++row) {
                if (!(b & 1ULL)) continue;
                segRowIdx[static_cast<size_t>(s)][k] = row;
                segXSum[static_cast<size_t>(s)][k] = stampSum[static_cast<size_t>(row)];
                segXCnt[static_cast<size_t>(s)][k] = stampCnt[static_cast<size_t>(row)];
                stampSum[static_cast<size_t>(row)] = 0.0;
                stampCnt[static_cast<size_t>(row)] = 0;
                ++k;
            }
        }
    }
    auto hasVotes = [&](int s) {
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            if (segVotes[static_cast<size_t>(s)][static_cast<size_t>(l)] > 0) return true;
        return false;
    };
    // —— 友军合并（union-find：共享行≥minRows 且均距≤friendXpx，有票段）——
    std::vector<int> parent(static_cast<size_t>(segCount));
    for (int s = 0; s < segCount; ++s) parent[static_cast<size_t>(s)] = s;
    auto find = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    long long mergePairs = 0;
    if (segCount > 0 && segCount <= 512) {
        // —— 行主序稠密对累加（S≤512；对 (a,b) 的 sum 累加序列＝行升序，与 GPU 宿主
        //    路径同一加法序列→逐位一致；实测达标对（n≥3）在长段数据上近乎全量，
        //    两阶段先数后算无收益，保持一阶段）——
        const int S = segCount;
        thread_local std::vector<double> psumP;
        thread_local std::vector<long long> pnP;
        thread_local std::vector<int> rowSegCntP, rowSegOffP, rowSegDataP, curP;
        thread_local std::vector<double> rowSegXbP;
        auto& psum = psumP;
        auto& pn = pnP;
        auto& rowSegCnt = rowSegCntP;
        auto& rowSegOff = rowSegOffP;
        auto& rowSegData = rowSegDataP;
        auto& rowSegXb = rowSegXbP;
        auto& cur = curP;
        psum.assign(static_cast<size_t>(S) * S, 0.0);
        pn.assign(static_cast<size_t>(S) * S, 0);
        rowSegCnt.assign(static_cast<size_t>(MAX_EPIPOLAR_ROWS) + 1, 0);
        for (int s = 0; s < S; ++s) {
            if (!hasVotes(s)) continue;
            for (int r : segRowIdx[static_cast<size_t>(s)])
                ++rowSegCnt[static_cast<size_t>(r)];
        }
        rowSegOff.assign(static_cast<size_t>(MAX_EPIPOLAR_ROWS) + 1, 0);
        for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r)
            rowSegOff[static_cast<size_t>(r) + 1] = rowSegOff[static_cast<size_t>(r)]
                                                    + rowSegCnt[static_cast<size_t>(r)];
        const int totalEntries = rowSegOff[static_cast<size_t>(MAX_EPIPOLAR_ROWS)];
        rowSegData.resize(static_cast<size_t>(totalEntries));   // 每槽全量覆写，免清
        rowSegXb.resize(static_cast<size_t>(totalEntries));
        cur.assign(rowSegOff.begin(), rowSegOff.end() - 1);     // 游标副本
        for (int s = 0; s < S; ++s) {
            if (!hasVotes(s)) continue;
            const auto& ridx = segRowIdx[static_cast<size_t>(s)];
            for (size_t k = 0; k < ridx.size(); ++k) {
                const size_t slot = static_cast<size_t>(cur[static_cast<size_t>(ridx[k])]++);
                rowSegData[slot] = s;
                rowSegXb[slot] = segXSum[static_cast<size_t>(s)][k]
                                 / segXCnt[static_cast<size_t>(s)][k];   // 预算 xbar（同操作数）
            }
        }
        for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
            const int beg = rowSegOff[static_cast<size_t>(r)];
            const int cnt = rowSegCnt[static_cast<size_t>(r)];
            for (int ia_ = beg; ia_ < beg + cnt; ++ia_) {
                const int a = rowSegData[static_cast<size_t>(ia_)];
                const double xa = rowSegXb[static_cast<size_t>(ia_)];
                for (int ib_ = ia_ + 1; ib_ < beg + cnt; ++ib_) {
                    const int b = rowSegData[static_cast<size_t>(ib_)];
                    const size_t pk = static_cast<size_t>(a) * S + b;
                    psum[pk] += std::fabs(xa - rowSegXb[static_cast<size_t>(ib_)]);
                    ++pn[pk];
                }
            }
        }
        for (int a = 0; a < S; ++a) {
            for (int b = a + 1; b < S; ++b) {
                const size_t pk = static_cast<size_t>(a) * S + b;
                if (pn[pk] >= cfg.friendMinRows && psum[pk] / pn[pk] <= cfg.friendXpx) {
                    parent[static_cast<size_t>(find(b))] = find(a); ++mergePairs;
                }
            }
        }
    } else {
        // 对主归并游走（大 S 回退路径；对 (a,b) 字典序＋共享行升序＝同一加法序列）
        for (int a = 0; a < segCount; ++a) {
            if (!hasVotes(a)) continue;
            for (int b = a + 1; b < segCount; ++b) {
                if (!hasVotes(b)) continue;
                const auto& ra = segRowIdx[static_cast<size_t>(a)];
                const auto& rb = segRowIdx[static_cast<size_t>(b)];
                const auto& sa = segXSum[static_cast<size_t>(a)];
                const auto& sb = segXSum[static_cast<size_t>(b)];
                const auto& ca = segXCnt[static_cast<size_t>(a)];
                const auto& cb = segXCnt[static_cast<size_t>(b)];
                double sum = 0; long long n = 0;
                size_t pa = 0, pb = 0;
                while (pa < ra.size() && pb < rb.size()) {
                    const int r1 = ra[pa], r2 = rb[pb];
                    if (r1 < r2) ++pa;
                    else if (r2 < r1) ++pb;
                    else {
                        sum += std::fabs(sa[pa] / ca[pa] - sb[pb] / cb[pb]);
                        ++n; ++pa; ++pb;
                    }
                }
                if (n >= cfg.friendMinRows && sum / n <= cfg.friendXpx) {
                    parent[static_cast<size_t>(find(b))] = find(a); ++mergePairs;
                }
            }
        }
    }
    *outMerged = static_cast<int>(mergePairs);
    // —— 单元组装（成员升序聚合；行数据稀疏化）——
    thread_local std::vector<CascadeUnit> unitsP;
    thread_local std::vector<int> rootToUnitP;
    auto& units = unitsP;
    auto& rootToUnit = rootToUnitP;
    units.reserve(static_cast<size_t>(segCount) + 8);   // 引用不失效保障
    rootToUnit.assign(static_cast<size_t>(segCount), -1);
    size_t uCount = 0;
    for (int s = 0; s < segCount; ++s) {
        if (!hasVotes(s)) continue;
        const int root = find(s);
        int ui = rootToUnit[static_cast<size_t>(root)];
        if (ui < 0) {
            ui = static_cast<int>(uCount);
            rootToUnit[static_cast<size_t>(root)] = ui;
            if (static_cast<size_t>(uCount) == units.size()) units.emplace_back();
            CascadeUnit& nu = units[uCount];
            nu.members.clear();
            nu.origVotes.assign(V3_MAX_LIDS, 0);
            nu.votes.assign(V3_MAX_LIDS, 0);
            nu.rows.assign(kRowWords, 0ULL);
            nu.determined = false; nu.banned = false; nu.detLid = -1;
            ++uCount;
        }
        CascadeUnit& u = units[static_cast<size_t>(ui)];
        u.members.push_back(s);
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            u.origVotes[static_cast<size_t>(l)] += segVotes[static_cast<size_t>(s)][static_cast<size_t>(l)];
        for (int w = 0; w < kRowWords; ++w)
            u.rows[static_cast<size_t>(w)] |= segRows[static_cast<size_t>(s)][static_cast<size_t>(w)];
    }
    for (size_t uu = 0; uu < uCount; ++uu) {
        CascadeUnit& u = units[uu];
        u.rowIdx.clear();
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = u.rows[static_cast<size_t>(w)];
            for (int row = w * 64; bits; bits >>= 1, ++row)
                if (bits & 1ULL) u.rowIdx.push_back(row);
        }
        const size_t n = u.rowIdx.size();
        u.rowXSum.assign(n, 0.0);
        u.rowXCnt.assign(n, 0);
        if (u.members.size() == 1) {
            const int m = u.members.front();
            u.rowXSum = segXSum[static_cast<size_t>(m)];
            u.rowXCnt = segXCnt[static_cast<size_t>(m)];
            continue;
        }
        for (int m : u.members) {
            const auto& ridx = segRowIdx[static_cast<size_t>(m)];
            const auto& xs = segXSum[static_cast<size_t>(m)];
            const auto& xc = segXCnt[static_cast<size_t>(m)];
            for (size_t k = 0; k < ridx.size(); ++k) {
                const size_t slot = static_cast<size_t>(
                    std::lower_bound(u.rowIdx.begin(), u.rowIdx.end(), ridx[k])
                    - u.rowIdx.begin());
                u.rowXSum[slot] += xs[k];   // 行槽加法序＝成员升序
                u.rowXCnt[slot] += xc[k];
            }
        }
    }
    auto rowsOverlap = [](const std::vector<unsigned long long>& A,
                          const std::vector<unsigned long long>& B) {
        for (size_t w = 0; w < A.size(); ++w) if (A[w] & B[w]) return true;
        return false;
    };
    long long bannedCnt = 0;
    // —— 安全迭代级联 ——
    thread_local std::vector<int> detP, violP;
    auto& det = detP;
    auto& viol = violP;
    for (int iter = 0; iter < 10; ++iter) {
        for (size_t uu = 0; uu < uCount; ++uu) {
            CascadeUnit& u = units[uu];
            u.votes = u.origVotes;
            u.determined = false; u.detLid = -1;
        }
        for (int round = 0; ; ++round) {
            int best = -1; double bestShare = -1, bestMg = 0;
            for (size_t ui = 0; ui < uCount; ++ui) {
                CascadeUnit& u = units[ui];
                if (u.determined || u.banned) continue;
                long long w = 0, tot = 0, second = 0; int wl = -1;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    tot += u.votes[static_cast<size_t>(l)];
                    if (u.votes[static_cast<size_t>(l)] > w) {
                        second = w; w = u.votes[static_cast<size_t>(l)]; wl = l + 1;
                    } else if (u.votes[static_cast<size_t>(l)] > second)
                        second = u.votes[static_cast<size_t>(l)];
                }
                if (w < cfg.gateH) continue;
                const double sh = tot ? static_cast<double>(w) / tot : 0;
                const double mg = second ? static_cast<double>(w) / second : 99.0;
                if (sh < cfg.gateShare || mg < cfg.gateMargin) continue;
                if (sh > bestShare || (sh == bestShare && mg > bestMg)) {
                    bestShare = sh; bestMg = mg; best = static_cast<int>(ui);
                }
            }
            if (best < 0) break;
            CascadeUnit& bu = units[static_cast<size_t>(best)];
            long long w = 0; int wl = -1;
            for (int l = 0; l < V3_MAX_LIDS; ++l)
                if (bu.votes[static_cast<size_t>(l)] > w) { w = bu.votes[static_cast<size_t>(l)]; wl = l + 1; }
            bu.determined = true; bu.detLid = wl;
            for (size_t uu = 0; uu < uCount; ++uu) {
                CascadeUnit& u = units[uu];
                if (u.determined || u.banned || &u == &bu) continue;
                if (rowsOverlap(bu.rows, u.rows) && u.votes[static_cast<size_t>(wl - 1)] > 0)
                    u.votes[static_cast<size_t>(wl - 1)] = 0;
            }
        }
        // ===== 事后安全检查（撞号 / 相邻间距）——行稀疏归并游走 =====
        det.clear();
        for (size_t ui = 0; ui < uCount; ++ui)
            if (units[ui].determined && !units[ui].banned) det.push_back(static_cast<int>(ui));
        viol.clear();
        for (size_t i = 0; i < det.size(); ++i) for (size_t j = i + 1; j < det.size(); ++j) {
            CascadeUnit& A = units[static_cast<size_t>(det[i])];
            CascadeUnit& B = units[static_cast<size_t>(det[j])];
            const int lidDiff = std::abs(A.detLid - B.detLid);
            if (lidDiff > 1) continue;                  // 非撞号非相邻：免归并
            if (!rowsOverlap(A.rows, B.rows)) continue;
            // 共享行升序归并（同一加法序列）
            const auto& ra = A.rowIdx;
            const auto& rb = B.rowIdx;
            double sum = 0; long long n = 0;
            size_t pa = 0, pb = 0;
            while (pa < ra.size() && pb < rb.size()) {
                const int r1 = ra[pa], r2 = rb[pb];
                if (r1 < r2) ++pa;
                else if (r2 < r1) ++pb;
                else {
                    sum += std::fabs(A.rowXSum[pa] / A.rowXCnt[pa]
                                     - B.rowXSum[pb] / B.rowXCnt[pb]);
                    ++n; ++pa; ++pb;
                }
            }
            const double mdx = n ? sum / n : 0;
            bool bad = false;
            if (A.detLid == B.detLid) bad = true;                       // 撞号
            else if (lidDiff == 1                                      // 相邻间距
                     && std::fabs(mdx - cfg.spacingPx) > cfg.spacingTol * cfg.spacingPx)
                bad = true;
            if (bad) { viol.push_back(det[i]); viol.push_back(det[j]); }
        }
        if (viol.empty()) break;
        std::sort(viol.begin(), viol.end());
        viol.erase(std::unique(viol.begin(), viol.end()), viol.end());
        for (int v : viol) { units[static_cast<size_t>(v)].banned = true; ++bannedCnt; }
    }
    *outBanned = static_cast<int>(bannedCnt);
    // —— 写回 ——
    int determinedCnt = 0;
    for (int s = 0; s < segCount; ++s) {
        winner[static_cast<size_t>(s)] = -1;
        dropped[static_cast<size_t>(s)] = 0;
        if (!hasVotes(s)) continue;
        const int root = find(s);
        const int uiW = rootToUnit[static_cast<size_t>(root)];
        if (uiW < 0) continue;
        CascadeUnit& u = units[static_cast<size_t>(uiW)];
        if (u.banned) { dropped[static_cast<size_t>(s)] = 1; continue; }
        if (u.determined) {
            winner[static_cast<size_t>(s)] = u.detLid;
            if (std::find(u.members.begin(), u.members.end(), s) == u.members.begin())
                ++determinedCnt;                       // 每单元计一次
        }
    }
    return determinedCnt;
}

} // namespace

// ============================================================================
// Impl
// ============================================================================

struct LaserMatchScanCpuV3::Impl {
    LaserMatchScanParamsV3 params_;

    struct TempTableEntry {
        double temperature = 0.0;
        std::vector<float> mapData;
        int entryCount = 0;
        int numLines = 0;
    };
    std::vector<TempTableEntry> tempTable_;
    bool tableLoaded_ = false;

    // 温表缓存判定（同源指针＋同温＋同 rowStep）
    std::shared_ptr<const LaserPlaneMapTempTable> srcHold_;
    const void* uploadedSrc_ = nullptr;
    double uploadedTemp_ = 0.0;
    float uploadedRowStep_ = -1.0f;
    double currentTemp_ = 25.0;
    bool tempSet_ = false;

    // —— 活动温表按行 CSR ＋ 栅格直达参数 ——
    // MapEnt（匿名 namespace 12B AoS）：锚点跳走/展开扫卡/uR 取值全走同一条缓存行流
    int activeMapCount_ = 0;
    int activeNumLines_ = 0;
    std::vector<MapEnt> mapEnt_;       // 行内按 xL 升序
    std::vector<int> mapRowStart_;     // MAX_EPIPOLAR_ROWS
    std::vector<int> mapRowCount_;
    std::vector<float> mapRowX0_;      // 每行首条目 x（栅格原点；空行 = 0）
    float mapInvStep_ = 0.0f;          // 1/epipolar_row_step（栅格直达猜测）
    std::vector<int> bucketPool_;      // 行内桶下界（相对行首；K=1024）
    std::vector<int> bucketRowOff_;    // 每行池偏移
    float bucketMinX_ = 0.0f, bucketInvW_ = 0.0f;
    int bucketK_ = 0;

    // —— uR 补偿（互斥：后设者生效）——
    std::array<float, V3_MAX_LIDS> urBias_{};
    bool urBiasActive_ = false;
    std::array<float, 4> biasParams_{};   // wx, wy, wz, dtx
    bool urBiasModelActive_ = false;

    // —— 右点行 CSR（量化键稳定序）——
    std::vector<float> rxSorted_;      // 行内按 (xq, origIdx) 稳定序
    std::vector<int> rRowStart_;       // MAX_EPIPOLAR_ROWS
    std::vector<int> rRowCount_;
    std::vector<int> rScatter_;        // 散射游标
    std::vector<int> rxXq_;            // 行排序期 xq 平行数组（复用缓冲）

    // —— 发卡产物（CSR 紧凑布局：容量式 grow-only）——
    // stride-32 四数组（17.6MB）使整帧工作集 ~28MB 超 L3（12MB）→ 稳态 DRAM 颠簸；
    // CSR 单数组 {lid,ridx,hitRx}（~1.9MB）＋逐点偏移。cardRes 无消费者已删。
    std::vector<LmCard> cardDat_;       // 全帧卡紧凑流（append-only）
    std::vector<int> cardPtr_;          // [leftCount+1] 每点卡起始偏移（CSR）
    std::vector<int> cardCount_;        // 每点有效卡数
    std::vector<int> leftStatus_;

    // —— 段结构 ——
    std::vector<int> segmap_;
    std::vector<int> uniqTracks_;
    std::vector<std::vector<int>> segVotes_;
    std::vector<BitRows> segRows_;
    std::vector<int> winner_;
    std::vector<int> second_;
    std::vector<char> dropped_;
    std::vector<int> ptRow_;
    std::vector<std::vector<int>> segPts_;
    std::vector<int> exclCnt_;

    // —— 冲突占胞（默认路径）——
    std::vector<int> cellOwner_;

    // —— 输出缓冲（容量式）——
    std::vector<cv::Point2f> outLeft_, outRight_;
    std::vector<int> outLid_, outTrack_;
    std::vector<int> rightStatus_;
    cv::Mat outLeftM_, outRightM_, outLidM_, outTrackM_;

    bool warmed_up_ = false;
    int warmup_left_ = 0, warmup_right_ = 0;

    explicit Impl(const LaserMatchScanParamsV3& params);
    ~Impl() = default;

    bool LoadTempTable(const std::string& jsonPath);
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void SetCurrentTemperature(double temperature);
    void buildActiveMap(const TempTableEntry& entry);
    void setParams(const LaserMatchScanParamsV3& params);
    void setuRBias(const std::vector<float>& biasPerLid);
    void setuRBiasModel(float wx, float wy, float wz, float dtx);
    const LaserMatchScanParamsV3& getParams() const { return params_; }
    void warmup(int maxLeftPoints, int maxRightPoints);
    void allocateBuffers(int leftCount, int rightCount);

    LaserMatchScanCpuResultV3 process(const cv::Mat& left_points,
                                      const cv::Mat& left_line_ids,
                                      const cv::Mat& right_points,
                                      const cv::Mat& right_line_ids);
};

// ============================================================================
// ctor / 温表
// ============================================================================

LaserMatchScanCpuV3::Impl::Impl(const LaserMatchScanParamsV3& params)
    : params_(params)
{
    params_.validate();
}

bool LaserMatchScanCpuV3::Impl::LoadTempTable(const std::string& jsonPath) {
    try {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            CALIB_LOG_ERROR("loadTempTable(): cannot open file: {}", jsonPath);
            return false;
        }

        nlohmann::json j;
        ifs >> j;

        auto table = std::make_shared<LaserPlaneMapTempTable>();
        const auto& arr = j.at("table");
        for (const auto& entry : arr) {
            const double temp = entry.at("temperature").get<double>();
            const auto& mapData = entry.at("mapData");
            const int n = static_cast<int>(mapData.size());

            LaserPlaneMap planeMap;
            planeMap.temperature = temp;
            planeMap.totalPairs = n;
            cv::Mat m(n, 4, CV_32FC1);
            for (int i = 0; i < n; ++i) {
                const auto& row = mapData[i];
                m.at<float>(i, 0) = row[0].get<float>();
                m.at<float>(i, 1) = row[1].get<float>();
                m.at<float>(i, 2) = row[2].get<float>();
                m.at<float>(i, 3) = row[3].get<float>();
            }
            planeMap.leftToRightMap = std::move(m);
            table->table[temp] = std::move(planeMap);
        }

        if (!SetTempTable(table)) {
            return false;
        }
        CALIB_LOG_INFO("loadTempTable(): loaded {} temperature entries from {}",
                       tempTable_.size(), jsonPath);
        return true;

    } catch (const std::exception& e) {
        CALIB_LOG_ERROR("loadTempTable(): exception: {}", e.what());
        return false;
    }
}

bool LaserMatchScanCpuV3::Impl::SetTempTable(
    std::shared_ptr<const LaserPlaneMapTempTable> table)
{
    if (!table || table->empty()) {
        CALIB_LOG_ERROR("SetTempTable(): null or empty temp table");
        return false;
    }

    // 同源重复注入跳过重转换（const 共享只读契约）
    if (tableLoaded_ && srcHold_ && srcHold_.get() == table.get()) {
        return true;
    }
    srcHold_ = table;
    uploadedSrc_ = nullptr;   // 新表：活动表缓存失效

    tempTable_.clear();

    for (const auto& kv : table->table) {
        const double temp = kv.first;
        const LaserPlaneMap& planeMap = kv.second;
        const cv::Mat& m = planeMap.leftToRightMap;
        if (m.empty()) {
            CALIB_LOG_WARN("SetTempTable(): skip temp={} (empty leftToRightMap)", temp);
            continue;
        }

        TempTableEntry te;
        te.temperature = temp;
        te.entryCount = m.rows;
        te.mapData.resize(static_cast<size_t>(te.entryCount) * 4);

        int maxLid = -1;
        for (int i = 0; i < te.entryCount; ++i) {
            const float* r = m.ptr<float>(i);
            te.mapData[static_cast<size_t>(i) * 4 + 0] = r[0];
            te.mapData[static_cast<size_t>(i) * 4 + 1] = r[1];
            te.mapData[static_cast<size_t>(i) * 4 + 2] = r[2];
            te.mapData[static_cast<size_t>(i) * 4 + 3] = r[3];
            const int lid = static_cast<int>(r[3]);
            if (lid > maxLid) maxLid = lid;
        }
        te.numLines = maxLid + 1;
        tempTable_.push_back(std::move(te));
    }

    if (tempTable_.empty()) {
        CALIB_LOG_ERROR("SetTempTable(): no valid entries after conversion");
        tableLoaded_ = false;
        return false;
    }

    std::sort(tempTable_.begin(), tempTable_.end(),
              [](const TempTableEntry& a, const TempTableEntry& b) {
                  return a.temperature < b.temperature;
              });

    tableLoaded_ = true;
    return true;
}

void LaserMatchScanCpuV3::Impl::SetCurrentTemperature(double temperature) {
    if (!tableLoaded_ || tempTable_.empty()) {
        throw std::runtime_error("[07-LaserMatchScanCpuV3] temp table not loaded");
    }

    if (temperature < tempTable_.front().temperature ||
        temperature > tempTable_.back().temperature) {
        CALIB_LOG_WARN("setCurrentTemperature(): temp={} out of range [{}, {}], clamping",
                       temperature,
                       tempTable_.front().temperature,
                       tempTable_.back().temperature);
    }

    auto it = std::lower_bound(tempTable_.begin(), tempTable_.end(), temperature,
        [](const TempTableEntry& e, double val) { return e.temperature < val; });

    if (it == tempTable_.end()) --it;
    if (it != tempTable_.begin()) {
        auto prev = std::prev(it);
        if (std::abs(prev->temperature - temperature) <
            std::abs(it->temperature - temperature)) {
            it = prev;
        }
    }

    currentTemp_ = it->temperature;
    tempSet_ = true;

    // 缓存命中（同源表＋同档温＋同 rowStep）跳过重排
    const bool cacheHit = (uploadedSrc_ != nullptr
                           && srcHold_ && srcHold_.get() == uploadedSrc_
                           && it->temperature == uploadedTemp_
                           && params_.epipolar_row_step == uploadedRowStep_);
    if (!cacheHit) {
        buildActiveMap(*it);
        uploadedSrc_ = srcHold_.get();
        uploadedTemp_ = it->temperature;
        uploadedRowStep_ = params_.epipolar_row_step;
    }
}

// 按行 CSR＋桶索引池（移植 uploadMapTable 宿主段；行内排序同比较器）
void LaserMatchScanCpuV3::Impl::buildActiveMap(const TempTableEntry& entry) {
    activeMapCount_ = entry.entryCount;
    activeNumLines_ = entry.numLines;
    const int N = activeMapCount_;
    const float rowStep = params_.epipolar_row_step;
    const float* src = entry.mapData.data();

    // 行分桶（输入序）＋行内按 xL 排序（与 GPU 宿主路径同比较器同序）
    std::vector<std::vector<int>> buckets(MAX_EPIPOLAR_ROWS);
    for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) buckets[r].reserve(64);
    for (int i = 0; i < N; ++i) {
        float yL = src[static_cast<size_t>(i) * 4 + 1];
        int row = (int)std::round(yL / rowStep);
        row = clampRow(row);
        buckets[row].push_back(i);
    }
    mapEnt_.resize(static_cast<size_t>(N));
    mapRowStart_.assign(MAX_EPIPOLAR_ROWS, 0);
    mapRowCount_.assign(MAX_EPIPOLAR_ROWS, 0);
    mapRowX0_.assign(MAX_EPIPOLAR_ROWS, 0.0f);
    int offset = 0;
    for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
        mapRowStart_[r] = offset;
        mapRowCount_[r] = static_cast<int>(buckets[r].size());
        if (!buckets[r].empty())
            mapRowX0_[r] = src[static_cast<size_t>(buckets[r].front()) * 4];   // 行首 x＝栅格原点
        std::sort(buckets[r].begin(), buckets[r].end(), [&](int a, int b) {
            return src[static_cast<size_t>(a) * 4] < src[static_cast<size_t>(b) * 4];
        });
        for (int idx : buckets[r]) {
            mapEnt_[offset].x = src[static_cast<size_t>(idx) * 4];
            mapEnt_[offset].uR = src[static_cast<size_t>(idx) * 4 + 2];
            mapEnt_[offset].lid = static_cast<int>(src[static_cast<size_t>(idx) * 4 + 3]);
            ++offset;
        }
    }
    mapInvStep_ = 1.0f / rowStep;

    // 桶索引池（K=1024：行池段 4KB 驻 L1、桶内跳走 ~3 步；锚点＝min(lower_bound,
    // 行末) 与桶数无关，对栅格多重性不均（每格 2-3 条目）稳健）
    bucketK_ = 0;
    bucketPool_.clear();
    bucketRowOff_.assign(MAX_EPIPOLAR_ROWS, 0);
    {
        float mnX = std::numeric_limits<float>::max(), mxX = -std::numeric_limits<float>::max();
        for (int i = 0; i < N; ++i) {
            const float x = src[static_cast<size_t>(i) * 4];
            if (x < mnX) mnX = x;
            if (x > mxX) mxX = x;
        }
        const float w = (mxX - mnX) / kCpuBucketK;
        if (w > 1e-6f) {
            bucketMinX_ = mnX;
            bucketInvW_ = 1.f / w;
            bucketK_ = kCpuBucketK;
            bucketPool_.reserve(1024 * (kCpuBucketK + 1));
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                if (mapRowCount_[r] == 0) continue;
                bucketRowOff_[r] = static_cast<int>(bucketPool_.size());
                const int s = mapRowStart_[r];
                const int cnt = mapRowCount_[r];
                int p = 0;
                for (int b = 0; b <= kCpuBucketK; ++b) {
                    const float xb = mnX + b * w;
                    while (p < cnt && mapEnt_[static_cast<size_t>(s + p)].x < xb)
                        ++p;
                    bucketPool_.push_back(p);
                }
            }
            if (bucketPool_.empty()) {
                bucketK_ = 0;   // 空表防御
            }
        }
    }
}

void LaserMatchScanCpuV3::Impl::setParams(const LaserMatchScanParamsV3& params) {
    params.validate();
    if (params.epipolar_row_step != params_.epipolar_row_step)
        uploadedRowStep_ = -1.0f;   // rowStep 影响分桶 → 活动表缓存失效
    params_ = params;
    warmed_up_ = false;
    warmup_left_ = 0;
    warmup_right_ = 0;
    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}

void LaserMatchScanCpuV3::Impl::setuRBias(const std::vector<float>& biasPerLid) {
    urBias_.fill(0.0f);
    const size_t n = std::min(biasPerLid.size(), static_cast<size_t>(V3_MAX_LIDS));
    for (size_t i = 0; i < n; ++i) urBias_[i] = biasPerLid[i];
    urBiasActive_ = !biasPerLid.empty();
    if (urBiasActive_) urBiasModelActive_ = false;   // 互斥：逐线表顶替 4 参数模型
    CALIB_LOG_INFO("setuRBias(): {} lids, active={}", n, urBiasActive_);
}

void LaserMatchScanCpuV3::Impl::setuRBiasModel(float wx, float wy, float wz, float dtx) {
    biasParams_ = {wx, wy, wz, dtx};
    urBiasModelActive_ = true;
    urBiasActive_ = false;                           // 互斥：模型顶替逐线表
    CALIB_LOG_INFO("setuRBiasModel(): wx={} wy={} wz={} dtx={} (model active, per-lid table off)",
                   wx, wy, wz, dtx);
}

void LaserMatchScanCpuV3::Impl::warmup(int maxLeftPoints, int maxRightPoints) {
    if (maxLeftPoints <= 0 || maxRightPoints <= 0) {
        CALIB_LOG_WARN("warmup(): invalid sizes (left={}, right={}), skipping",
                       maxLeftPoints, maxRightPoints);
        return;
    }
    allocateBuffers(maxLeftPoints, maxRightPoints);
    warmed_up_ = true;
    warmup_left_ = maxLeftPoints;
    warmup_right_ = maxRightPoints;
    CALIB_LOG_INFO("warmup(): allocated buffers for {} left, {} right points",
                   maxLeftPoints, maxRightPoints);
}

void LaserMatchScanCpuV3::Impl::allocateBuffers(int leftCount, int rightCount) {
    if (leftCount > static_cast<int>(cardCount_.size())) {
        cardPtr_.assign(static_cast<size_t>(leftCount) + 1, 0);
        cardCount_.assign(static_cast<size_t>(leftCount), 0);
        leftStatus_.assign(static_cast<size_t>(leftCount), 0);
        cardDat_.clear();
        cardDat_.reserve(static_cast<size_t>(leftCount) * 8);   // 均值 ~4.6 卡/点＋余量
    }
    if (rightCount > static_cast<int>(rScatter_.size())) {
        rScatter_.assign(static_cast<size_t>(rightCount), 0);
        rxSorted_.assign(static_cast<size_t>(rightCount), 0.0f);
        rxXq_.assign(static_cast<size_t>(rightCount), 0);
        rightStatus_.assign(static_cast<size_t>(rightCount), 0);
    }
}

// ============================================================================
// process — 全流程（单线程确定性）
// ============================================================================

LaserMatchScanCpuResultV3 LaserMatchScanCpuV3::Impl::process(
    const cv::Mat& left_points,
    const cv::Mat& left_line_ids,
    const cv::Mat& right_points,
    const cv::Mat& /*right_line_ids*/)
{
    LaserMatchScanCpuResultV3 result;

    // 临时分段计时（JMW_CPUPROF=1 输出；后续优化完成后移除）
    static const bool kProf_ = (std::getenv("JMW_CPUPROF") != nullptr);
    const auto tT0_ = std::chrono::steady_clock::now();
    auto tMark_ = tT0_;
    auto tSince_ = [&tMark_]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tMark_).count();
        tMark_ = now;
        return ms;
    };
    std::string tBd_;

    try {
        if (!tempSet_) {
            result.success = false;
            result.message = "Temperature not set: call setCurrentTemperature() first";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        const int leftCount = left_points.rows * left_points.cols;
        const int rightCount = right_points.rows * right_points.cols;
        result.totalLeftPoints = leftCount;
        result.totalRightPoints = rightCount;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "No points to match";
            result.matchedCount = 0;
            return result;
        }

        allocateBuffers(leftCount, rightCount);

        const cv::Point2f* lp = left_points.ptr<cv::Point2f>();
        const cv::Point2f* rp = right_points.ptr<cv::Point2f>();
        const int* lt = left_line_ids.ptr<int>();
        const float rowStep = params_.epipolar_row_step;
        const float vLtol = params_.vL_tolerance;
        const float matchWin = params_.match_threshold;

    // —— 右点行桶＋行内量化键稳定序（复刻 GPU 键排序语义）——
    // 行计数排序（8192 桶，稳定保序＝散射序）＋行内 xq 插入排序（稳定，n≈22 免堆
    // 分配）＝ CUB LSD 基数排序的 (row,xq,origIdx) 稳定全序
    {
        rRowStart_.assign(MAX_EPIPOLAR_ROWS, 0);
        rRowCount_.assign(MAX_EPIPOLAR_ROWS, 0);
        rxXq_.resize(static_cast<size_t>(rightCount));
        for (int i = 0; i < rightCount; ++i) {
            int rowKey = froundAwayI(rp[i].y / rowStep);
            if (rowKey < 0) rowKey = 0;                      // 钳位（同 kernelGenKeysIdxV3）
            if (rowKey < MAX_EPIPOLAR_ROWS) ++rRowCount_[rowKey];
        }
        {
            int offset = 0;
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                rRowStart_[r] = offset;
                offset += rRowCount_[r];
            }
        }
        rScatter_.assign(rRowStart_.begin(), rRowStart_.end() - 1);
        for (int i = 0; i < rightCount; ++i) {
            int rowKey = froundAwayI(rp[i].y / rowStep);
            if (rowKey < 0) rowKey = 0;
            if (rowKey >= MAX_EPIPOLAR_ROWS) continue;       // ≥8192 丢弃
            const size_t pos = static_cast<size_t>(rScatter_[rowKey]++);
            float x = (rp[i].x >= 0.0f) ? rp[i].x : 0.0f;    // 负 x 钳位（同键生成）
            int xq = froundEvenI(x * static_cast<float>(XQ_ONE));
            if (xq < 0) xq = 0;
            if (xq > static_cast<int>(XQ_MASK)) xq = static_cast<int>(XQ_MASK);
            rxSorted_[pos] = rp[i].x;
            rxXq_[pos] = xq;
        }
        for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
            const int beg = rRowStart_[r];
            const int cnt = rRowCount_[r];
            for (int k = 1; k < cnt; ++k) {
                const int kq = rxXq_[static_cast<size_t>(beg + k)];
                const float kx = rxSorted_[static_cast<size_t>(beg + k)];
                int j = k - 1;
                while (j >= 0 && rxXq_[static_cast<size_t>(beg + j)] > kq) {
                    rxXq_[static_cast<size_t>(beg + j + 1)] = rxXq_[static_cast<size_t>(beg + j)];
                    rxSorted_[static_cast<size_t>(beg + j + 1)] = rxSorted_[static_cast<size_t>(beg + j)];
                    --j;
                }
                rxXq_[static_cast<size_t>(beg + j + 1)] = kq;
                rxSorted_[static_cast<size_t>(beg + j + 1)] = kx;
            }
        }
    }
        if (kProf_) { char b[64]; std::snprintf(b, sizeof(b), " rightCsr=%.2f", tSince_()); tBd_ += b; }

        // ── Step 5：发卡（kernelIssueCardsV3 单线程移植；桶直达锚点）──
        // AVX2 运行时探测（HIT 掩码搜索 3×256b；无 AVX2 回退 8×128b SSE2）
        static const bool avx2 = []() {
            int regs[4];
            __cpuid(regs, 1);
            if (!(regs[2] & (1 << 28))) return false;   // AVX
            __cpuidex(regs, 7, 0);
            return (regs[1] & (1 << 5)) != 0;           // AVX2
        }();
        const bool useBuckets = (bucketK_ > 0 && !bucketPool_.empty());
        const float bMinX = bucketMinX_, bInvW = bucketInvW_;
        const int bK = bucketK_;
        static bool diagOnce_ = kProf_;
        if (diagOnce_) {
            diagOnce_ = false;
            int rowsWith = 0; long long entSum = 0; int maxEnt = 0;
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r)
                if (mapRowCount_[r] > 0) { ++rowsWith; entSum += mapRowCount_[r]; if (mapRowCount_[r] > maxEnt) maxEnt = mapRowCount_[r]; }
            int lRows = 0; long long lPt = 0;
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r)
                if (rRowCount_[r] > 0) { ++lRows; lPt += rRowCount_[r]; }
            std::fprintf(stderr, "[cpu-prof] mapRows=%d mapAvgEnt=%.0f mapMaxEnt=%d"
                         " rRows=%d rAvg=%.0f\n",
                         rowsWith, rowsWith ? double(entSum) / rowsWith : 0.0,
                         maxEnt, lRows, lRows ? double(lPt) / lRows : 0.0);
            // 表栅格结构诊断（等距→栅格直达锚点有效）
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                if (mapRowCount_[r] < 16) continue;
                const int s = mapRowStart_[r];
                std::fprintf(stderr, "[cpu-prof] row=%d cnt=%d x0..x3=",
                             r, mapRowCount_[r], mapRowX0_[r]);
                for (int k = 0; k < 4; ++k) std::fprintf(stderr, "%.4f ", mapEnt_[static_cast<size_t>(s + k)].x);
                std::fprintf(stderr, " d01=%.5f d12=%.5f\n",
                             mapEnt_[static_cast<size_t>(s + 1)].x - mapEnt_[static_cast<size_t>(s)].x,
                             mapEnt_[static_cast<size_t>(s + 2)].x - mapEnt_[static_cast<size_t>(s + 1)].x);
                break;
            }
        }
        const MapEnt* ent = mapEnt_.data();
        int noCardCnt = 0, noMapCnt = 0;   // 统计折入发卡主循环
        cardDat_.clear();
        cardPtr_[0] = 0;
        ptRow_.resize(static_cast<size_t>(leftCount));
        // 发卡主循环：轻量外壳（行守卫＋行级预取）＋noinline 单点处理（干净 RA）
        LmCard oneCards[V3_MAX_CARDS];
        int pfRow = -2;
        for (int i = 0; i < leftCount; ++i) {
            cardCount_[i] = 0;
            leftStatus_[i] = -1;
            cardPtr_[static_cast<size_t>(i)] = static_cast<int>(cardDat_.size());

            const cv::Point2f p = lp[i];
            const int row = froundAwayI(p.y / rowStep);
            ptRow_[static_cast<size_t>(i)] = row;        // 折入（无 clamp，同 GPU 口径）
            if (row < 0 || row >= MAX_EPIPOLAR_ROWS) { ++noMapCnt; continue; }
            const int mapCnt = mapRowCount_[row];
            if (mapCnt == 0) { ++noMapCnt; continue; }

            const int start = mapRowStart_[row];
            if (row != pfRow) {
                pfRow = row;
                // 行级批量预取：本行后续各点的桶跨行缓存背靠背发射（最大 MLP）
                if (useBuckets) {
                    const float rowY1 = (static_cast<float>(row) + 0.5f) * rowStep;
                    const float rowY0 = (static_cast<float>(row) - 0.5f) * rowStep;
                    const int poolBase = bucketRowOff_[row];
                    for (int k = i, kEnd = i + 64; k < leftCount && k < kEnd; ++k) {
                        const float yk = lp[k].y;   // 预取目标判定（近似即可——错取只是无效预取）
                        if (yk >= rowY1 || yk < rowY0) break;
                        int bk = static_cast<int>((lp[k].x - bMinX) * bInvW);
                        if (bk < 0) bk = 0;
                        if (bk > bK - 1) bk = bK - 1;
                        const int jk = start + bucketPool_[static_cast<size_t>(poolBase + bk)];
                        _mm_prefetch(reinterpret_cast<const char*>(&ent[jk - 5 < start ? start : jk - 5]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&ent[jk + 3]), _MM_HINT_T0);
                    }
                }
            }

            const auto r12 = issueOnePoint(p, row,
                                           ent, mapRowStart_.data(), mapRowCount_.data(),
                                           bucketPool_.data(), bucketRowOff_.data(),
                                           bMinX, bInvW, bK,
                                           vLtol, matchWin,
                                           rxSorted_.data(), rRowStart_.data(), rRowCount_.data(),
                                           urBiasModelActive_, urBiasActive_,
                                           urBias_.data(), biasParams_.data(),
                                           avx2, oneCards);
            const int out = r12.first;
            for (int c = 0; c < out; ++c) cardDat_.push_back(oneCards[c]);
            cardCount_[i] = out;
            leftStatus_[i] = out > 0 ? 1 : 0;
            if (out == 0) ++noCardCnt;
        }
        cardPtr_[static_cast<size_t>(leftCount)] = static_cast<int>(cardDat_.size());
        if (kProf_) { char b[64]; std::snprintf(b, sizeof(b), " issue=%.2f", tSince_()); tBd_ += b; }

        // ── Step 6：段表（sorted-unique rank；track 值域窄时 LUT 直映，否则排序＋哈希）──
        int segCount = 0;
        segmap_.resize(static_cast<size_t>(leftCount));
        {
            int mn = lt[0], mx = lt[0];
            for (int i = 1; i < leftCount; ++i) {
                const int t = lt[i];
                if (t < mn) mn = t;
                if (t > mx) mx = t;
            }
            const long long range = static_cast<long long>(mx) - mn + 1;
            if (mn >= 0 && range <= 4 * 1024 * 1024) {
                // LUT：present 前缀和＝sorted-unique rank（同 GPU 游程 rank，逐位一致）
                std::vector<int> lut(static_cast<size_t>(range) + 1, -1);
                for (int i = 0; i < leftCount; ++i) lut[lt[i] - mn] = 1;
                int run = 0;
                for (long long k = 0; k < range; ++k)
                    if (lut[static_cast<size_t>(k)] >= 0) lut[static_cast<size_t>(k)] = run++;
                segCount = run;
                for (int i = 0; i < leftCount; ++i)
                    segmap_[static_cast<size_t>(i)] = lut[lt[i] - mn];
                uniqTracks_.clear();
            } else {
                uniqTracks_.assign(lt, lt + leftCount);
                std::sort(uniqTracks_.begin(), uniqTracks_.end());
                uniqTracks_.erase(std::unique(uniqTracks_.begin(), uniqTracks_.end()), uniqTracks_.end());
                segCount = static_cast<int>(uniqTracks_.size());
                std::unordered_map<int, int> rankOf;
                rankOf.reserve(static_cast<size_t>(segCount) * 2);
                for (size_t k = 0; k < uniqTracks_.size(); ++k)
                    rankOf[uniqTracks_[k]] = static_cast<int>(k);
                for (int i = 0; i < leftCount; ++i)
                    segmap_[static_cast<size_t>(i)] = rankOf[lt[i]];
            }
        }

        // ── 投票＋行位图＋段点集（单遍；段内点升序 i；ptRow 已折入发卡循环）──
        segVotes_.assign(static_cast<size_t>(segCount), std::vector<int>(V3_MAX_LIDS, 0));
        segRows_.assign(static_cast<size_t>(segCount), BitRows(kRowWords, 0ULL));
        segPts_.assign(static_cast<size_t>(segCount), std::vector<int>());
        for (int i = 0; i < leftCount; ++i) {
            const int row = ptRow_[static_cast<size_t>(i)];
            const int seg = segmap_[i];
            if (seg >= 0 && seg < segCount) {
                segPts_[static_cast<size_t>(seg)].push_back(i);
                if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                    segRows_[static_cast<size_t>(seg)][static_cast<size_t>(row) / 64] |=
                        1ULL << (static_cast<size_t>(row) % 64);
            }
            const int nc = cardCount_[i];
            const LmCard* cards = cardDat_.data() + cardPtr_[i];
            for (int c = 0; c < nc; ++c) {
                const int lid = cards[c].lid;
                if (seg >= 0 && seg < segCount && lid >= 1 && lid <= V3_MAX_LIDS)
                    ++segVotes_[static_cast<size_t>(seg)][lid - 1];
            }
        }

        // winner/second 选举（argmax 平票取小 lid；second 供 relect 路径）
        winner_.assign(static_cast<size_t>(segCount), -1);
        second_.assign(static_cast<size_t>(segCount), -1);
        for (int s = 0; s < segCount; ++s) {
            const int* v = segVotes_[static_cast<size_t>(s)].data();
            winner_[static_cast<size_t>(s)] = electWinnerArgmax(v);
            if (winner_[static_cast<size_t>(s)] >= 1) {
                int best = 0, bestLid = -1, scnd = 0, scndLid = -1;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    const int vv = v[l];
                    if (vv > best) { scnd = best; scndLid = bestLid; best = vv; bestLid = l + 1; }
                    else if (vv > scnd) { scnd = vv; scndLid = l + 1; }
                }
                second_[static_cast<size_t>(s)] = scndLid;
            }
        }
        dropped_.assign(static_cast<size_t>(segCount), 0);

        if (params_.conflict_reelect) {
            if (kProf_) { char b[64]; std::snprintf(b, sizeof(b), " segmap=%.2f", tSince_()); tBd_ += b; }

            // ── P6b 前置 modifier（segment_split；候选预筛同判据；excl 达标后才算）──
            int p6bCandidates = 0;
            for (int s = 0; s < segCount; ++s) {
                if (dropped_[static_cast<size_t>(s)] || winner_[static_cast<size_t>(s)] < 1) continue;
                const int W = winner_[static_cast<size_t>(s)];
                const int* v = segVotes_[static_cast<size_t>(s)].data();
                int sv = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    if (l + 1 != W && v[l] > sv) sv = v[l];
                if (sv >= std::max(30, (v[W - 1] * 3) / 10)) { ++p6bCandidates; break; }
            }
            if (params_.segment_split && p6bCandidates > 0) {
                // exclCount 预备（kernelExclCountV3 判据；持 S 不持 W 点数）——仅对
                // 过双峰门槛①的候选段扫其段内点（等价：非候选段 W/S 组合不产生 excl）
                exclCnt_.assign(static_cast<size_t>(segCount), 0);
                for (int s = 0; s < segCount; ++s) {
                    if (dropped_[static_cast<size_t>(s)] || winner_[static_cast<size_t>(s)] < 1) continue;
                    const int W = winner_[static_cast<size_t>(s)];
                    const int S = second_[static_cast<size_t>(s)];
                    if (S < 1) continue;
                    const int* v = segVotes_[static_cast<size_t>(s)].data();
                    if (v[S - 1] < std::max(30, (v[W - 1] * 3) / 10)) continue;   // 同门槛①
                    const int seg = s;
                    for (int i : segPts_[static_cast<size_t>(seg)]) {
                        const int Wc = winner_[static_cast<size_t>(seg)];
                        const int Sc = second_[static_cast<size_t>(seg)];
                        const int nc = cardCount_[i];
                        const LmCard* cards = cardDat_.data() + cardPtr_[i];
                        bool hasW = false, hasS = false;
                        for (int c = 0; c < nc; ++c) {
                            const int lid = cards[c].lid;
                            if (lid == Wc) { hasW = true; break; }
                            if (lid == Sc) hasS = true;
                        }
                        if (hasS && !hasW) ++exclCnt_[static_cast<size_t>(seg)];
                    }
                }
                const int* exclPtr = exclCnt_.data();
                for (int r = 0; r < 3; ++r) {
                    const int n = splitBimodalSegments(
                        segmap_, lp, leftCount, rowStep,
                        cardDat_.data(), cardPtr_.data(), cardCount_.data(),
                        exclPtr,
                        segCount, segVotes_, segRows_, winner_, dropped_, ptRow_, segPts_);
                    if (n == 0) break;
                    exclPtr = nullptr;   // 旧扫描路径（卡已常驻）
                }
            }

            if (params_.cascade_elim) {
                if (kProf_) { char b[96]; std::snprintf(b, sizeof(b), " p6b=%.2f", tSince_()); tBd_ += b; }
                // ── P7：级联判定 ──
                CascadeCfg cp;
                cp.gateH = params_.cascade_gate_h;
                cp.gateShare = params_.cascade_gate_share;
                cp.gateMargin = params_.cascade_gate_margin;
                cp.friendXpx = params_.cascade_friend_xpx;
                cp.friendMinRows = params_.cascade_friend_minrows;
                cp.spacingPx = params_.cascade_spacing_px;
                cp.spacingTol = params_.cascade_spacing_tol;
                int banned = 0, merged = 0;
                const int determined = cascadeSolve(segCount, segmap_, lp, leftCount,
                                                    rowStep, ptRow_, segPts_,
                                                    segVotes_, segRows_,
                                                    winner_, dropped_, cp, &banned, &merged);
                CALIB_LOG_INFO("P7 cascade: segs={} determined={} merged={} banned={}",
                               segCount, determined, merged, banned);
            } else {
                // ── P5：冲突重选循环（≤10 轮）──
                int conflictsResolved = 0, reelected = 0, forcedDrop = 0;
                for (int iter = 0; iter < 10; ++iter)
                    if (!resolveConflictsOnce(segCount, segmap_, lp, leftCount,
                                              rowStep, segVotes_, segRows_,
                                              winner_, dropped_, &reelected, &forcedDrop))
                        break;
                    else ++conflictsResolved;
                CALIB_LOG_INFO("P5/P6b: segs={} conflictsResolved={} reelected={} forcedDrop={}",
                               segCount, conflictsResolved, reelected, forcedDrop);
            }
        } else {
            // ── 默认路径：同行同 winner 双弃（kernelConflictMarkV3 净语义）──
            // 首占者留名；任何后到异段 → 双弃；已弃段仍参与占胞（kernel 不查 drop）
            cellOwner_.assign(static_cast<size_t>(MAX_EPIPOLAR_ROWS) * V3_MAX_LIDS, -1);
            for (int i = 0; i < leftCount; ++i) {
                const int seg = segmap_[static_cast<size_t>(i)];
                if (seg < 0 || seg >= segCount) continue;
                const int w = winner_[static_cast<size_t>(seg)];
                if (w < 1) continue;
                const int row = ptRow_[i];
                if (row < 0 || row >= MAX_EPIPOLAR_ROWS) continue;
                const size_t cell = static_cast<size_t>(row) * V3_MAX_LIDS + (w - 1);
                const int prev = cellOwner_[cell];
                if (prev < 0) {
                    cellOwner_[cell] = seg;
                } else if (prev != seg) {
                    dropped_[static_cast<size_t>(seg)] = 1;
                    dropped_[static_cast<size_t>(prev)] = 1;
                }
            }
        }

        // ── Step 8/9：收集（保留段 winner 卡；原序追加＝DeviceSelect 保序）──
        if (kProf_) { char b[64]; std::snprintf(b, sizeof(b), " p5p7=%.2f", tSince_()); tBd_ += b; }
        outLeft_.clear();
        outRight_.clear();
        outLid_.clear();
        outTrack_.clear();
        std::fill(rightStatus_.begin(), rightStatus_.begin() + rightCount, 0);
        int noWinnerCard = 0;
        int rightUsed = 0;
        for (int i = 0; i < leftCount; ++i) {
            const int seg = segmap_[static_cast<size_t>(i)];
            if (seg < 0 || seg >= segCount) continue;
            if (dropped_[static_cast<size_t>(seg)] != 0) continue;   // dropped segment
            const int w = winner_[static_cast<size_t>(seg)];
            if (w < 1) continue;
            const int nc = cardCount_[i];
            const LmCard* cards = cardDat_.data() + cardPtr_[i];
            int wc = -1;
            for (int c = 0; c < nc; ++c) {
                if (cards[c].lid == w) { wc = c; break; }
            }
            if (wc < 0) {
                ++noWinnerCard;                        // 保留段无 winner 卡（原 gather 静默丢弃）
                continue;
            }
            outLeft_.push_back(lp[i]);
            outRight_.emplace_back(cards[wc].hitRx, lp[i].y);   // y＝左点 y（GPU 对齐）
            outLid_.push_back(w);
            outTrack_.push_back(lt[i]);
            const int ridx = cards[wc].ridx;
            if (rightStatus_[ridx] == 0) {
                rightStatus_[ridx] = 1;
                ++rightUsed;
            }
        }

        // ── Step 10：统计（noCard/noMap 已折入发卡循环；rightUsed 已折入收集）──
        int droppedSegs = 0, emptySegs = 0;
        for (int s = 0; s < segCount; ++s) {
            if (dropped_[static_cast<size_t>(s)] != 0) ++droppedSegs;
            if (winner_[static_cast<size_t>(s)] < 1) ++emptySegs;
        }
        const int noCard = noCardCnt;
        const int noMap = noMapCnt;

        result.matchedCount = static_cast<int>(outLeft_.size());
        result.excludedLeftCount = noCard;
        result.excludedRightCount = rightCount - rightUsed;
        result.noMapLeftCount = noMap;
        result.noWinnerCardCount = noWinnerCard;
        result.emptyWinnerSegCount = emptySegs;
        result.trackTotal = segCount;
        result.trackDropped = droppedSegs;

        // 输出 Mats（容量式：内存复用，ROI 视图语义无必要——值拷贝输出）
        const int n = result.matchedCount;
        if (n > 0) {
            outLeftM_.create(n, 1, CV_32FC2);
            std::memcpy(outLeftM_.ptr(), outLeft_.data(), sizeof(cv::Point2f) * n);
            outRightM_.create(n, 1, CV_32FC2);
            std::memcpy(outRightM_.ptr(), outRight_.data(), sizeof(cv::Point2f) * n);
            outLidM_.create(n, 1, CV_32SC1);
            std::memcpy(outLidM_.ptr(), outLid_.data(), sizeof(int) * n);
            outTrackM_.create(n, 1, CV_32SC1);
            std::memcpy(outTrackM_.ptr(), outTrack_.data(), sizeof(int) * n);
            result.matched_left = outLeftM_;
            result.matched_right = outRightM_;
            result.matched_line_ids = outLidM_;
            result.matched_track_ids = outTrackM_;
        }
        result.left_status.assign(leftStatus_.begin(), leftStatus_.begin() + leftCount);
        result.right_status.assign(rightStatus_.begin(), rightStatus_.begin() + rightCount);

        if (kProf_) {
            char b[160];
            std::snprintf(b, sizeof(b), " gather+stats=%.2f total=%.2f N=%d",
                          tSince_(),
                          std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tT0_).count(),
                          leftCount);
            tBd_ += b;
            std::fprintf(stderr, "[cpu-prof]%s\n", tBd_.c_str());
        }

        result.success = true;
        result.message = fmt::format(
            "OK: {} matched, {}/{} tracks kept, no-card={} no-map={} exclR={} "
            "noWinnerCard={} emptySeg={}",
            n, segCount - droppedSegs, segCount, noCard, noMap,
            rightCount - rightUsed, noWinnerCard, emptySegs);

        CALIB_LOG_DEBUG("process(): {}", result.message);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    return result;
}

// ============================================================================
// 公开接口桥接
// ============================================================================

LaserMatchScanCpuV3::LaserMatchScanCpuV3(const LaserMatchScanParamsV3& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserMatchScanCpuV3 initialized: threshold={}",
                   params.match_threshold);
}

LaserMatchScanCpuV3::~LaserMatchScanCpuV3() = default;

bool LaserMatchScanCpuV3::LoadTempTable(const std::string& jsonPath) {
    return pImpl_->LoadTempTable(jsonPath);
}

bool LaserMatchScanCpuV3::SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table) {
    return pImpl_->SetTempTable(std::move(table));
}

void LaserMatchScanCpuV3::SetCurrentTemperature(double temperature) {
    pImpl_->SetCurrentTemperature(temperature);
}

LaserMatchScanCpuResultV3 LaserMatchScanCpuV3::Execute(
    const cv::Mat& left_points,
    const cv::Mat& left_line_ids,
    const cv::Mat& right_points,
    const cv::Mat& right_line_ids)
{
    CALIB_LOG_DEBUG("Execute() called: left={}x{} type={}, right={}x{} type={}",
                    left_points.cols, left_points.rows, left_points.type(),
                    right_points.cols, right_points.rows, right_points.type());

    if (left_points.empty() || left_line_ids.empty() ||
        right_points.empty() || right_line_ids.empty()) {
        CALIB_LOG_WARN("Execute(): empty input, returning empty result");
        LaserMatchScanCpuResultV3 result;
        result.success = true;
        result.message = "Empty input, no points to match";
        return result;
    }

    if (left_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: left_points type={}, expected CV_32FC2", left_points.type());
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Left points must be CV_32FC2";
        return result;
    }

    if (right_points.type() != CV_32FC2) {
        CALIB_LOG_ERROR("Execute() failed: right_points type={}, expected CV_32FC2", right_points.type());
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Right points must be CV_32FC2";
        return result;
    }

    if (left_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: left_line_ids type={}, expected CV_32SC1", left_line_ids.type());
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Left line_ids must be CV_32SC1";
        return result;
    }

    if (right_line_ids.type() != CV_32SC1) {
        CALIB_LOG_ERROR("Execute() failed: right_line_ids type={}, expected CV_32SC1", right_line_ids.type());
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Right line_ids must be CV_32SC1";
        return result;
    }

    int leftPointCount = left_points.rows * left_points.cols;
    int leftIdCount = left_line_ids.rows * left_line_ids.cols;
    if (leftPointCount != leftIdCount) {
        CALIB_LOG_ERROR("Execute() failed: left points count={} != line_ids count={}",
                        leftPointCount, leftIdCount);
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Left points and line_ids must have same number of elements";
        return result;
    }

    int rightPointCount = right_points.rows * right_points.cols;
    int rightIdCount = right_line_ids.rows * right_line_ids.cols;
    if (rightPointCount != rightIdCount) {
        CALIB_LOG_ERROR("Execute() failed: right points count={} != line_ids count={}",
                        rightPointCount, rightIdCount);
        LaserMatchScanCpuResultV3 result;
        result.success = false;
        result.message = "Right points and line_ids must have same number of elements";
        return result;
    }

    return pImpl_->process(left_points, left_line_ids, right_points, right_line_ids);
}

void LaserMatchScanCpuV3::Destroy() {
    pImpl_.reset();
}

void LaserMatchScanCpuV3::Warmup(int maxLeftPoints, int maxRightPoints) {
    CALIB_LOG_INFO("Warmup() called: maxLeft={}, maxRight={}", maxLeftPoints, maxRightPoints);
    if (pImpl_) pImpl_->warmup(maxLeftPoints, maxRightPoints);
}

void LaserMatchScanCpuV3::Warmup(const calib::WarmupConfig& config) {
    int pointCount = config.maxPointCount > 0 ? config.maxPointCount : config.rows * config.cols;
    Warmup(pointCount, pointCount);
}

void LaserMatchScanCpuV3::SetParams(const LaserMatchScanParamsV3& params) {
    CALIB_LOG_INFO("SetParams(): threshold={}", params.match_threshold);
    if (pImpl_) pImpl_->setParams(params);
}

const LaserMatchScanParamsV3& LaserMatchScanCpuV3::GetParams() const {
    if (!pImpl_)
        throw std::runtime_error("[07-LaserMatchScanCpuV3] GetParams() after Destroy()");
    return pImpl_->getParams();
}

void LaserMatchScanCpuV3::SetuRBias(const std::vector<float>& biasPerLid) {
    if (pImpl_) pImpl_->setuRBias(biasPerLid);
}

void LaserMatchScanCpuV3::SetuRBiasModel(float wx, float wy, float wz, float dtx) {
    if (pImpl_) pImpl_->setuRBiasModel(wx, wy, wz, dtx);
}
