/**
 * @file laser_match_scan_v4.cpp
 * @brief 激光线匹配扫描 V4 — 裁判型 CPU 后段实现（REA 对标；纯 CPU 无 CUDA）
 *
 * 2026-11-05 自 laser_match_scan_v3_split_cpu.cpp 机械移植（该实现已经 GPU A/B
 * 精确一致验证）：P5/P6b/P7 宿主求解 helper 与投票/选举/冲突/收集/统计流程
 * 【逐字保留区】——判据与加法序列零改动（L0 恒等的前提）。
 *
 * v4 新增（详见 laser_match_scan_v4.h 头注）：
 *   - B-pass2：深度门卡过滤（card_depth_gate，无需温表，Q 经 SetStereoQ 注入）
 *   - N 阶段（D2 之后、E 之前——cascadeSolve 写回段会整体覆写 winner_/dropped_
 *     （v3_cpu.cpp 镜像 :896-913），仲裁前置会被级联吞掉，故后置）：
 *       争议段（W<margin×S 或 W<share×T）→ 单遍查表取 gap 样本＋逐卡残差 →
 *       自适应窗 mask（w_eff=max(floor,min(mt,gap/2))）→ 质量仲裁
 *       （meanRes argmin＋分离门＋单票否决）→ 分不开则弃权
 *   - 收集：winner 卡被 mask → 按无卡处理（adaptNoCardPts）；输出对附逐点残差
 *   - 温表双槽缓存（双管交替注入免重转换）＋ host 查表（移植自
 *     laser_match_scan_v3_cpu.cpp issueOnePoint 的锚点跳走＋双向展开段）
 *
 * 2026-10-10 残差加权证伪史防护：仅争议段介入 / 分离门 / 整数票仓与判定链
 * 逐字不动。非争议段与 v4 通道关闭时＝与 V3 后段逐位一致。
 *
 * ⚠ 沙箱定位：TEST/fullgpu_optimize 实验算子，不入 mod_operatorlib，不入产线。
 */

#include "laser_match_scan_v4.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace calib {

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanSplitCpuV4);

namespace {

// ============================================================================
// 常量（同值拷贝自 laser_match_scan_v3_cpu.cpp——勿漂移）
// ============================================================================

constexpr int MAX_EPIPOLAR_ROWS = 8192;
constexpr int V3_MAX_CARDS = 32;      // 每点卡上限（每 lid 就近单卡）
constexpr int V3_MAX_LIDS = 32;       // lid 桶距（lid 1..32）
constexpr int kRowWords = MAX_EPIPOLAR_ROWS / 64;   // 128（段行位图字数）

// roundf 半值远离零语义的 MSVC 内联等价（|x|<2^22 按位一致）
inline int froundAwayI(float x) { return static_cast<int>(x >= 0.0f ? x + 0.5f : x - 0.5f); }

// 发卡紧凑记录（CSR 元素：lid＋右排序系索引＋命中 x）

// ============================================================================
// P5/P6b/P7 宿主求解（自 laser_match_scan_v3_cpu.cpp 机械移植；匿名 namespace
// 防链接冲突——判据与加法序列逐字保留）
// ============================================================================

int electWinnerArgmax(const int* votes26) {          // 平票取小 lid
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
// 【池化】cellA/cellB/sumA/sumB/cntA/cntB thread_local（同尺寸 assign＝零重分配；
// 数值语义与逐次构造逐位一致——全部元素每轮重写/清零）
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
    thread_local std::vector<int> cellA, cellB;
    cellA.assign(static_cast<size_t>(MAX_EPIPOLAR_ROWS) * V3_MAX_LIDS, -1);
    cellB.assign(static_cast<size_t>(MAX_EPIPOLAR_ROWS) * V3_MAX_LIDS, -1);
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
            thread_local std::vector<double> sumA, sumB;
            thread_local std::vector<int> cntA, cntB;
            sumA.assign(MAX_EPIPOLAR_ROWS, 0.0);
            sumB.assign(MAX_EPIPOLAR_ROWS, 0.0);
            cntA.assign(MAX_EPIPOLAR_ROWS, 0);
            cntB.assign(MAX_EPIPOLAR_ROWS, 0);
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
            CALIB_LOG_DEBUG("P5 conflict: segs {}/{} lid={} votesA={} votesB={} rowVote {}/{} "
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
            CALIB_LOG_DEBUG("P5 conflict (no order): segs {}/{} lid={} votesA={} votesB={} "
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
// 【v4 参数化】p6bMinVotes/p6bVotePct＝双峰门槛（原硬编码 max(30, 30%)——
// 2026-11-06 线覆盖优化：收敛区真线少数派不足 30% 被吞，暴露为可调参数）
int splitBimodalSegments(std::vector<int>& h_segmap,
                         const cv::Point2f* h_left, int leftCount, float /*rowStep*/,
                                 const int* cardLid, const int* h_card_ptr,
                         const int* h_exclCnt,
                         int& segCount,
                         std::vector<std::vector<int>>& segVotes,
                         std::vector<BitRows>& segRows,
                         std::vector<int>& winner,
                         std::vector<char>& dropped,
                         const std::vector<int>& ptRow,
                         std::vector<std::vector<int>>& segPts,
                         int p6bMinVotes = 30,
                         int p6bVotePct = 30)
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
        if (S < 1 || sVotes < std::max(p6bMinVotes, (v[W - 1] * p6bVotePct) / 100)) continue;
        // ② 点可分：优先卡独占（持 S 不持 W）；不足量 → 行序分割兜底
        std::vector<int> exclPts;
        std::vector<unsigned long long> exclRows(kRowWords, 0ULL);
        int exclSz = -1;
        if (h_exclCnt != nullptr) exclSz = h_exclCnt[s];
        bool needCardScan = (exclSz < 0) || (exclSz >= (sVotes * 8) / 10);
        if (needCardScan) {
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int beg6 = h_card_ptr[i];
                const int nc = h_card_ptr[i + 1] - beg6;
                const int* lids = cardLid + beg6;
                bool hasW = false, hasS = false;
                for (int c = 0; c < nc; ++c) {
                    const int lid = lids[c];
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
                CALIB_LOG_DEBUG("P6b reject(seg {}): rowOrder qty=0/{} out of band (W={} S={})",
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
                    CALIB_LOG_DEBUG("P6b reject(seg {}): rowOrder qty={}/{} out of band (W={} S={})",
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
            const int begA = h_card_ptr[i];
            const int ncA = h_card_ptr[i + 1] - begA;
            const int* lidsA = cardLid + begA;
            for (int c = 0; c < ncA; ++c) {
                const int lid = lidsA[c];
                if (lid >= 1 && lid <= V3_MAX_LIDS) ++vA[lid - 1];
            }
        }
        for (int i : segPts[static_cast<size_t>(newSeg)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                rB[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            const int begB = h_card_ptr[i];
            const int ncB = h_card_ptr[i + 1] - begB;
            const int* lidsB = cardLid + begB;
            for (int c = 0; c < ncB; ++c) {
                const int lid = lidsB[c];
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

// 【插桩·仅测量】P7 子相位耗时（JMW_V3P7T=1 启用；默认零开销——env 静态查一次，
// 时钟读全部条件化）。子相位：stamp（行戳记+稀疏化）/ merge（友军合并）/
// unit+casc（单元组装+级联迭代）/ wb（写回）。结果经 thread_local 槽带回 process()
// 并入 lastBd_（conflict 段之后），同时独立打 [v3:p7t] 行（不依赖 JMW_FCBD）。
static bool p7TimingOn() {
    static const bool on = [] {
        const char* e = getenv("JMW_V3P7T");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
static std::string& p7BdSlot() { thread_local std::string s; return s; }

// 【B1】P7 友军合并标量臂开关（JMW_V3P7MS=1 → 原标量循环；默认 pair16 臂。
// env 静态查一次；两臂每槽加法序列相同 → 结果逐位一致，运行期可任意切换）
static bool p7MergeScalar() {
    static const bool on = [] {
        const char* e = getenv("JMW_V3P7MS");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

// 【B1】友军合并累加器 16B 打包槽：psum 与 pn 同槽连续——每对一次 loadu/storeu
struct P7PairSlot { double s; long long n; };
static_assert(sizeof(P7PairSlot) == 16, "P7PairSlot must be 16B (loadu/storeu)");

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
    // 【刀三-P7】行区间（rowsOverlap 区间门——区间不相交时免扫位图字）
    int rowMin = 0x7fffffff, rowMax = -1;
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
    const bool p7t = p7TimingOn();   // 【插桩】P7 子相位 T0（stamp 段起点）
    const auto p7tA = p7t ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
    std::chrono::steady_clock::time_point p7tB{}, p7tC{}, p7tD{};
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
        for (int w = 0; w < kRowWords; ++w)
            n += static_cast<size_t>(__popcnt64(bits_[static_cast<size_t>(w)]));
        if (n == 0) continue;
        segRowIdx[static_cast<size_t>(s)].resize(n);
        segXSum[static_cast<size_t>(s)].resize(n);
        segXCnt[static_cast<size_t>(s)].resize(n);
        size_t k = 0;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long b = bits_[static_cast<size_t>(w)];
            while (b) {
                unsigned long bit;
                _BitScanForward64(&bit, b);
                const int row = w * 64 + static_cast<int>(bit);
                segRowIdx[static_cast<size_t>(s)][k] = row;
                segXSum[static_cast<size_t>(s)][k] = stampSum[static_cast<size_t>(row)];
                segXCnt[static_cast<size_t>(s)][k] = stampCnt[static_cast<size_t>(row)];
                stampSum[static_cast<size_t>(row)] = 0.0;
                stampCnt[static_cast<size_t>(row)] = 0;
                ++k;
                b &= b - 1;   // 清最低置位——(k,row) 升序序列与原逐位相同
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
    long long mergePairs = 0, pairAdds = 0;
    if (p7t) p7tB = std::chrono::steady_clock::now();   // 【插桩】stamp→merge
    if (segCount > 0 && segCount <= 512) {
        // —— 行主序稠密对累加（S≤512；对 (a,b) 槽的 sum 加法序列＝行严格升序，
        //    与 GPU 宿主路径同一加法序列→逐位一致；实测达标对在长段数据上近乎
        //    全量，两阶段先数后算无收益，保持一阶段）——
        const int S = segCount;
        // 【B1】pair16 打包槽：psum 与 pn 同槽连续——每对一次 loadu/storeu 完成
        //「浮点累加＋整数计数」（探针 6 形态×5 种子全阵 memcmp 零差；密集档
        // 1.34-1.71×；AVX2 gather 路线已证伪不入）
        thread_local std::vector<P7PairSlot> pkP;
        thread_local std::vector<int> rowSegCntP, rowSegOffP, rowSegDataP, curP;
        thread_local std::vector<double> rowSegXbP;
        auto& pk16 = pkP;
        auto& rowSegCnt = rowSegCntP;
        auto& rowSegOff = rowSegOffP;
        auto& rowSegData = rowSegDataP;
        auto& rowSegXb = rowSegXbP;
        auto& cur = curP;
        pk16.assign(static_cast<size_t>(S) * S, P7PairSlot{ 0.0, 0 });
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
        if (p7t)   // 【B1·定档插桩】对加总数（密度档判据：1024/2048 带≈1.34-1.55×；wide8192 带收益<0.2ms）
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                const long long nr = rowSegCnt[static_cast<size_t>(r)];
                pairAdds += nr * (nr - 1) / 2;
            }
        if (!p7MergeScalar()) {
            // 【B1】pair16 主臂：4 对交错（每对 1 loadu＋1 storeu；psum 加法序/pn
            // 整数计数不变 → 位精确；blend 掩码 2＝{和, n+1}；add_epi64 双 lane +1
            // 的低位垃圾被 blend 丢弃——探针逐指令一致）
            const __m128i one64 = _mm_set1_epi64x(1);
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                const int beg = rowSegOff[static_cast<size_t>(r)];
                const int cnt = rowSegCnt[static_cast<size_t>(r)];
                for (int ia_ = beg; ia_ < beg + cnt; ++ia_) {
                    const int a = rowSegData[static_cast<size_t>(ia_)];
                    const double xa = rowSegXb[static_cast<size_t>(ia_)];
                    P7PairSlot* sbase = pk16.data() + static_cast<size_t>(a) * S;
                    const int* data = rowSegData.data();
                    const double* xb = rowSegXb.data();
                    const int nb = beg + cnt;
                    int ib_ = ia_ + 1;
                    for (; ib_ + 4 <= nb; ib_ += 4) {
                        P7PairSlot* t0 = sbase + data[ib_];
                        P7PairSlot* t1 = sbase + data[ib_ + 1];
                        P7PairSlot* t2 = sbase + data[ib_ + 2];
                        P7PairSlot* t3 = sbase + data[ib_ + 3];
                        const __m128d v0 = _mm_loadu_pd(&t0->s);
                        const __m128d v1 = _mm_loadu_pd(&t1->s);
                        const __m128d v2 = _mm_loadu_pd(&t2->s);
                        const __m128d v3 = _mm_loadu_pd(&t3->s);
                        const __m128d a0 = _mm_add_sd(v0, _mm_set_sd(std::fabs(xa - xb[ib_])));
                        const __m128d a1 = _mm_add_sd(v1, _mm_set_sd(std::fabs(xa - xb[ib_ + 1])));
                        const __m128d a2 = _mm_add_sd(v2, _mm_set_sd(std::fabs(xa - xb[ib_ + 2])));
                        const __m128d a3 = _mm_add_sd(v3, _mm_set_sd(std::fabs(xa - xb[ib_ + 3])));
                        const __m128i i0 = _mm_add_epi64(_mm_castpd_si128(v0), one64);
                        const __m128i i1 = _mm_add_epi64(_mm_castpd_si128(v1), one64);
                        const __m128i i2 = _mm_add_epi64(_mm_castpd_si128(v2), one64);
                        const __m128i i3 = _mm_add_epi64(_mm_castpd_si128(v3), one64);
                        _mm_storeu_pd(&t0->s, _mm_blend_pd(a0, _mm_castsi128_pd(i0), 2));
                        _mm_storeu_pd(&t1->s, _mm_blend_pd(a1, _mm_castsi128_pd(i1), 2));
                        _mm_storeu_pd(&t2->s, _mm_blend_pd(a2, _mm_castsi128_pd(i2), 2));
                        _mm_storeu_pd(&t3->s, _mm_blend_pd(a3, _mm_castsi128_pd(i3), 2));
                    }
                    for (; ib_ < nb; ++ib_) {          // 尾段：同构造单 16B RMW
                        P7PairSlot& t = sbase[data[ib_]];
                        const __m128d v = _mm_loadu_pd(&t.s);
                        const __m128d aa = _mm_add_sd(v, _mm_set_sd(std::fabs(xa - xb[ib_])));
                        const __m128i ii = _mm_add_epi64(_mm_castpd_si128(v), one64);
                        _mm_storeu_pd(&t.s, _mm_blend_pd(aa, _mm_castsi128_pd(ii), 2));
                    }
                }
            }
        } else {
            // 标量回退臂（JMW_V3P7MS=1；原累加序列逐字——直写打包槽）
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                const int beg = rowSegOff[static_cast<size_t>(r)];
                const int cnt = rowSegCnt[static_cast<size_t>(r)];
                for (int ia_ = beg; ia_ < beg + cnt; ++ia_) {
                    const int a = rowSegData[static_cast<size_t>(ia_)];
                    const double xa = rowSegXb[static_cast<size_t>(ia_)];
                    for (int ib_ = ia_ + 1; ib_ < beg + cnt; ++ib_) {
                        const int b = rowSegData[static_cast<size_t>(ib_)];
                        P7PairSlot& t = pk16[static_cast<size_t>(a) * S + b];
                        t.s += std::fabs(xa - rowSegXb[static_cast<size_t>(ib_)]);
                        ++t.n;
                    }
                }
            }
        }
        for (int a = 0; a < S; ++a) {
            for (int b = a + 1; b < S; ++b) {
                const P7PairSlot& t = pk16[static_cast<size_t>(a) * S + b];
                if (t.n >= cfg.friendMinRows && t.s / static_cast<double>(t.n) <= cfg.friendXpx) {
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
    if (p7t) p7tC = std::chrono::steady_clock::now();   // 【插桩】merge→unit+casc
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
    // 【刀三-P7a】行→槽戳记表（代际戳——免逐单元清表）：替代逐条目 lower_bound
    // 的 O(log n) 随机访问为 O(1) 直查；ridx[k] ∈ u.rowIdx 恒成立（u.rows＝成员
    // 位图之 OR），槽号与 lower_bound 结果一致 → 加法序列逐位不变
    thread_local std::vector<int> slotGenP, slotOfP;
    thread_local int slotGenCur = 0;
    const int kStampDomain = 8193;   // ptRow 值域上界（行号 ≤ 8192 同 interp 域）
    if (static_cast<int>(slotGenP.size()) < kStampDomain) {
        slotGenP.assign(static_cast<size_t>(kStampDomain), -1);
        slotOfP.assign(static_cast<size_t>(kStampDomain), 0);
    }
    for (size_t uu = 0; uu < uCount; ++uu) {
        CascadeUnit& u = units[uu];
        u.rowIdx.clear();
        u.rowMin = 0x7fffffff;
        u.rowMax = -1;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = u.rows[static_cast<size_t>(w)];
            for (int row = w * 64; bits; bits >>= 1, ++row)
                if (bits & 1ULL) {
                    u.rowIdx.push_back(row);
                    if (row < u.rowMin) u.rowMin = row;
                    if (row > u.rowMax) u.rowMax = row;
                }
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
        ++slotGenCur;
        const int gen = slotGenCur;
        static const bool p7Legacy_ = (getenv("JMW_P7_LEGACY") != nullptr
                                       && getenv("JMW_P7_LEGACY")[0] == '1');
        if (!p7Legacy_) {
            for (size_t k = 0; k < n; ++k) {
                const int row = u.rowIdx[k];
                if (row >= 0 && row < kStampDomain) {
                    slotGenP[static_cast<size_t>(row)] = gen;
                    slotOfP[static_cast<size_t>(row)] = static_cast<int>(k);
                }
            }
        }
        for (int m : u.members) {
            const auto& ridx = segRowIdx[static_cast<size_t>(m)];
            const auto& xs = segXSum[static_cast<size_t>(m)];
            const auto& xc = segXCnt[static_cast<size_t>(m)];
            for (size_t k = 0; k < ridx.size(); ++k) {
                const int row = ridx[k];
                size_t slot;
                if (!p7Legacy_ && row >= 0 && row < kStampDomain
                    && slotGenP[static_cast<size_t>(row)] == gen) {
                    slot = static_cast<size_t>(slotOfP[static_cast<size_t>(row)]);
                } else {
                    slot = static_cast<size_t>(
                        std::lower_bound(u.rowIdx.begin(), u.rowIdx.end(), row)
                        - u.rowIdx.begin());   // 旧路径/域外兜底（≥8193 物理不现）
                }
                u.rowXSum[slot] += xs[k];   // 行槽加法序＝成员升序
                u.rowXCnt[slot] += xc[k];
            }
        }
    }
    // 【刀三-P7b】区间门：区间不相交 ⇒ 位图必无公共位——布尔结果严格不变
    auto rowsOverlap = [](const CascadeUnit& A, const CascadeUnit& B) {
        if (A.rowMax < B.rowMin || B.rowMax < A.rowMin) return false;
        for (size_t w = 0; w < A.rows.size(); ++w)
            if (A.rows[w] & B.rows[w]) return true;
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
                if (rowsOverlap(bu, u) && u.votes[static_cast<size_t>(wl - 1)] > 0)
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
            if (!rowsOverlap(A, B)) continue;
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
    if (p7t) p7tD = std::chrono::steady_clock::now();   // 【插桩】unit+casc→wb
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
    if (p7t) {   // 【插桩】四子相位成串（ms；顺 process() 并入 lastBd_）
        const auto p7tE = std::chrono::steady_clock::now();
        using MsD = std::chrono::duration<double, std::milli>;
        p7BdSlot() = fmt::format(
            " p7stamp={:.3f} p7merge={:.3f} p7casc={:.3f} p7wb={:.3f} pairAdds={}",
            MsD(p7tB - p7tA).count(), MsD(p7tC - p7tB).count(),
            MsD(p7tD - p7tC).count(), MsD(p7tE - p7tD).count(), pairAdds);
    }
    return determinedCnt;
}

} // namespace

// ============================================================================
// Impl — 后段状态与主流程
// ============================================================================

struct LaserMatchScanSplitCpuV4::Impl {
    // —— v4 参数持有：v3 视图经引用绑定（【逐字保留区】全部经 params_ 访问 v3
    //    参数，代码零改动；v4 开关经 v4params_ 访问）——
    LaserMatchScanParamsV4 v4params_;
    LaserMatchScanParamsV3& params_ = v4params_.v3;

    // —— 输入组装（从前段 SoA → 本地工作布局；容量跨帧复用）——
    // 输入工作区（【B1 刀4】卡表直接消费前段 SoA，不落地 LmCard/cardPtr 副本）
    std::vector<int> ptRow_;            // [N]（froundAwayI(y/rowStep)，无 clamp）
    std::vector<int> segmap_;           // 工作副本（P6b 分裂原地改写）

    // —— 段结构 ——
    std::vector<std::vector<int>> segVotes_;
    std::vector<BitRows> segRows_;
    std::vector<int> winner_;
    std::vector<int> second_;
    std::vector<char> dropped_;
    std::vector<std::vector<int>> segPts_;
    std::vector<int> exclCnt_;

    // —— 冲突占胞（默认路径）——
    std::vector<int> cellOwner_;

    // —— 输出缓冲（容量式）——
    std::vector<cv::Point2f> outLeft_, outRight_;
    std::vector<int> outLid_, outTrack_;
    std::vector<int> rightStatus_;
    cv::Mat outLeftM_, outRightM_, outLidM_, outTrackM_;

    std::string lastBd_;

    // —— v4 裁判通道 ——
    std::vector<unsigned char> cardMask_;   // [totalCards] 1=被 v4 mask（窗/深度门）
    std::vector<float> resOut_;             // 输出对残差（容量复用；拷贝进 result）
    bool hasQ_ = false;
    double q23_ = 0.0, q32_ = 0.0, q33_ = 0.0;   // Z=Q(2,3)/(Q(3,2)·d+Q(3,3))

    // —— 温表双槽缓存（双管交替注入免重转换；judgeReady 门）——
    struct TempTableEntryV4 {
        double temperature = 0.0;
        std::vector<float> mapData;         // x,y,uR,lid 交错（展平自 leftToRightMap）
        int entryCount = 0;
        int numLines = 0;
    };
    struct MapEntV4 { float x, uR; int lid; };   // 行内条目 12B AoS（同 v3_cpu MapEnt）
    struct TubeSlot {
        std::shared_ptr<const LaserPlaneMapTempTable> hold;   // 源表（只读共享）
        std::vector<TempTableEntryV4> temps;                  // 逐温档展平
        bool loaded = false;
        // 活动 CSR＋桶索引池（buildActiveMap 产物）
        int mapCount = 0, numLines = 0;
        std::vector<MapEntV4> ent;
        std::vector<int> rowStart, rowCount, rowX0;
        std::vector<int> bucketPool, bucketRowOff;
        float bMinX = 0.0f, bInvW = 0.0f;
        int bK = 0;
        const void* builtSrc = nullptr;
        double builtTemp = 0.0;
        float builtRowStep = -1.0f;
    };
    TubeSlot slot_[2];
    int active_ = -1;        // 活动 slot 下标；-1=未激活
    double curTemp_ = 0.0;   // 最近解析温档
    bool tempSet_ = false;

    bool setTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void setCurrentTemperature(double temperature);
    void setStereoQ(const cv::Mat& Q);
    void buildActiveMap(TubeSlot& s, const TempTableEntryV4& entry);
    int queryPoint(float xL, int row,
                   int (&locLid)[V3_MAX_CARDS], float (&locU)[V3_MAX_CARDS]);
    bool judgeReady() const {
        return active_ >= 0 && tempSet_ && slot_[active_].builtSrc != nullptr;
    }

    explicit Impl(const LaserMatchScanParamsV4& params) : v4params_(params) {
        v4params_.validate();
    }

    LaserMatchScanCpuResultV4 process(const LaserMatchSplitFrontV3& f);
};

namespace {
inline int clampRowV4(int row) {
    return row < 0 ? 0 : (row >= MAX_EPIPOLAR_ROWS ? MAX_EPIPOLAR_ROWS - 1 : row);
}
}

// ============================================================================
// 温表装载与查表（移植自 laser_match_scan_v3_cpu.cpp —— 双槽化）
// ============================================================================

bool LaserMatchScanSplitCpuV4::Impl::setTempTable(
    std::shared_ptr<const LaserPlaneMapTempTable> table)
{
    if (!table || table->empty()) {
        CALIB_LOG_ERROR("setTempTable(): null or empty temp table");
        return false;
    }
    // 同源槽命中（双管交替：两槽各驻一表，切 active_ 即可，免重转换）
    for (int i = 0; i < 2; ++i) {
        if (slot_[i].loaded && slot_[i].hold && slot_[i].hold.get() == table.get()) {
            active_ = i;
            return true;
        }
    }
    // 驱逐非活动槽（活动槽受保护；active_==-1 时驱逐 0 号）
    const int victim = (active_ == 0) ? 1 : 0;
    TubeSlot& s = slot_[victim];
    s.hold = table;
    s.builtSrc = nullptr;
    s.temps.clear();
    s.ent.clear(); s.bucketPool.clear(); s.rowStart.clear(); s.rowCount.clear();
    s.rowX0.clear(); s.bucketRowOff.clear();
    s.bK = 0; s.mapCount = 0; s.numLines = 0; s.loaded = false;

    for (const auto& kv : table->table) {
        const double temp = kv.first;
        const LaserPlaneMap& planeMap = kv.second;
        const cv::Mat& m = planeMap.leftToRightMap;
        if (m.empty()) {
            CALIB_LOG_WARN("setTempTable(): skip temp={} (empty leftToRightMap)", temp);
            continue;
        }
        TempTableEntryV4 te;
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
        s.temps.push_back(std::move(te));
    }
    if (s.temps.empty()) {
        s.hold.reset();
        CALIB_LOG_ERROR("setTempTable(): no valid entries after conversion");
        return false;
    }
    std::sort(s.temps.begin(), s.temps.end(),
              [](const TempTableEntryV4& a, const TempTableEntryV4& b) {
                  return a.temperature < b.temperature;
              });
    s.loaded = true;
    active_ = victim;
    tempSet_ = false;   // 新表须重设温度
    return true;
}

void LaserMatchScanSplitCpuV4::Impl::setCurrentTemperature(double temperature) {
    if (active_ < 0 || !slot_[active_].loaded || slot_[active_].temps.empty()) {
        throw std::runtime_error("[07-LaserMatchSplitCpuV4] temp table not loaded");
    }
    TubeSlot& s = slot_[active_];
    if (temperature < s.temps.front().temperature ||
        temperature > s.temps.back().temperature) {
        CALIB_LOG_WARN("setCurrentTemperature(): temp={} out of range [{}, {}], clamping",
                       temperature, s.temps.front().temperature,
                       s.temps.back().temperature);
    }
    auto it = std::lower_bound(s.temps.begin(), s.temps.end(), temperature,
        [](const TempTableEntryV4& e, double val) { return e.temperature < val; });
    if (it == s.temps.end()) --it;
    if (it != s.temps.begin()) {
        auto prev = std::prev(it);
        if (std::abs(prev->temperature - temperature) < std::abs(it->temperature - temperature))
            it = prev;
    }
    curTemp_ = it->temperature;
    tempSet_ = true;
    // 缓存命中（同源表＋同档温＋同 rowStep）跳过重排
    const bool cacheHit = (s.builtSrc != nullptr && s.hold && s.hold.get() == s.builtSrc
                           && it->temperature == s.builtTemp
                           && params_.epipolar_row_step == s.builtRowStep);
    if (!cacheHit) {
        buildActiveMap(s, *it);
        s.builtSrc = s.hold.get();
        s.builtTemp = it->temperature;
        s.builtRowStep = params_.epipolar_row_step;
    }
}

void LaserMatchScanSplitCpuV4::Impl::setStereoQ(const cv::Mat& Q) {
    hasQ_ = false;
    if (Q.empty() || Q.rows != 4 || Q.cols != 4 || Q.type() != CV_64F) return;
    q23_ = Q.at<double>(2, 3);   // f
    q32_ = Q.at<double>(3, 2);   // 1/b
    q33_ = Q.at<double>(3, 3);
    hasQ_ = true;
}

void LaserMatchScanSplitCpuV4::Impl::buildActiveMap(TubeSlot& s,
                                                    const TempTableEntryV4& entry) {
    s.mapCount = entry.entryCount;
    s.numLines = entry.numLines;
    const int N = s.mapCount;
    const float rowStep = params_.epipolar_row_step;
    const float* src = entry.mapData.data();

    // 行分桶（输入序）＋行内按 xL 排序（与 v3_cpu 宿主路径同比较器同序）
    std::vector<std::vector<int>> buckets(MAX_EPIPOLAR_ROWS);
    for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) buckets[r].reserve(64);
    for (int i = 0; i < N; ++i) {
        const float yL = src[static_cast<size_t>(i) * 4 + 1];
        int row = (int)std::round(yL / rowStep);
        row = clampRowV4(row);
        buckets[row].push_back(i);
    }
    s.ent.resize(static_cast<size_t>(N));
    s.rowStart.assign(MAX_EPIPOLAR_ROWS, 0);
    s.rowCount.assign(MAX_EPIPOLAR_ROWS, 0);
    s.rowX0.assign(MAX_EPIPOLAR_ROWS, 0.0f);
    int offset = 0;
    for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
        s.rowStart[r] = offset;
        s.rowCount[r] = static_cast<int>(buckets[r].size());
        if (!buckets[r].empty())
            s.rowX0[r] = src[static_cast<size_t>(buckets[r].front()) * 4];
        std::sort(buckets[r].begin(), buckets[r].end(), [&](int a, int b) {
            return src[static_cast<size_t>(a) * 4] < src[static_cast<size_t>(b) * 4];
        });
        for (int idx : buckets[r]) {
            s.ent[static_cast<size_t>(offset)].x = src[static_cast<size_t>(idx) * 4];
            s.ent[static_cast<size_t>(offset)].uR = src[static_cast<size_t>(idx) * 4 + 2];
            s.ent[static_cast<size_t>(offset)].lid =
                static_cast<int>(src[static_cast<size_t>(idx) * 4 + 3]);
            ++offset;
        }
    }

    // 桶索引池（K=1024，同 v3_cpu：行池段 4KB 驻 L1、桶内跳走 ~3 步）
    constexpr int kCpuBucketKV4 = 1024;
    s.bK = 0;
    s.bucketPool.clear();
    s.bucketRowOff.assign(MAX_EPIPOLAR_ROWS, 0);
    {
        float mnX = std::numeric_limits<float>::max(),
              mxX = -std::numeric_limits<float>::max();
        for (int i = 0; i < N; ++i) {
            const float x = src[static_cast<size_t>(i) * 4];
            if (x < mnX) mnX = x;
            if (x > mxX) mxX = x;
        }
        const float w = (mxX - mnX) / kCpuBucketKV4;
        if (w > 1e-6f) {
            s.bMinX = mnX;
            s.bInvW = 1.f / w;
            s.bK = kCpuBucketKV4;
            s.bucketPool.reserve(1024 * (kCpuBucketKV4 + 1));
            for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                if (s.rowCount[r] == 0) continue;
                s.bucketRowOff[r] = static_cast<int>(s.bucketPool.size());
                const int st = s.rowStart[r];
                const int cnt = s.rowCount[r];
                int p = 0;
                for (int b = 0; b <= kCpuBucketKV4; ++b) {
                    const float xb = mnX + b * w;
                    while (p < cnt && s.ent[static_cast<size_t>(st + p)].x < xb)
                        ++p;
                    s.bucketPool.push_back(p);
                }
            }
            if (s.bucketPool.empty()) s.bK = 0;
        }
    }
}

int LaserMatchScanSplitCpuV4::Impl::queryPoint(
    float xL, int row, int (&locLid)[V3_MAX_CARDS], float (&locU)[V3_MAX_CARDS])
{
    if (active_ < 0) return 0;
    if (row < 0 || row >= MAX_EPIPOLAR_ROWS) return 0;
    const TubeSlot& s = slot_[active_];
    const int start = s.rowStart[row];
    const int mapCnt = s.rowCount[row];
    if (mapCnt <= 0) return 0;
    const MapEntV4* ent = s.ent.data();
    const int bK = s.bK;
    const float vLtol = params_.vL_tolerance;

    // 首个 x≥xL 锚点（桶直达＋8 步跳走＋线性收尾；与 GPU 默认路径逐位一致——
    // 移植自 v3_cpu.cpp issueOnePoint 锚点段）
    int lo;
    if (bK > 0) {
        int b = static_cast<int>((xL - s.bMinX) * s.bInvW);
        if (b < 0) b = 0;
        if (b > bK - 1) b = bK - 1;
        const int poolBase = s.bucketRowOff[row];
        int j = start + s.bucketPool[static_cast<size_t>(poolBase + b)];
        const int jEnd = start + s.bucketPool[static_cast<size_t>(poolBase + b + 1)];
        while (j + 8 <= jEnd && ent[j + 7].x < xL) j += 8;
        while (j < jEnd && ent[j].x < xL) ++j;
        if (j >= start + mapCnt) j = start + mapCnt - 1;
        lo = j;
    } else {
        lo = start;
        int hi = start + mapCnt - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ent[mid].x < xL) lo = mid + 1; else hi = mid;
        }
    }

    // 双向展开：vL_tolerance 窗内每 lid 就近条目（收 lid/uR；locDx 仅内部去重用）
    int nc = 0;
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
                    if (adx < locDx[c]) { locDx[c] = adx; locU[c] = uR; }
                }
            }
            q += step;
        }
    }
    return nc;
}

LaserMatchScanCpuResultV4 LaserMatchScanSplitCpuV4::Impl::process(
    const LaserMatchSplitFrontV3& f)
{
    LaserMatchScanCpuResultV4 result;

    const auto tT0 = std::chrono::steady_clock::now();
    auto tMark = tT0;
    auto tSince = [&tMark]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tMark).count();
        tMark = now;
        return ms;
    };
    std::string bd;
    auto fmtBd = [&bd, &tSince](const char* tag) {
        bd += fmt::format(" {}={:.2f}", tag, tSince());
    };

    try {
        // ── 前段失败短路（透传）──
        if (!f.success) {
            result.success = false;
            result.message = "front failed: " + f.message;
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        const int leftCount = f.leftCount;
        const int rightCount = f.rightCount;
        result.totalLeftPoints = leftCount;
        result.totalRightPoints = rightCount;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "No points to match";
            result.matchedCount = 0;
            return result;
        }

        // ── 防御一致性（前段产物自洽）──
        if (static_cast<int>(f.cardPtr.size()) != leftCount + 1 ||
            static_cast<int>(f.segmap.size()) != leftCount ||
            static_cast<int>(f.leftPts.size()) != leftCount ||
            static_cast<int>(f.leftTracks.size()) != leftCount ||
            static_cast<int>(f.leftStatus.size()) != leftCount ||
            static_cast<int>(f.cardLid.size()) != f.totalCards ||
            static_cast<int>(f.cardRx.size()) != f.totalCards ||      // 【B1·Y6】gather 直读防御
            static_cast<int>(f.cardRidx.size()) != f.totalCards ||
            f.cardPtr[leftCount] != f.totalCards) {
            result.success = false;
            result.message = "front output inconsistent (sizes/CSR mismatch)";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        int segCount = f.segCount;   // 原值（后段无段上限；P6b 分裂可扩）
        if (segCount < 0) segCount = 0;
        if (f.segOverflow != 0)
            CALIB_LOG_WARN("process(): front segOverflow=1 (segCount={} >4096)；"
                           "后段无段上限，按 GPU 段表原值继续（对齐原算子宿主重做路径语义）",
                           segCount);

        const float rowStep = params_.epipolar_row_step;
        const cv::Point2f* lp = f.leftPts.data();
        const int* lt = f.leftTracks.data();

        // ── ptRow 复算＋状态计数（【B1 刀4】卡表直接消费前段 SoA，不落地 LmCard；
        //    【B1 刀3】leftStatus 不拷副本——result 直写 f）──
        {
            ptRow_.resize(static_cast<size_t>(leftCount));
            int noMapCnt = 0, noCardCnt = 0;
            for (int i = 0; i < leftCount; ++i) {
                ptRow_[static_cast<size_t>(i)] = froundAwayI(
                    lp[static_cast<size_t>(i)].y / rowStep);   // 无 clamp（同 GPU 口径）
                const int st = f.leftStatus[static_cast<size_t>(i)];
                if (st == -1) ++noMapCnt;
                else if (st == 0) ++noCardCnt;
            }
            segmap_.assign(f.segmap.begin(), f.segmap.end());
            // 折出统计早值（供末段使用）
            result.noMapLeftCount = noMapCnt;
            result.excludedLeftCount = noCardCnt;
        }
        fmtBd("ptRow");

        // ── 投票＋行位图＋段点集（单遍；段内点升序 i；ptRow 无 clamp 存原始值）──
        segVotes_.assign(static_cast<size_t>(segCount), std::vector<int>(V3_MAX_LIDS, 0));
        segRows_.assign(static_cast<size_t>(segCount), BitRows(kRowWords, 0ULL));
        segPts_.assign(static_cast<size_t>(segCount), std::vector<int>());
        for (int i = 0; i < leftCount; ++i) {
            const int row = ptRow_[static_cast<size_t>(i)];
            const int seg = segmap_[static_cast<size_t>(i)];
            if (seg >= 0 && seg < segCount) {
                segPts_[static_cast<size_t>(seg)].push_back(i);
                if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                    segRows_[static_cast<size_t>(seg)][static_cast<size_t>(row) / 64] |=
                        1ULL << (static_cast<size_t>(row) % 64);
            }
            const int begV = f.cardPtr[static_cast<size_t>(i)];
            const int ncV = f.cardPtr[static_cast<size_t>(i) + 1] - begV;
            const int* lidsV = f.cardLid.data() + begV;
            for (int c = 0; c < ncV; ++c) {
                const int lid = lidsV[c];
                if (seg >= 0 && seg < segCount && lid >= 1 && lid <= V3_MAX_LIDS)
                    ++segVotes_[static_cast<size_t>(seg)][lid - 1];
            }
        }
        fmtBd("vote");

        // ──【v4·B-pass2】通道开关与深度门（深度门无需温表；judge 类机制依赖温表就绪）──
        const bool v4Depth = v4params_.card_depth_gate && hasQ_;
        const bool v4Judge = (v4params_.quality_elect || v4params_.adapt_window ||
                              v4params_.abstain) && judgeReady();
        const bool v4On = v4Depth || v4Judge;
        int v4DepthDropped = 0, v4WindowDropped = 0, v4AdaptNoCard = 0;
        int v4Flip = 0, v4Abstain = 0, v4Veto = 0, v4SpacingMiss = 0, v4Contested = 0;
        if (v4On)
            cardMask_.assign(static_cast<size_t>(f.totalCards), 0);
        if (v4Depth) {
            const double zmin = static_cast<double>(v4params_.card_depth_min);
            const double zmax = static_cast<double>(v4params_.card_depth_max);
            for (int i = 0; i < leftCount; ++i) {
                const int begV = f.cardPtr[static_cast<size_t>(i)];
                const int ncV = f.cardPtr[static_cast<size_t>(i) + 1] - begV;
                const float xL = lp[static_cast<size_t>(i)].x;
                for (int c = 0; c < ncV; ++c) {
                    const float d = xL - f.cardRx[static_cast<size_t>(begV + c)];
                    if (d <= 0.0f) {
                        cardMask_[static_cast<size_t>(begV + c)] = 1;
                        ++v4DepthDropped;
                        continue;
                    }
                    const double Z = q23_ / (q32_ * static_cast<double>(d) + q33_);
                    if (Z < zmin || Z > zmax) {
                        cardMask_[static_cast<size_t>(begV + c)] = 1;
                        ++v4DepthDropped;
                    }
                }
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
        fmtBd("winner");

        if (params_.conflict_reelect) {
            // ── P6b 前置 modifier（segment_split；候选预筛同判据；excl 达标后才算）──
            int p6bCandidates = 0;
            for (int s = 0; s < segCount; ++s) {
                if (dropped_[static_cast<size_t>(s)] || winner_[static_cast<size_t>(s)] < 1) continue;
                const int W = winner_[static_cast<size_t>(s)];
                const int* v = segVotes_[static_cast<size_t>(s)].data();
                int sv = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    if (l + 1 != W && v[l] > sv) sv = v[l];
                if (sv >= std::max(v4params_.p6b_min_votes,
                                   (v[W - 1] * v4params_.p6b_vote_pct) / 100)) { ++p6bCandidates; break; }
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
                    if (v[S - 1] < std::max(v4params_.p6b_min_votes,
                                             (v[W - 1] * v4params_.p6b_vote_pct) / 100)) continue;   // 同门槛①
                    const int seg = s;
                    for (int i : segPts_[static_cast<size_t>(seg)]) {
                        const int Wc = winner_[static_cast<size_t>(seg)];
                        const int Sc = second_[static_cast<size_t>(seg)];
                        const int begE = f.cardPtr[static_cast<size_t>(i)];
                        const int ncE = f.cardPtr[static_cast<size_t>(i) + 1] - begE;
                        const int* lidsE = f.cardLid.data() + begE;
                        bool hasW = false, hasS = false;
                        for (int c = 0; c < ncE; ++c) {
                            const int lid = lidsE[c];
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
                        f.cardLid.data(), f.cardPtr.data(),
                        exclPtr,
                        segCount, segVotes_, segRows_, winner_, dropped_, ptRow_, segPts_,
                        v4params_.p6b_min_votes, v4params_.p6b_vote_pct);
                    if (n == 0) break;
                    exclPtr = nullptr;   // 后续轮卡数据随 f 生命周期（无需重扫 exclCnt）
                }
            }
            fmtBd("p6b");

            if (params_.cascade_elim) {
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
                CALIB_LOG_DEBUG("P7 cascade: segs={} determined={} merged={} banned={}",
                                segCount, determined, merged, banned);
                if (p7TimingOn()) {   // 【插桩】子相位串并入 lastBd_＋独立行
                    bd += p7BdSlot();
                    fprintf(stderr, "[v3:p7t] segs=%d%s\n", segCount,
                            p7BdSlot().c_str());
                }
            } else {
                // ── P5：冲突重选循环（≤10 轮）──
                int conflictsResolved = 0, reelected = 0, forcedDrop = 0;
                for (int iter = 0; iter < 10; ++iter)
                    if (!resolveConflictsOnce(segCount, segmap_, lp, leftCount,
                                              rowStep, segVotes_, segRows_,
                                              winner_, dropped_, &reelected, &forcedDrop))
                        break;
                    else ++conflictsResolved;
                CALIB_LOG_DEBUG("P5/P6b: segs={} conflictsResolved={} reelected={} forcedDrop={}",
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
                const int row = ptRow_[static_cast<size_t>(i)];
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
        fmtBd("conflict");

        // ──【v4·N 阶段】争议段裁判（D2 之后、收集之前——cascadeSolve 写回段会整体
        //    覆写 winner_/dropped_（split_cpu.cpp:896-913 同源），仲裁前置会被级联
        //    吞掉，故后置）──
        const auto tV4 = std::chrono::steady_clock::now();
        if (v4Judge) {
            const float mt = params_.match_threshold;
            int locLid[V3_MAX_CARDS];
            float locU[V3_MAX_CARDS];
            struct CardRes { int cardIdx; float res; };
            std::vector<CardRes> segCards;   // 复用缓冲（容量跨帧）
            std::vector<float> gapBuf;       // 复用缓冲
            struct SegStat {
                int seg, Wl, Sl; double W, S, T;
                float gapMed; bool gapValid;
                int cnt[V3_MAX_LIDS]; float sum[V3_MAX_LIDS];
            };
            std::vector<SegStat> contestedSegs;
            for (int s2 = 0; s2 < segCount; ++s2) {
                if (dropped_[static_cast<size_t>(s2)] != 0) continue;
                const int Wl = winner_[static_cast<size_t>(s2)];
                if (Wl < 1) continue;
                const std::vector<int>& v = segVotes_[static_cast<size_t>(s2)];
                double W = 0, S = 0, T = 0; int Sl = -1;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    T += v[l];
                    if (l + 1 == Wl) { W = v[l]; continue; }
                    if (v[l] > S) { S = v[l]; Sl = l + 1; }
                }
                // 争议判据（证伪史防护①：非争议段零漂移放行）。
                // share 分支带次名门槛（S≥(1−τ)T）：只抓「两线主导」真歧义——
                // 多线分流（第三名以下分票）不触发（perf：否则几乎全段判争议，
                // N 阶段查表把后段拖慢一个数量级，ABBA 实测在案）
                const bool shareAmb = (T > 0) &&
                    W < v4params_.abstain_share * T &&
                    S >= (1.0 - v4params_.abstain_share) * T;
                const bool safe = (S <= 0) ||
                    (W >= v4params_.abstain_margin * S && !shareAmb);
                if (safe) continue;
                // 争议段：单遍查表——gap 样本（winner vs 次名线预测间距）＋逐卡残差
                segCards.clear();
                gapBuf.clear();
                for (int i : segPts_[static_cast<size_t>(s2)]) {
                    const int row = ptRow_[static_cast<size_t>(i)];
                    const int nc = queryPoint(lp[static_cast<size_t>(i)].x, row,
                                              locLid, locU);
                    float pw = 0.f, psv = 0.f; bool hw = false, hsv = false;
                    for (int q2 = 0; q2 < nc; ++q2) {
                        if (locLid[q2] == Wl) { pw = locU[q2]; hw = true; }
                        else if (Sl > 0 && locLid[q2] == Sl) { psv = locU[q2]; hsv = true; }
                    }
                    if (Sl > 0) {
                        if (hw && hsv) gapBuf.push_back(std::fabs(pw - psv));
                        else ++v4SpacingMiss;
                    }
                    const int begC = f.cardPtr[static_cast<size_t>(i)];
                    const int ncC = f.cardPtr[static_cast<size_t>(i) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        const int lid = f.cardLid[static_cast<size_t>(begC + c)];
                        bool found = false;
                        float uRpred = 0.f;
                        for (int q2 = 0; q2 < nc; ++q2)
                            if (locLid[q2] == lid) { uRpred = locU[q2]; found = true; break; }
                        if (!found) { ++v4SpacingMiss; continue; }   // 防御：卡 lid 查不到
                        segCards.push_back({begC + c,
                            std::fabs(f.cardRx[static_cast<size_t>(begC + c)] - uRpred)});
                    }
                }
                SegStat st;
                st.seg = s2; st.Wl = Wl; st.Sl = Sl; st.W = W; st.S = S; st.T = T;
                st.gapValid = !gapBuf.empty();
                if (st.gapValid) {
                    std::nth_element(gapBuf.begin(),
                                     gapBuf.begin() + static_cast<int>(gapBuf.size()) / 2,
                                     gapBuf.end());
                    st.gapMed = gapBuf[gapBuf.size() / 2];
                } else {
                    st.gapMed = 0.f;
                }
                // 自适应窗：争议段内统一 w_eff＝夹 [floor, mt] 的 gap 中位一半
                if (v4params_.adapt_window && st.gapValid) {
                    const float wEff = std::max(v4params_.adapt_window_floor,
                                                std::min(mt, st.gapMed * 0.5f));
                    for (const auto& cr : segCards) {
                        if (!cardMask_[static_cast<size_t>(cr.cardIdx)] &&
                            cr.res > wEff) {
                            cardMask_[static_cast<size_t>(cr.cardIdx)] = 1;
                            ++v4WindowDropped;
                        }
                    }
                }
                for (int l = 0; l < V3_MAX_LIDS; ++l) { st.cnt[l] = 0; st.sum[l] = 0.f; }
                for (const auto& cr : segCards) {
                    if (cardMask_[static_cast<size_t>(cr.cardIdx)]) continue;   // 窗/深度 mask 均跳
                    const int l = f.cardLid[static_cast<size_t>(cr.cardIdx)] - 1;
                    if (l < 0 || l >= V3_MAX_LIDS) continue;
                    ++st.cnt[l];
                    st.sum[l] += cr.res;
                }
                contestedSegs.push_back(st);
            }
            v4Contested = static_cast<int>(contestedSegs.size());
            // 可选：帧内逐 lid meanRes 中位归一（quality_lid_norm；证伪史防护②——
            // 去线号系统偏差只改比较量纲，不动匹配本体）
            float lidMedian[V3_MAX_LIDS];
            if (v4params_.quality_lid_norm) {
                std::vector<float> perLid[V3_MAX_LIDS];
                for (const auto& st : contestedSegs)
                    for (int l = 0; l < V3_MAX_LIDS; ++l)
                        if (st.cnt[l] >= v4params_.quality_cnt_veto)
                            perLid[l].push_back(st.sum[l] / st.cnt[l]);
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    if (perLid[l].empty()) { lidMedian[l] = 0.f; continue; }
                    std::nth_element(perLid[l].begin(),
                                     perLid[l].begin() +
                                         static_cast<int>(perLid[l].size()) / 2,
                                     perLid[l].end());
                    lidMedian[l] = perLid[l][perLid[l].size() / 2];
                }
            } else {
                for (int l = 0; l < V3_MAX_LIDS; ++l) lidMedian[l] = 0.f;
            }
            // 逐段仲裁：质量改选（分离门防护③）→ 分不开则弃权（宁缺毋滥）
            for (auto& st : contestedSegs) {
                const int s2 = st.seg;
                if (v4params_.quality_elect) {
                    int bestLid = -1; double bestMean = 0.0;
                    int sndLid = -1; double sndMean = 0.0;
                    for (int l = 0; l < V3_MAX_LIDS; ++l) {
                        const int cq = st.cnt[l];
                        if (cq < 1) continue;
                        if (cq < v4params_.quality_cnt_veto) { ++v4Veto; continue; }
                        const double mean = st.sum[l] / cq - lidMedian[l];
                        if (bestLid < 0 || mean < bestMean) {
                            sndLid = bestLid; sndMean = bestMean;
                            bestLid = l + 1; bestMean = mean;
                        } else if (sndLid < 0 || mean < sndMean) {
                            sndLid = l + 1; sndMean = mean;
                        }
                    }
                    bool resolved = false;
                    if (bestLid >= 1) {
                        if (sndLid < 1 ||
                            (sndMean - bestMean) >
                                static_cast<double>(v4params_.quality_separate_px)) {
                            if (bestLid != st.Wl) {
                                winner_[static_cast<size_t>(s2)] = bestLid;
                                ++v4Flip;
                            }
                            resolved = true;
                        }
                    }
                    if (!resolved && v4params_.abstain) {
                        dropped_[static_cast<size_t>(s2)] = 1;
                        ++v4Abstain;
                    }
                } else if (v4params_.abstain) {
                    // 纯弃权模式（R2b 式双门）
                    const bool gapBad = st.gapValid &&
                        st.gapMed < v4params_.abstain_gap_factor * mt;
                    const bool marginBad = st.W < v4params_.abstain_margin * st.S;
                    const bool shareBad = (st.T > 0) &&
                        st.W < v4params_.abstain_share * st.T;
                    if ((marginBad && gapBad) || shareBad) {
                        dropped_[static_cast<size_t>(s2)] = 1;
                        ++v4Abstain;
                    }
                }
            }
        }

        // ── Step 8/9：收集（保留段 winner 卡；原序追加＝DeviceSelect 保序）──
        outLeft_.clear();
        outRight_.clear();
        outLid_.clear();
        outTrack_.clear();
        if (rightStatus_.size() < static_cast<size_t>(rightCount))
            rightStatus_.resize(static_cast<size_t>(rightCount));
        std::fill(rightStatus_.begin(), rightStatus_.begin() + rightCount, 0);
        int noWinnerCard = 0;
        int rightUsed = 0;
        for (int i = 0; i < leftCount; ++i) {
            const int seg = segmap_[static_cast<size_t>(i)];
            if (seg < 0 || seg >= segCount) continue;
            if (dropped_[static_cast<size_t>(seg)] != 0) continue;   // dropped segment
            const int w = winner_[static_cast<size_t>(seg)];
            if (w < 1) continue;
            const int begG = f.cardPtr[static_cast<size_t>(i)];
            const int ncG = f.cardPtr[static_cast<size_t>(i) + 1] - begG;
            const int* lidsG = f.cardLid.data() + begG;
            const float* rxsG = f.cardRx.data() + begG;
            const int* ridxsG = f.cardRidx.data() + begG;
            int wc = -1;
            for (int c = 0; c < ncG; ++c) {
                if (lidsG[c] == w) { wc = c; break; }
            }
            if (wc < 0) {
                ++noWinnerCard;                        // 保留段无 winner 卡（原 gather 静默丢弃）
                // ──【v4】v4 通道下无 winner 卡且存在被 mask 的卡 → 归因 v4 计数 ──
                if (v4On) {
                    for (int c = 0; c < ncG; ++c)
                        if (cardMask_[static_cast<size_t>(begG) + c]) {
                            ++v4AdaptNoCard;
                            break;
                        }
                }
                continue;
            }
            // ──【v4】winner 卡被 mask（窗/深度）→ 按无卡处理（惰性删除语义）──
            if (v4On && cardMask_[static_cast<size_t>(begG) + wc]) {
                ++noWinnerCard;
                ++v4AdaptNoCard;
                continue;
            }
            outLeft_.push_back(lp[static_cast<size_t>(i)]);
            outRight_.emplace_back(rxsG[wc], lp[static_cast<size_t>(i)].y);   // y＝左点 y（GPU 对齐）
            outLid_.push_back(w);
            outTrack_.push_back(lt[static_cast<size_t>(i)]);
            // ──【v4】输出对残差通道（res_channel 开且 judge 激活时；标定观测用，
            //    每输出点一次查表——默认关，防 gather 热路径退化）──
            if (v4Judge && v4params_.res_channel) {
                int lLid[V3_MAX_CARDS];
                float lU[V3_MAX_CARDS];
                const int nq = queryPoint(lp[static_cast<size_t>(i)].x,
                                          ptRow_[static_cast<size_t>(i)], lLid, lU);
                float rres = -1.0f;
                for (int q2 = 0; q2 < nq; ++q2)
                    if (lLid[q2] == w) {
                        // 【E2·2026-11-08】改带符号（原 fabs）：>0＝实际右点在表
                        // 预测右侧＝表预测偏低——供逐 lid 偏差方向判别（单测断言
                        // 侧自裹 fabs，口径兼容）
                        rres = rxsG[wc] - lU[q2];
                        break;
                    }
                resOut_.push_back(rres);
            }
            const int ridx = ridxsG[wc];
            if (rightStatus_[static_cast<size_t>(ridx)] == 0) {
                rightStatus_[static_cast<size_t>(ridx)] = 1;
                ++rightUsed;
            }
        }
        fmtBd("gather");

        // ── Step 10：统计（noCard/noMap 已折入组装；rightUsed 已折入收集）──
        int droppedSegs = 0, emptySegs = 0;
        for (int s = 0; s < segCount; ++s) {
            if (dropped_[static_cast<size_t>(s)] != 0) ++droppedSegs;
            if (winner_[static_cast<size_t>(s)] < 1) ++emptySegs;
        }
        const int noCard = result.excludedLeftCount;
        const int noMap = result.noMapLeftCount;

        result.matchedCount = static_cast<int>(outLeft_.size());
        result.excludedRightCount = rightCount - rightUsed;
        result.noWinnerCardCount = noWinnerCard;
        result.emptyWinnerSegCount = emptySegs;
        result.trackTotal = segCount;
        result.trackDropped = droppedSegs;

        // 输出 Mats（容量式：内存复用，值拷贝输出）
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
        result.left_status.assign(f.leftStatus.begin(), f.leftStatus.begin() + leftCount);   //【B1 刀3】直写 f（副本已删）
        result.right_status.assign(rightStatus_.begin(), rightStatus_.begin() + rightCount);
        fmtBd("output");

        // ──【v4】诊断计数与残差通道落盘（机制关闭时计数全零、通道为空）──
        result.qualityFlipCount = v4Flip;
        result.abstainSegCount = v4Abstain;
        result.windowShrinkDropped = v4WindowDropped;
        result.cardDepthDropped = v4DepthDropped;
        result.adaptNoCardPts = v4AdaptNoCard;
        result.spacingMissCount = v4SpacingMiss;
        result.singleVoteVetoCount = v4Veto;
        if (!resOut_.empty()) {
            std::vector<float> med(resOut_);
            std::nth_element(med.begin(),
                             med.begin() + static_cast<int>(med.size()) / 2, med.end());
            result.resMedianPx = med[med.size() / 2];
            result.matched_res = resOut_;   // 拷贝（resOut_ 容量跨帧复用）
        }
        resOut_.clear();
        if (v4On)
            bd += fmt::format(" v4judge={:.2f} contested={} flip={} abstain={}",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tV4).count(),
                v4Contested, v4Flip, v4Abstain);
        // 诊断注记：v4 通道活跃时把关键计数并入 message（[fc:bd] 与日志可读）
        if (v4Judge)
            result.message += fmt::format(
                " | v4: contested={} flip={} abstain={} winDrop={} noCardPts={} "
                "spacingMiss={}",
                v4Contested, v4Flip, v4Abstain, v4WindowDropped, v4AdaptNoCard,
                v4SpacingMiss);

        bd += fmt::format(" total={:.2f} N={}",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tT0).count(), leftCount);
        lastBd_ = bd;

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
        lastBd_ = bd;
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        lastBd_ = bd;
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    return result;
}

// ============================================================================
// 公开接口桥接
// ============================================================================

LaserMatchScanSplitCpuV4::LaserMatchScanSplitCpuV4(const LaserMatchScanParamsV4& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserMatchScanSplitCpuV4 initialized: threshold={} "
                   "quality={} adaptwin={} abstain={}",
                   params.v3.match_threshold, params.quality_elect,
                   params.adapt_window, params.abstain);
}

LaserMatchScanSplitCpuV4::~LaserMatchScanSplitCpuV4() = default;

bool LaserMatchScanSplitCpuV4::SetTempTable(
    std::shared_ptr<const LaserPlaneMapTempTable> table)
{
    return pImpl_ ? pImpl_->setTempTable(std::move(table)) : false;
}

void LaserMatchScanSplitCpuV4::SetCurrentTemperature(double temperature) {
    if (pImpl_) pImpl_->setCurrentTemperature(temperature);
}

void LaserMatchScanSplitCpuV4::SetStereoQ(const cv::Mat& Q) {
    if (pImpl_) pImpl_->setStereoQ(Q);
}

LaserMatchScanCpuResultV4 LaserMatchScanSplitCpuV4::Execute(
    const LaserMatchSplitFrontV3& front)
{
    return pImpl_ ? pImpl_->process(front) : [] {
        LaserMatchScanCpuResultV4 r;
        r.success = false;
        r.message = "Execute() after Destroy()";
        return r;
    }();
}

void LaserMatchScanSplitCpuV4::Destroy() {
    pImpl_.reset();
}

void LaserMatchScanSplitCpuV4::SetParams(const LaserMatchScanParamsV4& params) {
    if (pImpl_) {
        pImpl_->v4params_ = params;
        pImpl_->v4params_.validate();
    }
}

const LaserMatchScanParamsV4& LaserMatchScanSplitCpuV4::GetParams() const {
    if (!pImpl_)
        throw std::runtime_error("[07-LaserMatchSplitCpuV4] GetParams() after Destroy()");
    return pImpl_->v4params_;
}

const std::string& LaserMatchScanSplitCpuV4::LastTimingBreakdown() const {
    static const std::string kEmpty;
    return pImpl_ ? pImpl_->lastBd_ : kEmpty;
}

} // namespace calib
