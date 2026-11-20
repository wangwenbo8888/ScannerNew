/**
 * @file laser_match_scan_v5.cpp
 * @brief 激光线匹配扫描 V5 — 人工 8 步骨架 CPU 后段实现（纯 CPU 无 CUDA）
 *
 * 2026-11-11 依设计方案 v2.1（专题-激光匹配算子laser_match_scan_v5-设计方案-
 * 2026-11-11.md §五）实现。骨架＝人工 8 步原文一字不动；补丁①~⑫以标注内联。
 * 撤销清单（v2.1）：无 P5/P6b/P7、无友军合并、无收尾终检（补丁②后消元结构上
 * 防撞号）、无 lid_norm（绝对值聚合减带符号偏差会减出负数）、无误差绝对帽
 * （发卡窗已保证恒不触发）。
 *
 * 实现口径备忘：
 *   - 确定性：所有并列取"线号小 → 段号小"（金样对拍要求同输入逐位同输出）
 *   - A 不随消元变化（分子分母都不动）；级联动力全在 B（消元只删计票板＋
 *     清备选位——补丁②）
 *   - 票数＝持该线（未掩蔽）卡的点数（每点每线至多一卡——前提见设计方案 §三）
 *   - 第7步票数因子：线性 c1/c2（人工原式）或对数 ln(c1)/ln(c2)（补丁⑨，
 *     c2=1 → ∞＝对手单票自动出局；c1=1 且 log 档 → ln(1)=0 → E=0 恒不过门）
 *   - 深度门取 Q 式（设计方案记 ZG 廉价式 ρ=P/(disp−D)，K2 无该组标定常数；
 *     Q 在 stereoRectifyTempTable 现成，一次除法同价——数学等价，头注已记）
 *   - 残差清洗（补丁⑪）中心＝带符号残差中位，自适应门＝内点散布×1.5×基准
 *     钳 [0.4,1.2]（无内点回退 0.8×基准）——ZG LineFilter 短路径同构；
 *     n<5 跳过（统计不成立）
 *
 * ⚠ 沙箱定位：TEST/fullgpu_optimize 实验算子，不入 mod_operatorlib，不入产线。
 */

#include "laser_match_scan_v5.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace calib {

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanSplitCpuV5);

namespace {

// 常量（同值拷贝自 laser_match_scan_v4.cpp——勿漂移）
constexpr int MAX_EPIPOLAR_ROWS = 8192;
constexpr int V3_MAX_LIDS = 32;      // lid 桶距（lid 1..32；产品 1..25）
constexpr int kRowWords = MAX_EPIPOLAR_ROWS / 64;   // 128（段行位图字数）
constexpr int kCardCap = 32;         // 每点卡上限（每 lid 就近单卡）

// roundf 半值远离零语义的 MSVC 内联等价（|x|<2^22 按位一致；同值拷贝——勿漂移）
inline int froundAwayI(float x) { return static_cast<int>(x >= 0.0f ? x + 0.5f : x - 0.5f); }

using BitRows = std::vector<unsigned long long>;    // kRowWords 位图

// 行位图判交（首个公共字即返回——段数溢出病理下的早退保险）
inline bool rowsIntersect(const BitRows& a, const BitRows& b) {
    for (int w = 0; w < kRowWords; ++w)
        if (a[static_cast<size_t>(w)] & b[static_cast<size_t>(w)]) return true;
    return false;
}

// 弃段原因码（诊断计数口径）
enum DropWhy : int {
    kDropNone = 0,
    kDropNoCard = 1,       // 补丁①：整段无卡（A 分母为 0）
    kDropNoSupport = 2,    // D1：候选集空（设备误差大信号）
    kDropElimAll = 3,      // 票被消元删光（线号已被邻居合法占用）
    kDropExit = 4,         // 兜底弃（第7步不入口或 E/分离门不达标）
};

} // namespace

// ============================================================================
// Impl
// ============================================================================

struct LaserMatchScanSplitCpuV5::Impl {
    LaserMatchScanParamsV5 params_;

    // 深度门 Q（补丁⑩；未注入/格式不符 → 自动关）
    bool hasQ_ = false;
    double q23_ = 0.0, q32_ = 0.0, q33_ = 0.0;

    // ── 三件套＋V5 附加（容量跨帧复用；P3 速度优先）──
    std::vector<int> ptRow_;                 // N：行号复算（同 GPU 口径不 clamp）
    std::vector<int> votes_;                 // segCount*V3_MAX_LIDS 扁平计票板
    std::vector<BitRows> segRows_;           // segCount：行位图（消元判交）
    std::vector<std::vector<int>> segPts_;   // seg → 点下标列表
    std::vector<int> segCarded_;             // seg：持有（未掩蔽）卡的点数＝A 分母（补丁①）
    std::vector<uint32_t> candMask_;         // seg：备选位图（bit=lid-1；补丁②＝A过门∧未消元）
    std::vector<unsigned char> cardMask_;    // totalCards：深度门卡掩蔽（补丁⑩）
    std::vector<int> winner_;                // seg：判定线号（-1=未定）
    std::vector<int> dropWhy_;               // seg：DropWhy
    std::vector<unsigned char> cleanSkip_;   // N：残差清洗点级跳过（补丁⑪）

    // 输出缓冲（容量复用）
    std::vector<cv::Point2f> outLeft_, outRight_;
    std::vector<int> outLid_, outTrack_;
    std::vector<float> resOut_;
    std::vector<int> rightStatus_;
    std::vector<float> medBuf_;
    cv::Mat outLeftM_, outRightM_, outLidM_, outTrackM_;

    std::string lastBd_;

    // ── 消元（第5步＋补丁②联动）：定下 lid 的段与 fromSeg 行位图有交集的
    //    未定段：删该 lid 选票＋清备选位。A 不动（静态资格）。──
    void eliminate(int fromSeg, int lid, int segCount) {
        const uint32_t bit = 1u << (lid - 1);
        const BitRows& rows = segRows_[static_cast<size_t>(fromSeg)];
        for (int t = 0; t < segCount; ++t) {
            if (t == fromSeg) continue;
            if (winner_[static_cast<size_t>(t)] >= 1) continue;
            if (dropWhy_[static_cast<size_t>(t)] != kDropNone) continue;
            if ((candMask_[static_cast<size_t>(t)] & bit) == 0 &&
                votes_[static_cast<size_t>(t) * V3_MAX_LIDS + (lid - 1)] == 0) continue;
            if (!rowsIntersect(rows, segRows_[static_cast<size_t>(t)])) continue;
            votes_[static_cast<size_t>(t) * V3_MAX_LIDS + (lid - 1)] = 0;
            candMask_[static_cast<size_t>(t)] &= ~bit;
        }
    }

    // ── 第3~6步：级联主循环（补丁③④）。返回本轮是否有定段（false＝无可定）。──
    // 每轮全量扫描未定未弃段取 B 最大者；典型段数（几十~二百）下
    // U×32×U 整数量级微秒级（P3 简化优先，不做脏集优化；段溢出病理另议）。
    bool cascadeRound(int segCount, int& determinedTotal) {
        const LaserMatchScanParamsV5& vp = params_;
        bool any = false;
        for (;;) {
            int bestSeg = -1;
            double bestB = -1.0;
            int bestLid = -1;
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] >= 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int* v = &votes_[static_cast<size_t>(s) * V3_MAX_LIDS];
                // 第一名（平票取小 lid——升序严格 > 天然保证）＋全量第二名
                int wLid = -1, w = 0, sLid = -1, sCnt = 0, total = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    total += v[l];
                    if (v[l] > w) { sCnt = w; sLid = wLid; w = v[l]; wLid = l + 1; }
                    else if (v[l] > sCnt) { sCnt = v[l]; sLid = l + 1; }
                }
                if (total == 0) {   // 票被消元删光（防御：消元处不逐段查和）
                    dropWhy_[static_cast<size_t>(s)] = kDropElimAll;
                    continue;
                }
                if (wLid < 1) continue;
                // 补丁③(a)：第一名须在备选中（不在→本轮不定，等消元）
                if (((candMask_[static_cast<size_t>(s)] >> (wLid - 1)) & 1u) == 0)
                    continue;
                // 补丁③(b)：票数下限（有第二名→votes_floor；全票→solo 下限）
                const int floor = (sCnt > 0) ? vp.votes_floor : vp.votes_floor_solo;
                if (w < floor) continue;
                // B 门（第二名不存在 → ∞ 恒过）
                const double B = (sCnt > 0) ? static_cast<double>(w) / sCnt
                                            : std::numeric_limits<double>::infinity();
                if (B < vp.margin_min) continue;
                // 取 B 最大（并列→线号小→段号小——确定性）
                if (bestSeg < 0 || B > bestB ||
                    (B == bestB && (wLid < bestLid))) {
                    bestSeg = s; bestB = B; bestLid = wLid;
                }
            }
            if (bestSeg < 0) break;
            winner_[static_cast<size_t>(bestSeg)] = bestLid;
            ++determinedTotal;
            eliminate(bestSeg, bestLid, segCount);
            any = true;
        }
        return any;
    }

    LaserMatchScanCpuResultV5 process(const LaserMatchSplitFrontV3& f);
};

// ============================================================================
// process — 8 步骨架主流程
// ============================================================================

LaserMatchScanCpuResultV5 LaserMatchScanSplitCpuV5::Impl::process(
    const LaserMatchSplitFrontV3& f)
{
    LaserMatchScanCpuResultV5 result;

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

        // ── 防御一致性（前段产物自洽；镜像 V4 具名字段口径）──
        if (static_cast<int>(f.cardPtr.size()) != leftCount + 1 ||
            static_cast<int>(f.segmap.size()) != leftCount ||
            static_cast<int>(f.leftPts.size()) != leftCount ||
            static_cast<int>(f.leftTracks.size()) != leftCount ||
            static_cast<int>(f.leftStatus.size()) != leftCount ||
            static_cast<int>(f.cardLid.size()) != f.totalCards ||
            static_cast<int>(f.cardRx.size()) != f.totalCards ||
            static_cast<int>(f.cardRidx.size()) != f.totalCards ||
            f.cardPtr[leftCount] != f.totalCards) {
            result.success = false;
            result.message = "front output inconsistent (sizes/CSR mismatch)";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }
        // 【V5 契约】cardRes 必须在位（骨架第0步核心输入）：空＝前段发射关＝
        // 配置矛盾（fail-safe——见 v2.1 §七 空态契约）
        if (static_cast<int>(f.cardRes.size()) != f.totalCards) {
            result.success = false;
            result.message = "front.cardRes missing/mismatch: V5 requires cardRes "
                             "emission (JMW_V3CARDRES=1 or SetEmitCardRes(true))";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        int segCount = f.segCount;
        if (segCount < 0) segCount = 0;
        if (f.segOverflow != 0)
            CALIB_LOG_WARN("process(): front segOverflow=1 (segCount={}); "
                           "后段无段上限，按段表原值继续", segCount);

        const LaserMatchScanParamsV5& vp = params_;
        const float rowStep = vp.v3.epipolar_row_step;
        const cv::Point2f* lp = f.leftPts.data();
        const int* lt = f.leftTracks.data();
        const float* cres = f.cardRes.data();

        // ── 第0步（备料）：行号复算＋状态清点（同 V4 口径；不 clamp）──
        ptRow_.resize(static_cast<size_t>(leftCount));
        int noMapCnt = 0, noCardCnt = 0;
        for (int i = 0; i < leftCount; ++i) {
            ptRow_[static_cast<size_t>(i)] = froundAwayI(
                lp[static_cast<size_t>(i)].y / rowStep);
            const int st = f.leftStatus[static_cast<size_t>(i)];
            if (st == -1) ++noMapCnt;
            else if (st == 0) ++noCardCnt;
        }
        result.noMapLeftCount = noMapCnt;
        result.excludedLeftCount = noCardCnt;
        fmtBd("ptRow");

        // ── 第0步（补丁⑩）：卡片级深度门（Q 式；机制关或无 Q → 全通过）──
        const bool v4Depth = vp.depth_gate && hasQ_;
        int v4DepthDropped = 0;
        if (v4Depth) {
            const double zmin = static_cast<double>(vp.depth_min);
            const double zmax = static_cast<double>(vp.depth_max);
            cardMask_.assign(static_cast<size_t>(f.totalCards), 0);
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
        } else {
            cardMask_.clear();
        }
        result.depthDroppedCards = v4DepthDropped;

        // ── 第1/2步：计票＋行位图＋点名单＋支持度 A＋备选（补丁①②）──
        votes_.assign(static_cast<size_t>(segCount) * V3_MAX_LIDS, 0);
        segRows_.assign(static_cast<size_t>(segCount), BitRows(kRowWords, 0ULL));
        segPts_.assign(static_cast<size_t>(segCount), std::vector<int>());
        segCarded_.assign(static_cast<size_t>(segCount), 0);
        candMask_.assign(static_cast<size_t>(segCount), 0u);
        winner_.assign(static_cast<size_t>(segCount), -1);
        dropWhy_.assign(static_cast<size_t>(segCount), kDropNone);
        for (int i = 0; i < leftCount; ++i) {
            const int seg = f.segmap[static_cast<size_t>(i)];
            if (seg < 0 || seg >= segCount) continue;
            const int begV = f.cardPtr[static_cast<size_t>(i)];
            const int ncV = f.cardPtr[static_cast<size_t>(i) + 1] - begV;
            // 本点未掩蔽卡（投票资格）；全掩蔽/无卡＝A 分母不计（补丁①）
            int live = 0;
            for (int c = 0; c < ncV; ++c) {
                if (v4Depth && cardMask_[static_cast<size_t>(begV + c)]) continue;
                const int lid = f.cardLid[static_cast<size_t>(begV + c)];
                if (lid >= 1 && lid <= V3_MAX_LIDS) {
                    ++votes_[static_cast<size_t>(seg) * V3_MAX_LIDS + (lid - 1)];
                    ++live;
                }
            }
            if (live == 0) continue;
            ++segCarded_[static_cast<size_t>(seg)];
            segPts_[static_cast<size_t>(seg)].push_back(i);
            const int row = ptRow_[static_cast<size_t>(i)];
            if (row >= 0 && row < MAX_EPIPOLAR_ROWS)
                segRows_[static_cast<size_t>(seg)][static_cast<size_t>(row) / 64] |=
                    1ULL << (static_cast<size_t>(row) % 64);
        }
        // 备选位图＝A>share_fast（严格大于——人工原文"大于"）；空候选集段弃（D1）
        int noCardSeg = 0, noSupSeg = 0;
        for (int s = 0; s < segCount; ++s) {
            const int carded = segCarded_[static_cast<size_t>(s)];
            if (carded == 0) {
                dropWhy_[static_cast<size_t>(s)] = kDropNoCard;
                ++noCardSeg;
                continue;
            }
            const int* v = &votes_[static_cast<size_t>(s) * V3_MAX_LIDS];
            uint32_t mask = 0u;
            for (int l = 0; l < V3_MAX_LIDS; ++l)
                if (static_cast<float>(v[l]) > vp.share_fast * static_cast<float>(carded))
                    mask |= (1u << l);
            candMask_[static_cast<size_t>(s)] = mask;
            if (mask == 0u) {
                dropWhy_[static_cast<size_t>(s)] = kDropNoSupport;
                ++noSupSeg;
            }
        }
        result.noCardSegCount = noCardSeg;
        result.noSupportSegCount = noSupSeg;
        if (segCount > 0 &&
            static_cast<float>(noSupSeg) >
                vp.no_support_warn_ratio * static_cast<float>(segCount)) {
            result.noSupportWarned = true;
            CALIB_LOG_WARN("process(): no-support segments {}/{} (> {}%) —— "
                           "设备误差偏大，建议重新校准（逐线号残差中位见 res 输出口径）",
                           noSupSeg, segCount,
                           static_cast<int>(vp.no_support_warn_ratio * 100.0f + 0.5f));
        }
        fmtBd("vote");

        // ── 第3~8步：级联 ⇄ 质量裁决回环 ──
        // 外层＝第8步（判后回 5/3/4/5/6）：级联跑尽 → 第7步挑一个最优裁决段
        //（补丁⑦：E 大者先）→ 判定 → 再级联；两级都无段可动 → 出循环。
        int determinedTotal = 0;
        int judged = 0, flipped = 0, elimAll = 0;
        const auto countElimAll = [&]() {
            int n = 0;
            for (int s = 0; s < segCount; ++s)
                if (dropWhy_[static_cast<size_t>(s)] == kDropElimAll) ++n;
            return n;
        };
        for (;;) {
            cascadeRound(segCount, determinedTotal);
            fmtBd("cascade");

            // ── 第7步：挑当前最优裁决段（补丁⑤入口/⑥分离门/⑨票数因子）──
            int bestSeg = -1;
            double bestE = -1.0;
            int bestLid = -1, bestVoteLid = -1;
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] >= 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int* v = &votes_[static_cast<size_t>(s) * V3_MAX_LIDS];
                int wLid = -1, w = 0, sLid = -1, sCnt = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    if (v[l] > w) { sCnt = w; sLid = wLid; w = v[l]; wLid = l + 1; }
                    else if (v[l] > sCnt) { sCnt = v[l]; sLid = l + 1; }
                }
                // 补丁⑤入口三条件：B 有限（有第二名）／可比编号≥2／第一名票数下限
                if (sCnt <= 0) continue;                 // B=∞ 段禁入（防洞②复活）
                if (w < vp.votes_floor) continue;
                const uint32_t compMask = candMask_[static_cast<size_t>(s)] &
                                          [&]{ uint32_t m = 0u;
                                               for (int l = 0; l < V3_MAX_LIDS; ++l)
                                                   if (v[l] > 0) m |= (1u << l);
                                               return m; }();
                int compCnt = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    if (compMask & (1u << l)) ++compCnt;
                if (compCnt < 2) continue;               // D 需两个操作数
                // 逐可比线号聚合平均 |cardRes|（第7步原文；掩蔽卡不进聚合——补丁⑩）
                float sumRes[V3_MAX_LIDS];
                int cntRes[V3_MAX_LIDS];
                for (int l = 0; l < V3_MAX_LIDS; ++l) { sumRes[l] = 0.0f; cntRes[l] = 0; }
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        const int lid = f.cardLid[static_cast<size_t>(begC + c)];
                        if (lid < 1 || lid > V3_MAX_LIDS) continue;
                        if ((compMask & (1u << (lid - 1))) == 0) continue;
                        sumRes[lid - 1] += std::fabs(cres[static_cast<size_t>(begC + c)]);
                        ++cntRes[lid - 1];
                    }
                }
                // 最小/次小平均误差（升序严格 < 天然平票取小 lid）
                int l1 = -1, l2 = -1;
                double m1 = 0.0, m2 = 0.0;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    if ((compMask & (1u << l)) == 0 || cntRes[l] < 1) continue;
                    const double m = static_cast<double>(sumRes[l]) / cntRes[l];
                    if (l1 < 0 || m < m1) { l2 = l1; m2 = m1; l1 = l + 1; m1 = m; }
                    else if (l2 < 0 || m < m2) { l2 = l + 1; m2 = m; }
                }
                if (l1 < 1 || l2 < 1) continue;          // 退化：只剩一个可比线号
                // D＝次小/最小（m1≤0 处理：双零跳过；m1=0 而 m2>0 → D=∞）
                double D;
                if (m1 <= 0.0 && m2 <= 0.0) continue;
                D = (m1 <= 0.0) ? std::numeric_limits<double>::infinity() : m2 / m1;
                // 票数因子（补丁⑨）：线性 c1/c2（人工原式）｜对数 ln(c1)/ln(c2)
                double Bfac;
                if (vp.vote_weight_log) {
                    if (sCnt == 1) Bfac = std::numeric_limits<double>::infinity();
                    else if (w <= 1) continue;           // ln(1)=0 → E=0 恒不过
                    else Bfac = std::log(static_cast<double>(w)) /
                                std::log(static_cast<double>(sCnt));
                } else {
                    Bfac = static_cast<double>(w) / sCnt;
                }
                const double E = D * Bfac;
                const double sep = m2 - m1;              // 绝对分离量（补丁⑥）
                if (E < vp.quality_e_min) continue;
                if (sep < vp.quality_sep_px) continue;
                if (bestSeg < 0 || E > bestE ||
                    (E == bestE && l1 < bestLid)) {      // 补丁⑦：E 大→线号小→段号小
                    bestSeg = s; bestE = E; bestLid = l1; bestVoteLid = wLid;
                }
            }
            if (bestSeg < 0) break;                      // 两级均无段可动 → 出回环
            // 判定为最小误差线号（≠投票第一名＝翻案——只改结论不改票数）
            winner_[static_cast<size_t>(bestSeg)] = bestLid;
            ++determinedTotal;
            ++judged;
            if (bestLid != bestVoteLid) ++flipped;
            eliminate(bestSeg, bestLid, segCount);       // 第5步：消元后回环
        }
        // 兜底出口（补丁④/⑤）：仍定不了的段全部弃掉＋计数（循环必终止的保证）
        int exitDrop = 0;
        for (int s = 0; s < segCount; ++s) {
            if (winner_[static_cast<size_t>(s)] < 1 &&
                dropWhy_[static_cast<size_t>(s)] == kDropNone) {
                dropWhy_[static_cast<size_t>(s)] = kDropExit;
                ++exitDrop;
            }
        }
        elimAll = countElimAll();
        result.qualityJudgedCount = judged;
        result.qualityFlipCount = flipped;
        result.eliminatedAllVotesSegCount = elimAll;
        result.exitDroppedSegCount = exitDrop;
        fmtBd("judge");

        // ── 收尾·补丁⑪：按段残差众数清洗（机制关跳过；n<5 跳过）──
        cleanSkip_.assign(static_cast<size_t>(leftCount), 0);
        int cleanDropped = 0;
        if (vp.res_clean) {
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] < 1) continue;
                const int wl = winner_[static_cast<size_t>(s)];
                // 收集本段赢家线号卡的带符号残差（未掩蔽）
                medBuf_.clear();
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (f.cardLid[static_cast<size_t>(begC + c)] != wl) continue;
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        medBuf_.push_back(cres[static_cast<size_t>(begC + c)]);
                        break;   // 每 lid 至多一卡
                    }
                }
                const int n = static_cast<int>(medBuf_.size());
                if (n < 5) continue;   // 统计不成立
                // 中心＝带符号残差中位（短段众数的稳健近似；ZG 短路径同构）
                std::vector<float> med(medBuf_);
                std::nth_element(med.begin(), med.begin() + n / 2, med.end());
                const double center = med[static_cast<size_t>(n / 2)];
                // 自适应门：内点(dev<1.0)散布×1.5×基准，钳 [0.4,1.2]；无内点 0.8×基准
                double acc = 0.0; int inl = 0;
                std::vector<double> dev(static_cast<size_t>(n));
                for (int k = 0; k < n; ++k) {
                    dev[static_cast<size_t>(k)] =
                        std::fabs(static_cast<double>(medBuf_[static_cast<size_t>(k)]) - center);
                    if (dev[static_cast<size_t>(k)] < 1.0) {
                        acc += dev[static_cast<size_t>(k)] * dev[static_cast<size_t>(k)];
                        ++inl;
                    }
                }
                double gate = (inl > 0)
                    ? std::sqrt(acc / inl) * 1.5 * vp.res_clean_thresh
                    : 0.8 * vp.res_clean_thresh;
                gate = std::max(0.4, std::min(1.2, gate));
                // 超门点打跳过标记（不搬数据——惰性，gather 处消费）
                int k = 0;
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (f.cardLid[static_cast<size_t>(begC + c)] != wl) continue;
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        if (dev[static_cast<size_t>(k)] > gate) {
                            cleanSkip_[static_cast<size_t>(pi)] = 1;
                            ++cleanDropped;
                        }
                        ++k;
                        break;
                    }
                }
            }
        }
        result.resCleanDropped = cleanDropped;
        fmtBd("clean");

        // ── 收集（第7步后保留段取赢家卡；V4 同语义）＋补丁⑫残差输出 ──
        outLeft_.clear(); outRight_.clear(); outLid_.clear(); outTrack_.clear();
        resOut_.clear();
        if (rightStatus_.size() < static_cast<size_t>(rightCount))
            rightStatus_.resize(static_cast<size_t>(rightCount));
        std::fill(rightStatus_.begin(), rightStatus_.begin() + rightCount, 0);
        int noWinnerCard = 0;
        int rightUsed = 0;
        for (int i = 0; i < leftCount; ++i) {
            const int seg = f.segmap[static_cast<size_t>(i)];
            if (seg < 0 || seg >= segCount) continue;
            if (winner_[static_cast<size_t>(seg)] < 1) continue;
            if (dropWhy_[static_cast<size_t>(seg)] != kDropNone) continue;
            const int w = winner_[static_cast<size_t>(seg)];
            if (cleanSkip_[static_cast<size_t>(i)]) continue;   // 补丁⑪（已计数）
            const int begG = f.cardPtr[static_cast<size_t>(i)];
            const int ncG = f.cardPtr[static_cast<size_t>(i) + 1] - begG;
            int wc = -1;
            for (int c = 0; c < ncG; ++c) {
                if (v4Depth && cardMask_[static_cast<size_t>(begG + c)]) continue;
                if (f.cardLid[static_cast<size_t>(begG + c)] == w) { wc = c; break; }
            }
            if (wc < 0) { ++noWinnerCard; continue; }    // 保留段无赢家卡（V4 静默口径）
            outLeft_.push_back(lp[static_cast<size_t>(i)]);
            outRight_.emplace_back(f.cardRx[static_cast<size_t>(begG + wc)],
                                   lp[static_cast<size_t>(i)].y);   // y＝左点 y（GPU 对齐）
            outLid_.push_back(w);
            outTrack_.push_back(lt[static_cast<size_t>(i)]);
            if (vp.res_output)   // 补丁⑫：带符号残差（>0＝表预测偏低）
                resOut_.push_back(cres[static_cast<size_t>(begG + wc)]);
            const int ridx = f.cardRidx[static_cast<size_t>(begG + wc)];
            if (ridx >= 0 && ridx < rightCount &&
                rightStatus_[static_cast<size_t>(ridx)] == 0) {
                rightStatus_[static_cast<size_t>(ridx)] = 1;
                ++rightUsed;
            }
        }
        fmtBd("gather");

        // ── 统计（第9步）──
        int droppedSegs = 0;
        for (int s = 0; s < segCount; ++s)
            if (dropWhy_[static_cast<size_t>(s)] != kDropNone) ++droppedSegs;

        result.matchedCount = static_cast<int>(outLeft_.size());
        result.excludedRightCount = rightCount - rightUsed;
        result.noWinnerCardCount = noWinnerCard;
        result.emptyWinnerSegCount = 0;   // V5 构造性为 0（定段必有赢家卡线号）
        result.trackTotal = segCount;
        result.trackDropped = droppedSegs;

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
        result.left_status.assign(f.leftStatus.begin(), f.leftStatus.begin() + leftCount);
        result.right_status.assign(rightStatus_.begin(), rightStatus_.begin() + rightCount);
        if (vp.res_output && !resOut_.empty()) {
            std::vector<float> med(resOut_);
            std::nth_element(med.begin(), med.begin() + static_cast<int>(med.size()) / 2,
                             med.end());
            result.resMedianPx = med[med.size() / 2];
            result.matched_res = resOut_;
        }
        fmtBd("output");

        result.success = true;
        result.message = fmt::format(
            "OK: {} matched, {}/{} segs kept | v5: noCard={} noSup={}{} elimAll={} "
            "exitDrop={} judge={} flip={} depthDropped={} cleanDropped={}",
            n, segCount - droppedSegs, segCount,
            result.noCardSegCount, result.noSupportSegCount,
            result.noSupportWarned ? "(WARN)" : "",
            result.eliminatedAllVotesSegCount, result.exitDroppedSegCount,
            result.qualityJudgedCount, result.qualityFlipCount,
            result.depthDroppedCards, result.resCleanDropped);
        bd += fmt::format(" total={:.2f} N={}",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tT0).count(), leftCount);
        lastBd_ = bd;
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

LaserMatchScanSplitCpuV5::LaserMatchScanSplitCpuV5(const LaserMatchScanParamsV5& params)
    : pImpl_(std::make_unique<Impl>())
{
    params.validate();
    pImpl_->params_ = params;
}

LaserMatchScanSplitCpuV5::~LaserMatchScanSplitCpuV5() = default;

void LaserMatchScanSplitCpuV5::SetStereoQ(const cv::Mat& Q) {
    if (Q.empty() || Q.rows != 4 || Q.cols != 4 || Q.type() != CV_64F) {
        pImpl_->hasQ_ = false;
        return;
    }
    pImpl_->q23_ = Q.at<double>(2, 3);
    pImpl_->q32_ = Q.at<double>(3, 2);
    pImpl_->q33_ = Q.at<double>(3, 3);
    pImpl_->hasQ_ = true;
}

LaserMatchScanCpuResultV5 LaserMatchScanSplitCpuV5::Execute(
    const LaserMatchSplitFrontV3& front)
{
    if (!pImpl_) {
        LaserMatchScanCpuResultV5 r;
        r.success = false;
        r.message = "Execute() after Destroy()";
        return r;
    }
    return pImpl_->process(front);
}

void LaserMatchScanSplitCpuV5::Destroy() {
    pImpl_.reset();
}

void LaserMatchScanSplitCpuV5::SetParams(const LaserMatchScanParamsV5& params) {
    params.validate();
    if (pImpl_) pImpl_->params_ = params;
}

const LaserMatchScanParamsV5& LaserMatchScanSplitCpuV5::GetParams() const {
    static const LaserMatchScanParamsV5 kDefault{};
    return pImpl_ ? pImpl_->params_ : kDefault;
}

const std::string& LaserMatchScanSplitCpuV5::LastTimingBreakdown() const {
    static const std::string kEmpty;
    return pImpl_ ? pImpl_->lastBd_ : kEmpty;
}

} // namespace calib
