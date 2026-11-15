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
    kDropTinySeg = 5,      // 【D″】入口丢单点段（carded==1，零复核）
    kDropElimCand = 6,     // 【D″/F2】候选被消元清空但票有残留 → 即时弃
    kDropSurvivor = 7,     // 【D″】逐点门后幸存<2 整段弃（防单点复活）
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
    std::vector<float> sumResT_;             // segCount*V3_MAX_LIDS 静态残差表（⑮性能）
    std::vector<int> cntResT_;               // 同上：每 (段,线号) 参与投票的卡数
    std::vector<BitRows> segRows_;           // segCount：行位图（消元判交）
    std::vector<std::vector<int>> segPts_;   // seg → 点下标列表
    std::vector<int> segCarded_;             // seg：持有（未掩蔽）卡的点数＝A 分母（补丁①）
    std::vector<uint32_t> candMask_;         // seg：备选位图（bit=lid-1；补丁②＝A过门∧未消元）
    std::vector<unsigned char> cardMask_;    // totalCards：深度门卡掩蔽（补丁⑩）
    std::vector<int> winner_;                // seg：判定线号（-1=未定）
    std::vector<int> dropWhy_;               // seg：DropWhy
    std::vector<unsigned char> cleanSkip_;   // N：残差清洗点级跳过（补丁⑪）
    std::vector<unsigned char> gateSkip_;    // N：【D″】逐点残差门跳过（与 cleanSkip_ 独立）

    // 【D″】三阶段分桶（0=大, 1=中, 2=小；big<=0 时全部归 0＝坍缩单阶段）
    // 口径＝segCarded_（有卡点数，与⑬一致）。1 点段已在入口丢弃，不进任何阶段。
    int stageOf(int s) const {
        const int carded = segCarded_[static_cast<size_t>(s)];
        if (params_.big_seg_min_pts <= 0 || carded >= params_.big_seg_min_pts)
            return 0;
        if (carded > params_.small_seg_max_pts)
            return 1;
        return 2;
    }

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
    //    【D″/F2】候选被清空但票有残留 → 即时弃（kDropElimCand）——
    //    票全零仍走既有 kDropElimAll 路径（cascadeRound 防御段处理）。
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
            // 【D″/F2】消元后候选空但票残留 → 即时弃（计数真实化＋防御）。
            // 不变量（A 静态＋成对清）下，当前票数第一名非候选 ⟺ 候选已清空。
            if (candMask_[static_cast<size_t>(t)] == 0u) {
                bool hasVotes = false;
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    if (votes_[static_cast<size_t>(t) * V3_MAX_LIDS + l] > 0) {
                        hasVotes = true; break;
                    }
                if (hasVotes) {
                    dropWhy_[static_cast<size_t>(t)] = kDropElimCand;
                }
            }
        }
    }

    // ── 第3~6步：级联主循环（补丁③④⑬⑯D″三阶段）──
    // 阶段（stageOf）：0=大(≥big) 1=中 2=小；big<=0 时全部归 0（坍缩）。
    // phaseLimit：0/1/2＝指定阶段；-1＝全量。返回本轮是否有定段。
    bool cascadeRound(int segCount, int& determinedTotal, int phaseLimit) {
        const LaserMatchScanParamsV5& vp = params_;
        bool any = false;
        for (;;) {
            int bestSeg = -1;
            double bestB = -1.0;
            int bestLid = -1;
            int bestStage = 3;   // 越小越优先
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] >= 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int stage = stageOf(s);
                if (phaseLimit >= 0 && stage != phaseLimit) continue;   // 【D″】阶段过滤
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
                // B 门（第二名不存在 → ∞ 恒过）。
                // 【补丁⑭】margin_cand_scope=true：第二名只在候选集内取——非候选对手
                //（A≤share，永远不可能赢本段）不行使否决权；无候选对手 → B=∞。
                const double B = [&]() -> double {
                    if (!vp.margin_cand_scope)
                        return (sCnt > 0) ? static_cast<double>(w) / sCnt
                                          : std::numeric_limits<double>::infinity();
                    int sCntC = 0;
                    for (int l = 0; l < V3_MAX_LIDS; ++l) {
                        if (l + 1 == wLid) continue;
                        if (((candMask_[static_cast<size_t>(s)] >> l) & 1u) == 0) continue;
                        if (v[l] > sCntC) sCntC = v[l];
                    }
                    return (sCntC > 0) ? static_cast<double>(w) / sCntC
                                       : std::numeric_limits<double>::infinity();
                }();
                if (B < vp.margin_min) continue;
                // 阶段优先 → B 最大 → 线号小 → 段号小（确定性；D″三阶段扩展）
                if (bestSeg < 0 || stage < bestStage ||
                    (stage == bestStage &&
                     (B > bestB || (B == bestB && wLid < bestLid)))) {
                    bestSeg = s; bestB = B; bestLid = wLid; bestStage = stage;
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
        // 【⑮性能】静态残差表（seg×lid 的 Σ|cardRes|/卡数——裁决道每轮逐点重聚是
        // 50 倍慢的根因；卡集/掩蔽帧内不变，表值静态，查表语义与逐点聚合逐位等价）
        sumResT_.assign(static_cast<size_t>(segCount) * V3_MAX_LIDS, 0.0f);
        cntResT_.assign(static_cast<size_t>(segCount) * V3_MAX_LIDS, 0);
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
                    sumResT_[static_cast<size_t>(seg) * V3_MAX_LIDS + (lid - 1)] +=
                        std::fabs(cres[static_cast<size_t>(begV + c)]);
                    ++cntResT_[static_cast<size_t>(seg) * V3_MAX_LIDS + (lid - 1)];
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
        // 【D″】carded==1（单点段）在入口显式丢弃（kDropTinySeg——独立计数，
        // 不混入兜底弃）。零复核、实测 2/3 是垃圾、零覆盖风险（全部线号有 ≥100 点大段认领）。
        int noCardSeg = 0, noSupSeg = 0, tinySeg = 0;
        for (int s = 0; s < segCount; ++s) {
            const int carded = segCarded_[static_cast<size_t>(s)];
            if (carded == 0) {
                dropWhy_[static_cast<size_t>(s)] = kDropNoCard;
                ++noCardSeg;
                continue;
            }
            if (carded == 1) {
                dropWhy_[static_cast<size_t>(s)] = kDropTinySeg;
                ++tinySeg;
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
        // 【诊断·JMW_V5VOTES=1】逐段票数明细（计票刚建好、未判定未消元的初始态）：
        // seg/track/点数/有卡点数/物理位置(x均值,y范围)/备选(A>share 且未消元)＋
        // 逐线号 票数(A百分比,备选*)。只读打印，零算法影响。
        if (getenv("JMW_V5VOTES")) {
            for (int s = 0; s < segCount; ++s) {
                const int carded = segCarded_[static_cast<size_t>(s)];
                const int* v = &votes_[static_cast<size_t>(s) * V3_MAX_LIDS];
                double sx = 0; float ymin = 1e9f, ymax = -1e9f;
                int npts = static_cast<int>(segPts_[static_cast<size_t>(s)].size());
                int trk = -1;
                if (npts > 0) {
                    trk = lt[segPts_[static_cast<size_t>(s)][0]];
                    for (int pi : segPts_[static_cast<size_t>(s)]) {
                        sx += lp[static_cast<size_t>(pi)].x;
                        ymin = std::min(ymin, lp[static_cast<size_t>(pi)].y);
                        ymax = std::max(ymax, lp[static_cast<size_t>(pi)].y);
                    }
                    sx /= npts;
                }
                std::string vs;
                // 逐线号平均|残差|（裁决道 D 的数据源；诊断专用——算法本体不读此处）
                float sumR[V3_MAX_LIDS] = {0.0f};
                int cntR[V3_MAX_LIDS] = {0};
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        const int lid = f.cardLid[static_cast<size_t>(begC + c)];
                        if (lid >= 1 && lid <= V3_MAX_LIDS) {
                            sumR[lid - 1] += std::fabs(cres[static_cast<size_t>(begC + c)]);
                            ++cntR[lid - 1];
                        }
                    }
                }
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    if (v[l] <= 0) continue;
                    const float a = carded > 0
                        ? static_cast<float>(v[l]) / static_cast<float>(carded) : 0.0f;
                    const bool cand = (candMask_[static_cast<size_t>(s)] >> l) & 1u;
                    const float mr = cntR[l] > 0 ? sumR[l] / cntR[l] : -1.0f;
                    vs += fmt::format(" L{}:{}({:.0f}%{})r{:.2f}", l + 1, v[l],
                                      a * 100.0f, cand ? "*" : "", mr);
                }
                std::fprintf(stderr,
                             "[v5:votes] seg=%d trk=%d pts=%d carded=%d x~%.0f y%.0f-%.0f"
                             " drop=%d |%s\n",
                             s, trk, npts, carded, sx, ymin, ymax,
                             dropWhy_[static_cast<size_t>(s)], vs.c_str());
            }
        }
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
        // 【补丁⑯＋D″】抢号竞速修复：大段裁决排在后续阶段投票之前——
        // margin≥1.5 时真主段（B<margin）只能等裁决，而小碎片被消元后成独苗
        // (B=∞) 可立即走投票道抢走线号、再经消元杀死真主。
        // 【D″三阶段】回环顺序：大投票→大裁决→中投票→中裁决→小投票→小裁决→全量兜底。
        auto adjudicateRound = [&](int stageOnly) -> bool {
            int bestSeg = -1;
            double bestScore = -1.0;
            int bestLid = -1, bestVoteLid = -1;
            for (int s = 0; s < segCount; ++s) {
                if (stageOnly >= 0 && stageOf(s) != stageOnly)
                    continue;   // 【D″】阶段过滤（-1＝全量兜底）
                if (winner_[static_cast<size_t>(s)] >= 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int* v = &votes_[static_cast<size_t>(s) * V3_MAX_LIDS];
                int wLid = -1, w = 0, sLid = -1, sCnt = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    if (v[l] > w) { sCnt = w; sLid = wLid; w = v[l]; wLid = l + 1; }
                    else if (v[l] > sCnt) { sCnt = v[l]; sLid = l + 1; }
                }
                // 入口：B 有限（有第二名——B=∞ 段归投票道，防复活）＋票数下限
                if (sCnt <= 0) continue;
                if (w < vp.votes_floor) continue;
                // 平均|残差|查静态表（票数>0 ⟹ 卡数>0，表必命中）
                const float* sumRes = &sumResT_[static_cast<size_t>(s) * V3_MAX_LIDS];
                const int* cntRes = &cntResT_[static_cast<size_t>(s) * V3_MAX_LIDS];
                const double r1 = static_cast<double>(sumRes[wLid - 1]) / cntRes[wLid - 1];
                const double r2 = static_cast<double>(sumRes[sLid - 1]) / cntRes[sLid - 1];
                if (r1 <= 0.0 && r2 <= 0.0) continue;    // 双零＝无差别，死区
                // 票数因子（补丁⑨）：线性 c1/c2（默认）｜对数 ln(c1)/ln(c2)
                double Bfac;
                if (vp.vote_weight_log) {
                    if (sCnt == 1) Bfac = std::numeric_limits<double>::infinity();
                    else if (w <= 1) continue;
                    else Bfac = std::log(static_cast<double>(w)) /
                                std::log(static_cast<double>(sCnt));
                } else {
                    Bfac = static_cast<double>(w) / sCnt;
                }
                const double E = Bfac * ((r1 > 0.0) ? r2 / r1
                                                   : std::numeric_limits<double>::infinity());
                int winLid = -1;
                double score = 0.0;                      // 置信＝离死区的距离（确认 E／翻案 1/E）
                if (E > vp.quality_e_min) {
                    // 【F2 守卫·D″】确认前查第一名须在候选集——防无资格借确认加冕。
                    // 消元清空候选时 eliminate() 已即时弃（kDropElimCand），
                    // 此处为纵深防御（不变量依赖"成对清"实现细节）。
                    if (((candMask_[static_cast<size_t>(s)] >> (wLid - 1)) & 1u) == 0)
                        continue;
                    winLid = wLid;                       // 确认票数第一名
                    score = E;
                } else if (E < 1.0 / vp.quality_e_min &&
                           ((candMask_[static_cast<size_t>(s)] >> (sLid - 1)) & 1u)) {
                    winLid = sLid;                       // 翻案（第二名须候选）
                    score = 1.0 / E;
                } else {
                    continue;                            // 死区——本轮不可定
                }
                if (bestSeg < 0 || score > bestScore ||
                    (score == bestScore && winLid < bestLid)) {
                    bestSeg = s; bestScore = score; bestLid = winLid; bestVoteLid = wLid;
                }
            }
            if (bestSeg < 0) return false;
            // 判定（≠投票第一名＝翻案——只改结论不改票数）
            winner_[static_cast<size_t>(bestSeg)] = bestLid;
            ++determinedTotal;
            ++judged;
            if (bestLid != bestVoteLid) ++flipped;
            eliminate(bestSeg, bestLid, segCount);       // 第5步：消元后回环
            return true;
        };
        for (;;) {
            if (cascadeRound(segCount, determinedTotal, 0)) { fmtBd("cascB"); continue; }
            if (adjudicateRound(0)) { fmtBd("judB"); continue; }
            if (cascadeRound(segCount, determinedTotal, 1)) { fmtBd("cascM"); continue; }
            if (adjudicateRound(1)) { fmtBd("judM"); continue; }
            if (cascadeRound(segCount, determinedTotal, 2)) { fmtBd("cascS"); continue; }
            if (adjudicateRound(2)) { fmtBd("judS"); continue; }
            if (adjudicateRound(-1)) { fmtBd("judA"); continue; }
            break;
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
        result.elimCandSegCount = [&]() {
            int n = 0;
            for (int s = 0; s < segCount; ++s)
                if (dropWhy_[static_cast<size_t>(s)] == kDropElimCand) ++n;
            return n;
        }();
        elimAll = countElimAll();
        result.tinyDroppedSegCount = tinySeg;
        result.qualityJudgedCount = judged;
        result.qualityFlipCount = flipped;
        result.eliminatedAllVotesSegCount = elimAll;
        result.exitDroppedSegCount = exitDrop;
        // 【诊断·JMW_V5VOTES=1】终态逐段结果（与 [v5:votes] 按 seg 对读）
        if (getenv("JMW_V5VOTES")) {
            for (int s = 0; s < segCount; ++s)
                std::fprintf(stderr, "[v5:vres] seg=%d winner=%d drop=%d\n",
                             s, winner_[static_cast<size_t>(s)],
                             dropWhy_[static_cast<size_t>(s)]);
        }
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
                // 自适应门：内点(dev<1.0)散布×1.5×基准，相对钳 [0.4,1.2]×基准
                //（扫参修正：ZG 原版绝对钳 [0.4,1.2]px 按其残差尺度定标；K2 烘表后
                // 残差亚像素级，绝对钳上界 1.2px 把任何基准都封顶—— thresh 2.5/3.0
                // 全部等效 1.2px（实测 8.88M 平台）。改随 thresh 缩放的相对钳。）
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
                gate = std::max(0.4 * vp.res_clean_thresh,
                                std::min(1.2 * vp.res_clean_thresh, gate));
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

        // ──【D″ 阶段3·按线号局部 τ】逐线号算大段赢家卡|res|中位 → τ 表 [1..25]──
        // 落位链：本线号大段 → 本线号中段 → 左右邻居 τ → 全局大段中位 → 门关。
        // 物理依据：激光线物理相邻（16/17/18 号扫同一片物体表面），邻居的
        // 标定偏差水平基本能代表自己——比全局共用一个 τ 更准（消除不同
        // 线号标定精度差异的错位：L1 标定准 vs L25 偏 2.9px）。
        double resGateTau = -1.0;                     // 全局兜底（诊断/结果字段用）
        std::vector<double> tauTable;                 // [0..25]：逐线号 τ（-1=无来源）
        if (vp.res_gate) {
            tauTable.assign(26, -1.0);
            // 逐线号收集大段赢家卡 |res|（stage=0 大段优先；空→stage=1 中段）
            std::vector<std::vector<float>> lidResBig(26), lidResMid(26);
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] < 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int st = stageOf(s);
                if (st > 1) continue;                 // 只收大段/中段
                const int wl = winner_[static_cast<size_t>(s)];
                auto& target = (st == 0) ? lidResBig[wl] : lidResMid[wl];
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (f.cardLid[static_cast<size_t>(begC + c)] != wl) continue;
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        target.push_back(std::fabs(
                            cres[static_cast<size_t>(begC + c)]));
                        break;
                    }
                }
            }
            // 逐线号算中位 → τ
            double globalMed = -1.0;
            {
                std::vector<float> allBig;
                for (int l = 1; l <= 25; ++l)
                    allBig.insert(allBig.end(), lidResBig[l].begin(), lidResBig[l].end());
                if (allBig.empty())
                    for (int l = 1; l <= 25; ++l)
                        allBig.insert(allBig.end(), lidResMid[l].begin(), lidResMid[l].end());
                if (!allBig.empty()) {
                    std::nth_element(allBig.begin(),
                                     allBig.begin() + static_cast<int>(allBig.size()) / 2,
                                     allBig.end());
                    globalMed = allBig[allBig.size() / 2];
                }
            }
            for (int l = 1; l <= 25; ++l) {
                auto& src = !lidResBig[l].empty() ? lidResBig[l] : lidResMid[l];
                if (src.empty()) continue;
                std::nth_element(src.begin(),
                                 src.begin() + static_cast<int>(src.size()) / 2,
                                 src.end());
                double localTau = std::max(
                    static_cast<double>(vp.res_gate_k) * src[src.size() / 2],
                    0.4 * static_cast<double>(vp.res_clean_thresh));
                // 【修正·261113】按线号只能比全局更紧，不能更松——
                // L25 等标定差的线号局部中位更高 → τ 更大 → 放过边界垃圾
                // 推高线均值（#330 实证）。取 min 确保局部精细化只用于
                // 收紧（对准线更精准），不用于放宽。
                if (globalMed > 0) {
                    const double globalTau = std::max(
                        static_cast<double>(vp.res_gate_k) * globalMed,
                        0.4 * static_cast<double>(vp.res_clean_thresh));
                    localTau = std::min(localTau, globalTau);
                }
                tauTable[l] = localTau;
            }
            // 落位填充：没有 τ 的线号 → 左右邻居 → 全局
            for (int l = 1; l <= 25; ++l) {
                if (tauTable[l] > 0) continue;
                for (int off = 1; off <= 24 && tauTable[l] <= 0; ++off) {
                    if (l + off <= 25 && tauTable[l + off] > 0) {
                        tauTable[l] = tauTable[l + off]; break;
                    }
                    if (l - off >= 1 && tauTable[l - off] > 0) {
                        tauTable[l] = tauTable[l - off]; break;
                    }
                }
                if (tauTable[l] <= 0 && globalMed > 0) {
                    tauTable[l] = std::max(
                        static_cast<double>(vp.res_gate_k) * globalMed,
                        0.4 * static_cast<double>(vp.res_clean_thresh));
                }
            }
            // 全局 τ（诊断用：有值的线号的中位）
            {
                std::vector<double> vals;
                for (int l = 1; l <= 25; ++l)
                    if (tauTable[l] > 0) vals.push_back(tauTable[l]);
                if (!vals.empty()) {
                    std::nth_element(vals.begin(),
                                     vals.begin() + static_cast<int>(vals.size()) / 2,
                                     vals.end());
                    resGateTau = vals[vals.size() / 2];
                }
            }
        }
        result.resGateTauPx = static_cast<float>(resGateTau);

        // ──【D″ 阶段4】逐点门＋幸存规则（按线号查 τ 表）──
        // 门范围＝赢家卡数 ≤4（与⑪的 n<5 豁免同界，共享常量——封住翻案段缝：
        // 大段被翻案到 ≤4 卡线号时 ⑪跳过、门须接管）。
        // 幸存规则＝所有已定段统一（含大段——对称）：可输出点<2 → 整段弃。
        gateSkip_.assign(static_cast<size_t>(leftCount), 0);
        int gateDroppedPts = 0, survivorDropped = 0;
        if (vp.res_gate) {
            for (int s = 0; s < segCount; ++s) {
                if (winner_[static_cast<size_t>(s)] < 1) continue;
                if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
                const int wl = winner_[static_cast<size_t>(s)];
                const double tauL = (wl >= 1 && wl <= 25 &&
                                     static_cast<size_t>(wl) < tauTable.size())
                                    ? tauTable[static_cast<size_t>(wl)] : -1.0;
                if (tauL <= 0) continue;              // 该线号无 τ → 门关放行
                // 【修复·261113】先数赢家卡，>4 跳过（大段走⑪自适应清洗）
                // ——原代码 continue 在标记之后＝死语句，大段被误加逐点门
                int wcCount = 0;
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        if (f.cardLid[static_cast<size_t>(begC + c)] == wl) {
                            ++wcCount; break;
                        }
                    }
                }
                if (wcCount > 4) continue;   // 范围外（≥5 走⑪自适应清洗）
                // 逐点检查（只对 ≤4 赢家卡的段）
                for (int pi : segPts_[static_cast<size_t>(s)]) {
                    const int begC = f.cardPtr[static_cast<size_t>(pi)];
                    const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                    for (int c = 0; c < ncC; ++c) {
                        if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                        if (f.cardLid[static_cast<size_t>(begC + c)] == wl) {
                            if (std::fabs(cres[static_cast<size_t>(begC + c)]) > tauL) {
                                gateSkip_[static_cast<size_t>(pi)] = 1;
                                ++gateDroppedPts;
                            }
                            break;
                        }
                    }
                }
            }
        }
        // 幸存规则：所有已定段统一——可输出点 = 赢家卡存在 && !cleanSkip && !gateSkip
        for (int s = 0; s < segCount; ++s) {
            if (winner_[static_cast<size_t>(s)] < 1) continue;
            if (dropWhy_[static_cast<size_t>(s)] != kDropNone) continue;
            const int wl = winner_[static_cast<size_t>(s)];
            int outputable = 0;
            for (int pi : segPts_[static_cast<size_t>(s)]) {
                if (cleanSkip_[static_cast<size_t>(pi)]) continue;
                if (gateSkip_[static_cast<size_t>(pi)]) continue;
                const int begC = f.cardPtr[static_cast<size_t>(pi)];
                const int ncC = f.cardPtr[static_cast<size_t>(pi) + 1] - begC;
                for (int c = 0; c < ncC; ++c) {
                    if (v4Depth && cardMask_[static_cast<size_t>(begC + c)]) continue;
                    if (f.cardLid[static_cast<size_t>(begC + c)] == wl) {
                        ++outputable; break;
                    }
                }
            }
            if (outputable < 2) {
                dropWhy_[static_cast<size_t>(s)] = kDropSurvivor;
                ++survivorDropped;
            }
        }
        result.resGateDroppedPts = gateDroppedPts;
        result.survivorDroppedSegCount = survivorDropped;
        fmtBd("gate");

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
            if (gateSkip_[static_cast<size_t>(i)]) continue;    // 【D″】逐点门（已计数）
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
            "OK: {} matched, {}/{} segs kept | v5: noCard={} noSup={}{} tiny={} elimAll={} "
            "elimCand={} exitDrop={} survivor={} judge={} flip={} depthDropped={} "
            "cleanDropped={} gateDrop={} tau={:.2f}",
            n, segCount - droppedSegs, segCount,
            result.noCardSegCount, result.noSupportSegCount,
            result.noSupportWarned ? "(WARN)" : "",
            result.tinyDroppedSegCount,
            result.eliminatedAllVotesSegCount, result.elimCandSegCount,
            result.exitDroppedSegCount, result.survivorDroppedSegCount,
            result.qualityJudgedCount, result.qualityFlipCount,
            result.depthDroppedCards, result.resCleanDropped,
            result.resGateDroppedPts, result.resGateTauPx);
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
