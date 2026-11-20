/**
 * @file test_laser_match_v5.cpp
 * @brief laser_match_scan_v5 CPU 后段（人工8步骨架＋补丁①~⑫）gtest
 *        —— 前段复用 LaserMatchScanSplitGpuV3 真跑（cardRes 发射开），
 *        同一 FrontV3 喂 V5 被测后段。
 *
 * 用例清单（对照设计方案 v2.1 §四 13 洞对照表＋§十二④）：
 *   V5CardResEmission       发射开→cardRes 尺寸/符号/数值断言（δ 即残差，①挂账项）
 *   V5RequiresCardRes       发射关/尺寸不符→fail-safe 失败（D8 空态契约）
 *   V5EmptyInput            零点 front→success 短路
 *   V5Determinism           同输入两次运行逐位一致（金样对拍前提）
 *   V5SoloFloorSmallSeg     全票 3 点段（votes<floor 但≥solo）能定——洞⑧/P2 铁丝
 *   V5VotesFloorThirdState  B 达标但票数不足→兜底弃（洞①死锁出口）
 *   V5FullCountBasis        全量口径：0.9/0.6 段不得定（洞⑩语义锚）
 *   V5CascadeUnlock         消元解锁：邻段定 lid1→本段 B 跳 ∞ 定 lid2（级联回环）
 *                           ＋B=∞ 不进裁决（洞②防复活，judged==0）
 *   V5QualityFlip           争议段误差可分→判最小误差线号＋翻案计数（骨架第7步）
 *   V5SeparationGateBlocks  两线误差都小但比值过 E 门→分离门拦（补丁⑥/洞③）
 *   V5DepthGate             Q 注入→界外深度卡掩蔽→争议段归一（补丁⑩）＋无 Q 自动关
 *   V5ResCleanDrops         段内离群点清洗（补丁⑪）
 *   V5NoSupportWarn         候选集空计数＋告警标记（D1）
 *   V5ResultSuperset        V5 结果可切片赋 V3 基类（继承兼容，编译期）
 *
 * 构造原理：合成温表 rowStep=0.5；每条目 (xL,yL,uR,lid)。左点在 xL 处查表得
 * 预测 uR；右点放在 uR+δ 处即命中（|δ|≤match_threshold）——**δ 就是该卡的
 * 带符号残差 cardRes**，测试对残差有精确数值控制。段＝连续同 track 点串
 * （前段 GPU 游程）；行重叠＝两段共占同一 row。
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "laser_match_scan_v3_split.h"   // LaserMatchScanSplitGpuV3 / CpuV3 / FrontV3
#include "laser_match_scan_v5.h"         // V5 后段（头自带 ParamsV5/ResultV5，带出 V3 族）

namespace {

// ============================================================================
// 0. 静态环境守卫（仿 test_laser_match_v4.cpp——本 TU 真跑 V3 前段，
//    GPU 侧 JMW_V3* 敏感变量清空；追加 JMW_LMV5 卫生清空）
// ============================================================================
struct V5EnvGuard {
    V5EnvGuard() {
        for (const char* v : {"JMW_V3GRAPH", "JMW_V3BISECT", "JMW_V3ANCHOR", "JMW_V3_SEGCAP",
                              "JMW_V3FULLGPU", "JMW_V3GPU", "JMW_V3PAIR_AB",
                              "JMW_NOTBD", "JMW_V3TIME", "JMW_K1B_SKIPSTAMPS",
                              "JMW_V3CARDRES", "JMW_LMV4", "JMW_LMV5"}) {
            std::string s = std::string(v) + "=";
            _putenv(s.c_str());
        }
    }
} g_v5EnvGuard;

using calib::LaserMatchSplitFrontV3;
using calib::LaserMatchScanCpuResultV3;
using calib::LaserMatchScanCpuResultV5;
using calib::LaserMatchScanParamsV5;

// ── 自定义行布局温表：rowStep=0.5；每行任意条目 (xL,uR,lid)（同 row 同 xL
//    多 lid 合法——构造双线争议/收敛场景）──
struct SynthEntry {
    float xL;
    float uR;
    int lid;
};

std::shared_ptr<const calib::LaserPlaneMapTempTable> MakeTableFromRows(
    const std::vector<std::pair<int, std::vector<SynthEntry>>>& rowSpec, double tempC = 25.0) {
    auto table = std::make_shared<calib::LaserPlaneMapTempTable>();
    calib::LaserPlaneMap m;
    m.temperature = tempC;
    const float rowStep = 0.5f;
    std::vector<cv::Vec4f> entries;
    for (const auto& rs : rowSpec) {
        const float yL = static_cast<float>(rs.first) * rowStep;
        for (const auto& e : rs.second)
            entries.push_back({e.xL, yL, e.uR, static_cast<float>(e.lid)});
    }
    m.leftToRightMap = cv::Mat(static_cast<int>(entries.size()), 1, CV_32FC4, entries.data()).clone();
    m.totalPairs = static_cast<int>(entries.size());
    table->table[tempC] = std::move(m);
    return table;
}

cv::Mat PointsMat(const std::vector<cv::Point2f>& pts) {
    cv::Mat m(static_cast<int>(pts.size()), 1, CV_32FC2);
    if (!pts.empty()) std::memcpy(m.ptr(), pts.data(), sizeof(cv::Point2f) * pts.size());
    return m;
}

cv::Mat IdsMat(const std::vector<int>& ids) {
    cv::Mat m(static_cast<int>(ids.size()), 1, CV_32SC1);
    if (!ids.empty()) std::memcpy(m.ptr(), ids.data(), sizeof(int) * ids.size());
    return m;
}

// 测试基参（rowStep 必须与合成表一致 0.5；窗口 7 容纳 |δ|≤3 的清洗离群构造）
LaserMatchScanParamsV5 BaseP5() {
    LaserMatchScanParamsV5 p;
    p.v3.match_threshold = 7.0f;
    p.v3.epipolar_row_step = 0.5f;
    p.v3.vL_tolerance = 0.5f;
    // V5 新键用代码默认（share 0.8/margin 2.0/E 1.9/floor 4/solo 1/sep 0.5）
    return p;
}

// 前段跑一次（发射 cardRes 可控）
LaserMatchSplitFrontV3 RunFront(calib::LaserMatchScanSplitGpuV3* gpu,
                                const std::vector<cv::Point2f>& left,
                                const std::vector<int>& ltrk,
                                const std::vector<cv::Point2f>& right) {
    cv::cuda::GpuMat dL, dLid, dR, dRid;
    dL.upload(PointsMat(left));
    dLid.upload(IdsMat(ltrk));
    dR.upload(PointsMat(right));
    dRid.upload(IdsMat(std::vector<int>(right.size(), 0)));
    LaserMatchSplitFrontV3 front;
    gpu->Execute(dL, dLid, dR, dRid, front);
    return front;
}

// 行 y（row=r → y=r*0.5）
inline float RowY(int r) { return static_cast<float>(r) * 0.5f; }

// 深度门 Q：Z = Q23 / (Q32*d + Q33)（V4 测试同款构造器）
cv::Mat MakeDepthQ(double q23, double q32, double q33) {
    cv::Mat q = cv::Mat::zeros(4, 4, CV_64F);
    q.at<double>(2, 3) = q23;
    q.at<double>(3, 2) = q32;
    q.at<double>(3, 3) = q33;
    return q;
}

// 逐位一致断言（matchedCount＋四数组 memcmp）
void ExpectIdentical(const LaserMatchScanCpuResultV5& a,
                     const LaserMatchScanCpuResultV5& b, const char* tag) {
    ASSERT_TRUE(a.success && b.success) << tag;
    ASSERT_EQ(a.matchedCount, b.matchedCount) << tag;
    const int n = a.matchedCount;
    if (n == 0) return;
    EXPECT_EQ(0, std::memcmp(a.matched_left.ptr(), b.matched_left.ptr(),
                             sizeof(cv::Point2f) * n)) << tag << " left";
    EXPECT_EQ(0, std::memcmp(a.matched_right.ptr(), b.matched_right.ptr(),
                             sizeof(cv::Point2f) * n)) << tag << " right";
    EXPECT_EQ(0, std::memcmp(a.matched_line_ids.ptr(), b.matched_line_ids.ptr(),
                             sizeof(int) * n)) << tag << " lids";
    EXPECT_EQ(0, std::memcmp(a.matched_track_ids.ptr(), b.matched_track_ids.ptr(),
                             sizeof(int) * n)) << tag << " tracks";
}

// 取第 i 个输出线号
int LidAt(const LaserMatchScanCpuResultV5& r, int i) {
    return r.matched_line_ids.ptr<int>()[i];
}

// ============================================================================
// 1. V5CardResEmission —— 发射开的数值断言（排期①挂账到④的验证项）
//    三点各持一卡；右点故意偏 δ → cardRes == hitRx − uR（带符号）。
//    构造：xL_i=100/102/104，各自独立条目 uR_i=10i，右点放 10i+δ_i。
// ============================================================================
TEST(LaserMatchV5, V5CardResEmission) {
    // 表：row100 三条目（每点一个 xL/uR——防右点串命中；全同一 lid=1，
    // 否则三点三线号 A=1/3 会触发无支持告警偏离本测试意图）
    auto table = MakeTableFromRows({
        {100, {{100.0f, 0.0f, 1}, {102.0f, 10.0f, 1}, {104.0f, 20.0f, 1}}},
    });
    const std::vector<float> delta = {+0.5f, -0.5f, 0.0f};
    std::vector<cv::Point2f> left = {{100, RowY(100)}, {102, RowY(100)}, {104, RowY(100)}};
    std::vector<int> trk = {7, 7, 7};
    std::vector<cv::Point2f> right = {
        {0.0f + delta[0], RowY(100)}, {10.0f + delta[1], RowY(100)},
        {20.0f + delta[2], RowY(100)}};

    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);   // 【V5 契约】发射开
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    // 尺寸契约：cardRes 与卡表同长
    ASSERT_EQ(front.cardRes.size(), static_cast<size_t>(front.totalCards));
    ASSERT_EQ(front.totalCards, 3);
    // 数值/符号：cardRes = hitRx − uR_pred = δ（>0＝实际右点在预测右侧）
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(front.cardRes[static_cast<size_t>(i)], delta[static_cast<size_t>(i)], 1e-4f)
            << "i=" << i;

    // 消费侧：V5 跑通（3 点全票 A=1.0 → B=∞ → solo floor 定）
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 3);
}

// ============================================================================
// 2. V5RequiresCardRes —— 发射关/尺寸不符 → fail-safe 失败（D8 空态契约）
// ============================================================================
TEST(LaserMatchV5, V5RequiresCardRes) {
    auto table = MakeTableFromRows({{100, {{100.0f, 0.0f, 1}}}});
    std::vector<cv::Point2f> left = {{100, RowY(100)}};
    std::vector<int> trk = {7};
    std::vector<cv::Point2f> right = {{0.5f, RowY(100)}};

    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    // 不 SetEmitCardRes → 发射关
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    EXPECT_TRUE(front.cardRes.empty());

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.message.find("cardRes"), std::string::npos) << r.message;

    // 尺寸不符路径（手工 front：totalCards=2 但 cardRes=1）
    LaserMatchSplitFrontV3 f2;
    f2.success = true;
    f2.leftCount = 1; f2.rightCount = 1; f2.segCount = 1; f2.totalCards = 2;
    f2.leftPts = left; f2.leftTracks = trk; f2.segmap = {0};
    f2.leftStatus = {1};
    f2.cardPtr = {0, 2};
    f2.cardLid = {1, 1}; f2.cardRx = {0.5f, 0.5f}; f2.cardRidx = {0, 0};
    f2.cardRes = {0.1f};   // 尺寸不符
    LaserMatchScanCpuResultV5 r2 = v5.Execute(f2);
    EXPECT_FALSE(r2.success);
    EXPECT_NE(r2.message.find("cardRes"), std::string::npos) << r2.message;
}

// ============================================================================
// 3. V5EmptyInput —— 零点 front → success 短路
// ============================================================================
TEST(LaserMatchV5, V5EmptyInput) {
    calib::LaserMatchScanSplitCpuV5 v5(BaseP5());
    LaserMatchSplitFrontV3 f;
    f.success = true;   // counts=0
    LaserMatchScanCpuResultV5 r = v5.Execute(f);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.matchedCount, 0);
}

// ============================================================================
// 4. V5SoloFloorSmallSeg —— 洞⑧/P2：3 点全票段（votes 3 < floor 4，
//    ≥ solo 2）能定。铁丝形态。【D″】solo 默认 1→2。
// ============================================================================
TEST(LaserMatchV5, V5SoloFloorSmallSeg) {
    auto table = MakeTableFromRows({{100, {{110.0f, 5.0f, 5}}}});
    std::vector<cv::Point2f> left(3, {110, RowY(100)});
    std::vector<int> trk = {9, 9, 9};
    std::vector<cv::Point2f> right = {{5.2f, RowY(100)}};

    LaserMatchScanParamsV5 p5 = BaseP5();
    EXPECT_EQ(p5.votes_floor, 4);      // 有对手下限 4
    EXPECT_EQ(p5.votes_floor_solo, 2); // 【D″】全票下限 2（1→2：单点串入口丢弃）
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 3);      // 3 点全输出（3≥2 全票下限过）
    for (int i = 0; i < 3; ++i) EXPECT_EQ(LidAt(r, i), 5);
}

// ============================================================================
// 5. V5VotesFloorThirdState —— 洞①：B 达标（3:1→B=3≥2）但票数 3<floor 4
//    → 定不了也进不了裁决 → 兜底弃（有出口，不死锁）。
// ============================================================================
TEST(LaserMatchV5, V5VotesFloorThirdState) {
    // p1/p2: xL=100 仅 lid1；p3: xL=102 持 lid1+lid7 → votes 3:1
    auto table = MakeTableFromRows({
        {100, {{100.0f, 0.0f, 1}, {102.0f, 2.0f, 1}, {102.0f, 32.0f, 7}}},
    });
    std::vector<cv::Point2f> left = {{100, RowY(100)}, {100, RowY(100)},
                                     {102, RowY(100)}};
    std::vector<int> trk = {1, 1, 1};
    std::vector<cv::Point2f> right = {{0.1f, RowY(100)}, {2.1f, RowY(100)},
                                      {32.1f, RowY(100)}};
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.totalCards, 4);   // 1+1+2

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 0);                    // 定不了→弃
    EXPECT_EQ(r.exitDroppedSegCount, 1);             // 兜底出口（不死循环）
    EXPECT_EQ(r.qualityJudgedCount, 0);              // 票数不足禁入裁决
}

// ============================================================================
// 6. V5FullCountBasis —— 洞⑩语义锚：第一名 A=0.9 但全量第二名有分量
//    （B=9/6=1.5<2）→ 不得立即定；无邻段可消 → 兜底弃。
// ============================================================================
TEST(LaserMatchV5, V5FullCountBasis) {
    // 10 点三行：p1-5（xL=100 双卡 lid1+lid7）、p6-9（xL=102 仅 lid1）、
    // p10（xL=104 仅 lid7）→ votes lid1=9（A=0.9）、lid7=6（A=0.6）
    const std::vector<SynthEntry> row = {
        {100.0f, 0.0f, 1}, {100.0f, 30.0f, 7},
        {102.0f, 2.0f, 1},
        {104.0f, 34.0f, 7}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    auto push = [&](float xL, int r) { left.emplace_back(xL, RowY(r)); trk.push_back(1); };
    push(100, 100); push(100, 100);
    push(100, 101); push(100, 101); push(100, 101);    // 5×双卡（xL=100）
    push(102, 101);
    push(102, 102); push(102, 102); push(102, 102);    // 4×仅 lid1（xL=102）
    push(104, 102);                                    // 1×仅 lid7（xL=104）
    std::vector<cv::Point2f> right;
    for (int r : {100, 101, 102}) {
        right.emplace_back(0.1f, RowY(r));    // lid1@xL=100 命中
        right.emplace_back(30.1f, RowY(r));   // lid7@xL=100 命中
        right.emplace_back(2.1f, RowY(r));    // lid1@xL=102 命中
        right.emplace_back(34.1f, RowY(r));   // lid7@xL=104 命中
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 0);            // B=9/6=1.5<2 → 不定
    EXPECT_EQ(r.exitDroppedSegCount, 1);
}

// ============================================================================
// 7. V5CascadeUnlock —— 消元解锁（骨架 3-6 步动力）＋B=∞ 不进裁决（洞②）。
//    SegA（3 点全票 lid1，B=∞）先定 → 行重叠的 SegB（6:6 争议）lid1 票被消
//    → SegB 剩 lid2 独大 B=∞ → 级联定 lid2（非裁决）。
// ============================================================================
TEST(LaserMatchV5, V5CascadeUnlock) {
    // 每行条目：SegA xL=100→(uR=0,lid1)；SegB xL=150→双条目 (uR=10,lid1)+(uR=40,lid2)
    const std::vector<SynthEntry> row = {
        {100.0f, 0.0f, 1},
        {150.0f, 10.0f, 1}, {150.0f, 40.0f, 2}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 102; ++r) { left.emplace_back(100, RowY(r)); trk.push_back(1); }
    for (int r = 100; r <= 102; ++r) { left.emplace_back(150, RowY(r)); trk.push_back(2); }
    std::vector<cv::Point2f> right;
    for (int r = 100; r <= 102; ++r) {
        right.emplace_back(0.2f, RowY(r));    // SegA lid1 命中
        right.emplace_back(10.9f, RowY(r));   // SegB lid1 命中（残差 +0.9）
        right.emplace_back(40.1f, RowY(r));   // SegB lid2 命中（残差 +0.1）
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.segCount, 2);
    ASSERT_EQ(front.totalCards, 9);   // 3×1 + 3×2

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 6);
    EXPECT_EQ(r.qualityJudgedCount, 0);   // B=∞ 段不进裁决（洞②防复活）
    EXPECT_EQ(r.qualityFlipCount, 0);
    // 轨迹→线号：track1→lid1、track2→lid2
    const int* tids = r.matched_track_ids.ptr<int>();
    for (int i = 0; i < r.matchedCount; ++i)
        EXPECT_EQ(LidAt(r, i), tids[i] == 1 ? 1 : 2) << "i=" << i;
}

// ============================================================================
// 8. V5QualityFlip —— 骨架第7步：争议段（6:6，B=1）误差可分（D=4.5，
//    E=4.5≥1.9，sep=0.7≥0.5）→ 判最小误差线号 lid2＝翻案（投票平票取小 lid1）。
// ============================================================================
TEST(LaserMatchV5, V5QualityFlip) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 1}, {200.0f, 30.0f, 2}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 102; ++r)
        for (int k = 0; k < 2; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(3); }
    std::vector<cv::Point2f> right;
    for (int r = 100; r <= 102; ++r) {
        right.emplace_back(0.9f, RowY(r));    // lid1 残差 +0.9
        right.emplace_back(30.2f, RowY(r));   // lid2 残差 +0.2
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.qualityJudgedCount, 1);
    EXPECT_EQ(r.qualityFlipCount, 1);      // lid1（投票平票小 lid）→ lid2（误差最优）
    EXPECT_EQ(r.matchedCount, 6);
    for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 2);
    // 补丁⑫：matched_res 与输出同序（lid2 残差 +0.2）
    ASSERT_EQ(r.matched_res.size(), static_cast<size_t>(r.matchedCount));
    for (float res : r.matched_res) EXPECT_NEAR(res, 0.2f, 1e-3f);
    EXPECT_NEAR(r.resMedianPx, 0.2f, 1e-3f);
}

// ============================================================================
// 9. V5SeparationGateBlocks —— 【261112 对称双门版语义改写】两线平均误差
//    0.24(票多) vs 0.10(票少)：E = 1.0×(0.10/0.24) = 0.42 < 1/1.9 → 翻案门
//    （比值 2.4 悬殊即翻，绝对差大小不再是门——分离门参数退役）。
// ============================================================================
TEST(LaserMatchV5, V5SeparationGateBlocks) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 1}, {200.0f, 30.0f, 2}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 102; ++r)
        for (int k = 0; k < 2; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(3); }
    std::vector<cv::Point2f> right;
    for (int r = 100; r <= 102; ++r) {
        right.emplace_back(0.24f, RowY(r));   // lid1（票多）残差 +0.24
        right.emplace_back(30.10f, RowY(r));  // lid2（票少）残差 +0.10
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.qualityJudgedCount, 1);
    EXPECT_EQ(r.qualityFlipCount, 1);         // E=0.42<1/1.9 → 翻案给票少的 lid2
    EXPECT_EQ(r.matchedCount, 6);
    for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 2);
}

// ============================================================================
// 10. V5DepthGate —— 补丁⑩：Q 注入（Z=1e5/d）使 lid2（d≈60→Z≈1672>1500）
//     卡全掩蔽 → 争议段退化为全票 lid1 → 级联定 lid1；无 Q 自动关（回到 Flip 形态）。
// ============================================================================
TEST(LaserMatchV5, V5DepthGate) {
    // lid1: uR=0（d≈199→Z≈1005 ✓）；lid2: uR=40（d≈160→Z≈625 ✓ 会保留）——
    // 改为差异化深度：lid2 uR=160 → d=200−160.2≈40 → Z≈2500 ✗ 掩蔽
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 1}, {200.0f, 160.0f, 2}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 102; ++r)
        for (int k = 0; k < 2; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(3); }
    std::vector<cv::Point2f> right;
    for (int r = 100; r <= 102; ++r) {
        right.emplace_back(0.2f, RowY(r));     // lid1 残差 +0.2（Z≈1005 ✓）
        right.emplace_back(160.9f, RowY(r));   // lid2 残差 +0.9（Z≈2535 ✗）
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    p5.depth_gate = true;
    p5.depth_min = 200.0f;
    p5.depth_max = 1500.0f;
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    // 无 Q → 深度门自动关 → 争议段走裁决翻 lid1（残差 0.2 更小）——对照臂
    {
        calib::LaserMatchScanSplitCpuV5 v5(p5);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.depthDroppedCards, 0);
        ASSERT_GT(r.matchedCount, 0);
        EXPECT_EQ(LidAt(r, 0), 1);   // 裁决取最小误差（0.2）
    }
    // 有 Q → lid2 卡掩蔽 → 退化为全票 lid1 级联定
    {
        calib::LaserMatchScanSplitCpuV5 v5(p5);
        v5.SetStereoQ(MakeDepthQ(1.0e5, 1.0, 0.0));
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.depthDroppedCards, 6);   // 6 点各 1 张 lid2 卡
        EXPECT_EQ(r.matchedCount, 6);
        for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 1);
    }
}

// ============================================================================
// 11. V5ResCleanDrops —— 补丁⑪：10 点全票段，1 点残差 +3.0 离群
//     （其余 +0.1）→ 自适应门（钳下限 0.4）剔离群点。
// ============================================================================
TEST(LaserMatchV5, V5ResCleanDrops) {
    // 每点独立 xL/uR（防右点串命中——uR 间距 20 > 2×窗口 7）：xL=100+2i，uR=20i；
    // δ=0.1×9＋3.0×1
    std::vector<SynthEntry> row100, row101;
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
    auto addPoint = [&](int i, int r, float delta) {
        const float xL = 100.0f + 2.0f * i;
        const float uR = 20.0f * i;
        (r == 100 ? row100 : row101).push_back({xL, uR, 1});
        left.emplace_back(xL, RowY(r));
        trk.push_back(4);
        right.emplace_back(uR + delta, RowY(r));
    };
    for (int i = 0; i < 10; ++i)
        addPoint(i, (i < 5) ? 100 : 101, (i == 7) ? 3.0f : 0.1f);
    auto table = MakeTableFromRows({{100, row100}, {101, row101}});

    LaserMatchScanParamsV5 p5 = BaseP5();
    p5.res_clean = true;
    p5.res_clean_thresh = 1.0f;
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.totalCards, 10);

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.resCleanDropped, 1);    // 离群点被剔
    EXPECT_EQ(r.matchedCount, 9);
    // 幸存者残差全为 +0.1（离群 +3.0 不在输出里）
    for (float res : r.matched_res) EXPECT_NEAR(res, 0.1f, 1e-3f);
}

// ============================================================================
// 12. V5NoSupportWarn —— D1：两段各 2:2 分票（A=0.5≤0.8）→ 候选集空
//     计数＋占比超限告警标记。
// ============================================================================
TEST(LaserMatchV5, V5NoSupportWarn) {
    // 两段，每段＝一个仅 lid1 点（xL=200）＋一个仅 lid2 点（xL=202）
    // → 各段 votes 1:1、A=0.5 ≤0.8 → 候选集空（不同 xL 才能分开持卡）
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 1}, {202.0f, 30.0f, 2}};
    auto table = MakeTableFromRows({{100, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int seg = 1; seg <= 2; ++seg) {
        left.emplace_back(200, RowY(100)); trk.push_back(seg);   // 仅 lid1 卡
        left.emplace_back(202, RowY(100)); trk.push_back(seg);   // 仅 lid2 卡
    }
    std::vector<cv::Point2f> right = {{0.1f, RowY(100)}, {30.1f, RowY(100)}};
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.segCount, 2);

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.noSupportSegCount, 2);      // 两段候选集空
    EXPECT_TRUE(r.noSupportWarned);         // 占比 100% > 30% → 告警
    EXPECT_EQ(r.matchedCount, 0);
}

// ============================================================================
// 13. V5Determinism —— 同一 front 两个实例各跑一次 → 逐位一致
//     （复用 Flip 场景，含裁决路径的确定性）。
// ============================================================================
TEST(LaserMatchV5, V5Determinism) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 1}, {200.0f, 30.0f, 2}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 102; ++r)
        for (int k = 0; k < 2; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(3); }
    std::vector<cv::Point2f> right;
    for (int r = 100; r <= 102; ++r) {
        right.emplace_back(0.9f, RowY(r));
        right.emplace_back(30.2f, RowY(r));
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5a(p5), v5b(p5);
    LaserMatchScanCpuResultV5 ra = v5a.Execute(front);
    LaserMatchScanCpuResultV5 rb = v5b.Execute(front);
    ASSERT_TRUE(ra.success && rb.success);
    ASSERT_GT(ra.matchedCount, 0);
    ExpectIdentical(ra, rb, "determinism");
}

// ============================================================================
// 13b. V5WireCageSynthetic —— 排期⑤：合成铁丝笼（镂空工件形态）。
//    物理形态：每根激光线打在笼子上被铁丝切成多个短段（每段 3~8 点、单线号、
//    不同段不共行）。100 段 × 25 线 → 全段必须配出、零丢弃、线号全对。
//    混合 3 点段（走 solo floor）与 5~8 点段（走 votes_floor）双路径。
// ============================================================================
TEST(LaserMatchV5, V5WireCageSynthetic) {
    // 25 线 × 每线 4 根丝：线 l 的丝 w 占 rows [100 + w*10 + (l-1), +2]（不同线
    // 的段允许共行——消元只删同线号票，异线号共行无害），xL=100+2l、uR=20l
    //（线间视差差 20px>2×窗口，防右点串命中）。每段 3~8 点轮转。
    std::vector<std::pair<int, std::vector<SynthEntry>>> rows;
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
    int track = 0;
    int expectedPts = 0;
    for (int l = 1; l <= 25; ++l) {
        const float xL = 100.0f + 2.0f * l;
        const float uR = 20.0f * l;
        for (int w = 0; w < 4; ++w) {
            const int nPts = 3 + ((l + w) % 6);   // 3..8 轮转
            const int r0 = 100 + w * 10 + (l - 1);
            ++track;
            for (int k = 0; k < nPts; ++k) {
                const int r = r0 + (k % 3);       // 段内 3 行内取点
                rows.push_back({r, {{xL, uR, l}}});
                left.emplace_back(xL, RowY(r));
                trk.push_back(track);
                right.emplace_back(uR + 0.1f, RowY(r));
                ++expectedPts;
            }
        }
    }
    // 合并同 row 的多条目（MakeTableFromRows 按行组织）
    {
        std::map<int, std::vector<SynthEntry>> byRow;
        for (auto& pr : rows) for (auto& e : pr.second) byRow[pr.first].push_back(e);
        rows.clear();
        for (auto& kv : byRow) rows.push_back({kv.first, kv.second});
    }
    auto table = MakeTableFromRows(rows);

    LaserMatchScanParamsV5 p5 = BaseP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.segCount, 100);   // 25 线 × 4 丝

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, expectedPts);          // 全点配出
    EXPECT_EQ(r.noCardSegCount, 0);
    EXPECT_EQ(r.noSupportSegCount, 0);               // 铁丝段全票不误报
    EXPECT_EQ(r.exitDroppedSegCount, 0);             // 零丢弃（P2 目标）
    EXPECT_EQ(r.eliminatedAllVotesSegCount, 0);
    EXPECT_FALSE(r.noSupportWarned);
    // 线号全对：track→line 映射按构造（track = (l-1)*4+w+1 → l=(track-1)/4+1）
    const int* tids = r.matched_track_ids.ptr<int>();
    for (int i = 0; i < r.matchedCount; ++i)
        EXPECT_EQ(LidAt(r, i), (tids[i] - 1) / 4 + 1) << "i=" << i;
}

// ============================================================================
// 13c. V5BigSegFirst —— 补丁⑬：两阶段硬控排序（frame_001 事故的微缩复现）。
//    场景：大段（70 点，L5:70 100%、L18:55 78.6%——折叠区双持）＋
//    行重叠的 2 点噪声碎片（孤立假全票 L5:2, B=∞）。
//    关（big_seg_min_pts=0，纯 B 序）：碎片 B=∞ 先定 5 → 大段 5 票被消 →
//      沦为 18（事故复现：5 号被标成 18）；
//    开（=60）：大段先判 5 ✓ → 碎片候选被消 → 静默出局。
// ============================================================================
TEST(LaserMatchV5, V5BigSegFirst) {
    // 行条目：xL=200 双卡(L5@uR0,L18@uR30)；xL=202 仅 L5；xL=205 噪声碎片仅 L5
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 5}, {200.0f, 30.0f, 18},
        {202.0f, 2.0f, 5},
        {205.0f, 5.0f, 5}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row},
                                    {103, row}, {104, row}});
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
    auto addPt = [&](float xL, int r, int trkId) {
        left.emplace_back(xL, RowY(r));
        trk.push_back(trkId);
    };
    // 大段 70 点（trk=1）：55 点 xL=200（双卡）＋15 点 xL=202（仅 L5）
    for (int r = 100; r <= 104; ++r) {
        for (int k = 0; k < 11; ++k) addPt(200, r, 1);   // 55 双卡
        for (int k = 0; k < 3; ++k) addPt(202, r, 1);    // 15 仅 L5
    }
    // 噪声碎片 2 点（trk=2，行 100~101 与大段重叠）
    addPt(205, 100, 2);
    addPt(205, 101, 2);
    for (int r = 100; r <= 104; ++r) {
        right.emplace_back(0.2f, RowY(r));    // L5@xL=200 命中
        right.emplace_back(30.1f, RowY(r));   // L18@xL=200 命中
        right.emplace_back(2.1f, RowY(r));    // L5@xL=202 命中
        right.emplace_back(5.1f, RowY(r));    // L5@xL=205 命中
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    p5.share_fast = 0.75f;   // A(L18)=55/70=78.6%>75% → 18 也入备选（折叠区形态）
    p5.margin_min = 1.05f;   // 大段 B=70/55=1.27 须过门（结构体默认 2.0 会把大段
                             // 挡在门外、事故在两臂同样发生——测试将失真）
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    // ── 关（纯 B 序）：事故复现——碎片(B=∞)先定 5，大段沦为 18 ──
    {
        LaserMatchScanParamsV5 p = p5;
        p.big_seg_min_pts = 0;
        calib::LaserMatchScanSplitCpuV5 v5(p);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        // 事故态：大段 70 点中仅 55 点持 18 卡可输出（15 点只持 5 卡→无赢家卡
        // 静默丢）＋碎片 2 点＝57——丢点本身即事故损伤的一部分
        ASSERT_EQ(r.matchedCount, 57);
        const int* tids = r.matched_track_ids.ptr<int>();
        int bigLid = -1, junkLid = -1;
        for (int i = 0; i < r.matchedCount; ++i)
            (tids[i] == 1 ? bigLid : junkLid) = LidAt(r, i);
        EXPECT_EQ(bigLid, 18);                     // 事故：真 5 号段被标成 18
        EXPECT_EQ(junkLid, 5);                     // 噪声碎片抢走了 5
    }
    // ── 开（阈值 60）：大段先判 5，碎片候选被消出局 ──
    {
        LaserMatchScanParamsV5 p = p5;
        p.big_seg_min_pts = 60;                    // 大段 70 ≥ 60 阶段一
        calib::LaserMatchScanSplitCpuV5 v5(p);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.matchedCount, 70);             // 仅大段输出
        for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 5);
        EXPECT_EQ(r.eliminatedAllVotesSegCount + r.exitDroppedSegCount >= 1, true);
    }
}

// ============================================================================
// 13d. V5MarginCandScope —— 补丁⑭：B 分母候选集豁免（0.8/2.0 臂 52/53 弃段死因）。
//    场景同 V5BigSegFirst 骨架：大段 70 点（L5:70 全票候选 ＋ L18:55=78.6% 非候选@0.8），
//    margin=2.0：
//    全量口径（false）：B=70/55=1.27<2 → 投票否决；裁决道可比候选仅 {L5} → 拒收 → 弃；
//    候选豁免（true）：非候选 L18 不计入分母 → 无候选对手 → B=∞ → 投票直判 L5。
// ============================================================================
TEST(LaserMatchV5, V5MarginCandScope) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 5}, {200.0f, 30.0f, 18},
        {210.0f, 500.0f, 5}};   // 单线点独立 uR——防 L5 卡残差被邻窗右点污染
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row},
                                    {103, row}, {104, row}});
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
    for (int r = 100; r <= 104; ++r) {
        for (int k = 0; k < 11; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(1); }
        for (int k = 0; k < 3; ++k)  { left.emplace_back(210, RowY(r)); trk.push_back(1); }
    }
    for (int r = 100; r <= 104; ++r) {
        right.emplace_back(0.2f, RowY(r));    // L5（票多）残差 +0.2 —— r1=0.2 全净
        right.emplace_back(30.9f, RowY(r));   // L18（票少）残差 +0.9 —— r2=0.9
        right.emplace_back(500.2f, RowY(r));  // 单线点 L5 残差 +0.2
    }
    LaserMatchScanParamsV5 p5 = BaseP5();
    p5.share_fast = 0.8f;     // L18 的 A=78.6% < 80% → 非候选
    p5.margin_min = 2.0f;
    p5.big_seg_min_pts = 60;
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    // 全量口径（⑭关）：投票被 B=1.27<2 挡 → 裁决门确认 L5
    //（E = 1.27×(0.9/0.543) ≈ 2.11 > 1.9——对称双门公式）
    {
        LaserMatchScanParamsV5 p = p5;
        p.margin_cand_scope = false;
        calib::LaserMatchScanSplitCpuV5 v5(p);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.matchedCount, 70);
        EXPECT_EQ(r.qualityJudgedCount, 1);   // 经裁决门确认（非投票道）
        for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 5);
    }
    // 候选豁免（补丁⑭）：B=∞ → 投票直判 L5，70 点全出（不经裁决）
    {
        LaserMatchScanParamsV5 p = p5;
        p.margin_cand_scope = true;
        calib::LaserMatchScanSplitCpuV5 v5(p);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.matchedCount, 70);
        EXPECT_EQ(r.qualityJudgedCount, 0);   // 投票道直接定，裁决门未动
        for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 5);
    }
}

// ============================================================================
// 13e. V5JudgeRivalAdmit —— 补丁⑮：裁决道对手准入＋翻案合法性门。
//    场景：大段 70 点（L5:70 全票候选 ＋ L18:55=78.6%），margin=2.0 → B=1.27<2 进裁决：
//    A) ⑮开＋L5 残差更小 → 判 L5（非候选对手输了，不碍事）→ 70 点全出
//    B) ⑮开＋L18 残差更小＋L18 非候选(@0.8) → 裁判想判 L18 但它无资格 → 整段弃
//    C) ⑮开＋share 0.75（L18 是候选）＋L18 残差更小 → 合法翻案 → 70 点全判 L18
// ============================================================================
TEST(LaserMatchV5, V5JudgeRivalAdmit) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 5}, {200.0f, 30.0f, 18},
        {202.0f, 2.0f, 5}};
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row},
                                    {103, row}, {104, row}});
    std::vector<cv::Point2f> left;
    std::vector<int> trk;
    for (int r = 100; r <= 104; ++r) {
        for (int k = 0; k < 11; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(1); }
        for (int k = 0; k < 3; ++k)  { left.emplace_back(202, RowY(r)); trk.push_back(1); }
    }
    auto runCase = [&](float share, float d5, float d18, int expLid, int expMatched) {
        std::vector<cv::Point2f> right;
        for (int r = 100; r <= 104; ++r) {
            right.emplace_back(0.0f + d5, RowY(r));
            right.emplace_back(30.0f + d18, RowY(r));
            right.emplace_back(2.0f + d5, RowY(r));
        }
        LaserMatchScanParamsV5 p5 = BaseP5();
        p5.share_fast = share;
        p5.margin_min = 2.0f;
        p5.big_seg_min_pts = 60;
        p5.judge_rival_admit = true;
        p5.quality_sep_px = 0.0f;   // 单线点残差被邻窗右点干扰（0.543 vs 0.9 差 0.36）
                                     // ——sep=0.5 会误拦，本测试只验证⑮准入/合法性语义
        calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
        EXPECT_TRUE(gpu.SetTempTable(table));
        gpu.SetCurrentTemperature(25.0);
        gpu.SetEmitCardRes(true);
        LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
        EXPECT_TRUE(front.success) << front.message;
        calib::LaserMatchScanSplitCpuV5 v5(p5);
        LaserMatchScanCpuResultV5 r = v5.Execute(front);
        EXPECT_TRUE(r.success) << r.message;
        EXPECT_EQ(r.matchedCount, expMatched);
        if (expMatched > 0)
            for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), expLid);
        return r;
    };
    // A：L5 残差好（E≈2.11>1.9）→ 确认门 → 定 L5 → 70 点全出
    runCase(0.8f, 0.2f, 0.9f, 5, 70);
    // B：L18 残差好（E≈0.27<1/1.9）→ 翻案方向，但 L18 非候选 → 死区 → 兜底弃
    LaserMatchScanCpuResultV5 rb = runCase(0.8f, 0.9f, 0.2f, -1, 0);
    EXPECT_EQ(rb.qualityJudgedCount, 0);      // 死区不算裁决（最终兜底弃）
    EXPECT_EQ(rb.exitDroppedSegCount, 1);
    // C：share 0.75 → L18 是候选 → 翻案成立。
    //    15 个只持 L5 卡的点无赢家卡被丢（真实折叠段同款行为）→ 55 点输出
    runCase(0.75f, 0.9f, 0.2f, 18, 55);
}

// ============================================================================
// 13f. V5SoloSnatchRace —— 补丁⑯：抢号竞速（334 墙主因的微缩复现）。
//    大段 70 点（B=1.27<2 投票受阻，E=5.7 裁判可确认）＋ 2 点独苗碎片（B=∞
//    投票道随时可判）。⑯前：碎片在阶段二投票抢走 L5 → 消元杀真主 → 输出 2 点；
//    ⑯后：大段裁决（judge1 轮）先于阶段二投票 → 真主确认 L5 → 碎片被消元出局。
// ============================================================================
TEST(LaserMatchV5, V5SoloSnatchRace) {
    const std::vector<SynthEntry> row = {
        {200.0f, 0.0f, 5}, {200.0f, 30.0f, 18},
        {210.0f, 500.0f, 5},   // 单线点独立 uR（残差干净）
        {205.0f, 5.0f, 5}};    // 碎片（2 点，仅 L5）
    auto table = MakeTableFromRows({{100, row}, {101, row}, {102, row},
                                    {103, row}, {104, row}});
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
    for (int r = 100; r <= 104; ++r) {
        for (int k = 0; k < 11; ++k) { left.emplace_back(200, RowY(r)); trk.push_back(1); }
        for (int k = 0; k < 3; ++k)  { left.emplace_back(210, RowY(r)); trk.push_back(1); }
    }
    left.emplace_back(205, RowY(100)); trk.push_back(2);   // 碎片点（行与大段重叠）
    left.emplace_back(205, RowY(101)); trk.push_back(2);
    for (int r = 100; r <= 104; ++r) {
        right.emplace_back(0.2f, RowY(r));     // L5@200 残差 0.2 → r1=0.2
        right.emplace_back(30.9f, RowY(r));    // L18 残差 0.9 → r2=0.9 → E=5.7
        right.emplace_back(500.2f, RowY(r));   // 单线点 L5 残差 0.2
    }
    right.emplace_back(5.1f, RowY(100));       // 碎片 L5 命中
    right.emplace_back(5.1f, RowY(101));
    LaserMatchScanParamsV5 p5 = BaseP5();
    p5.share_fast = 0.8f;
    p5.margin_min = 2.0f;
    p5.big_seg_min_pts = 60;
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;
    ASSERT_EQ(front.segCount, 2);

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 70);             // 真主 70 点全出（非碎片的 2 点）
    for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(LidAt(r, i), 5);
    const int* tids = r.matched_track_ids.ptr<int>();
    for (int i = 0; i < r.matchedCount; ++i) EXPECT_EQ(tids[i], 1);
    EXPECT_EQ(r.qualityJudgedCount, 1);        // 大段经 judge1 轮确认
    EXPECT_EQ(r.eliminatedAllVotesSegCount, 1);// 碎片被消元出局
}

// ============================================================================
// 14. V5ResultSuperset —— V5 结果可切片赋 V3 基类（编译期继承兼容）
// ============================================================================
TEST(LaserMatchV5, V5ResultSuperset) {
    static_assert(std::is_base_of<calib::LaserMatchScanCpuResultV3,
                                  LaserMatchScanCpuResultV5>::value,
                  "V5 result must derive from V3 cpu result");
    LaserMatchScanCpuResultV5 r5;
    LaserMatchScanCpuResultV3 slice = r5;   // 切片赋值合法
    EXPECT_EQ(slice.matchedCount, 0);
}

// ============================================================================
// 【步0·261114 红队 T0 补口】D″ 阶段3/4（τ 表＋逐点门＋幸存规则）单测。
// 此前 19 测全走 res_gate=false 路径，被重构代码零覆盖。
// 场景骨架：大段（60 点全票 lid1，δ=0.2，行 100/101）供 τ 表（τ=max(k×0.2,
// 0.4×thresh)=0.8@k=4.0,thresh=1.0）；目标段在行 200/201（与大段行不相交→
// 不被消元，各自独立定 lid1）。
// ============================================================================
namespace {
// 建场景：bigN 点大段＋目标段（tgtDeltas 每点 δ）。返回拼好的表/左右点/track。
struct GateScene {
    std::shared_ptr<const calib::LaserPlaneMapTempTable> table;
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
};
GateScene MakeGateScene(int bigN, const std::vector<float>& tgtDeltas) {
    GateScene s;
    std::vector<SynthEntry> row100, row101, row200, row201;
    for (int i = 0; i < bigN; ++i) {
        const float xL = 100.0f + 2.0f * i, uR = 20.0f * i;
        ((i % 2) ? row101 : row100).push_back({xL, uR, 1});
        s.left.emplace_back(xL, RowY(i % 2 ? 101 : 100));
        s.trk.push_back(1);
        s.right.emplace_back(uR + 0.2f, RowY(i % 2 ? 101 : 100));
    }
    for (size_t i = 0; i < tgtDeltas.size(); ++i) {
        const float xL = 500.0f + 2.0f * static_cast<float>(i);
        const float uR = 2000.0f + 20.0f * static_cast<float>(i);
        ((i % 2) ? row201 : row200).push_back({xL, uR, 1});
        s.left.emplace_back(xL, RowY(i % 2 ? 201 : 200));
        s.trk.push_back(2);
        s.right.emplace_back(uR + tgtDeltas[i], RowY(i % 2 ? 201 : 200));
    }
    s.table = MakeTableFromRows({{100, row100}, {101, row101},
                                 {200, row200}, {201, row201}});
    return s;
}
LaserMatchScanParamsV5 GateP5() {
    LaserMatchScanParamsV5 p = BaseP5();
    p.res_gate = true;            // 生产定稿开
    p.res_gate_k = 4.0f;          // 生产值（代码默认 2.7 与 config 无关）
    p.res_clean_thresh = 1.0f;    // τ 下限 0.4×1.0=0.4
    return p;                     // res_clean 保持默认 false＝隔离门行为
}
} // namespace

// 15. V5ResGateDropsSmallSegOutlier —— 4 点小段 1 点 |res|=3.0>τ=0.8 → 剔该点，
//     幸存 3≥2 段保留（逐点门只对小段生效）。
TEST(LaserMatchV5, V5ResGateDropsSmallSegOutlier) {
    GateScene s = MakeGateScene(60, {0.2f, 0.2f, 0.2f, 3.0f});
    LaserMatchScanParamsV5 p5 = GateP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.resGateDroppedPts, 1);          // 只剔超 τ 的 1 点
    EXPECT_EQ(r.survivorDroppedSegCount, 0);    // 幸存 3≥2 不弃段
    EXPECT_EQ(r.matchedCount, 60 + 3);
    EXPECT_NEAR(r.resGateTauPx, 0.8f, 0.05f);   // τ=max(4×0.2, 0.4)
    for (float res : r.matched_res) EXPECT_LT(std::fabs(res), 1.0f);
}

// 16. V5ResGateSurvivorDropsSeg —— 2 点小段全超 τ → 幸存 0<2 整段弃。
TEST(LaserMatchV5, V5ResGateSurvivorDropsSeg) {
    GateScene s = MakeGateScene(60, {3.0f, 3.0f});
    LaserMatchScanParamsV5 p5 = GateP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.resGateDroppedPts, 2);
    EXPECT_EQ(r.survivorDroppedSegCount, 1);    // 幸存 0<2 → kDropSurvivor
    EXPECT_EQ(r.matchedCount, 60);              // 只剩大段
}

// 17. V5ResGateExemptsAbove4 —— 5 赢家卡段（>small 上限）豁免逐点门：
//     |res|=3.0 离群点不被门剔（⑪也关）——门只管 ≤4 的段（261113 死语句修复锚）。
TEST(LaserMatchV5, V5ResGateExemptsAbove4) {
    GateScene s = MakeGateScene(60, {0.2f, 0.2f, 0.2f, 0.2f, 3.0f});
    LaserMatchScanParamsV5 p5 = GateP5();
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.resGateDroppedPts, 0);          // 5 卡段豁免
    EXPECT_EQ(r.survivorDroppedSegCount, 0);
    EXPECT_EQ(r.matchedCount, 60 + 5);          // 离群点保留在输出
    float mx = 0.0f;
    for (float res : r.matched_res) mx = std::max(mx, std::fabs(res));
    EXPECT_NEAR(mx, 3.0f, 0.05f);               // 3.0 离群确实在
}

// 18. V5ResGateOffPassesThrough —— res_gate=false：无门无 τ，超差点全放行
//     （幸存规则仍在跑但无事发生）——关态行为锚。
TEST(LaserMatchV5, V5ResGateOffPassesThrough) {
    GateScene s = MakeGateScene(60, {0.2f, 0.2f, 0.2f, 3.0f});
    LaserMatchScanParamsV5 p5 = BaseP5();       // res_gate 默认 false
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.resGateDroppedPts, 0);
    EXPECT_EQ(r.survivorDroppedSegCount, 0);
    EXPECT_EQ(r.matchedCount, 60 + 4);          // 全放行
    EXPECT_LT(r.resGateTauPx, 0.0f);            // 无 τ 表
}

// ============================================================================
// 19+. V5LineClean* —— 补丁⑰·线连续性清洗（三轮代理方案 v3·261118）：
//     按线对赢家卡残差序列 r(idx) 滑窗 Hampel（win=5/min_run=5/σ_floor=0.3 常量）。
//     构造原理同 δ 即 cardRes：⑰的检测对象就是本组构造的 δ 序列。
// ============================================================================

// 一条竖直激光线：row r0..r0+n-1 各一点（同 lid/同 track），uR 随行序缓变
// （模拟真实线形——⑰只看高频形貌）。deltas 不足处补 0。rows 留存供多线并表。
struct LineScene {
    std::shared_ptr<const calib::LaserPlaneMapTempTable> table;
    std::vector<std::pair<int, std::vector<SynthEntry>>> rows;
    std::vector<cv::Point2f> left, right;
    std::vector<int> trk;
};
LineScene MakeLineScene(int r0, int n, const std::vector<float>& deltas,
                        float slope = 0.05f, int lid = 1, int track = 1,
                        float xL = 100.0f, float uR0 = 20.0f) {
    LineScene s;
    for (int i = 0; i < n; ++i) {
        const int r = r0 + i;
        const float uR = uR0 + slope * static_cast<float>(i);
        s.rows.push_back({r, {{xL, uR, lid}}});
        s.left.emplace_back(xL, RowY(r));
        s.trk.push_back(track);
        const float d = i < static_cast<int>(deltas.size())
                            ? deltas[static_cast<size_t>(i)] : 0.0f;
        s.right.emplace_back(uR + d, RowY(r));
    }
    s.table = MakeTableFromRows(s.rows);
    return s;
}
// 多线并表（同 row 各线条目并列——MakeTableFromRows 逐条目平铺，天然支持）
LineScene MergeLineScenes(std::vector<LineScene> parts) {
    LineScene s = std::move(parts[0]);
    for (size_t k = 1; k < parts.size(); ++k) {
        s.rows.insert(s.rows.end(), parts[k].rows.begin(), parts[k].rows.end());
        s.left.insert(s.left.end(), parts[k].left.begin(), parts[k].left.end());
        s.right.insert(s.right.end(), parts[k].right.begin(), parts[k].right.end());
        s.trk.insert(s.trk.end(), parts[k].trk.begin(), parts[k].trk.end());
    }
    s.table = MakeTableFromRows(s.rows);
    return s;
}
LaserMatchScanParamsV5 LcP5() {
    LaserMatchScanParamsV5 p = BaseP5();
    p.line_clean = true;
    return p;
}
LaserMatchScanCpuResultV5 RunLc(const LineScene& s, const LaserMatchScanParamsV5& p5) {
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    EXPECT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    EXPECT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(p5);
    return v5.Execute(front);
}

// 19. V5LineCleanOffDefault —— OFF（默认参）：δ=5 坏点也直通，计数恒 0。
TEST(LaserMatchV5, V5LineCleanOffDefault) {
    LineScene s = MakeLineScene(100, 11, {0, 0, 0, 0, 0, 5.0f, 0, 0, 0, 0, 0});
    LaserMatchScanCpuResultV5 r = RunLc(s, BaseP5());       // line_clean 默认 false
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 0);
    EXPECT_EQ(r.lineCleanFallbackLines, 0);
    EXPECT_EQ(r.matchedCount, 11);
}

// 20. V5LineCleanDropsOutlier —— 中点 δ=5：窗 med≈0/MAD→σ_floor=0.3/门 1.05 → 剔 1
//     （21 点剔 1＝4.8%＜帽 5%——不触发回退）。
TEST(LaserMatchV5, V5LineCleanDropsOutlier) {
    std::vector<float> d(21, 0.0f);
    d[10] = 5.0f;
    LineScene s = MakeLineScene(100, 21, d);
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 1);
    EXPECT_EQ(r.lineCleanFallbackLines, 0);
    EXPECT_EQ(r.matchedCount, 20);
    for (float res : r.matched_res)
        EXPECT_LT(std::fabs(res), 1.0f) << "被剔点不应残留在输出";
}

// 21. V5LineCleanCleanDataUnchanged —— 干净数据上 ON 与 OFF 逐位一致（零误动锚）。
TEST(LaserMatchV5, V5LineCleanCleanDataUnchanged) {
    LineScene s = MakeLineScene(100, 11, {});
    calib::LaserMatchScanSplitGpuV3 gpu(BaseP5().v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5off(BaseP5());
    calib::LaserMatchScanSplitCpuV5 v5on(LcP5());
    LaserMatchScanCpuResultV5 off = v5off.Execute(front);
    LaserMatchScanCpuResultV5 on = v5on.Execute(front);
    ASSERT_TRUE(off.success && on.success);
    ExpectIdentical(off, on, "lc-clean");
    EXPECT_EQ(on.lineCleanDroppedPts, 0);
}

// 22. V5LineCleanSlopeImmune —— 线性斜坡 δ=0.5·i（旋转漂移形态）：滑窗中值吸收
//     斜坡，全存活（⑰动机的直接断言——同数据下⑪段中位口径两端 dev≈2.5px）。
TEST(LaserMatchV5, V5LineCleanSlopeImmune) {
    std::vector<float> ramp;
    for (int i = 0; i < 11; ++i) ramp.push_back(0.5f * static_cast<float>(i));
    LineScene s = MakeLineScene(100, 11, ramp);
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 0);
    EXPECT_EQ(r.lineCleanFallbackLines, 0);
    EXPECT_EQ(r.matchedCount, 11);
}

// 23. V5LineCleanKeepsSmallDev —— δ=0.5（<门 1.05）：保留（抖动门内放行）。
TEST(LaserMatchV5, V5LineCleanKeepsSmallDev) {
    LineScene s = MakeLineScene(100, 11, {0, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0});
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 0);
    EXPECT_EQ(r.matchedCount, 11);
}

// 24. V5LineCleanMinRunExempt —— 4 点短线带坏点：run<min_run=5 豁免（P2 镂空红线）。
TEST(LaserMatchV5, V5LineCleanMinRunExempt) {
    LineScene s = MakeLineScene(100, 4, {0, 5.0f, 0, 0});
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 0);
    EXPECT_EQ(r.matchedCount, 4);
}

// 25. V5LineCleanGapSplitsRuns —— 同 track 两 run（row 100-110 坏点／120-130 干净，
//     gap=10 行＞3）：统计不跨缝——run1 剔 1、run2 零误伤（若不切缝 med 被 5.0 污染）；
//     22 点剔 1＝4.5%＜帽——不回退。
TEST(LaserMatchV5, V5LineCleanGapSplitsRuns) {
    std::vector<float> d1(11, 0.0f);
    d1[5] = 5.0f;
    LineScene a = MakeLineScene(100, 11, d1);
    LineScene b = MakeLineScene(120, 11, {});
    LineScene s = MergeLineScenes({std::move(a), std::move(b)});   // 同 track=1
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 1);
    EXPECT_EQ(r.lineCleanFallbackLines, 0);
    EXPECT_EQ(r.matchedCount, 21);
}

// 26. V5LineCleanRateCapFallback —— 20 点中 5 个 δ=5（25%＞帽 5%）：整线回退
//     全放行（故障安全——防串线/系统性误杀成洞）。
TEST(LaserMatchV5, V5LineCleanRateCapFallback) {
    std::vector<float> d(20, 0.0f);
    for (int i : {2, 6, 10, 14, 18}) d[static_cast<size_t>(i)] = 5.0f;
    LineScene s = MakeLineScene(100, 20, d);
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanFallbackLines, 1);
    EXPECT_EQ(r.lineCleanDroppedPts, 0);
    EXPECT_EQ(r.matchedCount, 20);          // 全撤回放行
}

// 27. V5LineCleanMultiLidIsolated —— 双线（lid1 坏点/lid2 干净）：CSR 桶隔离，
//     只剔线 1 的坏点（21 点剔 1＝4.8%＜帽）。
TEST(LaserMatchV5, V5LineCleanMultiLidIsolated) {
    std::vector<float> d(21, 0.0f);
    d[10] = 5.0f;
    LineScene a = MakeLineScene(100, 21, d);
    LineScene b = MakeLineScene(100, 21, {}, 0.05f, 2, 2, 200.0f, 400.0f);
    LineScene s = MergeLineScenes({std::move(a), std::move(b)});
    LaserMatchScanCpuResultV5 r = RunLc(s, LcP5());
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.lineCleanDroppedPts, 1);
    EXPECT_EQ(r.lineCleanFallbackLines, 0);
    EXPECT_EQ(r.matchedCount, 41);
    int lid1 = 0, lid2 = 0;
    for (int i = 0; i < r.matchedCount; ++i)
        (LidAt(r, i) == 1) ? ++lid1 : ++lid2;
    EXPECT_EQ(lid1, 20);
    EXPECT_EQ(lid2, 21);
}

// 28. V5LineCleanDeterminism —— ON（含坏点）同 front 双跑逐位一致
//     （21 点剔 1＜帽——走真实剔路径）。
TEST(LaserMatchV5, V5LineCleanDeterminism) {
    std::vector<float> d(21, 0.0f);
    d[10] = 5.0f;
    d[20] = 0.5f;
    LineScene s = MakeLineScene(100, 21, d);
    calib::LaserMatchScanSplitGpuV3 gpu(LcP5().v3);
    ASSERT_TRUE(gpu.SetTempTable(s.table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, s.left, s.trk, s.right);
    ASSERT_TRUE(front.success) << front.message;
    calib::LaserMatchScanSplitCpuV5 v5(LcP5());
    LaserMatchScanCpuResultV5 a = v5.Execute(front);
    LaserMatchScanCpuResultV5 b = v5.Execute(front);
    ASSERT_TRUE(a.success && b.success);
    ExpectIdentical(a, b, "lc-det");
    EXPECT_EQ(a.lineCleanDroppedPts, b.lineCleanDroppedPts);
}

// 29. V5LineCleanParamsValidate —— 参数域校验。
TEST(LaserMatchV5, V5LineCleanParamsValidate) {
    auto bad = [](auto mutate) {
        LaserMatchScanParamsV5 p = BaseP5();
        mutate(p);
        EXPECT_THROW(p.validate(), std::invalid_argument);
    };
    bad([](auto& p) { p.line_clean = true; p.line_clean_k = 0.0f; });
    bad([](auto& p) { p.line_clean_gap_rows = 0; });
    bad([](auto& p) { p.line_clean_cap_ratio = 0.0f; });
    bad([](auto& p) { p.line_clean_cap_ratio = 1.5f; });
    LcP5().validate();     // 合法值不抛
    SUCCEED();
}

} // namespace
