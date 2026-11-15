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
//    ≥ solo 1）能定。铁丝形态。
// ============================================================================
TEST(LaserMatchV5, V5SoloFloorSmallSeg) {
    auto table = MakeTableFromRows({{100, {{110.0f, 5.0f, 5}}}});
    std::vector<cv::Point2f> left(3, {110, RowY(100)});
    std::vector<int> trk = {9, 9, 9};
    std::vector<cv::Point2f> right = {{5.2f, RowY(100)}};

    LaserMatchScanParamsV5 p5 = BaseP5();
    EXPECT_EQ(p5.votes_floor, 4);      // 有对手下限 4
    EXPECT_EQ(p5.votes_floor_solo, 1); // 全票下限 1
    calib::LaserMatchScanSplitGpuV3 gpu(p5.v3);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    gpu.SetEmitCardRes(true);
    LaserMatchSplitFrontV3 front = RunFront(&gpu, left, trk, right);
    ASSERT_TRUE(front.success) << front.message;

    calib::LaserMatchScanSplitCpuV5 v5(p5);
    LaserMatchScanCpuResultV5 r = v5.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.matchedCount, 3);      // 3 点全输出
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
// 9. V5SeparationGateBlocks —— 补丁⑥：两线平均误差 0.10/0.24（D=2.4，
//    E=2.4 过门）但绝对差 0.14 < 0.5px → 分离门拦 → 弃。
//    （δ 取 0.24/0.10 而非 0.19/0.10——后者 D≈1.8999998 会先死在 E 门，
//     测不到分离门本身。）
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
        right.emplace_back(0.24f, RowY(r));   // lid1 残差 +0.24
        right.emplace_back(30.10f, RowY(r));  // lid2 残差 +0.10
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
    EXPECT_EQ(r.qualityJudgedCount, 0);      // 分离门拦下
    EXPECT_EQ(r.matchedCount, 0);
    EXPECT_EQ(r.exitDroppedSegCount, 1);
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

} // namespace
