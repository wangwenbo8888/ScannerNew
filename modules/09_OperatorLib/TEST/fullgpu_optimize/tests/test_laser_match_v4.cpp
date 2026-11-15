/**
 * @file test_laser_match_v4.cpp
 * @brief laser_match_scan_v4 CPU 后段（V4 新机制：质量改选/自适应窗/弃权/深度门）
 *        gtest —— 前段复用 LaserMatchScanSplitGpuV3 真跑，同一 FrontV3 喂
 *        V3 参照后段与 V4 被测后段做对照。
 *
 * 用例清单（对照规格）：
 *   V4L0IdentityAllOff      v4 新机制全关＋注入温表 → 与 V3 后段逐位一致
 *   V4L0IdentityNoTable     机制开但不注入温表（自动降级全关）→ 仍逐位一致
 *   V4QualityFlip           争议段（票平局）质量分可分 → V3 取 lid1、V4 改选 lid2
 *   V4AbstainNotSeparable   争议段两线 res 相等分不开 → V4 弃段，V3 正常输出
 *   V4AdaptiveWindowDrop    双线收敛 w_eff 收缩 → lid1 卡被 mask＋单线行无幸存卡
 *   V4DepthGateAutoOffNoQ   card_depth_gate=true 但未 SetStereoQ → 行为与关一致
 *   V4DepthGateDrops        注入 4×4 Q 使某线 Z 落界外 → 该线卡被 mask 弃点
 *   V4QualityCntVeto        单票 lid（res 极小）被 cnt_veto=2 否决不得胜出
 *   V4RealDataParityL0      真实数据（缺席 GTEST_SKIP）：机制全关 V3/V4 逐位一致
 *   V4ResultSuperset        V4 结果可切片赋给 V3 结果（编译期验证继承兼容）
 *
 * 真实数据缺席 → GTEST_SKIP（date_in / out_golden_base 未入版本库，惯例同
 * test_laser_match_v3_split.cpp）。
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "laser_match_scan_v3_split.h"   // LaserMatchScanSplitGpuV3 / CpuV3 / FrontV3
#include "laser_match_scan_v4.h"         // V4 后段（头自带 ParamsV4/ResultV4，带出 V3 族）
#include "cmtt_table_loader.h"           // 真数据 CMTT 装载

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace {

// ============================================================================
// 0. 静态环境守卫：仿 v3 拆分测试——GPU 侧 JMW_V3* 敏感变量清空（本 TU 真跑 V3
//    前段），并追加 V4 新机制的 JMW_LMV4* 开关清空。_putenv("VAR=") 在 MSVC 删除
//    变量；本 TU 静态对象构造先于 gtest main，亦先于任何算子构造。
// ============================================================================
struct SplitEnvGuard {
    SplitEnvGuard() {
        for (const char* v : {"JMW_V3GRAPH", "JMW_V3BISECT", "JMW_V3ANCHOR", "JMW_V3_SEGCAP",
                              "JMW_V3FULLGPU", "JMW_V3GPU", "JMW_V3PAIR_AB",
                              "JMW_NOTBD", "JMW_V3TIME", "JMW_K1B_SKIPSTAMPS",
                              "JMW_LMV4_QUALITY", "JMW_LMV4_ADAPTWIN",
                              "JMW_LMV4_ABSTAIN", "JMW_LMV4_DEPTHGATE"}) {
            std::string s = std::string(v) + "=";
            _putenv(s.c_str());
        }
        // 【v4 默认开·2026-11-07】JMW_LMV4 不再删除而是显式置 0：harness 的
        // lmV4Serial 是进程级一次性冻结静态量，删除会被解读为"默认开"并污染
        // 全进程；与他 TU 的静态守卫无论先后都收敛到 0（v3 臂）。
        _putenv("JMW_LMV4=0");
    }
} g_v4EnvGuard;

using calib::LaserMatchScanParamsV3;
using calib::LaserMatchScanParamsV4;
using calib::LaserMatchScanSplitGpuV3;
using calib::LaserMatchScanSplitCpuV3;
using calib::LaserMatchScanSplitCpuV4;
using calib::LaserMatchSplitFrontV3;
using calib::LaserMatchScanCpuResultV3;
using calib::LaserMatchScanCpuResultV4;

// ── 合成小温表（照抄 v3 测试）：rowStep=0.5；表行 r∈{100,101,102}（y=r*0.5）──
// 每行 3 条：xL=100/200/300 → lid=1/2/3；uR=xL-100（视差 100）
std::shared_ptr<const calib::LaserPlaneMapTempTable> MakeSynthTable() {
    auto table = std::make_shared<calib::LaserPlaneMapTempTable>();
    calib::LaserPlaneMap m;
    m.temperature = 25.0;
    const float rowStep = 0.5f;
    std::vector<cv::Vec4f> entries;
    for (int r = 100; r <= 102; ++r) {
        const float yL = static_cast<float>(r) * rowStep;
        for (int k = 0; k < 3; ++k)
            entries.push_back({100.0f + 100.0f * k, yL, k * 100.0f, static_cast<float>(k + 1)});
        // uR = xL-100：xL=100→uR=0(lid1)；200→100(lid2)；300→200(lid3)
    }
    m.leftToRightMap = cv::Mat(static_cast<int>(entries.size()), 1, CV_32FC4, entries.data()).clone();
    m.totalPairs = static_cast<int>(entries.size());
    table->table[25.0] = std::move(m);
    return table;
}

// ── 自定义行布局温表：rowStep=0.5；每行可放任意条目（同 row 同 xL 多 lid 合法，
//    表按 (row,x,lid) 组织——用于构造双线收敛/单线行等物理场景）──
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

// 链内配置（同 v3 测试：config.json operators.laser_match_scan_v3）
LaserMatchScanParamsV3 ChainParams() {
    LaserMatchScanParamsV3 p;
    p.match_threshold = 7.0f;
    p.epipolar_row_step = 0.67f;
    p.vL_tolerance = 0.6f;
    p.conflict_reelect = true;
    p.segment_split = true;
    p.cascade_elim = true;
    p.cascade_gate_h = 30;
    p.cascade_gate_share = 0.4f;
    p.cascade_gate_margin = 1.3f;
    p.cascade_friend_xpx = 15.0f;
    p.cascade_friend_minrows = 3;
    p.cascade_spacing_px = 36.9f;
    p.cascade_spacing_tol = 0.55f;
    return p;
}

// 前段跑一次的便捷封装（同 v3 测试）
LaserMatchSplitFrontV3 RunFront(LaserMatchScanSplitGpuV3* gpu,
                                const std::vector<cv::Point2f>& left,
                                const std::vector<int>& ltrk,
                                const std::vector<cv::Point2f>& right,
                                const std::vector<int>& rtrk) {
    cv::cuda::GpuMat dL, dLid, dR, dRid;
    dL.upload(PointsMat(left));
    dLid.upload(IdsMat(ltrk));
    dR.upload(PointsMat(right));
    dRid.upload(IdsMat(rtrk));
    LaserMatchSplitFrontV3 front;
    gpu->Execute(dL, dLid, dR, dRid, front);
    return front;
}

// V4 全关参数（包裹同一份 v3 子参数——前后段一致性由构造保证）
LaserMatchScanParamsV4 AllOffV4(const LaserMatchScanParamsV3& p3) {
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = false;
    p4.adapt_window = false;
    p4.abstain = false;
    p4.card_depth_gate = false;
    return p4;
}

// L0 逐位一致断言：matchedCount＋四数组（连续内存 memcmp）
// V4 结果按基类切片参与比较（matched_res 等 V4 增量字段不进比较面）
void ExpectL0Identical(const LaserMatchScanCpuResultV3& r3,
                       const LaserMatchScanCpuResultV4& r4, const char* tag) {
    ASSERT_TRUE(r3.success) << tag << " V3 后段失败: " << r3.message;
    ASSERT_TRUE(r4.success) << tag << " V4 后段失败: " << r4.message;
    ASSERT_EQ(r3.matchedCount, r4.matchedCount) << tag << " matchedCount 不一致";
    const int n = r3.matchedCount;
    if (n == 0) return;

    struct FieldCheck {
        const char* name;
        const cv::Mat* m3;
        const cv::Mat* m4;
        size_t bytesPerElem;
    };
    ASSERT_TRUE(r3.matched_left.isContinuous() && r4.matched_left.isContinuous());
    ASSERT_TRUE(r3.matched_right.isContinuous() && r4.matched_right.isContinuous());
    ASSERT_TRUE(r3.matched_line_ids.isContinuous() && r4.matched_line_ids.isContinuous());
    ASSERT_TRUE(r3.matched_track_ids.isContinuous() && r4.matched_track_ids.isContinuous());
    const FieldCheck checks[] = {
        {"matched_left",      &r3.matched_left,      &r4.matched_left,      sizeof(cv::Point2f)},
        {"matched_right",     &r3.matched_right,     &r4.matched_right,     sizeof(cv::Point2f)},
        {"matched_line_ids",  &r3.matched_line_ids,  &r4.matched_line_ids,  sizeof(int)},
        {"matched_track_ids", &r3.matched_track_ids, &r4.matched_track_ids, sizeof(int)},
    };
    for (const auto& c : checks) {
        ASSERT_EQ(c.m3->type(), c.m4->type()) << tag << " " << c.name << " 类型不一致";
        ASSERT_EQ(c.m3->total(), c.m4->total()) << tag << " " << c.name << " 元素数不一致";
        ASSERT_EQ(c.m3->total(), static_cast<size_t>(n)) << tag << " " << c.name << " 元素数≠matchedCount";
        EXPECT_EQ(0, std::memcmp(c.m3->ptr(), c.m4->ptr(), static_cast<size_t>(n) * c.bytesPerElem))
            << tag << " " << c.name << " 逐位不一致 (n=" << n << ")";
    }
}

// 深度门 Q 矩阵：Z = Q23 / (Q32*d + Q33)（d＝卡视差；Q 索引为 0 起 row,col）
cv::Mat MakeDepthQ(double q23, double q32, double q33) {
    cv::Mat q = cv::Mat::zeros(4, 4, CV_64F);
    q.at<double>(2, 3) = q23;
    q.at<double>(3, 2) = q32;
    q.at<double>(3, 3) = q33;
    return q;
}

// ── 回放数据装载（同 test_laser_match_v3_split.cpp）──
struct ReplayFrame {
    std::vector<cv::Point2f> leftPts, rightPts;
    std::vector<int> leftTracks;
    std::vector<int> rightTracksZero;   // GPU 侧右 lid 只写不读 → 全 0 安全
};

bool LoadCsv(const std::string& path, std::vector<cv::Point2f>* pts, std::vector<int>* tracks) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double x = 0, y = 0; int t = 0;
        if (tracks) {
            if (!(ss >> x >> y >> t)) return false;
            tracks->push_back(t);
        } else {
            if (!(ss >> x >> y)) return false;
        }
        pts->emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
    return true;
}

std::string GoldenDir(int cycle) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/out_golden_base/cycle_%03d", SCR_SOURCE_DIR, cycle);
    return buf;
}

bool LoadFrame(int cycle, ReplayFrame* fr) {
    const std::string dir = GoldenDir(cycle);
    fr->leftPts.clear(); fr->rightPts.clear(); fr->leftTracks.clear();
    if (!LoadCsv(dir + "/L3-interp_L.csv", &fr->leftPts, nullptr)) return false;
    if (!LoadCsv(dir + "/L3-interp_R.csv", &fr->rightPts, nullptr)) return false;
    std::vector<cv::Point2f> dummy;
    if (!LoadCsv(dir + "/L3t-tracks_L.csv", &dummy, &fr->leftTracks)) return false;
    return fr->leftPts.size() == fr->leftTracks.size() && !fr->leftPts.empty();
}

std::shared_ptr<const calib::LaserPlaneMapTempTable> LoadCmttTable() {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    std::string err;
    auto t = calib::LoadTempTableFromCmtt(bin, 22.5, err);
    if (!t) ADD_FAILURE() << "CMTT 装载失败: " << err;
    return t;
}

} // namespace

// ============================================================================
// 1. V4L0IdentityAllOff —— v4 新机制全关＋注入温表
//    物理场景：合成 3 线表（rows 100-102，lid1/2/3），左右点覆盖三线＋一个
//    noMap 点（行 500 无表）＋一个游离右点（不被消费）。
//    预期：V3 与 V4 后段对同一 FrontV3 输出完全一致（matchedCount=4，四数组逐位）。
// ============================================================================
TEST(LaserMatchV4, V4L0IdentityAllOff) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    const LaserMatchScanParamsV4 p4 = AllOffV4(p3);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    auto table = MakeSynthTable();
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);

    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.2f, 50.2f},    // track 0：行 100/100 lid2（uR=100）
        {100.0f, 50.5f}, {300.0f, 51.0f},    // track 1：行 101 lid1 / 行 102 lid3
        {250.0f, 250.0f}                     // track 2：行 500 无表 → noMap
    };
    const std::vector<int> ltrk = {0, 0, 1, 1, 2};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.3f, 50.2f},    // 命中 lid2 卡（res=0/0.3）
        {0.0f, 50.5f}, {200.5f, 51.0f},      // 命中 lid1 / lid3 卡
        {999.9f, 60.0f}                      // 游离右点（行 120 无人消费）
    };
    auto front = RunFront(&gpu, left, ltrk, right, {0, 0, 0, 0, 0});
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    auto r4 = cpu4.Execute(front);
    ExpectL0Identical(r3, r4, "L0AllOff");
    // 实测 V3＝3（非 4）：track1 两点跨 lid1/lid3 两线，段投票（1:1 平票→lid1）
    // 吸收掉 lid3 点＝V3 既有正确行为；L0 恒等以 ExpectL0Identical 为唯一判据
    EXPECT_EQ(r3.matchedCount, 3);
    EXPECT_EQ(r3.noMapLeftCount, 1);
}

// ============================================================================
// 2. V4L0IdentityNoTable —— v4 机制开但不给 CPU 后段注入温表
//    物理场景：同用例 1 的点集；V4 后段 quality/adapt/abstain 全开但未
//    SetTempTable → 实现应自动降级全关（card_depth_gate 留关：其无 Q 降级
//    路径由 V4DepthGateAutoOffNoQ 单独覆盖）。
//    预期：结果仍与 V3 后段逐位一致（温表缺席不改变 L0 行为、不崩溃）。
// ============================================================================
TEST(LaserMatchV4, V4L0IdentityNoTable) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = true;    // 机制开——靠缺表降级
    p4.adapt_window = true;
    p4.abstain = true;
    p4.card_depth_gate = false;

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);       // 不 SetTempTable / 不设温度
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);

    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.2f, 50.2f}, {100.0f, 50.5f}, {300.0f, 51.0f}, {250.0f, 250.0f}
    };
    const std::vector<int> ltrk = {0, 0, 1, 1, 2};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.3f, 50.2f}, {0.0f, 50.5f}, {200.5f, 51.0f}, {999.9f, 60.0f}
    };
    auto front = RunFront(&gpu, left, ltrk, right, {0, 0, 0, 0, 0});
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    auto r4 = cpu4.Execute(front);
    ExpectL0Identical(r3, r4, "L0NoTable");
    EXPECT_EQ(r3.matchedCount, 3);   // 同用例 1：track1 跨线点被 V3 段投票吸收
}

// ============================================================================
// 3. V4QualityFlip —— 争议段（int 票平局）＋质量分可分 → 改选
//    物理场景：曲线收敛区——表行 100..105 每行两条目 {xL=100,lid=1,uR=0} 与
//    {xL=100,lid=2,uR=6}（两线在同 xL 处预测差 6px）；每行左点 (100,r*0.5)、
//    右点 (5,r*0.5)：lid1 卡 res=|5-0|=5、lid2 卡 res=|5-6|=1（thr=7 两卡均 HIT）。
//    同一 track → 单段，票 6:6 平局。多行复制制造票数（≥2 过 veto）。
//    预期：V3 平票取小 → winner=lid1 全输出；V4（quality 开，separate_px=1.0，
//    |5-1|=4≥1 可分）→ winner=lid2、qualityFlipCount≥1、matched_res 尺寸==matchedCount。
// ============================================================================
TEST(LaserMatchV4, V4QualityFlip) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = true;
    p4.quality_separate_px = 1.0f;
    p4.quality_cnt_veto = 2;
    p4.res_channel = true;     // 本用例断言 matched_res，显式开通道
    p4.adapt_window = false;
    p4.abstain = false;
    p4.card_depth_gate = false;

    // 表：rows 100..105 每行 {lid1:uR=0, lid2:uR=6}（收敛区两线预测差 6px）
    std::vector<std::pair<int, std::vector<SynthEntry>>> spec;
    for (int r = 100; r <= 105; ++r)
        spec.push_back({r, {{100.0f, 0.0f, 1}, {100.0f, 6.0f, 2}}});
    auto table = MakeTableFromRows(spec);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> left, right;
    for (int r = 100; r <= 105; ++r) {
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(5.0f, static_cast<float>(r) * 0.5f);
    }
    const std::vector<int> ltrk(left.size(), 0);   // 单段
    auto front = RunFront(&gpu, left, ltrk, right, std::vector<int>(right.size(), 0));
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    ASSERT_TRUE(r3.success) << r3.message;
    ASSERT_EQ(r3.matchedCount, 6) << "V3 构造前提：6 点全命中";
    for (int i = 0; i < r3.matchedCount; ++i)
        ASSERT_EQ(r3.matched_line_ids.ptr<int>()[i], 1) << "V3 平票取小 → lid1（idx=" << i << "）";

    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message;
    EXPECT_EQ(r4.matchedCount, 6) << "V4 质量改选不弃点";
    for (int i = 0; i < r4.matchedCount; ++i)
        EXPECT_EQ(r4.matched_line_ids.ptr<int>()[i], 2) << "V4 质量改选 → lid2（idx=" << i << "）";
    EXPECT_GE(r4.qualityFlipCount, 1) << "至少一段发生质量改选";
    // 用例 10 附带断言面：matched_res 尺寸==matchedCount 且数值合理（lid2 卡 |res|=1）
    ASSERT_EQ(r4.matched_res.size(), static_cast<size_t>(r4.matchedCount));
    for (size_t i = 0; i < r4.matched_res.size(); ++i) {
        EXPECT_TRUE(std::isfinite(r4.matched_res[i])) << "matched_res[" << i << "] 非有限值";
        EXPECT_NEAR(std::fabs(r4.matched_res[i]), 1.0f, 1e-3f)
            << "matched_res[" << i << "]=" << r4.matched_res[i] << " 应为 lid2 卡 |res|=1";
    }
}

// ============================================================================
// 4. V4AbstainNotSeparable —— 争议段两线预测差 2px，右点放中点 → res 相等分不开
//    物理场景：表行 100..103 每行 {xL=100,lid=1,uR=0} 与 {xL=100,lid=2,uR=2}；
//    每行左点 (100,r*0.5)、右点 (1,r*0.5)：lid1 卡 res=1、lid2 卡 res=1（相等，
//    差 0 < separate_px=1.0 不可分）。票 4:4 平局且质量分不开。
//    预期：V3 平票取小 → lid1 正常输出 4 点；V4（abstain 开）→ 该段弃：
//    matchedCount==0、abstainSegCount≥1（success 保持 true，弃段是正常输出路径）。
// ============================================================================
TEST(LaserMatchV4, V4AbstainNotSeparable) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = true;
    p4.quality_separate_px = 1.0f;
    p4.quality_cnt_veto = 2;
    p4.res_channel = true;     // 本用例断言 matched_res，显式开通道
    p4.adapt_window = false;
    p4.abstain = true;
    p4.abstain_margin = 1.3f;
    p4.abstain_share = 0.60f;
    p4.abstain_gap_factor = 2.0f;
    p4.card_depth_gate = false;

    std::vector<std::pair<int, std::vector<SynthEntry>>> spec;
    for (int r = 100; r <= 103; ++r)
        spec.push_back({r, {{100.0f, 0.0f, 1}, {100.0f, 2.0f, 2}}});
    auto table = MakeTableFromRows(spec);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> left, right;
    for (int r = 100; r <= 103; ++r) {
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(1.0f, static_cast<float>(r) * 0.5f);   // 中点：两线 res 均=1
    }
    auto front = RunFront(&gpu, left, std::vector<int>(left.size(), 0),
                          right, std::vector<int>(right.size(), 0));
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    ASSERT_TRUE(r3.success) << r3.message;
    EXPECT_EQ(r3.matchedCount, 4) << "V3 正常输出（平票取小 lid1）";

    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message << "（弃段应为正常输出路径，非失败）";
    EXPECT_EQ(r4.matchedCount, 0) << "V4 争议且分不开 → 整段弃";
    EXPECT_GE(r4.abstainSegCount, 1) << "弃权段计数≥1";
}

// ============================================================================
// 5. V4AdaptiveWindowDrop —— 双线收敛自适应窗收缩＋单线行无幸存卡
//    物理场景 A：表行 100..104 每行 {lid1:uR=0, lid2:uR=6}（预测差 6px →
//    w_eff=max(6/2, floor=2)=3px）；右点 (5,r*0.5)：lid1 卡 res=5>3 被 mask、
//    lid2 卡 res=1≤3 幸存。物理场景 B：表行 120 只有 {lid1:uR=0} 条目（lid2
//    该行缺失），右点 (5,60.0)：唯一卡 res=5>w_eff → 无幸存卡。
//    全部同一 track → 单段（w_eff 按段内竞争线对间距生效——单线行同受收缩窗）。
//    预期：V3（thr=7 全 HIT）输出 6 点；V4：windowShrinkDropped≥1、行 120
//    adaptNoCardPts≥1 且该点不输出、双线行 winner=lid2 → matchedCount==5。
// ============================================================================
TEST(LaserMatchV4, V4AdaptiveWindowDrop) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = true;
    p4.quality_separate_px = 1.0f;
    p4.quality_cnt_veto = 2;
    p4.res_channel = true;     // 本用例断言 matched_res，显式开通道
    p4.adapt_window = true;
    p4.adapt_window_floor = 2.0f;
    p4.abstain = false;
    p4.card_depth_gate = false;

    std::vector<std::pair<int, std::vector<SynthEntry>>> spec;
    for (int r = 100; r <= 104; ++r)
        spec.push_back({r, {{100.0f, 0.0f, 1}, {100.0f, 6.0f, 2}}});
    spec.push_back({120, {{100.0f, 0.0f, 1}}});   // 单线行：仅 lid1
    auto table = MakeTableFromRows(spec);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> left, right;
    std::vector<int> rows = {100, 101, 102, 103, 104, 120};
    for (int r : rows) {
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(5.0f, static_cast<float>(r) * 0.5f);
    }
    auto front = RunFront(&gpu, left, std::vector<int>(left.size(), 0),
                          right, std::vector<int>(right.size(), 0));
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    ASSERT_TRUE(r3.success) << r3.message;
    EXPECT_EQ(r3.matchedCount, 6) << "V3 对照：thr=7 全 HIT（含行 120 的 lid1）";

    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message;
    EXPECT_GE(r4.windowShrinkDropped, 1) << "lid1 卡（res=5>w_eff=3）应被收缩窗 mask";
    EXPECT_GE(r4.adaptNoCardPts, 1) << "行 120 唯一卡被 mask → 无幸存卡计数";
    EXPECT_EQ(r4.matchedCount, 5) << "双线行 5 点输出，行 120 点被弃";
    for (int i = 0; i < r4.matchedCount; ++i)
        EXPECT_EQ(r4.matched_line_ids.ptr<int>()[i], 2) << "幸存卡全部 lid2（idx=" << i << "）";
    // 行 120 点不输出：输出左点 y 不含 60.0
    for (int i = 0; i < r4.matchedCount; ++i)
        EXPECT_NE(r4.matched_left.ptr<cv::Point2f>()[i].y, 60.0f) << "行 120 点不应输出";
}

// ============================================================================
// 6. V4DepthGateAutoOffNoQ —— card_depth_gate=true 但未 SetStereoQ
//    物理场景：同用例 1 点集；三路对照：V3 后段 / V4 全关 / V4 仅开深度门但无 Q。
//    预期：无 Q 时深度门自动关闭——三者结果逐位一致、cardDepthDropped==0、不崩溃。
// ============================================================================
TEST(LaserMatchV4, V4DepthGateAutoOffNoQ) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4gate = AllOffV4(p3);
    p4gate.card_depth_gate = true;             // 开门但不 SetStereoQ
    const LaserMatchScanParamsV4 p4off = AllOffV4(p3);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4off(p4off);
    LaserMatchScanSplitCpuV4 cpu4gate(p4gate);
    auto table = MakeSynthTable();
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4off.SetTempTable(table));
    cpu4off.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4gate.SetTempTable(table));
    cpu4gate.SetCurrentTemperature(25.0);
    // cpu4gate 刻意不调 SetStereoQ

    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.2f, 50.2f}, {100.0f, 50.5f}, {300.0f, 51.0f}, {250.0f, 250.0f}
    };
    const std::vector<int> ltrk = {0, 0, 1, 1, 2};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.3f, 50.2f}, {0.0f, 50.5f}, {200.5f, 51.0f}, {999.9f, 60.0f}
    };
    auto front = RunFront(&gpu, left, ltrk, right, {0, 0, 0, 0, 0});
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    auto rOff = cpu4off.Execute(front);
    auto rGate = cpu4gate.Execute(front);
    ExpectL0Identical(r3, rOff, "L0AllOff(对照)");
    ExpectL0Identical(r3, rGate, "DepthGateNoQ");
    EXPECT_EQ(rGate.cardDepthDropped, 0) << "无 Q 深度门自动关闭，零丢弃";
}

// ============================================================================
// 7. V4DepthGateDrops —— 注入 4×4 Q 使某线重建深度 Z 落界外 → 卡被 mask
//    物理场景：Q 构造 Z=Q23/(Q32*d+Q33)，取 Q23=100000,Q32=1,Q33=0 → Z=100000/d
//    （d＝卡视差；本构造中表视差 xL-uR 与实际对视差 xL-hitRx 相同，实现取任一
//    口径结论一致）。
//      段 A（track0）：表行 100..102 {xL=100,lid=1,uR=0}，右点 x=0 → res=0，
//        d=100 → Z=1000 ∈[200,1500] → 保留；
//      段 B（track1）：表行 110..112 {xL=100,lid=2,uR=90}，右点 x=90 → res=0，
//        d=10 → Z=10000 >1500 → 越界丢弃。
//    预期：V3 两段各胜出（lid1/lid2）共 6 点；V4 段 B 全卡被深度门 mask →
//    cardDepthDropped≥1、matchedCount==3、输出线号全为 lid1。
// ============================================================================
TEST(LaserMatchV4, V4DepthGateDrops) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4 = AllOffV4(p3);
    p4.card_depth_gate = true;
    p4.card_depth_min = 200.0f;
    p4.card_depth_max = 1500.0f;

    std::vector<std::pair<int, std::vector<SynthEntry>>> spec;
    for (int r = 100; r <= 102; ++r) spec.push_back({r, {{100.0f, 0.0f, 1}}});    // Z=1000 保留
    for (int r = 110; r <= 112; ++r) spec.push_back({r, {{100.0f, 90.0f, 2}}});   // Z=10000 丢弃
    auto table = MakeTableFromRows(spec);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);
    cpu4.SetStereoQ(MakeDepthQ(100000.0, 1.0, 0.0));   // Z=100000/d

    std::vector<cv::Point2f> left, right;
    std::vector<int> ltrk;
    for (int r = 100; r <= 102; ++r) {   // 段 A：track0，右点 x=0（d=100→Z=1000）
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(0.0f, static_cast<float>(r) * 0.5f);
        ltrk.push_back(0);
    }
    for (int r = 110; r <= 112; ++r) {   // 段 B：track1，右点 x=90（d=10→Z=10000）
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(90.0f, static_cast<float>(r) * 0.5f);
        ltrk.push_back(1);
    }
    auto front = RunFront(&gpu, left, ltrk, right, std::vector<int>(right.size(), 0));
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    ASSERT_TRUE(r3.success) << r3.message;
    EXPECT_EQ(r3.matchedCount, 6) << "V3 无深度门：两段各 3 点全输出";

    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message;
    EXPECT_GE(r4.cardDepthDropped, 1) << "段 B 卡（Z=10000>1500）应被深度门 mask";
    EXPECT_EQ(r4.matchedCount, 3) << "段 B 点全部被弃，仅段 A 输出";
    for (int i = 0; i < r4.matchedCount; ++i)
        EXPECT_EQ(r4.matched_line_ids.ptr<int>()[i], 1) << "幸存段 A 线号 lid1（idx=" << i << "）";
}

// ============================================================================
// 8. V4QualityCntVeto —— 单票 lid 不得胜出（quality_cnt_veto=2）
//    物理场景（争议段版）：rows 100..101 每行 {xL=100,lid1:uR=5, lid2:uR=0}，
//    row 102 仅 {lid3:uR=4.6}；右点全部 x=4.5 → res：lid1=0.5、lid2=4.5、
//    lid3=0.1。同 track 单段，票 lid1=2 / lid2=2 / lid3=1 → W=2,S=2 平局
//    争议（W<1.3×S）。V3 平票取小 → winner=lid1（row102 点无 lid1 卡不输出）。
//    V4 质量：lid3 仅 1 卡 < veto=2 被否决（尽管 res 最小 0.1）；
//    lid1(0.5) vs lid2(4.5) 差 4≥separate_px → 改选/保持 lid1。
//    断言：vetoCount≥1；输出绝不出现 lid3（被否决者）；res 全有限。
// ============================================================================
TEST(LaserMatchV4, V4QualityCntVeto) {
    LaserMatchScanParamsV3 p3;
    p3.match_threshold = 7.0f;
    p3.vL_tolerance = 0.5f;
    LaserMatchScanParamsV4 p4;
    p4.v3 = p3;
    p4.quality_elect = true;
    p4.quality_separate_px = 1.0f;
    p4.quality_cnt_veto = 2;
    p4.res_channel = true;     // 本用例断言 matched_res，显式开通道
    p4.adapt_window = false;
    p4.abstain = true;      // 允许弃路径（否决后无可选候选时）
    p4.card_depth_gate = false;

    std::vector<std::pair<int, std::vector<SynthEntry>>> spec;
    for (int r = 100; r <= 101; ++r)
        spec.push_back({r, {{100.0f, 5.0f, 1}, {100.0f, 0.0f, 2}}});
    spec.push_back({102, {{100.0f, 4.6f, 3}}});   // 单票 lid3（res 最小但 cnt=1）
    auto table = MakeTableFromRows(spec);

    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> left, right;
    for (int r = 100; r <= 102; ++r) {
        left.emplace_back(100.0f, static_cast<float>(r) * 0.5f);
        right.emplace_back(4.5f, static_cast<float>(r) * 0.5f);
    }
    auto front = RunFront(&gpu, left, std::vector<int>(left.size(), 0),
                          right, std::vector<int>(right.size(), 0));
    ASSERT_TRUE(front.success) << front.message;

    auto r3 = cpu3.Execute(front);
    ASSERT_TRUE(r3.success) << r3.message;
    ASSERT_EQ(r3.matchedCount, 2) << "V3 构造前提：平票→lid1，row102 点无 lid1 卡不输出";

    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message;
    EXPECT_GE(r4.singleVoteVetoCount, 1) << "lid3（1 卡<2）应触发单票否决计数";
    for (int i = 0; i < r4.matchedCount; ++i)
        EXPECT_NE(r4.matched_line_ids.ptr<int>()[i], 3)
            << "单票 lid3 不得胜出（idx=" << i << "）";
    ASSERT_EQ(r4.matched_res.size(), static_cast<size_t>(r4.matchedCount));
    for (size_t i = 0; i < r4.matched_res.size(); ++i)
        EXPECT_TRUE(std::isfinite(r4.matched_res[i])) << "matched_res[" << i << "] 非有限值";
}

// ============================================================================
// 9. V4RealDataParityL0 —— 真实数据 L0 逐位一致（缺席 GTEST_SKIP）
//    物理场景：真 CMTT 温表（22.5 就近档）＋golden L3-interp CSV 造点左右点
//    （同 v3 测试回放模式）；链内配置（ChainParams），v4 机制全关，V3/V4
//    双后段对同一 FrontV3 逐位比对（cycle_000..002）。
//    预期：真实规模数据下 L0 恒等（V4 全关不改变既有行为）。
// ============================================================================
TEST(LaserMatchV4, V4RealDataParityL0) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good())
        GTEST_SKIP() << "数据缺席: " << bin;
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    const LaserMatchScanParamsV3 p3 = ChainParams();
    const LaserMatchScanParamsV4 p4 = AllOffV4(p3);
    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV3 cpu3(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(22.5);
    ASSERT_TRUE(cpu4.SetTempTable(table));
    cpu4.SetCurrentTemperature(22.5);

    int frames = 0;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (!LoadFrame(c, &fr)) continue;
        fr.rightTracksZero.assign(fr.rightPts.size(), 0);
        auto front = RunFront(&gpu, fr.leftPts, fr.leftTracks, fr.rightPts, fr.rightTracksZero);
        ASSERT_TRUE(front.success) << "cycle_" << c << " front 失败: " << front.message;
        auto r3 = cpu3.Execute(front);
        auto r4 = cpu4.Execute(front);
        ExpectL0Identical(r3, r4, ("chain[cycle_" + std::to_string(c) + "]").c_str());
        ++frames;
    }
    if (frames == 0) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
}

// ============================================================================
// 10. V4ResultSuperset —— V4 结果可切片赋给 V3 结果（继承兼容编译期验证）
//     物理场景：最小可匹配场景（合成表 2 点）跑 V4 后段。
//     预期：static_assert 基类关系成立；拷贝切片与赋值切片后 V3 基面字段
//     （matchedCount/数组）与 V4 原值逐位一致。
// ============================================================================
TEST(LaserMatchV4, V4ResultSuperset) {
    static_assert(std::is_base_of_v<calib::LaserMatchScanCpuResultV3,
                                    calib::LaserMatchScanCpuResultV4>,
                  "LaserMatchScanCpuResultV4 必须继承 LaserMatchScanCpuResultV3");

    LaserMatchScanParamsV3 p3;
    p3.vL_tolerance = 0.5f;
    const LaserMatchScanParamsV4 p4 = AllOffV4(p3);
    LaserMatchScanSplitGpuV3 gpu(p3);
    LaserMatchScanSplitCpuV4 cpu4(p4);
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);
    ASSERT_TRUE(cpu4.SetTempTable(MakeSynthTable()));
    cpu4.SetCurrentTemperature(25.0);

    const std::vector<cv::Point2f> left = {{200.0f, 50.0f}, {200.2f, 50.2f}};
    const std::vector<cv::Point2f> right = {{100.0f, 50.0f}, {100.1f, 50.2f}};
    auto front = RunFront(&gpu, left, {0, 0}, right, {0, 0});
    ASSERT_TRUE(front.success) << front.message;
    auto r4 = cpu4.Execute(front);
    ASSERT_TRUE(r4.success) << r4.message;
    ASSERT_EQ(r4.matchedCount, 2) << "构造前提：2 点全匹配";

    // 拷贝切片（V3 base = v4result）与赋值切片——两形态都验证继承兼容
    const calib::LaserMatchScanCpuResultV3 sliced = r4;
    calib::LaserMatchScanCpuResultV3 assigned;
    assigned = r4;
    EXPECT_EQ(sliced.matchedCount, r4.matchedCount);
    EXPECT_EQ(assigned.matchedCount, r4.matchedCount);
    ASSERT_EQ(sliced.matched_left.total(), r4.matched_left.total());
    EXPECT_EQ(0, std::memcmp(sliced.matched_left.ptr(), r4.matched_left.ptr(),
                             sizeof(cv::Point2f) * sliced.matchedCount));
    ASSERT_EQ(assigned.matched_line_ids.total(), r4.matched_line_ids.total());
    EXPECT_EQ(0, std::memcmp(assigned.matched_line_ids.ptr(), r4.matched_line_ids.ptr(),
                             sizeof(int) * assigned.matchedCount));
}
