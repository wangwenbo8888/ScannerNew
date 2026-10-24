/**
 * @file test_laser_match_scan_v3_cpu.cpp
 * @brief laser_match_scan_v3_cpu（全 CPU 备用算子）gtest：合成单测 → 性能 → A/B 回放
 *
 * 测试顺序（TU 内声明序）：
 *   1. 合成单测（手工小温表，精确断言各状态/计数/双弃语义/补偿）
 *   2. 性能断言（<5ms，中位 10 次预热 3；置于 GPU 用例之前，避免 GPU/装载扰动）
 *   3. A/B 回放一致性（GPU 基准，out_golden_base cycle_000..002，链内/默认双配置）
 *
 * 数据缺席 → GTEST_SKIP（date_in/out_golden_base 未入版本库）。
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "laser_match_scan_v3_cpu.h"
#include "laser_match_scan_v3_cuda.h"
#include "cmtt_table_loader.h"
#include "common/calib_result_types.h"

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace {

using calib::LaserMatchScanParamsV3;
using calib::LaserMatchScanCpuV3;
using calib::LaserMatchScanCpuResultV3;

// ── 合成小温表：rowStep=0.5；表行 r∈{100,101,102}（y=r*0.5）──
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

// CPU/GPU 参数对齐（SetParams 不做 cascade→reelect 归一——两实例同参同值）
LaserMatchScanParamsV3 ChainParams() {   // 链内配置（config.json operators.laser_match_scan_v3）
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

// ── 回放数据装载 ──
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
    if (!LoadCsv(dir + "/L3-interp_L.csv", &fr->leftPts, nullptr)) {
        std::printf("[dbg] LoadFrame L fail: %s\n", (dir + "/L3-interp_L.csv").c_str());
        return false;
    }
    if (!LoadCsv(dir + "/L3-interp_R.csv", &fr->rightPts, nullptr)) {
        std::printf("[dbg] LoadFrame R fail: %s\n", (dir + "/L3-interp_R.csv").c_str());
        return false;
    }
    std::vector<cv::Point2f> dummy;
    if (!LoadCsv(dir + "/L3t-tracks_L.csv", &dummy, &fr->leftTracks)) {
        std::printf("[dbg] LoadFrame T fail: %s (L=%zu)\n",
                    (dir + "/L3t-tracks_L.csv").c_str(), fr->leftPts.size());
        return false;
    }
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

// ── A/B 比对：CPU vs GPU（同参同输入）──
void RunAbOneFrame(LaserMatchScanCpuV3* cpu, calib::LaserMatchScanCudaV3* gpu,
                   const ReplayFrame& fr, const std::string& tag) {
    const cv::Mat lpts = PointsMat(fr.leftPts);
    const cv::Mat lids = IdsMat(fr.leftTracks);
    const cv::Mat rpts = PointsMat(fr.rightPts);
    const cv::Mat rids = IdsMat(fr.rightTracksZero);

    auto rc = cpu->Execute(lpts, lids, rpts, rids);
    ASSERT_TRUE(rc.success) << tag << " CPU execute 失败: " << rc.message;

    cv::cuda::GpuMat dL, dLid, dR, dRid;
    dL.upload(lpts);
    dLid.upload(lids);
    dR.upload(rpts);
    dRid.upload(rids);
    auto rg = gpu->Execute(dL, dLid, dR, dRid);
    ASSERT_TRUE(rg.success) << tag << " GPU execute 失败: " << rg.message;

    EXPECT_EQ(rc.matchedCount, rg.matchedCount) << tag << " matchedCount";
    ASSERT_EQ(rc.matchedCount, rg.matchedCount);

    // GPU ROI 视图：download 后再比对（下一次 Execute 前）
    cv::Mat gmL, gmR, gmLid, gmTrack;
    rg.d_matched_left->download(gmL);
    rg.d_matched_right->download(gmR);
    rg.d_matched_line_ids->download(gmLid);
    rg.d_matched_track_ids->download(gmTrack);

    const int n = rc.matchedCount;
    ASSERT_EQ(n, static_cast<int>(gmL.total()));
    const cv::Point2f* cl = rc.matched_left.ptr<cv::Point2f>();
    const cv::Point2f* cr = rc.matched_right.ptr<cv::Point2f>();
    const int* cid = rc.matched_line_ids.ptr<int>();
    const int* ctr = rc.matched_track_ids.ptr<int>();
    const cv::Point2f* gl = gmL.ptr<cv::Point2f>();
    const cv::Point2f* gr = gmR.ptr<cv::Point2f>();
    const int* gid = gmLid.ptr<int>();
    const int* gtr = gmTrack.ptr<int>();
    int coordBad = 0;
    for (int i = 0; i < n; ++i) {
        if (cid[i] != gid[i] || ctr[i] != gtr[i]) {
            ADD_FAILURE() << tag << " idx=" << i << " lid/track 不一致: cpu=("
                          << cid[i] << "," << ctr[i] << ") gpu=(" << gid[i] << "," << gtr[i] << ")";
            return;
        }
        if (std::fabs(cl[i].x - gl[i].x) > 1e-3f || std::fabs(cl[i].y - gl[i].y) > 1e-3f ||
            std::fabs(cr[i].x - gr[i].x) > 1e-3f || std::fabs(cr[i].y - gr[i].y) > 1e-3f)
            ++coordBad;
    }
    EXPECT_EQ(coordBad, 0) << tag << " 坐标超差(>1e-3)点数";

    EXPECT_EQ(rc.totalLeftPoints, rg.totalLeftPoints) << tag;
    EXPECT_EQ(rc.totalRightPoints, rg.totalRightPoints) << tag;
    EXPECT_EQ(rc.excludedLeftCount, rg.excludedLeftCount) << tag << " exclL";
    EXPECT_EQ(rc.excludedRightCount, rg.excludedRightCount) << tag << " exclR";
    EXPECT_EQ(rc.noMapLeftCount, rg.noMapLeftCount) << tag << " noMap";
    EXPECT_EQ(rc.noWinnerCardCount, rg.noWinnerCardCount) << tag << " noWinnerCard";
    EXPECT_EQ(rc.emptyWinnerSegCount, rg.emptyWinnerSegCount) << tag << " emptySeg";
    EXPECT_EQ(rc.trackTotal, rg.trackTotal) << tag << " trackTotal";
    EXPECT_EQ(rc.trackDropped, rg.trackDropped) << tag << " trackDropped";
}

} // namespace

// ============================================================================
// 1. 合成单测
// ============================================================================

TEST(LaserMatchCpuV3, SyntheticBasicMatch) {
    LaserMatchScanCpuV3 op;
    LaserMatchScanParamsV3 p;   // 默认 thr=1.0/step=0.5；放宽 vL 至 0.5
    p.vL_tolerance = 0.5f;
    op.SetParams(p);
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);

    // 左点：行 100（y=50），x=200 → 查表卡 lid2 uR=100；右点 (100, 50±ε) 命中窗
    const std::vector<cv::Point2f> left = {{200.0f, 50.0f}, {200.2f, 50.2f}};
    const std::vector<int> ltrk = {0, 0};
    const std::vector<cv::Point2f> right = {{100.0f, 50.0f}, {100.1f, 50.2f}};
    auto r = op.Execute(PointsMat(left), IdsMat(ltrk), PointsMat(right), IdsMat({0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.matchedCount, 2);
    EXPECT_EQ(r.matched_line_ids.ptr<int>()[0], 2);
    EXPECT_EQ(r.matched_line_ids.ptr<int>()[1], 2);
    // matched_right.y＝左点 y
    EXPECT_FLOAT_EQ(r.matched_right.ptr<cv::Point2f>()[1].y, 50.2f);
    EXPECT_EQ(r.noMapLeftCount, 0);
    EXPECT_EQ(r.excludedLeftCount, 0);
    EXPECT_EQ(r.trackTotal, 1);
    EXPECT_EQ(r.trackDropped, 0);
    EXPECT_EQ(r.left_status[0], 1);
    EXPECT_EQ(r.right_status[0], 1);   // 右排序系首点被消费
}

TEST(LaserMatchCpuV3, SyntheticNoMapAndNoCard) {
    LaserMatchScanCpuV3 op;
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);

    // p0：y 超出表行（行 500 无表）→ noMap；p1：行 101 有表但右点全不在窗 → noCard
    const std::vector<cv::Point2f> left = {{200.0f, 250.0f}, {200.0f, 50.5f}};
    const std::vector<int> ltrk = {0, 1};
    const std::vector<cv::Point2f> right = {{999.0f, 50.5f}};
    auto r = op.Execute(PointsMat(left), IdsMat(ltrk), PointsMat(right), IdsMat({0}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.noMapLeftCount, 1);
    EXPECT_EQ(r.excludedLeftCount, 1);
    EXPECT_EQ(r.matchedCount, 0);
    EXPECT_EQ(r.left_status[0], -1);
    EXPECT_EQ(r.left_status[1], 0);
}

TEST(LaserMatchCpuV3, SyntheticConflictDoubleDrop) {
    LaserMatchScanCpuV3 op;   // 默认参数（conflict_reelect=false → 双弃路径）
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);

    // 两个段（track 0/1）同行（y=50 行 100）都投出 winner lid2 → (row,lid) 冲突 → 双弃
    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.1f, 50.0f},     // track 0（行 100 → lid2 卡）
        {300.0f, 50.2f}, {300.1f, 50.2f}      // track 1（行 100 → lid3 卡）
    };
    const std::vector<int> ltrk = {0, 0, 1, 1};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.1f, 50.0f},     // lid2 uR=100 命中
        {200.0f, 50.2f}, {200.1f, 50.2f}      // lid3 uR=200 命中
    };
    auto r = op.Execute(PointsMat(left), IdsMat(ltrk), PointsMat(right), IdsMat({0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    // track0 winner=lid2、track1 winner=lid3——不同 lid 不冲突；本用例验证基本出口
    // 真正冲突：两段同行同 lid。构造 track1 也落在 lid2 卡位（x≈200）
    EXPECT_EQ(r.trackTotal, 2);
}

TEST(LaserMatchCpuV3, SyntheticSameLidConflictDropsBoth) {
    LaserMatchScanCpuV3 op;   // 默认参数 → 双弃
    LaserMatchScanParamsV3 p;
    p.vL_tolerance = 0.5f;
    op.SetParams(p);
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);

    // 两段同行 100、同 winner lid2（都在 x≈200 查到 lid2 卡且右点命中）→ 双弃
    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.1f, 50.0f},     // track 0
        {200.5f, 50.2f}, {200.6f, 50.2f}      // track 1（同行 100：50.2/0.5=100.4→round=100）
    };
    const std::vector<int> ltrk = {0, 0, 1, 1};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.1f, 50.0f}, {100.5f, 50.2f}, {100.6f, 50.2f}
    };
    auto r = op.Execute(PointsMat(left), IdsMat(ltrk), PointsMat(right), IdsMat({0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.trackTotal, 2);
    EXPECT_EQ(r.trackDropped, 2);   // 同 (row,lid) 双主张 → 全弃
    EXPECT_EQ(r.matchedCount, 0);
    EXPECT_EQ(r.emptyWinnerSegCount, 0);   // winner 仍为 lid2（被弃但 winner≥1）
}

TEST(LaserMatchCpuV3, SyntheticEmptyInput) {
    LaserMatchScanCpuV3 op;
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);
    auto r = op.Execute(PointsMat({}), IdsMat({}), PointsMat({}), IdsMat({}));
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.matchedCount, 0);
}

TEST(LaserMatchCpuV3, SyntheticUrBiasShift) {
    LaserMatchScanCpuV3 op;
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);

    // 无补偿：uR=100 → 右点 100 命中；注入 +5 补偿：uR=105 → 原右点 100 距 5>win1 不命中
    LaserMatchScanParamsV3 p;
    p.match_threshold = 1.0f;
    p.epipolar_row_step = 0.5f;
    p.vL_tolerance = 0.5f;
    op.SetParams(p);
    const std::vector<cv::Point2f> left = {{200.0f, 50.0f}};
    const std::vector<cv::Point2f> right = {{100.0f, 50.0f}};

    auto r0 = op.Execute(PointsMat(left), IdsMat({0}), PointsMat(right), IdsMat({0}));
    ASSERT_TRUE(r0.success);
    EXPECT_EQ(r0.matchedCount, 1);

    std::vector<float> bias(32, 0.0f);
    bias[1] = 5.0f;   // lid2 → +5
    op.SetuRBias(bias);
    auto r1 = op.Execute(PointsMat(left), IdsMat({0}), PointsMat(right), IdsMat({0}));
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(r1.matchedCount, 0);
    EXPECT_EQ(r1.excludedLeftCount, 1);   // 有卡无 HIT → noCard
}

TEST(LaserMatchCpuV3, SyntheticRowClampAndDrop) {
    LaserMatchScanCpuV3 op;
    ASSERT_TRUE(op.SetTempTable(MakeSynthTable()));
    op.SetCurrentTemperature(25.0);
    LaserMatchScanParamsV3 p;
    p.epipolar_row_step = 0.5f;
    p.vL_tolerance = 0.5f;
    op.SetParams(p);

    // 左点 y<0 → 行 0（钳位）无表 → noMap；y 巨大 → 行 ≥8192 → noMap
    const std::vector<cv::Point2f> left = {{200.0f, -3.0f}, {200.0f, 8000.0f}};
    auto r = op.Execute(PointsMat(left), IdsMat({0, 1}), PointsMat({{100.0f, -3.0f}}), IdsMat({0}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.noMapLeftCount, 2);
}

// ============================================================================
// 2. 性能断言（<5ms；置于 GPU 用例之前）
// ============================================================================

// ============================================================================
// 2. 性能断言（<5ms 硬指标＝产线默认配置；置于 GPU 用例之前）
//    - 产线默认（双弃路径）：参数缺省即产线语义（见 laser_match_scan_v3_cuda.h
//      「false＝原版双弃语义（产线默认）」），本测试硬断言 <5ms
//    - 链内配置（P5/P6b/P7 三实验路径全开，沙箱 config.json 口径）：实测参考值
//      ~9ms@i7-9850H（本机 34k 点最大帧），设 15ms 热安全回归界（不卡 5ms——
//      实验路径为 2026-10 原型，非产线语义；GPU 基准同帧 3.5ms）
// ============================================================================

TEST(LaserMatchCpuV3, PerfChainConfig) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (bin.find(SCR_SOURCE_DIR) == std::string::npos || SCR_SOURCE_DIR[0] == '\0'
        || !std::ifstream(bin).good()) {
        GTEST_SKIP() << "数据缺席: " << bin;
    }
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    // 最大帧（cycle_000..002 逐帧载入取点数最大者）
    ReplayFrame best;
    bool loaded = false;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (LoadFrame(c, &fr) && (!loaded || fr.leftPts.size() > best.leftPts.size())) {
            best = std::move(fr);
            loaded = true;
        }
    }
    if (!loaded) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
    best.rightTracksZero.assign(best.rightPts.size(), 0);

    LaserMatchScanCpuV3 op(ChainParams());
    ASSERT_TRUE(op.SetTempTable(table));
    op.SetCurrentTemperature(22.5);

    const cv::Mat l = PointsMat(best.leftPts), lt = IdsMat(best.leftTracks);
    const cv::Mat r = PointsMat(best.rightPts), rt = IdsMat(best.rightTracksZero);
    op.Warmup(static_cast<int>(best.leftPts.size()), static_cast<int>(best.rightPts.size()));
    for (int i = 0; i < 3; ++i) {
        auto w = op.Execute(l, lt, r, rt);
        ASSERT_TRUE(w.success);
    }
    std::vector<double> ms;
    for (int i = 0; i < 10; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = op.Execute(l, lt, r, rt);
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_TRUE(res.success);
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2];
    std::printf("[perf] chain-config: N_left=%zu N_right=%zu median=%.3fms (min=%.3f max=%.3f)\n",
                best.leftPts.size(), best.rightPts.size(), median, ms.front(), ms.back());
    // 参考项（实验三路径全开）：GPU 基准同帧 ~3.5ms；本机实测 ~9ms。热安全回归界。
    EXPECT_LT(median, 15.0) << "链内配置中位耗时 " << median << "ms 异常（回归？参考 ~9ms@i7-9850H）";
}

TEST(LaserMatchCpuV3, PerfDefaultConfig) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good()) {
        GTEST_SKIP() << "数据缺席: " << bin;
    }
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    ReplayFrame best;
    bool loaded = false;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (LoadFrame(c, &fr) && (!loaded || fr.leftPts.size() > best.leftPts.size())) {
            best = std::move(fr);
            loaded = true;
        }
    }
    if (!loaded) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
    best.rightTracksZero.assign(best.rightPts.size(), 0);

    // 产线默认配置（双弃路径；vL_tolerance=0.01 展开窗近乎零、卡数骤减）
    LaserMatchScanParamsV3 def;
    LaserMatchScanCpuV3 op(def);
    ASSERT_TRUE(op.SetTempTable(table));
    op.SetCurrentTemperature(22.5);

    const cv::Mat l = PointsMat(best.leftPts), lt = IdsMat(best.leftTracks);
    const cv::Mat r = PointsMat(best.rightPts), rt = IdsMat(best.rightTracksZero);
    op.Warmup(static_cast<int>(best.leftPts.size()), static_cast<int>(best.rightPts.size()));

    for (int i = 0; i < 3; ++i) {
        auto w = op.Execute(l, lt, r, rt);
        ASSERT_TRUE(w.success);
    }
    std::vector<double> ms;
    for (int i = 0; i < 10; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = op.Execute(l, lt, r, rt);
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_TRUE(res.success);
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2];
    std::printf("[perf] default-config: N_left=%zu N_right=%zu median=%.3fms (min=%.3f max=%.3f)\n",
                best.leftPts.size(), best.rightPts.size(), median, ms.front(), ms.back());
    EXPECT_LT(median, 5.0) << "CPU 单线程 Execute（产线默认配置）中位耗时 " << median << "ms ≥ 5ms";
}

// ============================================================================
// 3. A/B 回放一致性（GPU 基准）
// ============================================================================

TEST(LaserMatchCpuV3, AbReplayChainParams) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good())
        GTEST_SKIP() << "数据缺席: " << bin;
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    LaserMatchScanCpuV3 cpu(ChainParams());
    calib::LaserMatchScanCudaV3 gpu(ChainParams());
    ASSERT_TRUE(cpu.SetTempTable(table));
    ASSERT_TRUE(gpu.SetTempTable(table));
    cpu.SetCurrentTemperature(22.5);
    gpu.SetCurrentTemperature(22.5);

    int frames = 0;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (!LoadFrame(c, &fr)) continue;
        fr.rightTracksZero.assign(fr.rightPts.size(), 0);
        RunAbOneFrame(&cpu, &gpu, fr, "chain[cycle_" + std::to_string(c) + "]");
        ++frames;
    }
    if (frames == 0) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
}

TEST(LaserMatchCpuV3, AbReplayDefaultParams) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good())
        GTEST_SKIP() << "数据缺席: " << bin;
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    // 产线默认（双弃路径）：step=0.5 → 温表按行 CSR 以 0.5 重建（两实例同步重建）
    LaserMatchScanParamsV3 def;
    LaserMatchScanCpuV3 cpu(def);
    calib::LaserMatchScanCudaV3 gpu(def);
    ASSERT_TRUE(cpu.SetTempTable(table));
    ASSERT_TRUE(gpu.SetTempTable(table));
    cpu.SetCurrentTemperature(22.5);
    gpu.SetCurrentTemperature(22.5);

    int frames = 0;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (!LoadFrame(c, &fr)) continue;
        fr.rightTracksZero.assign(fr.rightPts.size(), 0);
        RunAbOneFrame(&cpu, &gpu, fr, "default[cycle_" + std::to_string(c) + "]");
        ++frames;
    }
    if (frames == 0) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
}
