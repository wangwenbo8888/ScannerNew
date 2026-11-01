/**
 * @file test_laser_match_v3_split.cpp
 * @brief laser_match_scan_v3 流水线拆分（GPU 前段＋CPU 后段）gtest
 *
 * 测试顺序（TU 内声明序）：
 *   0. 静态环境守卫（先于任何算子构造/static env 固化，清空 JMW_V3* 敏感变量）
 *   1. 合成单测（GPU 前段＋CPU 后段，手工小温表，精确断言）
 *   2. 性能断言（后段 <2ms 硬指标＝产线默认配置；链内配置软回归界）
 *   3. A/B 回放一致性（GPU 基准＝原算子 Execute，cycle_000..002 双配置）
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "laser_match_scan_v3_split.h"
#include "laser_match_scan_v3_cuda.h"
#include "cmtt_table_loader.h"
#include "common/calib_result_types.h"

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace {

// ============================================================================
// 0. 静态环境守卫：GPU 侧 JMW_V3* 敏感变量清空（含 static 固化时序——本 TU 静态
//    对象构造先于 gtest main，亦先于任何算子构造）。_putenv("VAR=") 在 MSVC 删除变量。
// ============================================================================
struct SplitEnvGuard {
    SplitEnvGuard() {
        for (const char* v : {"JMW_V3GRAPH", "JMW_V3BISECT", "JMW_V3ANCHOR", "JMW_V3_SEGCAP",
                              "JMW_V3FULLGPU", "JMW_V3GPU", "JMW_V3PAIR_AB",
                              "JMW_NOTBD", "JMW_V3TIME", "JMW_K1B_SKIPSTAMPS"}) {
            std::string s = std::string(v) + "=";
            _putenv(s.c_str());
        }
    }
} g_splitEnvGuard;

using calib::LaserMatchScanParamsV3;
using calib::LaserMatchScanSplitGpuV3;
using calib::LaserMatchScanSplitCpuV3;
using calib::LaserMatchSplitFrontV3;
using calib::LaserMatchScanCpuResultV3;
using calib::LaserMatchScanCudaV3;

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

// 前段跑一次的便捷封装
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

// ── 回放数据装载（同 test_laser_match_scan_v3_cpu.cpp）──
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

// ── A/B 比对：[前段＋后段] vs 原 GPU 算子 Execute（同参同输入）──
void RunSplitAbOneFrame(LaserMatchScanSplitGpuV3* sg, LaserMatchScanSplitCpuV3* sc,
                        LaserMatchScanCudaV3* gpuRef,
                        const ReplayFrame& fr, const std::string& tag) {
    const cv::Mat lpts = PointsMat(fr.leftPts);
    const cv::Mat lids = IdsMat(fr.leftTracks);
    const cv::Mat rpts = PointsMat(fr.rightPts);
    const cv::Mat rids = IdsMat(fr.rightTracksZero);

    // 被测：前段＋后段
    LaserMatchSplitFrontV3 front;
    {
        cv::cuda::GpuMat dL, dLid, dR, dRid;
        dL.upload(lpts);
        dLid.upload(lids);
        dR.upload(rpts);
        dRid.upload(rids);
        cv::cuda::Stream stream;
        sg->Execute(dL, dLid, dR, dRid, stream, front);
    }
    ASSERT_TRUE(front.success) << tag << " front 失败: " << front.message;
    auto rs = sc->Execute(front);
    ASSERT_TRUE(rs.success) << tag << " back 失败: " << rs.message;

    // 基准：原 GPU 算子
    cv::cuda::GpuMat dL, dLid, dR, dRid;
    dL.upload(lpts);
    dLid.upload(lids);
    dR.upload(rpts);
    dRid.upload(rids);
    auto rg = gpuRef->Execute(dL, dLid, dR, dRid);
    ASSERT_TRUE(rg.success) << tag << " GPU 基准失败: " << rg.message;

    EXPECT_EQ(rs.matchedCount, rg.matchedCount) << tag << " matchedCount";
    ASSERT_EQ(rs.matchedCount, rg.matchedCount);

    // GPU ROI 视图：download 后再比对（下一次 Execute 前）
    cv::Mat gmL, gmR, gmLid, gmTrack;
    rg.d_matched_left->download(gmL);
    rg.d_matched_right->download(gmR);
    rg.d_matched_line_ids->download(gmLid);
    rg.d_matched_track_ids->download(gmTrack);

    const int n = rs.matchedCount;
    ASSERT_EQ(n, static_cast<int>(gmL.total()));
    const cv::Point2f* cl = rs.matched_left.ptr<cv::Point2f>();
    const cv::Point2f* cr = rs.matched_right.ptr<cv::Point2f>();
    const int* cid = rs.matched_line_ids.ptr<int>();
    const int* ctr = rs.matched_track_ids.ptr<int>();
    const cv::Point2f* gl = gmL.ptr<cv::Point2f>();
    const cv::Point2f* gr = gmR.ptr<cv::Point2f>();
    const int* gid = gmLid.ptr<int>();
    const int* gtr = gmTrack.ptr<int>();
    int coordBad = 0;
    for (int i = 0; i < n; ++i) {
        if (cid[i] != gid[i] || ctr[i] != gtr[i]) {
            ADD_FAILURE() << tag << " idx=" << i << " lid/track 不一致: split=("
                          << cid[i] << "," << ctr[i] << ") gpu=(" << gid[i] << "," << gtr[i] << ")";
            return;
        }
        if (std::fabs(cl[i].x - gl[i].x) > 1e-3f || std::fabs(cl[i].y - gl[i].y) > 1e-3f ||
            std::fabs(cr[i].x - gr[i].x) > 1e-3f || std::fabs(cr[i].y - gr[i].y) > 1e-3f)
            ++coordBad;
    }
    EXPECT_EQ(coordBad, 0) << tag << " 坐标超差(>1e-3)点数";

    EXPECT_EQ(rs.totalLeftPoints, rg.totalLeftPoints) << tag;
    EXPECT_EQ(rs.totalRightPoints, rg.totalRightPoints) << tag;
    EXPECT_EQ(rs.excludedLeftCount, rg.excludedLeftCount) << tag << " exclL";
    EXPECT_EQ(rs.excludedRightCount, rg.excludedRightCount) << tag << " exclR";
    EXPECT_EQ(rs.noMapLeftCount, rg.noMapLeftCount) << tag << " noMap";
    EXPECT_EQ(rs.noWinnerCardCount, rg.noWinnerCardCount) << tag << " noWinnerCard";
    EXPECT_EQ(rs.emptyWinnerSegCount, rg.emptyWinnerSegCount) << tag << " emptySeg";
    EXPECT_EQ(rs.trackTotal, rg.trackTotal) << tag << " trackTotal";
    EXPECT_EQ(rs.trackDropped, rg.trackDropped) << tag << " trackDropped";
}

} // namespace

// ============================================================================
// 1. 合成单测（GPU 前段＋CPU 后段）
// ============================================================================

TEST(LaserMatchV3Split, SyntheticBasicMatch) {
    LaserMatchScanSplitGpuV3 gpu;
    LaserMatchScanSplitCpuV3 cpu;
    LaserMatchScanParamsV3 p;   // 默认 thr=1.0/step=0.5；放宽 vL 至 0.5
    p.vL_tolerance = 0.5f;
    gpu.SetParams(p);
    cpu.SetParams(p);
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);

    // 左点：行 100（y=50），x=200 → 查表卡 lid2 uR=100；右点 (100, 50±ε) 命中窗
    const std::vector<cv::Point2f> left = {{200.0f, 50.0f}, {200.2f, 50.2f}};
    const std::vector<int> ltrk = {0, 0};
    const std::vector<cv::Point2f> right = {{100.0f, 50.0f}, {100.1f, 50.2f}};
    auto front = RunFront(&gpu, left, ltrk, right, {0, 0});
    ASSERT_TRUE(front.success) << front.message;
    auto r = cpu.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
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

TEST(LaserMatchV3Split, SyntheticNoMapAndNoCard) {
    LaserMatchScanSplitGpuV3 gpu;
    LaserMatchScanSplitCpuV3 cpu;
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);

    // p0：y 超出表行（行 500 无表）→ noMap；p1：行 101 有表但右点全不在窗 → noCard
    const std::vector<cv::Point2f> left = {{200.0f, 250.0f}, {200.0f, 50.5f}};
    const std::vector<int> ltrk = {0, 1};
    const std::vector<cv::Point2f> right = {{999.0f, 50.5f}};
    auto front = RunFront(&gpu, left, ltrk, right, {0});
    ASSERT_TRUE(front.success) << front.message;
    auto r = cpu.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.noMapLeftCount, 1);
    EXPECT_EQ(r.excludedLeftCount, 1);
    EXPECT_EQ(r.matchedCount, 0);
    EXPECT_EQ(r.left_status[0], -1);
    EXPECT_EQ(r.left_status[1], 0);
}

TEST(LaserMatchV3Split, SyntheticSameLidConflictDropsBoth) {
    LaserMatchScanSplitGpuV3 gpu;
    LaserMatchScanSplitCpuV3 cpu;
    LaserMatchScanParamsV3 p;   // 默认参数 → 双弃
    p.vL_tolerance = 0.5f;
    gpu.SetParams(p);
    cpu.SetParams(p);
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);

    // 两段同行 100、同 winner lid2（都在 x≈200 查到 lid2 卡且右点命中）→ 双弃
    const std::vector<cv::Point2f> left = {
        {200.0f, 50.0f}, {200.1f, 50.0f},     // track 0
        {200.5f, 50.2f}, {200.6f, 50.2f}      // track 1（同行 100：50.2/0.5=100.4→round=100）
    };
    const std::vector<int> ltrk = {0, 0, 1, 1};
    const std::vector<cv::Point2f> right = {
        {100.0f, 50.0f}, {100.1f, 50.0f}, {100.5f, 50.2f}, {100.6f, 50.2f}
    };
    auto front = RunFront(&gpu, left, ltrk, right, {0, 0, 0, 0});
    ASSERT_TRUE(front.success) << front.message;
    auto r = cpu.Execute(front);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.trackTotal, 2);
    EXPECT_EQ(r.trackDropped, 2);   // 同 (row,lid) 双主张 → 全弃
    EXPECT_EQ(r.matchedCount, 0);
    EXPECT_EQ(r.emptyWinnerSegCount, 0);   // winner 仍为 lid2（被弃但 winner≥1）
}

TEST(LaserMatchV3Split, SyntheticEmptyInput) {
    LaserMatchScanSplitGpuV3 gpu;
    LaserMatchScanSplitCpuV3 cpu;
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);
    auto front = RunFront(&gpu, {}, {}, {}, {});
    EXPECT_TRUE(front.success);
    auto r = cpu.Execute(front);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.matchedCount, 0);
}

TEST(LaserMatchV3Split, SyntheticUrBiasShift) {
    LaserMatchScanSplitGpuV3 gpu;
    LaserMatchScanSplitCpuV3 cpu;
    LaserMatchScanParamsV3 p;
    p.match_threshold = 1.0f;
    p.epipolar_row_step = 0.5f;
    p.vL_tolerance = 0.5f;
    gpu.SetParams(p);
    cpu.SetParams(p);
    ASSERT_TRUE(gpu.SetTempTable(MakeSynthTable()));
    gpu.SetCurrentTemperature(25.0);

    // 无补偿：uR=100 → 右点 100 命中；注入 +5 补偿：uR=105 → 原右点 100 距 5>win1 不命中
    const std::vector<cv::Point2f> left = {{200.0f, 50.0f}};
    const std::vector<cv::Point2f> right = {{100.0f, 50.0f}};

    auto f0 = RunFront(&gpu, left, {0}, right, {0});
    ASSERT_TRUE(f0.success);
    auto r0 = cpu.Execute(f0);
    ASSERT_TRUE(r0.success);
    EXPECT_EQ(r0.matchedCount, 1);

    std::vector<float> bias(32, 0.0f);
    bias[1] = 5.0f;   // lid2 → +5
    gpu.SetuRBias(bias);
    auto f1 = RunFront(&gpu, left, {0}, right, {0});
    ASSERT_TRUE(f1.success);
    auto r1 = cpu.Execute(f1);
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(r1.matchedCount, 0);
    EXPECT_EQ(r1.excludedLeftCount, 1);   // 有卡无 HIT → noCard
}

// ============================================================================
// 2. 性能断言（后段 <2ms 硬指标＝产线默认配置；链内配置软回归界 4ms）
//    口径：back_ms＝后段 Execute（组装＋计算＋输出），D2H 计入 front_ms。
// ============================================================================

namespace {

ReplayFrame LoadBiggestFrame(bool* ok) {
    *ok = false;
    ReplayFrame best;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (LoadFrame(c, &fr) && (!*ok || fr.leftPts.size() > best.leftPts.size())) {
            best = std::move(fr);
            *ok = true;
        }
    }
    if (*ok) best.rightTracksZero.assign(best.rightPts.size(), 0);
    return best;
}

void RunPerf(const LaserMatchScanParamsV3& p, const char* tag, double hardBoundMs,
             double* outBackMs, double* outFrontMs) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good()) {
        GTEST_SKIP() << "数据缺席: " << bin;
    }
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";
    bool loaded = false;
    ReplayFrame best = LoadBiggestFrame(&loaded);
    if (!loaded) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";

    LaserMatchScanSplitGpuV3 gpu(p);
    LaserMatchScanSplitCpuV3 cpu(p);
    ASSERT_TRUE(gpu.SetTempTable(table));
    gpu.SetCurrentTemperature(22.5);

    const cv::Mat l = PointsMat(best.leftPts), lt = IdsMat(best.leftTracks);
    const cv::Mat r = PointsMat(best.rightPts), rt = IdsMat(best.rightTracksZero);
    gpu.Warmup(static_cast<int>(best.leftPts.size()), static_cast<int>(best.rightPts.size()));

    cv::cuda::GpuMat dL, dLid, dR, dRid;
    dL.upload(l);
    dLid.upload(lt);
    dR.upload(r);
    dRid.upload(rt);

    // 前段跑一次（预热＋测 front_ms）
    LaserMatchSplitFrontV3 front;
    {
        cv::cuda::Stream stream;
        gpu.Execute(dL, dLid, dR, dRid, stream, front);
    }
    ASSERT_TRUE(front.success) << front.message;

    // 后段预热 3＋测 10（中位）
    for (int i = 0; i < 3; ++i) {
        auto w = cpu.Execute(front);
        ASSERT_TRUE(w.success) << w.message;
    }
    std::vector<double> ms;
    for (int i = 0; i < 10; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = cpu.Execute(front);
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_TRUE(res.success);
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double backMs = ms[ms.size() / 2];

    // front_ms：预热后再测中位（3 次）
    std::vector<double> fms;
    for (int i = 0; i < 3; ++i) {
        LaserMatchSplitFrontV3 f2;
        const auto t0 = std::chrono::steady_clock::now();
        cv::cuda::Stream stream;
        gpu.Execute(dL, dLid, dR, dRid, stream, f2);
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_TRUE(f2.success);
        fms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        front = f2;   // 留最新一份（供外部复用）
    }
    std::sort(fms.begin(), fms.end());
    const double frontMs = fms[fms.size() / 2];

    std::printf("[perf-split] %s: N_left=%zu N_right=%zu segs=%d cards=%d\n",
                tag, best.leftPts.size(), best.rightPts.size(), front.segCount, front.totalCards);
    std::printf("[perf-split] %s: front_ms=%.3f back_ms=%.3f (back min=%.3f max=%.3f) "
                "serial=%.3f | back bd: %s | front bd: %s\n",
                tag, frontMs, backMs, ms.front(), ms.back(), frontMs + backMs,
                cpu.LastTimingBreakdown().c_str(), gpu.LastTimingBreakdown().c_str());

    EXPECT_LT(backMs, hardBoundMs)
        << tag << " 后段中位耗时 " << backMs << "ms ≥ " << hardBoundMs << "ms";
    if (outBackMs) *outBackMs = backMs;
    if (outFrontMs) *outFrontMs = frontMs;
}

} // namespace

TEST(LaserMatchV3Split, PerfBackDefaultConfig) {
    // 产线默认（双弃路径）：<2ms 硬断言（i7-9850H 参照机，Release）
    double backMs = 0, frontMs = 0;
    LaserMatchScanParamsV3 def;
    RunPerf(def, "default", 2.0, &backMs, &frontMs);
}

TEST(LaserMatchV3Split, PerfBackChainConfig) {
    // 链内配置（P5/P6b/P7 实验三路径全开）：实测记录＋热安全软回归界 4ms
    double backMs = 0, frontMs = 0;
    RunPerf(ChainParams(), "chain", 4.0, &backMs, &frontMs);
}

// ============================================================================
// 3. A/B 回放一致性（GPU 基准＝原算子 Execute）
// ============================================================================

TEST(LaserMatchV3Split, AbReplayChainParams) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good())
        GTEST_SKIP() << "数据缺席: " << bin;
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    LaserMatchScanSplitGpuV3 sg(ChainParams());
    LaserMatchScanSplitCpuV3 sc(ChainParams());
    LaserMatchScanCudaV3 gpuRef(ChainParams());   // 独立基准实例
    ASSERT_TRUE(sg.SetTempTable(table));
    ASSERT_TRUE(gpuRef.SetTempTable(table));
    sg.SetCurrentTemperature(22.5);
    gpuRef.SetCurrentTemperature(22.5);

    int frames = 0;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (!LoadFrame(c, &fr)) continue;
        fr.rightTracksZero.assign(fr.rightPts.size(), 0);
        RunSplitAbOneFrame(&sg, &sc, &gpuRef, fr, "chain[cycle_" + std::to_string(c) + "]");
        ++frames;
    }
    if (frames == 0) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
}

TEST(LaserMatchV3Split, AbReplayDefaultParams) {
    const std::string bin = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/curve_map_temp_table.bin";
    if (SCR_SOURCE_DIR[0] == '\0' || !std::ifstream(bin).good())
        GTEST_SKIP() << "数据缺席: " << bin;
    auto table = LoadCmttTable();
    if (!table) GTEST_SKIP() << "CMTT 温表装载失败";

    // 产线默认（双弃路径）：step=0.5 → 温表按行 CSR 以 0.5 重建（实例间同步重建）
    LaserMatchScanParamsV3 def;
    LaserMatchScanSplitGpuV3 sg(def);
    LaserMatchScanSplitCpuV3 sc(def);
    LaserMatchScanCudaV3 gpuRef(def);
    ASSERT_TRUE(sg.SetTempTable(table));
    ASSERT_TRUE(gpuRef.SetTempTable(table));
    sg.SetCurrentTemperature(22.5);
    gpuRef.SetCurrentTemperature(22.5);

    int frames = 0;
    for (int c = 0; c <= 2; ++c) {
        ReplayFrame fr;
        if (!LoadFrame(c, &fr)) continue;
        fr.rightTracksZero.assign(fr.rightPts.size(), 0);
        RunSplitAbOneFrame(&sg, &sc, &gpuRef, fr, "default[cycle_" + std::to_string(c) + "]");
        ++frames;
    }
    if (frames == 0) GTEST_SKIP() << "回放帧缺席 (out_golden_base/cycle_000..002)";
}
