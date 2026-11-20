/**
 * @file test_laser_reconstruct_cpu.cpp
 * @brief laser_reconstruct_cpu（P0 新写 CPU 版）gtest
 *
 * 测试顺序（TU 内声明序）：
 *   1. 合成单测（显式 Q＋double 精度独立推算基准：坐标/门槛/顺序/定容截断）
 *   2. 契约边界（空输入/尺寸不一致/非 CV_64F Q）
 *   3. A/B 合成（4 万随机点 vs CUDA 算子：计数/线号逐位、坐标 |Δ|≤1e-3×max(1,|z|)）
 *   4. A/B 真实数据（out_golden_base cycle_000..002 L4 配对点＋22.5℃ 档真 Q）
 *
 * 数据缺席 → GTEST_SKIP（date_in/out_golden_base 未入版本库）。
 * 对拍口径〔设计文档 §十 P0〕：计数一致＋逐点 |Δ|≤1e-6×max(1,|z|)（本测收紧为 1e-3
 * 之外的浮点健壮带；实测应 ≈1e-4 量级）。
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "laser_reconstruct_cuda.h"
#include "laser_reconstruct_cpu.h"

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace {

using calib::LaserReconstructParams;
using calib::LaserReconstructCpu;
using calib::LaserReconstructCpuResult;
using calib::LaserReconstructCuda;

// 合成 Q（结构＝标准立体矫正重投影矩阵；深度单位 mm）：
//   W = Q[14]*d + Q[15] = d/35；Z = 1400/W（d=100 → Z=490mm）
//   X = (x-844)/W；Y = (y-700)/W
cv::Mat MakeSynthQ() {
    return (cv::Mat_<double>(4, 4) <<
            1.0, 0.0, 0.0, -844.0,
            0.0, 1.0, 0.0, -700.0,
            0.0, 0.0, 0.0, 1400.0,
            0.0, 0.0, 1.0 / 35.0, 0.0);
}

struct OraclePt { double X, Y, Z; };

// double 精度独立推算（测试基准，非被测代码复用）
bool OracleReconstruct(float xl, float yl, float xr, const double Qd[16],
                       double minDepth, double maxDepth, OraclePt* out) {
    const double d = static_cast<double>(xl) - static_cast<double>(xr);
    const double W = Qd[14] * d + Qd[15];
    if (W <= 0.0) return false;
    const double X = (Qd[0] * xl + Qd[3]) / W;
    const double Y = (Qd[5] * yl + Qd[7]) / W;
    const double Z = Qd[11] / W;
    if (Z < minDepth || Z > maxDepth) return false;
    out->X = X; out->Y = Y; out->Z = Z;
    return true;
}

// ============================================================================
// 1. 合成单测
// ============================================================================

TEST(LaserReconstructCpu, SyntheticOracle) {
    const cv::Mat Q = MakeSynthQ();
    double Qd[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Qd[i * 4 + j] = Q.at<double>(i, j);

    LaserReconstructParams p;
    p.minDepth = 100.0f;    // 深度门（合成口径：d∈[15,300] → Z∈[163,32667]，门内取子集）
    p.maxDepth = 5000.0f;
    LaserReconstructCpu op(p);

    // 构造：正常点×5（d=60..140）＋W<=0 点×1（d=-10）＋超深度点×2（d=6 → Z≈8167 超上限；d=500 → Z=98 门内但先过 W>0）
    std::vector<cv::Point2f> L, R;
    std::vector<int> ids;
    std::vector<OraclePt> expected;

    auto add = [&](float x, float y, float d, int id) {
        L.emplace_back(x, y);
        R.emplace_back(x - d, y);
        ids.push_back(id);
        OraclePt o{};
        if (OracleReconstruct(x, y, x - d, Qd, p.minDepth, p.maxDepth, &o)) {
            expected.push_back(o);
        }
    };

    add(300.0f, 200.0f, 60.0f, 10);
    add(900.0f, 500.0f, 80.0f, 11);
    add(1500.0f, 1100.0f, 100.0f, 12);
    add(100.0f, 1300.0f, 120.0f, 13);
    add(1680.0f, 100.0f, 140.0f, 14);
    add(500.0f, 600.0f, -10.0f, 99);    // W<0 弃
    add(700.0f, 800.0f, 6.0f, 98);      // Z≈8167 超门弃
    add(400.0f, 900.0f, 500.0f, 15);    // Z=98 < minDepth=100 弃

    auto r = op.Execute(L, R, ids, Q);
    ASSERT_TRUE(r.success) << r.message;
    EXPECT_EQ(r.totalInput, 8);
    ASSERT_EQ(r.validCount, static_cast<int>(expected.size()));
    ASSERT_EQ(r.validCount, 5);

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(r.points3d[i].x, expected[i].X, 1e-3 * std::max(1.0, std::abs(expected[i].X)));
        EXPECT_NEAR(r.points3d[i].y, expected[i].Y, 1e-3 * std::max(1.0, std::abs(expected[i].Y)));
        EXPECT_NEAR(r.points3d[i].z, expected[i].Z, 1e-3 * std::max(1.0, std::abs(expected[i].Z)));
    }
    // 顺序保持（幸存线号按输入序）：d=-10/6/500 三帧分别因 W≤0/超上限/低于下限被弃
    EXPECT_EQ(r.validLineIds, (std::vector<int>{10, 11, 12, 13, 14})) << "线号顺序保持";
}

TEST(LaserReconstructCpu, ReconstructIntoTruncation) {
    const cv::Mat Q = MakeSynthQ();
    LaserReconstructParams p;   // 默认门 0..10000
    LaserReconstructCpu op(p);

    std::vector<cv::Point2f> L, R;
    std::vector<int> ids;
    for (int i = 0; i < 100; ++i) {
        L.emplace_back(100.0f + i, 200.0f);
        R.emplace_back(100.0f + i - 80.0f, 200.0f);
        ids.push_back(i);
    }

    // 容量充足：全量
    std::vector<float> xyz(3 * 100);
    std::vector<int> outIds(100);
    int n = LaserReconstructCpu::ReconstructInto(L.data(), R.data(), ids.data(),
                                                 100, Q, p.minDepth, p.maxDepth,
                                                 xyz.data(), outIds.data(), 100);
    EXPECT_EQ(n, 100);
    EXPECT_EQ(outIds[99], 99);

    // 容量 30：截断到 30（前 30 个幸存）
    n = LaserReconstructCpu::ReconstructInto(L.data(), R.data(), ids.data(),
                                             100, Q, p.minDepth, p.maxDepth,
                                             xyz.data(), outIds.data(), 30);
    EXPECT_EQ(n, 30);
    EXPECT_EQ(outIds[29], 29);

    // 与 Execute 一致性（前 30 点）
    auto r = op.Execute(L, R, ids, Q);
    ASSERT_TRUE(r.success);
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(outIds[i], r.validLineIds[i]);
        EXPECT_EQ(xyz[3 * i], r.points3d[i].x);
    }
}

// ============================================================================
// 2. 契约边界
// ============================================================================

TEST(LaserReconstructCpu, ContractEdges) {
    const cv::Mat Q = MakeSynthQ();
    LaserReconstructCpu op;

    // 空输入＝success（对齐 CUDA 版语义）
    auto r0 = op.Execute({}, {}, {}, Q);
    EXPECT_TRUE(r0.success);
    EXPECT_EQ(r0.validCount, 0);
    EXPECT_EQ(r0.totalInput, 0);

    // 尺寸不一致＝fail
    std::vector<cv::Point2f> L{{1, 2}}, R{{3, 4}};
    std::vector<int> ids{7, 8};
    auto r1 = op.Execute(L, R, ids, Q);
    EXPECT_FALSE(r1.success);

    // 非 CV_64F Q＝fail
    cv::Mat Qf;
    Q.convertTo(Qf, CV_32F);
    auto r2 = op.Execute(L, R, {7}, Qf);
    EXPECT_FALSE(r2.success);
}

// ============================================================================
// 3. A/B 合成（vs CUDA）
// ============================================================================

std::string MaxDiffReport(const std::vector<cv::Point3f>& cpu,
                          const cv::Mat& gpu, int count) {
    const cv::Point3f* g = gpu.ptr<cv::Point3f>();
    double maxAbs = 0, maxRel = 0;
    int worst = -1;
    for (int i = 0; i < count; ++i) {
        const double dx = std::abs(cpu[i].x - g[i].x);
        const double dy = std::abs(cpu[i].y - g[i].y);
        const double dz = std::abs(cpu[i].z - g[i].z);
        const double m = std::max(dx, std::max(dy, dz));
        const double rel = m / std::max(1.0, static_cast<double>(std::abs(g[i].z)));
        if (rel > maxRel) { maxRel = rel; maxAbs = m; worst = i; }
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "maxRel=%.3e maxAbs=%.6f @idx=%d(z=%.2f)",
                  maxRel, maxAbs, worst, worst >= 0 ? g[worst].z : 0.0);
    return buf;
}

TEST(LaserReconstructCpu, AbGpuSynthetic) {
    const cv::Mat Q = MakeSynthQ();
    LaserReconstructParams p;
    LaserReconstructCpu cpu(p);
    LaserReconstructCuda gpu(p);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xd(0.0f, 1688.0f);
    std::uniform_real_distribution<float> yd(0.0f, 1400.0f);
    std::uniform_real_distribution<float> dd(20.0f, 200.0f);
    std::uniform_int_distribution<int> idd(1, 40);

    const int N = 40000;
    std::vector<cv::Point2f> L(N), R(N);
    std::vector<int> ids(N);
    for (int i = 0; i < N; ++i) {
        L[i] = {xd(rng), yd(rng)};
        R[i] = {L[i].x - dd(rng), L[i].y};
        ids[i] = idd(rng);
    }

    auto rc = cpu.Execute(L, R, ids, Q);
    ASSERT_TRUE(rc.success) << rc.message;

    cv::Mat hL(N, 1, CV_32FC2, L.data());
    cv::Mat hR(N, 1, CV_32FC2, R.data());
    cv::Mat hI(N, 1, CV_32SC1, ids.data());
    cv::cuda::GpuMat dL, dR, dI;
    dL.upload(hL);
    dR.upload(hR);
    dI.upload(hI);
    auto rg = gpu.Execute(dL, dR, dI, Q);
    ASSERT_TRUE(rg.success) << rg.message;

    ASSERT_EQ(rc.validCount, rg.validCount) << "幸存计数 CPU=" << rc.validCount
                                            << " GPU=" << rg.validCount;
    ASSERT_EQ(rc.totalInput, rg.totalInput);

    cv::Mat gPts, gIds;
    rg.d_points3d->download(gPts);
    rg.d_valid_line_ids->download(gIds);
    const int* gid = gIds.ptr<int>();

    int idMismatch = 0;
    for (int i = 0; i < rc.validCount; ++i)
        if (rc.validLineIds[i] != gid[i]) ++idMismatch;
    EXPECT_EQ(idMismatch, 0) << "线号逐位不一致数";

    for (int i = 0; i < rc.validCount; ++i) {
        const cv::Point3f& g = gPts.ptr<cv::Point3f>()[i];
        EXPECT_NEAR(rc.points3d[i].x, g.x, 1e-3 * std::max(1.0f, std::abs(g.z)));
        EXPECT_NEAR(rc.points3d[i].y, g.y, 1e-3 * std::max(1.0f, std::abs(g.z)));
        EXPECT_NEAR(rc.points3d[i].z, g.z, 1e-3 * std::max(1.0f, std::abs(g.z)));
    }
    std::printf("[ab-synth] N=%d valid=%d %s\n", N, rc.validCount,
                MaxDiffReport(rc.points3d, gPts, rc.validCount).c_str());
}

// ============================================================================
// 4. A/B 真实数据（out_golden_base cycle_000..002）
// ============================================================================

bool LoadXyCsv(const std::string& path, std::vector<cv::Point2f>* pts) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    std::getline(f, line);   // header: x,y
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double x = 0, y = 0;
        if (!(ss >> x >> y)) return false;
        pts->emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
    return true;
}

bool LoadTracksCsv(const std::string& path, std::vector<int>* lids) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    std::getline(f, line);   // header: x,y,lid,track
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double x = 0, y = 0; int lid = 0, track = 0;
        if (!(ss >> x >> y >> lid >> track)) return false;
        lids->push_back(lid);
    }
    return true;
}

cv::Mat LoadRealQ(double wantTemp) {
    const std::string path = std::string(SCR_SOURCE_DIR)
        + "/date_in/laser-mark1688-1006/camera_calib.json";
    std::ifstream f(path);
    if (!f.is_open()) return {};
    nlohmann::json j;
    f >> j;
    if (!j.contains("stereoRectifyTempTable")) return {};
    for (const auto& t : j["stereoRectifyTempTable"]["table"]) {
        const double temp = t["temperature"].get<double>();
        if (std::abs(temp - wantTemp) < 0.25) {
            cv::Mat Q(4, 4, CV_64F);
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    Q.at<double>(r, c) = t["Q"][r][c].get<double>();
            return Q;
        }
    }
    return {};
}

TEST(LaserReconstructCpu, AbGpuRealData) {
    const cv::Mat Q = LoadRealQ(22.5);
    if (Q.empty()) GTEST_SKIP() << "camera_calib.json 缺席或无 22.5℃ 档";

    LaserReconstructParams p;   // 与 config operators.laser_reconstruct 同（0..10000）
    LaserReconstructCpu cpu(p);
    LaserReconstructCuda gpu(p);

    for (int cycle = 0; cycle <= 2; ++cycle) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/out_golden_base/cycle_%03d", SCR_SOURCE_DIR, cycle);
        const std::string dir = buf;

        std::vector<cv::Point2f> L, R;
        std::vector<int> ids;
        if (!LoadXyCsv(dir + "/L4-match_L.csv", &L) ||
            !LoadXyCsv(dir + "/L4-match_R.csv", &R) ||
            !LoadTracksCsv(dir + "/L4t-match-tracks.csv", &ids)) {
            GTEST_SKIP() << "回放配对数据缺席: " << dir;
        }
        ASSERT_EQ(L.size(), R.size()) << "cycle " << cycle << " L/R 行数不一致";
        ASSERT_EQ(L.size(), ids.size()) << "cycle " << cycle << " tracks 行数不一致";
        if (L.empty()) continue;

        const int N = static_cast<int>(L.size());
        auto rc = cpu.Execute(L, R, ids, Q);
        ASSERT_TRUE(rc.success) << "cycle " << cycle << ": " << rc.message;

        cv::Mat hL(N, 1, CV_32FC2, L.data());
        cv::Mat hR(N, 1, CV_32FC2, R.data());
        cv::Mat hI(N, 1, CV_32SC1, ids.data());
        cv::cuda::GpuMat dL, dR, dI;
        dL.upload(hL);
        dR.upload(hR);
        dI.upload(hI);
        auto rg = gpu.Execute(dL, dR, dI, Q);
        ASSERT_TRUE(rg.success) << "cycle " << cycle << ": " << rg.message;

        ASSERT_EQ(rc.validCount, rg.validCount) << "cycle " << cycle;
        ASSERT_EQ(rc.totalInput, rg.totalInput) << "cycle " << cycle;

        cv::Mat gPts, gIds;
        rg.d_points3d->download(gPts);
        rg.d_valid_line_ids->download(gIds);
        const int* gid = gIds.ptr<int>();

        int idMismatch = 0;
        for (int i = 0; i < rc.validCount; ++i)
            if (rc.validLineIds[i] != gid[i]) ++idMismatch;
        EXPECT_EQ(idMismatch, 0) << "cycle " << cycle << " 线号不一致";

        for (int i = 0; i < rc.validCount; ++i) {
            const cv::Point3f& g = gPts.ptr<cv::Point3f>()[i];
            EXPECT_NEAR(rc.points3d[i].x, g.x, 1e-3 * std::max(1.0f, std::abs(g.z)));
            EXPECT_NEAR(rc.points3d[i].y, g.y, 1e-3 * std::max(1.0f, std::abs(g.z)));
            EXPECT_NEAR(rc.points3d[i].z, g.z, 1e-3 * std::max(1.0f, std::abs(g.z)));
        }
        std::printf("[ab-real] cycle %d: N=%d valid=%d %s\n", cycle, N, rc.validCount,
                    MaxDiffReport(rc.points3d, gPts, rc.validCount).c_str());
    }
}

} // namespace
