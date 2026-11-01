// test_units.cpp — 纯逻辑单测（config/artifacts/frame_source/calib_loader）
// ＋ StegerDedup oracle（GPU 算子内去重 vs CPU 外挂位移式去重，合成平顶对照）
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

#include "artifacts.h"
#include "calib_loader.h"
#include "frame_source.h"
#include "scr_config.h"
#include "steger_dedup_merge.h"
#include "scr_test_helpers.h"

#include "core/laser/steger_fast/steger_fast.h"

namespace scr {
namespace fs = std::filesystem;

namespace {
fs::path tempDir() {
    static const auto d = fs::temp_directory_path() / "scr_units";
    fs::create_directories(d);
    return d;
}
} // namespace

// ============================ ScrConfig ============================

TEST(ScrConfig, ExpandSourceRootToken) {
    const auto p = ScrConfig::expandPath("{JMW_SOURCE_ROOT}/x/y.json");
    EXPECT_EQ(p.find("{JMW_SOURCE_ROOT}"), std::string::npos)
        << "占位符应被展开: " << p;
    EXPECT_NE(p.find("x/y.json"), std::string::npos);
    EXPECT_FALSE(p.empty());
}

TEST(ScrConfig, RoundtripAndDefaults) {
    const auto cfgPath = tempDir() / "cfg_roundtrip.json";
    {
        std::ofstream ofs(cfgPath);
        ofs << R"({
          "paths": { "dataDir": "{JMW_SOURCE_ROOT}/d", "cameraCalib": "{JMW_SOURCE_ROOT}/c.json",
                     "cmttSidecar": "{JMW_SOURCE_ROOT}/t.bin", "outDir": "out" },
          "runtime": { "maxCycles": 7, "nominalTemperature": 22.5 },
          "operators": { "ccl": { "minArea": 50 } }
        })";
    }
    ScrConfig cfg;
    std::string err;
    ASSERT_TRUE(ScrConfig::fromFile(cfgPath.string(), cfg, err)) << err;
    EXPECT_EQ(cfg.runtime.maxCycles, 7);
    EXPECT_DOUBLE_EQ(cfg.runtime.nominalTemperature, 22.5);
    EXPECT_EQ(cfg.runtime.warmupCycles, 1);       // 缺省默认
    EXPECT_EQ(cfg.opParams("ccl").at("minArea").get<int>(), 50);
    EXPECT_TRUE(cfg.opParams("no_such_op").empty());   // 缺键=空对象=算子默认
}

TEST(ScrConfig, MissingPathsFails) {
    ScrConfig cfg;
    std::string err;
    EXPECT_FALSE(ScrConfig::fromJson(nlohmann::json::object(), cfg, err));
    EXPECT_NE(err.find("paths"), std::string::npos);
}

// ============================ artifacts ============================

TEST(Artifacts, CsvP2Roundtrip) {
    const auto p = (tempDir() / "pts.csv").string();
    std::vector<cv::Point2f> in{{1.5f, -2.25f}, {100.125f, 0.0f}, {0, 999.5f}};
    ASSERT_TRUE(saveCsvP2f(p, in));
    std::vector<cv::Point2f> out;
    ASSERT_TRUE(loadCsvP2f(p, out));
    ASSERT_EQ(out.size(), in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i].x, in[i].x, 1e-3f);
        EXPECT_NEAR(out[i].y, in[i].y, 1e-3f);
    }
}

TEST(Artifacts, PlyVertexCount) {
    const auto p = (tempDir() / "pts.ply").string();
    std::vector<cv::Point3f> in{{0, 0, 0}, {1, 2, 3}};
    ASSERT_TRUE(savePlyXYZf(p, in));
    std::ifstream ifs(p);
    std::string line;
    int lines = 0, declared = -1;
    while (std::getline(ifs, line)) {
        ++lines;
        if (line.rfind("element vertex", 0) == 0)
            declared = std::stoi(line.substr(std::string("element vertex").size()));
    }
    EXPECT_EQ(declared, 2);
    EXPECT_EQ(lines, 7 + 2);   // 头 7 行 + 2 顶点
}

TEST(Artifacts, PngRoundtrip) {
    const auto p = (tempDir() / "img.png").string();
    cv::Mat img = cv::Mat::zeros(32, 32, CV_8UC1);
    img.at<uchar>(10, 10) = 255;
    ASSERT_TRUE(savePng(p, img));
    const cv::Mat back = cv::imread(p, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(back.empty());
    EXPECT_EQ(back.at<uchar>(10, 10), 255);
}

// ============================ frame_source ============================

TEST(FrameType, LaserBandVsDots) {
    cv::Mat laser = cv::Mat::zeros(240, 320, CV_8UC1);
    cv::line(laser, cv::Point(0, 120), cv::Point(319, 125), 255, 5);   // 横贯亮带
    EXPECT_TRUE(looksLikeLaserFrame(laser));

    cv::Mat dots = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int i = 0; i < 6; ++i)   // 间距 60px > 直径 16px：列簇有断口，不成横贯带
        cv::circle(dots, cv::Point(30 + i * 55, 50 + (i % 3) * 60), 8, 255, -1);
    EXPECT_FALSE(looksLikeLaserFrame(dots));
}

// ============================ calib_loader ============================

TEST(CalibLoader, MatFromNestedAndFlat) {
    nlohmann::json nested = nlohmann::json::parse("[[1,2,3],[4,5,6],[7,8,9]]");
    const cv::Mat m = matFromJson(nested, 3, 3);
    ASSERT_EQ(m.rows, 3);
    EXPECT_DOUBLE_EQ(m.at<double>(1, 1), 5.0);

    nlohmann::json flat = nlohmann::json::parse("[-131.2, 2.08, 22.5]");
    const cv::Mat t = matFromJson(flat, 3, 1);
    ASSERT_EQ(t.rows, 3);
    EXPECT_DOUBLE_EQ(t.at<double>(2, 0), 22.5);

    nlohmann::json bad = nlohmann::json::parse("[[1,2],[3,4]]");
    EXPECT_TRUE(matFromJson(bad, 3, 3).empty());
}

TEST(CalibLoader, SelectNearestTier) {
    nlohmann::json table = nlohmann::json::array();
    for (int k = 0; k < 5; ++k)
        table.push_back({{"temperature", 7.5 + k * 0.5}});
    const auto* t = selectTier(table, 8.6);   // 8.5 vs 9.0 → 就近 8.5
    ASSERT_NE(t, nullptr);
    EXPECT_DOUBLE_EQ(t->at("temperature").get<double>(), 8.5);
    const auto* lo = selectTier(table, 5.0);  // 越界 → 低温端
    EXPECT_DOUBLE_EQ(lo->at("temperature").get<double>(), 7.5);
}

TEST(CalibLoader, RowStepCheck) {
    std::string msg;
    EXPECT_EQ(checkRowStep(0.7f, 0.7f, msg), 0);
    EXPECT_EQ(checkRowStep(0.7f, 1.4f, msg), 1);   // 倍数 → warn
    EXPECT_EQ(checkRowStep(0.7f, 0.55f, msg), -1); // 无关系 → error
}

// ============================ StegerDedup oracle ============================
// （合成图辅助已迁 tests/scr_test_helpers.h——MakeFlatTopSigma4/Flatten/
//  PointsInWindow，2026-10-18；精度敏感配方勿改）

// GPU（算子内 dedupMode=1，生产默认路径）vs CPU（外挂 steger_dedup::mergePoints
// 位移式，对照 oracle）同图对照。已知机理差异（Task 4 考古＋本用例实测存档，
// 勿硬凑逐点相等）：
//   - GPU：Hessian 解析切向——合成双点跨列交错成段 → 质心 (x+0.5, yc)（沿线偏
//     0.5px、法向 0）；部分段 >8 链量化边界直通（保留原 (x, yc±δ) 双点）
//   - CPU：kNN-PCA 估计切向≈(1,0)——同列上下双点 along≈0 直配 → 质心 (x, yc)
//   ⇒ 两法逐列桶结构不同（GPU 半整数质心 vs CPU 整数质心；GPU 直通列 2 点 vs
//     CPU 每列 1 点），出点总数差＝GPU 直通段数。断言取设计不变量：双输出全
//     部落真线心法向 ≤0.5px ＋ 双向最近邻 ≤0.75px（切向质心归属差 0.5px＋余
//     量——两法切向来源不同，容差即设计口径）＋各自记账恒等；逐桶差异打印对照。
TEST(StegerDedupOracle, GpuOperatorMatchesCpuMergePoints) {
    cv::Mat mask;
    const cv::Mat gray = scr::MakeFlatTopSigma4(mask);
    const float yc = 120.5f;
    cv::cuda::GpuMat d_gray, d_mask;
    d_gray.upload(gray);
    d_mask.upload(mask);

    // a) 算子内 GPU 去重（dedupMode=1）；b) 同图 dedupMode=0 出点喂 CPU 合并
    calib::StegerParams pGpu;
    pGpu.sigma = 4.0f; pGpu.lowThreshold = 0.5f; pGpu.dedupMode = 1;
    calib::StegerExtractorFast opGpu(pGpu);
    opGpu.Warmup(gray.rows, gray.cols);
    const auto rGpu = opGpu.Execute(d_gray, d_mask, cv::cuda::Stream::Null(),
                                    calib::GroupMode::Flat);
    calib::StegerParams pOff;
    pOff.sigma = 4.0f; pOff.lowThreshold = 0.5f;
    calib::StegerExtractorFast opOff(pOff);
    opOff.Warmup(gray.rows, gray.cols);
    const auto rOff = opOff.Execute(d_gray, d_mask, cv::cuda::Stream::Null(),
                                    calib::GroupMode::Flat);
    ASSERT_TRUE(rGpu.success && rOff.success);
    ASSERT_EQ(rOff.mergedCount, 0);

    const std::vector<cv::Point2f> in = scr::Flatten(rOff);
    steger_dedup::Stats st;
    const std::vector<cv::Point2f> cpuOut = steger_dedup::mergePoints(in, &st);
    const std::vector<cv::Point2f> gpuOut = scr::Flatten(rGpu);

    // —— 输入结构前置（Task 4 配方复现）：窗内 ≥230 列恰 2 点 ——
    {
        std::map<int, int> twoPt;
        int twoPtCols = 0;
        for (const auto& q : scr::PointsInWindow(in)) ++twoPt[cvRound(q.x)];
        for (const auto& kv : twoPt)
            if (kv.second == 2) ++twoPtCols;
        EXPECT_GE(twoPtCols, 230);   // 实测 239（Task 4 同配方）
    }

    // —— 各自记账恒等：出点数 = 入点数 − 合并数 ——
    EXPECT_GT(rGpu.mergedCount, 100);                                        // 实测 ~204
    EXPECT_EQ(rGpu.totalPointCount, rOff.totalPointCount - rGpu.mergedCount);
    EXPECT_GT(st.mergedPoints, 100);
    EXPECT_EQ(st.inCount, rOff.totalPointCount);
    EXPECT_EQ(st.outCount, static_cast<int>(cpuOut.size()));
    EXPECT_EQ(st.outCount, st.inCount - st.mergedPoints);

    // —— 双输出全部落真线心（法向 ≤0.5px；窗口 [40,280)）——
    const auto gpuWin = scr::PointsInWindow(gpuOut);
    const auto cpuWin = scr::PointsInWindow(cpuOut);
    for (const auto& q : gpuWin) EXPECT_NEAR(q.y, yc, 0.5f) << "GPU x=" << q.x;
    for (const auto& q : cpuWin) EXPECT_NEAR(q.y, yc, 0.5f) << "CPU x=" << q.x;
    EXPECT_GT(static_cast<int>(gpuWin.size()), 230);
    EXPECT_GT(static_cast<int>(cpuWin.size()), 230);

    // —— 双向最近邻 ≤0.75px（切向质心归属差 ≤0.5px＋余量）——
    auto maxNnDist = [](const std::vector<cv::Point2f>& a,
                        const std::vector<cv::Point2f>& b) {
        float mx = 0.0f;
        for (const auto& p : a) {
            float best = std::numeric_limits<float>::max();
            for (const auto& q : b) {
                const float dx = p.x - q.x, dy = p.y - q.y;
                best = std::min(best, std::sqrt(dx * dx + dy * dy));
            }
            mx = std::max(mx, best);
        }
        return mx;
    };
    EXPECT_LE(maxNnDist(gpuWin, cpuWin), 0.75f) << "GPU→CPU 最近邻";
    EXPECT_LE(maxNnDist(cpuWin, gpuWin), 0.75f) << "CPU→GPU 最近邻";

    // —— 逐桶结构对照（诊断打印：归属机理差异，不作硬断言）——
    std::map<int, int> bGpu, bCpu;
    for (const auto& q : gpuWin) ++bGpu[cvRound(q.x)];
    for (const auto& q : cpuWin) ++bCpu[cvRound(q.x)];
    int mismatch = 0;
    for (const auto& kv : bCpu) {
        const auto it = bGpu.find(kv.first);
        const int g = it == bGpu.end() ? 0 : it->second;
        if (g != kv.second) ++mismatch;
    }
    std::cout << "[oracle] in=" << st.inCount
              << " | GPU out=" << rGpu.totalPointCount
              << " merged=" << rGpu.mergedCount
              << " | CPU out=" << st.outCount
              << " merged=" << st.mergedPoints
              << " | 桶数差异列=" << mismatch
              << "（GPU 直通/半整数质心 vs CPU 整列质心——机理差异，非缺陷）"
              << std::endl;
}

} // namespace scr
