/**
 * @file test_fusion_integration.cpp
 * @brief 【五链 P1】产线体素融合算子集成 gtest（串行链替换后的机读验收）
 *
 * 四指标（设计 §十一.1）：
 *   ① 逐帧喂点完整：fuseInputTotal == Σ(非hold成功帧 laserPoints3D)
 *   ② 导出一致：PLY 顶点数 == fuseFusedPointCount == fuseVoxelCount
 *   ③ NN 距离分布：无重合点（体素去重语义）；分布百分位落档
 *   ④ 体素覆盖率：fused/Σinput 落历史区间（仅记录）
 * 另：扩容/越限计数（sparse_30 口径 8M 槽应全 0）。
 * 数据缺席 → GTEST_SKIP。
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "calib_loader.h"
#include "frame_source.h"
#include "harness.h"
#include "scr_config.h"

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif

namespace {

using scr::ScrConfig;
struct FuseEnv {
    bool available = false, skipped = false, hardFail = false;
    std::string skipReason;
    ScrConfig cfg;
    std::unique_ptr<scr::CalibData> calib;
    std::unique_ptr<scr::FrameSource> src;
    std::unique_ptr<scr::Harness> harness;
};

FuseEnv& env() {
    static FuseEnv e = [] {
        FuseEnv x;
        std::string err;
        const std::string cfgPath = scr::ScrConfig::defaultConfigPath();
        if (!scr::ScrConfig::fromFile(cfgPath, x.cfg, err)) {
            x.skipped = true;
            x.skipReason = "config 装载失败: " + err;
            return x;
        }
        if (!x.cfg.paths.allExist(err)) {
            x.skipped = true;
            x.skipReason = "回放数据缺席: " + err;
            return x;
        }
        x.calib = std::make_unique<scr::CalibData>();
        if (!scr::loadCalib(x.cfg, *x.calib, err)) {
            x.hardFail = true;
            x.skipReason = "标定装载失败: " + err;
            return x;
        }
        x.src = scr::makeFrameSource(x.cfg, err);
        if (!x.src) {
            x.hardFail = true;
            x.skipReason = "帧源失败: " + err;
            return x;
        }
        x.harness = std::make_unique<scr::Harness>(x.cfg, *x.calib, *x.src, err);
        if (!err.empty()) {
            x.hardFail = true;
            x.skipReason = "Harness 构造失败: " + err;
            return x;
        }
        x.available = true;
        return x;
    }();
    return e;
}

struct FuseIntegrationTest : public ::testing::Test {
    void SetUp() override {
        auto& e = env();
        if (e.hardFail) FAIL() << e.skipReason;
        if (e.skipped || !e.available) GTEST_SKIP() << e.skipReason;
    }
};

bool IsFused(const std::string& s) {
    return s == "first" || s == "flow" || s == "fuse";
}

TEST_F(FuseIntegrationTest, FourMetricsRunAndExport) {
    auto& e = env();
    const int n = std::max(1, e.cfg.runtime.checkCycles);

    long long expectInput = 0;   // ① 期望 Σ＝逐帧（非hold成功帧）laserPoints3D
    int fusedFrames = 0, holdFrames = 0;
    for (int c = 0; c < n; ++c) {
        scr::CycleProducts p;
        std::string err;
        ASSERT_TRUE(e.harness->runCycle(c, p, /*recordTiming=*/false, err)) << err;
        if (IsFused(p.regSource) && p.laserPoints3D > 0) {
            expectInput += p.laserPoints3D;
            ++fusedFrames;
        } else if (p.regSource == "hold" || p.regSource == "skip") {
            ++holdFrames;
        }
        e.harness->accumulateGlobalLaser(p);   // 与 main.cpp 同口
    }

    // ① 喂点完整
    EXPECT_EQ(e.harness->fuseInputTotal(), static_cast<size_t>(expectInput))
        << "Σ inputCount 与 Σ laserPoints3D 不一致（漏喂/重喂）";
    EXPECT_EQ(e.harness->fuseFrameCount(), static_cast<size_t>(fusedFrames));

    // ② 导出一致
    const std::string ply = std::string(SCR_SOURCE_DIR) + "/out_fuse_test.ply";
    const size_t exported = e.harness->saveGlobalLaserPly(ply);
    ASSERT_GT(exported, 0u) << "导出点数为 0（融合帧=" << fusedFrames << "）";
    EXPECT_EQ(exported, e.harness->fuseFusedPointCount());
    EXPECT_EQ(exported, e.harness->fuseVoxelCount());

    // ④ 覆盖率落档（仅记录）
    const double coverage =
        static_cast<double>(exported) / static_cast<double>(expectInput);
    std::printf("[fuse-metrics] fusedFrames=%d hold=%d Σinput=%lld exported=%zu "
                "coverage=%.3f rehash=%zu overload=%zu\n",
                fusedFrames, holdFrames, expectInput, exported, coverage,
                e.harness->fuseRehashCount(), e.harness->fuseOverloadCount());
    EXPECT_GT(coverage, 0.1);
    EXPECT_LT(coverage, 1.0);

    // sparse_30 口径 8M 槽：零扩容零越限
    EXPECT_EQ(e.harness->fuseRehashCount(), 0u);
    EXPECT_EQ(e.harness->fuseOverloadCount(), 0u);

    // ③ NN 距离分布（x 排序＋扫描窗）
    std::ifstream f(ply);
    ASSERT_TRUE(f.is_open());
    std::string line;
    std::getline(f, line);   // ply
    std::getline(f, line);   // format
    long vertexCount = 0;
    while (std::getline(f, line)) {
        if (line.rfind("element vertex", 0) == 0)
            vertexCount = std::atol(line.c_str() + 15);
        if (line == "end_header") break;
    }
    ASSERT_EQ(vertexCount, static_cast<long>(exported));
    std::vector<std::array<double, 3>> pts;
    pts.reserve(exported);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        double x = 0, y = 0, z = 0;
        if (sscanf_s(line.c_str(), "%lf %lf %lf", &x, &y, &z) == 3)
            pts.push_back({x, y, z});
    }
    ASSERT_EQ(pts.size(), exported);
    std::sort(pts.begin(), pts.end(),
              [](const auto& a, const auto& b) { return a[0] < b[0]; });
    std::vector<double> nn;
    nn.reserve(pts.size());
    const double win = 2.0;   // mm 扫描半径（>体素对角线 0.43mm）
    for (size_t i = 0; i < pts.size(); ++i) {
        double best = 1e9;
        for (size_t j = i + 1; j < pts.size(); ++j) {
            const double dx = pts[j][0] - pts[i][0];
            if (dx >= win) break;
            const double dy = pts[j][1] - pts[i][1];
            const double dz = pts[j][2] - pts[i][2];
            if (std::abs(dy) >= win || std::abs(dz) >= win) continue;
            const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < best && d > 0) best = d;
        }
        if (best < 1e9) nn.push_back(best);
    }
    ASSERT_GT(nn.size(), 1000u) << "NN 样本不足";
    std::sort(nn.begin(), nn.end());
    const double p01 = nn[nn.size() / 100];
    const double p50 = nn[nn.size() / 2];
    const double p99 = nn[nn.size() * 99 / 100];
    std::printf("[fuse-nn] n=%zu P01=%.4f P50=%.4f P99=%.4f min=%.6f (mm)\n",
                nn.size(), p01, p50, p99, nn.front());
    // 门：无重合点（P01 ≥ 0.02mm≈0.08×体素；体素代表点理论最近≈面对角布置）
    EXPECT_GT(p01, 0.02) << "存在近重合点（体素去重失效？）";
}

} // namespace
