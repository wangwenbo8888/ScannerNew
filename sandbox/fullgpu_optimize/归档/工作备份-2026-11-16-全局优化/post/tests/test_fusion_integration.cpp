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

// ── 【B2·16B 标量拼包】count 永不清边界＋stats 逐字段回归 ──
// 守护目标：帧首 memset 只清 scalarPack[0..2]（12B stats），count 槽 [3] 跨帧
// 原子累积永不清（P2-0 脏读前科防线）；拼包 D2H 落地后各字段逐位正确。
TEST(FuseScalarPack, CountNeverClearedAndStatsFields) {
    calib::LaserCloudFuseCUDAParams p;
    p.voxelSize = 2.0f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 4096;
    p.growSlotCount = 4096;
    calib::LaserCloudFuseCuda fuse(p);

    auto makePoints = [](float off) {
        std::vector<cv::Vec3f> host;
        host.reserve(64);
        for (int i = 0; i < 64; ++i)
            host.push_back(cv::Vec3f(off + (i % 8) * 10.0f, off + (i / 8) * 10.0f, off));
        cv::Mat m(host);                       // 64×1 CV_32FC3
        cv::Mat row = m.reshape(3, 1);         // 1×64
        return cv::cuda::GpuMat(row);          // 上传（隐式默认流，同步语义）
    };

    // 帧1：64 新体素
    auto r1 = fuse.Execute(makePoints(0.0f));
    ASSERT_TRUE(r1.success) << r1.message;
    EXPECT_EQ(64, r1.newVoxelCount);
    EXPECT_EQ(64, r1.survivingCount);
    EXPECT_EQ(0, r1.deletedCount);
    EXPECT_EQ(64, r1.totalVoxelCount);

    // 帧2：同 64 点重喂——count 不变（永不清边界核心断言），无新体素
    auto r2 = fuse.Execute(makePoints(0.0f));
    ASSERT_TRUE(r2.success) << r2.message;
    EXPECT_EQ(0, r2.newVoxelCount);
    EXPECT_EQ(64, r2.survivingCount);
    EXPECT_EQ(0, r2.deletedCount);
    EXPECT_EQ(64, r2.totalVoxelCount) << "count 被帧首 memset 误清（B2 边界破坏）";

    // 帧3：+64 新点——单调不减＋新体素账目正确
    auto r3 = fuse.Execute(makePoints(500.0f));
    ASSERT_TRUE(r3.success) << r3.message;
    EXPECT_EQ(64, r3.newVoxelCount);
    EXPECT_EQ(128, r3.totalVoxelCount);
    EXPECT_GE(r3.totalVoxelCount, r2.totalVoxelCount);

    // removePoints 半数——count 槽显式写路径联动
    std::vector<uint32_t> half;
    for (uint32_t i = 0; i < 64; ++i) half.push_back(2 * i);   // 偶数下标
    auto rm = fuse.removePoints(half);
    ASSERT_TRUE(rm.success) << rm.message;
    EXPECT_EQ(64u, fuse.GetFusedPointCount());

    // Clear——全量复位（含 count 槽）
    fuse.Clear();
    EXPECT_EQ(0u, fuse.GetFusedPointCount());
}

// ── 【批F·fuse 段图化】rehash 毁图重捕守护＋图/逐发双实例逐字段对拍 ──
TEST(FuseGraph, RehashRecaptureAndParity) {
    _putenv("JMW_FUSEGRAPH=1");
    struct PutenvRestore { ~PutenvRestore() { _putenv("JMW_FUSEGRAPH=0"); } } raii;

    calib::LaserCloudFuseCUDAParams p;
    p.voxelSize = 2.0f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 64;    // 微容量——数帧内跨 70% 负载触发 rehash
    p.growSlotCount = 64;
    calib::LaserCloudFuseCuda fuseGraph(p);   // 图臂实例（构造器读 env）
    _putenv("JMW_FUSEGRAPH=0");
    calib::LaserCloudFuseCuda fuseEager(p);   // 逐发基准实例
    _putenv("JMW_FUSEGRAPH=1");

    auto makePoints = [](float off, int span) {
        std::vector<cv::Vec3f> host;
        host.reserve(128);
        for (int i = 0; i < 128; ++i)
            host.push_back(cv::Vec3f(off + (i % 8) * 10.0f,
                                     off + ((i / 8) % span) * 10.0f,
                                     off + (i / (8 * span)) * 10.0f));
        cv::Mat m(host);
        return cv::cuda::GpuMat(m.reshape(3, 1));
    };

    bool rehashSeen = false;
    int prevTotal = 0;
    for (int f = 0; f < 12; ++f) {
        auto g = fuseGraph.Execute(makePoints(100.0f * f, 4 + f % 3));
        auto e = fuseEager.Execute(makePoints(100.0f * f, 4 + f % 3));
        ASSERT_TRUE(g.success) << g.message;
        ASSERT_TRUE(e.success) << e.message;
        // 图/逐发逐字段对拍（同数据双实例）
        EXPECT_EQ(e.totalVoxelCount, g.totalVoxelCount) << "frame " << f;
        EXPECT_EQ(e.newVoxelCount, g.newVoxelCount) << "frame " << f;
        EXPECT_EQ(e.survivingCount, g.survivingCount) << "frame " << f;
        EXPECT_EQ(e.deletedCount, g.deletedCount) << "frame " << f;
        // count 永不清边界（跨 rehash）
        EXPECT_GE(g.totalVoxelCount, prevTotal) << "frame " << f;
        prevTotal = g.totalVoxelCount;
        rehashSeen = rehashSeen || g.rehashHappened;
    }
    EXPECT_TRUE(rehashSeen) << "微容量 12 帧应至少触发一次 rehash（重捕守护路径）";
    // rehash 后（重捕完成态）追加一致性帧
    auto g = fuseGraph.Execute(makePoints(5000.0f, 5));
    auto e = fuseEager.Execute(makePoints(5000.0f, 5));
    ASSERT_TRUE(g.success && e.success);
    EXPECT_EQ(e.totalVoxelCount, g.totalVoxelCount);
}

// ── 【根因②修复 B'·确定性单测】图臂稳态参数块必须拿当帧输入指针 ──
// 历史缺陷：executeDeferredImpl 稳态 writeFuseParamsPinned_ 误传 fuseSig_（建图
// 帧签名）→参数块 d_inputXyz 恒为建图帧旧指针→第 2 帧起内核每次读旧缓冲（帧
// N 位姿×帧 M 点＝五链 3-50mm 散布）。注意：图臂【仅存在于 ExecuteDeferred】
// ——既有 FuseGraph 用例走 Execute（fuseImpl 逐发路径）从未覆盖图臂；且其逐帧
// 新建即毁 GpuMat，分配器复用地址使新旧指针碰巧相同，即使走图臂也测不出。
// 本用例：ExecuteDeferred+Finish 打图臂＋两缓冲【同时存活】＝地址必不同→
// 确定性复现指针轮换；:596 修复前必红（第 2 帧 0 新体素），修复后必绿。
TEST(FuseGraph, InputRefreshRotatingBuffers) {
    _putenv("JMW_FUSEGRAPH=1");
    struct PutenvRestore { ~PutenvRestore() { _putenv("JMW_FUSEGRAPH=0"); } } raii;

    calib::LaserCloudFuseCUDAParams p;
    p.voxelSize = 2.0f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 4096;
    p.growSlotCount = 4096;
    calib::LaserCloudFuseCuda fuseGraph(p);   // 图臂（受测：ExecuteDeferred 路径）
    _putenv("JMW_FUSEGRAPH=0");
    calib::LaserCloudFuseCuda fuseEager(p);   // 逐发基准（Execute 直传当帧指针）
    _putenv("JMW_FUSEGRAPH=1");

    auto makePoints = [](float off) {
        std::vector<cv::Vec3f> host;
        host.reserve(64);
        for (int i = 0; i < 64; ++i)
            host.push_back(cv::Vec3f(off + (i % 8) * 10.0f, off + (i / 8) * 10.0f, off));
        cv::Mat m(host);
        return cv::cuda::GpuMat(m.reshape(3, 1));   // 上传后临时 Mat 析构、显存常驻
    };
    // 两块同时存活的输入（A/B 相距 1000 体素空间，地址必不同）
    auto bufA = makePoints(0.0f);
    auto bufB = makePoints(1000.0f);
    ASSERT_NE(bufA.ptr(), bufB.ptr()) << "分配器意外复用地址——用例前提破坏";

    const cv::Matx33d eyeR = cv::Matx33d::eye();
    const cv::Vec3d zeroT(0, 0, 0);
    cv::cuda::Stream s;

    // 图臂需 2 帧暖机（impl: fuseGraphWarm_>=2）：帧0 eager 暖机（喂 A）
    auto dw = fuseGraph.ExecuteDeferred(bufA, eyeR, zeroT, s);
    ASSERT_TRUE(dw.success) << dw.message;
    ASSERT_TRUE(fuseGraph.Finish().success);
    auto ew = fuseEager.Execute(bufA);
    ASSERT_TRUE(ew.success);
    ASSERT_EQ(64, ew.newVoxelCount);

    // 帧1：A 走图臂【建图】（fuseSig_=cur=A 指针；本帧参数源正确性＝建图路径）
    auto d1 = fuseGraph.ExecuteDeferred(bufA, eyeR, zeroT, s);
    ASSERT_TRUE(d1.success) << d1.message;
    ASSERT_NE(std::string::npos, d1.message.find("CUDA graph"))
        << "图臂未启用（用例前提破坏）： " << d1.message;
    auto g1 = fuseGraph.Finish();
    auto e1 = fuseEager.Execute(bufA);
    ASSERT_TRUE(g1.success && e1.success);
    EXPECT_EQ(e1.newVoxelCount, g1.newVoxelCount) << "建图帧（重融 A：均 0 新）";

    // 帧2：换 B——【稳态＝:596 bug 位点】修复前参数块仍指 A（重融 A：0 新体素）
    // ；修复后指 B（+64 新体素）
    ASSERT_TRUE(fuseGraph.ExecuteDeferred(bufB, eyeR, zeroT, s).success);
    auto g2 = fuseGraph.Finish();
    auto e2 = fuseEager.Execute(bufB);
    ASSERT_TRUE(g2.success && e2.success);
    EXPECT_EQ(e2.newVoxelCount, g2.newVoxelCount)
        << "图臂稳态未取当帧输入（陈旧指针复现：newVoxel graph="
        << g2.newVoxelCount << " eager=" << e2.newVoxelCount << "）";
    EXPECT_EQ(64, g2.newVoxelCount) << "稳态帧应新增 64 体素（B 区域）";
    EXPECT_EQ(e2.totalVoxelCount, g2.totalVoxelCount);

    // 帧3：回 A——再次轮换方向（B→A），同判
    ASSERT_TRUE(fuseGraph.ExecuteDeferred(bufA, eyeR, zeroT, s).success);
    auto g3 = fuseGraph.Finish();
    auto e3 = fuseEager.Execute(bufA);
    ASSERT_TRUE(g3.success && e3.success);
    EXPECT_EQ(e3.totalVoxelCount, g3.totalVoxelCount);
}

} // namespace
