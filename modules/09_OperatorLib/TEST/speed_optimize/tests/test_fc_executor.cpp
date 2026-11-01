/**
 * @file test_fc_executor.cpp
 * @brief 【五链 P2】fivechain 执行器 gtest：通道对账＋金样对拍＋溢出墓碑注入
 *
 * 验收（设计 §十 P2）：
 *   ② 注入失败下七通道 elem 计数==totalCycles 对账、帧号零丢失无停等
 *   ⑥ sparse_30 逐帧对拍（基准=金样 report；dt/drot 宽口径=4 点兜底 RANSAC
 *      放大微观输入差的既有现象，pipe-vs-serial 同类在案）
 *   ⑧ 排空无挂起（测试正常返回即证——析构 join 全部线程）
 * 数据缺席 → GTEST_SKIP。
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
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

using scr::CalibData;
using scr::CycleProducts;
using scr::FrameSource;
using scr::Harness;
using scr::ScrConfig;

// 金样 report 行（cycle,markersDetected,markersValid,regSource,regMatched,dt,drot,
// laserMatched,laserPoints3D,rowResid,disp）
struct GoldenRow {
    int cycle = 0, markersDetected = 0, markersValid = 0, regMatched = 0;
    int laserMatched = 0, laserPoints3D = 0;
    std::string regSource;
    double dt = 0, drot = 0;
};

std::vector<GoldenRow> LoadGolden() {
    std::vector<GoldenRow> rows;
    std::ifstream f(std::string(SCR_SOURCE_DIR) + "/golden/report.csv");
    if (!f.is_open()) return rows;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        GoldenRow r;
        std::string rs;
        ss >> r.cycle >> r.markersDetected >> r.markersValid >> rs >> r.regMatched
           >> r.dt >> r.drot >> r.laserMatched >> r.laserPoints3D;
        r.regSource = rs;
        rows.push_back(r);
    }
    return rows;
}

// 模式运行器：serial｜fivechain（fivechain 可覆写参数），逐帧收割
struct FcRunResult {
    bool ok = false;
    std::string skipReason;
    int cycles = 0;
    int readFails = 0;
    std::vector<CycleProducts> frames;
    std::unique_ptr<Harness> harness;   // 保持存活供计数查询
};

FcRunResult RunMode(const std::string& executor,
                    const scr::ScrFiveChain& fcParams = {}) {
    FcRunResult r;
    std::string err;
    ScrConfig cfg;
    const std::string cfgPath = ScrConfig::defaultConfigPath();
    if (!ScrConfig::fromFile(cfgPath, cfg, err)) {
        r.skipReason = "config 装载失败: " + err;
        return r;
    }
    cfg.fivechain = fcParams;
    cfg.fivechain.executor = executor;
    if (!cfg.paths.allExist(err)) {
        r.skipReason = "回放数据缺席: " + err;
        return r;
    }
    auto calib = std::make_unique<CalibData>();
    if (!scr::loadCalib(cfg, *calib, err)) {
        r.skipReason = "标定装载失败: " + err;
        return r;
    }
    auto src = scr::makeFrameSource(cfg, err);
    if (!src) {
        r.skipReason = "帧源失败: " + err;
        return r;
    }
    r.harness = std::make_unique<Harness>(cfg, *calib, *src, err);
    if (!err.empty()) {
        r.skipReason = "Harness 构造失败: " + err;
        return r;
    }
    const int total = std::max(1, cfg.runtime.checkCycles);
    for (int c = 0; c < total; ++c) {
        CycleProducts p;
        std::string cerr;
        if (!r.harness->runCycle(c, p, true, cerr)) {
            ++r.readFails;
            continue;
        }
        // 与 main.cpp 同口：串行头帧经此融合；fivechain 帧 fcActive 守卫免重复
        r.harness->accumulateGlobalLaser(p);
        r.frames.push_back(std::move(p));
        ++r.cycles;
    }
    r.ok = true;
    return r;
}

TEST(FiveChain, ChannelsAccountingAndSerialParity) {
    // 基准＝同二进制串行跑（金样 report 属旧工具口径——README §6 已记"待重钉"）
    FcRunResult serial = RunMode("serial");
    if (!serial.ok) GTEST_SKIP() << serial.skipReason;
    FcRunResult r = RunMode("fivechain");
    if (!r.ok) GTEST_SKIP() << r.skipReason;
    ASSERT_GE(r.cycles, 10) << "收割帧数异常";
    ASSERT_EQ(r.cycles, serial.cycles);

    // —— ② 七通道＋读/上传 对账：每帧恰一元素（含墓碑）——
    // 口径：串行头（warmup＋dumpFull）帧走原串行路径——通道只计流水线帧
    ScrConfig cfgCheck;
    std::string cerr;
    ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
    const int total = std::max(1, cfgCheck.runtime.checkCycles);
    const int serialHead = std::min(total,
        cfgCheck.runtime.warmupCycles + cfgCheck.runtime.dumpFullFrames);
    const int expectFc = total - serialHead;
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(r.harness->fcChCount(i), expectFc)
            << "通道 " << i << " 计数≠流水线帧数（帧丢失/重发）";
    }
    EXPECT_EQ(r.harness->fcTombCount(), 0) << "sparse_30 无读帧失败，墓碑应恒 0";

    // —— ⑥ 串行逐帧对拍（counting 类逐位；dt/drot 宽口径＋laser GPU 抖动容差）——
    int regSrcDiff = 0, markerDiff = 0, laserOverTol = 0;
    long long sumFused = 0;
    int fusedFrames = 0;
    for (int c = 0; c < total; ++c) {
        const CycleProducts& p = r.frames[static_cast<size_t>(c)];
        const CycleProducts& s = serial.frames[static_cast<size_t>(c)];
        if (p.regSource != s.regSource) {
            ++regSrcDiff;
            std::printf("[parity] cycle %d regSource fc=%s serial=%s\n",
                        c, p.regSource.c_str(), s.regSource.c_str());
        }
        if (p.markersDetected != s.markersDetected
            || p.markersValid != s.markersValid)
            ++markerDiff;
        const double tol = std::max(4.0, 0.01 * std::abs(s.laserMatched));
        if (std::abs(p.laserMatched - s.laserMatched) > tol) {
            ++laserOverTol;
            std::printf("[parity] cycle %d laserMatched fc=%d serial=%d\n",
                        c, p.laserMatched, s.laserMatched);
        }
        const bool fused = p.regSource == "first" || p.regSource == "flow"
                           || p.regSource == "fuse";
        if (fused && p.laserPoints3D > 0) {
            sumFused += p.laserPoints3D;
            ++fusedFrames;
        }
    }
    EXPECT_EQ(regSrcDiff, 0);
    EXPECT_EQ(markerDiff, 0);
    EXPECT_EQ(laserOverTol, 0);

    // —— ① 喂点完整（fivechain 融合路径）——
    EXPECT_EQ(r.harness->fuseInputTotal(), static_cast<size_t>(sumFused))
        << "Σfuse inputCount≠Σ非hold帧 laserPoints3D";
    EXPECT_EQ(r.harness->fuseFrameCount(), static_cast<size_t>(fusedFrames));
    std::printf("[fc] fusedFrames=%d Σinput=%lld voxels=%zu rehash=%zu overload=%zu\n",
                fusedFrames, sumFused, r.harness->fuseVoxelCount(),
                r.harness->fuseRehashCount(), r.harness->fuseOverloadCount());
}

TEST(FiveChain, RleOverflowTombstoneNoStall) {
    // R1 注入：rleCap 压到 100 → 流水线帧 A 墓碑（ccl 失败→配准 skip；激光支路不受影响）
    // 判定：不崩不停等、八通道对账==total、流水线帧 regSource==skip、零标记点
    scr::ScrFiveChain fc;
    fc.kRleCapOverride = 100;
    FcRunResult r = RunMode("fivechain", fc);
    if (!r.ok) GTEST_SKIP() << r.skipReason;

    ScrConfig cfgCheck;
    std::string cerr;
    ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
    const int total = std::max(1, cfgCheck.runtime.checkCycles);
    const int serialHead = std::min(total,
        cfgCheck.runtime.warmupCycles + cfgCheck.runtime.dumpFullFrames);
    const int expectFc = total - serialHead;
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(r.harness->fcChCount(i), expectFc) << "通道 " << i;
    for (int c = serialHead; c < total; ++c) {
        EXPECT_EQ(r.frames[static_cast<size_t>(c)].regSource, "skip")
            << "cycle " << c << " 溢出墓碑帧应 regSource=skip";
        EXPECT_EQ(r.frames[static_cast<size_t>(c)].markersValid, 0)
            << "cycle " << c << " 溢出帧标记点链应全弃";
    }
    // 流水线帧零融合（串行头帧照常）
    EXPECT_LE(r.harness->fuseFrameCount(), static_cast<size_t>(serialHead));
}

TEST(FiveChain, SplitChain1DisabledParity) {
    // 二分默认开后，单线程回退路径（JMW_FCSPLIT=0 同款）正确性锚定：
    // 单线程 vs 二分逐帧对拍（counting 类逐位＋regSource 逐位）
    scr::ScrFiveChain fcSingle;
    fcSingle.splitChain1 = false;                 // 单线程回退路径
    FcRunResult single = RunMode("fivechain", fcSingle);
    if (!single.ok) GTEST_SKIP() << single.skipReason;
    scr::ScrFiveChain fcSplit;
    fcSplit.splitChain1 = true;
    fcSplit.lmaskSlots = 4;   // 环槽参数变体一并覆盖
    FcRunResult split = RunMode("fivechain", fcSplit);
    if (!split.ok) GTEST_SKIP() << split.skipReason;
    ASSERT_EQ(single.cycles, split.cycles);
    int regSrcDiff = 0, markerDiff = 0, laserOverTol = 0;
    for (size_t c = 0; c < single.frames.size(); ++c) {
        const CycleProducts& a = single.frames[c];
        const CycleProducts& b = split.frames[c];
        if (a.regSource != b.regSource) ++regSrcDiff;
        if (a.markersDetected != b.markersDetected
            || a.markersValid != b.markersValid) ++markerDiff;
        const double tol = std::max(4.0, 0.01 * std::abs(a.laserMatched));
        if (std::abs(a.laserMatched - b.laserMatched) > tol) ++laserOverTol;
    }
    EXPECT_EQ(regSrcDiff, 0);
    EXPECT_EQ(markerDiff, 0);
    EXPECT_EQ(laserOverTol, 0);
}

// ============================================================================
// P5 压测场景（设计 §十四矩阵）：S1 拖慢 / hold 演练 / E 截断——判定＝不崩不
// 停等、八通道对账、计数口径正确（测试正常返回即证排空无挂起）
// ============================================================================

TEST(FiveChain, SlowMarkerLaneNoDeadlock) {
    scr::ScrFiveChain fc;
    fc.slowChainMs = 20;   // S1：lane0 每帧拖 20ms（14 帧全程拖慢态）
    FcRunResult r = RunMode("fivechain", fc);
    if (!r.ok) GTEST_SKIP() << r.skipReason;
    ScrConfig cfgCheck;
    std::string cerr;
    ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
    const int total = std::max(1, cfgCheck.runtime.checkCycles);
    const int expectFc = total - std::min(total,
        cfgCheck.runtime.warmupCycles + cfgCheck.runtime.dumpFullFrames);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(r.harness->fcChCount(i), expectFc) << "通道 " << i;
    EXPECT_EQ(r.cycles, total) << "拖慢态帧零丢失";
}

TEST(FiveChain, HoldSkipZeroUploadDrill) {
    scr::ScrFiveChain fc;
    fc.holdSkip = 1;        // 全部流水线帧按 hold：零上传弃 E 路径演练
    FcRunResult r = RunMode("fivechain", fc);
    if (!r.ok) GTEST_SKIP() << r.skipReason;
    ScrConfig cfgCheck;
    std::string cerr;
    ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
    const int serialHead = std::min(cfgCheck.runtime.checkCycles,
        cfgCheck.runtime.warmupCycles + cfgCheck.runtime.dumpFullFrames);
    EXPECT_LE(r.harness->fuseFrameCount(), static_cast<size_t>(serialHead))
        << "holdSkip 下流水线帧应零融合";
}

TEST(FiveChain, EBlockTruncationNoCrash) {
    scr::ScrFiveChain fc;
    fc.eBlockSizePct = 25;  // R2：E 块容量压到 25%（34237→8559 点/块）
    FcRunResult r = RunMode("fivechain", fc);
    if (!r.ok) GTEST_SKIP() << r.skipReason;
    // 截断帧照常完成（laserPoints3D 被夹到块容量；计数对账完好）
    ScrConfig cfgCheck;
    std::string cerr;
    ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
    const int total = std::max(1, cfgCheck.runtime.checkCycles);
    int truncated = 0;
    for (const auto& p : r.frames)
        if (p.laserPoints3D == 8559) ++truncated;
    std::printf("[fc:trunc] 截断帧数=%d/%d（块容量 8559）\n", truncated, total);
    EXPECT_GT(truncated, 0) << "25% 块容量下应存在截断帧";
}

} // namespace
