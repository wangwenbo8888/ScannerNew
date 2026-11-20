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

// 【T0-2·测试债①】单线程臂（splitChain1=false）RLE 溢出不死等——复用 RunMode，
// 判定同 RleOverflowTombstoneNoStall（N-2 后单线程臂 evGate_ 消费路径一并受验）
TEST(FiveChain, Fcsplit0OverflowNoStall) {
    scr::ScrFiveChain fc;
    fc.kRleCapOverride = 100;
    fc.splitChain1 = false;   // 单线程回退臂（chain1Loop）
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
    EXPECT_LE(r.harness->fuseFrameCount(), static_cast<size_t>(serialHead));
}

// 【T0-2·测试债②】submitFailInject：注入一次提交失败（submitLaserFront fail+
// return 0 → frontKind=0 →不排水→gpuDrained=false）——两臂（二分/单线程）各跑
// 一遍；判定：不死等（测试正常返回即证——析构 join 全线程）＋八通道对账＋
// 注入帧激光链失败（laserMatched==0 且无死等）
TEST(FiveChain, SubmitFailInjectFallbackNoDeadlock) {
    for (int arm = 0; arm < 2; ++arm) {
        scr::ScrFiveChain fc;
        fc.submitFailInject = 1;
        fc.splitChain1 = (arm == 0);   // arm0=二分（chain1bLoop）/arm1=单线程（chain1Loop）
        FcRunResult r = RunMode("fivechain", fc);
        if (!r.ok) GTEST_SKIP() << r.skipReason;

        ScrConfig cfgCheck;
        std::string cerr;
        ASSERT_TRUE(ScrConfig::fromFile(ScrConfig::defaultConfigPath(), cfgCheck, cerr));
        const int total = std::max(1, cfgCheck.runtime.checkCycles);
        const int serialHead = std::min(total,
            cfgCheck.runtime.warmupCycles + cfgCheck.runtime.dumpFullFrames);
        const int expectFc = total - serialHead;
        EXPECT_EQ(r.cycles, total) << "arm " << arm << " 帧零丢失";
        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(r.harness->fcChCount(i), expectFc)
                << "arm " << arm << " 通道 " << i;
        int laserZero = 0;
        for (int c = serialHead; c < total; ++c)
            if (r.frames[static_cast<size_t>(c)].laserMatched == 0) ++laserZero;
        EXPECT_GE(laserZero, 1) << "arm " << arm << " 注入帧激光产物应归零";
        std::printf("[fc:submitfail] arm=%d laserZero=%d/%d\n", arm, laserZero, expectFc);
    }
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

// 【批C·T5】【批H 后】RLE 槽图执行器级 A/B（红队B E4：RunMode 不收 env——
// _putenv 切换＋还原；H 翻默认后 off 臂须显式 =0；构造器每次重读 env 与 K12 配套）
TEST(FiveChain, RleGraphExecutorParity) {
    struct PutenvRestore { ~PutenvRestore() { _putenv("JMW_RLEGRAPH="); } } raii;
    _putenv("JMW_RLEGRAPH=0");
    FcRunResult off = RunMode("fivechain");
    if (!off.ok) GTEST_SKIP() << off.skipReason;
    _putenv("JMW_RLEGRAPH=1");
    FcRunResult on = RunMode("fivechain");
    _putenv("JMW_RLEGRAPH=");   // 立即还原（空值＝默认开）
    ASSERT_TRUE(on.ok) << on.skipReason;
    ASSERT_EQ(off.cycles, on.cycles);
    int regSrcDiff = 0, markerDiff = 0, laserOverTol = 0;
    for (size_t c = 0; c < off.frames.size(); ++c) {
        const CycleProducts& a = off.frames[c];
        const CycleProducts& b = on.frames[c];
        if (a.regSource != b.regSource) ++regSrcDiff;
        if (a.markersDetected != b.markersDetected
            || a.markersValid != b.markersValid) ++markerDiff;
        const double tol = std::max(4.0, 0.01 * std::abs(a.laserMatched));
        if (std::abs(a.laserMatched - b.laserMatched) > tol) ++laserOverTol;
    }
    EXPECT_EQ(regSrcDiff, 0) << "图臂 regSource 漂移";
    EXPECT_EQ(markerDiff, 0) << "图臂标记点计数漂移（RLE 结果错误会现形于此）";
    EXPECT_EQ(laserOverTol, 0) << "图臂激光计数超差";
}

// 【G′-a】【G′-b·翻默认后 off 臂须显式 =0】三合一大图执行器级 A/B：
// JMW_1BBIGGRAPH（flat+undistort+1B 三合一，默认开）=0 对照 vs 默认开逐帧对拍；
// 含回退注入变体（JMW_1BBIG_INJECTFAIL=1——首捕窗中体强制失败一次，C8 幂等
// 重做+次帧重捕恢复受验）
TEST(FiveChain, BigGraphExecutorParity) {
    struct PutenvRestore {
        ~PutenvRestore() { _putenv("JMW_1BBIGGRAPH="); _putenv("JMW_1BBIG_INJECTFAIL="); }
    } raii;
    _putenv("JMW_1BBIGGRAPH=0");
    FcRunResult off = RunMode("fivechain");
    _putenv("JMW_1BBIGGRAPH=");   // 立即还原（空值＝默认开）
    if (!off.ok) GTEST_SKIP() << off.skipReason;
    for (int variant = 0; variant < 2; ++variant) {
        _putenv("JMW_1BBIGGRAPH=1");
        _putenv(variant == 0 ? "JMW_1BBIG_INJECTFAIL=0" : "JMW_1BBIG_INJECTFAIL=1");
        FcRunResult on = RunMode("fivechain");
        _putenv("JMW_1BBIGGRAPH=");
        _putenv("JMW_1BBIG_INJECTFAIL=");
        ASSERT_TRUE(on.ok) << on.skipReason;
        ASSERT_EQ(off.cycles, on.cycles) << "variant " << variant;
        int regSrcDiff = 0, markerDiff = 0, laserOverTol = 0;
        for (size_t c = 0; c < off.frames.size(); ++c) {
            const CycleProducts& a = off.frames[c];
            const CycleProducts& b = on.frames[c];
            if (a.regSource != b.regSource) ++regSrcDiff;
            if (a.markersDetected != b.markersDetected
                || a.markersValid != b.markersValid) ++markerDiff;
            const double tol = std::max(4.0, 0.01 * std::abs(a.laserMatched));
            if (std::abs(a.laserMatched - b.laserMatched) > tol) ++laserOverTol;
        }
        EXPECT_EQ(regSrcDiff, 0) << "variant " << variant << " regSource 漂移";
        EXPECT_EQ(markerDiff, 0) << "variant " << variant << " 标记点计数漂移";
        EXPECT_EQ(laserOverTol, 0) << "variant " << variant << " 激光计数超差";
    }
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
