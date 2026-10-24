// test_scan_chain_replay.cpp — 链级回放 gtest 前端（断言策略：两步钉阈值）
//
// 首步（check 为空）：统计 report-only＋宽松下限（标 [PIN-PENDING]）；
// 钉阈值后（config runtime.check 填入）：按阈值硬断言。
// 数据缺失（CESHI260831 不在）→ 全部 SKIP；CUDA 初始化失败 → FAIL。
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "calib_loader.h"
#include "frame_source.h"
#include "harness.h"
#include "scr_config.h"

namespace scr {
namespace fs = std::filesystem;

namespace {

constexpr const char* kPinPending = "[PIN-PENDING]";

/// 链环境（SetUpTestSuite 一次装载，全 suite 共享）
struct ChainEnv {
    bool available = false;
    bool skipped = false;      // 数据缺席 → SKIP
    bool hardFail = false;     // 数据在但标定/构造失败 → FAIL
    std::string skipReason;
    ScrConfig cfg;
    std::unique_ptr<CalibData> calib;
    std::unique_ptr<FrameSource> src;
    std::unique_ptr<Harness> harness;   // Smoke/Stats 共用（prevState 链式推进）
    std::string outDir;
};

ChainEnv& env() {
    static ChainEnv e = [] {
        ChainEnv x;
        x.cfg = ScrConfig{};
        // 配置：--config 语义的测试版——exe 旁 → 源码模板
        const std::string cfgPath = ScrConfig::defaultConfigPath();
        std::string err;
        if (!ScrConfig::fromFile(cfgPath, x.cfg, err)) {
            x.skipped = true;
            x.skipReason = "config 装载失败: " + err + " (path=" + cfgPath + ")";
            return x;
        }
        if (!x.cfg.paths.allExist(err)) {
            x.skipped = true;
            x.skipReason = "回放数据缺席（SKIP 不算失败）: " + err;
            return x;
        }
        x.outDir = x.cfg.paths.outDir;
        x.calib = std::make_unique<CalibData>();
        if (!loadCalib(x.cfg, *x.calib, err)) {
            x.hardFail = true;   // 数据在而标定失败 → FAIL 而非 SKIP
            x.skipReason = "标定装载失败: " + err;
            return x;
        }
        x.src = makeFrameSource(x.cfg, err);
        if (!x.src) {
            x.hardFail = true;
            x.skipReason = "帧源失败: " + err;
            return x;
        }
        x.harness = std::make_unique<Harness>(x.cfg, *x.calib, *x.src, err);
        if (!err.empty()) {
            x.hardFail = true;
            x.skipReason = "Harness 构造失败: " + err;
            x.src.reset();
            return x;
        }
        x.available = true;
        return x;
    }();
    return e;
}

struct ChainReplayTest : public ::testing::Test {
    void SetUp() override {
        auto& e = env();
        if (e.hardFail)
            FAIL() << e.skipReason;
        if (e.skipped || !e.available)
            GTEST_SKIP() << e.skipReason;
    }
};

/// 逐 cycle 统计收集器
struct RunStats {
    int cycles = 0;
    int markerCyclesOk = 0;     // markersValid ≥ 1
    int laserCyclesOk = 0;      // laserPoints3D ≥ 1
    int regFlowOk = 0;
    std::vector<int> markersValid, laserPoints, laserMatched, regMatched;
    std::vector<double> rowResid, dtMm, drotDeg;
    int stepFailTotal = 0;
};

RunStats runCycles(Harness& h, int n, bool dumpFirst = false) {
    RunStats s;
    double prevR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, prevT[3] = {0, 0, 0};
    bool havePrev = false;
    for (int c = 0; c < n; ++c) {
        CycleProducts p;
        std::string err;
        if (!h.runCycle(c, p, true, err)) {
            ADD_FAILURE() << "cycle " << c << " 读取失败: " << err;
            return s;
        }
        s.cycles++;
        s.stepFailTotal += static_cast<int>(p.stepError.size());
        s.markersValid.push_back(p.markersValid);
        s.laserPoints.push_back(p.laserPoints3D);
        s.laserMatched.push_back(p.laserMatched);
        s.regMatched.push_back(p.regMatched);
        s.rowResid.push_back(p.rowResidMedPx);
        if (p.markersDetected > 0) s.markerCyclesOk++;   // 链路存活口径（检出即活）
        if (p.laserPoints3D > 0) s.laserCyclesOk++;
        if (p.regSource == "flow" || p.regSource == "first") s.regFlowOk++;
        if (havePrev) {
            const double dt = std::sqrt((p.T[0] - prevT[0]) * (p.T[0] - prevT[0])
                                        + (p.T[1] - prevT[1]) * (p.T[1] - prevT[1])
                                        + (p.T[2] - prevT[2]) * (p.T[2] - prevT[2]));
            const cv::Matx33d dR = cv::Matx33d(p.R) * cv::Matx33d(prevR).t();
            const double tr = dR(0, 0) + dR(1, 1) + dR(2, 2);
            const double cosHalf = std::min(1.0, std::max(-1.0, (tr - 1.0) * 0.5));
            s.dtMm.push_back(dt);
            s.drotDeg.push_back(std::acos(cosHalf) * 180.0 / CV_PI);
        }
        std::copy(std::begin(p.R), std::end(p.R), std::begin(prevR));
        std::copy(std::begin(p.T), std::end(p.T), std::begin(prevT));
        havePrev = true;
        if (dumpFirst && c == 0)
            h.dumpCycle(0, p, true);
    }
    return s;
}

double medianV(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}
double maxV(const std::vector<double>& v) {
    if (v.empty()) return -1.0;
    return *std::max_element(v.begin(), v.end());
}

/// check 节取阈值；空 → report-only 宽松下限
double checkOr(const nlohmann::json& check, const char* key, double looseDefault) {
    if (check.contains(key)) return check.at(key).get<double>();
    std::cout << kPinPending << " 阈值 " << key << " 未钉，用宽松值 " << looseDefault
              << std::endl;
    return looseDefault;
}

} // namespace

// ------------------------------------------------------------------
// 冒烟：checkCycles 个 cycle 全链可跑、成功占比、产物落盘
// ------------------------------------------------------------------
TEST_F(ChainReplayTest, SmokeAndProducts) {
    auto& e = env();
    const int n = std::max(1, std::min(e.cfg.runtime.checkCycles,
                                       static_cast<int>(e.src->cycles.size())));
    const RunStats s = runCycles(*e.harness, n, /*dumpFirst=*/true);

    EXPECT_GT(s.cycles, 0);
    // 断言1 冒烟：步骤失败总数受控（允许偶发 Degraded，不允许链性失败）
    EXPECT_LT(s.stepFailTotal, s.cycles)
        << "步骤失败过多: " << s.stepFailTotal << "/" << s.cycles << " cycles";

    // 断言2 标记点门禁（宽松下限：≥ 半数 cycle 有有效标记点）
    const double markerRatioMin = checkOr(e.cfg.runtime.check, "markerCycleRatioMin", 0.5);
    EXPECT_GE(static_cast<double>(s.markerCyclesOk) / s.cycles, markerRatioMin);

    // 断言4 激光门禁（宽松下限）
    const double laserRatioMin = checkOr(e.cfg.runtime.check, "laserCycleRatioMin", 0.5);
    EXPECT_GE(static_cast<double>(s.laserCyclesOk) / s.cycles, laserRatioMin);

    // 产物存在性（cycle_000 全量 dump；L*/M* 链前缀命名，全部 L|R 并合）
    const fs::path d0 = fs::path(e.outDir) / "cycle_000";
    EXPECT_TRUE(fs::exists(d0 / "summary.json"));
    EXPECT_TRUE(fs::exists(d0 / "0-frames.png"));
    EXPECT_TRUE(fs::exists(d0 / "M10-markers3d.ply"));
    EXPECT_TRUE(fs::exists(d0 / "L5-laser3d.ply"));
}

// ------------------------------------------------------------------
// 双链矫正系一致性（undistort_cpu 回显 vs 温表档注入）
// ------------------------------------------------------------------
TEST_F(ChainReplayTest, CalibrationConsistency) {
    auto& e = env();
    CycleProducts p;
    std::string err;
    ASSERT_TRUE(e.harness->runCycle(0, p, false, err)) << err;
    std::string detail;
    const double d = e.harness->checkConsistency(p, detail);
    std::cout << "[consistency] " << detail << std::endl;
    const double tol = checkOr(e.cfg.runtime.check, "consistencyMaxDiff", 1e-6);
    EXPECT_LT(d, tol) << "双链矫正系不一致（R1/R2/P1/P2/Q 注入 vs 回显）: " << detail;
}

// ------------------------------------------------------------------
// 统计断言（阈值钉后硬断言；未钉 → 打印 [PIN-PENDING] 统计）
// ------------------------------------------------------------------
TEST_F(ChainReplayTest, StatsGates) {
    auto& e = env();
    const int n = std::max(1, std::min(e.cfg.runtime.checkCycles,
                                       static_cast<int>(e.src->cycles.size())));
    const RunStats s = runCycles(*e.harness, n);

    const double medMarkers = medianV(std::vector<double>(s.markersValid.begin(),
                                                           s.markersValid.end()));
    const double medLaser = medianV(std::vector<double>(s.laserPoints.begin(),
                                                        s.laserPoints.end()));
    const double medRowResid = medianV(s.rowResid);
    const double maxDt = maxV(s.dtMm);
    const double maxDrot = maxV(s.drotDeg);

    std::cout << kPinPending
              << " 统计: markersValid 中位=" << medMarkers
              << " laser3D 中位=" << medLaser
              << " 行残差中位(px)=" << medRowResid
              << "（标定本底 0.2246px）"
              << " |Δt| 最大(mm)=" << maxDt
              << " |Δrot| 最大(deg)=" << maxDrot << std::endl;

    // 断言2：标记点中位数
    const double markersMin = checkOr(e.cfg.runtime.check, "markersValidMedianMin", 3.0);
    EXPECT_GE(medMarkers, markersMin);
    // 断言3：R/T 平滑
    const double dtMax = checkOr(e.cfg.runtime.check, "rtDtMmMax", 50.0);
    const double drotMax = checkOr(e.cfg.runtime.check, "rtDrotDegMax", 10.0);
    EXPECT_LT(maxDt, dtMax);
    EXPECT_LT(maxDrot, drotMax);
    // 断言4：激光点中位数
    const double laserMin = checkOr(e.cfg.runtime.check, "laserPointsMedianMin", 100.0);
    EXPECT_GE(medLaser, laserMin);
    // 断言5：行残差中位数（极线自洽；须 > 标定本底 0.2246px 的口径）
    const double rowResidMax = checkOr(e.cfg.runtime.check, "rowResidMedPxMax", 2.0);
    EXPECT_LT(medRowResid, rowResidMax);
}

// ------------------------------------------------------------------
// 确定性：全新 Harness 跑同两 cycle，末 cycle 3D 输出逐点一致（哈希→容差）
// ------------------------------------------------------------------
TEST_F(ChainReplayTest, DeterminismFreshInstances) {
    auto& e = env();
    const int n = std::min(2, static_cast<int>(e.src->cycles.size()));
    ASSERT_GE(n, 1);

    auto run = [&](CycleProducts& lastOut) {
        std::string err;
        Harness h(e.cfg, *e.calib, *e.src, err);
        EXPECT_TRUE(err.empty()) << err;
        for (int c = 0; c < n; ++c) ASSERT_TRUE(h.runCycle(c, lastOut, false, err)) << err;
    };

    CycleProducts a, b;
    run(a);
    run(b);

    // 标记点 3D（CPU 链；⚠ 实测跨实例存在计数 1↔2 边界翻转＋ccl GPU 标签顺序
    // 抖动（同索引≠同一物理点）——按 §7#6 降级口径：双向最近邻几何比对（对顺序
    // 不敏感，0.1mm 容差），两侧无对应点数 ≤2 视为通过；大幅发散仍报红）
    {
        auto validCenters = [](const calib::ConicReconstructCPUResult& r) {
            std::vector<cv::Point3d> out;
            for (const auto& m : r.markers)
                if (m.validCircle) out.emplace_back(m.centerX, m.centerY, m.centerZ);
            return out;
        };
        const auto va = validCenters(a.conic), vb = validCenters(b.conic);
        constexpr double kTol = 0.1;   // mm
        auto unmatchedCount = [](const std::vector<cv::Point3d>& from,
                                 const std::vector<cv::Point3d>& to) {
            int miss = 0;
            for (const auto& p : from) {
                double best = 1e30;
                for (const auto& q : to) {
                    const double dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
                    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (d < best) best = d;
                }
                if (best > 0.1) ++miss;
            }
            return miss;
        };
        const int missAB = unmatchedCount(va, vb);
        const int missBA = unmatchedCount(vb, va);
        std::cout << "[determinism] markers: A=" << va.size() << " B=" << vb.size()
                  << " 无对应点 A→B=" << missAB << " B→A=" << missBA << std::endl;
        EXPECT_LE(missAB, 2);
        EXPECT_LE(missBA, 2);
    }

    // 激光点数一致（⚠ v3/interp 链存在运行间轻微非确定性——GPU 归约/原子顺序，
    // 实测 3↔4 点抖动；按设计 §7#6 降级为容差判定：|Δ|≤2 视为通过）
    const int laserDiff = std::abs(a.laserPoints3D - b.laserPoints3D);
    std::cout << "[determinism] laser3D: A=" << a.laserPoints3D
              << " B=" << b.laserPoints3D << " |Δ|=" << laserDiff
              << (laserDiff == 0 ? "（逐位一致）" : "（容差内，GPU 非确定性）")
              << std::endl;
    EXPECT_LE(laserDiff, 2);
    EXPECT_LE(std::abs(a.laserMatched - b.laserMatched), 2);
    // R/T 一致
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(a.R[i], b.R[i], 1e-9) << "R[" << i << "]";
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(a.T[i], b.T[i], 1e-9) << "T[" << i << "]";
}

// ------------------------------------------------------------------
// 去重对齐：operator（算子内 GPU 去重）vs cpu（外挂位移式 oracle）双路对拍
// ------------------------------------------------------------------
struct DedupAlign : public ::testing::Test {
    void SetUp() override {
        auto& e = env();
        if (e.hardFail)
            FAIL() << e.skipReason;
        if (e.skipped || !e.available)
            GTEST_SKIP() << e.skipReason;
    }
};

TEST_F(DedupAlign, ParityOperatorVsCpu) {
    auto& e = env();
    const int k = std::max(1, std::min({3, e.cfg.runtime.checkCycles,
                                        static_cast<int>(e.src->cycles.size())}));

    // 双路径汇总：Σ(L+R) post-dedup 出点 / Σ mergedCount（cpu 路径 apply 不写
    // mergedCount，恒 0）/ 逐 cycle 下游产物
    struct AlignSum {
        long long sumOut = 0, sumMerged = 0;
        std::vector<int> matched, pts3d;
    };
    auto runPath = [&](const char* dedupPath, AlignSum& s, std::string& errMsg) {
        ScrConfig cfg = e.cfg;          // 全同复制，仅切去重路径
        cfg.runtime.dedupPath = dedupPath;
        Harness h(cfg, *e.calib, *e.src, errMsg);
        if (!errMsg.empty()) return false;
        for (int c = 0; c < k; ++c) {
            CycleProducts p;
            if (!h.runCycle(c, p, false, errMsg)) return false;
            s.sumOut += p.stegerL.totalPointCount + p.stegerR.totalPointCount;
            s.sumMerged += p.stegerL.mergedCount + p.stegerR.mergedCount;
            s.matched.push_back(p.laserMatched);
            s.pts3d.push_back(p.laserPoints3D);
        }
        return true;
    };
    AlignSum op, cpu;
    std::string err;
    ASSERT_TRUE(runPath("operator", op, err)) << "operator 路径: " << err;
    ASSERT_TRUE(runPath("cpu", cpu, err)) << "cpu 路径: " << err;

    // merged 对齐（恒等式桥）：op 侧记账 N = opOut+opMerged（mergedCount=N−N'
    // 恒等式；金样已证 Off 逐位稳定 → cpu 路径同帧 dedupMode=0 的原始 N 与之
    // 相等）；cpu 侧 apply 后 totalPointCount=cpuOut 且不写 mergedCount →
    // cpuMerged = N − cpuOut
    const long long nRaw = op.sumOut + op.sumMerged;
    const long long cpuMerged = nRaw - cpu.sumOut;
    const double mergedRatio =
        static_cast<double>(op.sumMerged) / static_cast<double>(cpuMerged);

    // 下游逐 cycle 偏差（绝对/相对双口径，相对分母 cpu 侧）
    int maxDM = 0, maxDP = 0;
    double maxRM = 0.0, maxRP = 0.0;
    for (int c = 0; c < k; ++c) {
        const int dm = std::abs(op.matched[c] - cpu.matched[c]);
        const int dp = std::abs(op.pts3d[c] - cpu.pts3d[c]);
        maxDM = std::max(maxDM, dm);
        maxDP = std::max(maxDP, dp);
        if (cpu.matched[c] > 0)
            maxRM = std::max(maxRM, dm / static_cast<double>(cpu.matched[c]));
        if (cpu.pts3d[c] > 0)
            maxRP = std::max(maxRP, dp / static_cast<double>(cpu.pts3d[c]));
    }
    std::cout << kPinPending
              << " 去重对齐(" << k << " cycles): opMerged=" << op.sumMerged
              << " cpuMerged=" << cpuMerged << " 比值=" << mergedRatio
              << " | 下游 max|Δmatched|=" << maxDM << "(rel=" << maxRM << ")"
              << " max|Δ3D|=" << maxDP << "(rel=" << maxRP << ")" << std::endl;

    // 防空转门：数据无平顶 → 对齐判据无意义
    if (cpuMerged < 100)
        GTEST_SKIP() << "数据无平顶（cpuMerged=" << cpuMerged << " < 100）";

    // 断言① 合并量对齐：opMerged ≥ ratioMin × cpuMerged
    const double ratioMin = checkOr(e.cfg.runtime.check, "dedupAlignMergedRatioMin", 0.0);
    EXPECT_GE(op.sumMerged, ratioMin * cpuMerged)
        << "opMerged=" << op.sumMerged << " cpuMerged=" << cpuMerged
        << " ratio=" << mergedRatio;

    // 断言② 下游对齐：|op−cpu| ≤ max(5, diffMax×cpu)（绝对 5 点兜底 GPU 抖动）
    const double diffMax = checkOr(e.cfg.runtime.check, "dedupAlignDownstreamDiffMax", 1.0);
    for (int c = 0; c < k; ++c) {
        EXPECT_LE(std::abs(op.matched[c] - cpu.matched[c]),
                  std::max(5.0, diffMax * cpu.matched[c]))
            << "cycle " << c << " laserMatched: op=" << op.matched[c]
            << " cpu=" << cpu.matched[c];
        EXPECT_LE(std::abs(op.pts3d[c] - cpu.pts3d[c]),
                  std::max(5.0, diffMax * cpu.pts3d[c]))
            << "cycle " << c << " laserPoints3D: op=" << op.pts3d[c]
            << " cpu=" << cpu.pts3d[c];
    }
}

// ------------------------------------------------------------------
// 激光中心提取算子双路径对拍（flat_center vs steger_fast，2026-10-19）
// 设计：docs/plans/2026-10-18-flat-center-extractor-design.md §6 链级层。
// 出点＝Σ(L+R) post-dedup totalPointCount（DedupAlign AlignSum.sumOut 口径）；
// 数据门（SKIP）只看 steger 侧（oracle），flat 侧为硬断言（不随被测失效 disarm）。
// ------------------------------------------------------------------
struct ExtractorParity : public ::testing::Test {
    void SetUp() override {
        auto& e = env();
        if (e.hardFail) FAIL() << e.skipReason;
        if (e.skipped || !e.available) GTEST_SKIP() << e.skipReason;
    }
};
TEST_F(ExtractorParity, FlatVsStegerGolden) {
    auto& e = env();
    const int k = std::max(1, std::min({3, e.cfg.runtime.checkCycles,
                                        static_cast<int>(e.src->cycles.size())}));
    struct Sum { long long out = 0, collapsed = 0; std::vector<int> matched, pts3d; };
    // dy=1 双点折叠（同列 Δy≤1.5 计一对）——本数据集 steger 去重未并平顶双肩对
    // （金样实证 L1r==L1c），折叠后 ≈ 段数＝flat 逐段单点的同口径基准
    auto collapsePairs = [](const calib::StegerResult& r) -> int {
        if (!r.d_centerPoints || r.d_centerPoints->empty()) return 0;
        cv::Mat h; r.d_centerPoints->download(h);
        std::map<int, std::vector<float>> byCol;
        const cv::Point2f* p = h.ptr<cv::Point2f>();
        for (int i = 0; i < h.cols; ++i) byCol[cvRound(p[i].x)].push_back(p[i].y);
        int n = 0;
        for (auto& kv : byCol) {
            std::sort(kv.second.begin(), kv.second.end());
            for (size_t i = 0; i < kv.second.size(); ) {
                if (i + 1 < kv.second.size() && kv.second[i+1] - kv.second[i] <= 1.5f) {
                    ++n; i += 2;                     // 双点折叠为 1
                } else { ++n; ++i; }
            }
        }
        return n;
    };
    auto runPath = [&](const char* extractor, Sum& s, std::string& errMsg) {
        ScrConfig cfg = e.cfg;
        cfg.runtime.extractor = extractor;
        cfg.runtime.dedupPath = "operator";   // flat 侧防组合校验；两路径同口径
        Harness h(cfg, *e.calib, *e.src, errMsg);
        if (!errMsg.empty()) return false;
        for (int c = 0; c < k; ++c) {
            CycleProducts p;
            if (!h.runCycle(c, p, false, errMsg)) return false;
            if (c == 0 && std::string(extractor) == "flat")   // 诊断：首帧退化原因
                std::cout << "[flat-msg L] " << p.stegerL.message
                          << " N=" << p.stegerL.totalPointCount << "\n";
            s.out += p.stegerL.totalPointCount + p.stegerR.totalPointCount;
            if (std::string(extractor) == "steger")
                s.collapsed += collapsePairs(p.stegerL) + collapsePairs(p.stegerR);
            s.matched.push_back(p.laserMatched);
            s.pts3d.push_back(p.laserPoints3D);
        }
        return true;
    };
    Sum st, fl; std::string err;
    ASSERT_TRUE(runPath("steger", st, err)) << "steger: " << err;
    ASSERT_TRUE(runPath("flat", fl, err)) << "flat: " << err;
    if (st.out < 100LL * k) GTEST_SKIP() << "数据出点不足（steger Σ=" << st.out << "）";
    // 点数门：对"折叠 dy=1 双点后的 steger"（≈段数）比对——flat 逐段单点即达同口径。
    // 下界 0.85：折叠口径把间距 2-3px 的散点对计为独立点（天然高估段数，实测
    // 高估 ~10%）；flat 对自身段覆盖 ≥98%（dropped/segments 实测 324/16256）
    EXPECT_GE(fl.out, static_cast<long long>(0.85 * st.collapsed));
    EXPECT_LE(fl.out, static_cast<long long>(1.1 * st.collapsed) + 1);
    // 下游门：单点喂链 vs 双点喂链的匹配结构性差异（实测 4-8%，v3 对双点输入的
    // 匹配利用）——先 10% 观察口径，单点适配研究后再收紧
    for (int c = 0; c < k; ++c) {
        EXPECT_LE(std::abs(fl.matched[c] - st.matched[c]),
                  std::max(5.0, 0.10 * st.matched[c])) << "cycle " << c << " matched";
        EXPECT_LE(std::abs(fl.pts3d[c] - st.pts3d[c]),
                  std::max(5.0, 0.10 * st.pts3d[c])) << "cycle " << c << " pts3d";
    }
    std::cout << "[extractor] k=" << k << " stegerOut=" << st.out
              << " collapsed=" << st.collapsed << " flatOut=" << fl.out
              << " ratio(vs raw)=" << static_cast<double>(fl.out) / st.out << std::endl;
}

TEST_F(ExtractorParity, SpeedFlatVsStegerSameSession) {
    auto& e = env();
    const int warm = std::max(0, e.cfg.runtime.warmupCycles);
    const int k = std::max(1, std::min({e.cfg.runtime.checkCycles,
                                        static_cast<int>(e.src->cycles.size())}));
    if (k <= warm) GTEST_SKIP() << "帧数不足";
    // 同进程双 Harness＝同会话口径（CLI 双跑＝跨进程，排除）；record=true 取
    // stepMs["steger_fast"]（槽位键恒为 steger_fast，L+R 合计→÷2 为每眼）
    auto medOf = [&](const char* extractor, std::string& errMsg) {
        std::vector<double> ms;
        ScrConfig cfg = e.cfg; cfg.runtime.extractor = extractor;
        cfg.runtime.dedupPath = "operator"; cfg.runtime.dumpFullFrames = 0;
        Harness h(cfg, *e.calib, *e.src, errMsg);
        if (!errMsg.empty()) return ms;
        for (int c = 0; c < k; ++c) {
            CycleProducts p;
            if (!h.runCycle(c, p, true, errMsg)) return std::vector<double>{};
            for (auto& kv : p.stepMs)
                if (kv.first == "steger_fast" && c >= warm) ms.push_back(kv.second / 2.0);
        }
        return ms;
    };
    std::string err;
    auto a = medOf("steger", err); ASSERT_FALSE(a.empty()) << err;
    auto b = medOf("flat", err);   ASSERT_FALSE(b.empty()) << err;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    const double medS = a[a.size() / 2], medF = b[b.size() / 2];
    std::cout << "[speed] steger/eye=" << medS << "ms flat/eye=" << medF
              << "ms speedup=" << medS / medF << "x\n";
    // 【计数设备侧化 2026-10-19】steger 去重链同步 2→1 后墙钟反超 flat——原
    // 「flat 必快」排序断言基于旧双同步结构，正当失效。改为同量级护栏
    // （flat 不劣于 steger 1.5×——防 flat 退化；速度比为信息性输出）
    EXPECT_LT(medF, medS * 1.5);
}

// ------------------------------------------------------------------
// frame_fuse 兜底在链内真实触发（2026-10-15 步2/步5：验证域＝使用域——
// 子集契约【视野捞点/三角索引】＋三角形 matchMode=1 在链内消费）。
// 隔离手段：关 relock（否则宽域重锁会抢在兜底之前把帧接走）。
namespace golden {

struct GFrame {
    std::vector<cv::Point3d> pos;
    std::vector<cv::Vec3d> nrm;
};

bool loadGolden(const std::string& dir, std::vector<GFrame>& frames) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir)) return false;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.size() == 21 && name.substr(0, 6) == "cycle_"
            && name.substr(9) == "_markers.csv")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.size() < 2) return false;
    for (const auto& f : files) {
        std::ifstream ifs(f);
        if (!ifs) return false;
        std::string line;
        std::getline(ifs, line);
        GFrame fr;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string tok;
            std::vector<std::string> c;
            while (std::getline(ss, tok, ',')) c.push_back(tok);
            if (c.size() < 7) continue;
            fr.pos.emplace_back(std::stod(c[1]), std::stod(c[2]), std::stod(c[3]));
            fr.nrm.emplace_back(std::stod(c[4]), std::stod(c[5]), std::stod(c[6]));
        }
        frames.push_back(std::move(fr));
    }
    return frames.size() >= 2;
}

} // namespace golden
// ------------------------------------------------------------------
TEST_F(ChainReplayTest, FusePathExercisedInChain) {
    auto& e = env();
    ScrConfig cfg = e.cfg;                        // 全同复制，仅改三处
    cfg.operators["frame_fuse"]["matchMode"] = 1; // ② 三角形模式（子集契约消费）
    std::string err;
    Harness h(cfg, *e.calib, *e.src, err);
    ASSERT_TRUE(err.empty()) << err;
    // —— 真值判卷（批1 F11b：旧断言 rmse<门 是套套逻辑——fuse 的前提就是已过
    //    同一条门。真裁判＝金样 CSV 相邻帧互相最近邻（60mm 门，sparse_30 位移
    //    30~60mm）＋ Kabsch 标准变换；fuse 结果映射 cur→全局系（锚＝cycle0），
    //    与 oracle(cur_c → cycle_0) 同向可比）——
    std::vector<golden::GFrame> frames;
    const std::string gdir = std::string(SCR_SOURCE_DIR)
        + "/tests/frame_fuse_opt/in/markers_normals_14";
    const bool hasGolden = golden::loadGolden(gdir, frames);
    // 相邻帧裁判＋链式复合：直接 c↔0 配对在跨两帧以上会因位移>点间距配错对
    // （g60 实验证实的歧义）；相邻段（位移≤60mm）可靠，复合出 cur_c→cycle0 真值
    auto kabschAdj = [](const golden::GFrame& curF, const golden::GFrame& prevF,
                        cv::Matx33d& R, cv::Vec3d& T) -> int {
        std::vector<std::pair<int, int>> pairs;
        std::vector<int> nn(curF.pos.size(), -1);
        std::vector<double> nd(curF.pos.size(), 1e30);
        for (size_t i = 0; i < curF.pos.size(); ++i) {
            for (size_t j = 0; j < prevF.pos.size(); ++j) {
                const double dx = curF.pos[i].x - prevF.pos[j].x;
                const double dy = curF.pos[i].y - prevF.pos[j].y;
                const double dz = curF.pos[i].z - prevF.pos[j].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < nd[i]) { nd[i] = d; nn[i] = static_cast<int>(j); }
            }
        }
        for (size_t j = 0; j < prevF.pos.size(); ++j) {
            int bi = -1;
            double bd = 1e30;
            for (size_t i = 0; i < curF.pos.size(); ++i) {
                const double dx = curF.pos[i].x - prevF.pos[j].x;
                const double dy = curF.pos[i].y - prevF.pos[j].y;
                const double dz = curF.pos[i].z - prevF.pos[j].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < bd) { bd = d; bi = static_cast<int>(i); }
            }
            if (bi >= 0 && nn[static_cast<size_t>(bi)] == static_cast<int>(j)
                && nd[static_cast<size_t>(bi)] < 60.0)
                pairs.emplace_back(bi, static_cast<int>(j));
        }
        if (pairs.size() < 3) return -1;   // 裁判不可用（3 对 Kabsch 段残差<0.2mm 已验）
        cv::Vec3d c1(0, 0, 0), c2(0, 0, 0);
        for (const auto& pr : pairs) {
            c1 += cv::Vec3d(curF.pos[static_cast<size_t>(pr.first)].x,
                            curF.pos[static_cast<size_t>(pr.first)].y,
                            curF.pos[static_cast<size_t>(pr.first)].z);
            c2 += cv::Vec3d(prevF.pos[static_cast<size_t>(pr.second)].x,
                            prevF.pos[static_cast<size_t>(pr.second)].y,
                            prevF.pos[static_cast<size_t>(pr.second)].z);
        }
        c1 /= static_cast<double>(pairs.size());
        c2 /= static_cast<double>(pairs.size());
        cv::Mat H = cv::Mat::zeros(3, 3, CV_64FC1);
        for (const auto& pr : pairs) {
            const cv::Vec3d a(curF.pos[static_cast<size_t>(pr.first)].x - c1(0),
                             curF.pos[static_cast<size_t>(pr.first)].y - c1(1),
                             curF.pos[static_cast<size_t>(pr.first)].z - c1(2));
            const cv::Vec3d b(prevF.pos[static_cast<size_t>(pr.second)].x - c2(0),
                             prevF.pos[static_cast<size_t>(pr.second)].y - c2(1),
                             prevF.pos[static_cast<size_t>(pr.second)].z - c2(2));
            for (int u = 0; u < 3; ++u)
                for (int v = 0; v < 3; ++v)
                    H.at<double>(u, v) += a(u) * b(v);
        }
        cv::Mat U, S, Vt;
        cv::SVD::compute(H, S, U, Vt);
        cv::Mat Rm = Vt.t() * U.t();
        if (cv::determinant(Rm) < 0) {
            cv::Mat Vf = Vt.clone();
            Vf.at<double>(2, 0) *= -1; Vf.at<double>(2, 1) *= -1; Vf.at<double>(2, 2) *= -1;
            Rm = Vf.t() * U.t();
        }
        R = cv::Matx33d(Rm.at<double>(0, 0), Rm.at<double>(0, 1), Rm.at<double>(0, 2),
                        Rm.at<double>(1, 0), Rm.at<double>(1, 1), Rm.at<double>(1, 2),
                        Rm.at<double>(2, 0), Rm.at<double>(2, 1), Rm.at<double>(2, 2));
        T = c2 - R * c1;
        return static_cast<int>(pairs.size());
    };
    // 预复合：M_c = cur_c → cycle_0（相邻段 K 依次左乘）
    std::vector<cv::Matx33d> chainR(1, cv::Matx33d::eye());
    std::vector<cv::Vec3d> chainT(1, cv::Vec3d(0, 0, 0));
    if (hasGolden) {
        for (size_t c = 1; c < frames.size(); ++c) {
            cv::Matx33d K;
            cv::Vec3d Kt;
            if (kabschAdj(frames[c], frames[c - 1], K, Kt) < 4) break;   // 链断（后续不可判）
            const cv::Matx33d& P = chainR.back();
            const cv::Vec3d& Pt = chainT.back();
            chainR.push_back(P * K);
            chainT.push_back(P * Kt + Pt);
        }
    }
    const int n = std::min(6, static_cast<int>(e.src->cycles.size()));
    int fuseCount = 0, holdCount = 0, slowPathCount = 0, judged = 0;
    double maxRmse = 0.0;
    for (int c = 0; c < n; ++c) {
        CycleProducts p;
        ASSERT_TRUE(h.runCycle(c, p, false, err)) << err;
        if (p.regSource == "fuse") {
            ++fuseCount;
            maxRmse = std::max(maxRmse, p.regFuse.rmse);
            if (p.fuseSlowPath) ++slowPathCount;
            if (hasGolden && c < static_cast<int>(chainR.size())) {
                ++judged;   // 真值判卷：fuse R/T 必须与链式复合裁判一致（真解才算数）
                const cv::Matx33d dR = cv::Matx33d(p.regFuse.R) * chainR[static_cast<size_t>(c)].t();
                const double tr = dR(0, 0) + dR(1, 1) + dR(2, 2);
                const double dAng = std::acos(std::min(1.0, std::max(-1.0,
                    (tr - 1.0) * 0.5))) * 180.0 / CV_PI;
                const double dTmm = cv::norm(cv::Vec3d(p.regFuse.T)
                                             - chainT[static_cast<size_t>(c)]);
                EXPECT_LT(dAng, 2.0) << "cycle " << c << " fuse 旋转 vs 裁判: " << dAng;
                EXPECT_LT(dTmm, 3.0) << "cycle " << c << " fuse 平移 vs 裁判: " << dTmm;
            }
        }
        if (p.regSource == "hold") ++holdCount;
    }
    // sparse_30 早段（1←0 真值 4 点等）应触发兜底且过链内门（matched≥4/rmse<2mm）
    EXPECT_GT(fuseCount, 0) << "frame_fuse 兜底未被触发——链内消费域回归失效";
    EXPECT_GE(judged, 1) << "无任何 fuse 帧被真值判卷（金样缺席或裁判不可用）";
    if (fuseCount > 0) EXPECT_LT(maxRmse, cfg.runtime.fuseMaxRmseMm);
    std::cout << "[fuse-path] cycles=" << n << " fuse=" << fuseCount
              << " hold=" << holdCount << " slowPath=" << slowPathCount
              << " judged=" << judged << " maxRmse=" << maxRmse << "mm" << std::endl;
}

// ------------------------------------------------------------------
// 金样判卷常驻回归（2026-10-15 步6）：ALL 全速率前 10 帧标记点（tests/
// frame_fuse_opt/in/markers_normals_10all，几 KB 入库）＋互相最近邻裁判。
// 断言：①三角形模式 9/9 全部零错配（红线）；②变异校验——同数据 PFH 模式
// 成功对数 ≤3（证明考题能杀死坏实现；若 PFH 将来变好，此断言提醒重估）。
// ------------------------------------------------------------------

TEST(GoldenTriangleJudge, TenAllZeroWrongMatch) {
    std::vector<golden::GFrame> frames;
#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR ""
#endif
    const std::string dir = std::string(SCR_SOURCE_DIR)
        + "/tests/frame_fuse_opt/in/markers_normals_10all";
    if (!golden::loadGolden(dir, frames))
        GTEST_SKIP() << "金样数据缺席: " << dir;
    ASSERT_GE(frames.size(), 10u);

    // 裁判：互相最近邻（<10mm）＝真值对应
    auto truthPairs = [](const golden::GFrame& cur, const golden::GFrame& prev) {
        std::vector<std::pair<int, int>> out;
        std::vector<int> nn(cur.pos.size(), -1);
        std::vector<double> nd(cur.pos.size(), 1e30);
        for (size_t i = 0; i < cur.pos.size(); ++i) {
            for (size_t j = 0; j < prev.pos.size(); ++j) {
                const double dx = cur.pos[i].x - prev.pos[j].x;
                const double dy = cur.pos[i].y - prev.pos[j].y;
                const double dz = cur.pos[i].z - prev.pos[j].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < nd[i]) { nd[i] = d; nn[i] = static_cast<int>(j); }
            }
        }
        for (size_t j = 0; j < prev.pos.size(); ++j) {
            int bi = -1;
            double bd = 1e30;
            for (size_t i = 0; i < cur.pos.size(); ++i) {
                const double dx = cur.pos[i].x - prev.pos[j].x;
                const double dy = cur.pos[i].y - prev.pos[j].y;
                const double dz = cur.pos[i].z - prev.pos[j].z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < bd) { bd = d; bi = static_cast<int>(i); }
            }
            if (bi >= 0 && nn[static_cast<size_t>(bi)] == static_cast<int>(j) && nd[static_cast<size_t>(bi)] < 10.0)
                out.emplace_back(bi, static_cast<int>(j));
        }
        return out;
    };

    calib::FrameFuseCPUParams pTri;
    pTri.matchMode = 1;
    calib::FrameFuseCPU opTri(pTri);
    calib::FrameFuseCPU opPfh;   // 默认 matchMode=0（变异校验）

    int triOk = 0, pfhOk = 0, pairsTotal = 0;
    for (size_t i = 1; i < 10; ++i) {
        calib::MarkerPointSet s1, s2;
        s1.positions = frames[i].pos;
        s1.normals = frames[i].nrm;
        s2.positions = frames[i - 1].pos;
        s2.normals = frames[i - 1].nrm;
        const auto truth = truthPairs(frames[i], frames[i - 1]);
        ASSERT_GE(truth.size(), 3u) << "裁判失效 pair " << i;
        pairsTotal += static_cast<int>(truth.size());

        auto rT = opTri.Execute(s1, s2);
        ASSERT_TRUE(rT.success) << "pair " << i << ": " << rT.message;
        int wrong = 0;
        for (const auto& pr : rT.correspondences) {
            bool hit = false;
            for (const auto& t : truth)
                if (t.first == pr.first && t.second == pr.second) { hit = true; break; }
            if (!hit) ++wrong;
        }
        EXPECT_EQ(wrong, 0) << "pair " << i << " 错配 " << wrong << " 对（红线）";
        EXPECT_GE(rT.matchedCount, 4u) << "pair " << i;
        EXPECT_LT(rT.rmse, 2.0) << "pair " << i;
        if (wrong == 0 && rT.matchedCount >= 4) ++triOk;

        auto rP = opPfh.Execute(s1, s2);
        if (rP.success && rP.matchedCount >= 4 && rP.rmse < 2.0) ++pfhOk;
    }
    EXPECT_EQ(triOk, 9) << "三角形模式金样 9 对须全绿";
    EXPECT_LE(pfhOk, 3) << "变异校验：PFH 在病灶形态应大多失败——若 PFH 已修好请重估本断言";
    std::cout << "[golden] tri=" << triOk << "/9 pfh=" << pfhOk
              << "/9 truthPairs=" << pairsTotal << std::endl;
}

// ============================================================================
// 规则网格板检测（2026-10-15 裁决：网格平移对称＝原理性不可分辨，宁拒勿错）
// ============================================================================
TEST(GridBoardDetector, LatticeRejected) {
    // 3×3 规则网格（100mm 格距）——最近邻间距全等于格距 → CV≈0 → 必须检出
    std::vector<cv::Point3d> grid;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            grid.emplace_back(c * 100.0, r * 100.0, 300.0);
    EXPECT_TRUE(detectRegularGrid(grid));

    // 带测量噪声（±0.5mm）的网格——CV 仍远小于阈值
    std::vector<cv::Point3d> noisy;
    const double jig[] = {0.3, -0.4, 0.2, 0.5, -0.2, -0.5, 0.4, 0.1, -0.3};
    for (size_t k = 0; k < grid.size(); ++k)
        noisy.emplace_back(grid[k].x + jig[k], grid[k].y - jig[k], grid[k].z + 0.1 * jig[k]);
    EXPECT_TRUE(detectRegularGrid(noisy));
}

TEST(GridBoardDetector, RandomBoardAccepted) {
    // 随机布点（取自真实数据 cycle_000 的标记点分布，间距 100~200mm 连续）——不误报
    std::vector<cv::Point3d> rand9 = {
        {206.4, -33.9, 322.0}, {190.1, -102.2, 321.3}, {155.9, -105.1, 321.2},
        {120.1, 357.2, 321.2}, {906.2, 546.6, 322.4}, {1358.6, 665.5, 323.1},
        {955.6, 870.7, 323.1}, {585.4, 872.5, 323.2}, {1257.9, 1069.0, 324.0}
    };
    EXPECT_FALSE(detectRegularGrid(rand9));

    // 点数不足 8——样本不足不判（返回 false 不误杀）
    std::vector<cv::Point3d> few = {{0,0,0}, {100,0,0}, {200,0,0}, {300,0,0}, {400,0,0}};
    EXPECT_FALSE(detectRegularGrid(few));
}

TEST(GridBoardDetector, RealWorldFalsePositiveFrameRegression) {
    // 回归钉死（2026-10-15 全量实测翻车帧）：ALL 数据 cycle_126 的 6 个真实标记点——
    // 可见子集恰巧间距较齐（NN 间距 CV=0.0179），曾误触旧阈值 0.04 导致其后 273 帧
    // 全部被锁存拒绝。校准后（阈值 0.012＋最少 8 点）该帧必须放行。
    // 全量口径：真实随机板 399 帧 CV 分布 min=0.0179 / P50=0.138；真网格（±0.5mm
    // 噪声/100mm 格距）CV≈0.005——阈值 0.012 两侧各留 ≥1.4 倍余量。
    std::vector<cv::Point3d> frame126 = {
        {102.88, -106.71, 318.79}, {205.16, -93.63, 316.16},
        {119.31, -38.62, 320.07}, {209.12, -25.79, 317.51},
        {-53.59, 45.53, 327.32}, {-57.90, 112.53, 329.13}
    };
    EXPECT_FALSE(detectRegularGrid(frame126));
}

} // namespace scr
