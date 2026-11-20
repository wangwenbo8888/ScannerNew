// ============================================================================
// test_global_optim_stage.cpp — 全局优化收尾阶段单测【2026-11-16】
// ============================================================================
// 用例口径（方案分析-全局优化接入-2026-11-16.md §三·终）：
//  ①配置 fail-safe（缺节/关/参数透传）   ②装配过滤（gid<0／观测不足剔帧）
//  ③合成收敛（GBA 端到端数值正确性）      ④规约对齐（整体刚体差消除）
//  ⑤分布统计精确性                       ⑥重融合位姿缺帧策略（GPU）
//  ⑦跨遍清零（--repeat 防增长）          ⑧轨迹 PLY 落盘
// 不依赖真实数据集（合成数据；缺席即跑，无 SKIP）。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "gba_records.h"
#include "global_optim_stage.h"
#include "laser_cloud_fuse_cuda.h"
#include "scr_config.h"

namespace fs = std::filesystem;

namespace {

/// 合成场景：nFrames 帧沿 X 平移扫过 m 个固定标记点；观测＝本机系坐标＋噪声
struct SyntheticScene {
    std::vector<scr::GbaObsRecord> obs;
    std::vector<scr::GbaPose> truth;          // 每帧真值位姿
    std::vector<cv::Point3d> markers;         // 全局系标记点
};

SyntheticScene makeScene(int nFrames, int nMarkers, double noiseMm,
                         double driftMm, unsigned seed = 7) {
    SyntheticScene s;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noiseMm);
    // 标记点：z=200mm 平面网格（模拟标定板距离）
    for (int i = 0; i < nMarkers; ++i)
        s.markers.emplace_back((i % 4) * 40.0 - 60.0, (i / 4) * 40.0 - 20.0, 200.0);
    for (int f = 0; f < nFrames; ++f) {
        // 真值位姿：沿 X 平移的扫描运动
        scr::GbaPose pose;
        pose.T = cv::Vec3d(f * 8.0, 0, 0);
        s.truth.push_back(pose);
        // 初值＝真值＋累积漂移（模拟链式配准漂移，单调 x 向）
        scr::GbaObsRecord rec;
        rec.frameId = static_cast<uint64_t>(f);
        const double drift = driftMm * f;
        rec.T[0] = pose.T(0) + drift;
        rec.T[1] = pose.T(1);
        rec.T[2] = pose.T(2);
        // rec.R＝单位（真值也单位——漂移只注入平移，隔离变量）
        for (const auto& X : s.markers) {
            const cv::Vec3d global(X.x, X.y, X.z);
            const cv::Vec3d local = pose.R.t() * (global - pose.T);   // 真值观测
            rec.local.emplace_back(local(0) + noise(rng), local(1) + noise(rng),
                                  local(2) + noise(rng));
            rec.gids.push_back(static_cast<int>(&X - &s.markers[0]));
        }
        s.obs.push_back(std::move(rec));
    }
    return s;
}

std::string writeTempJson(const std::string& body) {
    static int seq = 0;
    const auto p = fs::temp_directory_path() /
                   ("scr_gba_test_" + std::to_string(++seq) + ".json");
    std::ofstream(p) << body;
    return p.string();
}

}  // namespace

// ---------------------------------------------------------------- ① 配置
TEST(GbaStage, ConfigFailsafe) {
    std::string err;
    scr::ScrConfig cfg;
    // 缺节 → 关
    ASSERT_TRUE(scr::ScrConfig::fromFile(
        writeTempJson(R"({"paths":{"dataDir":"x"},"runtime":{},"operators":{}})"),
        cfg, err)) << err;
    EXPECT_FALSE(cfg.globalOptim.enabled);
    // 显式 false → 关（paths 节为 fromFile 必填，给空对象走默认值）
    ASSERT_TRUE(scr::ScrConfig::fromFile(
        writeTempJson(R"({"paths":{},"globalOptim":{"enabled":false}})"), cfg, err))
        << err;
    EXPECT_FALSE(cfg.globalOptim.enabled);
    // true ＋参数透传
    ASSERT_TRUE(scr::ScrConfig::fromFile(
        writeTempJson(R"({"paths":{},"globalOptim":{"enabled":true,"refuseLaser":false,
                       "keepUnoptimizedFrames":false,
                       "params":{"maxIterations":77,"sigmaObserved":0.02}}})"),
        cfg, err)) << err;
    EXPECT_TRUE(cfg.globalOptim.enabled);
    EXPECT_FALSE(cfg.globalOptim.refuseLaser);
    EXPECT_FALSE(cfg.globalOptim.keepUnoptimizedFrames);
    EXPECT_EQ(77, cfg.globalOptim.params.value("maxIterations", 0));
    EXPECT_DOUBLE_EQ(0.02, cfg.globalOptim.params.value("sigmaObserved", 0.0));
}

// ---------------------------------------------------------------- ② 装配过滤
TEST(GbaStage, AssembleFilters) {
    std::vector<scr::GbaObsRecord> obs(3);
    // 帧 0：4 点其中 1 点 gid=-1（候选）→ 有效 3 点过门
    obs[0].frameId = 0;
    obs[0].local = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {9, 9, 9}};
    obs[0].gids = {5, -1, 6, 7};
    // 帧 1：3 点但 2 点 gid=-1 → 有效 1 点 <3 → 整帧剔除
    obs[1].frameId = 1;
    obs[1].local = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
    obs[1].gids = {-1, -1, 8};
    // 帧 2：4 点全有效
    obs[2].frameId = 2;
    obs[2].local = {{4, 4, 4}, {5, 5, 5}, {6, 6, 6}, {7, 7, 7}};
    obs[2].gids = {9, 10, 11, 12};

    calib::GlobalBAParams params;   // minPointsPerFrame=3 默认
    std::vector<uint64_t> excluded;
    const auto input = scr::assembleGbaInput(obs, params, excluded);
    ASSERT_EQ(2u, input.frames.size());
    EXPECT_EQ(1u, excluded.size());
    EXPECT_EQ(1u, excluded[0]);
    EXPECT_EQ(0u, input.frames[0].frameId);
    EXPECT_EQ(2u, input.frames[1].frameId);
    EXPECT_EQ(3u, input.frames[0].markerObs.size());   // gid=-1 已滤
    EXPECT_EQ(4u, input.frames[1].markerObs.size());
}

// ---------------------------------------------------------------- ③ 合成收敛
TEST(GbaStage, SyntheticConverges) {
    const double noise = 0.02;    // mm（σ 与算子默认 sigmaObserved=0.01 同量级）
    const double drift = 0.6;     // mm/帧 累积漂移 → 末帧 ~18mm
    auto scene = makeScene(30, 12, noise, drift);
    ASSERT_EQ(30u, scene.obs.size());

    calib::GlobalBAParams params;
    params.maxSolveSecondsPerPhase = 30.0;   // 单测限时收紧
    scr::GbaStageOptions opts;               // outDir 空＝不落盘
    const auto res = scr::runGbaCore(scene.obs, nullptr, params, opts);

    EXPECT_TRUE(res.executed);
    ASSERT_TRUE(res.success) << res.message;
    EXPECT_GT(res.initialRMSE, 0.5);          // 初值漂移显著（>0.5mm）
    EXPECT_LT(res.finalRMSE, res.initialRMSE);  // 收敛（数值正确性的主断言）
    EXPECT_EQ(30, res.framesSent);
    EXPECT_EQ(30, res.framesKept);
    // 优化后位姿应贴近真值（漂移被吸收；留 5×噪声 + 1mm 容差）
    for (size_t f = 0; f < scene.truth.size(); ++f) {
        auto it = res.poses.find(static_cast<uint64_t>(f));
        ASSERT_NE(res.poses.end(), it);
        const double dtErr = cv::norm(it->second.T - scene.truth[f].T);
        EXPECT_LT(dtErr, 1.0 + 5 * noise) << "frame " << f;
    }
}

// ---------------------------------------------------------------- ④ 规约对齐
TEST(GbaStage, GaugeAlign) {
    // after = Rg0·before + tg0（整体刚体差）；对齐后 Δ 应≈0
    const double a = 0.4;   // rad
    cv::Matx33d Rg0(std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1);
    cv::Vec3d tg0(3.0, -2.0, 1.5);
    std::vector<std::pair<scr::GbaPose, scr::GbaPose>> pairs;
    for (int i = 0; i < 8; ++i) {
        scr::GbaPose b;
        // 非共线轨迹（共线＝旋转不可观测的退化场景，Umeyama 无唯一解）
        b.T = cv::Vec3d(i * 2.0, 3.0 * std::sin(0.7 * i), 95.0 + 0.9 * i);
        scr::GbaPose aft;
        aft.R = Rg0 * b.R;
        aft.T = Rg0 * b.T + tg0;
        pairs.emplace_back(b, aft);
    }
    cv::Matx33d Rg;
    cv::Vec3d tg;
    scr::gaugeAlign(pairs, Rg, tg);
    const auto st = scr::poseDeltaStats(pairs);          // raw：应含刚体差
    EXPECT_GT(st.maxDt, 2.0);
    // aligned：手工用 (Rg,tg) 重算（statsFromPairs 内部同式——这里经由公开 API
    // poseDeltaStats 只出 raw；aligned 验证改为：Rg·Rg0≈I 且 Rg·tg0+tg≈0）
    cv::Matx33d I = Rg * Rg0;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(r == c ? 1.0 : 0.0, I(r, c), 1e-9);
    const cv::Vec3d residual = Rg * tg0 + tg;
    EXPECT_LT(cv::norm(residual), 1e-9);
}

// ---------------------------------------------------------------- ⑤ 分布统计
TEST(GbaStage, PoseDeltaStatsExact) {
    std::vector<std::pair<scr::GbaPose, scr::GbaPose>> pairs;
    const double dts[] = {0.5, 1.0, 2.0, 4.0, 10.0};   // 5 对：中位=2.0 max=10.0
    for (double d : dts) {
        scr::GbaPose b, a;
        b.T = cv::Vec3d(0, 0, 0);
        a.T = cv::Vec3d(d, 0, 0);
        pairs.emplace_back(b, a);
    }
    const auto st = scr::poseDeltaStats(pairs);
    ASSERT_TRUE(st.valid);
    EXPECT_DOUBLE_EQ(2.0, st.medDt);
    EXPECT_DOUBLE_EQ(10.0, st.maxDt);
    EXPECT_DOUBLE_EQ(0.0, st.medDr);   // 旋转无差
    EXPECT_DOUBLE_EQ(0.0, st.maxDr);
}

// ---------------------------------------------------------------- ⑥ 重融合策略（GPU）
TEST(GbaStage, RefusePolicy) {
    calib::LaserCloudFuseCuda op;   // 默认参数（0.5mm 体素语义同产线）
    cv::cuda::Stream stream;
    // 两帧激光档：帧 0 无优化位姿（GBA 被剔），帧 1 有
    std::vector<scr::GbaLaserRecord> arch(2);
    arch[0].frameId = 0;
    arch[0].xyz = {0.f, 0.f, 300.f, 1.f, 0.f, 300.f, 0.f, 1.f, 300.f, 1.f, 1.f, 300.f};
    arch[1].frameId = 1;
    arch[1].xyz = {10.f, 0.f, 300.f, 11.f, 0.f, 300.f, 10.f, 1.f, 300.f, 11.f, 1.f, 300.f};
    std::map<uint64_t, scr::GbaPose> poses;
    poses[1] = scr::GbaPose{};      // 单位位姿

    // keepUnoptimized=true：帧 0 原位姿兜底 → 全融合
    auto st1 = scr::refuseWithOp(op, arch, poses, true, stream);
    EXPECT_TRUE(st1.executed);
    EXPECT_EQ(2, st1.framesFused);
    EXPECT_EQ(1, st1.framesNoPose);
    // keepUnoptimized=false：帧 0 跳过
    auto st2 = scr::refuseWithOp(op, arch, poses, false, stream);
    EXPECT_TRUE(st2.executed);
    EXPECT_EQ(1, st2.framesFused);
    EXPECT_EQ(1, st2.framesNoPose);
    // 空档拒执行（对账失败保原云）
    auto st3 = scr::refuseWithOp(op, {}, poses, true, stream);
    EXPECT_FALSE(st3.executed);
}

// ---------------------------------------------------------------- ⑦ 跨遍清零
TEST(GbaStage, RepeatPassReset) {
    std::vector<scr::GbaObsRecord> obs(3);
    std::vector<scr::GbaLaserRecord> laser(2);
    laser[0].xyz.resize(3 * 1000);   // 模拟一帧点云
    scr::gbaClearArchives(obs, laser);
    EXPECT_TRUE(obs.empty());
    EXPECT_TRUE(laser.empty());
    EXPECT_EQ(0u, obs.capacity());   // 容量一并释放（130MB 级不留）
    EXPECT_EQ(0u, laser.capacity());
}

// ---------------------------------------------------------------- ⑧ 轨迹 PLY
TEST(GbaStage, TrajectoryPlyWritten) {
    std::vector<scr::GbaPose> before(2), after(2);
    before[0].T = cv::Vec3d(0, 0, 0);
    before[1].T = cv::Vec3d(10, 0, 0);
    after[0].T = cv::Vec3d(0, 0, 0);
    after[1].T = cv::Vec3d(10.2, 0, 0);
    const auto p = fs::temp_directory_path() / "scr_gba_traj_test.ply";
    ASSERT_TRUE(scr::writeGbaTrajectoryPly(p.string(), before, after));
    {
        std::ifstream f(p);
        ASSERT_TRUE(f.good());
        std::string line;
        bool sawVertex = false, sawEdge = false;
        while (std::getline(f, line)) {
            if (line.find("element vertex 4") != std::string::npos) sawVertex = true;
            if (line.find("element edge 2") != std::string::npos) sawEdge = true;
        }
        EXPECT_TRUE(sawVertex);
        EXPECT_TRUE(sawEdge);
    }   // ifstream 出作用域关闭后再删（Windows 句柄语义）
    std::error_code ec;
    fs::remove(p, ec);
}
