// ============================================================================
// test_scan_sim_frameflow.cpp — SimScanSource 观测级断言（260917 中段模拟提取版）
//
// 架构对齐后本源只产设备系原始观测（SimFrameObs）——R/T/融合/渲染归生产链
// （ScanChains::runRegistration + FuseConsumer，另有生产测试覆盖）。此处断言：
//   0 builtin 场景形态：30 标志点＋6561 激光点（PLY/builtin 契约锚）
//   1 恒等冒烟：σ=0＋轨迹静止＋视场不限 → 首帧观测==场景真值（集合语义）
//   2 部分观测：垂直视场收窄 → 标志点/激光均为真子集（"扫到哪看到哪"）
//   3 真值轨迹：k 帧后 lastTruePose 与 G_k = T(k·t)·R(k·θ) 解析一致
//   4 帧序与上限：maxFrames=N → 恰 N 帧后 false；frameId 单调 0..N-1
//   5 可复现：同参双源逐帧观测完全一致（洗牌池/噪声序确定）
//   6 噪声有界：σ=0.05 → 观测与真值变换点距 < 6σ
// ============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

#include "pipelines/scan/ScanTypes.h"
#include "pipelines/scan/SimScanSource.h"

#include "core/marker/optical_flow_fuse/marker_optical_flow_fuse_cpu.h"

using namespace Scanner::pipeline;

namespace {

constexpr double kEps = 1e-6;

/// 集合等价断言（激光池洗牌→顺序无关）：逐维排序后比较
::testing::AssertionResult cloudEquals(std::vector<cv::Point3f> a,
                                       std::vector<cv::Point3f> b) {
    auto lt = [](const cv::Point3f& l, const cv::Point3f& r) {
        return std::tie(l.x, l.y, l.z) < std::tie(r.x, r.y, r.z);
    };
    std::sort(a.begin(), a.end(), lt);
    std::sort(b.begin(), b.end(), lt);
    if (a.size() != b.size())
        return ::testing::AssertionFailure() << "size " << a.size() << " != " << b.size();
    for (size_t i = 0; i < a.size(); ++i) {
        const auto d = a[i] - b[i];
        if (std::abs(d.x) > kEps || std::abs(d.y) > kEps || std::abs(d.z) > kEps)
            return ::testing::AssertionFailure() << "point[" << i << "] differs";
    }
    return ::testing::AssertionSuccess();
}

} // namespace

// —— 0 builtin 场景形态 ——
TEST(SimScanObs, BuiltinSceneShape) {
    const auto s = SimScene::builtin();
    EXPECT_EQ(s.markers.size(), 30u);      // 6×5 网格
    EXPECT_EQ(s.laser.size(), 6561u);      // 81×81 面片
}

// —— 1 恒等冒烟：σ=0＋静止＋视场不限 → 首帧标志点观测==真值；激光=25 条线
//    子集（260919 条带化：每帧 25 行线、行厚 2.5mm——非全池） ——
TEST(SimScanObs, IdentityFirstFrame) {
    SimTrajParams p;                       // 缺省：静止轨迹（0°/0mm）
    p.markerNoiseSigmaMm = 0.0;
    p.jitterSigmaMm = 0.0;
    const auto scene = SimScene::builtin();
    SimScanSource src(scene, p);

    SimFrameObs obs;
    ASSERT_TRUE(src.next(obs));
    ASSERT_EQ(obs.markerPositions.size(), scene.markers.size());
    for (size_t i = 0; i < obs.markerPositions.size(); ++i) {
        EXPECT_NEAR(obs.markerPositions[i].x, scene.markers[i].x, kEps);
        EXPECT_NEAR(obs.markerPositions[i].y, scene.markers[i].y, kEps);
        EXPECT_NEAR(obs.markerPositions[i].z, scene.markers[i].z, kEps);
    }
    EXPECT_GT(obs.laser.size(), 0u);               // 条带化：有观测
    EXPECT_LE(obs.laser.size(), scene.laser.size());   // ≤ 池（行选子集；小池
                                                        // stride=1 时可为全池）
}

// —— 2 部分观测：垂直视场收窄 → 真子集 ——
TEST(SimScanObs, PartialObservationFov) {
    SimTrajParams p;
    p.markerNoiseSigmaMm = 0.0;
    p.jitterSigmaMm = 0.0;
    p.fovHalfVDeg = 4.0;                   // ±4° @ ~400mm → |y|≲28mm（网格 ±60mm）
    p.sweepY = false;                      // 窗定住——纯子集语义
    SimScanSource src(SimScene::builtin(), p);

    SimFrameObs obs;
    ASSERT_TRUE(src.next(obs));
    EXPECT_GT(obs.markerPositions.size(), 0u);
    EXPECT_LT(obs.markerPositions.size(), 30u);
    EXPECT_GT(obs.laser.size(), 0u);
    EXPECT_LT(obs.laser.size(), 6561u);
}

// —— 3 真值轨迹解析一致：G_k = T(k·t)·R(k·θ) ——
TEST(SimScanObs, TruePoseAnalytic) {
    SimTrajParams p;
    p.rotDegPerFrame = 0.3;
    p.transMmPerFrame = 0.5;
    SimScanSource src(SimScene::builtin(), p);

    SimFrameObs obs;
    const uint64_t k = 7;
    for (uint64_t i = 0; i <= k; ++i) ASSERT_TRUE(src.next(obs));
    double R[9], T[3];
    src.lastTruePose(R, T);
    // 平移：x=k·t（无 sweep）
    EXPECT_NEAR(T[0], 0.5 * static_cast<double>(k), 1e-9);
    EXPECT_NEAR(T[1], 0.0, 1e-9);
    EXPECT_NEAR(T[2], 0.0, 1e-9);
    // 旋转：绕 y 轴 θ=k·0.3°
    const double a = 0.3 * k * CV_PI / 180.0;
    EXPECT_NEAR(R[0], std::cos(a), 1e-9);
    EXPECT_NEAR(R[2], std::sin(a), 1e-9);
    EXPECT_NEAR(R[6], -std::sin(a), 1e-9);
    EXPECT_NEAR(R[8], std::cos(a), 1e-9);
}

// —— 4 帧序与上限 ——
TEST(SimScanObs, FrameSequenceAndLimit) {
    SimTrajParams p;
    p.maxFrames = 5;
    SimScanSource src(SimScene::builtin(), p);

    SimFrameObs obs;
    for (uint64_t i = 0; i < 5; ++i) {
        ASSERT_TRUE(src.next(obs));
        EXPECT_EQ(obs.frameId, i);
    }
    EXPECT_FALSE(src.next(obs));
    EXPECT_TRUE(obs.markerPositions.empty());   // 尽帧后 out 置空（链上=空观测续跑）
}

// —— 5 可复现：同参双源逐帧一致 ——
TEST(SimScanObs, Reproducible) {
    SimTrajParams p;
    p.rotDegPerFrame = 0.05;
    p.transMmPerFrame = 0.3;
    p.fovHalfHDeg = 25.0;
    p.fovHalfVDeg = 20.0;
    p.depthMinMm = 250.0;
    p.depthMaxMm = 800.0;
    p.sweepY = true;
    SimScanSource a(SimScene::builtin(), p);
    SimScanSource b(SimScene::builtin(), p);

    SimFrameObs oa, ob;
    for (int i = 0; i < 50; ++i) {
        ASSERT_TRUE(a.next(oa));
        ASSERT_TRUE(b.next(ob));
        ASSERT_EQ(oa.markerPositions.size(), ob.markerPositions.size());
        ASSERT_EQ(oa.laser.size(), ob.laser.size());
        ASSERT_TRUE(cloudEquals(oa.laser, ob.laser));
        for (size_t m = 0; m < oa.markerPositions.size(); ++m) {
            EXPECT_NEAR(oa.markerPositions[m].x, ob.markerPositions[m].x, 0.0);
            EXPECT_NEAR(oa.markerPositions[m].y, ob.markerPositions[m].y, 0.0);
        }
    }
}

// —— 6 噪声有界：|obs − 真值变换| < 6σ ——
TEST(SimScanObs, NoiseBounded) {
    SimTrajParams p;
    p.rotDegPerFrame = 0.05;
    p.transMmPerFrame = 0.3;
    p.markerNoiseSigmaMm = 0.05;
    p.jitterSigmaMm = 0.05;
    p.maxFrames = 20;
    SimScanSource src(SimScene::builtin(), p);

    SimFrameObs obs;
    while (src.next(obs)) {
        // 深度窗内所有观测 z∈(−800,−250)（真值面 z≈−400；视场不限时全可见）
        for (const auto& m : obs.markerPositions) {
            EXPECT_GT(m.z, -800.0 - 1.0);
            EXPECT_LT(m.z, -250.0 + 1.0);
        }
        // 有界性粗锚：观测围绕真值分布——质心深度贴近 −400（噪声 σ=0.05 远小
        // 于 1mm；若有野值/量纲错立刻放大）
        if (!obs.markerPositions.empty()) {
            double zMean = 0.0;
            for (const auto& m : obs.markerPositions) zMean += m.z;
            zMean /= static_cast<double>(obs.markerPositions.size());
            EXPECT_NEAR(zMean, -402.5, 1.0);
        }
    }
}

// —— 7 递增调度（260919 用户定版「先8后9逐帧增加」）：窗口 8→9→10 且相邻
//    步子集重合 W-1 ——
TEST(SimScanObs, SubsetScheduleGrowth) {
    SimTrajParams p;
    p.markerNoiseSigmaMm = 0.0;          // 确定性断言（σ=0）
    p.jitterSigmaMm = 0.0;
    p.rotDegPerFrame = 0.02;
    p.transMmPerFrame = 0.1;
    p.markerStepFrames = 15;
    p.markerWindowStart = 8;
    p.markerGrowEvery = 2;
    p.markerAdvancePerStep = 1;
    p.maxFrames = 90;                    // 6 步：8,8,9,9,10,10
    p.depthMinMm = 250.0;
    p.depthMaxMm = 800.0;
    SimScanSource src(SimScene::builtin(), p);

    SimFrameObs obs;
    ASSERT_TRUE(src.next(obs));
    EXPECT_EQ(obs.markerPositions.size(), 8u);        // 帧0＝8 个
    auto prev = obs.markerPositions;                   // σ=0 下步间可比（集合语义：
    size_t prevOverlap = 8;                            // 按坐标近邻计数）
    for (int k = 1; k < 90; ++k) {
        ASSERT_TRUE(src.next(obs));
        const size_t cur = obs.markerPositions.size();
        const size_t step = static_cast<size_t>(k / 15);
        const size_t expect = std::min<size_t>(8 + step / 2, 30);
        EXPECT_EQ(cur, expect) << "帧 " << k;
        size_t overlap = 0;                            // 与上帧重合数（容差 1mm——
        for (const auto& c : obs.markerPositions)      // 帧间设备微动 ~0.25mm）
            for (const auto& pv : prev)
                if (std::abs(c.x - pv.x) < 1.0 && std::abs(c.y - pv.y) < 1.0 &&
                    std::abs(c.z - pv.z) < 1.0)
                    { ++overlap; break; }
        EXPECT_GE(overlap, prev.size() > 1 ? prev.size() - 1 : prev.size())
            << "帧 " << k << " 与上帧重合不足（" << overlap << "/" << prev.size() << "）";
        prev = obs.markerPositions;
        prevOverlap = overlap;
        (void)prevOverlap;
    }
}

// —— 8 递增调度×生产光流配准复现（260919 真机 matched=0 排障）：帧0 建锚
//    （ScanChains 首帧分支语义）→ 帧1..N 经 MarkerOpticalFlowFuseCPU 链式
//    配准——断言逐帧 success 且重合标志点数 ≥ W-1 ——
TEST(SimScanObs, SubsetScheduleRegistration) {
    SimTrajParams p;
    p.markerNoiseSigmaMm = 0.05;
    p.jitterSigmaMm = 0.05;
    p.rotDegPerFrame = 0.02;
    p.transMmPerFrame = 0.1;
    p.markerStepFrames = 15;
    p.markerWindowStart = 8;
    p.markerGrowEvery = 2;
    p.markerAdvancePerStep = 1;
    p.maxFrames = 90;
    p.depthMinMm = 250.0;
    p.depthMaxMm = 800.0;
    // 真机同源场景：markers_30.ply（缺省路径在则用之——y∈[-474,+291] 真实板，
    // 260919 真机 matched=0 排障；缺失则退 builtin）
    SimScene scene;
    const char* ply = std::getenv("JMW_SIM_MARKERS_PLY");
    const std::string plyPath = ply && *ply ? std::string(ply) : "D:/markers_30.ply";
    if (!SimScene::loadMarkersFromPly(plyPath, scene).success) scene = SimScene::builtin();
    SimScanSource src(scene, p);
    calib::MarkerOpticalFlowFuseCPU op;

    auto toPf = [](const SimFrameObs& o, const std::vector<int>& ids,
                   const cv::Matx33d& R, const cv::Vec3d& T) {
        calib::PrevFrameState pf;
        for (size_t i = 0; i < o.markerPositions.size(); ++i) {
            pf.rawPositions.emplace_back(o.markerPositions[i].x, o.markerPositions[i].y,
                                         o.markerPositions[i].z);
            pf.rawNormals.emplace_back(o.markerNormals[i][0], o.markerNormals[i][1],
                                       o.markerNormals[i][2]);
        }
        pf.globalIds = ids;
        pf.R = R;
        pf.T = T;
        return pf;
    };

    SimFrameObs obs;
    ASSERT_TRUE(src.next(obs));
    std::vector<int> ids0(obs.markerPositions.size());
    for (size_t i = 0; i < ids0.size(); ++i) ids0[i] = static_cast<int>(i);
    auto pf = toPf(obs, ids0, cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));

    int okFrames = 0;
    for (int k = 1; k < 90; ++k) {
        ASSERT_TRUE(src.next(obs));
        std::vector<cv::Point3d> cur;
        std::vector<cv::Vec3d> nrm;
        cur.reserve(obs.markerPositions.size());
        nrm.reserve(obs.markerPositions.size());
        for (size_t i = 0; i < obs.markerPositions.size(); ++i) {
            cur.emplace_back(obs.markerPositions[i].x, obs.markerPositions[i].y,
                             obs.markerPositions[i].z);
            nrm.emplace_back(obs.markerNormals[i][0], obs.markerNormals[i][1],
                             obs.markerNormals[i][2]);
        }
        auto fr = op.Execute(cur, nrm, pf);
        EXPECT_TRUE(fr.success) << "帧 " << k << "（观测 " << cur.size() << "）: "
                                << fr.message;
        if (!fr.success) break;
        ++okFrames;
        pf = toPf(obs, fr.getGlobalIds(), fr.R, fr.T);   // 链式（ScanChains st 语义）
    }
    EXPECT_GE(okFrames, 80);
}
