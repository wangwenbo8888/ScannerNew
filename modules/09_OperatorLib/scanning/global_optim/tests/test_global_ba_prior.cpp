// P4-T24: GBA 高精度已有點软先验测试（客户端扫描流水线.md §5.3 解法）
// 背景: GBA 现把位姿+标记点位置都设自由变量, 已有点会被扫描观测挪动;
//       软先验对命中 highPrecisionGlobalIds 的点 X 加残差 ‖X−X_existing‖²/σ²（σ 极小=高权重）。
#include "global_ba_cpu.h"
#include "ba_residuals.h"
#include <gtest/gtest.h>
#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <ceres/autodiff_cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/numeric_diff_options.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ringbuffer_sink.h>

using namespace calib;

// ===== 合成场景辅助（复刻 test_global_ba_cpu.cpp, 保持本测试 exe 自包含）=====
struct SyntheticGT {
    std::vector<cv::Point3d> points;                       // GT 全局点(frame-0 系)
    std::vector<std::pair<cv::Matx33d, cv::Vec3d>> poses;  // GT 每帧位姿 (R_i, t_i)
};

static GlobalBAInput makeSyntheticInput(int nPoints, int nFrames, int idsPerFrame,
                                        SyntheticGT* gtOut = nullptr,
                                        uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> posDist(-100.0, 100.0);  // 200mm 立方
    std::vector<cv::Point3d> Xgt(nPoints);
    for (auto& p : Xgt) p = cv::Point3d(posDist(rng), posDist(rng), posDist(rng));

    GlobalBAInput in;
    std::vector<std::pair<cv::Matx33d, cv::Vec3d>> gtPoses;
    std::normal_distribution<double> angNoise(0.0, 0.5 * CV_PI / 180.0);  // 0.5 deg
    std::normal_distribution<double> tNoise(0.0, 0.1);                     // 0.1 mm

    for (int i = 0; i < nFrames; ++i) {
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t(0, 0, 0);
        if (i > 0) {
            double ang = (i * 0.3) * CV_PI / 180.0;
            cv::Matx33d Rz(std::cos(ang), -std::sin(ang), 0,
                           std::sin(ang),  std::cos(ang), 0,
                           0, 0, 1);
            R = Rz;
            t = cv::Vec3d(i * 0.5, 0.0, 0.0);
        }
        gtPoses.push_back({R, t});

        GlobalBAFrame f;
        f.frameId = static_cast<uint64_t>(i);
        for (int k = 0; k < idsPerFrame; ++k) {
            int pid = (i + k) % nPoints;
            cv::Vec3d Xv(Xgt[pid].x, Xgt[pid].y, Xgt[pid].z);
            cv::Vec3d local = R.t() * (Xv - t);
            f.markerObs.push_back({cv::Point3d(local(0), local(1), local(2)), pid});
        }
        if (i == 0) {
            f.R_init = cv::Matx33d::eye();
            f.t_init = cv::Vec3d(0, 0, 0);
        } else {
            double da = angNoise(rng), db = angNoise(rng), dc = angNoise(rng);
            cv::Matx33d Rx(1, 0, 0, 0, std::cos(da), -std::sin(da), 0, std::sin(da), std::cos(da));
            cv::Matx33d Ry(std::cos(db), 0, std::sin(db), 0, 1, 0, -std::sin(db), 0, std::cos(db));
            cv::Matx33d Rz2(std::cos(dc), -std::sin(dc), 0, std::sin(dc), std::cos(dc), 0, 0, 0, 1);
            cv::Matx33d dR = Rz2 * Ry * Rx;
            f.R_init = dR * R;
            f.t_init = cv::Vec3d(t(0) + tNoise(rng), t(1) + tNoise(rng), t(2) + tNoise(rng));
        }
        in.frames.push_back(std::move(f));
    }
    if (gtOut) {
        gtOut->points = Xgt;
        gtOut->poses = gtPoses;
    }
    return in;
}

static double dist3(const cv::Point3d& a, const cv::Point3d& b) {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 按下标为 pid 的点全部观测加一致 x 向偏置(模拟该点观测系统性偏差/初值被扰动)
static void biasPointObs(GlobalBAInput& in, int pid, double bias) {
    for (auto& f : in.frames)
        for (auto& obs : f.markerObs)
            if (obs.globalId == pid)
                obs.local = cv::Point3d(obs.local.x + bias, obs.local.y, obs.local.z);
}

static const GlobalMarker& findMarker(const GlobalBAResult& r, int gid) {
    for (const auto& m : r.optimizedMarkers)
        if (m.globalId == gid) return m;
    ADD_FAILURE() << "marker " << gid << " not found";
    static GlobalMarker dummy;
    return dummy;
}

// ===== 0) MarkerPriorCost 残差值单测: r[i] = (X[i] − X_e[i]) / sigma =====
TEST(GlobalBASoftPrior, MarkerPriorCostResidualValue) {
    // X = X_e + (σ,0,0) → r = (1,0,0)
    // 注: 本 ceres 版 DynamicAutoDiffCostFunction 第二模板参数为 Stride, 残差数须 SetNumResiduals
    ceres::DynamicAutoDiffCostFunction<MarkerPriorCost> cost(
        new MarkerPriorCost{{1.0, 2.0, 3.0}, 0.001});
    cost.AddParameterBlock(3);
    cost.SetNumResiduals(3);
    double X[3] = {1.0 + 0.001, 2.0, 3.0};
    const double* params[1] = {X};
    double r[3] = {0, 0, 0};
    ASSERT_TRUE(cost.Evaluate(params, r, nullptr));
    EXPECT_NEAR(r[0], 1.0, 1e-12);
    EXPECT_NEAR(r[1], 0.0, 1e-12);
    EXPECT_NEAR(r[2], 0.0, 1e-12);
    // X == X_e → r == 0
    double X0[3] = {1.0, 2.0, 3.0};
    const double* params0[1] = {X0};
    ASSERT_TRUE(cost.Evaluate(params0, r, nullptr));
    EXPECT_NEAR(r[0], 0.0, 1e-12);
    EXPECT_NEAR(r[1], 0.0, 1e-12);
    EXPECT_NEAR(r[2], 0.0, 1e-12);
}

// ===== 1) 无先验输入: 行为与现状完全一致（两 Input 同跑 diff, 逐字节相同）=====
TEST(GlobalBASoftPrior, NoPriorRegression) {
    SyntheticGT gt;
    auto inA = makeSyntheticInput(20, 30, 15, &gt);
    GlobalBAInput inB = inA;  // 默认 ids 空 → 不走先验路径
    GlobalBAParams p;
    p.enablePoseGraphPreopt = false;
    GlobalBundleAdjustmentCPU op(p);
    auto rA = op.Execute(inA);
    auto rB = op.Execute(inB);
    ASSERT_TRUE(rA.success) << rA.message;
    ASSERT_TRUE(rB.success) << rB.message;
    // 位姿逐字节一致
    ASSERT_EQ(rA.optimizedPoses.size(), rB.optimizedPoses.size());
    for (size_t i = 0; i < rA.optimizedPoses.size(); ++i) {
        ASSERT_EQ(rA.optimizedPoses[i].frameId, rB.optimizedPoses[i].frameId);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                ASSERT_EQ(rA.optimizedPoses[i].R(a, b), rB.optimizedPoses[i].R(a, b));
        for (int k = 0; k < 3; ++k)
            ASSERT_EQ(rA.optimizedPoses[i].t(k), rB.optimizedPoses[i].t(k));
    }
    // 点逐字节一致
    ASSERT_EQ(rA.optimizedMarkers.size(), rB.optimizedMarkers.size());
    for (size_t j = 0; j < rA.optimizedMarkers.size(); ++j) {
        ASSERT_EQ(rA.optimizedMarkers[j].globalId, rB.optimizedMarkers[j].globalId);
        ASSERT_EQ(rA.optimizedMarkers[j].X.x, rB.optimizedMarkers[j].X.x);
        ASSERT_EQ(rA.optimizedMarkers[j].X.y, rB.optimizedMarkers[j].X.y);
        ASSERT_EQ(rA.optimizedMarkers[j].X.z, rB.optimizedMarkers[j].X.z);
    }
    EXPECT_EQ(rA.statistics.finalRMSE, rB.statistics.finalRMSE);
}

// ===== 2) 先验锚定: 高精度点观测一致偏置 0.03mm → 收敛后距 X_existing < 0.01mm =====
// 设计: pid 全部观测加 0.03mm 偏置(低于预清洗门限 0.05mm; chi2=3×9=27>11.34 会被末段剔除,
//       即便保留, 权重平衡下偏移 = 0.03×kσp²/(kσp²+σo²) ≪ 0.01)。
//       无先验实现时该点收敛到偏置位置(≈0.03mm 外) → 本用例 RED; 有先验(σ=0.001) → 锚回 GT。
TEST(GlobalBASoftPrior, PriorAnchorsPoint) {
    SyntheticGT gt;
    auto in = makeSyntheticInput(20, 30, 15, &gt);
    const int pid = 7;
    biasPointObs(in, pid, 0.03);
    in.highPrecisionGlobalIds = {pid};
    in.X_existing = {gt.points[pid].x, gt.points[pid].y, gt.points[pid].z};
    in.priorSigma.clear();  // 用 params 默认(0.001)

    GlobalBAParams p;
    p.enablePoseGraphPreopt = false;
    GlobalBundleAdjustmentCPU op(p);
    auto r = op.Execute(in);
    ASSERT_TRUE(r.success) << r.message;
    // 先验点: 几乎不动(距 X_existing < 0.01mm)
    EXPECT_LT(dist3(findMarker(r, pid).X, gt.points[pid]), 0.01);
    // 非先验点: 正常优化(靠近 GT)
    for (const auto& m : r.optimizedMarkers) {
        if (m.globalId == pid) continue;
        EXPECT_LT(dist3(m.X, gt.points[m.globalId]), 0.05) << "point " << m.globalId;
    }
}

// ===== 3) σ 可配: 先验位置偏离 GT 0.1mm（数据与先验"打架"）, σ 决定赢家 =====
//   σ=10(弱): 数据主导 → 点留在 GT 附近, 距 X_existing ≈ 0.1mm > 0.01mm(允许适度移动)
//   σ=1e-4(极小): 先验主导 → 点被锚死在 X_existing(< 0.01mm)
TEST(GlobalBASoftPrior, PriorSigmaHonored) {
    SyntheticGT gt;
    auto in = makeSyntheticInput(20, 30, 15, &gt);
    const int pid = 7;
    const cv::Point3d Xe(gt.points[pid].x + 0.1, gt.points[pid].y, gt.points[pid].z);
    in.highPrecisionGlobalIds = {pid};
    in.X_existing = {Xe.x, Xe.y, Xe.z};
    GlobalBAParams p;
    p.enablePoseGraphPreopt = false;

    // σ=10: 弱先验 → 数据赢
    in.priorSigma = {10.0};
    GlobalBundleAdjustmentCPU opWeak(p);
    auto rWeak = opWeak.Execute(in);
    ASSERT_TRUE(rWeak.success) << rWeak.message;
    const auto& mWeak = findMarker(rWeak, pid);
    EXPECT_GT(dist3(mWeak.X, Xe), 0.01) << "weak prior should allow movement";
    EXPECT_LT(dist3(mWeak.X, Xe), 0.15);
    EXPECT_LT(dist3(mWeak.X, gt.points[pid]), 0.02) << "data should dominate weak prior";

    // σ=1e-4: 极小先验 → 锚死
    in.priorSigma = {1e-4};
    GlobalBundleAdjustmentCPU opPin(p);
    auto rPin = opPin.Execute(in);
    ASSERT_TRUE(rPin.success) << rPin.message;
    EXPECT_LT(dist3(findMarker(rPin, pid).X, Xe), 0.01) << "tiny sigma should pin the point";
}

// ===== 4) 畸形 ids: 越界/长度不匹配 → 不崩、warn、有效部分生效 =====
TEST(GlobalBASoftPrior, MalformedIdsIgnored) {
    SyntheticGT gt;
    GlobalBAParams p;
    p.enablePoseGraphPreopt = false;

    // 场景 A: ids 含场景中不存在的 globalId(9999)
    {
        auto in = makeSyntheticInput(20, 30, 15, &gt);
        const int pid = 7;
        biasPointObs(in, pid, 0.03);
        in.highPrecisionGlobalIds = {pid, 9999};
        in.X_existing = {gt.points[pid].x, gt.points[pid].y, gt.points[pid].z,
                         0.0, 0.0, 0.0};  // 9999 槽位有值但点不存在
        // ringbuffer_sink 缓存经默认 logger 输出的消息(本 spdlog 版无 test_sink, 用其等价物)
        auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(4096);
        auto oldLogger = spdlog::default_logger();
        spdlog::set_default_logger(std::make_shared<spdlog::logger>("prior_test_a", sink));
        GlobalBundleAdjustmentCPU op(p);
        auto r = op.Execute(in);
        spdlog::set_default_logger(oldLogger);  // 先恢复全局 logger 再断言
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_LT(dist3(findMarker(r, pid).X, gt.points[pid]), 0.01);  // 有效部分生效
        bool warned = false;
        for (const auto& msg : sink->last_formatted())
            if (msg.find("9999") != std::string::npos) warned = true;
        EXPECT_TRUE(warned) << "expected warn about unknown globalId 9999";
    }
    // 场景 B: X_existing 长度不匹配(ids 2 个只给 3 值)
    {
        auto in = makeSyntheticInput(20, 30, 15, &gt);
        const int pid = 7, pid2 = 8;
        biasPointObs(in, pid, 0.03);
        in.highPrecisionGlobalIds = {pid, pid2};
        in.X_existing = {gt.points[pid].x, gt.points[pid].y, gt.points[pid].z};  // 只有 id[0] 的量
        auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(4096);
        auto oldLogger = spdlog::default_logger();
        spdlog::set_default_logger(std::make_shared<spdlog::logger>("prior_test_b", sink));
        GlobalBundleAdjustmentCPU op(p);
        auto r = op.Execute(in);
        spdlog::set_default_logger(oldLogger);
        ASSERT_TRUE(r.success) << r.message;
        EXPECT_LT(dist3(findMarker(r, pid).X, gt.points[pid]), 0.01);  // id[0] 有效生效
        bool warned = false;
        for (const auto& msg : sink->last_formatted())
            if (msg.find("too short") != std::string::npos) warned = true;
        EXPECT_TRUE(warned) << "expected warn about X_existing too short";
    }
}

// ===== 5) useSoftPrior=false 且字段已填 → 与无字段结果逐字节一致 =====
TEST(GlobalBASoftPrior, SoftPriorOffEqualsBaseline) {
    SyntheticGT gt;
    auto inBase = makeSyntheticInput(20, 30, 15, &gt);
    GlobalBAInput inFill = inBase;
    biasPointObs(inBase, 7, 0.03);
    biasPointObs(inFill, 7, 0.03);  // 两侧场景完全一致
    inFill.highPrecisionGlobalIds = {7, 8};
    inFill.X_existing = {gt.points[7].x, gt.points[7].y, gt.points[7].z,
                         gt.points[8].x, gt.points[8].y, gt.points[8].z};
    inFill.priorSigma = {0.001, 0.001};

    GlobalBAParams pOff;  pOff.enablePoseGraphPreopt = false;  pOff.useSoftPrior = false;
    GlobalBAParams pBase; pBase.enablePoseGraphPreopt = false;  // ids 空 → 亦无先验
    GlobalBundleAdjustmentCPU opOff(pOff);
    GlobalBundleAdjustmentCPU opBase(pBase);
    auto rOff = opOff.Execute(inFill);
    auto rBase = opBase.Execute(inBase);
    ASSERT_TRUE(rOff.success) << rOff.message;
    ASSERT_TRUE(rBase.success) << rBase.message;
    ASSERT_EQ(rOff.optimizedPoses.size(), rBase.optimizedPoses.size());
    for (size_t i = 0; i < rOff.optimizedPoses.size(); ++i) {
        ASSERT_EQ(rOff.optimizedPoses[i].frameId, rBase.optimizedPoses[i].frameId);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                ASSERT_EQ(rOff.optimizedPoses[i].R(a, b), rBase.optimizedPoses[i].R(a, b));
        for (int k = 0; k < 3; ++k)
            ASSERT_EQ(rOff.optimizedPoses[i].t(k), rBase.optimizedPoses[i].t(k));
    }
    ASSERT_EQ(rOff.optimizedMarkers.size(), rBase.optimizedMarkers.size());
    for (size_t j = 0; j < rOff.optimizedMarkers.size(); ++j) {
        ASSERT_EQ(rOff.optimizedMarkers[j].globalId, rBase.optimizedMarkers[j].globalId);
        ASSERT_EQ(rOff.optimizedMarkers[j].X.x, rBase.optimizedMarkers[j].X.x);
        ASSERT_EQ(rOff.optimizedMarkers[j].X.y, rBase.optimizedMarkers[j].X.y);
        ASSERT_EQ(rOff.optimizedMarkers[j].X.z, rBase.optimizedMarkers[j].X.z);
    }
    EXPECT_EQ(rOff.statistics.finalRMSE, rBase.statistics.finalRMSE);
}
