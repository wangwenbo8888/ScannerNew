// ============================================================================
// test_global_ba_zg.cpp — GBA 备用变体（ZG 思路: Schur/门控/σ₀）测试
// ============================================================================
// 覆盖: 合成网收敛(Schur 与 Cholesky 双路) / 入网门控抓飞点 / 后验 σ₀ 口径 /
// 与主路 GBA 的 A/B 同胜 / 退化输入拒绝。
// ============================================================================
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "global_ba_cpu.h"
#include "global_ba_zg.h"

using calib::GlobalBAFrame;
using calib::GlobalBAInput;
using calib::GlobalBAMarkerObs;
using calib::GlobalBAParams;
using calib::GlobalBAResult;
using calib::GlobalBundleAdjustmentCPU;
using calib::GlobalBundleAdjustmentZG;
using calib::ZgBaOptions;
using calib::ZgBaResult;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct SynthNet {
    GlobalBAInput input;
    int plantedOutliers = 0;
};

// 合成标志点网: Nf 个位姿绕圈看中央点云; 观测 = R'(X−T)+N(0,σ);
// plantedFrac 比例观测加 5mm 飞点; initPoseErr 给位姿初值加扰动。
SynthNet makeNet(int nFrames, int nPoints, double sigma, double plantedFrac,
                 double initRotDeg, double initTransMm, uint32_t seed) {
    std::mt19937 eng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // 真值点: 中央箱体
    std::vector<cv::Point3d> X(nPoints);
    for (auto& x : X)
        x = cv::Point3d(150 * gauss(eng), 150 * gauss(eng), 100 * gauss(eng));

    SynthNet net;
    net.plantedOutliers = 0;
    for (int i = 0; i < nFrames; ++i) {
        const double th = 2.0 * kPi * i / nFrames;
        cv::Matx33d R(std::cos(th), -std::sin(th), 0, std::sin(th), std::cos(th), 0, 0, 0, 1);
        const cv::Vec3d T(350 * std::cos(th), 350 * std::sin(th), 50);
        GlobalBAFrame f;
        f.frameId = (uint64_t)i;
        // 初值扰动: 平移 + 旋转(绕随机小轴)
        cv::Vec3d T0 = T + cv::Vec3d(gauss(eng), gauss(eng), gauss(eng)) * (initTransMm / 3.0);
        const double a = initRotDeg * kPi / 180.0 * gauss(eng) / 3.0;
        const double b = initRotDeg * kPi / 180.0 * gauss(eng) / 3.0;
        cv::Matx33d Rp(1, -b, a, b, 1, 0, -a, 0, 1);  // 小角近似
        f.R_init = R * Rp;
        f.t_init = T0;
        for (int j = 0; j < nPoints; ++j) {
            GlobalBAMarkerObs o;
            const cv::Vec3d pw(X[j].x - T(0), X[j].y - T(1), X[j].z - T(2));
            const cv::Vec3d z = R.t() * pw;
            o.local = cv::Point3d(z(0) + gauss(eng) * sigma, z(1) + gauss(eng) * sigma,
                                  z(2) + gauss(eng) * sigma);
            o.globalId = j;
            if (uni(eng) < plantedFrac) {
                o.local.x += 5.0 * gauss(eng);
                o.local.y += 5.0 * gauss(eng);
                o.local.z += 5.0 * gauss(eng);
                ++net.plantedOutliers;
            }
            f.markerObs.push_back(o);
        }
        net.input.frames.push_back(f);
    }
    return net;
}

}  // namespace

// ---------------------------------------------------------------------
// 好初值+飞点: Schur 求解收敛、门控抓飞点、σ₀ 口径合理
TEST(GlobalBaZg, ConvergeSchurWithGate) {
    auto net = makeNet(24, 60, 0.01, 0.04, 0.05, 0.05, 11u);
    GlobalBAParams bp;
    bp.maxIterations = 100;
    ZgBaOptions zo;
    zo.useSchur = true;
    GlobalBundleAdjustmentZG ba;
    const auto r = ba.Execute(net.input, bp, zo);
    ASSERT_TRUE(r.base.success) << r.base.message;
    EXPECT_LT(r.base.statistics.finalRMSE, 0.05) << "收敛后 RMS 应 <5σ(σ=0.01)";
    // 门控抓飞点: 5mm >> 门限(MAD 尺度≈0.02 → 门≈0.12mm)
    EXPECT_GE(r.obsGated, net.plantedOutliers * 9 / 10)
        << "入网门控应抓到 ≥90% 飞点 (" << r.obsGated << "/" << net.plantedOutliers << ")";
    // σ₀ 口径: 单位权后验 ≈1（噪声模型正确时）
    EXPECT_GT(r.sigma0FinalDimless, 0.3);
    EXPECT_LT(r.sigma0FinalDimless, 3.0);
    EXPECT_GT(r.dof, 0);
    EXPECT_EQ(r.solverName, "ITERATIVE_SCHUR+JACOBI");
}

// ---------------------------------------------------------------------
// 同网 Cholesky 路也收敛（求解器形态 A/B）
TEST(GlobalBaZg, ConvergeCholesky) {
    auto net = makeNet(24, 60, 0.01, 0.0, 0.05, 0.05, 22u);
    GlobalBAParams bp;
    ZgBaOptions zo;
    zo.useSchur = false;
    GlobalBundleAdjustmentZG ba;
    const auto r = ba.Execute(net.input, bp, zo);
    ASSERT_TRUE(r.base.success) << r.base.message;
    EXPECT_LT(r.base.statistics.finalRMSE, 0.05);
    EXPECT_EQ(r.solverName, "SPARSE_NORMAL_CHOLESKY");
}

// ---------------------------------------------------------------------
// 粗初值(1mm/0.5°)+无飞点: 鲁棒收敛（门控自动放弃也不崩）
TEST(GlobalBaZg, CoarseInitConverges) {
    auto net = makeNet(24, 60, 0.01, 0.0, 0.5, 1.0, 33u);
    GlobalBAParams bp;
    bp.maxIterations = 200;
    ZgBaOptions zo;
    GlobalBundleAdjustmentZG ba;
    const auto r = ba.Execute(net.input, bp, zo);
    ASSERT_TRUE(r.base.success) << r.base.message;
    EXPECT_LT(r.base.statistics.finalRMSE, 0.08);
}

// ---------------------------------------------------------------------
// A/B: 主路 GBA 与变体同输入都成功（对比评测入口可用性）
TEST(GlobalBaZg, AbBothSolversSucceed) {
    auto net = makeNet(16, 40, 0.01, 0.02, 0.1, 0.2, 44u);
    GlobalBAParams bp;
    GlobalBAResult mainRes;
    ZgBaResult zgRes;
    ASSERT_TRUE(GlobalBundleAdjustmentZG::RunAb(net.input, bp, {}, mainRes, zgRes));
    EXPECT_LT(mainRes.statistics.finalRMSE, 0.08);
    EXPECT_LT(zgRes.base.statistics.finalRMSE, 0.08);
    printf("[AbBoth] main RMSE=%.4f mm, zg RMSE=%.4f mm (sigma0=%.3f)\n",
           mainRes.statistics.finalRMSE, zgRes.base.statistics.finalRMSE,
           zgRes.sigma0FinalDimless);
}

// ---------------------------------------------------------------------
// 退化输入: 单帧拒绝
TEST(GlobalBaZg, DegenerateRejected) {
    GlobalBAInput in;
    GlobalBAFrame f;
    f.frameId = 0;
    f.markerObs.push_back({cv::Point3d(1, 2, 3), 7});
    in.frames.push_back(f);
    GlobalBundleAdjustmentZG ba;
    const auto r = ba.Execute(in, GlobalBAParams{}, ZgBaOptions{});
    EXPECT_FALSE(r.base.success);
    EXPECT_NE(r.base.message.find("frames"), std::string::npos);
}
