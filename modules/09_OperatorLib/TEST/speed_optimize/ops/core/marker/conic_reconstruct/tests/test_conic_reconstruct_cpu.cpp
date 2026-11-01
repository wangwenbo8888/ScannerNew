#include <gtest/gtest.h>
#include "../conic_reconstruct_cpu.h"
#include "common/calib_warmup_config.h"
#include "common/testutil/marker_synth.h"

#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace calib;
using calib::synth::CircleSpec;
using calib::synth::EllipseObs;
using calib::synth::MarkerSynth;
using calib::synth::Rig;

class ConicParamsTest : public ::testing::Test {
protected:
    ConicReconstructCPUParams p_;
};

TEST_F(ConicParamsTest, DefaultsPassValidate) { EXPECT_NO_THROW(p_.validate()); }

TEST_F(ConicParamsTest, InvalidThrows) {
    auto q = p_; q.nominalRadiusMm = 0;      EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.radiusToleranceRatio = 0;      EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.consistencyMax = 0;            EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.circleAxisRatioMax = 0;        EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.degeneracyGateAngleDeg = 0;    EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.depthMaxMm = 100;              EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.maxReprojResidualPx = 0;       EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.ringPointsForViz = -1;         EXPECT_THROW(q.validate(), std::invalid_argument);
}

// 行偏置预修正参数：JSON 键往返＋空 JSON 默认 0（默认关）＋validate 边界
// （|offset|≤50 过，±50.1 抛——超出合理标定残差范围）
TEST_F(ConicParamsTest, RowOffsetJsonRoundtrip) {
    p_.rowOffsetPx = 12.5;
    auto r = ConicReconstructCPUParams::fromJson(p_.toJson());
    EXPECT_DOUBLE_EQ(r.rowOffsetPx, 12.5);
    EXPECT_DOUBLE_EQ(ConicReconstructCPUParams::fromJson(nlohmann::json{}).rowOffsetPx, 0.0);
    auto q = p_; q.rowOffsetPx = 50.0;   EXPECT_NO_THROW(q.validate());
    q = p_; q.rowOffsetPx = 50.1;        EXPECT_THROW(q.validate(), std::invalid_argument);
    q = p_; q.rowOffsetPx = -50.0;       EXPECT_NO_THROW(q.validate());
    q = p_; q.rowOffsetPx = -50.1;       EXPECT_THROW(q.validate(), std::invalid_argument);
}

TEST_F(ConicParamsTest, JsonRoundtrip) {
    p_.nominalRadiusMm = 2.5; p_.maxReprojResidualPx = 1.2; p_.yPreGatePx = 50;
    auto r = ConicReconstructCPUParams::fromJson(p_.toJson());
    EXPECT_DOUBLE_EQ(r.nominalRadiusMm, 2.5);
    EXPECT_DOUBLE_EQ(r.maxReprojResidualPx, 1.2);
    EXPECT_DOUBLE_EQ(r.yPreGatePx, 50);
    EXPECT_NO_THROW(r.validate());
}

TEST_F(ConicParamsTest, JsonEmptyGivesDefaults) {
    auto r = ConicReconstructCPUParams::fromJson(nlohmann::json{});
    EXPECT_DOUBLE_EQ(r.nominalRadiusMm, 3.0);
    EXPECT_DOUBLE_EQ(r.maxReprojResidualPx, 1.5);
}

TEST(ConicContractTest, EmptyInputs) {
    const Rig rig = Rig::scanRig();
    ConicReconstructCPU op;
    op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);  // Task 3 起未标定直接 Execute → success=false
    auto r = op.Execute({}, {}, {});
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(r.markers.empty());
}

TEST(ConicContractTest, NoMatchesAllMinusOne) {
    const Rig rig = Rig::scanRig();
    ConicReconstructCPU op;
    op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
    std::vector<EllipseFitCPUResult> e(2);
    e[0].success = e[1].success = true;
    auto r = op.Execute(e, e, {-1, -1});
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.statistics.totalPairs, 0u);
}

// ============================================================
// Task 3：锥面/束/平面拆分/截面圆 核心数学（合成位姿回代）
// ============================================================

namespace {

// EllipseObs → EllipseFitCPUResult：a/b 无大小序（fitEllipse 的 width/height
// 不保证有序），直接 a→majorAxis/b→minorAxis 映射，锥面公式天然处理任意序
EllipseFitCPUResult obsToResult(const EllipseObs& o) {
    EllipseFitCPUResult e;
    e.success = true;
    e.centerX = o.cx;
    e.centerY = o.cy;
    e.majorAxis = o.a;
    e.minorAxis = o.b;
    e.angle = o.angleDeg;
    return e;
}

ConicReconstructCPUResult runOne(ConicReconstructCPU& op, const calib::synth::SyntheticMarker& m) {
    std::vector<EllipseFitCPUResult> left;
    left.push_back(obsToResult(m.left));
    std::vector<EllipseFitCPUResult> right;
    right.push_back(obsToResult(m.right));
    return op.Execute(left, right, {0});
}

// 断言 helper：取 r.markers[0]，圆心各轴 / 法向夹角 / 半径对 truth 回代
void expectCircle(const ConicReconstructCPUResult& r, const CircleSpec& t,
                  double tolC_mm, double tolN_deg, double tolR_mm, const char* label = "") {
    ASSERT_FALSE(r.markers.empty()) << label << ": no marker reconstructed";
    const MarkerReconstructResult& m = r.markers[0];
    EXPECT_NEAR(m.centerX, t.C(0), tolC_mm) << label;
    EXPECT_NEAR(m.centerY, t.C(1), tolC_mm) << label;
    EXPECT_NEAR(m.centerZ, t.C(2), tolC_mm) << label;
    // Task 5 起法向统一翻向左相机（n·C<0）——与 truth 法向只比轴线夹角，不比方向
    const double dot = std::abs(m.normalX * t.n(0) + m.normalY * t.n(1) + m.normalZ * t.n(2));
    const double angDeg = std::acos(std::max(-1.0, std::min(1.0, dot))) * 180.0 / CV_PI;
    EXPECT_LE(angDeg, tolN_deg) << label << ": angleDeg=" << angDeg;
    EXPECT_NEAR(m.circleFit.radius, t.r, tolR_mm) << label;
}

// helper：同一合成噪声对（45°/45°、0.1px）两次 Execute 的结果对（确定性比较用）
std::pair<ConicReconstructCPUResult, ConicReconstructCPUResult> runIdempotentPair() {
    const Rig rig = Rig::scanRig();
    ConicReconstructCPU op;
    op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
    MarkerSynth synth(rig, 42);
    const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0), 48, 0.1);
    std::vector<EllipseFitCPUResult> left;
    left.push_back(obsToResult(m.left));
    std::vector<EllipseFitCPUResult> right;
    right.push_back(obsToResult(m.right));
    auto r1 = op.Execute(left, right, {0});
    auto r2 = op.Execute(left, right, {0});
    return {std::move(r1), std::move(r2)};
}

} // namespace

// 确定性契约：同输入两次 Execute 逐位一致（含 GN 抛光路径——无随机性）
TEST(ConicContractTest, IdempotentSameInput) {
    auto rr = runIdempotentPair();
    const ConicReconstructCPUResult& r1 = rr.first;
    const ConicReconstructCPUResult& r2 = rr.second;
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.markers.size(), 1u);
    ASSERT_EQ(r2.markers.size(), 1u);
    const MarkerReconstructResult& m1 = r1.markers[0];
    const MarkerReconstructResult& m2 = r2.markers[0];
    EXPECT_EQ(m1.centerX, m2.centerX);
    EXPECT_EQ(m1.centerY, m2.centerY);
    EXPECT_EQ(m1.centerZ, m2.centerZ);
    EXPECT_EQ(m1.normalX, m2.normalX);
    EXPECT_EQ(m1.normalY, m2.normalY);
    EXPECT_EQ(m1.normalZ, m2.normalZ);
    EXPECT_EQ(m1.circleFit.radius, m2.circleFit.radius);
    EXPECT_EQ(m1.validCircle, m2.validCircle);
    EXPECT_EQ(m1.validPlane, m2.validPlane);
    ASSERT_EQ(r1.diagnostics.size(), 1u);
    ASSERT_EQ(r2.diagnostics.size(), 1u);
    const ConicPairDiagnostic& d1 = r1.diagnostics[0];
    const ConicPairDiagnostic& d2 = r2.diagnostics[0];
    EXPECT_EQ(std::make_pair(d1.leftIdx, d1.rightIdx), std::make_pair(d2.leftIdx, d2.rightIdx));
    EXPECT_EQ(d1.failReason, d2.failReason);
    EXPECT_EQ(d1.etaBaseline, d2.etaBaseline);
    EXPECT_EQ(d1.etaView, d2.etaView);
    EXPECT_EQ(d1.reconstructedRadiusMm, d2.reconstructedRadiusMm);
    EXPECT_EQ(d1.reprojResidualPx, d2.reprojResidualPx);
    EXPECT_EQ(d1.sectionAxisRatio, d2.sectionAxisRatio);
    EXPECT_EQ(d1.solutionCount, d2.solutionCount);
    EXPECT_EQ(r1.statistics.totalPairs, r2.statistics.totalPairs);
    // candidatePairs 已于 Task 5 接线（本用例=1：种子 1、无预筛新增）——确定性口径
    EXPECT_EQ(r1.statistics.candidatePairs, r2.statistics.candidatePairs);
    EXPECT_EQ(r1.statistics.validCount, r2.statistics.validCount);
    EXPECT_EQ(r1.statistics.rejectedCount, r2.statistics.rejectedCount);
    for (size_t k = 0; k < kFailReasonCount; ++k) {
        EXPECT_EQ(r1.statistics.failReasonHistogram[k], r2.statistics.failReasonHistogram[k]) << "k=" << k;
    }
}

class ConicSolveTest : public ::testing::Test {
protected:
    Rig rig_ = Rig::scanRig();
    ConicReconstructCPU op_;

    void SetUp() override { op_.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2); }
};

TEST_F(ConicSolveTest, FrontoParallel) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_FALSE(r.markers.empty());
    expectCircle(r, m.truth, 0.2, 1.5, 0.05);
    EXPECT_NEAR(std::abs(r.markers[0].normalZ), 1.0, 1e-3);
    // 截面轴比诊断：成功对 0 ≤ ratio ≤ 闸（正对精确合成，实测 2.2e-6 ≪ 0.03）
    ASSERT_EQ(r.diagnostics.size(), 1u);
    const double ratio = r.diagnostics[0].sectionAxisRatio;
    std::cout << "fronto sectionAxisRatio: " << ratio << "\n";
    EXPECT_GE(ratio, 0.0);
    EXPECT_LE(ratio, op_.GetParams().circleAxisRatioMax);
}

TEST_F(ConicSolveTest, Tilt45Az90) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(45, 90, 431, 3.0));
    expectCircle(runOne(op_, m), m.truth, 0.1, 0.5, 0.05);
}

TEST_F(ConicSolveTest, Tilt60Az0) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(60, 0, 431, 3.0));
    expectCircle(runOne(op_, m), m.truth, 0.2, 0.5, 0.05);
}

TEST_F(ConicSolveTest, Tilt15Az0) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(15, 0, 431, 3.0));
    expectCircle(runOne(op_, m), m.truth, 0.1, 1.5, 0.05);
}

TEST_F(ConicSolveTest, Noise01px) {
    for (uint64_t k = 0; k < 30; ++k) {
        MarkerSynth synth(rig_, 42 + k);
        auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0), 48, 0.1);
        auto r = runOne(op_, m);
        const std::string label = "seed=" + std::to_string(42 + k);
        ASSERT_FALSE(r.markers.empty()) << label;
        expectCircle(r, m.truth, 0.15, 0.8, 0.05, label.c_str());
    }
}

TEST_F(ConicSolveTest, PhysicalEllipseRejected) {
    MarkerSynth synth(rig_, 42);
    const CircleSpec spec = CircleSpec::pose(0, 0, 431, 3.0);
    const auto m = synth.make(spec);
    // 右目：同 3D 位置、面内 u 轴拉伸 1.35 的椭圆投影（内联生成边点 fitEllipse 构造右 EllipseObs）
    cv::Vec3d ax(1, 0, 0);
    if (std::abs(spec.n(0)) > 0.9) ax = cv::Vec3d(0, 1, 0);
    cv::Vec3d u = spec.n.cross(ax); u /= cv::norm(u);
    cv::Vec3d v = spec.n.cross(u);
    std::vector<cv::Point2f> pts;
    for (int i = 0; i < 48; ++i) {
        const double th = 2 * CV_PI * i / 48.0;
        const cv::Vec3d X = spec.C + spec.r * (1.35 * std::cos(th) * u + std::sin(th) * v);
        const cv::Vec3d xc = rig_.K2 * (X - rig_.C2);
        pts.emplace_back(static_cast<float>(xc(0) / xc(2)), static_cast<float>(xc(1) / xc(2)));
    }
    const cv::RotatedRect rr = cv::fitEllipse(pts);
    std::vector<EllipseFitCPUResult> left;
    left.push_back(obsToResult(m.left));
    std::vector<EllipseFitCPUResult> right;
    right.push_back(obsToResult(EllipseObs{rr.center.x, rr.center.y, rr.size.width, rr.size.height, rr.angle}));
    auto r = op_.Execute(left, right, {0});
    ASSERT_EQ(r.diagnostics.size(), 1u);
    const ConicFailReason fr = r.diagnostics[0].failReason;
    EXPECT_NE(fr, ConicFailReason::None);
    EXPECT_TRUE(fr == ConicFailReason::NotCircle || fr == ConicFailReason::Inconsistent)
        << "actual failReason=" << static_cast<int>(fr);
}

TEST(ConicGateTest, NotCalibrated) {
    ConicReconstructCPU op;  // 不调 SetProjectionMatrices 直接 Execute
    std::vector<EllipseFitCPUResult> e(1);
    e[0].success = true; e[0].centerX = 1000; e[0].centerY = 700;
    e[0].majorAxis = 24; e[0].minorAxis = 23; e[0].angle = 30;
    auto r = op.Execute(e, e, {0});
    EXPECT_FALSE(r.success);
}

// ============================================================
// Task 4：闸门链（cheirality/半径/Sampson 残差/双条件病态＋isfinite 前置＋灰区降级）
// 闸序：cheirality → radius → residual → degeneracy（设计 §2.6）
// ============================================================

class ConicGateChainTest : public ::testing::Test {
protected:
    Rig rig_ = Rig::scanRig();
    ConicReconstructCPU op_;   // 默认参数

    void SetUp() override { op_.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2); }
};

// ② 半径闸：真值 r=3.0、nominal=2.0 → 偏差 50% > 15% → Radius 拒
TEST_F(ConicGateChainTest, RadiusGateBoundary) {
    ConicReconstructCPUParams p;
    p.nominalRadiusMm = 2.0;
    ConicReconstructCPU op(p);
    op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    auto r = runOne(op, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::Radius);
    EXPECT_TRUE(r.markers.empty());
}

// ③ 残差闸：右目轴 ×1.15（同中心错锥）
// 实测钉（Task 4 探针）：默认阈值下该失配量级被上游截面非圆闸先拦——抛光后
// 右锥截面轴比 0.035~0.063 > circleAxisRatioMax=0.03，failReason=NotCircle(7)，
// 候选圆未达残差闸（reprojResidualPx=-1）。残差闸独立触发面见 ResidualGateFiresRelaxedSection
TEST_F(ConicGateChainTest, ResidualGatePreemptedBySectionGate) {
    MarkerSynth synth(rig_, 42);
    const auto m = MarkerSynth::axesScaleRight(
        synth.make(CircleSpec::pose(45, 45, 431, 3.0)), 1.15);
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::NotCircle);
    // 截面轴比诊断：拒对须给出超闸量值（NotCircle 不再只给枚举；实测 0.179 > 0.03，
    // 抛光后两锥截面取 max 口径）
    const double ratio = r.diagnostics[0].sectionAxisRatio;
    std::cout << "axesScale1.15 sectionAxisRatio: " << ratio << "\n";
    EXPECT_GT(ratio, op_.GetParams().circleAxisRatioMax);
    EXPECT_TRUE(r.markers.empty());
}

// ③ 残差闸独立触发：放宽截面闸（circleAxisRatioMax=0.5）暴露残差闸本体——
// 右椭圆单轴（各向异性）×1.3：刚圆不可吸收，大圆近距（r=10mm Z=200，半轴 92px）
// 放大像素杠杆。实测钉：max-median 残差 3.03px > 1.5px → failReason=Residual
TEST_F(ConicGateChainTest, ResidualGateFiresRelaxedSection) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 200, 10.0));
    m.right.a *= 1.3;   // 各向异性拉伸（单轴）
    ConicReconstructCPUParams p;
    p.nominalRadiusMm = 10.0;
    p.radiusToleranceRatio = 0.3;
    p.circleAxisRatioMax = 0.5;
    ConicReconstructCPU op(p);
    op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
    auto r = runOne(op, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::Residual);
    EXPECT_GT(r.diagnostics[0].reprojResidualPx, 1.5);   // 实测 3.03px（拒也填，审计）
    EXPECT_TRUE(r.markers.empty());
}

// 灰区·残差带：同 ResidualGateFiresRelaxedSection 构造，调 τ=maxReprojResidualPx=3.2
// 使实测残差 ~3.03px 落闭带 [0.9τ,τ]=[2.88,3.2] → 过闸输出＋qualityFlag==Degraded
TEST_F(ConicGateChainTest, GrayZoneResidualBand) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 200, 10.0));
    m.right.a *= 1.3;   // 各向异性拉伸（单轴），实测 max-median 残差 3.03px
    ConicReconstructCPUParams p;
    p.nominalRadiusMm = 10.0;
    p.radiusToleranceRatio = 0.3;
    p.circleAxisRatioMax = 0.5;
    p.maxReprojResidualPx = 3.2;   // 灰区带下沿 0.9×3.2=2.88 < 3.03 ≤ 3.2
    ConicReconstructCPU op(p);
    op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
    auto r = runOne(op, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
    ASSERT_EQ(r.markers.size(), 1u);
    const double res = r.diagnostics[0].reprojResidualPx;   // 成功对=幸存者残差
    EXPECT_GT(res, 0.9 * p.maxReprojResidualPx);            // 带内（实测 3.03）
    EXPECT_LE(res, p.maxReprojResidualPx);
    EXPECT_EQ(r.qualityFlag, QualityFlag::Degraded);        // 过闸但灰区 → 整批降级
}

// ④ 病态闸：pose(90,90)——n̂=(0,±1,0)，圆平面 y=0 含基线（etaBaseline≈0 且 etaView≈0）
// 实测钉（Task 4 探针）：该位形在闸序上游即被拦——平面含基线 ⇒ 两目投影共线 ⇒
// fitEllipse 退化为零面积椭圆 ⇒ buildCone 拒 → failReason=InputClass(14)。
// 纯闸触发位形见 DegeneracyFlatPureGateTrigger
TEST_F(ConicGateChainTest, DegeneracyFlatRejects) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(90, 90, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::InputClass);
    EXPECT_TRUE(r.markers.empty());
}

// ④ 病态闸纯触发：pose(85,90)——圆平面不含基线（上游数学可解出圆）但
// etaBaseline=|n̂·b̂|≈0 且 etaView=cos85°≈0.087 < sin8°≈0.139 → 双条件皆病态。
// 规格建议的 pose(82,90) 实测恰在闸界（cos82°≡sin8°，etaView=0.1392≥sinθg 放行）
// ——重建噪声下是掷硬币边界，故取 85°（余量 0.052，确定性拒）
TEST_F(ConicGateChainTest, DegeneracyFlatPureGateTrigger) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(85, 90, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    const ConicPairDiagnostic& d = r.diagnostics[0];
    EXPECT_EQ(d.failReason, ConicFailReason::Degeneracy);
    EXPECT_LT(d.etaBaseline, 0.02);   // 实测 0.000
    EXPECT_LT(d.etaView, 0.12);       // 实测 0.087 < sin8°≈0.139
    EXPECT_TRUE(r.markers.empty());
}

// ④ 回归锚：正对位形（n̂⊥基线、n̂∥视线）etaBaseline≈0 但 etaView≈1 → 双条件放行
TEST_F(ConicGateChainTest, FrontoParallelPassesDegeneracy) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    ASSERT_EQ(r.markers.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
    EXPECT_LT(r.diagnostics[0].etaBaseline, 0.01);
    EXPECT_GT(r.diagnostics[0].etaView, 0.99);
}

// ① cheirality：Z=−431（左目后方）→ 深度窗 [100,2000]mm 外
TEST_F(ConicGateChainTest, CheiralityBehindCamera) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, -431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::Cheirality);
}

// ① cheirality：Z=2500（r 同比放大维持表观尺寸）出 2000mm 深度窗
TEST_F(ConicGateChainTest, DepthWindowFar) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(0, 0, 2500, 20.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::Cheirality);
}

// 束判据：右光心=左光心（C2=0）＋左右同椭圆 → 两锥重合 → Coincident
TEST_F(ConicGateChainTest, CoincidentCones) {
    ConicReconstructCPU op;   // 默认参数
    op.SetProjectionMatrices(rig_.K1, rig_.K1, cv::Vec3d(0.0, 0.0, 0.0));
    MarkerSynth synth(rig_, 42);
    const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    std::vector<EllipseFitCPUResult> left;
    left.push_back(obsToResult(m.left));
    std::vector<EllipseFitCPUResult> right;
    right.push_back(obsToResult(m.left));   // 左右同椭圆
    auto r = op.Execute(left, right, {0});
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::Coincident);
}

// 束判据：右轴 ×1.8（强错锥）→ 无公共实圆 → NoRealRoot/Inconsistent/VirtualPlane 之一
TEST_F(ConicGateChainTest, VirtualPlaneOrNoRealRoot) {
    MarkerSynth synth(rig_, 42);
    const auto m = MarkerSynth::axesScaleRight(
        synth.make(CircleSpec::pose(45, 45, 431, 3.0)), 1.8);
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    const ConicFailReason fr = r.diagnostics[0].failReason;
    EXPECT_TRUE(fr == ConicFailReason::NoRealRoot || fr == ConicFailReason::Inconsistent ||
                fr == ConicFailReason::VirtualPlane)
        << "actual failReason=" << static_cast<int>(fr);
}

// isfinite 前置：混入 centerX=NaN 对 → NaN 对 Numerical 拒，其余对正常，批 Warning
TEST_F(ConicGateChainTest, NaNIsolated) {
    MarkerSynth synth(rig_, 42);
    const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    std::vector<EllipseFitCPUResult> left;
    left.push_back(obsToResult(m.left));
    left.push_back(obsToResult(m.left));
    left[1].centerX = std::numeric_limits<double>::quiet_NaN();
    std::vector<EllipseFitCPUResult> right;
    right.push_back(obsToResult(m.right));
    // Task 5 预筛起：右目重复椭圆会生成同左第二候选（几何全同→双解残差相等→判
    // Ambiguous，属新语义下的正确行为）。此处右[1]平移 +1000px 出视差窗，保持本
    // 用例「NaN 隔离」语义不变（(0,1) 预筛出局、(1,1) 种子照解）
    EllipseFitCPUResult rightFar = obsToResult(m.right);
    rightFar.centerX += 1000.0;
    right.push_back(std::move(rightFar));
    auto r = op_.Execute(left, right, {0, 1});
    EXPECT_TRUE(r.success);
    ASSERT_EQ(r.diagnostics.size(), 2u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
    EXPECT_EQ(r.diagnostics[1].failReason, ConicFailReason::Numerical);
    ASSERT_EQ(r.markers.size(), 1u);
    EXPECT_EQ(r.statistics.validCount, 1u);
    EXPECT_EQ(r.statistics.totalPairs, 2u);
    EXPECT_EQ(r.qualityFlag, QualityFlag::Warning);   // validCount < candidatePairs（Task 5 起分母为候选数口径）
}

// 灰区降级：调 nominal 使半径偏差恰为 0.14 ∈ [0.9,1.0)×0.15 → 过闸输出但整批 Degraded
TEST_F(ConicGateChainTest, GrayZoneDegraded) {
    ConicReconstructCPUParams p;
    p.nominalRadiusMm = 3.0 / 0.86;   // |3.0−nominal|/nominal = 0.14
    ConicReconstructCPU op(p);
    op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    auto r = runOne(op, m);
    ASSERT_EQ(r.markers.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
    EXPECT_EQ(r.qualityFlag, QualityFlag::Degraded);
}

// 灰区·病态余量带：无噪 pose(81.5,90)——etaBaseline≈0、etaView=cos81.5°≈0.148
// 落 [sin8°, sin8°/0.9)≈[0.139,0.155)（过闸但 margin·0.9<sinθg）→ Degraded
TEST_F(ConicGateChainTest, GrayZoneEtaBand) {
    MarkerSynth synth(rig_, 42);
    auto m = synth.make(CircleSpec::pose(81.5, 90, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
    ASSERT_EQ(r.markers.size(), 1u);
    const double sinGate = std::sin(8.0 * CV_PI / 180.0);
    const double etaView = r.diagnostics[0].etaView;        // 实测 0.1481（cos81.5°=0.14813）
    EXPECT_GE(etaView, sinGate);                            // 带下沿：过双条件闸
    EXPECT_LT(etaView, sinGate / 0.9);                      // 带上沿：灰区内（margin·0.9<sinθg）
    EXPECT_NEAR(etaView, 0.1481, 0.003);                    // 实测锚（无噪重建误差 ~1e-4）
    EXPECT_EQ(r.qualityFlag, QualityFlag::Degraded);
}

// 上轮审查核验点 4 回归：60° 倾斜＋0.1px 噪声、30 种子 → 接受率 ≥90% 且过闸点精度达标
// （实测 29/30：需抛光残差取两锥距离口径 XᵀΠX/‖ΠX‖ 拉平折衷——代数口径下右锥
//   截面比被牺牲到 0.035~0.063>0.03，接受率仅 14/30，见 conic_reconstruct_cpu.cpp 勘误注）
TEST_F(ConicSolveTest, Tilt60RealisticNoiseAcceptance) {
    int accepted = 0;
    for (uint64_t k = 0; k < 30; ++k) {
        MarkerSynth synth(rig_, 42 + k);
        auto m = synth.make(CircleSpec::pose(60, 0, 431, 3.0), 48, 0.1);
        auto r = runOne(op_, m);
        const std::string label = "seed=" + std::to_string(42 + k);
        if (!r.markers.empty()) {
            ++accepted;
            expectCircle(r, m.truth, 0.2, 0.5, 0.05, label.c_str());
        } else {
            ASSERT_FALSE(r.diagnostics.empty()) << label;   // 拒也须有诊断可审计
        }
    }
    EXPECT_GE(accepted, 27);   // ≥ 27/30（90%）
}

// ============================================================
// Task 5：候选生成（种子＋预筛 cap）／同左多右竞争消歧／结果组装完善
// ============================================================

namespace {

// 直造椭圆（脱离 synth：预筛/上限用例只需像素参数可预筛，不需真实可解）
EllipseFitCPUResult makeEllipse(double cx, double cy, double a, double b, double angDeg) {
    EllipseFitCPUResult e;
    e.success = true;
    e.centerX = cx;
    e.centerY = cy;
    e.majorAxis = a;
    e.minorAxis = b;
    e.angle = angDeg;
    return e;
}

EllipseObs scaleObs(const EllipseObs& o, double k) {
    EllipseObs r = o;
    r.a *= k;
    r.b *= k;
    return r;
}

// EllipseFitCPUResult 仅可移动——向量经 push_back 装配（禁 initializer_list 拷贝）
std::vector<EllipseFitCPUResult> obsVec(const EllipseObs& o1, const EllipseObs& o2) {
    std::vector<EllipseFitCPUResult> v;
    v.push_back(obsToResult(o1));
    v.push_back(obsToResult(o2));
    return v;
}
std::vector<EllipseFitCPUResult> obsVec(const EllipseObs& o) {
    std::vector<EllipseFitCPUResult> v;
    v.push_back(obsToResult(o));
    return v;
}

// 像素点到椭圆二次曲线的 Sampson 距离 px（半轴换算与算子内 ellipseToConic 同口径）
double ellipseSampsonPx(double u, double v, const EllipseObs& o) {
    const double sa = 0.5 * o.a, sb = 0.5 * o.b;
    const double rad = o.angleDeg * CV_PI / 180.0;
    const double cs = std::cos(rad), sn = std::sin(rad);
    const double ia2 = 1.0 / (sa * sa), ib2 = 1.0 / (sb * sb);
    const double A = cs * cs * ia2 + sn * sn * ib2;
    const double B = 2.0 * cs * sn * (ia2 - ib2);
    const double C = sn * sn * ia2 + cs * cs * ib2;
    const double D = -(A * o.cx + 0.5 * B * o.cy);
    const double E = -(0.5 * B * o.cx + C * o.cy);
    const double F = A * o.cx * o.cx + B * o.cx * o.cy + C * o.cy * o.cy - 1.0;
    const double num = std::abs(A * u * u + B * u * v + C * v * v + 2.0 * D * u + 2.0 * E * v + F);
    const double gx = 2.0 * A * u + B * v + 2.0 * D;
    const double gy = B * u + 2.0 * C * v + 2.0 * E;
    return num / std::hypot(gx, gy);
}

// 拒因名：单源 ConicFailReasonName（算子头导出——禁本地复制名表）

} // namespace

class ConicCandidateTest : public ::testing::Test {
protected:
    Rig rig_ = Rig::scanRig();
    ConicReconstructCPU op_;

    void SetUp() override { op_.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2); }
};

// 预筛恢复未配对真对：centerMatches 全 −1（marker_match 失配场景），左右各 1 个
// 真椭圆（视差 ~558px 落窗、|Δv|≈0）→ 预筛找到对并解出
TEST_F(ConicCandidateTest, PrescreenRecoversUnmatched) {
    MarkerSynth synth(rig_, 42);
    const auto m = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    auto r = op_.Execute(obsVec(m.left), obsVec(m.right), {-1});
    EXPECT_EQ(r.statistics.totalPairs, 0u);          // 种子口径：无种子
    EXPECT_EQ(r.statistics.candidatePairs, 1u);      // 预筛 1 对
    ASSERT_EQ(r.markers.size(), 1u);
    expectCircle(r, m.truth, 0.2, 1.5, 0.05);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
}

// 种子对必进＋预筛对并存：左 2 椭圆右 2，centerMatches 只指 1 对；第二标记点放
// v≈1403px（与第一标记 Δv≈640px>60 → 交叉对被 y 预筛门杀，真对 (1,1) 视差落窗收录）
TEST_F(ConicCandidateTest, SeedPairAlwaysIncluded) {
    MarkerSynth synth(rig_, 42);
    const auto mA = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    CircleSpec specB;
    specB.C = cv::Vec3d(0.0, 150.0, 431.0);   // v_L≈763+1838·150/431≈1403
    specB.n = cv::Vec3d(0.0, 0.0, 1.0);
    specB.r = 3.0;
    const auto mB = synth.make(specB);
    std::vector<EllipseFitCPUResult> left = obsVec(mA.left, mB.left);
    std::vector<EllipseFitCPUResult> right = obsVec(mA.right, mB.right);
    auto r = op_.Execute(left, right, {0, -1});
    EXPECT_EQ(r.statistics.totalPairs, 1u);          // 种子 (0,0)
    EXPECT_EQ(r.statistics.candidatePairs, 2u);      // 种子＋预筛 (1,1)
    ASSERT_EQ(r.markers.size(), 2u);
    EXPECT_EQ(r.markers[0].leftEllipseIdx, 0);
    EXPECT_EQ(r.markers[0].rightEllipseIdx, 0);
    EXPECT_NEAR(r.markers[0].centerX, 0.0, 0.2);
    EXPECT_NEAR(r.markers[0].centerY, 0.0, 0.2);
    EXPECT_NEAR(r.markers[0].centerZ, 431.0, 0.2);
    EXPECT_EQ(r.markers[1].leftEllipseIdx, 1);       // 预筛对也解出
    EXPECT_EQ(r.markers[1].rightEllipseIdx, 1);
    EXPECT_NEAR(r.markers[1].centerX, 0.0, 0.2);
    EXPECT_NEAR(r.markers[1].centerY, 150.0, 0.2);
    EXPECT_NEAR(r.markers[1].centerZ, 431.0, 0.2);
}

// 同左多右竞争：左 1 真椭圆；右 2（真投影＋同中心等比缩放变体——从 1.25 逐级下调
// 找「刚过数学闸」的尺度，实测钉 1.05 仍被截面非圆闸拦、1.04 刚过；Task 4 实测
// 1.15 亦被拦）。winner=真对、残差可区分（>1.5×）非 Ambiguous；loser 过闸（None）
// 但被竞争丢弃，只留诊断
TEST_F(ConicCandidateTest, SameLeftMultiRightBestWins) {
    MarkerSynth synth(rig_, 42);
    const auto m0 = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    std::vector<EllipseFitCPUResult> left = obsVec(m0.left);
    ConicReconstructCPUResult r;
    double chosen = 0.0;
    for (double k : {1.25, 1.15, 1.10, 1.08, 1.06, 1.05, 1.04, 1.03, 1.02}) {
        std::vector<EllipseFitCPUResult> right = obsVec(m0.right, scaleObs(m0.right, k));
        r = op_.Execute(left, right, {-1});
        const bool ok = r.markers.size() == 1u && r.markers[0].rightEllipseIdx == 0 &&
                        r.diagnostics.size() == 2u &&
                        r.diagnostics[0].failReason == ConicFailReason::None &&
                        r.diagnostics[1].failReason == ConicFailReason::None;
        if (ok) { chosen = k; break; }
    }
    ASSERT_GT(chosen, 0.0) << "no variant scale passes math gates (all rejected upstream)";
    std::cout << "chosen variant scale: " << chosen << "\n";
    EXPECT_EQ(r.statistics.candidatePairs, 2u);
    expectCircle(r, m0.truth, 0.15, 0.8, 0.05);      // winner=真对（圆心/法向/半径正确）
    const double rw = r.diagnostics[0].reprojResidualPx;
    const double rl = r.diagnostics[1].reprojResidualPx;
    EXPECT_LT(rw, rl);                               // 真对残差最小
    EXPECT_GT(rl, 1.5 * rw);                         // 次优可区分 → 非 Ambiguous
    EXPECT_FALSE(r.diagnostics[0].droppedByCompetition);   // winner 未落败
    EXPECT_TRUE(r.diagnostics[1].droppedByCompetition);    // loser 置位（failReason 仍 None）
    for (const auto& d : r.diagnostics) {
        EXPECT_NE(d.failReason, ConicFailReason::Ambiguous);
    }
}

// Ambiguous 双解：① 沿基线方位倾角扫描 tilt∈{50..75}×az=0 找「两平面候选都过全闸」
// 位形（数学层双解）；② 扫描找不到 → 改用「同左双右、两残差接近」构造：右目两椭圆
// 均为真投影的 ±0.2% 等比变体（两解残差比 ≤1.5 → winner 判跨候选 Ambiguous）。
// 扫描结论（实测）：tilt 50/55/60/65/70/75 全 None——第二平面候选（鬼影圆）均被
// 半径/cheirality 闸拦下（单幸存放行），全程无「双候选都过全闸」位形（轨迹见运行
// 输出 scan: 行），故由 ②构造触发断言
TEST_F(ConicCandidateTest, AmbiguousDualSolutions) {
    std::ostringstream scanLog;
    bool found = false;
    for (double tilt : {50.0, 55.0, 60.0, 65.0, 70.0, 75.0}) {
        MarkerSynth synth(rig_, 42);
        auto m = synth.make(CircleSpec::pose(tilt, 0.0, 431.0, 3.0));
        auto r = runOne(op_, m);
        scanLog << " tilt" << tilt << "->" << ConicFailReasonName(r.diagnostics[0].failReason);
        if (r.diagnostics[0].failReason == ConicFailReason::Ambiguous) {
            found = true;
            EXPECT_TRUE(r.markers.empty());   // 双幸存禁静默取一
            break;
        }
    }
    if (!found) {
        MarkerSynth synth(rig_, 42);
        const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
        std::vector<EllipseFitCPUResult> left = obsVec(m.left);
        std::vector<EllipseFitCPUResult> right = obsVec(scaleObs(m.right, 1.002),
                                                        scaleObs(m.right, 0.998));
        auto r = op_.Execute(left, right, {-1});
        EXPECT_EQ(r.statistics.candidatePairs, 2u);
        EXPECT_TRUE(r.markers.empty());               // Ambiguous 组不出标记
        ASSERT_EQ(r.diagnostics.size(), 2u);
        int ambCount = 0;
        double res[2] = {-1.0, -1.0};
        for (size_t k = 0; k < 2u; ++k) {
            ambCount += (r.diagnostics[k].failReason == ConicFailReason::Ambiguous) ? 1 : 0;
            res[k] = r.diagnostics[k].reprojResidualPx;
        }
        EXPECT_EQ(ambCount, 1);                       // winner 判 Ambiguous；loser 保留 None
        const double lo = std::min(res[0], res[1]), hi = std::max(res[0], res[1]);
        EXPECT_LE(hi, 1.5 * lo);                      // 两残差接近是触发条件本体
        // 竞争相位确定性见证：同一构造二次 Execute，诊断全字段逐位一致
        // （含 droppedByCompetition——Ambiguous winner 为 false，None loser 为 true）
        auto r2 = op_.Execute(left, right, {-1});
        ASSERT_EQ(r2.diagnostics.size(), r.diagnostics.size());
        for (size_t k = 0; k < r.diagnostics.size(); ++k) {
            const ConicPairDiagnostic& a = r.diagnostics[k];
            const ConicPairDiagnostic& b = r2.diagnostics[k];
            EXPECT_EQ(a.leftIdx, b.leftIdx);
            EXPECT_EQ(a.rightIdx, b.rightIdx);
            EXPECT_EQ(a.failReason, b.failReason);
            EXPECT_EQ(a.reconstructedRadiusMm, b.reconstructedRadiusMm);
            EXPECT_EQ(a.reprojResidualPx, b.reprojResidualPx);
            EXPECT_EQ(a.sectionAxisRatio, b.sectionAxisRatio);
            EXPECT_EQ(a.etaBaseline, b.etaBaseline);
            EXPECT_EQ(a.etaView, b.etaView);
            EXPECT_EQ(a.solutionCount, b.solutionCount);
            EXPECT_EQ(a.droppedByCompetition, b.droppedByCompetition);
        }
        EXPECT_TRUE(r2.markers.empty());
        EXPECT_EQ(r2.statistics.validCount, r.statistics.validCount);
        EXPECT_EQ(r2.statistics.candidatePairs, r.statistics.candidatePairs);
    }
    std::cout << "scan:" << scanLog.str() << "\n";   // 记录扫描轨迹供审计
    SUCCEED();
}

// 候选上限确定性：左右各 9 直造椭圆全交叉（d∈[600,1400] 全落窗、Δv=0）→ 81 预筛对
// 超 cap → 截 64；两轮运行候选序列逐项一致（截断规则：|d−窗中心| 升序＋(i,j) 字典序）
TEST_F(ConicCandidateTest, CandidateCapDeterministic) {
    std::vector<EllipseFitCPUResult> left, right;
    for (int i = 0; i < 9; ++i)
        left.push_back(makeEllipse(1400.0 + 50.0 * i, 763.0, 24.0 + 2.0 * i, 23.0 + 2.0 * i, 7.0 * i));
    for (int j = 0; j < 9; ++j)
        right.push_back(makeEllipse(400.0 + 50.0 * j, 763.0, 22.0 + 2.0 * j, 21.0 + 2.0 * j, 11.0 * j));
    const std::vector<int> noSeeds(9, -1);
    auto r1 = op_.Execute(left, right, noSeeds);
    auto r2 = op_.Execute(left, right, noSeeds);
    EXPECT_EQ(r1.statistics.totalPairs, 0u);
    EXPECT_EQ(r1.statistics.candidatePairs, 64u);
    EXPECT_EQ(r2.statistics.candidatePairs, 64u);
    ASSERT_EQ(r1.diagnostics.size(), 64u);
    ASSERT_EQ(r2.diagnostics.size(), 64u);
    for (size_t k = 0; k < 64u; ++k) {
        EXPECT_EQ(r1.diagnostics[k].leftIdx, r2.diagnostics[k].leftIdx) << "k=" << k;
        EXPECT_EQ(r1.diagnostics[k].rightIdx, r2.diagnostics[k].rightIdx) << "k=" << k;
        EXPECT_EQ(r1.diagnostics[k].failReason, r2.diagnostics[k].failReason) << "k=" << k;
    }
    for (size_t k = 0; k < kFailReasonCount; ++k) {
        EXPECT_EQ(r1.statistics.failReasonHistogram[k], r2.statistics.failReasonHistogram[k]) << "k=" << k;
    }
}

class ConicResultTest : public ::testing::Test {
protected:
    Rig rig_ = Rig::scanRig();
    ConicReconstructCPU op_;

    void SetUp() override { op_.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2); }
};

// 结果组装字段兼容：成功对的 leftEllipseIdx/rightEllipseIdx、36 点合成圆环（3D 落
// 真圆上、两目投影落椭圆 ±2px、reprojError=对级残差）、法向定向 n·C<0、半径、闸旗
TEST_F(ConicResultTest, MarkerFieldsCompatible) {
    MarkerSynth synth(rig_, 42);
    const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    auto r = runOne(op_, m);
    ASSERT_EQ(r.markers.size(), 1u);
    const MarkerReconstructResult& mk = r.markers[0];
    EXPECT_EQ(mk.leftEllipseIdx, 0);
    EXPECT_EQ(mk.rightEllipseIdx, 0);
    ASSERT_EQ(mk.reconstructedPoints.size(), 36u);
    ASSERT_EQ(r.diagnostics.size(), 1u);
    const double pairResid = r.diagnostics[0].reprojResidualPx;
    double maxSampson = 0.0;
    for (const auto& pt : mk.reconstructedPoints) {
        const double dc = std::hypot(std::hypot(pt.x - mk.centerX, pt.y - mk.centerY), pt.z - mk.centerZ);
        EXPECT_NEAR(dc, 3.0, 0.1);                                  // 环点在圆上
        maxSampson = std::max(maxSampson, ellipseSampsonPx(pt.leftU, pt.leftV, m.left));
        maxSampson = std::max(maxSampson, ellipseSampsonPx(pt.rightU, pt.rightV, m.right));
        EXPECT_LE(ellipseSampsonPx(pt.leftU, pt.leftV, m.left), 0.5);   // 实测 max≈2e-6px（精确合成，见下方打印）
        EXPECT_LE(ellipseSampsonPx(pt.rightU, pt.rightV, m.right), 0.5);
        EXPECT_FLOAT_EQ(pt.reprojError, static_cast<float>(pairResid));
    }
    std::cout << "max ring sampson px: " << maxSampson << "\n";
    const double nDotC = mk.normalX * mk.centerX + mk.normalY * mk.centerY + mk.normalZ * mk.centerZ;
    EXPECT_LT(nDotC, 0.0);                                          // 定向：翻向左相机
    EXPECT_NEAR(mk.circleFit.radius, 3.0, 0.05);
    EXPECT_TRUE(mk.validCircle);
    EXPECT_TRUE(mk.validPlane);
    EXPECT_EQ(mk.message, "ok");
}

// 逐帧重注盲区（设计 D6）：rig A（B=131）解一帧→换 rig B（B=90，K 同）→换回 A。
// 像素不变下 rig B 解为 rig A 的等比缩放（Z、r ×90/131）；换回 A 逐位复现首次结果
TEST_F(ConicResultTest, AlternateProjectionInjection) {
    const Rig rigA = Rig::scanRig();
    MarkerSynth synth(rigA, 42);
    const auto m = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    std::vector<EllipseFitCPUResult> left = obsVec(m.left);
    std::vector<EllipseFitCPUResult> right = obsVec(m.right);
    ConicReconstructCPU op;
    op.SetProjectionMatrices(rigA.K1, rigA.K2, rigA.C2);
    auto rA1 = op.Execute(left, right, {0});
    ASSERT_EQ(rA1.markers.size(), 1u);

    Rig rigB;
    rigB.K1 = rigA.K1;
    rigB.K2 = rigA.K2;
    rigB.C2 = cv::Vec3d(90.0, 0.0, 0.0);
    ConicReconstructCPUParams pB;                     // 半径闸随比例换算（r×90/131≈2.061）
    pB.nominalRadiusMm = 3.0 * 90.0 / 131.0;
    op.SetParams(pB);
    op.SetProjectionMatrices(rigB.K1, rigB.K2, rigB.C2);
    auto rB = op.Execute(left, right, {0});
    ASSERT_EQ(rB.markers.size(), 1u);
    EXPECT_NEAR(rB.markers[0].centerZ, 431.0 * 90.0 / 131.0, 1.0);  // ≈296.3
    EXPECT_NEAR(rB.markers[0].circleFit.radius, 3.0 * 90.0 / 131.0, 0.05);
    EXPECT_GT(rA1.markers[0].centerZ, rB.markers[0].centerZ + 100.0);

    op.SetParams(ConicReconstructCPUParams{});        // 参数原样复位
    op.SetProjectionMatrices(rigA.K1, rigA.K2, rigA.C2);
    auto rA2 = op.Execute(left, right, {0});
    ASSERT_EQ(rA2.markers.size(), 1u);
    const MarkerReconstructResult& a1 = rA1.markers[0];
    const MarkerReconstructResult& a2 = rA2.markers[0];
    EXPECT_EQ(a1.centerX, a2.centerX);                // 逐位一致（无状态串扰）
    EXPECT_EQ(a1.centerY, a2.centerY);
    EXPECT_EQ(a1.centerZ, a2.centerZ);
    EXPECT_EQ(a1.normalX, a2.normalX);
    EXPECT_EQ(a1.normalY, a2.normalY);
    EXPECT_EQ(a1.normalZ, a2.normalZ);
    EXPECT_EQ(a1.circleFit.radius, a2.circleFit.radius);
    ASSERT_EQ(a1.reconstructedPoints.size(), a2.reconstructedPoints.size());
    for (size_t k = 0; k < a1.reconstructedPoints.size(); ++k) {
        const ReconstructedPoint3D& p1 = a1.reconstructedPoints[k];
        const ReconstructedPoint3D& p2 = a2.reconstructedPoints[k];
        EXPECT_EQ(p1.x, p2.x);
        EXPECT_EQ(p1.y, p2.y);
        EXPECT_EQ(p1.z, p2.z);
        EXPECT_EQ(p1.leftU, p2.leftU);
        EXPECT_EQ(p1.leftV, p2.leftV);
        EXPECT_EQ(p1.rightU, p2.rightU);
        EXPECT_EQ(p1.rightV, p2.rightV);
        EXPECT_EQ(p1.reprojError, p2.reprojError);
    }
    EXPECT_EQ(rA1.statistics.candidatePairs, rA2.statistics.candidatePairs);
    EXPECT_EQ(rA1.statistics.validCount, rA2.statistics.validCount);
}

// 套件级拒因直方图自检：串跑一组已知构造，failReasonHistogram 须覆盖全部枚举。
// 各触发器依据（实测钉）：None=真对种子；Numerical=中心 NaN（isfinite 前置）；
// Cheirality=Z=−431；Radius=nominal=2.0；Residual=r10/Z200 单轴 ×1.3（放宽截面闸，
// 实测 3.03px>1.5px）；Degeneracy=pose(85,90)；InputClass=minorAxis→0（buildCone
// 拒：参数非正→左上 2×2 奇异）；Coincident=C2=0＋同椭圆；NotCircle=右轴 ×1.15
// （截面轴比 0.035~0.063>0.03）；Inconsistent=右轴 ×1.8；VirtualPlane=右目中心垂
// 直平移 +30px（束顶点 λ̂ 处 λ₁λ₂>0）；PlaneDegenerate=右目像素=左目像素（解平面
// 过左光心 |h|≈0）＋minPlaneDistMm=1500 参数驱动双触发；Ambiguous=同左双右
// ±0.2% 变体（跨候选残差比 ≤1.5）
TEST(ConicSuiteTest, SuiteHistogramSelfCheck) {
    const Rig rig = Rig::scanRig();
    MarkerSynth synth(rig, 42);
    std::array<size_t, kFailReasonCount> hist{};

    auto acc = [&](const ConicReconstructCPUResult& r) {
        for (size_t k = 0; k < kFailReasonCount; ++k)
            hist[k] += r.statistics.failReasonHistogram[k];
    };
    auto runPair = [&](const char* name, ConicReconstructCPU& op,
                       const EllipseObs& lo, const EllipseObs& ro) {
        auto r = op.Execute(obsVec(lo), obsVec(ro), {0});
        std::cout << "[hist] " << name << ":";
        for (const auto& d : r.diagnostics)
            std::cout << " " << ConicFailReasonName(d.failReason);
        std::cout << "\n";
        acc(r);
    };
    auto defaultOp = [&]() {
        auto op = std::make_unique<ConicReconstructCPU>();
        op->SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
        return op;
    };

    const auto m4545 = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    const auto m0 = synth.make(CircleSpec::pose(0, 0, 431, 3.0));
    {   // None：真对种子
        auto op = defaultOp();
        runPair("truePair", *op, m4545.left, m4545.right);
    }
    {   // Numerical：中心 NaN
        auto op = defaultOp();
        EllipseObs l = m4545.left;
        l.cx = std::numeric_limits<double>::quiet_NaN();
        runPair("nanCenter", *op, l, m4545.right);
    }
    {   // Cheirality：Z=−431 左目后方
        auto op = defaultOp();
        const auto m = synth.make(CircleSpec::pose(0, 0, -431, 3.0));
        runPair("behindCamera", *op, m.left, m.right);
    }
    {   // Radius：nominal=2.0 vs 真值 3.0
        ConicReconstructCPUParams p;
        p.nominalRadiusMm = 2.0;
        ConicReconstructCPU op(p);
        op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
        runPair("radiusGate", op, m0.left, m0.right);
    }
    {   // Residual：r=10mm Z=200 右目单轴 ×1.3＋放宽截面闸
        ConicReconstructCPUParams p;
        p.nominalRadiusMm = 10.0;
        p.radiusToleranceRatio = 0.3;
        p.circleAxisRatioMax = 0.5;
        ConicReconstructCPU op(p);
        op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
        const auto m = synth.make(CircleSpec::pose(0, 0, 200, 10.0));
        EllipseObs rr = m.right;
        rr.a *= 1.3;
        runPair("residualGate", op, m.left, rr);
    }
    {   // Degeneracy：pose(85,90)
        auto op = defaultOp();
        const auto m = synth.make(CircleSpec::pose(85, 90, 431, 3.0));
        runPair("degeneracy85", *op, m.left, m.right);
    }
    {   // InputClass：minorAxis→0 退化椭圆
        auto op = defaultOp();
        EllipseObs l = m4545.left;
        l.b = 0.0;
        runPair("degenerateMinor", *op, l, m4545.right);
    }
    {   // Coincident：C2=0＋左右同椭圆
        ConicReconstructCPU op;
        op.SetProjectionMatrices(rig.K1, rig.K1, cv::Vec3d(0.0, 0.0, 0.0));
        runPair("coincidentCones", op, m4545.left, m4545.left);
    }
    {   // NotCircle：右轴整体 ×1.15
        auto op = defaultOp();
        runPair("axesScale115", *op, m4545.left, scaleObs(m4545.right, 1.15));
    }
    {   // Ambiguous：同左双右 ±0.2% 变体
        auto op = defaultOp();
        auto r = op->Execute(obsVec(m4545.left),
                             obsVec(scaleObs(m4545.right, 1.002),
                                    scaleObs(m4545.right, 0.998)), {-1});
        std::cout << "[hist] ambigDual:";
        for (const auto& d : r.diagnostics)
            std::cout << " " << ConicFailReasonName(d.failReason);
        std::cout << "\n";
        acc(r);
    }
    {   // Inconsistent：右轴整体 ×1.8（强错锥）
        auto op = defaultOp();
        runPair("axesScale18", *op, m4545.left, scaleObs(m4545.right, 1.8));
    }
    {   // VirtualPlane：右目中心垂直平移 +30px
        auto op = defaultOp();
        EllipseObs rr = m4545.right;
        rr.cy += 30.0;
        runPair("centerShiftY30", *op, m4545.left, rr);
    }
    {   // PlaneDegenerate①：右目像素=左目像素（d=0）——解平面过左光心 |h|≈0
        auto op = defaultOp();
        runPair("selfPixels", *op, m0.left, m0.left);
    }
    {   // PlaneDegenerate②：minPlaneDistMm=1500 → 真平面 h=431<1500（参数驱动）
        ConicReconstructCPUParams p;
        p.minPlaneDistMm = 1500.0;
        ConicReconstructCPU op(p);
        op.SetProjectionMatrices(rig.K1, rig.K2, rig.C2);
        runPair("planeDistGate", op, m0.left, m0.right);
    }
    {   // NoRealRoot：右目中心垂直微偏 +2px（标定行错位量级——束根复化；与离线工具
        //   --row-offset scan 实证同源。Task 5 探针只扫 30px+ 大错位区得 VirtualPlane，
        //   2px 微错位区由 rowOffsetPx 修正案测试发现）
        auto op = defaultOp();
        EllipseObs rr = m4545.right;
        rr.cy += 2.0;
        runPair("rowShift2px", *op, m4545.left, rr);
    }

    // 覆盖断言：14 个可达枚举各 ≥1。NegativeR(8) 实测不可达（防御性死分支）：
    // ~600 构造探针（含 400 对轴组×角度网格、默认/放宽双闸、错标定注入、同心嵌套
    // 圆、高离心交叉锥、2px 行错位）均未触发——虚圆截面在 lmax>0 守卫处已归
    // NotCircle。断言恒 0 钉住该实证结论；若未来改动使其可达，本断言失败会强制
    // 重审覆盖口径。
    for (size_t k = 0; k < kFailReasonCount; ++k) {
        if (k == static_cast<size_t>(ConicFailReason::NegativeR)) {
            EXPECT_EQ(hist[k], 0u) << ConicFailReasonName(static_cast<ConicFailReason>(k))
                                   << " reported unreachable-dead-branch";
            continue;
        }
        EXPECT_GE(hist[k], 1u) << ConicFailReasonName(static_cast<ConicFailReason>(k))
                               << " (enum " << k << ") not covered";
    }
}

// ============================================================
// 行偏置预修正（批准修正案）：rowOffsetPx 参数
// ============================================================

// 右目 centerY 注入 +2.0px（模拟本数据标定系统性行错位 vR−vL≈+2.12px，离线工具
// --row-offset scan 实证 14/42 对可恢复）→ 默认参数下拒（failReason 实测钉）；
// rowOffsetPx=2.0 精确抵消后恢复（修正为像素域平移，与 f/Z 无关——scanRig 任意
// 位形均可验证，此处取 pose(45,45,431) 既有精度锚位形）
TEST_F(ConicCandidateTest, RowOffsetRecoversShiftedPair) {
    MarkerSynth synth(rig_, 42);
    const auto m = synth.make(CircleSpec::pose(45, 45, 431, 3.0));
    EllipseObs rShift = m.right;
    rShift.cy += 2.0;   // 模拟标定行差
    std::vector<EllipseFitCPUResult> left = obsVec(m.left);
    std::vector<EllipseFitCPUResult> right = obsVec(rShift);
    {
        ConicReconstructCPU op;
        op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
        auto r = op.Execute(left, right, {0});
        ASSERT_EQ(r.diagnostics.size(), 1u);
        const ConicFailReason fr = r.diagnostics[0].failReason;
        std::cout << "rowShift default failReason: " << ConicFailReasonName(fr) << "\n";
        EXPECT_TRUE(fr == ConicFailReason::NoRealRoot || fr == ConicFailReason::NotCircle)
            << "actual failReason=" << ConicFailReasonName(fr);
        EXPECT_TRUE(r.markers.empty());
    }
    {
        ConicReconstructCPUParams p;
        p.rowOffsetPx = 2.0;
        ConicReconstructCPU op(p);
        op.SetProjectionMatrices(rig_.K1, rig_.K2, rig_.C2);
        auto r = op.Execute(left, right, {0});
        ASSERT_EQ(r.markers.size(), 1u) << "rowOffsetPx=2.0 should recover the pair";
        const MarkerReconstructResult& mk = r.markers[0];
        std::cout << "rowShift recover err: dC=" << std::hypot(std::hypot(mk.centerX - m.truth.C(0),
                              mk.centerY - m.truth.C(1)), mk.centerZ - m.truth.C(2))
                  << "mm, dR=" << std::abs(mk.circleFit.radius - m.truth.r) << "mm\n";
        expectCircle(r, m.truth, 0.2, 1.0, 0.05);
        EXPECT_EQ(r.diagnostics[0].failReason, ConicFailReason::None);
        EXPECT_EQ(r.statistics.validCount, 1u);
    }
}
