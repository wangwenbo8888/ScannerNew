/**
 * @file conic_reconstruct_cpu.cpp
 * @brief 双目圆锥相交重建算子 - 实现文件——方案 A 结构化闭式＋GN 抛光
 *
 * 算法流程：
 *   1. 候选生成（设计 §2.0 D4）：centerMatches 种子对必进 ＋ 轻量预筛
 *      （视差窗 fx·B/Z∈[dMin,dMax] ＋ |Δv|≤yPreGatePx），总数 cap kMaxCandidatePairs
 *   2. 逐候选：isfinite 前置后左右椭圆整体参数→归一化锥（A1/A2）
 *   3. 双目锥束（A4/A5）→ 退化成员特征分解拆平面（A6，顶点 λ̂=−b/2a）
 *   4. 截面圆提取（A7，tr/det 闭式特征＋B 式半径）＋GN 抛光（两锥环残差）
 *      → 圆性闸（circleAxisRatioMax，两锥截面取 max）→ 候选圆
 *   5. 闸门链（Task 4）：cheirality → 半径 → Sampson 残差 → 双条件病态
 *      （每闸前 isfinite；灰区 ±10% 带 → 整批 Degraded；双幸存 Ambiguous）
 *   6. 同左多右竞争（设计 §2.7，Task 5）：过闸解按 leftIdx 分组取残差最小者；
 *      次优 ≤ kDistinctResidualFactor×最优 → 该 leftIdx 判 Ambiguous
 *   7. 结果组装：MarkerReconstructResult（法向定向 n·C<0＋合成圆环）＋逐对诊断
 */

#include "conic_reconstruct_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>

using namespace calib;

OperatorInfo getConicReconstructCPUInfo() {
    return OperatorInfo{"ConicReconstructCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(13, ConicReconstructCPU);

void ConicReconstructCPUParams::validate() const {
    if (nominalRadiusMm <= 0.0) throw std::invalid_argument("nominalRadiusMm must be > 0");
    if (radiusToleranceRatio <= 0.0 || radiusToleranceRatio >= 1.0) throw std::invalid_argument("radiusToleranceRatio must be in (0,1)");
    if (maxReprojResidualPx <= 0.0) throw std::invalid_argument("maxReprojResidualPx must be > 0");
    if (consistencyMax <= 0.0 || consistencyMax >= 1.0) throw std::invalid_argument("consistencyMax must be in (0,1)");
    if (circleAxisRatioMax <= 0.0 || circleAxisRatioMax >= 1.0) throw std::invalid_argument("circleAxisRatioMax must be in (0,1)");
    if (degeneracyGateAngleDeg <= 0.0 || degeneracyGateAngleDeg >= 45.0) throw std::invalid_argument("degeneracyGateAngleDeg must be in (0,45)");
    if (depthMinMm <= 0.0 || depthMaxMm <= depthMinMm) throw std::invalid_argument("depth window invalid");
    if (minPlaneDistMm < 0.0) throw std::invalid_argument("minPlaneDistMm must be >= 0");
    if (yPreGatePx <= 0.0) throw std::invalid_argument("yPreGatePx must be > 0");
    if (std::abs(rowOffsetPx) > 50.0) throw std::invalid_argument("rowOffsetPx must be within ±50px");
    if (ringPointsForViz < 0) throw std::invalid_argument("ringPointsForViz must be >= 0");
}

nlohmann::json ConicReconstructCPUParams::toJson() const {
    return {
        {"nominalRadiusMm", nominalRadiusMm}, {"radiusToleranceRatio", radiusToleranceRatio},
        {"maxReprojResidualPx", maxReprojResidualPx}, {"consistencyMax", consistencyMax},
        {"circleAxisRatioMax", circleAxisRatioMax}, {"degeneracyGateAngleDeg", degeneracyGateAngleDeg},
        {"depthMinMm", depthMinMm}, {"depthMaxMm", depthMaxMm}, {"minPlaneDistMm", minPlaneDistMm},
        {"yPreGatePx", yPreGatePx}, {"rowOffsetPx", rowOffsetPx}, {"ringPointsForViz", ringPointsForViz},
        {"collectStatistics", collectStatistics}};
}

ConicReconstructCPUParams ConicReconstructCPUParams::fromJson(const nlohmann::json& j) {
    ConicReconstructCPUParams p;
    if (j.contains("nominalRadiusMm")) p.nominalRadiusMm = j.at("nominalRadiusMm").get<double>();
    if (j.contains("radiusToleranceRatio")) p.radiusToleranceRatio = j.at("radiusToleranceRatio").get<double>();
    if (j.contains("maxReprojResidualPx")) p.maxReprojResidualPx = j.at("maxReprojResidualPx").get<double>();
    if (j.contains("consistencyMax")) p.consistencyMax = j.at("consistencyMax").get<double>();
    if (j.contains("circleAxisRatioMax")) p.circleAxisRatioMax = j.at("circleAxisRatioMax").get<double>();
    if (j.contains("degeneracyGateAngleDeg")) p.degeneracyGateAngleDeg = j.at("degeneracyGateAngleDeg").get<double>();
    if (j.contains("depthMinMm")) p.depthMinMm = j.at("depthMinMm").get<double>();
    if (j.contains("depthMaxMm")) p.depthMaxMm = j.at("depthMaxMm").get<double>();
    if (j.contains("minPlaneDistMm")) p.minPlaneDistMm = j.at("minPlaneDistMm").get<double>();
    if (j.contains("yPreGatePx")) p.yPreGatePx = j.at("yPreGatePx").get<double>();
    if (j.contains("rowOffsetPx")) p.rowOffsetPx = j.at("rowOffsetPx").get<double>();
    if (j.contains("ringPointsForViz")) p.ringPointsForViz = j.at("ringPointsForViz").get<int>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    return p;
}

// ============================================================
// 纯函数层（匿名 namespace）：锥面 / 束 / 平面拆分 / 截面圆
// 尺度口径：全部几何量以 B=‖C2‖ 为长度单位解算，末端 ×B 回 mm
// ============================================================
namespace {

// ---------- A1 椭圆参数 → 像素系点锥 xᵀQx=0 ----------
// a,b=全轴长（diameter），a 沿 angDeg 方向（cv::fitEllipse/RotatedRect 约定，
// width 沿 angle 方向；a/b 无大小序，公式天然处理任意序）
// ⚠ 勘误：规格原文 ia2=1/a² 未做直径→半轴换算，会把椭圆整体放大 2×（FrontoParallel
//   实测 r=6.0mm 的整 2× 特征证实）；此处换算到半轴 sa=a/2 后套用同一公式
cv::Matx33d ellipseToConic(double cx, double cy, double a, double b, double angDeg) {
    const double sa = 0.5 * a, sb = 0.5 * b;
    const double rad = angDeg * CV_PI / 180.0;
    const double cs = std::cos(rad), sn = std::sin(rad);
    const double ia2 = 1.0 / (sa * sa), ib2 = 1.0 / (sb * sb);
    const double A = cs * cs * ia2 + sn * sn * ib2;
    const double Bc = 2.0 * cs * sn * (ia2 - ib2);
    const double Cc = sn * sn * ia2 + cs * cs * ib2;
    const double D = -(A * cx + 0.5 * Bc * cy);
    const double E = -(0.5 * Bc * cx + Cc * cy);
    const double F = A * cx * cx + Bc * cx * cy + Cc * cy * cy - 1.0;
    return cv::Matx33d(A, 0.5 * Bc, D,
                       0.5 * Bc, Cc, E,
                       D, E, F);
}

// ---------- A2 椭圆 → 归一化锥 ----------
// Qn = Kᵀ·Qpix·K（方向：像点 x=K·x_n ⇒ 锥 Q_n=KᵀQK，禁反）
// 失败（秩亏 / 左上 2×2 非正定 / 参数非有限）→ false（调用方判 InputClass）
// QpixOut：可选输出像素系锥矩阵（Task 4 残差闸 Sampson 距离复用）
bool buildCone(const EllipseFitCPUResult& e, const cv::Matx33d& K, cv::Matx33d& Pi,
               cv::Matx33d* QpixOut = nullptr) {
    if (!(e.majorAxis > 0.0) || !(e.minorAxis > 0.0) ||
        !std::isfinite(e.majorAxis) || !std::isfinite(e.minorAxis) ||
        !std::isfinite(e.centerX) || !std::isfinite(e.centerY)) {
        return false;
    }
    const cv::Matx33d Qpix = ellipseToConic(e.centerX, e.centerY, e.majorAxis, e.minorAxis, e.angle);
    if (QpixOut) *QpixOut = Qpix;
    cv::Matx33d Qn = K.t() * Qpix * K;
    if (Qn(0, 0) + Qn(1, 1) < 0.0) Qn = -Qn;   // 符号规范
    const double fro = cv::norm(Qn);           // Frobenius
    if (std::abs(cv::determinant(Qn)) < 1e-12 * fro * fro * fro) return false;  // det≈0（标度）
    const double det2 = Qn(0, 0) * Qn(1, 1) - Qn(0, 1) * Qn(0, 1);
    if (Qn(0, 0) <= 0.0 || Qn(1, 1) <= 0.0 || det2 <= 0.0) return false;       // 左上 2×2 正定
    Pi = Qn * (1.0 / fro);
    return true;
}

// 3×3 伴随矩阵（cofactor 转置；det=0 时仍有效——束求值需要）
cv::Matx33d adjugate3x3(const cv::Matx33d& S) {
    const double a00 = (S(1, 1) * S(2, 2) - S(1, 2) * S(2, 1));
    const double a01 = -(S(0, 1) * S(2, 2) - S(0, 2) * S(2, 1));
    const double a02 = (S(0, 1) * S(1, 2) - S(0, 2) * S(1, 1));
    const double a10 = -(S(1, 0) * S(2, 2) - S(1, 2) * S(2, 0));
    const double a11 = (S(0, 0) * S(2, 2) - S(0, 2) * S(2, 0));
    const double a12 = -(S(0, 0) * S(1, 2) - S(0, 2) * S(1, 0));
    const double a20 = (S(1, 0) * S(2, 1) - S(1, 1) * S(2, 0));
    const double a21 = -(S(0, 0) * S(2, 1) - S(0, 1) * S(2, 0));
    const double a22 = (S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0));
    return cv::Matx33d(a00, a01, a02, a10, a11, a12, a20, a21, a22);
}

// ---------- A5 束多项式 g(λ) 三点求值 + 精确二次拟合 ----------
// p(λ)=det(M1−λM2)=λ·g(λ)，λ=0 为结构根（左锥自身秩亏）丢弃
// g(λ) = −[ κ·det(Pi1−λPi2) + λ·wᵀ·adj(Pi1−λPi2)·w ]，κ=CᵀPi2C，w=Pi2·C
struct QuadABC { double a = 0.0, b = 0.0, c = 0.0; };

QuadABC fitPencilQuadratic(const cv::Matx33d& Pi1, const cv::Matx33d& Pi2,
                           const cv::Vec3d& w, double kappa) {
    auto g = [&](double lam) {
        const cv::Matx33d S = Pi1 - lam * Pi2;
        return -(kappa * cv::determinant(S) + lam * w.dot(adjugate3x3(S) * w));
    };
    const double g0 = g(0.0), gp = g(1.0), gm = g(-1.0);
    QuadABC q;
    q.c = g0;
    q.a = 0.5 * (gp + gm) - g0;
    q.b = 0.5 * (gp - gm);
    return q;
}

// ---------- A7 截面圆提取（两锥，B 单位） ----------
struct CircleB {
    cv::Vec3d C{0, 0, 0};   // 圆心（B 单位）
    cv::Vec3d n{0, 0, 0};   // 单位法向（定向后）
    double r = 0.0;         // 半径（B 单位）
};

enum class PlaneOutcome { Circle, SkipPlaneDegenerate, SkipNotCircle, SkipNegativeR };

// 平面内正交基（确定性规则选种子轴）：e1,e2 ⊥ n̂ 且互相正交
void planeBasis(const cv::Vec3d& nhat, cv::Vec3d& e1, cv::Vec3d& e2) {
    const cv::Vec3d axes[3] = {cv::Vec3d(1, 0, 0), cv::Vec3d(0, 1, 0), cv::Vec3d(0, 0, 1)};
    int seedIdx = 0;
    double best = 2.0;
    for (int k = 0; k < 3; ++k) {
        const double d = std::abs(nhat.dot(axes[k]));
        if (d < best) { best = d; seedIdx = k; }
    }
    e1 = axes[seedIdx] - nhat.dot(axes[seedIdx]) * nhat;
    e1 /= cv::norm(e1);
    e2 = nhat.cross(e1);
}

// 平面 π（n̂,h）截锥 Π 的 2×2 截面矩阵参数（n̂ 须为单位向量）
void sectionMatrix(const cv::Vec3d& nhat, double h, const cv::Matx33d& Pi,
                   double& a11, double& a12, double& a22,
                   double& b1, double& b2, cv::Vec3d& X0) {
    cv::Vec3d e1, e2;
    planeBasis(nhat, e1, e2);
    X0 = h * nhat;
    a11 = e1.dot(Pi * e1);
    a12 = e1.dot(Pi * e2);
    a22 = e2.dot(Pi * e2);
    b1 = e1.dot(Pi * X0);
    b2 = e2.dot(Pi * X0);
}

// 圆面内切向基：u,v ⊥ n 且互相正交（确定性规则，抛光残差/雅可比共用）
void circleTangentBasis(const cv::Vec3d& n, cv::Vec3d& u, cv::Vec3d& v) {
    cv::Vec3d ax(1, 0, 0);
    if (std::abs(n(0)) > 0.9) ax = cv::Vec3d(0, 1, 0);
    u = n.cross(ax); u /= cv::norm(u);
    v = n.cross(u);
}

// ⚠ 规格外补强（偏差 3，详见报告）：顶点解 GN 抛光。
// 顶点束路径的平面提取对锥系数噪声呈 √ε 抖动（0.1px 椭圆噪声 → 平面 ~1°，
// 截面轴比被抬过 0.03 闸，接受率 60%）；而两锥所含真圆信息是 O(ε) 完整的
// （God 验证：对两噪声锥直接拟合最优圆 dr≤5e-6mm）。此处以候选圆 8 环点对
// 两锥的一阶距离残差（f=XᵀΠX/‖ΠX‖，B 单位，两锥同量纲）做 kPolishIterations
// 步高斯-牛顿（数值雅可比，确定性），恢复 O(ε) 精度。闸门值不动——轴比闸改
// 在抛光后平面上评（设计 §7.3「闸不吃好点」），且对两锥截面取 max（真圆两目
// 截面皆圆，单目圆不足以放行）。
// ⚠ 残差口径勘误（Task 4 实测）：Frobenius 归一锥的原始代数残差 XᵀΠX 在两锥
//   间尺度失衡——倾斜位形下优化器过度满足左锥（截面比 ~0.003）而牺牲右锥
//   （~0.035>0.03 闸），tilt60+0.1px 接受率仅 14/30；距离口径拉平折衷后 29/30
constexpr int kPolishIterations = 3;        // GN 步数上限
constexpr double kPolishTrustRegion = 0.02; // ‖dx‖∞ 上限：平移 0.02B≈2.6mm / 转 0.02rad≈1.1°（真圆修正量级；
                                            // 无此约束时幻影候选会被拉向「锥交线上 r→0 退化圆」伪极小）
constexpr double kPolishDivergenceBound = 1e3;  // 原始步长超过此值视为发散

bool polishCircle(cv::Vec3d& Cc, cv::Vec3d& nhat, double& r,
                  const cv::Matx33d& Pi1, const cv::Matx33d& Pi2, const cv::Vec3d& C2u) {
    // 残差口径：f = XᵀΠX/‖ΠX‖（一阶锥面距离，B 单位）——两锥同量纲。
    // ⚠ 若用原始代数值 XᵀΠX（Frobenius 归一锥），两锥残差尺度不均：倾斜位形下
    //   优化器会过度满足左锥（截面比 ~0.003）而牺牲右锥（~0.035 > 0.03 闸），
    //   tilt60+0.1px 接受率掉到 47%；距离口径把折衷拉平（两截面比同降入闸内）
    auto residuals = [&](const cv::Vec3d& C, const cv::Vec3d& n, double rr, double* out16) {
        cv::Vec3d u, v;
        circleTangentBasis(n, u, v);
        for (int i = 0; i < 8; ++i) {
            const double th = CV_PI * 0.25 * i;
            const cv::Vec3d X = C + rr * (std::cos(th) * u + std::sin(th) * v);
            const cv::Vec3d Xr = X - C2u;
            const cv::Vec3d g1 = Pi1 * X;
            const cv::Vec3d g2 = Pi2 * Xr;
            out16[2 * i] = X.dot(g1) / std::max(1e-12, cv::norm(g1));
            out16[2 * i + 1] = Xr.dot(g2) / std::max(1e-12, cv::norm(g2));
        }
    };

    const double hStep = 1e-6;
    double f0[16], fp[16];
    double cost0 = 0;
    residuals(Cc, nhat, r, f0);
    for (int k = 0; k < 16; ++k) cost0 += f0[k] * f0[k];
    for (int iter = 0; iter < kPolishIterations; ++iter) {
        residuals(Cc, nhat, r, f0);
        cv::Vec3d du, dv;
        circleTangentBasis(nhat, du, dv);
        double J[16][6];
        for (int c = 0; c < 6; ++c) {
            cv::Vec3d Cp = Cc, np = nhat;
            double rp = r;
            if (c < 3) {
                const cv::Vec3d dvec(c == 0 ? 1.0 : 0.0, c == 1 ? 1.0 : 0.0, c == 2 ? 1.0 : 0.0);
                Cp = Cc + hStep * dvec;
            } else if (c == 3) {
                np = nhat + hStep * du; np /= cv::norm(np);
            } else if (c == 4) {
                np = nhat + hStep * dv; np /= cv::norm(np);
            } else {
                rp = r + hStep;
            }
            residuals(Cp, np, rp, fp);
            for (int k = 0; k < 16; ++k) J[k][c] = (fp[k] - f0[k]) / hStep;
        }
        // 正规方程 6×6（高斯消元）
        double A[6][7];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                double s = 0;
                for (int k = 0; k < 16; ++k) s += J[k][i] * J[k][j];
                A[i][j] = s;
            }
            double g = 0;
            for (int k = 0; k < 16; ++k) g -= J[k][i] * f0[k];
            A[i][6] = g;
        }
        for (int i = 0; i < 6; ++i) {
            int p = i;
            for (int k = i + 1; k < 6; ++k)
                if (std::abs(A[k][i]) > std::abs(A[p][i])) p = k;
            if (!(std::abs(A[p][i]) > 0.0)) return false;
            if (p != i) for (int j = 0; j < 7; ++j) std::swap(A[i][j], A[p][j]);
            for (int k = i + 1; k < 6; ++k) {
                const double w = A[k][i] / A[i][i];
                for (int j = i; j < 7; ++j) A[k][j] -= w * A[i][j];
            }
        }
        double dx[6];
        for (int i = 5; i >= 0; --i) {
            double s = A[i][6];
            for (int j = i + 1; j < 6; ++j) s -= A[i][j] * dx[j];
            dx[i] = s / A[i][i];
        }
        // 信赖域：‖dx‖∞ ≤ kPolishTrustRegion。
        // 无此约束时幻影候选会被 GN 拉向「锥交线上半径→0 的退化圆」（残差趋零的伪极小）
        double dxMax = 0.0;
        for (int i = 0; i < 6; ++i) {
            if (!std::isfinite(dx[i])) return false;   // NaN/Inf → 弃抛光
            dxMax = std::max(dxMax, std::abs(dx[i]));
        }
        if (dxMax > kPolishDivergenceBound) return false;
        if (dxMax > kPolishTrustRegion) {
            const double s = kPolishTrustRegion / dxMax;
            for (int i = 0; i < 6; ++i) dx[i] *= s;
        }
        Cc = Cc + cv::Vec3d(dx[0], dx[1], dx[2]);
        nhat = nhat + dx[3] * du + dx[4] * dv;
        nhat /= cv::norm(nhat);
        r += dx[5];
        if (!(r > 0.0)) return false;
    }
    // 终验：抛光不降代价（含 NaN→比较为 false）则弃用
    double cost1 = 0;
    residuals(Cc, nhat, r, fp);
    for (int k = 0; k < 16; ++k) cost1 += fp[k] * fp[k];
    if (!(cost1 <= cost0)) return false;
    return cv::norm(Cc) > 0.0 && std::isfinite(r);
}

PlaneOutcome extractCircle(const cv::Vec4d& pi, const cv::Matx33d& Pi1, const cv::Matx33d& Pi2,
                           const cv::Vec3d& C2u, const ConicReconstructCPUParams& params,
                           double B, CircleB& out, double* axisRatioOut = nullptr) {
    if (axisRatioOut) *axisRatioOut = -1.0;   // 默认未达（早退路径不评）
    const cv::Vec3d n(pi(0), pi(1), pi(2));
    const double nn = cv::norm(n);
    if (!(nn >= 1e-12)) return PlaneOutcome::SkipPlaneDegenerate;
    cv::Vec3d nhat = n / nn;
    double h = -pi(3) / nn;   // 原点到平面距离（B 单位）
    if (std::abs(h) * B < params.minPlaneDistMm) return PlaneOutcome::SkipPlaneDegenerate;

    double a11, a12, a22, b1, b2;
    cv::Vec3d X0;
    sectionMatrix(nhat, h, Pi1, a11, a12, a22, b1, b2, X0);
    const double tr = a11 + a22;
    const double dt = a11 * a22 - a12 * a12;
    const double root = std::sqrt(std::max(0.0, tr * tr - 4.0 * dt));   // tr/det 闭式特征
    const double lmax = 0.5 * (tr + root);
    const double lmin = 0.5 * (tr - root);
    if (!(lmax > 0.0)) return PlaneOutcome::SkipNotCircle;
    if (std::abs(dt) < 1e-12 * lmax * lmax) return PlaneOutcome::SkipNotCircle;
    // s0 = −A2x2⁻¹·b2vec；粗解（B 式）作抛光初值
    const double s0x = -(a22 * b1 - a12 * b2) / dt;
    const double s0y = -(-a12 * b1 + a11 * b2) / dt;
    // r² = ‖s0‖² − c2/λmax 的等价重排：r² = −(CcᵀΠCc)/λ̄（λ̄=tr/2）。
    // ⚠ 数值勘误：原式在偏置截面（‖s₀‖≫r，如倾斜圆）呈大数相消，放大椭圆轴误差
    //   ~10⁴×；重排式与原式在精确圆截面处恒等，且对截面基旋转迹不变
    cv::Vec3d e1, e2;
    planeBasis(nhat, e1, e2);
    cv::Vec3d Cc = X0 + s0x * e1 + s0y * e2;
    const double lamBar = 0.5 * tr;
    double r2 = -Cc.dot(Pi1 * Cc) / lamBar;
    if (!(r2 >= 0.0)) return PlaneOutcome::SkipNegativeR;
    double r = std::sqrt(r2);

    // GN 抛光（失败保留粗解）
    cv::Vec3d CcP = Cc, nP = nhat;
    double rP = r;
    if (polishCircle(CcP, nP, rP, Pi1, Pi2, C2u)) {
        Cc = CcP;
        nhat = nP;
        r = rP;
        h = nhat.dot(Cc);   // 抛光平面：过 Cc、法向 n̂
    }

    // 抛光后平面重评闸门：平面距离＋两锥截面轴比（取 max）
    if (std::abs(h) * B < params.minPlaneDistMm) return PlaneOutcome::SkipPlaneDegenerate;
    double a11b, a12b, a22b, b1b, b2b;
    cv::Vec3d X0b;
    sectionMatrix(nhat, h, Pi1, a11b, a12b, a22b, b1b, b2b, X0b);
    const double tr1 = a11b + a22b;
    const double dt1 = a11b * a22b - a12b * a12b;
    const double root1 = std::sqrt(std::max(0.0, tr1 * tr1 - 4.0 * dt1));
    const double lmax1 = 0.5 * (tr1 + root1);
    const double lmin1 = 0.5 * (tr1 - root1);
    double a11r, a12r, a22r, b1r, b2r;
    cv::Vec3d X0r;
    sectionMatrix(nhat, h, Pi2, a11r, a12r, a22r, b1r, b2r, X0r);
    const double tr2 = a11r + a22r;
    const double dt2 = a11r * a22r - a12r * a12r;
    const double root2 = std::sqrt(std::max(0.0, tr2 * tr2 - 4.0 * dt2));
    const double lmax2 = 0.5 * (tr2 + root2);
    const double lmin2 = 0.5 * (tr2 - root2);
    if (!(lmax1 > 0.0) || !(lmax2 > 0.0)) return PlaneOutcome::SkipNotCircle;
    const double ratio1 = (lmax1 - lmin1) / lmax1;
    const double ratio2 = (lmax2 - lmin2) / lmax2;
    if (axisRatioOut) *axisRatioOut = std::max(ratio1, ratio2);   // 闸口径轴比（过/拒均填，诊断消费）
    if (std::max(ratio1, ratio2) > params.circleAxisRatioMax) return PlaneOutcome::SkipNotCircle;  // 轴比闸

    // 抛光平面上的最终截面（Π₁，B 式）
    const double dtF = dt1;
    if (std::abs(dtF) < 1e-12 * lmax1 * lmax1) return PlaneOutcome::SkipNotCircle;
    const double s0xF = -(a22b * b1b - a12b * b2b) / dtF;
    const double s0yF = -(-a12b * b1b + a11b * b2b) / dtF;
    cv::Vec3d e1F, e2F;
    planeBasis(nhat, e1F, e2F);
    const cv::Vec3d Cf = X0b + s0xF * e1F + s0yF * e2F;
    const double r2F = -Cf.dot(Pi1 * Cf) / (0.5 * tr1);
    if (!(r2F >= 0.0)) return PlaneOutcome::SkipNegativeR;   // 负 r²（含 NaN）不许 clamp
    out.r = std::sqrt(r2F);
    out.C = Cf;
    out.n = nhat;
    if (out.n.dot(out.C) < 0.0) out.n = -out.n;             // 定向：n̂·C_B ≥ 0
    return PlaneOutcome::Circle;
}

// ============================================================
// Task 4 闸门链（设计 §2.6）：cheirality → radius → residual → degeneracy
// 每道闸前对涉及输入量做 isfinite 检查（NaN 在 ratio>τ 比较下静默放行是已识别
// 漏洞）→ Numerical；过闸但余量落在阈值 ±10% 带内 → 灰区（整批 Degraded）
// ============================================================

struct GateTrace {
    ConicFailReason failReason = ConicFailReason::None;
    double reprojResidualPx = -1.0;   // max(medianL,medianR)；到达残差闸即填（拒也填，审计）
    bool grayZone = false;            // 任一闸余量在阈值 ±10% 带内（过闸侧）
};

bool vecFinite(const cv::Vec3d& v) {
    return std::isfinite(v(0)) && std::isfinite(v(1)) && std::isfinite(v(2));
}

// 椭圆整体参数 isfinite 前置（NaN/Inf → Numerical，先于 buildCone 的 InputClass）
bool ellipseFinite(const EllipseFitCPUResult& e) {
    return std::isfinite(e.centerX) && std::isfinite(e.centerY) &&
           std::isfinite(e.majorAxis) && std::isfinite(e.minorAxis) &&
           std::isfinite(e.angle);
}

// 中位数（空集 → 0.0；偶数个取中间两元均值）；确定性
double medianOf(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t hi = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(hi), v.end());
    double m = v[hi];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(hi - 1), v.end());
        m = 0.5 * (m + v[hi - 1]);
    }
    return m;
}

// 齐次像素点 x=(u,v,1) 到像素系锥 Qpix 的一阶（Sampson 型）距离 px：
// d = |xᵀQx| / ‖(2·Q·x)的前两分量‖；梯度范数 <1e-9 的点跳过（不计入中位数）；
// 返回 false = 数值坏点（d 非有限 → Numerical）
bool sampsonPx(const cv::Vec3d& xh, const cv::Matx33d& Qpix, std::vector<double>& out) {
    const double num = std::abs(xh.dot(Qpix * xh));
    const cv::Vec3d g = 2.0 * (Qpix * xh);
    const double gn = std::hypot(g(0), g(1));
    if (gn < 1e-9) return true;                 // 梯度守卫：跳过该点
    const double d = num / gn;
    if (!std::isfinite(d)) return false;
    out.push_back(d);
    return true;
}

// 闸门链本体。circle 为 B 单位；bhat=单位基线（左光心→右光心）；B=基线长 mm；
// K1/K2 像素内参；Qpix1/Qpix2 像素系锥（buildCone 产出）。
// 返回 false 时 tr.failReason=四拒因之一；tr.reprojResidualPx/grayZone 供诊断。
bool gateChain(const CircleB& c, const cv::Matx33d& Qpix1, const cv::Matx33d& Qpix2,
               const cv::Matx33d& K1, const cv::Matx33d& K2, const cv::Vec3d& bhat,
               double B, const ConicReconstructCPUParams& params, GateTrace& tr) {
    // ---- 前置：圆参数 isfinite（后续所有闸的公共输入）----
    if (!vecFinite(c.C) || !vecFinite(c.n) || !std::isfinite(c.r)) {
        tr.failReason = ConicFailReason::Numerical;
        return false;
    }
    cv::Vec3d e1, e2;
    circleTangentBasis(c.n, e1, e2);   // 圆平面正交基（与抛光残差同一口径）

    // ---- ① cheirality：圆上 8 采样点（45° 间隔，θ=0,45°,…,315°）两目深度 Z ∈ [depthMin,depthMax] ----
    for (int i = 0; i < 8; ++i) {
        const double th = CV_PI * 0.25 * i;
        const cv::Vec3d X = c.C + c.r * (std::cos(th) * e1 + std::sin(th) * e2);   // B 单位
        const cv::Vec3d Xmm = X * B;                                                // mm
        const cv::Vec3d Xr = Xmm - bhat * B;                                        // 右目系 mm
        if (!vecFinite(Xmm) || !vecFinite(Xr)) {
            tr.failReason = ConicFailReason::Numerical;
            return false;
        }
        if (!(Xmm(2) >= params.depthMinMm && Xmm(2) <= params.depthMaxMm) ||
            !(Xr(2) >= params.depthMinMm && Xr(2) <= params.depthMaxMm)) {
            tr.failReason = ConicFailReason::Cheirality;
            return false;
        }
    }

    // ---- ② 半径闸：|r_mm − nominal|/nominal ≤ radiusToleranceRatio ----
    const double rMm = c.r * B;
    const double dev = std::abs(rMm - params.nominalRadiusMm) / params.nominalRadiusMm;
    if (!(dev <= params.radiusToleranceRatio)) {
        tr.failReason = ConicFailReason::Radius;
        return false;
    }
    tr.grayZone = tr.grayZone || dev >= 0.9 * params.radiusToleranceRatio;

    // ---- ③ Sampson 残差闸：圆上 16 采样点（θ=k·22.5°）投影两目，max(median) ≤ maxReprojResidualPx ----
    std::vector<double> dl, dr;
    dl.reserve(16);
    dr.reserve(16);
    for (int k = 0; k < 16; ++k) {
        const double th = CV_PI * 0.125 * k;
        const cv::Vec3d X = c.C + c.r * (std::cos(th) * e1 + std::sin(th) * e2);
        const cv::Vec3d Xmm = X * B;
        const cv::Vec3d xl = K1 * Xmm;                 // 左目齐次像素
        const cv::Vec3d xr = K2 * (Xmm - bhat * B);    // 右目齐次像素
        if (!vecFinite(xl) || !vecFinite(xr) || xl(2) == 0.0 || xr(2) == 0.0) {
            tr.failReason = ConicFailReason::Numerical;
            return false;
        }
        if (!sampsonPx(cv::Vec3d(xl(0) / xl(2), xl(1) / xl(2), 1.0), Qpix1, dl) ||
            !sampsonPx(cv::Vec3d(xr(0) / xr(2), xr(1) / xr(2), 1.0), Qpix2, dr)) {
            tr.failReason = ConicFailReason::Numerical;
            return false;
        }
    }
    if (dl.empty() || dr.empty()) {   // 全点被梯度守卫跳过 → 残差不可评（medianOf 空集封死）
        tr.failReason = ConicFailReason::Numerical;
        return false;
    }
    tr.reprojResidualPx = std::max(medianOf(dl), medianOf(dr));   // 拒也填（审计）
    if (!(tr.reprojResidualPx <= params.maxReprojResidualPx)) {
        tr.failReason = ConicFailReason::Residual;
        return false;
    }
    tr.grayZone = tr.grayZone || tr.reprojResidualPx >= 0.9 * params.maxReprojResidualPx;

    // ---- ④ 双条件病态闸：etaBaseline<sinθg 且 etaView<sinθg 才拒 ----
    // （η 单指标会误拒正对位形——正对 n̂⊥基线但 n̂∥视线：etaBaseline≈0、etaView≈1 → 放行）
    const double etaBaseline = std::abs(c.n.dot(bhat));
    const double cn = cv::norm(c.C);
    const double etaView = cn > 0.0 ? std::abs(c.C.dot(c.n)) / cn : 0.0;
    if (!std::isfinite(etaBaseline) || !std::isfinite(etaView)) {
        tr.failReason = ConicFailReason::Numerical;
        return false;
    }
    const double sinGate = std::sin(params.degeneracyGateAngleDeg * CV_PI / 180.0);
    if (etaBaseline < sinGate && etaView < sinGate) {
        tr.failReason = ConicFailReason::Degeneracy;
        return false;
    }
    const double margin = std::max(etaBaseline, etaView);   // 拒 ⟺ margin < sinGate
    tr.grayZone = tr.grayZone || margin * 0.9 < sinGate;    // ⟺ margin < sinGate/0.9（放行余量 ∈ [1,1/0.9)·sinGate 带内）

    return true;
}

// ---------- solvePair：束判据 + 退化成员拆平面 + 截面圆 + 闸门链 ----------
struct SolveOutcome {
    ConicFailReason failReason = ConicFailReason::Numerical;
    int solutionCount = 0;
    CircleB circle;                  // 首候选（nCand≥1 时，B 单位；成功时=幸存者）
    bool grayZone = false;           // 幸存候选任一闸余量在阈值 ±10% 带内
    double reprojResidualPx = -1.0;  // max(medianL,medianR)（拒也填，审计）
    double sectionAxisRatio = -1.0;  // 幸存者/末次评估候选的截面轴比（口径同 reprojResidualPx）
};

SolveOutcome solvePair(const cv::Matx33d& Pi1, const cv::Matx33d& Pi2,
                       const cv::Matx33d& Qpix1, const cv::Matx33d& Qpix2,
                       const cv::Matx33d& K1, const cv::Matx33d& K2,
                       const cv::Vec3d& C, double B, const ConicReconstructCPUParams& params) {
    SolveOutcome out;
    // ---- A4 4×4 锥面（C=C2/B 单位向量；分块 −Pi2·C2 与 B 单位口径一致）----
    const cv::Vec3d w = Pi2 * C;
    const double kappa = C.dot(w);
    const cv::Matx44d M1(
        Pi1(0, 0), Pi1(0, 1), Pi1(0, 2), 0.0,
        Pi1(1, 0), Pi1(1, 1), Pi1(1, 2), 0.0,
        Pi1(2, 0), Pi1(2, 1), Pi1(2, 2), 0.0,
        0.0, 0.0, 0.0, 0.0);
    const cv::Matx44d M2(
        Pi2(0, 0), Pi2(0, 1), Pi2(0, 2), -w(0),
        Pi2(1, 0), Pi2(1, 1), Pi2(1, 2), -w(1),
        Pi2(2, 0), Pi2(2, 1), Pi2(2, 2), -w(2),
        -w(0), -w(1), -w(2), kappa);
    // ---- A5 束判据 ----
    const QuadABC q = fitPencilQuadratic(Pi1, Pi2, w, kappa);
    const double qInf = std::max(std::abs(q.a), std::max(std::abs(q.b), std::abs(q.c)));
    if (qInf < 1e-12) {
        out.failReason = ConicFailReason::Coincident;   // 两锥重合
        return out;
    }
    if (std::abs(q.a) < 1e-12 * std::max(std::abs(q.b), std::abs(q.c))) {
        // 束退化（g 近线性，仅一个有限退化成员）。取 Coincident：
        // 近重合锥是该形态最常见的物理诱因；NoRealRoot 会误报实根缺席
        out.failReason = ConicFailReason::Coincident;
        return out;
    }
    double disc = q.b * q.b - 4.0 * q.a * q.c;
    // ε=1%（规格未定值）：椭圆拟合噪声会把结构性二重根劈成 |disc|/b² ~ 噪声量级的
    // 小负值——属正常；只有 |disc|/b² 显著（真不一致锥）才判 NoRealRoot
    if (disc < -1e-2 * std::max(q.b * q.b, std::abs(4.0 * q.a * q.c))) {
        out.failReason = ConicFailReason::NoRealRoot;
        return out;
    }
    if (disc < 0.0) disc = 0.0;   // 容差内负判别式 → 相切（重合平面）
    // ⚠ 勘误：规格原文「根间距比 √disc/max(|λ±|) < 1e-3 → PlaneDegenerate 拒」与 A6 自相矛盾：
    //   共同圆锥曲线的两锥在束中呈结构性二重根（秩 2 平面成员处两条特征值支同时过零，
    //   det 根重数=2；数值与理论双证）——即成功路径 disc≡0，λ̂=−b/(2a) 取顶点正是为免疫
    //   此 cancellation；A6 的 |λ3|/|λ1|≤consistencyMax 更要求 D(λ̂) 秩 2（仅顶点为根时成立）。
    //   故「根过近」不是退化而是常态，真正的平面对退化由 A6 的 |λ2|/|λ1|<1e-3 把关，此处不拒。
    const double lamHat = -q.b / (2.0 * q.a);
    // ---- A6 退化成员拆平面（特征分解；伴随法在此形态永远错误）----
    const cv::Matx44d D = M1 - lamHat * M2;
    cv::Mat evals, evecs;
    if (!cv::eigen(D, evals, evecs) || evals.rows != 4) {
        out.failReason = ConicFailReason::Numerical;
        return out;
    }
    struct EV { double val; cv::Vec4d vec; };
    std::array<EV, 4> evs;
    for (int i = 0; i < 4; ++i) {
        evs[i].val = evals.at<double>(i);
        evs[i].vec = cv::Vec4d(evecs.at<double>(i, 0), evecs.at<double>(i, 1),
                               evecs.at<double>(i, 2), evecs.at<double>(i, 3));
    }
    auto firstNonzeroSign = [](const cv::Vec4d& v) {
        for (int k = 0; k < 4; ++k)
            if (v(k) != 0.0) return v(k) > 0.0 ? 1.0 : -1.0;
        return 1.0;
    };
    // |特征值| 降序；tie-break：值正负（正优先）、向量首个非零分量符号（正优先）——确定性
    std::sort(evs.begin(), evs.end(), [&firstNonzeroSign](const EV& x, const EV& y) {
        const double ax = std::abs(x.val), ay = std::abs(y.val);
        if (ax != ay) return ax > ay;
        if (x.val != y.val) return x.val > y.val;
        return firstNonzeroSign(x.vec) > firstNonzeroSign(y.vec);
    });
    const double l1 = evs[0].val, l2 = evs[1].val, l3 = evs[2].val;
    if (!(std::abs(l1) > 0.0)) {
        out.failReason = ConicFailReason::Numerical;   // 含 NaN
        return out;
    }
    if (std::abs(l3) / std::abs(l1) > params.consistencyMax) {
        out.failReason = ConicFailReason::Inconsistent;
        return out;
    }
    if (l1 * l2 > 0.0) {
        out.failReason = ConicFailReason::VirtualPlane;
        return out;
    }
    if (std::abs(l2) / std::abs(l1) < 1e-3) {
        out.failReason = ConicFailReason::PlaneDegenerate;
        return out;
    }
    const double alpha = std::sqrt(std::abs(l2) / std::abs(l1));
    const cv::Vec4d planes[2] = {evs[0].vec + alpha * evs[1].vec,
                                 evs[0].vec - alpha * evs[1].vec};
    // ---- A7 两平面各提一候选圆 ----
    CircleB cand[2];
    double candRatio[2] = {-1.0, -1.0};   // 每候选截面轴比（-1=未达评估）
    int nCand = 0;
    ConicFailReason lastSkip = ConicFailReason::Numerical;
    double lastRatioSeen = -1.0;          // 末次完成轴比评估的候选（含被轴比闸拒者）
    for (const cv::Vec4d& plane : planes) {
        CircleB c;
        double ratio = -1.0;
        switch (extractCircle(plane, Pi1, Pi2, C, params, B, c, &ratio)) {
            case PlaneOutcome::Circle: cand[nCand] = c; candRatio[nCand] = ratio; ++nCand; break;
            case PlaneOutcome::SkipPlaneDegenerate: lastSkip = ConicFailReason::PlaneDegenerate; break;
            case PlaneOutcome::SkipNotCircle: lastSkip = ConicFailReason::NotCircle; break;
            case PlaneOutcome::SkipNegativeR: lastSkip = ConicFailReason::NegativeR; break;
        }
        if (ratio >= 0.0) lastRatioSeen = ratio;
    }
    // ---- 闸门链（每候选独立）：单幸存→成功；双幸存→Ambiguous；全灭→末次拒因 ----
    int survivors = 0;
    CircleB winner{};
    double winResid = -1.0;   // 幸存者残差（首幸存候选）
    double winRatio = -1.0;   // 幸存者截面轴比（首幸存候选）
    ConicFailReason lastGateRej = ConicFailReason::Numerical;
    for (int k = 0; k < nCand; ++k) {
        GateTrace tr;
        if (gateChain(cand[k], Qpix1, Qpix2, K1, K2, C, B, params, tr)) {
            if (survivors == 0) {
                winner = cand[k];
                winResid = tr.reprojResidualPx;
                winRatio = candRatio[k];
            }
            ++survivors;
            out.grayZone = out.grayZone || tr.grayZone;
        } else {
            lastGateRej = tr.failReason;
            out.reprojResidualPx = tr.reprojResidualPx;   // 全灭审计：末次被拒候选（拒也填）
        }
    }
    if (survivors >= 1) {
        out.reprojResidualPx = winResid;        // 有幸存：幸存者残差（成功/Ambiguous）
        out.sectionAxisRatio = winRatio;        // 幸存者截面轴比（同口径）
    } else {
        out.sectionAxisRatio = lastRatioSeen;   // 全灭：末次完成轴比评估的候选（未评过=-1）
    }
    out.solutionCount = nCand;
    if (nCand >= 1) out.circle = cand[0];             // 诊断锚（拒因/eta/半径审计）
    if (survivors == 1) {
        out.failReason = ConicFailReason::None;       // 单候选过全闸
        out.circle = winner;
    } else if (survivors == 2) {
        out.failReason = ConicFailReason::Ambiguous;  // 双候选全过闸：禁静默取一
    } else {
        out.failReason = (nCand > 0) ? lastGateRej : lastSkip;
    }
    return out;
}

// ---------- Task 5：候选上限与同左竞争消歧阈值 ----------
constexpr size_t kMaxCandidatePairs = 64;        // 候选总数上限：种子全保留，预筛按
                                                // |d−窗中心| 升序截断（tie-break (i,j)
                                                // 字典序）——确定性简单规则
constexpr double kDistinctResidualFactor = 1.5;  // 次优与最优不可区分阈值：同 leftIdx
                                                // 多个过闸解中 runner-up 残差 ≤1.5×
                                                // winner 残差 → 该 leftIdx 判 Ambiguous

} // namespace

struct ConicReconstructCPU::Impl {
    ConicReconstructCPUParams params_;
    cv::Matx33d K1_ = cv::Matx33d::eye(), K2_ = cv::Matx33d::eye();
    cv::Vec3d C2_{131.0, 0.0, 0.0};
    bool calibrated_ = false;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const ConicReconstructCPUParams& p) : params_(p) { params_.validate(); }

    ConicReconstructCPUResult ExecuteImpl(const std::vector<EllipseFitCPUResult>& left,
                                          const std::vector<EllipseFitCPUResult>& right,
                                          const std::vector<int>& centerMatches) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif
        ConicReconstructCPUResult result;
        if (!calibrated_) {
            result.success = false;
            result.message = "projection matrices not set";
            return result;
        }
        result.success = true;
        result.message = "Conic reconstruct completed";
        const double B = cv::norm(C2_);
        const cv::Vec3d Cunit = B > 0.0 ? C2_ / B : cv::Vec3d(0.0, 0.0, 0.0);

        // ---- 行偏置预修正（批准修正案）：右目 centerY −rowOffsetPx（默认 0 关）。
        // 逐字段重建修正副本（EllipseFitCPUResult 不可拷贝；message/qualityFlag/
        // inlierPoints 下游不消费，不搬）；R 统一供种子/预筛/锥面/残差闸消费——
        // 只修一次，Qpix 与 Sampson 闸同源；左目不动 ----
        std::vector<EllipseFitCPUResult> rightCorr;
        const std::vector<EllipseFitCPUResult>* rightUse = &right;
        if (params_.rowOffsetPx != 0.0) {
            rightCorr.reserve(right.size());
            for (const EllipseFitCPUResult& e : right) {
                EllipseFitCPUResult c;
                c.success = e.success;
                c.groupId = e.groupId;
                c.centerX = e.centerX;
                c.centerY = e.centerY - params_.rowOffsetPx;
                c.majorAxis = e.majorAxis;
                c.minorAxis = e.minorAxis;
                c.angle = e.angle;
                c.inlierCount = e.inlierCount;
                c.totalPointCount = e.totalPointCount;
                rightCorr.push_back(std::move(c));
            }
            rightUse = &rightCorr;
        }
        const std::vector<EllipseFitCPUResult>& R = *rightUse;

        // ---- A. 候选生成（设计 §2.0 D4，Task 5）：种子对必进＋轻量预筛，cap 截断 ----
        struct Cand { int li; int rj; };
        std::vector<Cand> cands;
        const size_t n = std::min(left.size(), centerMatches.size());
        for (size_t i = 0; i < n; ++i) {
            const int j = centerMatches[i];
            if (j < 0 || static_cast<size_t>(j) >= R.size()) continue;
            if (!left[i].success || !R[static_cast<size_t>(j)].success) continue;
            ++result.statistics.totalPairs;   // 口径：种子对数（不含预筛）
            cands.push_back({static_cast<int>(i), j});
        }
        if (B > 0.0 && !left.empty() && !R.empty()) {
            // 预筛窗：d=uL−uR ∈ [fx·B/depthMax, fx·B/depthMin]（中心像素坐标差）
            // ＋ |vL−vR| ≤ yPreGatePx（vR 已吃行偏置修正）；NaN 坐标在 !(…) 比较下自然出局
            const double fx = K1_(0, 0);
            const double dMin = fx * B / params_.depthMaxMm;
            const double dMax = fx * B / params_.depthMinMm;
            struct Pre { int li; int rj; double d; };
            std::vector<Pre> pres;
            for (size_t i = 0; i < left.size(); ++i) {
                if (!left[i].success) continue;
                for (size_t j = 0; j < R.size(); ++j) {
                    if (!R[j].success) continue;
                    if (i < centerMatches.size() && centerMatches[i] == static_cast<int>(j))
                        continue;   // 种子对已收录，去重
                    const double d = left[i].centerX - R[j].centerX;
                    if (!(d >= dMin && d <= dMax)) continue;
                    if (!(std::abs(left[i].centerY - R[j].centerY) <= params_.yPreGatePx))
                        continue;
                    pres.push_back({static_cast<int>(i), static_cast<int>(j), d});
                }
            }
            if (cands.size() + pres.size() > kMaxCandidatePairs) {
                const double dCenter = 0.5 * (dMin + dMax);
                std::sort(pres.begin(), pres.end(), [dCenter](const Pre& a, const Pre& b) {
                    const double da = std::abs(a.d - dCenter), db = std::abs(b.d - dCenter);
                    if (da != db) return da < db;
                    if (a.li != b.li) return a.li < b.li;
                    return a.rj < b.rj;
                });
                pres.resize(cands.size() >= kMaxCandidatePairs
                                ? 0 : kMaxCandidatePairs - cands.size());
            }
            for (const Pre& p : pres) cands.push_back({p.li, p.rj});
        }
        result.statistics.candidatePairs = cands.size();   // 实际参与 solvePair（含种子）

        // ---- B. 逐候选解算：isfinite 前置 → 锥面 → 束解 → 闸门链 ----
        struct Solved {
            Cand c;
            ConicPairDiagnostic diag;
            SolveOutcome so;
            bool dropped = false;   // 同左竞争落败（诊断保留，不进 markers）
        };
        std::vector<Solved> solved;
        solved.reserve(cands.size());
        for (const Cand& cd : cands) {
            Solved s;
            s.c = cd;
            s.diag.leftIdx = cd.li;
            s.diag.rightIdx = cd.rj;
            try {
                cv::Matx33d Pi1, Pi2, Qpix1, Qpix2;
                if (!ellipseFinite(left[static_cast<size_t>(cd.li)]) ||
                    !ellipseFinite(R[static_cast<size_t>(cd.rj)])) {
                    s.so.failReason = ConicFailReason::Numerical;   // isfinite 前置
                } else if (!buildCone(left[static_cast<size_t>(cd.li)], K1_, Pi1, &Qpix1) ||
                           !buildCone(R[static_cast<size_t>(cd.rj)], K2_, Pi2, &Qpix2)) {
                    s.so.failReason = ConicFailReason::InputClass;
                } else {
                    s.so = solvePair(Pi1, Pi2, Qpix1, Qpix2, K1_, K2_, Cunit, B, params_);
                }
            } catch (const std::exception&) {
                s.so.failReason = ConicFailReason::Numerical;
                s.so.solutionCount = 0;
            }
            s.diag.failReason = s.so.failReason;
            s.diag.solutionCount = s.so.solutionCount;
            s.diag.reprojResidualPx = s.so.reprojResidualPx;
            s.diag.sectionAxisRatio = s.so.sectionAxisRatio;
            if (s.so.solutionCount >= 1) {
                s.diag.etaBaseline = std::abs(s.so.circle.n.dot(Cunit));   // |n̂·b̂|
                // η₂ 视线口径：|(L−C)·n̂|/‖L−C‖，L=左光心=原点 → |C·n̂|/‖C‖
                const double cn = cv::norm(s.so.circle.C);
                s.diag.etaView = cn > 0.0 ? std::abs(s.so.circle.C.dot(s.so.circle.n)) / cn : 0.0;
                s.diag.reconstructedRadiusMm = s.so.circle.r * B;
            }
            solved.push_back(std::move(s));
        }

        // ---- C. 同左多右竞争（设计 §2.7）：过闸解按 leftIdx 分组，跨 leftIdx 互不影响 ----
        std::map<int, std::vector<size_t>> groups;
        for (size_t k = 0; k < solved.size(); ++k)
            if (solved[k].so.failReason == ConicFailReason::None)
                groups[solved[k].c.li].push_back(k);
        for (auto& entry : groups) {
            std::vector<size_t>& idxs = entry.second;
            if (idxs.size() < 2) continue;
            std::sort(idxs.begin(), idxs.end(), [&solved](size_t a, size_t b) {
                const double ra = solved[a].so.reprojResidualPx;
                const double rb = solved[b].so.reprojResidualPx;
                if (ra != rb) return ra < rb;
                return solved[a].c.rj < solved[b].c.rj;   // 确定性 tie-break
            });
            const double winResid = solved[idxs[0]].so.reprojResidualPx;
            const double runResid = solved[idxs[1]].so.reprojResidualPx;
            if (runResid <= kDistinctResidualFactor * winResid) {
                // 次优与最优不可区分 → 该 leftIdx 整组不出标记（winner 判 Ambiguous）
                solved[idxs[0]].so.failReason = ConicFailReason::Ambiguous;
                solved[idxs[0]].diag.failReason = ConicFailReason::Ambiguous;
            }
            for (size_t t = 1; t < idxs.size(); ++t) {
                solved[idxs[t]].dropped = true;                 // 不进 markers，诊断保留
                solved[idxs[t]].diag.droppedByCompetition = true;   // 落败者标记（failReason 仍 None）
            }
        }

        // ---- D. 结果组装＋统计（灰区：任一输出对落灰区带 → 整批 Degraded）----
        bool anyGray = false;
        for (Solved& s : solved) {
            if (s.so.failReason == ConicFailReason::None && !s.dropped) {
                MarkerReconstructResult mk;
                mk.leftEllipseIdx = s.c.li;
                mk.rightEllipseIdx = s.c.rj;
                const cv::Vec3d Cmm = s.so.circle.C * B;
                cv::Vec3d nOut = s.so.circle.n;
                if (nOut.dot(Cmm) > 0.0) nOut = -nOut;   // 定向：统一翻向左相机（n·C<0）
                mk.centerX = Cmm(0);
                mk.centerY = Cmm(1);
                mk.centerZ = Cmm(2);
                mk.normalX = nOut(0);
                mk.normalY = nOut(1);
                mk.normalZ = nOut(2);
                mk.validPlane = true;
                mk.validCircle = true;
                mk.circleFit.centerX = mk.centerX;
                mk.circleFit.centerY = mk.centerY;
                mk.circleFit.centerZ = mk.centerZ;
                mk.circleFit.radius = s.so.circle.r * B;
                mk.planeFit.nx = mk.normalX;
                mk.planeFit.ny = mk.normalY;
                mk.planeFit.nz = mk.normalZ;
                mk.planeFit.d = -nOut.dot(Cmm);
                mk.planeFit.centroidX = mk.centerX;
                mk.planeFit.centroidY = mk.centerY;
                mk.planeFit.centroidZ = mk.centerZ;
                if (params_.ringPointsForViz > 0) {
                    // 合成圆环：ringPointsForViz 点均布圆上；两目像素投影＋对级残差
                    cv::Vec3d e1, e2;
                    circleTangentBasis(nOut, e1, e2);
                    const double rMm = s.so.circle.r * B;
                    mk.reconstructedPoints.reserve(static_cast<size_t>(params_.ringPointsForViz));
                    for (int t = 0; t < params_.ringPointsForViz; ++t) {
                        const double th = 2.0 * CV_PI * t / params_.ringPointsForViz;
                        const cv::Vec3d X = Cmm + rMm * (std::cos(th) * e1 + std::sin(th) * e2);
                        const cv::Vec3d xl = K1_ * X;
                        const cv::Vec3d xr = K2_ * (X - C2_);
                        ReconstructedPoint3D pt;
                        pt.x = X(0);
                        pt.y = X(1);
                        pt.z = X(2);
                        pt.leftU = xl(0) / xl(2);
                        pt.leftV = xl(1) / xl(2);
                        pt.rightU = xr(0) / xr(2);
                        pt.rightV = xr(1) / xr(2);
                        pt.reprojError = static_cast<float>(s.so.reprojResidualPx);
                        mk.reconstructedPoints.push_back(pt);
                    }
                }
                mk.message = "ok";
                result.markers.push_back(std::move(mk));
                ++result.statistics.validCount;
                anyGray = anyGray || s.so.grayZone;
            }
            ++result.statistics.failReasonHistogram[static_cast<size_t>(s.so.failReason)];
            result.diagnostics.push_back(std::move(s.diag));
        }
        result.statistics.rejectedCount =
            result.statistics.candidatePairs - result.statistics.validCount;
        // ---- 质量旗（设计 §4）：部分有效→Warning；灰区通过→Degraded（若原本 Normal）----
        if (result.statistics.validCount < result.statistics.candidatePairs) {
            result.qualityFlag = QualityFlag::Warning;
        }
        if (anyGray && result.qualityFlag == QualityFlag::Normal) {
            result.qualityFlag = QualityFlag::Degraded;
        }
        return result;
    }
};

ConicReconstructCPUResult ConicReconstructCPU::Execute(const std::vector<EllipseFitCPUResult>& l,
                                                       const std::vector<EllipseFitCPUResult>& r,
                                                       const std::vector<int>& cm) {
    return pImpl_->ExecuteImpl(l, r, cm);
}

ConicReconstructCPU::ConicReconstructCPU(const ConicReconstructCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params)) {
    CALIB_LOG_INFO("ConicReconstructCPU initialized (radius={}mm, residual={}px)",
                   params.nominalRadiusMm, params.maxReprojResidualPx);
}
ConicReconstructCPU::~ConicReconstructCPU() = default;
void ConicReconstructCPU::Destroy() { }

void ConicReconstructCPU::SetProjectionMatrices(const cv::Matx33d& K1, const cv::Matx33d& K2,
                                                const cv::Vec3d& rightCenter) {
    pImpl_->K1_ = K1; pImpl_->K2_ = K2; pImpl_->C2_ = rightCenter; pImpl_->calibrated_ = true;
    CALIB_LOG_INFO("SetProjectionMatrices called (|C2|={}mm)", cv::norm(rightCenter));
}
void ConicReconstructCPU::Warmup(int n) { pImpl_->warmed_up_ = true; CALIB_LOG_INFO("warmup: maxPairs={}", n); }
void ConicReconstructCPU::Warmup(const WarmupConfig& c) { Warmup(c.maxPointCount); }
void ConicReconstructCPU::SetParams(const ConicReconstructCPUParams& p) {
    pImpl_->params_ = p;
    pImpl_->params_.validate();
    pImpl_->warmed_up_ = false;
}
const ConicReconstructCPUParams& ConicReconstructCPU::GetParams() const { return pImpl_->params_; }
