#include "global_ba_zg.h"
#include "ba_residuals.h"
#include "common/calib_logging.h"

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace calib {

CALIB_DEFINE_LOG_TAG(13Z, GlobalBundleAdjustmentZG)

void ZgBaOptions::validate() const {
    if (huberDeltaSigma <= 0.0)
        throw std::invalid_argument("huberDeltaSigma must be > 0");
    if (preGateK < 1.0)
        throw std::invalid_argument("preGateK must be >= 1");
    if (preGateFloorSigma <= 0.0)
        throw std::invalid_argument("preGateFloorSigma must be > 0");
}

namespace {

// R(3x3)→四元数 [w,x,y,z]（Eigen）
inline void rotToQuat(const cv::Matx33d& R, double q[4]) {
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m(r, c) = R(r, c);
    Eigen::Quaterniond Q(m);
    Q.normalize();
    q[0] = Q.w(); q[1] = Q.x(); q[2] = Q.y(); q[3] = Q.z();
}
inline cv::Matx33d quatToRot(const double q[4]) {
    Eigen::Quaterniond Q(q[0], q[1], q[2], q[3]);
    Q.normalize();
    const Eigen::Matrix3d m = Q.toRotationMatrix();
    cv::Matx33d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R(r, c) = m(r, c);
    return R;
}

// 观测扁平化记录
struct ObsRec {
    int frameIdx;      // 帧序（0..Nf-1, 装配序）
    int pointIdx;      // 点序（0..Np-1）
    cv::Point3d z;     // 本机系观测
};

// 线性 LS 三角化单点: min Σ‖R_iᵀ(X−t_i) − z_i‖² → 3×3 正规方程闭式
bool triangulatePoint(const std::vector<ObsRec>& obsOfPoint,
                      const std::vector<cv::Matx33d>& Rs,
                      const std::vector<cv::Vec3d>& ts, double X[3]) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (const auto& o : obsOfPoint) {
        Eigen::Matrix3d R;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) R(r, c) = Rs[o.frameIdx](r, c);
        const Eigen::Vector3d t(ts[o.frameIdx](0), ts[o.frameIdx](1), ts[o.frameIdx](2));
        const Eigen::Vector3d z(o.z.x, o.z.y, o.z.z);
        // r = Rᵀ(X − t) − z = RᵀX − (Rᵀt + z)
        A += R * R.transpose();
        b += R * (R.transpose() * t + z);
    }
    const double det = A.determinant();
    if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
    const Eigen::Vector3d x = A.ldlt().solve(b);
    X[0] = x(0); X[1] = x(1); X[2] = x(2);
    return std::isfinite(X[0]) && std::isfinite(X[1]) && std::isfinite(X[2]);
}

// 残差（毫米向量）: r = Rᵀ(X − t) − z（与 PointPairResidual 同式）
inline Eigen::Vector3d residualOf(const ObsRec& o, const cv::Matx33d& R,
                                  const cv::Vec3d& t, const double X[3]) {
    const double ox = X[0] - t(0), oy = X[1] - t(1), oz = X[2] - t(2);
    const double px = R(0, 0) * ox + R(1, 0) * oy + R(2, 0) * oz;
    const double py = R(0, 1) * ox + R(1, 1) * oy + R(2, 1) * oz;
    const double pz = R(0, 2) * ox + R(1, 2) * oy + R(2, 2) * oz;
    return Eigen::Vector3d(px - o.z.x, py - o.z.y, pz - o.z.z);
}

}  // namespace

ZgBaResult GlobalBundleAdjustmentZG::Execute(const GlobalBAInput& input,
                                             const GlobalBAParams& params,
                                             const ZgBaOptions& opts) {
    opts.validate();
    ZgBaResult result;
    result.base.success = false;
    result.solverName = opts.useSchur ? "ITERATIVE_SCHUR+JACOBI"
                                      : "SPARSE_NORMAL_CHOLESKY";

    // ---- 装配: gid≥0 过滤 + 帧过滤（≥1 观测） ----
    if (input.frames.size() < 2) {
        result.base.message = "ZG-BA: need >= 2 frames";
        return result;
    }
    std::vector<cv::Matx33d> Rs;
    std::vector<cv::Vec3d> ts;
    std::vector<std::vector<ObsRec>> frameObs;  // 装配帧序 → 观测
    std::unordered_map<int, int> gidToPoint;    // globalId → 点序
    std::vector<int> pointGid;
    std::vector<ObsRec> allObs;
    for (const auto& f : input.frames) {
        std::vector<ObsRec> fo;
        fo.reserve(f.markerObs.size());
        for (const auto& m : f.markerObs) {
            if (m.globalId < 0) continue;
            auto it = gidToPoint.find(m.globalId);
            if (it == gidToPoint.end()) {
                it = gidToPoint.emplace(m.globalId, (int)pointGid.size()).first;
                pointGid.push_back(m.globalId);
            }
            ObsRec o;
            o.frameIdx = (int)Rs.size();
            o.pointIdx = it->second;
            o.z = m.local;
            fo.push_back(o);
        }
        if (fo.empty()) continue;  // 无效帧剔除（主路 minPointsPerFrame 语义的弱化版）
        Rs.push_back(f.R_init);
        ts.push_back(f.t_init);
        for (auto& o : fo) o.frameIdx = (int)(Rs.size() - 1);
        frameObs.push_back(std::move(fo));
    }
    const int Nf = (int)Rs.size();
    const int Np = (int)pointGid.size();
    if (Nf < 2 || Np < 1) {
        result.base.message = "ZG-BA: insufficient frames/points after assembly";
        return result;
    }
    for (auto& fo : frameObs)
        for (auto& o : fo) allObs.push_back(o);
    const int Nobs = (int)allObs.size();
    result.obsTotal = Nobs;

    // ---- 参数块初值: 帧 q/t 链乘累积; 点 = 线性 LS 三角化 ----
    std::vector<std::array<double, 4>> q(Nf);
    std::vector<std::array<double, 3>> tt(Nf);
    for (int i = 0; i < Nf; ++i) {
        rotToQuat(Rs[i], q[i].data());
        tt[i] = {ts[i](0), ts[i](1), ts[i](2)};
    }
    std::vector<std::vector<double>> X(Np, std::vector<double>(3, 0.0));
    {
        std::vector<std::vector<ObsRec>> byPoint(Np);
        for (const auto& o : allObs) byPoint[o.pointIdx].push_back(o);
        for (int p = 0; p < Np; ++p) {
            if (!triangulatePoint(byPoint[p], Rs, ts, X[p].data())) {
                // 退化（共线观测）: 取首观测反投影质心兜底
                Eigen::Vector3d acc(0, 0, 0);
                for (const auto& o : byPoint[p]) {
                    const auto& R = Rs[o.frameIdx];
                    const auto& t = ts[o.frameIdx];
                    Eigen::Matrix3d Rm;
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) Rm(r, c) = R(r, c);
                    const Eigen::Vector3d z(o.z.x, o.z.y, o.z.z);
                    acc += Rm * z + Eigen::Vector3d(t(0), t(1), t(2));
                }
                acc /= (double)byPoint[p].size();
                X[p] = {acc(0), acc(1), acc(2)};
            }
        }
    }

    // ---- 入网门控（ZG 思路②）: 初始残差 MAD 稳健尺度预剔 ----
    std::vector<char> keep(Nobs, 1);
    std::vector<double> r0(Nobs);
    for (int i = 0; i < Nobs; ++i) {
        const auto& o = allObs[i];
        r0[i] = residualOf(o, Rs[o.frameIdx], ts[o.frameIdx], X[o.pointIdx].data()).norm();
    }
    if (opts.usePreGate && Nobs >= 8) {
        std::vector<double> sorted(r0);
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted[sorted.size() / 2];
        std::vector<double> dev(Nobs);
        for (int i = 0; i < Nobs; ++i) dev[i] = std::fabs(r0[i] - med);
        std::sort(dev.begin(), dev.end());
        const double mad = dev[dev.size() / 2];
        const double scale = 1.4826 * mad;
        const double gate = std::max(opts.preGateK * scale,
                                     opts.preGateFloorSigma * params.sigmaObserved);
        int gated = 0;
        for (int i = 0; i < Nobs; ++i)
            if (r0[i] > gate) {
                keep[i] = 0;
                ++gated;
            }
        // 防御: 门控超过半数 → 初值太差, 放弃门控全量入网（交 Huber 降权）
        if (gated * 2 > Nobs) {
            std::fill(keep.begin(), keep.end(), 1);
            CALIB_LOG_WARN("[{}] pre-gate would drop {}/{} obs; init too coarse, gate skipped",
                           kLogTag, gated, Nobs);
            gated = 0;
        }
        result.obsGated = gated;
    }
    int nKept = 0;
    for (int i = 0; i < Nobs; ++i) nKept += keep[i] ? 1 : 0;
    result.obsKept = nKept;

    // ---- 残差统计辅助 ----
    auto computeStats = [&](bool initial, double& rmse, double& sigma0Dimless,
                            double& sigma0Mm, int& dofOut) {
        double s2 = 0, sn = 0;
        int n = 0;
        std::vector<cv::Matx33d> Rcur(Nf);
        std::vector<cv::Vec3d> tcur(Nf);
        for (int i = 0; i < Nf; ++i) {
            if (initial) {
                Rcur[i] = Rs[i];
                tcur[i] = ts[i];
            } else {
                Rcur[i] = quatToRot(q[i].data());
                tcur[i] = cv::Vec3d(tt[i][0], tt[i][1], tt[i][2]);
            }
        }
        for (int i = 0; i < Nobs; ++i) {
            if (!keep[i]) continue;
            const auto& o = allObs[i];
            const Eigen::Vector3d r = residualOf(o, Rcur[o.frameIdx], tcur[o.frameIdx],
                                                 X[o.pointIdx].data());
            s2 += r.squaredNorm() / (params.sigmaObserved * params.sigmaObserved);
            sn += r.squaredNorm();
            ++n;
        }
        if (n == 0) return;
        rmse = std::sqrt(sn / n);  // 3D 残差向量模 RMS（mm）
        int dof = 3 * n - 6 * (Nf - 1) - 3 * Np;
        if (dof < 1) dof = 1;
        dofOut = dof;
        sigma0Dimless = std::sqrt(s2 / dof);   // 单位权后验 σ₀̂
        sigma0Mm = std::sqrt(sn / dof);        // 毫米口径
    };
    double rmse0 = 0, s00 = 0, s0mm0 = 0;
    int dof0 = 0;
    computeStats(true, rmse0, s00, s0mm0, dof0);
    result.sigma0InitDimless = s00;
    result.base.statistics.initialRMSE = rmse0;

    // ---- 建题: PointPairResidual + Huber(δ=4σ)（ZG 口径）; 帧0位姿定标（规范） ----
    ceres::Problem problem;
    for (int i = 0; i < Nobs; ++i) {
        if (!keep[i]) continue;
        const auto& o = allObs[i];
        ceres::LossFunction* loss =
            new ceres::HuberLoss(opts.huberDeltaSigma);  // 残差已 ×1/σ → δ 即 σ 倍数
        auto* cost = new ceres::AutoDiffCostFunction<
            PointPairResidual, 3, 4, 3, 3>(
            new PointPairResidual(Eigen::Vector3d(o.z.x, o.z.y, o.z.z),
                                  params.sigmaObserved));
        problem.AddResidualBlock(cost, loss, q[o.frameIdx].data(), tt[o.frameIdx].data(),
                                 X[o.pointIdx].data());
    }
    problem.SetParameterBlockConstant(q[0].data());
    problem.SetParameterBlockConstant(tt[0].data());

    // P4-T24 软先验（与主路同语义: 高精度点加 ‖X−X_existing‖²/σ² 软约束, 不加鲁棒）
    if (params.useSoftPrior && !input.highPrecisionGlobalIds.empty()) {
        std::unordered_set<int> hp(input.highPrecisionGlobalIds.begin(),
                                   input.highPrecisionGlobalIds.end());
        int priors = 0;
        for (int p = 0; p < Np; ++p) {
            if (hp.find(pointGid[p]) == hp.end()) continue;
            const size_t k = std::find(input.highPrecisionGlobalIds.begin(),
                                       input.highPrecisionGlobalIds.end(), pointGid[p]) -
                             input.highPrecisionGlobalIds.begin();
            if (k >= input.X_existing.size() / 3) continue;
            const double sigma = (k < input.priorSigma.size())
                                     ? input.priorSigma[k]
                                     : params.defaultPriorSigma;
            auto* pc = new ceres::DynamicAutoDiffCostFunction<MarkerPriorCost, 3>(
                new MarkerPriorCost{input.X_existing[3 * k], input.X_existing[3 * k + 1],
                                    input.X_existing[3 * k + 2], sigma});
            pc->AddParameterBlock(3);
            pc->SetNumResiduals(3);
            problem.AddResidualBlock(pc, nullptr, X[p].data());
            ++priors;
        }
        if (priors > 0)
            CALIB_LOG_INFO("[{}] soft priors applied: {}", kLogTag, priors);
    }

    // ---- 求解（ZG 思路①: Schur 消点; 限时/取消沿用主路字段语义） ----
    ceres::Solver::Options so;
    if (opts.useSchur) {
        so.linear_solver_type = ceres::ITERATIVE_SCHUR;
        so.preconditioner_type = ceres::JACOBI;
    } else {
        so.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    so.max_num_iterations = params.maxIterations;
    so.function_tolerance = params.tolerance;
    so.minimizer_progress_to_stdout = false;
    if (params.maxSolveSecondsPerPhase > 0)
        so.max_solver_time_in_seconds = params.maxSolveSecondsPerPhase;
    // 进度（单阶段 0→1）+ 取消（取消=不采用半程解, 与主路 USER_FAILURE 语义一致）
    struct IterCb : ceres::IterationCallback {
        const GlobalBAParams* prm;
        bool cancelled = false;
        explicit IterCb(const GlobalBAParams& p) : prm(&p) {}
        ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override {
            if (prm->cancelRequested && prm->cancelRequested()) {
                cancelled = true;
                return ceres::SOLVER_ABORT;
            }
            if (prm->progressCb)
                prm->progressCb(std::min(1.0, (s.iteration + 1.0) / prm->maxIterations));
            return ceres::SOLVER_CONTINUE;
        }
    } iterCb(params);
    so.callbacks.push_back(&iterCb);

    ceres::Solver::Summary summary;
    ceres::Solve(so, &problem, &summary);
    result.base.statistics.ceresIterations = (int)summary.num_successful_steps +
                                             (int)summary.num_unsuccessful_steps;

    if (iterCb.cancelled) {
        result.base.message = "ZG-BA cancelled by request";
        return result;
    }
    if (summary.termination_type == ceres::FAILURE) {
        result.base.message = "ZG-BA solver failure: " + summary.BriefReport();
        return result;
    }

    // ---- 终态统计与产物 ----
    double rmse1 = 0, s01 = 0, s0mm1 = 0;
    int dof1 = 0;
    computeStats(false, rmse1, s01, s0mm1, dof1);
    result.sigma0FinalDimless = s01;
    result.sigma0FinalMm = s0mm1;
    result.dof = dof1;
    result.base.statistics.finalRMSE = rmse1;
    for (int i = 0; i < Nobs; ++i)
        if (!keep[i]) result.base.statistics.outlierObsIds.push_back(i);

    result.base.optimizedPoses.resize(Nf);
    for (int i = 0; i < Nf; ++i) {
        FramePose fp;
        fp.frameId = input.frames.empty() ? 0 : 0;
        fp.R = quatToRot(q[i].data());
        fp.t = cv::Vec3d(tt[i][0], tt[i][1], tt[i][2]);
        result.base.optimizedPoses[i] = fp;
    }
    // frameId 回填: 装配期剔除的帧不在产物中, 有效帧按装配序对应原 frames 里
    // 保留者（重新扫一遍与装配同规则）
    {
        std::vector<uint64_t> keptIds;
        for (const auto& f : input.frames) {
            int cnt = 0;
            for (const auto& m : f.markerObs)
                if (m.globalId >= 0) ++cnt;
            if (cnt > 0) keptIds.push_back(f.frameId);
        }
        for (int i = 0; i < Nf && i < (int)keptIds.size(); ++i)
            result.base.optimizedPoses[i].frameId = keptIds[(size_t)i];
    }
    result.base.optimizedMarkers.resize(Np);
    for (int p = 0; p < Np; ++p) {
        result.base.optimizedMarkers[p].X =
            cv::Point3d(X[p][0], X[p][1], X[p][2]);
        result.base.optimizedMarkers[p].globalId = pointGid[p];
    }
    result.base.success = true;
    result.base.message = summary.BriefReport();
    CALIB_LOG_INFO("[{}] done: {} frames, {} points, obs {}/{} kept, RMSE {:.4f}->{:.4f} mm, "
                   "sigma0 {} (dof {})",
                   kLogTag, Nf, Np, nKept, Nobs, rmse0, rmse1, s01, dof1);
    return result;
}

bool GlobalBundleAdjustmentZG::RunAb(const GlobalBAInput& input,
                                     const GlobalBAParams& params,
                                     const ZgBaOptions& opts,
                                     GlobalBAResult& outMain, ZgBaResult& outZg) {
    GlobalBundleAdjustmentCPU mainBa(params);
    outMain = mainBa.Execute(input);
    GlobalBundleAdjustmentZG zg;
    outZg = zg.Execute(input, params, opts);
    return outMain.success && outZg.base.success;
}

OperatorInfo getGlobalBundleAdjustmentZGInfo() {
    return OperatorInfo{"GlobalBundleAdjustmentZG", SCANNER_VERSION_MAJOR,
                        SCANNER_VERSION_MINOR, OperatorType::CPU};
}

}  // namespace calib
