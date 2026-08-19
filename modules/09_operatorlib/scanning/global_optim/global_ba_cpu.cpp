#include "global_ba_cpu.h"
#include "ba_residuals.h"
#include "pose_graph_residuals.h"
#include <spdlog/spdlog.h>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace calib {

namespace {

// Kabsch: 求 R,t 使 R*z_k + t ≈ z_i(成对点集, z_i/z_k 为同一 landmark 在帧 i/k 的本机系坐标)。
// 返回的 (R, t) 即 "frame-k-local → frame-i-local" 的相对位姿 (R_ik, t_ik), 直接喂给 RelativePoseResidual。
// SVD + 反射修正(determinant 符号), 保证 R 为正交旋转。
// 返回 false 表示退化: 点数<3 / 两集大小不一致 / 共线或近共线(H 秩亏损)。
//   退化时 SVD 会返回绕共享线的任意旋转, 是喂给 PGO(无鲁棒损失)的污染测量, 故调用方须跳过该边。
bool kabsch(const std::vector<cv::Point3d>& zi,
            const std::vector<cv::Point3d>& zk,
            cv::Matx33d& R, cv::Vec3d& t) {
    // 点数守卫: <3 点无法约束 3-DoF 旋转; 大小不一致为程序错误。零点会致除零。
    if (zi.size() < 3 || zi.size() != zk.size()) return false;
    // 质心
    cv::Point3d ci{}, ck{};
    for (size_t m = 0; m < zi.size(); ++m) { ci += zi[m]; ck += zk[m]; }
    ci *= 1.0 / static_cast<double>(zi.size());
    ck *= 1.0 / static_cast<double>(zk.size());
    // 去质心, 构造 H = Σ (zi-ci)(zk-ck)ᵀ
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t m = 0; m < zi.size(); ++m) {
        Eigen::Vector3d a(zi[m].x - ci.x, zi[m].y - ci.y, zi[m].z - ci.z);
        Eigen::Vector3d b(zk[m].x - ck.x, zk[m].y - ck.y, zk[m].z - ck.z);
        H += a * b.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // 退化条件数检查: 共线点集 → H 秩<=1 → 最小奇异值趋于 0。
    // 取最小/最大奇异值比 < 1e-3 判退化(对应条件数 > 1e3)。此时 SVD 返回的 R 绕共享线任意,
    // 是不可信测量。singularValues() 按降序排列: (0)最大, (2)最小。
    const double sMax = svd.singularValues()(0);
    const double sMin = svd.singularValues()(2);
    if (sMax <= 0.0 || sMin / sMax < 1e-3) return false;
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d D; D.setIdentity();
    D(2, 2) = (U.determinant() * V.determinant() < 0) ? -1.0 : 1.0;  // 反射修正
    Eigen::Matrix3d Re = U * D * V.transpose();                       // R: zk→zi
    Eigen::Vector3d ciE(ci.x, ci.y, ci.z);
    Eigen::Vector3d ckE(ck.x, ck.y, ck.z);
    Eigen::Vector3d te = ciE - Re * ckE;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) R(a, b) = Re(a, b);
    t = cv::Vec3d(te(0), te(1), te(2));
    return true;
}

} // namespace

void GlobalBAParams::validate() const {
    if (sigmaObserved <= 0.0) throw std::invalid_argument("sigmaObserved must be > 0");
    if (tukeyC <= 0.0) throw std::invalid_argument("tukeyC must be > 0");
    if (chiSquareGate <= 0.0) throw std::invalid_argument("chiSquareGate must be > 0");
    if (tolerance <= 0.0) throw std::invalid_argument("tolerance must be > 0");
    if (maxIterations < 1) throw std::invalid_argument("maxIterations must be >= 1");
    if (minCovisForLoopEdge < 3) throw std::invalid_argument("minCovisForLoopEdge must be >= 3");
    if (loopFrameGap < 1) throw std::invalid_argument("loopFrameGap must be >= 1");
    if (minPointsPerFrame < 3) throw std::invalid_argument("minPointsPerFrame must be >= 3");
    if (defaultPriorSigma <= 0.0) throw std::invalid_argument("defaultPriorSigma must be > 0");
}

nlohmann::json GlobalBAParams::toJson() const {
    return {
        {"enablePoseGraphPreopt", enablePoseGraphPreopt},
        {"sigmaObserved", sigmaObserved},
        {"tukeyC", tukeyC},
        {"chiSquareGate", chiSquareGate},
        {"tolerance", tolerance},
        {"maxIterations", maxIterations},
        {"minCovisForLoopEdge", minCovisForLoopEdge},
        {"loopFrameGap", loopFrameGap},
        {"minPointsPerFrame", minPointsPerFrame},
        {"centerOrigin", centerOrigin},
        {"useSoftPrior", useSoftPrior},
        {"defaultPriorSigma", defaultPriorSigma}
    };
}

GlobalBAParams GlobalBAParams::fromJson(const nlohmann::json& j) {
    GlobalBAParams p;
    if (j.contains("enablePoseGraphPreopt")) p.enablePoseGraphPreopt = j.at("enablePoseGraphPreopt").get<bool>();
    if (j.contains("sigmaObserved")) p.sigmaObserved = j.at("sigmaObserved").get<double>();
    if (j.contains("tukeyC")) p.tukeyC = j.at("tukeyC").get<double>();
    if (j.contains("chiSquareGate")) p.chiSquareGate = j.at("chiSquareGate").get<double>();
    if (j.contains("tolerance")) p.tolerance = j.at("tolerance").get<double>();
    if (j.contains("maxIterations")) p.maxIterations = j.at("maxIterations").get<int>();
    if (j.contains("minCovisForLoopEdge")) p.minCovisForLoopEdge = j.at("minCovisForLoopEdge").get<int>();
    if (j.contains("loopFrameGap")) p.loopFrameGap = j.at("loopFrameGap").get<int>();
    if (j.contains("minPointsPerFrame")) p.minPointsPerFrame = j.at("minPointsPerFrame").get<int>();
    if (j.contains("centerOrigin")) p.centerOrigin = j.at("centerOrigin").get<bool>();
    if (j.contains("useSoftPrior")) p.useSoftPrior = j.at("useSoftPrior").get<bool>();
    if (j.contains("defaultPriorSigma")) p.defaultPriorSigma = j.at("defaultPriorSigma").get<double>();
    p.validate();
    return p;
}

OperatorInfo getGlobalBundleAdjustmentCPUInfo() {
    return {"GlobalBundleAdjustmentCPU",
            SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

struct GlobalBundleAdjustmentCPU::Impl {
    GlobalBAParams params;
    GlobalBAStats  stats;
};

GlobalBundleAdjustmentCPU::GlobalBundleAdjustmentCPU(const GlobalBAParams& params)
    : pImpl_(std::make_unique<Impl>()) { params.validate(); pImpl_->params = params; }
GlobalBundleAdjustmentCPU::~GlobalBundleAdjustmentCPU() = default;

void GlobalBundleAdjustmentCPU::SetParams(const GlobalBAParams& params) {
    params.validate(); pImpl_->params = params;
}
const GlobalBAParams& GlobalBundleAdjustmentCPU::GetParams() const { return pImpl_->params; }
const GlobalBAStats& GlobalBundleAdjustmentCPU::GetStatistics() const noexcept { return pImpl_->stats; }
void GlobalBundleAdjustmentCPU::Destroy() {}

GlobalBAResult GlobalBundleAdjustmentCPU::Execute(const GlobalBAInput& input) {
    GlobalBAResult res;
    pImpl_->stats = GlobalBAStats{};  // 重置统计,避免多次调用累积陈旧状态
    if (input.frames.empty()) {
        res.message = "input frames empty";
        return res;
    }
    for (const auto& f : input.frames) {
        if (static_cast<int>(f.markerObs.size()) < pImpl_->params.minPointsPerFrame) {
            res.message = "frame " + std::to_string(f.frameId)
                        + " has fewer than minPointsPerFrame markers";
            return res;
        }
    }
    // 1) 为每帧建 globalId 集合(std::unordered_set<int>),存于 vector
    std::vector<std::unordered_set<int>> frameIds;
    frameIds.reserve(input.frames.size());
    for (const auto& f : input.frames) {
        std::unordered_set<int> s;
        for (const auto& obs : f.markerObs) {
            if (obs.globalId < 0) continue;  // 跳过未赋值的哨兵 globalId,避免误触发闭环
            s.insert(obs.globalId);
        }
        frameIds.push_back(std::move(s));
    }
    // 2) 检测闭环边:非相邻帧(|i-k|>loopFrameGap)共视 >= minCovisForLoopEdge
    bool loop = false;
    int n = static_cast<int>(input.frames.size());
    for (int i = 0; i < n && !loop; ++i) {
        for (int k = i + 1; k < n; ++k) {
            if (k - i <= pImpl_->params.loopFrameGap) continue;
            // 统计共视数
            int covis = 0;
            for (int gid : frameIds[i]) {
                if (frameIds[k].count(gid)) ++covis;
                if (covis >= pImpl_->params.minCovisForLoopEdge) break;
            }
            if (covis >= pImpl_->params.minCovisForLoopEdge) { loop = true; break; }
        }
    }
    pImpl_->stats.loopDetected = loop;
    // ===== 核心 Global Bundle Adjustment =====
    const auto& frames = input.frames;

    // 1) 收集唯一 globalId → 连续索引
    std::unordered_map<int, int> idToIdx;
    for (const auto& f : frames)
        for (const auto& obs : f.markerObs)
            if (obs.globalId >= 0 && !idToIdx.count(obs.globalId))
                idToIdx.emplace(obs.globalId, static_cast<int>(idToIdx.size()));
    const int nPoints = static_cast<int>(idToIdx.size());
    const auto nFrames = frames.size();

    // 退化问题守卫: 无任何有效观测(globalId>=0) → nPoints==0 → 无残差块, 直接失败
    if (nPoints == 0) {
        res.message = "no valid observations for GBA";
        return res;  // success=false
    }

    // 2) 参数存储(双精度):每帧 q[4] (w,x,y,z) + t[3];每点 X[3]
    std::vector<std::array<double, 4>> q(nFrames, {1, 0, 0, 0});
    std::vector<std::array<double, 3>> tt(nFrames, {0, 0, 0});
    std::vector<std::array<double, 3>> X(nPoints, {0, 0, 0});

    // 3) 初始化位姿:从 R_init/t_init
    for (size_t i = 0; i < nFrames; ++i) {
        Eigen::Matrix3d Rm;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) Rm(a, b) = frames[i].R_init(a, b);
        Eigen::Quaterniond qq(Rm);
        qq.normalize();
        q[i] = {qq.w(), qq.x(), qq.y(), qq.z()};
        tt[i] = {frames[i].t_init(0), frames[i].t_init(1), frames[i].t_init(2)};
    }
    // 初始化 X:对每点,用第一个观测它的帧反算 X = R*z + t
    std::vector<bool> xInit(nPoints, false);
    for (size_t i = 0; i < nFrames; ++i) {
        for (const auto& obs : frames[i].markerObs) {
            if (obs.globalId < 0) continue;
            int j = idToIdx[obs.globalId];
            if (xInit[j]) continue;
            cv::Vec3d z(obs.local.x, obs.local.y, obs.local.z);
            cv::Vec3d Xv = frames[i].R_init * z + frames[i].t_init;
            X[j] = {Xv(0), Xv(1), Xv(2)};
            xInit[j] = true;
        }
    }

    // ===== PGO: 闭环位姿图预优化(平滑增量链闭合处的位姿突变, 为 GBA 提供更优初值) =====
    //   仅在 loopDetected 且 enablePoseGraphPreopt 时执行。链式(loopDetected=false)直接进 GBA, 行为不变。
    //   流程: (1) 共视倒排索引 → (2) 对每对共视>=3 的 (i,k) Kabsch 求相对位姿测量 T_ik;
    //         (3) Ceres 位姿图优化(帧0锚定, 其余 QuaternionManifold);
    //         (4) 用精修后的 q/tt 重三角化所有 landmark(线性 LS 最小版, Task10 加固)。
    //   鲁棒性: 若 Ceres 硬失败, 还原 PGO 前的初值位姿并跳过重三角化, 退化为"无 PGO"路径,
    //           保证不劣化(不破坏既有行为)。
    if (pImpl_->stats.loopDetected && pImpl_->params.enablePoseGraphPreopt) {
        // 快照初值位姿(供 PGO 硬失败时回退)
        auto qSnap = q;
        auto ttSnap = tt;

        // (0) 每帧 globalId -> 本机系 local 映射(取共享点的成对坐标用)
        std::vector<std::unordered_map<int, cv::Point3d>> localById(nFrames);
        for (size_t i = 0; i < nFrames; ++i)
            for (const auto& obs : frames[i].markerObs)
                if (obs.globalId >= 0)
                    localById[i].emplace(obs.globalId, obs.local);

        // (1) 共视倒排索引: globalId -> 观测帧索引; 进而累计 (i,k) 共视数(i<k)
        std::unordered_map<int, std::vector<size_t>> gidToFrames;
        for (size_t i = 0; i < nFrames; ++i)
            for (const auto& obs : frames[i].markerObs)
                if (obs.globalId >= 0)
                    gidToFrames[obs.globalId].push_back(i);
        std::map<std::pair<size_t, size_t>, int> covisCount;
        for (auto& kv : gidToFrames) {
            const auto& fl = kv.second;
            for (size_t a = 0; a < fl.size(); ++a)
                for (size_t b = a + 1; b < fl.size(); ++b) {
                    size_t lo = std::min(fl[a], fl[b]), hi = std::max(fl[a], fl[b]);
                    if (lo != hi) ++covisCount[{lo, hi}];
                }
        }

        // (2) 构建共视边 + Kabsch 相对位姿测量(共视阈值 3, 小于闭环阈值 5, 便于铺满位姿图)
        struct PGOEdge { size_t i, k; Eigen::Matrix3d R_ik; Eigen::Vector3d t_ik; bool isLoop; };
        std::vector<PGOEdge> edges;
        const int pgoCovisThresh = 3;
        for (const auto& kv : covisCount) {
            if (kv.second < pgoCovisThresh) continue;
            size_t i = kv.first.first, k = kv.first.second;
            const auto& mi = localById[i];
            const auto& mk = localById[k];
            std::vector<cv::Point3d> zi, zk;
            const bool iSmaller = mi.size() <= mk.size();
            const auto& small = iSmaller ? mi : mk;
            const auto& big   = iSmaller ? mk : mi;
            for (const auto& entry : small) {
                auto it = big.find(entry.first);
                if (it == big.end()) continue;
                const cv::Point3d& pi = iSmaller ? entry.second : it->second;
                const cv::Point3d& pk = iSmaller ? it->second  : entry.second;
                zi.push_back(pi);
                zk.push_back(pk);
            }
            if (static_cast<int>(zi.size()) < pgoCovisThresh) continue;
            cv::Matx33d Rik; cv::Vec3d tik;
            // I2: Kabsch 退化保护 — 共线/近平面点集返回任意旋转, 会污染 PGO(无鲁棒损失)。
            //     退化时直接跳过该边(不加入相对位姿残差),宁缺勿滥。
            if (!kabsch(zi, zk, Rik, tik)) {
                spdlog::debug("[{}] PGO: skip degenerate Kabsch edge (i={},k={},shared={})",
                              GlobalBundleAdjustmentCPU::kLogTag, i, k, zi.size());
                continue;
            }
            Eigen::Matrix3d Rike;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) Rike(a, b) = Rik(a, b);
            bool isLoop = (static_cast<int>(k) - static_cast<int>(i)) > pImpl_->params.loopFrameGap;
            edges.push_back({i, k, Rike, Eigen::Vector3d(tik(0), tik(1), tik(2)), isLoop});
        }

        bool pgoOk = true;
        if (!edges.empty()) {
            // (2.5) I3 运行时告警: vee 形式只在 |θ|≲~0.3 rad 内可信(见 pose_graph_residuals.h)。
            //   求解前用各边的初始位姿算旋转残差模(未加权, 单位 rad), 取最大值。
            //   超 0.3 rad 说明该闭环初始旋转漂移过大, vee 近似可能失真, PGO 会欠矫正。
            //   仅告警不中止: PGO+GBA 是 best-effort, 精密收敛由 GBA plain-LM 兜底。
            {
                double maxRotRes = 0.0;
                for (const auto& e : edges) {
                    Eigen::Quaterniond Qi(q[e.i][0], q[e.i][1], q[e.i][2], q[e.i][3]);
                    Eigen::Quaterniond Qk(q[e.k][0], q[e.k][1], q[e.k][2], q[e.k][3]);
                    Qi.normalize(); Qk.normalize();
                    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                    Eigen::Matrix3d Rk = Qk.toRotationMatrix();
                    Eigen::Matrix3d Rerr = e.R_ik.transpose() * (Ri.transpose() * Rk);
                    double rn = so3Log(Rerr).norm();  // vee 形式, |sin(θ)·axis|
                    if (rn > maxRotRes) maxRotRes = rn;
                }
                if (maxRotRes > 0.3) {
                    spdlog::warn("[{}] PGO: max pre-solve rotation residual {:.4f} rad > 0.3 rad; "
                                 "vee-form rotation model may be inaccurate for this loop's drift "
                                 "(PGO proceeds, GBA does the precision)",
                                 GlobalBundleAdjustmentCPU::kLogTag, maxRotRes);
                }
            }
            // (3) Ceres 位姿图优化: 帧0锚定(6-DoF gauge), 其余帧加 QuaternionManifold
            ceres::Problem pgoProblem;
            pgoProblem.AddParameterBlock(q[0].data(), 4);
            pgoProblem.SetParameterBlockConstant(q[0].data());
            pgoProblem.AddParameterBlock(tt[0].data(), 3);
            pgoProblem.SetParameterBlockConstant(tt[0].data());
            for (size_t i = 1; i < nFrames; ++i) {
                pgoProblem.AddParameterBlock(q[i].data(), 4);
                pgoProblem.SetManifold(q[i].data(), new ceres::QuaternionManifold);
            }
            // 边权: 旋转残差(rad, ~1e-2)与平移残差(mm, ~1)尺度差 ~100x,
            //   给旋转较大权重使其与平移贡献可比。PGO 目标是平滑而非精密(精密由 GBA 完成)。
            const double rotW = 50.0;
            const double transW = 1.0;
            for (const auto& e : edges) {
                auto* cost = new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 4, 3, 4, 3>(
                    new RelativePoseResidual(e.R_ik, e.t_ik, rotW, transW));
                pgoProblem.AddResidualBlock(cost, nullptr,
                    q[e.i].data(), tt[e.i].data(), q[e.k].data(), tt[e.k].data());
            }
            ceres::Solver::Options opt;
            opt.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            opt.max_num_iterations = 100;
            opt.function_tolerance = 1e-9;
            opt.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary sum;
            ceres::Solve(opt, &pgoProblem, &sum);
            const bool hardFail = (sum.termination_type == ceres::FAILURE ||
                                   sum.termination_type == ceres::USER_FAILURE);
            spdlog::info("[{}] PGO: {} edges, {} iters, term={}, cost {:.6e}->{:.6e}",
                         GlobalBundleAdjustmentCPU::kLogTag, edges.size(),
                         static_cast<int>(sum.iterations.size()),
                         ceres::TerminationTypeToString(sum.termination_type),
                         sum.initial_cost, sum.final_cost);

            if (hardFail) {
                // 回退: 还原初值位姿, 后续按"无 PGO"路径进 GBA
                spdlog::warn("[{}] PGO hard-failed (term={}); revert to pre-PGO init poses",
                             GlobalBundleAdjustmentCPU::kLogTag,
                             ceres::TerminationTypeToString(sum.termination_type));
                q = qSnap;
                tt = ttSnap;
                pgoOk = false;
            } else {
                // 闭环残差统计: 闭环边上 6 维相对位姿残差的 RMS(混合尺度, 用于诊断)
                double loopSumSq = 0.0; long long loopCnt = 0;
                for (const auto& e : edges) {
                    if (!e.isLoop) continue;
                    double res[6];
                    RelativePoseResidual(e.R_ik, e.t_ik, rotW, transW)
                        (q[e.i].data(), tt[e.i].data(), q[e.k].data(), tt[e.k].data(), res);
                    for (int d = 0; d < 6; ++d) { loopSumSq += res[d] * res[d]; ++loopCnt; }
                }
                pImpl_->stats.loopClosureResidual =
                    (loopCnt > 0) ? std::sqrt(loopSumSq / static_cast<double>(loopCnt)) : 0.0;
            }
        }

        // (4) 重三角化所有 landmark(线性 LS 最小版, 用 PGO 精修后的 q/tt):
        //       模型 z = R_iᵀ(X − t_i); 关于 X 的正规方程: (Σ R_i R_iᵀ) X = Σ (t_i + R_i z_i)
        //     (R_i R_iᵀ ≈ I, 故 ≈ 各观测单帧反算的均值; 用 ldlt 数值稳定求解)
        if (pgoOk) {
            std::vector<Eigen::Matrix3d> Ajj(nPoints, Eigen::Matrix3d::Zero());
            std::vector<Eigen::Vector3d> bjj(nPoints, Eigen::Vector3d::Zero());
            std::vector<int> obsPerPt(nPoints, 0);
            for (size_t i = 0; i < nFrames; ++i) {
                Eigen::Quaterniond Qi(q[i][0], q[i][1], q[i][2], q[i][3]);
                Qi.normalize();
                Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                Eigen::Vector3d ti(tt[i][0], tt[i][1], tt[i][2]);
                for (const auto& obs : frames[i].markerObs) {
                    if (obs.globalId < 0) continue;
                    int j = idToIdx[obs.globalId];
                    Eigen::Vector3d z(obs.local.x, obs.local.y, obs.local.z);
                    Ajj[j] += Ri * Ri.transpose();
                    bjj[j] += ti + Ri * z;
                    ++obsPerPt[j];
                }
            }
            for (int j = 0; j < nPoints; ++j) {
                if (obsPerPt[j] < 1) continue;  // 保留原 X(单帧反算值)
                Eigen::Vector3d Xsolved = Ajj[j].ldlt().solve(bjj[j]);
                X[j] = {Xsolved(0), Xsolved(1), Xsolved(2)};
            }
        }
    }

    // 4) centerOrigin: 把全局原点平移到初始点质心(改善数值条件,防御大坐标下的灾难性抵消)
    //    local 观测对整体平移不变: Rᵀ((X−c)−(t−c)) = Rᵀ(X−t), 故只需同时平移 X 和 t。
    //    帧0锚点变为 t_init[0] − c, 仍是合法的 6-DoF gauge。
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    const bool applyCenter = pImpl_->params.centerOrigin;
    if (applyCenter) {
        for (int j = 0; j < nPoints; ++j)
            centroid += Eigen::Vector3d(X[j][0], X[j][1], X[j][2]);
        if (nPoints > 0) centroid /= static_cast<double>(nPoints);
        for (int j = 0; j < nPoints; ++j)
            for (int k = 0; k < 3; ++k) X[j][k] -= centroid(k);
        for (size_t i = 0; i < nFrames; ++i)
            for (int k = 0; k < 3; ++k) tt[i][k] -= centroid(k);
    }

    // ===== P4-T24 高精度已有點软先验（客户端扫描流水线.md §5.3）=====
    // 对 highPrecisionGlobalIds 命中的点 X 加残差 ‖X−X_existing‖²/σ²（σ 极小=高权重）,
    // 防止已有点被扫描观测挪动。useSoftPrior=false 或 ids 空 → 不加任何残差块, 行为与现状一致。
    // 鲁棒性: id 无对应点 / X_existing 长度不足 / σ 非法 → 忽略该 id 并 warn（不 fail）。
    // 注: X_existing 在原始全局系, 而优化中 X 已按 centerOrigin 平移 → 先验位置同步减质心。
    // 升级路径（仅注释未实现）: 某子集需绝对固定时, 对该子集参数块改用
    //   problem.SetParameterBlockConstant(X[pointIdx].data()) 替代本软先验。
    struct PriorEntry { int pointIdx; double Xe[3]; double sigma; };
    std::vector<PriorEntry> priorEntries;
    if (pImpl_->params.useSoftPrior && !input.highPrecisionGlobalIds.empty()) {
        const auto& priorIds = input.highPrecisionGlobalIds;
        const auto& xeAll    = input.X_existing;
        const auto& sgAll    = input.priorSigma;
        for (size_t i = 0; i < priorIds.size(); ++i) {
            auto it = idToIdx.find(priorIds[i]);
            if (it == idToIdx.end()) {
                spdlog::warn("[{}] soft prior: globalId {} has no matching point in scene; ignored",
                             GlobalBundleAdjustmentCPU::kLogTag, priorIds[i]);
                continue;
            }
            if (xeAll.size() < 3 * (i + 1)) {
                spdlog::warn("[{}] soft prior: X_existing too short ({} < 3*{}); globalId {} ignored",
                             GlobalBundleAdjustmentCPU::kLogTag, xeAll.size(), i + 1, priorIds[i]);
                continue;
            }
            double sigma = pImpl_->params.defaultPriorSigma;
            if (i < sgAll.size() && sgAll[i] > 0.0) {
                sigma = sgAll[i];
            } else if (i < sgAll.size()) {
                spdlog::warn("[{}] soft prior: priorSigma[{}] <= 0; fallback to default {}",
                             GlobalBundleAdjustmentCPU::kLogTag, i,
                             pImpl_->params.defaultPriorSigma);
            }
            priorEntries.push_back({it->second,
                xeAll[3 * i]     - (applyCenter ? centroid(0) : 0.0),
                xeAll[3 * i + 1] - (applyCenter ? centroid(1) : 0.0),
                xeAll[3 * i + 2] - (applyCenter ? centroid(2) : 0.0),
                sigma});
        }
        if (!priorEntries.empty()) {
            spdlog::info("[{}] soft prior: {} high-precision point(s) anchored",
                         GlobalBundleAdjustmentCPU::kLogTag, priorEntries.size());
        }
    }

    // 5) RMSE 计算:遍历所有观测,算 r = R(q_i)ᵀ(X_j − t_i) − z 的范数
    //    读取当前 q/tt/X 数组,故 Solve 前调用得 initialRMSE,Solve 后得 finalRMSE
    // 观测扁平化 + 外点簿记(三层外点处理: 预清洗 + Tukey + 卡方剔除再收敛)
    // 每个有效观测(globalId>=0)分配全局观测索引 = 其在 obsInfos 中的位置;
    // culled 一旦置真, 后续 阶段2建题/computeRMSE/卡方剔除 均跳过该观测。
    struct ObsInfo {
        size_t frameIdx;
        size_t obsIdx;
        int    pointIdx;
        bool   culled = false;
    };
    std::vector<ObsInfo> obsInfos;
    obsInfos.reserve(nFrames * 4);
    for (size_t i = 0; i < nFrames; ++i)
        for (size_t k = 0; k < frames[i].markerObs.size(); ++k)
            if (frames[i].markerObs[k].globalId >= 0)
                obsInfos.push_back({i, k, idToIdx[frames[i].markerObs[k].globalId], false});

    // 单观测残差(原始 mm, 未缩放): r = R(q_i)^T (X_j - t_i) - z
    auto residualVec = [&](const ObsInfo& o) -> Eigen::Vector3d {
        const auto& obs = frames[o.frameIdx].markerObs[o.obsIdx];
        Eigen::Quaterniond Qi(q[o.frameIdx][0], q[o.frameIdx][1], q[o.frameIdx][2], q[o.frameIdx][3]);
        Qi.normalize();
        Eigen::Vector3d ti(tt[o.frameIdx][0], tt[o.frameIdx][1], tt[o.frameIdx][2]);
        Eigen::Vector3d Xj(X[o.pointIdx][0], X[o.pointIdx][1], X[o.pointIdx][2]);
        Eigen::Vector3d z(obs.local.x, obs.local.y, obs.local.z);
        return Qi.conjugate() * (Xj - ti) - z;
    };

    // RMSE: 仅遍历"未被剔除"的观测。读取当前 q/tt/X, 故 Solve 前得 initialRMSE,
    // 收敛后(飞点已剔除)得 finalRMSE。残差对整体平移不变, centerOrigin 下数值等价。
    auto computeRMSE = [&]() {
        double sumSq = 0.0;
        long long nobs = 0;
        for (const auto& o : obsInfos) {
            if (o.culled) continue;
            Eigen::Vector3d r = residualVec(o);
            sumSq += r.squaredNorm();
            ++nobs;
        }
        return nobs > 0 ? std::sqrt(sumSq / static_cast<double>(nobs)) : 0.0;
    };
    pImpl_->stats.initialRMSE = computeRMSE();

    // 6) 求解:两阶段 Graduated Robust BA
    //   初值位姿含扰动,残差可达 ~1mm 量级,远大于 σ(0.01mm)。若阶段一直接上
    //   Tukey(c=3σ=0.03mm),所有残差被 Tukey 完全压制(权重→0),梯度为零,
    //   Ceres 零步"收敛"无法下降。故:
    //   阶段1(plain):无鲁棒损失,普通 LM 把残差从初值量级下降到 ~噪声底;
    //   阶段2(Tukey):残差已 ~σ,此时 c=3σ 才正确发挥离群点剔除作用并定稿。
    const double invSigma = 1.0 / pImpl_->params.sigmaObserved;
    const double preCleanGate = 5.0 * pImpl_->params.sigmaObserved;  // ||r|| > 5sigma 视为粗差飞点
    const double tukeyC_dimless = pImpl_->params.tukeyC * invSigma;  // 无量纲 c = 3σ/σ = 3

    struct PhaseResult {
        int iters = 0;
        ceres::TerminationType term = ceres::CONVERGENCE;
    };
    auto solvePhase = [&](bool useTukey) -> PhaseResult {
        ceres::Problem problem;
        // 帧0锚定:q0/t0 设为常量(6-DoF gauge)
        problem.AddParameterBlock(q[0].data(), 4);
        problem.SetParameterBlockConstant(q[0].data());
        problem.AddParameterBlock(tt[0].data(), 3);
        problem.SetParameterBlockConstant(tt[0].data());
        // 其余帧加 QuaternionManifold(参数布局 [w,x,y,z], 实部在前, 匹配本存储)
        for (size_t i = 1; i < nFrames; ++i) {
            problem.AddParameterBlock(q[i].data(), 4);
            problem.SetManifold(q[i].data(), new ceres::QuaternionManifold);
        }
        for (const auto& o : obsInfos) {
            if (o.culled) continue;  // 已剔除(预清洗/卡方)的观测不进入问题
            const auto& obs = frames[o.frameIdx].markerObs[o.obsIdx];
            Eigen::Vector3d z(obs.local.x, obs.local.y, obs.local.z);
            auto* cost = new ceres::AutoDiffCostFunction<PointPairResidual, 3, 4, 3, 3>(
                new PointPairResidual(z, pImpl_->params.sigmaObserved));
            ceres::LossFunction* loss = useTukey
                ? static_cast<ceres::LossFunction*>(new ceres::TukeyLoss(tukeyC_dimless))
                : nullptr;
            problem.AddResidualBlock(cost, loss,
                                     q[o.frameIdx].data(), tt[o.frameIdx].data(),
                                     X[o.pointIdx].data());
        }
        // P4-T24: 高精度已有點软先验残差块 r = (X − X_existing)/σ。
        // 不加鲁棒损失（先验可信, 不被 Tukey 压制）; 每阶段求解(plain/Tukey/重解)均重建题, 故每相都加。
        // 升级路径（仅注释未实现）: 需绝对固定时改 problem.SetParameterBlockConstant(X[pe.pointIdx].data())。
        for (const auto& pe : priorEntries) {
            auto* priorCost = new ceres::DynamicAutoDiffCostFunction<MarkerPriorCost>(
                new MarkerPriorCost{{pe.Xe[0], pe.Xe[1], pe.Xe[2]}, pe.sigma});
            priorCost->AddParameterBlock(3);
            priorCost->SetNumResiduals(3);
            problem.AddResidualBlock(priorCost, nullptr, X[pe.pointIdx].data());
        }
        ceres::Solver::Options opt;
        opt.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        opt.function_tolerance = pImpl_->params.tolerance;
        opt.gradient_tolerance = pImpl_->params.tolerance;
        opt.parameter_tolerance = pImpl_->params.tolerance;
        opt.max_num_iterations = pImpl_->params.maxIterations;
        opt.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(opt, &problem, &summary);
        // 收敛/用户成功 → info; 否则(未收敛/硬失败)→ warn, 便于排查退化解
        const char* phase = useTukey ? "Tukey" : "plain";
        const bool ok = summary.termination_type == ceres::CONVERGENCE
                     || summary.termination_type == ceres::USER_SUCCESS;
        if (ok) {
            spdlog::info("[{}] Ceres phase({}): {} iters, term={}, cost {:.6e}->{:.6e}",
                         GlobalBundleAdjustmentCPU::kLogTag, phase,
                         summary.iterations.size(),
                         ceres::TerminationTypeToString(summary.termination_type),
                         summary.initial_cost, summary.final_cost);
        } else {
            spdlog::warn("[{}] Ceres phase({}): {} iters, term={}, cost {:.6e}->{:.6e}",
                         GlobalBundleAdjustmentCPU::kLogTag, phase,
                         summary.iterations.size(),
                         ceres::TerminationTypeToString(summary.termination_type),
                         summary.initial_cost, summary.final_cost);
        }
        return { static_cast<int>(summary.iterations.size()), summary.termination_type };
    };
    auto r1 = solvePhase(false);

    // ===== Layer 1 预清洗: 阶段1(plain LM, 含全部观测)给出残差估计; ||r||>gate 者为粗差飞点 =====
    //   关键: 阶段1 会被粗差飞点污染——好观测残差可达 ~0.1mm(>>5sigma=0.05mm), 纯 5sigma 门限会
    //   误删大量好观测。故门限取 max(5sigma, k*median||r||) 自适应到当前残差尺度: 好观测(~median)
    //   保留, 飞点(~10x median)剔除。剔除后 plain 重解即可把好观测拉回 ~sigma(见下)。
    double preCleanThresh = preCleanGate;
    {
        std::vector<double> nr;
        nr.reserve(obsInfos.size());
        for (const auto& o : obsInfos)
            if (!o.culled) nr.push_back(residualVec(o).norm());
        if (!nr.empty()) {
            std::sort(nr.begin(), nr.end());
            double med = nr[nr.size() / 2];
            double adaptive = 5.0 * med;  // 飞点典型 ~10x median, 5x median 落在好/坏间隙
            if (adaptive > preCleanThresh) preCleanThresh = adaptive;
        }
    }
    bool anyPre = false;
    for (size_t idx = 0; idx < obsInfos.size(); ++idx) {
        if (obsInfos[idx].culled) continue;
        double nr = residualVec(obsInfos[idx]).norm();
        if (nr > preCleanThresh) {
            obsInfos[idx].culled = true;
            pImpl_->stats.outlierObsIds.push_back(static_cast<int>(idx));
            anyPre = true;
            spdlog::info("[{}] pre-clean: cull obs #{} (frame{},obs{}) ||r||={:.4f}>{:.4f}mm",
                         GlobalBundleAdjustmentCPU::kLogTag, idx,
                         obsInfos[idx].frameIdx, obsInfos[idx].obsIdx, nr, preCleanThresh);
        }
    }

    // 预清洗后 plain 重解: 移除飞点后好观测不再被污染, plain LM 即可恢复到 ~sigma。
    //   (Tukey c=3sigma 无法从 ~0.1mm 恢复: 0.1mm>>c 把所有好观测降权→零梯度→卡住, 故必须 plain 重解。)
    PhaseResult rRec;
    if (anyPre) rRec = solvePhase(false);

    // ===== Layer 2: Tukey(阶段2)精修(无飞点的好观测集, 残差 ~sigma < c, 正常收敛) =====
    PhaseResult r2 = solvePhase(true);

    // ===== Layer 3 卡方剔除 + 重解一次: chi2=||r/sigma||^2 > chiSquareGate 者剔除 =====
    //   此时好观测 ~sigma(chi2~1), 漏网飞点 chi2>>gate; 绝对门限 11.34(3-DoF 99%) 正确生效。
    PhaseResult r3;
    bool anyCulled = false;
    for (size_t idx = 0; idx < obsInfos.size(); ++idx) {
        if (obsInfos[idx].culled) continue;  // 预清洗已剔除
        double chi2 = (residualVec(obsInfos[idx]) * invSigma).squaredNorm();
        if (chi2 > pImpl_->params.chiSquareGate) {
            pImpl_->stats.outlierObsIds.push_back(static_cast<int>(idx));
            obsInfos[idx].culled = true;
            anyCulled = true;
            spdlog::info("[{}] chi2-cull: cull obs #{} (frame{},obs{}) chi2={:.3f}>{:.3f}",
                         GlobalBundleAdjustmentCPU::kLogTag, idx,
                         obsInfos[idx].frameIdx, obsInfos[idx].obsIdx,
                         chi2, pImpl_->params.chiSquareGate);
        }
    }
    if (anyCulled) r3 = solvePhase(true);

    pImpl_->stats.ceresIterations = r1.iters + rRec.iters + r2.iters + r3.iters;

    // ===== 求解失败处理 =====
    res.qualityFlag = QualityFlag::Normal;
    auto isHardFail = [](ceres::TerminationType t) {
        return t == ceres::FAILURE || t == ceres::USER_FAILURE;
    };
    // 硬失败(Ceres 报错, 参数块未更新): 不输出可能退化的几何, 直接失败
    if (isHardFail(r1.term)) {
        res.message = "GBA phase 1 (plain LM) failed: "
                    + std::string(ceres::TerminationTypeToString(r1.term));
        return res;  // success=false
    }
    if (isHardFail(r2.term)) {
        res.message = "GBA phase 2 (Tukey) failed: "
                    + std::string(ceres::TerminationTypeToString(r2.term));
        return res;  // success=false
    }
    if (anyPre && isHardFail(rRec.term)) {
        res.message = "GBA re-solve (post pre-clean) failed: "
                    + std::string(ceres::TerminationTypeToString(rRec.term));
        return res;  // success=false
    }
    if (anyCulled && isHardFail(r3.term)) {
        res.message = "GBA re-solve (post chi2-cull) failed: "
                    + std::string(ceres::TerminationTypeToString(r3.term));
        return res;  // success=false
    }
    // 未收敛(非硬失败, 仍更新了最佳参数): 保留结果但标记 Degraded
    const bool nonConv1 = (r1.term == ceres::NO_CONVERGENCE);
    const bool nonConv2 = (r2.term == ceres::NO_CONVERGENCE);
    const bool nonConvRec = anyPre && (rRec.term == ceres::NO_CONVERGENCE);
    const bool nonConv3 = anyCulled && (r3.term == ceres::NO_CONVERGENCE);
    if (nonConv1 || nonConv2 || nonConvRec || nonConv3) {
        res.qualityFlag = QualityFlag::Degraded;
        res.message = "GBA did not fully converge";
        if (nonConv1) res.message += " (phase1 NO_CONVERGENCE)";
        if (nonConv2) res.message += " (phase2 NO_CONVERGENCE)";
        if (nonConvRec) res.message += " (pre-clean re-solve NO_CONVERGENCE)";
        if (nonConv3) res.message += " (re-solve NO_CONVERGENCE)";
    }

    // 7) finalRMSE(优化后,仍在归一化坐标系下;RMSE 对整体平移不变,数值等价)
    pImpl_->stats.finalRMSE = computeRMSE();

    // 反 centerOrigin:结果加回质心,使输出回到原始全局坐标系
    if (applyCenter) {
        for (size_t i = 0; i < nFrames; ++i)
            for (int k = 0; k < 3; ++k) tt[i][k] += centroid(k);
        for (int j = 0; j < nPoints; ++j)
            for (int k = 0; k < 3; ++k) X[j][k] += centroid(k);
    }

    // 8) 填充输出
    res.optimizedPoses.resize(nFrames);
    for (size_t i = 0; i < nFrames; ++i) {
        Eigen::Quaterniond qq(q[i][0], q[i][1], q[i][2], q[i][3]);
        qq.normalize();
        Eigen::Matrix3d Rm = qq.toRotationMatrix();
        cv::Matx33d R;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) R(a, b) = Rm(a, b);
        res.optimizedPoses[i] = {frames[i].frameId, R,
                                 cv::Vec3d(tt[i][0], tt[i][1], tt[i][2])};
    }
    res.optimizedMarkers.resize(nPoints);
    // 统计每点共视数(covisCount)
    std::vector<int> covisCount(nPoints, 0);
    for (size_t i = 0; i < nFrames; ++i)
        for (const auto& obs : frames[i].markerObs)
            if (obs.globalId >= 0) ++covisCount[idToIdx[obs.globalId]];
    for (auto& kv : idToIdx) {
        int j = kv.second;
        res.optimizedMarkers[j] = {cv::Point3d(X[j][0], X[j][1], X[j][2]), kv.first,
                                   covisCount[j]};
    }
    res.statistics = pImpl_->stats;
    res.success = true;
    return res;
}

} // namespace calib
