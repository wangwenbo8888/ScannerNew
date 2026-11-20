// ============================================================================
// global_optim_stage.cpp — 回放收尾全局优化（GBA）阶段实现【2026-11-16 方案定稿】
// ============================================================================
// 定稿依据：归档/记录文档/方案分析-全局优化接入-2026-11-16.md §三·终。
// 门禁口径（R1C 五条之前两条，本文件自动评估）：
//   门① RMSE：final≤0.5×initial 且 ≤0.15mm；<0.01mm 反判"吸噪声过拟合"（fail）
//   门② 位姿修正分布（aligned 口径）：中位 Δt≤2mm｜P95≤10mm｜max<50mm｜
//          中位 Δrot≤0.5°｜P95≤2°｜max<10°（锚＝现有 rtDtMmMax/rtDrotDegMax 硬门）
// 门③（帧集合⊆可融合帧）由装配结构性保证＋调用方交叉核对；门④（点云对照）
// 与门⑤（时长）由调用方/验收脚本执行。
#include "global_optim_stage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "quality_flag.h"  // ops/core/common（qualityFlagToString）

namespace scr {
namespace {

double medianOf(std::vector<double> v) {   // 拷贝排序；空返回 0
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double p95Of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(0.95 * (v.size() - 1)))];
}
double maxOf(std::vector<double> v) {
    if (v.empty()) return 0;
    return *std::max_element(v.begin(), v.end());
}

/// dR = Rb^T·Ra 的旋转角（deg）
double rotAngleDeg(const cv::Matx33d& Rb, const cv::Matx33d& Ra) {
    const cv::Matx33d dR = Rb.t() * Ra;
    const double tr = dR(0, 0) + dR(1, 1) + dR(2, 2);
    const double c = std::min(1.0, std::max(-1.0, (tr - 1.0) * 0.5));
    return std::acos(c) * 180.0 / CV_PI;
}

GbaDeltaStats statsFromPairs(const std::vector<std::pair<GbaPose, GbaPose>>& pairs,
                             const cv::Matx33d& Rg, const cv::Vec3d& tg) {
    GbaDeltaStats s;
    if (pairs.empty()) return s;
    std::vector<double> dt, dr;
    dt.reserve(pairs.size());
    dr.reserve(pairs.size());
    for (const auto& [b, a] : pairs) {
        const cv::Vec3d ta = Rg * a.T + tg;      // after 对齐后平移
        dt.push_back(cv::norm(ta - b.T));        // mm（输入本位姿系即 mm）
        dr.push_back(rotAngleDeg(b.R, Rg * a.R));
    }
    s.medDt = medianOf(dt);  s.p95Dt = p95Of(dt);  s.maxDt = maxOf(dt);
    s.medDr = medianOf(dr);  s.p95Dr = p95Of(dr);  s.maxDr = maxOf(dr);
    s.valid = true;
    return s;
}

nlohmann::json statsJson(const GbaDeltaStats& s) {
    return {{"median_dt_mm", s.medDt}, {"p95_dt_mm", s.p95Dt}, {"max_dt_mm", s.maxDt},
            {"median_drot_deg", s.medDr}, {"p95_drot_deg", s.p95Dr},
            {"max_drot_deg", s.maxDr}, {"valid", s.valid}};
}

}  // namespace

// ----------------------------------------------------------------------------
// 装配（纯函数）
// ----------------------------------------------------------------------------
calib::GlobalBAInput assembleGbaInput(const std::vector<GbaObsRecord>& obs,
                                      const calib::GlobalBAParams& params,
                                      std::vector<uint64_t>& excludedFrames) {
    excludedFrames.clear();
    calib::GlobalBAInput input;
    input.frames.reserve(obs.size());
    for (const auto& rec : obs) {
        calib::GlobalBAFrame f;
        f.frameId = rec.frameId;
        f.R_init = cv::Matx33d(rec.R[0], rec.R[1], rec.R[2],
                               rec.R[3], rec.R[4], rec.R[5],
                               rec.R[6], rec.R[7], rec.R[8]);
        f.t_init = cv::Vec3d(rec.T[0], rec.T[1], rec.T[2]);
        const size_t n = std::min(rec.local.size(), rec.gids.size());
        f.markerObs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (rec.gids[i] < 0) continue;   // 候选期点不入图（R3I 边角①）
            f.markerObs.push_back({rec.local[i], rec.gids[i]});
        }
        if (static_cast<int>(f.markerObs.size()) < params.minPointsPerFrame) {
            excludedFrames.push_back(rec.frameId);   // 观测不足整帧剔除（记档不静默）
            continue;
        }
        input.frames.push_back(std::move(f));
    }
    // 软先验恒空＝行为与无先验一致（globalSet_ 本身是链式漂移产物，不作先验）
    return input;
}

// ----------------------------------------------------------------------------
// 分布统计＋规约对齐
// ----------------------------------------------------------------------------
GbaDeltaStats poseDeltaStats(const std::vector<std::pair<GbaPose, GbaPose>>& pairs) {
    cv::Matx33d Rg = cv::Matx33d::eye();
    cv::Vec3d tg(0, 0, 0);
    return statsFromPairs(pairs, Rg, tg);   // raw＝恒等对齐
}

void gaugeAlign(const std::vector<std::pair<GbaPose, GbaPose>>& pairs,
                cv::Matx33d& Rg, cv::Vec3d& tg) {
    Rg = cv::Matx33d::eye();
    tg = cv::Vec3d(0, 0, 0);
    if (pairs.size() < 3) return;
    // —— 旋转规约：R_b·R_a^T 的弦平均（Chordal L2 mean）——
    // 目标 Rg = argmax Σ trace(Rg^T·M_i)，M_i = R_b·R_a^T；ΣM = U·S·V^T 时
    // 最优解 Rg = U·V^T（⚠ 与平移 Umeyama 的 V·U^T 相反——2026-11-16 实测
    // 踩坑：写反＝规约反向，对齐统计反而恶化，单测 GaugeAlign 抓出）。
    cv::Matx33d Rsum = cv::Matx33d::zeros();
    for (const auto& [b, a] : pairs) Rsum += b.R * a.R.t();
    cv::Mat Rm(Rsum);
    cv::SVD svd(Rm, cv::SVD::FULL_UV);
    cv::Matx33d U, Vt;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            U(r, c) = svd.u.at<double>(r, c);
            Vt(r, c) = svd.vt.at<double>(r, c);
        }
    Rg = U * Vt;
    if (cv::determinant(Rg) < 0) {   // 反射修正（罕见；保 SO(3)）
        cv::Matx33d Ufix = U;
        Ufix(0, 2) *= -1; Ufix(1, 2) *= -1; Ufix(2, 2) *= -1;
        Rg = Ufix * Vt;
    }
    // —— 平移规约：在旋转规约 Rg 消除后的平移点集上闭式求解 ——
    // 对齐目标 min Σ||Rg·t_a + tg − t_b||² → tg = mean(t_b) − mean(Rg·t_a)
    cv::Vec3d ca(0, 0, 0), cb(0, 0, 0);
    for (const auto& [b, a] : pairs) { ca += Rg * a.T; cb += b.T; }
    ca *= 1.0 / pairs.size();
    cb *= 1.0 / pairs.size();
    tg = cb - ca;
}

// ----------------------------------------------------------------------------
// 双轨迹 PLY
// ----------------------------------------------------------------------------
bool writeGbaTrajectoryPly(const std::string& path,
                           const std::vector<GbaPose>& before,
                           const std::vector<GbaPose>& after) {
    const size_t nb = before.size(), na = after.size();
    if (nb < 2 || na < 2) return false;
    const size_t edges = (nb - 1) + (na - 1);
    std::ofstream f(path);
    if (!f) return false;
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << (nb + na) << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "element edge " << edges << "\n";
    f << "property int vertex1\nproperty int vertex2\nend_header\n";
    f << std::fixed << std::setprecision(3);
    for (const auto& p : before)
        f << p.T(0) << ' ' << p.T(1) << ' ' << p.T(2) << " 255 0 0\n";   // 优化前＝红
    for (const auto& p : after)
        f << p.T(0) << ' ' << p.T(1) << ' ' << p.T(2) << " 0 255 0\n";   // 优化后＝绿
    for (size_t i = 0; i + 1 < nb; ++i) f << i << ' ' << (i + 1) << '\n';
    for (size_t i = 0; i + 1 < na; ++i)
        f << (nb + i) << ' ' << (nb + i + 1) << '\n';
    return true;
}

// ----------------------------------------------------------------------------
// 阶段核心
// ----------------------------------------------------------------------------
GbaStageResult runGbaCore(const std::vector<GbaObsRecord>& obs,
                          const std::vector<GbaLaserRecord>* laser,
                          const calib::GlobalBAParams& params,
                          const GbaStageOptions& opts) {
    const auto t0 = std::chrono::steady_clock::now();
    GbaStageResult out;
    std::vector<std::pair<GbaPose, GbaPose>> pairs;   // (before, after)
    std::vector<GbaPose> trajBefore, trajAfter;

    // —— 装配 ——
    calib::GlobalBAInput input = assembleGbaInput(obs, params, out.excludedFrames);
    out.framesSent = static_cast<int>(input.frames.size());
    if (input.frames.size() < 2) {
        out.executed = false;
        out.message = "有效观测帧不足（<2）：obs=" + std::to_string(obs.size()) +
                      " sent=" + std::to_string(out.framesSent);
        return out;
    }
    out.executed = true;

    // —— 求解（纯 CPU；maxSolveSecondsPerPhase 兜底黑盒）——
    calib::GlobalBundleAdjustmentCPU op(params);
    calib::GlobalBAResult res = op.Execute(input);
    out.initialRMSE = res.statistics.initialRMSE;
    out.finalRMSE = res.statistics.finalRMSE;
    out.framesKept = static_cast<int>(res.optimizedPoses.size());
    if (!res.success) {
        out.message = std::string("GBA 失败/降级：") + res.message + "（qualityFlag=" +
                      calib::qualityFlagToString(res.qualityFlag) + "）";
        return out;   // 不做 poses/门评估；重融合由调用方据 success=false 跳过
    }
    out.success = true;

    // —— 位姿映射＋分布统计（raw＋规约对齐双口径）——
    for (const auto& fp : res.optimizedPoses) out.poses[fp.frameId] = {fp.R, fp.t};
    std::map<uint64_t, const calib::GlobalBAFrame*> byId;
    for (const auto& f : input.frames) byId[f.frameId] = &f;
    for (const auto& [fid, pose] : out.poses) {
        auto it = byId.find(fid);
        if (it == byId.end()) continue;
        pairs.emplace_back(GbaPose{it->second->R_init, it->second->t_init}, pose);
        trajBefore.emplace_back(it->second->R_init, it->second->t_init);
        trajAfter.emplace_back(pose);
    }
    out.deltaRaw = poseDeltaStats(pairs);
    cv::Matx33d Rg;
    cv::Vec3d tg;
    gaugeAlign(pairs, Rg, tg);
    out.deltaAligned = statsFromPairs(pairs, Rg, tg);

    // —— 门①②（R1C 口径）——
    const bool rmseHalved = out.finalRMSE <= 0.5 * out.initialRMSE;
    const bool rmseInBand = out.finalRMSE <= 0.15 && out.finalRMSE >= 0.01;
    const bool overfitSuspect = out.finalRMSE < 0.01;   // 反判：吸噪声（产线教训）
    out.rmseGate = rmseHalved && rmseInBand;
    const auto& a = out.deltaAligned;
    out.poseGate = a.medDt <= 2.0 && a.p95Dt <= 10.0 && a.maxDt < 50.0 &&
                   a.medDr <= 0.5 && a.p95Dr <= 2.0 && a.maxDr < 10.0;

    // —— 报告 json ——
    out.report = {
        {"stage", "global_optim"},
        {"framesObs", obs.size()},
        {"framesSent", out.framesSent},
        {"framesKept", out.framesKept},
        {"excludedFrames", out.excludedFrames},
        {"laserFrames", laser ? static_cast<int>(laser->size()) : -1},
        {"rmse",
         {{"initial_mm", out.initialRMSE}, {"final_mm", out.finalRMSE},
          {"iterations", res.statistics.ceresIterations},
          {"qualityFlag", calib::qualityFlagToString(res.qualityFlag)},
          {"loopDetected", res.statistics.loopDetected},
          {"loopClosureResidual", res.statistics.loopClosureResidual},
          {"outlierObs", static_cast<int>(res.statistics.outlierObsIds.size())},
          {"overfitSuspect", overfitSuspect}}},
        {"poseDelta", {{"raw", statsJson(out.deltaRaw)},
                       {"aligned", statsJson(out.deltaAligned)}}},
        {"gates", {{"rmseGate", out.rmseGate}, {"poseGate", out.poseGate},
                   {"rmseHalved", rmseHalved}, {"rmseInBand", rmseInBand}}},
        {"refuse", {{"requested", opts.refuseLaser},
                    {"executed", false},   // 调用方（重融合后）覆写
                    {"framesFused", 0}}},
        {"message", out.message}};

    // —— 落盘 ——
    if (!opts.outDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(opts.outDir, ec);
        const std::string rp = opts.reportPath.empty()
                                   ? (std::filesystem::path(opts.outDir) /
                                      "gba_result.json").string()
                                   : opts.reportPath;
        std::ofstream jf(rp);
        if (jf) jf << out.report.dump(2) << "\n";
        writeGbaTrajectoryPly((std::filesystem::path(opts.outDir) /
                               "gba_trajectory.ply").string(),
                              trajBefore, trajAfter);
    }

    // —— 控制台汇总段 ——
    const double sec = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n===== GBA 全局优化 =====\n";
    std::cout << "帧：obs " << obs.size() << "｜送入 " << out.framesSent << "｜保留 "
              << out.framesKept << "｜装配剔除 " << out.excludedFrames.size() << "\n";
    std::cout << "RMSE：initial " << out.initialRMSE << " → final " << out.finalRMSE
              << " mm（迭代 " << res.statistics.ceresIterations
              << "，qualityFlag=" << calib::qualityFlagToString(res.qualityFlag)
              << (res.statistics.loopDetected ? "，含回环" : "") << "）\n";
    std::cout << "位姿修正（aligned）：Δt 中位 " << a.medDt << " / P95 " << a.p95Dt
              << " / max " << a.maxDt << " mm｜Δrot 中位 " << a.medDr << " / P95 "
              << a.p95Dr << " / max " << a.maxDr << " deg\n";
    std::cout << "门①RMSE=" << (out.rmseGate ? "PASS" : "FAIL")
              << "｜门②位姿分布=" << (out.poseGate ? "PASS" : "FAIL");
    if (overfitSuspect) std::cout << "｜⚠ finalRMSE<0.01mm：吸噪声过拟合嫌疑";
    if (a.medDt < 1e-6 || a.p95Dt > 10.0)
        std::cout << "｜⚠ 欠定嫌疑（noMove/wildTail 症状）";
    std::cout << "\n阶段耗时 " << sec << "s";
    std::cout << "\n";
    return out;
}

// ----------------------------------------------------------------------------
// 重融合本体
// ----------------------------------------------------------------------------
GbaRefuseStats refuseWithOp(calib::LaserCloudFuseCuda& op,
                            const std::vector<GbaLaserRecord>& laserArch,
                            const std::map<uint64_t, GbaPose>& poses,
                            bool keepUnoptimized,
                            cv::cuda::Stream& stream) {
    GbaRefuseStats st;
    st.framesTotal = static_cast<int>(laserArch.size());
    if (laserArch.empty()) {
        st.reason = "激光档为空（对账失败保原云）";
        return st;
    }
    op.Clear();   // 【毁灭性动作前的对账已由调用方完成】Clear 自带 Finish
    for (const auto& rec : laserArch) {
        const size_t n = rec.xyz.size() / 3;
        if (n == 0) continue;
        cv::Matx33d R(rec.R[0], rec.R[1], rec.R[2],
                      rec.R[3], rec.R[4], rec.R[5],
                      rec.R[6], rec.R[7], rec.R[8]);
        cv::Vec3d T(rec.T[0], rec.T[1], rec.T[2]);
        auto it = poses.find(rec.frameId);
        if (it != poses.end()) {
            R = it->second.R;
            T = it->second.T;
        } else if (!keepUnoptimized) {
            ++st.framesNoPose;
            continue;                       // 宁缺毋错：无优化位姿的帧不融合
        } else {
            ++st.framesNoPose;              // 原位姿兜底（保帧覆盖）
        }
        const cv::Mat host(1, static_cast<int>(n), CV_32FC3,
                           const_cast<float*>(rec.xyz.data()));
        cv::cuda::GpuMat d;
        d.upload(host, stream);
        op.Execute(d, R, T, stream);
        ++st.framesFused;
    }
    stream.waitForCompletion();             // 导出前保证重融合全部落地
    st.executed = true;
    return st;
}

}  // namespace scr
