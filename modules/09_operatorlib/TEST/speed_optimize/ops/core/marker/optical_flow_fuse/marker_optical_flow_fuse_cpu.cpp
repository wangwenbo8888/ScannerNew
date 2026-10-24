#include "marker_optical_flow_fuse_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cassert>

using namespace calib;

OperatorInfo getMarkerOpticalFlowFuseCPUInfo() {
    return OperatorInfo{"MarkerOpticalFlowFuseCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(01, MarkerOpticalFlowFuseCPU);

// ============================================================
// Anonymous namespace helpers
// ============================================================

namespace {

inline void normalizeNormal(cv::Vec3d& n) {
    double len = std::sqrt(n.dot(n));
    if (len > 1e-12) {
        n /= len;
    } else {
        // 2026-10-15 清理②：坏数据不再静默——零/退化法线替换留痕（罕见事件；
        // 此前无日志，伪装成正常朝向混过质检时事后无线索）
        n = cv::Vec3d(0, 0, 1);
        CALIB_LOG_WARN("01-MarkerOpticalFlowFuseCPU",
                       "零/退化法线替换为 (0,0,1)——上游重建坏数据信号");
    }
}

double normalAngleDeg(const cv::Vec3d& n1, cv::Vec3d n2) {
    normalizeNormal(n2);
    cv::Vec3d a = n1;
    normalizeNormal(a);
    double cosA = std::min(1.0, std::abs(a.dot(n2)));
    return std::acos(cosA) * 180.0 / CV_PI;
}

double pointDistance(const cv::Point3d& a, const cv::Point3d& b) {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

cv::Point3d transformPoint(const cv::Matx33d& R, const cv::Vec3d& T, const cv::Point3d& p) {
    return cv::Point3d(
        R(0,0)*p.x + R(0,1)*p.y + R(0,2)*p.z + T(0),
        R(1,0)*p.x + R(1,1)*p.y + R(1,2)*p.z + T(1),
        R(2,0)*p.x + R(2,1)*p.y + R(2,2)*p.z + T(2));
}

cv::Vec3d transformNormal(const cv::Matx33d& R, const cv::Vec3d& n) {
    return cv::Vec3d(
        R(0,0)*n(0) + R(0,1)*n(1) + R(0,2)*n(2),
        R(1,0)*n(0) + R(1,1)*n(1) + R(1,2)*n(2),
        R(2,0)*n(0) + R(2,1)*n(1) + R(2,2)*n(2));
}

// useNormalWeight=false（默认，2026-10-15 裁决）：等权解算——默认 15° 朝向门下
// cos² 权重差仅 ~7% 无实际作用；=true 时按法线一致性软降级（朝向门放宽场景的保险丝）
bool weightedSVD(
    const std::vector<cv::Point3d>& src,
    const std::vector<cv::Point3d>& dst,
    const std::vector<cv::Vec3d>& srcNorm,
    const std::vector<cv::Vec3d>& dstNorm,
    bool useNormalWeight,
    cv::Matx33d& R, cv::Vec3d& T)
{
    int N = static_cast<int>(src.size());
    if (N < 3) return false;

    std::vector<double> weights(static_cast<size_t>(N), 1.0);
    double wTotal = 0;
    if (useNormalWeight) {
        for (int k = 0; k < N; ++k) {
            cv::Vec3d sn = srcNorm[static_cast<size_t>(k)]; normalizeNormal(sn);
            cv::Vec3d dn = dstNorm[static_cast<size_t>(k)]; normalizeNormal(dn);
            double cosA = sn.dot(dn);
            weights[static_cast<size_t>(k)] = cosA * cosA;
            wTotal += cosA * cosA;
        }
        if (wTotal < 1e-12) {
            std::fill(weights.begin(), weights.end(), 1.0);
            wTotal = static_cast<double>(N);
        }
    } else {
        wTotal = static_cast<double>(N);
    }

    cv::Vec3d cSrc(0, 0, 0), cDst(0, 0, 0);
    for (int k = 0; k < N; ++k) {
        double w = weights[static_cast<size_t>(k)];
        cSrc += w * cv::Vec3d(src[static_cast<size_t>(k)].x, src[static_cast<size_t>(k)].y, src[static_cast<size_t>(k)].z);
        cDst += w * cv::Vec3d(dst[static_cast<size_t>(k)].x, dst[static_cast<size_t>(k)].y, dst[static_cast<size_t>(k)].z);
    }
    cSrc /= wTotal;
    cDst /= wTotal;

    cv::Mat H = cv::Mat::zeros(3, 3, CV_64FC1);
    for (int k = 0; k < N; ++k) {
        double w = weights[static_cast<size_t>(k)];
        cv::Vec3d dSrc(src[static_cast<size_t>(k)].x - cSrc(0), src[static_cast<size_t>(k)].y - cSrc(1), src[static_cast<size_t>(k)].z - cSrc(2));
        cv::Vec3d dDst(dst[static_cast<size_t>(k)].x - cDst(0), dst[static_cast<size_t>(k)].y - cDst(1), dst[static_cast<size_t>(k)].z - cDst(2));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                H.at<double>(r, c) += w * dSrc(r) * dDst(c);
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(H, S, U, Vt);
    cv::Mat Rmat = Vt.t() * U.t();
    if (cv::determinant(Rmat) < 0) {
        cv::Mat VtFixed = Vt.clone();
        VtFixed.at<double>(2, 0) *= -1;
        VtFixed.at<double>(2, 1) *= -1;
        VtFixed.at<double>(2, 2) *= -1;
        Rmat = VtFixed.t() * U.t();
    }

    R = cv::Matx33d(
        Rmat.at<double>(0,0), Rmat.at<double>(0,1), Rmat.at<double>(0,2),
        Rmat.at<double>(1,0), Rmat.at<double>(1,1), Rmat.at<double>(1,2),
        Rmat.at<double>(2,0), Rmat.at<double>(2,1), Rmat.at<double>(2,2));
    T = cv::Vec3d(
        cDst(0) - R(0,0)*cSrc(0) - R(0,1)*cSrc(1) - R(0,2)*cSrc(2),
        cDst(1) - R(1,0)*cSrc(0) - R(1,1)*cSrc(1) - R(1,2)*cSrc(2),
        cDst(2) - R(2,0)*cSrc(0) - R(2,1)*cSrc(1) - R(2,2)*cSrc(2));
    return true;
}

double computeRMSE(const cv::Matx33d& R, const cv::Vec3d& T,
                   const std::vector<cv::Point3d>& src,
                   const std::vector<cv::Point3d>& dst) {
    if (src.empty()) return 0.0;
    double sumSq = 0.0;
    for (size_t k = 0; k < src.size(); ++k) {
        cv::Point3d tp = transformPoint(R, T, src[k]);
        double dx = tp.x - dst[k].x;
        double dy = tp.y - dst[k].y;
        double dz = tp.z - dst[k].z;
        sumSq += dx*dx + dy*dy + dz*dz;
    }
    return std::sqrt(sumSq / static_cast<double>(src.size()));
}

// 三维展度检查：中心化点集最小奇异值过小（≈共线）→ 刚体解算欠定，必须显式拒绝
// （否则 SVD 会"成功"返回一个绕线轴任意旋转的姿态）
bool pointsSpan3D(const std::vector<cv::Point3d>& pts) {
    if (pts.size() < 3) return false;
    cv::Vec3d c(0, 0, 0);
    for (const auto& p : pts) c += cv::Vec3d(p.x, p.y, p.z);
    c /= static_cast<double>(pts.size());
    cv::Mat A(3, static_cast<int>(pts.size()), CV_64FC1);
    for (size_t k = 0; k < pts.size(); ++k) {
        A.at<double>(0, static_cast<int>(k)) = pts[k].x - c(0);
        A.at<double>(1, static_cast<int>(k)) = pts[k].y - c(1);
        A.at<double>(2, static_cast<int>(k)) = pts[k].z - c(2);
    }
    cv::Mat w;
    cv::SVD::compute(A, w);   // 奇异值降序
    double sMax = w.at<double>(0);
    double s2 = w.at<double>(1);
    // 秩 ≥2 即可解刚体变换（3 点/共面点集合法，s3≈0 是平面的正常形态）；
    // 共线特征＝第二大伸展量归零。旧阈值 1e-6 只拦数学精确共线——近共线
    // （偏 1mm/150mm）解算病态且误差在远离标记点处放大（rmse 盲区），
    // 2026-10-15 提到物理量级：2% 相对＋0.5mm 绝对下限（≈10×点位噪声）
    constexpr double kNearCollinearRel = 0.02;
    constexpr double kNearCollinearAbsMm = 0.5;
    return s2 > std::max(kNearCollinearRel * sMax, kNearCollinearAbsMm);
}

// 逐对残差（解算后剔最差对用）
std::vector<double> pairResiduals(const cv::Matx33d& R, const cv::Vec3d& T,
                                  const std::vector<cv::Point3d>& src,
                                  const std::vector<cv::Point3d>& dst) {
    std::vector<double> out(src.size(), 0.0);
    for (size_t k = 0; k < src.size(); ++k) {
        cv::Point3d tp = transformPoint(R, T, src[k]);
        double dx = tp.x - dst[k].x;
        double dy = tp.y - dst[k].y;
        double dz = tp.z - dst[k].z;
        out[k] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return out;
}

} // anonymous namespace

// ============================================================
// Params
// ============================================================

void MarkerOpticalFlowFuseCPUParams::validate() const {
    if (matchDistThresh <= 0.0)
        throw std::invalid_argument("matchDistThresh must be > 0");
    if (normalAngleThresh <= 0.0 || normalAngleThresh >= 90.0)
        throw std::invalid_argument("normalAngleThresh must be in (0, 90)");
    if (minMatchedPoints < 3)
        throw std::invalid_argument("minMatchedPoints must be >= 3");
    if (maxMarkerCount == 0)
        throw std::invalid_argument("maxMarkerCount must be > 0");
    if (firstFrameSearchScale <= 0.0)
        throw std::invalid_argument("firstFrameSearchScale must be > 0");
    // —— 跨参数守卫（2026-10-15 加固：防"单独合法、组合爆炸"——20mm 门灾难的
    //    配置期闸门。派生阈值随门放大是事故根源，见 registration_optimize 参数域分析）——
    if (rmseMaxGate < 0.0)
        throw std::invalid_argument("rmseMaxGate must be >= 0 (negative silently disables the gate; use 0 to disable)");
    if (rmseMaxGate > 0.0 && rmseMaxGate < 0.25 * matchDistThresh)
        throw std::invalid_argument("rmseMaxGate too tight (< matchDistThresh/4): would reject healthy frames");
    if (normalAngleThresh >= 30.0 && !useNormalWeight)
        throw std::invalid_argument("normalAngleThresh >= 30 requires useNormalWeight=true (hard gate + soft degrade must not both be removed)");
    if (matchDistThresh * firstFrameSearchScale > 100.0)
        throw std::invalid_argument("matchDistThresh*firstFrameSearchScale > 100mm: first-frame search radius reaches marker-pitch scale (wrong-lock free-play; 20mm-gate disaster replay)");
    if (firstFrameSearchScale > 25.0)
        throw std::invalid_argument("firstFrameSearchScale > 25 (typo guard)");
    if (matchDistThresh > 50.0)
        throw std::invalid_argument("matchDistThresh > 50mm is pose-jump territory, not matching");
    if (static_cast<size_t>(minMatchedPoints) > maxMarkerCount)
        throw std::invalid_argument("minMatchedPoints > maxMarkerCount (unsatisfiable config)");
}

nlohmann::json MarkerOpticalFlowFuseCPUParams::toJson() const {
    nlohmann::json j;
    j["matchDistThresh"] = matchDistThresh;
    j["normalAngleThresh"] = normalAngleThresh;
    j["minMatchedPoints"] = minMatchedPoints;
    j["collectStatistics"] = collectStatistics;
    j["maxMarkerCount"] = maxMarkerCount;
    j["firstFrameSearchScale"] = firstFrameSearchScale;
    j["rmseMaxGate"] = rmseMaxGate;
    j["useNormalWeight"] = useNormalWeight;
    return j;
}

MarkerOpticalFlowFuseCPUParams MarkerOpticalFlowFuseCPUParams::fromJson(const nlohmann::json& j) {
    MarkerOpticalFlowFuseCPUParams p;
    if (j.contains("matchDistThresh")) p.matchDistThresh = j.at("matchDistThresh").get<double>();
    if (j.contains("normalAngleThresh")) p.normalAngleThresh = j.at("normalAngleThresh").get<double>();
    if (j.contains("minMatchedPoints")) p.minMatchedPoints = j.at("minMatchedPoints").get<int>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    if (j.contains("maxMarkerCount")) p.maxMarkerCount = j.at("maxMarkerCount").get<size_t>();
    if (j.contains("firstFrameSearchScale")) p.firstFrameSearchScale = j.at("firstFrameSearchScale").get<double>();
    if (j.contains("rmseMaxGate")) p.rmseMaxGate = j.at("rmseMaxGate").get<double>();
    if (j.contains("useNormalWeight")) p.useNormalWeight = j.at("useNormalWeight").get<bool>();
    p.validate();
    return p;
}

// ============================================================
// Result convenience methods
// ============================================================

std::vector<cv::Point3d> MarkerOpticalFlowFuseCPUResult::getTransformedPositions() const {
    std::vector<cv::Point3d> v;
    v.reserve(markers.size());
    for (const auto& m : markers) v.push_back(m.transformedPosition);
    return v;
}

std::vector<cv::Point3d> MarkerOpticalFlowFuseCPUResult::getRawPositions() const {
    std::vector<cv::Point3d> v;
    v.reserve(markers.size());
    for (const auto& m : markers) v.push_back(m.rawPosition);
    return v;
}

std::vector<int> MarkerOpticalFlowFuseCPUResult::getGlobalIds() const {
    std::vector<int> v;
    v.reserve(markers.size());
    for (const auto& m : markers) v.push_back(m.globalId);
    return v;
}

size_t MarkerOpticalFlowFuseCPUResult::getMatchedCount() const {
    size_t cnt = 0;
    for (const auto& m : markers) if (m.matched) ++cnt;
    return cnt;
}

// ============================================================
// Impl
// ============================================================

struct MarkerOpticalFlowFuseCPU::Impl {
    MarkerOpticalFlowFuseCPUParams params_;
    MarkerOpticalFlowFuseStats stats_;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const MarkerOpticalFlowFuseCPUParams& params) : params_(params) { params_.validate(); }   // 2026-10-15 清理④：出生即体检

    // 2026-10-15 清理①：collectStatistics 开关生效——false 时计时不上报（功能统计
    // rmse/matchedCount/eff* 快照不受门控，始终输出）
    double durMs(const std::chrono::high_resolution_clock::time_point& t0,
                 const std::chrono::high_resolution_clock::time_point& t1) const {
        if (!params_.collectStatistics) return 0.0;
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    MarkerOpticalFlowFuseCPUResult ExecuteImpl(const std::vector<cv::Point3d>& currentPositions,
                   const std::vector<cv::Vec3d>& currentNormals,
                   const PrevFrameState& prevState,
                   const GlobalMarkerSet* globalMarkers)
    {
#ifndef NDEBUG
        // 2026-10-15 清理③：恢复 Debug 并发误用守卫（成员与 SetParams 侧断言一直
        // 在，但置位块曾丢失——断言永不触发＝假防护。RAII 置位/复位，Release 编出）
        assert(!inProcess_.load() && "Concurrent call detected");
        struct ScopedFlag {
            std::atomic<bool>* f;
            explicit ScopedFlag(std::atomic<bool>* f_) : f(f_) { f->store(true); }
            ~ScopedFlag() { f->store(false); }
        };
        ScopedFlag inProcessGuard{&inProcess_};
#endif
        MarkerOpticalFlowFuseCPUResult result;
        stats_ = MarkerOpticalFlowFuseStats();
        // 生效参数快照（2026-10-15 加固）：门的派生量随填随记，事故复盘可对账
        stats_.effMatchDistThresh = params_.matchDistThresh;
        stats_.effGuardTolerance = 2.0 * params_.matchDistThresh;
        stats_.effRejectFloor = 0.5;
        stats_.effFirstFrameRadius = params_.matchDistThresh * params_.firstFrameSearchScale;
        stats_.effRmseMaxGate = params_.rmseMaxGate;
        stats_.effUseNormalWeight = params_.useNormalWeight;
        auto totalStart = std::chrono::high_resolution_clock::now();

        stats_.currentFrameCount = currentPositions.size();
        stats_.prevFrameCount = prevState.size();

        if (currentPositions.empty()) {
            result.success = false;
            result.message = "Empty current frame";
            return result;
        }
        if (currentPositions.size() > params_.maxMarkerCount) {
            result.success = false;
            result.message = "Point count exceeds maxMarkerCount";
            return result;
        }
        // 输入等长校验（防越界读）：位置与法线必须一一对应
        if (currentNormals.size() != currentPositions.size()) {
            result.success = false;
            result.message = "currentNormals size mismatch with currentPositions";
            return result;
        }
        if (!prevState.empty()
            && prevState.rawNormals.size() != prevState.rawPositions.size()) {
            result.success = false;
            result.message = "prevState rawNormals size mismatch with rawPositions";
            return result;
        }

        std::vector<cv::Vec3d> curNorms = currentNormals;
        for (auto& n : curNorms) normalizeNormal(n);

        if (prevState.empty()) {
            fuseFirstFrame(currentPositions, curNorms, globalMarkers, result);
            auto totalEnd = std::chrono::high_resolution_clock::now();
            stats_.totalTimeMs = durMs(totalStart, totalEnd);
            result.statistics = stats_;
            return result;
        }

        fuseSubsequentFrame(currentPositions, curNorms, prevState, globalMarkers, result);

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = durMs(totalStart, totalEnd);
        result.statistics = stats_;

        return result;
    }

    void fuseFirstFrame(const std::vector<cv::Point3d>& positions,
                        const std::vector<cv::Vec3d>& normals,
                        const GlobalMarkerSet* globalMarkers,
                        MarkerOpticalFlowFuseCPUResult& result)
    {
        size_t N = positions.size();

        if (!globalMarkers || globalMarkers->empty()) {
            // 2026-10-15 裁决：首帧建系强制 ≥max(minMatchedPoints,4) 点——首帧是
            // 全局基准，3 点零冗余（残差恒零质检失明），错首帧污染全会话
            const size_t firstFrameMin =
                std::max(static_cast<size_t>(params_.minMatchedPoints), static_cast<size_t>(4));
            if (N < firstFrameMin) {
                result.success = false;
                result.message = "First frame auto-init requires >= "
                                 + std::to_string(firstFrameMin) + " points (low-redundancy guard)";
                return;
            }
            result.R = cv::Matx33d::eye();
            result.T = cv::Vec3d(0, 0, 0);
            result.transform = cv::Matx44d::eye();
            result.markers.resize(N);
            for (size_t i = 0; i < N; ++i) {
                result.markers[i].rawPosition = positions[i];
                result.markers[i].rawNormal = normals[i];
                result.markers[i].transformedPosition = positions[i];
                result.markers[i].transformedNormal = normals[i];
                result.markers[i].globalId = static_cast<int>(i);
                result.markers[i].matched = true;
                result.markers[i].matchDistance = 0.0;
            }
            stats_.matchedCount = N;
            stats_.unmatchedCount = 0;
            stats_.rmse = 0.0;
            result.success = true;
            result.message = "First frame auto-initialized";
            result.qualityFlag = calib::QualityFlag::Normal;
            CALIB_LOG_INFO("First frame auto-init: {} markers", N);
            return;
        }

        // 首帧挂地图分支的宽松搜索半径（倍率参数化，历史口径 10 倍）
        double searchRadius = params_.matchDistThresh * params_.firstFrameSearchScale;
        double normalThresh = params_.normalAngleThresh;

        std::vector<cv::Point3d> matchedSrc, matchedDst;
        std::vector<cv::Vec3d> matchedSrcNorm, matchedDstNorm;

        result.markers.resize(N);
        for (size_t i = 0; i < N; ++i) {
            result.markers[i].rawPosition = positions[i];
            result.markers[i].rawNormal = normals[i];
        }

        // 一对一约束：全量候选按距离升序贪心分配，每个地图点只被认领一次
        struct FFMapCand { size_t i; size_t j; double dist; };
        std::vector<FFMapCand> cands;
        cands.reserve(N * globalMarkers->size());
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < globalMarkers->size(); ++j) {
                double dist = pointDistance(positions[i], globalMarkers->positions[j]);
                if (dist >= searchRadius) continue;
                if (normalAngleDeg(normals[i], globalMarkers->normals[j]) >= normalThresh) continue;
                cands.push_back({i, j, dist});
            }
        }
        std::sort(cands.begin(), cands.end(),
                  [](const FFMapCand& a, const FFMapCand& b) { return a.dist < b.dist; });
        std::vector<char> usedI(N, 0);
        std::vector<char> usedJ(globalMarkers->size(), 0);
        for (const auto& c : cands) {
            if (usedI[c.i] || usedJ[c.j]) continue;
            usedI[c.i] = 1;
            usedJ[c.j] = 1;
            result.markers[c.i].matched = true;
            result.markers[c.i].globalId = static_cast<int>(c.j);
            result.markers[c.i].matchDistance = c.dist;
            matchedSrc.push_back(positions[c.i]);
            matchedDst.push_back(globalMarkers->positions[c.j]);
            matchedSrcNorm.push_back(normals[c.i]);
            matchedDstNorm.push_back(globalMarkers->normals[c.j]);
        }
        for (size_t i = 0; i < N; ++i) {
            if (!usedI[i]) {
                result.markers[i].matched = false;
                result.markers[i].globalId = -1;
            }
        }

        size_t matched = matchedSrc.size();
        stats_.matchedCount = matched;
        stats_.unmatchedCount = N - matched;

        if (static_cast<int>(matched) < params_.minMatchedPoints) {
            result.success = false;
            result.message = "First frame: insufficient matches to global markers";
            return;
        }

        // 共线退化检查：欠定几何不得进 SVD
        if (!pointsSpan3D(matchedSrc)) {
            result.success = false;
            result.message = "First frame: degenerate (collinear) matched points";
            return;
        }

        cv::Matx33d R;
        cv::Vec3d T;
        if (!weightedSVD(matchedSrc, matchedDst, matchedSrcNorm, matchedDstNorm,
                    params_.useNormalWeight, R, T)) {
            result.success = false;
            result.message = "First frame: SVD failed";
            return;
        }

        result.R = R;
        result.T = T;
        result.transform = cv::Matx44d::eye();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) result.transform(r, c) = R(r, c);
            result.transform(r, 3) = T(r);
        }

        for (size_t i = 0; i < N; ++i) {
            result.markers[i].transformedPosition = transformPoint(R, T, positions[i]);
            result.markers[i].transformedNormal = transformNormal(R, normals[i]);
        }

        stats_.rmse = computeRMSE(R, T, matchedSrc, matchedDst);

        // 误差门（>0 启用）：数量够但解歪了同样必须拦下
        if (params_.rmseMaxGate > 0.0 && stats_.rmse > params_.rmseMaxGate) {
            result.success = false;
            result.message = "First frame: RMSE gate exceeded (rmse=" +
                             std::to_string(stats_.rmse) + ")";
            return;
        }

        result.success = true;
        result.message = "First frame registered to global markers";
        double overlap = static_cast<double>(matched) / static_cast<double>(N);
        result.qualityFlag = overlap >= 0.8 ? calib::QualityFlag::Normal :
                            (overlap >= 0.5 ? calib::QualityFlag::Degraded :
                             calib::QualityFlag::Warning);
    }

    void fuseSubsequentFrame(const std::vector<cv::Point3d>& currentPositions,
                             const std::vector<cv::Vec3d>& currentNormals,
                             const PrevFrameState& prevState,
                             const GlobalMarkerSet* globalMarkers,
                             MarkerOpticalFlowFuseCPUResult& result)
    {
        size_t N = currentPositions.size();

        std::vector<cv::Vec3d> prevNorms = prevState.rawNormals;
        for (auto& n : prevNorms) normalizeNormal(n);

        auto matchStart = std::chrono::high_resolution_clock::now();

        struct MatchResult { int prevIdx; double dist; };
        std::vector<MatchResult> matches(N, {-1, 0.0});

        // 一对一约束：全量候选按距离升序贪心分配，每个上帧点只被认领一次
        // （旧实现逐当前点独立取最近，两个当前点可重复认领同一上帧点污染解算）
        struct SubCand { size_t i; size_t j; double dist; };
        std::vector<SubCand> cands;
        cands.reserve(N * prevState.size());
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < prevState.size(); ++j) {
                double dist = pointDistance(currentPositions[i], prevState.rawPositions[j]);
                if (dist >= params_.matchDistThresh) continue;
                if (normalAngleDeg(currentNormals[i], prevNorms[j]) >= params_.normalAngleThresh) continue;
                cands.push_back({i, j, dist});
            }
        }
        std::sort(cands.begin(), cands.end(),
                  [](const SubCand& a, const SubCand& b) { return a.dist < b.dist; });
        std::vector<char> usedI(N, 0);
        std::vector<char> usedJ(prevState.size(), 0);
        for (const auto& c : cands) {
            if (usedI[c.i] || usedJ[c.j]) continue;
            usedI[c.i] = 1;
            usedJ[c.j] = 1;
            matches[c.i] = {static_cast<int>(c.j), c.dist};
        }

        auto matchEnd = std::chrono::high_resolution_clock::now();
        stats_.matchTimeMs = durMs(matchStart, matchEnd);

        std::vector<cv::Point3d> matchedSrc, matchedDst;
        std::vector<cv::Vec3d> matchedSrcNorm, matchedDstNorm;
        std::vector<size_t> srcIdxOfPair;   // 配对 ↔ 当前点下标（剔坏对时回标 unmatched）

        for (size_t i = 0; i < N; ++i) {
            if (matches[i].prevIdx < 0) continue;
            size_t j = static_cast<size_t>(matches[i].prevIdx);
            int gid = (j < prevState.globalIds.size()) ? prevState.globalIds[j] : -1;
            if (gid < 0) continue;

            cv::Point3d globalPos;
            cv::Vec3d globalNorm;
            if (globalMarkers && !globalMarkers->empty()) {
                // 编号-地图一致性守卫：编号必须落在地图内，且地图条目与上帧变换位置吻合
                // （正常维护下二者同源同帧，差值≈噪声；不符＝帧内下标语义混入/地图错位，
                //   宁可丢该对也不拿错点硬解——一次解算不得混用两套坐标）
                if (gid >= static_cast<int>(globalMarkers->size())) continue;
                globalPos = globalMarkers->positions[static_cast<size_t>(gid)];
                globalNorm = globalMarkers->normals[static_cast<size_t>(gid)];
                cv::Point3d prevGlobal =
                    transformPoint(prevState.R, prevState.T, prevState.rawPositions[j]);
                if (pointDistance(globalPos, prevGlobal) > 2.0 * params_.matchDistThresh)
                    continue;
            } else {
                // 无地图（历史口径）：上帧 R/T 变换后的累积坐标
                globalPos = transformPoint(prevState.R, prevState.T, prevState.rawPositions[j]);
                globalNorm = transformNormal(prevState.R, prevNorms[j]);
            }

            matchedSrc.push_back(currentPositions[i]);
            matchedDst.push_back(globalPos);
            matchedSrcNorm.push_back(currentNormals[i]);
            matchedDstNorm.push_back(globalNorm);
            srcIdxOfPair.push_back(i);
        }

        size_t matchedCount = matchedSrc.size();
        stats_.matchedCount = matchedCount;
        stats_.unmatchedCount = N - matchedCount;

        if (static_cast<int>(matchedCount) < params_.minMatchedPoints) {
            result.success = false;
            result.message = "Insufficient matched points: " + std::to_string(matchedCount) +
                             " < " + std::to_string(params_.minMatchedPoints);
            result.markers.resize(N);
            for (size_t i = 0; i < N; ++i) {
                result.markers[i].rawPosition = currentPositions[i];
                result.markers[i].rawNormal = currentNormals[i];
                result.markers[i].matched = false;
                result.markers[i].globalId = -1;
            }
            return;
        }

        // 共线退化检查：欠定几何不得进 SVD（共线三点解刚体变换有无穷解）
        if (!pointsSpan3D(matchedSrc)) {
            result.success = false;
            result.message = "Degenerate (collinear) matched points";
            result.markers.resize(N);
            for (size_t i = 0; i < N; ++i) {
                result.markers[i].rawPosition = currentPositions[i];
                result.markers[i].rawNormal = currentNormals[i];
                result.markers[i].matched = false;
                result.markers[i].globalId = -1;
            }
            return;
        }

        auto svdStart = std::chrono::high_resolution_clock::now();

        cv::Matx33d R;
        cv::Vec3d T;
        if (!weightedSVD(matchedSrc, matchedDst, matchedSrcNorm, matchedDstNorm,
                    params_.useNormalWeight, R, T)) {
            result.success = false;
            result.message = "SVD failed";
            return;
        }

        // 剔最坏对重解：残差同时超 [2×下中位残差] 与 [绝对下限] 才剔（防误剔正常噪声），
        // 且剔除后剩余须 ≥ minMatchedPoints。坏对来源＝匹配门内混入的错配。
        // 绝对下限 2026-10-15 与门限解耦（旧＝门/4：门开 10mm 时免死线抬到 2.5mm，
        // 1~2mm 半歪对永久免死——门限耦合缺陷）；固定 0.5mm≈好对残差上限
        constexpr double kOutlierRejectFloorMm = 0.5;
        int rejectedCurIdx = -1;
        if (matchedSrc.size() > static_cast<size_t>(params_.minMatchedPoints)) {
            auto resid = pairResiduals(R, T, matchedSrc, matchedDst);
            std::vector<double> sorted = resid;
            std::sort(sorted.begin(), sorted.end());
            double med = sorted[(sorted.size() - 1) / 2];
            size_t worst = 0;
            for (size_t k = 1; k < resid.size(); ++k)
                if (resid[k] > resid[worst]) worst = k;
            if (resid[worst] > 2.0 * med
                && resid[worst] > kOutlierRejectFloorMm) {
                rejectedCurIdx = static_cast<int>(srcIdxOfPair[worst]);
                matchedSrc.erase(matchedSrc.begin() + static_cast<long>(worst));
                matchedDst.erase(matchedDst.begin() + static_cast<long>(worst));
                matchedSrcNorm.erase(matchedSrcNorm.begin() + static_cast<long>(worst));
                matchedDstNorm.erase(matchedDstNorm.begin() + static_cast<long>(worst));
                srcIdxOfPair.erase(srcIdxOfPair.begin() + static_cast<long>(worst));
                matchedCount = matchedSrc.size();
                stats_.matchedCount = matchedCount;
                stats_.unmatchedCount = N - matchedCount;
                if (!pointsSpan3D(matchedSrc)
                    || !weightedSVD(matchedSrc, matchedDst, matchedSrcNorm, matchedDstNorm,
                    params_.useNormalWeight, R, T)) {
                    result.success = false;
                    result.message = "Re-solve after outlier rejection failed";
                    return;
                }
                CALIB_LOG_INFO("Outlier pair rejected (curIdx={}), re-solved with {} pairs",
                               rejectedCurIdx, matchedCount);
            }
        }

        auto svdEnd = std::chrono::high_resolution_clock::now();
        stats_.svdTimeMs = durMs(svdStart, svdEnd);

        auto transformStart = std::chrono::high_resolution_clock::now();

        result.R = R;
        result.T = T;
        result.transform = cv::Matx44d::eye();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) result.transform(r, c) = R(r, c);
            result.transform(r, 3) = T(r);
        }

        result.markers.resize(N);
        for (size_t i = 0; i < N; ++i) {
            result.markers[i].rawPosition = currentPositions[i];
            result.markers[i].rawNormal = currentNormals[i];
            result.markers[i].transformedPosition = transformPoint(R, T, currentPositions[i]);
            result.markers[i].transformedNormal = transformNormal(R, currentNormals[i]);

            if (matches[i].prevIdx >= 0) {
                size_t j = static_cast<size_t>(matches[i].prevIdx);
                int gid = (j < prevState.globalIds.size()) ? prevState.globalIds[j] : -1;
                // matched 语义＝「参与了最终解算」：被地图守卫丢弃/剔坏对/无编号的对都不算
                bool inSolve = gid >= 0
                    && std::find(srcIdxOfPair.begin(), srcIdxOfPair.end(), i) != srcIdxOfPair.end();
                result.markers[i].matched = inSolve;
                result.markers[i].globalId = inSolve ? gid : -1;
                result.markers[i].matchDistance = matches[i].dist;
            } else {
                result.markers[i].matched = false;
                result.markers[i].globalId = -1;
            }
        }

        auto transformEnd = std::chrono::high_resolution_clock::now();
        stats_.transformTimeMs = durMs(transformStart, transformEnd);

        stats_.rmse = computeRMSE(R, T, matchedSrc, matchedDst);

        // 误差门（>0 启用）：数量够但解歪了同样必须拦下
        if (params_.rmseMaxGate > 0.0 && stats_.rmse > params_.rmseMaxGate) {
            result.success = false;
            result.message = "RMSE gate exceeded (rmse=" + std::to_string(stats_.rmse) + ")";
            return;
        }

        result.success = true;
        result.message = "Frame fusion completed";
        double overlap = static_cast<double>(matchedCount) / static_cast<double>(N);
        result.qualityFlag = overlap >= 0.8 ? calib::QualityFlag::Normal :
                            (overlap >= 0.5 ? calib::QualityFlag::Degraded :
                             calib::QualityFlag::Warning);
        // 2026-10-15 裁决：恰好 minMatchedPoints 对＝零冗余解（残差恒≈0 质检失明），
        // 成功也不得报 Normal——强制降级，下游知情
        if (matchedCount == static_cast<size_t>(params_.minMatchedPoints)
            && result.qualityFlag == calib::QualityFlag::Normal) {
            result.qualityFlag = calib::QualityFlag::Degraded;
        }

        CALIB_LOG_DEBUG("Fuse completed: matched={}, overlap={:.1f}%, RMSE={:.4f}",
                       matchedCount, overlap * 100.0, stats_.rmse);
        // 注：totalTimeMs 在 ExecuteImpl 返回前才赋值，此处只报分解项（旧版曾在此打恒 0 的 total）
        CALIB_LOG_INFO("Timing breakdown (ms): match={:.3f} | svd={:.3f} | transform={:.3f}",
                       stats_.totalTimeMs, stats_.matchTimeMs, stats_.svdTimeMs, stats_.transformTimeMs);
    }
};

// ============================================================
// Constructor / Destructor
// ============================================================

MarkerOpticalFlowFuseCPU::MarkerOpticalFlowFuseCPU(const MarkerOpticalFlowFuseCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    // 2026-10-15 加固：构造/改参全量留痕（旧版只记 2 个字段，运行中改参零日志）
    CALIB_LOG_INFO("MarkerOpticalFlowFuseCPU initialized, params={}",
                   params.toJson().dump());
}

MarkerOpticalFlowFuseCPU::~MarkerOpticalFlowFuseCPU() = default;

void MarkerOpticalFlowFuseCPU::Destroy() { }

// ============================================================
// fuse methods
// ============================================================

MarkerOpticalFlowFuseCPUResult MarkerOpticalFlowFuseCPU::Execute(const std::vector<cv::Point3d>& currentPositions,
              const std::vector<cv::Vec3d>& currentNormals,
              const PrevFrameState& prevState) {
    CALIB_LOG_DEBUG("fuse(pos,norm,prev) called: current={}", currentPositions.size());
    return pImpl_->ExecuteImpl(currentPositions, currentNormals, prevState, nullptr);
}

MarkerOpticalFlowFuseCPUResult MarkerOpticalFlowFuseCPU::Execute(const PointReconstructCPUResult& currentFrame,
              const PrevFrameState& prevState) {
    CALIB_LOG_DEBUG("fuse(PointReconstructCPUResult) called: markers={}",
                    currentFrame.markerResults.size());

    std::vector<cv::Point3d> positions;
    std::vector<cv::Vec3d> normals;
    for (const auto& mr : currentFrame.markerResults) {
        if (!mr.validCircle) continue;
        positions.emplace_back(mr.centerX, mr.centerY, mr.centerZ);
        normals.emplace_back(mr.normalX, mr.normalY, mr.normalZ);
    }
    return pImpl_->ExecuteImpl(positions, normals, prevState, nullptr);
}

MarkerOpticalFlowFuseCPUResult MarkerOpticalFlowFuseCPU::Execute(const std::vector<cv::Point3d>& currentPositions,
              const std::vector<cv::Vec3d>& currentNormals,
              const PrevFrameState& prevState,
              const GlobalMarkerSet& globalMarkers) {
    CALIB_LOG_DEBUG("fuse(pos,norm,prev,global) called: current={}", currentPositions.size());
    return pImpl_->ExecuteImpl(currentPositions, currentNormals, prevState, &globalMarkers);
}

// ============================================================
// warmup
// ============================================================

void MarkerOpticalFlowFuseCPU::Warmup(int maxMarkerCount) {
    pImpl_->warmed_up_ = true;
    CALIB_LOG_INFO("warmup() completed: maxMarkerCount={}", maxMarkerCount);
}

void MarkerOpticalFlowFuseCPU::Warmup(const calib::WarmupConfig& config) {
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================

void MarkerOpticalFlowFuseCPU::SetParams(const MarkerOpticalFlowFuseCPUParams& params) {
#ifndef NDEBUG
    assert(!pImpl_->inProcess_.load() && "setParams() while processing");
#endif
    pImpl_->params_ = params;
    pImpl_->params_.validate();
    pImpl_->warmed_up_ = false;
    // 2026-10-15 加固：运行中改参全量留痕（旧版零日志——事故后无法对账"当时生效值"）
    CALIB_LOG_INFO("MarkerOpticalFlowFuseCPU SetParams, params={}",
                   pImpl_->params_.toJson().dump());
}

const MarkerOpticalFlowFuseCPUParams& MarkerOpticalFlowFuseCPU::GetParams() const {
    return pImpl_->params_;
}

const MarkerOpticalFlowFuseStats& MarkerOpticalFlowFuseCPU::GetStatistics() const noexcept {
    return pImpl_->stats_;
}

void MarkerOpticalFlowFuseCPU::ResetStatistics() noexcept {
    pImpl_->stats_ = MarkerOpticalFlowFuseStats();
}
