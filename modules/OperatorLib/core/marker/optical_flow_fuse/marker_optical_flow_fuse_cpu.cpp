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

void normalizeNormal(cv::Vec3d& n) {
    double len = std::sqrt(n.dot(n));
    if (len > 1e-12) n /= len;
    else n = cv::Vec3d(0, 0, 1);
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

bool weightedSVD(
    const std::vector<cv::Point3d>& src,
    const std::vector<cv::Point3d>& dst,
    const std::vector<cv::Vec3d>& srcNorm,
    const std::vector<cv::Vec3d>& dstNorm,
    cv::Matx33d& R, cv::Vec3d& T)
{
    int N = static_cast<int>(src.size());
    if (N < 3) return false;

    std::vector<double> weights(static_cast<size_t>(N));
    double wTotal = 0;
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
}

nlohmann::json MarkerOpticalFlowFuseCPUParams::toJson() const {
    nlohmann::json j;
    j["matchDistThresh"] = matchDistThresh;
    j["normalAngleThresh"] = normalAngleThresh;
    j["minMatchedPoints"] = minMatchedPoints;
    j["collectStatistics"] = collectStatistics;
    j["maxMarkerCount"] = maxMarkerCount;
    return j;
}

MarkerOpticalFlowFuseCPUParams MarkerOpticalFlowFuseCPUParams::fromJson(const nlohmann::json& j) {
    MarkerOpticalFlowFuseCPUParams p;
    if (j.contains("matchDistThresh")) p.matchDistThresh = j.at("matchDistThresh").get<double>();
    if (j.contains("normalAngleThresh")) p.normalAngleThresh = j.at("normalAngleThresh").get<double>();
    if (j.contains("minMatchedPoints")) p.minMatchedPoints = j.at("minMatchedPoints").get<int>();
    if (j.contains("collectStatistics")) p.collectStatistics = j.at("collectStatistics").get<bool>();
    if (j.contains("maxMarkerCount")) p.maxMarkerCount = j.at("maxMarkerCount").get<size_t>();
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

    explicit Impl(const MarkerOpticalFlowFuseCPUParams& params) : params_(params) {}

    MarkerOpticalFlowFuseCPUResult ExecuteImpl(const std::vector<cv::Point3d>& currentPositions,
                   const std::vector<cv::Vec3d>& currentNormals,
                   const PrevFrameState& prevState,
                   const GlobalMarkerSet* globalMarkers)
    {
        MarkerOpticalFlowFuseCPUResult result;
        stats_ = MarkerOpticalFlowFuseStats();
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

        std::vector<cv::Vec3d> curNorms = currentNormals;
        for (auto& n : curNorms) normalizeNormal(n);

        if (prevState.empty()) {
            fuseFirstFrame(currentPositions, curNorms, globalMarkers, result);
            auto totalEnd = std::chrono::high_resolution_clock::now();
            stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            result.statistics = stats_;
            return result;
        }

        fuseSubsequentFrame(currentPositions, curNorms, prevState, globalMarkers, result);

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
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

        double searchRadius = params_.matchDistThresh * 10.0;
        double normalThresh = params_.normalAngleThresh;

        std::vector<cv::Point3d> matchedSrc, matchedDst;
        std::vector<cv::Vec3d> matchedSrcNorm, matchedDstNorm;

        result.markers.resize(N);
        for (size_t i = 0; i < N; ++i) {
            result.markers[i].rawPosition = positions[i];
            result.markers[i].rawNormal = normals[i];

            int bestIdx = -1;
            double bestDist = searchRadius;
            for (size_t j = 0; j < globalMarkers->size(); ++j) {
                double dist = pointDistance(positions[i], globalMarkers->positions[j]);
                if (dist < bestDist) {
                    double angle = normalAngleDeg(normals[i], globalMarkers->normals[j]);
                    if (angle < normalThresh) {
                        bestDist = dist;
                        bestIdx = static_cast<int>(j);
                    }
                }
            }

            if (bestIdx >= 0) {
                result.markers[i].matched = true;
                result.markers[i].globalId = static_cast<int>(bestIdx);
                result.markers[i].matchDistance = bestDist;
                matchedSrc.push_back(positions[i]);
                matchedDst.push_back(globalMarkers->positions[static_cast<size_t>(bestIdx)]);
                matchedSrcNorm.push_back(normals[i]);
                matchedDstNorm.push_back(globalMarkers->normals[static_cast<size_t>(bestIdx)]);
            } else {
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

        cv::Matx33d R;
        cv::Vec3d T;
        if (!weightedSVD(matchedSrc, matchedDst, matchedSrcNorm, matchedDstNorm, R, T)) {
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

        for (size_t i = 0; i < N; ++i) {
            int bestIdx = -1;
            double bestDist = params_.matchDistThresh;
            for (size_t j = 0; j < prevState.size(); ++j) {
                double dist = pointDistance(currentPositions[i], prevState.rawPositions[j]);
                if (dist < bestDist) {
                    double angle = normalAngleDeg(currentNormals[i], prevNorms[j]);
                    if (angle < params_.normalAngleThresh) {
                        bestDist = dist;
                        bestIdx = static_cast<int>(j);
                    }
                }
            }
            matches[i] = {bestIdx, bestDist};
        }

        auto matchEnd = std::chrono::high_resolution_clock::now();
        stats_.matchTimeMs = std::chrono::duration<double, std::milli>(matchEnd - matchStart).count();

        std::vector<cv::Point3d> matchedSrc, matchedDst;
        std::vector<cv::Vec3d> matchedSrcNorm, matchedDstNorm;

        for (size_t i = 0; i < N; ++i) {
            if (matches[i].prevIdx >= 0) {
                size_t j = static_cast<size_t>(matches[i].prevIdx);
                int gid = (j < prevState.globalIds.size()) ? prevState.globalIds[j] : -1;
                if (gid < 0) continue;

                cv::Point3d globalPos;
                cv::Vec3d globalNorm;
                if (globalMarkers && !globalMarkers->empty() &&
                    gid < static_cast<int>(globalMarkers->size())) {
                    globalPos = globalMarkers->positions[static_cast<size_t>(gid)];
                    globalNorm = globalMarkers->normals[static_cast<size_t>(gid)];
                } else {
                    globalPos = transformPoint(prevState.R, prevState.T, prevState.rawPositions[j]);
                    globalNorm = transformNormal(prevState.R, prevNorms[j]);
                }

                matchedSrc.push_back(currentPositions[i]);
                matchedDst.push_back(globalPos);
                matchedSrcNorm.push_back(currentNormals[i]);
                matchedDstNorm.push_back(globalNorm);
            }
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

        auto svdStart = std::chrono::high_resolution_clock::now();

        cv::Matx33d R;
        cv::Vec3d T;
        if (!weightedSVD(matchedSrc, matchedDst, matchedSrcNorm, matchedDstNorm, R, T)) {
            result.success = false;
            result.message = "SVD failed";
            return;
        }

        auto svdEnd = std::chrono::high_resolution_clock::now();
        stats_.svdTimeMs = std::chrono::duration<double, std::milli>(svdEnd - svdStart).count();

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
                result.markers[i].matched = (gid >= 0);
                result.markers[i].globalId = gid;
                result.markers[i].matchDistance = matches[i].dist;
            } else {
                result.markers[i].matched = false;
                result.markers[i].globalId = -1;
            }
        }

        auto transformEnd = std::chrono::high_resolution_clock::now();
        stats_.transformTimeMs = std::chrono::duration<double, std::milli>(transformEnd - transformStart).count();

        stats_.rmse = computeRMSE(R, T, matchedSrc, matchedDst);

        result.success = true;
        result.message = "Frame fusion completed";
        double overlap = static_cast<double>(matchedCount) / static_cast<double>(N);
        result.qualityFlag = overlap >= 0.8 ? calib::QualityFlag::Normal :
                            (overlap >= 0.5 ? calib::QualityFlag::Degraded :
                             calib::QualityFlag::Warning);

        CALIB_LOG_INFO("Fuse completed: matched={}, overlap={:.1f}%, RMSE={:.4f}",
                       matchedCount, overlap * 100.0, stats_.rmse);
        CALIB_LOG_INFO("Timing breakdown (ms): total={:.3f} | match={:.3f} | svd={:.3f} | transform={:.3f}",
                       stats_.totalTimeMs, stats_.matchTimeMs, stats_.svdTimeMs, stats_.transformTimeMs);
    }
};

// ============================================================
// Constructor / Destructor
// ============================================================

MarkerOpticalFlowFuseCPU::MarkerOpticalFlowFuseCPU(const MarkerOpticalFlowFuseCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("MarkerOpticalFlowFuseCPU initialized (matchDistThresh={}, minMatchedPoints={})",
                   params.matchDistThresh, params.minMatchedPoints);
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
