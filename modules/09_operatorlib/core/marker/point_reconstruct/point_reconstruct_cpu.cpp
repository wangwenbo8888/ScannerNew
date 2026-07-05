/**
 * @file point_reconstruct_cpu.cpp
 * @brief 标记点法线和中心快速三维重建算子 - 实现文件
 *
 * 算法流程：
 *   1. 构建左右相机投影矩阵
 *   2. 对每组标记点的匹配点对进行三角测量
 *   3. 重投影误差过滤
 *   4. SVD 拟合平面
 *   5. 投影到平面 + Taubin 圆拟合
 *   6. 还原圆心三维坐标 + 提取法线
 */

#include "point_reconstruct_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

using namespace calib;

OperatorInfo getPointReconstructCPUInfo() {
    return OperatorInfo{"PointReconstructCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(11, PointReconstructCPU);

namespace {

void buildProjectionMatrices(const PointReconstructCPUParams& params,
                             cv::Mat& P1, cv::Mat& P2) {
    cv::Matx33d K1(params.fxLeft, 0, params.cxLeft,
                   0, params.fyLeft, params.cyLeft,
                   0, 0, 1);
    cv::Matx33d K2(params.fxRight, 0, params.cxRight,
                   0, params.fyRight, params.cyRight,
                   0, 0, 1);

    cv::Matx34d P1m(K1(0,0), K1(0,1), K1(0,2), 0,
                     K1(1,0), K1(1,1), K1(1,2), 0,
                     K1(2,0), K1(2,1), K1(2,2), 0);

    cv::Matx33d R = params.R;
    cv::Vec3d T = params.T;
    cv::Matx34d Rt(R(0,0), R(0,1), R(0,2), T(0),
                   R(1,0), R(1,1), R(1,2), T(1),
                   R(2,0), R(2,1), R(2,2), T(2));

    cv::Matx34d P2m = K2 * Rt;

    P1 = cv::Mat(P1m);
    P2 = cv::Mat(P2m);
}

void triangulateBatch(const std::vector<double>& leftU,
                      const std::vector<double>& leftV,
                      const std::vector<double>& rightU,
                      const std::vector<double>& rightV,
                      const cv::Mat& P1, const cv::Mat& P2,
                      std::vector<ReconstructedPoint3D>& outPts) {
    size_t N = leftU.size();
    if (N == 0) return;

    cv::Mat pts1(2, static_cast<int>(N), CV_64FC1);
    cv::Mat pts2(2, static_cast<int>(N), CV_64FC1);
    for (size_t i = 0; i < N; ++i) {
        pts1.at<double>(0, static_cast<int>(i)) = leftU[i];
        pts1.at<double>(1, static_cast<int>(i)) = leftV[i];
        pts2.at<double>(0, static_cast<int>(i)) = rightU[i];
        pts2.at<double>(1, static_cast<int>(i)) = rightV[i];
    }

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    outPts.resize(N);
    for (size_t i = 0; i < N; ++i) {
        int idx = static_cast<int>(i);
        double w = points4D.at<double>(3, idx);
        if (std::abs(w) < 1e-12) {
            outPts[i].x = 0; outPts[i].y = 0; outPts[i].z = 0;
            outPts[i].reprojError = 9999.0f;
            outPts[i].leftU = leftU[i];
            outPts[i].leftV = leftV[i];
            outPts[i].rightU = rightU[i];
            outPts[i].rightV = rightV[i];
            continue;
        }

        double X = points4D.at<double>(0, idx) / w;
        double Y = points4D.at<double>(1, idx) / w;
        double Z = points4D.at<double>(2, idx) / w;

        cv::Mat p3d = (cv::Mat_<double>(4, 1) << X, Y, Z, 1.0);
        cv::Mat rpL = P1 * p3d;
        cv::Mat rpR = P2 * p3d;

        double errLx = rpL.at<double>(0) / rpL.at<double>(2) - leftU[i];
        double errLy = rpL.at<double>(1) / rpL.at<double>(2) - leftV[i];
        double errRx = rpR.at<double>(0) / rpR.at<double>(2) - rightU[i];
        double errRy = rpR.at<double>(1) / rpR.at<double>(2) - rightV[i];

        float reprojErr = static_cast<float>(std::sqrt(
            (errLx*errLx + errLy*errLy + errRx*errRx + errRy*errRy) / 2.0));

        outPts[i].x = X;
        outPts[i].y = Y;
        outPts[i].z = Z;
        outPts[i].leftU = leftU[i];
        outPts[i].leftV = leftV[i];
        outPts[i].rightU = rightU[i];
        outPts[i].rightV = rightV[i];
        outPts[i].reprojError = reprojErr;
    }
}

bool fitPlane(const std::vector<ReconstructedPoint3D>& pts, PlaneFitResult& out) {
    size_t N = pts.size();
    if (N < 3) return false;

    double cx = 0, cy = 0, cz = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; cz += p.z; }
    cx /= N; cy /= N; cz /= N;

    cv::Mat A(3, static_cast<int>(N), CV_64FC1);
    for (size_t i = 0; i < N; ++i) {
        A.at<double>(0, static_cast<int>(i)) = pts[i].x - cx;
        A.at<double>(1, static_cast<int>(i)) = pts[i].y - cy;
        A.at<double>(2, static_cast<int>(i)) = pts[i].z - cz;
    }

    cv::Mat w, U, Vt;
    cv::SVD::compute(A, w, U, Vt);

    double nx = U.at<double>(0, 2);
    double ny = U.at<double>(1, 2);
    double nz = U.at<double>(2, 2);

    double nLen = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nLen < 1e-12) return false;
    nx /= nLen; ny /= nLen; nz /= nLen;

    double d = -(nx*cx + ny*cy + nz*cz);

    double fitErr = 0;
    for (const auto& p : pts) {
        double dist = nx*p.x + ny*p.y + nz*p.z + d;
        fitErr += dist * dist;
    }
    fitErr = std::sqrt(fitErr / N);

    out.nx = nx; out.ny = ny; out.nz = nz; out.d = d;
    out.centroidX = cx; out.centroidY = cy; out.centroidZ = cz;
    out.fitError = fitErr;
    out.singularValues[0] = w.at<double>(0);
    out.singularValues[1] = w.at<double>(1);
    out.singularValues[2] = w.at<double>(2);

    return true;
}

bool taubinCircleFit(const std::vector<double>& x, const std::vector<double>& y,
                     double& cx, double& cy, double& r) {
    size_t N = x.size();
    if (N < 3) return false;

    double sumX = 0, sumY = 0;
    double sumX2 = 0, sumY2 = 0, sumXY = 0;
    double sumX3 = 0, sumY3 = 0, sumX2Y = 0, sumXY2 = 0;
    double sumX4 = 0, sumX2Y2 = 0, sumY4 = 0;

    for (size_t i = 0; i < N; ++i) {
        double xi = x[i], yi = y[i];
        double x2 = xi*xi, y2 = yi*yi, xy = xi*yi;
        sumX += xi; sumY += yi;
        sumX2 += x2; sumY2 += y2; sumXY += xy;
        sumX3 += x2*xi; sumY3 += y2*yi;
        sumX2Y += x2*yi; sumXY2 += xi*y2;
        sumX4 += x2*x2; sumX2Y2 += x2*y2; sumY4 += y2*y2;
    }

    double D = N * sumX2 - sumX * sumX;
    double E = N * sumXY - sumX * sumY;
    double F = N * sumY2 - sumY * sumY;
    double G = 0.5 * (N * sumX3 + N * sumXY2 - sumX * sumX2 - sumX * sumY2);
    double H = 0.5 * (N * sumX2Y + N * sumY3 - sumY * sumX2 - sumY * sumY2);

    double denom = D * F - E * E;
    if (std::abs(denom) < 1e-12) return false;

    cx = (G * F - H * E) / denom;
    cy = (H * D - G * E) / denom;

    double sumCentDist2 = 0;
    for (size_t i = 0; i < N; ++i) {
        double dx = x[i] - cx;
        double dy = y[i] - cy;
        sumCentDist2 += dx*dx + dy*dy;
    }
    r = std::sqrt(sumCentDist2 / N);

    return r > 0;
}

bool fitCircleOnPlane(const std::vector<ReconstructedPoint3D>& pts,
                      const PlaneFitResult& plane,
                      CircleFitResult& out) {
    size_t N = pts.size();
    if (N < 3) return false;

    cv::Vec3d n(plane.nx, plane.ny, plane.nz);
    cv::Vec3d centroid(plane.centroidX, plane.centroidY, plane.centroidZ);

    cv::Vec3d ref(1.0, 0.0, 0.0);
    if (std::abs(n.dot(ref)) > 0.9) ref = cv::Vec3d(0.0, 1.0, 0.0);

    cv::Vec3d u = ref - n * n.dot(ref);
    double uLen = std::sqrt(u.dot(u));
    if (uLen < 1e-12) return false;
    u = u / uLen;
    cv::Vec3d v = n.cross(u);

    std::vector<double> localX(N), localY(N);
    for (size_t i = 0; i < N; ++i) {
        cv::Vec3d diff(pts[i].x - centroid(0),
                       pts[i].y - centroid(1),
                       pts[i].z - centroid(2));
        localX[i] = diff.dot(u);
        localY[i] = diff.dot(v);
    }

    double cx, cy, r;
    if (!taubinCircleFit(localX, localY, cx, cy, r)) return false;

    double fitErr = 0;
    for (size_t i = 0; i < N; ++i) {
        double dx = localX[i] - cx;
        double dy = localY[i] - cy;
        double dist = std::sqrt(dx*dx + dy*dy);
        fitErr += (dist - r) * (dist - r);
    }
    fitErr = std::sqrt(fitErr / N);

    cv::Vec3d center3D = centroid + cx * u + cy * v;

    out.centerLocalX = cx;
    out.centerLocalY = cy;
    out.radius = r;
    out.fitError = fitErr;
    out.centerX = center3D(0);
    out.centerY = center3D(1);
    out.centerZ = center3D(2);

    return true;
}

void collectStatistics(PointReconstructStats& stats,
                       const std::vector<MarkerReconstructResult>& results) {
    stats.totalMarkerPairs = results.size();
    stats.validMarkerCount = 0;
    stats.totalReconstructedPoints = 0;
    stats.avgReprojError = 0.0f;
    stats.avgPlaneFitError = 0.0;
    stats.avgCircleFitError = 0.0;
    stats.avgRadius = 0.0;
    stats.radiusStd = 0.0;

    float totalReproj = 0.0f;
    double totalPlaneErr = 0.0;
    double totalCircleErr = 0.0;
    std::vector<double> radii;

    for (const auto& mr : results) {
        stats.totalReconstructedPoints += mr.reconstructedPoints.size();
        for (const auto& pt : mr.reconstructedPoints) {
            totalReproj += pt.reprojError;
        }
        if (mr.validPlane) {
            totalPlaneErr += mr.planeFit.fitError;
        }
        if (mr.validCircle) {
            totalCircleErr += mr.circleFit.fitError;
            radii.push_back(mr.circleFit.radius);
            ++stats.validMarkerCount;
        }
    }

    if (stats.totalReconstructedPoints > 0) {
        stats.avgReprojError = totalReproj / static_cast<float>(stats.totalReconstructedPoints);
    }
    size_t validPlaneCount = 0;
    for (const auto& mr : results) { if (mr.validPlane) ++validPlaneCount; }
    if (validPlaneCount > 0) {
        stats.avgPlaneFitError = totalPlaneErr / static_cast<double>(validPlaneCount);
    }
    if (!radii.empty()) {
        stats.avgCircleFitError = totalCircleErr / static_cast<double>(radii.size());
        double sum = 0;
        for (double r : radii) sum += r;
        stats.avgRadius = sum / static_cast<double>(radii.size());
        if (radii.size() > 1) {
            double sqSum = 0;
            for (double r : radii) sqSum += (r - stats.avgRadius) * (r - stats.avgRadius);
            stats.radiusStd = std::sqrt(sqSum / static_cast<double>(radii.size()));
        }
    }
}

}

struct PointReconstructCPU::Impl {
    PointReconstructCPUParams params_;
    PointReconstructStats stats_;
    cv::Mat P1_;
    cv::Mat P2_;
    cv::Mat Q_;
    cv::Mat extP1_, extP2_, extQ_;
    bool matricesCached_ = false;
    bool hasExternalMatrices_ = false;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const PointReconstructCPUParams& params)
        : params_(params)
    {
        if (params_.fxLeft > 0 && params_.fyLeft > 0 &&
            params_.fxRight > 0 && params_.fyRight > 0) {
            params_.validate();
            buildProjectionMatrices(params_, P1_, P2_);
            matricesCached_ = true;
        }
    }

    void ensureMatrices() {
        if (!matricesCached_) {
            if (hasExternalMatrices_) {
                matricesCached_ = true;
            } else {
                params_.validate();
                buildProjectionMatrices(params_, P1_, P2_);
                matricesCached_ = true;
            }
        }
    }

    PointReconstructCPUResult ExecuteFromPoints(
        const std::vector<double>& leftU, const std::vector<double>& leftV,
        const std::vector<double>& rightU, const std::vector<double>& rightV,
        const std::vector<int>& leftGroupIds, const std::vector<int>& rightGroupIds,
        const std::vector<int>& centerMatches)
    {
        PointReconstructCPUResult result;
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

        auto totalStart = std::chrono::high_resolution_clock::now();
        stats_ = PointReconstructStats();

        ensureMatrices();

        std::unordered_map<int, std::vector<size_t>> leftGroupMap;
        for (size_t i = 0; i < leftGroupIds.size(); ++i) {
            leftGroupMap[leftGroupIds[i]].push_back(i);
        }
        std::unordered_map<int, std::vector<size_t>> rightGroupMap;
        for (size_t i = 0; i < rightGroupIds.size(); ++i) {
            rightGroupMap[rightGroupIds[i]].push_back(i);
        }

        result.markerResults.reserve(centerMatches.size());
        size_t processedMarkers = 0;

        for (size_t leftIdx = 0; leftIdx < centerMatches.size(); ++leftIdx) {
            int rightIdx = centerMatches[leftIdx];
            if (rightIdx < 0) continue;
            if (processedMarkers >= params_.maxMarkerCount) break;

            auto leftIt = leftGroupMap.find(static_cast<int>(leftIdx));
            auto rightIt = rightGroupMap.find(rightIdx);
            if (leftIt == leftGroupMap.end() || rightIt == rightGroupMap.end()) continue;

            const auto& lIdxs = leftIt->second;
            const auto& rIdxs = rightIt->second;
            size_t matchCount = std::min(lIdxs.size(), rIdxs.size());

            std::vector<double> lu, lv, ru, rv;
            lu.reserve(matchCount); lv.reserve(matchCount);
            ru.reserve(matchCount); rv.reserve(matchCount);

            for (size_t k = 0; k < matchCount; ++k) {
                size_t li = lIdxs[k];
                size_t ri = rIdxs[k];
                if (li >= leftU.size() || ri >= rightU.size()) continue;
                lu.push_back(leftU[li]);
                lv.push_back(leftV[li]);
                ru.push_back(rightU[ri]);
                rv.push_back(rightV[ri]);
            }

            if (lu.empty()) continue;

            auto tTriStart = std::chrono::high_resolution_clock::now();
            std::vector<ReconstructedPoint3D> reconPts;
            triangulateBatch(lu, lv, ru, rv, P1_, P2_, reconPts);
            auto tTriEnd = std::chrono::high_resolution_clock::now();
            stats_.triangulateTimeMs += std::chrono::duration<double, std::milli>(tTriEnd - tTriStart).count();

            auto tProjStart = std::chrono::high_resolution_clock::now();
            std::vector<ReconstructedPoint3D> filtered;
            filtered.reserve(reconPts.size());
            for (auto& pt : reconPts) {
                if (pt.reprojError <= static_cast<float>(params_.maxReprojError)) {
                    filtered.push_back(std::move(pt));
                }
            }
            auto tProjEnd = std::chrono::high_resolution_clock::now();
            stats_.projectionTimeMs += std::chrono::duration<double, std::milli>(tProjEnd - tProjStart).count();

            MarkerReconstructResult mr;
            mr.leftEllipseIdx = static_cast<int>(leftIdx);
            mr.rightEllipseIdx = rightIdx;
            mr.reconstructedPoints = std::move(filtered);

            if (mr.reconstructedPoints.size() >= static_cast<size_t>(params_.minPointsForPlaneFit)) {
                auto tPlaneStart = std::chrono::high_resolution_clock::now();
                if (fitPlane(mr.reconstructedPoints, mr.planeFit)) {
                    mr.validPlane = true;
                }
                auto tPlaneEnd = std::chrono::high_resolution_clock::now();
                stats_.planeFitTimeMs += std::chrono::duration<double, std::milli>(tPlaneEnd - tPlaneStart).count();
            }

            if (mr.validPlane &&
                mr.reconstructedPoints.size() >= static_cast<size_t>(params_.minPointsForCircleFit)) {
                auto tCircleStart = std::chrono::high_resolution_clock::now();
                if (fitCircleOnPlane(mr.reconstructedPoints, mr.planeFit, mr.circleFit)) {
                    mr.validCircle = true;
                    mr.centerX = mr.circleFit.centerX;
                    mr.centerY = mr.circleFit.centerY;
                    mr.centerZ = mr.circleFit.centerZ;
                    mr.normalX = mr.planeFit.nx;
                    mr.normalY = mr.planeFit.ny;
                    mr.normalZ = mr.planeFit.nz;
                }
                auto tCircleEnd = std::chrono::high_resolution_clock::now();
                stats_.circleFitTimeMs += std::chrono::duration<double, std::milli>(tCircleEnd - tCircleStart).count();
            }

            result.markerResults.push_back(std::move(mr));
            ++processedMarkers;
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        if (params_.collectStatistics) {
            collectStatistics(stats_, result.markerResults);
        }

        result.statistics = stats_;
        result.success = true;
        result.message = "Reconstruction completed";
        result.qualityFlag = calib::QualityFlag::Normal;

        if (stats_.validMarkerCount < stats_.totalMarkerPairs && stats_.totalMarkerPairs > 0) {
            result.qualityFlag = calib::QualityFlag::Warning;
        }

        return result;
    }

    void Warmup(int maxMarkerCount) {
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxMarkerCount={}", maxMarkerCount);
    }

    void SetParams(const PointReconstructCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        if (hasExternalMatrices_) {
            P1_ = extP1_;
            P2_ = extP2_;
        } else {
            params_.validate();
            buildProjectionMatrices(params_, P1_, P2_);
        }
        matricesCached_ = true;
        warmed_up_ = false;
    }

    void setProjectionMatrices(const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& Q) {
        extP1_ = P1.clone();
        extP2_ = P2.clone();
        extQ_ = Q.clone();
        P1_ = extP1_;
        P2_ = extP2_;
        Q_ = extQ_;
        hasExternalMatrices_ = true;
        matricesCached_ = true;
        CALIB_LOG_INFO("setProjectionMatrices() — external P1/P2/Q provided, buildProjectionMatrices skipped");
    }

    void clearProjectionMatrices() {
        extP1_.release();
        extP2_.release();
        extQ_.release();
        Q_.release();
        hasExternalMatrices_ = false;
        matricesCached_ = false;
        CALIB_LOG_INFO("clearProjectionMatrices() — reverting to buildProjectionMatrices");
    }

    const PointReconstructCPUParams& GetParams() const { return params_; }
    const PointReconstructStats& getStatistics() const noexcept { return stats_; }
    void resetStatistics() noexcept { stats_ = PointReconstructStats(); }
};

// ============================================================
// 构造 / 析构
// ============================================================
PointReconstructCPU::PointReconstructCPU(const PointReconstructCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("PointReconstructCPU initialized (fxLeft={}, maxMarkerCount={})",
                   params.fxLeft, params.maxMarkerCount);
}

PointReconstructCPU::~PointReconstructCPU() = default;

void PointReconstructCPU::Destroy() { }

// ============================================================
// reconstruct (EdgeMatchCPUResult)
// ============================================================
PointReconstructCPUResult PointReconstructCPU::Execute(const EdgeMatchCPUResult& edgeMatchResult) {
    CALIB_LOG_DEBUG("reconstruct(EdgeMatchCPUResult) called: ellipseResults={}",
                    edgeMatchResult.ellipseResults.size());

    if (edgeMatchResult.ellipseResults.empty()) {
        PointReconstructCPUResult result;
        result.success = true;
        result.message = "No marker pairs to reconstruct";
        result.qualityFlag = calib::QualityFlag::Normal;
        return result;
    }

    std::vector<double> leftU, leftV, rightU, rightV;
    std::vector<int> leftGroupIds, rightGroupIds, centerMatches;

    for (size_t ei = 0; ei < edgeMatchResult.ellipseResults.size(); ++ei) {
        const auto& eemr = edgeMatchResult.ellipseResults[ei];
        int leftIdx = eemr.leftEllipseIdx;
        int rightIdx = eemr.rightEllipseIdx;

        for (size_t pi = 0; pi < eemr.matchedPairs.size(); ++pi) {
            const auto& pair = eemr.matchedPairs[pi];
            leftU.push_back(pair.leftX);
            leftV.push_back(pair.leftY);
            rightU.push_back(pair.rightX);
            rightV.push_back(pair.rightY);
            leftGroupIds.push_back(leftIdx);
            rightGroupIds.push_back(rightIdx);
        }

        if (static_cast<size_t>(leftIdx) >= centerMatches.size()) {
            centerMatches.resize(leftIdx + 1, -1);
        }
        centerMatches[leftIdx] = rightIdx;
    }

    return pImpl_->ExecuteFromPoints(leftU, leftV, rightU, rightV,
                                   leftGroupIds, rightGroupIds, centerMatches);
}

// ============================================================
// reconstruct (Point2f + groups)
// ============================================================
PointReconstructCPUResult PointReconstructCPU::Execute(const std::vector<cv::Point2f>& leftPoints,
                                      const std::vector<cv::Point2f>& rightPoints,
                                      const std::vector<int>& leftGroupIds,
                                      const std::vector<int>& rightGroupIds,
                                      const std::vector<int>& centerMatches) {
    CALIB_LOG_DEBUG("reconstruct(Point2f) called: left={}, right={}",
                    leftPoints.size(), rightPoints.size());

    if (leftPoints.empty() || rightPoints.empty() || centerMatches.empty()) {
        PointReconstructCPUResult result;
        result.success = true;
        result.message = "Empty input";
        result.qualityFlag = calib::QualityFlag::Normal;
        return result;
    }

    std::vector<double> leftU(leftPoints.size()), leftV(leftPoints.size());
    std::vector<double> rightU(rightPoints.size()), rightV(rightPoints.size());
    for (size_t i = 0; i < leftPoints.size(); ++i) {
        leftU[i] = static_cast<double>(leftPoints[i].x);
        leftV[i] = static_cast<double>(leftPoints[i].y);
    }
    for (size_t i = 0; i < rightPoints.size(); ++i) {
        rightU[i] = static_cast<double>(rightPoints[i].x);
        rightV[i] = static_cast<double>(rightPoints[i].y);
    }

    return pImpl_->ExecuteFromPoints(leftU, leftV, rightU, rightV,
                                   leftGroupIds, rightGroupIds, centerMatches);
}

// ============================================================
// warmup
// ============================================================
void PointReconstructCPU::Warmup(int maxMarkerCount) {
    CALIB_LOG_INFO("warmup() called: maxMarkerCount={}", maxMarkerCount);
    pImpl_->Warmup(maxMarkerCount);
}

void PointReconstructCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// SetProjectionMatrices / ClearProjectionMatrices
// ============================================================
void PointReconstructCPU::SetProjectionMatrices(const cv::Mat& P1, const cv::Mat& P2,
                                                 const cv::Mat& Q) {
    pImpl_->setProjectionMatrices(P1, P2, Q);
}

void PointReconstructCPU::ClearProjectionMatrices() {
    pImpl_->clearProjectionMatrices();
}

// ============================================================
// setParams / getParams
// ============================================================
void PointReconstructCPU::SetParams(const PointReconstructCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const PointReconstructCPUParams& PointReconstructCPU::GetParams() const {
    return pImpl_->GetParams();
}

const PointReconstructStats& PointReconstructCPU::GetStatistics() const noexcept {
    return pImpl_->getStatistics();
}

void PointReconstructCPU::ResetStatistics() noexcept {
    pImpl_->resetStatistics();
}
