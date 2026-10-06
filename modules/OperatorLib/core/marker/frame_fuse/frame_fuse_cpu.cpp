/**
 * @file frame_fuse_cpu.cpp
 * @brief 单帧快速融合算子 - 实现文件
 *
 * 算法流程：
 *   1. KD-Tree 构建 + KNN 搜索
 *   2. 自适应距离阈值计算
 *   3. PFH 描述子计算（点+法线局部几何特征）
 *   4. 描述子匹配 + 法线预过滤
 *   5. RANSAC 3点粗配准
 *   6. 法线加权 SVD 精配准
 *   7. 质量评估
 */

#include "frame_fuse_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <cassert>

using namespace calib;

OperatorInfo getFrameFuseCPUInfo() {
    return OperatorInfo{"FrameFuseCPU", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CPU};
}

CALIB_DEFINE_LOG_TAG(12, FrameFuseCPU);

namespace {

// ============================================================
// KD-Tree helpers
// ============================================================

cv::Ptr<cv::flann::Index> buildKDTree(const std::vector<cv::Point3d>& points,
                                      cv::Mat& dataMat) {
    int N = static_cast<int>(points.size());
    dataMat.create(N, 3, CV_32FC1);
    for (int i = 0; i < N; ++i) {
        dataMat.at<float>(i, 0) = static_cast<float>(points[i].x);
        dataMat.at<float>(i, 1) = static_cast<float>(points[i].y);
        dataMat.at<float>(i, 2) = static_cast<float>(points[i].z);
    }
    cv::Ptr<cv::flann::Index> index = cv::makePtr<cv::flann::Index>(
        dataMat, cv::flann::KDTreeIndexParams(1));
    return index;
}

double computeMedianKNNDistance(cv::flann::Index& index, const cv::Mat& data,
                                int K) {
    int N = data.rows;
    if (N <= 1) return 1.0;

    int actualK = std::min(K, N);
    cv::Mat indices(N, actualK, CV_32SC1);
    cv::Mat dists(N, actualK, CV_32FC1);
    index.knnSearch(data, indices, dists, actualK,
                    cv::flann::SearchParams(32));

    std::vector<double> firstDists;
    firstDists.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        if (actualK >= 2) {
            firstDists.push_back(static_cast<double>(dists.at<float>(i, 1)));
        }
    }
    if (firstDists.empty()) return 1.0;

    std::sort(firstDists.begin(), firstDists.end());
    size_t mid = firstDists.size() / 2;
    return firstDists[mid];
}

// ============================================================
// PFH Descriptor
// ============================================================

std::vector<float> computePFHDescriptor(
    const cv::Point3d& pi, const cv::Vec3d& ni,
    const std::vector<int>& neighborIndices,
    const std::vector<cv::Point3d>& positions,
    const std::vector<cv::Vec3d>& normals,
    int B1, int B2, int B3)
{
    int descDim = B1 + B2 + B3;
    std::vector<float> descriptor(static_cast<size_t>(descDim), 0.0f);
    int validPairs = 0;

    for (int jIdx : neighborIndices) {
        if (jIdx < 0) continue;
        cv::Point3d pj = positions[static_cast<size_t>(jIdx)];
        cv::Vec3d nj = normals[static_cast<size_t>(jIdx)];

        cv::Vec3d diff(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
        double dist = std::sqrt(diff.dot(diff));
        if (dist < 1e-12) continue;

        // Darboux frame at pi
        cv::Vec3d v = ni;
        cv::Vec3d uRaw = diff.cross(v);
        double uLen = std::sqrt(uRaw.dot(uRaw));
        if (uLen < 1e-12) continue;
        cv::Vec3d u = uRaw / uLen;
        cv::Vec3d w = u.cross(v);

        // Three invariant angles
        double f1 = v.dot(nj);
        cv::Vec3d diffNorm = diff / dist;
        double f2 = u.dot(diffNorm);
        double f3 = std::atan2(w.dot(nj), u.dot(nj));

        // Clamp and bin
        int b1 = static_cast<int>(std::round((f1 + 1.0) / 2.0 * (B1 - 1)));
        int b2 = static_cast<int>(std::round((f2 + 1.0) / 2.0 * (B2 - 1)));
        int b3 = static_cast<int>(std::round((f3 + CV_PI) / (2.0 * CV_PI) * (B3 - 1)));

        b1 = std::max(0, std::min(B1 - 1, b1));
        b2 = std::max(0, std::min(B2 - 1, b2));
        b3 = std::max(0, std::min(B3 - 1, b3));

        descriptor[static_cast<size_t>(b1)] += 1.0f;
        descriptor[static_cast<size_t>(B1 + b2)] += 1.0f;
        descriptor[static_cast<size_t>(B1 + B2 + b3)] += 1.0f;
        validPairs++;
    }

    if (validPairs > 0) {
        float invPairs = 1.0f / static_cast<float>(validPairs);
        for (auto& v : descriptor) v *= invPairs;
    }

    // L2 normalize
    float normSq = 0.0f;
    for (float v : descriptor) normSq += v * v;
    if (normSq > 1e-12f) {
        float invNorm = 1.0f / std::sqrt(normSq);
        for (auto& v : descriptor) v *= invNorm;
    }

    return descriptor;
}

// ============================================================
// Descriptor Matching
// ============================================================

struct Correspondence {
    int idx1;
    int idx2;
    double distance;
};

std::vector<Correspondence> matchDescriptors(
    const std::vector<std::vector<float>>& desc1,
    const std::vector<std::vector<float>>& desc2,
    double loweRatio)
{
    if (desc1.empty() || desc2.empty()) return {};

    int N2 = static_cast<int>(desc2.size());
    int dim = static_cast<int>(desc2[0].size());

    cv::Mat desc2Mat(N2, dim, CV_32FC1);
    for (int i = 0; i < N2; ++i) {
        for (int d = 0; d < dim; ++d) {
            desc2Mat.at<float>(i, d) = desc2[static_cast<size_t>(i)][static_cast<size_t>(d)];
        }
    }

    cv::Ptr<cv::flann::Index> flannIdx = cv::makePtr<cv::flann::Index>(
        desc2Mat, cv::flann::KDTreeIndexParams(1));

    std::vector<Correspondence> result;
    for (int i = 0; i < static_cast<int>(desc1.size()); ++i) {
        cv::Mat query(1, dim, CV_32FC1);
        for (int d = 0; d < dim; ++d) {
            query.at<float>(0, d) = desc1[static_cast<size_t>(i)][static_cast<size_t>(d)];
        }
        cv::Mat indices, dists;
        flannIdx->knnSearch(query, indices, dists, 2, cv::flann::SearchParams(32));

        float bestDist = dists.at<float>(0, 0);
        float secondDist = dists.at<float>(0, 1);

        if (secondDist < 1e-12f || bestDist / secondDist < static_cast<float>(loweRatio)) {
            Correspondence c;
            c.idx1 = i;
            c.idx2 = indices.at<int>(0, 0);
            c.distance = static_cast<double>(bestDist);
            result.push_back(c);
        }
    }
    return result;
}

// ============================================================
// 3-Point SVD
// ============================================================

bool svd3Point(
    const cv::Point3d& p1a, const cv::Point3d& p1b, const cv::Point3d& p1c,
    const cv::Point3d& p2a, const cv::Point3d& p2b, const cv::Point3d& p2c,
    cv::Matx33d& R, cv::Vec3d& T)
{
    cv::Point3d c1((p1a.x + p1b.x + p1c.x) / 3.0,
                   (p1a.y + p1b.y + p1c.y) / 3.0,
                   (p1a.z + p1b.z + p1c.z) / 3.0);
    cv::Point3d c2((p2a.x + p2b.x + p2c.x) / 3.0,
                   (p2a.y + p2b.y + p2c.y) / 3.0,
                   (p2a.z + p2b.z + p2c.z) / 3.0);

    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        p1a.x - c1.x, p1b.x - c1.x, p1c.x - c1.x,
        p1a.y - c1.y, p1b.y - c1.y, p1c.y - c1.y,
        p1a.z - c1.z, p1b.z - c1.z, p1c.z - c1.z);
    cv::Mat B = (cv::Mat_<double>(3, 3) <<
        p2a.x - c2.x, p2b.x - c2.x, p2c.x - c2.x,
        p2a.y - c2.y, p2b.y - c2.y, p2c.y - c2.y,
        p2a.z - c2.z, p2b.z - c2.z, p2c.z - c2.z);

    cv::Mat H = A * B.t();
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
        Rmat.at<double>(0, 0), Rmat.at<double>(0, 1), Rmat.at<double>(0, 2),
        Rmat.at<double>(1, 0), Rmat.at<double>(1, 1), Rmat.at<double>(1, 2),
        Rmat.at<double>(2, 0), Rmat.at<double>(2, 1), Rmat.at<double>(2, 2));
    T = cv::Vec3d(
        c2.x - R(0, 0) * c1.x - R(0, 1) * c1.y - R(0, 2) * c1.z,
        c2.y - R(1, 0) * c1.x - R(1, 1) * c1.y - R(1, 2) * c1.z,
        c2.z - R(2, 0) * c1.x - R(2, 1) * c1.y - R(2, 2) * c1.z);
    return true;
}

bool isCollinear(const cv::Point3d& a, const cv::Point3d& b, const cv::Point3d& c,
                 double eps = 1e-6) {
    cv::Vec3d v1(b.x - a.x, b.y - a.y, b.z - a.z);
    cv::Vec3d v2(c.x - a.x, c.y - a.y, c.z - a.z);
    cv::Vec3d cross = v1.cross(v2);
    return std::sqrt(cross.dot(cross)) < eps;
}

// ============================================================
// Inlier collection
// ============================================================

size_t collectInliers(
    const cv::Matx33d& R, const cv::Vec3d& T,
    const std::vector<cv::Point3d>& pos1, const std::vector<cv::Vec3d>& norm1,
    const std::vector<cv::Point3d>& pos2, const std::vector<cv::Vec3d>& norm2,
    const std::vector<Correspondence>& corrs,
    double distThresh, double normalAngleThreshRad,
    std::vector<std::pair<int, int>>& inlierPairs)
{
    inlierPairs.clear();
    for (const auto& c : corrs) {
        size_t i = static_cast<size_t>(c.idx1);
        size_t j = static_cast<size_t>(c.idx2);
        if (i >= pos1.size() || j >= pos2.size()) continue;

        cv::Vec3d p1(pos1[i].x, pos1[i].y, pos1[i].z);
        cv::Vec3d p2(pos2[j].x, pos2[j].y, pos2[j].z);
        cv::Vec3d p1t(
            R(0, 0) * p1(0) + R(0, 1) * p1(1) + R(0, 2) * p1(2) + T(0),
            R(1, 0) * p1(0) + R(1, 1) * p1(1) + R(1, 2) * p1(2) + T(1),
            R(2, 0) * p1(0) + R(2, 1) * p1(1) + R(2, 2) * p1(2) + T(2));

        cv::Vec3d diff = p1t - p2;
        double dist = std::sqrt(diff.dot(diff));
        if (dist > distThresh) continue;

        cv::Vec3d n1t(
            R(0, 0) * norm1[i](0) + R(0, 1) * norm1[i](1) + R(0, 2) * norm1[i](2),
            R(1, 0) * norm1[i](0) + R(1, 1) * norm1[i](1) + R(1, 2) * norm1[i](2),
            R(2, 0) * norm1[i](0) + R(2, 1) * norm1[i](1) + R(2, 2) * norm1[i](2));
        double nLen = std::sqrt(n1t.dot(n1t));
        if (nLen < 1e-12) continue;
        n1t /= nLen;

        double cosAngle = std::abs(n1t.dot(norm2[j]));
        double angle = std::acos(std::min(1.0, cosAngle));
        if (angle > normalAngleThreshRad) continue;

        inlierPairs.emplace_back(c.idx1, c.idx2);
    }
    return inlierPairs.size();
}

// ============================================================
// Weighted SVD
// ============================================================

bool weightedSVD(
    const std::vector<cv::Point3d>& pos1, const std::vector<cv::Vec3d>& norm1,
    const std::vector<cv::Point3d>& pos2, const std::vector<cv::Vec3d>& norm2,
    const std::vector<std::pair<int, int>>& pairs,
    const cv::Matx33d& R_init, const cv::Vec3d& T_init,
    cv::Matx33d& R_out, cv::Vec3d& T_out)
{
    if (pairs.size() < 3) return false;

    // Compute weights from normal consistency
    std::vector<double> weights(pairs.size());
    double wTotal = 0.0;
    for (size_t k = 0; k < pairs.size(); ++k) {
        int i = pairs[k].first;
        int j = pairs[k].second;
        cv::Vec3d n1(norm1[static_cast<size_t>(i)]);
        cv::Vec3d n1t(
            R_init(0, 0) * n1(0) + R_init(0, 1) * n1(1) + R_init(0, 2) * n1(2),
            R_init(1, 0) * n1(0) + R_init(1, 1) * n1(1) + R_init(1, 2) * n1(2),
            R_init(2, 0) * n1(0) + R_init(2, 1) * n1(1) + R_init(2, 2) * n1(2));
        double nLen = std::sqrt(n1t.dot(n1t));
        if (nLen > 1e-12) n1t /= nLen;
        double cosA = n1t.dot(norm2[static_cast<size_t>(j)]);
        weights[k] = cosA * cosA;
        wTotal += weights[k];
    }

    if (wTotal < 1e-12) return false;

    // Weighted centroids
    cv::Vec3d c1(0, 0, 0), c2(0, 0, 0);
    for (size_t k = 0; k < pairs.size(); ++k) {
        double w = weights[k];
        int i = pairs[k].first;
        int j = pairs[k].second;
        c1 += w * cv::Vec3d(pos1[static_cast<size_t>(i)].x,
                             pos1[static_cast<size_t>(i)].y,
                             pos1[static_cast<size_t>(i)].z);
        c2 += w * cv::Vec3d(pos2[static_cast<size_t>(j)].x,
                             pos2[static_cast<size_t>(j)].y,
                             pos2[static_cast<size_t>(j)].z);
    }
    c1 /= wTotal;
    c2 /= wTotal;

    // Weighted cross-covariance
    cv::Mat H = cv::Mat::zeros(3, 3, CV_64FC1);
    for (size_t k = 0; k < pairs.size(); ++k) {
        double w = weights[k];
        int i = pairs[k].first;
        int j = pairs[k].second;
        cv::Vec3d p1(pos1[static_cast<size_t>(i)].x - c1(0),
                      pos1[static_cast<size_t>(i)].y - c1(1),
                      pos1[static_cast<size_t>(i)].z - c1(2));
        cv::Vec3d p2(pos2[static_cast<size_t>(j)].x - c2(0),
                      pos2[static_cast<size_t>(j)].y - c2(1),
                      pos2[static_cast<size_t>(j)].z - c2(2));
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                H.at<double>(r, c) += w * p1(r) * p2(c);
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

    R_out = cv::Matx33d(
        Rmat.at<double>(0, 0), Rmat.at<double>(0, 1), Rmat.at<double>(0, 2),
        Rmat.at<double>(1, 0), Rmat.at<double>(1, 1), Rmat.at<double>(1, 2),
        Rmat.at<double>(2, 0), Rmat.at<double>(2, 1), Rmat.at<double>(2, 2));
    T_out = cv::Vec3d(
        c2(0) - R_out(0, 0) * c1(0) - R_out(0, 1) * c1(1) - R_out(0, 2) * c1(2),
        c2(1) - R_out(1, 0) * c1(0) - R_out(1, 1) * c1(1) - R_out(1, 2) * c1(2),
        c2(2) - R_out(2, 0) * c1(0) - R_out(2, 1) * c1(1) - R_out(2, 2) * c1(2));
    return true;
}

// ============================================================
// RMSE
// ============================================================

double computeRMSE(const cv::Matx33d& R, const cv::Vec3d& T,
                   const std::vector<cv::Point3d>& pos1,
                   const std::vector<cv::Point3d>& pos2,
                   const std::vector<std::pair<int, int>>& pairs) {
    if (pairs.empty()) return 0.0;
    double sumSq = 0.0;
    for (const auto& p : pairs) {
        cv::Vec3d p1(pos1[static_cast<size_t>(p.first)].x,
                      pos1[static_cast<size_t>(p.first)].y,
                      pos1[static_cast<size_t>(p.first)].z);
        cv::Vec3d p2(pos2[static_cast<size_t>(p.second)].x,
                      pos2[static_cast<size_t>(p.second)].y,
                      pos2[static_cast<size_t>(p.second)].z);
        cv::Vec3d p1t(
            R(0, 0) * p1(0) + R(0, 1) * p1(1) + R(0, 2) * p1(2) + T(0),
            R(1, 0) * p1(0) + R(1, 1) * p1(1) + R(1, 2) * p1(2) + T(1),
            R(2, 0) * p1(0) + R(2, 1) * p1(1) + R(2, 2) * p1(2) + T(2));
        cv::Vec3d diff = p1t - p2;
        sumSq += diff.dot(diff);
    }
    return std::sqrt(sumSq / static_cast<double>(pairs.size()));
}

double computeNormalRMSE(const cv::Matx33d& R, const cv::Vec3d& /*T*/,
                         const std::vector<cv::Vec3d>& norm1,
                         const std::vector<cv::Vec3d>& norm2,
                         const std::vector<std::pair<int, int>>& pairs) {
    if (pairs.empty()) return 0.0;
    double sumAngleSq = 0.0;
    for (const auto& p : pairs) {
        cv::Vec3d n1 = norm1[static_cast<size_t>(p.first)];
        cv::Vec3d n1t(
            R(0, 0) * n1(0) + R(0, 1) * n1(1) + R(0, 2) * n1(2),
            R(1, 0) * n1(0) + R(1, 1) * n1(1) + R(1, 2) * n1(2),
            R(2, 0) * n1(0) + R(2, 1) * n1(1) + R(2, 2) * n1(2));
        double nLen = std::sqrt(n1t.dot(n1t));
        if (nLen > 1e-12) n1t /= nLen;
        double cosA = std::min(1.0, std::abs(n1t.dot(norm2[static_cast<size_t>(p.second)])));
        double angle = std::acos(cosA);
        sumAngleSq += angle * angle;
    }
    return std::sqrt(sumAngleSq / static_cast<double>(pairs.size())) * 180.0 / CV_PI;
}

// ============================================================
// Normal normalization
// ============================================================

void normalizeNormals(std::vector<cv::Vec3d>& normals) {
    for (auto& n : normals) {
        double len = std::sqrt(n.dot(n));
        if (len > 1e-12) {
            n /= len;
        } else {
            n = cv::Vec3d(0, 0, 1);
        }
    }
}

} // anonymous namespace

// ============================================================
// Impl
// ============================================================

struct FrameFuseCPU::Impl {
    FrameFuseCPUParams params_;
    FrameFuseStats stats_;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const FrameFuseCPUParams& params) : params_(params) {}

    FrameFuseCPUResult ExecuteImpl(const MarkerPointSet& set1, const MarkerPointSet& set2)
    {
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected");
        struct ScopedFlag {
            std::atomic<bool>* f;
            ScopedFlag(std::atomic<bool>* f_) : f(f_) {}
            ~ScopedFlag() { f->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        auto totalStart = std::chrono::high_resolution_clock::now();
        stats_ = FrameFuseStats();
        FrameFuseCPUResult result;

        stats_.set1PointCount = set1.size();
        stats_.set2PointCount = set2.size();

        // --- Input validation ---
        if (set1.size() < 3 || set2.size() < 3) {
            result.success = false;
            result.message = "Need at least 3 points in each set";
            return result;
        }
        if (set1.size() > params_.maxPointCount || set2.size() > params_.maxPointCount) {
            result.success = false;
            result.message = "Point count exceeds maxPointCount";
            return result;
        }

        // Normalize normals
        MarkerPointSet s1 = set1, s2 = set2;
        normalizeNormals(s1.normals);
        normalizeNormals(s2.normals);

        // --- Step 1: KD-Tree + KNN ---
        auto knnStart = std::chrono::high_resolution_clock::now();

        cv::Mat data1, data2;
        auto kd1 = buildKDTree(s1.positions, data1);
        auto kd2 = buildKDTree(s2.positions, data2);

        int K = std::min(params_.knnK, static_cast<int>(std::min(s1.size(), s2.size())) - 1);
        K = std::max(K, 3);

        cv::Mat idx1, dist1, idx2, dist2;
        kd1->knnSearch(data1, idx1, dist1, K, cv::flann::SearchParams(32));
        kd2->knnSearch(data2, idx2, dist2, K, cv::flann::SearchParams(32));

        double medianD1 = computeMedianKNNDistance(*kd1, data1, K);
        double medianD2 = computeMedianKNNDistance(*kd2, data2, K);
        double medianD = std::min(medianD1, medianD2);
        if (medianD < 1e-12) medianD = 1.0;

        double epsilonCoarse = 3.0 * medianD * 0.1;
        double epsilonFine = medianD * 0.1;
        stats_.adaptiveDistThreshCoarse = epsilonCoarse;
        stats_.adaptiveDistThreshFine = epsilonFine;

        auto knnEnd = std::chrono::high_resolution_clock::now();
        stats_.knnTimeMs = std::chrono::duration<double, std::milli>(knnEnd - knnStart).count();

        // --- Step 2: PFH Descriptors ---
        auto descStart = std::chrono::high_resolution_clock::now();

        int B1 = params_.descriptorBins1;
        int B2 = params_.descriptorBins2;
        int B3 = params_.descriptorBins3;

        std::vector<std::vector<float>> desc1(s1.size());
        for (size_t i = 0; i < s1.size(); ++i) {
            std::vector<int> neighbors(K);
            for (int k = 0; k < K; ++k) {
                neighbors[static_cast<size_t>(k)] = idx1.at<int>(static_cast<int>(i), k);
            }
            desc1[i] = computePFHDescriptor(s1.positions[i], s1.normals[i],
                                             neighbors, s1.positions, s1.normals,
                                             B1, B2, B3);
        }

        std::vector<std::vector<float>> desc2(s2.size());
        for (size_t i = 0; i < s2.size(); ++i) {
            std::vector<int> neighbors(K);
            for (int k = 0; k < K; ++k) {
                neighbors[static_cast<size_t>(k)] = idx2.at<int>(static_cast<int>(i), k);
            }
            desc2[i] = computePFHDescriptor(s2.positions[i], s2.normals[i],
                                             neighbors, s2.positions, s2.normals,
                                             B1, B2, B3);
        }

        auto descEnd = std::chrono::high_resolution_clock::now();
        stats_.descriptorTimeMs = std::chrono::duration<double, std::milli>(descEnd - descStart).count();

        // --- Step 3: Descriptor Matching ---
        auto matchStart = std::chrono::high_resolution_clock::now();

        auto correspondences = matchDescriptors(desc1, desc2, params_.loweRatio);
        stats_.descriptorCorrespondences = correspondences.size();

        // Normal pre-filter
        double preFilterRad = params_.normalPreFilterAngleDeg * CV_PI / 180.0;
        std::vector<Correspondence> filteredCorrs;
        for (const auto& c : correspondences) {
            size_t i = static_cast<size_t>(c.idx1);
            size_t j = static_cast<size_t>(c.idx2);
            double cosA = std::abs(s1.normals[i].dot(s2.normals[j]));
            double angle = std::acos(std::min(1.0, cosA));
            if (angle < preFilterRad) {
                filteredCorrs.push_back(c);
            }
        }
        stats_.normalFilteredCorrespondences = filteredCorrs.size();

        auto matchEnd = std::chrono::high_resolution_clock::now();
        stats_.matchingTimeMs = std::chrono::duration<double, std::milli>(matchEnd - matchStart).count();

        if (filteredCorrs.size() < 3) {
            result.success = false;
            result.message = "Insufficient correspondences after matching";
            result.totalCorrespondences = filteredCorrs.size();
            auto totalEnd = std::chrono::high_resolution_clock::now();
            stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            result.statistics = stats_;
            return result;
        }

        // --- Step 4: RANSAC ---
        auto ransacStart = std::chrono::high_resolution_clock::now();

        double normalThreshRad = 10.0 * CV_PI / 180.0;
        size_t numCorrs = filteredCorrs.size();
        std::mt19937 rng(42);

        size_t bestInliers = 0;
        cv::Matx33d bestR = cv::Matx33d::eye();
        cv::Vec3d bestT(0, 0, 0);
        std::vector<std::pair<int, int>> bestInlierPairs;
        int bestIter = 0;

        int maxIter = params_.ransacMaxIterations;

        for (int iter = 0; iter < maxIter; ++iter) {
            // Random sample 3 correspondences
            std::uniform_int_distribution<int> dist(0, static_cast<int>(numCorrs - 1));
            int a = dist(rng), b = dist(rng), c = dist(rng);
            if (a == b || b == c || a == c) continue;

            const auto& ca = filteredCorrs[static_cast<size_t>(a)];
            const auto& cb = filteredCorrs[static_cast<size_t>(b)];
            const auto& cc = filteredCorrs[static_cast<size_t>(c)];

            // Non-collinear check
            if (isCollinear(s1.positions[static_cast<size_t>(ca.idx1)],
                           s1.positions[static_cast<size_t>(cb.idx1)],
                           s1.positions[static_cast<size_t>(cc.idx1)])) continue;

            cv::Matx33d R_c;
            cv::Vec3d T_c;
            if (!svd3Point(
                    s1.positions[static_cast<size_t>(ca.idx1)],
                    s1.positions[static_cast<size_t>(cb.idx1)],
                    s1.positions[static_cast<size_t>(cc.idx1)],
                    s2.positions[static_cast<size_t>(ca.idx2)],
                    s2.positions[static_cast<size_t>(cb.idx2)],
                    s2.positions[static_cast<size_t>(cc.idx2)],
                    R_c, T_c)) continue;

            // Check det(R) ~ +1
            double det = cv::determinant(R_c);
            if (std::abs(det - 1.0) > 1e-3) continue;

            // Count inliers
            std::vector<std::pair<int, int>> inlierPairs;
            size_t inliers = collectInliers(R_c, T_c,
                                            s1.positions, s1.normals,
                                            s2.positions, s2.normals,
                                            filteredCorrs,
                                            epsilonCoarse, normalThreshRad,
                                            inlierPairs);

            if (inliers > bestInliers) {
                bestInliers = inliers;
                bestR = R_c;
                bestT = T_c;
                bestInlierPairs = std::move(inlierPairs);
                bestIter = iter;

                // Adaptive iteration count
                double w = static_cast<double>(bestInliers) / static_cast<double>(numCorrs);
                if (w > 0) {
                    double w3 = w * w * w;
                    if (w3 > 0 && w3 < 1.0) {
                        int adaptIter = static_cast<int>(
                            std::ceil(std::log(1.0 - params_.ransacConfidence) / std::log(1.0 - w3)));
                        maxIter = std::min(maxIter, adaptIter);
                    }
                }
            }
        }

        stats_.ransacInliers = bestInliers;
        stats_.ransacIterations = bestIter + 1;

        auto ransacEnd = std::chrono::high_resolution_clock::now();
        stats_.ransacTimeMs = std::chrono::duration<double, std::milli>(ransacEnd - ransacStart).count();

        if (bestInliers < static_cast<size_t>(params_.minInlierCount)) {
            result.success = false;
            result.message = "RANSAC found too few inliers";
            result.totalCorrespondences = numCorrs;
            auto totalEnd = std::chrono::high_resolution_clock::now();
            stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            result.statistics = stats_;
            return result;
        }

        stats_.initialRMSE = computeRMSE(bestR, bestT, s1.positions, s2.positions, bestInlierPairs);

        // --- Step 5: Weighted SVD Refinement ---
        auto refineStart = std::chrono::high_resolution_clock::now();

        cv::Matx33d curR = bestR;
        cv::Vec3d curT = bestT;
        double prevRMSE = stats_.initialRMSE;
        double fineNormalThreshRad = 5.0 * CV_PI / 180.0;

        for (int refIter = 0; refIter < params_.refineIterations; ++refIter) {
            // Re-collect inliers with finer threshold
            std::vector<std::pair<int, int>> refinedPairs;
            collectInliers(curR, curT,
                          s1.positions, s1.normals,
                          s2.positions, s2.normals,
                          filteredCorrs,
                          epsilonFine, fineNormalThreshRad,
                          refinedPairs);

            if (refinedPairs.size() < 3) break;

            cv::Matx33d newR;
            cv::Vec3d newT;
            if (!weightedSVD(s1.positions, s1.normals,
                            s2.positions, s2.normals,
                            refinedPairs, curR, curT, newR, newT)) break;

            curR = newR;
            curT = newT;

            double newRMSE = computeRMSE(curR, curT, s1.positions, s2.positions, refinedPairs);
            if (prevRMSE > 0 && std::abs(prevRMSE - newRMSE) / prevRMSE < params_.refineConvergeRatio) {
                stats_.refineIterationsActual = refIter + 1;
                break;
            }
            prevRMSE = newRMSE;
            stats_.refineIterationsActual = refIter + 1;
        }

        // Final inlier collection for output
        std::vector<std::pair<int, int>> finalPairs;
        collectInliers(curR, curT,
                      s1.positions, s1.normals,
                      s2.positions, s2.normals,
                      filteredCorrs,
                      epsilonFine, fineNormalThreshRad,
                      finalPairs);

        stats_.finalRMSE = computeRMSE(curR, curT, s1.positions, s2.positions, finalPairs);

        auto refineEnd = std::chrono::high_resolution_clock::now();
        stats_.refineTimeMs = std::chrono::duration<double, std::milli>(refineEnd - refineStart).count();

        // --- Step 6: Build Result ---
        result.R = curR;
        result.T = curT;

        // 4x4 homogeneous transform
        result.transform = cv::Matx44d::eye();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result.transform(r, c) = curR(r, c);
            }
            result.transform(r, 3) = curT(r);
        }

        result.rmse = stats_.finalRMSE;
        result.normalRMSE = computeNormalRMSE(curR, curT, s1.normals, s2.normals, finalPairs);
        result.matchedCount = finalPairs.size();
        result.totalCorrespondences = numCorrs;
        result.correspondences = std::move(finalPairs);

        size_t minPoints = std::min(s1.size(), s2.size());
        result.overlapRatio = (minPoints > 0)
            ? static_cast<double>(result.matchedCount) / static_cast<double>(minPoints)
            : 0.0;

        // Quality assessment
        result.success = true;
        result.message = "Frame fusion completed";
        if (result.overlapRatio >= 0.5) {
            result.qualityFlag = calib::QualityFlag::Normal;
        } else if (result.overlapRatio >= 0.3) {
            result.qualityFlag = calib::QualityFlag::Degraded;
        } else {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Low overlap ratio: " + std::to_string(result.overlapRatio);
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
        result.statistics = stats_;

        CALIB_LOG_INFO("Fuse completed: matched={}, overlap={:.1f}%, RMSE={:.4f}, time={:.1f}ms",
                       result.matchedCount, result.overlapRatio * 100.0,
                       result.rmse, stats_.totalTimeMs);

        return result;
    }

    void Warmup(int maxPointCount) {
        warmed_up_ = true;
        CALIB_LOG_INFO("warmup() completed: maxPointCount={}", maxPointCount);
    }

    void SetParams(const FrameFuseCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const FrameFuseCPUParams& GetParams() const { return params_; }
    const FrameFuseStats& getStatistics() const noexcept { return stats_; }
    void resetStatistics() noexcept { stats_ = FrameFuseStats(); }
};

// ============================================================
// Constructor / Destructor
// ============================================================

FrameFuseCPU::FrameFuseCPU(const FrameFuseCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("FrameFuseCPU initialized (knnK={}, maxPointCount={})",
                   params.knnK, params.maxPointCount);
}

FrameFuseCPU::~FrameFuseCPU() = default;

void FrameFuseCPU::Destroy() { }

// ============================================================
// fuse (MarkerPointSet)
// ============================================================

FrameFuseCPUResult FrameFuseCPU::Execute(const MarkerPointSet& set1,
                        const MarkerPointSet& set2) {
    CALIB_LOG_DEBUG("fuse(MarkerPointSet) called: set1={}, set2={}",
                    set1.size(), set2.size());
    return pImpl_->ExecuteImpl(set1, set2);
}

// ============================================================
// fuse (PointReconstructCPUResult)
// ============================================================

FrameFuseCPUResult FrameFuseCPU::Execute(const PointReconstructCPUResult& reconstruct1,
                        const PointReconstructCPUResult& reconstruct2) {
    CALIB_LOG_DEBUG("fuse(PointReconstructCPUResult) called: markers1={}, markers2={}",
                    reconstruct1.markerResults.size(),
                    reconstruct2.markerResults.size());

    auto extractSet = [](const PointReconstructCPUResult& rr) -> MarkerPointSet {
        MarkerPointSet ms;
        for (const auto& mr : rr.markerResults) {
            if (!mr.validCircle) continue;
            ms.positions.emplace_back(mr.centerX, mr.centerY, mr.centerZ);
            ms.normals.emplace_back(mr.normalX, mr.normalY, mr.normalZ);
        }
        return ms;
    };

    MarkerPointSet s1 = extractSet(reconstruct1);
    MarkerPointSet s2 = extractSet(reconstruct2);

    return pImpl_->ExecuteImpl(s1, s2);
}

// ============================================================
// warmup
// ============================================================

void FrameFuseCPU::Warmup(int maxPointCount) {
    CALIB_LOG_INFO("warmup() called: maxPointCount={}", maxPointCount);
    pImpl_->Warmup(maxPointCount);
}

void FrameFuseCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================

void FrameFuseCPU::SetParams(const FrameFuseCPUParams& params) {
    CALIB_LOG_INFO("setParams() called");
    pImpl_->SetParams(params);
}

const FrameFuseCPUParams& FrameFuseCPU::GetParams() const {
    return pImpl_->GetParams();
}

const FrameFuseStats& FrameFuseCPU::GetStatistics() const noexcept {
    return pImpl_->getStatistics();
}

void FrameFuseCPU::ResetStatistics() noexcept {
    pImpl_->resetStatistics();
}
