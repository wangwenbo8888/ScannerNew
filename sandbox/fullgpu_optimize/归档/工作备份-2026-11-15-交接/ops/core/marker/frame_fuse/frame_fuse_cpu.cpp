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
#include <array>
#include <cmath>
#include <chrono>
#include <map>
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
    // 提速（2026-10-15）：原实现走 cv::Mat＋cv::SVD 通用流程，单次 ~13µs 几乎全是
    // 库开销（3×3 数学本身 <0.1µs）。改三角架构法（封闭公式，零内存申请）：
    //   两侧各用三点构建正交坐标架 [e1 e2 e3]（e1=首边方向，e2=第三点对 e1 的
    //   垂直分量，e3=e1×e2 保右手），R = F_B·F_Aᵀ，T = 质心差。
    // 对精确对应与 Kabsch 结果一致（噪声下差 O(σ/边长)，远小于验证门）；候选
    // 仅用于复核筛选，最终 R/T 仍由下游加权 SVD 精修。退化（首边长度≈0）返 false。
    const cv::Vec3d ab1(p1b.x - p1a.x, p1b.y - p1a.y, p1b.z - p1a.z);
    const cv::Vec3d ac1(p1c.x - p1a.x, p1c.y - p1a.y, p1c.z - p1a.z);
    const double n1 = std::sqrt(ab1.dot(ab1));
    if (n1 < 1e-9) return false;
    const cv::Vec3d e1a = ab1 / n1;
    cv::Vec3d e2a = ac1 - e1a * ac1.dot(e1a);
    double n2 = std::sqrt(e2a.dot(e2a));
    if (n2 < 1e-9) return false;   // 三点共线（垂直分量退化）
    e2a /= n2;
    const cv::Vec3d e3a = e1a.cross(e2a);

    const cv::Vec3d ab2(p2b.x - p2a.x, p2b.y - p2a.y, p2b.z - p2a.z);
    const cv::Vec3d ac2(p2c.x - p2a.x, p2c.y - p2a.y, p2c.z - p2a.z);
    const double m1 = std::sqrt(ab2.dot(ab2));
    if (m1 < 1e-9) return false;
    const cv::Vec3d e1b = ab2 / m1;
    cv::Vec3d e2b = ac2 - e1b * ac2.dot(e1b);
    double m2 = std::sqrt(e2b.dot(e2b));
    if (m2 < 1e-9) return false;
    e2b /= m2;
    const cv::Vec3d e3b = e1b.cross(e2b);

    // R = F_B · F_Aᵀ（两架均右手正交 → det(R)=+1 恒成立，无需修正分支）
    const cv::Matx33d FA(e1a(0), e2a(0), e3a(0),
                         e1a(1), e2a(1), e3a(1),
                         e1a(2), e2a(2), e3a(2));
    const cv::Matx33d FB(e1b(0), e2b(0), e3b(0),
                         e1b(1), e2b(1), e3b(1),
                         e1b(2), e2b(2), e3b(2));
    R = FB * FA.t();

    const cv::Point3d c1((p1a.x + p1b.x + p1c.x) / 3.0,
                         (p1a.y + p1b.y + p1c.y) / 3.0,
                         (p1a.z + p1b.z + p1c.z) / 3.0);
    const cv::Point3d c2((p2a.x + p2b.x + p2c.x) / 3.0,
                         (p2a.y + p2b.y + p2c.y) / 3.0,
                         (p2a.z + p2b.z + p2c.z) / 3.0);
    T = c2 - R * c1;
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

// 零法线静默替换计数（上游质量信号——退化圆/坏法线的可观测出口）
int countZeroNormals(const std::vector<cv::Vec3d>& normals) {
    int c = 0;
    for (const auto& n : normals)
        if (std::sqrt(n.dot(n)) <= 1e-12) ++c;
    return c;
}

} // anonymous namespace

// ============================================================
// Triangle Correspondence Builder (matchMode=1)
// 稀疏共面标记点阵专用：穷举两侧三元组，按三边长度（旋转平移不变）配对，
// 三点闭式解出候选变换后全体互最近邻验证，取支持数最多的变换产对应。
// （源自 tests/frame_fuse_opt 赛马冠军方案 A，2026-10-14）
// ============================================================

namespace {   // 匿名命名空间续块：访问上方 svd3Point/Correspondence

double triDist(const cv::Point3d& a, const cv::Point3d& b) {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 有效边长容差：绝对项与相对项取大（重建噪声/库累积漂移随边长放大）
inline double sideTolOf(double edgeLen, double tolAbs, double tolRatio) {
    return std::max(tolAbs, tolRatio * edgeLen);
}

// 三角形预处理：三边排序＋退化检查＋顶点角色排列
// 角色约定 roles[0]=最短边对角点、roles[1]=最短边∩次短边公共点、roles[2]=最短边∩最长边公共点
// 三边长短分明（相邻边差 > 2×容差）→ 角色唯一，单排列（修复旧版按数组下标指派的
// 50% 错排 bug）；近等腰/等边（任两邻边差 ≤ 2×容差）→ 6 排列靠全体验证裁决。
// 近共线（最短高 < 2% 最长边）→ false（闭式解病态，弃——修复旧版绝对面积判据失效）。
struct TriRoles {
    double sides[3] = {0, 0, 0};                      // 排序后边长（短→长）
    std::vector<std::array<cv::Point3d, 3>> perms;    // 顶点角色排列（1 或 6 种）
};

bool triRoles(const cv::Point3d* P, double tolAbs, double tolRatio, TriRoles& out) {
    struct Edge { double d; int i, j; };
    Edge e[3] = {{triDist(P[0], P[1]), 0, 1},
                 {triDist(P[1], P[2]), 1, 2},
                 {triDist(P[0], P[2]), 0, 2}};
    if (e[0].d < e[1].d) std::swap(e[0], e[1]);
    if (e[1].d < e[2].d) std::swap(e[1], e[2]);
    if (e[0].d < e[1].d) std::swap(e[0], e[1]);       // e[0]≥e[1]≥e[2]
    const double longest = e[0].d;
    const cv::Vec3d ev1(P[1].x - P[0].x, P[1].y - P[0].y, P[1].z - P[0].z);
    const cv::Vec3d ev2(P[2].x - P[0].x, P[2].y - P[0].y, P[2].z - P[0].z);
    const cv::Vec3d cr = ev1.cross(ev2);
    const double h = std::sqrt(cr.dot(cr)) / longest;  // 最短高 = 2×面积/最长边
    if (h < 0.02 * longest) return false;
    out.sides[0] = e[2].d;
    out.sides[1] = e[1].d;
    out.sides[2] = e[0].d;
    auto commonOf = [](const Edge& x, const Edge& y) -> int {
        if (x.i == y.i || x.i == y.j) return x.i;
        return x.j;
    };
    const int v0 = 3 - e[2].i - e[2].j;                // 最短边对角
    const int v1 = commonOf(e[2], e[1]);               // 短∩中
    const int v2 = commonOf(e[2], e[0]);               // 短∩长
    const bool ambSM = (e[1].d - e[2].d) <= 2.0 * sideTolOf(e[1].d, tolAbs, tolRatio);
    const bool ambML = (e[0].d - e[1].d) <= 2.0 * sideTolOf(e[0].d, tolAbs, tolRatio);
    // 排列生成（提速 2026-10-15：分级展开——实测等腰多为 mid≈long 型）：
    //   三边分明 → 1 种（角色唯一）；
    //   仅 mid≈long → 2 种（最短边唯一，v1/v2 互换即可，无需全 6 种）；
    //   short≈mid → 6 种（"最短边"指认不稳，角色整体洗牌，只能全试）
    if (!ambSM && !ambML) {
        out.perms.push_back({P[v0], P[v1], P[v2]});
    } else if (!ambSM) {
        out.perms.push_back({P[v0], P[v1], P[v2]});
        out.perms.push_back({P[v0], P[v2], P[v1]});
    } else {
        int idx[3] = {v0, v1, v2};
        std::sort(idx, idx + 3);
        do {
            out.perms.push_back({P[idx[0]], P[idx[1]], P[idx[2]]});
        } while (std::next_permutation(idx, idx + 3));
    }
    return true;
}

// 全体验证：变换后双向互最近邻＋距离门＋有向法线同向门（镜像翻转解杀手）。
// 返回支持数；pairsOut 收集配对（供解去重与终局输出）。
size_t triVerify(const std::vector<cv::Point3d>& A, const std::vector<cv::Vec3d>& An,
                 const std::vector<cv::Point3d>& B, const std::vector<cv::Vec3d>& Bn,
                 const cv::Matx33d& R, const cv::Vec3d& T,
                 double gateMm, double normalMinCos,
                 std::vector<std::pair<int, int>>* pairsOut) {
    std::vector<cv::Vec3d> at(A.size());
    std::vector<cv::Vec3d> nt(A.size());
    for (size_t a = 0; a < A.size(); ++a) {
        at[a] = R * cv::Vec3d(A[a].x, A[a].y, A[a].z) + T;
        nt[a] = R * An[a];
    }
    size_t cnt = 0;
    for (size_t a = 0; a < at.size(); ++a) {
        double bd = 1e30;
        int bj = -1;
        for (size_t b = 0; b < B.size(); ++b) {
            const cv::Vec3d bp(B[b].x, B[b].y, B[b].z);
            const double d = cv::norm(at[a] - bp);
            if (d < bd) { bd = d; bj = static_cast<int>(b); }
        }
        if (bj < 0 || bd >= gateMm) continue;
        const cv::Vec3d bp(B[static_cast<size_t>(bj)].x, B[static_cast<size_t>(bj)].y,
                           B[static_cast<size_t>(bj)].z);
        double ad = 1e30;
        int ai = -1;
        for (size_t c = 0; c < at.size(); ++c) {
            const double d = cv::norm(at[c] - bp);
            if (d < ad) { ad = d; ai = static_cast<int>(c); }
        }
        if (ai != static_cast<int>(a)) continue;               // 反向互最近邻
        if (nt[a].dot(Bn[static_cast<size_t>(bj)]) < normalMinCos) continue;  // 有向同向
        ++cnt;
        if (pairsOut) pairsOut->emplace_back(static_cast<int>(a), bj);
    }
    return cnt;
}

// 三角形模式诊断（批1 埋点：裁决边际/封顶/胜出配对距离极值；
// 提速步①：分项计时＋计数——定位热点用）
struct TriDiag {
    int solutionMargin = -1;            // 胜出 − 最强竞争解独占对数（<2 已弃权）
    bool capped = false;                // 工作量封顶触发
    double winnerPairDistMaxMm = -1.0;  // 胜出解最大配对距离
    double collectMs = 0.0;             // 三角形枚举预处理
    double scanMs = 0.0;                // 边长门扫描（净）
    double evalMs = 0.0;                // 候选解算（SVD＋全点复核）
    double svdMs = 0.0;                 // 解算内部：三点闭式解（svd3Point）
    double verifyMs = 0.0;              // 解算内部：全点复核（triVerify）
    double judgeMs = 0.0;               // 裁决
    long long pairChecks = 0;           // ta×tb 组合比较次数
    long long evalCount = 0;            // 过门组合数
    long long svdCount = 0;             // 实际 SVD+复核调用数（含排列展开）
};

std::vector<Correspondence> buildTriangleCorrespondences(
    const MarkerPointSet& s1, const MarkerPointSet& s2,
    const FrameFuseCPUParams& p, std::string& note, TriDiag& diag) {
    std::vector<Correspondence> out;
    note.clear();
    diag = TriDiag{};
    const auto& A = s1.positions;
    const auto& An = s1.normals;
    const auto& B = s2.positions;
    const auto& Bn = s2.normals;

    // 枚举预算门（防误配守卫：C(n,3) 超预算直接拒——20000 对解算封顶只封
    // SVD 次数封不住枚举本身；n≈80 时 C(80,3)=82160 已超。子集契约下正常
    // 触不到，专防"triangleMaxPoints 被调大＋全库直配"的组合事故）
    constexpr size_t kMaxTrianglesPerSide = 50000;
    const auto nCk3 = [](size_t n) -> size_t { return n >= 3 ? n * (n - 1) * (n - 2) / 6 : 0; };
    if (nCk3(A.size()) > kMaxTrianglesPerSide || nCk3(B.size()) > kMaxTrianglesPerSide) {
        note = "枚举预算超限（C(n,3)>" + std::to_string(kMaxTrianglesPerSide)
               + "，检查点数上限与子集装配）";
        CALIB_LOG_WARN("triangle: {}", note);
        return out;
    }

    // 两侧三角形预处理（含退化剔除）——提速步① 计时：枚举预处理
    const auto tCollect0 = std::chrono::steady_clock::now();
    struct TriInfo { TriRoles r; };
    std::vector<TriInfo> ta, tb;
    auto collect = [&](const std::vector<cv::Point3d>& P, std::vector<TriInfo>& dst) {
        for (size_t i = 0; i < P.size(); ++i)
         for (size_t j = i + 1; j < P.size(); ++j)
          for (size_t k = j + 1; k < P.size(); ++k) {
            const cv::Point3d pts[3] = {P[i], P[j], P[k]};
            TriRoles r;
            if (!triRoles(pts, p.triangleSideTolMm, p.triangleSideTolRatio, r)) continue;
            dst.push_back({r});
         }
    };
    collect(A, ta);
    collect(B, tb);
    diag.collectMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tCollect0).count();
    if (ta.empty() || tb.empty()) {
        note = "无有效三角形（近共线/退化）";
        return out;
    }

    const double normalMinCos = std::cos(p.triangleNormalMaxAngleDeg * CV_PI / 180.0);
    // 批3 F10：验证门深度自适应——噪声随 Z² 放大（双目 σz ∝ Z²/(B·f)），
    // 固定门在远距会误杀真对。有效门 = max(基线, min(2.0, ratio × Z 中位))。
    // 近距 Z≈330 → 0.66 → max(1.0, 0.66) = 1.0（与批1定标一致，零行为变化）；
    // 远距 Z=1000 → 2.0（远距噪声余量）。Z 取 set1（当前帧）中位数——免费可得。
    double medianZ = 0.0;
    {
        std::vector<double> zs;
        zs.reserve(A.size());
        for (const auto& q : A) zs.push_back(std::abs(q.z));
        std::sort(zs.begin(), zs.end());
        medianZ = zs.empty() ? 0.0 : zs[zs.size() / 2];
    }
    const double effectiveGateMm = std::max(p.triangleVerifyGateMm,
        std::min(2.0, p.triangleVerifyGateRatio * medianZ));
    // 解账本：同一真解会被多个三角形候选反复发现——按配对集去重，防"自并列"误弃权
    std::map<std::vector<std::pair<int, int>>, std::pair<cv::Matx33d, cv::Vec3d>> solutions;
    size_t evalCount = 0;
    bool capped = false;
    // 提速步① 计时：扫描（净）＝组合比较；解算（SVD＋复核）单独累计后从总循环时间中扣除
    const auto tScan0 = std::chrono::steady_clock::now();
    double evalMsAcc = 0.0;
    std::vector<std::pair<int, int>> evalPairsBuf;   // 复用配对缓冲（免每候选堆分配）
    evalPairsBuf.reserve(32);
    for (const auto& xa : ta) {
        for (const auto& xb : tb) {
            ++diag.pairChecks;
            bool pass = true;
            for (int e = 0; e < 3; ++e) {
                if (std::abs(xa.r.sides[e] - xb.r.sides[e])
                    > sideTolOf(std::max(xa.r.sides[e], xb.r.sides[e]),
                                p.triangleSideTolMm, p.triangleSideTolRatio)) {
                    pass = false;
                    break;
                }
            }
            if (!pass) continue;
            const auto tEval0 = std::chrono::steady_clock::now();
            for (const auto& pa : xa.r.perms) {
                for (const auto& pb : xb.r.perms) {
                    if (++evalCount > static_cast<size_t>(p.triangleMaxCandidatePairs)) {
                        capped = true;
                        break;
                    }
                    cv::Matx33d Rc;
                    cv::Vec3d Tc;
                    const auto tSvd0 = std::chrono::steady_clock::now();
                    if (!svd3Point(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], Rc, Tc))
                        continue;   // 退化三点（共线/重合）——提速版现可拒
                    const auto tSvd1 = std::chrono::steady_clock::now();
                    diag.svdMs += std::chrono::duration<double, std::milli>(tSvd1 - tSvd0).count();
                    evalPairsBuf.clear();   // 复用缓冲（提速：免每候选堆分配）
                    if (triVerify(A, An, B, Bn, Rc, Tc, effectiveGateMm,
                                  normalMinCos, &evalPairsBuf) >= 3)
                        solutions.emplace(evalPairsBuf, std::make_pair(Rc, Tc));
                    diag.verifyMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tSvd1).count();
                    ++diag.svdCount;
                }
                if (capped) break;
            }
            evalMsAcc += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tEval0).count();
            ++diag.evalCount;
            if (capped) break;
        }
        if (capped) break;
    }
    {
        const double loopMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tScan0).count();
        diag.scanMs = loopMs - evalMsAcc;   // 净扫描＝总循环 − 解算
        diag.evalMs = evalMsAcc;
    }
    if (capped)
        CALIB_LOG_WARN("triangle: 工作量封顶（{} 对解已评估，上限 {}）——结果可能不完整",
                       evalCount, p.triangleMaxCandidatePairs);
    diag.capped = capped;

    // 裁决（2026-10-15 批1 升级为竞争边际制）：支持数最多者胜；
    // 最强竞争解＝与胜出解【不含公共配对】的解中独占对数最多者（真解的子集解
    // 独占=0 不构成竞争——真实序列中子集解合法存在，按全体解算次优会误伤）；
    // 边际（bestC−maxExclusive）<2 → 弃权：防"严格并列被噪声打破后错解险胜"
    // 的窗口（规则网格/镜像/重复 motif 场景），也覆盖旧版严格并列保护。
    const auto tJudge0 = std::chrono::steady_clock::now();
    if (solutions.empty()) {
        diag.judgeMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tJudge0).count();
        note = capped ? "工作量封顶且无共识解" : "无候选通过验证";
        return out;
    }
    const std::vector<std::pair<int, int>>* bestKey = nullptr;
    int bestC = -1;
    for (const auto& kv : solutions) {
        const int c = static_cast<int>(kv.first.size());
        if (c > bestC) { bestC = c; bestKey = &kv.first; }
    }
    if (bestKey == nullptr || bestC < 3) {
        note = "共识解支持数不足";
        return out;
    }
    int maxExclusive = 0;
    for (const auto& kv : solutions) {
        if (&kv.first == bestKey) continue;
        int ex = 0;
        for (const auto& pr : kv.first) {
            bool inBest = false;
            for (const auto& b : *bestKey)
                if (b == pr) { inBest = true; break; }
            if (!inBest) ++ex;
        }
        if (ex > maxExclusive) maxExclusive = ex;
    }
    diag.solutionMargin = bestC - maxExclusive;
    diag.judgeMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tJudge0).count();
    if (diag.solutionMargin < 2) {
        note = "歧义：胜出解边际不足（margin=" + std::to_string(diag.solutionMargin)
               + "，规则布点/对称/重复场景），弃权";
        out.clear();
        return out;
    }
    const auto& bestRT = solutions.at(*bestKey);
    for (const auto& pr : *bestKey) {
        const cv::Vec3d at = bestRT.first
            * cv::Vec3d(A[static_cast<size_t>(pr.first)].x,
                        A[static_cast<size_t>(pr.first)].y,
                        A[static_cast<size_t>(pr.first)].z) + bestRT.second;
        const cv::Vec3d bp(B[static_cast<size_t>(pr.second)].x,
                           B[static_cast<size_t>(pr.second)].y,
                           B[static_cast<size_t>(pr.second)].z);
        Correspondence c;
        c.idx1 = pr.first;
        c.idx2 = pr.second;
        c.distance = cv::norm(at - bp);
        diag.winnerPairDistMaxMm = std::max(diag.winnerPairDistMaxMm, c.distance);
        out.push_back(c);
    }
    return out;
}

} // anonymous namespace (triangle)

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

    explicit Impl(const FrameFuseCPUParams& params) : params_(params) {
        // 构造即校验（防误配守卫：fromJson 路径此前不校验，非法配置迟到失败
        // 无早告警——启动即炸优于运行中静默劣化）
        params_.validate();
    }

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
        // 尺寸一致＋坐标/法线有限性（负例防护：不抛异常不越界，干净失败）
        {
            auto finiteAll = [](const MarkerPointSet& s) {
                if (s.positions.size() != s.normals.size()) return false;
                for (const auto& p : s.positions)
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                        return false;
                for (const auto& n : s.normals)
                    if (!std::isfinite(n(0)) || !std::isfinite(n(1)) || !std::isfinite(n(2)))
                        return false;
                return true;
            };
            if (!finiteAll(set1) || !finiteAll(set2)) {
                result.success = false;
                result.message = "Invalid input: positions/normals size mismatch or non-finite values";
                return result;
            }
        }

        // Normalize normals（零法线静默替换计数——上游退化圆的可观测信号）
        MarkerPointSet s1 = set1, s2 = set2;
        stats_.zeroNormalCount = countZeroNormals(s1.normals) + countZeroNormals(s2.normals);
        if (stats_.zeroNormalCount > 0)
            CALIB_LOG_WARN("frame_fuse: 零法线输入 {} 个（上游退化圆？），"
                           "将以 (0,0,1) 替换参与校验", stats_.zeroNormalCount);
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

        // --- Step 2+3: 找对应（matchMode 分派；2026-10-14 增三角形模式）---
        std::vector<Correspondence> correspondences;
        std::vector<Correspondence> filteredCorrs;
        std::string triNote;
        const bool triangleActive = params_.matchMode == 1
            && std::min(s1.size(), s2.size())
                   <= static_cast<size_t>(params_.triangleMaxPoints);
        if (params_.matchMode == 1 && !triangleActive) {
            // 大声回退（不做静默换路——调用方须可感知）
            CALIB_LOG_WARN("frame_fuse: matchMode=1 但点数超 triangleMaxPoints({})，"
                           "回退 PFH 路径（稀疏共面形态大概率失败）",
                           params_.triangleMaxPoints);
        }
        if (triangleActive) {
            // 三角形模式：稀疏共面标记点阵专用（PFH 在此形态无区分度）；
            // 产物已经全体互最近邻＋有向法线双锁验证，跳过法线预滤
            auto triStart = std::chrono::high_resolution_clock::now();
            TriDiag triDiag;
            correspondences = buildTriangleCorrespondences(s1, s2, params_, triNote, triDiag);
            auto triEnd = std::chrono::high_resolution_clock::now();
            stats_.descriptorTimeMs = std::chrono::duration<double, std::milli>(
                triEnd - triStart).count();
            stats_.matchingTimeMs = 0.0;
            stats_.solutionMargin = triDiag.solutionMargin;
            stats_.triangleCapped = triDiag.capped;
            stats_.winnerPairDistMaxMm = triDiag.winnerPairDistMaxMm;
            stats_.triCollectMs = triDiag.collectMs;
            stats_.triScanMs = triDiag.scanMs;
            stats_.triEvalMs = triDiag.evalMs;
            stats_.triSvdMs = triDiag.svdMs;
            stats_.triVerifyMs = triDiag.verifyMs;
            stats_.triJudgeMs = triDiag.judgeMs;
            stats_.triPairChecks = triDiag.pairChecks;
            stats_.triEvalCount = triDiag.evalCount;
            stats_.triSvdCount = triDiag.svdCount;
            stats_.descriptorCorrespondences = correspondences.size();
            filteredCorrs = correspondences;
            stats_.normalFilteredCorrespondences = filteredCorrs.size();
        } else {
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

        correspondences = matchDescriptors(desc1, desc2, params_.loweRatio);
        stats_.descriptorCorrespondences = correspondences.size();

        // Normal pre-filter
        double preFilterRad = params_.normalPreFilterAngleDeg * CV_PI / 180.0;
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
        }   // else（legacy PFH 路径）

        if (filteredCorrs.size() < 3) {
            result.success = false;
            result.message = triangleActive
                ? ("triangle: " + (triNote.empty() ? "对应不足" : triNote))
                : "Insufficient correspondences after matching";
            result.totalCorrespondences = filteredCorrs.size();
            auto totalEnd = std::chrono::high_resolution_clock::now();
            stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
            result.statistics = stats_;
            return result;
        }

        // --- Step 4: RANSAC ---
        auto ransacStart = std::chrono::high_resolution_clock::now();

        // 三角形模式：法线门统一用三角门（60°）——帧内法线散布可达 7~35°（系统
        // 性偏置而非噪声），PFH 路径沿用的 10°/5° 硬门会把真对误扔（2026-10-15
        // 步1 数据定因：真实数据 matched 3~4/真值7 的损失全部来自下游 5° 门）
        double normalThreshRad = triangleActive
            ? params_.triangleNormalMaxAngleDeg * CV_PI / 180.0
            : 10.0 * CV_PI / 180.0;
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

                // 提速步①热点修复（2026-10-15）：全内点（w==1）时早退公式分母
                // log(0) 被旧守卫跳过 → 完美共识反而跑满 5000 轮（实测 20~38ms
                // 全耗在此）。全部配对皆内点＝不可能更好，直接收工。
                if (bestInliers == numCorrs) {
                    maxIter = iter + 1;   // 本轮已是最优，下一轮判定即退出
                    break;
                }
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
        double fineNormalThreshRad = triangleActive
            ? params_.triangleNormalMaxAngleDeg * CV_PI / 180.0
            : 5.0 * CV_PI / 180.0;

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

        CALIB_LOG_DEBUG("Fuse completed: matched={}, overlap={:.1f}%, RMSE={:.4f}, time={:.1f}ms",
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
