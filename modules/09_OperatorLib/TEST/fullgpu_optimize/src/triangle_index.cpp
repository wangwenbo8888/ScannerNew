// triangle_index.cpp — 三角形索引实现（frame_fuse 优化步5）
#include "triangle_index.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <utility>

namespace scr {

namespace {

double dist3(const cv::Point3d& a, const cv::Point3d& b) {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 有向法线夹角余弦（分正反——镜像/背面在此分离）
double orientedDot(const cv::Vec3d& a, const cv::Vec3d& b) { return a.dot(b); }

// 种子聚类分面：到任一种子有向余弦 ≥ cos(60°) 归该组；否则自立新种子。
// "到种子"（而非逐对连通）限制组直径，防曲面连锁把正反面串成一组。
std::vector<int> faceGroups(const std::vector<cv::Vec3d>& normals) {
    constexpr double kMinCos = 0.5;   // cos(60°)
    std::vector<cv::Vec3d> seeds;
    std::vector<int> gid(normals.size(), -1);
    for (size_t i = 0; i < normals.size(); ++i) {
        int g = -1;
        for (size_t s = 0; s < seeds.size(); ++s)
            if (orientedDot(normals[i], seeds[s]) >= kMinCos) { g = static_cast<int>(s); break; }
        if (g < 0) { g = static_cast<int>(seeds.size()); seeds.push_back(normals[i]); }
        gid[i] = g;
    }
    return gid;
}

// 一组共面点：质心去心 → SVD 最小特征向量为面法线 → (e1,e2) 平面投影
bool projectGroup(const std::vector<cv::Point3d>& P, const std::vector<int>& idx,
                  std::vector<cv::Point2f>& out2d) {
    if (idx.size() < 3) return false;
    cv::Vec3d c(0, 0, 0);
    for (int i : idx) c += cv::Vec3d(P[static_cast<size_t>(i)].x,
                                     P[static_cast<size_t>(i)].y,
                                     P[static_cast<size_t>(i)].z);
    c /= static_cast<double>(idx.size());
    cv::Mat A(static_cast<int>(idx.size()), 3, CV_64FC1);
    for (size_t r = 0; r < idx.size(); ++r) {
        A.at<double>(static_cast<int>(r), 0) = P[static_cast<size_t>(idx[r])].x - c(0);
        A.at<double>(static_cast<int>(r), 1) = P[static_cast<size_t>(idx[r])].y - c(1);
        A.at<double>(static_cast<int>(r), 2) = P[static_cast<size_t>(idx[r])].z - c(2);
    }
    cv::Mat U, S, Vt;
    cv::SVD::compute(A, S, U, Vt);        // 奇异值降序：Vt 行0/1＝面内主方向，行2＝法线
    cv::Vec3d e1(Vt.at<double>(0, 0), Vt.at<double>(0, 1), Vt.at<double>(0, 2));
    cv::Vec3d e2(Vt.at<double>(1, 0), Vt.at<double>(1, 1), Vt.at<double>(1, 2));
    out2d.clear();
    for (int i : idx) {
        const cv::Vec3d d(P[static_cast<size_t>(i)].x - c(0),
                          P[static_cast<size_t>(i)].y - c(1),
                          P[static_cast<size_t>(i)].z - c(2));
        out2d.emplace_back(static_cast<float>(d.dot(e1)), static_cast<float>(d.dot(e2)));
    }
    return true;
}

// 2D Delaunay：返回三角形顶点在组内的下标三元组（异常/退化返回空）
std::vector<std::array<int, 3>> delaunayGroup(const std::vector<cv::Point2f>& pts2d) {
    std::vector<std::array<int, 3>> out;
    try {
        if (pts2d.size() < 3) return out;
        float minx = pts2d[0].x, maxx = pts2d[0].x;
        float miny = pts2d[0].y, maxy = pts2d[0].y;
        for (const auto& q : pts2d) {
            minx = std::min(minx, q.x); maxx = std::max(maxx, q.x);
            miny = std::min(miny, q.y); maxy = std::max(maxy, q.y);
        }
        const cv::Rect rect(static_cast<int>(std::floor(minx)) - 10,
                            static_cast<int>(std::floor(miny)) - 10,
                            static_cast<int>(std::ceil(maxx)) - static_cast<int>(std::floor(minx)) + 20,
                            static_cast<int>(std::ceil(maxy)) - static_cast<int>(std::floor(miny)) + 20);
        if (rect.width <= 0 || rect.height <= 0) return out;
        cv::Subdiv2D subdiv(rect);
        for (const auto& q : pts2d) subdiv.insert(q);
        std::vector<cv::Vec6f> triList;
        subdiv.getTriangleList(triList);
        for (const auto& t : triList) {
            const cv::Point2f a(t[0], t[1]), b(t[2], t[3]), c(t[4], t[5]);
            int ia = -1, ib = -1, ic = -1;
            for (size_t k = 0; k < pts2d.size(); ++k) {
                if (ia < 0 && std::abs(pts2d[k].x - a.x) < 1e-4f
                    && std::abs(pts2d[k].y - a.y) < 1e-4f) ia = static_cast<int>(k);
                if (ib < 0 && std::abs(pts2d[k].x - b.x) < 1e-4f
                    && std::abs(pts2d[k].y - b.y) < 1e-4f) ib = static_cast<int>(k);
                if (ic < 0 && std::abs(pts2d[k].x - c.x) < 1e-4f
                    && std::abs(pts2d[k].y - c.y) < 1e-4f) ic = static_cast<int>(k);
            }
            if (ia < 0 || ib < 0 || ic < 0) continue;
            if (ia == ib || ib == ic || ia == ic) continue;
            out.push_back({ia, ib, ic});
        }
    } catch (const cv::Exception&) {
        // 近重复/共线输入——该组放弃剖分（调用方仍有其他组/快路径兜底）
    }
    return out;
}

} // namespace

TriangleLibraryIndex::BuildStat TriangleLibraryIndex::rebuild(
    const std::vector<cv::Point3d>& positions, const std::vector<cv::Vec3d>& normals) {
    BuildStat st;
    const auto t0 = std::chrono::high_resolution_clock::now();
    tris_.clear();
    edgeKeys_.clear();
    lastPointCount_ = static_cast<int>(positions.size());
    st.pointCount = lastPointCount_;
    if (positions.size() < 3 || positions.size() != normals.size()) return st;

    const std::vector<int> gid = faceGroups(normals);
    std::map<int, std::vector<int>> groups;
    for (size_t i = 0; i < positions.size(); ++i) groups[gid[i]].push_back(static_cast<int>(i));

    for (const auto& kv : groups) {
        const std::vector<int>& idx = kv.second;
        std::vector<cv::Point2f> pts2d;
        if (!projectGroup(positions, idx, pts2d)) continue;
        const auto local = delaunayGroup(pts2d);
        if (local.empty()) continue;
        ++st.groupCount;
        for (const auto& lt : local) {
            const cv::Point3d& P0 = positions[static_cast<size_t>(idx[static_cast<size_t>(lt[0])])];
            const cv::Point3d& P1 = positions[static_cast<size_t>(idx[static_cast<size_t>(lt[1])])];
            const cv::Point3d& P2 = positions[static_cast<size_t>(idx[static_cast<size_t>(lt[2])])];
            double s[3] = {dist3(P0, P1), dist3(P1, P2), dist3(P0, P2)};
            std::sort(s, s + 3);
            // 最短高门（相对判据，近共线弃——与算子同款）
            const cv::Vec3d e1(P1.x - P0.x, P1.y - P0.y, P1.z - P0.z);
            const cv::Vec3d e2(P2.x - P0.x, P2.y - P0.y, P2.z - P0.z);
            const cv::Vec3d cr = e1.cross(e2);
            if (std::sqrt(cr.dot(cr)) / s[2] < 0.02 * s[2]) continue;
            Tri t;
            t.v[0] = idx[static_cast<size_t>(lt[0])];
            t.v[1] = idx[static_cast<size_t>(lt[1])];
            t.v[2] = idx[static_cast<size_t>(lt[2])];
            t.s[0] = s[0]; t.s[1] = s[1]; t.s[2] = s[2];
            const int tid = static_cast<int>(tris_.size());
            tris_.push_back(t);
            edgeKeys_.push_back({s[0], s[1], tid});
            ++st.triangleCount;
        }
    }
    std::sort(edgeKeys_.begin(), edgeKeys_.end(),
              [](const EdgeKey& a, const EdgeKey& b) { return a.minEdge < b.minEdge; });
    st.buildMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return st;
}

std::vector<int> TriangleLibraryIndex::query(
    const std::vector<cv::Point3d>& curPositions, int maxCount, QueryStat* stat) const {
    std::vector<int> out;
    if (stat) *stat = QueryStat{};
    if (empty() || curPositions.size() < 3 || maxCount < 3) return out;
    const auto t0 = std::chrono::high_resolution_clock::now();

    // 当前帧三角形：单面假设 → 投影剖分；失败退化为全部三点组合（≤C(10,3)）
    std::vector<std::array<int, 3>> curTris;
    {
        std::vector<int> allIdx(curPositions.size());
        for (size_t i = 0; i < curPositions.size(); ++i) allIdx[i] = static_cast<int>(i);
        std::vector<cv::Point2f> pts2d;
        if (projectGroup(curPositions, allIdx, pts2d))
            curTris = delaunayGroup(pts2d);
        if (curTris.empty()) {
            for (size_t i = 0; i < curPositions.size(); ++i)
             for (size_t j = i + 1; j < curPositions.size(); ++j)
              for (size_t k = j + 1; k < curPositions.size(); ++k)
                curTris.push_back({static_cast<int>(i), static_cast<int>(j),
                                   static_cast<int>(k)});
        }
    }
    if (stat) stat->curTriangleCount = static_cast<int>(curTris.size());

    // 顶点投票
    std::map<int, int> votes;
    int candidateCount = 0;
    for (const auto& ct : curTris) {
        double s[3] = {dist3(curPositions[static_cast<size_t>(ct[0])],
                             curPositions[static_cast<size_t>(ct[1])]),
                       dist3(curPositions[static_cast<size_t>(ct[1])],
                             curPositions[static_cast<size_t>(ct[2])]),
                       dist3(curPositions[static_cast<size_t>(ct[0])],
                             curPositions[static_cast<size_t>(ct[2])])};
        std::sort(s, s + 3);
        const double tol = sideTolOf(std::max(s[0], s[1]));
        // 二分取 [s0-tol, s0+tol] 段 → 次边门
        EdgeKey lo{s[0] - tol, 0, 0}, hi{s[0] + tol, 0, 0};
        auto itLo = std::lower_bound(edgeKeys_.begin(), edgeKeys_.end(), lo,
                     [](const EdgeKey& a, const EdgeKey& b) { return a.minEdge < b.minEdge; });
        auto itHi = std::upper_bound(edgeKeys_.begin(), edgeKeys_.end(), hi,
                     [](const EdgeKey& a, const EdgeKey& b) { return a.minEdge < b.minEdge; });
        for (auto it = itLo; it != itHi; ++it) {
            if (std::abs(it->secondEdge - s[1]) > sideTolOf(std::max(it->secondEdge, s[1])))
                continue;
            // 第三边全等校验（两把钥匙不够时假候选多；三边齐比再投票）
            const Tri& cand = tris_[static_cast<size_t>(it->triId)];
            if (std::abs(cand.s[2] - s[2]) > sideTolOf(std::max(cand.s[2], s[2])))
                continue;
            ++candidateCount;
            for (int v = 0; v < 3; ++v) ++votes[cand.v[v]];
        }
    }
    if (stat) stat->candidateTriangleCount = candidateCount;
    if (static_cast<int>(votes.size()) < 3) return out;

    // top-K（票数降序，并列取小下标——确定性）
    std::vector<std::pair<int, int>> byVote(votes.begin(), votes.end());
    std::sort(byVote.begin(), byVote.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    for (size_t i = 0; i < byVote.size() && static_cast<int>(out.size()) < maxCount; ++i)
        out.push_back(byVote[i].first);
    if (stat)
        stat->queryMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

} // namespace scr
