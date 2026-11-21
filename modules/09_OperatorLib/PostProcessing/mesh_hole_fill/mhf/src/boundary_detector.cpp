#include "hole_filler_pro/boundary_detector.h"
#include "hole_filler_pro/utils.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {

// Boundary detection via arc-splitting walk.
//
// Design (S3a of the fix plan):
//  - The walk unit is a DIRECTED edge, but claims are booked on canonical
//    (undirected) edges so a single-edge arc can never be walked twice.
//  - Any boundary vertex whose boundary degree != 2 is a split point (pinch:
//    two rims meeting at one vertex; or a dangling end). Walks start at split
//    points and TERMINATE when they reach one; a pinch is never traversed
//    through, so two rims sharing a pinch can never merge into one arc.
//  - Residual unclaimed edges after split-point seeding form pure degree-2
//    closed loops.
//  - Deterministic: std::map ordering + sorted adjacency vectors make the
//    result independent of hash iteration order.
//  - Conservation invariant: every count==1 edge is claimed exactly once.
//    Violations print a warning (they indicate inconsistent input topology).
std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    // S1: table building extracted to buildEdgeCount (shared with the entry cleaner).
    long long self_edges = 0;
    std::unordered_map<Edge, int, pair_hash> edge_count = buildEdgeCount(F, &self_edges);
    if (self_edges > 0) {
        std::cerr << "[detectBoundaries] warning: ignored " << self_edges
                  << " degenerate self-edges (a==a)\n";
    }
    return detectBoundaries(V, F, edge_count);
}

namespace {
// [M1] shared walk body: takes pre-collected (unsorted) boundary edges.
std::vector<BoundaryLoop> detectFromBoundaryEdges(const Eigen::MatrixXd& V,
                                                  std::vector<Edge> boundary_edges) {
    std::vector<BoundaryLoop> result;
    std::sort(boundary_edges.begin(), boundary_edges.end());

    // Adjacency over boundary edges. std::map + sorted vectors make the walk
    // deterministic regardless of hash iteration order.
    std::map<int, std::vector<int>> next;
    for (const Edge& e : boundary_edges) {
        next[e.first].push_back(e.second);
        next[e.second].push_back(e.first);
    }
    for (auto& kv : next) std::sort(kv.second.begin(), kv.second.end());

    std::unordered_map<Edge, int, pair_hash> claim_count;
    auto claim = [&](int a, int b) { ++claim_count[canonicalEdge(a, b)]; };
    auto isClaimed = [&](int a, int b) {
        auto it = claim_count.find(canonicalEdge(a, b));
        return it != claim_count.end() && it->second > 0;
    };

    auto emit = [&](std::vector<int> path, bool closed, double perim) {
        if (path.size() < 2) return;
        BoundaryLoop bl;
        int m = static_cast<int>(path.size());
        int nedges = closed ? m : m - 1;
        bl.is_closed = closed;
        bl.perimeter = perim;
        bl.centroid = computeCentroid(path, V);
        bl.normal = computeNormalNewell(path, V, closed);
        bl.avg_edge_length = nedges > 0 ? perim / static_cast<double>(nedges) : 0.0;
        bl.vertex_indices = std::move(path);
        result.push_back(std::move(bl));
    };

    // Directed walk starting along edge (s -> t); claims each traversed edge.
    // Ends when it returns to s (closed arc: pure loop, or self-loop through a
    // single pinch) or reaches a boundary vertex of degree != 2 (open arc).
    std::function<void(int, int)> walk = [&](int s, int t) {
        std::vector<int> path{s};
        double perim = (V.row(s) - V.row(t)).norm();
        claim(s, t);
        int prev = s;
        int cur = t;
        while (true) {
            if (cur == s) {   // closed: check BEFORE push/degree (self-loop case)
                emit(std::move(path), true, perim);
                return;
            }
            path.push_back(cur);
            auto it = next.find(cur);
            if (it == next.end() || it->second.size() != 2) {
                emit(std::move(path), false, perim);   // open arc at split point
                return;
            }
            int n1 = it->second[0], n2 = it->second[1];
            int nxt = (n1 != prev) ? n1 : n2;
            if (isClaimed(cur, nxt)) {
                std::cerr << "[detectBoundaries] warning: re-claiming edge ("
                          << cur << "," << nxt << ") - inconsistent boundary topology\n";
            }
            perim += (V.row(cur) - V.row(nxt)).norm();
            claim(cur, nxt);
            prev = cur;
            cur = nxt;
        }
    };

    // Phase 1: seed from every split point (boundary degree != 2), each
    // unclaimed out-edge starts one walk.
    for (const auto& kv : next) {
        int v = kv.first;
        if (kv.second.size() == 2) continue;
        for (int w : kv.second) {
            if (!isClaimed(v, w)) walk(v, w);
        }
    }

    // Phase 2: residual unclaimed edges form pure degree-2 closed loops.
    for (const Edge& e : boundary_edges) {
        if (claim_count.count(e) == 0) walk(e.first, e.second);
    }

    // Conservation invariant: every boundary edge must be claimed exactly once.
    long long violations = 0;
    for (const Edge& e : boundary_edges) {
        auto it = claim_count.find(e);
        int c = (it == claim_count.end()) ? 0 : it->second;
        if (c != 1) {
            ++violations;
            std::cerr << "[detectBoundaries] warning: boundary edge (" << e.first
                      << "," << e.second << ") claimed " << c
                      << " times (expected 1)\n";
        }
    }
    return result;
}
} // anonymous namespace

std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                           const std::unordered_map<Edge, int, pair_hash>& edge_count) {
    std::vector<Edge> boundary_edges;
    boundary_edges.reserve(edge_count.size() / 8);
    for (const auto& kv : edge_count)
        if (kv.second == 1) boundary_edges.push_back(kv.first);
    return detectFromBoundaryEdges(V, std::move(boundary_edges));
}

std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                           const std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats) {
    // [M1] unified-table convention: self-edges are PRESENT (cnt>=1), filter here.
    std::vector<Edge> boundary_edges;
    boundary_edges.reserve(edge_stats.size() / 8);
    for (const auto& kv : edge_stats)
        if (kv.second.cnt == 1 && kv.first.first != kv.first.second)
            boundary_edges.push_back(kv.first);
    return detectFromBoundaryEdges(V, std::move(boundary_edges));
}

} // namespace hole_filler_pro
