#include "hole_filler_pro/utils.h"

namespace hole_filler_pro {

Edge canonicalEdge(int a, int b) {
    return (a < b) ? Edge(a, b) : Edge(b, a);
}

std::unordered_map<Edge, int, pair_hash> buildEdgeCount(const Eigen::MatrixXi& F,
                                                        long long* self_edges) {
    std::unordered_map<Edge, int, pair_hash> edge_count;
    edge_count.reserve(static_cast<std::size_t>(F.rows()) * 3);
    if (self_edges) *self_edges = 0;
    for (int f = 0; f < F.rows(); ++f) {
        for (int k = 0; k < 3; ++k) {
            int a = static_cast<int>(F(f, k));
            int b = static_cast<int>(F(f, (k + 1) % 3));
            if (a == b) {
                if (self_edges) ++*self_edges;
                continue;   // self-edge from a degenerate face: not walkable
            }
            ++edge_count[canonicalEdge(a, b)];
        }
    }
    return edge_count;
}

std::unordered_map<Edge, EdgeStat, pair_hash> buildEdgeStats(const Eigen::MatrixXd& V,
                                                             const Eigen::MatrixXi& F) {
    // [M1] bitwise replica of the legacy inline builder (hole_filler_pro.cpp
    // pre-M1): reserve F*2, no self-skip, first-seen len, operator[] insert.
    std::unordered_map<Edge, EdgeStat, pair_hash> edge_stats;
    edge_stats.reserve(static_cast<std::size_t>(F.rows()) * 2);
    for (int fi = 0; fi < F.rows(); ++fi) {
        for (int k = 0; k < 3; ++k) {
            int a = static_cast<int>(F(fi, k));
            int b = static_cast<int>(F(fi, (k + 1) % 3));
            EdgeStat& s = edge_stats[canonicalEdge(a, b)];
            if (s.cnt == 0) s.len = (V.row(a) - V.row(b)).norm();
            ++s.cnt;
        }
    }
    return edge_stats;
}

Eigen::Vector3d computeNormalNewell(const std::vector<int>& loop, const Eigen::MatrixXd& V,
                                    bool closed) {
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    std::size_t m = loop.size();
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t inext = i + 1;
        if (inext >= m) {
            if (!closed || m == 0) break;
            inext = 0;
        }
        const Eigen::Vector3d& cur = V.row(loop[i]);
        const Eigen::Vector3d& next = V.row(loop[inext]);
        n.x() += (cur.y() - next.y()) * (cur.z() + next.z());
        n.y() += (cur.z() - next.z()) * (cur.x() + next.x());
        n.z() += (cur.x() - next.x()) * (cur.y() + next.y());
    }
    double len = n.norm();
    if (len > 1e-12) return n / len;
    return Eigen::Vector3d::UnitZ();
}

Eigen::Vector3d computeCentroid(const std::vector<int>& loop, const Eigen::MatrixXd& V) {
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int idx : loop) c += V.row(idx);
    return c / static_cast<double>(loop.size());
}

double computeAvgEdgeLength(const std::vector<int>& loop, const Eigen::MatrixXd& V,
                            bool closed) {
    if (loop.size() < 2) return 0.0;
    double total = 0.0;
    std::size_t m = loop.size();
    std::size_t edges = 0;
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t inext = i + 1;
        if (inext >= m) {
            if (!closed) break;
            inext = 0;
        }
        total += (V.row(loop[i]) - V.row(loop[inext])).norm();
        ++edges;
    }
    return edges > 0 ? total / static_cast<double>(edges) : 0.0;
}

double triangleArea(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
}

} // namespace hole_filler_pro
