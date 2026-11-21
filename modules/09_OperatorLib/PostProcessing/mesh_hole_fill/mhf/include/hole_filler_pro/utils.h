#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hole_filler_pro {

using Edge = std::pair<int, int>;

Edge canonicalEdge(int a, int b);
// `closed`: when false, the loop is an open arc - the last->first wraparound edge
// is NOT a real edge and is excluded from the computation.
Eigen::Vector3d computeNormalNewell(const std::vector<int>& loop, const Eigen::MatrixXd& V,
                                    bool closed = true);
Eigen::Vector3d computeCentroid(const std::vector<int>& loop, const Eigen::MatrixXd& V);
double computeAvgEdgeLength(const std::vector<int>& loop, const Eigen::MatrixXd& V,
                            bool closed = true);
double triangleArea(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c);

// 64-bit combine + splitmix64 finalizer. The old `h1 ^ (h2 << 1)` had
// systematic collisions on MSVC (identity hash): (0,5) and (2,4) both hashed
// to 10, degrading the large edge tables. Determinism of pipeline RESULTS no
// longer depends on the hash (the arc walk iterates sorted structures), so
// this is a pure performance change.
struct pair_hash {
    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    std::size_t operator()(const std::pair<int, int>& p) const {
        std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.first)) << 32)
                          | static_cast<std::uint32_t>(p.second);
        return static_cast<std::size_t>(splitmix64(key));
    }
};

// S1: edge -> incident-face-count table over F (canonicalEdge keys, self-edges
// from degenerate faces skipped - SAME convention detectBoundaries uses, so the
// table can be shared between the entry cleaner and boundary detection, saving
// an O(F) hash pass). `self_edges` (optional) counts skipped a==a edges.
std::unordered_map<Edge, int, pair_hash> buildEdgeCount(const Eigen::MatrixXi& F,
                                                        long long* self_edges = nullptr);

// [M1] {len,cnt} edge table (bitwise replica of the legacy edge_stats builder
// formerly in hole_filler_pro.cpp). Contract: reserve F*2, NO self-skip,
// first-seen len, operator[] insertion order (fi asc, k 0..2) - lazyedge qsum
// iterates this map, the traversal order is load-bearing.
struct EdgeStat { double len = 0.0; int cnt = 0; };
std::unordered_map<Edge, EdgeStat, pair_hash> buildEdgeStats(const Eigen::MatrixXd& V,
                                                             const Eigen::MatrixXi& F);

} // namespace hole_filler_pro
