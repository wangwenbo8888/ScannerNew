#pragma once
#include "mesh_types.h"
#include "utils.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {
std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

// S1 overload: reuse a precomputed edge -> count table (buildEdgeCount). The
// table MUST have been built for exactly this F (same self-edge-skipping
// convention). fillHoles builds it once for the entry-cleaned mesh and hands
// it through, saving a full O(F) pass.
std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                           const std::unordered_map<Edge, int, pair_hash>& edge_count);

// [M1] {len,cnt} table overload. The unified table does NOT skip self-edges,
// so the boundary collection filters a==a here (representation-convention
// migration; provably complementary filter set - see speedopt R10B ruling).
std::vector<BoundaryLoop> detectBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                           const std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats);
}
