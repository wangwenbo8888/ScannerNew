#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>
#include <vector>

namespace hole_filler_pro {

struct TrimResult {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<BoundaryLoop> loops;
};

// Stage G1.5: remove `trim_rings` BFS rings of triangles around each boundary
// loop, then re-detect boundaries on the trimmed mesh. Loops whose trim regions
// overlap (thin walls / close holes) are left UN-trimmed as a group. Returns a
// self-contained trimmed mesh; downstream stages operate entirely on it.
// `protect_loops` (optional): rims that must survive untouched (holes the
// pipeline will skip anyway). A protection zone of `trim_rings + 1` BFS layers
// is grown around each; candidate trim bands are clipped against it, so a
// band can never eat into a protected opening's rim (the "one wrongly sealed
// 55.5-perimeter opening" failure mode of the minimal S8).
TrimResult trimBoundaryRings(const Eigen::MatrixXd& V,
                             const Eigen::MatrixXi& F,
                             const std::vector<BoundaryLoop>& loops,
                             int trim_rings,
                             const std::vector<BoundaryLoop>* protect_loops = nullptr);

} // namespace hole_filler_pro
