#pragma once
#include "mesh_types.h"
#include "utils.h"
#include <Eigen/Dense>
#include <functional>

namespace hole_filler_pro {

// Coarse triangulation via ear clipping (O(n^2)). Same FillResult contract as
// initialFill: boundary verts copied to local 0..n-1, triangles in local ids.
// Used as the unified initial triangulator (S5): handles convex/concave/
// collinear rims; quality is recovered downstream by density remesh +
// triharmonic fairing. Degenerate inputs (duplicate rim vertices, sliver rims
// projecting to ~zero area in every basis) are REJECTED (empty result).
// `base_occupied(a, b)` (optional, GLOBAL vertex ids): an ear whose base
// diagonal coincides with an already-occupied mesh edge is not clipped - the
// clipper triangulates around it (forbidden-diagonal avoidance).
FillResult initialFillEarClip(const BoundaryLoop& loop,
                              const Eigen::MatrixXd& V,
                              const std::function<bool(int, int)>& base_occupied = {});

} // namespace hole_filler_pro
