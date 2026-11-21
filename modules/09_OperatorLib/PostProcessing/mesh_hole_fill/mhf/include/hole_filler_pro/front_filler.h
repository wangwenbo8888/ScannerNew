#pragma once
#include "mesh_types.h"
#include "surface_guide.h"
#include <Eigen/Dense>
#include <functional>
#include <unordered_map>

namespace hole_filler_pro {

// [front] Advancing-front initial triangulation for large holes (2026-11-17
// design, docs/plans/2026-11-17-front-fill-design.md). Grows the patch inward
// from the rim ON the fitted guide surface: apex candidates are placed
// equilateral in the guide (u,v) frame and lifted with the exact height
// h(x,y), so the patch is born curvature-aware (vs ear clipping's
// topology-only patch). Same FillResult contract as initialFillEarClip:
// boundary verts at local 0..n-1 in rim order (ORIGINAL positions), interior
// verts appended (lifted positions). Deterministic: shortest-front-edge rule,
// ties by vertex id. Returns an EMPTY result on any guard failure (bad fit,
// degenerate projection, occupied rim edge, stalled front, runaway) - the
// caller falls back to ear clipping (bit-identical legacy path).
// `edge_occupied(a, b)` (optional, GLOBAL vertex ids): if any RIM edge of the
// loop is already occupied the front refuses to start (the ear clipper can
// triangulate around forbidden diagonals; the front cannot).
FillResult initialFillFront(const BoundaryLoop& loop,
                            const Eigen::MatrixXd& V,
                            const GuideFitResult& guide,
                            const std::unordered_map<int, double>& global_mean_edge,
                            const std::function<bool(int, int)>& edge_occupied = {},
                            bool delaunay_flips = false);

} // namespace hole_filler_pro
