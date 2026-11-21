#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>
#include <vector>

namespace hole_filler_pro {

// Triharmonic fairing. Solves L^requested_order on a joint graph of patch + ring
// vertices, with patch topological-boundary verts and ring verts as Dirichlet anchors.
// `requested_order` is clamped to >= 1 (1 = C^0 Laplacian, 2 = C^1 bi-Laplacian,
// 3 = C^2 tri-Laplacian). On success the solver tries order = requested_order and
// degrades to requested_order-1, ..., 1 if the system is singular at a higher order.
// If `achieved_order_out` is non-null it receives the order that succeeded, or 0 if
// all orders failed and vertices were left unchanged.
// `boundary_global_ids`: global vertex ids of the patch rim (new_vertices[0..n-1]).
// With it, ring triangles touching the rim are wired to the correct joint
// vertices instead of being silently dropped (S2 fix: the old code polluted the
// global->joint map with patch-local keys, so any ring global id < n_patch
// collided and every rim-touching ring triangle was discarded).
// [guide] Optional soft target term (2026-11-16 ruling): when `free_targets`
// (indexed by PATCH-LOCAL id, size == new_vertices.size(); entries for
// fixed/rim verts are ignored) and guide_alpha > 0 are given, solveBlock
// solves (Mff + lambda*I) x = rhs + lambda*t with
// lambda = guide_alpha * mean(diag(Mff)) computed per order attempt, so the
// soft pull stays active across the degradation ladder and scale-free vs
// each L^k. A wrong-size target set is dropped with a warning (legacy
// pure-fairing system, never half-applied). lambda == 0 (no targets or
// alpha == 0) leaves the system bit-identical to legacy.
void triharmonicFair(std::vector<Eigen::Vector3d>& new_vertices,
                     std::vector<Eigen::Vector3i>& new_triangles,
                     const NeighborhoodInfo& nb,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     int requested_order,
                     int* achieved_order_out = nullptr,
                     const std::vector<int>& boundary_global_ids = {},
                     const std::vector<Eigen::Vector3d>* free_targets = nullptr,
                     double guide_alpha = 0.0,
                     bool soft_rim = false);

}
