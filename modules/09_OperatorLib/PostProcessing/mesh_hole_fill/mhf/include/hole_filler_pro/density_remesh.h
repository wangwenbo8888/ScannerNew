#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>

namespace hole_filler_pro {
void remeshDensity(std::vector<Eigen::Vector3d>& new_vertices,
                   std::vector<Eigen::Vector3i>& new_triangles,
                   NeighborhoodInfo& nb,
                   const std::vector<int>& boundary_local_indices,
                   const FillParams& params);

// [flip] Topology-quality operator for fill patches (2026-11-18 rim-quality
// diagnosis, fix A+D): a Delaunay-style edge flip over INTERIOR patch edges.
// Two accepted gains, in priority order:
//   1. FOLD removal: current dihedral > 45deg and the flipped dihedral is
//      strictly smaller (kills the 179deg folded edges seen at rims);
//   2. min-angle improvement: the worse new triangle beats the worse old one.
// Legality guards: quad strictly convex in the average plane, opposite edge
// (c,d) not already present, both replacement triangles keep min-angle > 5deg,
// and a flip whose OPPOSITE vertices are BOTH boundary (patch rim) vertices
// is refused (it would mint a new rim-to-rim chord the occupancy ledger has
// never vetted). Deterministic: edges visited in sorted canonical order;
// bounded sweeps (<=5). Vertices are never moved and boundary edges never
// touched, so the patch rim stays bit-identical.
void flipImprove(std::vector<Eigen::Vector3d>& verts,
                 std::vector<Eigen::Vector3i>& tris,
                 const std::vector<int>& boundary_local_indices);

// [rim-lift] Moves interior apexes of rim-band sliver triangles away from
// the unsplit rim chord (tangentially, in the local tangent plane) to
// height ~= sqrt(3)/2 * min leg. Moves interior verts only; topology and
// rim stay bit-identical. No-op when no triangle is below the ratio.
// Runs AFTER fairing (the fairing solve would re-flatten any pre-fairing
// lift) and BEFORE the post-fairing flip (so flips judge post-lift
// geometry). Backoff with fold guard; displacement capped.
void liftRimApexes(std::vector<Eigen::Vector3d>& verts,
                   std::vector<Eigen::Vector3i>& tris,
                   const std::vector<int>& boundary_local_indices,
                   const FillParams& params);
}
