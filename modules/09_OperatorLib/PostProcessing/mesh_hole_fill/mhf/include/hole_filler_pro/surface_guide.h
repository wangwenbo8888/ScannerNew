#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {

// [guide] Robust guide surface for large curved-hole filling (2026-11-16
// merged ruling T1+T2). A bivariate cubic (10 coefficients, "C10", graded
// lexicographic basis) is fitted through a metric band of ORIGINAL mesh
// vertices around a big hole rim; the fairing stage then softly pulls the
// patch toward it. Band membership is decided per vertex during a BFS over
// FACE adjacency from the rim (<= max_rings), with three windows:
//   - radial: rho = min 3D distance to the RIM VERTEX SET. rho in
//     (kBandGap=0.5, band_width] enters the band; rho <= kBandGap is not
//     collected but keeps expanding; rho > band_width stops the BFS.
//     Rim vertices themselves are seeds only (rho == 0, never fitted).
//   - axial: |t| <= clamp(t_rim_max + 1.0, 0.3, 3.0) with t = (p-o).n,
//     o = loop.centroid, n = ref_normal (fixed BEFORE collection - no
//     chicken-and-egg with the band mean). Off-slab vertices are neither
//     collected nor expanded; rim seeds always expand.
//   - guard: vertices of OTHER boundary loops (guard_verts, the union of
//     all work-loop rims built by the orchestrator) are never collected
//     and never expanded (other rims can be torn/folded geometry).
// Thin-wall safety is topological: a back sheet sharing no faces with the
// rim can never enter the band, whatever its distance.
struct GuideFitResult {
    bool ok = false;                                        // a usable fit exists
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();       // frame origin (= loop.centroid)
    Eigen::Vector3d u = Eigen::Vector3d::UnitX();           // tangent axes (u, v, n orthonormal)
    Eigen::Vector3d v = Eigen::Vector3d::UnitY();
    Eigen::Vector3d n = Eigen::Vector3d::UnitZ();
    // Frame coordinates are divided by `scale` (the band radius actually
    // used) before fitting: the design matrix is built on (x/scale,
    // y/scale) and coeffs are stored in the NORMALIZED domain.
    // guideHeight/guideGrad take RAW frame mm coordinates.
    double scale = 1.0;
    // Graded order (i,j): (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3);
    // entries at/after coeff_count are zero.
    double coeffs[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int coeff_count = 0;     // 10 = cubic (C10), 6 = quadratic fallback (Q6)
    int used_pts = 0;        // points kept after the two 2-sigma cuts
    int rejected = 0;        // points dropped by the cuts (N - used_pts)
    double cond = 0.0;       // singular-value condition number of the final fit
};

// Sorted global vertex ids of the band (rim itself NOT included - it is the
// seed set). Empty on degenerate input (empty rim / bad ref_normal /
// band_width <= 0). Deterministic: ids are sorted before return.
std::vector<int> collectGuideBand(const BoundaryLoop& loop,
                                  const Eigen::MatrixXd& V,
                                  const Eigen::MatrixXi& F,
                                  const std::unordered_map<int, std::vector<int>>& global_v2t,
                                  const Eigen::Vector3d& ref_normal,
                                  double band_width,
                                  const std::unordered_set<int>* guard_verts = nullptr,
                                  int max_rings = 10);

// Robust fit of h(x,y) = sum_{i+j<=3} c_ij (x/scale)^i (y/scale)^j in the
// given orthonormal frame. Ladder per the ruling: 10-term cubic -> 6-term
// quadratic -> give up (NO plane rung). A rung is attempted only with
// >= max(50, 10*terms) points; each rung runs THREE fits with TWO 2-sigma
// rejection passes (sigma = inlier RMS of the latest fit). Final guards
// (internal constants): rejection rate <= 0.30, final RMS <= 0.5 mm,
// inliers >= 50; any failure demotes to the next rung. ok=false when every
// rung fails. Rank/condition failure threshold sigma_min/sigma_max <= 1e-6.
GuideFitResult fitRobustCubic(const std::vector<Eigen::Vector3d>& band_pts,
                              const Eigen::Vector3d& origin,
                              const Eigen::Vector3d& u,
                              const Eigen::Vector3d& v,
                              const Eigen::Vector3d& n,
                              double scale = 1.0);

// Convenience: frame build (origin = loop.centroid, n = ref_normal
// normalized, u = least-aligned canonical axis - argmin |a.n| with
// tie-break X < Y < Z - projected tangent, v = n x u) + collectGuideBand +
// fitRobustCubic (scale = band_width).
GuideFitResult fitGuideSurface(const BoundaryLoop& loop,
                               const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               const std::unordered_map<int, std::vector<int>>& global_v2t,
                               const Eigen::Vector3d& ref_normal,
                               double band_width,
                               const std::unordered_set<int>* guard_verts = nullptr);

// Fitted height at RAW frame coordinates (x, y). Valid only when fit.ok;
// returns 0.0 otherwise (coeffs all zero).
double guideHeight(const GuideFitResult& fit, double x, double y);

// Gradient (dh/dx, dh/dy) at RAW frame coordinates, mm/mm. Zero when !ok.
Eigen::Vector2d guideGrad(const GuideFitResult& fit, double x, double y);

} // namespace hole_filler_pro
