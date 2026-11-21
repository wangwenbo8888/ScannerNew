#pragma once
#include "mesh_types.h"
#include "utils.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace hole_filler_pro {

// [marker] One physical reference marker: cylinder axis anchored at `center`
// (a point on the surface), unit `normal` pointing OUT of the material.
struct MarkerSpec {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
};

// [marker] Result of the S0 cylinder cut (compacted mesh + counters).
struct MarkerCutResult {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    int faces_removed = 0;
    int verts_removed = 0;
};

// [marker] Load a marker .asc file: one marker per line "x y z nx ny nz";
// '#' comments / blank lines skipped; malformed rows are counted, warned and
// skipped; normals are normalized at load. Returns false on IO failure or
// when zero valid markers remain.
bool loadMarkersAsc(const std::string& path, std::vector<MarkerSpec>& markers);

// [marker] Ruling #4 normal-validity gate: accept the marker iff some face
// centroid lies within p.marker_surface_max_dist of the center (not
// floating), the area-weighted normal sum of faces within radius r is
// coherent (||n_sum|| > 0.25 * sum_area), and its direction agrees with the
// marker normal (n_sum_hat . m.normal >= p.marker_normal_min_dot). Rejecting
// a flipped marker here is what protects thin-wall back faces.
bool markerNormalValid(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                       const MarkerSpec& m, const FillParams& p);

// [marker] Ruling #2 cut: delete every face that has a vertex inside a
// marker cylinder (t in [-len_back, +len_front] along the axis, radial^2 <=
// r^2, r = diameter/2 * radius_scale) AND whose normalized face normal
// satisfies n_face . axis > p.marker_normal_min_dot. Degenerate faces
// (duplicate indices or |cross| < 1e-14) die unconditionally, but ONLY when
// the face owns a cylinder vertex (surgical - mesh-wide hygiene belongs to
// S1 entry clean). Surviving vertices are compacted (boundary_trimmer.cpp
// idiom). Vertex ownership is first-marker-wins (deterministic).
MarkerCutResult cutMarkerCylinders(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                   const std::vector<MarkerSpec>& markers,
                                   const FillParams& p);

// [marker] Ruling #5 geometric on-site rim match. A loop is a marker rim iff
// it is CLOSED, has 3..marker_match_max_edges vertices, perimeter within
// [0.25, 4.0] * 2*pi*r, |centroid - center| < r + marker_match_margin and
// |Newell(loop) . marker.normal| > marker_match_min_dot (Newell sign is
// arbitrary, hence |.|). One loop matches at most one marker and vice versa
// (first match wins, deterministic in loop/marker order). `matched_marker`
// optionally receives the marker index per loop (-1 = none).
std::vector<bool> matchMarkerLoops(const std::vector<BoundaryLoop>& loops,
                                   const std::vector<MarkerSpec>& markers,
                                   const FillParams& p,
                                   std::vector<int>* matched_marker = nullptr);

// [marker-bay] Telemetry for healMarkerBays (plain counters + grep-able
// refusal strings) and the ghost chords the S7 check must keep out of the
// rim-edge set (canonical form).
struct BayHealResult {
    std::size_t healed = 0;              // bays sealed (cap loop inserted)
    std::vector<std::string> refused;    // "#<mi> reason=Rxx_name"
    std::vector<Edge> chord_edges;       // canonicalEdge(X, Y) per healed cap
    bool modified = false;               // any change to `loops`
};

// [marker-bay] S0.5 bay heal (2026-11-16 three-round ruling). A cut rim
// merged with a dead-end crack survives as ONE interior run of ONE open arc
// (the "bay"). For every marker that is NOT claimed by a closed loop at
// call time (heal-time matchMarkerLoops view - the trigger), find the
// unique open carrier arc, extract the maximal run of consecutive vertices
// inside the capture disc (R_cap = r + marker_bay_cap_margin, axial window
// = cut window relaxed by the same margin), close that run with a ghost
// chord into an artificial closed loop built to pass matchMarkerLoops
// verbatim, and replace the carrier in place with [residual head, cap,
// residual tail] (open arcs, emit-convention metadata). Guards (any reject
// leaves `loops` untouched, refusal recorded): carrier uniqueness; exactly
// one in-disc run; run strictly interior (arc endpoints are pinch/dangling);
// 3 <= run length <= marker_match_max_edges; chord <= 2*R_cap*sin(mouth/2);
// run perimeter <= 4*2*pi*r; in-rim share (radial <= r+1.5) >= 0.6; angular
// sweep >= marker_bay_min_sweep_deg; cap re-check via matchMarkerLoops must
// claim exactly this marker. Two markers never share a carrier (first in
// marker-file order wins, later ones refused R12). Deterministic: marker
// order, loop order, descending apply by loop index. Only `loops` mutates -
// never the mesh.
BayHealResult healMarkerBays(std::vector<BoundaryLoop>& loops,
                             const Eigen::MatrixXd& V,
                             const std::vector<MarkerSpec>& markers,
                             const FillParams& p);

// [marker-bay][P3-R6] lost_to_crack hit test, vertex-based: some arc vertex
// lies within r + marker_match_margin of an UNCLAIMED marker's center (a
// giant merged arc's centroid is far away and never fires the old centroid
// test). Returns the marker id or -1. `marker_claimed` is the main-loop
// match view - sealed rims and healed caps must not report losses.
int markerLostToCrack(const BoundaryLoop& arc, const Eigen::MatrixXd& V,
                      const std::vector<MarkerSpec>& markers,
                      const std::vector<char>& marker_claimed,
                      const FillParams& p);

}  // namespace hole_filler_pro
