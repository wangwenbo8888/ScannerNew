#pragma once
#include "mesh_types.h"
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace hole_filler_pro {

// [P0-3] Per-group telemetry for stitchSharedEndpointArcs (who was stitched,
// which guard rejected whom). Plain counts - safe to copy and assert on.
struct ArcStitchResult {
    std::size_t stitched = 0;         // pairs successfully merged
    std::size_t rejected_k = 0;       // guard 1: endpoint pair has != 2 arcs
    std::size_t rejected_gates = 0;   // guard 2: own double gate (edges/perim)
    std::size_t rejected_domain = 0;  // guard 3: empty-domain (see .cpp spec)
};

// [P0-3] Pinch-split hole recovery: merge OPEN ARC pairs that share BOTH
// endpoints into one closed loop (the hole rim was torn in two at pinch
// vertices; the tracer records it as two arcs which are never filled).
// Three guards (six-agent reviewed design, mustclean 2026-11-14):
//   1. EXACTLY two arcs on an endpoint pair (k==2). k>=3 (theta hubs) has
//      pair ambiguity and 4/5 real theta groups contain gate-passing wrong
//      subsets - "best pair" variants are FORBIDDEN (E: load-bearing wall).
//   2. The merged loop must pass its own gates BEFORE merging:
//      (n1+n2-2 <= max_hole_edges) and (p1+p2 <= max_hole_perimeter).
//      Perimeters add exactly - no ghost closing edge (both arcs share the
//      two endpoint vertices, the merged rim is all real boundary edges).
//   3. Empty domain (E-revised): project F's triangle centroids onto the
//      ring's Newell plane; an offender counts ONLY if it shares NO vertex
//      with the rim AND lies within 0.3mm of the plane (far-field curved
//      surfaces, rim-adjacent faces and tangential walls are legitimate and
//      must NOT block - the literal centroid-inside rule killed 17/20
//      eligible holes on the 50W part). A synthetic mirrored flap (shares
//      no rim vertex, h~0) IS caught. Plus D's warpage gate: when the rim
//      bows more than 0.15*avgEdge off its Newell plane the projection is
//      unreliable -> refuse (conservative: no stitch = status quo).
// Direction rule (D1): compare endpoints - if both arcs start at the same
// endpoint, reverse the second (A..B + B..A -> A..B..A). Never rely on the
// tracer's same-direction invariant. Internal vertices are provably
// disjoint (shared internal vertex would be a pinch and split the arcs).
// Determinism: groups ordered by canonical (min,max) endpoint key; the
// merged loop REPLACES the lower-index arc in place (order-preserving
// erase of the other); the rim starts from min(A,B).
// Metadata (perimeter/centroid/normal/avg_edge) is recomputed; is_closed
// becomes true. Mutates ONLY the given loop list (never the mesh).
ArcStitchResult stitchSharedEndpointArcs(std::vector<BoundaryLoop>& loops,
                                         const Eigen::MatrixXd& V,
                                         const Eigen::MatrixXi& F,
                                         const FillParams& params);

}  // namespace hole_filler_pro
