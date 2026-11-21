#pragma once
#include "mesh_types.h"
#include "utils.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {

// What the S1 entry clean did (all zero / V_out == V_in for an already-clean
// input). Plain int/bool counters - safe to copy and assert on.
struct EntryCleanReport {
    // S1a: exactly-duplicate vertices merged (exact double key, +-0 fused).
    int verts_merged = 0;
    // S1b: duplicate faces removed. Same-winding duplicates keep the FIRST
    // face; an opposite-winding pair deletes BOTH faces (fake overlap).
    int faces_dup_removed = 0;
    // S1c: zero-area / repeated-index / out-of-range faces removed.
    int faces_degenerate_removed = 0;
    // S2: interior-bowtie fan splits + non-manifold edge pruning, bounded by
    // entry_bowtie_max_iters. An iteration that would increase the
    // boundary-edge count rolls back wholesale.
    int bowtie_iters_done = 0;
    int verts_bowtie_split = 0;        // extra vertex copies created
    int faces_nonmanifold_removed = 0;
    bool bowtie_rolled_back = false;
    // S3: small connected components deleted (opt-in stage).
    int small_cc_removed = 0;
    int faces_small_cc_removed = 0;
    // S2.5: dangling-flap faces peeled (opt-in stage).
    int faces_flap_peeled = 0;
    // S2.6: spike/pit vertices snapped to neighbor centroid (opt-in stage).
    int spikes_snapped = 0;
    // S2.7: sparse-veil faces removed (opt-in stage).
    int faces_veil_removed = 0;
    // Size totals.
    int V_in = 0;
    int F_in = 0;
    int V_out = 0;
    int F_out = 0;
};

struct EntryCleanResult {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    EntryCleanReport report;
};

// S1 entry clean: normalize a raw mesh at the pipeline entry so the boundary
// walk sees sane topology.
//   S1a  merge exactly-duplicate vertices   (exact key; -0.0 fuses with +0.0)
//   S1b  remove duplicate faces             (winding-aware, see report)
//   S1c  remove zero-area faces
//   S2   split INTERIOR bowtie vertices into per-fan copies (each copy offset
//        i*eps along its fan normal, eps = 1e-6 * max|coord| - copies stay
//        pairwise distinct so nothing can merge them back) + prune non-manifold
//        edges (count > 2 -> keep the 2 best faces), at most
//        params.entry_bowtie_max_iters rounds; a round that would RAISE the
//        boundary-edge count is rolled back wholesale. Boundary (pinch/rim)
//        vertices are deliberately NEVER split - the arc-walk depends on them.
//   S3   delete small connected components  (opt-in via
//        entry_small_cc_min_faces > 0; largest CC never deleted)
//   S2.5 dangling-flap peel (opt-in via entry_flap_peel): repeatedly remove
//        faces with >= 2 boundary edges - thin strips / flaps hanging off the
//        main surface by an edge (BPV2 stitch remnants). Small-CC removal
//        cannot touch them because they are edge-CONNECTED to the main body.
//        Capped at 0.5% of faces; runs between S2 and S3.
// Vertices left unreferenced are compacted away.
//
// `edge_stats` is an in/out scratch table: on return it holds the {len,cnt}
// edge table of the RETURNED mesh (M1 unified convention: self-edges present,
// consumers filter a==a at read sites), ready for the EdgeStat overloads of
// detectBoundaries / countNmEdges without another O(F) pass.
EntryCleanResult cleanMeshEntry(const Eigen::MatrixXd& V,
                                const Eigen::MatrixXi& F,
                                const FillParams& params,
                                std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats);

// [exit-nm] mustclean task (R2-A V-A6): count non-manifold edges (cnt >= 3,
// self-edges skipped - buildEdgeCount convention) in an edge table. Read-only.
int countNmEdges(const std::unordered_map<Edge, int, pair_hash>& edge_count);
// [M1] {len,cnt} table overload (self-edges filtered here - read-side guard).
int countNmEdges(const std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats);

// [exit-nm] mustclean task (R2-A V-A7): main-exit second-pass non-manifold
// trim, replicating the S2 scorer (ref normal = largest-area face normal;
// score = n . ref_n desc, then area desc, then fi asc - keep best 2 per edge).
// A loser face is deleted ONLY if each of its other two edges CURRENTLY has
// cnt >= 3 (the decrement keeps it >= 2) or cnt == 1 (the edge belongs to
// this face alone and disappears WITH it - a rim sliver, no interior
// surface is opened). Only cnt == 2 exactly is refused (the edge would drop
// to 1 and open interior surface). Deletions decrement edge_count
// immediately (query -> delete -> decrement takes effect at once).
// nm edges are processed in canonical (first, second) order for determinism.
// tri_list is compacted in place, PRESERVING order; verts are NEVER deleted.
// Returns the number of faces removed.
int pruneNmEdgesExit(const std::vector<Eigen::Vector3d>& verts,
                     std::vector<Eigen::Vector3i>& tri_list,
                     std::unordered_map<Edge, int, pair_hash>& edge_count);

} // namespace hole_filler_pro

