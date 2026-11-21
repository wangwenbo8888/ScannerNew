#pragma once
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {

struct MarkerSpec;   // [port-261121] marker_cutter.h 前置声明（内存标志点直取）

struct BoundaryLoop {
    std::vector<int> vertex_indices;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double avg_edge_length = 0.0;
    // True when the closing edge (last->first vertex) is a real boundary edge
    // (pure closed loop, or self-loop passing exactly once through a pinch).
    // False for open arcs that terminate at pinch/dangling vertices and must
    // NOT be treated as closed polygons downstream (no ghost closing edge).
    bool is_closed = true;
    // Exact edge-sum perimeter (closed: m edges incl. closing edge; open: m-1).
    double perimeter = 0.0;
};

struct NeighborhoodInfo {
    std::vector<int> ring_vertices;
    // Per-vertex local target edge length (density field).
    // KEYED BY GLOBAL MESH VERTEX ID (not patch-local index).
    // Downstream consumers (density_remesh, triharmonic_fairing) must remap to patch-local
    // indices via loop.vertex_indices in the orchestrator before lookup.
    std::unordered_map<int, double> target_edge;
    std::vector<int> ring_triangles;
    Eigen::Vector3d ref_normal = Eigen::Vector3d::UnitZ();
};

struct FillParams {
    int neighborhood_rings = 2;
    int density_remesh_passes = 3;
    double split_factor = 1.4142135623730951;
    // Reserved for future short-edge collapse; NOT implemented (documented).
    double collapse_factor = 0.70710678118654752;
    int fairing_order = 3;
    int max_hole_edges = 2000;
    double max_hole_perimeter = 50.0;
    int trim_rings = 2;
    // ---- S1 entry clean (plan v5 搂2.1). Default OFF: the 44 existing tests and
    // the library's old behavior stay untouched; the chain enables it via
    // config ("entry_clean": true). When ON, fillHoles normalizes the input
    // before boundary detection (dedup -> bowtie split / nm-edge prune ->
    // optional small-CC removal), replacing the four Open3D cleanup calls that
    // used to run only on the fillHolesFile path.
    bool entry_clean = false;
    // Max S2 bowtie-split iterations; an iteration that would RAISE the
    // boundary-edge count rolls back wholesale. 0 disables the S2 stage.
    int entry_bowtie_max_iters = 3;
    // S3 small-component removal (0 = disabled): delete a CC when it has fewer
    // faces than this AND less area than entry_small_cc_area_ratio x largest.
    // Largest CC is never deleted. Chain default: 50 + 0.01.
    int entry_small_cc_min_faces = 0;
    double entry_small_cc_area_ratio = 0.01;
    // S2.5 dangling-flap peel: remove faces with >=2 boundary edges (thin
    // strips hanging by an edge - BPV2 stitch remnants). Capped 3 rounds /
    // 0.25% of faces. Default OFF (chain enables it).
    bool entry_flap_peel = false;
    // S2.6 spike/pit removal (ported from archived SmoothingCore SpikeRemover
    // v0.2): non-boundary vertex with ring-neighbor plane-deviation ratio
    // |d_v|/d_mean above the threshold snaps to the neighbor centroid. Kills
    // needle/sliver structures (manifold + edge-connected - untouchable by
    // the topological stages). 5 rounds, 1% vertex cap. Default OFF.
    bool entry_spike_remove = false;
    double entry_spike_ratio = 10.0;
    int entry_spike_ring = 2;
    // S2.7 sparse-veil removal: delete faces whose 2nd-longest edge exceeds
    // entry_veil_edge_mult x the global median edge (membranes spanning empty
    // space; ~0.001% of faces on real data). Default OFF.
    bool entry_veil_remove = false;
    double entry_veil_edge_mult = 8.0;

    // [clean-opt] equivalence-preserving fast paths for the S2 stage
    // (docs/清扫优化专题-2026-11-09.md §7/§10). Default false = original
    // code path; both verified byte-identical on the golden input.
    bool s2short = true;    // skip rollback-gate rebuild when iteration changed nothing [M5 verified 50W+25M]
    bool s2merge = true;    // find count>2 edges from the existing table + build face lists only for them [M5 verified]
    bool spikedirty = true; // spike rounds 1..N re-scan only the closed ring-ball of previously snapped vertices [M5 verified]
    bool shareec = true;    // one shared edge-count table for S2/S2.6/S2.5 (incrementally maintained) [M5 verified]
    bool lazyedge = true;   // compute global_mean_edge only for vertices the hole loop will query [M5 verified]
    bool s2fan = true;      // CSR vertex->faces + fast single-fan pre-check (cycle trace) [M5 verified]

    // [cleanup-opt] R1/R2 batch switches (default false = original code path;
    // equivalence verified via golden SHA gates). See cleanup_opt\PROGRESS.md.
    bool s1hash = false;    // S1a/S1b: unordered_map with FNV-1a+splitmix64 hash (map->hash, keep-first identical)
    bool nrmhash = false;   // normal embedding exact-match: std::map -> unordered_map with coord_key_hash
    bool gatecheap = false; // watertight gate: reuse edge_stats instead of rebuilding O(F) edge table
    bool edge_unify = false;  // [M1] build {len,cnt} exit table at compact_emit, move-reuse as edge_stats (saves one O(F) build; traversal-order-safe by identical construction)

    // [exit-nm] exit non-manifold audit/prune (mustclean task, R2-A V-A6/V-A7).
    // audit  = every fillHoles exit path counts cnt>=3 edges O(E) (print only);
    //          covers paths the S2 rollback flag cannot see (max-iters exit,
    //          both early returns, entry_clean=false).
    // prune  = main-exit second-pass trim reusing the S2 scorer (guard: a loser
    //          face is deleted ONLY if its other two edges currently have
    //          cnt>=3, so no new boundary edge is ever created; triangles
    //          only - vertices are NEVER deleted). Default OFF (M1 full-off gate).
    bool exit_nm_audit = false;
    bool exit_nm_prune = false;

    // [P0-2] mustclean follow-up (2026-11-14, three-agent reviewed): exempt
    // CLOSED loops from being unioned with OPEN ARCS in the component
    // perimeter gate - pinhole rims that merely touch the crack/arc network
    // through a pinch vertex were dragged over the total-perimeter limit
    // (120/124 unfilled pinholes on the 50W part). Closed-closed chaining is
    // KEPT (a vertex-pinch dumbbell = two closed loops sharing a vertex must
    // stay gated, S8), and every loop still passes its own <=max_hole_edges
    // and <=max_hole_perimeter gates. Default false = legacy full union.
    bool component_exempt_closed = false;

    // [P0-3] mustclean follow-up (2026-11-14, six-agent reviewed): stitch
    // OPEN-ARC pairs sharing both endpoints into closed loops (pinch-split
    // hole rims) before the fill gates. Guards: k==2 exactly / own double
    // gate pre-check / empty-domain (rim-incident faces exempt, offender
    // must be within 0.3mm of the ring plane) / warpage gate. Interacts
    // with component_exempt_closed: without it the stitched (closed) loop
    // unions with the crack-arc network and the rescue silently dies - a
    // runtime warning is printed in that case. Default false = legacy.
    bool arc_stitch = false;

    // [marker] S0 marker cut & exempt fill (2026-11-15, three-round ruled).
    // Scanners cannot see the surface under reference-marker stickers, so the
    // mesh there is garbage. When ON, fillHoles loads the marker list, checks
    // each marker's normal against the local (area-weighted) surface normals,
    // then - BEFORE S1 entry clean - deletes every face with a vertex inside
    // the marker cylinder (axis = marker normal, t in [-marker_len_back,
    // +marker_len_front], radial <= marker_diameter/2*marker_radius_scale)
    // whose face normal points along the axis (dot > marker_normal_min_dot;
    // thin-wall back faces point against it and survive). The cut rims are
    // then re-detected, geometrically matched (centroid near a marker center,
    // |Newell . axis| coherent, size-bounded) and EXEMPT from trim, the
    // max_hole_edges / max_hole_perimeter / component-perimeter gates, and
    // fillSingleHole's own size gate, so they are always filled. Default OFF:
    // the OFF path is byte-identical to legacy (golden K3 gate).
    bool marker_cut = false;
    // Marker list file: ASCII, one marker per line "x y z nx ny nz",
    // '#' comment lines skipped, normals normalized at load.
    std::string marker_file;
    // [port-261121] 产品路径：内存标志点列表（marker_cloud_fuse 输出直取）。
    // 非空时优先生效，跳过 marker_file 磁盘加载（文件路径仅沙盒链使用）。
    // 生命周期：指向调用方持有的列表，须覆盖 fillHoles 调用期。
    std::vector<MarkerSpec>* marker_list_mem = nullptr;
    // Physical marker diameter, mm (> 0). Cut radius = diameter/2 * scale.
    double marker_diameter = 10.0;
    // Cut-radius scale - slightly > 1 to also remove the noisy ring around
    // the sticker edge (user ruling: "cut a bit more for de-noising").
    // 1.2 verified 10/10 on 50W; 1.5 FAILS there (marker #2 has a crack
    // 7.51mm from center - the bigger rim merges into it and stays open).
    double marker_radius_scale = 1.2;
    // Cylinder extent along +normal (out of the surface), mm (>= 0).
    double marker_len_front = 2.0;
    // Cylinder extent along -normal (into the material), mm (>= 0). Longer
    // than front on purpose (surface bends away on the -n side).
    double marker_len_back = 3.0;
    // Face dies iff face_normal . marker_normal > this (strictly). Default
    // 0.1 (J3 ruling: kills the near-tangential band where the dot SIGN is
    // numerical noise - a dot>0 gate would checkerboard-cut such faces;
    // measured identical output on 50W for both values).
    double marker_normal_min_dot = 0.1;
    // Exempt match: |loop.centroid - marker.center| < r + margin, mm (>= 0).
    double marker_match_margin = 2.0;
    // Exempt match: |Newell(loop) . marker.normal| > this (Newell sign is
    // arbitrary, hence the absolute value). In [0, 1].
    double marker_match_min_dot = 0.5;
    // Exempt match: loops with more vertices never match (crack guard).
    int marker_match_max_edges = 256;
    // Marker sanity: reject markers whose nearest face centroid is farther
    // than this from the marker center, mm (0 = off).
    double marker_surface_max_dist = 2.0;

    // [marker-bay] S0.5 bay heal (2026-11-16, three-round ruled). When a cut
    // rim merges with a dead-end crack, the rim survives as an interior run
    // of ONE open/closed arc (the "bay"). healMarkerBays (right after
    // arc_stitch, before pre_loops) splits that arc into residual arc A +
    // artificial closed cap + residual arc B; the cap passes matchMarkerLoops
    // verbatim (re-checked at build time) and its ghost wraparound chord is
    // kept OUT of the S7 rim set (it lands in the prior==0/own<=2 branch and
    // becomes the residual crack's new legal end after commit). Guards: cap
    // discs of two markers never overlap; exactly one carrier arc with
    // exactly one in-disc run of 3..marker_match_max_edges vertices; open
    // arcs never run into their endpoints; chord <= 2*R_cap*sin(mouth/2);
    // sub-perimeter <= 4*2*pi*r; in-rim share >= 0.6; angular sweep >=
    // marker_bay_min_sweep_deg (a passing straight crack sweeps <= 180 deg,
    // a real bay ~290-310 deg); predicate re-check. Any reject leaves
    // work_loops untouched. Dormant unless marker_cut is on; at scales where
    // every rim closes normally (e.g. 1.2 on 50W) every marker is already
    // claimed and the stage is a no-op (byte-identical output).
    bool marker_bay_heal = true;
    // Capture-disc radius for the bay run = r + this margin, mm (>= 0.5).
    // Must clear the rim polyline overshoot (~1 edge length) so the rim does
    // not fragment into multiple runs on the R_cap circle.
    double marker_bay_cap_margin = 2.0;
    // Minimum angular sweep of the in-disc run around the marker axis,
    // degrees (0..360). 200 separates real bays (~290+) from passing
    // straight cracks (<= 180).
    double marker_bay_min_sweep_deg = 200.0;
    // Maximum bay-mouth opening angle, degrees (0..180). The run's chord
    // must satisfy chord <= 2*R_cap*sin(mouth/2); 90 deg = the mouth may not
    // be wider than a half circle of the capture disc.
    double marker_bay_max_mouth_deg = 90.0;

    // [guide] Large-hole curved-surface guided fill (2026-11-16 ruling).
    // Plain triharmonic fairing pins the rim and minimizes bending energy;
    // on large curved holes the membrane undershoots the true surface
    // (center sag grows with hole size; ~0.5-1.0 mm on the 12-16 mm marker
    // patches of 50W). When ON, fillSingleHole additionally collects a band
    // of ORIGINAL mesh vertices around big rims (face-adjacency BFS from
    // the rim, metric window with inner gap + axial slab, never crossing
    // into other boundary loops), fits a robust bivariate cubic (10 terms,
    // two 2-sigma rejection passes, C10->Q6 degradation on rank failure)
    // through it, and adds a soft target term to the fairing solve:
    // (Mff + lambda*I) x = rhs + lambda*t with
    // lambda = fill_guide_lambda_alpha * mean(diag(Mff)), recomputed per
    // degradation order. Engages only for closed rims with perimeter >=
    // fill_guide_perimeter_min; marker-exempt rims are force-enabled via a
    // per-call threshold clamp in the orchestrator. A failed fit, a refused
    // guard, or alpha == 0 degrades bitwise to the legacy system. Default
    // OFF: the OFF path is byte-identical to legacy (golden gates).
    bool fill_surface_guided = false;
    // Perimeter threshold, mm: closed holes at/above this get the guided
    // fill (0 = every closed hole; marker-exempt rims ignore this via the
    // orchestrator clamp). Small holes below it keep the fast legacy path.
    double fill_guide_perimeter_min = 25.0;
    // Soft-pull strength alpha: lambda = alpha * mean(diag(Mff)) per order.
    // 0 = off (bitwise legacy); larger = follow the guide surface more.
    double fill_guide_lambda_alpha = 0.5;
    // Band outer radius around the rim, mm (inner gap 0.5 and axial slab
    // are internal constants).
    double fill_guide_band_width = 9.0;

    // [front] Advancing-front initial triangulation for large holes (2026-11-17
    // design, docs/plans/2026-11-17-front-fill-design.md). When ON and the rim
    // perimeter >= fill_front_perimeter_min, stage 1 grows the patch inward ON
    // the fitted guide surface (every new vertex projected onto it) instead of
    // ear clipping. Any guard failure (bad fit / stalled front / occupied rim
    // edge / runaway) returns empty and the caller falls back to ear clipping
    // (bit-identical legacy). Default OFF.
    bool fill_front = false;
    double fill_front_perimeter_min = 25.0;

    // [front-flip] 2D Delaunay edge flips inside the front filler (2026-11-20
    // sliver fix). Ear clipping guarantees TOPOLOGY only: on nasty rims its
    // long chords leave sliver strips that neither edge splitting nor
    // tangential relaxation can re-topologize (a strip's vertices stay in the
    // strip). Flipping interior edges to the Delaunay criterion in the guide
    // (u,v) frame restores well-shaped 2D triangles BEFORE refinement; the
    // guided lift preserves them. Rim boundary edges are excluded
    // automatically (single incident triangle); new rim<->rim diagonals
    // respect edge_occupied exactly like the ear clipper's diagonals.
    // Default OFF (bit-identical legacy).
    bool fill_front_flip = false;

    // [rim-soft] Soft-anchor rim vertices during fairing (2026-11-19
    // curvature fix, Agent-1 proposal #2). When ON, rim verts join the FREE
    // set with their original positions as soft targets: the existing λI
    // spring in solveBlock pulls them home while the bending energy from
    // both sides (patch interior + fixed ring) negotiates tangent
    // continuity across the seam — upgrading the junction from C0 toward
    // C1. Default OFF = legacy hard Dirichlet (byte-identical).
    bool rim_soft_anchor = false;

    // [rim-gradient] Density gradient at the patch rim (2026-11-20 rim-quality
    // fix). When ON, remeshDensity seeds its per-pass target cache with a
    // field T(v) = T_in + (T_rim - T_in) * smoothstep(clamp((W-d)/W, 0, 1)),
    // d = BFS ring distance from the rim, W = rim_gradient_rings,
    // T_in = rim_gradient_ratio * mean(rim targets). Rim verts keep their
    // OWN measured target. Split and relaxation read the SAME field.
    // Default OFF = legacy uniform path (byte-identical).
    bool rim_gradient = false;
    double rim_gradient_rings = 4.0;   // transition width, BFS rings (> 0)
    double rim_gradient_ratio = 0.75;  // interior/rim target ratio (0..1]

    // [rim-lift] Rim-band apex height correction (2026-11-20 sliver fix).
    // When ON, interior apexes of rim-band sliver triangles (height <
    // rim_lift_min_ratio * min leg) are lifted tangentially away from the
    // unsplit rim chord to height ~ sqrt(3)/2 * min leg. Runs AFTER fairing.
    // Interior verts only; rim and topology bit-identical. Default OFF.
    bool rim_lift = false;
    double rim_lift_min_ratio = 0.3;   // trigger: h < ratio * min(|AM|,|BM|)
    double rim_lift_max_move  = 1.0;   // cap: |Delta| <= ratio * min leg

    // [collapse] Short-edge collapse inside density remesh (2026-11-20
    // density-floor fix). Runs after the split phase of each pass
    // (split → collapse → relax → flip). Collapses interior patch edges
    // shorter than collapse_factor × local target onto their midpoint;
    // rim verts/edges untouched (rim stays bit-identical). Default OFF.
    bool remesh_collapse = false;

    // [flip] Delaunay-style edge-flip quality pass for fill patches
    // (2026-11-18 rim-quality diagnosis, fix A+D). Runs at the end of each
    // density-remesh pass and once more after fairing; flips interior patch
    // edges only (rim edges/vertices untouched, so the rim stays
    // bit-identical). Default OFF = legacy behaviour untouched (golden
    // gates).
    bool remesh_flip = false;
};

struct FillResult {
    std::vector<Eigen::Vector3i> new_triangles;
    std::vector<Eigen::Vector3d> new_vertices;
};

} // namespace hole_filler_pro

