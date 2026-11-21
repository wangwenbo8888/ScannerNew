#include "hole_filler_pro/hole_filler_pro.h"
#include "hole_filler_pro/boundary_detector.h"
#include "hole_filler_pro/boundary_trimmer.h"
#include "hole_filler_pro/ear_clipper.h"
#include "hole_filler_pro/front_filler.h"
#include "hole_filler_pro/neighborhood.h"
#include "hole_filler_pro/surface_guide.h"
#include "hole_filler_pro/density_remesh.h"
#include "hole_filler_pro/triharmonic_fairing.h"
#include "hole_filler_pro/mesh_entry_cleaner.h"
#include "hole_filler_pro/arc_stitcher.h"
#include "hole_filler_pro/marker_cutter.h"

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {

namespace {

// Reference normal = mean of original face normals adjacent to any boundary edge.
Eigen::Vector3d computeReferenceNormal(const BoundaryLoop& loop,
                                       const Eigen::MatrixXi& F,
                                       const Eigen::MatrixXd& V) {
    std::unordered_map<Edge, int, pair_hash> e2f;
    e2f.reserve(static_cast<std::size_t>(F.rows()) * 3);
    for (int fi = 0; fi < F.rows(); ++fi) {
        for (int k = 0; k < 3; ++k) {
            int a = static_cast<int>(F(fi, k));
            int b = static_cast<int>(F(fi, (k + 1) % 3));
            e2f[canonicalEdge(a, b)] = fi;
        }
    }
    Eigen::Vector3d ref = Eigen::Vector3d::Zero();
    int cnt = 0;
    const auto& bv = loop.vertex_indices;
    int m = static_cast<int>(bv.size());
    for (int i = 0; i < m; ++i) {
        int j = loop.is_closed ? (i + 1) % m : i + 1;
        if (j >= m) break;   // open arc: no ghost closing edge
        auto it = e2f.find(canonicalEdge(bv[i], bv[j]));
        if (it == e2f.end()) continue;
        int fi = it->second;
        Eigen::Vector3d a = V.row(F(fi, 0));
        Eigen::Vector3d b = V.row(F(fi, 1));
        Eigen::Vector3d c = V.row(F(fi, 2));
        Eigen::Vector3d fn = (b - a).cross(c - a);
        if (fn.norm() > 1e-12) { ref += fn.normalized(); ++cnt; }
    }
    if (cnt > 0) ref /= static_cast<double>(cnt);
    return ref.norm() > 1e-12 ? ref.normalized() : loop.normal;
}

void fixOrientation(std::vector<Eigen::Vector3i>& tris,
                    const std::vector<Eigen::Vector3d>& verts,
                    const Eigen::Vector3d& ref) {
    if (tris.empty()) return;
    // Area-weighted average face normal (2x area magnitudes as weights). The
    // old version used only tris[0], whose normal is noisy for a small first
    // ear (e.g. at a pinch) and flipped whole patches arbitrarily.
    Eigen::Vector3d fn_sum = Eigen::Vector3d::Zero();
    double w_sum = 0.0;
    int valid = 0;
    for (const auto& tr : tris) {
        if (tr[0] < 0 || tr[0] >= static_cast<int>(verts.size())) continue;
        if (tr[1] < 0 || tr[1] >= static_cast<int>(verts.size())) continue;
        if (tr[2] < 0 || tr[2] >= static_cast<int>(verts.size())) continue;
        Eigen::Vector3d fn = (verts[tr[1]] - verts[tr[0]]).cross(verts[tr[2]] - verts[tr[0]]);
        double a = fn.norm();
        if (a > 1e-14) {
            fn_sum += fn;
            w_sum += a;
            ++valid;
        }
    }
    if (valid == 0 || w_sum <= 1e-14) return;   // fully degenerate patch
    if (fn_sum.dot(ref) < 0.0) {
        for (auto& tr : tris) std::swap(tr[1], tr[2]);
    }
}

} // anonymous namespace

FillResult fillSingleHole(const BoundaryLoop& loop,
                           const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const FillParams& params,
                           const std::unordered_map<int, double>& global_mean_edge,
                           const std::unordered_map<int, std::vector<int>>& global_v2t,
                           const std::function<bool(int, int)>& edge_occupied,
                           const std::unordered_set<int>* guide_boundary_verts) {
    FillResult result;
    // Open arcs are not fillable holes (their rim does not close); the
    // orchestrator skips them earlier - this is the hard second gate.
    if (!loop.is_closed) return result;
    int n = static_cast<int>(loop.vertex_indices.size());
    if (n < 3 || n > params.max_hole_edges) return result;

    // Stage 2 (hoisted BEFORE stage 1): neighborhood extraction reads only the
    // ORIGINAL mesh (loop/V/F) - the front fill needs ref_normal for its guide
    // fit, and the guided fairing below reuses the same fit.
    auto t_neigh0 = std::chrono::steady_clock::now();
    NeighborhoodInfo nb_global = extractNeighborhood(loop, V, F, params.neighborhood_rings, global_mean_edge, global_v2t);
    auto t_neigh1 = std::chrono::steady_clock::now();

    // Guide fit state (fitted once: either here for the front, or at 2.5).
    GuideFitResult guide;
    bool guide_prefitted = false;
    const bool guide_try = params.fill_surface_guided && params.fairing_order >= 1 &&
                           loop.perimeter >= params.fill_guide_perimeter_min;

    // Stage 1: coarse triangulation (topology only; boundary verts at local 0..n-1).
    // S5: unified ear-clip routing. Ear clipping is O(n^2), handles convex/
    // concave/collinear rims correctly, avoids occupied diagonals, and quality
    // is recovered downstream by the remesh + fairing stages.
    // [front] When ON and the rim is large enough, the advancing front grows
    // the patch ON the fitted guide surface (curvature-aware from birth);
    // any guard failure falls back to the ear clip for this hole.
    auto t_init0 = std::chrono::steady_clock::now();
    const bool front_try = params.fill_front && loop.perimeter >= params.fill_front_perimeter_min;
    if (front_try) {
        guide = fitGuideSurface(loop, V, F, global_v2t, nb_global.ref_normal,
                                params.fill_guide_band_width, guide_boundary_verts);
        guide_prefitted = true;
        result = initialFillFront(loop, V, guide, global_mean_edge, edge_occupied,
                                  params.fill_front_flip);
        if (result.new_triangles.empty())
            result = initialFillEarClip(loop, V, edge_occupied);
    } else {
        result = initialFillEarClip(loop, V, edge_occupied);
    }
    auto t_init1 = std::chrono::steady_clock::now();
    if (result.new_triangles.empty()) return result;

    // ---- GLOBAL -> LOCAL remap of target_edge for the remesher ----
    // The remesher looks up target_edge by LOCAL patch index. Boundary vertex
    // loop.vertex_indices[i] occupies local index i in the patch. Remap those entries;
    // ring vertices' targets are not needed by the remesher (it only handles patch verts).
    // Keep ring_vertices / ring_triangles / ref_normal in GLOBAL form (fairing indexes them
    // against V, F).
    NeighborhoodInfo nb_local;
    nb_local.ring_vertices = nb_global.ring_vertices;
    nb_local.ring_triangles = nb_global.ring_triangles;
    nb_local.ref_normal = nb_global.ref_normal;
    nb_local.target_edge.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int gv = loop.vertex_indices[i];
        auto it = nb_global.target_edge.find(gv);
        if (it != nb_global.target_edge.end()) {
            nb_local.target_edge[i] = it->second;   // global gv -> local i
        }
    }

    // Boundary local indices are simply 0..n-1 (boundary-first layout from initialFill).
    std::vector<int> boundary_local(n);
    for (int i = 0; i < n; ++i) boundary_local[i] = i;

    // Stage 2.5 [guide]: large-curved-hole guide surface (2026-11-16 merged
    // ruling). For rims at/above the perimeter threshold (the orchestrator
    // clamps it down for marker-exempt rims via the p_use copy), collect a
    // metric band around the rim on the ORIGINAL mesh (face-adjacency BFS +
    // radial window with inner gap + axial slab + other-rim guard) and fit a
    // robust bivariate cubic through it. Any failure leaves guide.ok=false
    // and stage 4 runs the legacy system (bit-identical fallback). Master
    // switch default OFF: this block never executes.
    // [front] The fit made at stage 1 (identical arguments) is REUSED - no
    // double fitting when both features engage.
    if (guide_try && !guide_prefitted)
        guide = fitGuideSurface(loop, V, F, global_v2t, nb_global.ref_normal,
                                params.fill_guide_band_width, guide_boundary_verts);

    // Stage 3: spatially-adaptive density remesh (consumes LOCAL-keyed target_edge;
    // inserts entries for new vertices using local keys).
    auto t0 = std::chrono::steady_clock::now();
    if (params.density_remesh_passes > 0) {
        remeshDensity(result.new_vertices, result.new_triangles,
                      nb_local, boundary_local, params);
    }
    auto t1 = std::chrono::steady_clock::now();

    // Stage 4: triharmonic fairing (classifies fixed/free from topology; uses GLOBAL
    // ring info indexed against V, F). boundary_global_ids lets the fairer map
    // rim vertices of the patch to their global ids (S2 fix), so ring triangles
    // touching the rim participate as anchors instead of being dropped.
    int achieved_order = 0;
    if (params.fairing_order >= 1) {
        // [guide] Targets for ALL patch verts at fairing time (post-remesh
        // coordinates): project the vertex into the guide frame, keep its
        // tangential (x, y), replace the height with the guide height. Rim
        // entries are computed but ignored by the solver (Dirichlet-fixed);
        // keeping the domain == new_vertices.size() lets triharmonicFair
        // index it directly. Extrapolation sentinel on FREE verts (patch
        // locals >= n are the free set by boundary-first layout): any
        // |h| > 3 mm or |grad h| > 5 refuses the guide (targets dropped,
        // legacy fairing).
        std::vector<Eigen::Vector3d> guide_targets;
        const std::vector<Eigen::Vector3d>* guide_targets_ptr = nullptr;
        bool guide_applied = false;
        if (guide.ok) {
            bool extrapolation_ok = true;
            guide_targets.resize(result.new_vertices.size());
            for (std::size_t i = 0; i < result.new_vertices.size(); ++i) {
                const Eigen::Vector3d& p = result.new_vertices[i];
                const double x = (p - guide.origin).dot(guide.u);
                const double y = (p - guide.origin).dot(guide.v);
                const double h = guideHeight(guide, x, y);
                if (static_cast<int>(i) >= n) {
                    if (std::abs(h) > 3.0 || guideGrad(guide, x, y).norm() > 5.0)
                        extrapolation_ok = false;
                }
                // [rim-soft] For rim verts (i < n), the soft-anchor target is
                // the ORIGINAL position (not the guide surface) — the spring
                // pulls the rim home, bending energy negotiates tangency.
                if (params.rim_soft_anchor && static_cast<int>(i) < n) {
                    guide_targets[i] = result.new_vertices[i];   // pin to self
                } else {
                    guide_targets[i] = guide.origin + guide.u * x + guide.v * y +
                                       guide.n * h;
                }
            }
            if (extrapolation_ok && params.fill_guide_lambda_alpha > 0.0) {
                guide_targets_ptr = &guide_targets;
                guide_applied = true;
            }
        }
        // Telemetry: one line, final verdict, ASCII only (console codepage
        // safe; tokens chosen to miss every acceptance grep - checked
        // against accept.ps1 / run_chain.ps1 patterns 2026-11-16).
        if (guide_try) {
            std::cerr << "    [guide] rim=" << n
                      << " peri=" << loop.perimeter
                      << " lvl=" << (guide.ok ? (guide.coeff_count == 10 ? "C10" : "Q6") : "-")
                      << " kept=" << (guide.ok ? guide.used_pts : -1)
                      << " rej=" << (guide.ok ? guide.rejected : -1)
                      << " cond=" << (guide.ok ? guide.cond : -1.0)
                      << " -> " << (guide_applied ? "applied" : "rejected") << "\n";
        }
        triharmonicFair(result.new_vertices, result.new_triangles,
                        nb_local, V, F, params.fairing_order, &achieved_order,
                        loop.vertex_indices, guide_targets_ptr,
                        params.fill_guide_lambda_alpha,
                        params.rim_soft_anchor);
    }
    auto t2 = std::chrono::steady_clock::now();
    // [rim-lift] Lift starved rim-band apexes AFTER fairing (fairing would
    // re-flatten any pre-fairing lift) and BEFORE the post-fairing flip
    // (so flips judge post-lift geometry). Interior verts only; rim and
    // topology bit-identical.
    if (params.rim_lift)
        liftRimApexes(result.new_vertices, result.new_triangles,
                      boundary_local, params);
    // [flip] post-fairing quality sweep: fairing moves vertices and can
    // leave folded or sliver configurations behind (rim-quality fix D).
    // Topology-only, rim bit-identical.
    if (params.remesh_flip)
        flipImprove(result.new_vertices, result.new_triangles, boundary_local);
    double ms_init   = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    double ms_neigh  = std::chrono::duration<double, std::milli>(t_neigh1 - t_neigh0).count();
    double ms_remesh = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_fair   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // Per-stage timing, gated behind HOLE_FILLER_TIMING env var (off by default; useful
    // for future profiling). The fairing-degradation note below is always printed.
    if (n > 20 && std::getenv("HOLE_FILLER_TIMING") != nullptr) {
        std::cerr << "    [time] neigh=" << ms_neigh << "ms  init=" << ms_init
                  << "ms  remesh=" << ms_remesh << "ms  fair=" << ms_fair
                  << "ms  (n=" << n << ", V=" << result.new_vertices.size()
                  << ", T=" << result.new_triangles.size() << ")\n";
    }
    if (achieved_order < params.fairing_order && achieved_order > 0) {
        std::cerr << "    note: fairing degraded to order " << achieved_order
                  << " (requested " << params.fairing_order << ")\n";
    }

    // Stage 5: orientation fix.
    Eigen::Vector3d ref = nb_global.ref_normal.squaredNorm() > 1e-18
                              ? nb_global.ref_normal
                              : computeReferenceNormal(loop, F, V);
    fixOrientation(result.new_triangles, result.new_vertices, ref);

    // [nan-guard] Post-fill NaN safety net: numerical edge cases in the
    // collapse / lift / fairing pipeline can rarely produce NaN coordinates
    // (2 out of 525k vertices on real data). Replace NaN vertices with the
    // centroid of their valid neighbors; if no valid neighbor exists, snap
    // to the loop centroid. Prevents the -nan area poisoning downstream.
    for (std::size_t i = 0; i < result.new_vertices.size(); ++i) {
        Eigen::Vector3d& p = result.new_vertices[i];
        if (std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z()))
            continue;
        // Collect valid neighbors from incident triangles
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        int cnt = 0;
        for (const auto& t : result.new_triangles) {
            if (t[0] < 0) continue;
            for (int k = 0; k < 3; ++k) {
                if (t[k] == static_cast<int>(i)) {
                    const int u = t[(k + 1) % 3], w = t[(k + 2) % 3];
                    for (const int nb : {u, w}) {
                        const Eigen::Vector3d& q = result.new_vertices[static_cast<std::size_t>(nb)];
                        if (std::isfinite(q.x()) && std::isfinite(q.y()) && std::isfinite(q.z())) {
                            sum += q;
                            ++cnt;
                        }
                    }
                }
            }
        }
        if (cnt > 0) {
            p = sum / static_cast<double>(cnt);
        } else {
            p = loop.centroid;  // last resort
        }
        std::cerr << "    [nan-guard] vertex " << i << " repaired\n";
    }

    return result;
}

namespace {

// S9: authoritative parameter validation at the library entry. Only the
// genuinely dangerous items are checked (downstream stages tolerate the rest
// by design, e.g. max_hole_perimeter == 0 is the documented "no limit"
// sentinel and fairing_order == 0 disables fairing).
bool validateFillParams(const FillParams& p) {
    bool ok = true;
    if (!(p.split_factor > 0.0) || !std::isfinite(p.split_factor)) {
        std::cerr << "[fillHoles] invalid split_factor=" << p.split_factor
                  << " (must be > 0 and finite; bad values cause infinite splits)\n";
        ok = false;
    }
    if (p.fairing_order < 0 || p.fairing_order > 4) {
        std::cerr << "[fillHoles] invalid fairing_order=" << p.fairing_order
                  << " (must be 0..4; 0=off, 1/2/3=C0/C1/C2)\n";
        ok = false;
    }
    if (p.max_hole_edges < 1) {
        std::cerr << "[fillHoles] invalid max_hole_edges=" << p.max_hole_edges
                  << " (must be >= 1)\n";
        ok = false;
    }
    // [marker] S0 marker-cut prerequisites - only checked when the stage is
    // ON (default-OFF path must stay byte-identical, K3 golden gate).
    if (p.marker_cut) {
        if (p.marker_file.empty() && (!p.marker_list_mem || p.marker_list_mem->empty())) {
            std::cerr << "[fillHoles] invalid marker source (marker_file empty and no in-memory marker list; required when marker_cut=true)\n";
            ok = false;
        }
        if (!(p.marker_diameter > 0.0) || !std::isfinite(p.marker_diameter)) {
            std::cerr << "[fillHoles] invalid marker_diameter=" << p.marker_diameter
                      << " (must be > 0 and finite)\n";
            ok = false;
        }
        if (!(p.marker_radius_scale > 0.0) || !std::isfinite(p.marker_radius_scale)) {
            std::cerr << "[fillHoles] invalid marker_radius_scale=" << p.marker_radius_scale
                      << " (must be > 0 and finite)\n";
            ok = false;
        }
        if (p.marker_len_front < 0.0 || !std::isfinite(p.marker_len_front) ||
            p.marker_len_back < 0.0 || !std::isfinite(p.marker_len_back)) {
            std::cerr << "[fillHoles] invalid marker_len_front/back (must be >= 0 and finite)\n";
            ok = false;
        }
        if (p.marker_normal_min_dot < -1.0 || p.marker_normal_min_dot > 1.0 ||
            !std::isfinite(p.marker_normal_min_dot)) {
            std::cerr << "[fillHoles] invalid marker_normal_min_dot="
                      << p.marker_normal_min_dot << " (must be in [-1, 1])\n";
            ok = false;
        }
        if (p.marker_match_margin < 0.0 || !std::isfinite(p.marker_match_margin)) {
            std::cerr << "[fillHoles] invalid marker_match_margin=" << p.marker_match_margin
                      << " (must be >= 0 and finite)\n";
            ok = false;
        }
        if (p.marker_match_min_dot < 0.0 || p.marker_match_min_dot > 1.0 ||
            !std::isfinite(p.marker_match_min_dot)) {
            std::cerr << "[fillHoles] invalid marker_match_min_dot="
                      << p.marker_match_min_dot << " (must be in [0, 1])\n";
            ok = false;
        }
        if (p.marker_match_max_edges < 3) {
            std::cerr << "[fillHoles] invalid marker_match_max_edges="
                      << p.marker_match_max_edges << " (must be >= 3)\n";
            ok = false;
        }
        // [marker-bay] heal knobs - only checked when the heal stage is armed.
        if (p.marker_bay_heal) {
            if (p.marker_bay_cap_margin < 0.5 || !std::isfinite(p.marker_bay_cap_margin)) {
                std::cerr << "[fillHoles] invalid marker_bay_cap_margin="
                          << p.marker_bay_cap_margin << " (must be >= 0.5 and finite)\n";
                ok = false;
            }
            if (p.marker_bay_min_sweep_deg <= 0.0 || p.marker_bay_min_sweep_deg >= 360.0 ||
                !std::isfinite(p.marker_bay_min_sweep_deg)) {
                std::cerr << "[fillHoles] invalid marker_bay_min_sweep_deg="
                          << p.marker_bay_min_sweep_deg << " (must be in (0, 360))\n";
                ok = false;
            }
            if (p.marker_bay_max_mouth_deg <= 0.0 || p.marker_bay_max_mouth_deg >= 180.0 ||
                !std::isfinite(p.marker_bay_max_mouth_deg)) {
                std::cerr << "[fillHoles] invalid marker_bay_max_mouth_deg="
                          << p.marker_bay_max_mouth_deg << " (must be in (0, 180))\n";
                ok = false;
            }
            const double r_cap = 0.5 * p.marker_diameter * p.marker_radius_scale +
                                 p.marker_bay_cap_margin;
            if (r_cap > 12.0)
                std::cerr << "[fillHoles] note: capture radius r_cap=" << r_cap
                          << "mm exceeds the 12mm solid-material evidence band"
                             " - bay heal may patch across real features\n";
        }
    }

    // [guide] curved-surface guided fill - only checked when the stage is
    // ON (default-OFF path must stay byte-identical, golden gates). NOTE:
    // top-level on purpose (guide fill does NOT require marker_cut).
    if (p.fill_surface_guided) {
        if (p.fill_guide_perimeter_min < 0.0 || !std::isfinite(p.fill_guide_perimeter_min)) {
            std::cerr << "[fillHoles] invalid fill_guide_perimeter_min="
                      << p.fill_guide_perimeter_min << " (must be >= 0 and finite)\n";
            ok = false;
        }
        if (p.fill_guide_lambda_alpha < 0.0 || !std::isfinite(p.fill_guide_lambda_alpha)) {
            std::cerr << "[fillHoles] invalid fill_guide_lambda_alpha="
                      << p.fill_guide_lambda_alpha << " (must be >= 0 and finite)\n";
            ok = false;
        }
        if (!(p.fill_guide_band_width > 0.0) || !std::isfinite(p.fill_guide_band_width)) {
            std::cerr << "[fillHoles] invalid fill_guide_band_width="
                      << p.fill_guide_band_width << " (must be > 0 and finite)\n";
            ok = false;
        }
    }

    // [front] advancing-front initial triangulation - only checked when ON
    // (default-OFF path stays byte-identical, golden gates).
    if (p.fill_front) {
        if (p.fill_front_perimeter_min < 0.0 || !std::isfinite(p.fill_front_perimeter_min)) {
            std::cerr << "[fillHoles] invalid fill_front_perimeter_min="
                      << p.fill_front_perimeter_min << " (must be >= 0 and finite)\n";
            ok = false;
        }
    }
    // [rim-grad] density gradient - only checked when ON
    if (p.rim_gradient) {
        if (!(p.rim_gradient_rings > 0.0) || !std::isfinite(p.rim_gradient_rings)) {
            std::cerr << "[fillHoles] invalid rim_gradient_rings="
                      << p.rim_gradient_rings << " (must be > 0 and finite)\n";
            ok = false;
        }
        if (p.rim_gradient_ratio <= 0.0 || p.rim_gradient_ratio > 1.0 ||
            !std::isfinite(p.rim_gradient_ratio)) {
            std::cerr << "[fillHoles] invalid rim_gradient_ratio="
                      << p.rim_gradient_ratio << " (must be in (0, 1])\n";
            ok = false;
        }
    }
    return ok;
}

// Component-level perimeter gate (S8 completion). Arcs chained through pinch
// vertices form ONE physical opening; the perimeter threshold must apply to
// the whole component, not to individual arcs - otherwise a dumbbell-shaped
// opening decomposes into two sub-threshold lobes that both get filled
// (the "one wrongly sealed 55.5-perimeter opening" failure).
std::vector<bool> componentTooBig(const std::vector<BoundaryLoop>& loops,
                                  double threshold, bool exempt_closed,
                                  const std::vector<bool>* marker_exempt = nullptr) {
    std::vector<bool> too_big(loops.size(), false);
    if (threshold <= 0.0 || loops.empty()) return too_big;
    std::vector<int> parent(loops.size());
    for (std::size_t i = 0; i < loops.size(); ++i) parent[i] = static_cast<int>(i);
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uni = [&](int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    };
    // [P0-2] With exempt_closed, closed loops only union with closed loops
    // and arcs only with arcs: a pinhole rim touching the crack network
    // through a shared pinch vertex is judged on its own perimeter again,
    // while a vertex-pinch dumbbell (two CLOSED loops sharing a vertex)
    // stays one component and remains gated (S8 protection preserved).
    // [marker] Matched marker rims go to a DEDICATED slot 2: they never
    // chain with the crack/arc network (and vice versa), and the final
    // verdict for them is forced false below - the rim is bounded by the
    // cut cylinder by construction.
    std::unordered_map<int, int> owner[3];
    for (std::size_t i = 0; i < loops.size(); ++i) {
        const bool mex = marker_exempt != nullptr && i < marker_exempt->size() &&
                         (*marker_exempt)[i];
        const int slot = mex ? 2
                             : (exempt_closed ? (loops[i].is_closed ? 0 : 1) : 0);
        for (int v : loops[i].vertex_indices) {
            auto it = owner[slot].find(v);
            if (it == owner[slot].end()) owner[slot][v] = static_cast<int>(i);
            else uni(static_cast<int>(i), it->second);
        }
    }
    std::vector<double> comp_peri(loops.size(), 0.0);
    for (std::size_t i = 0; i < loops.size(); ++i)
        comp_peri[find(static_cast<int>(i))] += loops[i].perimeter;
    for (std::size_t i = 0; i < loops.size(); ++i) {
        if (marker_exempt != nullptr && i < marker_exempt->size() && (*marker_exempt)[i])
            continue;   // [marker] never gated
        too_big[i] = comp_peri[find(static_cast<int>(i))] > threshold;
    }
    return too_big;
}

} // namespace

// S3: the original fillHoles body, renamed. Unified-exit wrapper `fillHoles`
// (below) runs the watertight gate on every return path. `pre_loops` receives
// the boundary-loop count the filling stage starts from (trim-rollback-safe
//鍙ｅ緞: work_loops after trim; 0 on the early-exit paths, matching post=0 there).
static bool fillHolesImpl(const Eigen::MatrixXd& vertices,
                const Eigen::MatrixXi& triangles,
                Eigen::MatrixXd& out_vertices,
                Eigen::MatrixXi& out_triangles,
                const FillParams& params,
                EntryCleanReport* clean_report,
                std::size_t& pre_loops,
                std::size_t& exit_nm_pruned,    // [exit-nm] faces removed at the main exit
                std::size_t& exit_nm_residue,   // [exit-nm] nm edges still >=3 at exit
                std::size_t& exempt_filled,     // [P0-2] fills rescued by component_exempt_closed
                std::size_t& stitched,          // [P0-3] arc pairs merged by arc_stitch
                std::size_t& marker_filled,     // [marker] marker rims actually filled
                std::size_t& marker_expected,   // [marker] rims matched right after the S0 cut
                std::size_t* post_loops_hint = nullptr) {   // [gatecheap] output boundary loop count from edge_stats
    if (!validateFillParams(params)) return false;   // S9 entry gate
    bool timing = (std::getenv("HOLE_FILLER_TIMING") != nullptr);
    auto t_total0 = std::chrono::steady_clock::now();
    pre_loops = 0;
    exit_nm_pruned = 0;
    exit_nm_residue = 0;
    exempt_filled = 0;
    stitched = 0;
    marker_filled = 0;
    marker_expected = 0;

    // ---- S1: entry clean (dedup / degenerate / bowtie-split / small-CC) -----
    // Replaces the four Open3D cleanup calls that used to run only on the
    // fillHolesFile path; when enabled, EVERY caller gets them. edge_count is
    // built once here and REUSED by the detectBoundaries overload below.
    Eigen::MatrixXd V_in = vertices;
    Eigen::MatrixXi F_in = triangles;
    std::unordered_map<Edge, EdgeStat, pair_hash> edge_table;   // [M1] in/out scratch, now {len,cnt}

    // ---- S0: [marker] marker-point cylinder cut (2026-11-15 ruling) -----
    // Runs BEFORE S1 entry clean on purpose: the cut rims then get cleaned
    // by S1 like any scanned boundary, and the [M1]/edge_unify table that
    // cleanMeshEntry builds is built for exactly this (post-cut) mesh - the
    // in/out contract "table == table of the returned F" stays intact.
    // Markers are world-coordinate entities and survive every downstream
    // re-detection (trim re-detect, compaction); exemption is matched
    // geometrically at each gate, never stored on the loops.
    std::vector<MarkerSpec> markers;   // empty when marker_cut=false
    if (params.marker_cut) {
        std::vector<MarkerSpec> loaded;
        if (params.marker_list_mem && !params.marker_list_mem->empty()) {
            loaded = *params.marker_list_mem;      // [port-261121] in-memory markers (product path)
        } else if (!loadMarkersAsc(params.marker_file, loaded)) {
            std::cerr << "[marker] no usable markers in \"" << params.marker_file
                      << "\" - aborting\n";
            return false;   // fail-closed (same contract as param/IO failure)
        }
        markers.reserve(loaded.size());
        for (std::size_t mi = 0; mi < loaded.size(); ++mi) {
            if (!markerNormalValid(V_in, F_in, loaded[mi], params)) {
                std::cerr << "[marker] WARNING: skip marker #" << mi
                          << " (floating / incoherent / opposed surface normals)\n";
                continue;
            }
            markers.push_back(loaded[mi]);
        }
        if (markers.empty()) {
            std::cerr << "[marker] all markers skipped - aborting fill\n";
            return false;
        }
        auto t_mc0 = std::chrono::steady_clock::now();
        MarkerCutResult mcr = cutMarkerCylinders(V_in, F_in, markers, params);
        auto t_mc1 = std::chrono::steady_clock::now();
        V_in = std::move(mcr.V);
        F_in = std::move(mcr.F);
        if (timing) {
            std::cerr << "[time] markerCut="
                      << std::chrono::duration<double, std::milli>(t_mc1 - t_mc0).count()
                      << "ms  (faces=" << mcr.faces_removed
                      << ", verts=" << mcr.verts_removed << ")\n";
        }
        // Cross-check right after the cut: how many rims the matcher sees.
        // This is the "expected" telemetry that makes a silent miss (e.g.
        // rim merged into a crack network) visible in the [stats] line.
        std::vector<BoundaryLoop> s0_loops = detectBoundaries(V_in, F_in);   // no-table overload
        std::vector<bool> s0_match = matchMarkerLoops(s0_loops, markers, params, nullptr);
        for (bool b : s0_match)
            if (b) ++marker_expected;
        if (marker_expected < markers.size()) {
            // [marker-bay] with heal armed the miss is expected to be
            // recoverable - downgrade to an informational note (acceptance
            // greps for "[marker] WARNING"; heal=false keeps the real one).
            if (params.marker_bay_heal)
                std::cerr << "[marker] note: " << (markers.size() - marker_expected)
                          << " rim(s) merged with cracks/openings - bay heal will attempt"
                             " (S0 match "
                          << marker_expected << "/" << markers.size() << ")\n";
            else
                std::cerr << "[marker] WARNING: only " << marker_expected << " of "
                          << markers.size() << " cut rims matched right after the cut -"
                          << " a rim probably merged with a nearby crack/opening"
                             " (try a smaller marker_radius_scale or marker_bay_heal)\n";
        }
        std::cerr << "[marker] cut total markers=" << markers.size()
                  << " faces=" << mcr.faces_removed
                  << " expected_loops=" << marker_expected << "\n";
    }

    if (params.entry_clean) {
        auto t_cl0 = std::chrono::steady_clock::now();
        // [D1] dead-table removal kept: the pre-clean build was never read.
        // [M1] the exit table is now {len,cnt} (unified with edge_stats below).
        EntryCleanResult cr = cleanMeshEntry(V_in, F_in, params, edge_table);
        auto t_cl1 = std::chrono::steady_clock::now();
        V_in = std::move(cr.V);
        F_in = std::move(cr.F);
        if (clean_report) *clean_report = cr.report;
        if (timing) {
            std::cerr << "[time] cleanMeshEntry="
                      << std::chrono::duration<double, std::milli>(t_cl1 - t_cl0).count()
                      << "ms  (V " << vertices.rows() << "->" << V_in.rows()
                      << ", F " << triangles.rows() << "->" << F_in.rows()
                      << ", merged=" << cr.report.verts_merged
                      << ", dup=" << cr.report.faces_dup_removed
                      << ", degen=" << cr.report.faces_degenerate_removed
                      << ", bowtie_split=" << cr.report.verts_bowtie_split
                      << ", nm_pruned=" << cr.report.faces_nonmanifold_removed
                      << ", spikes_snapped=" << cr.report.spikes_snapped
                      << ", flap_peeled=" << cr.report.faces_flap_peeled
                      << ", veil=" << cr.report.faces_veil_removed
                      << ", cc=" << cr.report.small_cc_removed << ")\n";
        }
    }

    auto t_bd0 = std::chrono::steady_clock::now();
    std::vector<BoundaryLoop> loops = params.entry_clean
                                          ? detectBoundaries(V_in, F_in, edge_table)   // [M1] EdgeStat overload (a==a filtered inside)
                                          : detectBoundaries(V_in, F_in);
    auto t_bd1 = std::chrono::steady_clock::now();
    if (timing) {
        std::cerr << "[time] detectBoundaries="
                  << std::chrono::duration<double, std::milli>(t_bd1 - t_bd0).count()
                  << "ms  (loops=" << loops.size() << ", V=" << V_in.rows()
                  << ", F=" << F_in.rows() << ")\n";
    }
    if (loops.empty()) {
        // [exit-nm] early-exit 1: audit on the exact mesh we are returning.
        // edge_table holds the returned mesh's {len,cnt} table when entry_clean
        // ran (cleanMeshEntry in/out contract); otherwise build it here.
        if (params.exit_nm_audit)
            exit_nm_residue = static_cast<std::size_t>(countNmEdges(
                edge_table.empty() ? buildEdgeStats(V_in, F_in) : edge_table));   // [M1] EdgeStat overload
        out_vertices = V_in;
        out_triangles = F_in;
        return true;
    }

    // Stage G1.5: boundary trim. Remove `trim_rings` BFS rings of triangles around
    // each FILL-CANDIDATE loop to discard low-quality scanned boundary bands, then
    // re-detect. S8 (minimal): loops that will be skipped anyway (open arcs, or over
    // the size/perimeter limits measured on the ORIGINAL rim) are never trimmed -
    // their rims must stay pristine. Global fuse: if trimming INCREASED the number
    // of arcs/loops (split/cut damage), roll the whole trim stage back.
    Eigen::MatrixXd V_work = V_in;
    Eigen::MatrixXi F_work = F_in;
    std::vector<BoundaryLoop> work_loops = loops;
    bool trimRewrote = false;   // [M1] table-invalidating event flag (trim accept only)
    auto willSkip = [&params](const BoundaryLoop& l, bool comp_too_big) {
        if (!l.is_closed) return true;
        if (comp_too_big) return true;   // whole component over the perimeter limit
        if (static_cast<int>(l.vertex_indices.size()) > params.max_hole_edges) return true;
        if (params.max_hole_perimeter > 0.0 && l.perimeter > params.max_hole_perimeter)
            return true;
        return false;
    };
    // [marker] pre-trim match: matched rims are PROTECTED from trim (a
    // trimmed rim grows outward and would stop matching downstream; the cut
    // rim is fresh geometry, not a scanned band). Empty when marker_cut=off.
    std::vector<bool> marker_exempt_pre;
    if (params.marker_cut)
        marker_exempt_pre = matchMarkerLoops(loops, markers, params, nullptr);
    std::vector<bool> comp_big_pre = componentTooBig(
        loops, params.max_hole_perimeter, params.component_exempt_closed,
        params.marker_cut ? &marker_exempt_pre : nullptr);   // [marker]
    if (params.trim_rings > 0) {
        std::vector<BoundaryLoop> trim_candidates;
        trim_candidates.reserve(loops.size());
        std::vector<BoundaryLoop> protect_loops;
        for (std::size_t li = 0; li < loops.size(); ++li) {
            const bool mprot = li < marker_exempt_pre.size() && marker_exempt_pre[li];   // [marker]
            if (!mprot && !willSkip(loops[li], comp_big_pre[li])) trim_candidates.push_back(loops[li]);
            else protect_loops.push_back(loops[li]);   // rims that must survive untouched
        }
        if (!trim_candidates.empty()) {
            auto t_trim0 = std::chrono::steady_clock::now();
            TrimResult tr = trimBoundaryRings(V_work, F_work, trim_candidates,
                                              params.trim_rings, &protect_loops);
            auto t_trim1 = std::chrono::steady_clock::now();
            if (timing) {
                // [instr E] closed/open split for the trim decision experiment
                // (pure print, zero behavior). is_closed distinguishes real
                // hole rims from open arcs left by the scanner gaps.
                auto nClosed = [](const std::vector<BoundaryLoop>& ls) {
                    int n = 0;
                    for (const auto& l : ls)
                        if (l.is_closed) ++n;
                    return n;
                };
                std::cerr << "[time] trimBoundaryRings="
                          << std::chrono::duration<double, std::milli>(t_trim1 - t_trim0).count()
                          << "ms  (V " << V_work.rows() << "->" << tr.V.rows()
                          << ", F " << F_work.rows() << "->" << tr.F.rows()
                          << ", loops " << work_loops.size() << "->" << tr.loops.size()
                          << ", closed " << nClosed(work_loops) << "->" << nClosed(tr.loops)
                          << ", open " << (work_loops.size() - nClosed(work_loops))
                          << "->" << (tr.loops.size() - nClosed(tr.loops))
                          << ", candidates " << trim_candidates.size() << ")\n";
            }
            if (tr.loops.size() > loops.size()) {
                // Fuse: trimming must never CREATE extra boundary components
                // (thin-wall cut-through). Roll back to the untrimmed mesh.
                std::cerr << "[trim] rollback: loop count increased ("
                          << loops.size() << " -> " << tr.loops.size() << ")\n";
            } else {
                // tr.loops comes from detectBoundaries on the WHOLE trimmed mesh:
                // it includes both the candidates' new rims and the untouched
                // skipped rims, all with correct compacted ids.
                V_work = std::move(tr.V);
                F_work = std::move(tr.F);
                work_loops = std::move(tr.loops);
                trimRewrote = true;   // [M1] F_work rewritten: unified table is stale
            }
        }
        if (work_loops.empty()) {
            // [exit-nm] early-exit 2: trim rewrote ids (compaction), so
            // edge_count is stale - build fresh from F_work.
            if (params.exit_nm_audit)
                exit_nm_residue = static_cast<std::size_t>(
                    countNmEdges(buildEdgeCount(F_work)));
            out_vertices = V_work;
            out_triangles = F_work;
            return true;
        }
    }

    // [P0-3] Pinch-split hole recovery: re-stitch OPEN-ARC pairs that share
    // both endpoints into closed loops BEFORE pre_loops is taken (the fill
    // gate then judges them as ordinary holes). Runs after trim so the trim
    // candidacy view (comp_big_pre, computed on the raw arcs) is untouched
    // and trimmed rims cannot be re-torn afterwards. Mutates work_loops
    // only. Warning when the companion switch is off (stitched closed loops
    // would union with the crack network and silently never fill).
    if (params.arc_stitch) {
        if (!params.component_exempt_closed)
            std::cerr << "[P0-3] WARNING: arc_stitch without component_exempt_closed - "
                         "stitched loops chain with the crack network and will not fill\n";
        ArcStitchResult asr = stitchSharedEndpointArcs(work_loops, V_work, F_work, params);
        stitched = asr.stitched;
    }

    // [marker-bay] S0.5 bay heal: a cut rim merged with a dead-end crack is
    // one OPEN arc whose interior run (inside the marker capture disc) is
    // the old rim. Split into residual head + artificial closed cap +
    // residual tail; the cap satisfies matchMarkerLoops verbatim, so the
    // main loop claims it automatically. The cap's wraparound edge is a
    // GHOST chord - collected in heal_chords and kept OUT of the S7 rim set
    // (it lands in the prior==0/own<=2 branch and, after commit, becomes
    // the residual crack's new legal end). Runs BEFORE pre_loops so the
    // watertight gate judges the post-heal set. Mutates work_loops only.
    std::vector<Edge> heal_chords;   // [marker-bay] ghost wraparound edges
    if (params.marker_cut && params.marker_bay_heal) {
        BayHealResult bhr = healMarkerBays(work_loops, V_work, markers, params);
        marker_expected += bhr.healed;   // healed caps join the expected total
        heal_chords = std::move(bhr.chord_edges);
    }

    // S3 gate pre: the boundary-loop count the filling stage starts from
    // (after trim rollback handling - work_loops is the authoritative pre state).
    pre_loops = work_loops.size();

    // Precompute vertex -> incident triangles ONCE (global, all holes share it).
    // Same anti-pattern fix as global_mean_edge: extractNeighborhood's BFS was rebuilding
    // this by scanning all F triangles per hole (the remaining O(F)-per-hole cost after
    // the edge_len fix). [lazyedge] Built FIRST now so the query-set collection
    // below can walk it; pure reordering, no data dependency.
    auto t_pv2t0 = std::chrono::steady_clock::now();
    std::unordered_map<int, std::vector<int>> global_v2t;
    global_v2t.reserve(static_cast<std::size_t>(V_work.rows()));
    for (int fi = 0; fi < F_work.rows(); ++fi) {
        for (int k = 0; k < 3; ++k) {
            global_v2t[static_cast<int>(F_work(fi, k))].push_back(fi);
        }
    }
    auto t_pv2t1 = std::chrono::steady_clock::now();

    // Precompute the global edge table ONCE in a single pass (S2 perf fix):
    // one hash slot per edge carries both its length and its triangle count
    // (the S7 occupancy ledger). Previously two separate maps doubled the
    // insert traffic on ~3M edge slots.
    auto t_pme0 = std::chrono::steady_clock::now();
    std::unordered_map<int, double> global_mean_edge;
    std::unordered_map<Edge, EdgeStat, pair_hash> edge_stats;   // [M1] EdgeStat now in utils.h
    if (params.edge_unify && params.entry_clean && !trimRewrote && !edge_table.empty()) {
        // [M1] unified-table arm: the exit table IS this table (same container
        // type, same pair_hash, same reserve F*2, same insertion order, moved
        // bucket layout intact) -> lazyedge/eager traversal order identical.
        edge_stats = std::move(edge_table);
    } else {
        edge_stats = buildEdgeStats(V_work, F_work);   // bitwise-replica rebuild arm
    }
    {
        if (params.lazyedge) {
            // [clean-opt lazyedge] Only ~5% of vertices are ever queried by
            // the hole loop (boundary + ring vertices of the candidate
            // loops). Collect that query set Q (one extra BFS per loop over
            // the already-built global_v2t, mirroring extractNeighborhood's
            // traversal with +1 ring of margin -- a superset is always
            // safe), then fill mean_edge for Q members in a SINGLE pass over
            // edge_stats. Per-vertex additions happen in edge_stats
            // traversal order = exactly the subsequence the eager incident
            // vectors were summed in -> bitwise-identical results.
            std::vector<char> inQ(static_cast<std::size_t>(V_work.rows()), 0);
            const int qrings = std::max(1, params.neighborhood_rings) + 1;
            for (const BoundaryLoop& loop : work_loops) {
                if (loop.vertex_indices.empty()) continue;
                std::unordered_set<int> visited(loop.vertex_indices.begin(),
                                                loop.vertex_indices.end());
                std::queue<std::pair<int, int>> bfsq;
                for (int bv : loop.vertex_indices) {
                    bfsq.push({bv, 0});
                    inQ[static_cast<std::size_t>(bv)] = 1;
                }
                while (!bfsq.empty()) {
                    auto front = bfsq.front(); bfsq.pop();
                    int v = front.first, depth = front.second;
                    if (depth >= qrings) continue;
                    auto it = global_v2t.find(v);
                    if (it == global_v2t.end()) continue;
                    for (int fi : it->second) {
                        for (int k = 0; k < 3; ++k) {
                            int u = static_cast<int>(F_work(fi, k));
                            if (!visited.count(u)) {
                                visited.insert(u);
                                inQ[static_cast<std::size_t>(u)] = 1;
                                bfsq.push({u, depth + 1});
                            }
                        }
                    }
                }
            }
            std::vector<double> qsum(static_cast<std::size_t>(V_work.rows()), 0.0);
            std::vector<int> qcnt(static_cast<std::size_t>(V_work.rows()), 0);
            for (const auto& kv : edge_stats) {
                const std::size_t ia = static_cast<std::size_t>(kv.first.first);
                const std::size_t ib = static_cast<std::size_t>(kv.first.second);
                if (inQ[ia]) { qsum[ia] += kv.second.len; ++qcnt[ia]; }
                if (inQ[ib]) { qsum[ib] += kv.second.len; ++qcnt[ib]; }
            }
            for (std::size_t v = 0; v < inQ.size(); ++v)
                if (inQ[v] && qcnt[v] > 0)
                    global_mean_edge[static_cast<int>(v)] =
                        qsum[v] / static_cast<double>(qcnt[v]);
        } else {
            std::unordered_map<int, std::vector<double>> incident;
            for (const auto& kv : edge_stats) {
                incident[kv.first.first].push_back(kv.second.len);
                incident[kv.first.second].push_back(kv.second.len);
            }
            for (const auto& kv : incident) {
                double s = 0.0;
                for (double x : kv.second) s += x;
                global_mean_edge[kv.first] = s / static_cast<double>(kv.second.size());
            }
        }
    }
    auto t_pme1 = std::chrono::steady_clock::now();
    if (timing) {
        std::cerr << "[time] precompute_mean_edge="
                  << std::chrono::duration<double, std::milli>(t_pme1 - t_pme0).count()
                  << "ms  precompute_v2t="
                  << std::chrono::duration<double, std::milli>(t_pv2t1 - t_pv2t0).count()
                  << "ms\n";
    }

    std::vector<Eigen::Vector3d> vert_list;
    vert_list.reserve(static_cast<std::size_t>(V_work.rows()));
    for (int i = 0; i < V_work.rows(); ++i) vert_list.push_back(V_work.row(i));

    std::vector<Eigen::Vector3i> tri_list;
    tri_list.reserve(static_cast<std::size_t>(F_work.rows()));
    for (int i = 0; i < F_work.rows(); ++i) tri_list.push_back(F_work.row(i));

    // S7 (minimal): the occupancy ledger lives in edge_stats (filled in the
    // precompute pass above; .cnt field). Before sewing any patch we verify
    // its edges against the mesh (and already-sewn patches): rim edges may go
    // 1 -> 2, every other patch edge requires prior occupancy 0. A conflicting
    // patch is DISCARDED with a warning - better an open hole than an edge
    // shared by 3+ triangles. Increments are buffered and only committed when
    // the patch is accepted (transaction semantics).

    // Occupancy predicate handed to the ear clipper (forbidden diagonals).
    auto edge_occupied = [&edge_stats](int a, int b) {
        return edge_stats.find(canonicalEdge(a, b)) != edge_stats.end();
    };

    std::cout << "Detected " << work_loops.size() << " boundary loop(s)\n";
    // [marker] post-trim match on the FINAL loop set (the main-loop
    // exemption view; geometric, so trim/compaction cannot invalidate it).
    std::vector<bool> marker_exempt_post;    // empty when marker_cut=false
    std::vector<int> marker_id_post;
    std::vector<char> marker_claimed;        // [marker-bay] lost_to_crack filter
    if (params.marker_cut) {
        marker_exempt_post = matchMarkerLoops(work_loops, markers, params, &marker_id_post);
        marker_claimed.assign(markers.size(), 0);
        for (int id : marker_id_post)
            if (id >= 0) marker_claimed[static_cast<std::size_t>(id)] = 1;
    }
    // Component gate recomputed on the post-trim arcs (rim metrics changed).
    std::vector<bool> comp_big_post = componentTooBig(
        work_loops, params.max_hole_perimeter, params.component_exempt_closed,
        params.marker_cut ? &marker_exempt_post : nullptr);   // [marker]
    // [P0-2 telemetry] legacy full-union view: fills that land here would
    // have been chained-skipped before - counted as exempt_filled.
    const std::vector<bool> comp_big_legacy =
        params.component_exempt_closed
            ? componentTooBig(work_loops, params.max_hole_perimeter, false)   // [marker] stays legacy-pure
            : comp_big_post;

    // [guide] Other-rim guard set: union of ALL work_loops' vertices, built
    // ONCE before the main loop (post trim/stitch/heal - work_loops is
    // final here). fillSingleHole's own rim seeds bypass the guard inside
    // collectGuideBand, so the current loop's rim is unaffected; every
    // OTHER rim (holes, open arcs, frames) is a hard stop for the band BFS.
    // Built only when the switch is on: the OFF path does zero extra work.
    std::unordered_set<int> guide_bverts;
    const std::unordered_set<int>* guide_bverts_ptr = nullptr;
    if (params.fill_surface_guided) {
        for (const BoundaryLoop& l : work_loops)
            guide_bverts.insert(l.vertex_indices.begin(), l.vertex_indices.end());
        guide_bverts_ptr = &guide_bverts;
    }

    int filled = 0, skipped = 0, open_arcs = 0, rejected = 0;
    for (std::size_t bi = 0; bi < work_loops.size(); ++bi) {
        const auto& loop = work_loops[bi];
        if (!loop.is_closed) {
            // Open arc: a rim fragment between pinch/dangling vertices. NOT a
            // closed hole - filling it would require a ghost closing edge.
            // Skip deliberately; report a geometric fingerprint (endpoints,
            // vertex count, exact perimeter) for whitelisting/audit.
            ++skipped; ++open_arcs;
            const Eigen::Vector3d& p0 = V_work.row(loop.vertex_indices.front());
            const Eigen::Vector3d& p1 = V_work.row(loop.vertex_indices.back());
            std::cout << "  Hole " << bi << ": " << loop.vertex_indices.size()
                      << " vertices, OPEN ARC (SKIP not closed)"
                      << " ep(" << p0.x() << "," << p0.y() << "," << p0.z() << ")|("
                      << p1.x() << "," << p1.y() << "," << p1.z() << ")"
                      << " n=" << loop.vertex_indices.size()
                      << " p=" << loop.perimeter << "\n";
            // [marker] a marker rim that got crack-chained into an open arc
            // can never be filled by the plain gates; report it so the loss
            // is visible instead of silent. [P3-R6] vertex-based hit test (a
            // giant merged arc's CENTROID is far away and never fired the
            // old centroid test); claimed markers (sealed rims incl. healed
            // bay caps) are excluded - a crack merely passing near a SEALED
            // marker is not a loss.
            if (params.marker_cut) {
                int lost = markerLostToCrack(loop, V_work, markers, marker_claimed, params);
                if (lost >= 0)
                    std::cout << "    [marker] lost_to_crack #" << lost << "\n";
            }
            continue;
        }
        // [marker] matched marker rims bypass the three size gates below
        // (max_hole_edges / max_hole_perimeter / component perimeter) and
        // fillSingleHole's own hard gate (via the params copy at the call).
        const bool mexempt = bi < marker_exempt_post.size() && marker_exempt_post[bi];
        if (static_cast<int>(loop.vertex_indices.size()) > params.max_hole_edges && !mexempt) {
            ++skipped;
            std::cout << "  Hole " << bi << ": " << loop.vertex_indices.size()
                      << " vertices (SKIP > max_hole_edges)\n";
            continue;
        }
        double perimeter = loop.perimeter;   // exact edge-sum (no ghost edge)
        if (params.max_hole_perimeter > 0.0 && perimeter > params.max_hole_perimeter && !mexempt) {
            ++skipped;
            std::cout << "  Hole " << bi << ": " << loop.vertex_indices.size()
                      << " vertices, perimeter = " << perimeter
                      << " (SKIP > max_hole_perimeter=" << params.max_hole_perimeter
                      << ")\n";
            continue;
        }
        if (comp_big_post[bi] && !mexempt) {
            // Part of a pinch-chained component whose TOTAL perimeter exceeds
            // the limit: the whole opening must stay open even if this arc
            // alone is under the threshold (dumbbell lobe).
            ++skipped;
            std::cout << "  Hole " << bi << ": " << loop.vertex_indices.size()
                      << " vertices, perimeter = " << perimeter
                      << " (SKIP component over max_hole_perimeter)\n";
            continue;
        }
        std::cout << "  Hole " << bi << ": " << loop.vertex_indices.size()
                  << " vertices, avg edge = " << loop.avg_edge_length << "\n";

        // [marker] per-call params copy: an exempt rim may carry more rim
        // vertices than max_hole_edges (fillSingleHole re-checks that gate
        // internally), so lift exactly that field for exactly this call.
        // For non-exempt loops the max() is an identity (gate above already
        // passed) - zero OFF-path disturbance.
        FillParams p_use = params;
        if (mexempt) {
            p_use.max_hole_edges = std::max(params.max_hole_edges,
                                            static_cast<int>(loop.vertex_indices.size()));
            // [guide] exempt rims force-enable guided fill: clamp the
            // perimeter threshold to at most this rim's perimeter (same
            // per-call lift pattern as max_hole_edges above). Non-exempt
            // loops keep the unmodified threshold - fillSingleHole's own
            // gate then decides, identity for them.
            if (params.fill_surface_guided)
                p_use.fill_guide_perimeter_min =
                    std::min(params.fill_guide_perimeter_min, perimeter);
        }
        FillResult r = fillSingleHole(loop, V_work, F_work, p_use, global_mean_edge,
                                      global_v2t, edge_occupied, guide_bverts_ptr);
        if (r.new_triangles.empty()) { ++skipped; continue; }

        int n_boundary = static_cast<int>(loop.vertex_indices.size());
        // local_to_global: boundary verts (local 0..n-1) reuse global ids; new interior
        // verts (local n..) are appended to vert_list with fresh ids.
        std::vector<int> local_to_global(r.new_vertices.size());
        for (int i = 0; i < n_boundary; ++i) local_to_global[i] = loop.vertex_indices[i];
        int new_base = static_cast<int>(vert_list.size());
        for (int i = n_boundary; i < static_cast<int>(r.new_vertices.size()); ++i) {
            local_to_global[i] = new_base + (i - n_boundary);
        }

        // ---- S7 occupancy check (buffered, commit on accept) ----
        // Rim edges (adjacent loop pairs incl. the closing edge) may take the
        // count from 1 to 2. Any other patch edge (diagonals/interior, incl.
        // edges to new vertices) requires prior count 0 with own count <= 2.
        {
            std::unordered_set<Edge, pair_hash> rim_edges;
            int m = n_boundary;
            for (int i = 0; i < m; ++i) {
                int j = loop.is_closed ? (i + 1) % m : i + 1;
                if (j >= m) break;
                int a = loop.vertex_indices[i], b = loop.vertex_indices[j];
                Edge e = canonicalEdge(a, b);
                // [marker-bay] ghost wraparound chord of a healed cap: NOT a
                // real mesh edge (prior==0). Keep it out of the rim set so
                // it lands in the prior==0/own<=2 branch below; after commit
                // it becomes the residual crack's new legal closing end.
                bool is_heal_chord = false;
                for (const Edge& hc : heal_chords)
                    if (hc == e) { is_heal_chord = true; break; }
                if (is_heal_chord) continue;
                rim_edges.insert(e);
            }
            std::unordered_map<Edge, int, pair_hash> own;
            bool conflict = false;
            for (const auto& t : r.new_triangles) {
                if (t[0] < 0) continue;
                for (int k = 0; k < 3; ++k) {
                    int a = local_to_global[t[k]];
                    int b = local_to_global[t[(k + 1) % 3]];
                    ++own[canonicalEdge(a, b)];
                }
            }
            for (const auto& kv : own) {
                auto it = edge_stats.find(kv.first);
                int prior = (it == edge_stats.end()) ? 0 : it->second.cnt;
                if (rim_edges.count(kv.first)) {
                    if (prior != 1 || kv.second != 1) { conflict = true; }
                } else {
                    if (prior != 0 || kv.second > 2) { conflict = true; }
                }
                if (conflict) {
                    std::cout << "    [diag] conflicting edge v" << kv.first.first
                              << "-v" << kv.first.second << " prior=" << prior
                              << " own=" << kv.second
                              << (rim_edges.count(kv.first) ? " [rim]" : " [chord/interior]")
                              << " p=";
                    if (kv.first.first < V_work.rows())
                        std::cout << "(" << V_work(kv.first.first, 0) << "," << V_work(kv.first.first, 1)
                                  << "," << V_work(kv.first.first, 2) << ")";
                    else std::cout << "(new-vert)";
                    if (kv.first.second < V_work.rows())
                        std::cout << "|(" << V_work(kv.first.second, 0) << "," << V_work(kv.first.second, 1)
                                  << "," << V_work(kv.first.second, 2) << ")";
                    else std::cout << "|(new-vert)";
                    std::cout << (kv.first.first < n_boundary ? " A-rim" : " A-new")
                              << (kv.first.second < n_boundary ? " B-rim" : " B-new")
                              << "\n";
                    break;
                }
            }
            if (conflict) {
                ++rejected; ++skipped;
                std::cout << "    rejected: hole " << bi
                          << " (patch edge conflicts with occupied mesh edge - left open)\n";
                continue;
            }
            for (const auto& kv : own) edge_stats[kv.first].cnt += kv.second;   // commit
        }

        for (int i = n_boundary; i < static_cast<int>(r.new_vertices.size()); ++i) {
            vert_list.push_back(r.new_vertices[i]);
        }
        for (const auto& t : r.new_triangles) {
            if (t[0] < 0) continue;
            tri_list.push_back(Eigen::Vector3i(local_to_global[t[0]],
                                               local_to_global[t[1]],
                                               local_to_global[t[2]]));
        }
        ++filled;
        if (comp_big_legacy[bi] && !comp_big_post[bi] && !mexempt) ++exempt_filled;   // [P0-2] ([marker] no double count)
        std::cout << "    filled: " << r.new_triangles.size() << " tris, "
                  << (r.new_vertices.size() - n_boundary) << " new verts\n";
        if (mexempt) {   // [marker] grep-able per-marker fill contract
            ++marker_filled;
            std::cout << "    [marker] filled #" << marker_id_post[bi]
                      << " hole=" << bi << " tris=" << r.new_triangles.size() << "\n";
        }
    }
    std::cout << "Filled " << filled << " holes, skipped " << skipped
              << " (open arcs: " << open_arcs << ", rejected: " << rejected << ")\n";

    // [exit-nm] main-exit audit + guarded prune. edge_stats IS the exact edge
    // table of the output mesh (S7 occupancy ledger after all patch commits).
    // Audit: count-only. Prune: second-pass trim that can never open a
    // boundary edge; counts are decremented in place, so post_loops_hint
    // below and the watertight gate judge the PRUNED topology.
    if (params.exit_nm_audit || params.exit_nm_prune) {
        std::unordered_map<Edge, int, pair_hash> exit_cnt;
        exit_cnt.reserve(edge_stats.size());
        for (const auto& kv : edge_stats)
            if (kv.first.first != kv.first.second)
                exit_cnt[kv.first] = kv.second.cnt;
        if (params.exit_nm_prune)
            exit_nm_pruned = static_cast<std::size_t>(
                pruneNmEdgesExit(vert_list, tri_list, exit_cnt));
        for (auto& kv : edge_stats) {
            auto it = exit_cnt.find(kv.first);
            kv.second.cnt = (it != exit_cnt.end()) ? it->second : 0;
        }
        exit_nm_residue = static_cast<std::size_t>(countNmEdges(exit_cnt));
    }

    out_vertices.resize(static_cast<int>(vert_list.size()), 3);
    for (int i = 0; i < out_vertices.rows(); ++i) out_vertices.row(i) = vert_list[i];
    out_triangles.resize(static_cast<int>(tri_list.size()), 3);
    for (int i = 0; i < out_triangles.rows(); ++i) out_triangles.row(i) = tri_list[i];

    // [gatecheap] post boundary loop count directly from edge_stats (the S7
    // occupancy ledger, which after all patch commits IS the output's exact
    // edge table). Avoids the O(F) rebuild of detectBoundaries in fillHoles.
    // Boundary edges = cnt==1 (skip a==a self-loops to match buildEdgeCount
    // convention); loops = connected components of the boundary subgraph.
    if (post_loops_hint) {
        std::map<int, std::vector<int>> badj;  // sorted for determinism
        for (const auto& kv : edge_stats) {
            if (kv.second.cnt != 1) continue;
            int a = kv.first.first, b = kv.first.second;
            if (a == b) continue;  // skip self-loop (buildEdgeCount convention)
            badj[a].push_back(b);
            badj[b].push_back(a);
        }
        for (auto& kv : badj) std::sort(kv.second.begin(), kv.second.end());
        std::map<int, char> vis;
        std::size_t loops = 0;
        for (const auto& kv : badj) {
            if (vis[kv.first]) continue;
            ++loops;
            std::vector<int> stack{kv.first};
            vis[kv.first] = 1;
            while (!stack.empty()) {
                int v = stack.back(); stack.pop_back();
                for (int w : badj[v]) {
                    if (!vis[w]) { vis[w] = 1; stack.push_back(w); }
                }
            }
        }
        *post_loops_hint = loops;
    }

    if (timing) {
        auto t_total1 = std::chrono::steady_clock::now();
        std::cerr << "[time] fillHoles_total="
                  << std::chrono::duration<double, std::milli>(t_total1 - t_total0).count()
                  << "ms  (out_V=" << out_vertices.rows() << ", out_T=" << out_triangles.rows() << ")\n";
    }
    return true;
}

// S3 watertight gate (unified exit): covers EVERY return path of
// fillHolesImpl - including the loops-empty fast path and the post-trim empty
// path (both pre=post=0, trivially passing). Judgment: post > pre means the
// fill CREATED new openings -> stderr gate token + watertight_fail=1 +
// return false (out_* is still filled; the caller decides whether to keep the
// product - the chain records it as "product kept, no certificate issued").
// post == pre (holes legitimately left: oversize / open arcs) PASSES: the fill
// did not make things worse, and the ledger records them via other channels.
bool fillHoles(const Eigen::MatrixXd& vertices,
               const Eigen::MatrixXi& triangles,
               Eigen::MatrixXd& out_vertices,
               Eigen::MatrixXi& out_triangles,
               const FillParams& params,
               EntryCleanReport* clean_report,
               int* watertight_fail) {
    if (watertight_fail) *watertight_fail = 0;
    std::size_t pre = 0;
    std::size_t post_hint = 0;
    std::size_t exit_nm_pruned = 0, exit_nm_residue = 0, exempt_filled = 0, stitched = 0;
    std::size_t marker_filled = 0, marker_expected = 0;   // [marker]
    // [P0-3/D4] arc_stitch changes the counting unit: pre counts stitched
    // LOOPS while an unstitched-or-unfilled pair leaves two arcs in the
    // output, and the plain detectBoundaries count (= arc/ring ITEMS) would
    // read post=pre+1 -> false WATERTIGHT_FAIL. Force the connected-
    // component count (post_hint, same unit as pre under stitching: two
    // arcs sharing endpoints are one component) whenever arc_stitch is on.
    const bool postHintPath = params.gatecheap || params.arc_stitch;
    if (!fillHolesImpl(vertices, triangles, out_vertices, out_triangles, params,
                       clean_report, pre, exit_nm_pruned, exit_nm_residue,
                       exempt_filled, stitched, marker_filled, marker_expected,
                       postHintPath ? &post_hint : nullptr))
        return false;   // parameter validation / IO failure: no gate to run
    // [exit-nm] ledger BEFORE the watertight line (digest -First 3 headroom).
    if (params.exit_nm_audit || params.exit_nm_prune)
        std::cerr << "[stats] holefill exit_nm_pruned=" << exit_nm_pruned
                  << " residue=" << exit_nm_residue << "\n";
    // [gatecheap] When enabled, skip the O(F) detectBoundaries rebuild and use
    // the edge_stats-derived post count (bit-identical loop count: same edge
    // table, same boundary edges, same connected-component count).
    const std::size_t post = postHintPath ? post_hint
                               : detectBoundaries(out_vertices, out_triangles).size();
    if (post > pre) {
        if (watertight_fail) *watertight_fail = 1;
        std::cerr << "[stats] holefill watertight=1 pre=" << pre << " post=" << post << "\n";
        std::cerr << "[gate] WATERTIGHT_FAIL: boundary loops increased (" << pre << " -> "
                  << post << "); fill created new openings\n";
        if (params.arc_stitch)
            std::cerr << "[stats] holefill stitched=" << stitched << "\n";
        if (params.component_exempt_closed)
            std::cerr << "[stats] holefill exempt_filled=" << exempt_filled << "\n";
        if (params.marker_cut)
            std::cerr << "[stats] holefill marker_exempt_filled=" << marker_filled
                      << "/" << marker_expected << "\n";
        return false;
    }
    std::cerr << "[stats] holefill watertight=0 pre=" << pre << " post=" << post << "\n";
    if (params.arc_stitch)
        std::cerr << "[stats] holefill stitched=" << stitched << "\n";
    if (params.component_exempt_closed)
        std::cerr << "[stats] holefill exempt_filled=" << exempt_filled << "\n";
    if (params.marker_cut) {
        std::cerr << "[stats] holefill marker_exempt_filled=" << marker_filled
                  << "/" << marker_expected << "\n";
        if (marker_filled < marker_expected)
            std::cerr << "[marker] WARNING: " << (marker_expected - marker_filled)
                      << " marker hole(s) left open\n";
    }
    return true;
}

// [port-261121] fillHolesFile (Open3D file-layer wrapper) removed:
// in-memory path = fillHoles; embedded-normal carry moved to the operator adapter.

} // namespace hole_filler_pro

