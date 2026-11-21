#include "hole_filler_pro/front_filler.h"
#include "hole_filler_pro/ear_clipper.h"
#include "hole_filler_pro/utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {
namespace {

// ---- Algorithm constants (design plan §3; deliberately NOT FillParams) ----
constexpr int    kRefinePasses     = 4;     // enough rounds to create interior
                                            // vertices for the guide lift
constexpr double kSplitFactor      = 2.0;   // MATCH downstream split_factor so
                                            // both stages work at one density
constexpr double kRelaxStep        = 0.5;   // full relaxation for angle quality
                                            // (density is controlled by the split
                                            // threshold, not by skipping relax)
constexpr double kMaxHeightMM      = 3.0;   // extrapolation sentinels (H2.5 parity)
constexpr double kMaxGradNorm      = 5.0;
constexpr long   kVertsCapMul      = 40;    // runaway guards
constexpr long   kVertsCapBase     = 4000;

// [front-flip] 2D Delaunay edge flips in the guide (u,v) frame. Ear clipping
// guarantees topology only; on nasty rims its long chords leave sliver strips
// that neither splitting nor relaxation can re-topologize. Flipping interior
// edges to the Delaunay criterion restores well-shaped 2D triangles BEFORE
// refinement; the guided lift preserves them. Rim boundary edges are excluded
// automatically (single incident triangle); new rim<->rim diagonals respect
// `edge_occupied` exactly like the ear clipper's diagonals (interior verts are
// new and can never be occupied). Bounded sweeps; each flip strictly improves
// the local Delaunay measure, so a handful of sweeps converge.
int delaunayFlipPass(std::vector<Eigen::Vector2d>& p2d,
                     std::vector<Eigen::Vector3i>& tris,
                     const std::vector<int>& rim_global,
                     const std::function<bool(int, int)>& edge_occupied) {
    if (p2d.empty() || tris.empty()) return 0;
    const int nrim = static_cast<int>(rim_global.size());
    auto orient = [](const Eigen::Vector2d& P, const Eigen::Vector2d& Q,
                     const Eigen::Vector2d& R) {
        return (Q.x() - P.x()) * (R.y() - P.y()) -
               (Q.y() - P.y()) * (R.x() - P.x());
    };
    int total = 0;
    for (int sweep = 0; sweep < 12; ++sweep) {
        std::unordered_map<Edge, std::vector<int>, pair_hash> e2t;
        e2t.reserve(tris.size() * 3);
        for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
            const auto& t = tris[static_cast<size_t>(ti)];
            if (t[0] < 0) continue;
            for (int k = 0; k < 3; ++k)
                e2t[canonicalEdge(t[k], t[(k + 1) % 3])].push_back(ti);
        }
        std::vector<char> dirty(tris.size(), 0);
        std::unordered_set<Edge, pair_hash> created;   // diagonals added this sweep
        int flips = 0;
        for (const auto& kv : e2t) {
            if (kv.second.size() != 2) continue;      // rim boundary / non-manifold
            const int t0i = kv.second[0], t1i = kv.second[1];
            if (dirty[static_cast<size_t>(t0i)] || dirty[static_cast<size_t>(t1i)]) continue;
            const Eigen::Vector3i t0 = tris[static_cast<size_t>(t0i)];
            const Eigen::Vector3i t1 = tris[static_cast<size_t>(t1i)];
            // ordered shared edge (a,b) as it appears in t0's winding; c, d = opposite
            int a = -1, b = -1, c = -1, d = -1;
            for (int k = 0; k < 3 && a < 0; ++k) {
                const int p = t0[k], q = t0[(k + 1) % 3];
                const bool pin = (t1[0] == p || t1[1] == p || t1[2] == p);
                const bool qin = (t1[0] == q || t1[1] == q || t1[2] == q);
                if (pin && qin) { a = p; b = q; c = t0[(k + 2) % 3]; }
            }
            if (a < 0) continue;
            for (int k = 0; k < 3; ++k)
                if (t1[k] != a && t1[k] != b && t1[k] != c) d = t1[k];
            if (d < 0) continue;                      // duplicated triangle
            const Eigen::Vector2d& A = p2d[static_cast<size_t>(a)];
            const Eigen::Vector2d& B = p2d[static_cast<size_t>(b)];
            const Eigen::Vector2d& C = p2d[static_cast<size_t>(c)];
            const Eigen::Vector2d& D = p2d[static_cast<size_t>(d)];
            const double oABC = orient(A, B, C);
            const double oABD = orient(A, B, D);
            if (oABC == 0.0 || oABD == 0.0) continue;
            if (oABC * oABD > 0.0) continue;          // same side: concave quad
            const double oCDA = orient(C, D, A);
            const double oCDB = orient(C, D, B);
            if (oCDA * oCDB >= 0.0) continue;         // new edge degenerate / invalid
            // Delaunay test (orientation-aware): D strictly inside circumcircle
            // of (A,B,C) <=> incircle * oABC > 0.
            const double adx = A.x() - D.x(), ady = A.y() - D.y();
            const double bdx = B.x() - D.x(), bdy = B.y() - D.y();
            const double cdx = C.x() - D.x(), cdy = C.y() - D.y();
            const double aa = adx * adx + ady * ady;
            const double bb = bdx * bdx + bdy * bdy;
            const double cc = cdx * cdx + cdy * cdy;
            const double inc = adx * (bdy * cc - cdy * bb) -
                               ady * (bdx * cc - cdx * bb) +
                               aa * (bdx * cdy - bdy * cdx);
            if (inc * oABC <= 0.0) continue;          // already Delaunay / cocircular
            // Duplicate-diagonal guard: ear clipping on degenerate rims can
            // produce geometrically OVERLAPPING (but topologically manifold)
            // triangles; flipping in such a region would create an edge that
            // already exists -> 4 triangles per edge -> S7 rejects the whole
            // patch. Skip if (c,d) exists now or was created earlier this
            // sweep.
            const Edge cd = canonicalEdge(c, d);
            if (e2t.count(cd) || created.count(cd)) continue;
            if (c < nrim && d < nrim && edge_occupied &&
                edge_occupied(rim_global[static_cast<size_t>(c)],
                             rim_global[static_cast<size_t>(d)]))
                continue;                            // would create an occupied diagonal
            // orientation-preserving replacement (quad loop a,d,b,c)
            tris[static_cast<size_t>(t0i)] = Eigen::Vector3i(a, d, c);
            tris[static_cast<size_t>(t1i)] = Eigen::Vector3i(d, b, c);
            dirty[static_cast<size_t>(t0i)] = 1;
            dirty[static_cast<size_t>(t1i)] = 1;
            created.insert(cd);
            ++flips;
        }
        total += flips;
        if (flips == 0) break;
    }
    return total;
}

    // [front-flip] Fit-failure fallback: a robust cubic cannot span giant
    // holes (extent ~136mm) within its RMS guard — previously such holes fell
    // back SILENTLY to ear clipping, whose unconverged sliver core survives
    // the entire downstream pipeline (79% thin triangles on the 50W case).
    // Build a planar guide frame (PCA of the rim, h == 0) instead: the front
    // filler still delivers Delaunay topology + converged density, and the
    // geometric curvature is delegated to the downstream fairing (rim soft
    // anchor + ring bending energy interpolate smoothly across the hole).
    GuideFitResult planarGuideFromRim(const BoundaryLoop& loop,
                                      const Eigen::MatrixXd& V) {
        GuideFitResult r;
        const int nr = static_cast<int>(loop.vertex_indices.size());
        if (nr < 3) return r;
        Eigen::Vector3d c(0.0, 0.0, 0.0);
        for (int gv : loop.vertex_indices) c += V.row(gv);
        c /= static_cast<double>(nr);
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int gv : loop.vertex_indices) {
            Eigen::Vector3d d = V.row(gv).transpose() - c;
            cov += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix3d E = es.eigenvectors();      // ascending eigenvalues
        Eigen::Vector3d u = E.col(2);
        Eigen::Vector3d nrm = E.col(0);
        Eigen::Vector3d v = nrm.cross(u).normalized();
        nrm = u.cross(v).normalized();
        r.origin = c;
        r.u = u; r.v = v; r.n = nrm;
        double radius = 0.0;
        for (int gv : loop.vertex_indices)
            radius = std::max(radius, (V.row(gv).transpose() - c).norm());
        r.scale = std::max(radius, 1e-6);
        r.coeff_count = 1;                          // h(u,v) == 0 everywhere
        for (int t = 0; t < 10; ++t) r.coeffs[t] = 0.0;
        r.ok = true;
        r.used_pts = nr;
        return r;
    }

} // anonymous namespace

// [front] Implementation note (2026-11-17): the design doc sketched a classic
// advancing-front grower; the first prototype tangled into self-intersecting
// fronts on simple rims (front-collision resolution is the hard 20% of that
// algorithm). Path switch per the plan's guardrail: keep the proven ear-clip
// for TOPOLOGY, refine it to the local target density with long-edge splits
// (same rule as the density remesher, applied here in the guide frame), and
// lift EVERY interior vertex exactly onto the fitted guide surface (the
// height-field projection is exact in one step because u,v are orthogonal to
// n). The patch is therefore curvature-aware from birth, converges by
// construction, and inherits the ear clipper's robustness on nasty rims.
FillResult initialFillFront(const BoundaryLoop& loop,
                            const Eigen::MatrixXd& V,
                            const GuideFitResult& guide,
                            const std::unordered_map<int, double>& global_mean_edge,
                            const std::function<bool(int, int)>& edge_occupied,
                            bool delaunay_flips) {
    FillResult result;
    GuideFitResult g = guide;
    if (!g.ok && delaunay_flips) {
        g = planarGuideFromRim(loop, V);
        if (g.ok)
            std::cerr << "[front] rim=" << loop.vertex_indices.size()
                      << " peri=" << loop.perimeter
                      << " -> planar guide (polynomial fit failed)\n";
    }
    if (!g.ok) return result;

    // Stage A: topology from the proven ear clipper (handles convex/concave/
    // collinear rims, occupied-diagonal avoidance, boundary-first layout).
    result = initialFillEarClip(loop, V, edge_occupied);
    if (result.new_triangles.empty()) return result;

    const int n = static_cast<int>(loop.vertex_indices.size());
    const double fallback_t =
        (loop.avg_edge_length > 1e-12) ? loop.avg_edge_length : 1.0;
    const long verts_cap = kVertsCapMul * n + kVertsCapBase;

    // Work set: 2D guide-frame coordinates + per-vertex target edge length.
    std::vector<Eigen::Vector2d> p2d(result.new_vertices.size());
    std::vector<double> targ(result.new_vertices.size());
    for (size_t i = 0; i < result.new_vertices.size(); ++i) {
        const Eigen::Vector3d& p = result.new_vertices[i];
        p2d[i] = Eigen::Vector2d((p - g.origin).dot(g.u),
                                 (p - g.origin).dot(g.v));
        targ[i] = fallback_t;
    }
    for (int i = 0; i < n; ++i) {                      // boundary targets
        auto it = global_mean_edge.find(loop.vertex_indices[static_cast<size_t>(i)]);
        if (it != global_mean_edge.end()) targ[static_cast<size_t>(i)] = it->second;
    }

    // [front-flip] re-topologize the ear-clip result to the Delaunay
    // criterion BEFORE refinement: chords between rim vertices leave sliver
    // strips that splitting/relaxation cannot fix.
    int flips_total = 0;
    if (delaunay_flips)
        flips_total += delaunayFlipPass(p2d, result.new_triangles,
                                        loop.vertex_indices, edge_occupied);

    int splits = 0;
    bool converged_early = false;
    for (int pass = 0; pass < kRefinePasses; ++pass) {
        // ---- split long interior edges (count >= 2 patch triangles) ----
        // [front-flip] With Delaunay flips ON, split RECURSIVELY until every
        // interior edge is below the threshold (bounded by sub levels and
        // verts_cap): a huge hole (perimeter ~300mm) needs ~9 halvings to
        // reach the ~1mm target, which 4 single-split passes can never
        // deliver — the unconverged core is exactly where the giant slivers
        // live. Legacy path (flips OFF) keeps exactly one split sweep per
        // pass (byte-identical).
        const int sub_levels = delaunay_flips ? 64 : 1;
        bool hit_cap = false;
        for (int sub = 0; sub < sub_levels; ++sub) {
        std::unordered_map<Edge, std::vector<int>, pair_hash> e2t;
        e2t.reserve(result.new_triangles.size() * 3);
        for (int ti = 0; ti < static_cast<int>(result.new_triangles.size()); ++ti) {
            const auto& t = result.new_triangles[static_cast<size_t>(ti)];
            if (t[0] < 0) continue;
            for (int k = 0; k < 3; ++k)
                e2t[canonicalEdge(t[k], t[(k + 1) % 3])].push_back(ti);
        }
        std::vector<Edge> to_split;
        for (const auto& kv : e2t) {
            if (kv.second.size() < 2) continue;        // patch-boundary edge
            const Edge& e = kv.first;
            const double L = (p2d[static_cast<size_t>(e.first)] -
                              p2d[static_cast<size_t>(e.second)]).norm();
            const double t =
                0.5 * (targ[static_cast<size_t>(e.first)] + targ[static_cast<size_t>(e.second)]);
            if (L > kSplitFactor * t) to_split.push_back(e);
        }
        if (to_split.empty()) {
            if (pass > 0 && sub == 0) converged_early = true;   // legacy convergence break
            break;
        }
        for (const Edge& e : to_split) {
            const int a = e.first, b = e.second;
            if (a >= static_cast<int>(p2d.size()) ||
                b >= static_cast<int>(p2d.size())) continue;
            if (static_cast<long>(p2d.size()) > verts_cap) { hit_cap = true; break; }
            const int mid = static_cast<int>(p2d.size());
            p2d.push_back(0.5 * (p2d[static_cast<size_t>(a)] + p2d[static_cast<size_t>(b)]));
            targ.push_back(0.5 * (targ[static_cast<size_t>(a)] + targ[static_cast<size_t>(b)]));
            result.new_vertices.emplace_back();        // position fixed by the lift below
            // Split every triangle CURRENTLY carrying edge (a,b) (live scan,
            // same pattern as the density remesher).
            for (int ti = 0; ti < static_cast<int>(result.new_triangles.size()); ++ti) {
                Eigen::Vector3i tri = result.new_triangles[static_cast<size_t>(ti)];
                if (tri[0] < 0) continue;
                const bool ha = (tri[0] == a || tri[1] == a || tri[2] == a);
                const bool hb = (tri[0] == b || tri[1] == b || tri[2] == b);
                if (!(ha && hb)) continue;
                int other = -1, pa = -1, pb = -1;
                for (int k = 0; k < 3; ++k) {
                    if (tri[k] == a) pa = k;
                    if (tri[k] == b) pb = k;
                    if (tri[k] != a && tri[k] != b) other = tri[k];
                }
                if (other < 0) continue;
                const bool ab_order = ((pa + 1) % 3 == pb);
                result.new_triangles[static_cast<size_t>(ti)] = {-1, -1, -1};
                if (ab_order) {
                    result.new_triangles.push_back({other, a, mid});
                    result.new_triangles.push_back({other, mid, b});
                } else {
                    result.new_triangles.push_back({other, b, mid});
                    result.new_triangles.push_back({other, mid, a});
                }
                ++splits;
            }
            result.new_triangles.erase(
                std::remove_if(result.new_triangles.begin(), result.new_triangles.end(),
                               [](const Eigen::Vector3i& t) { return t[0] < 0; }),
                result.new_triangles.end());
        }
        if (hit_cap) break;
        }
        if (converged_early) break;    // legacy: exit pass loop WITHOUT relax

        // ---- light 2D tangential relaxation (boundary fixed) ----
        std::vector<std::vector<int>> adj(p2d.size());
        for (const auto& t : result.new_triangles) {
            adj[static_cast<size_t>(t[0])].push_back(t[1]);
            adj[static_cast<size_t>(t[0])].push_back(t[2]);
            adj[static_cast<size_t>(t[1])].push_back(t[0]);
            adj[static_cast<size_t>(t[1])].push_back(t[2]);
            adj[static_cast<size_t>(t[2])].push_back(t[0]);
            adj[static_cast<size_t>(t[2])].push_back(t[1]);
        }
        std::vector<Eigen::Vector2d> upd = p2d;
        for (size_t v = static_cast<size_t>(n); v < p2d.size(); ++v) {
            if (adj[v].empty()) continue;
            Eigen::Vector2d c(0.0, 0.0);
            for (int u : adj[v]) c += p2d[static_cast<size_t>(u)];
            c /= static_cast<double>(adj[v].size());
            upd[v] = p2d[v] + kRelaxStep * (c - p2d[v]);
        }
        p2d = std::move(upd);

        // [front-flip] splits+relax break local Delaunay-ness; restore it so
        // the topology stays well-shaped for the next pass.
        if (delaunay_flips)
            flips_total += delaunayFlipPass(p2d, result.new_triangles,
                                            loop.vertex_indices, edge_occupied);
    }

    // [front-flip] Manifold audit (belt & braces): the ear clipper can emit
    // geometrically overlapping triangles on degenerate rims, and any local
    // operator on such a region risks an edge with >2 incident triangles,
    // which S7 would reject wholesale. If the refined patch is not edge-
    // manifold, fall back to the plain ear-clip result (safe legacy path).
    if (delaunay_flips) {
        std::unordered_map<Edge, int, pair_hash> cnt;
        cnt.reserve(result.new_triangles.size() * 3);
        for (const auto& t : result.new_triangles)
            for (int k = 0; k < 3; ++k)
                ++cnt[canonicalEdge(t[k], t[(k + 1) % 3])];
        for (const auto& kv : cnt)
            if (kv.second > 2) {
                std::cerr << "[front] rim=" << n << " peri=" << loop.perimeter
                          << " -> fallback (non-manifold edge x" << kv.second
                          << " after refine)\n";
                result = initialFillEarClip(loop, V, edge_occupied);
                return result;
            }
    }

    // ---- lift EVERY interior vertex exactly onto the guide surface ----
    for (size_t i = static_cast<size_t>(n); i < p2d.size(); ++i) {
        const double h = guideHeight(g, p2d[i].x(), p2d[i].y());
        const Eigen::Vector2d gr = guideGrad(g, p2d[i].x(), p2d[i].y());
        if (std::abs(h) > kMaxHeightMM || gr.norm() > kMaxGradNorm) {
            result.new_triangles.clear();
            result.new_vertices.clear();
            std::cerr << "[front] rim=" << n << " peri=" << loop.perimeter
                      << " -> fallback (extrapolation sentinel at interior vertex)\n";
            return result;
        }
        result.new_vertices[i] = g.origin + g.u * p2d[i].x() +
                                 g.v * p2d[i].y() + g.n * h;
    }

    std::cerr << "[front] rim=" << n << " peri=" << loop.perimeter
              << " -> applied (tris=" << result.new_triangles.size() << ", verts="
              << result.new_vertices.size() << ", splits=" << splits
              << ", flips=" << flips_total << ")\n";
    return result;
}

} // namespace hole_filler_pro
