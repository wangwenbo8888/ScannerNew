#include "hole_filler_pro/density_remesh.h"
#include "hole_filler_pro/utils.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hole_filler_pro {

namespace {

// Look up the local target edge length for a patch vertex (local index).
// Falls back to neighbor mean, then to patch-wide average. Memoized per-pass.
double lookupTarget(int local_idx,
                    const std::vector<Eigen::Vector3d>& verts,
                    const std::vector<Eigen::Vector3i>& tris,
                    const NeighborhoodInfo& nb,
                    double fallback,
                    std::unordered_map<int, double>& cache) {
    auto cit = cache.find(local_idx);
    if (cit != cache.end()) return cit->second;

    double result;
    auto it = nb.target_edge.find(local_idx);
    if (it != nb.target_edge.end()) {
        result = it->second;
    } else {
        // Interpolate from neighbors that DO have a target.
        std::unordered_set<int> neighbors;
        for (const auto& t : tris) {
            for (int k = 0; k < 3; ++k) {
                if (t[k] == local_idx) {
                    neighbors.insert(t[(k + 1) % 3]);
                    neighbors.insert(t[(k + 2) % 3]);
                }
            }
        }
        double s = 0.0;
        int c = 0;
        for (int nb_v : neighbors) {
            auto nit = nb.target_edge.find(nb_v);
            if (nit != nb.target_edge.end()) { s += nit->second; ++c; }
        }
        result = (c > 0) ? s / static_cast<double>(c) : fallback;
    }
    cache[local_idx] = result;
    return result;
}

} // anonymous namespace

// [rim-grad] Multi-source BFS over patch vertex adjacency from the rim.
// dist[v] = ring hops from the rim (rim = 0); INT_MAX = unreached.
namespace {
std::vector<int> rimRingDistances(const std::vector<Eigen::Vector3i>& tris,
                                  const std::unordered_set<int>& rim,
                                  std::size_t nverts) {
    std::vector<std::vector<int>> adj(nverts);
    for (const auto& t : tris) {
        adj[static_cast<size_t>(t[0])].push_back(t[1]);
        adj[static_cast<size_t>(t[0])].push_back(t[2]);
        adj[static_cast<size_t>(t[1])].push_back(t[0]);
        adj[static_cast<size_t>(t[1])].push_back(t[2]);
        adj[static_cast<size_t>(t[2])].push_back(t[0]);
        adj[static_cast<size_t>(t[2])].push_back(t[1]);
    }
    std::vector<int> dist(nverts, INT_MAX);
    std::queue<int> q;
    for (int b : rim) {
        if (b >= 0 && static_cast<std::size_t>(b) < nverts) {
            dist[static_cast<std::size_t>(b)] = 0;
            q.push(b);
        }
    }
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int u : adj[static_cast<std::size_t>(v)])
            if (dist[static_cast<std::size_t>(u)] == INT_MAX) {
                dist[static_cast<std::size_t>(u)] = dist[static_cast<std::size_t>(v)] + 1;
                q.push(u);
            }
    }
    return dist;
}
} // anonymous namespace

// [collapse] Short-edge collapse operator: merges interior patch edges
// shorter than collapse_factor × target onto their midpoint. Rim verts
// are NEVER collapsed. Link condition + fan-flip test guard manifoldness.
// Vertex compaction after the pass loop removes orphans.
namespace {

// Build vertex -> incident live triangle list (helper for collapse).
std::vector<std::vector<int>> buildV2T(const std::vector<Eigen::Vector3i>& tris,
                                       std::size_t nverts) {
    std::vector<std::vector<int>> v2t(nverts);
    for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        v2t[static_cast<std::size_t>(t[0])].push_back(ti);
        v2t[static_cast<std::size_t>(t[1])].push_back(ti);
        v2t[static_cast<std::size_t>(t[2])].push_back(ti);
    }
    return v2t;
}

// Attempt to collapse edge (a,b): move both to midpoint, remove the two
// triangles carrying the edge, rewrite b->a in the fan. Returns true on
// commit. All guards: rim, link condition, (c,d)-not-exist, fan-flip.
bool tryCollapse(int a, int b,
                std::vector<Eigen::Vector3d>& verts,
                std::vector<Eigen::Vector3i>& tris,
                const std::vector<std::vector<int>>& v2t,
                const std::unordered_set<int>& boundary_set,
                NeighborhoodInfo& nb) {
    // Rim guard (re-check at live execution — cheap)
    if (boundary_set.count(a) || boundary_set.count(b)) return false;

    // Live resolve: exactly 2 live triangles contain BOTH a and b
    std::vector<int> shared;
    for (int ti : v2t[static_cast<std::size_t>(a)]) {
        if (ti >= static_cast<int>(tris.size())) continue;
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        bool ha = (t[0] == a || t[1] == a || t[2] == a);
        bool hb = (t[0] == b || t[1] == b || t[2] == b);
        if (ha && hb) shared.push_back(ti);
    }
    if (shared.size() != 2) return false;

    // Resolve the two opposite vertices c, d (winding-aware)
    int c = -1, d = -1;
    for (int ti : shared) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        for (int k = 0; k < 3; ++k) {
            if (t[k] != a && t[k] != b) {
                if (c < 0) c = t[k]; else if (d < 0) d = t[k];
            }
        }
    }
    if (c < 0 || d < 0 || c == d) return false;

    // Link condition: N(a) ∩ N(b) == {c, d}
    // (1-ring neighbors from CURRENT live triangles)
    std::unordered_set<int> Na, Nb;
    for (int ti : v2t[static_cast<std::size_t>(a)]) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        for (int k = 0; k < 3; ++k) if (t[k] != a) Na.insert(t[k]);
    }
    for (int ti : v2t[static_cast<std::size_t>(b)]) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        for (int k = 0; k < 3; ++k) if (t[k] != b) Nb.insert(t[k]);
    }
    std::unordered_set<int> link;
    for (int v : Na) if (Nb.count(v)) link.insert(v);
    if (link.size() != 2 || !link.count(c) || !link.count(d)) return false;

    // Merged edge (c,d) must not already exist (tetrahedron guard)
    for (int ti : v2t[static_cast<std::size_t>(c)]) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        bool hc = (t[0] == c || t[1] == c || t[2] == c);
        bool hd = (t[0] == d || t[1] == d || t[2] == d);
        if (hc && hd) return false;
    }

    // Geometric validity: tentative midpoint move, whole merged fan must
    // stay sane (positive area, no normal flip, min-angle >= 5 deg).
    const Eigen::Vector3d mid = 0.5 * (verts[static_cast<std::size_t>(a)] +
                                       verts[static_cast<std::size_t>(b)]);
    const Eigen::Vector3d saveA = verts[static_cast<std::size_t>(a)];
    verts[static_cast<std::size_t>(a)] = mid;
    bool ok = true;
    for (int ti : v2t[static_cast<std::size_t>(a)]) {
        if (ti >= static_cast<int>(tris.size())) continue;
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        // Replace b with a in the triangle copy for checking
        Eigen::Vector3i tc = t;
        for (int k = 0; k < 3; ++k) if (tc[k] == b) tc[k] = a;
        // Skip the two dying triangles
        if (ti == shared[0] || ti == shared[1]) continue;
        // Check: positive area
        Eigen::Vector3d fn = (verts[static_cast<std::size_t>(tc[1])] -
                              verts[static_cast<std::size_t>(tc[0])])
                             .cross(verts[static_cast<std::size_t>(tc[2])] -
                                    verts[static_cast<std::size_t>(tc[0])]);
        if (fn.norm() < 1e-14) { ok = false; break; }
        // Check: normal doesn't flip (compare old vs new)
        Eigen::Vector3d fo = (verts[static_cast<std::size_t>(t[1])] -
                              saveA)
                             .cross(verts[static_cast<std::size_t>(t[2])] -
                                    verts[static_cast<std::size_t>(t[0])]);
        if (fo.norm() > 1e-14 && fn.normalized().dot(fo.normalized()) <= 0.0) {
            ok = false; break;
        }
    }
    // Also check b's fan (excluding the two dying triangles)
    if (ok) {
        for (int ti : v2t[static_cast<std::size_t>(b)]) {
            if (ti >= static_cast<int>(tris.size())) continue;
            const auto& t = tris[static_cast<std::size_t>(ti)];
            if (t[0] < 0) continue;
            if (ti == shared[0] || ti == shared[1]) continue;
            // Rewrite b->a for check
            Eigen::Vector3i tc = t;
            for (int k = 0; k < 3; ++k) if (tc[k] == b) tc[k] = a;
            Eigen::Vector3d fn = (verts[static_cast<std::size_t>(tc[1])] -
                                  verts[static_cast<std::size_t>(tc[0])])
                                 .cross(verts[static_cast<std::size_t>(tc[2])] -
                                        verts[static_cast<std::size_t>(tc[0])]);
            if (fn.norm() < 1e-14) { ok = false; break; }
            Eigen::Vector3d fo = (verts[static_cast<std::size_t>(t[1])] -
                                  verts[static_cast<std::size_t>(t[0])])
                                 .cross(verts[static_cast<std::size_t>(t[2])] -
                                        verts[static_cast<std::size_t>(t[0])]);
            if (fo.norm() > 1e-14 && fn.normalized().dot(fo.normalized()) <= 0.0) {
                ok = false; break;
            }
        }
    }
    if (!ok) { verts[static_cast<std::size_t>(a)] = saveA; return false; }

    // COMMIT: b merges into a (a already at midpoint)
    for (int ti : shared) tris[static_cast<std::size_t>(ti)] = {-1, -1, -1};
    for (int ti : v2t[static_cast<std::size_t>(b)]) {
        if (ti >= static_cast<int>(tris.size())) continue;
        auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        if (ti == shared[0] || ti == shared[1]) continue;
        for (int k = 0; k < 3; ++k) if (t[k] == b) t[k] = a;
    }
    nb.target_edge.erase(b);
    return true;
}

} // anonymous namespace

void remeshDensity(std::vector<Eigen::Vector3d>& new_vertices,
                   std::vector<Eigen::Vector3i>& new_triangles,
                   NeighborhoodInfo& nb,
                   const std::vector<int>& boundary_local_indices,
                   const FillParams& params) {
    if (new_triangles.empty() || new_vertices.empty()) return;

    // Patch-wide fallback target = mean of all known targets.
    double fallback = 0.0;
    if (!nb.target_edge.empty()) {
        for (const auto& kv : nb.target_edge) fallback += kv.second;
        fallback /= static_cast<double>(nb.target_edge.size());
    }

    std::unordered_set<int> boundary_set(boundary_local_indices.begin(),
                                         boundary_local_indices.end());

    for (int pass = 0; pass < params.density_remesh_passes; ++pass) {
        // Per-pass cache of resolved vertex targets (reset each pass; new vertices
        // appear mid-pass and their targets are seeded into both nb and this cache).
        std::unordered_map<int, double> target_cache;

        // [rim-grad] Seed the cache with the gradient field BEFORE any
        // lookupTarget call. lookupTarget is cache-first, so every later
        // call (split test, midpoint inheritance, weighted relaxation)
        // returns the field value unchanged. Rim verts (d==0) are skipped:
        // they fall through to their own nb.target_edge entry.
        if (params.rim_gradient && !boundary_local_indices.empty()) {
            const double W = std::max(params.rim_gradient_rings, 1e-6);
            double t_rim_mean = 0.0; int rc = 0;
            for (int b : boundary_local_indices) {
                auto it = nb.target_edge.find(b);
                if (it != nb.target_edge.end()) { t_rim_mean += it->second; ++rc; }
            }
            if (rc > 0) {
                t_rim_mean /= static_cast<double>(rc);
                const double t_in = params.rim_gradient_ratio * t_rim_mean;
                const auto dist = rimRingDistances(new_triangles, boundary_set,
                                                   new_vertices.size());
                for (int v = 0; v < static_cast<int>(new_vertices.size()); ++v) {
                    if (dist[static_cast<std::size_t>(v)] == 0) continue;     // rim: own target
                    if (dist[static_cast<std::size_t>(v)] == INT_MAX) {
                        target_cache[v] = t_in; continue;
                    }
                    const double t = std::min(std::max(
                        (W - dist[static_cast<std::size_t>(v)]) / W, 0.0), 1.0);
                    const double w = t * t * (3.0 - 2.0 * t);      // smoothstep
                    target_cache[v] = t_in + (t_rim_mean - t_in) * w;
                }
            }
        }

        // Build edge -> triangle list.
        std::unordered_map<Edge, std::vector<int>, pair_hash> e2t;
        e2t.reserve(new_triangles.size() * 3);
        for (int ti = 0; ti < static_cast<int>(new_triangles.size()); ++ti) {
            const auto& t = new_triangles[ti];
            if (t[0] < 0) continue;
            for (int k = 0; k < 3; ++k) {
                e2t[canonicalEdge(t[k], t[(k + 1) % 3])].push_back(ti);
            }
        }

        bool changed = false;

        // ---- SPLIT long edges ----
        // Collect edges to split first (don't mutate while iterating the map).
        // ONLY interior patch edges (shared by >= 2 patch triangles) may be split:
        // both adjacent triangles are patch triangles and both get updated below, so
        // the split stays manifold. Patch-boundary edges (count < 2) connect to the
        // read-only original mesh on the other side. Splitting such an edge inserts a
        // midpoint that the ORIGINAL mesh triangles do not reference, so the
        // midpoint-to-endpoint sub-edges would be shared by only 1 triangle -> open
        // edges -> non-watertight fill. They must therefore NOT be split here.
        std::vector<Edge> to_split;
        for (const auto& kv : e2t) {
            if (kv.second.size() < 2) continue;   // only interior patch edges
            const Edge& e = kv.first;
            double L = (new_vertices[e.first] - new_vertices[e.second]).norm();
            double target = 0.5 * (lookupTarget(e.first, new_vertices, new_triangles, nb, fallback, target_cache) +
                                   lookupTarget(e.second, new_vertices, new_triangles, nb, fallback, target_cache));
            if (L > params.split_factor * target) {
                to_split.push_back(e);
            }
        }

        for (const Edge& e : to_split) {
            int a = e.first, b = e.second;
            if (a >= static_cast<int>(new_vertices.size()) ||
                b >= static_cast<int>(new_vertices.size())) continue;

            Eigen::Vector3d mid = 0.5 * (new_vertices[a] + new_vertices[b]);
            int mid_idx = static_cast<int>(new_vertices.size());
            new_vertices.push_back(mid);

            // New vertex inherits the mean target of its endpoints.
            double ta = lookupTarget(a, new_vertices, new_triangles, nb, fallback, target_cache);
            double tb = lookupTarget(b, new_vertices, new_triangles, nb, fallback, target_cache);
            double mid_target = 0.5 * (ta + tb);
            nb.target_edge[mid_idx] = mid_target;
            target_cache[mid_idx] = mid_target;  // seed cache so later same-pass lookups hit

            // Midpoints are always interior (free) vertices: only interior edges
            // (count >= 2) are ever split, so the midpoint never lies on the patch
            // boundary. Do NOT add it to boundary_set.

            // Split every triangle CURRENTLY sharing edge (a,b).
            // Live-scan (not the stale e2t) so triangles created by an earlier split
            // in this same pass that still carry edge (a,b) are also split; otherwise
            // long edges get stranded as dangling count-1 edges and never subdivide.
            for (int ti = 0; ti < static_cast<int>(new_triangles.size()); ++ti) {
                Eigen::Vector3i tri = new_triangles[ti];  // copy: push_back below may realloc
                if (tri[0] < 0) continue;
                bool has_a = (tri[0] == a || tri[1] == a || tri[2] == a);
                bool has_b = (tri[0] == b || tri[1] == b || tri[2] == b);
                if (!(has_a && has_b)) continue;

                int other = -1, pos_a = -1, pos_b = -1;
                for (int k = 0; k < 3; ++k) {
                    if (tri[k] == a) pos_a = k;
                    if (tri[k] == b) pos_b = k;
                    if (tri[k] != a && tri[k] != b) other = tri[k];
                }
                if (other < 0) continue;

                bool ab_order = ((pos_a + 1) % 3 == pos_b);
                new_triangles[ti] = {-1, -1, -1};  // mark dead before push_back (no aliasing)
                if (ab_order) {
                    // original winding ... a -> b ... ; insert mid between them
                    new_triangles.push_back({other, a, mid_idx});
                    new_triangles.push_back({other, mid_idx, b});
                } else {
                    new_triangles.push_back({other, b, mid_idx});
                    new_triangles.push_back({other, mid_idx, a});
                }
                changed = true;
            }
        }

        // Purge invalidated triangles.
        new_triangles.erase(
            std::remove_if(new_triangles.begin(), new_triangles.end(),
                           [](const Eigen::Vector3i& t) { return t[0] < 0; }),
            new_triangles.end());

        // ---- [collapse] SHORT-EDGE COLLAPSE (density floor) ----
        if (params.remesh_collapse && params.collapse_factor > 0.0) {
            e2t.clear();
            for (int ti = 0; ti < static_cast<int>(new_triangles.size()); ++ti) {
                const auto& t = new_triangles[static_cast<std::size_t>(ti)];
                if (t[0] < 0) continue;
                for (int k = 0; k < 3; ++k)
                    e2t[canonicalEdge(t[k], t[(k + 1) % 3])].push_back(ti);
            }
            auto v2t = buildV2T(new_triangles, new_vertices.size());
            std::vector<Edge> cands;
            for (const auto& kv : e2t) {
                if (kv.second.size() != 2) continue;
                const Edge& e = kv.first;
                if (boundary_set.count(e.first) || boundary_set.count(e.second)) continue;
                const double L = (new_vertices[static_cast<std::size_t>(e.first)] -
                                  new_vertices[static_cast<std::size_t>(e.second)]).norm();
                const double tt = 0.5 * (lookupTarget(e.first, new_vertices, new_triangles,
                                                      nb, fallback, target_cache) +
                                         lookupTarget(e.second, new_vertices, new_triangles,
                                                      nb, fallback, target_cache));
                if (L < params.collapse_factor * tt) cands.push_back(e);
            }
            std::sort(cands.begin(), cands.end());
            bool collapsed = false;
            for (const Edge& e : cands) {
                if (tryCollapse(e.first, e.second, new_vertices, new_triangles,
                                v2t, boundary_set, nb)) {
                    changed = true;
                    collapsed = true;
                }
            }
            if (collapsed) {
                new_triangles.erase(
                    std::remove_if(new_triangles.begin(), new_triangles.end(),
                                   [](const Eigen::Vector3i& t) { return t[0] < 0; }),
                    new_triangles.end());
            }
        }

        // ---- TANGENTIAL RELAXATION (boundary fixed) ----
        // Build adjacency AND vertex->triangles in one pass (avoids the O(V*T)
        // full-triangle scan for the local normal).
        std::vector<std::unordered_set<int>> adj(new_vertices.size());
        std::vector<std::vector<int>> v2t(new_vertices.size());
        for (int ti = 0; ti < static_cast<int>(new_triangles.size()); ++ti) {
            const auto& t = new_triangles[ti];
            adj[t[0]].insert(t[1]); adj[t[0]].insert(t[2]);
            adj[t[1]].insert(t[0]); adj[t[1]].insert(t[2]);
            adj[t[2]].insert(t[0]); adj[t[2]].insert(t[1]);
            v2t[t[0]].push_back(ti);
            v2t[t[1]].push_back(ti);
            v2t[t[2]].push_back(ti);
        }
        std::vector<Eigen::Vector3d> updated = new_vertices;
        for (int v = 0; v < static_cast<int>(new_vertices.size()); ++v) {
            if (boundary_set.count(v)) continue;       // boundary stays fixed
            if (adj[v].empty()) continue;
            Eigen::Vector3d c(0, 0, 0);
            if (params.rim_gradient) {
                // [rim-grad] Density-weighted centroid: w = 1/target settles
                // vertices at distances proportional to local targets,
                // consistent with the split criterion. Same cache the split
                // pass used => same field, structurally no divergence.
                double wsum = 0.0;
                for (int u : adj[v]) {
                    const double tu = std::max(
                        lookupTarget(u, new_vertices, new_triangles,
                                     nb, fallback, target_cache), 1e-6);
                    const double w = 1.0 / tu;
                    c += new_vertices[u] * w;
                    wsum += w;
                }
                c /= wsum;
            } else {
                for (int u : adj[v]) c += new_vertices[u];
                c /= static_cast<double>(adj[v].size());
            }
            // Local normal at v = sum of incident face normals (iterate only incident tris).
            Eigen::Vector3d nrm(0, 0, 0);
            for (int ti : v2t[v]) {
                const auto& t = new_triangles[ti];
                Eigen::Vector3d fn = (new_vertices[t[1]] - new_vertices[t[0]])
                                     .cross(new_vertices[t[2]] - new_vertices[t[0]]);
                nrm += fn;
            }
            if (nrm.norm() > 1e-12) {
                nrm.normalize();
                Eigen::Vector3d d = c - new_vertices[v];
                d -= nrm * d.dot(nrm);   // project onto tangent plane
                updated[v] = new_vertices[v] + 0.5 * d;
            }
        }
        new_vertices = updated;

        // [flip] quality sweep after each pass (rim untouched).
        if (params.remesh_flip)
            flipImprove(new_vertices, new_triangles, boundary_local_indices);

        if (!changed) break;   // converged
    }

    // [collapse] Post-loop vertex compaction: remove orphaned vertices
    // (collapsed away, no longer referenced by any triangle). Rim indices
    // 0..n-1 are preserved by the boundary-first layout (rim verts are
    // always referenced by rim triangles whose rim-rim edges are never
    // collapsible). Re-key nb.target_edge to compacted indices.
    if (params.remesh_collapse) {
        std::vector<int> old2new(new_vertices.size(), -1);
        std::vector<bool> used(new_vertices.size(), false);
        for (const auto& t : new_triangles) {
            if (t[0] < 0) continue;
            used[static_cast<std::size_t>(t[0])] = true;
            used[static_cast<std::size_t>(t[1])] = true;
            used[static_cast<std::size_t>(t[2])] = true;
        }
        int k = 0;
        for (std::size_t i = 0; i < new_vertices.size(); ++i)
            if (used[i]) old2new[i] = k++;
        // Remap triangle indices
        for (auto& t : new_triangles) {
            if (t[0] < 0) continue;
            t[0] = old2new[static_cast<std::size_t>(t[0])];
            t[1] = old2new[static_cast<std::size_t>(t[1])];
            t[2] = old2new[static_cast<std::size_t>(t[2])];
        }
        // Compact vertex array
        std::vector<Eigen::Vector3d> compacted;
        compacted.reserve(static_cast<std::size_t>(k));
        for (std::size_t i = 0; i < new_vertices.size(); ++i)
            if (used[i]) compacted.push_back(new_vertices[i]);
        new_vertices = std::move(compacted);
        // Re-key nb.target_edge
        NeighborhoodInfo tmp;
        for (const auto& kv : nb.target_edge) {
            const int oldIdx = kv.first;
            if (oldIdx >= 0 && static_cast<std::size_t>(oldIdx) < old2new.size() &&
                old2new[static_cast<std::size_t>(oldIdx)] >= 0)
                tmp.target_edge[old2new[static_cast<std::size_t>(oldIdx)]] = kv.second;
        }
        nb.target_edge = std::move(tmp.target_edge);
    }
}

void flipImprove(std::vector<Eigen::Vector3d>& verts,
                 std::vector<Eigen::Vector3i>& tris,
                 const std::vector<int>& boundary_local_indices) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kFoldDeg = 45.0;     // above this a fold is worth fixing
    constexpr double kMinKeepDeg = 5.0;   // never mint a triangle below this
    constexpr int kMaxSweeps = 5;

    if (verts.empty() || tris.empty()) return;
    std::unordered_set<int> bnd(boundary_local_indices.begin(), boundary_local_indices.end());

    auto minAng = [&](int a, int b, int c) {
        const Eigen::Vector3d& A = verts[static_cast<size_t>(a)];
        const Eigen::Vector3d& B = verts[static_cast<size_t>(b)];
        const Eigen::Vector3d& C = verts[static_cast<size_t>(c)];
        double m = 180.0;
        const Eigen::Vector3d* p[3] = {&A, &B, &C};
        for (int k = 0; k < 3; ++k) {
            const Eigen::Vector3d u = *p[(k + 1) % 3] - *p[k];
            const Eigen::Vector3d w = *p[(k + 2) % 3] - *p[k];
            const double cs = u.normalized().dot(w.normalized());
            m = std::min(m, std::acos(std::max(-1.0, std::min(1.0, cs))) * 180.0 / kPi);
        }
        return m;
    };
    auto faceNormal = [&](int a, int b, int c) {
        const Eigen::Vector3d& A = verts[static_cast<size_t>(a)];
        const Eigen::Vector3d& B = verts[static_cast<size_t>(b)];
        const Eigen::Vector3d& C = verts[static_cast<size_t>(c)];
        Eigen::Vector3d n = (B - A).cross(C - A);
        const double nn = n.norm();
        if (nn > 1e-18) { n /= nn; return n; }
        return Eigen::Vector3d(Eigen::Vector3d::UnitZ());
    };
    auto dihedralDeg = [&](int a, int b, int c, int d) {
        // angle between face (a,b,c) and face (b,a,d)
        return std::acos(std::max(-1.0, std::min(1.0,
                   faceNormal(a, b, c).dot(faceNormal(b, a, d))))) * 180.0 / kPi;
    };

    // edge key -> the (up to 2) triangles carrying it
    std::map<std::pair<int, int>, std::vector<int>> e2t;
    auto key = [](int x, int y) {
        return std::make_pair(std::min(x, y), std::max(x, y));
    };
    auto rebuild = [&]() {
        e2t.clear();
        for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
            const auto& t = tris[static_cast<size_t>(ti)];
            if (t[0] < 0) continue;
            for (int k = 0; k < 3; ++k) e2t[key(t[k], t[(k + 1) % 3])].push_back(ti);
        }
    };
    rebuild();

    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        // Deterministic visit order: snapshot of interior edges, sorted.
        std::vector<std::pair<int, int>> cands;
        for (const auto& kv : e2t)
            if (kv.second.size() == 2) cands.push_back(kv.first);
        std::sort(cands.begin(), cands.end());
        bool flipped = false;

        for (const auto& ek : cands) {
            auto it = e2t.find(ek);
            if (it == e2t.end() || it->second.size() != 2) continue;  // stale
            const int t1 = it->second[0], t2 = it->second[1];
            const int a = ek.first, b = ek.second;

            // Resolve the winding-aware opposites: in t1 find the ordered
            // pair (a->b) with opposite c; in t2 the pair (b->a) with
            // opposite d. If the windings do not line up, skip.
            int c = -1, d = -1;
            for (const int ti : {t1, t2}) {
                const auto& t = tris[static_cast<size_t>(ti)];
                for (int k = 0; k < 3; ++k) {
                    const int u = t[k], w = t[(k + 1) % 3], o = t[(k + 2) % 3];
                    if (u == a && w == b && c < 0) { c = o; break; }
                    if (u == b && w == a && d < 0) { d = o; break; }
                }
            }
            if (c < 0 || d < 0 || c == d) continue;

            // Rim-chord guard: refuse flips that would mint a new
            // boundary-to-boundary edge the occupancy ledger never vetted.
            if (bnd.count(c) && bnd.count(d)) continue;
            // The replacement edge must not already exist.
            if (e2t.find(key(c, d)) != e2t.end()) continue;

            // Metrics: fold first, angle second.
            const double dihOld = dihedralDeg(a, b, c, d);
            const double dihNew = dihedralDeg(c, d, b, a);
            const double angOld = std::min(minAng(a, b, c), minAng(b, a, d));
            const double angNew = std::min(minAng(a, d, c), minAng(c, d, b));
            const bool foldGain = dihOld > kFoldDeg && dihNew < dihOld - 1e-6;
            const bool angleGain = angNew > angOld + 1e-9;
            if (!foldGain && !angleGain) continue;
            if (angNew < kMinKeepDeg) continue;

            // Strict convexity of the quad a-d-b-c in the plane of face 1.
            const Eigen::Vector3d n = faceNormal(a, b, c);
            Eigen::Vector3d u = verts[static_cast<size_t>(b)] - verts[static_cast<size_t>(a)];
            u -= n * u.dot(n);
            const double un = u.norm();
            if (un < 1e-12) continue;
            u /= un;
            const Eigen::Vector3d v = n.cross(u);
            auto proj = [&](int x) {
                const Eigen::Vector3d r = verts[static_cast<size_t>(x)] - verts[static_cast<size_t>(a)];
                return Eigen::Vector2d(r.dot(u), r.dot(v));
            };
            const int quad[4] = {a, d, b, c};
            Eigen::Vector2d p2[4];
            for (int k = 0; k < 4; ++k) p2[k] = proj(quad[k]);
            bool convex = true;
            int sign = 0;
            for (int k = 0; k < 4; ++k) {
                const Eigen::Vector2d e1 = p2[(k + 1) % 4] - p2[k];
                const Eigen::Vector2d e2 = p2[(k + 2) % 4] - p2[(k + 1) % 4];
                const double cr = e1.x() * e2.y() - e1.y() * e2.x();
                if (std::abs(cr) < 1e-14) { convex = false; break; }
                const int s = (cr > 0) ? 1 : -1;
                if (sign == 0) sign = s;
                else if (s != sign) { convex = false; break; }
            }
            if (!convex) continue;

            // Commit: replace (a,b,c)+(b,a,d) with (a,d,c)+(c,d,b) in place.
            tris[static_cast<size_t>(t1)] = Eigen::Vector3i(a, d, c);
            tris[static_cast<size_t>(t2)] = Eigen::Vector3i(c, d, b);
            // Incremental edge-table fixups for the four surviving sides.
            auto swapRef = [&](const std::pair<int, int>& k, int from, int to) {
                auto e = e2t.find(k);
                if (e == e2t.end()) return;
                for (auto& x : e->second)
                    if (x == from) { x = to; break; }
            };
            e2t.erase(key(a, b));
            swapRef(key(b, c), t1, t2);
            swapRef(key(a, d), t2, t1);
            e2t[key(c, d)] = {t1, t2};
            flipped = true;
        }
        if (!flipped) break;
    }
}

// [rim-lift] Implementation: detect starved apexes over rim chords and
// lift them tangentially. See density_remesh.h for the contract.
void liftRimApexes(std::vector<Eigen::Vector3d>& verts,
                   std::vector<Eigen::Vector3i>& tris,
                   const std::vector<int>& boundary_local_indices,
                   const FillParams& params) {
    if (!params.rim_lift || tris.empty() || verts.empty()) return;
    std::unordered_set<int> bnd(boundary_local_indices.begin(),
                                boundary_local_indices.end());

    // Build vertex -> incident triangles (skip dead marks)
    std::vector<std::vector<int>> v2t(verts.size());
    for (int ti = 0; ti < static_cast<int>(tris.size()); ++ti) {
        const auto& t = tris[static_cast<std::size_t>(ti)];
        if (t[0] < 0) continue;
        v2t[static_cast<std::size_t>(t[0])].push_back(ti);
        v2t[static_cast<std::size_t>(t[1])].push_back(ti);
        v2t[static_cast<std::size_t>(t[2])].push_back(ti);
    }

    // Vertex normals (area-weighted star sum — robust even when the
    // sliver's own face normal is near-zero)
    std::vector<Eigen::Vector3d> vnrm(verts.size(), Eigen::Vector3d::Zero());
    for (const auto& t : tris) {
        if (t[0] < 0) continue;
        Eigen::Vector3d fn = (verts[static_cast<std::size_t>(t[1])] -
                              verts[static_cast<std::size_t>(t[0])])
                             .cross(verts[static_cast<std::size_t>(t[2])] -
                                    verts[static_cast<std::size_t>(t[0])]);
        for (int k = 0; k < 3; ++k)
            vnrm[static_cast<std::size_t>(t[k])] += fn;
    }

    // Collect lift proposals: for each rim-band triangle (exactly 2 rim
    // verts A,B + 1 interior apex M), if M's height over chord AB is
    // below ratio × min leg, propose a tangential lift.
    std::map<int, Eigen::Vector3d> proposals;
    std::map<int, double> caps;

    for (const auto& t : tris) {
        if (t[0] < 0) continue;
        const int a = t[0], b = t[1], c = t[2];
        const bool ba = bnd.count(a) > 0, bb = bnd.count(b) > 0, bc = bnd.count(c) > 0;
        const int nBnd = (ba ? 1 : 0) + (bb ? 1 : 0) + (bc ? 1 : 0);
        if (nBnd != 2) continue;  // need exactly 2 rim verts + 1 interior

        int A, B, M;
        if (!ba && bb && bc)      { A = b; B = c; M = a; }
        else if (ba && !bb && bc) { A = c; B = a; M = b; }
        else if (ba && bb && !bc) { A = a; B = b; M = c; }
        else continue;

        const double AM = (verts[static_cast<std::size_t>(M)] -
                           verts[static_cast<std::size_t>(A)]).norm();
        const double BM = (verts[static_cast<std::size_t>(M)] -
                           verts[static_cast<std::size_t>(B)]).norm();
        const double s = std::min(AM, BM);
        if (s < 1e-9) continue;

        // Chord direction
        Eigen::Vector3d chord = verts[static_cast<std::size_t>(B)] -
                                verts[static_cast<std::size_t>(A)];
        const double chordLen = chord.norm();
        if (chordLen < 1e-9) continue;
        const Eigen::Vector3d tdir = chord / chordLen;

        // Vertex normal at M (star sum is well-conditioned even for slivers)
        Eigen::Vector3d n = vnrm[static_cast<std::size_t>(M)];
        if (n.norm() < 1e-12) continue;
        n.normalize();

        // In-plane perpendicular to chord
        Eigen::Vector3d w = n.cross(tdir);
        if (w.norm() < 1e-9) continue;  // chord parallel to normal
        w.normalize();

        // Which side of the chord is M on?
        const double side = (verts[static_cast<std::size_t>(M)] -
                             verts[static_cast<std::size_t>(A)]).dot(w);
        if (std::abs(side) < 1e-9) continue;
        if (side < 0) { w = -w; }  // w now points to M's side

        // Height over chord
        const double h = std::abs(side);
        if (h >= params.rim_lift_min_ratio * s) continue;  // not starved

        // Proposed lift: push M to height ≈ √3/2 × s
        const double target_h = std::sqrt(3.0) / 2.0 * s;
        Eigen::Vector3d delta = (target_h - h) * w;
        const double cap = params.rim_lift_max_move * s;
        if (delta.norm() > cap) delta *= cap / delta.norm();

        proposals[M] += delta;
        if (caps[M] < cap) caps[M] = cap;
    }

    // Apply with backoff + fold guard
    int lifted = 0, skipped = 0;
    for (auto& kv : proposals) {
        const int M = kv.first;
        Eigen::Vector3d delta = kv.second;
        if (delta.norm() < 1e-9) continue;
        auto cit = caps.find(M);
        if (cit != caps.end() && delta.norm() > cit->second)
            delta *= cit->second / delta.norm();

        const Eigen::Vector3d old = verts[static_cast<std::size_t>(M)];
        bool applied = false;
        for (double step : {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125}) {
            verts[static_cast<std::size_t>(M)] = old + step * delta;
            bool ok = true;
            for (int ti : v2t[static_cast<std::size_t>(M)]) {
                if (ti >= static_cast<int>(tris.size())) continue;
                const auto& t = tris[static_cast<std::size_t>(ti)];
                if (t[0] < 0) continue;
                // New face normal
                Eigen::Vector3d fn = (verts[static_cast<std::size_t>(t[1])] -
                                      verts[static_cast<std::size_t>(t[0])])
                                     .cross(verts[static_cast<std::size_t>(t[2])] -
                                            verts[static_cast<std::size_t>(t[0])]);
                if (fn.norm() < 1e-14) { ok = false; break; }  // collapsed
                // Old face normal (M at old position, SAME triangle winding)
                Eigen::Vector3d va = (t[0] == M) ? old : verts[static_cast<std::size_t>(t[0])];
                Eigen::Vector3d vb = (t[1] == M) ? old : verts[static_cast<std::size_t>(t[1])];
                Eigen::Vector3d vc = (t[2] == M) ? old : verts[static_cast<std::size_t>(t[2])];
                Eigen::Vector3d fo = (vb - va).cross(vc - va);
                if (fo.norm() > 1e-14 &&
                    fn.normalized().dot(fo.normalized()) <= 0.0) {
                    ok = false; break;  // fold/invert
                }
            }
            if (ok) { applied = true; ++lifted; break; }
        }
        if (!applied) {
            verts[static_cast<std::size_t>(M)] = old;
            ++skipped;
        }
    }
    if (lifted + skipped > 0) {
        std::cerr << "    [rim-lift] lifted=" << lifted << " skipped=" << skipped << "\n";
    }
}

} // namespace hole_filler_pro
