#include "hole_filler_pro/mesh_entry_cleaner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace hole_filler_pro {

namespace {

// Zero-area gate on the face cross product |(b-a)x(c-a)| == 2*area.
// Scan meshes are millimeter-scale (coords ~1e-1..1e3): the smallest real
// triangle (edges ~1e-6 mm) has 2*area ~1e-12, while exactly-degenerate faces
// produce 0 or a denormal through cancellation. 1e-30 kills only mathematical
// degeneracies, never small-but-real faces.
const double kDegenerateCross = 1e-30;

// Bowtie fan-split displacement scale: copy i moves i * (1e-6 * max|coord|)
// along its fan normal - geometrically negligible (< 0.001 mm at meter-scale
// coords), pairwise distinct, survives float32 STL roundtrips.
const double kEntrySplitEps = 1e-6;

// Exact-key normalization: -0.0 and +0.0 are numerically equal but bitwise
// different; force positive zero so both spellings map to the same key.
inline double normZero(double x) { return x == 0.0 ? 0.0 : x; }

using CoordKey = std::tuple<double, double, double>;
using FaceKey = std::tuple<int, int, int>;

// [s1hash] 64-bit FNV-1a over RAW double bits + splitmix64 final mix.
// Equality is still tuple exact-compare (unordered_map semantics) — hash
// collision auto-falls back to exact compare; "find/emplace only, no
// iteration" means container choice cannot change results.
struct coord_key_hash {
    std::size_t operator()(const CoordKey& k) const {
        std::uint64_t h = 1469598103934665603ULL;
        const double d[3] = { std::get<0>(k), std::get<1>(k), std::get<2>(k) };
        for (double x : d) {
            std::uint64_t b;
            std::memcpy(&b, &x, 8);
            h ^= b;
            h *= 1099511628211ULL;
        }
        // splitmix64 final mix (same mixer as pair_hash in utils.h)
        h += 0x9E3779B97F4A7C15ULL;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
        h ^= (h >> 31);
        return static_cast<std::size_t>(h);
    }
};

struct face_key_hash {
    std::size_t operator()(const FaceKey& k) const {
        std::uint64_t h = 1469598103934665603ULL;
        const int v[3] = { std::get<0>(k), std::get<1>(k), std::get<2>(k) };
        for (int x : v) {
            std::uint64_t b = static_cast<std::uint64_t>(static_cast<std::uint32_t>(x));
            h ^= b;
            h *= 1099511628211ULL;
        }
        h += 0x9E3779B97F4A7C15ULL;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
        h ^= (h >> 31);
        return static_cast<std::size_t>(h);
    }
};

// Edge counting over the internal vector form (same convention as
// buildEdgeCount: canonical key, self-edges skipped).
std::unordered_map<Edge, int, pair_hash> edgeCountOf(const std::vector<Eigen::Vector3i>& tris) {
    std::unordered_map<Edge, int, pair_hash> ec;
    ec.reserve(tris.size() * 3);
    for (const Eigen::Vector3i& t : tris) {
        for (int k = 0; k < 3; ++k) {
            int a = t[k];
            int b = t[(k + 1) % 3];
            if (a == b) continue;
            ++ec[canonicalEdge(a, b)];
        }
    }
    return ec;
}

int boundaryEdgeCount(const std::unordered_map<Edge, int, pair_hash>& ec) {
    int n = 0;
    for (const auto& kv : ec)
        if (kv.second == 1) ++n;
    return n;
}

// Rotate a triangle so its minimum index comes first, PRESERVING cyclic order.
// Same winding + same vertex set -> same key regardless of start vertex;
// reversed winding -> different key. (For distinct a,b,c the min-first
// representative of each directed cycle is unique.)
Eigen::Vector3i canonicalCyclic(const Eigen::Vector3i& t) {
    int m = std::min(t[0], std::min(t[1], t[2]));
    if (t[0] == m) return t;
    if (t[1] == m) return Eigen::Vector3i(t[1], t[2], t[0]);
    return Eigen::Vector3i(t[2], t[0], t[1]);
}

FaceKey faceKey(const Eigen::Vector3i& t) {
    return FaceKey(t[0], t[1], t[2]);
}

// Iterative union-find with path halving (same role as componentTooBig's
// find/uni pair in hole_filler_pro.cpp).
struct DisjointSet {
    std::vector<int> parent;
    explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)) {
        for (int i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i;
    }
    int find(int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    }
    void uni(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[static_cast<std::size_t>(ra)] = rb;
    }
};

} // namespace

EntryCleanResult cleanMeshEntry(const Eigen::MatrixXd& V,
                                const Eigen::MatrixXi& F,
                                const FillParams& params,
                                std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats) {
    EntryCleanResult result;
    EntryCleanReport& rep = result.report;
    rep.V_in = static_cast<int>(V.rows());
    rep.F_in = static_cast<int>(F.rows());

    // ---- [instrumentation only] per-stage timers ----------------------------
    // Prints "[time] clean.<stage>=...ms (...)" per stage when the env var
    // HOLE_FILLER_TIMING is set (same gate as the other [time] lines).
    // Zero effect on geometry; zero output when the env var is absent.
    const bool t_on = (std::getenv("HOLE_FILLER_TIMING") != nullptr);
    std::chrono::steady_clock::time_point t_prev = std::chrono::steady_clock::now();
    char t_buf[160];
    auto t_mark = [&](const char* stage, const char* extra) {
        if (!t_on) return;
        auto t_now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
        t_prev = t_now;
        if (extra && extra[0])
            std::fprintf(stderr, "[time] clean.%s=%.2fms (%s)\n", stage, ms, extra);
        else
            std::fprintf(stderr, "[time] clean.%s=%.2fms\n", stage, ms);
    };

    // Working copies in vector form (the stages below mutate heavily).
    std::vector<Eigen::Vector3d> verts(static_cast<std::size_t>(V.rows()));
    for (int i = 0; i < V.rows(); ++i) verts[static_cast<std::size_t>(i)] = V.row(i);

    // Defensive: faces referencing indices outside V are unusable garbage.
    std::vector<Eigen::Vector3i> tris;
    tris.reserve(static_cast<std::size_t>(F.rows()));
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3i t(static_cast<int>(F(i, 0)),
                          static_cast<int>(F(i, 1)),
                          static_cast<int>(F(i, 2)));
        bool ok = true;
        for (int k = 0; k < 3; ++k)
            if (t[k] < 0 || t[k] >= static_cast<int>(verts.size())) ok = false;
        if (!ok) {
            ++rep.faces_degenerate_removed;
            continue;
        }
        tris.push_back(t);
    }
    std::snprintf(t_buf, sizeof t_buf, "range_bad=%d tris_in=%d",
                  rep.faces_degenerate_removed, static_cast<int>(tris.size()));
    t_mark("prologue", t_buf);

    // ---- S1a: merge exactly-duplicate vertices ------------------------------
    if (!verts.empty()) {
        std::vector<int> vmap(verts.size());
        if (params.s1hash) {
            // [s1hash] unordered_map path: O(1) expected vs map's O(log n);
            // keep-first semantics identical (scan order unchanged, only
            // find/emplace, no iteration) -> bit-identical results.
            std::unordered_map<CoordKey, int, coord_key_hash> first_at;
            first_at.reserve(verts.size() * 2);
            for (std::size_t i = 0; i < verts.size(); ++i) {
                CoordKey key(normZero(verts[i].x()), normZero(verts[i].y()),
                             normZero(verts[i].z()));
                auto it = first_at.find(key);
                if (it == first_at.end()) {
                    first_at.emplace(key, static_cast<int>(i));
                    vmap[i] = static_cast<int>(i);
                } else {
                    vmap[i] = it->second;
                    ++rep.verts_merged;
                }
            }
        } else {
        std::map<CoordKey, int> first_at;
        for (std::size_t i = 0; i < verts.size(); ++i) {
            CoordKey key(normZero(verts[i].x()), normZero(verts[i].y()),
                         normZero(verts[i].z()));
            auto it = first_at.find(key);
            if (it == first_at.end()) {
                first_at[key] = static_cast<int>(i);
                vmap[i] = static_cast<int>(i);
            } else {
                vmap[i] = it->second;
                ++rep.verts_merged;
            }
        }
        }
        if (rep.verts_merged > 0) {
            for (Eigen::Vector3i& t : tris)
                t = Eigen::Vector3i(vmap[static_cast<std::size_t>(t[0])],
                                    vmap[static_cast<std::size_t>(t[1])],
                                    vmap[static_cast<std::size_t>(t[2])]);
        }
    }

    std::snprintf(t_buf, sizeof t_buf, "merged=%d", rep.verts_merged);
    t_mark("S1a_merge_verts", t_buf);

    // ---- S1b: winding-aware duplicate face removal ---------------------------
    {
        std::vector<bool> dead(tris.size(), false);
        if (params.s1hash) {
            // [s1hash] unordered_map path (same keep-first / both-go semantics)
            std::unordered_map<FaceKey, int, face_key_hash> seen;
            seen.reserve(tris.size() * 2);
            for (std::size_t fi = 0; fi < tris.size(); ++fi) {
                const Eigen::Vector3i& t = tris[fi];
                FaceKey fwd_key = faceKey(canonicalCyclic(t));
                FaceKey rev_key = faceKey(canonicalCyclic(Eigen::Vector3i(t[0], t[2], t[1])));
                auto itf = seen.find(fwd_key);
                if (itf != seen.end()) {
                    dead[fi] = true;
                    ++rep.faces_dup_removed;
                    continue;
                }
                auto itr = seen.find(rev_key);
                if (itr != seen.end()) {
                    dead[fi] = true;
                    if (!dead[static_cast<std::size_t>(itr->second)]) {
                        dead[static_cast<std::size_t>(itr->second)] = true;
                        ++rep.faces_dup_removed;
                    }
                    ++rep.faces_dup_removed;
                    continue;
                }
                seen.emplace(fwd_key, static_cast<int>(fi));
            }
        } else {
        std::map<FaceKey, int> seen;   // forward cyclic key -> first face id
        for (std::size_t fi = 0; fi < tris.size(); ++fi) {
            const Eigen::Vector3i& t = tris[fi];
            FaceKey fwd_key = faceKey(canonicalCyclic(t));
            FaceKey rev_key = faceKey(canonicalCyclic(Eigen::Vector3i(t[0], t[2], t[1])));
            auto itf = seen.find(fwd_key);
            if (itf != seen.end()) {
                dead[fi] = true;              // same-direction dup: keep first
                ++rep.faces_dup_removed;
                continue;
            }
            auto itr = seen.find(rev_key);
            if (itr != seen.end()) {
                dead[fi] = true;              // opposite winding: both go
                if (!dead[static_cast<std::size_t>(itr->second)]) {
                    dead[static_cast<std::size_t>(itr->second)] = true;
                    ++rep.faces_dup_removed;
                }
                ++rep.faces_dup_removed;
                continue;
            }
            seen[fwd_key] = static_cast<int>(fi);
        }
        }
        if (rep.faces_dup_removed > 0) {
            std::vector<Eigen::Vector3i> kept;
            kept.reserve(tris.size());
            for (std::size_t fi = 0; fi < tris.size(); ++fi)
                if (!dead[fi]) kept.push_back(tris[fi]);
            tris.swap(kept);
        }
    }

    std::snprintf(t_buf, sizeof t_buf, "dup=%d", rep.faces_dup_removed);
    t_mark("S1b_dup_faces", t_buf);

    const int t_degen_pre = rep.faces_degenerate_removed;  // [instr] S1c delta baseline

    // ---- S1c: zero-area face removal (also sweeps repeated-index faces) ------
    {
        std::vector<Eigen::Vector3i> kept;
        kept.reserve(tris.size());
        for (const Eigen::Vector3i& t : tris) {
            if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
                ++rep.faces_degenerate_removed;
                continue;
            }
            const Eigen::Vector3d& a = verts[static_cast<std::size_t>(t[0])];
            const Eigen::Vector3d& b = verts[static_cast<std::size_t>(t[1])];
            const Eigen::Vector3d& c = verts[static_cast<std::size_t>(t[2])];
            if ((b - a).cross(c - a).norm() < kDegenerateCross) {
                ++rep.faces_degenerate_removed;
                continue;
            }
            kept.push_back(t);
        }
        tris.swap(kept);
    }

    std::snprintf(t_buf, sizeof t_buf, "degen=%d", rep.faces_degenerate_removed - t_degen_pre);
    t_mark("S1c_degenerate", t_buf);

    // ---- [clean-opt shareec] group-shared edge table ------------------------
    // Built once after S1c and reused by S2 (iteration 0), S2.6 (boundary
    // marks) and S2.5 (flap rounds). Every face DELETION below (S2.6 collapse
    // sweep -> S2.7 veil -> S2.5 rounds, the real execution order) decrements
    // the table instead of invalidating it; any topology REWRITE (S2 split /
    // prune / rollback, or an invalid entry state) makes consumers fall back
    // to building their own table exactly as in the original code.
    std::unordered_map<Edge, int, pair_hash> ec_shared;
    bool ec_valid = false;
    if (params.shareec && !tris.empty()) {
        ec_shared = edgeCountOf(tris);
        ec_valid = true;
    }
    auto ec_dec_face = [&](const Eigen::Vector3i& t) {
        if (!(params.shareec && ec_valid)) return;
        for (int k = 0; k < 3; ++k) {
            int a = t[k], b = t[(k + 1) % 3];
            if (a == b) continue;
            Edge e = canonicalEdge(a, b);
            auto it = ec_shared.find(e);
            if (it != ec_shared.end() && --(it->second) <= 0) ec_shared.erase(it);
        }
    };

    // ---- S2: bowtie split + non-manifold edge pruning (bounded iterations) ---
    for (int iter = 0; iter < params.entry_bowtie_max_iters && !tris.empty(); ++iter) {
        const std::vector<Eigen::Vector3d> verts_snap = verts;
        const std::vector<Eigen::Vector3i> tris_snap = tris;

        // [clean-opt shareec] iteration 0 reuses the shared table while it is
        // still valid (tris unchanged since S1c); later iterations and any
        // invalid state build their own table exactly as before.
        std::unordered_map<Edge, int, pair_hash> ec_pre_build;
        const bool ec_reuse = (params.shareec && ec_valid && iter == 0);
        if (!ec_reuse) ec_pre_build = edgeCountOf(tris);
        const std::unordered_map<Edge, int, pair_hash>& ec_pre =
            ec_reuse ? ec_shared : ec_pre_build;
        const int boundary_pre = boundaryEdgeCount(ec_pre);

        // (1) Boundary degree per vertex: endpoints of count==1 edges.
        std::vector<int> bdeg(verts.size(), 0);
        for (const auto& kv : ec_pre) {
            if (kv.second == 1) {
                ++bdeg[static_cast<std::size_t>(kv.first.first)];
                ++bdeg[static_cast<std::size_t>(kv.first.second)];
            }
        }

        // Mesh scale for the split offset.
        double max_abs = 0.0;
        for (const Eigen::Vector3d& p : verts)
            max_abs = std::max(max_abs, p.cwiseAbs().maxCoeff());
        if (!(max_abs > 0.0)) max_abs = 1.0;
        const double eps = kEntrySplitEps * max_abs;

        // vertex -> incident faces (ids ascend).
        // [clean-opt s2fan] When on: CSR counting sort (two passes) replaces
        // the hash map — faces per vertex end up in ascending fi order because
        // we scan fi ascending during the scatter pass. Saves the ~600-1000ms
        // hash-map construction on 525K vertices.
        std::vector<int> vf_off, vf_lst;   // CSR: [vf_off[v], vf_off[v+1]) = faces of v
        std::unordered_map<int, std::vector<int>> v2t;
        if (params.s2fan) {
            const int nv = static_cast<int>(verts.size());
            vf_off.assign(static_cast<std::size_t>(nv) + 1, 0);
            for (const Eigen::Vector3i& t : tris)
                for (int k = 0; k < 3; ++k)
                    ++vf_off[static_cast<std::size_t>(t[k]) + 1];
            for (int i = 0; i < nv; ++i)
                vf_off[static_cast<std::size_t>(i) + 1] += vf_off[static_cast<std::size_t>(i)];
            vf_lst.assign(vf_off[static_cast<std::size_t>(nv)], 0);
            {
                std::vector<int> cursor(vf_off.begin(), vf_off.end() - 1);
                for (std::size_t fi = 0; fi < tris.size(); ++fi)
                    for (int k = 0; k < 3; ++k) {
                        int vi = tris[fi][k];
                        vf_lst[static_cast<std::size_t>(cursor[static_cast<std::size_t>(vi)]++)] =
                            static_cast<int>(fi);
                    }
            }
        } else {
            for (std::size_t fi = 0; fi < tris.size(); ++fi)
                for (int k = 0; k < 3; ++k)
                    v2t[tris[fi][k]].push_back(static_cast<int>(fi));
        }

        // (2) Fan split at INTERIOR bowtie vertices (boundary degree == 0).
        int splits_this_iter = 0;
        const int nverts = static_cast<int>(verts.size());   // copies are not candidates

        // [clean-opt s2fan] Fast single-fan pre-check (cycle trace).
        // For an interior vertex where all edges have count==2, the faces
        // around it form a graph of degree 2 (each face connected to exactly
        // 2 others through its 2 v-edges). A single fan = one cycle visiting
        // all faces; a bowtie = multiple cycles. We try to trace one cycle
        // in O(d²); if it visits all d faces, the vertex is manifold → skip
        // the expensive BFS. Conservative: any anomaly falls through.
        auto isSingleFan = [&](int v, const int* inc, int d) -> bool {
            if (d < 2) return true;   // trivially manifold
            // Find v's two neighbors in face inc[0]
            const Eigen::Vector3i& t0 = tris[static_cast<std::size_t>(inc[0])];
            int nb0 = -1, nb1 = -1;
            for (int k = 0; k < 3; ++k) {
                if (t0[k] != v) {
                    if (nb0 < 0) nb0 = t0[k];
                    else nb1 = t0[k];
                }
            }
            if (nb0 < 0 || nb1 < 0) return false;   // degenerate (S1c should have removed)
            // Trace: enter face 0 through (v,nb0), will exit through (v,nb1)
            int enter_via = nb1;   // exit edge of face 0 = entry of next face
            int visited = 1;
            int cur_idx = 0;
            std::vector<char> vis(static_cast<std::size_t>(d), 0);
            vis[0] = 1;
            while (visited < d) {
                // Find the next face: the one sharing edge (v, enter_via) with
                // the current face, excluding the current face. Scan inc.
                int next_idx = -1;
                for (int j = 0; j < d; ++j) {
                    if (vis[j]) continue;
                    const Eigen::Vector3i& tj = tris[static_cast<std::size_t>(inc[j])];
                    for (int k = 0; k < 3; ++k)
                        if (tj[k] == enter_via) { next_idx = j; break; }
                    if (next_idx >= 0) break;
                }
                if (next_idx < 0) return false;   // no unvisited neighbor → not single cycle
                // Find the exit edge: the other v-neighbor in the next face
                const Eigen::Vector3i& tn = tris[static_cast<std::size_t>(inc[next_idx])];
                int exit_via = -1;
                for (int k = 0; k < 3; ++k) {
                    if (tn[k] != v && tn[k] != enter_via) {
                        exit_via = tn[k];
                        break;
                    }
                }
                if (exit_via < 0) return false;   // degenerate
                vis[next_idx] = 1;
                ++visited;
                enter_via = exit_via;
                cur_idx = next_idx;
            }
            return true;   // visited all d faces in one cycle
        };

        for (int v = 0; v < nverts; ++v) {
            if (bdeg[static_cast<std::size_t>(v)] != 0) continue;

            // [s2fan] CSR path: no hash-map lookup
            const int* inc_data = nullptr;
            int inc_size = 0;
            if (params.s2fan) {
                inc_data = &vf_lst[static_cast<std::size_t>(vf_off[static_cast<std::size_t>(v)])];
                inc_size = vf_off[static_cast<std::size_t>(v) + 1] -
                          vf_off[static_cast<std::size_t>(v)];
                if (inc_size < 2) continue;
                // Fast manifold check: if single fan, skip the expensive BFS
                if (isSingleFan(v, inc_data, inc_size)) continue;
                // Fall through: might be a bowtie → run the original BFS
                // (build a temp vector from the CSR for the BFS code below)
            } else {
                auto it = v2t.find(v);
                if (it == v2t.end() || it->second.size() < 2) continue;
            }
            // Build inc vector from whichever source we used
            std::vector<int> inc;
            if (params.s2fan) {
                inc.assign(inc_data, inc_data + inc_size);
            } else {
                inc = v2t[v];   // copy (the original code used a const ref)
            }

            // Faces around v connected through an edge CONTAINING v form one
            // fan; bowtie sheets share no such edge -> separate fans.
            std::unordered_map<Edge, std::vector<int>, pair_hash> edge_faces;
            for (int fi : inc) {
                const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
                for (int k = 0; k < 3; ++k) {
                    int a = t[k], b = t[(k + 1) % 3];
                    if (a != b && (a == v || b == v))
                        edge_faces[canonicalEdge(a, b)].push_back(fi);
                }
            }
            std::vector<std::vector<int>> fans;
            std::vector<char> visited(inc.size(), 0);
            for (std::size_t i = 0; i < inc.size(); ++i) {
                if (visited[i]) continue;
                std::vector<int> fan;
                std::queue<std::size_t> q;
                q.push(i);
                visited[i] = 1;
                while (!q.empty()) {
                    std::size_t u = q.front();
                    q.pop();
                    fan.push_back(inc[u]);
                    const Eigen::Vector3i& t = tris[static_cast<std::size_t>(inc[u])];
                    for (int k = 0; k < 3; ++k) {
                        int a = t[k], b = t[(k + 1) % 3];
                        if (a == b || (a != v && b != v)) continue;
                        for (int nb : edge_faces[canonicalEdge(a, b)]) {
                            auto pos = std::lower_bound(inc.begin(), inc.end(), nb);
                            std::size_t ni = static_cast<std::size_t>(pos - inc.begin());
                            if (!visited[ni]) {
                                visited[ni] = 1;
                                q.push(ni);
                            }
                        }
                    }
                }
                std::sort(fan.begin(), fan.end());
                fans.push_back(std::move(fan));
            }
            if (fans.size() < 2) continue;   // manifold disk: nothing to split

            // Fan 0 keeps v. Fan j >= 1 gets a copy displaced j*eps along its
            // own area-weighted normal (distinct magnitudes -> distinct coords).
            for (std::size_t j = 1; j < fans.size(); ++j) {
                Eigen::Vector3d nsum = Eigen::Vector3d::Zero();
                for (int fi : fans[j]) {
                    const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
                    nsum += (verts[static_cast<std::size_t>(t[1])] -
                             verts[static_cast<std::size_t>(t[0])])
                                .cross(verts[static_cast<std::size_t>(t[2])] -
                                       verts[static_cast<std::size_t>(t[0])]);
                }
                Eigen::Vector3d unit = nsum.norm() > kDegenerateCross
                                           ? nsum.normalized()
                                           : Eigen::Vector3d::UnitZ();
                const int copy_id = static_cast<int>(verts.size());
                verts.push_back(verts[static_cast<std::size_t>(v)] +
                                static_cast<double>(j) * eps * unit);
                for (int fi : fans[j]) {
                    Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
                    for (int k = 0; k < 3; ++k)
                        if (t[k] == v) t[k] = copy_id;
                }
                ++splits_this_iter;
            }
        }

        // (3) Non-manifold edge pruning (count > 2): keep the 2 faces that
        // best agree with the largest-area face's normal (ties: larger area,
        // then lower face id). Deterministic sorted edge order.
        int pruned_this_iter = 0;
        {
            std::unordered_map<Edge, std::vector<int>, pair_hash> e2f;
            std::vector<Edge> bad_edges;
            if (params.s2merge && splits_this_iter == 0) {
                // [clean-opt s2merge] Fast path: bad edges from the already
                // built ec_pre table (no full edge->faces pass), then face
                // lists ONLY for those edges. ec_pre is valid here because
                // the fan-split step made no change this iteration; with any
                // split we fall back to the full build below (correctness
                // first). Face ids are pushed in ascending fi order, same as
                // the full build produces.
                for (const auto& kv : ec_pre)
                    if (kv.second > 2) bad_edges.push_back(kv.first);
                std::sort(bad_edges.begin(), bad_edges.end());
                if (!bad_edges.empty()) {
                    for (std::size_t fi = 0; fi < tris.size(); ++fi) {
                        const Eigen::Vector3i& t = tris[fi];
                        for (int k = 0; k < 3; ++k) {
                            int a = t[k], b = t[(k + 1) % 3];
                            if (a != b) {
                                Edge e = canonicalEdge(a, b);
                                if (std::binary_search(bad_edges.begin(),
                                                        bad_edges.end(), e))
                                    e2f[e].push_back(static_cast<int>(fi));
                            }
                        }
                    }
                }
            } else {
                e2f.reserve(tris.size() * 3);
                for (std::size_t fi = 0; fi < tris.size(); ++fi) {
                    const Eigen::Vector3i& t = tris[fi];
                    for (int k = 0; k < 3; ++k) {
                        int a = t[k], b = t[(k + 1) % 3];
                        if (a != b) e2f[canonicalEdge(a, b)].push_back(static_cast<int>(fi));
                    }
                }
                for (const auto& kv : e2f)
                    if (kv.second.size() > 2) bad_edges.push_back(kv.first);
                std::sort(bad_edges.begin(), bad_edges.end());
            }

            std::vector<bool> dead(tris.size(), false);
            for (const Edge& e : bad_edges) {
                struct Cand {
                    int fi;
                    double score;
                    double area;
                    Eigen::Vector3d n;
                };
                std::vector<Cand> cands;
                Eigen::Vector3d ref_n = Eigen::Vector3d::UnitZ();
                double ref_area = -1.0;
                for (int fi : e2f[e]) {
                    if (dead[static_cast<std::size_t>(fi)]) continue;
                    const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
                    Eigen::Vector3d cr =
                        (verts[static_cast<std::size_t>(t[1])] -
                         verts[static_cast<std::size_t>(t[0])])
                            .cross(verts[static_cast<std::size_t>(t[2])] -
                                   verts[static_cast<std::size_t>(t[0])]);
                    double area = 0.5 * cr.norm();
                    Eigen::Vector3d n = cr.norm() > kDegenerateCross
                                            ? cr.normalized()
                                            : Eigen::Vector3d::UnitZ();
                    if (area > ref_area) {
                        ref_area = area;
                        ref_n = n;
                    }
                    cands.push_back(Cand{fi, 0.0, area, n});
                }
                if (cands.size() <= 2) continue;
                for (Cand& c : cands) c.score = c.n.dot(ref_n);
                std::sort(cands.begin(), cands.end(),
                          [](const Cand& x, const Cand& y) {
                              if (x.score != y.score) return x.score > y.score;
                              if (x.area != y.area) return x.area > y.area;
                              return x.fi < y.fi;
                          });
                for (std::size_t i = 2; i < cands.size(); ++i) {
                    dead[static_cast<std::size_t>(cands[i].fi)] = true;
                    ++pruned_this_iter;
                }
            }
            if (pruned_this_iter > 0) {
                std::vector<Eigen::Vector3i> kept;
                kept.reserve(tris.size());
                for (std::size_t fi = 0; fi < tris.size(); ++fi)
                    if (!dead[fi]) kept.push_back(tris[fi]);
                tris.swap(kept);
            }
        }

        // (4) Rollback gate: this stage must never open new boundary edges.
        // [clean-opt s2short] When this iteration changed nothing, tris is
        // identical to the mesh ec_pre was built from, so the boundary count
        // equals boundary_pre by construction -- skip the full rebuild.
        int boundary_post;
        if (params.s2short && splits_this_iter == 0 && pruned_this_iter == 0) {
            boundary_post = boundary_pre;
        } else {
            boundary_post = boundaryEdgeCount(edgeCountOf(tris));
        }
        ++rep.bowtie_iters_done;
        if (boundary_post > boundary_pre) {
            verts = verts_snap;
            tris = tris_snap;
            rep.bowtie_rolled_back = true;
            // Restored mesh == the state ec_shared was built on when iter 0
            // reused it; any other rollback leaves the table stale.
            if (!ec_reuse) ec_valid = false;
            break;
        }
        rep.verts_bowtie_split += splits_this_iter;
        rep.faces_nonmanifold_removed += pruned_this_iter;
        if (splits_this_iter > 0 || pruned_this_iter > 0)
            ec_valid = false;   // topology rewritten: shared table is stale
        if (splits_this_iter == 0 && pruned_this_iter == 0)
            break;   // converged: another pass would change nothing
    }

    std::snprintf(t_buf, sizeof t_buf, "iters=%d split=%d nm_pruned=%d rolled_back=%d",
                  rep.bowtie_iters_done, rep.verts_bowtie_split,
                  rep.faces_nonmanifold_removed, rep.bowtie_rolled_back ? 1 : 0);
    t_mark("S2_bowtie_nm", t_buf);

    // ---- S2.6: spike/pit removal (ported from archived SmoothingCore -------
    // SpikeRemover v0.2, unified absolute-value mode): non-boundary vertex far
    // off its ring-neighbors' PCA plane (ratio = |d_v|/d_mean > threshold) is
    // snapped to the neighbor centroid. Kills needle/sliver structures the
    // topological stages cannot touch (they are manifold and edge-connected).
    // After snapping, a degenerate sweep removes collapsed faces.
    if (params.entry_spike_remove && !tris.empty()) {
        const int ring = std::max(1, params.entry_spike_ring);
        const double ratioThr = params.entry_spike_ratio;
        const size_t cap = std::max<size_t>(1, verts.size() / 100);   // 1% cap
        const double eps = 1e-15;
        int snapped = 0;
        const bool spikeDbg = (std::getenv("SPIKE_DBG") != nullptr);   // hoisted out of the hot loop

        // vertex adjacency + boundary marks
        std::vector<std::vector<int>> vadj(verts.size());
        {
            for (const Eigen::Vector3i& t : tris) {
                int a = t[0], b = t[1], c = t[2];
                if (a != b) { vadj[(size_t)a].push_back(b); vadj[(size_t)b].push_back(a); }
                if (b != c) { vadj[(size_t)b].push_back(c); vadj[(size_t)c].push_back(b); }
                if (a != c) { vadj[(size_t)a].push_back(c); vadj[(size_t)c].push_back(a); }
            }
        }
        std::vector<char> isBoundary(verts.size(), 0);
        {
            // [clean-opt shareec] boundary marks from the shared table when
            // valid (S2.6 only moves positions, never topology, before its
            // collapse sweep); otherwise build exactly as before.
            std::unordered_map<Edge, int, pair_hash> ec_local;
            const std::unordered_map<Edge, int, pair_hash>* ecp;
            if (params.shareec && ec_valid) {
                ecp = &ec_shared;
            } else {
                ec_local = edgeCountOf(tris);
                ecp = &ec_local;
            }
            for (const auto& kv : *ecp) {
                if (kv.second == 1) {
                    isBoundary[(size_t)kv.first.first] = 1;
                    isBoundary[(size_t)kv.first.second] = 1;
                }
            }
        }

        // [clean-opt spikedirty] Round 0 always scans everything; rounds 1..N
        // only re-scan the closed ring-ball of the vertices snapped in the
        // previous round. Undirected adjacency makes the ball relation
        // symmetric, so a vertex outside the dirty union had zero input
        // change since its last (no-snap) evaluation -- constructive
        // equivalence, see docs/清扫优化专题-2026-11-09.md §10/M1.
        std::vector<char> dirty;   // empty => flag off (or round 0): no skipping
        for (int round = 0; round < 5 && (size_t)snapped < cap; ++round) {
            std::vector<std::pair<int, Eigen::Vector3d>> toSnap;
            std::vector<int> stamp(verts.size(), -1);
            std::vector<int> bfsQueue;
            for (size_t v = 0; v < verts.size(); ++v) {
                if ((size_t)snapped + toSnap.size() >= cap) break;
                if (round > 0 && !dirty.empty() && !dirty[v]) continue;
                if (isBoundary[v]) continue;
                const std::vector<int>& adj1 = vadj[v];
                if (adj1.size() < 4) continue;
                // ring BFS (iterative, stamped)
                bfsQueue.clear();
                bfsQueue.push_back((int)v);
                stamp[v] = (int)v;
                for (int depth = 0; depth < ring; ++depth) {
                    size_t layerEnd = bfsQueue.size();
                    for (size_t qi = 0; qi < layerEnd; ++qi) {
                        int u = bfsQueue[qi];
                        for (int w : vadj[(size_t)u]) {
                            if (stamp[(size_t)w] == (int)v) continue;
                            stamp[(size_t)w] = (int)v;
                            bfsQueue.push_back(w);
                        }
                    }
                }
                std::vector<Eigen::Vector3d> nb;
                nb.reserve(bfsQueue.size());
                for (int u : bfsQueue)
                    if (u != (int)v) nb.push_back(verts[(size_t)u]);
                if (nb.size() < 5) continue;
                Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
                for (const auto& p : nb) centroid += p;
                centroid /= (double)nb.size();
                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& p : nb) {
                    Eigen::Vector3d d = p - centroid;
                    cov += d * d.transpose();
                }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
                if (solver.info() != Eigen::Success) continue;
                Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
                if (!normal.allFinite()) continue;
                double d_v = (verts[v] - centroid).dot(normal);
                double d_sum = 0.0;
                for (const auto& p : nb) d_sum += std::abs((p - centroid).dot(normal));
                double d_mean = d_sum / (double)nb.size();
                double ratio = std::abs(d_v) / std::max(d_mean, eps);
                if (spikeDbg && ratio > 0.5)
                    std::fprintf(stderr, "[spikedbg] v=%zu nb=%zu d_v=%.3f d_mean=%.3f ratio=%.2f round=%d c=(%.2f,%.2f,%.2f) n=(%.3f,%.3f,%.3f) vz=%.2f\n",
                                 v, nb.size(), d_v, d_mean, ratio, round,
                                 centroid.x(), centroid.y(), centroid.z(),
                                 normal.x(), normal.y(), normal.z(), verts[v].z());
                if (ratio > ratioThr) toSnap.emplace_back((int)v, centroid);
            }
            if (toSnap.empty()) break;
            for (const auto& c : toSnap) verts[(size_t)c.first] = c.second;
            snapped += (int)toSnap.size();
            if (params.spikedirty) {
                // Next round's dirty set = union over snapped vertices of
                // their closed ring-ball (radius = same `ring` the evaluation
                // uses, snapping itself included).
                if (dirty.empty()) dirty.assign(verts.size(), 0);
                else std::fill(dirty.begin(), dirty.end(), 0);
                std::vector<int> dstamp(verts.size(), -1);
                std::vector<int> dq;
                for (const auto& c : toSnap) {
                    const int mark = c.first;   // unique per snapped source
                    dq.clear();
                    dq.push_back(c.first);
                    dstamp[(size_t)c.first] = mark;
                    for (int depth = 0; depth < ring; ++depth) {
                        size_t layerEnd = dq.size();
                        for (size_t qi = 0; qi < layerEnd; ++qi) {
                            int u = dq[qi];
                            for (int w : vadj[(size_t)u]) {
                                if (dstamp[(size_t)w] == mark) continue;
                                dstamp[(size_t)w] = mark;
                                dq.push_back(w);
                            }
                        }
                    }
                    for (int w : dq) dirty[(size_t)w] = 1;
                }
            }
        }
        rep.spikes_snapped = snapped;

        // collapsed-face sweep (same gate as S1c)
        {
            std::vector<Eigen::Vector3i> kept;
            kept.reserve(tris.size());
            for (const Eigen::Vector3i& t : tris) {
                const Eigen::Vector3d& a = verts[(size_t)t[0]];
                const Eigen::Vector3d& b = verts[(size_t)t[1]];
                const Eigen::Vector3d& c = verts[(size_t)t[2]];
                if ((b - a).cross(c - a).norm() < kDegenerateCross) {
                    ++rep.faces_degenerate_removed;
                    ec_dec_face(t);   // [shareec] keep the shared table exact
                    continue;
                }
                kept.push_back(t);
            }
            tris.swap(kept);
        }
    }

    std::snprintf(t_buf, sizeof t_buf, "spikes=%d", rep.spikes_snapped);
    t_mark("S2.6_spike", t_buf);

    // ---- S2.7: sparse-veil removal (opt-in) ----------------------------------
    // Membrane faces spanning empty space (BPV2 boundary-expansion remnants):
    // long in TWO edge directions (unlike normal slivers: long in one). Rule:
    // 2nd-longest edge > entry_veil_edge_mult x GLOBAL median edge. On real
    // data the class is 0.001% of faces and the threshold sits above anything
    // the ball radius can legitimately produce - near-zero false positives.
    if (params.entry_veil_remove && params.entry_veil_edge_mult > 0.0 && !tris.empty()) {
        std::vector<double> lens;
        lens.reserve(tris.size() * 3);
        for (const Eigen::Vector3i& t : tris) {
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (a == b) continue;
                lens.push_back((verts[(size_t)a] - verts[(size_t)b]).norm());
            }
        }
        if (!lens.empty()) {
            std::nth_element(lens.begin(), lens.begin() + (std::ptrdiff_t)(lens.size() / 2), lens.end());
            const double median = lens[lens.size() / 2];
            const double thr = params.entry_veil_edge_mult * median;
            const size_t cap = std::max<size_t>(16, tris.size() / 100);   // 1% sanity cap
            size_t removed = 0;
            std::vector<Eigen::Vector3i> kept;
            kept.reserve(tris.size());
            for (const Eigen::Vector3i& t : tris) {
                if (removed < cap) {
                    double e[3];
                    for (int k = 0; k < 3; ++k) {
                        int a = t[k], b = t[(k + 1) % 3];
                        e[k] = (a == b) ? 0.0 : (verts[(size_t)a] - verts[(size_t)b]).norm();
                    }
                    std::sort(e, e + 3);
                    if (e[1] > thr) {   // 2nd-longest over threshold = veil
                        ++removed;
                        ec_dec_face(t);   // [shareec] keep the shared table exact
                        continue;
                    }
                }
                kept.push_back(t);
            }
            if (removed > 0) {
                tris.swap(kept);
                rep.faces_veil_removed = (int)removed;
            }
        }
    }

    std::snprintf(t_buf, sizeof t_buf, "veil=%d", rep.faces_veil_removed);
    t_mark("S2.7_veil", t_buf);

    // ---- S2.5: dangling-flap peel (opt-in) -----------------------------------
    // Repeatedly remove faces with >= 2 boundary edges: thin strips / flaps
    // hanging off the surface by an edge (BPV2 stitch remnants - small-CC
    // removal cannot touch them because they are edge-connected to the body).
    // NOTE: convex rim corners also match and lose a few layers; the trim
    // stage removes boundary rings around holes anyway, so this erosion is
    // inconsequential HERE. Hard caps: 3 rounds, 0.25% of faces.
    if (params.entry_flap_peel && !tris.empty()) {
        const size_t cap = std::max<size_t>(16, tris.size() / 400);
        int peeled = 0;
        for (int round = 0; round < 3; ++round) {
            // [clean-opt shareec] flap rounds reuse the (incrementally
            // maintained) shared table when valid; otherwise rebuild exactly
            // as before.
            std::unordered_map<Edge, int, pair_hash> ec_build;
            const std::unordered_map<Edge, int, pair_hash>* ec_ptr;
            if (params.shareec && ec_valid) {
                ec_ptr = &ec_shared;
            } else {
                ec_build = edgeCountOf(tris);
                ec_ptr = &ec_build;
            }
            const std::unordered_map<Edge, int, pair_hash>& ec = *ec_ptr;
            std::vector<bool> dead(tris.size(), false);
            size_t rm = 0;
            for (std::size_t fi = 0; fi < tris.size(); ++fi) {
                const Eigen::Vector3i& t = tris[fi];
                int b = 0;
                for (int k = 0; k < 3; ++k) {
                    int a = t[k], c = t[(k + 1) % 3];
                    if (a == c) { ++b; continue; }
                    auto it = ec.find(canonicalEdge(a, c));
                    if (it == ec.end() || it->second == 1) ++b;
                }
                if (b >= 2) { dead[fi] = true; ++rm; }
            }
            if (rm == 0 || peeled + rm > cap) break;
            for (std::size_t fi = 0; fi < tris.size(); ++fi)
                if (dead[fi]) ec_dec_face(tris[fi]);   // [shareec] decrement BEFORE filtering
            std::vector<Eigen::Vector3i> kept;
            kept.reserve(tris.size() - rm);
            for (std::size_t fi = 0; fi < tris.size(); ++fi)
                if (!dead[fi]) kept.push_back(tris[fi]);
            tris.swap(kept);
            peeled += (int)rm;
        }
        rep.faces_flap_peeled = peeled;
    }

    std::snprintf(t_buf, sizeof t_buf, "flap=%d", rep.faces_flap_peeled);
    t_mark("S2.5_flap", t_buf);

    // ---- S3: small connected components (opt-in) ----------------------------
    if (params.entry_small_cc_min_faces > 0 && tris.size() > 1) {
        const int nf = static_cast<int>(tris.size());
        DisjointSet dsu(nf);
        std::unordered_map<Edge, int, pair_hash> last_on_edge;
        for (int fi = 0; fi < nf; ++fi) {
            const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (a == b) continue;
                Edge e = canonicalEdge(a, b);
                auto it = last_on_edge.find(e);
                if (it == last_on_edge.end()) last_on_edge[e] = fi;
                else dsu.uni(fi, it->second);
            }
        }
        std::map<int, std::pair<int, double>> cc;   // root -> (faces, area)
        for (int fi = 0; fi < nf; ++fi) {
            const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
            double area = triangleArea(verts[static_cast<std::size_t>(t[0])],
                                       verts[static_cast<std::size_t>(t[1])],
                                       verts[static_cast<std::size_t>(t[2])]);
            std::pair<int, double>& st = cc[dsu.find(fi)];
            st.first += 1;
            st.second += area;
        }
        if (cc.size() > 1) {
            int keep_root = cc.begin()->first;
            double max_area = -1.0;
            for (const auto& kv : cc) {
                if (kv.second.second > max_area) {
                    max_area = kv.second.second;
                    keep_root = kv.first;
                }
            }
            const double area_thresh = params.entry_small_cc_area_ratio * max_area;
            std::vector<bool> dead(tris.size(), false);
            for (int fi = 0; fi < nf; ++fi) {
                int root = dsu.find(fi);
                if (root == keep_root) continue;
                const std::pair<int, double>& st = cc[root];
                if (st.first < params.entry_small_cc_min_faces &&
                    st.second < area_thresh) {
                    dead[static_cast<std::size_t>(fi)] = true;
                    ++rep.faces_small_cc_removed;
                }
            }
            int removed_cc = 0;
            if (rep.faces_small_cc_removed > 0) {
                std::vector<Eigen::Vector3i> kept;
                kept.reserve(tris.size());
                for (int fi = 0; fi < nf; ++fi) {
                    if (!dead[static_cast<std::size_t>(fi)]) {
                        kept.push_back(tris[static_cast<std::size_t>(fi)]);
                    } else if (fi == dsu.find(fi)) {
                        ++removed_cc;
                    }
                }
                tris.swap(kept);
            }
            rep.small_cc_removed = removed_cc;
        }
    }

    std::snprintf(t_buf, sizeof t_buf, "cc=%d cc_faces=%d",
                  rep.small_cc_removed, rep.faces_small_cc_removed);
    t_mark("S3_small_cc", t_buf);

    // ---- compact + emit (boundary_trimmer idiom) ----------------------------
    std::vector<bool> vert_used(verts.size(), false);
    for (const Eigen::Vector3i& t : tris)
        for (int k = 0; k < 3; ++k)
            vert_used[static_cast<std::size_t>(t[k])] = true;

    std::vector<int> old_to_new(verts.size(), -1);
    int new_count = 0;
    for (std::size_t i = 0; i < verts.size(); ++i)
        if (vert_used[i]) old_to_new[i] = new_count++;

    result.V.resize(new_count, 3);
    for (std::size_t i = 0; i < verts.size(); ++i)
        if (old_to_new[i] >= 0) result.V.row(old_to_new[i]) = verts[i];

    result.F.resize(static_cast<int>(tris.size()), 3);
    int ri = 0;
    for (const Eigen::Vector3i& t : tris) {
        result.F(ri, 0) = old_to_new[static_cast<std::size_t>(t[0])];
        result.F(ri, 1) = old_to_new[static_cast<std::size_t>(t[1])];
        result.F(ri, 2) = old_to_new[static_cast<std::size_t>(t[2])];
        ++ri;
    }
    rep.V_out = static_cast<int>(result.V.rows());
    rep.F_out = static_cast<int>(result.F.rows());

    // Refresh the caller's table for the OUTPUT mesh (ids changed by compaction).
    // [M1] unified {len,cnt} table (bitwise-replica builder, self-edges kept).
    edge_stats = buildEdgeStats(result.V, result.F);
    t_mark("compact_emit", nullptr);
    return result;
}

// ============================================================================
// [exit-nm] exit non-manifold audit + prune (mustclean task). The S2 rollback
// gate can PASS non-manifold edges through (rollback restores faces the
// scorer had already killed; the max-iters exit and both early returns never
// run the scorer at all), so the fillHoles exit re-checks the final mesh.
// Audit = read-only count; prune = guarded second-pass trim that can never
// open a new boundary edge (see header contract).
// ============================================================================

int countNmEdges(const std::unordered_map<Edge, int, pair_hash>& edge_count) {
    int n = 0;
    for (const auto& kv : edge_count)
        if (kv.second >= 3 && kv.first.first != kv.first.second) ++n;
    return n;
}

int countNmEdges(const std::unordered_map<Edge, EdgeStat, pair_hash>& edge_stats) {
    // [M1] unified-table overload (self-edges filtered at read site).
    int n = 0;
    for (const auto& kv : edge_stats)
        if (kv.second.cnt >= 3 && kv.first.first != kv.first.second) ++n;
    return n;
}

int pruneNmEdgesExit(const std::vector<Eigen::Vector3d>& verts,
                     std::vector<Eigen::Vector3i>& tri_list,
                     std::unordered_map<Edge, int, pair_hash>& edge_count) {
    // 1) nm edges in canonical order -> deterministic processing order.
    std::vector<Edge> nm;
    for (const auto& kv : edge_count)
        if (kv.second >= 3 && kv.first.first != kv.first.second)
            nm.push_back(kv.first);
    std::sort(nm.begin(), nm.end());
    if (nm.empty()) return 0;

    // 2) Face lists for the nm edges only (one O(F) pass; fi ascending - the
    //    same ordering the full-table build in S2 produces).
    std::unordered_map<Edge, std::vector<int>, pair_hash> e2f;
    e2f.reserve(nm.size() * 2);
    for (const Edge& e : nm) e2f[e] = {};
    for (std::size_t fi = 0; fi < tri_list.size(); ++fi) {
        const Eigen::Vector3i& t = tri_list[fi];
        for (int k = 0; k < 3; ++k) {
            int a = t[k], b = t[(k + 1) % 3];
            if (a == b) continue;
            auto it = e2f.find(canonicalEdge(a, b));
            if (it != e2f.end()) it->second.push_back(static_cast<int>(fi));
        }
    }

    // 3) Score (S2 scorer replica) + guard + delete.
    std::vector<bool> dead(tri_list.size(), false);
    int deleted = 0;
    for (const Edge& e : nm) {
        std::vector<int> live;   // a previous deletion may have fixed this edge
        for (int fi : e2f[e])
            if (!dead[static_cast<std::size_t>(fi)]) live.push_back(fi);
        if (live.size() <= 2) continue;

        struct Cand { int fi; double score; double area; Eigen::Vector3d n; bool degenerate; };
        std::vector<Cand> cands;
        Eigen::Vector3d ref_n = Eigen::Vector3d::UnitZ();
        double ref_area = -1.0;
        for (int fi : live) {
            const Eigen::Vector3i& t = tri_list[static_cast<std::size_t>(fi)];
            bool degen = (t[0] == t[1] || t[1] == t[2] || t[2] == t[0]);
            Eigen::Vector3d cr =
                degen ? Eigen::Vector3d::Zero()
                      : (verts[static_cast<std::size_t>(t[1])] -
                         verts[static_cast<std::size_t>(t[0])])
                            .cross(verts[static_cast<std::size_t>(t[2])] -
                                   verts[static_cast<std::size_t>(t[0])]);
            double area = 0.5 * cr.norm();
            Eigen::Vector3d n = cr.norm() > kDegenerateCross
                                    ? cr.normalized()
                                    : Eigen::Vector3d::UnitZ();
            if (area > ref_area) {
                ref_area = area;
                ref_n = n;
            }
            cands.push_back(Cand{fi, 0.0, area, n, degen});
        }
        for (Cand& c : cands) c.score = c.n.dot(ref_n);
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& x, const Cand& y) {
                      if (x.score != y.score) return x.score > y.score;
                      if (x.area != y.area) return x.area > y.area;
                      return x.fi < y.fi;
                  });
        for (std::size_t i = 2; i < cands.size(); ++i) {
            const Eigen::Vector3i& t = tri_list[static_cast<std::size_t>(cands[i].fi)];
            if (!cands[i].degenerate) {
                bool ok = true;
                for (int k = 0; k < 3; ++k) {
                    int a = t[k], b = t[(k + 1) % 3];
                    if (a == b) continue;
                    Edge e2 = canonicalEdge(a, b);
                    if (e2 == e) continue;   // the nm edge itself
                    auto it = edge_count.find(e2);
                    const int c = (it == edge_count.end()) ? 0 : it->second;
                    if (c == 2) {
                        ok = false;          // would drop to 1 - opening
                        break;               // interior surface - refuse
                    }
                    // c == 1: edge belongs to this face alone (rim sliver) -
                    //          it vanishes WITH the face, nothing is opened.
                    // c >= 3: decrements to >= 2 - stays manifold.
                }
                if (!ok) continue;
            }
            dead[static_cast<std::size_t>(cands[i].fi)] = true;
            ++deleted;
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (a == b) continue;
                auto it = edge_count.find(canonicalEdge(a, b));
                if (it != edge_count.end() && it->second > 0) --it->second;
            }
        }
    }
    if (deleted > 0) {
        std::vector<Eigen::Vector3i> kept;
        kept.reserve(tri_list.size());
        for (std::size_t fi = 0; fi < tri_list.size(); ++fi)
            if (!dead[fi]) kept.push_back(tri_list[fi]);
        tri_list.swap(kept);
    }
    return deleted;
}

} // namespace hole_filler_pro
