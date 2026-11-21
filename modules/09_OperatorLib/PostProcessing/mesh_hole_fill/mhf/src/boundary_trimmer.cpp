#include "hole_filler_pro/boundary_trimmer.h"
#include "hole_filler_pro/boundary_detector.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {

TrimResult trimBoundaryRings(const Eigen::MatrixXd& V,
                             const Eigen::MatrixXi& F,
                             const std::vector<BoundaryLoop>& loops,
                             int trim_rings,
                             const std::vector<BoundaryLoop>* protect_loops) {
    TrimResult result;
    if (trim_rings <= 0 || loops.empty()) {
        result.V = V;
        result.F = F;
        result.loops = loops;
        return result;
    }

    // Temporary vertex -> incident triangles map (used only for the trim BFS).
    std::unordered_map<int, std::vector<int>> v2t;
    v2t.reserve(static_cast<std::size_t>(V.rows()));
    for (int fi = 0; fi < F.rows(); ++fi) {
        for (int k = 0; k < 3; ++k) {
            v2t[static_cast<int>(F(fi, k))].push_back(fi);
        }
    }

    // Per-loop: BFS `trim_rings` layers over vertex adjacency, collecting every
    // triangle the BFS touches (same skeleton as extractNeighborhood).
    // OPEN ARCS (is_closed == false) are never trimmed: they are skipped holes
    // whose rims must stay untouched (S3b), and trimming an open path's rim
    // would deform the region we deliberately left unfilled.
    std::vector<std::unordered_set<int>> per_loop_tris(loops.size());
    for (std::size_t li = 0; li < loops.size(); ++li) {
        const auto& loop = loops[li];
        if (!loop.is_closed) continue;   // open arc: no trim band
        std::unordered_set<int> visited(loop.vertex_indices.begin(),
                                        loop.vertex_indices.end());
        std::queue<std::pair<int, int>> q;   // (vertex, depth)
        for (int bv : loop.vertex_indices) q.push({bv, 0});

        while (!q.empty()) {
            auto front = q.front(); q.pop();
            int v = front.first;
            int depth = front.second;
            if (depth >= trim_rings) continue;
            auto it = v2t.find(v);
            if (it == v2t.end()) continue;
            for (int fi : it->second) {
                per_loop_tris[li].insert(fi);
                for (int k = 0; k < 3; ++k) {
                    int u = static_cast<int>(F(fi, k));
                    if (!visited.count(u)) {
                        visited.insert(u);
                        q.push({u, depth + 1});
                    }
                }
            }
        }
    }

    // Protection zones (S8): grow trim_rings + 1 BFS layers around every
    // protected rim and clip the candidate bands against them, so no band can
    // reach into a to-be-skipped opening's rim.
    if (protect_loops && !protect_loops->empty()) {
        std::unordered_set<int> protect_tris;
        for (const auto& loop : *protect_loops) {
            std::unordered_set<int> visited(loop.vertex_indices.begin(),
                                            loop.vertex_indices.end());
            std::queue<std::pair<int, int>> q;
            for (int bv : loop.vertex_indices) q.push({bv, 0});
            while (!q.empty()) {
                auto front = q.front(); q.pop();
                int v = front.first;
                int depth = front.second;
                if (depth >= trim_rings + 1) continue;
                auto it = v2t.find(v);
                if (it == v2t.end()) continue;
                for (int fi : it->second) {
                    protect_tris.insert(fi);
                    for (int k = 0; k < 3; ++k) {
                        int u = static_cast<int>(F(fi, k));
                        if (!visited.count(u)) {
                            visited.insert(u);
                            q.push({u, depth + 1});
                        }
                    }
                }
            }
        }
        if (!protect_tris.empty()) {
            for (auto& tris : per_loop_tris) {
                for (auto it = tris.begin(); it != tris.end();) {
                    if (protect_tris.count(*it)) it = tris.erase(it);
                    else ++it;
                }
            }
        }
    }

    // Merge detection: if any triangle is touched by 2+ loops, those loops would
    // merge when trimmed -> leave the whole group UN-trimmed.
    std::unordered_map<int, std::vector<int>> tri_loops;   // tri -> touching loop indices
    for (std::size_t li = 0; li < loops.size(); ++li) {
        for (int fi : per_loop_tris[li]) {
            tri_loops[fi].push_back(static_cast<int>(li));
        }
    }
    std::vector<bool> skip_trim(loops.size(), false);
    for (const auto& kv : tri_loops) {
        if (kv.second.size() > 1) {
            for (int li : kv.second) skip_trim[li] = true;
        }
    }

    // Union of triangles to delete (only from non-skipped loops).
    std::unordered_set<int> to_delete;
    for (std::size_t li = 0; li < loops.size(); ++li) {
        if (skip_trim[li]) continue;
        for (int fi : per_loop_tris[li]) to_delete.insert(fi);
    }

    if (to_delete.empty()) {
        result.V = V;
        result.F = F;
        result.loops = loops;
        return result;
    }

    // Remove deleted triangles; compact vertices.
    std::vector<bool> tri_alive(static_cast<std::size_t>(F.rows()), true);
    for (int fi : to_delete) tri_alive[fi] = false;

    std::vector<bool> vert_used(static_cast<std::size_t>(V.rows()), false);
    int alive_count = 0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        if (!tri_alive[fi]) continue;
        ++alive_count;
        for (int k = 0; k < 3; ++k) vert_used[F(fi, k)] = true;
    }

    std::vector<int> old_to_new(static_cast<std::size_t>(V.rows()), -1);
    int new_count = 0;
    for (int i = 0; i < V.rows(); ++i) {
        if (vert_used[i]) old_to_new[i] = new_count++;
    }

    result.V.resize(new_count, 3);
    for (int i = 0; i < V.rows(); ++i) {
        if (old_to_new[i] >= 0) result.V.row(old_to_new[i]) = V.row(i);
    }
    result.F.resize(alive_count, 3);
    int ri = 0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        if (!tri_alive[fi]) continue;
        result.F(ri, 0) = old_to_new[F(fi, 0)];
        result.F(ri, 1) = old_to_new[F(fi, 1)];
        result.F(ri, 2) = old_to_new[F(fi, 2)];
        ++ri;
    }

    // Re-detect boundaries on the trimmed mesh.
    result.loops = detectBoundaries(result.V, result.F);
    return result;
}

} // namespace hole_filler_pro
