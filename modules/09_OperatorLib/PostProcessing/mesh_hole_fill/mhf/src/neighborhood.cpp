#include "hole_filler_pro/neighborhood.h"
#include "hole_filler_pro/utils.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>

namespace hole_filler_pro {

NeighborhoodInfo extractNeighborhood(const BoundaryLoop& loop,
                                     const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     int rings,
                                     const std::unordered_map<int, double>& global_mean_edge,
                                     const std::unordered_map<int, std::vector<int>>& global_v2t) {
    NeighborhoodInfo nb;
    if (loop.vertex_indices.empty()) return nb;

    const auto& v2t = global_v2t;   // precomputed once in fillHoles (no per-hole O(F) rebuild)

    // BFS from boundary vertices over vertex adjacency (vertices share a triangle).
    std::unordered_set<int> boundary_set(loop.vertex_indices.begin(),
                                         loop.vertex_indices.end());
    std::unordered_set<int> visited(boundary_set.begin(), boundary_set.end());
    std::queue<std::pair<int, int>> q;   // (vertex, depth)
    for (int bv : loop.vertex_indices) q.push({bv, 0});
    std::unordered_set<int> ring_vert_set;
    std::unordered_set<int> ring_tri_set;

    while (!q.empty()) {
        auto front = q.front(); q.pop();
        int v = front.first;
        int depth = front.second;
        if (depth >= rings) continue;
        auto it = v2t.find(v);
        if (it == v2t.end()) continue;
        for (int fi : it->second) {
            ring_tri_set.insert(fi);
            for (int k = 0; k < 3; ++k) {
                int u = static_cast<int>(F(fi, k));
                if (!visited.count(u)) {
                    visited.insert(u);
                    ring_vert_set.insert(u);
                    q.push({u, depth + 1});
                }
            }
        }
    }

    nb.ring_vertices.assign(ring_vert_set.begin(), ring_vert_set.end());
    nb.ring_triangles.assign(ring_tri_set.begin(), ring_tri_set.end());

    // Local target edge length per vertex from the PRECOMPUTED global map (O(1) per vertex,
    // NOT a per-hole O(F) scan).
    auto meanEdge = [&](int v) -> double {
        auto it = global_mean_edge.find(v);
        if (it != global_mean_edge.end()) return it->second;
        return loop.avg_edge_length;   // fallback for isolated verts
    };
    for (int bv : loop.vertex_indices) nb.target_edge[bv] = meanEdge(bv);
    for (int rv : nb.ring_vertices) nb.target_edge[rv] = meanEdge(rv);

    // Reference normal from triangles adjacent to boundary edges (bounded by v2t, not O(F)).
    std::unordered_set<Edge, pair_hash> bedges;
    int m = static_cast<int>(loop.vertex_indices.size());
    for (int i = 0; i < m; ++i) {
        int j = loop.is_closed ? (i + 1) % m : i + 1;
        if (j >= m) break;   // open arc: no ghost closing edge
        bedges.insert(canonicalEdge(loop.vertex_indices[i], loop.vertex_indices[j]));
    }
    Eigen::Vector3d ref = Eigen::Vector3d::Zero();
    int cnt = 0;
    std::unordered_set<int> done_faces;
    for (int bv : loop.vertex_indices) {
        auto it = v2t.find(bv);
        if (it == v2t.end()) continue;
        for (int fi : it->second) {
            if (!done_faces.insert(fi).second) continue;   // each face once
            bool touches = false;
            for (int k = 0; k < 3; ++k) {
                int a = static_cast<int>(F(fi, k));
                int b = static_cast<int>(F(fi, (k + 1) % 3));
                if (bedges.count(canonicalEdge(a, b))) { touches = true; break; }
            }
            if (!touches) continue;
            Eigen::Vector3d a = V.row(F(fi, 0));
            Eigen::Vector3d b = V.row(F(fi, 1));
            Eigen::Vector3d c = V.row(F(fi, 2));
            Eigen::Vector3d fn = (b - a).cross(c - a);
            if (fn.norm() > 1e-12) { ref += fn.normalized(); ++cnt; }
        }
    }
    if (cnt > 0) ref /= static_cast<double>(cnt);
    nb.ref_normal = ref.norm() > 1e-12 ? ref.normalized() : loop.normal;
    return nb;
}

} // namespace hole_filler_pro
