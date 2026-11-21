#include "hole_filler_pro/triharmonic_fairing.h"
#include "hole_filler_pro/utils.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {

namespace {

using SparseMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// Cotangent-weight Laplacian on a vertex set with given triangles. Vertices indexed 0..N-1.
SparseMat buildCotangentLaplacian(int N,
                                  const std::vector<Eigen::Vector3d>& verts,
                                  const std::vector<Eigen::Vector3i>& tris) {
    std::vector<std::unordered_map<int, double>> w(static_cast<std::size_t>(N));
    for (const auto& t : tris) {
        int v0 = t[0], v1 = t[1], v2 = t[2];
        if (v0 < 0 || v1 < 0 || v2 < 0) continue;
        if (v0 >= N || v1 >= N || v2 >= N) continue;
        auto cot = [&](int i, int j, int k) -> double {
            Eigen::Vector3d e1 = verts[i] - verts[k];
            Eigen::Vector3d e2 = verts[j] - verts[k];
            double c = e1.dot(e2);
            double s = e1.cross(e2).norm();
            if (s < 1e-12) return 0.0;
            return c / s;
        };
        double w01 = 0.5 * cot(v0, v1, v2);
        double w12 = 0.5 * cot(v1, v2, v0);
        double w20 = 0.5 * cot(v2, v0, v1);
        w[v0][v1] += w01; w[v1][v0] += w01;
        w[v1][v2] += w12; w[v2][v1] += w12;
        w[v2][v0] += w20; w[v0][v2] += w20;
    }
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(N) * 8);
    for (int i = 0; i < N; ++i) {
        double diag = 0.0;
        for (const auto& kv : w[i]) {
            trips.emplace_back(i, kv.first, -kv.second);
            diag += kv.second;
        }
        if (diag < 1e-12) diag = 1.0;   // isolated vertex -> identity row (stable)
        trips.emplace_back(i, i, diag);
    }
    SparseMat L(N, N);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// Solve (M_ff + lambda*I) X_free = -M_fc*X_fixed + lambda*T_free. Returns
// true on success. lambda engages only when free_targets != nullptr AND
// guide_alpha > 0: lambda = guide_alpha * mean(diag(Mff)) of the CURRENT
// order's free block (scale-free across the L^k degradation ladder).
// lambda == 0 leaves triplets, rhs and solver input bit-identical to
// legacy. free_targets is indexed by PATCH-LOCAL id (caller guarantees
// size == patch verts; every free vertex is a patch vertex).
bool solveBlock(const SparseMat& M,
                const std::vector<int>& free_idx,
                const std::vector<int>& fixed_idx,
                const std::vector<Eigen::Vector3d>& verts,
                Eigen::MatrixXd& X_free_out,
                const std::vector<Eigen::Vector3d>* free_targets = nullptr,
                double guide_alpha = 0.0, bool soft_rim = false) {
    int nf = static_cast<int>(free_idx.size());
    int nfx = static_cast<int>(fixed_idx.size());
    if (nf == 0) return false;

    std::unordered_map<int, int> fmap, cmap;
    fmap.reserve(nf);
    cmap.reserve(nfx);
    for (int i = 0; i < nf; ++i) fmap[free_idx[i]] = i;
    for (int i = 0; i < nfx; ++i) cmap[fixed_idx[i]] = i;

    std::vector<Triplet> ff, fc;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SparseMat::InnerIterator it(M, k); it; ++it) {
            int r = static_cast<int>(it.row());
            int c = static_cast<int>(it.col());
            double v = it.value();
            auto rit = fmap.find(r);
            if (rit == fmap.end()) continue;   // only free rows
            auto cit_f = fmap.find(c);
            if (cit_f != fmap.end()) {
                ff.emplace_back(rit->second, cit_f->second, v);
            } else {
                auto cit_c = cmap.find(c);
                if (cit_c != cmap.end()) {
                    fc.emplace_back(rit->second, cit_c->second, v);
                }
            }
        }
    }

    // [guide] soft-term lambda from the CURRENT order's Mff diagonal.
    // Every free row has one: buildCotangentLaplacian emits (i,i,diag) even
    // in the isolated-vertex fallback (diag=1.0). mean_diag MUST be summed
    // BEFORE the lambda triplets are appended - setFromTriplets SUMS
    // duplicate (i,i) entries, so appending first would pollute the mean.
    double lambda = 0.0;
    if (guide_alpha > 0.0 && free_targets != nullptr) {
        double mean_diag = 0.0;
        for (const Triplet& t : ff)
            if (t.row() == t.col()) mean_diag += t.value();
        mean_diag /= static_cast<double>(nf);
        if (std::isfinite(mean_diag) && mean_diag > 0.0) {
            lambda = guide_alpha * mean_diag;
            for (int i = 0; i < nf; ++i) ff.emplace_back(i, i, lambda);
        }
    }

    SparseMat Mff(nf, nf); Mff.setFromTriplets(ff.begin(), ff.end());
    SparseMat Mfc(nf, nfx); Mfc.setFromTriplets(fc.begin(), fc.end());

    Eigen::MatrixXd Xfixed(nfx, 3);
    for (int i = 0; i < nfx; ++i) Xfixed.row(i) = verts[fixed_idx[i]];
    Eigen::MatrixXd rhs = -Mfc * Xfixed;
    if (lambda > 0.0) {   // [guide] soft pull toward the guide-surface targets
        for (int i = 0; i < nf; ++i)
            rhs.row(i) += (lambda * (*free_targets)[free_idx[i]]).transpose();
    }

    Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(Mff);
    if (solver.info() != Eigen::Success) return false;
    X_free_out = solver.solve(rhs);
    return solver.info() == Eigen::Success;
}

} // anonymous namespace

void triharmonicFair(std::vector<Eigen::Vector3d>& new_vertices,
                     std::vector<Eigen::Vector3i>& new_triangles,
                     const NeighborhoodInfo& nb,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     int requested_order,
                     int* achieved_order_out,
                     const std::vector<int>& boundary_global_ids,
                     const std::vector<Eigen::Vector3d>* free_targets,
                     double guide_alpha, bool soft_rim) {
    if (achieved_order_out) *achieved_order_out = 0;   // default: total failure
    if (new_vertices.empty() || new_triangles.empty()) return;
    if (requested_order < 1) requested_order = 1;

    int n_patch = static_cast<int>(new_vertices.size());

    // ---- Build joint graph: patch verts (0..n_patch-1) + ring verts (appended) ----
    // Key domain: GLOBAL vertex ids ONLY (S2 fix). Rim vertices of the patch
    // (global ids from boundary_global_ids) map to their patch-local slot;
    // ring-only globals get fresh ids >= n_patch. Patch-INTERIOR locals never
    // enter this map - the old code inserted all locals 0..n_patch-1 as keys,
    // so any ring global id < n_patch collided with a local key and every
    // rim-touching ring triangle was silently dropped.
    std::unordered_map<int, int> global_to_joint;
    int next_id = n_patch;
    {
        int nb_rim = static_cast<int>(std::min(boundary_global_ids.size(),
                                               new_vertices.size()));
        for (int i = 0; i < nb_rim; ++i) {
            int gv = boundary_global_ids[i];
            if (gv < 0 || gv >= V.rows()) continue;
            global_to_joint[gv] = i;   // rim global id -> its patch-local slot
        }
    }
    // Validate ring vertex ids: only those in [0, V.rows()) participate.
    // (Out-of-range ids would otherwise leave uninitialized jverts entries — Eigen fixed-size
    //  default ctor does not zero-init.)
    for (int gv : nb.ring_vertices) {
        if (gv < 0 || gv >= V.rows()) continue;   // skip invalid
        if (global_to_joint.find(gv) == global_to_joint.end()) {
            global_to_joint[gv] = next_id++;
        }
    }
    int N = next_id;

    std::vector<Eigen::Vector3d> jverts(static_cast<std::size_t>(N));
    for (int i = 0; i < n_patch; ++i) jverts[i] = new_vertices[i];
    for (int gv : nb.ring_vertices) {
        if (gv < 0 || gv >= V.rows()) continue;
        int li = global_to_joint[gv];
        if (li >= n_patch) jverts[li] = V.row(gv);   // ring slot, now guaranteed assigned
    }

    std::vector<Eigen::Vector3i> jtris = new_triangles;
    for (int fi : nb.ring_triangles) {
        if (fi < 0 || fi >= F.rows()) continue;
        int a = static_cast<int>(F(fi, 0));
        int b = static_cast<int>(F(fi, 1));
        int c = static_cast<int>(F(fi, 2));
        auto ia = global_to_joint.find(a);
        auto ib = global_to_joint.find(b);
        auto ic = global_to_joint.find(c);
        if (ia == global_to_joint.end() || ib == global_to_joint.end() ||
            ic == global_to_joint.end()) continue;
        jtris.push_back({ia->second, ib->second, ic->second});
    }

    // ---- Classify fixed vs free from PATCH topology (NOT target_edge) ----
    // Patch topological-boundary edge = shared by < 2 patch triangles.
    std::unordered_map<Edge, int, pair_hash> patch_edge_count;
    for (const auto& t : new_triangles) {
        if (t[0] < 0) continue;
        for (int k = 0; k < 3; ++k) {
            ++patch_edge_count[canonicalEdge(t[k], t[(k + 1) % 3])];
        }
    }
    std::unordered_set<int> patch_boundary_verts;
    for (const auto& kv : patch_edge_count) {
        if (kv.second < 2) {
            patch_boundary_verts.insert(kv.first.first);
            patch_boundary_verts.insert(kv.first.second);
        }
    }

    std::vector<int> free_idx, fixed_idx;
    for (int i = 0; i < n_patch; ++i) {
        if (patch_boundary_verts.count(i)) {
            // [rim-soft] When soft_rim is ON, rim verts join the FREE set
            // with their original positions as soft targets (set by the
            // caller via free_targets). The λI spring in solveBlock pulls
            // them home while bending energy from both sides (patch
            // interior + fixed ring) negotiates tangent continuity across
            // the seam — upgrading the junction from C0 toward C1.
            if (soft_rim && free_targets != nullptr)
                free_idx.push_back(i);
            else
                fixed_idx.push_back(i);
        }
        else free_idx.push_back(i);
    }
    // Ring vertices -> fixed anchors (dedup by joint id).
    std::unordered_set<int> fixed_set(fixed_idx.begin(), fixed_idx.end());
    for (int gv : nb.ring_vertices) {
        auto it = global_to_joint.find(gv);
        if (it != global_to_joint.end() && !fixed_set.count(it->second)) {
            fixed_idx.push_back(it->second);
            fixed_set.insert(it->second);
        }
    }
    if (free_idx.empty()) return;   // nothing to smooth

    // [guide] free_targets domain check: PATCH-LOCAL ids 0..n_patch-1
    // (every free vertex is a patch-interior vertex, so free_idx entries
    // index it directly). Wrong size -> drop with a warning, degrade to
    // the legacy pure-fairing system (never crash, never half-applied).
    if (free_targets != nullptr &&
        free_targets->size() != new_vertices.size()) {
        std::cerr << "[guide] free_targets size " << free_targets->size()
                  << " != patch verts " << new_vertices.size()
                  << " - ignoring guide targets\n";
        free_targets = nullptr;
    }

    // ---- Build Laplacian and try orders 3,2,1 with degradation ----
    SparseMat L = buildCotangentLaplacian(N, jverts, jtris);

    for (int order = requested_order; order >= 1; --order) {
        SparseMat M = L;
        for (int k = 1; k < order; ++k) {
            M = (M * L).eval();   // .eval() avoids aliasing in sparse product
        }
        Eigen::MatrixXd X_free;
        if (solveBlock(M, free_idx, fixed_idx, jverts, X_free,
                       free_targets, guide_alpha)) {            for (int i = 0; i < static_cast<int>(free_idx.size()); ++i) {
                int li = free_idx[i];
                jverts[li] = Eigen::Vector3d(X_free(i, 0), X_free(i, 1), X_free(i, 2));
            }
            // Write patch verts back.
            for (int i = 0; i < n_patch; ++i) new_vertices[i] = jverts[i];
            if (achieved_order_out) *achieved_order_out = order;
            return;
        }
        // else: singular at this order, degrade to order-1 and retry
        std::cerr << "[fair] order " << order << " solve FAILED (patch verts="
                  << n_patch << " free=" << free_idx.size()
                  << " fixed=" << fixed_idx.size() << ") - degrading\n";
    }
    // All orders failed -> leave vertices unchanged (no crash).
    std::cerr << "[fair] ALL ORDERS FAILED - patch vertices left unchanged\n";
}

} // namespace hole_filler_pro
