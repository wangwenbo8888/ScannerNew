#include "hole_filler_pro/surface_guide.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

namespace hole_filler_pro {
namespace {

// Graded monomial order (must match GuideFitResult::coeffs documentation):
// 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3  (lexicographic).
struct Term { int ix, iy; };
const Term kTerms[10] = {{0,0},{1,0},{0,1},{2,0},{1,1},{0,2},{3,0},{2,1},{1,2},{0,3}};

inline double mono(int ix, int iy, double x, double y) {
    double r = 1.0;
    for (int k = 0; k < ix; ++k) r *= x;
    for (int k = 0; k < iy; ++k) r *= y;
    return r;
}

inline double evalPoly(int k, const Eigen::VectorXd& c, double x, double y) {
    double s = 0.0;
    for (int t = 0; t < k; ++t)
        s += c(t) * mono(kTerms[t].ix, kTerms[t].iy, x, y);
    return s;
}

// SVD least squares on rows `idx` of a k-term graded design (coordinates
// already normalized by the caller). false on rank/condition failure
// (sigma_min/sigma_max <= 1e-6). JacobiSVD is deterministic; singular
// values are sorted descending.
bool svdFit(const Eigen::MatrixXd& Xy, const Eigen::VectorXd& H,
            const std::vector<int>& idx, int k,
            Eigen::VectorXd& c_out, double& cond_out) {
    const int N = static_cast<int>(idx.size());
    Eigen::MatrixXd A(N, k);
    Eigen::VectorXd h(N);
    for (int i = 0; i < N; ++i) {
        const double x = Xy(idx[i], 0), y = Xy(idx[i], 1);
        h(i) = H(idx[i]);
        for (int t = 0; t < k; ++t)
            A(i, t) = mono(kTerms[t].ix, kTerms[t].iy, x, y);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double smax = svd.singularValues()(0);
    const double smin = svd.singularValues()(k - 1);
    if (!(smax > 0.0) || !(smin > 0.0) || smin / smax <= 1e-6) return false;
    c_out = svd.solve(h);
    cond_out = smax / smin;
    return true;
}

} // namespace

std::vector<int> collectGuideBand(const BoundaryLoop& loop,
                                  const Eigen::MatrixXd& V,
                                  const Eigen::MatrixXi& F,
                                  const std::unordered_map<int, std::vector<int>>& global_v2t,
                                  const Eigen::Vector3d& ref_normal,
                                  double band_width,
                                  const std::unordered_set<int>* guard_verts,
                                  int max_rings) {
    std::vector<int> band;
    if (loop.vertex_indices.empty()) return band;
    if (!(band_width > 0.0)) return band;
    Eigen::Vector3d n = ref_normal;
    if (!n.allFinite() || !(n.norm() > 1e-9)) return band;
    n.normalize();
    const Eigen::Vector3d o = loop.centroid;   // [ruling] fixed before collection
    // Axial slab half-width from the rim's own extent above the centroid
    // plane (t_rim_max is |t| of the worst rim vertex, so every rim seed is
    // inside its own slab unless clamped down).
    double t_rim_max = 0.0;
    for (int gv : loop.vertex_indices)
        t_rim_max = std::max(t_rim_max, std::abs((Eigen::Vector3d(V.row(gv)) - o).dot(n)));
    const double slab = std::min(3.0, std::max(0.3, t_rim_max + 1.0));
    // Rim coordinates for the radial window (min 3D distance to the rim
    // VERTEX SET - the merged ruling's simplification of T1's 2D polyline
    // distance).
    std::vector<Eigen::Vector3d> rim;
    rim.reserve(loop.vertex_indices.size());
    for (int gv : loop.vertex_indices) rim.push_back(V.row(gv));
    auto rimDist = [&](int gv) {
        const Eigen::Vector3d p = V.row(gv);
        double best = std::numeric_limits<double>::max();
        for (const Eigen::Vector3d& q : rim) best = std::min(best, (p - q).norm());
        return best;
    };
    const double kBandGap = 0.5;   // [ruling] inner gap, mm (internal constant)
    // BFS over face adjacency from the rim. Depth-0 seeds (the rim) always
    // expand and are never collected (rho == 0 <= gap). Every other vertex
    // must pass guard + radial expansion bound + axial slab to be visited;
    // membership additionally requires rho > gap.
    std::unordered_set<int> visited(loop.vertex_indices.begin(),
                                    loop.vertex_indices.end());
    std::queue<std::pair<int, int>> q;   // (vertex, ring depth)
    for (int bv : loop.vertex_indices) q.push({bv, 0});
    while (!q.empty()) {
        auto front = q.front(); q.pop();
        const int v = front.first, depth = front.second;
        if (depth > 0) {
            if (guard_verts != nullptr && guard_verts->count(v) > 0) continue;
            const Eigen::Vector3d p = V.row(v);
            const double rho = rimDist(v);
            if (rho > band_width) continue;                  // outer bound: stop
            if (std::abs((p - o).dot(n)) > slab) continue;   // axial slab: stop
            if (rho > kBandGap) band.push_back(v);           // inner gap: expand-only
        }
        if (depth >= max_rings) continue;
        auto it = global_v2t.find(v);
        if (it == global_v2t.end()) continue;
        for (int fi : it->second)
            for (int k = 0; k < 3; ++k) {
                const int w = static_cast<int>(F(fi, k));
                if (visited.insert(w).second) q.push({w, depth + 1});
            }
    }
    // [rim-fit] 2026-11-19: include the RIM VERTICES as fit data points.
    // Previously they were BFS seeds only (rho==0, never collected) — the
    // guide polynomial was unconstrained at the rim, leaving a systematic
    // height offset (fit residual ~0.5mm) that produced a visible C0-only
    // kink at the patch boundary. Adding them here pins the guide surface
    // to the actual rim heights, eliminating the offset at its source.
    for (int bv : loop.vertex_indices) band.push_back(bv);
    std::sort(band.begin(), band.end());   // determinism (BFS order is irrelevant)
    band.erase(std::unique(band.begin(), band.end()), band.end());  // belt & braces
    return band;
}

GuideFitResult fitRobustCubic(const std::vector<Eigen::Vector3d>& band_pts,
                              const Eigen::Vector3d& origin,
                              const Eigen::Vector3d& u,
                              const Eigen::Vector3d& v,
                              const Eigen::Vector3d& n,
                              double scale) {
    GuideFitResult r;
    r.origin = origin; r.u = u; r.v = v; r.n = n; r.scale = scale;
    if (!(scale > 0.0)) return r;
    const int N = static_cast<int>(band_pts.size());
    if (N < 50) return r;   // both rungs need >= max(50, 10*terms) >= 60

    // Frame coordinates; the design matrix uses (x/scale, y/scale), heights
    // stay in mm (guards are in mm).
    Eigen::MatrixXd Xy(N, 2);
    Eigen::VectorXd H(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d d = band_pts[i] - origin;
        Xy(i, 0) = d.dot(u) / scale;
        Xy(i, 1) = d.dot(v) / scale;
        H(i) = d.dot(n);
    }
    std::vector<int> all(N);
    for (int i = 0; i < N; ++i) all[i] = i;

    const int ladder[2] = {10, 6};   // [ruling] C10 -> Q6 -> give up
    for (int k : ladder) {
        const int min_pts = std::max(50, 10 * k);
        if (N < min_pts) continue;
        // Three fits, two 2-sigma cuts (sigma = inlier RMS of the latest fit).
        std::vector<int> keep = all;
        Eigen::VectorXd c;
        double cond = 0.0;
        bool fit_ok = true;
        for (int round = 0; round < 3; ++round) {
            if (!svdFit(Xy, H, keep, k, c, cond)) { fit_ok = false; break; }
            if (round == 2) break;   // final fit done
            const int M = static_cast<int>(keep.size());
            double s2 = 0.0;
            std::vector<double> res(M);
            for (int i = 0; i < M; ++i) {
                res[i] = H(keep[i]) - evalPoly(k, c, Xy(keep[i], 0), Xy(keep[i], 1));
                s2 += res[i] * res[i];
            }
            const double sigma = std::sqrt(s2 / static_cast<double>(M));
            if (sigma < 1e-10) break;   // [fix] exact fit: no rejection needed
            std::vector<int> next;
            next.reserve(keep.size());
            for (int i = 0; i < M; ++i)
                if (std::abs(res[i]) <= 2.0 * sigma) next.push_back(keep[i]);
            if (static_cast<int>(next.size()) < min_pts) { fit_ok = false; break; }
            keep.swap(next);
        }
        if (!fit_ok) continue;   // rank/condition/too-few -> demote
        // Final-state guards (internal constants, mm domain).
        const int kept = static_cast<int>(keep.size());
        const int rejected = N - kept;
        double s2 = 0.0;
        for (int i = 0; i < kept; ++i) {
            const double res = H(keep[i]) - evalPoly(k, c, Xy(keep[i], 0), Xy(keep[i], 1));
            s2 += res * res;
        }
        const double rms = std::sqrt(s2 / static_cast<double>(kept));
        if (rejected > 0.30 * static_cast<double>(N)) continue;   // rejection rate <= 30%
        if (!(rms <= 0.5)) continue;                              // final RMS <= 0.5 mm
        if (kept < 50) continue;                                  // inliers >= 50
        for (int t = 0; t < 10; ++t) r.coeffs[t] = 0.0;
        for (int t = 0; t < k; ++t) r.coeffs[t] = c(t);
        r.coeff_count = k;
        r.used_pts = kept;
        r.rejected = rejected;
        r.cond = cond;
        r.ok = true;
        return r;
    }
    return r;   // every rung failed
}

GuideFitResult fitGuideSurface(const BoundaryLoop& loop,
                               const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               const std::unordered_map<int, std::vector<int>>& global_v2t,
                               const Eigen::Vector3d& ref_normal,
                               double band_width,
                               const std::unordered_set<int>* guard_verts) {
    GuideFitResult r;
    Eigen::Vector3d n = ref_normal;
    if (!n.allFinite() || !(n.norm() > 1e-9)) return r;
    n.normalize();
    // Deterministic tangent pair: the canonical axis least aligned with n
    // (argmin |a.n|, ties broken by axis order X < Y < Z), projected onto
    // the tangent plane. For a unit n at least one axis has |a.n| <=
    // 1/sqrt(3), so the projection can never vanish.
    const double c[3] = {std::abs(n.x()), std::abs(n.y()), std::abs(n.z())};
    int ax = 0;
    if (c[1] < c[ax]) ax = 1;
    if (c[2] < c[ax]) ax = 2;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    a[ax] = 1.0;
    Eigen::Vector3d u = a - n * a.dot(n);
    if (!(u.norm() > 1e-9)) return r;
    u.normalize();
    const Eigen::Vector3d v = n.cross(u);
    const std::vector<int> band_ids =
        collectGuideBand(loop, V, F, global_v2t, ref_normal, band_width, guard_verts);
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(band_ids.size());
    for (int gv : band_ids) pts.push_back(V.row(gv));   // sorted ids -> deterministic
    return fitRobustCubic(pts, loop.centroid, u, v, n, band_width);
}

double guideHeight(const GuideFitResult& fit, double x, double y) {
    const double xn = x / fit.scale, yn = y / fit.scale;
    double h = 0.0;
    for (int t = 0; t < 10; ++t)
        h += fit.coeffs[t] * mono(kTerms[t].ix, kTerms[t].iy, xn, yn);
    return h;   // coeffs at/after coeff_count are zero; !ok -> all zero -> 0.0
}

Eigen::Vector2d guideGrad(const GuideFitResult& fit, double x, double y) {
    const double xn = x / fit.scale, yn = y / fit.scale;
    double hx = 0.0, hy = 0.0;
    for (int t = 0; t < 10; ++t) {
        const int ix = kTerms[t].ix, iy = kTerms[t].iy;
        if (ix > 0) hx += fit.coeffs[t] * ix * mono(ix - 1, iy, xn, yn);
        if (iy > 0) hy += fit.coeffs[t] * iy * mono(ix, iy - 1, xn, yn);
    }
    // d/dx(raw) = (1/scale) * d/d(xn)
    return Eigen::Vector2d(hx / fit.scale, hy / fit.scale);
}

} // namespace hole_filler_pro
