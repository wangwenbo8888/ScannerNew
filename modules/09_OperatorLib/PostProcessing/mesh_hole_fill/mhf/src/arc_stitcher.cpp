// arc_stitcher.cpp - [P0-3] pinch-split hole recovery (six-agent reviewed
// design; see arc_stitcher.h for the guard contract and PROGRESS.md for the
// two-round review evidence). std + Eigen only.
#include "hole_filler_pro/arc_stitcher.h"
#include "hole_filler_pro/utils.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_set>

namespace hole_filler_pro {

namespace {

// E-calibrated empty-domain height window: an offending face must lie within
// this distance of the ring plane to be a real overlap (a far surface merely
// projecting through the ring column does not block). 0.3mm separates every
// legitimate case (nearest non-rim offender 0.41mm) from real overlaps (~0).
constexpr double kDomainHeightMm = 0.3;
// D warpage gate: rim bows more than this fraction of its average edge off
// the Newell plane -> projection unreliable -> refuse (conservative).
constexpr double kWarpFrac = 0.15;

// Even-odd ray cast on the 2D projection; points within eps of the boundary
// count as OUTSIDE (boundary band - rim-adjacent slivers must not block).
bool pointInPolygon(const std::vector<double>& px, const std::vector<double>& py,
                    double x, double y, double eps) {
    bool inside = false;
    const std::size_t n = px.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const double yi = py[i], yj = py[j], xi = px[i], xj = px[j];
        if (std::fabs(yi - y) <= eps && std::fabs(yj - y) <= eps &&
            std::fabs(xi - xj) <= eps)
            return false;   // horizontal edge through the band -> outside
        if ((yi > y) != (yj > y)) {
            const double t = (y - yi) / (yj - yi);
            const double xint = xi + t * (xj - xi);
            if (std::fabs(xint - x) <= eps) return false;   // on an edge
            if (x < xint) inside = !inside;
        }
    }
    return inside;
}

}  // namespace

ArcStitchResult stitchSharedEndpointArcs(std::vector<BoundaryLoop>& loops,
                                         const Eigen::MatrixXd& V,
                                         const Eigen::MatrixXi& F,
                                         const FillParams& params) {
    ArcStitchResult r;

    // 1) Group OPEN arcs by canonical endpoint pair (deterministic order).
    std::map<std::pair<int, int>, std::vector<int>> groups;
    for (std::size_t i = 0; i < loops.size(); ++i) {
        const BoundaryLoop& l = loops[i];
        if (l.is_closed) continue;
        if (l.vertex_indices.size() < 2) continue;
        const int a = l.vertex_indices.front(), b = l.vertex_indices.back();
        groups[{std::min(a, b), std::max(a, b)}].push_back(static_cast<int>(i));
    }

    // 2) Per-group: guards, merge, in-place replacement at the lower index.
    //    Collect merge actions first, then apply back-to-front so indices of
    //    pending actions stay valid (erasing shifts later indices only).
    struct Action { int keep; int eraseIdx; BoundaryLoop merged; };
    std::vector<Action> actions;
    std::unordered_set<int> consumed;   // loop indices already used by an action

    for (const auto& kv : groups) {
        if (kv.second.size() != 2) { ++r.rejected_k; continue; }
        const int i1 = kv.second[0], i2 = kv.second[1];
        if (consumed.count(i1) || consumed.count(i2)) { ++r.rejected_k; continue; }
        const BoundaryLoop& arc1 = loops[static_cast<std::size_t>(i1)];
        const BoundaryLoop& arc2 = loops[static_cast<std::size_t>(i2)];

        // Direction rule (D1): align arc2 to run B..A against arc1's A..B.
        const int A = arc1.vertex_indices.front();
        const int B = arc1.vertex_indices.back();
        std::vector<int> rim = arc1.vertex_indices;
        std::vector<int> tail = arc2.vertex_indices;
        if (tail.front() == A && tail.back() == B) {
            std::reverse(tail.begin(), tail.end());
        } else if (tail.front() == B && tail.back() == A) {
            // already B..A
        } else {
            ++r.rejected_k;   // endpoints mismatch (cannot happen via grouping)
            continue;
        }
        // rim = A..B then tail's interior (skip tail's B and A endpoints).
        rim.insert(rim.end(), tail.begin() + 1, tail.end() - 1);
        const std::size_t m = rim.size();

        // Defensive duplicate-vertex check (topologically impossible; see
        // header - the conservation invariant keeps arcs internally disjoint).
        {
            std::unordered_set<int> seen;
            bool dup = false;
            for (int v : rim)
                if (!seen.insert(v).second) { dup = true; break; }
            if (dup) { ++r.rejected_k; continue; }
        }

        // Guard 2: own double gate BEFORE merging (vertex count == edge count
        // for a closed loop; perimeters add exactly - no ghost edge).
        const double perim = arc1.perimeter + arc2.perimeter;
        if (static_cast<int>(m) > params.max_hole_edges ||
            (params.max_hole_perimeter > 0.0 && perim > params.max_hole_perimeter)) {
            ++r.rejected_gates;
            continue;
        }

        // Guard 3: empty domain (E-revised) + warpage gate (D).
        // a) Newell normal over the merged rim.
        const Eigen::Vector3d nrm = computeNormalNewell(rim, V, true);
        const double nLen = nrm.norm();
        if (nLen < 1e-12) { ++r.rejected_domain; continue; }
        const Eigen::Vector3d nHat = nrm / nLen;
        // b) Warpage gate: max signed distance of rim vertices off the plane.
        Eigen::Vector3d c0 = Eigen::Vector3d::Zero();
        for (int v : rim) c0 += V.row(v);
        c0 /= static_cast<double>(m);
        double maxD = 0.0;
        for (int v : rim)
            maxD = std::max(maxD, std::fabs((V.row(v) - c0.transpose()).dot(nHat)));
        const double avgEdge = perim / static_cast<double>(m);
        if (maxD > kWarpFrac * avgEdge) { ++r.rejected_domain; continue; }
        // c) Projection basis: dominant axis of the normal.
        const int axis = (std::fabs(nHat.x()) >= std::fabs(nHat.y()) &&
                          std::fabs(nHat.x()) >= std::fabs(nHat.z()))
                             ? 0
                             : (std::fabs(nHat.y()) >= std::fabs(nHat.z()) ? 1 : 2);
        const int u = (axis == 0) ? 1 : 0;
        const int w = (axis == 2) ? 1 : 2;
        std::vector<double> rx(m), ry(m);
        double minU = 1e300, maxU = -1e300, minW = 1e300, maxW = -1e300;
        for (std::size_t k = 0; k < m; ++k) {
            rx[k] = V.row(rim[k])[u];
            ry[k] = V.row(rim[k])[w];
            minU = std::min(minU, rx[k]); maxU = std::max(maxU, rx[k]);
            minW = std::min(minW, ry[k]); maxW = std::max(maxW, ry[k]);
        }
        const double eps = 1e-9 * std::max(avgEdge, 1e-12);
        const double pad = avgEdge;   // bbox inflate: boundary-band tolerance
        // d) Rim vertex set for the rim-incident exemption (rule 1).
        std::unordered_set<int> rimSet(rim.begin(), rim.end());
        // e) Scan triangles: bbox prefilter, then plane distance + point-in-poly.
        bool blocked = false;
        for (int fi = 0; fi < F.rows() && !blocked; ++fi) {
            const int a = F(fi, 0), b = F(fi, 1), c = F(fi, 2);
            // non-degenerate only
            if (a == b || b == c || a == c) continue;
            const double fu = (V.row(a)[u] + V.row(b)[u] + V.row(c)[u]) / 3.0;
            const double fw = (V.row(a)[w] + V.row(b)[w] + V.row(c)[w]) / 3.0;
            if (fu < minU - pad || fu > maxU + pad || fw < minW - pad || fw > maxW + pad)
                continue;
            // rule 1: rim-incident faces are exempt (their centroids legally
            // project inside; E: 5/20 legit groups die without this).
            if (rimSet.count(a) || rimSet.count(b) || rimSet.count(c)) continue;
            // rule 2: must be CLOSE to the ring plane to be a real overlap
            // (far-field curved surfaces merely project through the column).
            const Eigen::Vector3d cen = (V.row(a) + V.row(b) + V.row(c)).transpose() / 3.0;
            const double h = std::fabs((cen - c0).dot(nHat));
            if (h > kDomainHeightMm) continue;
            if (pointInPolygon(rx, ry, fu, fw, eps)) blocked = true;
        }
        if (blocked) { ++r.rejected_domain; continue; }

        // 3) Build the merged closed loop (metadata recomputed, B5).
        BoundaryLoop merged;
        merged.vertex_indices = rim;
        merged.is_closed = true;
        merged.perimeter = perim;
        merged.centroid = computeCentroid(rim, V);
        merged.normal = nHat;
        merged.avg_edge_length = perim / static_cast<double>(m);
        actions.push_back({std::min(i1, i2), std::max(i1, i2), merged});
        consumed.insert(i1);
        consumed.insert(i2);
        ++r.stitched;
    }

    // 4) Apply actions back-to-front (higher indices first) so earlier
    //    pending indices stay valid; the merged loop replaces the LOWER
    //    index in place (D7: minimal downstream order perturbation).
    std::sort(actions.begin(), actions.end(),
              [](const Action& x, const Action& y) { return x.eraseIdx > y.eraseIdx; });
    for (const Action& act : actions) {
        loops[static_cast<std::size_t>(act.eraseIdx)] = BoundaryLoop();  // neutralize
        loops[static_cast<std::size_t>(act.keep)] = act.merged;
    }
    // order-preserving removal of the neutralized slots
    std::vector<BoundaryLoop> kept;
    kept.reserve(loops.size());
    for (const BoundaryLoop& l : loops)
        if (!(l.vertex_indices.empty())) kept.push_back(l);
    loops.swap(kept);
    return r;
}

}  // namespace hole_filler_pro
