#define _USE_MATH_DEFINES
#include "hole_filler_pro/ear_clipper.h"
#include "hole_filler_pro/utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace hole_filler_pro {
namespace {

constexpr double kEps = 1e-12;

// 2D cross product (b - a) x (c - a).
inline double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

// Ear blocker test for CCW ear (a=prev, b=cur, c=next). A ring vertex t blocks
// the ear iff it lies STRICTLY inside the ear, or on the INTERIOR of the base
// edge (a,c) (a T-junction would make the clipped triangles overlap). Points
// merely touching the two rim edges (a,b)/(b,c) of the ear no longer reject
// it - that false rejection used to wedge the clipper on collinear chains.
inline bool blocksEar(const Eigen::Vector2d& t,
                      const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                      const Eigen::Vector2d& c) {
    double d1 = cross2(a, b, t);
    double d2 = cross2(b, c, t);
    double d3 = cross2(c, a, t);
    bool strictlyInside = (d1 > kEps) && (d2 > kEps) && (d3 > kEps);
    bool onBaseInterior = (std::abs(d3) <= kEps) && (d1 > kEps) && (d2 > kEps);
    return strictlyInside || onBaseInterior;
}

struct ClipOutcome {
    std::vector<Eigen::Vector3i> tris;
    bool clean = false;   // polygon fully consumed (no guard exhaustion)
};

// One ear-clipping attempt over a CCW sequence of local indices given 2D
// projections. Fixes vs the old implementation:
//  1) collinear vertex: clip it WITH its (flat) triangle instead of silently
//     dropping it - dropping left the vertex's two rim edges uncovered and
//     tore open seams in the patch;
//  2) stuck/aborted clip returns clean=false so the caller can retry with a
//     rotated projection basis; the fan fallback is applied by the caller for
//     ANY leftover polygon (remaining > 3), not only when zero triangles were
//     produced.
ClipOutcome clipPolygonCCW(const std::vector<Eigen::Vector2d>& proj,
                           const std::vector<int>& seq,
                           const std::function<bool(int, int)>& baseBlocked) {
    ClipOutcome out;
    int m = static_cast<int>(seq.size());
    if (m < 3) return out;

    std::vector<int> prev(m), next(m);
    for (int i = 0; i < m; ++i) { prev[i] = (i - 1 + m) % m; next[i] = (i + 1) % m; }

    int remaining = m;
    int cur = 0;
    int sinceClip = 0;
    long long guard = 0;
    long long maxGuard = 4LL * m + 16;

    while (remaining > 3 && guard++ < maxGuard) {
        int p = prev[cur];
        int q = next[cur];
        const Eigen::Vector2d& pa = proj[seq[p]];
        const Eigen::Vector2d& ca = proj[seq[cur]];
        const Eigen::Vector2d& qa = proj[seq[q]];
        double cross = cross2(pa, ca, qa);

        bool clip = false;
        if (std::abs(cross) <= kEps) {
            // Collinear: still clip with a flat triangle so BOTH rim edges of
            // this vertex keep their patch coverage - unless the flat base
            // edge (p,q) is already occupied in the mesh.
            clip = !baseBlocked(seq[p], seq[q]);
        } else if (cross > 0.0) {
            bool isEar = true;
            for (int t = next[q]; t != p; t = next[t]) {
                if (blocksEar(proj[seq[t]], pa, ca, qa)) { isEar = false; break; }
            }
            if (isEar && baseBlocked(seq[p], seq[q])) isEar = false;
            clip = isEar;
        }

        if (clip) {
            // Convex ears emit a proper triangle; collinear vertices emit a
            // flat (zero 2D area) triangle so their rim edges stay covered.
            out.tris.push_back(Eigen::Vector3i(seq[p], seq[cur], seq[q]));
            next[p] = q; prev[q] = p;
            --remaining;
            cur = q;
            sinceClip = 0;
            continue;
        }
        cur = q;
        if (++sinceClip > remaining) break;   // no ear found in a full sweep
    }

    if (remaining == 3) {
        out.tris.push_back(Eigen::Vector3i(seq[prev[cur]], seq[cur], seq[next[cur]]));
        out.clean = true;
    }
    return out;
}

} // namespace

FillResult initialFillEarClip(const BoundaryLoop& loop, const Eigen::MatrixXd& V,
                              const std::function<bool(int, int)>& base_occupied) {
    FillResult result;
    int n = static_cast<int>(loop.vertex_indices.size());
    if (n < 3) return result;

    // Reject degenerate inputs (S5): duplicate rim vertices. (After S3a the
    // detector never produces duplicates inside one arc, but this function is
    // public and may be called with hand-built loops.)
    {
        std::vector<int> sorted = loop.vertex_indices;
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] == sorted[i - 1]) return result;
    }

    // Copy boundary vertices to local 0..n-1 (same layout as initialFill).
    result.new_vertices.resize(n);
    for (int i = 0; i < n; ++i) {
        result.new_vertices[i] = V.row(loop.vertex_indices[i]);
    }

    // Build 2D bases on the loop's best-fit plane: primary + rotated fallbacks
    // (in-plane rotations), used to retry when the primary projection wedges.
    Eigen::Vector3d nrm = loop.normal;
    if (nrm.norm() < kEps) nrm = Eigen::Vector3d::UnitZ();
    nrm.normalize();
    Eigen::Vector3d refAxis = (std::abs(nrm.x()) < 0.9) ? Eigen::Vector3d::UnitX()
                                                        : Eigen::Vector3d::UnitY();
    Eigen::Vector3d basisU0 = nrm.cross(refAxis).normalized();
    Eigen::Vector3d basisV0 = nrm.cross(basisU0).normalized();
    Eigen::Vector3d origin = loop.centroid;

    const double rotations[] = {0.0, M_PI / 6.0, -M_PI / 6.0, M_PI / 3.0, -M_PI / 3.0};
    bool anyValidProjection = false;
    std::vector<int> firstValidSeq;

    // Forbidden-diagonal predicate (global vertex ids): an ear base that
    // coincides with an occupied mesh edge must not be created (it would push
    // that edge to 3+ triangles). Rim edges of THIS loop can never be a base
    // (a base spans non-adjacent ring vertices), so no rim exemption is needed.
    auto baseBlocked = [&](int lp, int lq) {
        if (!base_occupied) return false;
        return base_occupied(loop.vertex_indices[lp], loop.vertex_indices[lq]);
    };

    for (double ang : rotations) {
        Eigen::Vector3d basisU = basisU0 * std::cos(ang) + basisV0 * std::sin(ang);
        Eigen::Vector3d basisV = -basisU0 * std::sin(ang) + basisV0 * std::cos(ang);

        std::vector<Eigen::Vector2d> proj(n);
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d d = result.new_vertices[i] - origin;
            proj[i] = Eigen::Vector2d(basisU.dot(d), basisV.dot(d));
        }

        std::vector<int> seq(n);
        for (int i = 0; i < n; ++i) seq[i] = i;
        double signedArea = 0.0;
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            signedArea += proj[i].x() * proj[j].y() - proj[j].x() * proj[i].y();
        }
        // Relative degeneracy threshold: a real hole projects to an area on the
        // order of its edge lengths squared; a sliver rim projects to ~0.
        double scale = loop.avg_edge_length * loop.avg_edge_length * n + 1.0;
        if (std::abs(signedArea) < 1e-12 * scale) continue;   // sliver: try next basis
        if (!anyValidProjection) {
            anyValidProjection = true;
            firstValidSeq = seq;
        }
        if (signedArea < 0.0) std::reverse(seq.begin(), seq.end());

        ClipOutcome out = clipPolygonCCW(proj, seq, baseBlocked);
        if (out.clean) {
            result.new_triangles = std::move(out.tris);
            return result;
        }
    }

    // All projections wedged or sliver. Sliver-only -> reject (empty result).
    if (!anyValidProjection) {
        result.new_vertices.clear();
        return result;
    }
    // At least one non-sliver projection but clipping wedged: fan fallback
    // covering ALL rim vertices (keeps every rim edge covered).
    std::vector<int>& seq = firstValidSeq;
    int m = static_cast<int>(seq.size());
    for (int i = 1; i + 1 < m; ++i) {
        result.new_triangles.push_back(Eigen::Vector3i(seq[0], seq[i], seq[i + 1]));
    }
    return result;
}

} // namespace hole_filler_pro
