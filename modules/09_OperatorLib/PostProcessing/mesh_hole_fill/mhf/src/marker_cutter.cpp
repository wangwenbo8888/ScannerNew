#include "hole_filler_pro/marker_cutter.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace hole_filler_pro {
namespace {

double markerRadius(const FillParams& p) {
    return 0.5 * p.marker_diameter * p.marker_radius_scale;
}

constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kPi = 3.14159265358979323846;

// [marker-bay] Order-preserving open-arc slice [from..to] (inclusive, both
// endpoints shared with the neighbouring slices on purpose - pinch points)
// with metadata per detectBoundaries' emit convention (open: m-1 edges).
BoundaryLoop makeOpenSlice(const std::vector<int>& src, std::size_t from, std::size_t to,
                           const Eigen::MatrixXd& V) {
    BoundaryLoop l;
    l.vertex_indices.assign(src.begin() + static_cast<long>(from),
                            src.begin() + static_cast<long>(to) + 1);
    l.is_closed = false;
    double perim = 0.0;
    for (std::size_t k = from; k < to; ++k) {
        const Eigen::Vector3d a = V.row(src[k]);
        const Eigen::Vector3d b = V.row(src[k + 1]);
        perim += (a - b).norm();
    }
    l.perimeter = perim;
    l.centroid = computeCentroid(l.vertex_indices, V);
    l.normal = computeNormalNewell(l.vertex_indices, V, false);
    l.avg_edge_length = l.vertex_indices.size() > 1
                            ? perim / static_cast<double>(l.vertex_indices.size() - 1)
                            : 0.0;
    return l;
}

}  // namespace

bool loadMarkersAsc(const std::string& path, std::vector<MarkerSpec>& markers) {
    markers.clear();
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[marker] cannot open marker file: " << path << "\n";
        return false;
    }
    std::string line;
    int lineno = 0;
    int skipped = 0;
    while (std::getline(ifs, line)) {
        ++lineno;
        // Strip a UTF-8 BOM on the very first line (Windows toolchain export).
        if (lineno == 1 && line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }
        if (!line.empty() && line[0] == '#') continue;
        std::istringstream iss(line);
        double x, y, z, nx, ny, nz;
        MarkerSpec m;
        if (!(iss >> x >> y >> z >> nx >> ny >> nz) ||
            !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
            !std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
            if (line.find_first_not_of(" \t\r\n") != std::string::npos) {
                std::cerr << "[marker] WARNING: line " << lineno
                          << " malformed (want \"x y z nx ny nz\") - skipped\n";
                ++skipped;
            }
            continue;
        }
        m.center = Eigen::Vector3d(x, y, z);
        m.normal = Eigen::Vector3d(nx, ny, nz);
        if (m.normal.norm() < 1e-12) {
            std::cerr << "[marker] WARNING: line " << lineno
                      << " zero normal - skipped\n";
            ++skipped;
            continue;
        }
        m.normal.normalize();
        markers.push_back(m);
    }
    if (skipped > 0)
        std::cerr << "[marker] " << skipped << " malformed line(s) skipped in "
                  << path << "\n";
    if (markers.empty()) {
        std::cerr << "[marker] no valid markers parsed from " << path << "\n";
        return false;
    }
    return true;
}

bool markerNormalValid(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                       const MarkerSpec& m, const FillParams& p) {
    const double r = markerRadius(p);
    Eigen::Vector3d n_sum = Eigen::Vector3d::Zero();
    double area_sum = 0.0;
    bool near_surface = false;
    for (int fi = 0; fi < F.rows(); ++fi) {
        const Eigen::Vector3d a = V.row(F(fi, 0));
        const Eigen::Vector3d b = V.row(F(fi, 1));
        const Eigen::Vector3d c = V.row(F(fi, 2));
        const Eigen::Vector3d ctr = (a + b + c) / 3.0;
        const double d = (ctr - m.center).norm();
        if (p.marker_surface_max_dist > 0.0 && d <= p.marker_surface_max_dist)
            near_surface = true;
        if (d > r) continue;
        const Eigen::Vector3d fn = (b - a).cross(c - a);  // 2*area weighted
        n_sum += fn;
        area_sum += 0.5 * fn.norm();
    }
    if (p.marker_surface_max_dist > 0.0 && !near_surface) return false;
    if (area_sum <= 1e-12) return false;
    if (n_sum.norm() <= 0.25 * area_sum) return false;  // incoherent patch
    return n_sum.normalized().dot(m.normal) >= p.marker_normal_min_dot;
}

MarkerCutResult cutMarkerCylinders(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                   const std::vector<MarkerSpec>& markers,
                                   const FillParams& p) {
    MarkerCutResult res;
    const double r = markerRadius(p);
    const double r2 = r * r;

    // 1) First-marker-wins cylinder ownership per vertex (deterministic).
    std::vector<int> owner(static_cast<std::size_t>(V.rows()), -1);
    for (std::size_t mi = 0; mi < markers.size(); ++mi) {
        const MarkerSpec& m = markers[mi];
        for (int i = 0; i < V.rows(); ++i) {
            if (owner[static_cast<std::size_t>(i)] >= 0) continue;
            const Eigen::Vector3d pos = V.row(i);
            const Eigen::Vector3d d = pos - m.center;
            const double t = d.dot(m.normal);
            if (t < -p.marker_len_back || t > p.marker_len_front) continue;
            if ((d - t * m.normal).squaredNorm() <= r2)
                owner[static_cast<std::size_t>(i)] = static_cast<int>(mi);
        }
    }

    // 2) Face deletion - surgical: only faces owning a cylinder vertex are
    //    even considered; everything outside stays byte-identical.
    std::vector<char> tri_alive(static_cast<std::size_t>(F.rows()), 1);
    std::vector<int> per_marker(markers.size(), 0);
    int removed = 0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        const int a = F(fi, 0), b = F(fi, 1), c = F(fi, 2);
        int om = -1;
        if (owner[static_cast<std::size_t>(a)] >= 0)
            om = owner[static_cast<std::size_t>(a)];
        else if (owner[static_cast<std::size_t>(b)] >= 0)
            om = owner[static_cast<std::size_t>(b)];
        else if (owner[static_cast<std::size_t>(c)] >= 0)
            om = owner[static_cast<std::size_t>(c)];
        if (om < 0) continue;  // no cylinder contact - untouched
        const Eigen::Vector3d fa = V.row(a), fb = V.row(b), fc = V.row(c);
        const Eigen::Vector3d cr = (fb - fa).cross(fc - fa);
        bool die = false;
        if (a == b || b == c || a == c) {
            die = true;  // topologically dead face inside the cylinder
        } else if (cr.norm() < 1e-14) {
            die = true;  // degenerate (zero-area) face: ruling deletes it
        } else if (cr.normalized().dot(markers[static_cast<std::size_t>(om)].normal) >
                   p.marker_normal_min_dot) {
            die = true;  // same-direction face (back faces point against n)
        }
        if (!die) continue;
        tri_alive[static_cast<std::size_t>(fi)] = 0;
        ++per_marker[static_cast<std::size_t>(om)];
        ++removed;
    }

    // Per-marker telemetry (ruling #7).
    for (std::size_t mi = 0; mi < markers.size(); ++mi)
        std::cerr << "[marker] cut #" << mi << " at (" << markers[mi].center.x()
                  << ", " << markers[mi].center.y() << ", " << markers[mi].center.z()
                  << ") faces=" << per_marker[mi] << "\n";

    // 3) Compaction - boundary_trimmer.cpp:134-164 idiom.
    std::vector<char> vert_used(static_cast<std::size_t>(V.rows()), 0);
    int alive_count = 0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        if (!tri_alive[static_cast<std::size_t>(fi)]) continue;
        ++alive_count;
        for (int k = 0; k < 3; ++k)
            vert_used[static_cast<std::size_t>(F(fi, k))] = 1;
    }
    std::vector<int> old_to_new(static_cast<std::size_t>(V.rows()), -1);
    int new_count = 0;
    for (int i = 0; i < V.rows(); ++i)
        if (vert_used[static_cast<std::size_t>(i)])
            old_to_new[static_cast<std::size_t>(i)] = new_count++;
    res.V.resize(new_count, 3);
    for (int i = 0; i < V.rows(); ++i)
        if (old_to_new[static_cast<std::size_t>(i)] >= 0)
            res.V.row(old_to_new[static_cast<std::size_t>(i)]) = V.row(i);
    res.F.resize(alive_count, 3);
    int ri = 0;
    for (int fi = 0; fi < F.rows(); ++fi) {
        if (!tri_alive[static_cast<std::size_t>(fi)]) continue;
        res.F(ri, 0) = old_to_new[static_cast<std::size_t>(F(fi, 0))];
        res.F(ri, 1) = old_to_new[static_cast<std::size_t>(F(fi, 1))];
        res.F(ri, 2) = old_to_new[static_cast<std::size_t>(F(fi, 2))];
        ++ri;
    }
    res.faces_removed = removed;
    res.verts_removed = static_cast<int>(V.rows()) - new_count;
    return res;
}

std::vector<bool> matchMarkerLoops(const std::vector<BoundaryLoop>& loops,
                                   const std::vector<MarkerSpec>& markers,
                                   const FillParams& p,
                                   std::vector<int>* matched_marker) {
    std::vector<bool> flag(loops.size(), false);
    if (matched_marker) matched_marker->assign(loops.size(), -1);
    if (markers.empty()) return flag;
    const double r = markerRadius(p);
    std::vector<char> taken(markers.size(), 0);
    for (std::size_t li = 0; li < loops.size(); ++li) {
        const BoundaryLoop& l = loops[li];
        if (!l.is_closed) continue;
        const int m = static_cast<int>(l.vertex_indices.size());
        if (m < 3 || m > p.marker_match_max_edges) continue;
        if (l.perimeter < 0.25 * kTwoPi * r || l.perimeter > 4.0 * kTwoPi * r)
            continue;
        for (std::size_t mi = 0; mi < markers.size(); ++mi) {
            if (taken[mi]) continue;
            if ((l.centroid - markers[mi].center).norm() > r + p.marker_match_margin)
                continue;
            // l.normal is the Newell normal; its sign is arbitrary.
            if (std::abs(l.normal.dot(markers[mi].normal)) <= p.marker_match_min_dot)
                continue;
            flag[li] = true;
            if (matched_marker) (*matched_marker)[li] = static_cast<int>(mi);
            taken[mi] = 1;
            break;
        }
    }
    return flag;
}

BayHealResult healMarkerBays(std::vector<BoundaryLoop>& loops,
                             const Eigen::MatrixXd& V,
                             const std::vector<MarkerSpec>& markers,
                             const FillParams& p) {
    BayHealResult res;
    if (markers.empty() || loops.empty()) return res;
    const double r = markerRadius(p);
    const double capR = r + p.marker_bay_cap_margin;
    const double capR2 = capR * capR;
    // Axial window: the cut window relaxed by the same margin so tilted rim
    // vertices just outside [-len_back, +len_front] still register (K3-T5).
    const double t_lo = -p.marker_len_back - p.marker_bay_cap_margin;
    const double t_hi = p.marker_len_front + p.marker_bay_cap_margin;
    const double mouth_bound =
        2.0 * capR * std::sin(p.marker_bay_max_mouth_deg * 0.5 * kPi / 180.0);

    // Heal-time trigger view: which markers are already claimed by a closed
    // loop RIGHT NOW. Claimed markers are healthy and never touched (this is
    // what keeps the 1.2-configuration output byte-identical: every rim
    // closes normally there, the stage is a silent no-op).
    std::vector<int> mid;
    std::vector<bool> mex = matchMarkerLoops(loops, markers, p, &mid);
    std::vector<char> claimed(markers.size(), 0);
    for (std::size_t li = 0; li < loops.size(); ++li)
        if (li < mex.size() && mex[li] && mid[li] >= 0)
            claimed[static_cast<std::size_t>(mid[li])] = 1;

    struct Action {
        std::size_t loop_idx;
        BoundaryLoop head, cap, tail;
        bool has_head = false, has_tail = false;
    };
    std::vector<Action> actions;
    std::vector<char> loop_used(loops.size(), 0);   // one carrier per loop (N3)

    for (std::size_t mi = 0; mi < markers.size(); ++mi) {
        if (claimed[mi]) continue;   // G1: healthy rim - silent
        const Eigen::Vector3d& c = markers[mi].center;

        // G10: capture discs of two markers must never overlap (symmetric).
        bool near_other = false;
        for (std::size_t mk = 0; mk < markers.size() && !near_other; ++mk) {
            if (mk == mi) continue;
            if ((c - markers[mk].center).norm() <= 2.0 * capR) near_other = true;
        }
        if (near_other) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R10_marker_proximity");
            continue;
        }

        // Carrier: the unique OPEN arc (v1: closed carriers are refused -
        // S7 chord semantics are order-sensitive there, K3-N8) that is not
        // matched to any marker, not already reserved, and has a vertex in
        // the capture cylinder.
        int carrier = -1;
        int carriers = 0;
        for (std::size_t li = 0; li < loops.size(); ++li) {
            if (loop_used[li]) continue;
            const BoundaryLoop& L = loops[li];
            if (L.is_closed) continue;                       // N8
            if (li < mex.size() && mex[li]) continue;        // someone's rim
            bool hit = false;
            for (int v : L.vertex_indices) {
                const Eigen::Vector3d pv = V.row(v);
                const Eigen::Vector3d d = pv - c;
                const double t = d.dot(markers[mi].normal);
                if (t < t_lo || t > t_hi) continue;
                if ((d - t * markers[mi].normal).squaredNorm() <= capR2) { hit = true; break; }
            }
            if (hit) {
                ++carriers;
                if (carrier < 0) carrier = static_cast<int>(li);
            }
        }
        if (carriers == 0) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R01_no_carrier");
            continue;
        }
        if (carriers > 1) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R02_multi_carrier");
            continue;
        }
        const std::size_t li = static_cast<std::size_t>(carrier);
        const BoundaryLoop& L = loops[li];
        const std::size_t M = L.vertex_indices.size();

        // Radial profile along the arc (arc order), crossings and runs.
        std::vector<char> inside(M, 0);
        for (std::size_t k = 0; k < M; ++k) {
            const Eigen::Vector3d pv = V.row(L.vertex_indices[k]);
            const Eigen::Vector3d d = pv - c;
            const double t = d.dot(markers[mi].normal);
            if (t < t_lo || t > t_hi) continue;
            if ((d - t * markers[mi].normal).squaredNorm() <= capR2) inside[k] = 1;
        }
        int crossings = 0;
        for (std::size_t k = 0; k + 1 < M; ++k)
            if (inside[k] != inside[k + 1]) ++crossings;
        if (crossings != 2) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R05_crossings");
            continue;
        }
        std::size_t i0 = 0, j0 = 0;
        int runs = 0;
        for (std::size_t k = 0; k < M; ) {
            if (inside[k]) {
                std::size_t k2 = k;
                while (k2 + 1 < M && inside[k2 + 1]) ++k2;
                i0 = k; j0 = k2; ++runs;
                k = k2 + 1;
            } else {
                ++k;
            }
        }
        if (runs != 1) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R04_runs");
            continue;
        }
        if (i0 == 0 || j0 == M - 1) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R14_run_touches_endpoint");
            continue;
        }
        const std::size_t subN = j0 - i0 + 1;
        if (subN < 3) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R09_too_small");
            continue;
        }
        if (subN > static_cast<std::size_t>(p.marker_match_max_edges)) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R08_too_large");
            continue;
        }

        // Sub-path, perimeter, chord.
        std::vector<int> sub(L.vertex_indices.begin() + static_cast<long>(i0),
                             L.vertex_indices.begin() + static_cast<long>(j0) + 1);
        {   // R03 defensive: duplicate vertex ids would break everything downstream.
            std::vector<int> srt = sub;
            std::sort(srt.begin(), srt.end());
            bool dup = false;
            for (std::size_t k = 1; k < srt.size(); ++k)
                if (srt[k] == srt[k - 1]) { dup = true; break; }
            if (dup) {
                res.refused.push_back("#" + std::to_string(mi) + " reason=R03_duplicate");
                continue;
            }
        }
        double subPerim = 0.0;
        for (std::size_t k = 0; k + 1 < sub.size(); ++k) {
            const Eigen::Vector3d a = V.row(sub[k]);
            const Eigen::Vector3d b = V.row(sub[k + 1]);
            subPerim += (a - b).norm();
        }
        const Eigen::Vector3d px = V.row(sub.front());
        const Eigen::Vector3d py = V.row(sub.back());
        const double chord = (px - py).norm();
        if (chord > mouth_bound) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R06_mouth_too_wide");
            continue;
        }
        if (subPerim > 4.0 * kTwoPi * r) {
            res.refused.push_back("#" + std::to_string(mi) + " reason=R13_perimeter");
            continue;
        }
        // In-rim share (cheap prefilter; sweep is the primary discriminator).
        {
            std::size_t in_rim = 0;
            const double rim2 = (r + 1.5) * (r + 1.5);
            for (int v : sub) {
                const Eigen::Vector3d pv = V.row(v);
                const Eigen::Vector3d d = pv - c;
                const double t = d.dot(markers[mi].normal);
                if ((d - t * markers[mi].normal).squaredNorm() <= rim2) ++in_rim;
            }
            if (static_cast<double>(in_rim) < 0.6 * static_cast<double>(subN)) {
                res.refused.push_back("#" + std::to_string(mi) + " reason=R07_not_rim_like");
                continue;
            }
        }
        // Angular sweep along the sub path (in order, unwrapped increments).
        {
            const Eigen::Vector3d& n = markers[mi].normal;
            const Eigen::Vector3d a0 =
                (std::abs(n.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
            const Eigen::Vector3d e1 = (a0 - n * n.dot(a0)).normalized();
            const Eigen::Vector3d e2 = n.cross(e1);
            double sweep = 0.0;
            double prev = std::atan2((px - c).dot(e2), (px - c).dot(e1));
            for (std::size_t k = 1; k < sub.size(); ++k) {
                const Eigen::Vector3d pv = V.row(sub[k]);
                const double cur = std::atan2((pv - c).dot(e2), (pv - c).dot(e1));
                double dth = cur - prev;
                if (dth > kPi) dth -= kTwoPi;
                if (dth < -kPi) dth += kTwoPi;
                sweep += std::abs(dth);
                prev = cur;
            }
            if (sweep < p.marker_bay_min_sweep_deg * kPi / 180.0) {
                res.refused.push_back("#" + std::to_string(mi) + " reason=R15_sweep");
                continue;
            }
        }

        // Cap: artificial closed loop, emit-convention metadata.
        BoundaryLoop cap;
        cap.vertex_indices = sub;
        cap.is_closed = true;
        cap.perimeter = subPerim + chord;
        cap.centroid = computeCentroid(cap.vertex_indices, V);
        cap.normal = computeNormalNewell(cap.vertex_indices, V, true);
        cap.avg_edge_length = cap.perimeter / static_cast<double>(subN);
        {   // N2: the EXACT matcher predicate arbitrates - zero drift.
            std::vector<BoundaryLoop> probe;
            probe.push_back(cap);
            std::vector<int> pid;
            std::vector<bool> pm = matchMarkerLoops(probe, markers, p, &pid);
            if (pm.size() != 1 || !pm[0] || pid[0] != static_cast<int>(mi)) {
                res.refused.push_back("#" + std::to_string(mi) + " reason=R11_predicate");
                continue;
            }
        }

        Action act;
        act.loop_idx = li;
        act.cap = cap;
        act.has_head = i0 > 0;                     // R14 guarantees i0 >= 1
        act.has_tail = j0 + 1 < M;                 // R14 guarantees j0 <= M-2
        if (act.has_head) act.head = makeOpenSlice(L.vertex_indices, 0, i0, V);
        if (act.has_tail) act.tail = makeOpenSlice(L.vertex_indices, j0, M - 1, V);
        actions.push_back(act);
        loop_used[li] = 1;
        res.chord_edges.push_back(canonicalEdge(cap.vertex_indices.front(),
                                                cap.vertex_indices.back()));
        ++res.healed;
        claimed[mi] = 1;
        std::cerr << "[marker] bay healed #" << mi << " loop=" << li
                  << " n=" << subN << " chord=" << chord << "mm capR=" << capR << "mm\n";
    }

    // Apply: replace carrier slots in place (descending loop index; one
    // action per loop by construction). Refusals never reach this point.
    if (!actions.empty()) {
        std::sort(actions.begin(), actions.end(),
                  [](const Action& a, const Action& b) { return a.loop_idx > b.loop_idx; });
        std::vector<BoundaryLoop> out;
        out.reserve(loops.size() + 2 * actions.size());
        std::size_t ai = actions.size();
        for (std::size_t li = 0; li < loops.size(); ++li) {
            if (ai > 0 && actions[ai - 1].loop_idx == li) {
                const Action& a = actions[--ai];
                if (a.has_head) out.push_back(a.head);
                out.push_back(a.cap);
                if (a.has_tail) out.push_back(a.tail);
            } else {
                out.push_back(loops[li]);
            }
        }
        loops.swap(out);
        res.modified = true;
    }
    for (const std::string& rf : res.refused)
        std::cerr << "[marker] bay refused " << rf << "\n";
    return res;
}

int markerLostToCrack(const BoundaryLoop& arc, const Eigen::MatrixXd& V,
                      const std::vector<MarkerSpec>& markers,
                      const std::vector<char>& marker_claimed,
                      const FillParams& p) {
    if (markers.empty()) return -1;
    const double rr = markerRadius(p) + p.marker_match_margin;
    for (std::size_t mi = 0; mi < markers.size(); ++mi) {
        if (mi < marker_claimed.size() && marker_claimed[mi]) continue;
        for (int v : arc.vertex_indices) {
            const Eigen::Vector3d pv = V.row(v);
            if ((pv - markers[mi].center).norm() <= rr)
                return static_cast<int>(mi);
        }
    }
    return -1;
}

}  // namespace hole_filler_pro
