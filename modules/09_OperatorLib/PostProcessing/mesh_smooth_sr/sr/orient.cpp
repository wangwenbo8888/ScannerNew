#include "orient.hpp"

#include "edge_table.hpp"   // [S1'] EdgeTable/buildEdgeTable
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sr {


namespace {

inline uint64_t ekey(uint32_t a, uint32_t b) {
    return a < b ? ((uint64_t)a << 32) | b : ((uint64_t)b << 32) | a;
}

// Shared 16B edge table entry: a<b normalized; t1/t2 = first two incident
// faces (-1 = none). 3rd+ faces go to the nmExtra side map so the main table
// stays 16B ("16B shared table", plan v4 搂2.2).
struct EdgeEntry {
    uint32_t a, b;
    int32_t t1, t2;
};

OrientStats orientRun(std::vector<uint32_t>& tris, const std::vector<double>& verts,
                      size_t numVerts, size_t numTris, bool direct) {
    OrientStats st;
    const size_t nt = numTris ? numTris : tris.size() / 3;
    const size_t nv = numVerts ? numVerts : verts.size() / 3;

    // ---- (1) per-face normal + area; degenerate faces counted, kept ----
    std::vector<double> fnx(nt, 0.0), fny(nt, 0.0), fnz(nt, 0.0);
    for (size_t t = 0; t < nt; t++) {
        uint32_t a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
        if ((size_t)a >= nv || (size_t)b >= nv || (size_t)c >= nv) { st.degen++; continue; }
        if (a == b || b == c || a == c) { st.degen++; continue; }
        double ax = verts[a * 3], ay = verts[a * 3 + 1], az = verts[a * 3 + 2];
        double ux = verts[b * 3] - ax, uy = verts[b * 3 + 1] - ay, uz = verts[b * 3 + 2] - az;
        double vx = verts[c * 3] - ax, vy = verts[c * 3 + 1] - ay, vz = verts[c * 3 + 2] - az;
        double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
        double l = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (0.5 * l < 1e-300) { st.degen++; continue; }
        fnx[t] = cx / l; fny[t] = cy / l; fnz[t] = cz / l;
    }

    // ---- (2) 16B shared edge table ----
    std::unordered_map<uint64_t, uint32_t> eidx;
    eidx.reserve(nt * 3);
    std::vector<EdgeEntry> edges;
    edges.reserve(nt * 3);
    std::unordered_map<uint32_t, int> nmExtra;   // edgeIdx -> count of 3rd+ faces
    for (size_t t = 0; t < nt; t++) {
        for (int k = 0; k < 3; k++) {
            uint32_t a = tris[t * 3 + (size_t)k], b = tris[t * 3 + ((size_t)k + 1) % 3];
            if (a == b) continue;
            uint64_t k2 = ekey(a, b);
            auto it = eidx.find(k2);
            if (it == eidx.end()) {
                eidx.emplace(k2, (uint32_t)edges.size());
                edges.push_back({a, b, (int32_t)t, -1});
            } else if (edges[it->second].t2 < 0) {
                edges[it->second].t2 = (int32_t)t;
            } else {
                nmExtra[it->second]++;
            }
        }
    }

    // ---- (3) edge groups (open/nm) + duplicate face pairs (counted, not removed) ----
    std::unordered_set<uint64_t> dupPairs;
    for (size_t e = 0; e < edges.size(); e++) {
        int extra = 0;
        auto nx = nmExtra.find((uint32_t)e);
        if (nx != nmExtra.end()) extra = nx->second;
        int cnt = (edges[e].t2 >= 0 ? 2 : 1) + extra;
        if (cnt == 1) { st.open++; continue; }
        if (cnt >= 3) { st.nm++; continue; }
        int32_t t1 = edges[e].t1, t2 = edges[e].t2;
        uint32_t s1[3] = {tris[(size_t)t1 * 3], tris[(size_t)t1 * 3 + 1], tris[(size_t)t1 * 3 + 2]};
        uint32_t s2[3] = {tris[(size_t)t2 * 3], tris[(size_t)t2 * 3 + 1], tris[(size_t)t2 * 3 + 2]};
        std::sort(s1, s1 + 3);
        std::sort(s2, s2 + 3);
        if (s1[0] == s2[0] && s1[1] == s2[1] && s1[2] == s2[2]) {
            uint32_t lo = (uint32_t)(t1 < t2 ? t1 : t2);
            uint32_t hi = (uint32_t)(t1 < t2 ? t2 : t1);
            dupPairs.insert(((uint64_t)lo << 32) | hi);   // 3 shared edges per pair: count once
        }
    }
    st.dup = (long long)dupPairs.size();

    // ---- (4) BFS consistent orientation via manifold edges only ----
    // snapshot original windings for the majority-original domain pass (6.5)
    const std::vector<uint32_t> trisOrig = tris;
    std::vector<int32_t> faceCC(nt, -1);
    std::vector<std::vector<int32_t>> ccFaces;
    for (size_t seed = 0; seed < nt; seed++) {
        if (faceCC[seed] >= 0) continue;
        int32_t ccId = (int32_t)ccFaces.size();
        ccFaces.push_back({});
        std::vector<size_t> stack;
        stack.push_back(seed);
        faceCC[seed] = ccId;
        while (!stack.empty()) {
            size_t f = stack.back();
            stack.pop_back();
            ccFaces[(size_t)ccId].push_back((int32_t)f);
            for (int k = 0; k < 3; k++) {
                uint32_t a = tris[f * 3 + (size_t)k], b = tris[f * 3 + ((size_t)k + 1) % 3];
                if (a == b) continue;
                auto it = eidx.find(ekey(a, b));
                if (it == eidx.end()) continue;
                if (edges[it->second].t2 < 0) continue;       // boundary: no propagation
                if (nmExtra.count(it->second)) continue;      // non-manifold: no crossing
                int g = (edges[it->second].t1 == (int32_t)f) ? edges[it->second].t2
                                                             : edges[it->second].t1;
                if (g == (int32_t)f || faceCC[(size_t)g] >= 0) continue;
                // f traverses a->b; g must traverse b->a. If g also has a->b, flip g.
                bool sameDir = false, hasEdge = false;
                for (int kk = 0; kk < 3; kk++) {
                    uint32_t p = tris[(size_t)g * 3 + (size_t)kk],
                             q = tris[(size_t)g * 3 + ((size_t)kk + 1) % 3];
                    if (p == a && q == b) { sameDir = true; hasEdge = true; break; }
                    if (p == b && q == a) { hasEdge = true; break; }
                }
                if (!hasEdge) continue;
                if (sameDir) {
                    std::swap(tris[(size_t)g * 3 + 1], tris[(size_t)g * 3 + 2]);
                    fnx[(size_t)g] = -fnx[(size_t)g];
                    fny[(size_t)g] = -fny[(size_t)g];
                    fnz[(size_t)g] = -fnz[(size_t)g];
                    st.flipped++;
                }
                faceCC[(size_t)g] = ccId;
                stack.push_back((size_t)g);
            }
        }
    }
    st.cc = (long long)ccFaces.size();

    // ---- (5) closed CCs: signed volume; <0 && direct -> flip whole CC ----
    std::vector<uint8_t> ccClosed(ccFaces.size(), 1);
    for (size_t t = 0; t < nt; t++) {
        size_t cc = (size_t)faceCC[t];
        if (!ccClosed[cc]) continue;
        for (int k = 0; k < 3; k++) {
            uint32_t a = tris[t * 3 + (size_t)k], b = tris[t * 3 + ((size_t)k + 1) % 3];
            if (a == b) { ccClosed[cc] = 0; break; }
            auto it = eidx.find(ekey(a, b));
            if (it == eidx.end()) continue;
            if (edges[it->second].t2 < 0 || nmExtra.count(it->second)) { ccClosed[cc] = 0; break; }
        }
    }
    for (size_t ci = 0; ci < ccFaces.size(); ci++) {
        if (!ccClosed[ci]) continue;
        double vol = 0.0;
        for (int32_t t : ccFaces[ci]) {
            uint32_t a = tris[(size_t)t * 3], b = tris[(size_t)t * 3 + 1], c = tris[(size_t)t * 3 + 2];
            double ax = verts[(size_t)a * 3], ay = verts[(size_t)a * 3 + 1], az = verts[(size_t)a * 3 + 2];
            double bx = verts[(size_t)b * 3], by = verts[(size_t)b * 3 + 1], bz = verts[(size_t)b * 3 + 2];
            double cx = verts[(size_t)c * 3], cy = verts[(size_t)c * 3 + 1], cz = verts[(size_t)c * 3 + 2];
            vol += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) +
                    az * (bx * cy - by * cx)) / 6.0;
        }
        if (vol < 0.0) {
            st.inward++;
            if (direct) {
                for (int32_t t : ccFaces[ci]) {
                    std::swap(tris[(size_t)t * 3 + 1], tris[(size_t)t * 3 + 2]);
                    fnx[(size_t)t] = -fnx[(size_t)t];
                    fny[(size_t)t] = -fny[(size_t)t];
                    fnz[(size_t)t] = -fnz[(size_t)t];
                    st.flipped++;
                }
            }
        }
    }

    // ---- (6) re-scan manifold edges: still same-direction => conflicted ----
    std::vector<uint32_t> conflictedEdges;   // seams for the domain pass (6.5)
    for (size_t e = 0; e < edges.size(); e++) {
        if (edges[e].t2 < 0 || nmExtra.count((uint32_t)e)) continue;
        uint32_t a = edges[e].a, b = edges[e].b;
        int32_t t1 = edges[e].t1, t2 = edges[e].t2;
        int d1 = 0, d2 = 0;
        for (int k = 0; k < 3; k++) {
            uint32_t p = tris[(size_t)t1 * 3 + (size_t)k], q = tris[(size_t)t1 * 3 + ((size_t)k + 1) % 3];
            if (p == a && q == b) d1 = 1;
            else if (p == b && q == a) d1 = -1;
            p = tris[(size_t)t2 * 3 + (size_t)k]; q = tris[(size_t)t2 * 3 + ((size_t)k + 1) % 3];
            if (p == a && q == b) d2 = 1;
            else if (p == b && q == a) d2 = -1;
        }
        if (d1 != 0 && d1 == d2) {
            st.conflicted++;
            conflictedEdges.push_back((uint32_t)e);
        }
    }

    // ---- (6.5) domain decomposition + majority-ORIGINAL-winding reference ----
    // 2026-11-06 bugfix (large-area flipped normals). Root cause: the input
    // arrives with large mutually-opposite winding domains (BPV2 stitch seams
    // + NE global-orientation arbitrariness). BFS makes everything match the
    // SEED face - if the seed sits in the minority camp, the majority gets
    // inverted wholesale (74% -> worse). Odd-cycle seams (conflicted edges)
    // stay unsatisfiable. Fix, two layers:
    //   (a) cut conflicted edges -> orientation DOMAINS (may equal CCs when
    //       even paths bypass the seams);
    //   (b) per domain, trust the ORIGINAL winding of the MAJORITY of its
    //       faces (the scanner-side majority was 74% outward): flip the whole
    //       domain when orient inverted more faces than it kept. Radial vote
    //       (away from bbox center) only breaks exact ties.
    if (direct) {
        std::unordered_set<uint32_t> cut;
        cut.reserve(conflictedEdges.size());
        for (uint32_t e : conflictedEdges) cut.insert(e);

        // re-partition faces: BFS along manifold edges that are NOT cut
        std::vector<int32_t> domain(nt, -1);
        long long nDomains = 0;
        for (size_t seed = 0; seed < nt; seed++) {
            if (domain[seed] >= 0) continue;
            std::vector<size_t> stack;
            stack.push_back(seed);
            domain[seed] = (int32_t)nDomains;
            while (!stack.empty()) {
                size_t f = stack.back();
                stack.pop_back();
                for (int k = 0; k < 3; k++) {
                    uint32_t a = tris[f * 3 + (size_t)k], b = tris[f * 3 + ((size_t)k + 1) % 3];
                    if (a == b) continue;
                    auto it = eidx.find(ekey(a, b));
                    if (it == eidx.end()) continue;
                    if (edges[it->second].t2 < 0) continue;          // boundary
                    if (nmExtra.count(it->second)) continue;         // non-manifold
                    if (cut.count(it->second)) continue;             // conflicted seam
                    int g = (edges[it->second].t1 == (int32_t)f) ? edges[it->second].t2
                                                                 : edges[it->second].t1;
                    if (g == (int32_t)f || domain[(size_t)g] >= 0) continue;
                    domain[(size_t)g] = (int32_t)nDomains;
                    stack.push_back((size_t)g);
                }
            }
            ++nDomains;
        }

        // bbox center for the radial tiebreak only
        double minX = 1e300, minY = 1e300, minZ = 1e300, maxX = -1e300, maxY = -1e300, maxZ = -1e300;
        for (size_t v = 0; v < nv; v++) {
            double x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        double bcx = (minX + maxX) / 2, bcy = (minY + maxY) / 2, bcz = (minZ + maxZ) / 2;

        // per-domain: closed domains -> signed volume is the absolute authority
        // (fixes the inverted-cube case: volume-fixed CCs must not be
        // majority-flipped back); open domains -> majority of ORIGINAL
        // windings (scanner-side majority was outward), radial breaks ties.
        std::vector<long long> domSame((size_t)nDomains, 0), domChanged((size_t)nDomains, 0);
        std::vector<double> domRadial((size_t)nDomains, 0.0);
        std::vector<double> domVol((size_t)nDomains, 0.0);
        std::vector<long long> domBoundaryEdges((size_t)nDomains, 0);
        for (size_t t = 0; t < nt; t++) {
            size_t d = (size_t)domain[t];
            uint32_t a0 = trisOrig[t * 3], b0 = trisOrig[t * 3 + 1], c0 = trisOrig[t * 3 + 2];
            uint32_t a1 = tris[t * 3], b1 = tris[t * 3 + 1], c1 = tris[t * 3 + 2];
            bool same;
            if (a1 == a0) same = (b1 == b0 && c1 == c0);
            else if (a1 == b0) same = (b1 == c0 && c1 == a0);
            else if (a1 == c0) same = (b1 == a0 && c1 == b0);
            else same = false;
            if (same) domSame[d]++; else domChanged[d]++;
            domVol[d] += (verts[(size_t)a1 * 3] * (verts[(size_t)b1 * 3 + 1] * verts[(size_t)c1 * 3 + 2] -
                          verts[(size_t)b1 * 3 + 2] * verts[(size_t)c1 * 3 + 1]) +
                         verts[(size_t)a1 * 3 + 1] * (verts[(size_t)b1 * 3 + 2] * verts[(size_t)c1 * 3] -
                          verts[(size_t)b1 * 3] * verts[(size_t)c1 * 3 + 2]) +
                         verts[(size_t)a1 * 3 + 2] * (verts[(size_t)b1 * 3] * verts[(size_t)c1 * 3 + 1] -
                          verts[(size_t)b1 * 3 + 1] * verts[(size_t)c1 * 3])) / 6.0;
            for (int k = 0; k < 3; k++) {
                uint32_t a = tris[t * 3 + (size_t)k], b = tris[t * 3 + ((size_t)k + 1) % 3];
                if (a == b) { domBoundaryEdges[d]++; continue; }
                auto it = eidx.find(ekey(a, b));
                if (it == eidx.end()) continue;
                if (edges[it->second].t2 < 0 || nmExtra.count(it->second) || cut.count(it->second))
                    domBoundaryEdges[d]++;
            }
            if (!(fnx[t] == 0.0 && fny[t] == 0.0 && fnz[t] == 0.0)) {
                double fx = (verts[(size_t)a1 * 3] + verts[(size_t)b1 * 3] + verts[(size_t)c1 * 3]) / 3 - bcx;
                double fy = (verts[(size_t)a1 * 3 + 1] + verts[(size_t)b1 * 3 + 1] + verts[(size_t)c1 * 3 + 1]) / 3 - bcy;
                double fz = (verts[(size_t)a1 * 3 + 2] + verts[(size_t)b1 * 3 + 2] + verts[(size_t)c1 * 3 + 2]) / 3 - bcz;
                domRadial[d] += fnx[t] * fx + fny[t] * fy + fnz[t] * fz;
            }
        }
        long long domainFlips = 0;
        for (long long d = 0; d < nDomains; d++) {
            bool flip;
            if (domBoundaryEdges[(size_t)d] == 0) {
                flip = (domVol[(size_t)d] < 0.0);   // closed: volume authority
            } else if (domSame[d] != domChanged[d]) {
                flip = (domChanged[d] > domSame[d]); // open: majority-original
            } else {
                flip = (domRadial[(size_t)d] < 0.0); // tie: radial
            }
            if (flip) {
                for (size_t t = 0; t < nt; t++) {
                    if (domain[t] != (int32_t)d) continue;
                    std::swap(tris[t * 3 + 1], tris[t * 3 + 2]);
                    fnx[t] = -fnx[t]; fny[t] = -fny[t]; fnz[t] = -fnz[t];
                    st.flipped++;
                    domainFlips++;
                }
            }
        }
        std::fprintf(stderr, "[stats] orient domains=%lld domain_flips=%lld (majority-original)\n",
                     nDomains, domainFlips);
    }

    // ---- (7) glued vertices: shared across CCs with opposing normals ----
    if (nv > 0 && nt > 0) {
        std::vector<size_t> voff(nv + 1, 0);
        for (size_t t = 0; t < nt; t++)
            for (int k = 0; k < 3; k++) voff[(size_t)tris[t * 3 + (size_t)k] + 1]++;
        for (size_t v = 0; v < nv; v++) voff[v + 1] += voff[v];
        std::vector<uint32_t> vlist(voff[nv]);
        {
            std::vector<size_t> cur(voff.begin(), voff.end() - 1);
            for (size_t t = 0; t < nt; t++)
                for (int k = 0; k < 3; k++) vlist[cur[(size_t)tris[t * 3 + (size_t)k]]++] = (uint32_t)t;
        }
        for (size_t v = 0; v < nv; v++) {
            bool glued = false;
            for (size_t i = voff[v]; i < voff[v + 1] && !glued; i++) {
                int32_t f1 = (int32_t)vlist[i];
                for (size_t j = i + 1; j < voff[v + 1]; j++) {
                    int32_t f2 = (int32_t)vlist[j];
                    if (faceCC[(size_t)f1] == faceCC[(size_t)f2]) continue;
                    if (fnx[(size_t)f1] * fnx[(size_t)f2] + fny[(size_t)f1] * fny[(size_t)f2] +
                        fnz[(size_t)f1] * fnz[(size_t)f2] < 0.0) { glued = true; break; }
                }
            }
            if (glued) st.glued++;
        }
    }

    // ---- (8)(9) stats line (stderr, chain ledger) ----
    std::fprintf(stderr,
                 "[stats] orient flipped=%lld cc=%lld conflicted=%lld nm=%lld "
                 "open=%lld degen=%lld dup=%lld inward=%lld glued=%lld\n",
                 st.flipped, st.cc, st.conflicted, st.nm, st.open, st.degen, st.dup,
                 st.inward, st.glued);
    return st;
}

}  // namespace

// truth-pinned propagation core (shared by the sidecar and embedded variants)
// [S1'] et: optional prebuilt shared edge table (flip-replica convention: no
// self-skip, nmExtra recorded). nullptr -> local fallback build (legacy path).
static long long propagateFromPinned(Mesh& mesh, std::vector<char>& pinned,
                                     const EdgeTable* et = nullptr);

OrientStats orientMesh(Mesh& mesh, bool direct) {
    return orientRun(mesh.triangles, mesh.verts, mesh.numVerts, mesh.numTris, direct);
}

long long orientByReference(Mesh& mesh, const std::string& nrmPath, EdgeTable* et) {
    std::ifstream nf(nrmPath, std::ios::binary);
    if (!nf.is_open()) {
        std::fprintf(stderr, "[stats] orient ref=load_fail %s\n", nrmPath.c_str());
        return -1;
    }
    char magic[6] = {0};
    nf.read(magic, 5);
    if (std::strcmp(magic, "NRM1\n") != 0) {
        std::fprintf(stderr, "[stats] orient ref=bad_magic %s\n", nrmPath.c_str());
        return -1;
    }
    uint32_t cnt = 0;
    nf.read(reinterpret_cast<char*>(&cnt), 4);
    if (cnt == 0) return -1;
    // key = bit-exact combine of the three float32 coordinate words; SR's
    // verts are doubles recovered from float32 STL, so casting back to float
    // reproduces the sidecar bits exactly.
    std::unordered_map<uint64_t, uint32_t> lookup;
    lookup.reserve((size_t)cnt * 2);
    std::vector<float> refN((size_t)cnt * 3);
    std::vector<float> buf(6);
    for (uint32_t i = 0; i < cnt; ++i) {
        nf.read(reinterpret_cast<char*>(buf.data()), 6 * sizeof(float));
        if (!nf) break;
        int32_t ix, iy, iz;
        std::memcpy(&ix, &buf[0], 4);
        std::memcpy(&iy, &buf[1], 4);
        std::memcpy(&iz, &buf[2], 4);
        uint64_t key = (uint64_t)(uint32_t)ix * 0x9E3779B97F4A7C15ULL ^
                       (uint64_t)(uint32_t)iy * 0xC2B2AE3D27D4EB4FULL ^
                       (uint64_t)(uint32_t)iz;
        lookup[key] = i;
        refN[(size_t)i * 3] = buf[3];
        refN[(size_t)i * 3 + 1] = buf[4];
        refN[(size_t)i * 3 + 2] = buf[5];
    }

    // ---- extended reference: vertices without an exact sidecar entry (fill
    // patches, eps-copies) borrow the sidecar normal of the NEAREST covered
    // mesh vertex (spatial hash, expanding ring). Geometry-based, order-free,
    // and immune to the twisted-seam paths that mislead pure propagation
    // (2026-11-06 AA-patch incident: patch was reachable but aligned through
    // an odd-cycle seam -> wrong; nearest-covered borrowing pins it directly).
    std::vector<int> vref(mesh.numVerts, -1);   // mesh vertex -> sidecar idx
    long long extended = 0;
    {
        // exact pass
        std::vector<double> vx((size_t)mesh.numVerts), vy((size_t)mesh.numVerts), vz((size_t)mesh.numVerts);
        for (size_t v = 0; v < (size_t)mesh.numVerts; ++v) {
            float fx = (float)mesh.verts[v * 3], fy = (float)mesh.verts[v * 3 + 1], fz = (float)mesh.verts[v * 3 + 2];
            vx[v] = fx; vy[v] = fy; vz[v] = fz;
            int32_t ix, iy, iz;
            std::memcpy(&ix, &fx, 4);
            std::memcpy(&iy, &fy, 4);
            std::memcpy(&iz, &fz, 4);
            uint64_t key = (uint64_t)(uint32_t)ix * 0x9E3779B97F4A7C15ULL ^
                           (uint64_t)(uint32_t)iy * 0xC2B2AE3D27D4EB4FULL ^
                           (uint64_t)(uint32_t)iz;
            auto it = lookup.find(key);
            if (it != lookup.end()) vref[v] = (int)it->second;
        }
        // spatial hash of covered vertices (8mm cells)
        const double cellSz = 8.0;
        std::unordered_map<int64_t, std::vector<uint32_t>> cg;
        for (size_t v = 0; v < (size_t)mesh.numVerts; ++v) {
            if (vref[v] < 0) continue;
            int64_t cxk = (int64_t)std::floor(vx[v] / cellSz), cyk = (int64_t)std::floor(vy[v] / cellSz), czk = (int64_t)std::floor(vz[v] / cellSz);
            int64_t ck = cxk * 73856093LL ^ cyk * 19349663LL ^ czk * 83492791LL;
            cg[ck].push_back((uint32_t)v);
        }
        // nearest covered vertex per uncovered vertex (expanding rings to 24 cells = 192mm)
        for (size_t v = 0; v < (size_t)mesh.numVerts; ++v) {
            if (vref[v] >= 0) continue;
            int64_t cxk = (int64_t)std::floor(vx[v] / cellSz), cyk = (int64_t)std::floor(vy[v] / cellSz), czk = (int64_t)std::floor(vz[v] / cellSz);
            double bestD = 1e30;
            int bestV = -1;
            for (int r = 0; r <= 24 && bestV < 0; ++r) {
                for (int64_t dx = -r; dx <= r; ++dx)
                for (int64_t dy = -r; dy <= r; ++dy)
                for (int64_t dz = -r; dz <= r; ++dz) {
                    if (std::max(std::llabs(dx), std::max(std::llabs(dy), std::llabs(dz))) != r) continue;   // shell only
                    int64_t ck = (cxk + dx) * 73856093LL ^ (cyk + dy) * 19349663LL ^ (czk + dz) * 83492791LL;
                    auto it = cg.find(ck);
                    if (it == cg.end()) continue;
                    for (uint32_t u : it->second) {
                        double ddx = vx[u] - vx[v], ddy = vy[u] - vy[v], ddz = vz[u] - vz[v];
                        double d = ddx * ddx + ddy * ddy + ddz * ddz;
                        if (d < bestD) { bestD = d; bestV = (int)u; }
                    }
                }
            }
            if (bestV >= 0) {
                vref[v] = vref[(size_t)bestV];   // borrow the neighbor's sidecar entry
                ++extended;
            }
        }
    }

    const size_t nt = mesh.triangles.size() / 3;
    long long flips = 0, covered = 0;
    std::vector<char> pinned(nt, 0);   // truth-pinned by reference (or propagation)
    for (size_t t = 0; t < nt; ++t) {
        uint32_t vi0 = mesh.triangles[t * 3], vi1 = mesh.triangles[t * 3 + 1], vi2 = mesh.triangles[t * 3 + 2];
        int r0 = vref[(size_t)vi0], r1i = vref[(size_t)vi1], r2i = vref[(size_t)vi2];
        if (r0 < 0 || r1i < 0 || r2i < 0) continue;
        const float* n0 = &refN[(size_t)r0 * 3];
        const float* n1 = &refN[(size_t)r1i * 3];
        const float* n2 = &refN[(size_t)r2i * 3];
        // mutual-consistency guard v2: pairwise dots reject legitimate sharp
        // corners (a cube face's diagonal corners have dot -1/3). Instead each
        // vertex reference must agree with the MEAN direction - thin-sheet
        // opposite-side borrowing still fails (minority side has n.mean<0).
        double rx0 = n0[0] + n1[0] + n2[0];
        double ry0 = n0[1] + n1[1] + n2[1];
        double rz0 = n0[2] + n1[2] + n2[2];
        double mlen = std::sqrt(rx0 * rx0 + ry0 * ry0 + rz0 * rz0);
        if (mlen < 1e-12) continue;
        if (n0[0] * rx0 + n0[1] * ry0 + n0[2] * rz0 <= 0.0) continue;
        if (n1[0] * rx0 + n1[1] * ry0 + n1[2] * rz0 <= 0.0) continue;
        if (n2[0] * rx0 + n2[1] * ry0 + n2[2] * rz0 <= 0.0) continue;
        double rx = rx0, ry = ry0, rz = rz0;
        ++covered;
        uint32_t a = mesh.triangles[t * 3], b = mesh.triangles[t * 3 + 1], c = mesh.triangles[t * 3 + 2];
        double ax = mesh.verts[(size_t)a * 3], ay = mesh.verts[(size_t)a * 3 + 1], az = mesh.verts[(size_t)a * 3 + 2];
        double ux = mesh.verts[(size_t)b * 3] - ax, uy = mesh.verts[(size_t)b * 3 + 1] - ay, uz = mesh.verts[(size_t)b * 3 + 2] - az;
        double vx2 = mesh.verts[(size_t)c * 3] - ax, vy2 = mesh.verts[(size_t)c * 3 + 1] - ay, vz2 = mesh.verts[(size_t)c * 3 + 2] - az;
        double nx = uy * vz2 - uz * vy2, ny = uz * vx2 - ux * vz2, nz = ux * vy2 - uy * vx2;
        if (nx * rx + ny * ry + nz * rz < 0.0) {
            std::swap(mesh.triangles[t * 3 + 1], mesh.triangles[t * 3 + 2]);
            ++flips;
            pinned[t] = true;   // truth-pinned by the reference
        } else {
            pinned[t] = true;
        }
    }
    std::fprintf(stderr, "[stats] orient ref_ext_vertices=%lld\n", extended);

    long long propFlips = propagateFromPinned(mesh, pinned, et);   // [S1'] thread table through
    std::fprintf(stderr, "[stats] orient prop_flips=%lld\n", propFlips);
    flips += propFlips;
    std::fprintf(stderr, "[stats] orient ref_faces=%lld/%zu ref_flips=%lld\n",
                 covered, nt, flips);
    return flips;
}

// Embedded variant: the mesh itself carries per-vertex normals (PLY input with
// nx/ny/nz - index-aligned with the vertices, NO coordinate matching, NO
// borrowing). Preferred whenever mesh.vnormals is populated; the sidecar
// variant stays as fallback for STL inputs.
long long orientByReferenceEmbedded(Mesh& mesh, EdgeTable* et) {
    if (mesh.vnormals.size() != mesh.numVerts * 3) {
        std::fprintf(stderr, "[stats] orient embedded=absent\n");
        return 0;
    }
    const size_t nt = mesh.triangles.size() / 3;
    std::vector<char> pinned(nt, 0);
    long long flips = 0, covered = 0;
    for (size_t t = 0; t < nt; ++t) {
        uint32_t a = mesh.triangles[t * 3], b = mesh.triangles[t * 3 + 1], c = mesh.triangles[t * 3 + 2];
        const double* n0 = &mesh.vnormals[(size_t)a * 3];
        const double* n1 = &mesh.vnormals[(size_t)b * 3];
        const double* n2 = &mesh.vnormals[(size_t)c * 3];
        // zero normals (unfilled entries) do not vote
        if ((n0[0] == 0 && n0[1] == 0 && n0[2] == 0) ||
            (n1[0] == 0 && n1[1] == 0 && n1[2] == 0) ||
            (n2[0] == 0 && n2[1] == 0 && n2[2] == 0)) continue;
        // mutual-consistency guard v2 (same as sidecar variant: mean-agreement
        // instead of pairwise - legitimate sharp corners pass, thin-sheet
        // opposite-side borrowing fails)
        double rx = n0[0] + n1[0] + n2[0];
        double ry = n0[1] + n1[1] + n2[1];
        double rz = n0[2] + n1[2] + n2[2];
        double mlen = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (mlen < 1e-12) continue;
        if (n0[0] * rx + n0[1] * ry + n0[2] * rz <= 0.0) continue;
        if (n1[0] * rx + n1[1] * ry + n1[2] * rz <= 0.0) continue;
        if (n2[0] * rx + n2[1] * ry + n2[2] * rz <= 0.0) continue;
        ++covered;
        double ax = mesh.verts[(size_t)a * 3], ay = mesh.verts[(size_t)a * 3 + 1], az = mesh.verts[(size_t)a * 3 + 2];
        double ux = mesh.verts[(size_t)b * 3] - ax, uy = mesh.verts[(size_t)b * 3 + 1] - ay, uz = mesh.verts[(size_t)b * 3 + 2] - az;
        double vx2 = mesh.verts[(size_t)c * 3] - ax, vy2 = mesh.verts[(size_t)c * 3 + 1] - ay, vz2 = mesh.verts[(size_t)c * 3 + 2] - az;
        double nx = uy * vz2 - uz * vy2, ny = uz * vx2 - ux * vz2, nz = ux * vy2 - uy * vx2;
        if (nx * rx + ny * ry + nz * rz < 0.0) {
            std::swap(mesh.triangles[t * 3 + 1], mesh.triangles[t * 3 + 2]);
            ++flips;
        }
        pinned[t] = true;
    }
    long long propFlips = propagateFromPinned(mesh, pinned, et);   // [S1'] thread table through
    std::fprintf(stderr, "[stats] orient embedded_faces=%lld/%zu emb_flips=%lld prop_flips=%lld\n",
                 covered, nt, flips, propFlips);
    return flips + propFlips;
}

// Truth-anchored propagation: uncovered faces inherit orientation from
// adjacent truth-pinned faces across manifold edges (see incident notes at the
// call sites). Returns the number of faces flipped.
static long long propagateFromPinned(Mesh& mesh, std::vector<char>& pinned,
                                     const EdgeTable* et) {
    long long propFlips = 0;
    {
        std::vector<uint32_t>& T = mesh.triangles;
        const size_t nt2 = T.size() / 3;
        // [S1'] shared table (flip-replica: no self-skip + nmExtra). Legacy
        // orient builder skipped a==a and dropped 3rd+ faces silently; for the
        // t1/t2 read surface the shared table exposes the SAME pairs (3rd+
        // faces live in nmExtra, invisible here), and the BFS body guards
        // a==b at the read site -> representation migration, gated by SHA.
        EdgeTable localTab;
        if (!et) localTab = buildEdgeTable(T.data(), nt2);
        const EdgeTable& ET = et ? *et : localTab;
        const auto& eidx2 = ET.eidx;
        const auto& eds = ET.edges;
        std::vector<size_t> queue;
        for (size_t t = 0; t < nt2; ++t)
            if (pinned[t]) queue.push_back(t);
        while (!queue.empty()) {
            size_t f = queue.back();
            queue.pop_back();
            for (int k = 0; k < 3; ++k) {
                uint32_t a = T[f * 3 + (size_t)k], b = T[f * 3 + ((size_t)k + 1) % 3];
                if (a == b) continue;
                auto it = eidx2.find(ekey(a, b));
                if (it == eidx2.end()) continue;
                if (eds[it->second].t2 < 0) continue;    // boundary edge
                int g = (eds[it->second].t1 == (int32_t)f) ? eds[it->second].t2 : eds[it->second].t1;
                if (g == (int32_t)f || pinned[(size_t)g]) continue;
                // f traverses a->b; g must traverse b->a
                bool sameDir = false, hasEdge = false;
                for (int kk = 0; kk < 3; ++kk) {
                    uint32_t p = T[(size_t)g * 3 + (size_t)kk], q = T[(size_t)g * 3 + ((size_t)kk + 1) % 3];
                    if (p == a && q == b) { sameDir = true; hasEdge = true; break; }
                    if (p == b && q == a) { hasEdge = true; break; }
                }
                if (!hasEdge) continue;
                if (sameDir) {
                    std::swap(T[(size_t)g * 3 + 1], T[(size_t)g * 3 + 2]);
                    ++propFlips;
                }
                pinned[(size_t)g] = true;
                queue.push_back((size_t)g);
            }
        }
    }
    return propFlips;
}

OrientStats orientVerify(const Mesh& mesh) {
    std::vector<uint32_t> tris = mesh.triangles;   // analyze on a copy: no flips
    return orientRun(tris, mesh.verts, mesh.numVerts, mesh.numTris, false);
}

} // namespace sr
