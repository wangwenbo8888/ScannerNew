#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geom.hpp"
#include "stl.hpp"
#include "edge_table.hpp"   // [S1'] EdgeR/EdgeTable shared topology

namespace sr {


namespace {

// [S1'] EdgeR/ekeyT moved to edge_table.hpp (bitwise replica); thin local
// alias keeps the rest of this file untouched.
inline uint64_t ekey(uint32_t a, uint32_t b) {
    return ekeyT(a, b);
}

double distSeg2(const double q[3], const double A[3], const double B[3]) {
    double ex = B[0] - A[0], ey = B[1] - A[1], ez = B[2] - A[2];
    double qx = q[0] - A[0], qy = q[1] - A[1], qz = q[2] - A[2];
    double ee = ex * ex + ey * ey + ez * ez;
    double t = ee > 1e-300 ? (qx * ex + qy * ey + qz * ez) / ee : 0.0;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    double dx = qx - t * ex, dy = qy - t * ey, dz = qz - t * ez;
    return dx * dx + dy * dy + dz * dz;
}

void wmEdgeQ(const double A[3], const double B[3], const double NA[3], const double NB[3],
             double Q1[3], double Q2[3]) {
    double dx = B[0] - A[0], dy = B[1] - A[1], dz = B[2] - A[2];
    double L = std::sqrt(dx * dx + dy * dy + dz * dz);
    double hx = dx / L, hy = dy / L, hz = dz / L;
    double k1 = NA[0] * hx + NA[1] * hy + NA[2] * hz;
    double k2 = NB[0] * hx + NB[1] * hy + NB[2] * hz;
    double c = NA[0] * NB[0] + NA[1] * NB[1] + NA[2] * NB[2];
    double den = 4.0 - c * c;
    double alpha = 6.0 * (2.0 * k1 + k2 * c) / den;
    double beta = 6.0 * (2.0 * k2 + k1 * c) / den;
    double s = L / 18.0;
    Q1[0] = A[0] + s * (6.0 * hx - 2.0 * alpha * NA[0] + beta * NB[0]);
    Q1[1] = A[1] + s * (6.0 * hy - 2.0 * alpha * NA[1] + beta * NB[1]);
    Q1[2] = A[2] + s * (6.0 * hz - 2.0 * alpha * NA[2] + beta * NB[2]);
    Q2[0] = B[0] - s * (6.0 * hx + alpha * NA[0] - 2.0 * beta * NB[0]);
    Q2[1] = B[1] - s * (6.0 * hy + alpha * NA[1] - 2.0 * beta * NB[1]);
    Q2[2] = B[2] - s * (6.0 * hz + alpha * NA[2] - 2.0 * beta * NB[2]);
}

double wmEnergy(const double A[3], const double B[3], const double NA[3], const double NB[3]) {
    double dx = B[0] - A[0], dy = B[1] - A[1], dz = B[2] - A[2];
    double L2 = dx * dx + dy * dy + dz * dz;
    if (L2 < 1e-16) return 0.0;
    double Q1[3], Q2[3];
    wmEdgeQ(A, B, NA, NB, Q1, Q2);
    double d1 = distSeg2(Q1, A, B);
    double d2 = distSeg2(Q2, A, B);
    return d1 > d2 ? d1 : d2;
}

void faceNA(const double* P, uint32_t t0, uint32_t t1, uint32_t t2, double* n, double* area) {
    double ax = P[t1 * 3] - P[t0 * 3], ay = P[t1 * 3 + 1] - P[t0 * 3 + 1], az = P[t1 * 3 + 2] - P[t0 * 3 + 2];
    double bx = P[t2 * 3] - P[t0 * 3], by = P[t2 * 3 + 1] - P[t0 * 3 + 1], bz = P[t2 * 3 + 2] - P[t0 * 3 + 2];
    double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    double l = std::sqrt(nx * nx + ny * ny + nz * nz);
    *area = 0.5 * l;
    if (l > 1e-300) {
        n[0] = nx / l;
        n[1] = ny / l;
        n[2] = nz / l;
    } else {
        n[0] = 0;
        n[1] = 0;
        n[2] = 1;
    }
}

bool flatnessOK(const double* P, uint32_t a, uint32_t b, uint32_t c) {
    double ax = P[b * 3] - P[a * 3], ay = P[b * 3 + 1] - P[a * 3 + 1], az = P[b * 3 + 2] - P[a * 3 + 2];
    double bx = P[c * 3] - P[a * 3], by = P[c * 3 + 1] - P[a * 3 + 1], bz = P[c * 3 + 2] - P[a * 3 + 2];
    double cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
    double cross2 = cx * cx + cy * cy + cz * cz;
    if (cross2 < 2.22e-16) return false;
    double e0 = ax * ax + ay * ay + az * az;
    double e1 = bx * bx + by * by + bz * bz;
    double ex = P[c * 3] - P[b * 3], ey = P[c * 3 + 1] - P[b * 3 + 1], ez = P[c * 3 + 2] - P[b * 3 + 2];
    double e2 = ex * ex + ey * ey + ez * ez;
    double mx = std::max(e0, std::max(e1, e2));
    return !(mx / std::sqrt(cross2) > 20.0);
}

}  // namespace

std::vector<uint32_t> optimizeEdges(const std::vector<double>& pos, const std::vector<uint32_t>& triIn,
                                    const std::vector<double>& origVerts,
                                    const std::vector<double>& origNrmIn, long long* flipCount,
                                    double coplanarDeg, int nm_guard, EdgeTable* tab) {
    std::vector<uint32_t> T = triIn;
    const size_t nt = T.size() / 3;
    const size_t nv = pos.size() / 3;
    std::vector<double> vNrm = origNrmIn;
    std::vector<double> vNrm0 = origNrmIn;

    std::vector<double> faceN(nt * 3);
    std::vector<double> faceA(nt);
    #pragma omp parallel for schedule(static)   // S3: per-face, writes disjoint
    for (int tt = 0; tt < (int)nt; tt++) {
        size_t t = (size_t)tt;
        faceNA(pos.data(), T[t * 3], T[t * 3 + 1], T[t * 3 + 2], &faceN[t * 3], &faceA[t]);
    }

    // [S1'] shared-table arm: tab != nullptr -> take ownership (mutations land
    // on the caller's table); nullptr -> local fallback build (legacy path,
    // bitwise identical construction via the shared builder).
    EdgeTable localTab;
    if (!tab) localTab = buildEdgeTable(T.data(), nt);
    EdgeTable& ET = tab ? *tab : localTab;
    std::unordered_map<uint64_t, uint32_t>& eidx = ET.eidx;
    std::vector<EdgeR>& edges = ET.edges;
    // [mustclean nm-guard] 3rd+ face on an edge (orient.cpp nmExtra pattern):
    // EdgeR only records t1/t2, so faces beyond the second are dropped on the
    // floor - and such an edge is NOT manifold (edge_flip would treat its
    // arbitrary t1/t2 as an interior manifold edge and mis-flip it). Counted
    // in nmExtra by the builder; when nm_guard is on the edge is never
    // activated (all THREE activation sites). Flips never change face counts,
    // so the count needs no incremental maintenance.
    std::unordered_map<uint32_t, int>& nmExtra = ET.nmExtra;
    const bool nmGuardOn = (nm_guard != 0) && !nmExtra.empty();
    auto nmBlocked = [&](uint32_t ei) { return nmGuardOn && nmExtra.count(ei) > 0; };
    if (nm_guard != 0)
        std::printf("edgeflip: nm_guard=1 nm_edges=%zu\n", nmExtra.size());
    std::vector<double> energy(edges.size());
    std::vector<uint8_t> active(edges.size(), 0);
    long long activeCnt = 0;   // S3: integer reduction - exact
    #pragma omp parallel for schedule(static) reduction(+:activeCnt)
    for (int ee = 0; ee < (int)edges.size(); ee++) {
        size_t e = (size_t)ee;
        energy[e] = wmEnergy(&pos[edges[e].a * 3], &pos[edges[e].b * 3], &vNrm[edges[e].a * 3],
                             &vNrm[edges[e].b * 3]);
        if (edges[e].t2 >= 0 && !nmBlocked((uint32_t)e)) {
            active[e] = 1;
            activeCnt++;
        }
    }
    std::vector<std::vector<uint32_t>> vertTris(nv);
    for (size_t t = 0; t < nt; t++) {
        vertTris[T[t * 3]].push_back((uint32_t)t);
        vertTris[T[t * 3 + 1]].push_back((uint32_t)t);
        vertTris[T[t * 3 + 2]].push_back((uint32_t)t);
    }
    std::printf("edgeflip: %zu edges (%lld interior active)\n", edges.size(), activeCnt);

    auto rehook = [&](uint32_t a, uint32_t b, int32_t from, int32_t to) {
        auto it = eidx.find(ekey(a, b));
        if (it == eidx.end()) return;
        EdgeR& r = edges[it->second];
        if (r.t1 == from) r.t1 = to;
        else if (r.t2 == from) r.t2 = to;
    };
    auto vremove = [&](std::vector<uint32_t>& vec, uint32_t t) {
        for (size_t i = 0; i < vec.size(); i++)
            if (vec[i] == t) {
                vec[i] = vec.back();
                vec.pop_back();
                return;
            }
    };

    const double COS1 = std::cos(coplanarDeg * 3.14159265358979323846 / 180.0);
    std::unordered_set<uint64_t> banned;
    long long flips = 0;
    std::vector<std::pair<uint32_t, uint32_t>> flipLog;
    const char* flipLogPath = std::getenv("SR_FLIP_LOG");
    int pass = 0;
    while (activeCnt > 0 && pass < 50) {
        pass++;
        std::printf("edgeflip pass %d: active=%lld flips=%lld\n", pass, activeCnt, flips);
        for (size_t e = 0; e < edges.size(); e++) {
            if ((e & 0xFFFFF) == 0) std::fprintf(stderr, "scan e=%zu/%zu t=%.1fs flips=%lld act=%lld\n", e, edges.size(), (double)clock()/CLOCKS_PER_SEC, flips, activeCnt);
            if (!active[e]) continue;
            active[e] = 0;
            activeCnt--;
            if (edges[e].t1 < 0 || edges[e].t2 < 0) continue;
            uint32_t A = edges[e].a, B = edges[e].b;
            int32_t t1 = edges[e].t1, t2 = edges[e].t2;
            uint32_t C = 0, D = 0;
            for (int k = 0; k < 3; k++) {
                uint32_t v = T[t1 * 3 + k];
                if (v != A && v != B) C = v;
            }
            for (int k = 0; k < 3; k++) {
                uint32_t v = T[t2 * 3 + k];
                if (v != A && v != B) D = v;
            }
            if (C == D || C == A || C == B || D == A || D == B) continue;
            if (eidx.count(ekey(C, D))) continue;
            if (banned.count(ekey(C, D))) continue;

            uint32_t nT1[3] = {T[t1 * 3], T[t1 * 3 + 1], T[t1 * 3 + 2]};
            uint32_t nT2[3] = {T[t2 * 3], T[t2 * 3 + 1], T[t2 * 3 + 2]};
            for (int k = 0; k < 3; k++) {
                if (nT1[k] == B) nT1[k] = D;
                if (nT2[k] == A) nT2[k] = C;
            }
            double m1[3], m2[3], a1, a2;
            faceNA(pos.data(), nT1[0], nT1[1], nT1[2], m1, &a1);
            faceNA(pos.data(), nT2[0], nT2[1], nT2[2], m2, &a2);

            double fn = m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2];
            if (fn < 0) continue;
            double bDot = vNrm0[C * 3] * vNrm0[D * 3] + vNrm0[C * 3 + 1] * vNrm0[D * 3 + 1] +
                          vNrm0[C * 3 + 2] * vNrm0[D * 3 + 2];
            if (fn > COS1 && bDot > COS1) continue;

            bool ok = true;
            for (uint32_t v : {A, B, C, D}) {
                if (m1[0] * vNrm[v * 3] + m1[1] * vNrm[v * 3 + 1] + m1[2] * vNrm[v * 3 + 2] < 0 ||
                    m2[0] * vNrm[v * 3] + m2[1] * vNrm[v * 3 + 1] + m2[2] * vNrm[v * 3 + 2] < 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            double c1[3], c2[3];
            c1[0] = m1[1] * m2[2] - m1[2] * m2[1];
            c1[1] = m1[2] * m2[0] - m1[0] * m2[2];
            c1[2] = m1[0] * m2[1] - m1[1] * m2[0];
            c2[0] = vNrm[A * 3 + 1] * vNrm[B * 3 + 2] - vNrm[A * 3 + 2] * vNrm[B * 3 + 1];
            c2[1] = vNrm[A * 3 + 2] * vNrm[B * 3] - vNrm[A * 3] * vNrm[B * 3 + 2];
            c2[2] = vNrm[A * 3] * vNrm[B * 3 + 1] - vNrm[A * 3 + 1] * vNrm[B * 3];
            if (c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2] < 0) continue;

            if (!flatnessOK(pos.data(), nT1[0], nT1[1], nT1[2])) continue;
            if (!flatnessOK(pos.data(), nT2[0], nT2[1], nT2[2])) continue;

            uint32_t oldT1[3] = {T[t1 * 3], T[t1 * 3 + 1], T[t1 * 3 + 2]};
            uint32_t oldT2[3] = {T[t2 * 3], T[t2 * 3 + 1], T[t2 * 3 + 2]};
            T[t1 * 3] = nT1[0]; T[t1 * 3 + 1] = nT1[1]; T[t1 * 3 + 2] = nT1[2];
            T[t2 * 3] = nT2[0]; T[t2 * 3 + 1] = nT2[1]; T[t2 * 3 + 2] = nT2[2];

            double pN[4][3];
            uint32_t quad4[4] = {A, B, C, D};
            bool pNok = true;
            for (int q = 0; q < 4; q++) {
                uint32_t v = quad4[q];
                double sx = 0, sy = 0, sz = 0;
                for (uint32_t t : vertTris[v]) {
                    if ((int32_t)t == t1 || (int32_t)t == t2) continue;
                    sx += faceN[t * 3] * faceA[t];
                    sy += faceN[t * 3 + 1] * faceA[t];
                    sz += faceN[t * 3 + 2] * faceA[t];
                }
                if (nT1[0] == v || nT1[1] == v || nT1[2] == v) {
                    sx += m1[0] * a1;
                    sy += m1[1] * a1;
                    sz += m1[2] * a1;
                }
                if (nT2[0] == v || nT2[1] == v || nT2[2] == v) {
                    sx += m2[0] * a2;
                    sy += m2[1] * a2;
                    sz += m2[2] * a2;
                }
                double l = std::sqrt(sx * sx + sy * sy + sz * sz);
                if (l > 1e-300) {
                    pN[q][0] = sx / l;
                    pN[q][1] = sy / l;
                    pN[q][2] = sz / l;
                } else {
                    pN[q][0] = vNrm[v * 3];
                    pN[q][1] = vNrm[v * 3 + 1];
                    pN[q][2] = vNrm[v * 3 + 2];
                }
            }
            auto qidx = [&](uint32_t v) -> int {
                for (int q = 0; q < 4; q++)
                    if (quad4[q] == v) return q;
                return -1;
            };
            auto nrmOf = [&](uint32_t v, double* out) {
                int q = qidx(v);
                if (q >= 0) {
                    out[0] = pN[q][0];
                    out[1] = pN[q][1];
                    out[2] = pN[q][2];
                } else {
                    out[0] = vNrm[v * 3];
                    out[1] = vNrm[v * 3 + 1];
                    out[2] = vNrm[v * 3 + 2];
                }
            };

            double eCD = wmEnergy(&pos[C * 3], &pos[D * 3], pN[qidx(C)], pN[qidx(D)]);
            std::vector<std::pair<uint32_t, uint32_t>> touched;
            for (int k = 0; k < 3; k++) {
                touched.push_back({nT1[k], nT1[(k + 1) % 3]});
                touched.push_back({nT2[k], nT2[(k + 1) % 3]});
            }
            for (uint32_t v : quad4) {
                for (uint32_t t : vertTris[v]) {
                    if ((int32_t)t == t1 || (int32_t)t == t2) continue;
                    for (int k = 0; k < 3; k++)
                        touched.push_back({T[t * 3 + k], T[t * 3 + (k + 1) % 3]});
                }
            }
            std::sort(touched.begin(), touched.end());
            touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

            double gain = energy[e] - eCD;
            std::vector<std::pair<uint32_t, double>> energyUpdates;
            for (auto& pr : touched) {
                uint64_t k2 = ekey(pr.first, pr.second);
                auto it = eidx.find(k2);
                if (it == eidx.end() || it->second == e) continue;
                uint32_t ei2 = it->second;
                double na[3], nb[3];
                nrmOf(pr.first, na);
                nrmOf(pr.second, nb);
                double eAft = wmEnergy(&pos[pr.first * 3], &pos[pr.second * 3], na, nb);
                gain += energy[ei2] - eAft;
                energyUpdates.push_back({ei2, eAft});
            }
            if (gain <= 0) {
                T[t1 * 3] = oldT1[0]; T[t1 * 3 + 1] = oldT1[1]; T[t1 * 3 + 2] = oldT1[2];
                T[t2 * 3] = oldT2[0]; T[t2 * 3 + 1] = oldT2[1]; T[t2 * 3 + 2] = oldT2[2];
                continue;
            }

            faceN[t1 * 3] = m1[0]; faceN[t1 * 3 + 1] = m1[1]; faceN[t1 * 3 + 2] = m1[2];
            faceA[t1] = a1;
            faceN[t2 * 3] = m2[0]; faceN[t2 * 3 + 1] = m2[1]; faceN[t2 * 3 + 2] = m2[2];
            faceA[t2] = a2;
            for (int q = 0; q < 4; q++) {
                vNrm[quad4[q] * 3] = pN[q][0];
                vNrm[quad4[q] * 3 + 1] = pN[q][1];
                vNrm[quad4[q] * 3 + 2] = pN[q][2];
            }
            vremove(vertTris[B], t1);
            vertTris[D].push_back(t1);
            vremove(vertTris[A], t2);
            vertTris[C].push_back(t2);
            rehook(A, D, t2, t1);
            rehook(B, C, t1, t2);
            for (auto& up : energyUpdates) energy[up.first] = up.second;

            eidx.erase(ekey(A, B));
            banned.insert(ekey(A, B));
            uint32_t newEi = (uint32_t)edges.size();
            edges.push_back({C, D, t1, t2});
            energy.push_back(eCD);
            active.push_back(0);
            eidx[ekey(C, D)] = newEi;

            for (int k = 0; k < 3; k++) {
                uint32_t e1v[2];
                e1v[0] = T[t1 * 3 + k];
                e1v[1] = T[t1 * 3 + (k + 1) % 3];
                uint32_t e2v[2];
                e2v[0] = T[t2 * 3 + k];
                e2v[1] = T[t2 * 3 + (k + 1) % 3];
                for (int ei = 0; ei < 2; ei++) {
                    const uint32_t* ev = (ei == 0) ? e1v : e2v;
                    if ((ev[0] == C && ev[1] == D) || (ev[0] == D && ev[1] == C)) continue;
                    uint64_t bk = ekey(ev[0], ev[1]);
                    auto it = eidx.find(bk);
                    if (it == eidx.end()) continue;
                    EdgeR& be = edges[it->second];
                    if (be.t2 >= 0 && !nmBlocked(it->second) && !active[it->second]) {
                        active[it->second] = 1;
                        activeCnt++;
                    }
                    if (be.t2 < 0) continue;
                    int32_t opp = (be.t1 == t1 || be.t1 == t2) ? be.t2 : be.t1;
                    if (opp == t1 || opp == t2) continue;
                    for (int kk = 0; kk < 3; kk++) {
                        uint32_t u1 = T[opp * 3 + kk], u2 = T[opp * 3 + (kk + 1) % 3];
                        auto it3 = eidx.find(ekey(u1, u2));
                        if (it3 == eidx.end()) continue;
                        if (edges[it3->second].t2 >= 0 && !nmBlocked(it3->second) &&
                            !active[it3->second]) {
                            active[it3->second] = 1;
                            activeCnt++;
                        }
                    }
                }
            }
            flips++;
            flipLog.push_back({C, D});
        }
    }
    if (flipLogPath) {
        FILE* lf = std::fopen(flipLogPath, "w");
        if (lf) {
            for (auto& p : flipLog) std::fprintf(lf, "%u %u\n", p.first, p.second);
            std::fclose(lf);
        }
    }
    std::printf("edgeflip: %lld flips in %d passes\n", flips, pass);
    if (flipCount) *flipCount = flips;
    return T;
}

} // namespace sr
