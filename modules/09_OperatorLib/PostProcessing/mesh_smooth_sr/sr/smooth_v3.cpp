#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>   // [C1] omp_get_max_threads / omp_get_thread_num

namespace sr {


namespace {

struct Csr {
    std::vector<uint32_t> off;
    std::vector<uint32_t> nbr;
};

Csr buildCsr(const std::vector<uint32_t>& triangles, size_t numVerts) {
    std::vector<std::vector<uint32_t>> adj(numVerts);
    for (size_t t = 0; t < triangles.size(); t += 3) {
        uint32_t a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
        adj[a].push_back(b); adj[a].push_back(c);
        adj[b].push_back(a); adj[b].push_back(c);
        adj[c].push_back(a); adj[c].push_back(b);
    }
    Csr csr;
    csr.off.resize(numVerts + 1, 0);
    for (size_t i = 0; i < numVerts; i++) {
        std::vector<uint32_t>& v = adj[i];
        std::vector<uint32_t>::iterator it = std::unique(v.begin(), v.end());
        v.erase(it, v.end());
        csr.off[i + 1] = csr.off[i] + (uint32_t)v.size();
    }
    csr.nbr.resize(csr.off[numVerts]);
    for (size_t i = 0; i < numVerts; i++)
        std::copy(adj[i].begin(), adj[i].end(), csr.nbr.begin() + csr.off[i]);
    return csr;
}

bool solve6(double A[6][7]) {
    for (int col = 0; col < 6; col++) {
        int piv = col;
        for (int r = col + 1; r < 6; r++)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-300) return false;
        if (std::fabs(A[piv][col]) < 1e-12) A[piv][col] = A[piv][col] < 0 ? -1e-12 : 1e-12;
        if (piv != col)
            for (int c = col; c < 7; c++) std::swap(A[col][c], A[piv][c]);
        for (int r = 0; r < 6; r++) {
            if (r == col) continue;
            double f = A[r][col] / A[col][col];
            for (int c = col; c < 7; c++) A[r][c] -= f * A[col][c];
        }
    }
    return true;
}

struct Ticket {              // [C1] 32B   :219-222/:238-240  
    uint32_t j;
    uint32_t pad;            //  
    double wx, wy, wz;
};
struct StageScratch {        // [C1]   scratch  stage  stampArr  
                             // 0xFFFFFFFF  stage  R6A  1.2 
    std::vector<uint32_t> stampArr;
    std::vector<uint8_t>  ringOf;
    std::vector<uint32_t> collected;
    std::vector<double> xy, zloc, fitZ;
};
}  // namespace

std::vector<double> runSmoothV3(const Mesh& mesh, double tolDeg1, double tolDeg2, double sigma,
                                int maxRing, int outerIters, bool parallel) {
    const size_t n = mesh.numVerts;
    const std::vector<double>& verts = mesh.verts;

    std::vector<double> pos(mesh.verts);
    std::vector<uint32_t> cornerCount(n, 0);
    for (uint32_t t : mesh.triangles) cornerCount[t]++;
    Csr csr = buildCsr(mesh.triangles, n);

    const double gates[3] = {std::cos(tolDeg1 * 3.14159265358979323846 / 180.0),
                             std::cos(tolDeg2 * 3.14159265358979323846 / 180.0), -2.0};
    const double sigmas[3] = {sigma, sigma, 1.7976931348623157e+308};

    const int    T     = parallel ? omp_get_max_threads() : 1;  // [C1]  =  1  
    const size_t CHUNK = 4096;    // [C1]   i  R3A:   stage  A/B   2048-65536

    for (int outer = 0; outer < outerIters; outer++) {
    // [INS-b] per-outer segment timers (zero behavior, boundary-only)
    double stgMs[3] = {0.0, 0.0, 0.0};
    double nrmMsO = 0.0;
    auto tN0 = std::chrono::steady_clock::now();
    std::vector<double> nrm = vertexNormals(pos, mesh.triangles, n);
    nrmMsO = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tN0).count();
    std::vector<double> sumX(n, 0.0), sumY(n, 0.0), sumZ(n, 0.0);
    std::vector<uint32_t> cnt(n, 0);

    for (int stage = 0; stage < 3; stage++) {
        if (stage == 1 && tolDeg2 <= 0) continue;
        const double gate = gates[stage];
        const double sg = sigmas[stage];
        auto tStg0 = std::chrono::steady_clock::now();    // [INS-b]
        std::vector<StageScratch> thr(T);                 // [C1]   stage  
        for (int t = 0; t < T; ++t) {
            thr[t].stampArr.assign(n, 0xFFFFFFFFu);
            thr[t].ringOf.assign(n, 0);
            thr[t].collected.reserve(1024);
        }
        std::vector<std::vector<Ticket>> logs(T);         // [C1] chunk thread  
        for (size_t c0 = 0; c0 < n; c0 += CHUNK) {        // [C1]   i  
            const size_t c1 = std::min(c0 + CHUNK, n);
            for (int t = 0; t < T; ++t) logs[t].clear();  // clear  
            #pragma omp parallel num_threads(T) default(none) \
                shared(nrm, pos, csr, cornerCount, thr, logs) \
                firstprivate(gate, sg, maxRing, n, c0, c1)
            {
                const int tid = omp_get_thread_num();
                StageScratch& S = thr[tid];               //  
                #pragma omp for schedule(static)   // [C1] static=tid   dynamic/guided
                for (int ii = (int)c0; ii < (int)c1; ++ii) {
                    const size_t i = (size_t)ii;
            double nx = nrm[i * 3], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
            double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl < 1e-12) continue;
            nx /= nl; ny /= nl; nz /= nl;

            S.collected.clear();
            S.collected.push_back((uint32_t)i);
            S.stampArr[i] = (uint32_t)i;
            S.ringOf[i] = 0;
            size_t frontierStart = 0, frontierEnd = 1;
            for (int r = 1; r <= maxRing && frontierStart < frontierEnd; r++) {
                size_t nextStart = S.collected.size();
                for (size_t fi = frontierStart; fi < frontierEnd; fi++) {
                    uint32_t v = S.collected[fi];
                    for (uint32_t jj = csr.off[v]; jj < csr.off[v + 1]; jj++) {
                        uint32_t w = csr.nbr[jj];
                        if (S.stampArr[w] == (uint32_t)i) continue;
                        if (gate > -1.5) {
                            double dotv = nrm[w * 3] * nx + nrm[w * 3 + 1] * ny + nrm[w * 3 + 2] * nz;
                            if (dotv < gate) {
                                S.stampArr[w] = (uint32_t)i;
                                continue;
                            }
                        }
                        S.stampArr[w] = (uint32_t)i;
                        S.ringOf[w] = (uint8_t)r;
                        S.collected.push_back(w);
                    }
                }
                frontierStart = nextStart;
                frontierEnd = S.collected.size();
            }

            size_t nb = S.collected.size() - 1;
            if (nb < 7) continue;

            double ux = 0, uy = 0, uz = 0;
            if (std::fabs(nx) < 0.9) { ux = 1; } else { uy = 1; }
            double px = uy * nz - uz * ny, py = uz * nx - ux * nz, pz = ux * ny - uy * nx;
            double pl = std::sqrt(px * px + py * py + pz * pz);
            if (pl < 1e-300) continue;
            px /= pl; py /= pl; pz /= pl;
            double qx = ny * pz - nz * py, qy = nz * px - nx * pz, qz = nx * py - ny * px;

            double px0 = pos[S.collected[1] * 3] - pos[i * 3];
            double py0 = pos[S.collected[1] * 3 + 1] - pos[i * 3 + 1];
            double pz0 = pos[S.collected[1] * 3 + 2] - pos[i * 3 + 2];
            double a0 = std::atan2(px0 * qx + py0 * qy + pz0 * qz, px0 * px + py0 * py + pz0 * pz);
            int oct[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (size_t k = 1; k <= nb; k++) {
                uint32_t j = S.collected[k];
                double dx = pos[j * 3] - pos[i * 3];
                double dy = pos[j * 3 + 1] - pos[i * 3 + 1];
                double dz = pos[j * 3 + 2] - pos[i * 3 + 2];
                double ang = std::atan2(dx * qx + dy * qy + dz * qz, dx * px + dy * py + dz * pz) - a0;
                int sector = (int)std::floor(ang / (3.14159265358979323846 / 4.0));
                oct[((sector % 8) + 8) % 8] = 1;
            }
            int occ = 0;
            for (int k = 0; k < 8; k++) occ += oct[k];
            if (occ < 3) continue;

            double A[6][7] = {{0}};
            std::vector<double>& xy = S.xy; std::vector<double>& zloc = S.zloc;
            S.xy.resize(2 * nb); S.zloc.resize(nb);
            for (size_t k = 0; k < nb; k++) {
                uint32_t j = S.collected[k + 1];
                double dx = pos[j * 3] - pos[i * 3];
                double dy = pos[j * 3 + 1] - pos[i * 3 + 1];
                double dz = pos[j * 3 + 2] - pos[i * 3 + 2];
                double x = dx * px + dy * py + dz * pz;
                double y = dx * qx + dy * qy + dz * qz;
                double z = dx * nx + dy * ny + dz * nz;
                xy[2 * k] = x;
                xy[2 * k + 1] = y;
                zloc[k] = z;
                double f[6] = {x * x, y * y, x * y, x, y, 1.0};
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) A[r][c] += f[r] * f[c];
                    A[r][6] += f[r] * z;
                }
            }
            if (!solve6(A)) continue;
            double ca = A[0][6] / A[0][0];
            double cb = A[1][6] / A[1][1];
            double cc = A[2][6] / A[2][2];
            double cd = A[3][6] / A[3][3];
            double ce = A[4][6] / A[4][4];
            double cf = A[5][6] / A[5][5];

            double resSum = 0;
            std::vector<double>& fitZ = S.fitZ; S.fitZ.resize(nb);   // [C1]  
            for (size_t k = 0; k < nb; k++) {
                double x = xy[2 * k], y = xy[2 * k + 1];
                double f = ca * x * x + cb * y * y + cc * x * y + cd * x + ce * y + cf;
                fitZ[k] = f;
                resSum += std::fabs(f - zloc[k]);
            }
            if (resSum / nb > sg) continue;
            double ringExtent = 0;
            for (size_t k = 0; k < nb; k++) {
                ringExtent = std::max(ringExtent, std::fabs(xy[2*k]));
                ringExtent = std::max(ringExtent, std::fabs(xy[2*k+1]));
            }
            double maxDrop = 0;
            for (size_t k = 0; k < nb; k++) maxDrop = std::max(maxDrop, std::fabs(fitZ[k] - zloc[k]));
            if (maxDrop > 2.0 * ringExtent) continue;

            int e = maxRing;
            int sViol = -1;
            for (int r = 1; r <= maxRing; r++) {
                size_t ringBegin = 1, ringEnd2 = 1;
                bool foundB = false;
                for (size_t k = 1; k <= nb; k++) {
                    if (!foundB && S.ringOf[S.collected[k]] == r) { ringBegin = k; foundB = true; }
                    if (S.ringOf[S.collected[k]] == r) ringEnd2 = k + 1;
                }
                if (!foundB) continue;
                size_t ringPts = ringEnd2 - ringBegin;
                size_t viol = 0;
                for (size_t k = ringBegin; k < ringEnd2; k++)
                    if (std::fabs(fitZ[k] - zloc[k]) > sg) viol++;
                if ((double)viol > (double)ringPts * 1.0) { sViol = r; break; }
            }

            double sumCx = pos[i * 3] + nx * cf;
            double sumCy = pos[i * 3 + 1] + ny * cf;
            double sumCz = pos[i * 3 + 2] + nz * cf;
            logs[tid].push_back(Ticket{(uint32_t)i, 0u, sumCx, sumCy, sumCz});   // [C1]  
            for (size_t k = 0; k < nb; k++) {
                uint32_t j = S.collected[k + 1];
                int rk = S.ringOf[j];
                double w = 1.0;
                if (sViol >= 0 && rk > sViol) {
                    double t = (rk - sViol) / (double)(e - sViol);
                    if (t < 0) t = 0;
                    if (t > 1) t = 1;
                    if (e > sViol) w = 1.0 - 3.0 * t * t + 2.0 * t * t * t;
                }
                double x = xy[2 * k], y = xy[2 * k + 1];
                double zNew = zloc[k] + w * (fitZ[k] - zloc[k]);
                double wx = pos[i * 3] + px * x + qx * y + nx * zNew;
                double wy = pos[i * 3 + 1] + py * x + qy * y + ny * zNew;
                double wz = pos[i * 3 + 2] + pz * x + qz * y + nz * zNew;
                if (cornerCount[j] >= 3)
                    logs[tid].push_back(Ticket{j, 0u, wx, wy, wz});   // [C1]   if  
            }
        }
        }   // omp parallel [C1]
        for (int t = 0; t < T; ++t)        // [C1]  tid   i  
            for (const Ticket& tk : logs[t]) {   //  =  (stage,i)  
                sumX[tk.j] += tk.wx;
                sumY[tk.j] += tk.wy;
                sumZ[tk.j] += tk.wz;
                ++cnt[tk.j];
            }
        }   // chunk loop [C1]
        stgMs[stage] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStg0).count();   // [INS-b]
    }
    std::printf("[time] sr smooth outer=%d nrm=%.1fms s0=%.1f s1=%.1f s2=%.1fms\n",
                outer, nrmMsO, stgMs[0], stgMs[1], stgMs[2]);     // [INS-b]

    #pragma omp parallel for schedule(static)   // S3: disjoint per-vertex writes
    for (int ii = 0; ii < (int)n; ii++) {
        size_t i = (size_t)ii;
        if (cnt[i] > 0 && cornerCount[i] >= 3) {
            pos[i * 3] = sumX[i] / cnt[i];
            pos[i * 3 + 1] = sumY[i] / cnt[i];
            pos[i * 3 + 2] = sumZ[i] / cnt[i];
        }
    }
    }
    return pos;
}

} // namespace sr
