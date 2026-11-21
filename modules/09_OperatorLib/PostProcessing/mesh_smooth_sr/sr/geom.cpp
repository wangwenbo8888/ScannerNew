#include "geom.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_set>

namespace sr {


Topology buildTopology(const std::vector<uint32_t>& triangles, size_t numVerts, int maxRing) {
    Topology topo;
    topo.adj.resize(numVerts);
    std::vector<std::set<uint32_t>> sets(numVerts);
    for (size_t t = 0; t < triangles.size(); t += 3) {
        uint32_t a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
        sets[a].insert(b); sets[a].insert(c);
        sets[b].insert(a); sets[b].insert(c);
        sets[c].insert(a); sets[c].insert(b);
    }
    for (size_t i = 0; i < numVerts; i++)
        topo.adj[i].assign(sets[i].begin(), sets[i].end());

    topo.rings.resize(numVerts);
    std::vector<uint32_t> visited;
    for (size_t i = 0; i < numVerts; i++) {
        std::vector<uint32_t> current{static_cast<uint32_t>(i)};
        std::unordered_set<uint32_t> seen{static_cast<uint32_t>(i)};
        for (int r = 0; r < maxRing; r++) {
            std::vector<uint32_t> next;
            for (uint32_t v : current)
                for (uint32_t w : topo.adj[v])
                    if (seen.insert(w).second) next.push_back(w);
            if (next.empty()) break;
            topo.rings[i].push_back(next);
            current = next;
        }
    }
    return topo;
}

std::vector<double> vertexNormals(const std::vector<double>& verts,
                                  const std::vector<uint32_t>& triangles, size_t numVerts) {
    std::vector<double> nrm(numVerts * 3, 0.0);
    for (size_t t = 0; t < triangles.size(); t += 3) {
        uint32_t a = triangles[t], b = triangles[t + 1], c = triangles[t + 2];
        double ax = verts[a * 3], ay = verts[a * 3 + 1], az = verts[a * 3 + 2];
        double ux = verts[b * 3] - ax, uy = verts[b * 3 + 1] - ay, uz = verts[b * 3 + 2] - az;
        double vx = verts[c * 3] - ax, vy = verts[c * 3 + 1] - ay, vz = verts[c * 3 + 2] - az;
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        for (uint32_t v : {a, b, c}) {
            nrm[v * 3] += nx;
            nrm[v * 3 + 1] += ny;
            nrm[v * 3 + 2] += nz;
        }
    }
    #pragma omp parallel for schedule(static)   // S3: per-vertex, writes disjoint
    for (int ii = 0; ii < (int)numVerts; ii++) {
        size_t i = (size_t)ii;
        double x = nrm[i * 3], y = nrm[i * 3 + 1], z = nrm[i * 3 + 2];
        double l = std::sqrt(x * x + y * y + z * z);
        if (l > 0) {
            nrm[i * 3] = x / l;
            nrm[i * 3 + 1] = y / l;
            nrm[i * 3 + 2] = z / l;
        } else {
            nrm[i * 3] = 0;
            nrm[i * 3 + 1] = 0;
            nrm[i * 3 + 2] = 1;
        }
    }
    return nrm;
}

void umbrellaNormalPass(std::vector<double>& nrm, const Topology& topo, double lambda) {
    size_t n = topo.adj.size();
    std::vector<double> out(n * 3);
    #pragma omp parallel for schedule(static)   // S3: Jacobi double buffer - race-free
    for (int ii = 0; ii < (int)n; ii++) {
        size_t i = (size_t)ii;
        const auto& nb = topo.adj[i];
        double sx = 0, sy = 0, sz = 0;
        for (uint32_t j : nb) {
            sx += nrm[j * 3];
            sy += nrm[j * 3 + 1];
            sz += nrm[j * 3 + 2];
        }
        if (!nb.empty()) {
            sx /= nb.size();
            sy /= nb.size();
            sz /= nb.size();
        }
        out[i * 3] = (1 - lambda) * nrm[i * 3] + lambda * sx;
        out[i * 3 + 1] = (1 - lambda) * nrm[i * 3 + 1] + lambda * sy;
        out[i * 3 + 2] = (1 - lambda) * nrm[i * 3 + 2] + lambda * sz;
        double x = out[i * 3], y = out[i * 3 + 1], z = out[i * 3 + 2];
        double l = std::sqrt(x * x + y * y + z * z);
        if (l > 0) {
            out[i * 3] = x / l;
            out[i * 3 + 1] = y / l;
            out[i * 3 + 2] = z / l;
        } else {
            out[i * 3] = 0;
            out[i * 3 + 1] = 0;
            out[i * 3 + 2] = 1;
        }
    }
    nrm = std::move(out);
}

void tangentFrame(double nx, double ny, double nz, double out[9]) {
    double ux = 0, uy = 0, uz = 0;
    if (std::fabs(nx) < 0.9) { ux = 1; } else { uy = 1; }
    double px = uy * nz - uz * ny, py = uz * nx - ux * nz, pz = ux * ny - uy * nx;
    double pl = std::sqrt(px * px + py * py + pz * pz);
    if (pl < 1e-300) pl = 1;
    px /= pl; py /= pl; pz /= pl;
    double qx = ny * pz - nz * py, qy = nz * px - nx * pz, qz = nx * py - ny * px;
    out[0] = px; out[1] = py; out[2] = pz;
    out[3] = qx; out[4] = qy; out[5] = qz;
    out[6] = nx; out[7] = ny; out[8] = nz;
}

bool fitParaboloid(const std::vector<double>& pts2d, double coeff[6]) {
    double A[36] = {0}, b[6] = {0};
    auto addPt = [&](double x, double y, double z) {
        double f[6] = {1, x, y, x * x, x * y, y * y};
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) A[r * 6 + c] += f[r] * f[c];
            b[r] += f[r] * z;
        }
    };
    for (size_t i = 0; i + 2 < pts2d.size(); i += 3) addPt(pts2d[i], pts2d[i + 1], pts2d[i + 2]);
    double M[6][7];
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) M[r][c] = A[r * 6 + c];
        M[r][6] = b[r];
    }
    for (int col = 0; col < 6; col++) {
        int piv = col;
        for (int r = col + 1; r < 6; r++)
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        if (std::fabs(M[piv][col]) < 1e-14) return false;
        if (piv != col) for (int c = col; c < 7; c++) std::swap(M[col][c], M[piv][c]);
        for (int r = 0; r < 6; r++) {
            if (r == col) continue;
            double f = M[r][col] / M[col][col];
            for (int c = col; c < 7; c++) M[r][c] -= f * M[col][c];
        }
    }
    for (int r = 0; r < 6; r++) coeff[r] = M[r][6] / M[r][r];
    return true;
}

double fitResidual(const double coeff[6], const std::vector<double>& pts2d) {
    double sum = 0;
    size_t cnt = 0;
    for (size_t i = 0; i + 2 < pts2d.size(); i += 3) {
        double x = pts2d[i], y = pts2d[i + 1], z = pts2d[i + 2];
        double f = coeff[0] + coeff[1] * x + coeff[2] * y + coeff[3] * x * x + coeff[4] * x * y + coeff[5] * y * y;
        sum += (f - z) * (f - z);
        cnt++;
    }
    return cnt ? std::sqrt(sum / cnt) : 0.0;
}

} // namespace sr
