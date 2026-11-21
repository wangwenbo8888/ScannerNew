// ============================================================================
// mesh_normal_est.cpp — 阶段 0 实现（M1 CPU 先行版；GPU 五件套随后）
//
// 算法忠实移植自 PCMP 沙盒 NormalEstimation（金标准 run_20261121_50Wasc 同源）：
//   点距估计（NN 中位数）→ 赶野点（kNN 统计，法线清零点保留）→ 拔钉（平面
//   拟合 + 原位贴回，点数不变）→ 加权 PCA（σ=radius/3，链默认）→ BFS 全局
//   定向（种子=最近质心）→ 符号修复（入口法线为参考，链 head 的 per-point
//   sign fix 语义）→ 补缺失法线 → 法线质量门禁（2000 采样 k=10）。
//
// 契约：入口 io.normals 若与顶点数匹配则作为定向参考（扫描仪/融合法线——
//       "局部噪但全局朝外"）；出口 io.normals 为唯一权威源（R2）。
//       点数与索引全程不变（与沙盒链逐点对齐，下游索引语义稳定）。
// ============================================================================
#include "mesh_normal_est.h"

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace calib {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Vector3d = Eigen::Vector3d;

constexpr double kSigmaScale = 1.0 / 3.0;   // 加权 PCA σ = radius × 此值（链默认，GPU 同源）
constexpr size_t kSpacingSamples = 5000;
constexpr size_t kGateSamples = 2000;
constexpr int    kGateK = 10;

// ============================================================
// CSR 均匀栅格（CPU，沙盒 UniformGrid 语义等价实现）
// ============================================================
struct UniGrid {
    double bbMinX = 0, bbMinY = 0, bbMinZ = 0, cell = 1.0;
    int gx = 1, gy = 1, gz = 1;
    std::vector<int> cellStart;   // (gx*gy*gz + 1)
    std::vector<int> entries;     // 按格排序的点索引

    int cellOf(const double* p) const {
        int cx = std::max(0, std::min(gx - 1, (int)((p[0] - bbMinX) / cell)));
        int cy = std::max(0, std::min(gy - 1, (int)((p[1] - bbMinY) / cell)));
        int cz = std::max(0, std::min(gz - 1, (int)((p[2] - bbMinZ) / cell)));
        return (cx * gy + cy) * gz + cz;
    }

    void build(const double* xyz, size_t n, double cellSize) {
        double mn[3] = {xyz[0], xyz[1], xyz[2]}, mx[3] = {xyz[0], xyz[1], xyz[2]};
        for (size_t i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
                mn[k] = std::min(mn[k], xyz[3 * i + k]);
                mx[k] = std::max(mx[k], xyz[3 * i + k]);
            }
        }
        bbMinX = mn[0]; bbMinY = mn[1]; bbMinZ = mn[2];
        cell = std::max(cellSize, 1e-12);
        gx = std::max(1, (int)((mx[0] - mn[0]) / cell) + 1);
        gy = std::max(1, (int)((mx[1] - mn[1]) / cell) + 1);
        gz = std::max(1, (int)((mx[2] - mn[2]) / cell) + 1);
        const size_t nc = (size_t)gx * gy * gz;
        cellStart.assign(nc + 1, 0);
        std::vector<int> cellOfIdx(n);
        for (size_t i = 0; i < n; ++i) {
            int c = cellOf(xyz + 3 * i);
            cellOfIdx[i] = c;
            ++cellStart[c + 1];
        }
        for (size_t c = 0; c < nc; ++c) cellStart[c + 1] += cellStart[c];
        entries.resize(n);
        std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
        for (size_t i = 0; i < n; ++i)
            entries[(size_t)cursor[cellOfIdx[i]]++] = (int)i;
    }

    /// 半径检索：按 [p−r, p+r] 盒裁剪计算格范围（任意 r ≤ cell 或 r > cell 均正确）
    template <typename F>
    void forNeighbors(const double* p, double r, F&& fn) const {
        const double r2 = r * r;
        int cx0 = (int)std::floor((p[0] - r - bbMinX) / cell), cx1 = (int)std::floor((p[0] + r - bbMinX) / cell);
        int cy0 = (int)std::floor((p[1] - r - bbMinY) / cell), cy1 = (int)std::floor((p[1] + r - bbMinY) / cell);
        int cz0 = (int)std::floor((p[2] - r - bbMinZ) / cell), cz1 = (int)std::floor((p[2] + r - bbMinZ) / cell);
        cx0 = std::max(0, cx0); cx1 = std::min(gx - 1, cx1);
        cy0 = std::max(0, cy0); cy1 = std::min(gy - 1, cy1);
        cz0 = std::max(0, cz0); cz1 = std::min(gz - 1, cz1);
        for (int z = cz0; z <= cz1; ++z)
            for (int y = cy0; y <= cy1; ++y)
                for (int x = cx0; x <= cx1; ++x) {
                    const int c = (x * gy + y) * gz + z;
                    for (int e = cellStart[c]; e < cellStart[c + 1]; ++e)
                        fn(entries[e], r2);
                }
    }
};

inline double dotDist2(const double* a, const double* b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

// ============================================================
// 各阶段（语义对齐沙盒全 GPU 流水线）
// ============================================================

/// 点距：体积法初估 → NN 中位数采样（沙盒 computeAvgSpacing 同源）
double estimateSpacing(const std::vector<double>& xyz) {
    const size_t n = xyz.size() / 3;
    if (n == 0) return 0.001;
    double mn[3] = {xyz[0], xyz[1], xyz[2]}, mx[3] = {xyz[0], xyz[1], xyz[2]};
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) { mn[k] = std::min(mn[k], xyz[3*i+k]); mx[k] = std::max(mx[k], xyz[3*i+k]); }
    const double vol = std::max(0.0, (mx[0]-mn[0])) * std::max(0.0, (mx[1]-mn[1])) * std::max(0.0, (mx[2]-mn[2]));
    const double volSpacing = (vol > 0 && n > 1) ? std::cbrt(vol / (double)n) : 0.1;

    UniGrid grid;
    grid.build(xyz.data(), n, volSpacing * 4.0);
    const size_t nSample = std::min(kSpacingSamples, n);
    std::vector<double> nnD(nSample);
    const size_t NT = nSample;
    #pragma omp parallel for schedule(dynamic)
    for (int64_t s = 0; s < (int64_t)NT; ++s) {
        const size_t si = (size_t)s * n / nSample;
        const double* p = xyz.data() + 3 * si;
        double best = std::numeric_limits<double>::max();
        grid.forNeighbors(p, volSpacing * 4.0, [&](int j, double) {
            if ((size_t)j == si) return;
            best = std::min(best, dotDist2(p, xyz.data() + 3 * j));
        });
        nnD[s] = (best < std::numeric_limits<double>::max()) ? std::sqrt(best) : 0.0;
    }
    std::sort(nnD.begin(), nnD.end());
    const double median = nnD[nSample / 2];
    return (median > 0) ? median : volSpacing;
}

/// 赶野点：精确 kNN(k+1 含自身) 平均距离（排除自身后取 k 个）> μ+factor·σ →
/// 法线清零（点保留）。GPU 版镜像：首环 r=cellSize=radius，环扩直至找满。
std::vector<char> detectOutliers(const std::vector<double>& xyz, double radius,
                                 int k, double factor, size_t* removedOut) {
    const size_t n = xyz.size() / 3;
    std::vector<char> mask(n, 1);
    if (n < (size_t)k + 1) return mask;

    UniGrid grid;
    grid.build(xyz.data(), n, radius);
    std::vector<double> meanD(n, 0.0);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        const double* p = xyz.data() + 3 * i;
        double best[65]; int cnt = 0;                       // top-(k+1) 最大距离堆替代：收集后排序
        double d2buf[2048];
        int nb = 0;
        double r = radius;
        while (r <= radius * 1024.0) {
            nb = 0;
            grid.forNeighbors(p, r, [&](int j, double r2) {
                if (nb >= 2048) return;
                d2buf[nb++] = dotDist2(p, xyz.data() + 3 * j);
            });
            if (nb >= k + 1) break;
            r *= 2.0;
        }
        (void)best; (void)cnt;
        // 含自身的 k+1 最近 → 排除自身（d2≈0 的一个）→ k 个
        std::sort(d2buf, d2buf + nb);
        const int take = std::min(k + 1, nb);
        double sum = 0; int used = 0; int skippedSelf = 0;
        for (int t = 0; t < take && used < k; ++t) {
            if (d2buf[t] <= 0.0 && !skippedSelf) { skippedSelf = 1; continue; }   // 自身
            sum += std::sqrt(d2buf[t]);
            ++used;
        }
        meanD[i] = (used > 0) ? sum / used : 0.0;
    }

    double sumD = 0, sumD2 = 0; size_t valid = 0;
    for (size_t i = 0; i < n; ++i)
        if (meanD[i] > 0) { sumD += meanD[i]; sumD2 += meanD[i] * meanD[i]; ++valid; }
    if (valid < (size_t)k) return mask;   // 与 GPU computeThresholdKernel 同门
    const double mu = sumD / valid;
    const double sigma = std::sqrt(std::max(0.0, sumD2 / valid - mu * mu));
    const double thr = mu + factor * sigma;
    size_t removed = 0;
    for (size_t i = 0; i < n; ++i)
        if (meanD[i] > 0 && meanD[i] > thr) { mask[i] = 0; ++removed; }
    *removedOut = removed;
    return mask;
}

/// 拔钉（GPU spikeDetectKernel/spikeSnapKernel 镜像）：
///   邻域（radius 圆盘内）排除自身/离群点/已标钉 → nJ≥12 → 等权 PCA 平面 →
///   planarity=2λ0/(trC−λ0)≤0.04 → |dv|≥minAbs 且 ratio=|dv|/max(dmean,noise)≥thr
///   且 similar(|dj−dv|≤max(0.5|dv|,minAbs))<2 → 钉。检测循环（mask 累积，
///   原始坐标）→ 循环后一次贴回（p −= dv·n）。cap=max(1,N/100)，0 新增即停。
size_t removeSpikes(std::vector<double>& xyz, const std::vector<char>& outlierMask,
                    const MeshNormalEstParams& prm, double radius,
                    bool includeOutliers, size_t cap, int* roundsOut,
                    size_t* candidatesOut, std::vector<char>* spikeMaskOut,
                    const std::function<bool()>& cancel) {
    struct SpikeConst { int kMin = 12; double planarMax = 0.04; };
    static const SpikeConst sc;
    const size_t n = xyz.size() / 3;
    if (n < 64) return 0;
    std::vector<char> spikeMask(n, 0);
    UniGrid grid;                                       // 原始坐标，检测全程不重建（GPU 同源）
    grid.build(xyz.data(), n, radius);
    size_t totalSpikes = 0, qualifying = 0;
    int rounds = 0;

    for (int round = 0; round < prm.spikeMaxRounds; ++round) {
        if (cancel && cancel()) break;
        std::vector<int> hits;
        #pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < (int64_t)n; ++i) {
            if (spikeMask[i]) continue;
            if (!includeOutliers && !outlierMask[i]) continue;
            const double* p = xyz.data() + 3 * i;
            std::vector<int> nb;
            grid.forNeighbors(p, radius, [&](int j, double r2) {
                if ((size_t)j == (size_t)i) return;
                if (dotDist2(p, xyz.data() + 3 * j) > r2) return;      // 圆盘
                if (!outlierMask[j]) return;                            // 邻居排离群（无条件）
                if (spikeMask[j]) return;                               // 邻居排已标钉
                nb.push_back(j);
            });
            if ((int)nb.size() < sc.kMin) continue;
            Vector3d c(0, 0, 0);
            for (int j : nb) c += Eigen::Map<const Vector3d>(xyz.data() + 3 * (size_t)j);
            c /= (double)nb.size();
            Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
            for (int j : nb) {
                const Vector3d d = Eigen::Map<const Vector3d>(xyz.data() + 3 * (size_t)j) - c;
                C += d * d.transpose();
            }
            const double trC = C.trace();
            if (std::fabs(C(0,0)) + std::fabs(C(1,1)) + std::fabs(C(2,2)) <= 1e-30) continue;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
            const Vector3d nrm = eig.eigenvectors().col(0);
            const double lambda0 = eig.eigenvalues()(0);
            const double denom = trC - lambda0;
            const double planarity = (denom > 1e-300) ? (2.0 * lambda0 / denom) : 1e30;
            if (!(planarity <= sc.planarMax)) continue;
            const Vector3d pv(p[0], p[1], p[2]);
            const double dv = (pv - c).dot(nrm);
            double dsum = 0;
            std::vector<double> dj(nb.size());
            for (size_t t = 0; t < nb.size(); ++t) {
                const Vector3d d = Eigen::Map<const Vector3d>(xyz.data() + 3 * (size_t)nb[t]) - c;
                dj[t] = d.dot(nrm);
                dsum += std::fabs(dj[t]);
            }
            const double dmean = dsum / (double)nb.size();
            const double ratio = std::fabs(dv) / std::max(dmean, (double)prm.spikeNoiseFloorMm);
            const double tol = std::max(0.5 * std::fabs(dv), (double)prm.spikeMinAbsMm);   // similarTol=minAbs（R1A）
            int similar = 0;
            for (double v : dj)
                if (std::fabs(v - dv) <= tol) ++similar;
            const bool isSpike = std::fabs(dv) >= (double)prm.spikeMinAbsMm
                              && ratio >= (double)prm.spikeRatioThreshold
                              && similar < 2;
            if (isSpike) {
                #pragma omp critical(spike_hit)
                { hits.push_back((int)i); ++qualifying; }
            }
        }
        std::sort(hits.begin(), hits.end());
        const size_t budget = (cap > totalSpikes) ? cap - totalSpikes : 0;
        if (hits.size() > budget) hits.resize(budget);
        for (int i : hits) spikeMask[i] = 1;
        totalSpikes += hits.size();
        ++rounds;
        if (hits.empty() || totalSpikes >= cap) break;
    }

    // 一次贴回（检测循环结束后；GPU spikeSnapKernel 同源——干净邻居集重拟合平面投影）
    size_t snapped = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!spikeMask[i]) continue;
        const double* p = xyz.data() + 3 * i;
        std::vector<int> nb;
        grid.forNeighbors(p, radius, [&](int j, double r2) {
            if ((size_t)j == i) return;
            if (dotDist2(p, xyz.data() + 3 * j) > r2) return;
            if (!outlierMask[j]) return;
            if (spikeMask[j]) return;
            nb.push_back((int)j);
        });
        if ((int)nb.size() < sc.kMin) continue;
        Vector3d c(0, 0, 0);
        for (int j : nb) c += Eigen::Map<const Vector3d>(xyz.data() + 3 * (size_t)j);
        c /= (double)nb.size();
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        for (int j : nb) {
            const Vector3d d = Eigen::Map<const Vector3d>(xyz.data() + 3 * (size_t)j) - c;
            C += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
        const Vector3d nrm = eig.eigenvectors().col(0);
        const Vector3d pv(p[0], p[1], p[2]);
        const Vector3d proj = pv - (pv - c).dot(nrm) * nrm;
        xyz[3 * i] = proj.x(); xyz[3 * i + 1] = proj.y(); xyz[3 * i + 2] = proj.z();
        ++snapped;
    }
    *roundsOut = rounds;
    if (candidatesOut) *candidatesOut = qualifying;
    if (spikeMaskOut) *spikeMaskOut = std::move(spikeMask);
    return totalSpikes;
}

/// 加权 PCA 法线（w = exp(-d²/σ²)，σ = radius×kSigmaScale；不足 minNeighbors → 零）
void computeWeightedPca(const std::vector<double>& xyz, double radius, int minNeighbors,
                        const Vector3d& viewpoint, std::vector<double>& nrm,
                        const std::function<bool()>& cancel, size_t batch) {
    const size_t n = xyz.size() / 3;
    UniGrid grid;
    grid.build(xyz.data(), n, radius);
    const double sigma = radius * kSigmaScale;
    const double inv2s2 = 1.0 / (2.0 * sigma * sigma);
    nrm.assign(n * 3, 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        if (((i & (int64_t)batch) == 0) && cancel && cancel()) continue;   // 取消检查点
        const double* p = xyz.data() + 3 * i;
        double w[128]; double qx[128], qy[128], qz[128];
        int cnt = 0;
        grid.forNeighbors(p, radius, [&](int j, double r2) {
            const double d2 = dotDist2(p, xyz.data() + 3 * j);
            if (d2 > r2 || cnt >= 128) return;
            w[cnt] = std::exp(-d2 * inv2s2);
            qx[cnt] = xyz[3 * j]; qy[cnt] = xyz[3 * j + 1]; qz[cnt] = xyz[3 * j + 2];
            ++cnt;
        });
        if (cnt < minNeighbors) continue;   // 零法线（交补漏）
        double sw = 0, cx = 0, cy = 0, cz = 0;
        for (int t = 0; t < cnt; ++t) {
            sw += w[t]; cx += w[t] * qx[t]; cy += w[t] * qy[t]; cz += w[t] * qz[t];
        }
        cx /= sw; cy /= sw; cz /= sw;
        double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
        for (int t = 0; t < cnt; ++t) {
            const double dx = qx[t] - cx, dy = qy[t] - cy, dz = qz[t] - cz;
            a00 += w[t] * dx * dx; a01 += w[t] * dx * dy; a02 += w[t] * dx * dz;
            a11 += w[t] * dy * dy; a12 += w[t] * dy * dz; a22 += w[t] * dz * dz;
        }
        Eigen::Matrix3d C;
        C << a00, a01, a02, a01, a11, a12, a02, a12, a22;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
        Vector3d nv = eig.eigenvectors().col(0);
        if (nv.dot(viewpoint - Vector3d(p[0], p[1], p[2])) < 0) nv = -nv;
        nrm[3 * i] = nv.x(); nrm[3 * i + 1] = nv.y(); nrm[3 * i + 2] = nv.z();
    }
}

/// BFS 全局定向（逐连通组件版，261121 真机修复）：
///   原 bug：单种子 BFS——未连到的组件保持视点初值，与主流域不一致
///   （真机 208 万点 bad=19.2% 的重要来源）；且整片朝内域无法自愈。
///   现口径：每组件独立 BFS 一致化 → 组件级外向投票（采样 dot(n, p−云质心)
///   多数为负＝整片朝内 → 整组件翻转）。凹腔件仍有残余（视向通道是根治，
///   见移植计划挂账）。
size_t bfsOrient(const std::vector<double>& xyz, std::vector<double>& nrm, double radius) {
    const size_t n = xyz.size() / 3;
    if (n < 2) return 0;
    Vector3d centroid(0, 0, 0);
    for (size_t i = 0; i < n; ++i)
        centroid += Eigen::Map<const Vector3d>(xyz.data() + 3 * i);
    centroid /= (double)n;

    UniGrid grid;
    grid.build(xyz.data(), n, radius * 6.0);
    std::vector<unsigned char> visited(n, 0);
    std::vector<int> queue;
    std::vector<int> nb;
    std::vector<int> comp;                 // 当前组件成员（外向投票用）
    comp.reserve(1024);
    size_t flipped = 0;

    for (size_t seedAll = 0; seedAll < n; ++seedAll) {
        if (visited[seedAll]) continue;
        // —— 组件 BFS（种子取向：朝外于云质心）——
        comp.clear();
        queue.clear();
        queue.push_back((int)seedAll);
        visited[seedAll] = 1;
        {
            const double* sn = nrm.data() + 3 * seedAll;
            const Vector3d sp(xyz[3*seedAll], xyz[3*seedAll+1], xyz[3*seedAll+2]);
            if (Vector3d(sn[0], sn[1], sn[2]).dot(sp - centroid) < 0) {
                nrm[3*seedAll] = -nrm[3*seedAll];
                nrm[3*seedAll+1] = -nrm[3*seedAll+1];
                nrm[3*seedAll+2] = -nrm[3*seedAll+2];
                ++flipped;
            }
        }
        size_t head = 0;
        while (head < queue.size()) {
            const int i = queue[head++];
            comp.push_back(i);
            const double* pi = xyz.data() + 3 * (size_t)i;
            const double* ni = nrm.data() + 3 * (size_t)i;
            nb.clear();
            grid.forNeighbors(pi, radius, [&](int j, double r2) {
                if ((size_t)j == (size_t)i) return;
                if (dotDist2(pi, xyz.data() + 3 * j) <= r2) nb.push_back(j);
            });
            for (int j : nb) {
                if (visited[j]) continue;
                const double* nj = nrm.data() + 3 * (size_t)j;
                if (ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2] < 0) {
                    nrm[3 * j] = -nrm[3 * j]; nrm[3 * j + 1] = -nrm[3 * j + 1]; nrm[3 * j + 2] = -nrm[3 * j + 2];
                    ++flipped;
                }
                visited[j] = 1;
                queue.push_back(j);
            }
        }
        // —— 组件级外向投票（多数朝内 → 整组件翻转）——
        if (!comp.empty()) {
            double vote = 0;
            const size_t step = std::max<size_t>(1, comp.size() / 1024);   // 采样 ≤1024
            size_t sampled = 0;
            for (size_t k = 0; k < comp.size(); k += step) {
                const int vi = comp[k];
                const double* pn = nrm.data() + 3 * (size_t)vi;
                const Vector3d pv(xyz[3*vi], xyz[3*vi+1], xyz[3*vi+2]);
                vote += Vector3d(pn[0], pn[1], pn[2]).dot(pv - centroid);
                ++sampled;
            }
            if (sampled > 0 && vote / (double)sampled < -1e-12) {          // 平均朝内
                for (int vi : comp) {
                    nrm[3*vi] = -nrm[3*vi]; nrm[3*vi+1] = -nrm[3*vi+1]; nrm[3*vi+2] = -nrm[3*vi+2];
                    ++flipped;
                }
            }
        }
    }
    return flipped;
}

/// 补缺失法线：零法线取邻域非零法线均值
size_t fillMissing(const std::vector<double>& xyz, std::vector<double>& nrm, double radius) {
    const size_t n = xyz.size() / 3;
    UniGrid grid;
    grid.build(xyz.data(), n, radius);
    std::vector<int> missing;
    for (size_t i = 0; i < n; ++i)
        if (nrm[3 * i] == 0 && nrm[3 * i + 1] == 0 && nrm[3 * i + 2] == 0) missing.push_back((int)i);
    size_t filled = 0;
    for (int i : missing) {
        const double* p = xyz.data() + 3 * (size_t)i;
        Vector3d mean(0, 0, 0);
        grid.forNeighbors(p, radius, [&](int j, double r2) {
            if (dotDist2(p, xyz.data() + 3 * j) > r2) return;
            const double* nj = nrm.data() + 3 * (size_t)j;
            if (nj[0] != 0 || nj[1] != 0 || nj[2] != 0) mean += Vector3d(nj[0], nj[1], nj[2]);
        });
        if (mean.squaredNorm() > 1e-24) {
            mean.normalize();
            nrm[3 * i] = mean.x(); nrm[3 * i + 1] = mean.y(); nrm[3 * i + 2] = mean.z();
            ++filled;
        }
    }
    return filled;
}

/// 法线质量门禁（链 normalGate 同源：2000 采样、k=10、法线与邻均反向为坏）
double normalGate(const std::vector<double>& xyz, const std::vector<double>& nrm,
                  double spacing, size_t* badOut) {
    const size_t n = xyz.size() / 3;
    UniGrid grid;
    grid.build(xyz.data(), n, spacing * 4.0);
    const size_t samples = std::min(kGateSamples, n);
    size_t bad = 0;
    for (size_t s = 0; s < samples; ++s) {
        const size_t i = s * n / samples;
        const double* ni = nrm.data() + 3 * i;
        const double n2 = ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2];
        if (n2 < 1e-24) { ++bad; continue; }
        const double* p = xyz.data() + 3 * i;
        std::vector<std::pair<double, int>> cand;
        grid.forNeighbors(p, spacing * 4.0, [&](int j, double) {
            if ((size_t)j == i) return;
            cand.emplace_back(dotDist2(p, xyz.data() + 3 * j), j);
        });
        if (cand.size() < 3) { ++bad; continue; }
        const size_t k = std::min((size_t)kGateK, cand.size());
        std::nth_element(cand.begin(), cand.begin() + (std::ptrdiff_t)k - 1, cand.end());
        Vector3d mean(0, 0, 0);
        for (size_t t = 0; t < k; ++t)
            mean += Eigen::Map<const Vector3d>(nrm.data() + 3 * (size_t)cand[t].second);
        if (mean.squaredNorm() < 1e-24) { ++bad; continue; }
        mean.normalize();
        const Vector3d nvi(ni[0], ni[1], ni[2]);
        if (nvi.normalized().dot(mean) < 0.0) ++bad;
    }
    *badOut = bad;
    return (samples > 0) ? (double)bad / (double)samples : 1.0;
}

} // namespace

// ============================================================
// Params
// ============================================================
void MeshNormalEstParams::validate() const {
    if (radiusMultiplier <= 0.0f)
        throw std::invalid_argument("MeshNormalEstParams: radiusMultiplier must be > 0");
    if (minNeighbors < 1)
        throw std::invalid_argument("MeshNormalEstParams: minNeighbors must be >= 1");
    if (minNeighbors > 127)
        throw std::invalid_argument("MeshNormalEstParams: minNeighbors must be <= 127");
    if (knownSpacingMm < 0.0f)
        throw std::invalid_argument("MeshNormalEstParams: knownSpacingMm must be >= 0");
    if (outlierK < 1)
        throw std::invalid_argument("MeshNormalEstParams: outlierK must be >= 1");
    if (outlierFactor <= 0.0f)
        throw std::invalid_argument("MeshNormalEstParams: outlierFactor must be > 0");
    if (spikeMaxRounds < 0)
        throw std::invalid_argument("MeshNormalEstParams: spikeMaxRounds must be >= 0");
    if (!(gateWarnPct >= 0.0f && gateWarnPct <= gateFailPct && gateFailPct <= 100.0f))
        throw std::invalid_argument("MeshNormalEstParams: require 0 <= gateWarnPct <= gateFailPct <= 100");
}

nlohmann::json MeshNormalEstParams::toJson() const {
    nlohmann::json j;
    j["radiusMultiplier"] = radiusMultiplier;
    j["minNeighbors"] = minNeighbors;
    j["knownSpacingMm"] = knownSpacingMm;
    j["removeOutliers"] = removeOutliers;
    j["outlierK"] = outlierK;
    j["outlierFactor"] = outlierFactor;
    j["removeSpikes"] = removeSpikes;
    j["spikeReportOnly"] = spikeReportOnly;
    j["spikeIncludeOutliers"] = spikeIncludeOutliers;
    j["spikeRatioThreshold"] = spikeRatioThreshold;
    j["spikeMinAbsMm"] = spikeMinAbsMm;
    j["spikeNoiseFloorMm"] = spikeNoiseFloorMm;
    j["spikeMaxRounds"] = spikeMaxRounds;
    j["gateWarnPct"] = gateWarnPct;
    j["gateFailPct"] = gateFailPct;
    j["numThreads"] = numThreads;
    return j;
}

MeshNormalEstParams MeshNormalEstParams::fromJson(const nlohmann::json& j) {
    MeshNormalEstParams p;
    auto g = [&j](const char* k, auto& v) {
        auto it = j.find(k);
        if (it != j.end()) it->get_to(v);
    };
    g("radiusMultiplier", p.radiusMultiplier);
    g("minNeighbors", p.minNeighbors);
    g("knownSpacingMm", p.knownSpacingMm);
    g("removeOutliers", p.removeOutliers);
    g("outlierK", p.outlierK);
    g("outlierFactor", p.outlierFactor);
    g("removeSpikes", p.removeSpikes);
    g("spikeReportOnly", p.spikeReportOnly);
    g("spikeIncludeOutliers", p.spikeIncludeOutliers);
    g("spikeRatioThreshold", p.spikeRatioThreshold);
    g("spikeMinAbsMm", p.spikeMinAbsMm);
    g("spikeNoiseFloorMm", p.spikeNoiseFloorMm);
    g("spikeMaxRounds", p.spikeMaxRounds);
    g("gateWarnPct", p.gateWarnPct);
    g("gateFailPct", p.gateFailPct);
    g("numThreads", p.numThreads);
    p.validate();
    return p;
}

// ============================================================
// Operator：实装（CPU 先行版）
// ============================================================
struct MeshNormalEst::Impl {
    MeshNormalEstParams params;
    MeshNormalEstStats stats;
    std::function<bool()> cancelProbe;

    bool cancelled() const { return cancelProbe && cancelProbe(); }
    int threads() const {
        return params.numThreads > 0 ? params.numThreads : omp_get_max_threads();
    }
};

MeshNormalEst::MeshNormalEst(const MeshNormalEstParams& params) : pImpl_(std::make_unique<Impl>()) {
    params.validate();
    pImpl_->params = params;
}

MeshNormalEst::~MeshNormalEst() = default;

MeshNormalEstResult MeshNormalEst::Execute(IndexedMesh& io) {
    auto t0 = Clock::now();
    MeshNormalEstResult r;
    const size_t n = io.vertexCount();
    pImpl_->stats = MeshNormalEstStats{};
    pImpl_->stats.pointsIn = n;

    if (n == 0) {
        r.success = false;
        r.message = std::string(kLogTag) + ": 空点云，无法重算法线";
        return r;
    }
    if (pImpl_->cancelled()) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 执行前已取消";
        return r;
    }
    const auto& prm = pImpl_->params;
    if (prm.numThreads > 0)
        omp_set_num_threads(prm.numThreads);

    // host double 工作区（入口 float → double）
    std::vector<double> xyz(n * 3);
    for (size_t i = 0; i < n * 3; ++i) xyz[i] = (double)io.xyz[i];

    // 定向参考：入口法线（扫描仪/融合法线——局部噪但全局朝外）
    const bool hasRefNormals = io.hasNormals();
    std::vector<double> refNrm;
    if (hasRefNormals) {
        refNrm.resize(n * 3);
        for (size_t i = 0; i < n * 3; ++i) refNrm[i] = (double)io.normals[i];
    }

    // —— 1) 点距 ——
    const double spacing = prm.knownSpacingMm > 0.0f ? (double)prm.knownSpacingMm
                                                     : estimateSpacing(xyz);
    const double radius = spacing * (double)prm.radiusMultiplier;
    pImpl_->stats.spacingMm = (float)spacing;

    // —— 2) 赶野点（法线清零语义由 PCA 后统一承接）——
    std::vector<char> outlierMask(n, 1);
    if (prm.removeOutliers) {
        size_t removed = 0;
        outlierMask = detectOutliers(xyz, radius, prm.outlierK, prm.outlierFactor, &removed);
        pImpl_->stats.outliersRemoved = removed;
    }

    // —— 3) 拔钉（检测循环＋一次贴回；点数不变；mask 供 PCA 后并入清零）——
    std::vector<char> spikeMask(n, 0);
    if (prm.removeSpikes && !prm.spikeReportOnly) {
        const size_t cap = std::max<size_t>(1, n / 100);
        int rounds = 0;
        size_t candidates = 0;
        const size_t removed = removeSpikes(xyz, outlierMask, prm, radius,
                                            prm.spikeIncludeOutliers, cap, &rounds,
                                            &candidates, &spikeMask,
                                            [this]() { return pImpl_->cancelled(); });
        pImpl_->stats.spikesRemoved = removed;
        pImpl_->stats.spikeRounds = rounds;
    }

    // —— 4) 加权 PCA ——
    // 自动视点：质心（链 auto-viewpoint 同源语义；后续 BFS+符号修复为最终定向权威）
    Vector3d viewpoint(0, 0, 0);
    for (size_t i = 0; i < n; ++i)
        viewpoint += Eigen::Map<const Vector3d>(xyz.data() + 3 * i);
    viewpoint /= (double)n;

    std::vector<double> nrm;
    computeWeightedPca(xyz, radius, prm.minNeighbors, viewpoint, nrm,
                       [this]() { return pImpl_->cancelled(); }, 0xFFFF);

    // 离群 ∪ 钉 → 法线清零（GPU zeroOutlierNormals＋mergeSpikeMask 语义；交补漏继承邻域）
    for (size_t i = 0; i < n; ++i)
        if (!outlierMask[i] || spikeMask[i])
            nrm[3 * i] = nrm[3 * i + 1] = nrm[3 * i + 2] = 0.0;

    // —— 5) BFS 全局定向 ——
    pImpl_->stats.orientFlips = bfsOrient(xyz, nrm, radius);

    // —— 6) 符号修复（入口法线为参考；链 head per-point sign fix 同源）——
    size_t signFlips = 0;
    if (hasRefNormals) {
        for (size_t i = 0; i < n; ++i) {
            const double d = nrm[3*i] * refNrm[3*i] + nrm[3*i+1] * refNrm[3*i+1] + nrm[3*i+2] * refNrm[3*i+2];
            if (d < 0.0) {
                nrm[3*i] = -nrm[3*i]; nrm[3*i+1] = -nrm[3*i+1]; nrm[3*i+2] = -nrm[3*i+2];
                ++signFlips;
            }
        }
    }

    // —— 7) 补缺失法线 ——
    pImpl_->stats.fillMissing = fillMissing(xyz, nrm, radius);

    // —— 8) 法线质量门禁 ——
    size_t bad = 0;
    const double badRate = normalGate(xyz, nrm, spacing, &bad);
    pImpl_->stats.badNormalPct = badRate * 100.0;

    // —— 出口：xyz（贴回后坐标）与 normals 写回（点数不变，索引对齐）——
    io.normals.resize(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        io.xyz[i] = (float)xyz[i];
        io.normals[i] = (float)nrm[i];
    }
    pImpl_->stats.pointsOut = n;
    pImpl_->stats.totalTimeMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // —— 结果口径（§5：单一失败模型）——
    // 门禁语义＝链头 hybrid_cli 对齐：badRate > failPct 是 exit 3＝**degraded
    // 继续跑**（261121 真机首测踩中：208 万点 BFS-only 定向 bad=19.2% 被本
    // 算子误判 fail 致链中止无产物——链上该口径是降级放行，交下游+出口绕向
    // 收尾兜底，由人视产物定夺）
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "%s: pts=%zu spacing=%.4f outliers=%zu spikes=%zu bfsFlips=%zu "
                  "signFlips=%zu filled=%zu bad=%.2f%% %.0fms",
                  kLogTag, n, spacing, pImpl_->stats.outliersRemoved,
                  pImpl_->stats.spikesRemoved, pImpl_->stats.orientFlips,
                  signFlips, pImpl_->stats.fillMissing, pImpl_->stats.badNormalPct,
                  pImpl_->stats.totalTimeMs);
    r.message = msg;
    if (badRate * 100.0 > (double)prm.gateFailPct) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message += "（法线门禁超限——降级继续，产物质量由出口门与人工复核兜底）";
    } else if (badRate * 100.0 > (double)prm.gateWarnPct) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
    } else {
        r.success = true;
        r.qualityFlag = QualityFlag::Normal;
    }
    if (pImpl_->cancelled() && r.success) {
        r.qualityFlag = QualityFlag::Degraded;
        r.message += "（中途取消）";
    }
    r.statistics = pImpl_->stats;
    return r;
}

void MeshNormalEst::Destroy() {
    // CPU 先行版无外部资源；保持 idempotent（§2.5.3）
}

void MeshNormalEst::SetParams(const MeshNormalEstParams& params) {
    params.validate();
    pImpl_->params = params;
}

const MeshNormalEstParams& MeshNormalEst::GetParams() const { return pImpl_->params; }
const MeshNormalEstStats& MeshNormalEst::GetStatistics() const noexcept { return pImpl_->stats; }
void MeshNormalEst::ResetStatistics() noexcept { pImpl_->stats = MeshNormalEstStats{}; }
void MeshNormalEst::SetCancelProbe(const std::function<bool()>& probe) { pImpl_->cancelProbe = probe; }

OperatorInfo getMeshNormalEstInfo() {
    return {"MeshNormalEst", 1, 0, OperatorType::Hybrid};
}

} // namespace calib
