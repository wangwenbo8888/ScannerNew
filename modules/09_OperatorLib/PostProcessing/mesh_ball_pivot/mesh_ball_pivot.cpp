// ============================================================================
// mesh_ball_pivot.cpp — 阶段 1 实现（M1 CPU-only 版）
//
// 编排忠实移植 PCMP HybridPipeline::Run 的 K=1 路径（金标准同源）：
//   最长轴按 gpuFraction 分 A/B owner ＋ halo(2×maxR) →
//   A 块（原 GPU owner）：FastBallPivoting 单块（CPU 引擎替代 GPU，输出后
//   owner-only 过滤——与 GPU 块机制等价）→
//   B 块：ParallelBallPivoting（cpuBlocks=12，链上实际生效值；skipStitch）→
//   B 块 owner-only 过滤 → 合并＋球心预计算 → 全局缝合（tuple 去重 +
//   loadTrianglesParallel + expandBordersMultiRadius）→ 折缝剔除 → 出口。
// 单位契约 mm（疑似米制拒收）；22.7M 三角形生存天花板＝degraded。
// ============================================================================
#include "mesh_ball_pivot.h"
#include "bp_ball_pivoting.h"
#include "bp_seam_fold_remover.h"

#include <omp.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace calib {
namespace {

using Clock = std::chrono::high_resolution_clock;
using Vector3d = Eigen::Vector3d;
using Vector3i = Eigen::Vector3i;

constexpr size_t kMaxTriangles = 22700000;      // 生存天花板（链同源）
constexpr double kSpacingFallbackMm = 0.406;    // 体积法兜底（链 v5 §9.1.8）
constexpr double kSpacingWarnMm = 2.0;
constexpr size_t kSpacingSamples = 5000;
constexpr size_t kUnitSamples = 10000;

/// 点距估计（hybrid_cli estimateSpacing 逐行移植：nn<0.2×vol 用 nn；
/// vol>2.0 兜底 0.406；否则 vol；人工值直通）
double estimateSpacing(const std::vector<double>& xyz, double manual) {
    const size_t n = xyz.size() / 3;
    double mn[3] = {xyz[0], xyz[1], xyz[2]}, mx[3] = {xyz[0], xyz[1], xyz[2]};
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], xyz[3 * i + k]);
            mx[k] = std::max(mx[k], xyz[3 * i + k]);
        }
    const double volSpacing = std::cbrt(std::max(1e-300,
        std::max(0.0, mx[0] - mn[0]) * std::max(0.0, mx[1] - mn[1]) *
        std::max(0.0, mx[2] - mn[2]) / (double)std::max<size_t>(1, n)));

    // NN 中位数采样（栅格 27 邻域盒，与链 SampleGrid 同构）
    // 注意：bp::UniformGrid 持有点云指针——pts 必须与 grid 同生命周期
    std::vector<Vector3d> gridPts(n);
    for (size_t i = 0; i < n; ++i)
        gridPts[i] = Vector3d(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    bp::UniformGrid grid;
    grid.build(gridPts, volSpacing * 4.0);
    const size_t nSample = std::min(kSpacingSamples, n);
    double sumNN = 0.0;
    size_t valid = 0;
    #pragma omp parallel for reduction(+:sumNN,valid) schedule(dynamic)
    for (int64_t s = 0; s < (int64_t)nSample; ++s) {
        const size_t si = (size_t)s * n / nSample;
        const Vector3d p(xyz[3 * si], xyz[3 * si + 1], xyz[3 * si + 2]);
        std::vector<int> idx;
        std::vector<double> d2;
        grid.searchRadius(p, volSpacing * 4.0, idx, d2);
        double best = 1e300;
        for (size_t t = 0; t < idx.size(); ++t) {
            if ((size_t)idx[t] == si) continue;
            best = std::min(best, d2[t]);
        }
        if (best < 1e300) { sumNN += std::sqrt(best); ++valid; }
    }
    const double nn = (valid > 0) ? sumNN / (double)valid : 0.0;

    if (manual > 0) return manual;
    if (nn > 0 && nn < volSpacing * 0.2) return nn;   // 表面数据：NN 胜出
    if (volSpacing > kSpacingWarnMm) return kSpacingFallbackMm;
    return volSpacing;
}

/// 单位自检（毫米契约；中位最大坐标 <1 → 疑似米制）
double medianMaxAbsCoord(const std::vector<double>& xyz) {
    const size_t n = xyz.size() / 3;
    const size_t samples = std::min(kUnitSamples, n);
    std::vector<double> v;
    v.reserve(samples);
    for (size_t s = 0; s < samples; ++s) {
        const size_t i = s * n / samples;
        v.push_back(std::max(std::max(std::fabs(xyz[3 * i]),
                                      std::fabs(xyz[3 * i + 1])),
                             std::fabs(xyz[3 * i + 2])));
    }
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)(v.size() / 2), v.end());
    return v[v.size() / 2];
}

/// 球心计算（HybridPipeline::computeBallCenterGlobal 逐行移植）
bool computeBallCenterGlobal(const std::vector<Vector3d>& pts,
                             int v0, int v1, int v2, double radius,
                             Vector3d& center) {
    const Vector3d& p0 = pts[v0];
    const Vector3d ac = pts[v1] - p0, bc = pts[v2] - p0;
    Vector3d n = ac.cross(bc);
    const double nLen = n.norm();
    if (nLen < 1e-16) return false;
    n /= nLen;
    const double aLen2 = ac.squaredNorm(), bLen2 = bc.squaredNorm(), ab = ac.dot(bc);
    const double d = aLen2 * bLen2 - ab * ab;
    if (d < 1e-20) return false;
    const double s0 = (bLen2 * (aLen2 - ab)) / (2.0 * d);
    const double s1 = (aLen2 * (bLen2 - ab)) / (2.0 * d);
    const Vector3d circum = p0 + ac * s0 + bc * s1;
    const double circumR2 = (circum - p0).squaredNorm();
    const double h2 = radius * radius - circumR2;
    if (h2 < 0) return false;
    center = circum + n * std::sqrt(h2);
    return true;
}

struct BlockOutput {
    std::vector<Vector3i> triangles;
    std::vector<Vector3d> centers;
};

} // namespace

// ============================================================
// Params
// ============================================================
void MeshBallPivotParams::validate() const {
    if (radiusBase <= 0.0f)
        throw std::invalid_argument("MeshBallPivotParams: radiusBase must be > 0");
    if (radiusLevels < 1 || radiusLevels > 3)
        throw std::invalid_argument("MeshBallPivotParams: radiusLevels must be in [1,3]");
    if (gpuFraction < 0.0f || gpuFraction > 1.0f)
        throw std::invalid_argument("MeshBallPivotParams: gpuFraction must be in [0,1]");
    if (gpuBlocks < 0 || cpuBlocks < 0)
        throw std::invalid_argument("MeshBallPivotParams: blocks must be >= 0");
    if (halo <= 0.0f)
        throw std::invalid_argument("MeshBallPivotParams: halo must be > 0");
    if (manualSpacingMm < 0.0f)
        throw std::invalid_argument("MeshBallPivotParams: manualSpacingMm must be >= 0");
}

nlohmann::json MeshBallPivotParams::toJson() const {
    nlohmann::json j;
    j["radiusBase"] = radiusBase;
    j["radiusLevels"] = radiusLevels;
    j["gpuFraction"] = gpuFraction;
    j["gpuBlocks"] = gpuBlocks;
    j["cpuBlocks"] = cpuBlocks;
    j["cpuThreads"] = cpuThreads;
    j["halo"] = halo;
    j["seamRemove"] = seamRemove;
    j["seamReportOnly"] = seamReportOnly;
    j["seamMaxRemove"] = seamMaxRemove;
    j["seamMinDot"] = seamMinDot;
    j["manualSpacingMm"] = manualSpacingMm;
    return j;
}

MeshBallPivotParams MeshBallPivotParams::fromJson(const nlohmann::json& j) {
    MeshBallPivotParams p;
    auto g = [&j](const char* k, auto& v) {
        auto it = j.find(k);
        if (it != j.end()) it->get_to(v);
    };
    g("radiusBase", p.radiusBase);
    g("radiusLevels", p.radiusLevels);
    g("gpuFraction", p.gpuFraction);
    g("gpuBlocks", p.gpuBlocks);
    g("cpuBlocks", p.cpuBlocks);
    g("cpuThreads", p.cpuThreads);
    g("halo", p.halo);
    g("seamRemove", p.seamRemove);
    g("seamReportOnly", p.seamReportOnly);
    g("seamMaxRemove", p.seamMaxRemove);
    g("seamMinDot", p.seamMinDot);
    g("manualSpacingMm", p.manualSpacingMm);
    p.validate();
    return p;
}

// ============================================================
// Operator：实装（CPU-only M1）
// ============================================================
struct MeshBallPivot::Impl {
    MeshBallPivotParams params;
    MeshBallPivotStats stats;
    std::function<bool()> cancelProbe;

    bool cancelled() const { return cancelProbe && cancelProbe(); }
};

MeshBallPivot::MeshBallPivot(const MeshBallPivotParams& params) : pImpl_(std::make_unique<Impl>()) {
    params.validate();
    pImpl_->params = params;
}

MeshBallPivot::~MeshBallPivot() = default;

MeshBallPivotResult MeshBallPivot::Execute(IndexedMesh& io) {
    auto t0 = Clock::now();
    MeshBallPivotResult r;
    pImpl_->stats = MeshBallPivotStats{};
    const auto& prm = pImpl_->params;

    const size_t n = io.vertexCount();
    pImpl_->stats.trisGpu = 0; // 计数待填
    if (n < 3) {
        r.success = false;
        r.message = std::string(kLogTag) + ": 点数不足（<3）";
        return r;
    }
    if (!io.hasNormals()) {
        r.success = false;
        r.message = std::string(kLogTag) + ": 缺法线（阶段 0 应已产出权威法线）";
        return r;
    }
    if (pImpl_->cancelled()) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 执行前已取消";
        return r;
    }
    if (prm.cpuThreads > 0)
        omp_set_num_threads(prm.cpuThreads);

    // —— host double 工作区 ——
    std::vector<Vector3d> pts(n), nrm(n);
    for (size_t i = 0; i < n; ++i) {
        pts[i] = Vector3d(io.xyz[3 * i], io.xyz[3 * i + 1], io.xyz[3 * i + 2]);
        nrm[i] = Vector3d(io.normals[3 * i], io.normals[3 * i + 1], io.normals[3 * i + 2]);
    }

    // —— 单位自检（毫米契约）＋ 扁平坐标工作区 ——
    std::vector<double> flat(n * 3);
    for (size_t i = 0; i < n * 3; ++i) flat[i] = io.xyz[i];
    if (medianMaxAbsCoord(flat) < 1.0) {
        r.success = false;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 单位疑似米制（链契约为毫米，请 ×1000）";
        return r;
    }

    // —— 点距与半径序列 ——
    const double spacing = estimateSpacing(flat, (double)prm.manualSpacingMm);
    const double kRadMults[3] = {1.0, 2.0, 4.0};
    std::vector<double> radii;
    for (int i = 0; i < prm.radiusLevels; ++i)
        radii.push_back(kRadMults[i] * (double)prm.radiusBase * spacing);
    const double maxR = *std::max_element(radii.begin(), radii.end());
    pImpl_->stats.spacingMm = (float)spacing;

    // —— K=1 分区：最长轴按 gpuFraction 分 A/B owner ＋ halo ——
    const int N = (int)n;
    double mn[3] = {pts[0].x(), pts[0].y(), pts[0].z()};
    double mx[3] = {mn[0], mn[1], mn[2]};
    for (const auto& p : pts) {
        mn[0] = std::min(mn[0], p.x()); mx[0] = std::max(mx[0], p.x());
        mn[1] = std::min(mn[1], p.y()); mx[1] = std::max(mx[1], p.y());
        mn[2] = std::min(mn[2], p.z()); mx[2] = std::max(mx[2], p.z());
    }
    const double ex = mx[0] - mn[0], ey = mx[1] - mn[1], ez = mx[2] - mn[2];
    const int axis = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);

    std::vector<int> sortedIdx(N);
    for (int i = 0; i < N; ++i) sortedIdx[i] = i;
    std::sort(sortedIdx.begin(), sortedIdx.end(),
              [&](int a, int b) { return pts[a](axis) < pts[b](axis); });

    const int aOwnerEnd = static_cast<int>((double)N * (double)prm.gpuFraction);
    std::vector<char> isAOwner(N, 0), isBOwner(N, 0);
    for (int i = 0; i < aOwnerEnd; ++i) isAOwner[sortedIdx[i]] = 1;
    for (int i = aOwnerEnd; i < N; ++i) isBOwner[sortedIdx[i]] = 1;

    const double haloRadius = (double)prm.halo * maxR;
    auto expandHalo = [&](const std::vector<char>& ownerFlag,
                          std::vector<int>& outIdx, std::vector<char>& outMask) {
        double lo = 1e30, hi = -1e30;
        for (int i = 0; i < N; ++i)
            if (ownerFlag[i]) {
                lo = std::min(lo, pts[i](axis));
                hi = std::max(hi, pts[i](axis));
            }
        const double extLo = lo - haloRadius, extHi = hi + haloRadius;
        for (int i = 0; i < N; ++i) {
            const double c = pts[i](axis);
            if (ownerFlag[i]) { outIdx.push_back(i); outMask.push_back(1); }
            else if (c >= extLo && c <= extHi) { outIdx.push_back(i); outMask.push_back(0); }
        }
    };
    std::vector<int> aIdx, bIdx;
    std::vector<char> aMask, bMask;
    expandHalo(isAOwner, aIdx, aMask);
    expandHalo(isBOwner, bIdx, bMask);

    // —— A/B 块并行执行（A=单块 CPU 引擎＋输出后 owner 过滤；B=多块 PBP＋owner 过滤）——
    BlockOutput aOut, bOut;
    auto runA = [&]() {
        const size_t M = aIdx.size();
        if (M < 3) return;
        std::vector<Vector3d> subPts(M), subNrm(M);
        for (size_t i = 0; i < M; ++i) {
            subPts[i] = pts[aIdx[i]];
            subNrm[i] = nrm[aIdx[i]];
        }
        bp::FastBallPivoting fbp(subPts, subNrm);
        fbp.setUseGrid(true);
        fbp.Run(radii);
        const auto& local = fbp.triangles();
        for (const auto& tri : local) {
            const int l0 = tri(0), l1 = tri(1), l2 = tri(2);
            if (aMask[l0] && aMask[l1] && aMask[l2]) {   // owner-only 过滤（GPU 块机制）
                const int g0 = aIdx[l0], g1 = aIdx[l1], g2 = aIdx[l2];
                aOut.triangles.emplace_back(g0, g1, g2);
                Vector3d c;
                computeBallCenterGlobal(pts, g0, g1, g2, maxR, c);
                aOut.centers.push_back(c);
            }
        }
    };
    auto runB = [&]() {
        const size_t M = bIdx.size();
        if (M < 3) return;
        std::vector<Vector3d> subPts(M), subNrm(M);
        for (size_t i = 0; i < M; ++i) {
            subPts[i] = pts[bIdx[i]];
            subNrm[i] = nrm[bIdx[i]];
        }
        bp::ParallelBallPivoting pbp(subPts, subNrm);
        pbp.setUseGrid(true);
        bp::ParallelBallPivoting::Params pp;
        pp.numBlocks = prm.cpuBlocks;      // 链 K=1 路径实际生效值 12
        pp.numThreads = prm.cpuThreads;
        pp.haloFactor = 2.0;
        pp.skipStitch = true;
        pbp.Run(radii, pp);
        const auto& local = pbp.triangles();
        for (const auto& tri : local) {
            const int g0 = bIdx[tri(0)], g1 = bIdx[tri(1)], g2 = bIdx[tri(2)];
            if (g0 < N && g1 < N && g2 < N &&
                isBOwner[g0] && isBOwner[g1] && isBOwner[g2]) {
                bOut.triangles.emplace_back(g0, g1, g2);
                Vector3d c;
                computeBallCenterGlobal(pts, g0, g1, g2, maxR, c);
                bOut.centers.push_back(c);
            }
        }
    };
    {
        std::thread tA(runA);
        std::thread tB(runB);
        tA.join();
        tB.join();
    }
    if (pImpl_->cancelled()) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 中途取消";
        return r;
    }
    pImpl_->stats.trisGpu = aOut.triangles.size();   // A 块（原 GPU owner 段）
    pImpl_->stats.trisCpu = bOut.triangles.size();

    // —— 合并 ＋ 缝合（tuple 去重 → loadTrianglesParallel → 多半径扩边）——
    std::vector<Vector3i> allTris, dedupTris;
    std::vector<Vector3d> allCenters, dedupCenters;
    allTris.reserve(aOut.triangles.size() + bOut.triangles.size());
    allCenters.reserve(allTris.capacity());
    allTris.insert(allTris.end(), aOut.triangles.begin(), aOut.triangles.end());
    allTris.insert(allTris.end(), bOut.triangles.begin(), bOut.triangles.end());
    allCenters.insert(allCenters.end(), aOut.centers.begin(), aOut.centers.end());
    allCenters.insert(allCenters.end(), bOut.centers.begin(), bOut.centers.end());
    const size_t beforeDedup = allTris.size();

    {
        using TriKey = std::tuple<int, int, int>;
        std::vector<std::pair<TriKey, int>> keyed;
        keyed.reserve(allTris.size());
        for (size_t i = 0; i < allTris.size(); ++i) {
            const auto& t = allTris[i];
            int v0 = t(0), v1 = t(1), v2 = t(2);
            if (v0 > v1) std::swap(v0, v1);
            if (v1 > v2) std::swap(v1, v2);
            if (v0 > v1) std::swap(v0, v1);
            keyed.emplace_back(std::make_tuple(v0, v1, v2), (int)i);
        }
        std::sort(keyed.begin(), keyed.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        keyed.erase(std::unique(keyed.begin(), keyed.end(),
                                [](const auto& a, const auto& b) { return a.first == b.first; }),
                    keyed.end());
        dedupTris.reserve(keyed.size());
        dedupCenters.reserve(keyed.size());
        for (const auto& k : keyed) {
            dedupTris.push_back(allTris[k.second]);
            dedupCenters.push_back(allCenters[k.second]);
        }
    }
    pImpl_->stats.dedupRemoved = beforeDedup - dedupTris.size();

    std::vector<Vector3i> stitched;
    {
        bp::FastBallPivoting stitcher(pts, nrm);
        stitcher.buildGrid(maxR);
        stitcher.loadTrianglesParallel(dedupTris, maxR, dedupCenters);
        stitcher.expandBordersMultiRadius(radii);
        stitched = stitcher.triangles();
    }
    pImpl_->stats.trisStitched = stitched.size();

    // —— 折缝剔除（只删三角形不动顶点）——
    if (prm.seamRemove) {
        bp::seamfold::SeamFoldParams sp;
        sp.report_only = prm.seamReportOnly;
        sp.min_dot = (double)prm.seamMinDot;
        sp.max_remove = prm.seamMaxRemove;
        sp.ref_eps = 1e-12;
        sp.cross_eps = 1e-12;
        sp.debug = false;
        auto st = bp::seamfold::removeSeamFolds(pts, nrm, stitched, sp);
        pImpl_->stats.seamCandidates = (size_t)st.bands;
        pImpl_->stats.seamRemoved = (size_t)(st.del + st.nm_del);
    }

    // —— 出口 ——
    io.triangles.clear();
    io.triangles.reserve(stitched.size() * 3);
    for (const auto& t : stitched) {
        io.triangles.push_back((uint32_t)t(0));
        io.triangles.push_back((uint32_t)t(1));
        io.triangles.push_back((uint32_t)t(2));
    }
    pImpl_->stats.totalTimeMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    char msg[320];
    std::snprintf(msg, sizeof(msg),
                  "%s: pts=%zu spacing=%.4f radii=%.3f/%.3f trisA=%zu trisB=%zu "
                  "dedup=%zu stitched=%zu seam(bands=%zu del=%zu) %.0fms",
                  kLogTag, n, spacing, radii.front(), radii.back(),
                  pImpl_->stats.trisGpu, pImpl_->stats.trisCpu,
                  pImpl_->stats.dedupRemoved, pImpl_->stats.trisStitched,
                  pImpl_->stats.seamCandidates, pImpl_->stats.seamRemoved,
                  pImpl_->stats.totalTimeMs);
    r.message = msg;

    if (stitched.empty()) {
        r.success = false;
        r.qualityFlag = QualityFlag::Degraded;
    } else if (stitched.size() > kMaxTriangles) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message += "（超 22.7M 生存天花板，规模受限降级）";
    } else {
        r.success = true;
        r.qualityFlag = QualityFlag::Normal;
    }
    r.statistics = pImpl_->stats;
    return r;
}

void MeshBallPivot::Destroy() {
    // CPU-only 版无外部资源；保持 idempotent（§2.5.3）
}

void MeshBallPivot::SetParams(const MeshBallPivotParams& params) {
    params.validate();
    pImpl_->params = params;
}

const MeshBallPivotParams& MeshBallPivot::GetParams() const { return pImpl_->params; }
const MeshBallPivotStats& MeshBallPivot::GetStatistics() const noexcept { return pImpl_->stats; }
void MeshBallPivot::ResetStatistics() noexcept { pImpl_->stats = MeshBallPivotStats{}; }
void MeshBallPivot::SetCancelProbe(const std::function<bool()>& probe) { pImpl_->cancelProbe = probe; }

OperatorInfo getMeshBallPivotInfo() {
    return {"MeshBallPivot", 1, 0, OperatorType::Hybrid};
}

} // namespace calib
