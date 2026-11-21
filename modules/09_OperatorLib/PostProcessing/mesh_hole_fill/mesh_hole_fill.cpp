// ============================================================================
// mesh_hole_fill.cpp — 阶段 2 实现（M2，PCMP MeshHoleFillerPro 机械移植适配）
//
// 库本体：mhf/（沙盒 MeshHoleFillerPro 机械拷贝，命名空间保留
// hole_filler_pro——09 仓"3DSCANNER 机械迁入"同款口径）。Open3D 剥离：
// fillHolesFile 文件壳层已删，本适配层直接调内存 API fillHoles()；
// 标志点经 FillParams.marker_list_mem 内存直取（marker_cloud_fuse 输出同构）；
// 嵌入法线搬运（精确匹配＋栅格最近邻）自 fillHolesFile 移植于此。
// R1 重试语义＝链 CLI（仅水密门触发，保守方向：trim-1 / edges 减半）。
// ============================================================================
#include "mesh_hole_fill.h"

#include "hole_filler_pro/hole_filler_pro.h"
#include "hole_filler_pro/mesh_entry_cleaner.h"
#include "hole_filler_pro/marker_cutter.h"
#include "hole_filler_pro/mesh_types.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace calib {
namespace {

namespace hfp = ::hole_filler_pro;   // 命名空间别名（库命名空间按 09 机械迁入惯例保留）
using Clock = std::chrono::high_resolution_clock;

/// MeshHoleFillParams → 库 FillParams（字段一一对应；链实测口径默认值）
hfp::FillParams toFillParams(const MeshHoleFillParams& p) {
    hfp::FillParams f;
    f.neighborhood_rings = p.neighborhoodRings;
    f.density_remesh_passes = p.densityRemeshPasses;
    f.split_factor = (double)p.splitFactor;
    f.collapse_factor = (double)p.collapseFactor;
    f.fairing_order = p.fairingOrder;
    f.max_hole_edges = p.maxHoleEdges;
    f.max_hole_perimeter = (double)p.maxHolePerimeterMm;
    f.trim_rings = p.trimRings;
    f.entry_clean = p.entryClean;
    f.entry_bowtie_max_iters = p.entryBowtieMaxIters;
    f.entry_small_cc_min_faces = p.entrySmallCcMinFaces;
    f.entry_small_cc_area_ratio = (double)p.entrySmallCcAreaRatio;
    f.entry_flap_peel = p.entryFlapPeel;
    f.entry_spike_remove = p.entrySpikeRemove;
    f.entry_spike_ratio = (double)p.entrySpikeRatio;
    f.entry_spike_ring = p.entrySpikeRing;
    f.entry_veil_remove = p.entryVeilRemove;
    f.entry_veil_edge_mult = (double)p.entryVeilEdgeMult;
    f.s1hash = p.s1Hash;
    f.nrmhash = p.nrmHash;
    f.edge_unify = p.edgeUnify;
    f.exit_nm_audit = p.exitNmAudit;
    f.exit_nm_prune = p.exitNmPrune;
    f.component_exempt_closed = p.componentExemptClosed;
    f.arc_stitch = p.arcStitch;
    f.fill_surface_guided = p.fillSurfaceGuided;
    f.fill_guide_perimeter_min = (double)p.fillGuidePerimeterMinMm;
    f.fill_guide_lambda_alpha = (double)p.fillGuideLambdaAlpha;
    f.fill_guide_band_width = (double)p.fillGuideBandWidthMm;
    f.fill_front = p.fillFront;
    f.fill_front_flip = p.fillFrontFlip;
    f.fill_front_perimeter_min = (double)p.fillFrontPerimeterMinMm;
    f.remesh_flip = p.remeshFlip;
    f.rim_soft_anchor = p.rimSoftAnchor;
    f.rim_gradient = p.rimGradient;
    f.rim_gradient_rings = (double)p.rimGradientRings;
    f.rim_gradient_ratio = (double)p.rimGradientRatio;
    f.remesh_collapse = p.remeshCollapse;
    f.rim_lift = p.rimLift;
    f.marker_cut = p.markerCut;
    f.marker_diameter = (double)p.markerDiameterMm;
    f.marker_radius_scale = (double)p.markerRadiusScale;
    return f;
}

} // namespace

// ============================================================
// Params
// ============================================================
void MeshHoleFillParams::validate() const {
    if (entrySmallCcMinFaces < 0)
        throw std::invalid_argument("MeshHoleFillParams: entrySmallCcMinFaces must be >= 0");
    if (entryBowtieMaxIters < 0)
        throw std::invalid_argument("MeshHoleFillParams: entryBowtieMaxIters must be >= 0");
    if (entryVeilEdgeMult <= 0.0f)
        throw std::invalid_argument("MeshHoleFillParams: entryVeilEdgeMult must be > 0");
    if (neighborhoodRings < 1)
        throw std::invalid_argument("MeshHoleFillParams: neighborhoodRings must be >= 1");
    if (densityRemeshPasses < 0)
        throw std::invalid_argument("MeshHoleFillParams: densityRemeshPasses must be >= 0");
    if (fairingOrder < 0 || fairingOrder > 4)
        throw std::invalid_argument("MeshHoleFillParams: fairingOrder must be in [0,4]");
    if (maxHoleEdges < 1)
        throw std::invalid_argument("MeshHoleFillParams: maxHoleEdges must be >= 1");
    if (maxHolePerimeterMm < 0.0f)
        throw std::invalid_argument("MeshHoleFillParams: maxHolePerimeterMm must be >= 0");
    if (trimRings < 0 || trimRings > 4)
        throw std::invalid_argument("MeshHoleFillParams: trimRings must be in [0,4]");
    if (markerDiameterMm <= 0.0f)
        throw std::invalid_argument("MeshHoleFillParams: markerDiameterMm must be > 0");
    if (markerRadiusScale <= 0.0f)
        throw std::invalid_argument("MeshHoleFillParams: markerRadiusScale must be > 0");
}

nlohmann::json MeshHoleFillParams::toJson() const {
    nlohmann::json j;
    j["entryClean"] = entryClean; j["entrySmallCcMinFaces"] = entrySmallCcMinFaces;
    j["entrySmallCcAreaRatio"] = entrySmallCcAreaRatio; j["entryBowtieMaxIters"] = entryBowtieMaxIters;
    j["entryFlapPeel"] = entryFlapPeel; j["entrySpikeRemove"] = entrySpikeRemove;
    j["entrySpikeRatio"] = entrySpikeRatio; j["entrySpikeRing"] = entrySpikeRing;
    j["entryVeilRemove"] = entryVeilRemove; j["entryVeilEdgeMult"] = entryVeilEdgeMult;
    j["neighborhoodRings"] = neighborhoodRings; j["densityRemeshPasses"] = densityRemeshPasses;
    j["splitFactor"] = splitFactor; j["collapseFactor"] = collapseFactor;
    j["fairingOrder"] = fairingOrder; j["maxHoleEdges"] = maxHoleEdges;
    j["maxHolePerimeterMm"] = maxHolePerimeterMm; j["trimRings"] = trimRings;
    j["exitNmAudit"] = exitNmAudit; j["exitNmPrune"] = exitNmPrune;
    j["componentExemptClosed"] = componentExemptClosed; j["arcStitch"] = arcStitch;
    j["s1Hash"] = s1Hash; j["nrmHash"] = nrmHash; j["edgeUnify"] = edgeUnify;
    j["fillSurfaceGuided"] = fillSurfaceGuided;
    j["fillGuidePerimeterMinMm"] = fillGuidePerimeterMinMm;
    j["fillGuideLambdaAlpha"] = fillGuideLambdaAlpha;
    j["fillGuideBandWidthMm"] = fillGuideBandWidthMm;
    j["fillFront"] = fillFront; j["fillFrontFlip"] = fillFrontFlip;
    j["fillFrontPerimeterMinMm"] = fillFrontPerimeterMinMm;
    j["remeshFlip"] = remeshFlip; j["rimSoftAnchor"] = rimSoftAnchor;
    j["rimGradient"] = rimGradient; j["rimGradientRings"] = rimGradientRings;
    j["rimGradientRatio"] = rimGradientRatio; j["remeshCollapse"] = remeshCollapse;
    j["rimLift"] = rimLift;
    j["markerCut"] = markerCut; j["markerDiameterMm"] = markerDiameterMm;
    j["markerRadiusScale"] = markerRadiusScale;
    j["numThreads"] = numThreads;
    return j;
}

MeshHoleFillParams MeshHoleFillParams::fromJson(const nlohmann::json& j) {
    MeshHoleFillParams p;
    auto g = [&j](const char* k, auto& v) {
        auto it = j.find(k);
        if (it != j.end()) it->get_to(v);
    };
    g("entryClean", p.entryClean); g("entrySmallCcMinFaces", p.entrySmallCcMinFaces);
    g("entrySmallCcAreaRatio", p.entrySmallCcAreaRatio); g("entryBowtieMaxIters", p.entryBowtieMaxIters);
    g("entryFlapPeel", p.entryFlapPeel); g("entrySpikeRemove", p.entrySpikeRemove);
    g("entrySpikeRatio", p.entrySpikeRatio); g("entrySpikeRing", p.entrySpikeRing);
    g("entryVeilRemove", p.entryVeilRemove); g("entryVeilEdgeMult", p.entryVeilEdgeMult);
    g("neighborhoodRings", p.neighborhoodRings); g("densityRemeshPasses", p.densityRemeshPasses);
    g("splitFactor", p.splitFactor); g("collapseFactor", p.collapseFactor);
    g("fairingOrder", p.fairingOrder); g("maxHoleEdges", p.maxHoleEdges);
    g("maxHolePerimeterMm", p.maxHolePerimeterMm); g("trimRings", p.trimRings);
    g("exitNmAudit", p.exitNmAudit); g("exitNmPrune", p.exitNmPrune);
    g("componentExemptClosed", p.componentExemptClosed); g("arcStitch", p.arcStitch);
    g("s1Hash", p.s1Hash); g("nrmHash", p.nrmHash); g("edgeUnify", p.edgeUnify);
    g("fillSurfaceGuided", p.fillSurfaceGuided);
    g("fillGuidePerimeterMinMm", p.fillGuidePerimeterMinMm);
    g("fillGuideLambdaAlpha", p.fillGuideLambdaAlpha);
    g("fillGuideBandWidthMm", p.fillGuideBandWidthMm);
    g("fillFront", p.fillFront); g("fillFrontFlip", p.fillFrontFlip);
    g("fillFrontPerimeterMinMm", p.fillFrontPerimeterMinMm);
    g("remeshFlip", p.remeshFlip); g("rimSoftAnchor", p.rimSoftAnchor);
    g("rimGradient", p.rimGradient); g("rimGradientRings", p.rimGradientRings);
    g("rimGradientRatio", p.rimGradientRatio); g("remeshCollapse", p.remeshCollapse);
    g("rimLift", p.rimLift);
    g("markerCut", p.markerCut); g("markerDiameterMm", p.markerDiameterMm);
    g("markerRadiusScale", p.markerRadiusScale);
    g("numThreads", p.numThreads);
    p.validate();
    return p;
}

// ============================================================
// Operator：实装（mhf 内存 API 适配）
// ============================================================
struct MeshHoleFill::Impl {
    MeshHoleFillParams params;
    MeshHoleFillStats stats;
    std::function<bool()> cancelProbe;

    bool cancelled() const { return cancelProbe && cancelProbe(); }
};

MeshHoleFill::MeshHoleFill(const MeshHoleFillParams& params) : pImpl_(std::make_unique<Impl>()) {
    params.validate();
    pImpl_->params = params;
}

MeshHoleFill::~MeshHoleFill() = default;

MeshHoleFillResult MeshHoleFill::Execute(IndexedMesh& io,
                                         const std::vector<MarkerHoleRef>& markers) {
    auto t0 = Clock::now();
    MeshHoleFillResult r;
    pImpl_->stats = MeshHoleFillStats{};
    const auto& prm = pImpl_->params;

    const size_t nv = io.vertexCount();
    const size_t nf = io.triangleCount();
    if (nv < 3 || nf < 1) {
        r.success = false;
        r.message = std::string(kLogTag) + ": 输入网格为空（应先经阶段 1 封装）";
        return r;
    }
    if (pImpl_->cancelled()) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 执行前已取消";
        return r;
    }

    // —— IndexedMesh → 库矩阵 ——
    Eigen::MatrixXd V((int)nv, 3), Vout;
    Eigen::MatrixXi F((int)nf, 3), Fout;
    for (size_t i = 0; i < nv; ++i) {
        V((int)i, 0) = io.xyz[3 * i];
        V((int)i, 1) = io.xyz[3 * i + 1];
        V((int)i, 2) = io.xyz[3 * i + 2];
    }
    for (size_t i = 0; i < nf; ++i) {
        F((int)i, 0) = (int)io.triangles[3 * i];
        F((int)i, 1) = (int)io.triangles[3 * i + 1];
        F((int)i, 2) = (int)io.triangles[3 * i + 2];
    }

    // —— 标志点内存直取 ——
    std::vector<hfp::MarkerSpec> memMarkers;
    if (prm.markerCut && !markers.empty()) {
        memMarkers.reserve(markers.size());
        for (const auto& m : markers) {
            hfp::MarkerSpec s;
            s.center = Eigen::Vector3d(m.x, m.y, m.z);
            s.normal = Eigen::Vector3d(m.nx, m.ny, m.nz);
            if (s.normal.norm() > 1e-12) s.normal.normalize();
            memMarkers.push_back(s);
        }
    }

    hfp::FillParams fp = toFillParams(prm);
    fp.marker_list_mem = &memMarkers;

    // —— 水密门预检计数（进门口径）——
    const int pre = (int)hfp::detectBoundaries(V, F).size();
    pImpl_->stats.watertightPre = pre;
    pImpl_->stats.loopsDetected = pre;

    // —— 主填充 ＋ R1 重试（链 CLI 语义：仅水密门触发，一次，保守方向）——
    hfp::EntryCleanReport clr;
    int wf = 0;
    bool ok = hfp::fillHoles(V, F, Vout, Fout, fp, &clr, &wf);
    if (!ok && wf == 1) {
        hfp::FillParams p2 = fp;
        if (p2.trim_rings > 0) p2.trim_rings -= 1;
        if (p2.max_hole_edges > 1) p2.max_hole_edges /= 2;
        ++pImpl_->stats.retryR1;
        int wf2 = 0;
        ok = hfp::fillHoles(V, F, Vout, Fout, p2, nullptr, &wf2);
        wf = wf2;
    }
    if (!ok) {
        r.success = false;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": fillHoles 失败（参数/内部错误）";
        return r;
    }

    const int post = (int)hfp::detectBoundaries(Vout, Fout).size();
    pImpl_->stats.watertightPost = post;
    // holesFilled 代理口径：pre−post（精确分项计数留库遥测接入时细化）
    pImpl_->stats.holesFilled = std::max(0, pre - post);
    pImpl_->stats.openArcs = post;
    pImpl_->stats.markersCut = prm.markerCut ? (int)memMarkers.size() : 0;

    // —— 嵌入法线搬运（fillHolesFile 移植：精确匹配＋栅格最近邻借法线）——
    const bool hasInNormals = io.hasNormals();
    std::vector<double> outNrm;
    if (hasInNormals) {
        const int nIn = (int)nv, nOut = (int)Vout.rows();
        auto makeKey = [](double x, double y, double z) {
            return std::make_tuple(x, y, z);
        };
        struct H {
            std::size_t operator()(const std::tuple<double, double, double>& k) const {
                std::uint64_t h = 1469598103934665603ULL;
                const double d[3] = { std::get<0>(k), std::get<1>(k), std::get<2>(k) };
                for (double x : d) {
                    std::uint64_t b;
                    std::memcpy(&b, &x, 8);
                    h ^= b;
                    h *= 1099511628211ULL;
                }
                h += 0x9E3779B97F4A7C15ULL;
                h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
                h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
                h ^= (h >> 31);
                return static_cast<std::size_t>(h);
            }
        };
        std::unordered_map<std::tuple<double, double, double>, int, H> exactMap(10);
        exactMap.reserve(nIn * 2);
        const double cellSz = 8.0;
        std::unordered_map<long long, std::vector<int>> grid;
        for (int i = 0; i < nIn; ++i) {
            exactMap.emplace(makeKey(V(i, 0), V(i, 1), V(i, 2)), i);
            long long ck = (long long)std::floor(V(i, 0) / cellSz) * 73856093LL ^
                           (long long)std::floor(V(i, 1) / cellSz) * 19349663LL ^
                           (long long)std::floor(V(i, 2) / cellSz) * 83492791LL;
            grid[ck].push_back(i);
        }
        outNrm.resize((size_t)nOut * 3);
        for (int i = 0; i < nOut; ++i) {
            const Eigen::Vector3d p(Vout(i, 0), Vout(i, 1), Vout(i, 2));
            auto it = exactMap.find(makeKey(p.x(), p.y(), p.z()));
            int src = -1;
            if (it != exactMap.end()) {
                src = it->second;                        // 原顶点：坐标精确一致
            } else {
                long long cxk = (long long)std::floor(p.x() / cellSz),
                          cyk = (long long)std::floor(p.y() / cellSz),
                          czk = (long long)std::floor(p.z() / cellSz);
                double bestD = 1e30;
                for (int rr = 0; rr <= 24 && src < 0; ++rr) {
                    for (long long dx = -rr; dx <= rr; ++dx)
                    for (long long dy = -rr; dy <= rr; ++dy)
                    for (long long dz = -rr; dz <= rr; ++dz) {
                        if (std::max(std::llabs(dx), std::max(std::llabs(dy), std::llabs(dz))) != rr) continue;
                        long long ck = (cxk + dx) * 73856093LL ^ (cyk + dy) * 19349663LL ^ (czk + dz) * 83492791LL;
                        auto git = grid.find(ck);
                        if (git == grid.end()) continue;
                        for (int gi : git->second) {
                            const Eigen::Vector3d q(V(gi, 0), V(gi, 1), V(gi, 2));
                            double d = (q - p).squaredNorm();
                            if (d < bestD) { bestD = d; src = gi; }
                        }
                    }
                }
            }
            if (src >= 0) {
                outNrm[3 * i] = io.normals[3 * src];
                outNrm[3 * i + 1] = io.normals[3 * src + 1];
                outNrm[3 * i + 2] = io.normals[3 * src + 2];
            }
        }
    }

    // —— 库产物 → IndexedMesh（顶点数允许增长：补丁新增顶点，R1 索引网格口径）——
    const int onv = (int)Vout.rows();
    const int onf = (int)Fout.rows();
    io.xyz.resize((size_t)onv * 3);
    for (int i = 0; i < onv; ++i) {
        io.xyz[3 * i] = (float)Vout(i, 0);
        io.xyz[3 * i + 1] = (float)Vout(i, 1);
        io.xyz[3 * i + 2] = (float)Vout(i, 2);
    }
    if (hasInNormals) {
        io.normals.assign(outNrm.begin(), outNrm.end());
    } else {
        io.normals.clear();   // 无法搬运时清空（下游 SR 需法线时应回退拓扑定向）
    }
    io.triangles.clear();
    io.triangles.reserve((size_t)onf * 3);
    for (int i = 0; i < onf; ++i) {
        io.triangles.push_back((uint32_t)Fout(i, 0));
        io.triangles.push_back((uint32_t)Fout(i, 1));
        io.triangles.push_back((uint32_t)Fout(i, 2));
    }
    pImpl_->stats.totalTimeMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "%s: in(v=%zu f=%zu) out(v=%d f=%d) loops=%d filled=%d post=%d "
                  "markers=%d retryR1=%d %.0fms",
                  kLogTag, nv, nf, onv, onf, pre,
                  (int)pImpl_->stats.holesFilled, post,
                  (int)pImpl_->stats.markersCut, (int)pImpl_->stats.retryR1,
                  pImpl_->stats.totalTimeMs);
    r.message = msg;

    // 门禁口径（验收协议 v1.0：post ≤ max(10, 5%·pre)）
    const int gate = std::max(10, pre / 20);
    if (post > gate) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message += "（水密门超限）";
    } else {
        r.success = true;
        r.qualityFlag = QualityFlag::Normal;
    }
    r.statistics = pImpl_->stats;
    return r;
}

void MeshHoleFill::Destroy() {
    // 无外部资源；保持 idempotent（§2.5.3）
}

void MeshHoleFill::SetParams(const MeshHoleFillParams& params) {
    params.validate();
    pImpl_->params = params;
}

const MeshHoleFillParams& MeshHoleFill::GetParams() const { return pImpl_->params; }
const MeshHoleFillStats& MeshHoleFill::GetStatistics() const noexcept { return pImpl_->stats; }
void MeshHoleFill::ResetStatistics() noexcept { pImpl_->stats = MeshHoleFillStats{}; }
void MeshHoleFill::SetCancelProbe(const std::function<bool()>& probe) { pImpl_->cancelProbe = probe; }

OperatorInfo getMeshHoleFillInfo() {
    return {"MeshHoleFill", 1, 0, OperatorType::CPU};
}

} // namespace calib
