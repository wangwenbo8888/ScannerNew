// ============================================================================
// mesh_smooth_sr.cpp — 阶段 3 实现（M3，smooth_reconstruction 机械移植适配）
//
// 库本体：sr/（沙盒 smooth_reconstruction/cpp/src 核心五件机械拷贝，包
// namespace sr——原全局类型按 §7.2 收编；C1 清洗：零友商商标字样）。
// 链路径（smooth3 子命令语义）：
//   共享边表（topo_share=1）→ 定向（链恒 --skip-topo-orient：拓扑遍跳过，
//   嵌入法线参考 orientByReferenceEmbedded 为权威——R2）→ runSmoothV3
//   （三遍扫描投票）→ optimizeEdges（调线六关卡，1 pass）→ 出口。
// ⚠ 合规前置门（移植计划 §6-C2）：参数标定值法务裁定前不入发布分支。
// ============================================================================
#include "mesh_smooth_sr.h"

#include "sr/pipeline.hpp"
#include "sr/orient.hpp"
#include "sr/edge_table.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace calib {
namespace {

using Clock = std::chrono::high_resolution_clock;

} // namespace

// ============================================================
// Params
// ============================================================
void MeshSmoothSrParams::validate() const {
    if (ring < 1)
        throw std::invalid_argument("MeshSmoothSrParams: ring must be >= 1");
    if (angleDeg < 0.0f || angleDeg > 90.0f)
        throw std::invalid_argument("MeshSmoothSrParams: angleDeg must be in [0,90]");
    if (angle2Deg < 0.0f || angle2Deg > angleDeg)
        throw std::invalid_argument("MeshSmoothSrParams: require 0 <= angle2Deg <= angleDeg");
    if (toleranceMm <= 0.0f)
        throw std::invalid_argument("MeshSmoothSrParams: toleranceMm must be > 0");
    if (outer < 1)
        throw std::invalid_argument("MeshSmoothSrParams: outer must be >= 1");
    if (coplanarDeg < 0.0f || coplanarDeg > 90.0f)
        throw std::invalid_argument("MeshSmoothSrParams: coplanarDeg must be in [0,90]");
}

nlohmann::json MeshSmoothSrParams::toJson() const {
    nlohmann::json j;
    j["ring"] = ring;
    j["angleDeg"] = angleDeg;
    j["angle2Deg"] = angle2Deg;
    j["toleranceMm"] = toleranceMm;
    j["outer"] = outer;
    j["enableOrient"] = enableOrient;
    j["enableFlip"] = enableFlip;
    j["coplanarDeg"] = coplanarDeg;
    j["flipNmGuard"] = flipNmGuard;
    j["numThreads"] = numThreads;
    return j;
}

MeshSmoothSrParams MeshSmoothSrParams::fromJson(const nlohmann::json& j) {
    MeshSmoothSrParams p;
    auto g = [&j](const char* k, auto& v) {
        auto it = j.find(k);
        if (it != j.end()) it->get_to(v);
    };
    g("ring", p.ring);
    g("angleDeg", p.angleDeg);
    g("angle2Deg", p.angle2Deg);
    g("toleranceMm", p.toleranceMm);
    g("outer", p.outer);
    g("enableOrient", p.enableOrient);
    g("enableFlip", p.enableFlip);
    g("coplanarDeg", p.coplanarDeg);
    g("flipNmGuard", p.flipNmGuard);
    g("numThreads", p.numThreads);
    p.validate();
    return p;
}

// ============================================================
// Operator：实装（sr 库适配，链 smooth3 语义）
// ============================================================
struct MeshSmoothSr::Impl {
    MeshSmoothSrParams params;
    MeshSmoothSrStats stats;
    std::function<bool()> cancelProbe;

    bool cancelled() const { return cancelProbe && cancelProbe(); }
};

MeshSmoothSr::MeshSmoothSr(const MeshSmoothSrParams& params) : pImpl_(std::make_unique<Impl>()) {
    params.validate();
    pImpl_->params = params;
}

MeshSmoothSr::~MeshSmoothSr() = default;

MeshSmoothSrResult MeshSmoothSr::Execute(IndexedMesh& io) {
    auto t0 = Clock::now();
    MeshSmoothSrResult r;
    pImpl_->stats = MeshSmoothSrStats{};
    const auto& prm = pImpl_->params;

    const size_t nv = io.vertexCount();
    const size_t nf = io.triangleCount();
    pImpl_->stats.trisIn = nf;
    if (nv < 3 || nf < 1) {
        r.success = false;
        r.message = std::string(kLogTag) + ": 输入网格为空（应先经阶段 2 补洞）";
        return r;
    }
    if (pImpl_->cancelled()) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message = std::string(kLogTag) + ": 执行前已取消";
        return r;
    }
    if (prm.numThreads > 0)
        omp_set_num_threads(prm.numThreads);

    // —— IndexedMesh → sr::Mesh ——
    sr::Mesh mesh;
    mesh.numVerts = nv;
    mesh.numTris = nf;
    mesh.verts.resize(nv * 3);
    for (size_t i = 0; i < nv * 3; ++i) mesh.verts[i] = (double)io.xyz[i];
    mesh.triangles = io.triangles;                       // uint32 直通
    const bool hasEmbedded = io.hasNormals();
    if (hasEmbedded) {
        mesh.vnormals.resize(nv * 3);
        for (size_t i = 0; i < nv * 3; ++i) mesh.vnormals[i] = (double)io.normals[i];
    }

    // —— 共享边表（链 topo_share=1：一次构建，定向读＋调线首遍用）——
    auto tTab0 = Clock::now();
    sr::EdgeTable etab = sr::buildEdgeTable(mesh.triangles.data(), nf);
    pImpl_->stats.totalMsOrient =
        std::chrono::duration<double, std::milli>(Clock::now() - tTab0).count();

    // —— 定向（链 --skip-topo-orient：拓扑遍恒跳过；嵌入法线参考为权威 R2）——
    auto tOr0 = Clock::now();
    if (hasEmbedded) {
        pImpl_->stats.orientFlips =
            (size_t)std::max(0LL, sr::orientByReferenceEmbedded(mesh, &etab));
        pImpl_->stats.orientEmbeddedFaces = nf;
    } else if (prm.enableOrient) {
        const auto os = sr::orientMesh(mesh, true);
        pImpl_->stats.orientFlips = (size_t)std::max(0LL, os.flipped);
    }
    pImpl_->stats.totalMsOrient +=
        std::chrono::duration<double, std::milli>(Clock::now() - tOr0).count();

    // —— 三遍扫描投票光顺 ——
    auto tSm0 = Clock::now();
    const bool smoothPar = (prm.numThreads != 1);
    std::vector<double> pos = sr::runSmoothV3(mesh,
                                              (double)prm.angleDeg,
                                              (double)prm.angle2Deg,
                                              (double)prm.toleranceMm,
                                              prm.ring, prm.outer, smoothPar);
    pImpl_->stats.totalMsSmooth =
        std::chrono::duration<double, std::milli>(Clock::now() - tSm0).count();

    // —— 位移上界（max|Δv|；验收协议 ≤10×tol）——
    double maxDv = 0.0;
    for (size_t i = 0; i < nv * 3; ++i) {
        const double d = std::fabs(pos[i] - mesh.verts[i]);
        if (d > maxDv) maxDv = d;
    }
    pImpl_->stats.maxDeltaVmm = (float)maxDv;

    // —— 调线（六道关卡；链 1 pass）——
    std::vector<uint32_t> trisOut = mesh.triangles;
    auto tFl0 = Clock::now();
    if (prm.enableFlip) {
        std::vector<double> origNrm = sr::vertexNormals(mesh.verts, mesh.triangles, mesh.numVerts);
        long long fc = 0;
        trisOut = sr::optimizeEdges(pos, trisOut, mesh.verts, origNrm, &fc,
                                    (double)prm.coplanarDeg, prm.flipNmGuard ? 1 : 0, &etab);
        pImpl_->stats.edgeFlips = (size_t)std::max(0LL, fc);
        pImpl_->stats.edgeFlipPasses = 1;
    }
    pImpl_->stats.totalMsFlip =
        std::chrono::duration<double, std::milli>(Clock::now() - tFl0).count();

    // —— 261121 绕向一致化收尾（链上增强，非金标准复刻）——
    // 依据：金标准链以性能由跳过拓扑定向遍（--skip-topo-orient），实测残留翻转边
    // 1989/1.64M（本移植同量级 1873）——正反面交错显示的根因。此处在 SR 出口
    // 补一次 BFS 全局一致化：种子取向承自法线参考定向（朝外性保持），整连通
    // 片绕向一致由构造保证；百万面级 ~1-2s。
    char wmsg[96] = "";
    {
        sr::Mesh om;
        om.numVerts = pos.size() / 3;
        om.numTris = trisOut.size() / 3;
        om.verts = pos;
        om.triangles = trisOut;
        const auto os = sr::orientMesh(om, true);
        trisOut = std::move(om.triangles);
        pImpl_->stats.orientFlips += (size_t)std::max(0LL, os.flipped);
        std::snprintf(wmsg, sizeof(wmsg), " | winding-fix: faces=%lld conflicted=%lld",
                      os.flipped, os.conflicted);
    }
    pImpl_->stats.trisOut = trisOut.size() / 3;

    // —— 出口：位置/拓扑写回；法线按终态几何重算（R2 权威源保持）——
    const size_t onv = pos.size() / 3;
    io.xyz.resize(onv * 3);
    for (size_t i = 0; i < onv * 3; ++i) io.xyz[i] = (float)pos[i];
    io.triangles = std::move(trisOut);
    std::vector<double> finalNrm = sr::vertexNormals(pos, io.triangles, onv);
    io.normals.resize(onv * 3);
    for (size_t i = 0; i < onv * 3; ++i) io.normals[i] = (float)finalNrm[i];

    pImpl_->stats.totalTimeMs =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "%s: v=%zu tris=%zu->%zu orientFlips=%zu flips=%zu maxDv=%.4f "
                  "(tol %.2f, %.1fx) %.0fms",
                  kLogTag, onv, nf, pImpl_->stats.trisOut,
                  pImpl_->stats.orientFlips, pImpl_->stats.edgeFlips,
                  pImpl_->stats.maxDeltaVmm, (double)prm.toleranceMm,
                  (double)pImpl_->stats.maxDeltaVmm / (double)prm.toleranceMm,
                  pImpl_->stats.totalTimeMs);
    r.message = msg;
    r.message += wmsg;   // 绕向一致化收尾遥测（261121 链上增强）

    // 位移防爆门（5mm；SR 抢救遍设计上无残差上限，链历史事故量级 41.5——只拦爆炸不拦常态）
    if ((double)pImpl_->stats.maxDeltaVmm > 5.0) {
        r.success = true;
        r.qualityFlag = QualityFlag::Degraded;
        r.message += "（位移超限降级）";
    } else {
        r.success = true;
        r.qualityFlag = QualityFlag::Normal;
    }
    r.statistics = pImpl_->stats;
    return r;
}

void MeshSmoothSr::Destroy() {
    // 无外部资源；保持 idempotent（§2.5.3）
}

void MeshSmoothSr::SetParams(const MeshSmoothSrParams& params) {
    params.validate();
    pImpl_->params = params;
}

const MeshSmoothSrParams& MeshSmoothSr::GetParams() const { return pImpl_->params; }
const MeshSmoothSrStats& MeshSmoothSr::GetStatistics() const noexcept { return pImpl_->stats; }
void MeshSmoothSr::ResetStatistics() noexcept { pImpl_->stats = MeshSmoothSrStats{}; }
void MeshSmoothSr::SetCancelProbe(const std::function<bool()>& probe) { pImpl_->cancelProbe = probe; }

OperatorInfo getMeshSmoothSrInfo() {
    return {"MeshSmoothSr", 1, 0, OperatorType::CPU};
}

} // namespace calib
