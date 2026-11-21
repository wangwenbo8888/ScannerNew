// ============================================================================
// MeshStageOps.cpp — 09 网格四族算子适配实现（M4 集成）
// ============================================================================
#include "pipelines/postprocess/MeshStageOps.h"

#include <utility>

#include "mesh_normal_est.h"
#include "mesh_ball_pivot.h"
#include "mesh_hole_fill.h"
#include "mesh_smooth_sr.h"

namespace Scanner::pipeline {
namespace {

/// MeshData → IndexedMesh（字段同构，move 零拷贝）
calib::IndexedMesh takeMesh(MeshData& io) {
    calib::IndexedMesh m;
    m.xyz = std::move(io.xyz);
    m.normals = std::move(io.normals);
    m.triangles = std::move(io.triangles);
    return m;
}

/// IndexedMesh → MeshData（回填；顶点数允许增长＝补洞补丁顶点，R1 口径）
void putMesh(MeshData& io, calib::IndexedMesh&& m) {
    io.xyz = std::move(m.xyz);
    io.normals = std::move(m.normals);
    io.triangles = std::move(m.triangles);
}

/// 算子 Result → 07 Result（§5 单一失败模型对齐）
template <typename OpResult>
Result toPipelineResult(const char* stage, OpResult&& r) {
    const std::string msg = std::string(stage) + ": " + r.message;
    if (!r.success)
        return Result::fail(msg);
    if (r.qualityFlag == calib::QualityFlag::Degraded ||
        r.qualityFlag == calib::QualityFlag::Warning)
        return Result::degraded(msg);
    return Result::ok(msg);
}

} // namespace

// ============================================================
// 阶段 0：法线重算＋定向＋清理
// ============================================================
Result MeshNormalEstStage::run(MeshData& io, const CancelToken& cancel) {
    if (io.pointCount() == 0)
        return Result::fail("MeshNormalEstStage: 空点云");
    if (cancel.cancelled())
        return Result::degraded("MeshNormalEstStage 已取消（执行前检查点）");

    calib::MeshNormalEst op;
    op.SetCancelProbe([&cancel] { return cancel.cancelled(); });
    calib::IndexedMesh m = takeMesh(io);
    auto r = op.Execute(m);
    putMesh(io, std::move(m));
    return toPipelineResult("MeshNormalEstStage", std::move(r));
}

// ============================================================
// 阶段 1：滚球封装
// ============================================================
Result MeshBallPivotStage::run(MeshData& io, const CancelToken& cancel) {
    if (io.pointCount() < 3)
        return Result::fail("MeshBallPivotStage: 点数不足（<3）");
    if (cancel.cancelled())
        return Result::degraded("MeshBallPivotStage 已取消（执行前检查点）");

    calib::MeshBallPivot op;
    op.SetCancelProbe([&cancel] { return cancel.cancelled(); });
    calib::IndexedMesh m = takeMesh(io);
    auto r = op.Execute(m);
    putMesh(io, std::move(m));
    return toPipelineResult("MeshBallPivotStage", std::move(r));
}

// ============================================================
// 阶段 2：补洞（标志点可选）
// ============================================================
Result MeshHoleFillStage::run(MeshData& io, const CancelToken& cancel) {
    if (io.triangles.empty())
        return Result::fail("MeshHoleFillStage: 输入网格为空（应先经封装）");
    if (cancel.cancelled())
        return Result::degraded("MeshHoleFillStage 已取消（执行前检查点）");

    calib::MeshHoleFillParams prm;                    // 链实测口径默认值
    if (markers_.empty()) {
        prm.markerCut = false;                        // 无标志点＝跳过切割段（消息注记）
    }
    calib::MeshHoleFill op(prm);
    op.SetCancelProbe([&cancel] { return cancel.cancelled(); });
    calib::IndexedMesh m = takeMesh(io);
    auto r = op.Execute(m, markers_);
    putMesh(io, std::move(m));
    return toPipelineResult("MeshHoleFillStage", std::move(r));
}

// ============================================================
// 阶段 3：光顺
// ============================================================
Result MeshSmoothSrStage::run(MeshData& io, const CancelToken& cancel) {
    if (io.triangles.empty())
        return Result::fail("MeshSmoothSrStage: 输入网格为空（应先经补洞）");
    if (cancel.cancelled())
        return Result::degraded("MeshSmoothSrStage 已取消（执行前检查点）");

    calib::MeshSmoothSr op;
    op.SetCancelProbe([&cancel] { return cancel.cancelled(); });
    calib::IndexedMesh m = takeMesh(io);
    auto r = op.Execute(m);
    putMesh(io, std::move(m));
    return toPipelineResult("MeshSmoothSrStage", std::move(r));
}

} // namespace Scanner::pipeline
