#pragma once
// ============================================================================
// MeshStageOps.h — 09 网格四族算子 → 07 IMeshStageOp 适配层（M4 集成）
//
// 阶段注入（04 装配期 setStageOp）：
//   0 MeshNormalEstStage   calib::MeshNormalEst   （法线定向清理，替代旧 NormalStage 适配）
//   1 MeshBallPivotStage   calib::MeshBallPivot   （滚球封装）
//   2 MeshHoleFillStage    calib::MeshHoleFill    （补洞＋标志点切割）
//   3 MeshSmoothSrStage    calib::MeshSmoothSr    （光顺）
//   4 边界优化：D3 形状一——留桩＋04 默认 skipStages bit4（显式跳过不降级）
//
// MeshData ↔ calib::IndexedMesh 字段同构（xyz/normals/triangles），move 平移零拷贝。
// 取消：CancelToken → 算子 SetCancelProbe（阶段前检查点＋算子内粗粒度检查）。
// ============================================================================
#include <memory>
#include <string>
#include <vector>

#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"   // CancelToken
#include "pipelines/postprocess/PostProcessPipeline.h"  // IMeshStageOp / MeshData

#include "mesh_stage_types.h"

namespace Scanner::pipeline {

/// 阶段 0：法线重算＋定向＋清理（入口法线若在则作符号参考）
class MeshNormalEstStage : public IMeshStageOp {
public:
    std::string name() const override { return "MeshNormalEst"; }
    Result run(MeshData& io, const CancelToken& cancel) override;
};

/// 阶段 1：滚球封装（消费定向点云，产出索引三角形）
class MeshBallPivotStage : public IMeshStageOp {
public:
    std::string name() const override { return "MeshBallPivot"; }
    Result run(MeshData& io, const CancelToken& cancel) override;
};

/// 阶段 2：补洞（标志点可选；无标志点时 markerCut 自动关闭并在消息注记）
class MeshHoleFillStage : public IMeshStageOp {
public:
    std::string name() const override { return "MeshHoleFill"; }
    Result run(MeshData& io, const CancelToken& cancel) override;

    /// 标志点注入（marker_cloud_fuse 输出同构；空＝不切割）
    void SetMarkers(std::vector<calib::MarkerHoleRef> markers) { markers_ = std::move(markers); }

private:
    std::vector<calib::MarkerHoleRef> markers_;
};

/// 阶段 3：光顺（⚠ 法务门 C2 未决前不入发布分支——见移植计划 §6）
class MeshSmoothSrStage : public IMeshStageOp {
public:
    std::string name() const override { return "MeshSmoothSr"; }
    Result run(MeshData& io, const CancelToken& cancel) override;
};

} // namespace Scanner::pipeline
