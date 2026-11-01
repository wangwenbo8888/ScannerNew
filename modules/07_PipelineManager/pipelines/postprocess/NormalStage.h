#pragma once
// ============================================================================
// NormalStage.h — E 五阶段·阶段0 法线重算（P5-T27 实接 09 laser_cloud_normal_cpu）
//
// 适配决策：09 CPU 法线算子输入为融合累积器 LaserCloudFuseCPU&（非裸点云），
// 但累积器公开接口本身吃 host 点数组（Execute(frame)）——故走适配不新建：
// MeshData.xyz → 临时累积器单帧灌入（自适应体素 cbrt(包围盒体积/N)，夹
// [diag/1024, diag/32]）→ 全体素法线重算 → 逐点就近体素代表点映射回写
// io.normals（数量==点数）。即 docs/模块功能/04-后处理.md "输入适配 or 新建"
// 二选一中的"输入适配"。
//
// 法线方向符号：算子 PCA 特征向量符号不定（± 等价），如实写回不定向；一致
// 朝向（如 STL 渲染需要）由消费侧定向（04 梳理定夺）。
//
// 线程安全：run 无跨调用状态（每次 run 重建临时累积器与算子实例），多实例
// 可并行（09 算子每实例非线程安全，本阶段每次调用独占新实例）。
// ============================================================================
#include <string>

#include "base/types.h"
#include "pipelines/postprocess/PostProcessPipeline.h"   // IMeshStageOp/MeshData/CancelToken

namespace Scanner::pipeline {

class NormalStage : public IMeshStageOp {
public:
    std::string name() const override { return "法线重算"; }

    /// 空点云/融合零体素→fail；算子失败→fail；法线退化比例 ≥10%→degraded
    /// （如实携带统计）；取消（映射回写途中检查点）→degraded（B 链惯例）
    Result run(MeshData& io, const CancelToken& cancel) override;
};

} // namespace Scanner::pipeline
