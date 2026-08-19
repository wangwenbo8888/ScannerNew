#pragma once
// ============================================================================
// QualityGate.h — B 标定质量门禁（阈值参数化；纯函数评估）
// ============================================================================
// 门禁项（8 项，客户端标定流水线.md §一/设计方案 §4.2-B）：
//   intrinsicRmsL / intrinsicRmsR   3-2 各目 rms（≤ 上限）
//   stereoReprojError               3-3 立体重投影误差（≤ 上限）
//   pjcConditionNumber              PJC JᵀJ 条件数（≤ 上限；>1e10 警示 t_z 不可信）
//   pjcRms                          PJC finalSampsonRms（≤ 上限；同算子 anomaly 阈）
//   pjcImprovementRatio             PJC improvementRatio（≥ 下限；≥1=优化未变差）
//   tempTableEntries                三档表（3-5/4-13/5-3 L,R）档数取 min（≥ 下限）
//   rectifyValidRoi                 3-4 有效 ROI 面积 L/R 取 min（≥1=非零）
// overall：全过=Normal；有 fail 项但链产物齐=Degraded（可复检）；
//          关键缺产物（立体参数 / PJC）=Fault。
// 默认阈值待联调（任务书 §1）——装配层经 CalibComputePipeline::Config 覆盖。
#include "pipelines/calibcompute/CalibComputeTypes.h"

namespace Scanner::pipeline::gate {

struct Thresholds {
    double intrinsicRmsMax     = 0.5;   // 3-2 各目 rms 上限（px）
    double stereoReprojMax     = 1.0;   // 3-3 立体重投影误差上限（px）
    double pjcConditionMax     = 1e10;  // PJC 条件数上限
    double pjcRmsMax           = 0.15;  // PJC finalSampsonRms 上限（mm）
    double pjcImprovementMin   = 1.0;   // PJC improvementRatio 下限
    int    tempTableEntriesMin = 1;     // 三档表最少档数
    // rectifyValidRoi 为固定判据（非零面积），无阈值参数
};

// 纯函数：对 B 输出逐项评估并合成 overall/summary（不修改输入）
QualityReport evaluate(const CalibComputeOutput& out, const Thresholds& th);

} // namespace Scanner::pipeline::gate
