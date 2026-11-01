// ============================================================================
// QualityGate.cpp — B 标定质量门禁实现（纯函数；项清单/判据见头文件注）
// ============================================================================
#include "pipelines/calibcompute/QualityGate.h"

#include <algorithm>
#include <string>

namespace Scanner::pipeline::gate {

namespace {

// 立体产物缺（3-4 未定案）：核心矩阵任一为空
bool stereoProductMissing(const CalibComputeOutput& o) {
    return o.stereo.cameraMatrixL.empty() || o.stereo.cameraMatrixR.empty() ||
           o.stereo.R1.empty() || o.stereo.P1.empty() || o.stereo.Q.empty();
}

// PJC 产物缺（激光链未产出或 PJC 算子失败）
bool pjcProductMissing(const CalibComputeOutput& o) {
    return !o.laserValid || !o.pjc.success;
}

// 三档表档数取 min：3-5（tableSize）/ 4-13（tableSize）/ 5-3（L,R table.size）
int tempTableMinEntries(const CalibComputeOutput& o) {
    const int rectify   = o.rectifyTempTable.success ? o.rectifyTempTable.tableSize : 0;
    const int planeMap  = o.planeMapTempTable.success ? o.planeMapTempTable.tableSize : 0;
    const int laserL = o.laserExtrinsicTempTable.success
                           ? static_cast<int>(o.laserExtrinsicTempTable.leftResult.table.size())
                           : 0;
    const int laserR = o.laserExtrinsicTempTable.success
                           ? static_cast<int>(o.laserExtrinsicTempTable.rightResult.table.size())
                           : 0;
    return std::min(std::min(rectify, planeMap), std::min(laserL, laserR));
}

// 3-4 有效 ROI 面积 L/R 取 min
double rectifyRoiMinArea(const CalibComputeOutput& o) {
    return std::min(static_cast<double>(o.rectifyValidRoiL.area()),
                    static_cast<double>(o.rectifyValidRoiR.area()));
}

QualityGateItem mkItem(const std::string& name, double value, double threshold,
                       bool pass, const std::string& note) {
    return QualityGateItem{name, value, threshold, pass, note};
}

std::string fmt(double v) { return std::to_string(v); }

} // namespace

QualityReport evaluate(const CalibComputeOutput& out, const Thresholds& th) {
    QualityReport r;
    const bool stereoMiss = stereoProductMissing(out);
    const bool pjcMiss = pjcProductMissing(out);

    // —— 3-2 各目 rms（≤ 上限）——
    r.items.push_back(mkItem("intrinsicRmsL", out.intrinsicRmsL, th.intrinsicRmsMax,
                             out.intrinsicRmsL <= th.intrinsicRmsMax, "rms<=thr"));
    r.items.push_back(mkItem("intrinsicRmsR", out.intrinsicRmsR, th.intrinsicRmsMax,
                             out.intrinsicRmsR <= th.intrinsicRmsMax, "rms<=thr"));

    // —— 3-3 立体重投影（≤ 上限；立体缺产物强制 fail）——
    r.items.push_back(mkItem("stereoReprojError", out.stereo.reprojError, th.stereoReprojMax,
                             !stereoMiss && out.stereo.reprojError <= th.stereoReprojMax,
                             stereoMiss ? "stereo product missing" : "err<=thr"));

    // —— PJC 诊断（条件数/rms ≤ 上限；改善比 ≥ 下限；PJC 缺产物强制 fail）——
    r.items.push_back(mkItem("pjcConditionNumber", out.pjc.jacobianConditionNumber,
                             th.pjcConditionMax,
                             !pjcMiss && out.pjc.jacobianConditionNumber <= th.pjcConditionMax,
                             pjcMiss ? "PJC product missing" : "cond<=thr"));
    r.items.push_back(mkItem("pjcRms", out.pjc.finalSampsonRms, th.pjcRmsMax,
                             !pjcMiss && out.pjc.finalSampsonRms <= th.pjcRmsMax,
                             pjcMiss ? "PJC product missing" : "rms<=thr"));
    r.items.push_back(mkItem("pjcImprovementRatio", out.pjc.improvementRatio,
                             th.pjcImprovementMin,
                             !pjcMiss && out.pjc.improvementRatio >= th.pjcImprovementMin,
                             pjcMiss ? "PJC product missing" : "ratio>=thr"));

    // —— 三档表档数（≥ 下限，min over 3-5/4-13/5-3(L,R)）——
    const int entries = tempTableMinEntries(out);
    r.items.push_back(mkItem("tempTableEntries", static_cast<double>(entries),
                             static_cast<double>(th.tempTableEntriesMin),
                             entries >= th.tempTableEntriesMin, "entries>=thr"));

    // —— 3-4 有效 ROI（非零面积；立体缺产物强制 fail）——
    const double roiArea = rectifyRoiMinArea(out);
    r.items.push_back(mkItem("rectifyValidRoi", roiArea, 1.0,
                             !stereoMiss && roiArea >= 1.0,
                             stereoMiss ? "stereo product missing" : "area>=1"));

    // —— overall 合成 + summary ——
    int passCount = 0;
    std::string fails;
    for (const auto& it : r.items) {
        if (it.pass) {
            ++passCount;
            continue;
        }
        if (!fails.empty()) fails += ", ";
        fails += it.name + "(" + fmt(it.value) + "/" + fmt(it.threshold) + " " + it.note + ")";
    }
    const int total = static_cast<int>(r.items.size());
    if (passCount == total) {
        r.ok = true;
        r.overall = Scanner::QualityFlag::Normal;
        r.summary = "calib gate " + std::to_string(passCount) + "/" +
                    std::to_string(total) + " pass";
    } else {
        r.ok = false;
        r.overall = (stereoMiss || pjcMiss) ? Scanner::QualityFlag::Fault
                                            : Scanner::QualityFlag::Degraded;
        r.summary = "calib gate " + std::to_string(passCount) + "/" +
                    std::to_string(total) + " pass | fail: " + fails;
    }
    return r;
}

} // namespace Scanner::pipeline::gate
