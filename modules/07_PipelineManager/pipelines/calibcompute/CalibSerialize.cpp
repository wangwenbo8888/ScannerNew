// ============================================================================
// CalibSerialize.cpp — B 输出 → 06 仓库 JSON 实现（键结构见头文件注）
// ============================================================================
#include "pipelines/calibcompute/CalibSerialize.h"

#include <string>

#include "common/json_utils.h"     // calib::matToJson

namespace Scanner::pipeline {

namespace {

nlohmann::json matOrEmpty(const cv::Mat& m) {
    return m.empty() ? nlohmann::json::array() : calib::matToJson(m);
}

nlohmann::json rectToJson(const cv::Rect& r) {
    return {{"x", r.x}, {"y", r.y}, {"width", r.width}, {"height", r.height}};
}

// StereoParams 无 toJson → 手写（相机链半区：矩阵逐元素 + 门禁诊断量）
nlohmann::json stereoToJson(const CalibComputeOutput& out) {
    const auto& sp = out.stereo;
    return {
        {"cameraMatrixL", matOrEmpty(sp.cameraMatrixL)},
        {"distCoeffsL", matOrEmpty(sp.distCoeffsL)},
        {"cameraMatrixR", matOrEmpty(sp.cameraMatrixR)},
        {"distCoeffsR", matOrEmpty(sp.distCoeffsR)},
        {"R", matOrEmpty(sp.R)},
        {"T", matOrEmpty(sp.T)},
        {"R1", matOrEmpty(sp.R1)},
        {"R2", matOrEmpty(sp.R2)},
        {"P1", matOrEmpty(sp.P1)},
        {"P2", matOrEmpty(sp.P2)},
        {"Q", matOrEmpty(sp.Q)},
        {"reprojError", sp.reprojError},
        {"intrinsicRmsL", out.intrinsicRmsL},
        {"intrinsicRmsR", out.intrinsicRmsR},
        {"rectifyValidRoiL", rectToJson(out.rectifyValidRoiL)},
        {"rectifyValidRoiR", rectToJson(out.rectifyValidRoiR)},
    };
}

// ProjectorJointCalibResult 无 toJson → 手写占位（denoisedPoints 只记数量）
nlohmann::json pjcToJson(const calib::ProjectorJointCalibResult& p) {
    nlohmann::json coeffs = nlohmann::json::array();
    for (double c : p.emissionCurve.coeffs) coeffs.push_back(c);
    return {
        {"success", p.success},
        {"message", p.message},
        {"projectorT", {p.projectorT(0), p.projectorT(1), p.projectorT(2)}},
        {"initialT", {p.initialT(0), p.initialT(1), p.initialT(2)}},
        {"emissionCurve",
         {{"coeffs", coeffs},
          {"discriminant", p.emissionCurve.discriminant},
          {"sampsonRms", p.emissionCurve.sampsonRms},
          {"pointCount", p.emissionCurve.pointCount}}},
        {"initialSampsonRms", p.initialSampsonRms},
        {"finalSampsonRms", p.finalSampsonRms},
        {"improvementRatio", p.improvementRatio},
        {"poseCount", p.poseCount},
        {"totalPointCount", p.totalPointCount},
        {"jacobianConditionNumber", p.jacobianConditionNumber},
        {"denoisedPointCount", p.denoisedPoints.size()},
    };
}

// PlaneMapResult（4-12）无 toJson → 手写占位（GpuMat 设备数据不落盘，头文件注）
nlohmann::json planeMapToJson(const calib::PlaneMapResult& m) {
    nlohmann::json stats = nlohmann::json::array();
    for (const auto& st : m.lineStats)
        stats.push_back({{"lineId", st.lineId}, {"numPairs", st.numPairs},
                         {"uMin", st.uMin}, {"uMax", st.uMax},
                         {"vMin", st.vMin}, {"vMax", st.vMax}});
    return {
        {"success", m.success},
        {"message", m.message},
        {"totalPairs", m.totalPairs},
        {"lineStats", stats},
        {"note", "GpuMat device data (d_left_to_right/d_right_u) not serialized"},
    };
}

const char* qualityFlagName(Scanner::QualityFlag q) {
    switch (q) {
    case Scanner::QualityFlag::Normal:   return "Normal";
    case Scanner::QualityFlag::Degraded: return "Degraded";
    case Scanner::QualityFlag::Warning:  return "Warning";
    case Scanner::QualityFlag::Fault:    return "Fault";
    default:                             return "Unknown";
    }
}

nlohmann::json qualityToJson(const QualityReport& qr) {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& it : qr.items)
        items.push_back({{"name", it.name},
                         {"value", it.value},
                         {"threshold", it.threshold},
                         {"pass", it.pass},
                         {"note", it.note}});
    return {
        {"ok", qr.ok},
        {"overall", qualityFlagName(qr.overall)},
        {"summary", qr.summary},
        {"items", items},
    };
}

} // namespace

nlohmann::json serializeCalib(const CalibComputeOutput& out) {
    return {
        {"stereo", stereoToJson(out)},
        {"tempTables",
         {{"intrinsic", out.intrinsicTempTable.toJson()},          // 5-1
          {"extrinsic", out.extrinsicTempTable.toJson()},          // 5-2
          {"rectify", out.rectifyTempTable.toJson()},              // 3-5
          {"laserExtrinsic", out.laserExtrinsicTempTable.toJson()} // 5-3
         }},
        {"pjc", pjcToJson(out.pjc)},
        {"planeMap",
         {{"map", planeMapToJson(out.planeMap)},                   // 4-12（占位）
          {"tempTable", out.planeMapTempTable.toJson()}            // 4-13 档表
         }},
        {"quality", qualityToJson(out.quality)},
    };
}

} // namespace Scanner::pipeline
