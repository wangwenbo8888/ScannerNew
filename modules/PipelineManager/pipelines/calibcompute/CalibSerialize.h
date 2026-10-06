#pragma once
// ============================================================================
// CalibSerialize.h — B 输出 → 06 仓库 JSON（ICalibRepoWriter::write 载荷）
// ============================================================================
// 顶层键：{stereo, tempTables, pjc, planeMap, quality}
//   stereo     ：StereoParams（matToJson 逐元素）+ rms/ROI 门禁诊断量
//   tempTables ：相机链三表（5-1/5-2/3-5）+ 5-3 —— 各算子 Result 自带 toJson
//   pjc        ：ProjectorJointCalibResult 无 toJson → 此处手写（占位，见 .cpp 注）
//   planeMap   ：{map: 4-12（PlaneMapResult 无 toJson → 手写占位）,
//                tempTable: 4-13 档表 toJson}
//   quality    ：QualityReport（items 数组逐项）
// ⚠ 不落盘：4-12/4-13 的 GpuMat 设备数据（d_left_to_right/d_right_u，设备指针
//   不可移植——扫描侧按表参数重算或经 06 设备资源交接）与 PJC denoisedPoints
//   （诊断点云，只记数量）。
#include <nlohmann/json.hpp>

#include "pipelines/calibcompute/CalibComputeTypes.h"

namespace Scanner::pipeline {

// 序列化（const，不修改输入；json → 字符串由调用方 .dump()）
nlohmann::json serializeCalib(const CalibComputeOutput& out);

} // namespace Scanner::pipeline
