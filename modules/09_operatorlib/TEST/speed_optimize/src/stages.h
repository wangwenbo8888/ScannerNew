#pragma once
// ============================================================================
// stages.h — 步骤注册表（--step 名 → 链内身份）
// ============================================================================
// 与流水线文档§一调用序一一对应；per-eye 步骤在 harness 内对 L/R 各执行一次。
// cacheable：产物可从磁盘缓存再载入（单步模式加速；配准步跨帧禁单跑）。
#include <string>
#include <vector>

namespace scr {

struct StageInfo {
    const char* name;
    const char* deps;        // 逗号分隔的直接前置（文档用；执行序在 harness 固化）
    bool perEye;
    bool cacheable;
};

/// 16 步注册表（顺序＝链执行序）
inline const std::vector<StageInfo>& stageTable() {
    static const std::vector<StageInfo> kStages = {
        {"mask_separation",    "frame",                     true,  true},
        {"ccl",                "mask_separation",           true,  true},
        {"image_split",        "ccl",                       true,  true},
        {"zernike_edge",       "image_split",               true,  true},
        {"image_merge",        "zernike_edge",              true,  true},
        {"undistort_cpu",      "image_merge",               false, true},
        {"ellipse_fit",        "undistort_cpu",             true,  true},
        {"marker_match",       "ellipse_fit",               false, true},
        {"conic_reconstruct",  "marker_match",              false, true},
        {"registration",       "conic_reconstruct",         false, false},
        {"steger_fast",        "mask_separation",           true,  true},
        {"undistort_cuda",     "steger_fast",               true,  true},
        {"epipolar_interp_dual", "undistort_cuda",          true,  true},
        {"laser_match_scan_v3", "epipolar_interp_dual",     false, true},
        {"laser_reconstruct",  "laser_match_scan_v3",       false, true},
        // {"frame_source"} 隐含前置，不单列
    };
    return kStages;
}

inline bool isValidStage(const std::string& name) {
    for (const auto& s : stageTable())
        if (name == s.name) return true;
    return false;
}

} // namespace scr
