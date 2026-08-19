#pragma once
// ============================================================================
// PostureTypes.h — A 姿态判断流水线自有类型（纯数据，无逻辑）
// ============================================================================
// PostureData        ：单姿态确认时保存的完整周期（B 重建的输入之一）
// PostureSessionData ：A 输出、B 输入的会话级收口数据（25 目标姿态簿）
//
// ⚠ 初始参数组同组约定（2-6 与 3-1 严格同组）：本结构只承数据，不持有参数组
//   引用——由 PosturePipeline 装配期另途注入 B，装配层保证同组。
#include <array>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "CycleUnit.h"

namespace Scanner::pipeline {

struct PostureData {
    uint64_t cycleId = 0;                     // 确认时刻周期号
    Scanner::data::CycleUnit cycle;           // 完整周期（标记点+激光管帧+温度）
    double R[9] = {1,0,0, 0,1,0, 0,0,1};      // 命中时设备姿态（row-major 3×3）
    double T[3] = {0,0,0};                    // 命中时设备平移
    std::vector<cv::Point2f> ellipseCentersL, ellipseCentersR;  // 姿态合格帧 2-7 输出（供 B 3-1 消费）
};

struct PostureSessionData {
    static constexpr int kTargetCount = 25;
    std::array<PostureData, kTargetCount> poses;
    std::array<bool, kTargetCount> collected{};      // 姿态 i 是否已确认
    int collectedCount = 0;
};

} // namespace Scanner::pipeline
