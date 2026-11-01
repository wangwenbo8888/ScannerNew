#pragma once
// ============================================================================
// CycleUnit.h — 采集周期单元（DataPlane · A 姿态判断流水线 01-⑤ 消费单元）
// ============================================================================
// 标记点帧 + 全部激光管帧 + 温度整单元。激光管帧作为"乘客"随单元传递——
// 流水线仅处理标记点帧，命中姿态时整单元保存。
// laserFrames 扁平存 N*2 张：约定偶数下标 = L、奇数下标 = R（N 运行期定）。

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace Scanner::data {

struct CycleUnit {
    uint64_t id = 0;                       // 周期号（单调）
    cv::Mat markerL, markerR;              // 标记点帧（host）
    // device 副本预留（08 接入期真上传，06 不碰 CUDA）
    void* d_markerL = nullptr;
    void* d_markerR = nullptr;
    std::vector<cv::Mat> laserFrames;      // N 路激光管帧（"乘客"，N*2 张：偶=L / 奇=R）
    double temperature = 0.0;              // 周期温度（℃）
    uint64_t timestamp = 0;                // 周期时间戳
};

} // namespace Scanner::data
