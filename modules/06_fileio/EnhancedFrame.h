#pragma once
// ============================================================================
// EnhancedFrame.h — 增强帧（DataPlane · 扫描工作流 02-⑤ 出口产物）
//
// 06 出口查表产物：帧数据 + 按帧温命中两张补偿表（立体矫正表 + 激光平面
// 映射表）的标定快照——算子无查表动作，消费方直接用 snapshot。
// ============================================================================

#include <cstdint>
#include <opencv2/core.hpp>

namespace Scanner::data {

// 标定快照（查表结果）：立体矫正档参数 + 两表命中的温度档索引
struct CalibSnapshot {
    cv::Matx33d R1, R2;
    cv::Matx34d P1, P2;
    cv::Matx44d Q;
    int stereoTier = 0;      // 命中的立体表温度档索引（表空时 -1）
    int laserTier = 0;       // 命中的激光映射表温度档索引（表空时 -1）
};

struct EnhancedFrame {
    uint64_t frameId = 0;
    cv::Mat grayL, grayR;                  // host 副本（clone 深拷贝）
    // device 副本以不透明指针预留（08 接入期真上传；06 不碰 CUDA）
    void* d_grayL = nullptr;
    void* d_grayR = nullptr;
    double temperature = 0.0;              // 查表用帧温（℃）
    CalibSnapshot snapshot;
};

} // namespace Scanner::data
