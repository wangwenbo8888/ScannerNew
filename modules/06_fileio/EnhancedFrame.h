#pragma once
// ============================================================================
// EnhancedFrame.h — 增强帧（DataPlane · 扫描工作流 02-⑤ 出口产物）
//
// 06 出口查表产物：帧数据 + 按帧温命中两张补偿表（立体矫正表 + 激光平面
// 映射表）的标定快照——算子无查表动作，消费方直接用 snapshot。
// ============================================================================

#include <cstdint>
#include <opencv2/core.hpp>
#include "base/types.h"

namespace Scanner::data {

// 标定快照（查表结果）：立体矫正档参数 + 两表命中的温度档索引
struct CalibSnapshot {
    cv::Matx33d R1 = cv::Matx33d::zeros();
    cv::Matx33d R2 = cv::Matx33d::zeros();
    cv::Matx34d P1 = cv::Matx34d::zeros();
    cv::Matx34d P2 = cv::Matx34d::zeros();
    cv::Matx44d Q  = cv::Matx44d::zeros();
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
    // T/V 激光组判定（260927 时间戳奇偶法，CameraControl 算好透传）：
    // tvKnown=false＝未知（sim 帧/未锚定）——消费方（07 ScanChains）回退帧号奇偶
    bool tvKnown = false;
    bool tvLeftSkew = true;                // true=左斜 T 组 / false=右斜 V 组
    // 采集模式随帧贯通（260927：08 captureMode 原子快照随帧下行）——07 帧级激光
    // 种类分派依据：面片=T/V 交替（tvKnown 判）、精细=D 管帧帧同族、深孔=C 管
    Scanner::ScanMode scanMode = Scanner::ScanMode::MarkerPlusLaser;
    CalibSnapshot snapshot;
};

} // namespace Scanner::data
