#pragma once
// ============================================================================
// scr_test_helpers.h — 合成图辅助共享头（自 test_units.cpp 匿名 namespace 迁出，2026-10-18）
// ⚠️ MakeFlatTopSigma4 为精度敏感考古配方（w=8/edge=3.5/断言窗 [40,280)），勿改数值。
// 函数一律 inline＋namespace scr：tests/*.cpp 经 CMake glob 编入同一 exe，
// 非 inline 的外部链接函数在多 TU 重复定义会 LNK2005。
// ============================================================================
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include "common/steger_types.h"

namespace scr {

// 合成平顶图（复刻 test_steger_fast.cpp Task 4 考古配方）：
// yc=N+0.5 半像素相位 + e=3.5 缓边 + σ4/lowThreshold 0.5 → λ 微区泰勒数值不
// 稳，每列恰 2 点 {yc-δ, yc+δ}（δ≈0.003px，分居真中心两侧）——两法去重共同
// 的输入结构。线 x∈[30,290)；mask 行 [95,145] 覆盖 w=8＋缓边完整支撑区；
// 断言窗 [40,280) 避开 mask 水平边界的图像阶跃伪响应。
inline cv::Mat MakeFlatTopSigma4(cv::Mat& mask) {
    const float yc = 120.5f, w = 8.0f, edge = 3.5f, amp = 255.0f;
    cv::Mat gray = cv::Mat::zeros(240, 320, CV_8UC1);
    mask = cv::Mat::zeros(240, 320, CV_8UC1);
    for (int x = 30; x < 290; ++x) {
        for (int y = 0; y < gray.rows; ++y) {
            const float d = std::fabs(static_cast<float>(y) - yc) - w * 0.5f;
            const float v = d <= 0.0f ? 1.0f : std::exp(-(d * d) / (2.0f * edge * edge));
            gray.at<uchar>(y, x) =
                cv::saturate_cast<uchar>(gray.at<uchar>(y, x) + amp * v);
        }
        for (int y = 95; y <= 145; ++y) mask.at<uchar>(y, x) = 255;
    }
    return gray;
}

inline std::vector<cv::Point2f> Flatten(const calib::StegerResult& r) {
    std::vector<cv::Point2f> v;
    for (const auto& kv : r.centerPoints)
        for (const auto& q : kv.second) v.push_back(q);
    return v;
}

inline std::vector<cv::Point2f> PointsInWindow(const std::vector<cv::Point2f>& v) {
    std::vector<cv::Point2f> w;
    for (const auto& q : v)
        if (q.x >= 40.0f && q.x < 280.0f) w.push_back(q);
    return w;
}

} // namespace scr
