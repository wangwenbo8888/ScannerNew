// steger_dedup_merge.h — steger 输出位移式去重（scan_chain_replay 去重变体专用）
//
// 与原版 scan_chain_replay 的唯一差异件：steger_fast 出点后在本模块做 CPU 合并，
// 重建 StegerResult 的 CPU/GPU 双输出（Flat 模式单标签）。
// 算法＝steger_dedup_probe 验证版：kNN-PCA 切向 + 位移式判据
//   |Δp·t̂ᵢ|≤0.5（沿线）∧ |Δp·n̂ᵢ|≤1.5（法向）∧ |t̂ᵢ·t̂ⱼ|≥0.5（切向一致）
//   簇门：size≤8 ∧ 法向跨度≤2.5px；输出：1 点直通 / 2 点均值 / ≥3 点中位
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "core/laser/steger_fast/steger_fast.h"

namespace steger_dedup {

struct Stats {
    int inCount = 0, outCount = 0;
    int mergedPoints = 0, mergedClusters = 0;
    int bypassTangent = 0, residDupPairs = 0;
};

/// 对 StegerResult 原地应用去重（仅 Flat 模式语义）。
/// 重建 d_centerPoints/d_line_ids/centerPoints/totalPointCount；
/// msg 形如 "27194->24517 (merged 2677, bypass 544, residDup 18)"。
/// 返回 false 仅当输入为空等硬错；无合并时原样返回 true。
bool apply(calib::StegerResult& r, cv::cuda::Stream& stream, std::string& msg);

/// 纯算法入口（测试可单独调）
std::vector<cv::Point2f> mergePoints(const std::vector<cv::Point2f>& in, Stats* st = nullptr);

}  // namespace steger_dedup
