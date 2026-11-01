// ============================================================================
// epipolar_pair_cpu.cpp — 同行配对实现（y 桶最近邻；从 07 ScanChains 还账
// 搬运 2026-09-01，算法口径不变）
// ============================================================================
#include "epipolar_pair_cpu.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace calib {

EpipolarPairCPU::EpipolarPairCPU(const EpipolarPairParams& params)
    : params_(params) {
    params_.validate();
}

EpipolarPairCPU::~EpipolarPairCPU() = default;

void EpipolarPairCPU::SetParams(const EpipolarPairParams& params) {
    params.validate();
    params_ = params;
}

EpipolarPairResult EpipolarPairCPU::Execute(
    const std::vector<cv::Point2f>& left,
    const std::vector<cv::Point2f>& right,
    const std::vector<int>& leftIds)
{
    EpipolarPairResult result;
    try {
        params_.validate();
        if (left.empty() || right.empty()) {
            result.message = "empty input";
            return result;
        }

        // 右点按 y 分桶（2px 粒度）
        std::unordered_map<int, std::vector<int>> rows;
        for (int j = 0; j < static_cast<int>(right.size()); ++j)
            rows[static_cast<int>(right[static_cast<size_t>(j)].y * 0.5f)]
                .push_back(j);

        for (int i = 0; i < static_cast<int>(left.size()); ++i) {
            auto it = rows.find(static_cast<int>(
                left[static_cast<size_t>(i)].y * 0.5f));
            if (it == rows.end()) continue;
            int best = -1;
            float bestDy = std::numeric_limits<float>::max();
            for (int j : it->second) {
                const float dy = std::abs(left[static_cast<size_t>(i)].y -
                                          right[static_cast<size_t>(j)].y);
                const float disp = left[static_cast<size_t>(i)].x -
                                   right[static_cast<size_t>(j)].x;
                if (dy < params_.yTolerance && disp > params_.dispMin &&
                    disp < params_.dispMax && dy < bestDy) {
                    bestDy = dy;
                    best = j;
                }
            }
            if (best >= 0) {
                result.matchedLeft.push_back(left[static_cast<size_t>(i)]);
                result.matchedRight.push_back(right[static_cast<size_t>(best)]);
                result.matchedIds.push_back(
                    static_cast<size_t>(i) < leftIds.size()
                        ? leftIds[static_cast<size_t>(i)] : -1);
            }
        }

        if (static_cast<int>(result.matchedIds.size()) < params_.minPairs) {
            result.message = "insufficient pairs: " +
                             std::to_string(result.matchedIds.size());
            return result;
        }
        result.success = true;
        result.message = "paired " + std::to_string(result.matchedIds.size());
    } catch (const std::exception& e) {
        result.message = std::string("Err: ") + e.what();
        result.qualityFlag = QualityFlag::Degraded;
    }
    return result;
}

} // namespace calib
