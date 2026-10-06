/**
 * @file edge_match_cpu.cpp
 * @brief 椭圆边界边缘点匹配算子 - 实现文件
 *
 * 算法流程：
 *   1. 输入左右相机极线交点集 + 中心匹配映射
 *   2. 遍历每组已匹配的椭圆对
 *   3. 按 epipolarIndex 建立左右交点索引映射
 *   4. 对共有极线索引上的交点按 X 排序后顺序配对
 *   5. 计算视差和置信度
 *   6. 汇总统计信息
 */

#include "edge_match_cpu.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

using namespace calib;


CALIB_DEFINE_LOG_TAG(10, EdgeMatchCPU);

namespace {

constexpr float EPSILON = 1e-6f;
constexpr float Y_CONFIDENCE_WEIGHT = 0.6f;
constexpr float CONSISTENCY_WEIGHT = 0.4f;

struct EpiGroup {
    std::vector<size_t> indices;
};

using EpiIndexMap = std::unordered_map<int, EpiGroup>;

EpiIndexMap buildEpiIndexMap(const std::vector<EpipolarIntersectPoint>& pts) {
    EpiIndexMap map;
    for (size_t i = 0; i < pts.size(); ++i) {
        map[pts[i].epipolarIndex].indices.push_back(i);
    }
    for (auto& kv : map) {
        auto& idxs = kv.second.indices;
        std::sort(idxs.begin(), idxs.end(), [&pts](size_t a, size_t b) {
            return pts[a].x < pts[b].x;
        });
    }
    return map;
}

float computeConfidence(double leftY, double rightY, float yTolerance,
                        float disparity,
                        const std::vector<float>& neighborDisparities,
                        float disparityMaxDiff) {
    float yDist = static_cast<float>(std::abs(leftY - rightY));
    if (yDist > yTolerance) return 0.0f;
    float yConf = 1.0f - yDist / yTolerance;

    if (neighborDisparities.empty()) return yConf;

    float avgDisp = 0.0f;
    for (float d : neighborDisparities) avgDisp += d;
    avgDisp /= static_cast<float>(neighborDisparities.size());

    float deviation = std::abs(disparity - avgDisp);
    float maxDev = disparityMaxDiff;
    if (maxDev < EPSILON) maxDev = EPSILON;

    float consistConf = 0.0f;
    if (deviation < maxDev) {
        consistConf = 1.0f - deviation / maxDev;
    }

    return yConf * Y_CONFIDENCE_WEIGHT + consistConf * CONSISTENCY_WEIGHT;
}

void collectStatistics(EdgeMatchStats& stats,
                       const std::vector<EllipseEdgeMatchResult>& ellipseResults) {
    stats.totalEllipsePairs = ellipseResults.size();
    stats.matchedPairs = 0;
    stats.skippedPairs = 0;
    stats.totalEpipolarLines = 0;

    std::vector<float> allDisparities;
    float confidenceSum = 0.0f;

    for (const auto& er : ellipseResults) {
        stats.matchedPairs += er.matchedPairs.size();
        for (const auto& mp : er.matchedPairs) {
            allDisparities.push_back(mp.disparity);
            confidenceSum += mp.confidence;
        }
    }

    stats.avgDisparity = 0.0f;
    stats.disparityStd = 0.0f;
    stats.avgConfidence = 0.0f;
    stats.matchRate = 0.0f;

    if (!allDisparities.empty()) {
        float sum = 0.0f;
        for (float d : allDisparities) sum += d;
        stats.avgDisparity = sum / static_cast<float>(allDisparities.size());
        stats.avgConfidence = confidenceSum / static_cast<float>(allDisparities.size());

        if (allDisparities.size() > 1) {
            float sqSum = 0.0f;
            for (float d : allDisparities) {
                sqSum += (d - stats.avgDisparity) * (d - stats.avgDisparity);
            }
            stats.disparityStd = std::sqrt(sqSum / static_cast<float>(allDisparities.size() - 1));
        }
    }

    size_t totalPoints = stats.matchedPairs + stats.skippedPairs;
    if (totalPoints > 0) {
        stats.matchRate = static_cast<float>(stats.matchedPairs) / static_cast<float>(totalPoints);
    }
}

} // anonymous namespace

struct EdgeMatchCPU::Impl {
    EdgeMatchCPUParams params_;
    EdgeMatchStats stats_;
    bool warmed_up_ = false;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const EdgeMatchCPUParams& params)
        : params_(params)
    {
        params_.validate();
    }

    EdgeMatchCPUResult ExecuteImpl(const std::vector<EllipseIntersectResult>& leftIntersect,
                   const std::vector<EllipseIntersectResult>& rightIntersect,
                   const std::vector<int>& centerMatches)
    {
        EdgeMatchCPUResult result;
#ifndef NDEBUG
        assert(!inProcess_.load() && "Concurrent call detected - NOT thread-safe!");
        struct ScopedFlag {
            std::atomic<bool>* flag;
            ScopedFlag(std::atomic<bool>* f) : flag(f) {}
            ~ScopedFlag() { flag->store(false); }
        };
        ScopedFlag guard(&inProcess_);
        inProcess_.store(true);
#endif

        auto totalStart = std::chrono::high_resolution_clock::now();

        stats_ = EdgeMatchStats();

        if (centerMatches.empty()) {
            result.success = true;
            result.message = "No center matches provided";
            result.qualityFlag = calib::QualityFlag::Normal;
            result.statistics = stats_;
            return result;
        }

        result.ellipseResults.reserve(centerMatches.size());
        size_t totalMatched = 0;

        for (size_t leftIdx = 0; leftIdx < centerMatches.size(); ++leftIdx) {
            int rightIdx = centerMatches[leftIdx];
            if (rightIdx < 0) continue;
            if (leftIdx >= leftIntersect.size()) continue;
            if (static_cast<size_t>(rightIdx) >= rightIntersect.size()) continue;

            const auto& leftE = leftIntersect[leftIdx];
            const auto& rightE = rightIntersect[static_cast<size_t>(rightIdx)];

            if (leftE.intersectPts.empty() || rightE.intersectPts.empty()) continue;

            EllipseEdgeMatchResult ellipseResult;
            ellipseResult.leftEllipseIdx = static_cast<int>(leftIdx);
            ellipseResult.rightEllipseIdx = rightIdx;
            ellipseResult.leftCenterX = leftE.centerX;
            ellipseResult.leftCenterY = leftE.centerY;
            ellipseResult.rightCenterX = rightE.centerX;
            ellipseResult.rightCenterY = rightE.centerY;

            EpiIndexMap leftMap = buildEpiIndexMap(leftE.intersectPts);
            EpiIndexMap rightMap = buildEpiIndexMap(rightE.intersectPts);

            ellipseResult.matchedPairs.reserve(
                std::min(leftE.intersectPts.size(), rightE.intersectPts.size()));

            std::vector<float> neighborDisparities;
            neighborDisparities.reserve(64);

            for (const auto& kv : leftMap) {
                int epiIdx = kv.first;
                auto rightIt = rightMap.find(epiIdx);
                if (rightIt == rightMap.end()) continue;

                const auto& leftPtsIdx = kv.second.indices;
                const auto& rightPtsIdx = rightIt->second.indices;

                size_t matchCount = std::min(leftPtsIdx.size(), rightPtsIdx.size());

                for (size_t k = 0; k < matchCount; ++k) {
                    const auto& lp = leftE.intersectPts[leftPtsIdx[k]];
                    const auto& rp = rightE.intersectPts[rightPtsIdx[k]];

                    float disparity = static_cast<float>(lp.x - rp.x);

                    float conf = computeConfidence(
                        lp.y, rp.y, params_.yTolerance,
                        disparity, neighborDisparities, params_.disparityMaxDiff);

                    EdgeMatchPair pair;
                    pair.leftX = lp.x;
                    pair.leftY = lp.y;
                    pair.rightX = rp.x;
                    pair.rightY = rp.y;
                    pair.disparity = disparity;
                    pair.confidence = conf;
                    pair.epipolarIndex = epiIdx;
                    pair.leftEllipseIdx = static_cast<int>(leftIdx);
                    pair.rightEllipseIdx = rightIdx;
                    pair.side = static_cast<int>(k);

                    ellipseResult.matchedPairs.push_back(pair);
                    neighborDisparities.push_back(disparity);
                    ++totalMatched;

                    if (totalMatched >= params_.maxMatchPairs) break;
                }

                if (totalMatched >= params_.maxMatchPairs) break;
            }

            if (!ellipseResult.matchedPairs.empty()) {
                result.ellipseResults.push_back(std::move(ellipseResult));
            }
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        if (params_.collectStatistics) {
            collectStatistics(stats_, result.ellipseResults);
        }

        result.statistics = stats_;
        result.success = true;
        result.message = "Edge matching completed";
        result.qualityFlag = calib::QualityFlag::Normal;

        if (stats_.matchRate < 0.3f && stats_.matchedPairs > 5) {
            result.qualityFlag = calib::QualityFlag::Warning;
            result.message = "Low match rate: " + std::to_string(static_cast<int>(stats_.matchRate * 100)) + "%";
        }

        return result;
    }

    EdgeMatchCPUResult ExecutePointsImpl(const std::vector<cv::Point2f>& leftPoints,
                          const std::vector<cv::Point2f>& rightPoints,
                          const std::vector<int>& leftGroupIds,
                          const std::vector<int>& rightGroupIds,
                          const std::vector<int>& centerMatches,
                          const std::vector<int>& leftEpipolarIndices,
                          const std::vector<int>& rightEpipolarIndices)
    {
        EdgeMatchCPUResult result;
        auto totalStart = std::chrono::high_resolution_clock::now();
        stats_ = EdgeMatchStats();

        if (leftPoints.empty() || rightPoints.empty() || centerMatches.empty()) {
            result.success = true;
            result.message = "Empty input";
            result.qualityFlag = calib::QualityFlag::Normal;
            result.statistics = stats_;
            return result;
        }

        std::unordered_map<int, std::vector<size_t>> leftGroupMap;
        for (size_t i = 0; i < leftGroupIds.size(); ++i) {
            leftGroupMap[leftGroupIds[i]].push_back(i);
        }

        std::unordered_map<int, std::vector<size_t>> rightGroupMap;
        for (size_t i = 0; i < rightGroupIds.size(); ++i) {
            rightGroupMap[rightGroupIds[i]].push_back(i);
        }

        result.ellipseResults.reserve(centerMatches.size());
        size_t totalMatched = 0;

        for (size_t leftIdx = 0; leftIdx < centerMatches.size(); ++leftIdx) {
            int rightIdx = centerMatches[leftIdx];
            if (rightIdx < 0) continue;

            auto leftIt = leftGroupMap.find(static_cast<int>(leftIdx));
            auto rightIt = rightGroupMap.find(rightIdx);
            if (leftIt == leftGroupMap.end() || rightIt == rightGroupMap.end()) continue;

            const auto& leftIdxs = leftIt->second;
            const auto& rightIdxs = rightIt->second;

            std::unordered_map<int, std::vector<size_t>> leftEpiMap;
            for (size_t idx : leftIdxs) {
                if (idx < leftEpipolarIndices.size()) {
                    leftEpiMap[leftEpipolarIndices[idx]].push_back(idx);
                }
            }

            std::unordered_map<int, std::vector<size_t>> rightEpiMap;
            for (size_t idx : rightIdxs) {
                if (idx < rightEpipolarIndices.size()) {
                    rightEpiMap[rightEpipolarIndices[idx]].push_back(idx);
                }
            }

            for (auto& kv : leftEpiMap) {
                auto& vec = kv.second;
                std::sort(vec.begin(), vec.end(), [&leftPoints](size_t a, size_t b) {
                    return leftPoints[a].x < leftPoints[b].x;
                });
            }
            for (auto& kv : rightEpiMap) {
                auto& vec = kv.second;
                std::sort(vec.begin(), vec.end(), [&rightPoints](size_t a, size_t b) {
                    return rightPoints[a].x < rightPoints[b].x;
                });
            }

            EllipseEdgeMatchResult ellipseResult;
            ellipseResult.leftEllipseIdx = static_cast<int>(leftIdx);
            ellipseResult.rightEllipseIdx = rightIdx;

            std::vector<float> neighborDisparities;
            neighborDisparities.reserve(64);

            for (const auto& kv : leftEpiMap) {
                int epiIdx = kv.first;
                auto rightEpiIt = rightEpiMap.find(epiIdx);
                if (rightEpiIt == rightEpiMap.end()) continue;

                const auto& lIdxs = kv.second;
                const auto& rIdxs = rightEpiIt->second;
                size_t matchCount = std::min(lIdxs.size(), rIdxs.size());

                for (size_t k = 0; k < matchCount; ++k) {
                    const auto& lp = leftPoints[lIdxs[k]];
                    const auto& rp = rightPoints[rIdxs[k]];

                    float disparity = lp.x - rp.x;
                    float conf = computeConfidence(
                        lp.y, rp.y, params_.yTolerance,
                        disparity, neighborDisparities, params_.disparityMaxDiff);

                    EdgeMatchPair pair;
                    pair.leftX = lp.x;
                    pair.leftY = lp.y;
                    pair.rightX = rp.x;
                    pair.rightY = rp.y;
                    pair.disparity = disparity;
                    pair.confidence = conf;
                    pair.epipolarIndex = epiIdx;
                    pair.leftEllipseIdx = static_cast<int>(leftIdx);
                    pair.rightEllipseIdx = rightIdx;
                    pair.side = static_cast<int>(k);

                    ellipseResult.matchedPairs.push_back(pair);
                    neighborDisparities.push_back(disparity);
                    ++totalMatched;

                    if (totalMatched >= params_.maxMatchPairs) break;
                }
                if (totalMatched >= params_.maxMatchPairs) break;
            }

            if (!ellipseResult.matchedPairs.empty()) {
                result.ellipseResults.push_back(std::move(ellipseResult));
            }
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        stats_.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        if (params_.collectStatistics) {
            collectStatistics(stats_, result.ellipseResults);
        }

        result.statistics = stats_;
        result.success = true;
        result.message = "Edge matching completed";
        result.qualityFlag = calib::QualityFlag::Normal;

        return result;
    }

    void Warmup(int maxEllipsePairs) {
        warmed_up_ = true;
        CALIB_LOG_INFO("Warmup() completed: maxEllipsePairs={}", maxEllipsePairs);
    }

    void SetParams(const EdgeMatchCPUParams& params) {
#ifndef NDEBUG
        assert(!inProcess_.load() && "setParams() while processing - NOT thread-safe!");
#endif
        params_ = params;
        params_.validate();
        warmed_up_ = false;
    }

    const EdgeMatchCPUParams& GetParams() const { return params_; }
    const EdgeMatchStats& getStatistics() const noexcept { return stats_; }
    void resetStatistics() noexcept { stats_ = EdgeMatchStats(); }
};

// ============================================================
// 构造 / 析构
// ============================================================
EdgeMatchCPU::EdgeMatchCPU(const EdgeMatchCPUParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("EdgeMatchCPU initialized (yTolerance={}, maxMatchPairs={})",
                   params.yTolerance, params.maxMatchPairs);
}

EdgeMatchCPU::~EdgeMatchCPU() = default;

void EdgeMatchCPU::Destroy() { }

// ============================================================
// match (EllipseIntersectResult)
// ============================================================
EdgeMatchCPUResult EdgeMatchCPU::Execute(const std::vector<EllipseIntersectResult>& leftIntersect,
                         const std::vector<EllipseIntersectResult>& rightIntersect,
                         const std::vector<int>& centerMatches)
{
    CALIB_LOG_DEBUG("Execute(EllipseIntersectResult) called: left={}, right={}, matches={}",
                    leftIntersect.size(), rightIntersect.size(), centerMatches.size());
    return pImpl_->ExecuteImpl(leftIntersect, rightIntersect, centerMatches);
}

// ============================================================
// match (Point2f + groups)
// ============================================================
EdgeMatchCPUResult EdgeMatchCPU::Execute(const std::vector<cv::Point2f>& leftPoints,
                         const std::vector<cv::Point2f>& rightPoints,
                         const std::vector<int>& leftGroupIds,
                         const std::vector<int>& rightGroupIds,
                         const std::vector<int>& centerMatches,
                         const std::vector<int>& leftEpipolarIndices,
                         const std::vector<int>& rightEpipolarIndices)
{
    CALIB_LOG_DEBUG("Execute(Point2f) called: left={}, right={}",
                    leftPoints.size(), rightPoints.size());
    return pImpl_->ExecutePointsImpl(leftPoints, rightPoints, leftGroupIds, rightGroupIds,
                            centerMatches, leftEpipolarIndices, rightEpipolarIndices);
}

// ============================================================
// warmup
// ============================================================
void EdgeMatchCPU::Warmup(int maxEllipsePairs) {
    CALIB_LOG_INFO("Warmup() called: maxEllipsePairs={}", maxEllipsePairs);
    pImpl_->Warmup(maxEllipsePairs);
}

void EdgeMatchCPU::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("warmup(WarmupConfig) called: maxPointCount={}", config.maxPointCount);
    Warmup(config.maxPointCount);
}

// ============================================================
// setParams / getParams
// ============================================================
void EdgeMatchCPU::SetParams(const EdgeMatchCPUParams& params) {
    CALIB_LOG_INFO("SetParams() called");
    pImpl_->SetParams(params);
}

const EdgeMatchCPUParams& EdgeMatchCPU::GetParams() const {
    return pImpl_->GetParams();
}

const EdgeMatchStats& EdgeMatchCPU::GetStatistics() const noexcept {
    return pImpl_->getStatistics();
}

void EdgeMatchCPU::ResetStatistics() noexcept {
    pImpl_->resetStatistics();
}
