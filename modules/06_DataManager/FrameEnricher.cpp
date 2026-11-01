// ============================================================================
// FrameEnricher.cpp — 出口查表器实现（详见 FrameEnricher.h 契约）
// ============================================================================

#include "FrameEnricher.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace Scanner::data {
namespace {

// 选最近档：|t - tier.tempC| 最小；并列（含浮点噪声，kTieEps 内视为并列）取
// 低温档。越出 [首档, 末档] → clamp 首档/末档并置 clamped。tiers 须已升序。
template <typename TierT>
int nearestTier(const std::vector<TierT>& tiers, double t, bool& clamped) {
    clamped = false;
    if (t < tiers.front().tempC) { clamped = true; return 0; }
    if (t > tiers.back().tempC) {
        clamped = true;
        return static_cast<int>(tiers.size()) - 1;
    }
    constexpr double kTieEps = 1e-9;      // 吸收双精度表示噪声（~1e-14），真档距 0.2℃
    int best = 0;
    double bestDiff = std::abs(t - tiers[0].tempC);
    for (int i = 1; i < static_cast<int>(tiers.size()); ++i) {
        const double diff = std::abs(t - tiers[i].tempC);
        if (diff < bestDiff - kTieEps) {  // 仅显著更近才更新：并列保留低温档
            best = i;
            bestDiff = diff;
        }
    }
    return best;
}

} // namespace

Result enrich(const cv::Mat& grayL, const cv::Mat& grayR, double temperatureC,
              const StereoTempTable& stereoTable, const PlaneMapTempTableRef& laserTable,
              uint64_t frameId, EnhancedFrame& out) {
    if (stereoTable.tiers.empty() && laserTable.tiers.empty())
        return Result::fail("enrich: 立体/激光温度表均为空，无法查表");

    // 防御：实现内按档温升序（拷贝排序，不改调用方表）
    std::vector<StereoTempTier> stereo = stereoTable.tiers;
    std::stable_sort(stereo.begin(), stereo.end(),
                     [](const StereoTempTier& a, const StereoTempTier& b) {
                         return a.tempC < b.tempC;
                     });
    std::vector<PlaneMapTempTierRef> laser = laserTable.tiers;
    std::stable_sort(laser.begin(), laser.end(),
                     [](const PlaneMapTempTierRef& a, const PlaneMapTempTierRef& b) {
                         return a.tempC < b.tempC;
                     });

    bool warn = false;
    std::string msg;

    // 立体侧：填档数据 + 档索引
    if (!stereo.empty()) {
        bool clamped = false;
        const int idx = nearestTier(stereo, temperatureC, clamped);
        out.snapshot.R1 = stereo[idx].R1;
        out.snapshot.R2 = stereo[idx].R2;
        out.snapshot.P1 = stereo[idx].P1;
        out.snapshot.P2 = stereo[idx].P2;
        out.snapshot.Q = stereo[idx].Q;
        out.snapshot.stereoTier = idx;
        if (clamped) {
            warn = true;
            msg += "stereo 档越界 clamp; ";
        }
    } else {
        out.snapshot.stereoTier = -1;
        warn = true;
        msg += "stereo 表空，tier=-1; ";
    }

    // 激光映射侧：只填档索引（表数据由 07/09 侧持有）
    if (!laser.empty()) {
        bool clamped = false;
        const int idx = nearestTier(laser, temperatureC, clamped);
        out.snapshot.laserTier = idx;
        if (clamped) {
            warn = true;
            msg += "laser 档越界 clamp; ";
        }
    } else {
        out.snapshot.laserTier = -1;
        warn = true;
        msg += "laser 表空，tier=-1; ";
    }

    out.frameId = frameId;
    out.temperature = temperatureC;
    out.grayL = grayL.clone();
    out.grayR = grayR.clone();
    out.d_grayL = nullptr;                 // 08 接入期真上传，06 不碰 CUDA
    out.d_grayR = nullptr;

    if (warn) return Result::warning(msg);
    return Result::ok();
}

} // namespace Scanner::data
