// ============================================================================
// PostureConfirmTable — 连续命中确认簿记（streak）+ CAS 防重复 + 集齐回调
// 行为契约见 PostureConfirmTable.h 头注释（两阈值与门、严格 < 边界语义）
// ============================================================================
#define _USE_MATH_DEFINES
#include <cmath>

#include <algorithm>
#include <utility>

#include "pipelines/posture/PostureConfirmTable.h"

namespace Scanner::pipeline {

PostureConfirmTable::PostureConfirmTable(const double (*targets)[16], int targetCount,
                                         Config cfg, OnCollected onCollected, OnComplete onComplete)
    : targetCount_(targetCount), cfg_(cfg), onCollected_(std::move(onCollected)),
      onComplete_(std::move(onComplete)), targets_(targetCount > 0 ? targetCount : 0),
      mu_(targetCount > 0 ? targetCount : 0), streak_(targetCount > 0 ? targetCount : 0, 0),
      poses_(targetCount > 0 ? targetCount : 0), collected_(targetCount > 0 ? targetCount : 0) {
    for (int i = 0; i < targetCount_; ++i) {
        const double* m = targets[i];
        TargetPose& tp = targets_[i];
        tp.R[0] = m[0];  tp.R[1] = m[1];  tp.R[2] = m[2];
        tp.R[3] = m[4];  tp.R[4] = m[5];  tp.R[5] = m[6];
        tp.R[6] = m[8];  tp.R[7] = m[9];  tp.R[8] = m[10];
        tp.T[0] = m[3];  tp.T[1] = m[7];  tp.T[2] = m[11];
    }
}

int PostureConfirmTable::match(const double liveR[9], const double liveT[3]) const {
    int best = -1;
    double bestDt = 0.0;
    for (int i = 0; i < targetCount_; ++i) {
        const TargetPose& tp = targets_[i];
        const double dx = tp.T[0] - liveT[0];
        const double dy = tp.T[1] - liveT[1];
        const double dz = tp.T[2] - liveT[2];
        const double dT = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!(dT < cfg_.transThresholdMm)) continue;      // 严格 <：恰=阈值不命中

        // 角度差：R_rel = R_target^T · R_live → trace = Σ_ij R_t[i][j]·R_l[i][j]
        // angle = acos(clamp((trace-1)/2))，钳域防浮点越界
        double tr = 0.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) tr += tp.R[r * 3 + c] * liveR[r * 3 + c];
        double cosA = (tr - 1.0) / 2.0;
        cosA = std::max(-1.0, std::min(1.0, cosA));
        const double angDeg = std::acos(cosA) * 180.0 / M_PI;
        if (!(angDeg < cfg_.angleThresholdDeg)) continue; // 严格 <

        if (best < 0 || dT < bestDt) {
            best = i;
            bestDt = dT;
        }
    }
    return best;
}

int PostureConfirmTable::report(const double liveR[9], const double liveT[3],
                                Scanner::data::CycleUnit&& cycle,
                                std::vector<cv::Point2f>&& ellipseCentersL,
                                std::vector<cv::Point2f>&& ellipseCentersR) {
    const int best = match(liveR, liveT);
    if (best < 0) {
        // 无候选：清当前 streak（复位 lastBestIdx 所在行后置 -1，下次命中从 1 计）
        const int last = lastBestIdx_.exchange(-1);
        if (last >= 0 && last < targetCount_) {
            std::lock_guard<std::mutex> lk(mu_[last]);
            streak_[last] = 0;
        }
        return -1;
    }

    // streak 簿记：同目标递增、换目标重计 1（lastBestIdx 语义见头注释）
    int streakNow;
    {
        std::lock_guard<std::mutex> lk(mu_[best]);
        if (lastBestIdx_.load(std::memory_order_acquire) == best) ++streak_[best];
        else streak_[best] = 1;
        lastBestIdx_.store(best, std::memory_order_release);
        streakNow = streak_[best];
    }
    if (streakNow < cfg_.confirmStreak) return -1;

    // CAS 防重复（§9.3）：恰一线程 exchange 成功者独占写该姿态
    bool expected = false;
    if (!collected_[best].compare_exchange_strong(expected, true)) return -1;

    {
        std::lock_guard<std::mutex> lk(mu_[best]);
        PostureData& p = poses_[best];
        p.cycleId = cycle.id;
        std::copy(liveR, liveR + 9, p.R);
        std::copy(liveT, liveT + 3, p.T);
        p.cycle = std::move(cycle);
        p.ellipseCentersL = std::move(ellipseCentersL);
        p.ellipseCentersR = std::move(ellipseCentersR);
    }

    const int n = collectedCount_.fetch_add(1) + 1;
    if (onCollected_) onCollected_(best);
    if (n == targetCount_ && !completeFired_.exchange(true) && onComplete_) onComplete_();
    return best;
}

bool PostureConfirmTable::isComplete() const {
    return collectedCount_.load(std::memory_order_acquire) == targetCount_;
}

int PostureConfirmTable::collectedCount() const {
    return collectedCount_.load(std::memory_order_acquire);
}

PostureSessionData PostureConfirmTable::takeSessionData() const {
    PostureSessionData s;
    const int n = std::min(targetCount_, PostureSessionData::kTargetCount);
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lk(mu_[i]);
        s.poses[i] = poses_[i];
        s.collected[i] = collected_[i].load(std::memory_order_acquire);
    }
    s.collectedCount = collectedCount_.load(std::memory_order_acquire);
    return s;
}

} // namespace Scanner::pipeline
