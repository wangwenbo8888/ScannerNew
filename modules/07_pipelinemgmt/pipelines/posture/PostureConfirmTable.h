#pragma once
// ============================================================================
// PostureConfirmTable.h — 姿态确认簿记（A 姿态判断 · 01-D3 + 姿态调度 §9.3）
// ============================================================================
// 职责（E 核终段逐周期调用 report）：
//   1. 匹配：对 N 目标逐一算 ΔT（平移欧氏距 mm）+ 角度差（R_rel = R_t^T·R_live
//      的转角 deg，acos 钳域）。命中候选 = ΔT < transThresholdMm 且
//      angle < angleThresholdDeg（两阈值与门、严格小于——恰等于阈值不算命中）；
//      bestIdx 取候选中 ΔT 最小者；无候选 → 清 streak 返回 -1。
//   2. streak 确认：与 lastBestIdx 相同 → streak++；不同 → streak=1 记新
//      lastBestIdx。streak ≥ confirmStreak 且未确认 → CAS exchange(true)：
//      成功者独占写 poses[bestIdx]（cycle 移入）+ 计数 + 回调；失败不写。
//   3. 集齐回调：collectedCount == targetCount → onComplete 恰一次
//      （atomic flag 防重）。
//
// 线程模型：E 核单线程逐周期 report 为主用法；并发仅要求同一姿态确认恰一次
// （§9.3 CAS 防重复）。簿记为 per-pose 细粒度锁（竞争≈0）+ collected 原子位。
#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

#include "pipelines/posture/PostureTypes.h"

namespace Scanner::pipeline {

class PostureConfirmTable {
public:
    struct Config {
        int confirmStreak = 3;                // 连续命中 N 次才确认（可配，联调标定）
        double transThresholdMm = 5.0;        // 姿态匹配阈值（平移差，严格 <）
        double angleThresholdDeg = 5.0;       // 姿态匹配阈值（角度差，严格 <）
    };

    using OnCollected = std::function<void(int poseIdx)>;   // 单姿态确认回调（渲染切目标）
    using OnComplete = std::function<void()>;               // N 全齐回调

    /// targets：targetCount×16 row-major 4×4 目标姿态（R 左 3×3、T 第 4 列）
    /// targetCount 须 ≤ PostureSessionData::kTargetCount（导出定长数组）
    PostureConfirmTable(const double (*targets)[16], int targetCount,
                        Config cfg, OnCollected onCollected, OnComplete onComplete);

    /// E 核终段逐周期调用。liveR/liveT=pose_estimate 输出；cycle=待存周期数据
    /// （确认时移走，否则丢弃）；ellipseCenters=本帧 2-7 输出（确认时存）。
    /// 返回：命中且已确认的新姿态 idx；-1=无新确认（含已确认过的姿态）。
    int report(const double liveR[9], const double liveT[3],
               Scanner::data::CycleUnit&& cycle,
               std::vector<cv::Point2f>&& ellipseCentersL,
               std::vector<cv::Point2f>&& ellipseCentersR);

    bool isComplete() const;
    int collectedCount() const;
    PostureSessionData takeSessionData() const;    // 收口时导出

private:
    int match(const double liveR[9], const double liveT[3]) const;  // bestIdx 或 -1

    struct TargetPose {
        double R[9] = {1,0,0, 0,1,0, 0,0,1};
        double T[3] = {0,0,0};
    };

    int targetCount_ = 0;
    Config cfg_;
    OnCollected onCollected_;
    OnComplete onComplete_;
    std::vector<TargetPose> targets_;                  // 预转 R/T 缓存

    // per-pose 细粒度簿记：行锁护 streak/poses 写（takeSessionData 亦锁行，故 mutable）；
    // collected 原子位承 CAS
    mutable std::vector<std::mutex> mu_;
    std::vector<int> streak_;
    std::vector<PostureData> poses_;
    std::vector<std::atomic<bool>> collected_;
    std::atomic<int> lastBestIdx_{-1};
    std::atomic<int> collectedCount_{0};
    std::atomic<bool> completeFired_{false};
};

} // namespace Scanner::pipeline
