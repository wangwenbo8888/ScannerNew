#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>
#include "CycleUnit.h"
#include "pipelines/posture/PostureConfirmTable.h"
#include "pipelines/posture/PostureTypes.h"

using Scanner::data::CycleUnit;
using Scanner::pipeline::PostureConfirmTable;
using Scanner::pipeline::PostureSessionData;

namespace {

constexpr int kTargets = PostureSessionData::kTargetCount;   // 25

// 目标 i：单位旋转 + T=(i*100, 0, 0)（间隔 100mm >> 阈值 5mm，互不串扰）
void targetPose(int i, double R[9], double T[3]) {
    const double M[9] = {1,0,0, 0,1,0, 0,0,1};
    std::copy(M, M + 9, R);
    T[0] = i * 100.0; T[1] = 0.0; T[2] = 0.0;
}

// 25×16 row-major 目标矩阵（row-major 4×4：R 在左 3×3、T 在第 4 列）
void fillTargets(double t[kTargets][16]) {
    for (int i = 0; i < kTargets; ++i) {
        double R[9], T[3]; targetPose(i, R, T);
        auto& m = t[i];
        std::fill(m, m + 16, 0.0);
        m[0] = R[0]; m[1] = R[1]; m[2] = R[2];   m[3] = T[0];
        m[4] = R[3]; m[5] = R[4]; m[6] = R[5];   m[7] = T[1];
        m[8] = R[6]; m[9] = R[7]; m[10] = R[8];  m[11] = T[2];
        m[15] = 1.0;
    }
}

// 绕 Z 轴旋转（deg）的 row-major 3×3
void rotZ(double deg, double R[9]) {
    const double a = deg * M_PI / 180.0, c = std::cos(a), s = std::sin(a);
    R[0] = c;    R[1] = -s;   R[2] = 0.0;
    R[3] = s;    R[4] = c;    R[5] = 0.0;
    R[6] = 0.0;  R[7] = 0.0;  R[8] = 1.0;
}

// id 唯一、内容可校验的周期单元
CycleUnit makeCycle(uint64_t id) {
    CycleUnit c;
    c.id = id;
    c.temperature = 20.0 + static_cast<double>(id);
    c.timestamp = 1000 + id;
    c.markerL = cv::Mat(8, 8, CV_8UC1, cv::Scalar(static_cast<double>(id)));
    c.markerR = cv::Mat(8, 8, CV_8UC1, cv::Scalar(static_cast<double>(id + 1)));
    c.laserFrames.resize(4);
    return c;
}

std::vector<cv::Point2f> centers(float tag) { return {{tag, tag}, {tag + 1, tag + 2}}; }

// 回调计数夹具（onCollected/onComplete 线程安全计数）
struct Counters {
    std::atomic<int> collected{0};
    std::atomic<int> lastIdx{-1};
    std::atomic<int> complete{0};
    std::mutex mu;
    std::vector<int> idxLog;
    PostureConfirmTable::OnCollected onCollected() {
        return [this](int i) {
            collected.fetch_add(1);
            lastIdx.store(i);
            std::lock_guard<std::mutex> lk(mu);
            idxLog.push_back(i);
        };
    }
    PostureConfirmTable::OnComplete onComplete() {
        return [this]() { complete.fetch_add(1); };
    }
};

// 精确命中目标 i（ΔT=0、角度 0）
int reportHit(PostureConfirmTable& t, int i, uint64_t cycleId) {
    double R[9], T[3]; targetPose(i, R, T);
    return t.report(R, T, makeCycle(cycleId), centers(static_cast<float>(cycleId)),
                    centers(static_cast<float>(cycleId + 0.5f)));
}

} // namespace

// 语义 1：连续命中同一目标 3 次 → 第 3 次 report 返回该 idx、计数 1、回调(0) 一次
TEST(PostureConfirm, ThreeStrikesConfirms) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    EXPECT_EQ(reportHit(table, 0, 1), -1);
    EXPECT_EQ(reportHit(table, 0, 2), -1);
    EXPECT_EQ(reportHit(table, 0, 3), 0);
    EXPECT_EQ(table.collectedCount(), 1);
    EXPECT_EQ(ct.collected.load(), 1);
    EXPECT_EQ(ct.lastIdx.load(), 0);
    EXPECT_EQ(ct.complete.load(), 0);
    EXPECT_FALSE(table.isComplete());
}

// 语义 2：命中 2 次→miss（无候选）→streak 清零；再 3 次命中才确认
TEST(PostureConfirm, StreakResetOnMiss) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    EXPECT_EQ(reportHit(table, 0, 1), -1);
    EXPECT_EQ(reportHit(table, 0, 2), -1);
    // miss：远离全部目标（T 偏 500mm，无候选）——周期被丢弃，streak 清零
    double R[9], T[3]; targetPose(0, R, T); T[0] += 500.0;
    EXPECT_EQ(table.report(R, T, makeCycle(99), centers(99), centers(99)), -1);
    EXPECT_EQ(reportHit(table, 0, 4), -1);   // streak 重新从 1 计：第 3 次整体击中不确认
    EXPECT_EQ(reportHit(table, 0, 5), -1);
    EXPECT_EQ(reportHit(table, 0, 6), 0);    // 新 streak 第 3 次才确认
    EXPECT_EQ(table.collectedCount(), 1);
    EXPECT_EQ(ct.collected.load(), 1);
}

// 语义 3：命中目标 0 两次→改报目标 1 →streak 随目标切换（目标 1 从 1 计，3 次确认）
TEST(PostureConfirm, StreakResetOnSwitch) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    EXPECT_EQ(reportHit(table, 0, 1), -1);
    EXPECT_EQ(reportHit(table, 0, 2), -1);
    EXPECT_EQ(reportHit(table, 1, 3), -1);   // 整体第 3 连击但换了目标：不确认
    EXPECT_EQ(reportHit(table, 1, 4), -1);
    EXPECT_EQ(reportHit(table, 1, 5), 1);    // 目标 1 自身 streak 到 3
    EXPECT_EQ(table.collectedCount(), 1);
    EXPECT_EQ(ct.lastIdx.load(), 1);
}

// 语义 4：ΔT=10mm（>5）无候选不确认；角度 10deg（>5）同理
TEST(PostureConfirm, NoConfirmBeyondThreshold) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    // 平移超阈
    double R[9], T[3]; targetPose(0, R, T); T[0] += 10.0;
    for (uint64_t id = 1; id <= 5; ++id)
        EXPECT_EQ(table.report(R, T, makeCycle(id), centers(1), centers(1)), -1);
    // 角度超阈（ΔT=0，绕 Z 10°）
    double Rz[9]; rotZ(10.0, Rz); targetPose(0, R, T);
    for (uint64_t id = 10; id <= 15; ++id)
        EXPECT_EQ(table.report(Rz, T, makeCycle(id), centers(1), centers(1)), -1);
    EXPECT_EQ(table.collectedCount(), 0);
    EXPECT_EQ(ct.collected.load(), 0);
}

// 语义 5：streak 已=2 时 8 线程并发各报一次 → 恰一确认（CAS 防重；poses[0] 单次写入）
TEST(PostureConfirm, ConcurrentDuplicateConfirmOnce) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    ASSERT_EQ(reportHit(table, 0, 1), -1);
    ASSERT_EQ(reportHit(table, 0, 2), -1);   // streak=2 就绪

    constexpr int kThreads = 8;
    std::vector<std::thread> pool;
    for (int k = 0; k < kThreads; ++k) {
        pool.emplace_back([&, k] {
            double R[9], T[3]; targetPose(0, R, T);
            table.report(R, T, makeCycle(100 + k), centers(static_cast<float>(100 + k)),
                         centers(static_cast<float>(100 + k)));
        });
    }
    for (auto& th : pool) th.join();

    EXPECT_EQ(table.collectedCount(), 1);
    EXPECT_EQ(ct.collected.load(), 1);
    EXPECT_EQ(ct.complete.load(), 0);

    auto s = table.takeSessionData();
    ASSERT_TRUE(s.collected[0]);
    EXPECT_EQ(s.collectedCount, 1);
    // 单次写入：cycleId 是 8 个并发周期之一，且内部一致（cycle.id == cycleId）
    const uint64_t got = s.poses[0].cycleId;
    EXPECT_TRUE(got >= 100 && got <= 107) << "cycleId=" << got;
    EXPECT_EQ(s.poses[0].cycle.id, got);
}

// 语义 6：24 姿态已确认 → 第 25 确认时 onComplete 恰一次；再报不重触发
TEST(PostureConfirm, CompleteFiresOnce) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    for (int i = 1; i < kTargets; ++i) {          // 预置 1..24 已确认
        ASSERT_EQ(reportHit(table, i, 10 + i), -1);
        ASSERT_EQ(reportHit(table, i, 20 + i), -1);
        ASSERT_EQ(reportHit(table, i, 30 + i), i);
    }
    EXPECT_EQ(ct.complete.load(), 0);
    EXPECT_EQ(reportHit(table, 0, 1), -1);
    EXPECT_EQ(reportHit(table, 0, 2), -1);
    EXPECT_EQ(reportHit(table, 0, 3), 0);          // 第 25 个 → complete
    EXPECT_TRUE(table.isComplete());
    EXPECT_EQ(ct.complete.load(), 1);
    EXPECT_EQ(table.collectedCount(), kTargets);
    EXPECT_EQ(reportHit(table, 0, 4), -1);         // 已确认姿态再报：无新确认
    EXPECT_EQ(ct.complete.load(), 1);              // 不重触发
    EXPECT_EQ(ct.collected.load(), kTargets);
}

// 语义 7：确认 2 姿态后 takeSessionData 导出——字段/collected/计数一致
TEST(PostureConfirm, TakeSessionDataRoundtrip) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    const auto l3 = centers(3), r3 = centers(3.5f);
    const auto l7 = centers(7), r7 = centers(7.5f);
    {
        double R[9], T[3]; targetPose(0, R, T);
        table.report(R, T, makeCycle(3), centers(3), centers(3.5f));
        table.report(R, T, makeCycle(4), centers(3), centers(3.5f));
        table.report(R, T, makeCycle(5), centers(3), centers(3.5f));   // 确认 0（末帧 cycleId=5）
    }
    {
        double R[9], T[3]; targetPose(1, R, T);
        table.report(R, T, makeCycle(7), centers(7), centers(7.5f));
        table.report(R, T, makeCycle(8), centers(7), centers(7.5f));
        table.report(R, T, makeCycle(9), centers(7), centers(7.5f));   // 确认 1（cycleId=9）
    }

    auto s = table.takeSessionData();
    ASSERT_EQ(s.collectedCount, 2);
    EXPECT_TRUE(s.collected[0]);
    EXPECT_TRUE(s.collected[1]);
    EXPECT_FALSE(s.collected[2]);
    EXPECT_FALSE(s.collected[24]);

    // pose 0：确认时刻的 R/T/cycle/椭圆中心
    EXPECT_EQ(s.poses[0].cycleId, 5u);
    EXPECT_EQ(s.poses[0].cycle.id, 5u);
    EXPECT_DOUBLE_EQ(s.poses[0].cycle.temperature, 25.0);
    EXPECT_DOUBLE_EQ(s.poses[0].T[0], 0.0);
    EXPECT_EQ(s.poses[0].ellipseCentersL, l3);
    EXPECT_EQ(s.poses[0].ellipseCentersR, r3);
    // pose 1
    EXPECT_EQ(s.poses[1].cycleId, 9u);
    EXPECT_EQ(s.poses[1].cycle.id, 9u);
    EXPECT_DOUBLE_EQ(s.poses[1].T[0], 100.0);
    EXPECT_EQ(s.poses[1].ellipseCentersL, l7);
    EXPECT_EQ(s.poses[1].ellipseCentersR, r7);
}

// 语义 8：边界语义——ΔT 恰=5.0mm 不算命中（严格 <）；4.9mm 命中
TEST(PostureConfirm, ThresholdBoundary) {
    double targets[kTargets][16]; fillTargets(targets);
    Counters ct;
    PostureConfirmTable table(targets, kTargets,
                              PostureConfirmTable::Config{}, ct.onCollected(), ct.onComplete());
    // ΔT 恰为 5.0（sqrt(25) 精确）：不命中——连报不确认
    double R[9], T[3]; targetPose(0, R, T); T[0] += 5.0;
    for (uint64_t id = 1; id <= 4; ++id)
        EXPECT_EQ(table.report(R, T, makeCycle(id), centers(1), centers(1)), -1);
    EXPECT_EQ(table.collectedCount(), 0);
    // ΔT=4.9（< 5）：命中
    targetPose(0, R, T); T[0] += 4.9;
    EXPECT_EQ(table.report(R, T, makeCycle(5), centers(1), centers(1)), -1);
    EXPECT_EQ(table.report(R, T, makeCycle(6), centers(1), centers(1)), -1);
    EXPECT_EQ(table.report(R, T, makeCycle(7), centers(1), centers(1)), 0);
    EXPECT_EQ(table.collectedCount(), 1);
}
