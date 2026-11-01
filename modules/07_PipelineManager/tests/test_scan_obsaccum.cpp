// ============================================================================
// test_scan_obsaccum.cpp — FrameObsAccumulator（逐帧观测累加器 + 激光帧缓存）
// 覆盖：往返导出 / 预算超限降级 / 预算边界(<=) / 快照深拷贝稳定 / clear 复位 / 空激光
// ============================================================================
#include <gtest/gtest.h>

#include <vector>

#include "pipelines/scan/FrameObsAccumulator.h"

using namespace Scanner::pipeline;

namespace {

FrameObs makeObs(uint64_t frameId, int nMarkers) {
    FrameObs fo;
    fo.frameId = frameId;
    fo.R_init[0] = 1.0 * frameId;
    fo.R_init[4] = 2.0 * frameId;
    fo.t_init[0] = 0.5 * frameId;
    fo.t_init[2] = -0.25 * frameId;
    for (int i = 0; i < nMarkers; ++i) {
        MarkerObs mo;
        mo.xyz[0] = 0.1 * frameId + i;
        mo.xyz[1] = 0.2 * frameId - i;
        mo.xyz[2] = 0.3 * frameId + 10 * i;
        mo.globalId = static_cast<int>(100 * frameId + i);
        mo.isHighPrecision = (i % 2 == 0);
        fo.markerObs.push_back(mo);
    }
    return fo;
}

void expectObsEquals(const FrameObs& got, const FrameObs& want) {
    EXPECT_EQ(got.frameId, want.frameId);
    for (int i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(got.R_init[i], want.R_init[i]);
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(got.t_init[i], want.t_init[i]);
    ASSERT_EQ(got.markerObs.size(), want.markerObs.size());
    for (size_t i = 0; i < want.markerObs.size(); ++i) {
        const auto& g = got.markerObs[i];
        const auto& w = want.markerObs[i];
        for (int k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(g.xyz[k], w.xyz[k]);
        EXPECT_EQ(g.globalId, w.globalId);
        EXPECT_EQ(g.isHighPrecision, w.isHighPrecision);
    }
}

} // namespace

// 1：push 3 帧（第 2 帧带激光 9 float）→ snapshot：obs 3 条字段逐帧一致；laserFrames[第2帧槽]==9 float
TEST(FrameObsAccumTest, PushThenSnapshotRoundtrip) {
    FrameObsAccumulator acc(1 << 20);
    std::vector<FrameObs> want;
    want.push_back(makeObs(1, 2));
    want.push_back(makeObs(2, 3));
    want.push_back(makeObs(3, 1));

    const std::vector<float> laser = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    acc.push(want[0], {});
    acc.push(want[1], laser);
    acc.push(want[2], {});
    ASSERT_EQ(acc.frameCount(), 3u);

    auto snap = acc.snapshot();
    ASSERT_EQ(snap.obs.size(), 3u);
    for (int i = 0; i < 3; ++i) expectObsEquals(snap.obs[i], want[i]);
    EXPECT_EQ(snap.obs[0].laserCacheSlot, FrameObs::kNoLaserSlot);   // 无激光帧
    EXPECT_EQ(snap.obs[2].laserCacheSlot, FrameObs::kNoLaserSlot);
    const size_t slot1 = snap.obs[1].laserCacheSlot;
    ASSERT_NE(slot1, FrameObs::kNoLaserSlot);
    ASSERT_LT(slot1, snap.laserFrames.size());
    EXPECT_EQ(snap.laserFrames[slot1], laser);
    EXPECT_EQ(acc.laserBytesUsed(), laser.size() * sizeof(float));
    EXPECT_FALSE(acc.degradedLaser());
}

// 2：预算 64 字节 → 第 1 帧小激光可入；第 2 帧大激光 → slot=kNoLaserSlot、degraded、markerObs 照常 2 帧
TEST(FrameObsAccumTest, LaserBudgetExceededDegrades) {
    FrameObsAccumulator acc(64);
    const std::vector<float> small(4, 1.f);         // 16 字节
    const std::vector<float> big(100, 2.f);         // 400 字节
    auto o1 = makeObs(1, 2);
    auto o2 = makeObs(2, 3);
    acc.push(o1, small);
    ASSERT_FALSE(acc.degradedLaser());
    acc.push(o2, big);

    EXPECT_TRUE(acc.degradedLaser());               // 超限停累加记 Degraded
    ASSERT_EQ(acc.frameCount(), 2u);                // 整体不失败
    auto snap = acc.snapshot();
    ASSERT_EQ(snap.obs.size(), 2u);
    EXPECT_EQ(snap.obs[0].laserCacheSlot, 0u);      // 第 1 帧小激光已入槽
    EXPECT_EQ(snap.obs[0].markerObs.size(), 2u);    // markerObs 照常累加
    EXPECT_EQ(snap.obs[1].laserCacheSlot, FrameObs::kNoLaserSlot);
    EXPECT_EQ(snap.obs[1].markerObs.size(), 3u);    // 超限帧 markerObs 也照常
    ASSERT_EQ(snap.laserFrames.size(), 1u);         // 大激光未入缓存
    EXPECT_EQ(snap.laserFrames[0].size(), 4u);
    EXPECT_EQ(acc.laserBytesUsed(), 16u);
}

// 3：恰好等于预算的帧可入（<= 判定）
TEST(FrameObsAccumTest, BudgetBoundaryExact) {
    const std::vector<float> exact(9, 7.f);         // 36 字节
    FrameObsAccumulator acc(exact.size() * sizeof(float));
    acc.push(makeObs(7, 1), exact);
    EXPECT_FALSE(acc.degradedLaser());
    EXPECT_EQ(acc.laserBytesUsed(), exact.size() * sizeof(float));
    auto snap = acc.snapshot();
    ASSERT_EQ(snap.obs.size(), 1u);
    EXPECT_EQ(snap.obs[0].laserCacheSlot, 0u);
    ASSERT_EQ(snap.laserFrames.size(), 1u);
    EXPECT_EQ(snap.laserFrames[0], exact);
}

// 4：snapshot 后再 push 不影响已导出快照（深拷贝验证）
TEST(FrameObsAccumTest, SnapshotStableAfterMorePush) {
    FrameObsAccumulator acc(1 << 20);
    const std::vector<float> laser = {9.f, 8.f, 7.f};
    auto o1 = makeObs(1, 1);
    acc.push(o1, laser);
    auto s1 = acc.snapshot();

    acc.push(makeObs(2, 4), {});                    // 快照后再 push
    acc.push(makeObs(3, 0), {1.f});

    // s1 不受后续 push 影响
    ASSERT_EQ(s1.obs.size(), 1u);
    expectObsEquals(s1.obs[0], o1);
    ASSERT_EQ(s1.laserFrames.size(), 1u);
    EXPECT_EQ(s1.laserFrames[0], laser);

    // s1 是深拷贝：改 s1 不影响累加器内部状态
    s1.obs[0].markerObs.clear();
    s1.laserFrames[0][0] = -999.f;
    auto s2 = acc.snapshot();
    ASSERT_EQ(s2.obs.size(), 3u);
    expectObsEquals(s2.obs[0], o1);
    ASSERT_EQ(s2.laserFrames.size(), 2u);
    EXPECT_EQ(s2.laserFrames[0][0], 9.f);
}

// 5：clear 后 frameCount=0、degraded=false、usedBytes=0（且可继续使用）
TEST(FrameObsAccumTest, ClearResets) {
    FrameObsAccumulator acc(64);
    acc.push(makeObs(1, 1), std::vector<float>(100, 1.f));   // 超限 → degraded
    ASSERT_TRUE(acc.degradedLaser());
    acc.push(makeObs(2, 1), {});
    ASSERT_EQ(acc.frameCount(), 2u);

    acc.clear();
    EXPECT_EQ(acc.frameCount(), 0u);
    EXPECT_FALSE(acc.degradedLaser());
    EXPECT_EQ(acc.laserBytesUsed(), 0u);
    auto snap = acc.snapshot();
    EXPECT_TRUE(snap.obs.empty());
    EXPECT_TRUE(snap.laserFrames.empty());

    acc.push(makeObs(3, 1), {1.f, 2.f});            // clear 复位后预算恢复可用
    EXPECT_EQ(acc.frameCount(), 1u);
    EXPECT_EQ(acc.laserBytesUsed(), 2 * sizeof(float));
    EXPECT_FALSE(acc.degradedLaser());
}

// 6：laserXyz 空 → slot=kNoLaserSlot、不占预算
TEST(FrameObsAccumTest, NullLaserFrame) {
    FrameObsAccumulator acc(1024);
    acc.push(makeObs(1, 2), {});
    EXPECT_EQ(acc.frameCount(), 1u);
    EXPECT_EQ(acc.laserBytesUsed(), 0u);
    EXPECT_FALSE(acc.degradedLaser());
    auto snap = acc.snapshot();
    ASSERT_EQ(snap.obs.size(), 1u);
    EXPECT_EQ(snap.obs[0].laserCacheSlot, FrameObs::kNoLaserSlot);
    EXPECT_TRUE(snap.laserFrames.empty());
}

// —— 编辑账本（05 D4·obs 侧，实施计划 P2）——

// 剔除后 snapshot 出口过滤已删 id；恢复（undo 路径）后回到全量
//（makeObs id 公式＝100*frameId+i：帧1→{100,101}，帧2→{200,201,202}）
TEST(FrameObsAccumTest, ExcludeMarkerObsFiltersSnapshot) {
    FrameObsAccumulator acc(1 << 20);
    acc.push(makeObs(1, 2), {});                      // ids 100,101
    acc.push(makeObs(2, 3), {});                      // ids 200,201,202
    ASSERT_EQ(acc.frameCount(), 2u);

    acc.excludeMarkerObs({101, 202}, true);
    EXPECT_EQ(acc.excludedMarkerObsCount(), 2u);

    auto snap = acc.snapshot();
    ASSERT_EQ(snap.obs.size(), 2u);
    ASSERT_EQ(snap.obs[0].markerObs.size(), 1u);      // 100 留
    EXPECT_EQ(snap.obs[0].markerObs[0].globalId, 100);
    ASSERT_EQ(snap.obs[1].markerObs.size(), 2u);      // 200,201 留
    EXPECT_EQ(snap.obs[1].markerObs[0].globalId, 200);
    EXPECT_EQ(snap.obs[1].markerObs[1].globalId, 201);

    acc.excludeMarkerObs({101, 202}, false);          // undo 恢复
    EXPECT_EQ(acc.excludedMarkerObsCount(), 0u);
    auto snap2 = acc.snapshot();
    ASSERT_EQ(snap2.obs[0].markerObs.size(), 2u);
    ASSERT_EQ(snap2.obs[1].markerObs.size(), 3u);
}

// 未知 id 幂等；负 id（链断观测）不参与；clear 清剔除集
TEST(FrameObsAccumTest, ExcludeMarkerObsIdempotentAndClear) {
    FrameObsAccumulator acc(1 << 20);
    acc.push(makeObs(1, 2), {});

    acc.excludeMarkerObs({999}, true);                // 未知 id
    EXPECT_EQ(acc.excludedMarkerObsCount(), 1u);
    EXPECT_EQ(acc.snapshot().obs[0].markerObs.size(), 2u);  // 不影响现有

    acc.excludeMarkerObs({-1}, true);                 // 负 id 忽略
    EXPECT_EQ(acc.excludedMarkerObsCount(), 1u);

    acc.excludeMarkerObs({999}, true);                // 重复剔除幂等
    EXPECT_EQ(acc.excludedMarkerObsCount(), 1u);

    acc.clear();
    EXPECT_EQ(acc.excludedMarkerObsCount(), 0u);
}