// ============================================================================
// test_sched_framesource.cpp — IFrameSource 两副面孔单测（调度底座）
// GrabLatestSource：扫描面孔（各 lane 自持 counter；落后超阈值跳最新丢中间）
// SequentialSource：姿态面孔（claim→waitFor→read 顺序全取不跳；超时 nullptr）
// ============================================================================
// myReadCounter 契约（grabLatest）：in/out——传入"下一个待读帧号"（初始 0）；
// 成功取帧后推进为已取帧号+1（lane 循环不得重复消费同一帧）。
// 指导用例 2 原文"counter 变 19"据此修正为 20（跳帧目标=19，取完推进=20）。
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "SlotRing.h"
#include "sched/IFrameSource.h"

using Scanner::data::SlotRing;
using Scanner::pipeline::sched::GrabLatestSource;
using Scanner::pipeline::sched::SequentialSource;

namespace {
struct Frame {
    uint64_t id;
    int payload;
};
auto mkFrame = [](uint64_t id, int payload = 0) {
    return std::make_shared<Frame>(Frame{id, payload});
};
} // namespace

// 语义 1：GrabLatest 正常路径——落后 3 ≤ 16：取自己领的帧 2，不跳
TEST(GrabLatest, NormalPathTakesOwnFrame) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 16);
    for (uint64_t i = 0; i < 5; ++i) ring.write(mkFrame(i, static_cast<int>(i) * 11));
    uint64_t counter = 2;
    size_t skipped = 999;
    auto f = src.grabLatest(counter, skipped);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->id, 2u);
    EXPECT_EQ(f->payload, 22);
    EXPECT_EQ(skipped, 0u);
    EXPECT_EQ(counter, 3u);          // counter = 已取帧号 + 1（下次从 3 继续，不重复取 2）
}

// 语义 2：GrabLatest 落后超阈值——lag=20 > 4：跳到最新帧 19，skipped=lag-1=19
TEST(GrabLatest, SkipsToLatestWhenLagExceedsThreshold) {
    SlotRing<Frame> ring(8, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 4);
    for (uint64_t i = 0; i < 20; ++i) ring.write(mkFrame(i));
    uint64_t counter = 0;
    size_t skipped = 0;
    auto f = src.grabLatest(counter, skipped);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->id, 19u);           // 最新帧
    EXPECT_EQ(skipped, 19u);         // 帧 0..18 被丢弃 = lag-1
    EXPECT_EQ(counter, 20u);         // 跳帧目标 19 取完推进到 20
    EXPECT_EQ(src.writePtr(), 20u);
}

// 语义 2b：GrabLatest 竞态重试分支——落后 10 未超阈值 100（不触发前置跳帧），
//          但帧 0 在容量 4 环中确定已被覆盖（read(0)=null）→ 走"跳最新重试一次"
//          路径：取到帧 9、skipped=9（重试路径补记 0..8）、counter=10
TEST(GrabLatest, RetryOnOverwritten) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 100);         // 阈值宽松：不触发前置跳帧
    for (uint64_t i = 0; i < 10; ++i) ring.write(mkFrame(i));
    EXPECT_EQ(ring.read(0), nullptr);               // 前置：帧 0 确定已被覆盖
    uint64_t counter = 0;
    size_t skipped = 0;
    auto f = src.grabLatest(counter, skipped);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->id, 9u);                           // 重试跳最新取到帧 9
    EXPECT_EQ(skipped, 9u);                         // 重试分支补记 0..8 被丢
    EXPECT_EQ(counter, 10u);
}

// 语义 3：GrabLatest 空环——无帧返回 nullptr 不崩；计数不动；grabNext 不支持恒 nullptr
TEST(GrabLatest, EmptyRingReturnsNull) {
    SlotRing<Frame> ring(4);
    GrabLatestSource<Frame> src(ring, 16);
    uint64_t counter = 0;
    size_t skipped = 99;
    EXPECT_EQ(src.grabLatest(counter, skipped), nullptr);
    EXPECT_EQ(skipped, 0u);
    EXPECT_EQ(counter, 0u);          // 无帧不动计数
    EXPECT_EQ(src.writePtr(), 0u);
    // 扫描面孔不支持顺序消费：grabNext 恒 nullptr 且不动 counter
    EXPECT_EQ(src.grabNext(counter), nullptr);
    EXPECT_EQ(counter, 0u);
}

// 语义 4：多 lane 各自 counter 独立推进——同一 source 上两 lane 分别取 3、5 帧，
//         各自从自己的 counter 续取，互不干扰（counter 由 lane 外部持有）
TEST(GrabLatest, LaneCountersAdvanceIndependently) {
    SlotRing<Frame> ring(16, SlotRing<Frame>::WriterMode::Overwrite);
    GrabLatestSource<Frame> src(ring, 100);
    for (uint64_t i = 0; i < 8; ++i) ring.write(mkFrame(i));
    uint64_t cA = 0, cB = 0;
    size_t skipped = 99;
    std::vector<uint64_t> gotA, gotB;
    for (int i = 0; i < 3; ++i) {
        auto f = src.grabLatest(cA, skipped);
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(skipped, 0u);
        gotA.push_back(f->id);
    }
    for (int i = 0; i < 5; ++i) {
        auto f = src.grabLatest(cB, skipped);
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(skipped, 0u);
        gotB.push_back(f->id);
    }
    EXPECT_EQ(gotA, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(gotB, (std::vector<uint64_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(cA, 3u);
    EXPECT_EQ(cB, 5u);
    auto f = src.grabLatest(cA, skipped);   // A 从自己的 3 续取，不受 B 推进影响
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->id, 3u);
    EXPECT_EQ(cA, 4u);
    EXPECT_EQ(cB, 5u);
}

// 语义 5：Sequential 顺序全取不跳——写 10 取 10 得 0..9 无缺；写侧继续写
//         不影响已取引用；顺序继续推进；Overwrite 环 done() 空操作无害
TEST(Sequential, InOrderNoSkipAndWriterDoesNotDisturb) {
    SlotRing<Frame> ring(16, SlotRing<Frame>::WriterMode::Overwrite);
    SequentialSource<Frame> src(ring, std::chrono::milliseconds(100));
    for (uint64_t i = 0; i < 10; ++i) ring.write(mkFrame(i));
    uint64_t counter = 0;
    std::shared_ptr<const Frame> kept;
    for (uint64_t expect = 0; expect < 10; ++expect) {
        auto f = src.grabNext(counter);
        ASSERT_NE(f, nullptr) << "第 " << expect << " 帧未取到";
        EXPECT_EQ(f->id, expect);
        EXPECT_EQ(counter, expect);         // 回写本次取到的帧号
        if (expect == 0) kept = f;
    }
    for (uint64_t i = 10; i < 20; ++i) ring.write(mkFrame(i));  // 写侧继续写（覆盖旧槽）
    ASSERT_NE(kept, nullptr);
    EXPECT_EQ(kept->id, 0u);                // 已取引用在覆盖后依旧安全
    auto more = src.grabNext(counter);      // 顺序继续：下一帧是 10
    ASSERT_NE(more, nullptr);
    EXPECT_EQ(more->id, 10u);
    EXPECT_EQ(counter, 10u);
    EXPECT_EQ(src.writePtr(), 20u);
    EXPECT_EQ(ring.donePtr(), 0u);          // Overwrite：done 空操作，不记账
}

// 语义 6：Sequential 空环超时——waitFor 超时返回 nullptr、不推进帧号、不回写 counter；
//         之后补写，重试应取首个 claim 到的 0 号帧（证明超时未丢号）
TEST(Sequential, TimeoutReturnsNullAndRetriesSameFrame) {
    SlotRing<Frame> ring(4);
    SequentialSource<Frame> src(ring, std::chrono::milliseconds(50));
    // 姿态面孔不支持跳最新：grabLatest 恒 nullptr 且不动 counter/skipped
    uint64_t counter = 123;
    size_t skipped = 7;
    EXPECT_EQ(src.grabLatest(counter, skipped), nullptr);
    EXPECT_EQ(skipped, 0u);
    EXPECT_EQ(counter, 123u);

    const auto t0 = std::chrono::steady_clock::now();
    auto f = src.grabNext(counter);
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_EQ(f, nullptr);                  // 空环：超时 nullptr（lane 检查停止标志后重试）
    EXPECT_EQ(counter, 123u);               // 未取到不回写
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 45);

    ring.write(mkFrame(0));
    auto f2 = src.grabNext(counter);
    ASSERT_NE(f2, nullptr);
    EXPECT_EQ(f2->id, 0u);                  // 重试取同帧号 0（而非重新 claim 到 1）
    EXPECT_EQ(counter, 0u);
}

// 语义 7：Sequential 反压全量消费——4 槽 Backpressure 环写 30 帧，
//         每帧 done() 腾位驱动写侧走完；消费侧 30 帧全到、序号无缺无重
TEST(Sequential, ConsumesAllWithBackpressureDone) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Backpressure);
    SequentialSource<Frame> src(ring, std::chrono::milliseconds(50));
    constexpr uint64_t kTotal = 30;
    std::atomic<uint64_t> written{0};
    std::thread writer([&] {
        for (uint64_t i = 0; i < kTotal; ++i) {
            ring.write(mkFrame(i));
            written.store(i + 1, std::memory_order_release);
        }
    });
    uint64_t counter = 0;
    std::set<uint64_t> got;
    for (int attempts = 0; got.size() < kTotal && attempts < 200; ++attempts) {
        auto f = src.grabNext(counter);
        if (f) got.insert(f->id);
    }
    if (got.size() != kTotal) {             // done() 未按帧腾位 → 写侧仍阻塞：分离防挂死
        writer.detach();
        FAIL() << "仅消费 " << got.size() << " 帧（done() 未腾位，写侧卡死于 "
               << written.load(std::memory_order_acquire) << "）";
        return;
    }
    writer.join();                          // 消费到 29 ⇒ 第 30 次 write() 已返回，join 安全
    EXPECT_EQ(written.load(), kTotal);
    EXPECT_EQ(got.size(), static_cast<size_t>(kTotal));
    EXPECT_EQ(*got.begin(), 0u);
    EXPECT_EQ(*got.rbegin(), kTotal - 1);
    EXPECT_EQ(ring.donePtr(), kTotal);      // 每成功一帧调一次 done()
    EXPECT_EQ(ring.writePtr(), kTotal);
    EXPECT_EQ(counter, kTotal - 1);         // 回写最后一帧号 29
}
