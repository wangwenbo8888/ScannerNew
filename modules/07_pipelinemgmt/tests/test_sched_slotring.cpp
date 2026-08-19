#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <vector>
#include "SlotRing.h"
using Scanner::data::SlotRing;

namespace {
struct Frame {
    uint64_t id;
    int payload;
};
auto mkFrame = [](uint64_t id, int payload) {
    return std::make_shared<Frame>(Frame{id, payload});
};
} // namespace

// 语义 1：Overwrite 写读回环——写 3 帧按帧号读回，字段各对
TEST(SlotRing, WriteReadRoundtrip) {
    SlotRing<Frame> ring(4, SlotRing<Frame>::WriterMode::Overwrite);
    for (uint64_t i = 0; i < 3; ++i) ring.write(mkFrame(i, static_cast<int>(i) * 11));
    EXPECT_EQ(ring.writePtr(), 3u);
    for (uint64_t i = 0; i < 3; ++i) {
        auto f = ring.read(i);
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(f->id, i);
        EXPECT_EQ(f->payload, static_cast<int>(i) * 11);
    }
}

// 语义 2：覆盖后旧引用仍安全——覆盖前取得的 shared_ptr 不悬空、值不变；
//         确定已覆盖的帧号 read 返回 nullptr；claim 从 0 号发起
TEST(SlotRing, OverwriteKeepsRefs) {
    SlotRing<Frame> ring(2, SlotRing<Frame>::WriterMode::Overwrite);
    ring.write(mkFrame(0, 100));
    auto kept = ring.read(0);                       // 覆盖前持引用
    ASSERT_NE(kept, nullptr);
    ring.write(mkFrame(1, 101));
    ring.write(mkFrame(2, 102));                    // 覆盖槽 0
    EXPECT_EQ(ring.writePtr(), 3u);
    ASSERT_NE(kept, nullptr);                       // 旧对象不悬空
    EXPECT_EQ(kept->id, 0u);                        // 值仍是第 0 帧
    EXPECT_EQ(kept->payload, 100);
    EXPECT_EQ(ring.read(0), nullptr);               // 0 + 2 < 3：确定已覆盖
    EXPECT_EQ(ring.claim(), 0u);                    // 领 0 号成功
}

// 语义 3：claim 多线程原子领号——4 线程各 2500 次，无重复且 min=0 max=9999
TEST(SlotRing, ClaimUniqueMonotonic) {
    SlotRing<Frame> ring(8);
    constexpr int kThreads = 4, kPerThread = 2500;
    std::vector<std::vector<uint64_t>> got(kThreads);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&ring, &got, t, kPerThread] {
            for (int i = 0; i < kPerThread; ++i) got[t].push_back(ring.claim());
        });
    }
    for (auto& th : ts) th.join();

    std::set<uint64_t> all;
    for (const auto& v : got) all.insert(v.begin(), v.end());
    ASSERT_EQ(all.size(), static_cast<size_t>(kThreads) * kPerThread);
    EXPECT_EQ(*all.begin(), 0u);
    EXPECT_EQ(*all.rbegin(), static_cast<uint64_t>(kThreads * kPerThread - 1));
}

// 语义 4：Backpressure 写满阻塞——第 3 写阻塞至 done() 两次腾位后完成
TEST(SlotRing, BackpressureBlocksUntilDone) {
    SlotRing<Frame> ring(2, SlotRing<Frame>::WriterMode::Backpressure);
    ring.write(mkFrame(0, 0));
    ring.write(mkFrame(1, 1));                      // 已满：2 - 0 >= 2
    std::atomic<bool> entered{false}, finished{false};
    std::thread writer([&] {
        entered = true;
        ring.write(mkFrame(2, 2));                  // 应阻塞
        finished = true;
    });
    while (!entered.load()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(finished.load());                  // 仍阻塞
    ring.done();
    ring.done();                                    // 腾两位
    writer.join();
    EXPECT_TRUE(finished.load());
    EXPECT_EQ(ring.donePtr(), 2u);
    EXPECT_EQ(ring.writePtr(), 3u);
}

// 语义 5：waitFor 超时与兑现——无写入超时 false；写入后立即 true
TEST(SlotRing, WaitForTimeoutAndSignal) {
    SlotRing<Frame> ring(8);
    EXPECT_FALSE(ring.waitFor(5, std::chrono::milliseconds(50)));   // 无写入
    for (uint64_t i = 0; i < 6; ++i) ring.write(mkFrame(i, static_cast<int>(i)));
    EXPECT_TRUE(ring.waitFor(5, std::chrono::milliseconds(100)));   // writePtr=6 > 5
}

// 语义 6：未写入帧号 read 返回 nullptr
TEST(SlotRing, ReadUnwrittenIsNull) {
    SlotRing<Frame> ring(4);
    EXPECT_EQ(ring.read(99), nullptr);
    EXPECT_EQ(ring.read(0), nullptr);
}
