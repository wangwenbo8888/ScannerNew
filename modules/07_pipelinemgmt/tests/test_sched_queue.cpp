#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "sched/FrameResultQueue.h"
using Scanner::pipeline::sched::FrameResultQueue;

TEST(Queue, PushPopBasics) {
    FrameResultQueue<int> q(4);
    q.push(1); q.push(2);
    auto a = q.pop(std::chrono::milliseconds(50));
    auto b = q.pop(std::chrono::milliseconds(50));
    ASSERT_TRUE(a && b); EXPECT_EQ(*a, 1); EXPECT_EQ(*b, 2);
}
TEST(Queue, PopTimeoutReturnsNullopt) {
    FrameResultQueue<int> q(4);
    EXPECT_FALSE(q.pop(std::chrono::milliseconds(30)).has_value());
}
TEST(Queue, OverflowOverwritesOldest) {
    FrameResultQueue<int> q(2);
    q.push(1); q.push(2); q.push(3);          // 1 被覆盖
    EXPECT_EQ(q.dropped(), 1u);
    EXPECT_EQ(*q.pop(std::chrono::milliseconds(10)), 2);
    EXPECT_EQ(*q.pop(std::chrono::milliseconds(10)), 3);
}
TEST(Queue, PushNeverBlocksWhenFull) {
    FrameResultQueue<int> q(2);
    for (int i = 0; i < 1000; ++i) q.push(i);
    EXPECT_EQ(q.dropped(), 998u);
}
TEST(Queue, MpmcStress) {
    FrameResultQueue<int> q(64);
    std::atomic<int> produced{0}, consumed{0};
    std::atomic<bool> prod1_done{false}, prod2_done{false};
    auto producers_done = [&] { return prod1_done.load() && prod2_done.load(); };

    std::thread prod1([&] {
        for (int i = 0; i < 2000; ++i) { q.push(i); ++produced; }
        prod1_done = true;
    });
    std::thread prod2([&] {
        for (int i = 0; i < 2000; ++i) { q.push(i); ++produced; }
        prod2_done = true;
    });

    // 消费者：成功弹出则计数；超时空队列且生产已结束才退出（无死锁退出条件）
    auto consume = [&] {
        while (true) {
            if (q.pop(std::chrono::milliseconds(2)).has_value()) { ++consumed; continue; }
            if (producers_done()) break;
        }
    };
    std::thread cons1(consume);
    std::thread cons2(consume);

    prod1.join(); prod2.join(); cons1.join(); cons2.join();

    // 不变式：生产 == 消费 + 丢弃 + 残留（不死锁、不崩溃、不凭空增减）
    EXPECT_EQ(produced.load(), 4000);
    EXPECT_EQ(static_cast<size_t>(produced.load()),
              static_cast<size_t>(consumed.load()) + q.dropped() + q.size());
}
