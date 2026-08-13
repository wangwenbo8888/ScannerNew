#include <gtest/gtest.h>
#include "data/RingBuffer.h"
#include <thread>
#include <vector>
#include <atomic>

using namespace Scanner::data;

TEST(RingBufferTest, PushPopBasic) {
    RingBuffer<int> buf(4);
    EXPECT_TRUE(buf.push(1));
    EXPECT_TRUE(buf.push(2));
    EXPECT_TRUE(buf.push(3));
    EXPECT_EQ(buf.size(), 3u);

    auto val = buf.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 1);
}

TEST(RingBufferTest, DropOldestWhenFull) {
    RingBuffer<int> buf(2, OverflowPolicy::DropOldest);
    buf.push(1);
    buf.push(2);
    buf.push(3);  // drops 1

    EXPECT_EQ(buf.size(), 2u);
    auto val = buf.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 2);
}

TEST(RingBufferTest, DropNewestWhenFull) {
    RingBuffer<int> buf(2, OverflowPolicy::DropNewest);
    buf.push(1);
    buf.push(2);
    EXPECT_FALSE(buf.push(3));  // rejected

    EXPECT_EQ(buf.size(), 2u);
}

TEST(RingBufferTest, ClearEmptiesBuffer) {
    RingBuffer<int> buf(4);
    buf.push(1);
    buf.push(2);
    buf.clear();
    EXPECT_TRUE(buf.empty());
}

TEST(RingBufferTest, ProducerConsumer) {
    const int capacity = 1024;
    const int count = 1000;
    RingBuffer<int> buf(capacity, OverflowPolicy::Block);

    std::atomic<int> consumed{0};

    std::thread producer([&]() {
        for (int i = 0; i < count; ++i) {
            buf.push(i);
        }
    });

    std::vector<int> received;
    std::thread consumer([&]() {
        while (consumed.load() < count) {
            auto val = buf.pop(std::chrono::milliseconds(500));
            if (val.has_value()) {
                received.push_back(*val);
                consumed.fetch_add(1);
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(static_cast<int>(received.size()), count);
}
