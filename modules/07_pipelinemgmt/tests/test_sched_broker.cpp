#include <gtest/gtest.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "sched/PCoreBroker.h"
using Scanner::pipeline::sched::PCoreBroker;
using Scanner::Result;

TEST(Broker, StartRejectsBadArgs) {
    PCoreBroker b;
    EXPECT_FALSE(b.start(0, {}).success);          // 0 worker 非法
    EXPECT_FALSE(b.start(-1, {}).success);
}
TEST(Broker, ExecutesTasksAndReturnsFuture) {
    PCoreBroker b;
    ASSERT_TRUE(b.start(3, {}).success);           // 空 masks=不绑核（可测性）
    auto f = b.submit([] { return Result::ok("done"); });
    EXPECT_TRUE(f.get().success);
    b.shutdown();
}
TEST(Broker, ConcurrencyBoundedByWorkers) {
    PCoreBroker b; b.start(2, {});
    std::atomic<int> inFlight{0}, maxInFlight{0};
    std::vector<std::future<Result>> fs;
    for (int i = 0; i < 20; ++i)
        fs.push_back(b.submit([&] {
            int cur = ++inFlight;
            int old = maxInFlight.load();
            while (cur > old && !maxInFlight.compare_exchange_weak(old, cur)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            --inFlight; return Result::ok();
        }));
    for (auto& f : fs) f.get();
    EXPECT_LE(maxInFlight.load(), 2);
    b.shutdown();
}
TEST(Broker, ShutdownDrainsPending) {
    PCoreBroker b; b.start(1, {});
    std::atomic<int> executed{0};
    std::vector<std::future<Result>> fs;
    for (int i = 0; i < 10; ++i)
        fs.push_back(b.submit([&] { ++executed; return Result::ok(); }));
    b.shutdown();                                   // 排空后停
    EXPECT_EQ(executed.load(), 10);
}
TEST(Broker, FutureCapturesException) {
    PCoreBroker b; b.start(1, {});
    auto f = b.submit([]() -> Result { throw std::runtime_error("boom"); });
    EXPECT_THROW(f.get(), std::runtime_error);     // packaged_task 异常传递
    b.shutdown();
}
TEST(Broker, SubmitDuringShutdownNoHang) {
    // C1 回归：submit×shutdown 竞态不得孤儿化任务（future 永久悬挂）
    for (int round = 0; round < 100; ++round) {
        PCoreBroker b;
        ASSERT_TRUE(b.start(2, {}).success);
        std::vector<std::future<Result>> fs;
        std::mutex fsMutex;
        std::atomic<bool> stopFire{false};
        std::thread fire([&] {
            while (!stopFire.load()) {
                try {
                    auto f = b.submit([] { return Result::ok(); });
                    std::lock_guard<std::mutex> lk(fsMutex);
                    fs.push_back(std::move(f));
                } catch (const std::logic_error&) {
                    // shutdown 中/后提交被拒：正常
                }
            }
        });
        std::this_thread::sleep_for(std::chrono::microseconds(200 * (round % 10 + 1)));
        b.shutdown();                              // 随机（轮次交错）时刻停
        stopFire.store(true);
        fire.join();
        for (auto& f : fs) {                       // 成功拿到的 future 必须全部 ready
            ASSERT_EQ(f.wait_for(std::chrono::milliseconds(500)),
                      std::future_status::ready);
        }
    }
}
TEST(Broker, RestartAfterShutdown) {
    PCoreBroker b;
    for (int cycle = 0; cycle < 3; ++cycle) {
        ASSERT_TRUE(b.start(2, {}).success);
        auto f = b.submit([] { return Result::ok("cycle"); });
        EXPECT_TRUE(f.get().success);
        b.shutdown();
        EXPECT_FALSE(b.isRunning());
    }
}
