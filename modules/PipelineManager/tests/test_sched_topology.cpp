// ============================================================================
// test_sched_topology.cpp — CpuTopology 单测（computeLanes 纯函数 + 真机冒烟）
// ============================================================================
#include <atomic>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include "sched/CpuTopology.h"
using namespace Scanner::pipeline::sched;

TEST(Lanes, AutoDetectClamp) {
    EXPECT_EQ(computeLanes(8, 8, 0), 7);      // min(7,8)
    EXPECT_EQ(computeLanes(4, 2, 0), 2);      // E 核不足→E
    EXPECT_EQ(computeLanes(2, 0, 0), 1);      // 无 E 核退化→下限 1
    EXPECT_EQ(computeLanes(16, 16, 0), 8);    // 上限 8
    EXPECT_EQ(computeLanes(4, 4, 3), 3);      // 人工覆盖
}
TEST(Lanes, ZeroPcores) { EXPECT_EQ(computeLanes(1, 8, 0), 1); }
TEST(Topology, DetectRunsOnRealMachine) {
    auto t = CpuTopology::detect();           // 真机冒烟：不崩且总数>0
    // SUCCEED 消息仅在失败/XML 时显示，另 cout 一份保证控制台可见（人工核对混合架构）
    SUCCEED() << "真机探测: pCores=" << t.pCores << " eCores=" << t.eCores
              << " hybrid=" << (t.hybrid ? "true" : "false");
    std::cout << "[真机探测] pCores=" << t.pCores << " eCores=" << t.eCores
              << " hybrid=" << (t.hybrid ? "true" : "false") << std::endl;
    EXPECT_GT(t.pCores + t.eCores, 0);
}
TEST(Topology, PinAndRealtimeOnRealThread) {
    std::atomic<bool> ran{false};
    std::thread t([&] { ran = true; });
    CpuTopology::pinThread(t, 0x1);           // 绑核 0（真机必有）
    CpuTopology::setRealtime(t);              // 提优先级
    t.join();
    EXPECT_TRUE(ran.load());                  // 不崩即过（API 失败仅 warn 不抛）
}
