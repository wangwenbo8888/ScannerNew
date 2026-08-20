// ============================================================================
// test_self_check_collector.cpp — SelfCheckCollector 上位机自检采集单测（H-T15）
//
// 契约钉死（2026-08-20 08 设计 §5.2；可测部分）：
//   - 只采集不判定（阈值归 10-PerfMonitor）；Honest：取不到 -1 不编数
//   - 内存/磁盘：真机 Win32 采集（Windows 必过）+ 探针注入双防线
//   - CPU 温度：WMI 降级不采 → cpuTempC 恒 -1（口径钉死）
//   - CPU 占用：PDH 首调 -1（仅基线样本）；二次起 ∈ [-1,100]
//   - GPU：NVML 动态加载，无 nvml.dll → -1（有卡真值/无卡 -1 两态皆合法）
//   - 心跳：注册制；beat 未注册名静默忽略不崩；超时判定 nowMs 注入（steady 同域）
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/SelfCheckCollector.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace Scanner::device;

namespace {
bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}
} // namespace

// —— 1. MemDiskReal：真机 Win32 采集 → 内存水位 (0,100]、磁盘剩余 >0、时间戳回填 ——
TEST(SelfCheckCollector, MemDiskReal) {
    SelfCheckCollector c;
    const auto m = c.collect(123456);
    EXPECT_GT(m.memPercent, 0.0);
    EXPECT_LE(m.memPercent, 100.0);
    EXPECT_GT(m.diskFreeGB, 0.0);
    EXPECT_EQ(m.timestampMs, 123456);
}

// —— 2. CpuTempHonestNegative：CPU 温度降级口径钉死恒 -1（WMI 不采） ——
TEST(SelfCheckCollector, CpuTempHonestNegative) {
    SelfCheckCollector c;
    EXPECT_DOUBLE_EQ(c.collect(1).cpuTempC, -1.0);
}

// —— 3. CpuPercentSecondCall：首调 -1（基线口径）；间隔 >100ms 二次调 ∈ [-1,100] ——
TEST(SelfCheckCollector, CpuPercentSecondCall) {
    SelfCheckCollector c;
    EXPECT_DOUBLE_EQ(c.collect(1000).cpuPercent, -1.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto m = c.collect(2000);
    EXPECT_GE(m.cpuPercent, -1.0);
    EXPECT_LE(m.cpuPercent, 100.0);
}

// —— 4. DiskProbeInjectable：磁盘探针注入 → diskFreeGB 原样回传 ——
TEST(SelfCheckCollector, DiskProbeInjectable) {
    SelfCheckCollector c;
    c.setDiskProbe([](const std::string& /*path*/) { return 42.5; });
    EXPECT_DOUBLE_EQ(c.collect(7).diskFreeGB, 42.5);
}

// —— 5. MemProbeInjectable：内存探针注入 → memPercent 原样回传 ——
TEST(SelfCheckCollector, MemProbeInjectable) {
    SelfCheckCollector c;
    c.setMemProbe([]() { return 66.0; });
    EXPECT_DOUBLE_EQ(c.collect(8).memPercent, 66.0);
}

// —— 6. HeartbeatStale：A(100ms)/B(1000ms) 同拍 beat → +200ms 仅 A 超时，+1500ms 双超时 ——
TEST(SelfCheckCollector, HeartbeatStale) {
    SelfCheckCollector c;
    c.registerHeartbeat("A", 100);
    c.registerHeartbeat("B", 1000);
    const int64_t t0 = c.steadyNowMs();
    c.beat("A");
    c.beat("B");

    auto stale = c.staleHeartbeats(t0 + 200);
    ASSERT_EQ(stale.size(), 1u);
    EXPECT_EQ(stale[0], "A");

    stale = c.staleHeartbeats(t0 + 1500);
    ASSERT_EQ(stale.size(), 2u);
    EXPECT_EQ(stale[0], "A");
    EXPECT_EQ(stale[1], "B");
}

// —— 7. HeartbeatUnknownBeat：beat 未注册名 → 静默忽略不崩、不进超时名单 ——
TEST(SelfCheckCollector, HeartbeatUnknownBeat) {
    SelfCheckCollector c;
    c.beat("ghost");
    EXPECT_TRUE(c.staleHeartbeats(c.steadyNowMs() + 100000).empty());
}

// —— 8. GpuNoNvmlOrReal：有卡真值/无卡 -1 两态皆合法（≥-1；带值则界内） ——
TEST(SelfCheckCollector, GpuNoNvmlOrReal) {
    SelfCheckCollector c;
    const auto m = c.collect(9);
    EXPECT_GE(m.gpuTempC, -1.0);
    EXPECT_GE(m.gpuMemPercent, -1.0);
    if (m.gpuTempC > -1.0) EXPECT_LE(m.gpuTempC, 150.0);
    if (m.gpuMemPercent > -1.0) EXPECT_LE(m.gpuMemPercent, 100.0);
}
