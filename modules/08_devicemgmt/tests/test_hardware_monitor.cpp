// ============================================================================
// test_hardware_monitor.cpp — HardwareMonitor 升级测（H-T16：删 IMCU 轮询/
// 帧率三件套/HealthMetrics 快照出口；08 设计 §5.2）
//
// 链接 mod_devicemgmt + mod_fileio（DeviceStateCache 真件；测试 exe 是叶子
// 消费方，引 06 头不违「06 不链 08」铁律——库间依赖方向不变）。
// 假件仅一处边界：LocalFakeCam（IScannerCamera 全接口空壳，isOpen/温度可控）。
// 时序敏感用例统一 200ms 裕量 + 10ms 轮询等待（waitUntil），防 flaky。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/HardwareMonitor.h"
#include "modules/08_devicemgmt/IScannerCamera.h"
#include "modules/08_devicemgmt/SelfCheckCollector.h"
#include "modules/08_devicemgmt/serial/McuFrame.h"
#include "modules/06_fileio/DeviceStateCache.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

using namespace Scanner::device;
using Scanner::data::DeviceStateCache;
using Scanner::DeviceState;

namespace {

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// 轮询等待：200ms 裕量内每 10ms 探一次（超时后再探一次兜底）
bool waitUntil(const std::function<bool()>& pred, int timeoutMs = 200) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        sleepMs(10);
    }
    return pred();
}

// —— 最小假相机：全接口空壳 + isOpen/温度可控（自建，不复用 test_device_manager）——
struct LocalFakeCam : Scanner::hal::IScannerCamera {
    bool openState = true;
    double tempC = 33.5;

    std::string getDeviceName() const override { return "FakeCam"; }
    std::string getSerialNumber() const override { return "FAKE-CAM"; }
    Scanner::Result open() override { openState = true; return Scanner::Result::ok(); }
    Scanner::Result close() override { openState = false; return Scanner::Result::ok(); }
    bool isOpen() const override { return openState; }
    Scanner::Result setExposure(double) override { return Scanner::Result::ok(); }
    Scanner::Result setGain(double) override { return Scanner::Result::ok(); }
    Scanner::Result setResolution(int, int) override { return Scanner::Result::ok(); }
    Scanner::Result setCalibration(const Scanner::hal::CameraIntrinsics&,
                                   const Scanner::hal::CameraIntrinsics&,
                                   const Scanner::hal::StereoExtrinsics&) override { return Scanner::Result::ok(); }
    Scanner::hal::CameraIntrinsics getLeftIntrinsics() const override { return {}; }
    Scanner::hal::CameraIntrinsics getRightIntrinsics() const override { return {}; }
    Scanner::hal::StereoExtrinsics getStereoExtrinsics() const override { return {}; }
    Scanner::Result startCapture() override { return Scanner::Result::ok(); }
    Scanner::Result stopCapture() override { return Scanner::Result::ok(); }
    bool isCapturing() const override { return false; }
    Scanner::Result grabFrame(Scanner::hal::StereoFrame&, int) override { return Scanner::Result::fail("未实现"); }
    Scanner::Result startAsyncCapture(Scanner::hal::FrameCallback) override { return Scanner::Result::ok(); }
    Scanner::Result stopAsyncCapture() override { return Scanner::Result::ok(); }
    double getTemperature() const override { return tempC; }
    std::string getPlatform() const override { return "Windows"; }
};

serial::TempFrame makeTemps(double c0, double c1, double c2, double c3, uint8_t ch) {
    serial::TempFrame tf{};
    tf.celsius[0] = c0; tf.celsius[1] = c1; tf.celsius[2] = c2; tf.celsius[3] = c3;
    tf.channels = ch;
    return tf;
}

} // namespace

// 1. start/stop 干净起落（线程起落；重复 stop 幂等）
TEST(HardwareMonitorTest, StartStop) {
    HardwareMonitor mon;
    EXPECT_FALSE(mon.isRunning());
    mon.start(50);
    EXPECT_TRUE(mon.isRunning());
    mon.stop();
    EXPECT_FALSE(mon.isRunning());
    mon.stop();  // 幂等
    EXPECT_FALSE(mon.isRunning());
}

// 2. 相机行照旧：isOpen/state/温度/fps + droppedFrames 累计口径
TEST(HardwareMonitorTest, CameraRowWritten) {
    DeviceStateCache dsc;
    LocalFakeCam cam;
    HardwareMonitor mon;
    mon.setDeviceStateCache(&dsc);
    mon.setCamera(&cam);
    mon.setFrameCounter([] { return 30; });
    mon.setDropCounter([] { return 7.0; });
    mon.start(50);
    ASSERT_TRUE(waitUntil([&] { return dsc.getState("Camera").state == DeviceState::Connected; }))
        << "巡检周期内未写入 Camera 行";
    mon.stop();

    auto st = dsc.getState("Camera");
    EXPECT_EQ(st.deviceType, "ScannerCamera");
    EXPECT_EQ(st.state, DeviceState::Connected);
    EXPECT_DOUBLE_EQ(st.temperature, 33.5);
    EXPECT_DOUBLE_EQ(st.fps, 30.0);
    EXPECT_EQ(st.droppedFrames, 7);
    EXPECT_GT(st.timestamp, 0u);
}

// 3. 温度改注入：4 路 TempFrame → MCU_T0..T3 行 + 首路同步写 MCU 行（MainWindow 兼容）
TEST(HardwareMonitorTest, McuTempRowsWritten) {
    DeviceStateCache dsc;
    HardwareMonitor mon;
    mon.setDeviceStateCache(&dsc);
    const auto tf = makeTemps(25.1, 26.2, 27.3, 28.4, 4);
    mon.setLastTemps([tf] { return tf; });
    mon.start(50);
    ASSERT_TRUE(waitUntil([&] { return dsc.getState("MCU").state == DeviceState::Connected; }))
        << "巡检周期内未写入 MCU 行";
    mon.stop();

    for (int i = 0; i < 4; ++i) {
        auto st = dsc.getState("MCU_T" + std::to_string(i));
        EXPECT_EQ(st.state, DeviceState::Connected) << "MCU_T" << i;
        EXPECT_DOUBLE_EQ(st.temperature, tf.celsius[i]) << "MCU_T" << i;
    }
    auto mcu = dsc.getState("MCU");
    EXPECT_EQ(mcu.deviceType, "MCU");
    EXPECT_DOUBLE_EQ(mcu.temperature, tf.celsius[0]);
}

// 4. 心跳经注入回调上报（巡检只调不发命令）：50ms 周期 200ms 内 ≥2 次
TEST(HardwareMonitorTest, HeartbeatHookCalled) {
    std::atomic<int> calls{0};
    HardwareMonitor mon;
    mon.setHeartbeatCheck([&] { calls.fetch_add(1); });
    mon.start(50);
    sleepMs(200);
    mon.stop();
    EXPECT_GE(calls.load(), 2);
}

// 5. 快照三件套回填 + sc=null 时 cpu 侧字段缺省 -1
TEST(HardwareMonitorTest, SnapshotFields) {
    HardwareMonitor mon;
    mon.setSelfCheck(nullptr);
    mon.setFrameCounter([] { return 25; });
    mon.setProcessCounter([] { return 20.5; });
    mon.setDropCounter([] { return 3.0; });
    mon.start(50);
    ASSERT_TRUE(waitUntil([&] { return mon.snapshot().captureFps > 0.0; }))
        << "巡检周期内快照未回填";
    mon.stop();

    auto m = mon.snapshot();
    EXPECT_DOUBLE_EQ(m.captureFps, 25.0);
    EXPECT_DOUBLE_EQ(m.processFps, 20.5);
    EXPECT_DOUBLE_EQ(m.dropRate, 3.0);
    EXPECT_DOUBLE_EQ(m.cpuTempC, -1.0);
    EXPECT_DOUBLE_EQ(m.cpuPercent, -1.0);
    EXPECT_DOUBLE_EQ(m.memPercent, -1.0);
    EXPECT_DOUBLE_EQ(m.diskFreeGB, -1.0);
    EXPECT_GT(m.timestampMs, 0);
}

// 6. 真 SelfCheckCollector（探针注假值）→ 快照 memPercent/diskFreeGB 正确 + 三件套并存
TEST(HardwareMonitorTest, SnapshotWithSelfCheck) {
    SelfCheckCollector sc;
    sc.setMemProbe([] { return 42.5; });
    sc.setDiskProbe([](const std::string&) { return 100.25; });
    // 预热：首次 collect 需付一次性 PDH 基线 + NVML 动态加载（本机实测数百 ms），
    // 先手工采一遍再起巡检——后续 collect 毫秒级，200ms 裕量才能确定性成立
    (void)sc.collect(0);
    HardwareMonitor mon;
    mon.setSelfCheck(&sc);
    mon.setFrameCounter([] { return 10; });
    mon.start(50);
    ASSERT_TRUE(waitUntil([&] { return mon.snapshot().memPercent > 0.0; }))
        << "巡检周期内自检字段未入快照";
    mon.stop();

    auto m = mon.snapshot();
    EXPECT_DOUBLE_EQ(m.memPercent, 42.5);
    EXPECT_DOUBLE_EQ(m.diskFreeGB, 100.25);
    EXPECT_DOUBLE_EQ(m.captureFps, 10.0);
}

// 7. 垫片行为延续：注入源空不写 MCU 行；channels 按位裁剪（2 路只写 T0/T1）
TEST(HardwareMonitorTest, NoSourceNoMcuRowAndChannelClip) {
    DeviceStateCache dsc;
    HardwareMonitor mon;
    mon.setDeviceStateCache(&dsc);
    mon.start(50);
    sleepMs(200);
    mon.stop();
    EXPECT_EQ(dsc.getState("MCU").state, DeviceState::Offline);
    EXPECT_EQ(dsc.getState("MCU_T0").state, DeviceState::Offline);

    const auto tf2 = makeTemps(11.0, 12.0, 0.0, 0.0, 2);
    mon.setLastTemps([tf2] { return tf2; });
    mon.start(50);
    ASSERT_TRUE(waitUntil([&] { return dsc.getState("MCU_T1").state == DeviceState::Connected; }));
    mon.stop();
    EXPECT_DOUBLE_EQ(dsc.getState("MCU_T0").temperature, 11.0);
    EXPECT_DOUBLE_EQ(dsc.getState("MCU_T1").temperature, 12.0);
    EXPECT_EQ(dsc.getState("MCU_T2").state, DeviceState::Offline);  // 超出 channels 不写
    EXPECT_EQ(dsc.getState("MCU_T3").state, DeviceState::Offline);
}
