// ============================================================================
// test_mcu_driver.cpp — MCUDriver 组合壳单测（协议 260831 批1-C 重写）
//
// 假 IO 注入：writeOverride 捕获下行帧 + testInjectRaw 回灌上行字节——
// 测试模式（writeOverride 非空）不开真串口、不起 rx 线程，全程单线程确定论。
// 用例 = typed 下行四条（含 N12/N13 域钳制）/ G01-G03 分流 / 旧协议帧拒收 /
// 心跳 / 手势环满丢新 / reopen 无残留。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/MCUDriver.h"

#include <string>
#include <vector>

using namespace Scanner::device;
using namespace Scanner::device::serial;
using Scanner::hal::CaptureParams;
using Scanner::hal::McuUplink;

namespace {

// 假 IO：记录全部下行帧（盲发——写即成功）
struct FrameLog {
    std::vector<std::string> frames;
    bool write(const std::string& f) { frames.push_back(f); return true; }
};

// 上行回调记账
struct UplinkLog {
    int temp = 0, gesture = 0, shot = 0;
    TempFrame lastT{};
    GestureEvent lastG{};
    ShotCountFrame lastS{};

    McuUplink uplink() {
        McuUplink h;
        h.onTemp     = [this](const TempFrame& t)     { ++temp;     lastT = t; };
        h.onGesture  = [this](const GestureEvent& g)  { ++gesture;  lastG = g; };
        h.onShotCount = [this](const ShotCountFrame& s) { ++shot;   lastS = s; };
        return h;
    }
};

} // namespace

// —— 1. TypedPayloads：typed 下行四条拼帧（N10 七参含空格；N12/N13 域钳制在
//      MCUDriver：T 5-1000 / S 0-80）——
TEST(MCUDriver, TypedPayloads) {
    FrameLog io;
    MCUDriver d([&](const std::string& f) { return io.write(f); });

    d.setCaptureParams(CaptureParams{60, 10, 1, 1, 0, 0, 40}, {});
    d.stopScan({});
    d.setTempReportPeriod(100, {});
    d.setTempReportPeriod(2, {});        // 下钳 5
    d.setTempReportPeriod(2000, {});     // 上钳 1000
    d.setHeatTarget(99, {});             // 上钳 80
    d.setHeatTarget(-5, {});             // 下钳 0

    ASSERT_EQ(io.frames.size(), 7u);
    EXPECT_EQ(io.frames[0], "N10 H60 B10 T1 V1 C0 D0 L40;");
    EXPECT_EQ(io.frames[1], "N11 H0;");
    EXPECT_EQ(io.frames[2], "N12 T100;");
    EXPECT_EQ(io.frames[3], "N12 T5;");
    EXPECT_EQ(io.frames[4], "N12 T1000;");
    EXPECT_EQ(io.frames[5], "N13 S80;");
    EXPECT_EQ(io.frames[6], "N13 S0;");
}

// —— 2. DispatchG：G01/G02/G03 上行帧经 pump 分流到 Uplink 三回调；
//      G03 计数同步入 lastShot_ 记账 ——
TEST(MCUDriver, DispatchG) {
    MCUDriver d;
    UplinkLog log;
    d.setUplink(log.uplink());

    d.testInjectRaw("G01 U1;");
    d.testInjectRaw("G02 A1 B2 C3 D4;");
    d.testInjectRaw("G03 S500;");
    d.pump();

    EXPECT_EQ(log.gesture, 1);
    EXPECT_EQ(log.lastG.key, KeyId::Up);
    EXPECT_EQ(log.lastG.gesture, Gesture::Short);
    EXPECT_EQ(log.temp, 1);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[0], 1.0);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[1], 2.0);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[2], 3.0);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[3], 4.0);
    EXPECT_EQ(log.shot, 1);
    EXPECT_EQ(log.lastS.count, 500u);
    EXPECT_EQ(d.shotCount(), 500u);      // 最近 G03 快照（记账口径 D6）
}

// —— 3. OldFrameRejected：旧协议 T/K/S 帧全部弃收（parseFail 计数增长、三回调零次）——
TEST(MCUDriver, OldFrameRejected) {
    MCUDriver d;
    UplinkLog log;
    d.setUplink(log.uplink());

    const uint64_t before = d.parseFailCount();
    d.testInjectRaw("T25.3,24.8;");      // 旧 v2 单路温度
    d.testInjectRaw("KM1,1234;");        // 旧 K 按键
    d.testInjectRaw("S0A;");             // 旧 S 状态
    d.pump();
    EXPECT_EQ(d.parseFailCount(), before + 3);
    EXPECT_EQ(log.temp + log.gesture + log.shot, 0);
}

// —— 4. Heartbeat：任何完整帧刷新通讯心跳时间戳（§4-4）——
TEST(MCUDriver, HeartbeatUpdated) {
    MCUDriver d;
    EXPECT_EQ(d.lastRxTime(), 0u);
    d.testInjectRaw("G03 S1;");
    EXPECT_GT(d.lastRxTime(), 0u);
}

// —— 5. GestureRingDrop：手势环容 64——不 pump 连灌 70 帧 → 满丢新计数 >0 ——
TEST(MCUDriver, GestureRingDrop) {
    MCUDriver d;
    for (int i = 0; i < 70; ++i) d.testInjectRaw("G01 U1;");
    EXPECT_GT(d.gestureRingDropped(), 0u);
}

// —— 6. Reopen：open/close×2 无残留——心跳复位、环排空，上一会话数据不串染 ——
TEST(MCUDriver, ReopenNoResidue) {
    FrameLog io;
    MCUDriver d([&](const std::string& f) { return io.write(f); });
    UplinkLog log;
    d.setUplink(log.uplink());

    d.open("");
    d.testInjectRaw("G01 M1;");
    d.pump();
    EXPECT_EQ(log.gesture, 1);                     // 正常收到

    d.testInjectRaw("G01 U1;");                    // 残留（未 pump 即关）
    EXPECT_GT(d.lastRxTime(), 0u);
    d.close();
    d.open("");
    EXPECT_EQ(d.lastRxTime(), 0u);                 // 心跳复位
    d.pump();                                      // 残留已被 open 排空
    EXPECT_EQ(log.gesture, 1);                     // 无任何回调再触发
}

// —— 7. OpenFailCleanup（真串口路径·无硬件确定论）：开口必败的口名 → open 返回
//      fail 且完全收口（写线程起→停、串口未开、close 幂等）——open 失败分支与
//      自动搜口全败路径共用 stopWriteThread 收口（P1 修复路径同源）。
//      注：探测「首口未命中→次口命中」需真实 COM 口枚举——WriteOverride 测试缝
//      在写线程之前短路（无队列无串口），SerialPort 无口枚举注入缝，不可无硬件
//      复现；该路径以 closePortOnly 拆分＋三条路径推演覆盖（见审查修复报告）。
TEST(MCUDriver, OpenFailCleanupOnBadPort) {
    MCUDriver d;                                   // 无 override：真串口路径（写线程起→收口）
    const auto r = d.open("NOSUCHPORT");
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(d.isOpen());
    EXPECT_TRUE(d.close().success);                // 失败后收口幂等
    EXPECT_FALSE(d.isOpen());
}
