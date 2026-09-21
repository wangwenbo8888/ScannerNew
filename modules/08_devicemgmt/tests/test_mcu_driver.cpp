// ============================================================================
// test_mcu_driver.cpp — MCUDriver 组合壳单测（S-T5）
//
// 假 IO 注入：writeOverride 捕获下行帧 + testInjectRaw 回灌上行字节——
// 测试模式（writeOverride 非空）不开真串口、不起 rx 线程，全程单线程确定论。
// 用例 = 实施计划 Task 5 Step 4 六条 + v2 温度单路兼容 + v2 旧 E 帧忽略。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/MCUDriver.h"
#include "modules/08_devicemgmt/serial/FrameCodec.h"

#include <string>
#include <vector>

using namespace Scanner::device;
using namespace Scanner::device::serial;
using Scanner::hal::CaptureParams;
using Scanner::hal::McuUplink;
using FCodec = Scanner::device::serial::FrameCodec;

namespace {

// 假 IO：记录全部下行帧
struct FrameLog {
    std::vector<std::string> frames;
    bool write(const std::string& f) { frames.push_back(f); return true; }
};

// 上行回调记账
struct UplinkLog {
    int temp = 0, gesture = 0, shot = 0;
    TempFrame lastT{};
    GestureEvent lastG{};
    ShotCountFrame lastSC{};

    McuUplink uplink() {
        McuUplink h;
        h.onTemp      = [this](const TempFrame& t)    { ++temp;    lastT  = t; };
        h.onGesture   = [this](const GestureEvent& g) { ++gesture; lastG  = g; };
        h.onShotCount = [this](const ShotCountFrame& s) { ++shot;  lastSC = s; };
        return h;
    }
};

} // namespace

// —— 1. TypedPayload：V3 下 typed 命令拼帧与 encode 一致（seq 依下发次序 0,1,2…）——
TEST(MCUDriver, TypedPayloadV3) {
    FrameLog io;
    MCUDriver d([&](const std::string& f) { return io.write(f); });
    d.setProtocolVersion(FCodec::Version::V3);
    FCodec enc(FCodec::Version::V3);

    d.setCaptureParams(CaptureParams{60, 80, 1, 1, 0, 0, 120}, {});
    d.startScan({});
    d.enterStandby({});
    d.setHeatTarget(60, {});
    d.queryTemperature(2);

    ASSERT_EQ(io.frames.size(), 5u);
    EXPECT_EQ(io.frames[0], enc.encode("N10 H60 B80 T1 V1 C0 D0 L120", 0));
    EXPECT_EQ(io.frames[1], enc.encode("N11 H1", 1));
    EXPECT_EQ(io.frames[2], enc.encode("N13 E1", 2));
    EXPECT_EQ(io.frames[3], enc.encode("N14 T60", 3));
    EXPECT_EQ(io.frames[4], enc.encode("N15 V2", 4));
}

// —— 2. PumpDispatch：T/G01/G03 上行经 pump 分流到 Uplink 回调（G01 手势/ G03 触发
//      计数文本行直收；K 原始链停用；S 为 v3 遗留环无回调——260831 协议）——
TEST(MCUDriver, PumpDispatchUplink) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V3);
    FCodec enc(FCodec::Version::V3);
    UplinkLog log;
    d.setUplink(log.uplink());

    d.testInjectRaw(enc.encode("T25.5", 1));
    d.testInjectTextLine("G01 M1");
    d.testInjectTextLine("G03 S500");
    d.pump();

    EXPECT_EQ(log.temp, 1);
    EXPECT_EQ(log.lastT.channels, 1);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[0], 25.5);
    EXPECT_EQ(log.gesture, 1);
    EXPECT_EQ(log.lastG.key, KeyId::Middle);
    EXPECT_EQ(log.lastG.gesture, GestureEvent::Gesture::Short);
    EXPECT_EQ(log.shot, 1);
    EXPECT_EQ(log.lastSC.count, 500u);
}

// —— 3. AckRouting：A 帧经 pump 回填 CommandChannel → 挂表命令 onDone(true) ——
TEST(MCUDriver, AckRouting) {
    FrameLog io;
    MCUDriver d([&](const std::string& f) { return io.write(f); });
    d.setProtocolVersion(FCodec::Version::V3);
    FCodec enc(FCodec::Version::V3);

    bool done = false, ok = false;
    d.startScan([&](bool o, const std::string&) { done = true; ok = o; });
    EXPECT_FALSE(done);                        // 非阻塞：send 后未 ACK 不回调
    d.testInjectRaw(enc.encode("A00", 9));     // ackedSeq=0x00 命中首条 seq
    EXPECT_FALSE(done);                        // pump 前不消化
    d.pump();
    EXPECT_TRUE(done);
    EXPECT_TRUE(ok);
}

// —— 4. V2AnonymousKeyDropped：v2 匿名 K1; 原始按键链停用 → 落 onParseFail，手势不触发 ——
TEST(MCUDriver, V2AnonymousKeyDropped) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V2);
    UplinkLog log;
    d.setUplink(log.uplink());
    d.testInjectRaw("K1;");
    d.pump();
    EXPECT_EQ(log.gesture, 0);
}

// —— 4b. MalformedG01Dropped：G01 格式不符（键位缺失/手势位非法）→ 不入环、不触发 ——
TEST(MCUDriver, MalformedG01Dropped) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V2);
    UplinkLog log;
    d.setUplink(log.uplink());
    d.testInjectTextLine("G01 U");
    d.testInjectTextLine("G01 U9");
    d.testInjectTextLine("G01 X1");
    d.pump();
    EXPECT_EQ(log.gesture, 0);
}

// —— 5. LastRxUpdated：任何有效帧刷新通讯心跳时间戳（设计方案 §4-4）——
TEST(MCUDriver, LastRxUpdated) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V3);
    FCodec enc(FCodec::Version::V3);
    EXPECT_EQ(d.lastRxTime(), 0u);
    d.testInjectRaw(enc.encode("T25.5", 1));
    EXPECT_GT(d.lastRxTime(), 0u);
}

// —— 6. ShotCountGapCounted：G03 帧计数跳变（丢帧）对账（§6.2-9/260831 改 G03 源；
//      首帧立基线、跳 1=一帧之不差、回绕重基线）——
TEST(MCUDriver, ShotCountGapCounted) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V2);
    d.testInjectTextLine("G03 S1");      // 首帧立基线
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 0u);
    d.testInjectTextLine("G03 S5");      // 1→5 跳变 = 丢 3 帧
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 3u);
    d.testInjectTextLine("G03 S6");      // 连续无新差
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 3u);
    d.testInjectTextLine("G03 S4");      // 回绕（MCU 复位）重基线
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 3u);
    d.testInjectTextLine("G03 S4");      // 回绕后连续
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 3u);
    d.testInjectTextLine("G03 S8");      // 4→8 = 丢 3
    d.pump();
    EXPECT_EQ(d.seqGapCount(), 6u);
}

// —— 7. V2TempSingleChannel：v2 旧 "T25.3;" 单路温度天然兼容（parse 1 通道）——
TEST(MCUDriver, V2TempSingleChannel) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V2);
    UplinkLog log;
    d.setUplink(log.uplink());
    d.testInjectRaw("T25.3;");
    d.pump();
    ASSERT_EQ(log.temp, 1);
    EXPECT_EQ(log.lastT.channels, 1);
    EXPECT_DOUBLE_EQ(log.lastT.celsius[0], 25.3);
}

// —— 8. V2LegacyEIgnored：v2 旧 "E1;" 急停牌已删——忽略，不触发任何回调 ——
TEST(MCUDriver, V2LegacyEIgnored) {
    MCUDriver d;
    d.setProtocolVersion(FCodec::Version::V2);
    UplinkLog log;
    d.setUplink(log.uplink());
    d.testInjectRaw("E1;");
    d.pump();
    EXPECT_EQ(log.temp + log.gesture + log.shot, 0);
}

// —— 9. ReopenDrainsRings：close 前未消费的环残留经 reopen 排空、对账基线/
//      心跳复位——上一会话数据不得串染下一会话 ——
TEST(MCUDriver, ReopenDrainsRings) {
    FrameLog io;
    MCUDriver d([&](const std::string& f) { return io.write(f); });
    d.setProtocolVersion(FCodec::Version::V3);
    UplinkLog log;
    d.setUplink(log.uplink());

    d.open("");
    d.testInjectTextLine("G01 U1");
    d.pump();
    EXPECT_EQ(log.gesture, 1);                          // 正常收到

    d.testInjectTextLine("G01 U2");                    // 残留（未 pump 即关）
    EXPECT_GT(d.lastRxTime(), 0u);
    d.close();
    d.open("");
    EXPECT_EQ(d.lastRxTime(), 0u);                      // 心跳复位
    d.pump();                                           // 残留已被 open 排空
    EXPECT_EQ(log.gesture, 1);                          // 无任何回调再触发
}
