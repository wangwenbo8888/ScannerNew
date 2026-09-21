// ============================================================================
// test_mcu_frame.cpp — 上行载荷解析 + SpscRing 满丢新/空读 单测（S-T1；D-T12a 口径）
//
// 解析用例基于 v3 默认口径（载荷已剥去 '$'/seq/crc/';'）；
// 环的并发正确性由 T12 集成测覆盖，此处单线程验证语义。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/serial/McuFrame.h"

using namespace Scanner::device::serial;

// —— parseTempPayload：T 后 1~4 个逗号分隔浮点 ——
TEST(McuFrameParse, TempLegal) {
    TempFrame f;
    ASSERT_TRUE(parseTempPayload("T25.3,24.8", f));
    EXPECT_EQ(f.channels, 2);
    EXPECT_DOUBLE_EQ(f.celsius[0], 25.3);
    EXPECT_DOUBLE_EQ(f.celsius[1], 24.8);
}

TEST(McuFrameParse, TempIllegal) {
    TempFrame f;
    EXPECT_FALSE(parseTempPayload("T25.x", f));       // 非数字
    EXPECT_FALSE(parseTempPayload("T1,2,3,4,5", f));  // 超 4 通道
    EXPECT_FALSE(parseTempPayload("T25.3,", f));      // 尾部空字段
    EXPECT_FALSE(parseTempPayload("X25.3", f));       // 前缀非 T
}

// —— parseTempPayload 严格性（S-T5 前置清理：from_chars 口径——locale 无关，
//    拒 '+'/前导空白/inf/nan/内嵌 NUL 残留；strtod 曾全部放过）——
TEST(McuFrameParse, TempStrictRejects) {
    TempFrame f;
    EXPECT_FALSE(parseTempPayload("T+25.3", f));                     // '+' 前缀
    EXPECT_FALSE(parseTempPayload("T 25.3", f));                     // 前导空白
    EXPECT_FALSE(parseTempPayload("Tinf", f));                       // 非有限
    EXPECT_FALSE(parseTempPayload("Tnan", f));                       // 非有限
    EXPECT_FALSE(parseTempPayload("T1e999", f));                     // 溢出
    EXPECT_FALSE(parseTempPayload(std::string("T25.3\0x", 7), f));   // 内嵌 NUL 残留
}

// —— parseG01Payload：G01 <键 U/L/M/R><手势位 1短/2双/3长>（键位前容忍空白）——
TEST(McuFrameParse, G01Legal) {
    GestureEvent e;
    ASSERT_TRUE(parseG01Payload("G01 U1", e));
    EXPECT_EQ(e.key, KeyId::Up);
    EXPECT_EQ(e.gesture, GestureEvent::Gesture::Short);
    GestureEvent g;
    ASSERT_TRUE(parseG01Payload("G01 M2", g));
    EXPECT_EQ(g.key, KeyId::Middle);
    EXPECT_EQ(g.gesture, GestureEvent::Gesture::Double);
    GestureEvent h;
    ASSERT_TRUE(parseG01Payload("G01R3", h));     // 无空白变体
    EXPECT_EQ(h.key, KeyId::Right);
    EXPECT_EQ(h.gesture, GestureEvent::Gesture::Hold);
}

TEST(McuFrameParse, G01Illegal) {
    GestureEvent e;
    EXPECT_FALSE(parseG01Payload("G01 X1", e));   // 键号非法
    EXPECT_FALSE(parseG01Payload("G01 U9", e));   // 手势位非 1/2/3
    EXPECT_FALSE(parseG01Payload("G01 U12", e));  // 多余字符
    EXPECT_FALSE(parseG01Payload("G01 U", e));    // 缺手势位
    EXPECT_FALSE(parseG01Payload("G02 U1", e));   // 前缀不符
}

// —— parseG03Payload：G03 S<十进制累计触发/快门计数>（帧计数对账源）——
TEST(McuFrameParse, G03Legal) {
    ShotCountFrame e;
    ASSERT_TRUE(parseG03Payload("G03 S500", e));
    EXPECT_EQ(e.count, 500u);
    ShotCountFrame g;
    ASSERT_TRUE(parseG03Payload("G03S1", g));              // 无空白变体
    EXPECT_EQ(g.count, 1u);
    ShotCountFrame h;
    ASSERT_TRUE(parseG03Payload("G03 S123456789012345678", h));   // 大数不溢出
    EXPECT_EQ(h.count, 123456789012345678ull);
}

TEST(McuFrameParse, G03Illegal) {
    ShotCountFrame e;
    EXPECT_FALSE(parseG03Payload("G03 X1", e));   // 缺 S 前缀
    EXPECT_FALSE(parseG03Payload("G03 S", e));    // 计数空
    EXPECT_FALSE(parseG03Payload("G03 S1a", e));  // 非纯数字
    EXPECT_FALSE(parseG03Payload("G01 S1", e));   // 前缀不符
}

// —— parseStatusPayload：S 后 1~2 个 hex 字符 ——
TEST(McuFrameParse, StatusLegal) {
    StatusFrame f;
    ASSERT_TRUE(parseStatusPayload("S0A", f));
    EXPECT_EQ(f.code, 0x0A);
    StatusFrame g;
    ASSERT_TRUE(parseStatusPayload("S3", g));
    EXPECT_EQ(g.code, 0x3);
}

TEST(McuFrameParse, StatusIllegal) {
    StatusFrame f;
    EXPECT_FALSE(parseStatusPayload("S0A2", f));  // 超 2 个 hex 字符
    EXPECT_FALSE(parseStatusPayload("SG", f));    // 非 hex 字符
    EXPECT_FALSE(parseStatusPayload("S", f));     // 无载荷
}

// —— parseAckPayload：A 后 1~2 个 hex 字符 = 被确认命令 seq ——
TEST(McuFrameParse, AckLegal) {
    AckFrame f;
    ASSERT_TRUE(parseAckPayload("A0B", f));
    EXPECT_EQ(f.ackedSeq, 0x0B);
    AckFrame g;
    ASSERT_TRUE(parseAckPayload("AF", g));
    EXPECT_EQ(g.ackedSeq, 0xF);
}

TEST(McuFrameParse, AckIllegal) {
    AckFrame f;
    EXPECT_FALSE(parseAckPayload("A0BC", f));  // 超 2 个 hex 字符
    EXPECT_FALSE(parseAckPayload("AZ", f));    // 非 hex 字符
    EXPECT_FALSE(parseAckPayload("B0B", f));   // 前缀非 A
}

// —— SpscRing：满丢新 + 空读 false（浪费一格：N=4 稳态存 3）——
TEST(SpscRing, FullDropsNewest) {
    SpscRing<int, 4> ring;
    for (int v = 1; v <= 3; ++v) EXPECT_TRUE(ring.push(v));
    EXPECT_FALSE(ring.push(4));                // 满环丢新：不入队
    EXPECT_FALSE(ring.push(5));
    EXPECT_EQ(ring.dropped(), 2u);             // 第 4/5 次各丢一个新帧
    int out = 0;
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 1);   // 旧序 N-1 条
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 2);
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 3);
    EXPECT_FALSE(ring.pop(out));               // 取尽
}

TEST(SpscRing, PopEmptyReturnsFalse) {
    SpscRing<int, 4> ring;
    int out = -1;
    EXPECT_TRUE(ring.empty());
    EXPECT_FALSE(ring.pop(out));
}
