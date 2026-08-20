// ============================================================================
// test_mcu_frame.cpp — 上行载荷解析 + SpscRing 丢最旧/空读 单测（S-T1）
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

// —— parseKeyPayload：K+键号(U/L/M/R)+按下(0/1)+','+毫秒 ——
TEST(McuFrameParse, KeyLegal) {
    RawKeyEvent e;
    ASSERT_TRUE(parseKeyPayload("KM1,1234", e));
    EXPECT_EQ(e.key, KeyId::Middle);
    EXPECT_TRUE(e.pressed);
    EXPECT_EQ(e.mcuMs, 1234u);
}

TEST(McuFrameParse, KeyIllegal) {
    RawKeyEvent e;
    EXPECT_FALSE(parseKeyPayload("KX1,1234", e));   // 键号非法
    EXPECT_FALSE(parseKeyPayload("KM2,1234", e));   // 按下位非 0/1
    EXPECT_FALSE(parseKeyPayload("KM1,12a4", e));   // 毫秒非纯数字
    EXPECT_FALSE(parseKeyPayload("KM1", e));        // 缺逗号段
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

// —— SpscRing：满丢最旧 + 空读 false（浪费一格：N=4 稳态存 3）——
TEST(SpscRing, FullDropsOldest) {
    SpscRing<int, 4> ring;
    for (int v = 1; v <= 5; ++v) ring.push(v);
    EXPECT_EQ(ring.dropped(), 2u);            // 第 4/5 次入环各丢一个最旧
    int out = 0;
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 3);
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 4);
    EXPECT_TRUE(ring.pop(out)); EXPECT_EQ(out, 5);
    EXPECT_FALSE(ring.pop(out));              // 取尽
}

TEST(SpscRing, PopEmptyReturnsFalse) {
    SpscRing<int, 4> ring;
    int out = -1;
    EXPECT_TRUE(ring.empty());
    EXPECT_FALSE(ring.pop(out));
}
