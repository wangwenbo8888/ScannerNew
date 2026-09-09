// test_mcu_frame.cpp — G01/G02/G03 解析器合法/非法 ＋ SpscRing 满丢新
#include "serial/McuFrame.h"
#include <gtest/gtest.h>

namespace sd = Scanner::device::serial;

TEST(G01Parse, AllTwelveGestures) {
    const char* keys[] = {"U", "L", "M", "R"};
    for (int k = 0; k < 4; ++k)
        for (int g = 1; g <= 3; ++g) {
            sd::GestureEvent ev;
            ASSERT_TRUE(sd::parseGesturePayload("G01 " + std::string(keys[k]) + std::to_string(g), ev));
            EXPECT_EQ(static_cast<int>(ev.key), k);
            EXPECT_EQ(static_cast<int>(ev.gesture), g);
        }
}

TEST(G01Parse, RejectsMalformed) {
    sd::GestureEvent ev;
    EXPECT_FALSE(sd::parseGesturePayload("G01 U0", ev));    // 手势位 0 非法
    EXPECT_FALSE(sd::parseGesturePayload("G01 U4", ev));    // 手势位 4 非法
    EXPECT_FALSE(sd::parseGesturePayload("G01 X1", ev));    // 未知键
    EXPECT_FALSE(sd::parseGesturePayload("G01U1", ev));     // 缺空格
    EXPECT_FALSE(sd::parseGesturePayload("K M1", ev));      // 旧 K 帧
    EXPECT_FALSE(sd::parseGesturePayload("", ev));
}

TEST(G02Parse, FourChannels) {
    sd::TempFrame t;
    ASSERT_TRUE(sd::parseTempPayload("G02 A25.3 B25.4 C26.0 D24.5", t));
    EXPECT_DOUBLE_EQ(t.celsius[0], 25.3);
    EXPECT_DOUBLE_EQ(t.celsius[1], 25.4);
    EXPECT_DOUBLE_EQ(t.celsius[2], 26.0);
    EXPECT_DOUBLE_EQ(t.celsius[3], 24.5);
}

TEST(G02Parse, RejectsMalformed) {
    sd::TempFrame t;
    EXPECT_FALSE(sd::parseTempPayload("G02 A25.3 B25.4 C26.0", t));      // 缺 D 路
    EXPECT_FALSE(sd::parseTempPayload("G02 A25.3 B25.4 C26.0 D", t));    // D 空值
    EXPECT_FALSE(sd::parseTempPayload("T25.3,24.8", t));                 // 旧 T 帧
    EXPECT_FALSE(sd::parseTempPayload("G02 A25.3 B25.4 C26.0 E24.5", t));// 键错
    EXPECT_FALSE(sd::parseTempPayload("G02 A B C D", t));                // 非数值
}

TEST(G03Parse, CountAndMalformed) {
    sd::ShotCountFrame f;
    ASSERT_TRUE(sd::parseShotCountPayload("G03 S500", f));
    EXPECT_EQ(f.count, 500u);
    EXPECT_TRUE(sd::parseShotCountPayload("G03 S0", f));                  // 0 合法
    EXPECT_EQ(f.count, 0u);
    EXPECT_FALSE(sd::parseShotCountPayload("G03 S", f));
    EXPECT_FALSE(sd::parseShotCountPayload("G03 S-1", f));
    EXPECT_FALSE(sd::parseShotCountPayload("S0A", f));                    // 旧 S 帧
}

TEST(SpscRing, FullDropsNew) {
    sd::SpscRing<int, 3> r;
    EXPECT_TRUE(r.push(1)); EXPECT_TRUE(r.push(2));
    EXPECT_FALSE(r.push(3));                       // 容量 3 有效 2，满丢新
    EXPECT_EQ(r.dropped(), 1u);
    int v = 0;
    EXPECT_TRUE(r.pop(v)); EXPECT_EQ(v, 1);
    EXPECT_TRUE(r.push(4));
    EXPECT_FALSE(r.empty());
}
