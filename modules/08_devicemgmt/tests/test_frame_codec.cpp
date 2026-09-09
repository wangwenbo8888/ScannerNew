// test_frame_codec.cpp — 裸 ';' 分帧：分帧/粘帧/分段/空帧/空格载荷/半帧超时/encode
#include "serial/FrameCodec.h"
#include <gtest/gtest.h>

namespace sd = Scanner::device::serial;

TEST(FrameCodec, SplitsOnSemicolon) {
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed("G01 U1;G03 S500;", out);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].payload, "G01 U1");
    EXPECT_EQ(out[1].payload, "G03 S500");
}

TEST(FrameCodec, SpacePayloadKept) {                     // 新协议载荷含空格
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed("G02 A25.3 B25.4 C26.0 D24.5;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "G02 A25.3 B25.4 C26.0 D24.5");
}

TEST(FrameCodec, SegmentedFeedAccumulates) {             // 分段到达
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed("N10 ", out);
    c.feed("H30", out);
    ASSERT_TRUE(out.empty());
    c.feed(" B60;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "N10 H30 B60");
}

TEST(FrameCodec, LeadingGarbageKeptInFrame) {            // 裸流无同步头：';' 前噪声并入帧（内容判别归解析层）
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed("xxG01 U1;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "xxG01 U1");
}

TEST(FrameCodec, EmptyFrameNotEmitted) {                 // ";;" 空帧不出
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed(";;G01 U1;;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "G01 U1");
}

TEST(FrameCodec, HalfFrameTimeoutDrops) {
    sd::FrameCodec c;
    std::vector<sd::FrameCodec::Frame> out;
    c.feed("G01 U1", out);
    EXPECT_FALSE(c.advanceTimeout(49));
    ASSERT_TRUE(c.advanceTimeout(1));                    // 50ms 丢挂起
    c.feed(";G03 S1;", out);                             // 迟到 ';' 只出一个 G03
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "G03 S1");
}

TEST(FrameCodec, EncodeAppendsSemicolon) {
    sd::FrameCodec c;
    EXPECT_EQ(c.encode("N10 H30 B60 T1 V1 C1 D1 L50"), "N10 H30 B60 T1 V1 C1 D1 L50;");
}
