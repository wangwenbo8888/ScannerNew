// ============================================================================
// test_frame_codec.cpp — v2/v3 成拆帧单测（S-T2）
//
// 用例 = 设计 §2.1 全约束 + §7 单元行：CRC 已知向量 / happy / 粘连 / 分段 /
// 垃圾丢弃 / 半帧超时 / CRC 错 / 尾错位 / 32B 上限 / v2 直通 / encode 禁字符。
// 期望帧 "$T25.3,24.80A34E9;" 的 CRC 值由独立参考实现预先算得（非本库自证）。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/serial/FrameCodec.h"

using namespace Scanner::device::serial;

// —— 1. CRC16-CCITT-FALSE 已知向量（poly 0x1021 / init 0xFFFF）——
TEST(FrameCodecCrc, KnownVector) {
    EXPECT_EQ(FrameCodec::crc16ccitt("123456789", 0xFFFF), 0x29B1);
}

// —— 2. v3 happy path：组帧定长格式 + 整帧拆出 ——
TEST(FrameCodecV3, EncodeFormat) {
    FrameCodec c(FrameCodec::Version::V3);
    EXPECT_EQ(c.version(), FrameCodec::Version::V3);
    EXPECT_EQ(c.encode("T25.3,24.8", 0x0A), "$T25.3,24.80A34E9;");
}

TEST(FrameCodecV3, FeedWholeFrame) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("$T25.3,24.80A34E9;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "T25.3,24.8");
    EXPECT_EQ(out[0].seq, 0x0A);
}

// —— 3. 粘连：两帧一次喂入 → 吐 2 帧 ——
TEST(FrameCodecV3, StickyFrames) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed(c.encode("T25.3,24.8", 0x0A) + c.encode("KM1,1234", 0x0B), out);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].payload, "T25.3,24.8");
    EXPECT_EQ(out[0].seq, 0x0A);
    EXPECT_EQ(out[1].payload, "KM1,1234");
    EXPECT_EQ(out[1].seq, 0x0B);
}

// —— 4. 分段：先半帧无输出 → 补尾吐 1 帧 ——
TEST(FrameCodecV3, SegmentedFeed) {
    FrameCodec c(FrameCodec::Version::V3);
    std::string frame = c.encode("T25.3,24.8", 0x0A);
    std::vector<FrameCodec::Frame> out;
    c.feed(frame.substr(0, 4), out);  // "$T25"
    EXPECT_TRUE(out.empty());
    c.feed(frame.substr(4), out);     // ".3,24.80A34E9;"
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "T25.3,24.8");
    EXPECT_EQ(out[0].seq, 0x0A);
}

// —— 5. 垃圾字节：非 '$' 前导全丢（含 ';'）；之后合法帧仍可恢复 ——
TEST(FrameCodecV3, GarbageWithoutDollarDropped) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("AB\x02" "CD;", out);
    EXPECT_TRUE(out.empty());
    c.feed(c.encode("T25.3,24.8", 0x01), out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "T25.3,24.8");
}

TEST(FrameCodecV3, DollarRestartsPending) {  // 半帧中出现新 '$' → 重新同步
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("$XYZ$", out);
    EXPECT_TRUE(out.empty());
    c.feed(c.encode("T25.3,24.8", 0x02), out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].seq, 0x02);
}

// —— 6. 半帧超时：50ms 推进丢弃挂起缓冲，尾巴不复活 ——
TEST(FrameCodecV3, HalfFrameTimeout) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("$T25.3", out);
    EXPECT_TRUE(out.empty());
    EXPECT_FALSE(c.advanceTimeout(30));  // 累计 30ms 未到 50
    EXPECT_TRUE(c.advanceTimeout(30));   // 累计 60ms → 丢弃
    c.feed(",24.80A34E9;", out);         // 缓冲已弃，尾巴不再成帧
    EXPECT_TRUE(out.empty());
    EXPECT_FALSE(c.advanceTimeout(60));  // 无挂起 → false
}

// —— 7. CRC 错：篡改一位 → 丢弃 ——
TEST(FrameCodecV3, CrcMismatchDropped) {
    FrameCodec c(FrameCodec::Version::V3);
    std::string frame = c.encode("T25.3,24.8", 0x0A);
    char& firstCrc = frame[frame.size() - 5];
    firstCrc = (firstCrc == '0') ? '1' : '0';
    std::vector<FrameCodec::Frame> out;
    c.feed(frame, out);
    EXPECT_TRUE(out.empty());
    c.feed("$T25.3,24.80A34E8;", out);  // 已知帧 CRC 差 1
    EXPECT_TRUE(out.empty());
}

// —— 8. 尾错位：载荷尾不足 2+4 hex / 非 hex → 丢弃 ——
TEST(FrameCodecV3, MalformedTailDropped) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("$A01;", out);       // 尾仅 3 位
    c.feed("$;", out);          // 空体
    c.feed("$T2ZZABCD;", out);  // seq 位非 hex
    EXPECT_TRUE(out.empty());
}

// —— 9. 帧长上限：encode 不拦、拆帧丢；总长 32 边界仍合法 ——
TEST(FrameCodecV3, OversizeFrameDropped) {
    FrameCodec c(FrameCodec::Version::V3);
    std::string payload25 = "TABCDEFGHIJKLMNOPQRSTUVWX";  // 25B → 总长 33
    std::string big = c.encode(payload25, 0x01);
    EXPECT_NE(big, "");  // encode 侧不拦
    EXPECT_EQ(big.size(), 33u);
    std::vector<FrameCodec::Frame> out;
    c.feed(big, out);
    EXPECT_TRUE(out.empty());  // 拆帧侧丢
    std::string payload24 = payload25.substr(0, 24);  // 总长 32 边界
    c.feed(c.encode(payload24, 0x02), out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, payload24);
}

// —— 10. v2 直通：';' 分帧、无校验、seq 恒 0 ——
TEST(FrameCodecV2, Passthrough) {
    FrameCodec c(FrameCodec::Version::V2);
    EXPECT_EQ(c.version(), FrameCodec::Version::V2);
    std::vector<FrameCodec::Frame> out;
    c.feed("T25.3;K1;", out);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].payload, "T25.3");
    EXPECT_EQ(out[0].seq, 0);
    EXPECT_EQ(out[1].payload, "K1");
    EXPECT_EQ(c.encode("T25.3", 7), "T25.3;");  // 忽略 seq、不检查载荷
}

// —— 11. encode 禁字符：V3 拒 '$'/';' → 空串 ——
TEST(FrameCodecEncode, ForbiddenPayloadChars) {
    FrameCodec v3(FrameCodec::Version::V3);
    EXPECT_EQ(v3.encode("T2$5", 1), "");
    EXPECT_EQ(v3.encode("T2;5", 1), "");
}

// —— 12. v2 空段不吐帧（裁定钉死）——
TEST(FrameCodecV2, EmptySegmentNoFrame) {
    FrameCodec c(FrameCodec::Version::V2);
    std::vector<FrameCodec::Frame> out;
    c.feed(";;T25.3;", out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, "T25.3");
}

// —— 13. v2 挂起同样受 50ms 半帧超时（裁定钉死）——
TEST(FrameCodecV2, PendingTimeoutApplies) {
    FrameCodec c(FrameCodec::Version::V2);
    std::vector<FrameCodec::Frame> out;
    c.feed("T25.3", out);                  // 无 ';' 挂起
    EXPECT_TRUE(out.empty());
    EXPECT_TRUE(c.advanceTimeout(60));     // 60ms ≥ 50 → 丢弃挂起
    c.feed(";", out);                      // 尾巴不再成帧
    EXPECT_TRUE(out.empty());
}

// —— 14. v2 无 32B 长度检查：超长段照常吐帧（与 v3 超长丢弃对照）——
TEST(FrameCodecV2, NoLengthLimit) {
    FrameCodec c(FrameCodec::Version::V2);
    std::string seg = "T" + std::string(40, 'x') + ";";  // 42B 段
    std::vector<FrameCodec::Frame> out;
    c.feed(seg, out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].payload, seg.substr(0, seg.size() - 1));
}

// —— 15. 无尾超长流：挂起缓冲早弃封顶（内存有界），后续正常帧仍可解
//      （S-T5 前置清理回归钉——防御加固，可观测行为不变）——
TEST(FrameCodecV3, RunawayPendingCappedThenRecovers) {
    FrameCodec c(FrameCodec::Version::V3);
    std::vector<FrameCodec::Frame> out;
    c.feed("$" + std::string(64, 'A'), out);   // 64B 无 ';' 垃圾流
    EXPECT_TRUE(out.empty());
    c.feed("BBBB;", out);                       // 尾巴（无 '$' 前导）不复活
    EXPECT_TRUE(out.empty());
    c.feed(c.encode("T25.3,24.8", 0x03), out);  // 正常帧照解
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].seq, 0x03);
}
