// ============================================================================
// test_command_channel.cpp — 下行盲发通道单测（协议 260831 批2：无 ACK——
// CommandChannel 物理降级为「组帧＋写＋同步回告」）
//
// 用例：send 写成败同步回告 / 载荷回传原值（写败路径） / fire-and-forget /
// 组链 happy+中段写败短路+空组 / codec 空裸载荷直发 / write 空恒失败 /
// 回调内再 send（同步重入冒烟）。
// 假件：假写（计数+帧序录制+可置失败/按帧内容置失败）。
// 批2 删尽（机制壳随 reliable 物理删除）：挂表/ACK 销项/超时重传/3 败 Fault/
// 表容量逐出/seq——相关旧用例整组退役。
// ============================================================================

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "modules/08_devicemgmt/serial/CommandChannel.h"

using namespace Scanner::device::serial;

namespace {

// 假写件：计数+帧序录制；writeOk 全局置成败；failPayload 按帧内容定向置败
struct Harness {
    FrameCodec codec;
    int writes = 0;
    bool writeOk = true;
    std::string failPayload;                     // 非空=该载荷组帧后写败
    std::vector<std::string> frames;

    CommandChannel::Deps deps() {
        CommandChannel::Deps d;
        d.codec = &codec;
        d.write = [this](const std::string& f) {
            ++writes;
            frames.push_back(f);
            if (!failPayload.empty() && f == codec.encode(failPayload)) return false;
            return writeOk;
        };
        return d;
    }

    std::string enc(const std::string& payload) const { return codec.encode(payload); }
};

} // namespace

// —— 1. send 写成功：onDone(true,"未确认") 恰一次（同步），帧串=encode(payload) ——
TEST(CommandChannel, SendReportsWriteSuccess) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = false;
    std::string payload;
    ch.send("N11 H0", [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    EXPECT_EQ(done, 1);                          // 同步回告：send 返回前已触发
    EXPECT_TRUE(ok);
    EXPECT_EQ(payload, "未确认");                // 盲发口径：写成功=已出线未确认
    EXPECT_EQ(h.writes, 1);
    ASSERT_EQ(h.frames.size(), 1u);
    EXPECT_EQ(h.frames[0], h.enc("N11 H0"));     // 帧串经 codec 组帧
}

// —— 2. send 写失败：onDone(false, payload)、载荷回传原值（V2WriteFail 语义保留）——
TEST(CommandChannel, SendReportsWriteFail) {
    Harness h;
    h.writeOk = false;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = true;
    std::string payload;
    ch.send("N11 H0", [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(payload, "N11 H0");                // 载荷回传原值（IMCU「同步回告写成败」契约）
    EXPECT_EQ(h.writes, 1);
}

// —— 3. fire-and-forget：无回调、帧串正确 ——
TEST(CommandChannel, FireAndForgetNoCallback) {
    Harness h;
    CommandChannel ch(h.deps());
    ch.sendFireAndForget("N12 T100");
    EXPECT_EQ(h.writes, 1);
    ASSERT_EQ(h.frames.size(), 1u);
    EXPECT_EQ(h.frames[0], h.enc("N12 T100"));
}

// —— 4. 组链 happy：3 帧依序写、onGroupDone(true, 最后payload) 恰一次 ——
TEST(CommandChannel, GroupAllPass) {
    Harness h;
    CommandChannel ch(h.deps());
    int groupDone = 0;
    bool groupOk = false;
    std::string groupPayload;
    ch.sendGroup({"N12 T100", "N10 H5", "N11 H0"}, [&](bool o, const std::string& p) {
        groupDone++;
        groupOk = o;
        groupPayload = p;
    });
    EXPECT_EQ(h.writes, 3);                      // 同步循环：sendGroup 返回前全写完
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc("N12 T100"), h.enc("N10 H5"), h.enc("N11 H0")}));
    EXPECT_EQ(groupDone, 1);
    EXPECT_TRUE(groupOk);
    EXPECT_EQ(groupPayload, "N11 H0");           // 组成功载荷=末步载荷
}

// —— 5. 组中段写败短路：第 2 帧写败 → 第 3 帧不写、onGroupDone(false, 第2payload)、
//    写次数=2 ——
TEST(CommandChannel, GroupMidFailShortCircuits) {
    Harness h;
    h.failPayload = "N10 H5";
    CommandChannel ch(h.deps());
    const std::string A = "N12 T100", B = "N10 H5", C = "N11 H0";
    int groupDone = 0;
    bool groupOk = true;
    std::string groupPayload;
    ch.sendGroup({A, B, C}, [&](bool o, const std::string& p) {
        groupDone++;
        groupOk = o;
        groupPayload = p;
    });
    EXPECT_EQ(h.writes, 2);                      // A+B；C 短路未写
    ASSERT_EQ(h.frames.size(), 2u);
    EXPECT_EQ(h.frames[0], h.enc(A));
    EXPECT_EQ(h.frames[1], h.enc(B));
    EXPECT_EQ(groupDone, 1);
    EXPECT_FALSE(groupOk);
    EXPECT_EQ(groupPayload, B);                  // 败步载荷=第 2 帧
}

// —— 6. 空组：onGroupDone(true, "") 恰一次，无 write ——
TEST(CommandChannel, GroupEmpty) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = false;
    std::string payload = "x";
    ch.sendGroup({}, [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    EXPECT_EQ(done, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(payload, "");
    EXPECT_EQ(h.writes, 0);
}

// —— 7. codec 空：裸载荷直发（不加 ';'——防护口径延续），write 收到 payload 原串 ——
TEST(CommandChannel, NullCodecBareSend) {
    int writes = 0;
    std::string received;
    CommandChannel::Deps d;
    d.codec = nullptr;
    d.write = [&](const std::string& f) {
        ++writes;
        received = f;
        return true;
    };
    CommandChannel ch(d);
    int done = 0;
    ch.send("N11 H0", [&](bool, const std::string&) { done++; });
    EXPECT_EQ(writes, 1);
    EXPECT_EQ(received, "N11 H0");               // 裸载荷原串（无组帧无尾分号）
    EXPECT_EQ(done, 1);
}

// —— 8. write 空：钳为恒 false → send 恒 onDone(false, payload) ——
TEST(CommandChannel, NullWriteAlwaysFails) {
    CommandChannel ch(CommandChannel::Deps{});   // codec/write 全空
    int done = 0;
    bool ok = true;
    std::string payload;
    ch.send("N11 H0", [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(payload, "N11 H0");
    ch.sendFireAndForget("N12 T100");            // 不崩（无回调路径）
    ch.sendGroup({"A", "B"}, [](bool o, const std::string& p) {
        EXPECT_FALSE(o);
        EXPECT_EQ(p, "A");                       // 首步即败短路
    });
}

// —— 9. 回调内再 send：同步重入冒烟——不死锁不崩溃，嵌套 send 正常回告 ——
TEST(CommandChannel, CallbackResendSafe) {
    Harness h;
    CommandChannel ch(h.deps());
    int nestedDone = 0;
    bool nestedOk = false;
    ch.send("P1", [&](bool o, const std::string&) {
        if (o) ch.send("P2", [&](bool o2, const std::string&) {
            nestedDone++;
            nestedOk = o2;
        });
    });
    EXPECT_EQ(h.writes, 2);                      // 嵌套 send 已同步出线
    EXPECT_EQ(nestedDone, 1);
    EXPECT_TRUE(nestedOk);
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc("P1"), h.enc("P2")}));
}
