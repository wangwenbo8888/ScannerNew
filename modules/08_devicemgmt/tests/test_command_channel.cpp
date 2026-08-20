// ============================================================================
// test_command_channel.cpp — 下行可靠通道单测（S-T4）
//
// 用例 = 设计 §2.3 + §7 单元行 + 二轮 R2-A2 组步链：非阻塞首发 / ACK 销项 /
// 超时重传 / 3 败 Fault / 表容量 8 逐出 / 查询不挂表 / 组步链 happy+中段败 /
// v2 降级 / seq 循环 / 未知 ACK / write 失败口径 / v2 组 / 回调重入。
// 假件：假写（计数+帧序录制+可置失败）、假钟（闭包持有 int64 可推进）。
// seq 分配自 0 顺序可预期（nextSeq 契约），ACK 用已知 seq 回填。
// 口径（钉死）：write 失败=消耗一次尝试（不立即补发，由 tick 推进至 1+maxRetries
// 发用尽判败）；判败与最后一发同 tick 收口（3 tick→4 write→fail）。
// ============================================================================

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "modules/08_devicemgmt/serial/CommandChannel.h"

using namespace Scanner::device::serial;

namespace {

struct Harness {
    explicit Harness(FrameCodec::Version v = FrameCodec::Version::V3, bool reliable = true)
        : codec(v), reliable(reliable) {}

    FrameCodec codec;
    bool reliable;
    int64_t clock = 0;
    int writes = 0;
    bool writeOk = true;
    std::vector<std::string> frames;

    CommandChannel::Deps deps() {
        CommandChannel::Deps d;
        d.codec = &codec;
        d.write = [this](const std::string& f) {
            writes++;
            frames.push_back(f);
            return writeOk;
        };
        d.nowMs = [this] { return clock; };
        d.reliable = reliable;
        return d;
    }

    void advance(int64_t ms) { clock += ms; }
    std::string enc(const std::string& payload, uint16_t seq) const { return codec.encode(payload, seq); }
};

} // namespace

// —— 1. 非阻塞铁律（T10 守护）：send 立即返回，write 恰 1 次，onDone 未触发 ——
TEST(CommandChannel, SendImmediateReturn) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    ch.send("N11H1", [&](bool, const std::string&) { done++; });
    EXPECT_EQ(h.writes, 1);
    EXPECT_EQ(done, 0);
}

// —— 2. ACK 命中：onDone(true, 原载荷) 恰一次；已销项后重复 ACK 无效 ——
TEST(CommandChannel, AckMatches) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = false;
    std::string payload;
    ch.send("N11H1", [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    ch.onAck(0);  // 首发 seq=0
    EXPECT_EQ(done, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(payload, "N11H1");
    ch.onAck(0);  // 已销项
    EXPECT_EQ(done, 1);
}

// —— 3. 超时重传：假钟 +100ms → tick → 第 2 次 write（同 seq 同帧），尚未判败 ——
TEST(CommandChannel, AckTimeoutRetransmit) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    ch.send("N11H1", [&](bool, const std::string&) { done++; });
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 2);
    EXPECT_EQ(done, 0);
    EXPECT_EQ(h.frames[0], h.frames[1]);
}

// —— 4. 3 败 Fault：+100ms×3 tick → 共 4 次 write → onDone(false) 一次 +
//    onFault(payload, 4) 一次；后再 tick 无动作 ——
TEST(CommandChannel, ThreeFailsFault) {
    Harness h;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = true;
    int faults = 0;
    std::string faultPayload;
    int faultAttempts = 0;
    ch.onFault = [&](const std::string& p, int a) {
        faults++;
        faultPayload = p;
        faultAttempts = a;
    };
    ch.send("N13E0", [&](bool o, const std::string&) {
        done++;
        ok = o;
    });
    EXPECT_EQ(h.writes, 1);
    for (int k = 1; k <= 3; ++k) {
        h.advance(100);
        ch.tick();
        EXPECT_EQ(h.writes, k + 1);
        if (k < 3) EXPECT_EQ(done, 0);
    }
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(faults, 1);
    EXPECT_EQ(faultPayload, "N13E0");
    EXPECT_EQ(faultAttempts, 4);
    h.advance(100);
    ch.tick();  // 判败后再 tick 无动作
    EXPECT_EQ(h.writes, 4);
    EXPECT_EQ(done, 1);
    EXPECT_EQ(faults, 1);
}

// —— 5. 表容量 8：连续 send 9 条（不 ACK 不 tick）→ 第 9 条挂入时最旧被逐出
//    判超时（onDone(false)+onFault(payload,1) 口径钉死），其余照常可 ACK ——
TEST(CommandChannel, TableCap8) {
    Harness h;
    CommandChannel ch(h.deps());
    std::vector<int> okCount(9, 0), failCount(9, 0);
    int faults = 0;
    std::string faultPayload;
    int faultAttempts = 0;
    ch.onFault = [&](const std::string& p, int a) {
        faults++;
        faultPayload = p;
        faultAttempts = a;
    };
    for (int i = 0; i < 9; ++i) {
        ch.send("P" + std::to_string(i + 1), [&, i](bool o, const std::string&) {
            (o ? okCount : failCount)[i]++;
        });
    }
    EXPECT_EQ(h.writes, 9);
    EXPECT_EQ(failCount[0], 1);  // P1（最旧）被逐出判超时
    EXPECT_EQ(okCount[0], 0);
    for (int i = 1; i < 9; ++i) EXPECT_EQ(failCount[i], 0) << "P" << i + 1;
    EXPECT_EQ(faults, 1);
    EXPECT_EQ(faultPayload, "P1");
    EXPECT_EQ(faultAttempts, 1);
    ch.onAck(1);  // P2（seq=1）正常销项
    EXPECT_EQ(okCount[1], 1);
}

// —— 6. 查询类：sendFireAndForget → write 1 次、不挂表（tick 无重传）、onAck 任何 seq 无副作用 ——
TEST(CommandChannel, FireAndForget) {
    Harness h;
    CommandChannel ch(h.deps());
    ch.sendFireAndForget("N15V2");
    EXPECT_EQ(h.writes, 1);
    for (int k = 0; k < 3; ++k) {
        h.advance(100);
        ch.tick();
    }
    EXPECT_EQ(h.writes, 1);
    ch.onAck(0);
    ch.onAck(123);
    EXPECT_EQ(h.writes, 1);
}

// —— 7. 组步链 happy：仅 A 先发；ACK(A)→B、ACK(B)→C、ACK(C)→onGroupDone(true) 一次；
//    三步严格先后（write 序列断言）——
TEST(CommandChannel, GroupHappyPath) {
    Harness h;
    CommandChannel ch(h.deps());
    const std::string A = "N13E0", B = "N10H1", C = "N11H1";
    int groupDone = 0;
    bool groupOk = false;
    std::string groupPayload;
    ch.sendGroup({A, B, C}, [&](bool o, const std::string& p) {
        groupDone++;
        groupOk = o;
        groupPayload = p;
    });
    EXPECT_EQ(h.writes, 1);  // 步链：前条 ACK 前不发下一条
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc(A, 0)}));
    ch.onAck(0);
    EXPECT_EQ(h.writes, 2);
    ch.onAck(1);
    EXPECT_EQ(h.writes, 3);
    EXPECT_EQ(groupDone, 0);  // C 未 ACK，组未完成
    ch.onAck(2);
    EXPECT_EQ(groupDone, 1);
    EXPECT_TRUE(groupOk);
    EXPECT_EQ(groupPayload, C);  // 组成功载荷=末步载荷（口径钉死）
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc(A, 0), h.enc(B, 1), h.enc(C, 2)}));
    ch.onAck(2);  // 已完成，重复 ACK 无效
    EXPECT_EQ(groupDone, 1);
}

// —— 8. 组中段 3 败：A ACK→B 发出→B 3 败（tick×3）→ C 的 write 从未发生 +
//    onGroupDone(false) 一次 + B 步自身 onFault(B, 4) ——
TEST(CommandChannel, GroupMidFail) {
    Harness h;
    CommandChannel ch(h.deps());
    const std::string A = "N13E0", B = "N10H1", C = "N11H1";
    int groupDone = 0;
    bool groupOk = true;
    int faults = 0;
    std::string faultPayload;
    int faultAttempts = 0;
    ch.onFault = [&](const std::string& p, int a) {
        faults++;
        faultPayload = p;
        faultAttempts = a;
    };
    ch.sendGroup({A, B, C}, [&](bool o, const std::string&) {
        groupDone++;
        groupOk = o;
    });
    ch.onAck(0);  // A ACK → B 发出
    EXPECT_EQ(h.writes, 2);
    for (int k = 0; k < 3; ++k) {
        h.advance(100);
        ch.tick();
    }
    EXPECT_EQ(h.writes, 5);  // A×1 + B×4
    EXPECT_EQ(groupDone, 1);
    EXPECT_FALSE(groupOk);
    EXPECT_EQ(faults, 1);
    EXPECT_EQ(faultPayload, B);
    EXPECT_EQ(faultAttempts, 4);
    EXPECT_EQ(h.frames,
              (std::vector<std::string>{h.enc(A, 0), h.enc(B, 1), h.enc(B, 1), h.enc(B, 1), h.enc(B, 1)}));
    h.advance(100);
    ch.tick();  // 组已短路，后再 tick 无动作
    EXPECT_EQ(h.writes, 5);
    EXPECT_EQ(groupDone, 1);
}

// —— 9. v2 降级：send 立即 onDone(true,"未确认")，无重传（tick 后 write 仍 1），onAck 无效 ——
TEST(CommandChannel, V2Degraded) {
    Harness h(FrameCodec::Version::V2, false);
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = false;
    std::string payload;
    ch.send("N11H1", [&](bool o, const std::string& p) {
        done++;
        ok = o;
        payload = p;
    });
    EXPECT_EQ(done, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(payload, "未确认");
    EXPECT_EQ(h.writes, 1);
    ASSERT_EQ(h.frames.size(), 1u);
    EXPECT_EQ(h.frames[0], "N11H1;");
    h.advance(100);
    ch.tick();
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 1);
    ch.onAck(0);
    EXPECT_EQ(done, 1);
}

// —— 10. seq 0~255 循环：连调 257 次 → 0..255,0,1 ——
TEST(CommandChannel, SeqWraps) {
    Harness h;
    CommandChannel ch(h.deps());
    for (int i = 0; i < 257; ++i) EXPECT_EQ(ch.nextSeq(), i % 256);
}

// —— 11. 未知 ACK：空表/未命中 seq 无副作用无崩溃；条目仍在表（tick 照常重传）——
TEST(CommandChannel, AckIgnoresUnknown) {
    Harness h;
    CommandChannel ch(h.deps());
    ch.onAck(200);  // 空表
    int done = 0;
    ch.send("N11H1", [&](bool, const std::string&) { done++; });
    ch.onAck(99);  // 未命中
    EXPECT_EQ(done, 0);
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 2);  // 未销项 → 照常重传
    ch.onAck(0);
    EXPECT_EQ(done, 1);
}

// —— 12. write 失败口径（钉死）：视同消耗一次尝试（首发即败不立即补发），
//    到期由 tick 重传推进，1+maxRetries 发用尽即 3 败（onDone(false)+onFault）——
TEST(CommandChannel, WriteFailFastPath) {
    Harness h;
    h.writeOk = false;
    CommandChannel ch(h.deps());
    int done = 0;
    bool ok = true;
    int faults = 0;
    int faultAttempts = 0;
    ch.onFault = [&](const std::string&, int a) {
        faults++;
        faultAttempts = a;
    };
    ch.send("N11H1", [&](bool o, const std::string&) {
        done++;
        ok = o;
    });
    EXPECT_EQ(h.writes, 1);  // 失败不触发立即补发
    EXPECT_EQ(done, 0);
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 2);
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 3);
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 4);
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(faults, 1);
    EXPECT_EQ(faultAttempts, 4);
}

// —— 13. v2 组：步链仍同步走完（每条立即"确认"）→ onGroupDone(true) 一次，无重传 ——
TEST(CommandChannel, GroupV2) {
    Harness h(FrameCodec::Version::V2, false);
    CommandChannel ch(h.deps());
    int groupDone = 0;
    bool groupOk = false;
    std::string groupPayload;
    ch.sendGroup({"N13E0", "N10H1", "N11H1"}, [&](bool o, const std::string& p) {
        groupDone++;
        groupOk = o;
        groupPayload = p;
    });
    EXPECT_EQ(h.writes, 3);
    EXPECT_EQ(h.frames, (std::vector<std::string>{"N13E0;", "N10H1;", "N11H1;"}));
    EXPECT_EQ(groupDone, 1);
    EXPECT_TRUE(groupOk);
    EXPECT_EQ(groupPayload, "未确认");  // 末步 v2 完成载荷（口径钉死）
    h.advance(100);
    ch.tick();
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 3);
}

// —— 14. 回调重入（回调短平快前提下的最小保证）：ACK 回调内嵌套 send /
//    tick 判败回调内嵌套 send——不崩溃、新命令挂表可 ACK ——
TEST(CommandChannel, ReentrancyGuard) {
    // a) ACK 完成回调内嵌套 send：先销项再回调 → 新命令正常挂表
    {
        Harness h;
        CommandChannel ch(h.deps());
        int nestedDone = 0;
        ch.send("P1", [&](bool o, const std::string&) {
            if (o) ch.send("P2", [&](bool, const std::string&) { nestedDone++; });
        });
        EXPECT_EQ(h.writes, 1);
        ch.onAck(0);  // 回调内发 P2（seq=1）
        EXPECT_EQ(h.writes, 2);
        ch.onAck(1);
        EXPECT_EQ(nestedDone, 1);
    }
    // b) tick 判败回调内嵌套 send：新命令挂表不被本 tick 波及，可正常 ACK
    {
        Harness h;
        CommandChannel ch(h.deps());
        int nestedDone = 0;
        bool nestedOk = false;
        ch.send("Q1", [&](bool o, const std::string&) {
            if (!o) ch.send("Q2", [&](bool o2, const std::string&) {
                nestedDone++;
                nestedOk = o2;
            });
        });
        for (int k = 0; k < 3; ++k) {
            h.advance(100);
            ch.tick();
        }
        EXPECT_EQ(h.writes, 5);  // Q1×4 + Q2×1
        ch.onAck(1);             // Q2（seq=1）正常销项
        EXPECT_EQ(nestedDone, 1);
        EXPECT_TRUE(nestedOk);
        h.advance(100);
        ch.tick();
        EXPECT_EQ(h.writes, 5);  // Q2 已 ACK，无重传
    }
}
