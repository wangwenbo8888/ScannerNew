// ============================================================================
// test_command_channel.cpp — 下行可靠通道单测（S-T4）
//
// 用例 = 设计 §2.3 + §7 单元行 + 二轮 R2-A2 组步链：非阻塞首发 / ACK 销项 /
// 超时重传 / 3 败 Fault / 表容量 8 逐出 / 查询不挂表 / 组步链 happy+中段败 /
// v2 降级 / seq 循环 / 未知 ACK / write 失败口径 / v2 组 / 回调重入。
// 假件：假写（计数+帧序录制+可置失败+可同步重入）、假钟（闭包持有 int64 可推进）。
// seq 分配自 0 顺序可预期（nextSeq 契约），ACK 用已知 seq 回填。
// 口径（钉死）：write 失败=消耗一次尝试（不立即补发，由 tick 推进至 1+maxRetries
// 发用尽判败）；判败与最后一发同 tick 收口（3 tick→4 write→fail）。
// 二轮修复钉死：重传写重入（onAck 销项/挪位 + send 扩容）不悬垂；Deps 空容忍
// 构造 clamp（write 空→恒 false、nowMs 空→恒 0、codec 空→裸载荷）；ackTimeoutMs
// 钳 ≥1（防 tick 活锁）；空组立即成功。
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
    CommandChannel* target = nullptr;                 // 重入目标（装配后回填）
    int reenterOnWrite = 0;                           // 第 N 次 write 触发同步重入（0=不触发）
    std::function<void(CommandChannel&)> reenter;     // 重入动作（write 回调内同步调）

    CommandChannel::Deps deps() {
        CommandChannel::Deps d;
        d.codec = &codec;
        d.write = [this](const std::string& f) {
            const int n = ++writes;
            frames.push_back(f);
            if (reenter && target && n == reenterOnWrite) reenter(*target);
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

// —— 15. 重传写重入销项（Important #1 回归·单条）：tick 重传的第 2 次 write 回调内
//    同步 onAck(seq) → 不崩溃；该条按 ACK 收口（onDone(true) 一次、无 Fault、
//    不再计 attempts——后续 tick 无重传无判败）——
TEST(CommandChannel, RetransmitWriteReentrantAck) {
    Harness h;
    CommandChannel ch(h.deps());
    h.target = &ch;
    int done = 0;
    bool ok = false;
    int faults = 0;
    ch.onFault = [&](const std::string&, int) { faults++; };
    ch.send("N11H1", [&](bool o, const std::string&) {
        done++;
        ok = o;
    });
    h.advance(100);
    h.reenterOnWrite = 2;  // 第 2 次 write = 首次重传
    h.reenter = [&](CommandChannel& c) { c.onAck(0); };
    ch.tick();
    EXPECT_EQ(h.writes, 2);
    EXPECT_EQ(done, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(faults, 0);
    h.advance(1000);
    ch.tick();
    EXPECT_EQ(h.writes, 2);
    EXPECT_EQ(faults, 0);
}

// —— 16. 重传写重入销项·错位（Important #1 回归）：A/B 双挂表，重传 A 的 write
//    回调内 onAck(A) → A 销项后 B 前移；B 的 attempts 不得被错位 ++（同 tick B
//    依序正常重传 1 次；判败前 B 恰 4 次写）——
TEST(CommandChannel, RetransmitWriteReentrantAckShift) {
    Harness h;
    CommandChannel ch(h.deps());
    h.target = &ch;
    int faults = 0;
    int faultAttempts = 0;
    ch.onFault = [&](const std::string&, int a) {
        faults++;
        faultAttempts = a;
    };
    int doneB = 0;
    bool okB = true;
    ch.send("A", [](bool, const std::string&) {});
    ch.send("B", [&](bool o, const std::string&) {
        doneB++;
        okB = o;
    });
    EXPECT_EQ(h.writes, 2);
    h.advance(100);
    h.reenterOnWrite = 3;  // 第 3 次 write = A 的首次重传
    h.reenter = [&](CommandChannel& c) { c.onAck(0); };
    ch.tick();
    EXPECT_EQ(h.writes, 4);  // A 重传 + B 依序重传（错位 ++ 会漏 B 这 1 写）
    EXPECT_EQ(doneB, 0);
    h.reenterOnWrite = 0;
    int ticksNeeded = 0;
    for (int k = 1; k <= 2; ++k) {  // B 再 2 轮重传达 4 尝试 → 3 败
        h.advance(100);
        ch.tick();
        if (doneB) {
            ticksNeeded = k;
            break;
        }
    }
    EXPECT_EQ(ticksNeeded, 2);
    EXPECT_EQ(h.writes, 6);  // A:2 + B:4（1 首发+3 重传）
    EXPECT_FALSE(okB);
    EXPECT_EQ(faults, 1);
    EXPECT_EQ(faultAttempts, 4);
}

// —— 17. 重传写重入 send·扩容（Important #1 回归）：重传 A 的 write 回调内同步
//    send(C)（push_back 扩容搬移表内元素）→ A 计数不丢（不重发不漏发），C 正常 ——
TEST(CommandChannel, RetransmitWriteReentrantSend) {
    Harness h;
    CommandChannel ch(h.deps());
    h.target = &ch;
    int doneA = 0;
    bool okA = true;
    int faults = 0;
    ch.onFault = [&](const std::string&, int) { faults++; };
    ch.send("A", [&](bool o, const std::string&) {
        doneA++;
        okA = o;
    });
    h.advance(100);
    h.reenterOnWrite = 2;  // 第 2 次 write = A 首次重传
    h.reenter = [&](CommandChannel& c) { c.send("C", [](bool, const std::string&) {}); };
    ch.tick();
    EXPECT_EQ(h.writes, 3);  // A 首发 + A 重传恰 1 + C 首发（悬垂会致 A 漏记重发）
    ch.onAck(1);             // C（seq=1）销项，隔离后续断言
    h.reenterOnWrite = 0;
    int ticksNeeded = 0;
    for (int k = 1; k <= 2; ++k) {  // A 已 2 尝试，再 2 轮重传 → 3 败
        h.advance(100);
        ch.tick();
        if (doneA) {
            ticksNeeded = k;
            break;
        }
    }
    EXPECT_EQ(ticksNeeded, 2);
    EXPECT_EQ(h.writes, 5);
    EXPECT_FALSE(okA);
    EXPECT_EQ(faults, 1);
    h.advance(100);
    ch.tick();
    EXPECT_EQ(h.writes, 5);  // A 已败 C 已销 → 无动作
}

// —— 18. Deps 空容忍（构造 clamp 钉死）：codec/write/nowMs 全空 → 不抛不崩、
//    发不挂账、tick 安全（write 空=恒 false、nowMs 空=恒 0、codec 空=裸载荷直发）——
TEST(CommandChannel, NullDepsClamped) {
    CommandChannel ch(CommandChannel::Deps{});
    int done = 0;
    ch.onFault = [](const std::string&, int) {};
    EXPECT_NO_THROW({
        ch.send("N11H1", [&](bool, const std::string&) { done++; });
        ch.sendFireAndForget("N15V2");
        ch.tick();
    });
    EXPECT_EQ(done, 0);
}

// —— 19. ackTimeoutMs 钳 ≥1（构造时）：0 注入 → t=0 tick 不重传（未钳则到期时刻
//    0 触发活锁式连发）；钳后按 1ms 节奏正常推进 ——
TEST(CommandChannel, AckTimeoutClampedMin1) {
    Harness h;
    auto d = h.deps();
    d.ackTimeoutMs = 0;
    d.maxRetries = 1;
    CommandChannel ch(d);
    int done = 0;
    bool ok = true;
    int faults = 0;
    ch.onFault = [&](const std::string&, int) { faults++; };
    ch.send("N11H1", [&](bool o, const std::string&) {
        done++;
        ok = o;
    });
    ch.tick();  // t=0：dueMs=0+1=1 > 0 → 不重传（防活锁）
    EXPECT_EQ(h.writes, 1);
    h.advance(1);
    ch.tick();  // t=1：到期 → 重传+发尽判败同 tick
    EXPECT_EQ(h.writes, 2);
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(faults, 1);
}

// —— 20. 空组（#5 钉死）：sendGroup({}, cb) → 立即 cb(true, "") 恰一次，无 write ——
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
