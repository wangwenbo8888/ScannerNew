// ============================================================================
// test_command_channel.cpp — 下行可靠通道单测（S-T4）
//
// 用例 = 设计 §2.3 + §7 单元行 + 二轮 R2-A2 组步链：非阻塞首发 / ACK 销项 /
// 超时重传 / 3 败 Fault / 表容量 8 逐出 / 查询不挂表 / 组步链 happy+中段败 /
// v2 降级 / 未知 ACK / write 失败口径 / v2 组 / 回调重入。
// 假件：假写（计数+帧序录制+可置失败+可同步重入）、假钟（闭包持有 int64 可推进）。
// 260831 批1-C 最小编译适配（reliable 机制壳留，批2 随机制整删）：
//   - FrameCodec 无版本（裸 ';' 帧）→ Harness/enc 单参化；
//   - nextSeq 已删 → SeqWraps 用例删；
//   - 表项 seq 恒 0 → ACK 恒命中表首（onAck(1)/(2) 改 onAck(0)）；依赖 seq 身份
//     的重入错位/扩容两用例（原 16/17）删——批2 机制物理删除时整组重写。
// 口径（钉死）：write 失败=消耗一次尝试（不立即补发，由 tick 推进至 1+maxRetries
// 发用尽判败）；判败与最后一发同 tick 收口（3 tick→4 write→fail）。
// 二轮修复钉死：重传写重入（onAck 销项/挪位 + send 扩容）不悬垂；Deps 空容忍
// 构造 clamp；ackTimeoutMs 钳 ≥1（防 tick 活锁）；空组立即成功。
// ============================================================================

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "modules/08_devicemgmt/serial/CommandChannel.h"

using namespace Scanner::device::serial;

namespace {

struct Harness {
    explicit Harness(bool reliable = true)
        : reliable(reliable) {}

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
    std::string enc(const std::string& payload) const { return codec.encode(payload); }
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
    ch.onAck(0);  // P2（表首，seq 恒 0）正常销项
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
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc(A)}));
    ch.onAck(0);
    EXPECT_EQ(h.writes, 2);
    ch.onAck(0);
    EXPECT_EQ(h.writes, 3);
    EXPECT_EQ(groupDone, 0);  // C 未 ACK，组未完成
    ch.onAck(0);
    EXPECT_EQ(groupDone, 1);
    EXPECT_TRUE(groupOk);
    EXPECT_EQ(groupPayload, C);  // 组成功载荷=末步载荷（口径钉死）
    EXPECT_EQ(h.frames, (std::vector<std::string>{h.enc(A), h.enc(B), h.enc(C)}));
    ch.onAck(0);  // 已完成，重复 ACK 无效
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
              (std::vector<std::string>{h.enc(A), h.enc(B), h.enc(B), h.enc(B), h.enc(B)}));
    h.advance(100);
    ch.tick();  // 组已短路，后再 tick 无动作
    EXPECT_EQ(h.writes, 5);
    EXPECT_EQ(groupDone, 1);
}

// —— 9. v2 降级：send 立即 onDone(true,"未确认")，无重传（tick 后 write 仍 1），onAck 无效 ——
TEST(CommandChannel, V2Degraded) {
    Harness h(false);
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

// —— 9b. 盲发写失败（D8 钉死）：reliable=false + write 返回 false →
//    onDone(false, payload) 同步回告、载荷回传原 payload（IMCU.h「完成回调
//    同步回告写成败」契约——DeviceManager 0x0807 三条写败路径的可达性凭据）——
TEST(CommandChannel, V2WriteFail) {
    Harness h(false);
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
    EXPECT_EQ(h.writes, 1);
    EXPECT_EQ(done, 1);
    EXPECT_FALSE(ok);
    EXPECT_EQ(payload, "N11 H0");
}

// —— 10. seq 循环用例已删（260831 无 seq——nextSeq 随批1-A/B 退役）———

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
    Harness h(false);
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
        ch.onAck(0);  // 回调内发 P2（表首，seq 恒 0）
        EXPECT_EQ(h.writes, 2);
        ch.onAck(0);
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
        ch.onAck(0);             // Q2（表首，seq 恒 0）正常销项
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

// —— 16/17（重传写重入错位/扩容）已删：260831 表项 seq 恒 0——双条目 seq 身份
//    不可辨，两用例口径失效；批2 reliable 机制物理删除时整组重写 ——

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
