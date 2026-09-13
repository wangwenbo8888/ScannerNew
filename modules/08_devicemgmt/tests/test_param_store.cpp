// ============================================================================
// test_param_store.cpp — ParamStore 参数账本单测（D-T10）
//
// 契约钉死（2026-08-18 设计 §2.5 + 2026-08-20 设计 §4-5 确认更新制）：
//   - 开机装载：Load 有档→按档改账（confirmed=false, source=Boot；未知/坏档段
//     忽略；越界钳回 spec 范围）；无档→默认值。均逐参数广播 onParamChanged；
//     bootstrap 清在途（会话重启语义——迟到回调一律 gen 失配作废）；
//   - 设值：唯一入口（UI 滑条/按键步进同队列串行——单线程属主=逻辑线程）；
//     入口即钳 spec 范围；未登记 key 不下发不广播；
//   - Dispatch 完成回调口径 void(bool ok, bool confirmed)（D-T10 裁定）：
//     v3=cb(ACK结果, ACK结果)；v2=cb(true, false)。ok→改账+广播 onParamChanged；
//     !ok→保旧值+onReject(key, oldValue) 恰一次、无广播；
//   - 在途后值胜出：同 key 再 setValue 覆盖在途（gen 判据）——旧回调到达
//     （无论成败）不改账不弹回；
//   - persist/bootstrap 手写极简格式 "key=value;..."（08 不引 json 库），
//     存取全经注入回调（06 无直链）；人工触发才落盘；
//   - setEntryDirect：直接改账不经下发、不广播（bootstrap 同源的静默写路径）。
// 用例 = 实施计划 Task 10 十二条 + 钳档/直写两条补钉。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/ParamStore.h"

#include <string>
#include <utility>
#include <vector>

using namespace Scanner::device;
using Source = ParamEntry::Source;

namespace {

std::vector<ParamSpec> demoSpecs() {
    // exposure: 默认 10，范围 [1,100]；laser: 默认 200，范围 [0,255]
    return {{"exposure", 10.0, 1.0, 100.0}, {"laser", 200.0, 0.0, 255.0}};
}

// 记录型 Dispatch：记调用序 + 暂存完成回调由测试择机触发
struct Recorder {
    std::vector<std::pair<std::string, double>> calls;
    std::vector<ParamStore::Done> cbs;
    ParamStore::Dispatch dispatch() {
        return [this](const std::string& k, double v, ParamStore::Done done) {
            calls.emplace_back(k, v);
            cbs.push_back(std::move(done));
        };
    }
};

} // namespace

// —— 1. BootstrapNoFile：Load 返空串 → 全默认值 + onParamChanged 每参数恰一次 ——
TEST(ParamStore, BootstrapNoFile) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.bootstrap([] { return std::string(""); });
    EXPECT_EQ(ps.get("exposure").value, 10.0);       // 默认值
    EXPECT_FALSE(ps.get("exposure").confirmed);
    EXPECT_EQ(ps.get("exposure").source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);
    EXPECT_FALSE(ps.get("laser").confirmed);
    ASSERT_EQ(changed.size(), 2u);                    // 每参数恰一次
    EXPECT_TRUE(ps.has("exposure") && ps.has("laser"));
    EXPECT_TRUE(rec.calls.empty());                   // 装载不下发
}

// —— 2. BootstrapFromFile：按档改账 confirmed=false；未知 key/坏数值段忽略 ——
TEST(ParamStore, BootstrapFromFile) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.bootstrap([] { return std::string("exposure=15;laser=abc;foo=1;"); });
    EXPECT_EQ(ps.get("exposure").value, 15.0);        // 按档
    EXPECT_FALSE(ps.get("exposure").confirmed);       // 读档不等于 ACK 确认
    EXPECT_EQ(ps.get("exposure").source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);          // laser=abc 坏段忽略 → 默认
    EXPECT_FALSE(ps.has("foo"));                      // 未知 key 忽略
    ASSERT_EQ(changed.size(), 2u);                    // 仍每参数广播一次
}

// —— 3. SetAckConfirm：setValue → dispatch(key,期望值) → cb(true,true) 改账+广播 ——
TEST(ParamStore, SetAckConfirm) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);                  // 下发恰一次
    EXPECT_EQ(rec.calls[0].first, "exposure");
    EXPECT_EQ(rec.calls[0].second, 15.0);
    EXPECT_TRUE(ps.pending("exposure"));              // 在途未决
    EXPECT_EQ(ps.get("exposure").value, 10.0);        // ACK 前不改账
    rec.cbs[0](true, true);                           // v3：ACK 成功
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 15.0);                         // 改账
    EXPECT_TRUE(e.confirmed);                         // confirmed=ok
    EXPECT_EQ(e.source, Source::Ui);
    EXPECT_FALSE(ps.pending("exposure"));             // 已决
    ASSERT_EQ(changed.size(), 1u);                    // 广播恰一次（改账后）
    EXPECT_EQ(changed[0].first, "exposure");
    EXPECT_EQ(changed[0].second.value, 15.0);
    EXPECT_TRUE(changed[0].second.confirmed);
}

// —— 4. SetThreeFailReject：cb(false,*) → 账保旧值 + onReject 恰一次 + 无广播 ——
TEST(ParamStore, SetThreeFailReject) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    std::vector<std::pair<std::string, double>> rejects;
    ps.onReject = [&](const std::string& k, double oldV) { rejects.emplace_back(k, oldV); };
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    rec.cbs[0](false, false);                         // 3 败终报（重试归 CommandChannel）
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 10.0);                         // 保旧值（默认账）
    EXPECT_FALSE(e.confirmed);
    EXPECT_EQ(e.source, Source::Boot);
    ASSERT_EQ(rejects.size(), 1u);                    // 弹回恰一次
    EXPECT_EQ(rejects[0].first, "exposure");
    EXPECT_EQ(rejects[0].second, 10.0);               // 旧值随行（UI 弹回用）
    EXPECT_TRUE(changed.empty());                     // 无改账广播
    EXPECT_FALSE(ps.pending("exposure"));
}

// —— 5. V2ImmediateConfirmedFalse：cb(true,false) → 改账 confirmed=false（v2 期望值）——
TEST(ParamStore, V2ImmediateConfirmedFalse) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    std::vector<std::pair<std::string, double>> rejects;
    ps.onReject = [&](const std::string& k, double oldV) { rejects.emplace_back(k, oldV); };
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    rec.cbs[0](true, false);                          // v2：无 ACK 立即成功但不确认
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 15.0);                         // 立即改账
    EXPECT_FALSE(e.confirmed);                        // 期望值（未读回确认）
    EXPECT_EQ(e.source, Source::Ui);
    ASSERT_EQ(changed.size(), 1u);                    // 改账仍广播
    EXPECT_TRUE(rejects.empty());                     // 不弹回
}

// —— 6. InFlightLastWins：后值覆盖在途——旧回调（含 3 败）到达不改账不弹回 ——
TEST(ParamStore, InFlightLastWins) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    std::vector<std::pair<std::string, double>> rejects;
    ps.onReject = [&](const std::string& k, double oldV) { rejects.emplace_back(k, oldV); };
    ps.setValue("exposure", 5.0, Source::Ui);         // gen1（不回调）
    ps.setValue("exposure", 7.0, Source::Ui);         // gen2 取代 gen1
    ASSERT_EQ(rec.calls.size(), 2u);
    EXPECT_TRUE(ps.pending("exposure"));
    rec.cbs[0](false, false);                         // gen1 的 3 败迟到 → 已被取代
    EXPECT_EQ(ps.get("exposure").value, 10.0);        // 不改账
    EXPECT_TRUE(rejects.empty());                     // 不弹回
    EXPECT_TRUE(changed.empty());
    EXPECT_TRUE(ps.pending("exposure"));              // gen2 仍在途
    rec.cbs[1](true, true);                           // gen2 ACK 成功
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 7.0);
    EXPECT_TRUE(e.confirmed);
    EXPECT_FALSE(ps.pending("exposure"));
    ASSERT_EQ(changed.size(), 1u);
}

// —— 7. RangeClamp：setValue 越界 → 入口即钳，dispatch 收到钳后值，落账同钳 ——
TEST(ParamStore, RangeClampAtEntry) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.setValue("exposure", 150.0, Source::Ui);       // max=100
    ASSERT_EQ(rec.calls.size(), 1u);
    EXPECT_EQ(rec.calls[0].second, 100.0);            // dispatch 收到 100
    ps.setValue("laser", -5.0, Source::Key);          // min=0
    ASSERT_EQ(rec.calls.size(), 2u);
    EXPECT_EQ(rec.calls[1].second, 0.0);
    rec.cbs[0](true, true);
    rec.cbs[1](true, true);
    EXPECT_EQ(ps.get("exposure").value, 100.0);       // 落账也是钳后值
    EXPECT_EQ(ps.get("laser").value, 0.0);
}

// —— 8. TwoSourcesSameQueue：Ui 源与 Key 源先后设值 → 同队列按序下发、各自落账 ——
TEST(ParamStore, TwoSourcesSameQueue) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.setValue("exposure", 5.0, Source::Ui);         // 滑条入口
    ps.setValue("laser", 220.0, Source::Key);         // 按键步进入口
    ASSERT_EQ(rec.calls.size(), 2u);                  // 两入口同队列、按序不交错
    EXPECT_EQ(rec.calls[0].first, "exposure");
    EXPECT_EQ(rec.calls[0].second, 5.0);
    EXPECT_EQ(rec.calls[1].first, "laser");
    EXPECT_EQ(rec.calls[1].second, 220.0);
    rec.cbs[0](true, true);
    rec.cbs[1](true, true);
    EXPECT_EQ(ps.get("exposure").source, Source::Ui);
    EXPECT_EQ(ps.get("laser").source, Source::Key);
    EXPECT_EQ(ps.get("exposure").value, 5.0);
    EXPECT_EQ(ps.get("laser").value, 220.0);
}

// —— 9. PersistRoundTrip：改几笔 → persist 串 → 新 store bootstrap 同串 → 值一致 ——
TEST(ParamStore, PersistRoundTrip) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    rec.cbs[0](true, true);                           // 确认值 15
    ps.setValue("laser", 220.0, Source::Key);
    ASSERT_EQ(rec.calls.size(), 2u);
    rec.cbs[1](true, false);                          // v2 期望值 220
    std::string json;
    const bool ok = ps.persist([&](const std::string& s) { json = s; return true; });
    EXPECT_TRUE(ok);
    EXPECT_EQ(json, "exposure=15.000000;laser=220.000000;");  // 手写极简格式（口径钉死）
    Recorder rec2;
    ParamStore ps2(demoSpecs(), rec2.dispatch());
    ps2.bootstrap([&] { return json; });
    EXPECT_EQ(ps2.get("exposure").value, 15.0);       // 往返一致
    EXPECT_EQ(ps2.get("laser").value, 220.0);
    EXPECT_FALSE(ps2.get("exposure").confirmed);      // 档不携带确认位
    EXPECT_EQ(ps2.get("exposure").source, Source::Boot);
    EXPECT_TRUE(rec2.calls.empty());
}

// —— 10. UnknownKeyIgnored：setValue 未登记 key → 无 dispatch 无广播 ——
TEST(ParamStore, UnknownKeyIgnored) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.setValue("foo", 5.0, Source::Ui);
    EXPECT_TRUE(rec.calls.empty());                   // 不下发
    EXPECT_TRUE(changed.empty());                     // 不广播
    EXPECT_FALSE(ps.has("foo"));
    EXPECT_EQ(ps.get("foo").value, 0.0);              // 未登记 get → 零值
    EXPECT_FALSE(ps.get("foo").confirmed);
    EXPECT_FALSE(ps.pending("foo"));
}

// —— 11. PendingQuery：在途 true；回调决出后 false；未登记恒 false ——
TEST(ParamStore, PendingQuery) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    EXPECT_FALSE(ps.pending("exposure"));             // 无在途
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    EXPECT_TRUE(ps.pending("exposure"));              // 在途
    EXPECT_FALSE(ps.pending("laser"));
    rec.cbs[0](true, true);
    EXPECT_FALSE(ps.pending("exposure"));             // 已决
    ps.setValue("exposure", 20.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 2u);
    rec.cbs[1](false, false);                         // 败也出队
    EXPECT_FALSE(ps.pending("exposure"));
}

// —— 12. BootstrapOverwritesRuntime：运行中有值再 bootstrap → 按档覆盖、清在途、
//        迟到回调作废（会话重启语义）——
TEST(ParamStore, BootstrapOverwritesRuntime) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    rec.cbs[0](true, true);                           // 15 已确认入账
    EXPECT_EQ(ps.get("exposure").value, 15.0);
    ps.setValue("exposure", 17.0, Source::Ui);        // gen2 在途不回调
    ASSERT_EQ(rec.calls.size(), 2u);
    EXPECT_TRUE(ps.pending("exposure"));
    ps.bootstrap([] { return std::string("exposure=20;"); });
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 20.0);                         // 按档覆盖
    EXPECT_FALSE(e.confirmed);
    EXPECT_EQ(e.source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);          // 档缺 → 默认
    EXPECT_FALSE(ps.pending("exposure"));             // 在途清空
    rec.cbs[1](true, true);                           // gen2 迟到 → 作废
    EXPECT_EQ(ps.get("exposure").value, 20.0);        // 账不动
}

// —— 13. BootstrapClampsOutOfRange：档内越界值钳回 spec 范围（与入口即钳同口径）——
TEST(ParamStore, BootstrapClampsOutOfRange) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.bootstrap([] { return std::string("exposure=500;laser=-5;"); });
    EXPECT_EQ(ps.get("exposure").value, 100.0);       // max=100
    EXPECT_EQ(ps.get("laser").value, 0.0);            // min=0
}

// —— 14. SetEntryDirect：直接改账不经下发、不广播（测试/bootstrap 同源静默写）——
TEST(ParamStore, SetEntryDirect) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.setEntryDirect("exposure", 42.0, true, Source::Key);
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 42.0);
    EXPECT_TRUE(e.confirmed);
    EXPECT_EQ(e.source, Source::Key);
    EXPECT_TRUE(rec.calls.empty());                   // 不经下发
    EXPECT_TRUE(changed.empty());                     // 不广播
    ps.setEntryDirect("exposure", 500.0, false, Source::Ui);
    EXPECT_EQ(ps.get("exposure").value, 100.0);       // 直写同样钳范围
    ps.setEntryDirect("foo", 1.0, true, Source::Ui);
    EXPECT_FALSE(ps.has("foo"));                      // 未登记不动账
}
