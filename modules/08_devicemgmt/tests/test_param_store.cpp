// ============================================================================
// test_param_store.cpp — ParamStore 参数账本单测（D-T10；批2 去 confirmed）
//
// 契约钉死（协议 260831：无 ACK——Done=写成败直通）：
//   - 开机装载：Load 有档→按档改账（source=Boot；未知/坏档段忽略；越界钳回
//     spec 范围）；无档→默认值。均逐参数广播 onParamChanged；bootstrap 清在途
//     （会话重启语义——迟到回调一律 gen 失配作废）；
//   - 设值：唯一入口（UI 滑条/按键步进同队列串行——单线程属主=逻辑线程）；
//     入口即钳 spec 范围；未登记 key 不下发不广播；
//   - Dispatch 完成回调口径 void(bool ok)（批2 裁定）：ok=写成败（盲发直通）。
//     ok→改账+广播 onParamChanged；!ok→保旧值+onReject(key, oldValue) 恰一次、
//     无广播；
//   - 在途后值胜出：同 key 再 setValue 覆盖在途（gen 判据）——旧回调到达
//     （无论成败）不改账不弹回；
//   - persist/bootstrap 手写极简格式 "key=value;..."（08 不引 json 库），
//     存取全经注入回调（06 无直链）；人工触发才落盘；
//   - setEntryDirect：直接改账不经下发、不广播（bootstrap 同源的静默写路径）。
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
    EXPECT_EQ(ps.get("exposure").source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);
    ASSERT_EQ(changed.size(), 2u);                    // 每参数恰一次
    EXPECT_TRUE(ps.has("exposure") && ps.has("laser"));
    EXPECT_TRUE(rec.calls.empty());                   // 装载不下发
}

// —— 2. BootstrapFromFile：按档改账 source=Boot；未知 key/坏数值段忽略 ——
TEST(ParamStore, BootstrapFromFile) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    std::vector<std::pair<std::string, ParamEntry>> changed;
    ps.onParamChanged = [&](const std::string& k, const ParamEntry& e) {
        changed.emplace_back(k, e);
    };
    ps.bootstrap([] { return std::string("exposure=15;laser=abc;foo=1;"); });
    EXPECT_EQ(ps.get("exposure").value, 15.0);        // 按档
    EXPECT_EQ(ps.get("exposure").source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);          // laser=abc 坏段忽略 → 默认
    EXPECT_FALSE(ps.has("foo"));                      // 未知 key 忽略
    ASSERT_EQ(changed.size(), 2u);                    // 仍每参数广播一次
}

// —— 3. SetWriteSuccess（原 SetAckConfirm 改写）：setValue → dispatch(key,期望值)
//    → cb(true) 改账+广播（写成败直通——无 ACK 语义）——
TEST(ParamStore, SetWriteSuccess) {
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
    EXPECT_EQ(ps.get("exposure").value, 10.0);        // 回调前不改账
    rec.cbs[0](true);                                 // 写成功（直通）
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 15.0);                         // 改账
    EXPECT_EQ(e.source, Source::Ui);
    EXPECT_FALSE(ps.pending("exposure"));             // 已决
    ASSERT_EQ(changed.size(), 1u);                    // 广播恰一次（改账后）
    EXPECT_EQ(changed[0].first, "exposure");
    EXPECT_EQ(changed[0].second.value, 15.0);
    EXPECT_EQ(changed[0].second.source, Source::Ui);
}

// —— 4. SetWriteFailReject（原 SetThreeFailReject 改写）：cb(false) → 账保旧值 +
//    onReject 恰一次 + 无广播（写失败即终报——重试机制已随批2 删）——
TEST(ParamStore, SetWriteFailReject) {
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
    rec.cbs[0](false);                                // 写失败即终报
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 10.0);                         // 保旧值（默认账）
    EXPECT_EQ(e.source, Source::Boot);
    ASSERT_EQ(rejects.size(), 1u);                    // 弹回恰一次
    EXPECT_EQ(rejects[0].first, "exposure");
    EXPECT_EQ(rejects[0].second, 10.0);               // 旧值随行（UI 弹回用）
    EXPECT_TRUE(changed.empty());                     // 无改账广播
    EXPECT_FALSE(ps.pending("exposure"));
}

// —— 5.（原 V2ImmediateConfirmedFalse 删：v3/v2 双模式随 confirmed 亡——写成败
//    直通单一语义，已并入用例 3）——

// —— 6. InFlightLastWins：后值覆盖在途——旧回调（含写败）到达不改账不弹回 ——
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
    rec.cbs[0](false);                                // gen1 的写败迟到 → 已被取代
    EXPECT_EQ(ps.get("exposure").value, 10.0);        // 不改账
    EXPECT_TRUE(rejects.empty());                     // 不弹回
    EXPECT_TRUE(changed.empty());
    EXPECT_TRUE(ps.pending("exposure"));              // gen2 仍在途
    rec.cbs[1](true);                                 // gen2 写成功
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 7.0);
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
    rec.cbs[0](true);
    rec.cbs[1](true);
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
    rec.cbs[0](true);
    rec.cbs[1](true);
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
    rec.cbs[0](true);                                 // 写成功入账 15
    ps.setValue("laser", 220.0, Source::Key);
    ASSERT_EQ(rec.calls.size(), 2u);
    rec.cbs[1](true);
    std::string json;
    const bool ok = ps.persist([&](const std::string& s) { json = s; return true; });
    EXPECT_TRUE(ok);
    EXPECT_EQ(json, "exposure=15.000000;laser=220.000000;");  // 手写极简格式（口径钉死）
    Recorder rec2;
    ParamStore ps2(demoSpecs(), rec2.dispatch());
    ps2.bootstrap([&] { return json; });
    EXPECT_EQ(ps2.get("exposure").value, 15.0);       // 往返一致
    EXPECT_EQ(ps2.get("laser").value, 220.0);
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
    rec.cbs[0](true);
    EXPECT_FALSE(ps.pending("exposure"));             // 已决
    ps.setValue("exposure", 20.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 2u);
    rec.cbs[1](false);                                // 败也出队
    EXPECT_FALSE(ps.pending("exposure"));
}

// —— 12. BootstrapOverwritesRuntime：运行中有值再 bootstrap → 按档覆盖、清在途、
//        迟到回调作废（会话重启语义）——
TEST(ParamStore, BootstrapOverwritesRuntime) {
    Recorder rec;
    ParamStore ps(demoSpecs(), rec.dispatch());
    ps.setValue("exposure", 15.0, Source::Ui);
    ASSERT_EQ(rec.calls.size(), 1u);
    rec.cbs[0](true);                                 // 15 已入账
    EXPECT_EQ(ps.get("exposure").value, 15.0);
    ps.setValue("exposure", 17.0, Source::Ui);        // gen2 在途不回调
    ASSERT_EQ(rec.calls.size(), 2u);
    EXPECT_TRUE(ps.pending("exposure"));
    ps.bootstrap([] { return std::string("exposure=20;"); });
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 20.0);                         // 按档覆盖
    EXPECT_EQ(e.source, Source::Boot);
    EXPECT_EQ(ps.get("laser").value, 200.0);          // 档缺 → 默认
    EXPECT_FALSE(ps.pending("exposure"));             // 在途清空
    rec.cbs[1](true);                                 // gen2 迟到 → 作废
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
    ps.setEntryDirect("exposure", 42.0, Source::Key);
    const ParamEntry e = ps.get("exposure");
    EXPECT_EQ(e.value, 42.0);
    EXPECT_EQ(e.source, Source::Key);
    EXPECT_TRUE(rec.calls.empty());                   // 不经下发
    EXPECT_TRUE(changed.empty());                     // 不广播
    ps.setEntryDirect("exposure", 500.0, Source::Ui);
    EXPECT_EQ(ps.get("exposure").value, 100.0);       // 直写同样钳范围
    ps.setEntryDirect("foo", 1.0, Source::Ui);
    EXPECT_FALSE(ps.has("foo"));                      // 未登记不动账
}

// —— 15. LegacyFileMigration（协议批3）：旧档 laserSelectA/B 键作废丢弃；
//      laserLevel>100 旧量纲（0-255）读时钳 100——值口径断言（日志口径不强
//      断言）；迁移单向：persist 只写 specs 内键，作废键不回写档 ——
TEST(ParamStore, LegacyFileMigration) {
    Recorder rec;
    const std::vector<ParamSpec> specs = {
        {"exposure", 10.0, 1.0, 100.0},
        {"laserLevel", 40.0, 0.0, 100.0},
    };
    ParamStore ps(specs, rec.dispatch());
    ps.bootstrap([] {
        return std::string("laserSelectA=1;laserSelectB=2;laserLevel=150;exposure=12;");
    });
    EXPECT_FALSE(ps.has("laserSelectA"));             // 旧协议键作废：不入账
    EXPECT_FALSE(ps.has("laserSelectB"));
    EXPECT_EQ(ps.get("laserLevel").value, 100.0);     // 旧量纲迁移：>100 钳 100
    EXPECT_EQ(ps.get("exposure").value, 12.0);        // 其余键不受影响
    std::string out;
    EXPECT_TRUE(ps.persist([&](const std::string& s) { out = s; return true; }));
    EXPECT_EQ(out.find("laserSelect"), std::string::npos);   // 作废键不回写
    EXPECT_NE(out.find("laserLevel=100"), std::string::npos); // 钳后值入档
}
