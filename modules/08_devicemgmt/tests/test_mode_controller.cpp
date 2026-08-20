// ============================================================================
// test_mode_controller.cpp — ModeController 模式黑板单测（D-T9）
//
// 契约钉死（2026-08-18 设计 §2.1 + 2026-08-20 设计 §4-1「ACK 后落板」）：
//   - request 先问门禁：拒→fail 原因透传且板不动；过→ok 但不落板（防「板已切
//     命令没到」——落板只能由 commit 在命令组成功回调里做）；
//   - commit 落板并广播 onChange(旧,新)；same-mode commit 不广播（口径钉死：
//     防重复命令组重复刷 UI）；直接 commit（未经 request）也落板（黑板只管记账
//     不管流程，信任调用方命令已发成功）；
//   - 采集子态 capturing 原子读写、与全局态互相独立（S4/S5 真相源，10 不记账）；
//   - gate 为空 → request 直接 ok（无门禁场景/测试便利）；request 同值 mode 照样
//     过（黑板不拦，防死锁——已扫描中再 enterScan 由门禁拦）。
// 用例 = 实施计划 Task 9 九条。
// ============================================================================

#include <gtest/gtest.h>

#include "modules/08_devicemgmt/ModeController.h"

using namespace Scanner::device;
using Scanner::Result;

namespace {

Result gatePass(const std::string&) { return Result::ok(); }
Result gateReject(const std::string&) { return Result::fail(-7, "状态机忙"); }

} // namespace

// —— 1. DefaultIdle：默认 Idle + isCapturing=false ——
TEST(ModeController, DefaultIdle) {
    ModeController m(gatePass);
    EXPECT_EQ(m.mode(), DeviceMode::Idle);
    EXPECT_FALSE(m.isCapturing());
}

// —— 2. RequestGateReject：门禁拒 → request fail 原因透传、板不动 ——
TEST(ModeController, RequestGateReject) {
    ModeController m(gateReject);
    const Result r = m.request(DeviceMode::Scanning, "enter_scan");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.errorCode, -7);
    EXPECT_EQ(r.message, "状态机忙");               // 原因透传（拒因直达调用方）
    EXPECT_EQ(m.mode(), DeviceMode::Idle);          // 板不变
}

// —— 3. RequestGatePassNoCommit：门禁过 → ok 但不落板 ——
TEST(ModeController, RequestGatePassNoCommit) {
    ModeController m(gatePass);
    const Result r = m.request(DeviceMode::Scanning, "enter_scan");
    EXPECT_TRUE(r.success);
    EXPECT_EQ(m.mode(), DeviceMode::Idle);          // 过了也不落板（commit 才落）
}

// —— 4. CommitLandsAndBroadcasts：commit 落板 + onChange(旧,新) 恰一次 ——
TEST(ModeController, CommitLandsAndBroadcasts) {
    ModeController m(gatePass);
    int calls = 0;
    DeviceMode oldArg = DeviceMode::Scanning, newArg = DeviceMode::Idle;
    m.onChange = [&](DeviceMode o, DeviceMode n) { ++calls; oldArg = o; newArg = n; };
    m.commit(DeviceMode::Scanning);
    EXPECT_EQ(m.mode(), DeviceMode::Scanning);      // 落板
    EXPECT_EQ(calls, 1);                            // 恰一次
    EXPECT_EQ(oldArg, DeviceMode::Idle);            // 旧值
    EXPECT_EQ(newArg, DeviceMode::Scanning);        // 新值
}

// —— 5. CommitWithoutRequest：直接 commit 也落板（黑板不管流程只管记账）——
TEST(ModeController, CommitWithoutRequest) {
    ModeController m(nullptr);                      // 甚至无门禁
    int calls = 0;
    m.onChange = [&](DeviceMode, DeviceMode) { ++calls; };
    m.commit(DeviceMode::Calibrating);              // 未经 request
    EXPECT_EQ(m.mode(), DeviceMode::Calibrating);   // 落板
    EXPECT_EQ(calls, 1);                            // onChange 照发
}

// —— 6. CaptureSubstate：采集子态原子读写、与 mode 互相独立、幂等 ——
TEST(ModeController, CaptureSubstate) {
    ModeController m(gatePass);
    m.setCapturing(true);
    EXPECT_TRUE(m.isCapturing());
    EXPECT_EQ(m.mode(), DeviceMode::Idle);          // 子态不扰全局态
    m.commit(DeviceMode::Scanning);
    EXPECT_TRUE(m.isCapturing());                   // 改板不扰子态
    m.setCapturing(false);
    EXPECT_FALSE(m.isCapturing());
    m.setCapturing(false);                          // 幂等
    EXPECT_FALSE(m.isCapturing());
    EXPECT_EQ(m.mode(), DeviceMode::Scanning);
}

// —— 7. GateQueryNull：gate 为空 → request 直接 ok（仍不落板）——
TEST(ModeController, GateQueryNull) {
    ModeController m(nullptr);
    const Result r = m.request(DeviceMode::Scanning, "enter_scan");
    EXPECT_TRUE(r.success);
    EXPECT_EQ(m.mode(), DeviceMode::Idle);          // 依旧不落板
}

// —— 8. SameModeCommitNoBroadcast（口径钉死）：commit 当前值 → 不广播 ——
TEST(ModeController, SameModeCommitNoBroadcast) {
    ModeController m(gatePass);
    int calls = 0;
    m.onChange = [&](DeviceMode, DeviceMode) { ++calls; };
    m.commit(DeviceMode::Idle);                     // 与当前同值
    EXPECT_EQ(m.mode(), DeviceMode::Idle);
    EXPECT_EQ(calls, 0);                            // 不广播（防重复命令组重复刷 UI）
}

// —— 9. RequestSameMode：request 当前 mode → ok 且门禁照问（黑板不拦防死锁）——
TEST(ModeController, RequestSameMode) {
    int gateCalls = 0;
    std::string seenOp;
    ModeController m([&](const std::string& op) {
        ++gateCalls;
        seenOp = op;
        return Result::ok();
    });
    m.commit(DeviceMode::Scanning);
    const Result r = m.request(DeviceMode::Scanning, "enter_scan");  // 同值
    EXPECT_TRUE(r.success);                         // 黑板不拦（拦重复是门禁的事）
    EXPECT_EQ(gateCalls, 1);                        // 门禁照问
    EXPECT_EQ(seenOp, "enter_scan");                // op 透传给门禁
    EXPECT_EQ(m.mode(), DeviceMode::Scanning);
}
