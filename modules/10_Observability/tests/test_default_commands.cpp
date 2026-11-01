// ============================================================================
// test_default_commands.cpp — 首批命令目录注册表单测（P1-T6）
//
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §3.2 命令目录表
//   · 7 命令全部过 registerCommand 组合校验（含收尾禁用型 system_ready）
//   · submit 逐条切态验证：S1→S2 / S2→S3 / S2→S4|S5（payload 0/1）/ S2→S6
//   · 触发型（started=0）submit 不切态；非法态被三态门禁键拦
//     （scanning/calibrating/postprocessing 仅各自工作态放行）
// ============================================================================
#include <gtest/gtest.h>
#include "DefaultCommands.h"
#include "StateMachine.h"
#include "base/EventBus.h"

using Scanner::service::CommandGate;
using Scanner::service::StateMachine;
using Scanner::service::SystemState;
using Scanner::service::makeDefaultCommandSpecs;
using Scanner::infra::EventBus;

namespace {

// 全量注册 7 命令（每用例新 SM+Gate，注册表隔离）
void registerAll(CommandGate& gate) {
    for (const auto& spec : makeDefaultCommandSpecs()) {
        ASSERT_TRUE(gate.registerCommand(spec).success) << spec.name;
    }
}

// 经目录内 system_ready 驱动到 S2（顺带验证内部命令可用）
void driveToStandby(CommandGate& gate, StateMachine& sm) {
    ASSERT_TRUE(gate.submit("system_ready").success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Standby);
}

} // namespace

TEST(DefaultCommands, AllSevenRegisterOk) {
    StateMachine sm;
    CommandGate gate(&sm);
    int n = 0;
    for (const auto& spec : makeDefaultCommandSpecs()) {
        EXPECT_TRUE(gate.registerCommand(spec).success) << spec.name;
        ++n;
    }
    EXPECT_EQ(n, 7);
}

TEST(DefaultCommands, SystemReadySwitchesInitToStandby) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Init);
    EXPECT_TRUE(gate.submit("system_ready").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
    // 收尾禁用型：无 finishedEvent，notifyCompleted fail（态不因误调被破坏）
    EXPECT_FALSE(gate.notifyCompleted("system_ready", true).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(DefaultCommands, StartCalibrationSwitchesToCalibrating) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    EXPECT_TRUE(gate.submit("start_calibration").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
}

TEST(DefaultCommands, StartScanPayloadSelectsScanState) {
    {   // payload 0 → S4（ScanMarker）
        StateMachine sm;
        CommandGate gate(&sm);
        registerAll(gate);
        driveToStandby(gate, sm);
        EXPECT_TRUE(gate.submit("start_scan", 0).success);
        EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
    }
    {   // payload 1 → S5（ScanMarkerLaser）
        StateMachine sm;
        CommandGate gate(&sm);
        registerAll(gate);
        driveToStandby(gate, sm);
        EXPECT_TRUE(gate.submit("start_scan", 1).success);
        EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarkerLaser);
    }
}

TEST(DefaultCommands, StartPostprocessSwitchesToPostProcessing) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    EXPECT_TRUE(gate.submit("start_postprocess").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::PostProcessing);
}

TEST(DefaultCommands, FinishScanTriggerKeepsStateUntilNotify) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    ASSERT_TRUE(gate.submit("start_scan", 0).success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
    // 触发型 submit 只点火，终态仍 S4
    EXPECT_TRUE(gate.submit("finish_scan").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::ScanMarker);
    // ⑩ 工作流以起始命令名收尾 → S2
    EXPECT_TRUE(gate.notifyCompleted("start_scan", true).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(DefaultCommands, FinishCalibrationTriggerThenCompleteByStartName) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    ASSERT_TRUE(gate.submit("start_calibration").success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::Calibrating);
    // 触发型 submit 只点火，态仍 S3
    EXPECT_TRUE(gate.submit("finish_calibration").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
    // 工作流以起始命令名收尾 → S2
    EXPECT_TRUE(gate.notifyCompleted("start_calibration", true).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(DefaultCommands, PostprocessCompletesBackToStandby) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    ASSERT_TRUE(gate.submit("start_postprocess").success);
    ASSERT_EQ(sm.getCurrentState(), SystemState::PostProcessing);
    // S6 下以起始命令名收尾 → S2
    EXPECT_TRUE(gate.notifyCompleted("start_postprocess", true).success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(DefaultCommands, FinishScanDeniedInStandby) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    // S2 下 finish_scan：gateOp="scanning" 仅 S4/S5 放行
    EXPECT_FALSE(gate.submit("finish_scan").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}

TEST(DefaultCommands, FinishPostprocessDeniedInCalibrating) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    ASSERT_TRUE(gate.submit("start_calibration").success);
    // S3 下 finish_postprocess：gateOp="postprocessing" 仅 S6 放行
    EXPECT_FALSE(gate.submit("finish_postprocess").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Calibrating);
}

TEST(DefaultCommands, FinishCalibrationDeniedInStandby) {
    StateMachine sm;
    CommandGate gate(&sm);
    registerAll(gate);
    driveToStandby(gate, sm);
    // S2 下 finish_calibration：gateOp="calibrating" 仅 S3 放行
    EXPECT_FALSE(gate.submit("finish_calibration").success);
    EXPECT_EQ(sm.getCurrentState(), SystemState::Standby);
}
