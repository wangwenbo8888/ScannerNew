#include <gtest/gtest.h>
#include "common/types.h"
#include "algorithm/operator_convention.h"
#include "data/IFrameSink.h"
#include "data/IDeviceStateSink.h"
#include "data/IPointCloudSink.h"
#include "data/RingBuffer.h"
#include "hal/IScannerCamera.h"
#include "hal/IMCU.h"
#include "infra/EventBus.h"
#include "infra/Scheduler.h"
#include "service/IState.h"
#include "workflow/IWorkflow.h"
#include "ui/IView.h"
#include "crosscut/IAuth.h"

using namespace Scanner;

// ============================================================================
// QualityFlag 枚举值测试
// ============================================================================
TEST(QualityFlagTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(QualityFlag::Normal), 0);
    EXPECT_EQ(static_cast<uint8_t>(QualityFlag::Degraded), 1);
    EXPECT_EQ(static_cast<uint8_t>(QualityFlag::Warning), 2);
    EXPECT_EQ(static_cast<uint8_t>(QualityFlag::Fault), 3);
}

// ============================================================================
// Result 类型测试
// ============================================================================
TEST(ResultTest, OkResult) {
    auto r = Result::ok("success");
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.errorCode, 0);
    EXPECT_EQ(r.message, "success");
    EXPECT_EQ(r.qualityFlag, QualityFlag::Normal);
}

TEST(ResultTest, FailResult) {
    auto r = Result::fail(42, "error");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.errorCode, 42);
    EXPECT_EQ(r.qualityFlag, QualityFlag::Fault);
}

TEST(ResultTest, DegradedResult) {
    auto r = Result::degraded("slow");
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(r.isDegraded());
}

// ============================================================================
// Frame 数据结构测试
// ============================================================================
TEST(FrameTest, DefaultConstruction) {
    data::FrameData frame;
    EXPECT_EQ(frame.frameId, 0u);
    EXPECT_EQ(frame.timestamp, 0u);
    EXPECT_TRUE(frame.leftGray.empty());
    EXPECT_TRUE(frame.rightGray.empty());
}

// ============================================================================
// WorkflowStatus 枚举值测试
// ============================================================================
TEST(WorkflowStatusTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Idle), 0);
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Running), 1);
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Paused), 2);
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Stopping), 3);
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Completed), 4);
    EXPECT_EQ(static_cast<uint8_t>(workflow::WorkflowState::Error), 5);
}

// ============================================================================
// ScannerState 枚举值测试
// ============================================================================
TEST(ScannerStateTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(service::ScannerState::Init), 0);
    EXPECT_EQ(static_cast<uint8_t>(service::ScannerState::DeviceReady), 1);
    EXPECT_EQ(static_cast<uint8_t>(service::ScannerState::Scanning), 4);
    EXPECT_EQ(static_cast<uint8_t>(service::ScannerState::EmergencyStop), 8);
}

// ============================================================================
// 接口抽象性测试（确保所有接口是纯虚的）
// ============================================================================
TEST(InterfaceTest, IFrameSinkIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<data::IFrameSink>));
}

TEST(InterfaceTest, IDeviceStateSinkIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<data::IDeviceStateSink>));
}

TEST(InterfaceTest, IPointCloudSinkIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<data::IPointCloudSink>));
}

TEST(InterfaceTest, IScannerCameraIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<hal::IScannerCamera>));
}

TEST(InterfaceTest, IMCUIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<hal::IMCU>));
}

TEST(InterfaceTest, IStateIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<service::IState>));
}

TEST(InterfaceTest, IWorkflowIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<workflow::IWorkflow>));
}

TEST(InterfaceTest, IViewIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<ui::IView>));
}

TEST(InterfaceTest, IAuthIsAbstract) {
    EXPECT_TRUE((std::is_abstract_v<crosscut::IAuth>));
}

// ============================================================================
// AlgorithmRegistry 测试
// ============================================================================
TEST(AlgorithmRegistryTest, RegisterAndRetrieve) {
    algorithm::AlgorithmRegistry<int()> registry;
    registry.registerOperator("test_op", []() { return 42; });
    EXPECT_TRUE(registry.hasOperator("test_op"));
    auto op = registry.getOperator("test_op");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op(), 42);
}

TEST(AlgorithmRegistryTest, UnknownOperatorReturnsNull) {
    algorithm::AlgorithmRegistry<int()> registry;
    EXPECT_FALSE(registry.hasOperator("nonexistent"));
    EXPECT_EQ(registry.getOperator("nonexistent"), nullptr);
}
