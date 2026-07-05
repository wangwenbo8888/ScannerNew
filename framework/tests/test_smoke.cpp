// smoke 测试：验证骨架各层契约可 include + 关键语义
// 骨架阶段无业务逻辑，测试聚焦"契约形状"（编译期 + 简单运行期断言）
#include <gtest/gtest.h>
#include <type_traits>

#include "framework/common/quality_flag.h"
#include "framework/common/types.h"
#include "framework/ui/IView.h"
#include "framework/workflow/IWorkflow.h"
#include "framework/service/IService.h"
#include "framework/algorithm/operator_convention.h"
#include "framework/data/IDataStore.h"
#include "framework/hal/ICamera.h"
#include "framework/infra/EventBus.h"
#include "framework/crosscut/IAuth.h"
#include "sdk/IScannerSDK.h"

// ── common ────────────────────────────────────────────────
TEST(Common, QualityFlagValues) {
    EXPECT_EQ(static_cast<int>(Scanner::QualityFlag::Normal),   0);
    EXPECT_EQ(static_cast<int>(Scanner::QualityFlag::Degraded), 1);
    EXPECT_EQ(static_cast<int>(Scanner::QualityFlag::Warning),  2);
    EXPECT_EQ(static_cast<int>(Scanner::QualityFlag::Fault),    3);
}

TEST(Common, FrameDefaults) {
    Scanner::Frame f;
    EXPECT_EQ(f.frameId, 0u);
    EXPECT_EQ(f.timestamp, 0u);
    EXPECT_DOUBLE_EQ(f.temperature, 25.0);
    EXPECT_EQ(f.leftGray, nullptr);
    EXPECT_EQ(f.rightGray, nullptr);
}

// ── workflow ──────────────────────────────────────────────
TEST(Workflow, StatusEnum) {
    EXPECT_NE(Scanner::workflow::WorkflowStatus::Idle,
              Scanner::workflow::WorkflowStatus::Running);
}

TEST(Workflow, IWorkflowIsAbstract) {
    // IWorkflow 是纯虚接口，不可实例化
    EXPECT_TRUE(std::is_abstract_v<Scanner::workflow::IWorkflow>);
}

// ── algorithm（G5：纯函数约定，无 IOperator 基类）─────────
TEST(Algorithm, RegistryTemplateInstantiates) {
    // G5 决策：算子无基类派生；AlgorithmRegistry 是模板，可实例化任意类型
    struct DummyOp { int execute() { return 42; } };
    Scanner::algorithm::AlgorithmRegistry<DummyOp> reg;
    DummyOp op;
    reg.registerOperator("dummy", &op);  // 不抛即通过
    SUCCEED();
}

// ── data / hal / service / ui / infra / crosscut / sdk ──
// 这些层桩是空接口/类，编译通过（上面的 #include 已证）。
// 这里各加一个最小断言证明类型可被命名（链接期存在）。
TEST(Data, SinkTypesExist) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::data::IFrameSink>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::data::IDataStore>);
}

TEST(Hal, InterfacesAreAbstract) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::hal::ICamera>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::hal::IMCU>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::hal::IPlatform>);
}

TEST(Service, IServiceIsAbstract) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::service::IService>);
}

TEST(Ui, IViewIsAbstract) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::ui::IView>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::ui::IUIController>);
}

TEST(Crosscut, InterfacesAreAbstract) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::crosscut::IAuth>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::crosscut::ILogger>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::crosscut::IPerfMonitor>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::crosscut::ICrashHandler>);
    EXPECT_TRUE(std::is_abstract_v<Scanner::crosscut::IConfig>);
}

TEST(Sdk, IScannerSDKIsAbstract) {
    EXPECT_TRUE(std::is_abstract_v<Scanner::sdk::IScannerSDK>);
}
