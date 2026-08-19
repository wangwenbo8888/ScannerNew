#include <gtest/gtest.h>
#include <atomic>
#include "pipelines/PipelineEventSink.h"
#include "base/EventBus.h"
using namespace Scanner::pipeline;
using Scanner::infra::EventBus;

TEST(EventSink, MapsQualityToSeverityAndPublishes) {
    EventBus bus;
    std::atomic<int> got{0};
    std::atomic<int> lastSeverity{-1};
    auto sub = bus.subscribe(Scanner::EventType::FaultOccurred, [&](const Scanner::Event& e) {
        lastSeverity = static_cast<int>(e.param2); ++got; });   // param2 携带 severity（见下契约）
    {
        EventBusEventSink sink(&bus);
        sink.report(Scanner::QualityFlag::Degraded, 1001, "帧降级");
        sink.report(Scanner::QualityFlag::Fault, 2002, "GPU 失败");
        sink.report(Scanner::QualityFlag::Normal, 0, "正常不打");
    }
    EXPECT_EQ(got.load(), 2);                       // Normal 不发布
    EXPECT_EQ(lastSeverity.load(), static_cast<int>(Scanner::FaultSeverity::Error));
    bus.unsubscribe(sub);
}
TEST(EventSink, NullEventBusSafe) {
    EventBusEventSink sink(nullptr);
    EXPECT_NO_FATAL_FAILURE(sink.report(Scanner::QualityFlag::Fault, 1, "x"));   // 不崩
}
TEST(EventSink, ProgressInfoPath) {
    EventBus bus;
    std::atomic<int> got{0};
    auto sub = bus.subscribe(Scanner::EventType::FaultOccurred, [&](const Scanner::Event&) { ++got; });
    EventBusEventSink sink(&bus);
    sink.report(Scanner::QualityFlag::Degraded, 3003, "配准降级光流→frame_fuse");
    EXPECT_EQ(got.load(), 1);
    bus.unsubscribe(sub);
}
