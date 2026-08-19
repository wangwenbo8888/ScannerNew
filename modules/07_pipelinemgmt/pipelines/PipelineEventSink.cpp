// PipelineEventSink.cpp — EventBusEventSink 实现（Event 映射契约见头文件注）
#include "pipelines/PipelineEventSink.h"

#include "base/EventBus.h"

namespace Scanner::pipeline {

EventBusEventSink::EventBusEventSink(Scanner::infra::EventBus* bus) : bus_(bus) {}

void EventBusEventSink::report(Scanner::QualityFlag q, int32_t code, const std::string& msg) {
    if (!bus_) return;

    Scanner::FaultSeverity severity;
    switch (q) {
    case Scanner::QualityFlag::Degraded:
        severity = Scanner::FaultSeverity::Warning;
        break;
    case Scanner::QualityFlag::Warning:
        severity = Scanner::FaultSeverity::Warning;
        break;
    case Scanner::QualityFlag::Fault:
        severity = Scanner::FaultSeverity::Error;
        break;
    case Scanner::QualityFlag::Normal:
    default:
        return;                                  // Normal 不发布
    }

    Scanner::Event e;
    e.type = Scanner::EventType::FaultOccurred;
    e.sourceId = kPipelineEventSourceId;
    e.param1 = code;
    e.param2 = static_cast<int64_t>(severity);
    bus_->publish(e);
    (void)msg;                                    // EventBus 控制通道不携带载荷，msg 仅作调用侧日志语义
}

} // namespace Scanner::pipeline
