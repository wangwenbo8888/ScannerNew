// PipelineEventSink.cpp — EventBusEventSink 实现（Event 映射契约见头文件注）
#include "pipelines/PipelineEventSink.h"

#include "base/EventBus.h"

#include <spdlog/spdlog.h>

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

    // EventBus 为控制通道不携带载荷，msg 落 spdlog 留存（防静默丢失）
    if (severity == Scanner::FaultSeverity::Error) {
        spdlog::error("[pipemgmt] code={} msg={}", code, msg);
    } else {
        spdlog::warn("[pipemgmt] code={} msg={}", code, msg);
    }
    bus_->publish(e);
}

} // namespace Scanner::pipeline
