// PipelineEventSink.cpp — EventBusEventSink 实现（Event 映射契约见头文件注）
#include "pipelines/PipelineEventSink.h"

#include "base/EventBus.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"

namespace Scanner::pipeline {

EventBusEventSink::EventBusEventSink(Scanner::infra::EventBus* bus) : bus_(bus) {}

void EventBusEventSink::report(Scanner::QualityFlag q, int32_t code,
                               const std::string& msg) {
    if (!bus_) return;
    if (q == Scanner::QualityFlag::Normal) return;   // 帧级 Normal 静默（原契约）
    publish(q, code, msg);
}

void EventBusEventSink::reportCompletion(Scanner::QualityFlag q, int32_t code,
                                         const std::string& msg) {
    if (!bus_) return;
    publish(q, code, msg);                           // Normal→Info 也发（完成事件）
}

void EventBusEventSink::publish(Scanner::QualityFlag q, int32_t code,
                                const std::string& msg) {
    Scanner::FaultSeverity severity;
    switch (q) {
    case Scanner::QualityFlag::Degraded:
    case Scanner::QualityFlag::Warning:
        severity = Scanner::FaultSeverity::Warning;
        break;
    case Scanner::QualityFlag::Fault:
        severity = Scanner::FaultSeverity::Error;
        break;
    case Scanner::QualityFlag::Normal:
    default:
        severity = Scanner::FaultSeverity::Info;
        break;
    }

    Scanner::Event e;
    e.type = Scanner::EventType::FaultOccurred;
    e.sourceId = kPipelineEventSourceId;
    e.param1 = code;
    e.param2 = static_cast<int64_t>(severity);

    // EventBus 为控制通道不携带载荷，msg 落 spdlog 留存（防静默丢失）
    if (severity == Scanner::FaultSeverity::Error) {
        JMW_LOG_ERROR("07-EventSink", "[pipemgmt] code={} msg={}", code, msg);
    } else if (severity == Scanner::FaultSeverity::Info) {
        JMW_LOG_INFO("07-EventSink", "[pipemgmt] code={} msg={}", code, msg);
    } else {
        JMW_LOG_WARN("07-EventSink", "[pipemgmt] code={} msg={}", code, msg);
    }
    bus_->publish(e);
}

} // namespace Scanner::pipeline
