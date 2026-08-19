// PipelineEventSink.h — 流水线事件上报（07 统一出口；默认实现走 base EventBus）
#pragma once

#include "base/types.h"
#include <cstdint>
#include <string>

namespace Scanner::infra { class EventBus; }

namespace Scanner::pipeline {

constexpr uint32_t kPipelineEventSourceId = 0x07;

class PipelineEventSink {                        // 抽象（测试/自定义可替）
public:
    virtual ~PipelineEventSink() = default;
    virtual void report(Scanner::QualityFlag q, int32_t code, const std::string& msg) = 0;
};

class EventBusEventSink : public PipelineEventSink {
public:
    explicit EventBusEventSink(Scanner::infra::EventBus* bus);   // nullptr 安全（全 no-op）
    void report(Scanner::QualityFlag q, int32_t code, const std::string& msg) override;
private:
    Scanner::infra::EventBus* bus_;
    // 映射：Normal→不发布；Degraded→FaultSeverity::Warning；Warning→Warning；Fault→Error
    // Event{type=FaultOccurred, sourceId=kPipelineEventSourceId(0x07), param1=code, param2=severity}
};

} // namespace Scanner::pipeline
