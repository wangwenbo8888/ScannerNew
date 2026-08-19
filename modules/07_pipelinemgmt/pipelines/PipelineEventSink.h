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

    // 一次性完成事件（非帧级，每 run 至多一次）：Normal 也发布（Info）——
    // 区别于帧级 report() 对 Normal 静默（防逐帧刷屏）。缺省退化=report()
    // （Normal 静默，兼容旧实现）。
    virtual void reportCompletion(Scanner::QualityFlag q, int32_t code,
                                  const std::string& msg) {
        report(q, code, msg);
    }
};

class EventBusEventSink : public PipelineEventSink {
public:
    explicit EventBusEventSink(Scanner::infra::EventBus* bus);   // nullptr 安全（全 no-op）
    void report(Scanner::QualityFlag q, int32_t code, const std::string& msg) override;
    void reportCompletion(Scanner::QualityFlag q, int32_t code,
                          const std::string& msg) override;      // Normal→Info 也发
private:
    void publish(Scanner::QualityFlag q, int32_t code, const std::string& msg);
    Scanner::infra::EventBus* bus_;
    // 映射：Normal→不发布（report 路径）/Info（reportCompletion 路径）；
    //       Degraded→Warning；Warning→Warning；Fault→Error
    // Event{type=FaultOccurred, sourceId=kPipelineEventSourceId(0x07), param1=code, param2=severity}
};

} // namespace Scanner::pipeline
