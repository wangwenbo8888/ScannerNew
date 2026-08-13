#pragma once
// ============================================================================
// FaultHandler.h — 故障处理器（Service 层）
//
// 订阅 EventBus 故障事件 → 主动响应 → 状态机转 Error → 发布指示灯控制
// ============================================================================

#include "base/types.h"
#include <functional>

namespace Scanner::infra { class EventBus; }
namespace Scanner::service { class StateMachine; }

namespace Scanner::service {

class FaultHandler {
public:
    FaultHandler();
    ~FaultHandler();

    void setEventBus(infra::EventBus* bus) { eventBus_ = bus; }
    void setStateMachine(StateMachine* sm) { stateMachine_ = sm; }

    void start();
    void stop();

    // 手动上报故障
    void reportFault(const std::string& source, FaultSeverity severity, const std::string& message);
    void clearFault(const std::string& faultId);

    using FaultCallback = std::function<void(FaultSeverity, const std::string&)>;
    void onFault(FaultCallback cb) { faultCallback_ = std::move(cb); }

private:
    infra::EventBus* eventBus_ = nullptr;
    StateMachine* stateMachine_ = nullptr;
    FaultCallback faultCallback_;
    uint32_t subscriberId_ = 0;
    bool started_ = false;
};

} // namespace Scanner::service
