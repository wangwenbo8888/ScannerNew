#include "FaultHandler.h"
#include "StateMachine.h"
#include "infra/EventBus.h"
#include <spdlog/spdlog.h>

namespace Scanner::service {

FaultHandler::FaultHandler() {}
FaultHandler::~FaultHandler() { stop(); }

void FaultHandler::start() {
    if (started_ || !eventBus_) return;
    subscriberId_ = eventBus_->subscribe(EventType::FaultOccurred,
        [this](const Event& evt) {
            FaultSeverity severity = static_cast<FaultSeverity>(evt.param1);
            std::string msg = "Fault " + std::to_string(evt.param2);

            spdlog::error("[FaultHandler] 故障: severity={} source={}", evt.param1, evt.param2);

            // 主动响应：状态机转 Error
            if (stateMachine_ && severity >= FaultSeverity::Error) {
                stateMachine_->transition(EventType::FaultOccurred);
            }

            // 发布指示灯控制（故障=红灯）
            if (eventBus_) {
                Event ledEvt;
                ledEvt.type = EventType::EmergencyStop;  // 复用急停通道触发红灯
                ledEvt.param1 = 0;  // red
                eventBus_->publish(ledEvt);
            }

            if (faultCallback_) faultCallback_(severity, msg);
        });
    started_ = true;
    spdlog::info("[FaultHandler] 已启动");
}

void FaultHandler::stop() {
    if (!started_ || !eventBus_) return;
    eventBus_->unsubscribe(subscriberId_);
    started_ = false;
}

void FaultHandler::reportFault(const std::string& source, FaultSeverity severity, const std::string& message) {
    spdlog::error("[FaultHandler] {} : {} ({})", source, message, static_cast<int>(severity));

    if (eventBus_) {
        Event evt;
        evt.type = EventType::FaultOccurred;
        evt.param1 = static_cast<int64_t>(severity);
        evt.param2 = 0;
        eventBus_->publish(evt);
    }

    if (stateMachine_ && severity >= FaultSeverity::Error) {
        stateMachine_->transition(EventType::FaultOccurred);
    }

    if (faultCallback_) faultCallback_(severity, message);
}

void FaultHandler::clearFault(const std::string& faultId) {
    spdlog::info("[FaultHandler] 故障已清除: {}", faultId);
    if (eventBus_) {
        Event evt;
        evt.type = EventType::FaultCleared;
        eventBus_->publish(evt);
    }
    if (stateMachine_) {
        stateMachine_->transition(EventType::FaultCleared);
    }
}

} // namespace Scanner::service
