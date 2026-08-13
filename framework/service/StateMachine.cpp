#include "StateMachine.h"
#include <spdlog/spdlog.h>

namespace Scanner::service {

StateMachine::StateMachine(infra::EventBus* bus) : eventBus_(bus) {}

ScannerState StateMachine::getCurrentState() const {
    return state_.load(std::memory_order_acquire);
}

std::string StateMachine::getStateName() const {
    return stateToString(state_.load());
}

std::string StateMachine::stateToString(ScannerState s) {
    switch (s) {
        case ScannerState::Init:           return "Init";
        case ScannerState::DeviceReady:    return "DeviceReady";
        case ScannerState::Calibrating:    return "Calibrating";
        case ScannerState::Calibrated:     return "Calibrated";
        case ScannerState::Scanning:       return "Scanning";
        case ScannerState::Paused:         return "Paused";
        case ScannerState::PostProcessing: return "PostProcessing";
        case ScannerState::Error:          return "Error";
        case ScannerState::EmergencyStop:  return "EmergencyStop";
    }
    return "Unknown";
}

bool StateMachine::isValidTransition(ScannerState from, EventType event, ScannerState& to) {
    switch (event) {
        case EventType::DeviceConnected:
            if (from == ScannerState::Init) { to = ScannerState::DeviceReady; return true; }
            break;
        case EventType::DeviceDisconnected:
            if (from != ScannerState::Init) { to = ScannerState::Init; return true; }
            break;
        case EventType::ScanStarted:
            if (from == ScannerState::Calibrated || from == ScannerState::DeviceReady) {
                to = ScannerState::Scanning; return true;
            }
            break;
        case EventType::ScanStopped:
            if (from == ScannerState::Scanning) { to = ScannerState::Calibrated; return true; }
            break;
        case EventType::ScanPaused:
            if (from == ScannerState::Scanning) { to = ScannerState::Paused; return true; }
            break;
        case EventType::FaultOccurred:
            to = ScannerState::Error; return true;
        case EventType::FaultCleared:
            if (from == ScannerState::Error) { to = ScannerState::DeviceReady; return true; }
            break;
        case EventType::EmergencyStop:
            to = ScannerState::EmergencyStop; return true;
        case EventType::SessionStarted:
            if (from == ScannerState::Calibrating) { to = ScannerState::Calibrated; return true; }
            break;
        default:
            break;
    }
    return false;
}

Result StateMachine::transition(EventType event, int64_t param) {
    ScannerState current = state_.load(std::memory_order_acquire);
    ScannerState next;
    if (!isValidTransition(current, event, next)) {
        spdlog::warn("[StateMachine] 非法转换: {} + {}", stateToString(current),
                     static_cast<int>(event));
        return Result::fail("非法状态转换");
    }

    state_.store(next, std::memory_order_release);
    notifyChange(current, next);

    spdlog::info("[StateMachine] {} → {}", stateToString(current), stateToString(next));

    if (eventBus_) {
        Event evt;
        evt.type = EventType::StateChanged;
        evt.param1 = static_cast<int64_t>(current);
        evt.param2 = static_cast<int64_t>(next);
        eventBus_->publish(evt);
    }

    return Result::ok();
}

bool StateMachine::canOperate(const std::string& operation) const {
    auto s = state_.load();
    if (operation == "scan")       return s == ScannerState::Calibrated || s == ScannerState::DeviceReady;
    if (operation == "calibrate")  return s == ScannerState::DeviceReady;
    if (operation == "postprocess") return s == ScannerState::Calibrated;
    if (operation == "connect")    return s == ScannerState::Init || s == ScannerState::Error;
    return false;
}

void StateMachine::onStateChange(StateChangeCallback cb) {
    std::lock_guard lock(cbMutex_);
    callback_ = std::move(cb);
}

void StateMachine::notifyChange(ScannerState oldState, ScannerState newState) {
    std::lock_guard lock(cbMutex_);
    if (callback_) callback_(oldState, newState);
}

} // namespace Scanner::service
