#include "StateMachine.h"
#include <spdlog/spdlog.h>

namespace Scanner::service {

namespace {
// 转换表＝测试 FullMatrix 的 kLegal 期望表逐条对应（19 条合法边中 18 条入表，
// S7 行的 SelfCheckPassed 特判不入表，见 resolveNext）。
// 铁律：S6 无 Fault/Disconnect 边（免疫）；S7 无 Disconnect/DeviceConnected 边；
// ScanStarted 的 param 0/1 区分在 resolveNext 里判。
struct Edge { SystemState from; EventType ev; SystemState to; };

constexpr Edge kEdges[] = {
    {SystemState::Init,            EventType::SystemReady,         SystemState::Standby},
    {SystemState::Init,            EventType::FaultOccurred,       SystemState::FaultSelfCheck},
    {SystemState::Standby,         EventType::CalibStarted,        SystemState::Calibrating},
    {SystemState::Standby,         EventType::ScanStarted,         SystemState::ScanMarker},
    {SystemState::Standby,         EventType::ScanStarted,         SystemState::ScanMarkerLaser},
    {SystemState::Standby,         EventType::PostProcessStarted,  SystemState::PostProcessing},
    {SystemState::Standby,         EventType::FaultOccurred,       SystemState::FaultSelfCheck},
    {SystemState::Standby,         EventType::DeviceDisconnected,  SystemState::Init},
    {SystemState::Calibrating,     EventType::CalibFinished,       SystemState::Standby},
    {SystemState::Calibrating,     EventType::FaultOccurred,       SystemState::FaultSelfCheck},
    {SystemState::Calibrating,     EventType::DeviceDisconnected,  SystemState::Init},
    {SystemState::ScanMarker,      EventType::ScanStopped,         SystemState::Standby},
    {SystemState::ScanMarker,      EventType::FaultOccurred,       SystemState::FaultSelfCheck},
    {SystemState::ScanMarker,      EventType::DeviceDisconnected,  SystemState::Init},
    {SystemState::ScanMarkerLaser, EventType::ScanStopped,         SystemState::Standby},
    {SystemState::ScanMarkerLaser, EventType::FaultOccurred,       SystemState::FaultSelfCheck},
    {SystemState::ScanMarkerLaser, EventType::DeviceDisconnected,  SystemState::Init},
    {SystemState::PostProcessing,  EventType::PostProcessFinished, SystemState::Standby},
};
} // namespace

StateMachine::StateMachine(infra::EventBus* bus) : eventBus_(bus) {}

SystemState StateMachine::getCurrentState() const {
    return state_.load(std::memory_order_acquire);
}

std::string StateMachine::getStateName() const {
    return stateToString(state_.load());
}

std::string StateMachine::stateToString(SystemState s) {
    switch (s) {
        case SystemState::Init:            return "Init";
        case SystemState::Standby:         return "Standby";
        case SystemState::Calibrating:     return "Calibrating";
        case SystemState::ScanMarker:      return "ScanMarker";
        case SystemState::ScanMarkerLaser: return "ScanMarkerLaser";
        case SystemState::PostProcessing:  return "PostProcessing";
        case SystemState::FaultSelfCheck:  return "FaultSelfCheck";
    }
    return "Unknown";
}

bool StateMachine::resolveNext(SystemState from, EventType event, int64_t param, SystemState& to) {
    // S7 恢复特判：回 lastHealthy 续作（§4.4）；永不回 S6/S7（当前矩阵下不可达，防御分支）
    if (event == EventType::SelfCheckPassed) {
        if (from != SystemState::FaultSelfCheck) return false;
        to = lastHealthy_.load(std::memory_order_acquire);
        if (to == SystemState::FaultSelfCheck || to == SystemState::PostProcessing) {
            to = SystemState::Standby;
        }
        return true;
    }
    // ScanStarted param 0/1 区分（base ScanMode 契约仅 0/1，其余值无边）
    if (event == EventType::ScanStarted && param != 0 && param != 1) return false;

    for (const auto& e : kEdges) {
        if (e.from != from || e.ev != event) continue;
        if (event == EventType::ScanStarted) {
            const SystemState want = (param == 0) ? SystemState::ScanMarker
                                                  : SystemState::ScanMarkerLaser;
            if (e.to != want) continue;
        }
        to = e.to;
        return true;
    }
    return false;
}

Result StateMachine::transition(EventType event, int64_t param) {
    SystemState from = state_.load(std::memory_order_acquire);
    for (;;) {
        SystemState to;
        if (!resolveNext(from, event, param, to)) {
            spdlog::warn("[StateMachine] 非法转换: {} + event {}", stateToString(from),
                         static_cast<int>(event));
            return Result::fail("非法状态转换");
        }
        // lastHealthy 先写后 CAS：CAS 败者重试的残留 store 无害（lastHealthy 仅 S7 出口
        // 被读，真正入 S7 前的下一次 store 会覆盖），换取「入 S7 时 from 必已记录」无竞态
        if (to == SystemState::FaultSelfCheck && from != SystemState::FaultSelfCheck) {
            lastHealthy_.store(from, std::memory_order_release);
        }
        if (state_.compare_exchange_weak(from, to, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            notifyChange(from, to);
            if (eventBus_) {
                Event evt;
                evt.type = EventType::StateChanged;
                evt.param1 = static_cast<int64_t>(from);
                evt.param2 = static_cast<int64_t>(to);
                eventBus_->publish(evt);
            }
            spdlog::info("[StateMachine] {} → {}", stateToString(from), stateToString(to));
            return Result::ok();
        }
        // CAS 失败：from 已被 compare_exchange_weak 更新为最新态，重判
    }
}

bool StateMachine::canOperate(const std::string& operation) const {
    const auto s = state_.load(std::memory_order_acquire);
    if (operation == "calibrate" || operation == "scan" ||
        operation == "postprocess" || operation == "edit") {
        return s == SystemState::Standby;
    }
    return false;
}

void StateMachine::onStateChange(StateChangeCallback cb) {
    std::lock_guard lock(cbMutex_);
    callback_ = std::move(cb);
}

void StateMachine::notifyChange(SystemState oldState, SystemState newState) {
    StateChangeCallback cb;
    {
        std::lock_guard lock(cbMutex_);
        cb = callback_;
    }
    if (cb) cb(oldState, newState);
}

} // namespace Scanner::service
