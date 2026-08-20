#pragma once
// ============================================================================
// StateMachine.h — 全局门禁状态机实现（7 态 S1–S7，表驱动 + CAS 原子转态）
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §2/§4.4/§4.5
//   · S6 后处理免疫故障/断连；S7 出口唯 SelfCheckPassed→lastHealthy（续作语义）
// ============================================================================

#include "IState.h"
#include "base/EventBus.h"
#include <atomic>
#include <mutex>
#include <functional>

namespace Scanner::service {

class StateMachine : public IState {
public:
    explicit StateMachine(infra::EventBus* bus = nullptr);
    ~StateMachine() override = default;

    SystemState getCurrentState() const override;
    std::string getStateName() const override;
    Result transition(EventType event, int64_t param = 0) override;
    bool canOperate(const std::string& operation) const override;

    using StateChangeCallback = std::function<void(SystemState oldState, SystemState newState)>;
    void onStateChange(StateChangeCallback cb);

    static std::string stateToString(SystemState s);

private:
    std::atomic<SystemState> state_{SystemState::Init};
    std::atomic<SystemState> lastHealthy_{SystemState::Standby}; // 进 S7 前的健康态，恢复续作目标
    infra::EventBus* eventBus_ = nullptr;
    StateChangeCallback callback_;
    mutable std::mutex cbMutex_;

    bool resolveNext(SystemState from, EventType event, int64_t param, SystemState& to);
    void notifyChange(SystemState oldState, SystemState newState);
};

} // namespace Scanner::service
