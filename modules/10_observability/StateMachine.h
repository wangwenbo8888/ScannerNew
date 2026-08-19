#pragma once
// ============================================================================
// StateMachine.h — 系统状态机实现（Service 层）
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

    ScannerState getCurrentState() const override;
    std::string getStateName() const override;
    Result transition(EventType event, int64_t param = 0) override;
    bool canOperate(const std::string& operation) const override;

    using StateChangeCallback = std::function<void(ScannerState oldState, ScannerState newState)>;
    void onStateChange(StateChangeCallback cb);

    static std::string stateToString(ScannerState s);

private:
    std::atomic<ScannerState> state_{ScannerState::Init};
    infra::EventBus* eventBus_ = nullptr;
    StateChangeCallback callback_;
    mutable std::mutex cbMutex_;

    bool isValidTransition(ScannerState from, EventType event, ScannerState& to);
    void notifyChange(ScannerState oldState, ScannerState newState);
};

} // namespace Scanner::service
