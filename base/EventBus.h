#pragma once
// ============================================================================
// base/EventBus.h — 事件总线（base 共享内核）
//
// 控制/通知通道，不携带载荷。Critical 通道用 publishSync。
// Phase 1：自 framework/infra/EventBus.h 迁入。
// ============================================================================

#include "base/types.h"
#include <functional>
#include <vector>
#include <mutex>
#include <atomic>

namespace Scanner::infra {

using SubscriberId = uint32_t;
using EventHandler = std::function<void(const Event&)>;

class EventBus {
public:
    EventBus() = default;
    ~EventBus() = default;

    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;

    Result publish(const Event& event);
    Result publishSync(const Event& event);
    SubscriberId subscribe(EventType type, EventHandler handler);
    SubscriberId subscribeAll(EventHandler handler);
    void unsubscribe(SubscriberId id);
    void clear();
    size_t getSubscriberCount() const;

private:
    struct Subscription {
        SubscriberId id;
        EventType type;
        EventHandler handler;
    };

    // 261017 嵌套发布崩溃根因修复：原 std::mutex（非递归）——订阅者回调内再
    // publish（如 p1=2000→pause()→syncState 故障上报）＝同线程二次加锁 UB，
    // MSVC 抛 system_error→未捕获→terminate→abort（真机 18:15 崩溃实证）。
    // recursive_mutex：同线程重入合法（嵌套事件内联分发），跨线程仍串行；
    // 原「锁内不得 transition（重入死锁）」限制随之解除（QTimer 延后兜底兼容不撤）
    mutable std::recursive_mutex mutex_;
    std::vector<Subscription> subscribers_;
    std::atomic<SubscriberId> nextId_{1};
};

} // namespace Scanner::infra
