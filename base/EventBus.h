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

    mutable std::mutex mutex_;
    std::vector<Subscription> subscribers_;
    std::atomic<SubscriberId> nextId_{1};
};

} // namespace Scanner::infra
