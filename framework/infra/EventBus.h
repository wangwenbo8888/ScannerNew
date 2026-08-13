#pragma once
// ============================================================================
// EventBus.h — 事件总线（infra 层）
//
// 控制/通知通道，不携带载荷（载荷走共享内存/直接调用）。
// Critical 通道用 publishSync（急停硬实时）。
// ============================================================================

#include "common/types.h"
#include <functional>
#include <vector>
#include <mutex>
#include <atomic>

namespace Scanner::infra {

// ============================================================================
// 订阅句柄
// ============================================================================
using SubscriberId = uint32_t;
using EventHandler = std::function<void(const Event&)>;

// ============================================================================
// EventBus（单例或注入）
// ============================================================================
class EventBus {
public:
    EventBus() = default;
    ~EventBus() = default;

    // 禁止拷贝/移动
    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;

    /// 普通发布（异步，排队）
    Result publish(const Event& event);

    /// 同步发布（Critical 通道，阻塞等所有订阅者执行完）
    Result publishSync(const Event& event);

    /// 订阅指定事件类型
    SubscriberId subscribe(EventType type, EventHandler handler);

    /// 订阅所有事件
    SubscriberId subscribeAll(EventHandler handler);

    /// 取消订阅
    void unsubscribe(SubscriberId id);

    /// 清空所有订阅
    void clear();

    /// 查询订阅数
    size_t getSubscriberCount() const;

private:
    struct Subscription {
        SubscriberId id;
        EventType type;         // 0 = all
        EventHandler handler;
    };

    mutable std::mutex mutex_;
    std::vector<Subscription> subscribers_;
    std::atomic<SubscriberId> nextId_{1};
};

} // namespace Scanner::infra
