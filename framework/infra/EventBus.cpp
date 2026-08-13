#include "EventBus.h"
#include <algorithm>

namespace Scanner::infra {

Result EventBus::publish(const Event& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& sub : subscribers_) {
        if (sub.type == EventType::UserDefined || sub.type == event.type) {
            sub.handler(event);
        }
    }
    return Result::ok();
}

Result EventBus::publishSync(const Event& event) {
    return publish(event);
}

SubscriberId EventBus::subscribe(EventType type, EventHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    SubscriberId id = nextId_++;
    subscribers_.push_back({id, type, std::move(handler)});
    return id;
}

SubscriberId EventBus::subscribeAll(EventHandler handler) {
    return subscribe(EventType::UserDefined, std::move(handler));
}

void EventBus::unsubscribe(SubscriberId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.erase(
        std::remove_if(subscribers_.begin(), subscribers_.end(),
            [id](const Subscription& s) { return s.id == id; }),
        subscribers_.end());
}

void EventBus::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.clear();
}

size_t EventBus::getSubscriberCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return subscribers_.size();
}

} // namespace Scanner::infra
