#pragma once
// ============================================================================
// RingBuffer.h — 线程安全环形缓冲区（Data 层）
// ============================================================================

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>

namespace Scanner::data {

enum class OverflowPolicy {
    DropOldest,   // 丢弃最旧帧
    DropNewest,   // 丢弃最新帧（拒绝入队）
    Block         // 阻塞等待
};

template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity, OverflowPolicy policy = OverflowPolicy::DropOldest)
        : capacity_(capacity), policy_(policy) {}

    /// 推入元素（可能丢弃旧元素）
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (capacity_ == 0) return false;

        if (buffer_.size() >= capacity_) {
            switch (policy_) {
                case OverflowPolicy::DropOldest:
                    buffer_.pop();
                    break;
                case OverflowPolicy::DropNewest:
                    return false;
                case OverflowPolicy::Block:
                    notFull_.wait(lock, [this] { return buffer_.size() < capacity_; });
                    break;
            }
        }
        buffer_.push(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    /// 弹出元素（超时返回 nullopt）
    std::optional<T> pop(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!notEmpty_.wait_for(lock, timeout, [this] { return !buffer_.empty(); })) {
            return std::nullopt;
        }
        T item = std::move(buffer_.front());
        buffer_.pop();
        notFull_.notify_one();
        return item;
    }

    /// 当前大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    /// 是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    /// 清空
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!buffer_.empty()) buffer_.pop();
        notFull_.notify_all();
    }

private:
    size_t capacity_;
    OverflowPolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::queue<T> buffer_;
};

} // namespace Scanner::data
