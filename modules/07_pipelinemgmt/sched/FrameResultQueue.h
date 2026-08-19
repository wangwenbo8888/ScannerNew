#pragma once
// ============================================================================
// FrameResultQueue.h — 流水线结果输出队列（调度底座）
// MPMC：多生产者多消费者；满时覆盖最旧并计数，生产侧永不阻塞。
// ============================================================================
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace Scanner::pipeline::sched {

template<typename T>
class FrameResultQueue {
public:
    /// capacity==0 视为 1（防除零/永久空）
    explicit FrameResultQueue(size_t capacity)
        : capacity_(capacity == 0 ? 1 : capacity) {}

    /// 推入元素；队列满时覆盖最旧并计入 dropped（永不阻塞）
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.size() >= capacity_) {
            buffer_.pop_front();
            ++dropped_;
        }
        buffer_.push_back(std::move(item));
        notEmpty_.notify_one();
    }

    /// 弹出元素；超时未取到返回 nullopt
    std::optional<T> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!notEmpty_.wait_for(lock, timeout, [this] { return !buffer_.empty(); })) {
            return std::nullopt;
        }
        T item = std::move(buffer_.front());
        buffer_.pop_front();
        return item;
    }

    /// 因满被覆盖丢弃的元素总数
    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    /// 当前积压元素数
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::deque<T> buffer_;
    size_t dropped_{0};
};

} // namespace Scanner::pipeline::sched
