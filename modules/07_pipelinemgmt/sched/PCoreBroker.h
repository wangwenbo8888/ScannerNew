#pragma once
// ============================================================================
// PCoreBroker.h — P 核任务代理（调度底座）
// X-1 个 worker 线程池（各可绑 1 个 P 核 + Above Normal 优先级），
// 接收 E 核提交的 CPU 重活任务并返回 future；先到先得；
// shutdown 置停后排空队列再 join（排空语义）。
// ============================================================================
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "base/types.h"

namespace Scanner::pipeline::sched {

using PTask = std::function<Result()>;

class PCoreBroker {
public:
    /// workers: 线程数；coreMasks: 每线程亲和掩码，空 vector 或元素 0 = 不绑核。
    /// 已在运行时再次 start 返回 fail。失败不改变当前状态。
    Result start(int workers, const std::vector<uint64_t>& coreMasks);

    /// 提交任务（先到先得）；返回 packaged_task 的 future，任务内异常经 future 传递。
    /// 未 start 或已 shutdown 时抛 std::logic_error。
    /// 队列满（容量 64）时阻塞提交方（E 核提交阻塞可接受）。
    std::future<Result> submit(PTask task);

    /// 置停 + 排空队列 + join；幂等，可重复调用。
    void shutdown();

    bool isRunning() const;

    /// 未 shutdown 则析构时自动 shutdown（防 join 悬挂）
    ~PCoreBroker();

    PCoreBroker() = default;
    PCoreBroker(const PCoreBroker&) = delete;
    PCoreBroker& operator=(const PCoreBroker&) = delete;

private:
    void workerLoop();

    /// 有界任务队列容量（Block 策略）
    static constexpr size_t kQueueCapacity = 64;

    mutable std::mutex mutex_;
    std::condition_variable notEmpty_;
    std::condition_variable notFull_;
    std::deque<std::shared_ptr<std::packaged_task<Result()>>> queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
};

} // namespace Scanner::pipeline::sched
