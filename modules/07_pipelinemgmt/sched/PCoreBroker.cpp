// ============================================================================
// PCoreBroker.cpp — P 核任务代理实现（Windows 绑核/提优先级）
// ============================================================================
#include "sched/PCoreBroker.h"

#include <windows.h>

namespace Scanner::pipeline::sched {

Result PCoreBroker::start(int workers, const std::vector<uint64_t>& coreMasks) {
    if (workers <= 0) {
        return Result::fail("PCoreBroker::start: workers must be > 0");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_.load()) {
        return Result::fail("PCoreBroker::start: already running");
    }
    stop_.store(false);
    running_.store(true);
    for (int i = 0; i < workers; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
        auto handle = workers_.back().native_handle();
        // mask 非 0 才绑核（空 vector 或元素 0 = 不绑，保可测性）
        const uint64_t mask = (static_cast<size_t>(i) < coreMasks.size()) ? coreMasks[i] : 0;
        if (mask != 0) {
            SetThreadAffinityMask(handle, static_cast<DWORD_PTR>(mask));
        }
        // P 核 worker 固定提一档优先级（后续 SchedConfig 接线再参数化）
        SetThreadPriority(handle, THREAD_PRIORITY_ABOVE_NORMAL);
    }
    return Result::ok();
}

std::future<Result> PCoreBroker::submit(PTask task) {
    if (!isRunning()) {
        throw std::logic_error("PCoreBroker::submit: broker not running");
    }
    // packaged_task 只移动 → 存 shared_ptr 入队
    auto sp = std::make_shared<std::packaged_task<Result()>>(std::move(task));
    auto fut = sp->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // 队列满阻塞在 notFull（Block 策略）；stop_ 兜底防 shutdown 后永久阻塞
        notFull_.wait(lock, [this] {
            return queue_.size() < kQueueCapacity || stop_.load();
        });
        if (queue_.size() < kQueueCapacity) {
            queue_.push_back(std::move(sp));
        }
        // stop_ 且队列仍满：放弃入队，sp 析构 → future.get() 抛 broken_promise
    }
    notEmpty_.notify_one();
    return fut;
}

void PCoreBroker::workerLoop() {
    while (true) {
        std::shared_ptr<std::packaged_task<Result()>> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            notEmpty_.wait(lock, [this] { return stop_.load() || !queue_.empty(); });
            if (queue_.empty()) {
                return;  // stop 且队列已排空 → 退出（排空语义）
            }
            task = std::move(queue_.front());
            queue_.pop_front();
            notFull_.notify_one();  // 唤醒阻塞在满队列上的提交方
        }
        (*task)();  // 任务内异常由 packaged_task 捕获，经 future 传递
    }
}

void PCoreBroker::shutdown() {
    std::vector<std::thread> toJoin;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_.load()) {
            return;  // 未 start 或已 shutdown：幂等
        }
        stop_.store(true);
        running_.store(false);
        toJoin = std::move(workers_);
        workers_.clear();
    }
    notEmpty_.notify_all();  // 唤醒 worker 继续排空；唤醒阻塞在 notFull 的提交方
    notFull_.notify_all();
    for (auto& t : toJoin) {
        if (t.joinable()) {
            t.join();
        }
    }
}

bool PCoreBroker::isRunning() const {
    return running_.load();
}

PCoreBroker::~PCoreBroker() {
    shutdown();
}

} // namespace Scanner::pipeline::sched
