#include "Scheduler.h"

namespace Scanner::infra {

Scheduler::Scheduler(size_t threadCount) {
    for (size_t i = 0; i < threadCount; ++i) {
        workers_.emplace_back(&Scheduler::workerLoop, this);
    }
}

Scheduler::~Scheduler() {
    shutdown();
}

Result Scheduler::submit(Task task) {
    if (shutdown_) return Result::fail("Scheduler is shut down");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    condition_.notify_one();
    return Result::ok();
}

Result Scheduler::submitPriority(Task task, int) {
    return submit(std::move(task));
}

void Scheduler::waitAll() {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty() && activeTasks_ == 0) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

size_t Scheduler::getThreadCount() const {
    return workers_.size();
}

size_t Scheduler::getPendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tasks_.size();
}

void Scheduler::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    condition_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

bool Scheduler::isShutdown() const {
    return shutdown_;
}

void Scheduler::workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return shutdown_ || !tasks_.empty(); });
            if (shutdown_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        activeTasks_.fetch_add(1);
        task();
        activeTasks_.fetch_sub(1);
    }
}

} // namespace Scanner::infra
