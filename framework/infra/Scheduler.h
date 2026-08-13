#pragma once
// ============================================================================
// Scheduler.h — 线程调度器（infra 层）
//
// 线程池 / CPU 亲和 / CUDA Stream 管理。
// Workflow 层消费 Scheduler 的线程资源执行 Stage。
// ============================================================================

#include "common/types.h"
#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Scanner::infra {

using Task = std::function<void()>;

class Scheduler {
public:
    explicit Scheduler(size_t threadCount = std::thread::hardware_concurrency());
    ~Scheduler();

    // 禁止拷贝/移动
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// 提交任务到线程池
    Result submit(Task task);

    /// 提交优先级任务
    Result submitPriority(Task task, int priority = 0);

    /// 等待所有任务完成
    void waitAll();

    /// 获取线程数
    size_t getThreadCount() const;

    /// 获取待执行任务数
    size_t getPendingCount() const;

    /// 停止（等待正在执行的任务完成）
    void shutdown();

    /// 是否已停止
    bool isShutdown() const;

private:
    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> activeTasks_{0};
};

} // namespace Scanner::infra
