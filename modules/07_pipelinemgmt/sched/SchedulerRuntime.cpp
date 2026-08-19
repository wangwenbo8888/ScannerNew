// ============================================================================
// SchedulerRuntime.cpp — 非模板成员实现（生命周期与统计）
// 模板成员（start/laneLoop）见 SchedulerRuntime.h。
// 本翻译单元兼作 STATIC 库编译锚点。
// ============================================================================
#include "sched/SchedulerRuntime.h"

namespace Scanner::pipeline::sched {

SchedulerRuntime::~SchedulerRuntime() {
    requestStop();
    drainAndShutdown();
}

Result SchedulerRuntime::setGpuStreamFactory(GpuSlotService::StreamFactory factory,
                                             GpuSlotService::StreamDestroyer destroyer) {
    if (running_.load()) {
        return Result::fail("SchedulerRuntime::setGpuStreamFactory: cannot change while running");
    }
    gpuFactory_ = std::move(factory);
    gpuDestroyer_ = std::move(destroyer);
    return Result::ok();
}

void SchedulerRuntime::requestStop() {
    stopFlag_.store(true);
}

void SchedulerRuntime::drainAndShutdown() {
    std::lock_guard<std::mutex> lifecycleLock(lifecycleMutex_);
    stopFlag_.store(true);                       // 幂等：未 requestStop 直接 drain 也安全
    for (auto& t : lanes_) {
        if (t.joinable()) t.join();              // lane 跑完在飞帧后退出（排空在飞）
    }
    lanes_.clear();
    broker_.shutdown();                          // 排空任务队列后 join workers
    if (gpu_) {
        gpu_->shutdown();                        // 逆序收摊：lanes → broker → gpu
        gpu_.reset();                            // 一次性服务销毁（下次 start 重建）
    }
    running_.store(false);
}

SchedulerRuntime::Stats SchedulerRuntime::stats() const {
    Stats s;
    s.processed = processed_.load();
    s.droppedSkips = droppedSkips_.load();
    s.gpuRejects = gpuRejects_.load();
    s.finalizeFails = finalizeFails_.load();
    return s;
}

bool SchedulerRuntime::isRunning() const {
    return running_.load();
}

} // namespace Scanner::pipeline::sched
