// ============================================================================
// SchedulerRuntime.cpp — 非模板成员实现（生命周期与统计）
// 模板成员（start/laneLoop）见 SchedulerRuntime.h。
// 本翻译单元兼作 STATIC 库编译锚点。
// ============================================================================
#include "sched/SchedulerRuntime.h"

#include <spdlog/spdlog.h>

#include <future>

namespace Scanner::pipeline::sched {

namespace {
int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

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
    drainAndShutdown(std::chrono::milliseconds(-1));   // 不限时（历史行为：阻塞 join）
}

void SchedulerRuntime::drainAndShutdown(std::chrono::milliseconds laneJoinTimeout) {
    std::lock_guard<std::mutex> lifecycleLock(lifecycleMutex_);
    stopFlag_.store(true);                       // 幂等：未 requestStop 直接 drain 也安全
    watchdogStop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();

    bool joinedInTime = true;
    if (laneJoinTimeout <= std::chrono::milliseconds(0)) {
        for (auto& t : lanes_) {
            if (t.joinable()) t.join();          // lane 跑完在飞帧后退出（排空在飞）
        }
    } else {
        // 限时 join：joiner 线程顺序 join，主线程等限定时——超时 detach 余 lane
        // （僵尸泄漏：正常路径不达；hang 恢复的兜底语义，注释见头文件）
        std::promise<void> joined;
        std::future<void> done = joined.get_future();
        std::thread joiner([this, p = std::move(joined)]() mutable {
            for (auto& t : lanes_) {
                if (t.joinable()) t.join();
            }
            p.set_value();
        });
        if (done.wait_for(laneJoinTimeout) != std::future_status::ready) {
            joinedInTime = false;
            spdlog::error("[SchedulerRuntime] lane join 超时（{}ms）——detach 余 lane 僵尸，"
                          "建议进程级重启收尾", static_cast<long>(laneJoinTimeout.count()));
        }
        joiner.join();                           // joiner 必返（超时后它仍在 join，等它收完
                                                  // 已退 lane；卡死 lane 的 join 由 detach 兜底）
        if (!joinedInTime) {
            for (auto& t : lanes_) {
                if (t.joinable()) t.detach();    // 僵尸：泄漏线程，对象生命周期须覆盖
            }
        }
    }
    lanes_.clear();
    broker_.shutdown();                          // 排空任务队列后 join workers
    if (gpu_) {
        gpu_->shutdown();                        // 逆序收摊：lanes → broker → gpu
        gpu_.reset();                            // 一次性服务销毁（下次 start 重建）
    }
    running_.store(false);
}

void SchedulerRuntime::watchdogLoop(int hangTimeoutMs) {
    constexpr auto kCheckInterval = std::chrono::milliseconds(250);
    while (!watchdogStop_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kCheckInterval);
        if (watchdogStop_.load(std::memory_order_acquire)) break;
        if (hangLatched_.load(std::memory_order_acquire)) continue;   // 一次性纪律
        if (activeLanes_.load() == 0) continue;                        // 未跑/全退：无心跳可查
        const int64_t now = nowMs();
        for (size_t i = 0; i < laneBeats_.size(); ++i) {
            const int64_t stale = now - laneBeats_[i]->load(std::memory_order_acquire);
            if (stale > static_cast<int64_t>(hangTimeoutMs)) {
                bool exp = false;
                if (!hangLatched_.compare_exchange_strong(exp, true)) break;
                hangDetected_.store(true, std::memory_order_release);
                spdlog::error("[SchedulerRuntime] lane {} 心跳静默 {}ms（阈 {}ms）——requestStop",
                              i, stale, hangTimeoutMs);
                requestStop();                   // 可见 stopFlag 的 lane 自行退出；死循环算子须限时 drain
                if (onHang) onHang(static_cast<int>(i), stale);
                break;
            }
        }
    }
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

uint64_t SchedulerRuntime::lastCounter() const {
    return lastCounter_.load();
}

bool SchedulerRuntime::lanesExited() const {
    return activeLanes_.load() == 0;
}

} // namespace Scanner::pipeline::sched
