// ============================================================================
// GpuSlotService.cpp — cudaStream 槽池实现（空闲池承担计数信号量 + RAII 归还）
// ============================================================================
#include "sched/GpuSlotService.h"

namespace Scanner::pipeline::sched {

Result GpuSlotService::start(int slots, StreamFactory factory, StreamDestroyer destroyer) {
    if (slots <= 0) {
        return Result::fail("GpuSlotService::start: slots must be > 0");
    }
#ifdef JMW_BUILD_CUDA
    if (!factory)   factory   = [](StreamHandle* s) { return static_cast<int>(cudaStreamCreate(s)); };
    if (!destroyer) destroyer = [](StreamHandle s) { cudaStreamDestroy(s); };
#else
    // CUDA 关闭：无默认真流实现，必须注入工厂（假工厂测试不受影响）
    if (!factory || !destroyer) {
        return Result::fail("GpuSlotService::start: factory/destroyer required when CUDA disabled");
    }
#endif
    // 流创建全程在锁外（M1+M2：工厂可能慢/抛异常，锁内调用会阻塞 acquire/shutdown；
    // 半成品只存在于局部 vector，任一流失败/异常 → 锁外销毁已建部分返回 fail，
    // 成员不触碰——并发 shutdown/acquire 看不到半初始化状态）
    std::vector<StreamHandle> created;
    try {
        created.reserve(static_cast<size_t>(slots));
        for (int i = 0; i < slots; ++i) {
            StreamHandle s{};
            if (factory(&s) != 0) {
                for (auto c : created) destroyer(c);
                return Result::fail("GpuSlotService::start: stream factory failed");
            }
            created.push_back(s);
        }
    } catch (...) {
        for (auto c : created) destroyer(c);
        return Result::fail("GpuSlotService::start: stream factory threw");
    }
    // 全部成功才持锁提交；提交时复查状态（锁外创建期间可能已被并发 start/shutdown 抢先）
    std::unique_lock<std::mutex> lock(mutex_);
    if (running_.load() || stopped_) {
        lock.unlock();
        for (auto c : created) destroyer(c);
        return Result::fail(running_.load() ? "GpuSlotService::start: already running"
                                            : "GpuSlotService::start: not restartable after shutdown");
    }
    destroyer_ = std::move(destroyer);
    freeStreams_ = std::move(created);
    running_.store(true);
    return Result::ok();
}

std::optional<GpuSlotService::SlotGuard> GpuSlotService::acquire(std::chrono::milliseconds timeout) {
    if (!running_.load()) {
        return std::nullopt;  // 未 start / 已 shutdown
    }
    std::unique_lock<std::mutex> lock(mutex_);
    if (!slotAvailable_.wait_for(lock, timeout, [this] { return !freeStreams_.empty() || stopped_; })) {
        return std::nullopt;  // 超时
    }
    if (stopped_ || freeStreams_.empty()) {
        return std::nullopt;  // shutdown 唤醒：不再发放
    }
    SlotGuard g;
    g.svc_ = this;
    g.stream = freeStreams_.back();
    freeStreams_.pop_back();
    return g;
}

void GpuSlotService::releaseSlot(SlotGuard& g) {
    StreamHandle s = g.stream;
    g.svc_ = nullptr;  // 先摘钩保证幂等（二次 reset/dtor 无操作）
    g.stream = {};
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
        // 已 shutdown：归还即销毁（不回池）
        lock.unlock();
        destroyer_(s);
        return;
    }
    freeStreams_.push_back(s);
    lock.unlock();
    slotAvailable_.notify_one();
}

void GpuSlotService::shutdown() {
    std::vector<StreamHandle> toDestroy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_.load()) {
            return;  // 未 start 或已 shutdown：幂等
        }
        stopped_ = true;
        running_.store(false);
        toDestroy = std::move(freeStreams_);
        freeStreams_.clear();
    }
    slotAvailable_.notify_all();  // 等待中的 acquire 立刻返回 nullopt
    for (auto s : toDestroy) destroyer_(s);
}

bool GpuSlotService::isRunning() const {
    return running_.load();
}

GpuSlotService::~GpuSlotService() {
    shutdown();
}

void GpuSlotService::SlotGuard::reset() {
    if (svc_) svc_->releaseSlot(*this);
}

GpuSlotService::SlotGuard::~SlotGuard() {
    reset();
}

GpuSlotService::SlotGuard::SlotGuard(SlotGuard&& other) noexcept
    : stream(other.stream), svc_(other.svc_) {
    other.svc_ = nullptr;
    other.stream = {};
}

GpuSlotService::SlotGuard& GpuSlotService::SlotGuard::operator=(SlotGuard&& other) noexcept {
    if (this != &other) {
        reset();  // 归还当前持有的槽（若有）
        stream = other.stream;
        svc_ = other.svc_;
        other.svc_ = nullptr;
        other.stream = {};
    }
    return *this;
}

} // namespace Scanner::pipeline::sched
