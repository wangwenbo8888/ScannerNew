#pragma once
// ============================================================================
// GpuSlotService.h — cudaStream 槽池（调度底座）
// 计数信号量（先到先得）+ RAII SlotGuard 归还（支持 reset() 显式提前归还，幂等）。
// 头文件经别名 StreamHandle 隔离 CUDA：仅 JMW_BUILD_CUDA 时映射 cudaStream_t，
// BUILD_CUDA=OFF 时为 void*（本文件不含 CUDA 头亦可编译，此分支无 CUDA 依赖）。
//
// shutdown 语义：置停并唤醒全部等待者（acquire 立刻返回 nullopt），
// 销毁全部未占用 stream；不等待在飞 guard —— shutdown 后 in-flight guard
// 归还即销毁该 stream（不回池）；服务不可重启。
// 线程安全约定：shutdown()/析构不可与其他线程的 shutdown()/析构并发；
// guard 不得越过服务生命周期存活（归还时解引用 svc_）。
// ============================================================================
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include "base/types.h"

#ifdef JMW_BUILD_CUDA
#include <cuda_runtime_api.h>
#endif

namespace Scanner::pipeline::sched {

class GpuSlotService {
public:
    /// CUDA 隔离别名（类内导出，BUILD_CUDA 无关可用）
#ifdef JMW_BUILD_CUDA
    using StreamHandle = cudaStream_t;
#else
    using StreamHandle = void*;
#endif
    using StreamFactory   = std::function<int(StreamHandle*)>;  // 0=成功（仿 cudaStreamCreate 签名）
    using StreamDestroyer = std::function<void(StreamHandle)>;

    /// 槽句柄守卫：析构或 reset() 归还；可移动不可拷贝。
    struct SlotGuard {
        StreamHandle stream = {};
        void reset();                       // 显式归还（幂等）
        ~SlotGuard();                       // 归还
        SlotGuard(SlotGuard&&) noexcept;    // 仅移动
        SlotGuard& operator=(SlotGuard&&) noexcept;
        SlotGuard(const SlotGuard&) = delete;
        SlotGuard& operator=(const SlotGuard&) = delete;
        SlotGuard() = default;              // 空 guard（optional 用）
    private:
        friend class GpuSlotService;
        GpuSlotService* svc_ = nullptr;
    };

    /// slots<=0 fail；factory/destroyer 为空时用默认实现（JMW_BUILD_CUDA 时
    /// cudaStreamCreate/cudaStreamDestroy；否则返回 fail —— CUDA 关闭时不支持
    /// 真 GPU 槽）。已在运行或 shutdown 后再次 start 均返回 fail。
    Result start(int slots, StreamFactory factory = {}, StreamDestroyer destroyer = {});

    /// 阻塞至有空槽或超时；超时/未 start/已 shutdown 返回 nullopt
    std::optional<SlotGuard> acquire(std::chrono::milliseconds timeout);

    /// 幂等；未 shutdown 则析构时自动调用
    void shutdown();

    bool isRunning() const;

    ~GpuSlotService();

    GpuSlotService() = default;
    GpuSlotService(const GpuSlotService&) = delete;
    GpuSlotService& operator=(const GpuSlotService&) = delete;

private:
    void releaseSlot(SlotGuard& g);

    mutable std::mutex mutex_;
    std::condition_variable slotAvailable_;
    std::vector<StreamHandle> freeStreams_;  // 空闲槽池（占用中不在内）
    StreamDestroyer destroyer_;
    int available_{0};                       // 计数信号量（<= freeStreams_.size()）
    bool stopped_{false};                    // shutdown 置位后不可重启
    std::atomic<bool> running_{false};
};

} // namespace Scanner::pipeline::sched
