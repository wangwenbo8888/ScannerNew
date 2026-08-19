#pragma once
// ============================================================================
// SchedulerRuntime.h — 调度底座组合运行时（E 核编排循环 + 启停逆序 + 统计）
// ============================================================================
// 组合 T4~T7 底座件（GpuSlotService / PCoreBroker / CpuTopology / IFrameSource）
// 为完整运行时。每 lane 一线程，单帧生命周期：
//
//   抓帧（互斥串行 + 共享计数器：每帧恰送一条 lane，多 lane 分工不重复消费）
//   → acquire GPU guard（超时丢帧，计 gpuRejects）
//   → onFrontReady 非空：先调 onFrontReady 并即刻提交 pChain 拿 future
//     （P 链与 GPU 链帧内并行；LaneHooks 签名无中途回调通道，"ccl 就绪点"的
//       前端准备由 onFrontready 钩子自身完成——提交即并行）
//     onFrontReady 空：gpuChain 返回 true 后才提交（A 模式推迟提交）
//   → gpuChain（false=帧销毁：计 gpuRejects，已提交 future 弃置不 get）
//   → future.get() → eFinalize（成功 processed++）→ guard 析构 → 下一帧
//   （帧边界查 stop：抓帧前与迭代退出）
//
// ⚠ 前端并行契约：onFrontReady 非空时，pChain（P 核线程）与 gpuChain（E 核
//   线程）并发访问同一 lane 的 front；gpuChain 返回 false 的已弃置 pChain 任务
//   亦可能与下一帧的 gpuChain 并发。TFront 须分区设计（pChain 只读写自己分区，
//   gpuChain 只写其余分区），否则数据竞争。
//
// 其余约定：
//   - TFront/TResult 须可默认构造；front 为每 lane 一份（跨帧复用），
//     result 为每帧一份（经 shared_ptr 与 P 任务共享所有权）
//   - queue 由调用方钩子（eFinalize 内 push）持有使用，运行时不触碰（A 模式可空）
//   - 统计三计数在每次成功 start 时清零
//   - source 及钩子捕获物的生命周期须覆盖运行期（至 drainAndShutdown 返回）
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "base/types.h"
#include "sched/CpuTopology.h"
#include "sched/FrameResultQueue.h"
#include "sched/GpuSlotService.h"
#include "sched/IFrameSource.h"
#include "sched/PCoreBroker.h"
#include "sched/SchedConfig.h"

#include <spdlog/spdlog.h>

namespace Scanner::pipeline::sched {

template<typename TFrame, typename TFront, typename TResult>
struct LaneHooks {
    std::function<bool(typename GpuSlotService::SlotGuard&, const std::shared_ptr<const TFrame>&, TFront&)> gpuChain;
    std::function<void(const std::shared_ptr<const TFrame>&, TFront&)> onFrontReady;
    std::function<Result(const std::shared_ptr<const TFrame>&, TFront&, TResult&)> pChain;
    std::function<Result(const std::shared_ptr<const TFrame>&, TFront&, TResult&, std::future<Result>&)> eFinalize;
};

class SchedulerRuntime {
public:
    struct Stats {
        uint64_t processed = 0;     // eFinalize 成功帧数
        uint64_t droppedSkips = 0;  // grabLatest 落后超阈值跳帧累计
        uint64_t gpuRejects = 0;    // acquire 超时 + gpuChain false 帧销毁
    };

    SchedulerRuntime() = default;
    ~SchedulerRuntime();

    /// GPU 流工厂注入（测试假工厂路径）；默认空 = 生产 CUDA 工厂。
    /// 须在 start 前调用；运行期修改无效。
    void setGpuStreamFactory(GpuSlotService::StreamFactory factory,
                             GpuSlotService::StreamDestroyer destroyer = {});

    /// sequential=true 用 grabNext（A 姿态顺序反压），false 用 grabLatest（C 扫描跳最新）；
    /// queue 可空（A 模式：结果在 eFinalize 内自行收集）。已运行时返回 fail；
    /// drainAndShutdown 后可再次 start（restart）。
    template<typename TFrame, typename TFront, typename TResult>
    Result start(const SchedConfig& cfg, IFrameSource<TFrame>& source, bool sequential,
                 FrameResultQueue<TResult>* queue, const LaneHooks<TFrame, TFront, TResult>& hooks);

    /// 置停标志（lane 跑完在飞帧后退出）
    void requestStop();

    /// join lanes → broker.shutdown() → gpu_.shutdown()（逆序收摊）；
    /// 幂等；未 requestStop 直接调用也安全（内部先置停）
    void drainAndShutdown();

    /// 三计数快照
    Stats stats() const;

    bool isRunning() const;

    SchedulerRuntime(const SchedulerRuntime&) = delete;
    SchedulerRuntime& operator=(const SchedulerRuntime&) = delete;

private:
    template<typename TFrame, typename TFront, typename TResult>
    void laneLoop(IFrameSource<TFrame>& source, bool sequential,
                  std::chrono::milliseconds gpuTimeout, const LaneHooks<TFrame, TFront, TResult>& hooks,
                  std::mutex& grabMutex, uint64_t& sharedCounter,
                  const std::shared_ptr<TFront>& front);

    mutable std::mutex lifecycleMutex_;               // start / drainAndShutdown 互斥
    std::vector<std::thread> lanes_;
    std::atomic<bool> stopFlag_{false};
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> processed_{0};
    std::atomic<uint64_t> droppedSkips_{0};
    std::atomic<uint64_t> gpuRejects_{0};
    PCoreBroker broker_;
    std::unique_ptr<GpuSlotService> gpu_;             // 一次性服务：每运行周期重建（支持 restart）
    GpuSlotService::StreamFactory gpuFactory_;        // 可注入（测试假工厂；空=生产默认）
    GpuSlotService::StreamDestroyer gpuDestroyer_;
};

// ---------------------------------------------------------------------------
// 模板成员实现（start 为模板，无法置 .cpp）
// ---------------------------------------------------------------------------
template<typename TFrame, typename TFront, typename TResult>
Result SchedulerRuntime::start(const SchedConfig& cfg, IFrameSource<TFrame>& source, bool sequential,
                               FrameResultQueue<TResult>* /*queue：钩子持有，运行时不触碰*/,
                               const LaneHooks<TFrame, TFront, TResult>& hooks) {
    std::lock_guard<std::mutex> lifecycleLock(lifecycleMutex_);
    if (running_.load()) {
        return Result::fail("SchedulerRuntime::start: already running");
    }
    if (!hooks.gpuChain || !hooks.pChain || !hooks.eFinalize) {
        return Result::fail("SchedulerRuntime::start: gpuChain/pChain/eFinalize hooks required");
    }

    const TopologyInfo topo = CpuTopology::detect();
    const int lanes = computeLanes(topo.pCores, topo.eCores, cfg.lanes);
    if (lanes <= 0) {
        return Result::fail("SchedulerRuntime::start: lanes must be > 0");
    }

    // GPU 槽池（一次性服务，每周期重建；工厂可注入）
    gpu_ = std::make_unique<GpuSlotService>();
    const Result gres = gpu_->start(cfg.gpuSlots, gpuFactory_, gpuDestroyer_);
    if (!gres.success) {
        gpu_.reset();
        return Result::fail("SchedulerRuntime::start: gpu service start failed: " + gres.message);
    }

    // P 核代理：workers = lanes-1；lanes==1 时不可为 0 → 用 1。
    // 掩码取 P 核 masks[1..]（首核让给系统；仅 hybrid 有值，不足部分 broker 补 0 不绑）
    const int workers = (lanes > 1) ? lanes - 1 : 1;
    std::vector<uint64_t> brokerMasks;
    if (topo.hybrid && topo.pMasks.size() > 1) {
        brokerMasks.assign(topo.pMasks.begin() + 1, topo.pMasks.end());
    }
    const Result bres = broker_.start(workers, brokerMasks);
    if (!bres.success) {
        gpu_->shutdown();
        gpu_.reset();
        return Result::fail("SchedulerRuntime::start: broker start failed: " + bres.message);
    }

    processed_.store(0);
    droppedSkips_.store(0);
    gpuRejects_.store(0);
    stopFlag_.store(false);

    // 抓帧互斥 + 共享计数器：每帧恰送一条 lane（顺序面孔下单 SequentialSource
    // 实例多 lane 也安全；跳最新面孔下多 lane 分工不重复消费）
    auto grabMutex = std::make_shared<std::mutex>();
    auto sharedCounter = std::make_shared<uint64_t>(0);
    auto hooksp = std::make_shared<LaneHooks<TFrame, TFront, TResult>>(hooks);

    try {
        for (int i = 0; i < lanes; ++i) {
            lanes_.emplace_back([this, &source, sequential, timeout = cfg.gpuAcquireTimeout, hooksp,
                                 grabMutex, sharedCounter] {
                // front 每 lane 一份（跨帧复用；经 shared_ptr 与 P 任务共享所有权）
                auto front = std::make_shared<TFront>();
                laneLoop<TFrame, TFront, TResult>(source, sequential, timeout, *hooksp,
                                                  *grabMutex, *sharedCounter, front);
            });
            auto& t = lanes_.back();
            // 绑 E 核 eMasks 轮转；无 E 核（非 hybrid）不绑；绑核（mask≠0）才提实时优先级
            const uint64_t mask = (topo.hybrid && !topo.eMasks.empty())
                                      ? topo.eMasks[static_cast<size_t>(i) % topo.eMasks.size()]
                                      : 0;
            if (mask != 0) {
                CpuTopology::pinThread(t, mask);
                CpuTopology::setRealtime(t);
            }
        }
    } catch (...) {
        // 线程创建失败（资源耗尽等）：逆序回收已建部分
        stopFlag_.store(true);
        for (auto& t : lanes_) {
            if (t.joinable()) t.join();
        }
        lanes_.clear();
        broker_.shutdown();
        gpu_->shutdown();
        gpu_.reset();
        return Result::fail("SchedulerRuntime::start: lane thread creation failed");
    }

    running_.store(true);
    return Result::ok();
}

template<typename TFrame, typename TFront, typename TResult>
void SchedulerRuntime::laneLoop(IFrameSource<TFrame>& source, bool sequential,
                                std::chrono::milliseconds gpuTimeout,
                                const LaneHooks<TFrame, TFront, TResult>& hooks,
                                std::mutex& grabMutex, uint64_t& sharedCounter,
                                const std::shared_ptr<TFront>& front) {
    while (!stopFlag_.load()) {                       // 帧边界检查点（抓帧前）
        std::shared_ptr<const TFrame> frame;
        {
            std::lock_guard<std::mutex> g(grabMutex); // 串行抓帧：帧不重复分发
            if (sequential) {
                frame = source.grabNext(sharedCounter);
            } else {
                size_t skipped = 0;
                frame = source.grabLatest(sharedCounter, skipped);
                if (skipped > 0) droppedSkips_.fetch_add(skipped);
            }
        }
        if (!frame) {
            // sequential：grabNext 内部已阻塞一个超时周期，直接重查 stop；
            // grabLatest 空转（无新帧）小睡防忙转
            if (!sequential) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto guard = gpu_->acquire(gpuTimeout);       // 超时丢帧
        if (!guard) {
            gpuRejects_.fetch_add(1);
            continue;
        }

        auto result = std::make_shared<TResult>();    // 每帧一份（与 P 任务共享所有权）
        std::future<Result> fut;
        const bool frontPath = static_cast<bool>(hooks.onFrontReady);
        if (frontPath) {                              // 前端就绪即提交：P 链与 GPU 链帧内并行
            hooks.onFrontReady(frame, *front);
            auto pChain = hooks.pChain;
            fut = broker_.submit([pChain, frame, front, result] {
                return pChain(frame, *front, *result);
            });
        }
        if (!hooks.gpuChain(*guard, frame, *front)) { // false=帧销毁
            gpuRejects_.fetch_add(1);
            continue;                                 // guard 归还槽；已提交 future 弃置不 get
        }
        if (!frontPath) {                             // 空回调：gpuChain 返回后才提交
            auto pChain = hooks.pChain;
            fut = broker_.submit([pChain, frame, front, result] {
                return pChain(frame, *front, *result);
            });
        }
        try {
            (void)fut.get();                          // 等 P 核结果（异常路径见 catch）
        } catch (const std::exception& e) {
            spdlog::warn("SchedulerRuntime: pChain 任务异常, 帧弃置: {}", e.what());
            continue;
        } catch (...) {
            spdlog::warn("SchedulerRuntime: pChain 任务未知异常, 帧弃置");
            continue;
        }
        if (hooks.eFinalize(frame, *front, *result, fut).success) {
            processed_.fetch_add(1);
        }
    }                                                 // guard 析构 → 下一帧
}

} // namespace Scanner::pipeline::sched
