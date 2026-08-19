#pragma once
// ============================================================================
// SchedulerRuntime.h — 调度底座组合运行时（E 核编排循环 + 启停逆序 + 统计）
// ============================================================================
// 组合 T4~T7 底座件（GpuSlotService / PCoreBroker / CpuTopology / IFrameSource）
// 为完整运行时。每 lane 一线程，单帧生命周期：
//
//   抓帧（互斥串行 + 共享计数器：每帧恰送一条 lane，多 lane 分工不重复消费）
//   → acquire GPU guard（超时丢帧，计 gpuRejects）
//   → gpuChain（持槽执行；在"ccl 就绪点"调用 runtime 注入的 frontReady()
//     回调——runtime 由此刻提交 pChain 拿 future，激光链剩余段与 P 链帧内并行）
//       · gpuChain 返回 true 且未触发 frontReady → 返回后兜底提交（A 模式路径）
//       · gpuChain 返回 false=帧销毁：计 gpuRejects；已触发则先 fut.wait()
//         等孤儿 P 任务完成再进下一帧（防与下一帧并发读写 per-lane TFront）。
//         注：wait() 无超时——依赖 Broker pChain 任务正常返回；任务卡死属
//         Broker 层故障，本层不设看门狗
//   → eFinalize（收有效 future：pChain 的 Result 与异常均由钩子经 get() 消费
//     与处理；runtime 不预取 get，eFinalize 返回后不再触碰 future）
//     成功 processed++ → guard 析构 → 下一帧（帧边界查 stop）
//
// ⚠ ccl 就绪点契约：frontReady 由 gpuChain 恰调一次或零次；多次调用幂等
//   （第二次起 no-op）。frontReady() 之后 pChain（P 核线程）与 gpuChain 剩余段
//   （E 核线程）并发访问同一 lane 的 TFront —— TFront 须分区设计（pChain 只读写
//   自己分区，激光链只写其余分区），否则数据竞争。
//
// 异常契约：帧主体任一钩子（抓帧/gpuChain/frontReady 提交/eFinalize）抛异常
//   → runtime 顶层捕获：spdlog error + finalizeFails++ + requestStop（异常即停
//   防连环崩）；pChain 任务内异常经 future 携带，由 eFinalize 钩子自行处理。
//
// 其余约定：
//   - TFront/TResult 须可默认构造；front 为每 lane 一份（跨帧复用），
//     result 为每帧一份（经 shared_ptr 与 P 任务共享所有权）
//   - queue 参数预留：当前由 eFinalize 钩子自行 push（运行时不触碰，A 模式可
//     空）；P2 接入时再决定是否由 runtime 统一发布
//   - 统计四计数在每次成功 start 时清零
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
    // GPU 段（已持 GpuSlotService::SlotGuard；返回 false=帧销毁，直接下一帧）。
    // frontReady：runtime 注入的"ccl 就绪点"提交回调——gpuChain 在前端数据
    // （ccl 产物）就绪的时刻调用它，此后激光链续跑与 P 链帧内并行；
    // 恰调一次或零次；多次调用幂等（第二次起 no-op）。
    std::function<bool(typename GpuSlotService::SlotGuard&, const std::shared_ptr<const TFrame>&, TFront&,
                       std::function<void()> frontReady)>
        gpuChain;
    // P 核段（PCoreBroker 上执行；A=标记点链，C=标记点链+配准）
    std::function<Result(const std::shared_ptr<const TFrame>&, TFront&, TResult&)> pChain;
    // E 核终段（A=融合+判姿+收集；C=汇合组装+发布）。future 为有效 future（runtime
    // 不预取 get）：可能携带 pChain 异常，钩子负责 get 与异常处理；runtime 在
    // eFinalize 返回后不再触碰 future。
    std::function<Result(const std::shared_ptr<const TFrame>&, TFront&, TResult&, std::future<Result>&)> eFinalize;
};

class SchedulerRuntime {
public:
    struct Stats {
        uint64_t processed = 0;     // eFinalize 成功帧数
        uint64_t droppedSkips = 0;  // grabLatest 落后超阈值跳帧累计
        uint64_t gpuRejects = 0;    // acquire 超时 + gpuChain false 帧销毁
        uint64_t finalizeFails = 0; // 帧主体钩子异常计数（异常即停机）
    };

    SchedulerRuntime() = default;
    ~SchedulerRuntime();

    /// GPU 流工厂注入（测试假工厂路径）；默认空 = 生产 CUDA 工厂。
    /// 须在 start 前调用；运行期调用返回 fail。⚠ 与 start 并发调用存在数据
    /// 竞争窗口（本 setter 无锁写 gpuFactory_ vs start 锁内拷贝读）：约定仅在
    /// 装配阶段（同线程、start 之前）调用，与 start 并发调用属 UB。
    Result setGpuStreamFactory(GpuSlotService::StreamFactory factory,
                               GpuSlotService::StreamDestroyer destroyer = {});

    /// sequential=true 用 grabNext（A 姿态顺序反压），false 用 grabLatest（C 扫描跳最新）；
    /// queue 参数预留（见文件头），A 模式可传空。已运行时返回 fail；
    /// drainAndShutdown 后可再次 start（restart）。
    /// startCounter：共享抓帧计数器初值（消费水位注入——pause/resume 后跳过已消费
    /// 帧）。语义随面孔：GrabLatest=下一待读帧号（lane 循环推进为已取帧号+1）；
    /// Sequential=最近已取帧号（源内部 nextId_ 推进）。默认 0=从头消费。
    template<typename TFrame, typename TFront, typename TResult>
    Result start(const SchedConfig& cfg, IFrameSource<TFrame>& source, bool sequential,
                 FrameResultQueue<TResult>* queue, const LaneHooks<TFrame, TFront, TResult>& hooks,
                 uint64_t startCounter = 0);

    /// 置停标志（lane 跑完在飞帧后退出）
    void requestStop();

    /// join lanes → broker.shutdown() → gpu_.shutdown()（逆序收摊）；
    /// 幂等；未 requestStop 直接调用也安全（内部先置停）
    void drainAndShutdown();

    /// 四计数快照
    Stats stats() const;

    /// 当前共享抓帧计数器值（每次成功 grab 后随锁更新；drain 后即最终消费水位，
    /// 供 restart 作 startCounter 注入）
    uint64_t lastCounter() const;

    /// lanes 是否已全部退出（含"从未 start"的空集语义）——正常 requestStop 停与
    /// 异常即停均会置位；运行中 false。供上层（ScanPipeline）惰性发现 runtime 自灭
    bool lanesExited() const;

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
    std::atomic<uint64_t> finalizeFails_{0};
    std::atomic<uint64_t> lastCounter_{0};           // 共享抓帧计数器镜像（随锁单调）
    std::atomic<int> activeLanes_{0};                // 在飞 lane 数（0=已全退/未启动）
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
                               FrameResultQueue<TResult>* /*queue：参数预留，eFinalize 钩子自行 push*/,
                               const LaneHooks<TFrame, TFront, TResult>& hooks,
                               uint64_t startCounter) {
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
    finalizeFails_.store(0);
    lastCounter_.store(startCounter, std::memory_order_relaxed);   // 消费水位基线（restart 注入）
    activeLanes_.store(lanes);                        // lane 退出时递减（0=全退）
    stopFlag_.store(false);
    running_.store(true);                             // 置位提前：线程创建前（失败路径回退）

    // 抓帧互斥 + 共享计数器：每帧恰送一条 lane（顺序面孔下单 SequentialSource
    // 实例多 lane 也安全；跳最新面孔下多 lane 分工不重复消费）
    auto grabMutex = std::make_shared<std::mutex>();
    auto sharedCounter = std::make_shared<uint64_t>(startCounter);
    auto hooksp = std::make_shared<LaneHooks<TFrame, TFront, TResult>>(hooks);

    try {
        for (int i = 0; i < lanes; ++i) {
            lanes_.emplace_back([this, &source, sequential, timeout = cfg.gpuAcquireTimeout, hooksp,
                                 grabMutex, sharedCounter] {
                // front 每 lane 一份（跨帧复用；经 shared_ptr 与 P 任务共享所有权）
                auto front = std::make_shared<TFront>();
                laneLoop<TFrame, TFront, TResult>(source, sequential, timeout, *hooksp,
                                                  *grabMutex, *sharedCounter, front);
                activeLanes_.fetch_sub(1, std::memory_order_release);   // 退出即减（异常停可被上层发现）
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
        running_.store(false);
        return Result::fail("SchedulerRuntime::start: lane thread creation failed");
    }

    return Result::ok();
}

template<typename TFrame, typename TFront, typename TResult>
void SchedulerRuntime::laneLoop(IFrameSource<TFrame>& source, bool sequential,
                                std::chrono::milliseconds gpuTimeout,
                                const LaneHooks<TFrame, TFront, TResult>& hooks,
                                std::mutex& grabMutex, uint64_t& sharedCounter,
                                const std::shared_ptr<TFront>& front) {
    while (!stopFlag_.load()) {                       // 帧边界检查点（抓帧前）
        try {                                         // 顶层异常捕获：钩子异常不蔓延到线程
            std::shared_ptr<const TFrame> frame;
            {
                std::lock_guard<std::mutex> g(grabMutex);  // 串行抓帧：帧不重复分发
                if (sequential) {
                    frame = source.grabNext(sharedCounter);
                } else {
                    size_t skipped = 0;
                    frame = source.grabLatest(sharedCounter, skipped);
                    if (skipped > 0) droppedSkips_.fetch_add(skipped);
                }
                // 水位镜像（锁内随 counter 单调推进）：drain 后 lastCounter() 即最终值
                lastCounter_.store(sharedCounter, std::memory_order_relaxed);
            }
            if (!frame) {
                // sequential：grabNext 内部已阻塞一个超时周期，直接重查 stop；
                // grabLatest 空转（无新帧）小睡防忙转
                if (!sequential) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto guard = gpu_->acquire(gpuTimeout);   // 超时丢帧
            if (!guard) {
                gpuRejects_.fetch_add(1);
                continue;
            }

            auto result = std::make_shared<TResult>();  // 每帧一份（与 P 任务共享所有权）
            std::future<Result> fut;                  // 有效 future 直传 eFinalize（不预取 get）
            std::atomic<bool> submitted{false};       // frontReady 恰一次防护
            auto frontReady = [this, &hooks, &frame, &front, &result, &fut, &submitted] {
                bool exp = false;
                if (!submitted.compare_exchange_strong(exp, true)) {
                    return;                           // 幂等：第二次起 no-op
                }
                auto pChain = hooks.pChain;
                fut = broker_.submit([pChain, frame, front, result] {
                    return pChain(frame, *front, *result);
                });
            };

            const bool gpuOk = hooks.gpuChain(*guard, frame, *front, frontReady);
            if (!gpuOk) {                             // false=帧销毁
                gpuRejects_.fetch_add(1);
                if (submitted.load()) {
                    fut.wait();                       // 等孤儿 P 任务完成再进下一帧（防 TFront 并发）；
                                                     // 无超时——依赖 pChain 正常返回，卡死属 Broker 层故障
                }
                continue;                             // guard 归还槽；future 弃置（未 get 不抛）
            }
            if (!submitted.load()) {
                frontReady();                         // 兜底提交：gpuChain 返回 true 且未触发（A 模式）
            }
            if (hooks.eFinalize(frame, *front, *result, fut).success) {
                processed_.fetch_add(1);
            }
        } catch (const std::exception& e) {
            spdlog::error("SchedulerRuntime: lane 帧主体钩子异常, 停机: {}", e.what());
            finalizeFails_.fetch_add(1);
            requestStop();                            // 异常即停防连环崩
        } catch (...) {
            spdlog::error("SchedulerRuntime: lane 帧主体钩子未知异常, 停机");
            finalizeFails_.fetch_add(1);
            requestStop();
        }
    }                                                 // guard 析构 → 下一帧
}

} // namespace Scanner::pipeline::sched
