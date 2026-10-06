#pragma once
// ============================================================================
// IFrameSource.h — 抓帧策略两副面孔（调度底座；缓冲容器本体在 06 SlotRing）
// ============================================================================
// 设计 §3.4：07 不造容器，只做 06 SlotRing<T> 之上的薄消费策略。
// 计划原拆 IFrameSource.h + FrameSources.h 两文件，此处合并为单头（接口仅
// 3 纯虚、两实现各约 30 行，拆分无收益；T8 LaneRunner 统一 include 本头）。
//
//   - GrabLatestSource（扫描面孔）：X 条 lane 各持自己的 myReadCounter，
//     lag = writePtr - myReadCounter；lag ≤ dropThreshold 取自己领的帧，
//     超过则跳到最新一帧（丢中间，报 skipped 供统计）。
//   - SequentialSource（姿态面孔）：claim→waitFor→read 顺序取全部帧不跳；
//     waitFor 超时返回 nullptr（lane 用以检查停止标志后重试）；每成功一帧
//     调 done() 记账（Backpressure 腾位驱动写侧；Overwrite 下空操作无害）。
//
// myReadCounter 语义：
//   GrabLatest（in/out）：传入"下一个待读帧号"（lane 初始 0）；成功取帧后
//     推进为已取帧号+1（lane 循环不得重复消费）；跳帧时先跳至 writePtr-1。
//   Sequential（out）：仅回写本次取到的帧号供 lane 记账；顺序推进由源内部
//     nextId_ 维护（首个 = claim()），超时/异常不推进（重试仍取同一帧号）。
//
// 线程安全：GrabLatest 可多 lane 并发（无共享可变状态，counter 由各 lane
//   外部持有）；SequentialSource 设计为单线程顺序消费（姿态链单消费者）；
//   多 lane 顺序消费须每 lane 一实例（nextId_ 为实例私有状态），claim 帧号
//   唯一性由 ring.claim() 原子保证。
//   ⚠ SequentialSource 不得配 Overwrite 环在落后>slots 工况下使用：帧被
//   覆盖后 read 永久 null 且 nextId_ 不推进，grabNext 将死循环空转（须配
//   Backpressure 环——写满阻塞写侧，由 done() 腾位保证帧不被覆盖丢失）。
#include <chrono>
#include <cstdint>
#include <memory>

#include "SlotRing.h"

namespace Scanner::pipeline::sched {

template<typename T>
class IFrameSource {                       // 抓帧策略两副面孔（缓冲容器本体在 06 SlotRing）
public:
    virtual ~IFrameSource() = default;
    /// 扫描面孔：抓最新；lag>阈值跳到最新丢中间。返回 nullptr=无帧可取
    virtual std::shared_ptr<const T> grabLatest(uint64_t& myReadCounter, size_t& skipped) = 0;
    /// 姿态面孔：顺序领下一单元（claim→waitFor→read）；无新帧（超时）返回 nullptr
    virtual std::shared_ptr<const T> grabNext(uint64_t& myReadCounter) = 0;
    /// 已写入总数（lag 计算/诊断）
    virtual uint64_t writePtr() const = 0;
};

// ---------------------------------------------------------------------------
// GrabLatestSource — 扫描面孔（Overwrite 环上的多 lane 抓最新）
// ---------------------------------------------------------------------------
template<typename T>
class GrabLatestSource : public IFrameSource<T> {
public:
    GrabLatestSource(Scanner::data::SlotRing<T>& ring, size_t dropThreshold)
        : ring_(ring), dropThreshold_(dropThreshold) {}

    std::shared_ptr<const T> grabLatest(uint64_t& myReadCounter, size_t& skipped) override {
        skipped = 0;
        const uint64_t wp = ring_.writePtr();
        if (myReadCounter >= wp) return nullptr;        // lag==0（或误传超界）：无新帧
        const uint64_t lag = wp - myReadCounter;
        if (lag > dropThreshold_) {                     // 落后过多：跳到最新，丢中间
            myReadCounter = wp - 1;
            skipped = static_cast<size_t>(lag - 1);
        }
        auto frame = ring_.read(myReadCounter);
        if (!frame) {                                   // 竞态被覆盖（理论少见）：跳最新重试一次
            const uint64_t wp2 = ring_.writePtr();
            skipped += static_cast<size_t>(wp2 - 1 - myReadCounter);
            myReadCounter = wp2 - 1;
            frame = ring_.read(myReadCounter);
            if (!frame) return nullptr;                 // 极端竞态：放弃本轮（counter 停在最新）
        }
        ++myReadCounter;                                // counter = 已取帧号+1：lane 循环不重复消费
        return frame;
    }

    /// 扫描面孔无顺序消费语义：不支持（恒 nullptr，不动 counter）
    std::shared_ptr<const T> grabNext(uint64_t&) override { return nullptr; }

    uint64_t writePtr() const override { return ring_.writePtr(); }

private:
    Scanner::data::SlotRing<T>& ring_;
    size_t dropThreshold_;
};

// ---------------------------------------------------------------------------
// SequentialSource — 姿态面孔（Backpressure 环上的单顺序消费者）
// ---------------------------------------------------------------------------
template<typename T>
class SequentialSource : public IFrameSource<T> {
public:
    explicit SequentialSource(Scanner::data::SlotRing<T>& ring,
                              std::chrono::milliseconds timeout = std::chrono::milliseconds(100))
        : ring_(ring), timeout_(timeout) {}

    std::shared_ptr<const T> grabNext(uint64_t& myReadCounter) override {
        if (!claimed_) {                                // 首个帧号 = claim() 领号
            nextId_ = ring_.claim();
            claimed_ = true;
        }
        if (!ring_.waitFor(nextId_, timeout_)) return nullptr;  // 超时：不推进（重试同帧号）
        auto frame = ring_.read(nextId_);
        if (!frame) return nullptr;                     // 异常（如被覆盖）：不推进
        myReadCounter = nextId_;                        // 回写本次帧号供 lane 记账
        ++nextId_;
        ring_.done();                                   // Backpressure 腾位；Overwrite 空操作
        return frame;
    }

    /// 姿态面孔无"跳最新"语义：不支持（恒 nullptr，不动 counter/skipped）
    std::shared_ptr<const T> grabLatest(uint64_t&, size_t& skipped) override {
        skipped = 0;
        return nullptr;
    }

    uint64_t writePtr() const override { return ring_.writePtr(); }

private:
    Scanner::data::SlotRing<T>& ring_;
    std::chrono::milliseconds timeout_;
    uint64_t nextId_{0};
    bool claimed_{false};
};

} // namespace Scanner::pipeline::sched
