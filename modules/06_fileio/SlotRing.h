#pragma once
// ============================================================================
// SlotRing.h — 原子槽位环（单生产者 / 多消费者；Data 层容器）
// ============================================================================
// 供 07 抓帧策略消费的调度底座容器（07 设计方案 §5.2 契约）：
//   - P0 采集线程单写（write 单调递增 writePtr）；X 条流水线并发读
//   - 槽位为 shared_ptr，原子换入（std::atomic_store，C++17 自由函数）；
//     已取出的引用在覆盖后依旧安全（引用计数保活）
//   - 两种写策略：Overwrite（扫描：满则覆盖最旧，永不阻塞）/
//     Backpressure（姿态：满则阻塞写线程，等 done() 记账腾位）
//
// read() 的近似语义约定（调用方须知）：
//   read 仅确定性挡掉两类帧号——"未写入"（frameId >= writePtr）与
//   "确定已被覆盖"（frameId + slots < writePtr）；窗口边缘若与 write()
//   并发，槽内可能已换入更新的帧，read 可能返回新帧内容。
//   正确用法：先 waitFor(frameId) 确认写入，再 read(frameId)。
//
// 线程安全：write（单线程约定，持锁期间亦容忍误用多生产者串行化）/
//   claim / read / done / waitFor 并发安全。
//
// 锁布局：单 mutex（mu_）+ 双 condition_variable——
//   wrCv_：Backpressure 写侧等腾位，done() 通知；
//   notEmptyCv_：waitFor 读侧等新帧，write() 通知。
//   两 cv 共用一把锁且所有等待均在锁内谓词等待（wait 释放锁），
//   写阻塞不阻碍 done()/waitFor() 拿锁，无死锁。

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace Scanner::data {

template<typename T>
class SlotRing {
public:
    enum class WriterMode { Overwrite, Backpressure };

    explicit SlotRing(size_t slots, WriterMode mode = WriterMode::Overwrite)
        : slots_(slots ? slots : 1), mode_(mode), buf_(slots ? slots : 1) {}
    SlotRing(const SlotRing&) = delete;
    SlotRing& operator=(const SlotRing&) = delete;

    /// 生产者写入（P0 单线程调用）。
    /// Overwrite：覆盖 writePtr-slots 槽，永不阻塞；
    /// Backpressure：满（writePtr-donePtr >= slots）阻塞等 done() 腾位。
    void write(std::shared_ptr<T> item) {
        std::unique_lock<std::mutex> lock(mu_);
        if (mode_ == WriterMode::Backpressure) {
            wrCv_.wait(lock, [this] {
                return writePtr_.load(std::memory_order_relaxed)
                     - donePtr_.load(std::memory_order_relaxed) < slots_;
            });
        }
        const uint64_t wp = writePtr_.load(std::memory_order_relaxed);
        std::atomic_store(&buf_[wp % slots_], std::move(item));
        writePtr_.fetch_add(1, std::memory_order_release);
        lock.unlock();
        notEmptyCv_.notify_all();
    }

    /// 顺序消费者原子领号（全局唯一递增，多线程安全）
    uint64_t claim() { return claimPtr_.fetch_add(1, std::memory_order_relaxed); }

    /// 读指定帧号：未写入 / 确定已覆盖 → nullptr；其余返回槽内容
    ///（近似语义见文件头注释；持引用者在覆盖后依旧安全）
    std::shared_ptr<const T> read(uint64_t frameId) const {
        const uint64_t wp = writePtr_.load(std::memory_order_acquire);
        if (frameId >= wp) return nullptr;            // 未写入
        if (frameId + slots_ < wp) return nullptr;    // 确定已覆盖
        return std::atomic_load(&buf_[frameId % slots_]);
    }

    /// 本帧消费完毕记账：Backpressure 腾一位并唤醒写侧；Overwrite 空操作。
    /// 约定：done 次数须与已消费的已写入帧一一对应（多于写入会压死写侧）。
    void done() {
        if (mode_ != WriterMode::Backpressure) return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            donePtr_.fetch_add(1, std::memory_order_relaxed);
        }
        wrCv_.notify_all();
    }

    /// 等新帧：writePtr > frameId 或超时（超时返回 false）
    bool waitFor(uint64_t frameId, std::chrono::milliseconds timeout) const {
        std::unique_lock<std::mutex> lock(mu_);
        return notEmptyCv_.wait_for(lock, timeout, [this, frameId] {
            return writePtr_.load(std::memory_order_acquire) > frameId;
        });
    }

    /// 已写入总数
    uint64_t writePtr() const { return writePtr_.load(std::memory_order_acquire); }
    /// 已消费总数（Backpressure）
    uint64_t donePtr() const { return donePtr_.load(std::memory_order_acquire); }

private:
    size_t slots_;
    WriterMode mode_;
    std::vector<std::shared_ptr<T>> buf_;
    std::atomic<uint64_t> writePtr_{0};
    std::atomic<uint64_t> claimPtr_{0};
    std::atomic<uint64_t> donePtr_{0};
    mutable std::mutex mu_;                       // 护 Backpressure 谓词与 waitFor 等待
    std::condition_variable wrCv_;                // 写侧：Backpressure 等腾位（done 通知）
    mutable std::condition_variable notEmptyCv_;  // 读侧：waitFor 等新帧（write 通知）
};

} // namespace Scanner::data
