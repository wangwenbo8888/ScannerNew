#pragma once
#include <atomic>
#include <memory>
#include <vector>
#include <cstdint>

namespace calib {

struct MarkerPoint3D {
    double x, y, z;
    double nx, ny, nz;
    int globalId;
};

struct AtomicFrameState {
    std::vector<MarkerPoint3D> rawPoints;
    std::vector<double> normals;
    std::vector<int> globalIds;

    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double T[3] = {0,0,0};
    uint64_t frameId = 0;

    static std::shared_ptr<AtomicFrameState> load(
        const std::shared_ptr<AtomicFrameState>& src) {
        return std::atomic_load(&src);
    }

    static void store(std::shared_ptr<AtomicFrameState>& dst,
                      std::shared_ptr<AtomicFrameState> state) {
        std::atomic_store(&dst, std::move(state));
    }

    /// 帧号守卫写入（2026-10-15 生产加固）：已存状态的帧号 ≥ 本帧则放弃写入——
    /// 多 lane 乱序完成时，慢线程的旧帧结果不得覆盖快线程的新帧结果
    /// （调试记录曾载 7-lane 旧帧覆盖新帧致 prevState 锚错位、配准全败）。
    /// 本帧观测不受影响（逐帧缓冲另有通道），只是配准基准不被回拨。
    /// CAS 循环保证判优与写入原子；同帧号重写同样拒绝（幂等保护）。
    static void storeIfNewer(std::shared_ptr<AtomicFrameState>& dst,
                             std::shared_ptr<AtomicFrameState> state) {
        std::shared_ptr<AtomicFrameState> cur = std::atomic_load(&dst);
        for (;;) {
            if (cur && cur->frameId >= state->frameId) return;
            if (std::atomic_compare_exchange_weak(&dst, &cur, state)) return;
            // CAS 失败时 cur 已被刷新为当前值，循环重判
        }
    }

    bool isFirstFrame() const { return frameId == 0; }
};

} // namespace calib
