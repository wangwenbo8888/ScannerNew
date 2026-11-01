#pragma once
// ============================================================================
// SchedConfig.h — 调度底座配置（A/C 两模式仅此不同）
// ============================================================================
#include <chrono>
#include <cstddef>

namespace Scanner::pipeline::sched {

struct SchedConfig {
    int lanes{0};                 // 流水线条数 X；0=自动探测（clamp(min(P-1,E),1,8)）
    int gpuSlots{1};              // GPU 槽数：姿态=1 / 扫描=2
    size_t queueCapacity{16};     // FrameResultQueue 容量
    size_t dropThreshold{0};      // 扫描丢帧阈值；0=2*lanes（仅 GrabLatestSource 用）
    std::chrono::milliseconds gpuAcquireTimeout{2000};
    int pcorePriorityBoost{1};    // worker 线程是否提优先级
    int hangTimeoutMs{10000};     // 看门狗：lane 心跳静默超此值判 hang（0=关）；
                                   //   检出即 onHang 回调＋requestStop（恢复路径见 drain 限时变体）
};

} // namespace Scanner::pipeline::sched
