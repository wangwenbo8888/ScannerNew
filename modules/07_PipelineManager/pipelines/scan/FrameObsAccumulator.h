#pragma once
// ============================================================================
// FrameObsAccumulator.h — 逐帧观测累加器 + 激光帧缓存（GBA 输入 / D 重融合）
//
// GBA 需要逐帧观测 {R_init, t_init, markerObs[]}（设计方案 §4.3）：融合累积器的
// 体素去重云已丢逐帧结构，不可替代。D 重融合另需逐帧激光 3D（host float xyz），
// 按字节预算缓存，超限停累加并记 Degraded（markerObs 照常累加，整体不失败）。
//
// 线程模型：push 为 FuseConsumer 单线程写；snapshot/clear 与 push 经 mu_ 互斥。
// ⚠ 内存增长：obsList_ 无上限增长、snapshot 深拷贝瞬时双倍内存（万帧级实测后评估 reserve）。
// ============================================================================
#include <atomic>
#include <cstddef>
#include <mutex>
#include <set>
#include <vector>

#include "pipelines/scan/ScanTypes.h"

namespace Scanner::pipeline {

class FrameObsAccumulator {
public:
    /// laserBudgetBytes：激光帧缓存预算（字节）
    explicit FrameObsAccumulator(size_t laserBudgetBytes);

    /// laserXyz 为 host 拷贝（float xyz 三元组数组；空=无激光帧）。
    /// 预算内：拷入缓存槽，obs.laserCacheSlot=槽号；
    /// 超限/已降级：slot=kNoLaserSlot 且 degradedLaser 置位（整体不失败）。
    void push(FrameObs obs, const std::vector<float>& laserXyz);

    bool degradedLaser() const;              // 激光缓存是否已停累加（只置位，clear 复位）
    size_t laserBytesUsed() const;
    size_t frameCount() const;

    struct Snapshot {
        std::vector<FrameObs> obs;                        // 全部逐帧观测
        std::vector<std::vector<float>> laserFrames;      // 下标=槽号（kNoLaserSlot 越界不访问）
    };
    Snapshot snapshot() const;               // mutex 护拷贝导出（快照后只读语义）

    void clear();

    // —— 会话检查点（崩溃恢复链；ScanPipeline::save/restoreCheckpoint 消费）——
    /// 二进制落盘（obs 全量＋激光缓存全量＋降级标志；锁内直写）。格式：魔术字
    /// "JMWFOBS1"＋版本＋逐段计数。obs 数 MB 级、激光缓存可达预算上限（GB 级）——
    /// 写盘耗时与体积成正比，调方控制触发时机（stop/pause/周期）。
    Scanner::Result saveCheckpoint(const std::string& path) const;
    /// 读盘替换全量内容（清空后载入；预算仍用构造值）。坏档 fail 不崩。
    Scanner::Result loadCheckpoint(const std::string& path);
    /// 快照回填（restore 路径：替换 obsList_/laserFrames_/降级标志；预算不变）
    void replace(Snapshot&& snap, bool degraded);

    // —— 编辑账本（05 D4 双账本·obs 侧，实施计划 P2）——
    /// 按 globalId 剔除/恢复逐帧观测：snapshot()（GBA 输入）出口过滤已剔除
    /// id 的 MarkerObs（激光帧缓存不动）；removed=false 恢复（undo 路径）；
    /// 未知 id 幂等；id<0（链断观测）不参与剔除。clear() 同步清剔除集；
    /// 检查点 save/load 不含剔除集（崩溃恢复后需重放剔除——编辑会话与
    /// 检查点不并发的口径下可接受）。
    void excludeMarkerObs(const std::vector<int>& globalIds, bool removed);
    /// 当前剔除 id 数（观测口径：会话统计/排障）
    size_t excludedMarkerObsCount() const;

private:
    mutable std::mutex mu_;                  // push(写) 与 snapshot(读) 互斥
    std::vector<FrameObs> obsList_;
    std::vector<std::vector<float>> laserFrames_;
    std::set<int> excludedIds_;              // 编辑剔除的 marker globalId（GBA 出口过滤）
    size_t budgetBytes_;
    size_t usedBytes_ = 0;
    std::atomic<bool> degradedLaser_{false}; // 只置位不复位（clear 复位）
};

} // namespace Scanner::pipeline
