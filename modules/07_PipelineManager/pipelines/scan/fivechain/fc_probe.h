#pragma once
// fc_probe.h — 五链中间结果抽检转储（design: docs/plans/2026-11-03-五链中间结果抽检转储-design.md）
// 帧号命中 config/fc_dump.json probe.frames 时，把该帧全链中间结果按 scan_replay
// 沙箱命名落盘 diagnostics/fc_probe/cycle_NNNN/。默认关＝零开销。
//
// 画法/字段遍历照抄沙箱 harness.cpp dumpCycle 段（同仓库移植）——保证在线/
// 离线同文件名同语义，可直接 diff。全部落盘经 scr::artifacts 工具箱
// （artifacts.h/.cpp，上一任务搬入）；算子与公共头零改动。
#ifndef JMW_PM_FIVECHAIN_FC_PROBE_H
#define JMW_PM_FIVECHAIN_FC_PROBE_H

#include <cstdint>
#include <set>
#include <string>
#include "harness.h"

namespace jmw::probe {

/// probe 配置（config/fc_dump.json 的 probe 段；任何异常静默回默认＝关）
struct ProbeConfig {
    bool enabled = false;
    long maxFrames = 1000;          // app 侧限量停采阈值（probe 侧仅透传/记录）
    std::set<long> frames;          // 挑帧帧号集合（1 起，流水线处理序号）
    static ProbeConfig load();      // 读 config/fc_dump.json 的 probe 段
};

/// 挑帧落盘器：fc_executor 在段出口按帧调用（Task 4 接线）。
/// 非命中帧仅做 set 查找＝零开销；命中帧 GPU→宿主拷贝＋写盘（几 MB 级）。
class FrameProbe {
public:
    void configure(const ProbeConfig& c) { cfg_ = c; }
    bool enabled() const { return cfg_.enabled; }
    size_t frameCount() const { return cfg_.frames.size(); }
    /// idx 0 起，帧号=idx+1；命中 probe.frames 则落该帧全套中间产物
    void maybeDump(long idx, const scr::CycleProductsCompat& p);
    /// probe 开时逐帧追加 rt.csv（列格式同 JMW_FC_FRAME_DUMPS 的 rt.csv；
    /// 帧列＝idx+1，与 cycle_NNNN 目录号一致）
    void appendRt(long idx, const scr::CycleProductsCompat& p);

private:
    ProbeConfig cfg_;
};

} // namespace jmw::probe

#endif // JMW_PM_FIVECHAIN_FC_PROBE_H
