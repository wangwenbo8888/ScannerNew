// ============================================================================
// CpuTopology.cpp — CPU 拓扑探测与线程亲和实现（Windows x64）
// ============================================================================
#ifndef NOMINMAX
#define NOMINMAX  // windows.h min/max 宏与 std::min/std::max 冲突
#endif

#include "sched/CpuTopology.h"

#include <algorithm>
#include <vector>

#include <spdlog/spdlog.h>
#include "jmw_logging.h"
#include <windows.h>

namespace Scanner::pipeline::sched {

namespace {

/// API 失败时的核数兜底估计（降级路径）：hardware_concurrency 返回的是
/// 逻辑核数（含超线程，约为物理核 2 倍）——降级时按逻辑核数估计，pCores
/// 偏多为已知保守偏差（computeLanes 会钳位，不会超发 lane）
int fallbackCoreEstimate() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 1;
}

} // namespace

int computeLanes(int pCores, int eCores, int overrideLanes) {
    if (overrideLanes > 0) {
        return overrideLanes;  // 人工覆盖：原样返回不钳位
    }
    // E==0（无 E 核/探测失败）退化：P 核取半保守值
    const int lanes = (eCores > 0) ? std::min(pCores - 1, eCores)
                                   : std::min(pCores - 1, pCores / 2);
    return std::max(1, std::min(lanes, 8));
}

TopologyInfo CpuTopology::detect() {
    TopologyInfo info;
    // 首调取所需尺寸：buffer=nullptr 必失败且 GetLastError=ERROR_INSUFFICIENT_BUFFER
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
        JMW_LOG_WARN("07-CpuTopology", "CpuTopology::detect: 查询物理核信息失败, err={}, 按纯 P 核兜底", GetLastError());
        info.pCores = fallbackCoreEstimate();
        return info;
    }
    // vector 堆分配满足结构体对齐；枚举覆盖全部处理器组的物理核
    // （每条记录一个物理核，GroupCount 恒 1，跨组也逐条计数）
    std::vector<BYTE> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()),
            &bytes)) {
        JMW_LOG_WARN("07-CpuTopology", "CpuTopology::detect: 枚举物理核失败, err={}, 按纯 P 核兜底", GetLastError());
        info.pCores = fallbackCoreEstimate();
        return info;
    }
    int total = 0, perf = 0, eff = 0;
    std::vector<uint64_t> perfMasks, effMasks;
    std::vector<uint64_t> allMasks;   // 同构 40/60 分组用（物理核粒度，含 SMT 对）
    DWORD offset = 0;
    while (offset < bytes) {
        auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (rec->Relationship != RelationProcessorCore || rec->Size == 0) {
            break;  // 防御：请求即 Core，异常记录停遍历
        }
        ++total;
        // 每物理核掩码（GroupCount 恒 1，目标机单处理器组；防御性下标检查）
        if (rec->Processor.GroupCount >= 1) {
            const uint64_t mask = static_cast<uint64_t>(rec->Processor.GroupMask[0].Mask);
            allMasks.push_back(mask);
            // 官方语义：值越大性能越高，仅异构系统非零 → 非零=P 核，0=E 核
            if (rec->Processor.EfficiencyClass != 0) {
                ++perf;
                perfMasks.push_back(mask);
            } else {
                ++eff;
                effMasks.push_back(mask);
            }
        }
        offset += rec->Size;
    }
    if (perf > 0) {
        // 异构：P/E 已区分（Intel 混合架构 P=1, E=0）；掩码一并导出供绑核
        info.hybrid = true;
        info.pCores = perf;
        info.eCores = eff;
        info.pMasks = std::move(perfMasks);
        info.eMasks = std::move(effMasks);
    } else {
        // 全同构（EfficiencyClass 全 0）——逻辑分组（2026-09-01 用户口径）：
        // 按物理核 40%/60% 分 P/E（P:E=1:1.5，对齐主流大小核比值；SMT 对
        // 天然同组——mask 即物理核粒度）。此前 eCores=0 走 computeLanes 回退
        // 且 masks 全空=绑核失效；分组后 lanes=min(P-1,E) 且绑核生效
        const int pCount = std::max(1, total * 2 / 5);
        info.hybrid = false;
        info.pCores = pCount;
        info.eCores = total - pCount;
        for (int i = 0; i < total; ++i) {
            (i < pCount ? perfMasks : effMasks).push_back(allMasks[static_cast<size_t>(i)]);
        }
        info.pMasks = std::move(perfMasks);
        info.eMasks = std::move(effMasks);
        JMW_LOG_INFO("07-CpuTopology",
                     "CpuTopology::detect: 同构 CPU {} 核——40/60 逻辑分组 P={} E={}（lanes=min(P-1,E)）",
                     total, info.pCores, info.eCores);
    }
    return info;
}

void CpuTopology::pinThread(std::thread& t, uint64_t mask) {
    if (mask == 0) {
        return;  // 0 = 不绑
    }
    // 单 DWORD_PTR 不跨处理器组（目标机单组；>64 逻辑核需 group 扩展）
    if (SetThreadAffinityMask(t.native_handle(), static_cast<DWORD_PTR>(mask)) == 0) {
        JMW_LOG_WARN("07-CpuTopology", "CpuTopology: SetThreadAffinityMask(mask={:#x}) 失败, err={}", mask, GetLastError());
    }
}

void CpuTopology::setRealtime(std::thread& t) {
    const HANDLE handle = t.native_handle();
    if (!SetThreadPriority(handle, THREAD_PRIORITY_TIME_CRITICAL)) {
        if (!SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)) {
            JMW_LOG_WARN("07-CpuTopology", "CpuTopology: SetThreadPriority TIME_CRITICAL 与 HIGHEST 均失败, err={}", GetLastError());
        }
    }
}

} // namespace Scanner::pipeline::sched
