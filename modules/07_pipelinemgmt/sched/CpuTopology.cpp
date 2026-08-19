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
#include <windows.h>

namespace Scanner::pipeline::sched {

namespace {

/// API 失败时的核数兜底估计：硬件并发数（逻辑处理器数，Win10 前/Ex API 不可用场景）
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
        spdlog::warn("CpuTopology::detect: 查询物理核信息失败, err={}, 按纯 P 核兜底", GetLastError());
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
        spdlog::warn("CpuTopology::detect: 枚举物理核失败, err={}, 按纯 P 核兜底", GetLastError());
        info.pCores = fallbackCoreEstimate();
        return info;
    }
    int total = 0, perf = 0, eff = 0;
    std::vector<uint64_t> perfMasks, effMasks;
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
        // 全 0：非混合架构或 Win10 前系统 → 全部按 P 核计（warn 一次）
        info.hybrid = false;
        info.pCores = total;
        info.eCores = 0;
        spdlog::warn("CpuTopology::detect: 未区分出 P/E（物理核 {} 全为 EfficiencyClass 0），按纯 P 核处理", total);
    }
    return info;
}

void CpuTopology::pinThread(std::thread& t, uint64_t mask) {
    if (mask == 0) {
        return;  // 0 = 不绑
    }
    // 单 DWORD_PTR 不跨处理器组（目标机单组；>64 逻辑核需 group 扩展）
    if (SetThreadAffinityMask(t.native_handle(), static_cast<DWORD_PTR>(mask)) == 0) {
        spdlog::warn("CpuTopology: SetThreadAffinityMask(mask={:#x}) 失败, err={}", mask, GetLastError());
    }
}

void CpuTopology::setRealtime(std::thread& t) {
    const HANDLE handle = t.native_handle();
    if (!SetThreadPriority(handle, THREAD_PRIORITY_TIME_CRITICAL)) {
        if (!SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST)) {
            spdlog::warn("CpuTopology: SetThreadPriority(TimeCritical→Highest) 失败, err={}", GetLastError());
        }
    }
}

} // namespace Scanner::pipeline::sched
