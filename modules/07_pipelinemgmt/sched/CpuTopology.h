#pragma once
// ============================================================================
// CpuTopology.h — CPU 拓扑探测 + 线程亲和/优先级工具（调度底座）
// detect(): GetLogicalProcessorInformationEx(RelationProcessorCore) 枚举物理核，
//           按 EfficiencyClass 区分 P/E（官方语义：值越大性能越高，仅异构系统
//           非零；Intel 混合架构 1=Performance(P核), 0=Efficiency(E核)）。
//           全 0（非混合/Win10 前）或 API 失败 → hybrid=false、全部按 P 核计。
// computeLanes(): 流水线数 X 纯函数（见 .cpp）。
// pinThread()/setRealtime(): Windows 绑核/提优先级，失败仅 spdlog warn 不抛。
// ============================================================================
#include <cstdint>
#include <thread>

namespace Scanner::pipeline::sched {

struct TopologyInfo {
    int pCores{0};        // 物理 P 核数（不含超线程兄弟）
    int eCores{0};        // 物理 E 核数
    bool hybrid{false};   // 是否成功区分 P/E
};

/// 流水线数 X 纯函数：override>0 直接返回；否则 clamp(min(P-1,E),1,8)；
/// E==0（无 E 核/探测失败）退化：clamp(min(P-1,P/2),1,8) —— P 核机器取半保守值
int computeLanes(int pCores, int eCores, int overrideLanes);

class CpuTopology {
public:
    /// 探测物理核拓扑（Windows 10+ 才有 EfficiencyClass；失败安全退化为纯 P 核）
    static TopologyInfo detect();

    /// mask=0 不绑；失败 spdlog warn 不抛
    /// （单 DWORD_PTR 不跨处理器组，>64 逻辑核需 group 扩展，目标机单组）
    static void pinThread(std::thread& t, uint64_t mask);

    /// TIME_CRITICAL；失败降 HIGHEST，再失败 warn 不抛
    static void setRealtime(std::thread& t);
};

} // namespace Scanner::pipeline::sched
