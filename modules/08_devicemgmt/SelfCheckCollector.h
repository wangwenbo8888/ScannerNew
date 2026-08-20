#pragma once
// ============================================================================
// SelfCheckCollector.h — 上位机自检采集（H-T15；2026-08-20 08 设计 §5.2）
//
// CPU/GPU/内存/磁盘/线程心跳 → HealthMetrics 快照（base/types.h）。
// 阈值判定归 10-PerfMonitor——08 只采集；Honest 原则：取不到的字段留 -1（不编数）。
//   - CPU 占用：PDH \Processor(_Total)\% Processor Time；首次 collect 返 -1
//     （PdhCollectQueryData 两次采样才有差值，首调只建基线——口径钉死）
//   - CPU 温度：WMI MSAcpi 采集降级不采（工控机该接口多数不可用）→ 恒 -1
//   - GPU：NVML 动态加载（LoadLibraryA，无链接依赖）；无 nvml.dll → GPU 字段 -1
//   - 内存：GlobalMemoryStatusEx dwMemoryLoad；磁盘：GetDiskFreeSpaceExA GB 换算
//   - 帧率三件套（captureFps/processFps/dropRate）归 HardwareMonitor 填——本类不碰
// 头文件零 Win32 依赖：PDH/NVML 句柄域藏 .cpp 的 Native（pimpl，仅巡检线程触碰）。
// ============================================================================

#include "base/types.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Scanner::device {

class SelfCheckCollector {
public:
    SelfCheckCollector();
    ~SelfCheckCollector();

    SelfCheckCollector(const SelfCheckCollector&) = delete;
    SelfCheckCollector& operator=(const SelfCheckCollector&) = delete;

    // 工作盘（默认 C: 盘符——磁盘剩余检查对象；可换）
    void setDiskPath(const std::string& path);

    // 线程心跳注册制：业务线程周期 beat(name)；超时清单判定
    void registerHeartbeat(const std::string& name, int64_t timeoutMs = 5000);
    void beat(const std::string& name);                                   // 未注册名静默忽略（口径）
    std::vector<std::string> staleHeartbeats() const;                     // 内部时钟（巡检线程用）
    std::vector<std::string> staleHeartbeats(int64_t nowMs) const;        // 时钟注入（测试用，同 steady 域）

    // 一次性采集（巡检线程 1s 调）：PDH/NVML/Win32 尽力而为
    Scanner::HealthMetrics collect(int64_t nowMs) const;

    // 测试缝：磁盘/内存换算钩子（注入假实现替代 Win32；缺省真采）
    void setDiskProbe(std::function<double(const std::string&)> gbFree);  // 缺省 GetDiskFreeSpaceExA
    void setMemProbe(std::function<double()> percent);                    // 缺省 GlobalMemoryStatusEx

    // 心跳同源时基（steady 域毫秒）——beat 记账/staleHeartbeats()/测试注入口共用
    int64_t steadyNowMs() const;

private:
    struct Heartbeat {
        int64_t lastBeatMs{0};
        int64_t timeoutMs{5000};
    };

    // Win32/PDH/NVML 句柄域（.cpp 内定义；仅巡检线程触碰——单线程域，不加锁）
    struct Native;
    std::unique_ptr<Native> native_;

    // 采集辅助（.cpp 实装；静态成员方可用私有 Native 形参）
    static double cpuPercentNative(Native& n);
    static void   gpuProbeNative(Native& n, double& tempC, double& memPercent);

    mutable std::mutex hbMutex_;          // beat 被各业务线程并发调，map 读写互斥
    std::map<std::string, Heartbeat> heartbeats_;

    std::string diskPath_ = "C:\\";
    std::function<double(const std::string&)> diskProbe_;   // 空 → Win32 真采
    std::function<double()> memProbe_;                      // 空 → Win32 真采
};

} // namespace Scanner::device
