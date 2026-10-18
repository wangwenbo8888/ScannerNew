// ============================================================================
// SelfCheckCollector.cpp — 上位机自检采集实现（H-T15）
//
// 各源实装口径（设计 §5.2）：
//   内存  GlobalMemoryStatusEx → dwMemoryLoad（百分比，0-100）
//   磁盘  GetDiskFreeSpaceExA → AvailableToCaller（0 则 Free）/ 2^30 GB
//   CPU%  PDH \Processor(_Total)\% Processor Time（英文计数器名防本地化）；
//         首次 collect 只建基线样本返 -1，第二秒起出值；_Total 微溢出钳 [0,100]
//   CPU温度 WMI MSAcpi_ThermalZoneTemperature —— 降级不采（恒 -1）：
//         工控机该接口多数不实现，YAGNI 留给后续真有需求再接
//   GPU   NVML 动态加载（LoadLibraryA("nvml.dll") 一次缓存；失败置 tried 不再重试）；
//         本地 typedef 照官方 nvml.h 签名声明，不 include nvml.h（无链接依赖）；
//         设备 0（单卡口径）→ 温度℃ + 显存 used/total 百分比；任一步败对应 -1
//   心跳  map + mutex（beat 多业务线程并发）；注册即视为存活一拍；
//         beat 未注册名静默忽略（口径钉死）；时钟统一 steady 域毫秒
// ============================================================================

#include "modules/DeviceManager/SelfCheckCollector.h"

// Win32 头只在本 .cpp（纪律：LEAN_AND_MEAN/NOMINMAX，不污染他处）
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <pdh.h>

#include <chrono>
#include <utility>

namespace Scanner::device {

// —— NVML 本地声明（不 include nvml.h；签名照官方头照抄，ABI：x64 单一调用约定）——
using NvmlDevice = struct nvmlDevice_st*;
using NvmlReturn = int;  // nvmlReturn_t 枚举按 int 传值；NVML_SUCCESS == 0
struct NvmlMemory {      // 现行 nvmlMemory_t 布局（total/reserved/used，字长 ull）
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long used;
};
struct NvmlUtilization { // nvmlUtilization_t 布局（261018 增：SM/显存/编解利用率）
    unsigned int gpu;      // SM 利用率 %（相机预览监视显示）
    unsigned int memory;
    unsigned int encoder;
    unsigned int decoder;
};
using PFN_nvmlInit_v2               = NvmlReturn(*)(void);
using PFN_nvmlDeviceGetHandleByIndex = NvmlReturn(*)(unsigned int, NvmlDevice*);
using PFN_nvmlDeviceGetTemperature   = NvmlReturn(*)(NvmlDevice, int, unsigned int*);  // int=传感器类型，GPU=0
using PFN_nvmlDeviceGetMemoryInfo    = NvmlReturn(*)(NvmlDevice, NvmlMemory*);
using PFN_nvmlDeviceGetUtilizationRates = NvmlReturn(*)(NvmlDevice, NvmlUtilization*);

// —— 句柄域（仅巡检线程触碰：collect 由 1s 巡检线程单线程调，不加锁）——
struct SelfCheckCollector::Native {
    // PDH（CPU 占用）
    PDH_HQUERY   pdhQuery{nullptr};
    PDH_HCOUNTER pdhCpu{nullptr};
    bool         pdhSeeded{false};   // 首次基线样本已采（本实例首调 collect 返 -1 口径）

    // PDH（261018 CUDA 引擎利用率——任务管理器同源 WDDM 计数器，多进程实例求和）
    PDH_HQUERY   pdhGpuQuery{nullptr};
    PDH_HCOUNTER pdhGpuCuda{nullptr};

    // NVML（GPU）
    bool       nvmlTried{false};     // LoadLibraryA 只试一次，失败不再重试
    HMODULE    nvmlLib{nullptr};
    PFN_nvmlInit_v2                nvmlInit{nullptr};
    PFN_nvmlDeviceGetHandleByIndex nvmlGetHandle{nullptr};
    PFN_nvmlDeviceGetTemperature   nvmlGetTemp{nullptr};
    PFN_nvmlDeviceGetMemoryInfo    nvmlGetMem{nullptr};
    PFN_nvmlDeviceGetUtilizationRates nvmlGetUtil{nullptr};   // 261018（可选——旧驱动缺此出口不挡温度/显存）
    NvmlDevice  nvmlDev{nullptr};    // 设备 0（单卡工控机口径）

    ~Native() {
        if (pdhQuery) PdhCloseQuery(pdhQuery);
        if (pdhGpuQuery) PdhCloseQuery(pdhGpuQuery);
        if (nvmlLib) FreeLibrary(nvmlLib);
    }
};

// —— Win32 缺省探针（可被 setMemProbe/setDiskProbe 测试缝替换）——
static double win32MemPercent() {
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (!GlobalMemoryStatusEx(&st)) return -1.0;
    return static_cast<double>(st.dwMemoryLoad);
}

static double win32DiskFreeGB(const std::string& path) {
    ULARGE_INTEGER availToCaller{}, total{}, free{};
    if (!GetDiskFreeSpaceExA(path.c_str(), &availToCaller, &total, &free)) return -1.0;
    const unsigned long long bytes = availToCaller.QuadPart ? availToCaller.QuadPart : free.QuadPart;
    return static_cast<double>(bytes) / 1073741824.0;  // 2^30 → GB
}

// CPU 占用：PDH 两次采样才有差值——首次只建基线返 -1
double SelfCheckCollector::cpuPercentNative(Native& n) {
    if (!n.pdhQuery) {
        if (PdhOpenQueryA(nullptr, 0, &n.pdhQuery) != ERROR_SUCCESS) {
            n.pdhQuery = nullptr;
            return -1.0;
        }
        if (PdhAddEnglishCounterA(n.pdhQuery, "\\Processor(_Total)\\% Processor Time",
                                  0, &n.pdhCpu) != ERROR_SUCCESS) {
            PdhCloseQuery(n.pdhQuery);
            n.pdhQuery = nullptr;
            n.pdhCpu = nullptr;
            return -1.0;
        }
    }
    if (PdhCollectQueryData(n.pdhQuery) != ERROR_SUCCESS) return -1.0;
    if (!n.pdhSeeded) {
        n.pdhSeeded = true;   // 本次即基线样本，无差值可算——Honest 返 -1
        return -1.0;
    }
    PDH_FMT_COUNTERVALUE fmt{};
    if (PdhGetFormattedCounterValue(n.pdhCpu, PDH_FMT_DOUBLE, nullptr, &fmt) != ERROR_SUCCESS)
        return -1.0;
    double v = fmt.doubleValue;
    if (v < 0.0) v = 0.0;     // _Total 偶见微溢出，钳回 [0,100]
    if (v > 100.0) v = 100.0;
    return v;
}

// 261018 CUDA 引擎利用率（WDDM 口径＝任务管理器同源）：PDH 通配计数器
// \GPU Engine(*engtype_Cuda)\Utilization Percentage，多进程实例求和（每实例
// 0-100，任务管理器显示的就是各实例合计）。任一步败（旧系统无 GPU 计数器/
// 首次无实例）返 -1——上层回退 NVML SM 口径
double SelfCheckCollector::cudaEnginePercentNative(Native& n) {
    if (!n.pdhGpuQuery) {
        if (PdhOpenQueryA(nullptr, 0, &n.pdhGpuQuery) != ERROR_SUCCESS) {
            n.pdhGpuQuery = nullptr;
            return -1.0;
        }
        if (PdhAddEnglishCounterA(n.pdhGpuQuery,
                                  "\\GPU Engine(*engtype_Cuda)\\Utilization Percentage",
                                  0, &n.pdhGpuCuda) != ERROR_SUCCESS) {
            PdhCloseQuery(n.pdhGpuQuery);
            n.pdhGpuQuery = nullptr;
            n.pdhGpuCuda = nullptr;
            return -1.0;
        }
    }
    if (PdhCollectQueryData(n.pdhGpuQuery) != ERROR_SUCCESS) return -1.0;
    // 数组取值：两次调用模式（先探体积再取）；实例集随进程启停变化
    DWORD bufBytes = 0, itemCount = 0;
    auto st = PdhGetFormattedCounterArrayW(n.pdhGpuCuda, PDH_FMT_DOUBLE,
                                           &bufBytes, &itemCount, nullptr);
    if (st != PDH_MORE_DATA || bufBytes == 0) return -1.0;
    std::vector<unsigned char> buf(bufBytes);
    st = PdhGetFormattedCounterArrayW(n.pdhGpuCuda, PDH_FMT_DOUBLE,
                                      &bufBytes, &itemCount,
                                      reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data()));
    if (st != ERROR_SUCCESS || itemCount == 0) return -1.0;
    const auto* items = reinterpret_cast<const PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
    double sum = 0.0;
    int valid = 0;
    for (DWORD i = 0; i < itemCount; ++i) {
        const double v = items[i].FmtValue.doubleValue;
        if (v >= 0.0 && v <= 100.0) { sum += v; ++valid; }
    }
    if (valid == 0) return -1.0;
    if (sum > 100.0) sum = 100.0;   // 多实例合计理论可超（多引擎）——钳到满载
    return sum;
}

// GPU：lazy 加载一次；任一步败对应字段留 -1
void SelfCheckCollector::gpuProbeNative(Native& n, double& tempC, double& memPercent,
                                        double& utilPercent) {
    tempC = -1.0;
    memPercent = -1.0;
    utilPercent = -1.0;
    if (!n.nvmlTried) {
        n.nvmlTried = true;
        n.nvmlLib = LoadLibraryA("nvml.dll");
        if (!n.nvmlLib) return;
        n.nvmlInit      = reinterpret_cast<PFN_nvmlInit_v2>(
            GetProcAddress(n.nvmlLib, "nvmlInit_v2"));
        n.nvmlGetHandle = reinterpret_cast<PFN_nvmlDeviceGetHandleByIndex>(
            GetProcAddress(n.nvmlLib, "nvmlDeviceGetHandleByIndex"));
        n.nvmlGetTemp   = reinterpret_cast<PFN_nvmlDeviceGetTemperature>(
            GetProcAddress(n.nvmlLib, "nvmlDeviceGetTemperature"));
        n.nvmlGetMem    = reinterpret_cast<PFN_nvmlDeviceGetMemoryInfo>(
            GetProcAddress(n.nvmlLib, "nvmlDeviceGetMemoryInfo"));
        n.nvmlGetUtil   = reinterpret_cast<PFN_nvmlDeviceGetUtilizationRates>(
            GetProcAddress(n.nvmlLib, "nvmlDeviceGetUtilizationRates"));
        if (!n.nvmlInit || !n.nvmlGetHandle || !n.nvmlGetTemp || !n.nvmlGetMem) return;
        if (n.nvmlInit() != 0) return;
        if (n.nvmlGetHandle(0, &n.nvmlDev) != 0) {
            n.nvmlDev = nullptr;
            return;
        }
    }
    if (!n.nvmlLib || !n.nvmlDev) return;

    unsigned int t = 0;
    if (n.nvmlGetTemp(n.nvmlDev, 0 /*NVML_TEMPERATURE_GPU*/, &t) == 0)
        tempC = static_cast<double>(t);

    NvmlMemory mem{};
    if (n.nvmlGetMem(n.nvmlDev, &mem) == 0 && mem.total > 0)
        memPercent = static_cast<double>(mem.used) * 100.0 / static_cast<double>(mem.total);

    NvmlUtilization util{};
    if (n.nvmlGetUtil && n.nvmlGetUtil(n.nvmlDev, &util) == 0)
        utilPercent = static_cast<double>(util.gpu);
}

// ============================================================================
// 公开接口
// ============================================================================

SelfCheckCollector::SelfCheckCollector() : native_(std::make_unique<Native>()) {}

SelfCheckCollector::~SelfCheckCollector() = default;

void SelfCheckCollector::setDiskPath(const std::string& path) { diskPath_ = path; }

void SelfCheckCollector::registerHeartbeat(const std::string& name, int64_t timeoutMs) {
    const int64_t now = steadyNowMs();
    std::lock_guard<std::mutex> lock(hbMutex_);
    heartbeats_[name] = Heartbeat{now, timeoutMs};   // 注册即视为存活一拍
}

void SelfCheckCollector::beat(const std::string& name) {
    const int64_t now = steadyNowMs();
    std::lock_guard<std::mutex> lock(hbMutex_);
    auto it = heartbeats_.find(name);
    if (it == heartbeats_.end()) return;   // 未注册名静默忽略（口径钉死）
    it->second.lastBeatMs = now;
}

std::vector<std::string> SelfCheckCollector::staleHeartbeats() const {
    return staleHeartbeats(steadyNowMs());
}

std::vector<std::string> SelfCheckCollector::staleHeartbeats(int64_t nowMs) const {
    std::lock_guard<std::mutex> lock(hbMutex_);
    std::vector<std::string> stale;
    for (const auto& kv : heartbeats_) {
        if (nowMs - kv.second.lastBeatMs > kv.second.timeoutMs) stale.push_back(kv.first);
    }
    return stale;   // map 天然按名有序——名单确定性输出
}

Scanner::HealthMetrics SelfCheckCollector::collect(int64_t nowMs) const {
    Scanner::HealthMetrics m;   // 缺省全 -1（Honest：取不到不编数）

    m.memPercent  = memProbe_  ? memProbe_()  : win32MemPercent();
    m.diskFreeGB  = diskProbe_ ? diskProbe_(diskPath_) : win32DiskFreeGB(diskPath_);
    m.cpuPercent  = cpuPercentNative(*native_);
    m.cpuTempC    = -1.0;      // WMI 温度降级不采（口径钉死，见文件头）
    double nvmlUtil = -1.0;
    gpuProbeNative(*native_, m.gpuTempC, m.gpuMemPercent, nvmlUtil);
    // 261018 CUDA 利用率口径（用户裁定）：优先 WDDM Cuda 引擎（任务管理器同源），
    // 拿不到回退 NVML SM 忙时率
    m.gpuUtilPercent = cudaEnginePercentNative(*native_);
    if (m.gpuUtilPercent < 0.0) m.gpuUtilPercent = nvmlUtil;

    // 帧率三件套（captureFps/processFps/dropRate）归 HardwareMonitor 填——留 -1
    m.timestampMs = nowMs;
    return m;
}

void SelfCheckCollector::setDiskProbe(std::function<double(const std::string&)> gbFree) {
    diskProbe_ = std::move(gbFree);
}

void SelfCheckCollector::setMemProbe(std::function<double()> percent) {
    memProbe_ = std::move(percent);
}

int64_t SelfCheckCollector::steadyNowMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace Scanner::device
