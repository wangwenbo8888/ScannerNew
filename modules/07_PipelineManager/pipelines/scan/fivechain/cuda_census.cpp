// ============================================================================
// cuda_census v3 - CUPTI runtime API census. See cuda_census.h.
// Column order == CSV header order == StageRow field order (single source of truth).
// ============================================================================
#include "cuda_census.h"

#include <cupti.h>
// functionParams＝meta 值存储（非指针数组）——方向分类须 cast 成 <API>_params 窄读
// kind（二轮-census修复规格定案；旧「*pr[3] 解引用」写法会崩，红队C 证伪）。
// <generated_cuda_runtime_api_meta.h> 由 cupti.h 传递包含（其自身无 include guard，
// 显式二次包含会 C2011 重定义）——勿再显式 include。

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
// 【分配现场抓拍】JMW_CENSUS_TRACE=1 时对每次设备分配抓 8 帧调用栈并符号化
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace jmw {
namespace census {

namespace {

struct StageRow {
    long launch = 0, mcH2D = 0, mcD2H = 0, mcD2D = 0;
    long mcaH2D = 0, mcaD2H = 0, mcaD2D = 0;
    // 【v3 旁列】2D 拷贝补计——与 .h Counters 同锚点（toCounters 整拷），不入 total
    long mc2D = 0, mca2D = 0;
    long memset = 0, memsetA = 0, sym = 0, symA = 0;
    long hostReg = 0, hostUnreg = 0, mallocHost = 0, freeHost = 0;
    // 【v3 旁列】cudaHostAlloc 补计——不入 total（四轮算术终审：入 mallocHost 列则
    // C0 +7×实例数 破「A 批 C2 不变」门；红队C 原案「归 mallocHost」据此改旁列）
    long hostAlloc = 0;
    long mallocAsync = 0, freeAsync = 0;
    long graphLaunch = 0, graphInst = 0, beginCap = 0, endCap = 0;
    long graphUpload = 0, graphExecUpd = 0, hostFunc = 0;
    long streamCreate = 0, streamDestroy = 0, streamQuery = 0;
    long streamSync = 0, devSync = 0;
    long evCreate = 0, evDestroy = 0, evRecord = 0, evQuery = 0, evSync = 0;
    long evElapsed = 0, evWait = 0;
    long malloc = 0, free = 0;
    long total() const {
        return launch + mcH2D + mcD2H + mcD2D + mcaH2D + mcaD2H + mcaD2D +
               memset + memsetA + sym + symA + hostReg + hostUnreg +
               mallocHost + freeHost + mallocAsync + freeAsync +
               graphLaunch + graphInst + beginCap + endCap + graphUpload +
               graphExecUpd + hostFunc + streamCreate + streamDestroy +
               streamQuery + streamSync + devSync + evCreate + evDestroy +
               evRecord + evQuery + evSync + evElapsed + evWait + malloc + free;
    }
    void toCounters(Counters& c) const {
        std::memcpy(&c, this, sizeof(StageRow));
    }
};
static_assert(sizeof(StageRow) == sizeof(Counters), "StageRow/Counters isomorphic");

std::mutex g_mu;
std::map<std::string, StageRow> g_byStage;
std::atomic<bool> g_enabled{false};
std::atomic<bool> g_subscribed{false};
std::atomic<long> g_frames{0};
thread_local StageRow* t_cnt = nullptr;
thread_local const char* t_stage = nullptr;

StageRow* acquireRow(const char* name) {
    static thread_local std::map<const char*, StageRow*> t_cache;
    auto it = t_cache.find(name);
    if (it != t_cache.end()) return it->second;
    std::lock_guard<std::mutex> lk(g_mu);
    StageRow* row = &g_byStage[name];
    t_cache[name] = row;
    return row;
}

#ifdef _WIN32
static FILE* g_traceFile = nullptr;
static void traceAlloc(const char* api, size_t sz) {
    if (!g_traceFile) return;
    void* frames[16];
    USHORT n = CaptureStackBackTrace(0, 16, frames, NULL);
    fprintf(g_traceFile, "[%s] sz=%zu\n", api, sz);
    for (USHORT i = 1; i < n; ++i) {   // 跳过探针自身帧
        char sym[256] = {0};
        DWORD64 disp = 0;
        auto si = reinterpret_cast<SYMBOL_INFO*>(sym);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = sizeof(sym) - sizeof(SYMBOL_INFO);
        if (SymFromAddr(GetCurrentProcess(), (DWORD64)frames[i], &disp, si)) {
            fprintf(g_traceFile, "  #%u %s+0x%llx\n", (unsigned)i, si->Name,
                    (unsigned long long)disp);
        } else {
            HMODULE m = NULL;
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                  (LPCWSTR)frames[i], &m) && m) {
                wchar_t wpath[MAX_PATH] = {0};
                GetModuleFileNameW(m, wpath, MAX_PATH);
                const wchar_t* base = wcsrchr(wpath, L'\\');
                base = base ? base + 1 : wpath;
                char mname[MAX_PATH] = {0};
                WideCharToMultiByte(CP_UTF8, 0, base, -1, mname, MAX_PATH, NULL, NULL);
                fprintf(g_traceFile, "  #%u %s!+0x%llx\n", (unsigned)i, mname,
                        (unsigned long long)((BYTE*)frames[i] - (BYTE*)m));
            }
        }
    }
    fflush(g_traceFile);
}
#define JMW_TRACE_ALLOC(api, sz) if (g_traceFile) traceAlloc(api, (size_t)(sz))

// 【P2-0·崩溃现场打印机】未处理异常时打印 32 帧符号化堆栈到 stderr（复用 dbghelp）
// 【案件三增强】追加寄存器现场＋AV 访问目标（ExceptionInformation），离线对号用
static LONG WINAPI CrashPrinter(EXCEPTION_POINTERS* ep) {
    void* frames[32];
    USHORT n = CaptureStackBackTrace(0, 32, frames, NULL);
    fprintf(stderr, "\n[JMW_CRASH] code=0x%08lx addr=%p\n",
            ep->ExceptionRecord->ExceptionCode,
            ep->ExceptionRecord->ExceptionAddress);
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
        ep->ExceptionRecord->NumberParameters >= 2) {
        fprintf(stderr, "  AV op=%s target=0x%llx\n",
                ep->ExceptionRecord->ExceptionInformation[0] == 0 ? "read" :
                ep->ExceptionRecord->ExceptionInformation[0] == 1 ? "write" : "dep",
                (unsigned long long)ep->ExceptionRecord->ExceptionInformation[1]);
    }
#if defined(_M_X64) || defined(__x86_64__)
    if (ep->ContextRecord) {
        CONTEXT* c = ep->ContextRecord;
        fprintf(stderr, "  regs rip=%llx rbp=%llx rsp=%llx\n"
                        "       rax=%llx rbx=%llx rcx=%llx rdx=%llx\n"
                        "       rsi=%llx rdi=%llx r8=%llx r9=%llx\n"
                        "       r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n",
                (unsigned long long)c->Rip, (unsigned long long)c->Rbp,
                (unsigned long long)c->Rsp,
                (unsigned long long)c->Rax, (unsigned long long)c->Rbx,
                (unsigned long long)c->Rcx, (unsigned long long)c->Rdx,
                (unsigned long long)c->Rsi, (unsigned long long)c->Rdi,
                (unsigned long long)c->R8, (unsigned long long)c->R9,
                (unsigned long long)c->R10, (unsigned long long)c->R11,
                (unsigned long long)c->R12, (unsigned long long)c->R13,
                (unsigned long long)c->R14, (unsigned long long)c->R15);
    }
#endif
    for (USHORT i = 0; i < n; ++i) {
        char sym[256] = {0};
        DWORD64 disp = 0;
        auto si = reinterpret_cast<SYMBOL_INFO*>(sym);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = sizeof(sym) - sizeof(SYMBOL_INFO);
        HMODULE m = NULL;
        wchar_t wpath[MAX_PATH] = {0};
        char mname[MAX_PATH] = "??";
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCWSTR)frames[i], &m) && m) {
            GetModuleFileNameW(m, wpath, MAX_PATH);
            const wchar_t* b = wcsrchr(wpath, L'\\');
            b = b ? b + 1 : wpath;
            WideCharToMultiByte(CP_UTF8, 0, b, -1, mname, MAX_PATH, NULL, NULL);
        }
        if (SymFromAddr(GetCurrentProcess(), (DWORD64)frames[i], &disp, si)) {
            fprintf(stderr, "  #%u %s!%s+0x%llx\n", (unsigned)i, mname, si->Name,
                    (unsigned long long)disp);
        } else {
            fprintf(stderr, "  #%u %s!+0x%llx\n", (unsigned)i, mname,
                    (unsigned long long)(m ? (BYTE*)frames[i] - (BYTE*)m : 0));
        }
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;   // 交回系统默认处理（不吞异常）
}
static void installCrashPrinter() {
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), NULL, TRUE);
    SetUnhandledExceptionFilter(CrashPrinter);
}
// 常驻安装（静态初始化器：exe 与 gtest 二进制加载即装，不依赖 init()/env）
static const bool s_crashPrinterInstalled = (installCrashPrinter(), true);
#else
#define JMW_TRACE_ALLOC(api, sz)
#endif

// 【CUPTI 回调纪律】本回调在 CUPTI 内部上下文执行——严禁发起任何 CUDA 运行时/驱动
// API 调用（死锁/递归回调未定义行为）；只允许纯内存读（meta cast 窄读）＋原子计数＋
// fprintf。TRACE 分支同样只做栈回溯与写文件。
void CUPTIAPI callback(void* /*userdata*/, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const void* cbData) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    const CUpti_CallbackData* cb = static_cast<const CUpti_CallbackData*>(cbData);
    if (cb->callbackSite != CUPTI_API_ENTER) return;
    StageRow* c = t_cnt;
    if (!c) c = acquireRow("misc");
    const void* const* pr = static_cast<const void* const*>(cb->functionParams);
    switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: ++c->launch; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000: {
            // meta cast 窄读 kind（旧 reinterpret_cast<intptr_t>(pr[3]) 整读 8B
            // ＝kind(4B)+padding 垃圾 → 恒落 default＝方向分类病根）
            auto* p = static_cast<const cudaMemcpy_v3020_params*>(cb->functionParams);
            switch (p ? p->kind : cudaMemcpyHostToDevice) {
                case cudaMemcpyHostToDevice: ++c->mcH2D; break;
                case cudaMemcpyDeviceToHost: ++c->mcD2H; break;
                case cudaMemcpyDeviceToDevice: ++c->mcD2D; break;
                default: ++c->mcH2D; break;   // 保守多计防虚降
            }
        } break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000: {
            auto* p = static_cast<const cudaMemcpyAsync_v3020_params*>(cb->functionParams);
            switch (p ? p->kind : cudaMemcpyHostToDevice) {
                case cudaMemcpyHostToDevice: ++c->mcaH2D; break;
                case cudaMemcpyDeviceToHost: ++c->mcaD2H; break;
                case cudaMemcpyDeviceToDevice: ++c->mcaD2D; break;
                default: ++c->mcaH2D; break;   // 保守多计防虚降
            }
        } break;
        // 【v3 旁列】2D 拷贝补计（纯计数不读 kind；2D 方向在参数表第 7 字段偏移 48，
        // stage 行已天然区分生产位点，无需分向）
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000: ++c->mc2D; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000: ++c->mca2D; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020: ++c->memset; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020: ++c->memsetA; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020: ++c->sym; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020: ++c->symA; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020: ++c->sym; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020: ++c->symA; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000: ++c->hostReg; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaHostUnregister_v4000: ++c->hostUnreg; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020: ++c->mallocHost; break;
        // 【v3 旁列】steger 锁页缓冲分配侧补计（steger_fast_impl.cu:1188/1359-1364 双实例）
        case CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020: ++c->hostAlloc; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020: ++c->freeHost; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020:
            ++c->mallocAsync; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020:
            ++c->freeAsync; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000:
            ++c->graphLaunch; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040:
            ++c->graphInst; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_ptsz_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_v12030:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_ptsz_v12030:
            ++c->beginCap; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_ptsz_v10000:
            ++c->endCap; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_ptsz_v10000:
            ++c->graphUpload; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020: ++c->graphExecUpd; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000:
            ++c->hostFunc; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050:
            ++c->streamCreate; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050:
            ++c->streamDestroy; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000:
            ++c->streamQuery; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020: ++c->streamSync; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020: ++c->devSync; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020:
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020:
            ++c->evCreate; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020: ++c->evDestroy; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020: ++c->evRecord; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020: ++c->evQuery; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020: ++c->evSync; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020: ++c->evElapsed; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020: ++c->evWait; break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
            ++c->malloc;
            JMW_TRACE_ALLOC("cudaMalloc", pr ? (size_t)pr[1] : 0);
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020: ++c->free; break;
        default: break;
    }
}

bool subscribeAll() {
    if (g_subscribed.exchange(true)) return true;
    CUpti_SubscriberHandle sub = nullptr;
    if (cuptiSubscribe(&sub, callback, nullptr) != CUPTI_SUCCESS) {
        g_subscribed.store(false);
        return false;
    }
    const CUpti_CallbackId ids[] = {
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000,
        CUPTI_RUNTIME_TRACE_CBID_cudaHostUnregister_v4000,
        CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020,
        CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020,
        CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_ptsz_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_v12030,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_ptsz_v12030,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_ptsz_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_ptsz_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020,
    };
    int ok = 0;
    const int n = static_cast<int>(sizeof(ids) / sizeof(ids[0]));
    for (int i = 0; i < n; ++i) {
        if (cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, ids[i]) == CUPTI_SUCCESS)
            ++ok;
    }
    if (ok < n) fprintf(stderr, "[JMW_CENSUS] warning: %d/%d callbacks enabled\n", ok, n);
    return true;
}

const char* kCsvCols =
    "stage,launch,mcH2D,mcD2H,mcD2D,mcaH2D,mcaD2H,mcaD2D,mc2D,mca2D,memset,memsetA,"
    "sym,symA,hostReg,hostUnreg,mallocHost,freeHost,hostAlloc,mallocAsync,freeAsync,"
    "graphLaunch,graphInst,beginCap,endCap,graphUpload,graphExecUpd,hostFunc,"
    "streamCreate,streamDestroy,streamQuery,streamSync,devSync,"
    "evCreate,evDestroy,evRecord,evQuery,evSync,evElapsed,evWait,malloc,free,total,frames";

void writeRow(FILE* f, const char* name, const StageRow& c, long frames) {
    fprintf(f, "%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,"
               "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,"
               "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
            name, c.launch, c.mcH2D, c.mcD2H, c.mcD2D, c.mcaH2D, c.mcaD2H,
            c.mcaD2D, c.mc2D, c.mca2D, c.memset, c.memsetA, c.sym, c.symA,
            c.hostReg, c.hostUnreg, c.mallocHost, c.freeHost, c.hostAlloc,
            c.mallocAsync, c.freeAsync, c.graphLaunch,
            c.graphInst, c.beginCap, c.endCap, c.graphUpload, c.graphExecUpd,
            c.hostFunc, c.streamCreate, c.streamDestroy, c.streamQuery,
            c.streamSync, c.devSync, c.evCreate, c.evDestroy, c.evRecord,
            c.evQuery, c.evSync, c.evElapsed, c.evWait, c.malloc, c.free,
            c.total(), frames);
}

void report() {
    if (!g_enabled.exchange(false)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    std::map<std::string, Counters> m;
    for (auto& kv : g_byStage) {
        Counters cc;
        kv.second.toCounters(cc);
        m.emplace(kv.first, cc);
    }
    Derived d = derive(m, g_frames.load());
    const char* outPath = getenv("JMW_CENSUS_OUT");
    FILE* f = fopen(outPath ? outPath : "jmw_census.csv", "w");
    fprintf(stderr, "\n[JMW_CENSUS] ===== CUDA runtime API census v3 =====\n[%s]\n", kCsvCols);
    if (f) fprintf(f, "%s\n", kCsvCols);
    for (auto& kv : g_byStage) {
        writeRow(stderr, kv.first.c_str(), kv.second, 0);
        if (f) writeRow(f, kv.first.c_str(), kv.second, 0);
    }
    // 【v3 代际行】置 COLLAB 三行之前（四轮定案：消解红队A「尾部」/红队C 表述冲突；
    // 消费侧按表头名解析勿按列号——v2 41 列 → v3 44 列，第 9/11 列起有位移）
    fprintf(stderr, "GEN,v3,meta-cast+ptds-sub+hostalloc-side+2d-side\n"
                    "COLLAB_C0,total=%ld\nCOLLAB_C1,total=%ld\nCOLLAB_C2,total=%.2f,frames=%ld\n",
            d.c0, d.c1, d.c2, g_frames.load());
    if (f) fprintf(f, "GEN,v3,meta-cast+ptds-sub+hostalloc-side+2d-side\n"
                      "COLLAB_C0,,total=%ld\nCOLLAB_C1,,total=%ld\nCOLLAB_C2,,total=%.2f,frames=%ld\n",
                   d.c0, d.c1, d.c2, g_frames.load());
    if (f) {
        fclose(f);
        fprintf(stderr, "[JMW_CENSUS] csv -> %s\n", outPath ? outPath : "jmw_census.csv");
    }
}

}  // namespace

void init() {
    const char* e = getenv("JMW_CENSUS");
    if (!e || e[0] == '0') return;   // 崩溃打印机已由静态初始化器常驻安装
    if (!subscribeAll()) {
        fprintf(stderr, "[JMW_CENSUS] cuptiSubscribe failed, census disabled\n");
        return;
    }
    g_enabled.store(true);
#ifdef _WIN32
    if (getenv("JMW_CENSUS_TRACE")) {
        g_traceFile = fopen(getenv("JMW_CENSUS_TRACE"), "w");
        if (g_traceFile) fprintf(stderr, "[JMW_CENSUS] alloc trace -> %s\n", getenv("JMW_CENSUS_TRACE"));
    }
#endif
    atexit(report);
    fprintf(stderr, "[JMW_CENSUS] v3 enabled\n");
}

bool enableForTest() {
    if (!subscribeAll()) return false;
    g_enabled.store(true);
    return true;
}

void setStage(const char* name) {
    if (!g_enabled.load(std::memory_order_relaxed)) return;
    if (t_stage == name) return;
    t_stage = name;
    t_cnt = acquireRow(name);
}

void noteFrames(long n) { g_frames.store(n); }

bool snapshot(std::map<std::string, Counters>& out) {
    if (!g_enabled.load()) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    out.clear();
    for (auto& kv : g_byStage) {
        Counters cc;
        kv.second.toCounters(cc);
        out.emplace(kv.first, cc);
    }
    return true;
}

Derived derive(const std::map<std::string, Counters>& m, long frames) {
    Derived d;
    long mm = 0, mf = 0;
    auto it = m.find("main");
    if (it != m.end()) {
        mm = it->second.malloc;
        mf = it->second.free;
    }
    for (auto& kv : m) d.c0 += kv.second.total();
    d.c1 = d.c0 - mm - mf;
    d.c2 = frames > 0 ? static_cast<double>(d.c1) / static_cast<double>(frames) : -1.0;
    return d;
}

}  // namespace census
}  // namespace jmw
