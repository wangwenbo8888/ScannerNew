// fc_executor.cpp — 【五链 P2】五链流水线执行器实现
// 设计：五链并行流水线重构-设计方案-2026-10-21.md §二/§十 P2。头文件见 fc_executor.h。
// 2026-10-21 调优会话：①二分 1a/1b 默认开（A/B/A/B 实测 131.8-137.2 vs 单线程
// 113.5-118.7 FPS；历史 42.8 病理未复现，诱因模式已改 cv 阻塞＋lmask 多槽环）
// ②Job 池化零分配 ③配对/重排 map→定容环 ④逐帧指标 fc_metrics/fc_summary CSV。
#include "fc_executor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

#include <opencv2/core/cuda_stream_accessor.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#include <vector>
#endif

#include "gpu_accum.h"

namespace scr {
namespace fc {

// ============================================================================
// 【P4】绑核＋优先级（默认开；JMW_AFFINITY=0 关——A/B 复证用）
// ============================================================================
static bool fcCoreFirstLPs(std::vector<DWORD_PTR>& coreFirstLP,
                           std::vector<std::vector<DWORD_PTR>>& coreLPs) {
#ifdef _WIN32
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(64);
    DWORD cb = static_cast<DWORD>(buf.size() * sizeof(buf[0]));
    while (GetLogicalProcessorInformation(buf.data(), &cb) == FALSE) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
        buf.resize(cb / sizeof(buf[0]) + 1);
        cb = static_cast<DWORD>(buf.size() * sizeof(buf[0]));
    }
    for (size_t i = 0; i < cb / sizeof(buf[0]); ++i)
        if (buf[i].Relationship == RelationProcessorCore) {
            DWORD_PTR m = buf[i].ProcessorMask;
            std::vector<DWORD_PTR> lps;
            for (DWORD_PTR x = m; x; x &= x - 1) lps.push_back(x & (~(x - 1)));
            coreLPs.push_back(lps);
            coreFirstLP.push_back(lps[0]);
        }
    return !coreFirstLP.empty();
#else
    (void)coreFirstLP; (void)coreLPs;
    return false;
#endif
}

// 【H1·优先级解耦验证臂·2026-10-24】env JMW_PRIO_DECOUPLE=1 时 marker×3/ch2
// 降 THREAD_PRIORITY_ABOVE_NORMAL（替代 HIGHEST；1a/1b 保持 HIGHEST——喂料
// 链节拍最高，生产 PCoreBroker 同哲学）。不设/非 1＝现状全 HIGHEST（零行为
// 差异）。static 一次 getenv（首次绑核线程初始化，线程安全 magic static）。
// 回退：不设 env；彻底回退＝删本函数＋fcApplyPin 内改回固定 HIGHEST。
static int fcPrioFor(const char* who) {
#ifdef _WIN32
    static const bool decouple = [] {
        const char* e = getenv("JMW_PRIO_DECOUPLE");
        return e != nullptr && e[0] == '1';
    }();
    if (decouple
        && (strcmp(who, "fc:marker") == 0 || strcmp(who, "fc:ch2") == 0))
        return THREAD_PRIORITY_ABOVE_NORMAL;
    return THREAD_PRIORITY_HIGHEST;
#else
    (void)who;
    return 0;
#endif
}

static void fcApplyPin(DWORD_PTR mask, const char* who) {
#ifdef _WIN32
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        fprintf(stderr, "[fc:pin] %s 绑核失败: %lu\n", who, GetLastError());
        return;
    }
    const int prio = fcPrioFor(who);   // 【H1】优先级按角色＋JMW_PRIO_DECOUPLE 派生
    if (SetThreadPriority(GetCurrentThread(), prio) == 0)
        fprintf(stderr, "[fc:pin] %s 提优先级失败: %lu\n", who, GetLastError());
#else
    (void)mask; (void)who;
#endif
}

// 【语义统一·2026-10-24】JMW_AFFINITY 陷阱修复：旧语义「设任意值即关 fc 绑核」
// 与 pipe 侧「=1 才开」相反——设 =1 想开 pipe 绑核时会误关 fc 绑核（跨执行器
// A/B 脚枪）。统一为「='0' 才关」：与档案既有用法（=0 关）兼容，=1 不再误伤。
static bool fcAffinityOff() {
    const char* e = getenv("JMW_AFFINITY");
    return e != nullptr && e[0] == '0';
}

void fcPinThread(int roleIdx, const char* who) {
#ifdef _WIN32
    if (fcAffinityOff()) return;   // JMW_AFFINITY=0 显式关
    std::vector<DWORD_PTR> coreFirstLP;
    std::vector<std::vector<DWORD_PTR>> coreLPs;
    if (!fcCoreFirstLPs(coreFirstLP, coreLPs)) return;
    const size_t target = static_cast<size_t>(roleIdx) + 1;   // 跳过 0 号物理核
    if (target >= coreFirstLP.size()) {                        // 核不够→不绑（保底）
        // 【S4·显式降级告警】旧静默回退是产线静默塌方源之一
        fprintf(stderr, "[fc:pin] %s 绑核失败：roleIdx=%d 超出可用物理核数 %zu"
                        "——该线程将浮动（性能口径改变）\n",
                who, roleIdx, coreFirstLP.size());
        return;
    }
    fcApplyPin(coreFirstLP[target], who);
#else
    (void)roleIdx; (void)who;
#endif
}

void fcPinThreadLP(int coreIdx, int lpOffset, const char* who) {
#ifdef _WIN32
    if (fcAffinityOff()) return;   // JMW_AFFINITY=0 显式关（语义统一·2026-10-24）
    std::vector<DWORD_PTR> coreFirstLP;
    std::vector<std::vector<DWORD_PTR>> coreLPs;
    if (!fcCoreFirstLPs(coreFirstLP, coreLPs)) return;
    if (coreIdx < 0 || static_cast<size_t>(coreIdx) >= coreLPs.size()) {
        fprintf(stderr, "[fc:pin] %s 绑核失败：coreIdx=%d 超出物理核数 %zu——浮动\n",
                who, coreIdx, coreLPs.size());
        return;
    }
    const auto& lps = coreLPs[static_cast<size_t>(coreIdx)];
    if (lpOffset < 0 || static_cast<size_t>(lpOffset) >= lps.size()) {   // 不回退 LP0
        fprintf(stderr, "[fc:pin] %s 绑核失败：核%d 无 LP%d（SMT 不可用）——浮动\n",
                who, coreIdx, lpOffset);
        return;
    }
    fcApplyPin(lps[static_cast<size_t>(lpOffset)], who);
#else
    (void)coreIdx; (void)lpOffset; (void)who;
#endif
}

static double medianOf(std::vector<double>& v) {
    if (v.empty()) return -1.0;
    const size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(n), v.end());
    return v[n];
}

static long long steadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

long long FiveChainExecutor::nowNs() { return steadyNs(); }

// ============================================================================
// 【调优②】提交序计数器等待：cv 阻塞（原自旋 yield×THREAD_PRIORITY_HIGHEST
// 双线程互踢＝WDDM 乒乓病理疑点——42.8 FPS 会话实测在案，当前环境未复现但
// 诱因模式仍在；阻塞等待消除争用且省 CPU）
// ============================================================================
void FiveChainExecutor::waitCounterGe_(const std::atomic<int>& ctr, int target) {
    if (ctr.load(std::memory_order_acquire) >= target) return;
    std::unique_lock<std::mutex> lk(gateMtx_);
    gateCv_.wait(lk, [&] {
        return ctr.load(std::memory_order_acquire) >= target || stop_.load();
    });
}

void FiveChainExecutor::notifyGate_() {
    gateCv_.notify_all();
}

// ============================================================================
// Job 池（零分配）：acquire 阻塞于空闲栈＝原 live_ 在飞闸语义（池容量=在飞上限）
// ============================================================================
FcJob* FiveChainExecutor::acquireJob() {
    std::unique_lock<std::mutex> lk(jobMtx_);
    jobCv_.wait(lk, [&] {
        return !jobFree_.empty() || stop_.load();
    });
    if (jobFree_.empty()) return nullptr;   // stop 路径
    FcJob* j = jobFree_.back();
    jobFree_.pop_back();
    return j;
}

void FiveChainExecutor::releaseJob(FcJob* j) {
    j->resetForReuse();
    {
        std::lock_guard<std::mutex> lk(jobMtx_);
        jobFree_.push_back(j);
    }
    jobCv_.notify_all();
}

FiveChainExecutor::FiveChainExecutor(Harness& h, int firstIdx, int totalCycles)
    : h_(h), firstIdx_(firstIdx), totalCycles_(totalCycles),
      maxInflight_(h.cfg_.fivechain.maxInflight > 0 ? h.cfg_.fivechain.maxInflight : 56),
      graySlots_(std::max(2, h.cfg_.fivechain.graySlots)),
      rleCap_(h.cfg_.fivechain.kRleCapOverride > 0
                  ? std::max(8, h.cfg_.fivechain.kRleCapOverride)
                  : std::max(1024, h.cfg_.fivechain.rleCap)),
      chRead_(8), chGpu_(graySlots_), ch1ab_(4),
      chB_(std::max(2, h.cfg_.fivechain.bCap)),
      chC_(std::max(2, h.cfg_.fivechain.cCap)), chD_(std::max(2, h.cfg_.fivechain.dQueueCap)),
      chE_(std::max(2, h.cfg_.fivechain.eQueueCap)), chF_(std::max(2, h.cfg_.fivechain.fCap)),
      fedIdx_(firstIdx - graySlots_ - 1), nextFuse_(firstIdx), nextRegIdx_(firstIdx) {
    h_.fcActive_ = true;
    h_.ensureGrayLanes(graySlots_);

    // 【P3】CPU链1 ×mkLanes：idx%lanes 子队列＋邻队窃取；B＝MPSC（closeProducer）
    mkLanes_ = std::max(1, std::min(8, h.cfg_.fivechain.markerLanes));
    for (int i = 1; i < mkLanes_; ++i) chB_.addProducer();
    aQs_.resize(static_cast<size_t>(mkLanes_));
    for (auto& q : aQs_)
        q = std::make_unique<BoundedQueue>(std::max(2, h.cfg_.fivechain.rleSlots));
    h_.markerLanes_.clear();
    h_.markerLanes_.resize(static_cast<size_t>(mkLanes_));
    for (int i = 0; i < mkLanes_; ++i)
        h_.markerLanes_[static_cast<size_t>(i)] = h_.makeMarkerOps();

    // 【零分配】Job 池（容量=在飞上限——acquire 阻塞＝原 live_ 闸门语义）＋
    // 配对/重排定容环（在飞上限+8——读帧闸保证同环槽两代不同时在飞）
    jobs_.reserve(static_cast<size_t>(maxInflight_));
    for (int i = 0; i < maxInflight_; ++i)
        jobs_.emplace_back(std::make_unique<FcJob>());
    for (auto& j : jobs_) jobFree_.push_back(j.get());
    pendCap_ = maxInflight_ + 8;
    pendC_.assign(static_cast<size_t>(pendCap_), nullptr);
    pendE_.assign(static_cast<size_t>(pendCap_), nullptr);
    reorder_.assign(static_cast<size_t>(pendCap_), nullptr);

    evUpl_.resize(static_cast<size_t>(graySlots_));
    evSlot_.resize(static_cast<size_t>(graySlots_));
    for (auto& e : evUpl_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    for (auto& e : evSlot_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    // 【P4.5】二分交接事件＋【调优①】lmask 多槽环（默认 2 槽——放松在飞距）
    cudaEventCreateWithFlags(&evGate_, cudaEventDisableTiming);
    lmaskSlots_ = std::max(2, std::min(4, h.cfg_.fivechain.lmaskSlots));
    evLmaskRing_.resize(static_cast<size_t>(lmaskSlots_));
    lmaskRingL_.resize(static_cast<size_t>(lmaskSlots_));
    lmaskRingR_.resize(static_cast<size_t>(lmaskSlots_));
    for (auto& e : evLmaskRing_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

    // 【P4.5】RLE 环形槽（设备＋锁页；拷出计数闸门）
    rleSlots_ = std::max(2, std::min(8, h.cfg_.fivechain.rleSlots));
    // 【E7】按实拷窗口：钳 [64, rleCap_]；kRleCapOverride 注入小 cap 时窗口随缩
    //（min-cap=8 → 窗口 64 仍 >cap → 每帧补拷＝欠预测天然压测臂）
    rleWin_ = std::max(64, std::min(h.cfg_.fivechain.rleWindow, rleCap_));
    // 261025 实时适配：totalCycles=INT_MAX 哨兵（无界实时会话）时钳 reserve
    rleNAll_.reserve(static_cast<size_t>(
        std::max(1, std::min(totalCycles_, 65536))));
    dRleRingL_.resize(static_cast<size_t>(rleSlots_));
    dRleRingR_.resize(static_cast<size_t>(rleSlots_));
    evRleRing_.resize(static_cast<size_t>(rleSlots_));
    for (int i = 0; i < rleSlots_; ++i) {
        dRleRingL_[static_cast<size_t>(i)].create(1, 3 * rleCap_, CV_32SC1);
        dRleRingR_[static_cast<size_t>(i)].create(1, 3 * rleCap_, CV_32SC1);
        cudaEventCreateWithFlags(&evRleRing_[static_cast<size_t>(i)],
                                 cudaEventDisableTiming);
    }
    cudaMallocHost(&rleCntRing_, 2ull * static_cast<unsigned long long>(rleSlots_) * sizeof(int));
    cudaMallocHost(&rleRunsRing_, 6ull * rleSlots_
                                 * static_cast<unsigned long long>(rleCap_) * sizeof(int));
    rleCopied_.store(firstIdx - 1);   // 首轮 rleSlots_ 帧免等（"此前全已拷出"哨兵）

    eCapPts_ = std::max(1024, h.cfg_.fivechain.eMaxPoints
                                  * std::max(1, std::min(100, h.cfg_.fivechain.eBlockSizePct))
                                  / 100);
    eBlocks_.resize(static_cast<size_t>(std::max(2, h.cfg_.fivechain.ePoolBlocks)));
    for (auto& b : eBlocks_)
        cudaMallocHost(&b, 3ull * static_cast<unsigned long long>(eCapPts_) * sizeof(float));
    for (int i = static_cast<int>(eBlocks_.size()) - 1; i >= 0; --i) eFree_.push_back(i);
    d_pts_.create(1, eCapPts_, CV_32FC3);

    // fivechain 专属算子（参数与 harness v3/recon 同源同值——对拍前提）
    const auto v3p = calib::LaserMatchScanParamsV3::fromJson(
        h_.cfg_.opParams("laser_match_scan_v3"));
    splitGpu_ = std::make_unique<calib::LaserMatchScanSplitGpuV3>(v3p);
    splitCpu_ = std::make_unique<calib::LaserMatchScanSplitCpuV3>(v3p);
    reconP_ = calib::LaserReconstructParams::fromJson(
        h_.cfg_.opParams("laser_reconstruct"));
    reconCpu_ = std::make_unique<calib::LaserReconstructCpu>(reconP_);
    splitGpu_->Warmup(40000, 40000);   // 冷启动防护（P2 交付项）
    // 【N4·测量卫生】温表预注：双管表构造期注入，把 140MB 级上传＋流同步挪出
    // 首帧计时窗（此前落在 fc 帧 3/4 的 1b 线程——Warmup 只分配缓冲不注表；
    // 两管共表时第二次注入走 skip-fast-path 天然无害）。2026-10-23 第三轮 N4。
    splitGpu_->SetTempTable(h_.tableOdd_);
    splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
    splitGpu_->SetTempTable(h_.tableEven_);
    splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);

    // 【指标】逐帧记录＋GPU busy 近似事件对（timing 环 8 深，延迟查询）
    metricsOn_ = !(getenv("JMW_FCMETRICS") != nullptr
                   && std::string(getenv("JMW_FCMETRICS")) == "0");
    busyProbe_ = (getenv("JMW_FCBUSY") != nullptr
                  && getenv("JMW_FCBUSY")[0] != '0');   // 【R3-低8】默认关；=0 显式关（全仓约定）
    if (const char* e = getenv("JMW_FCDBG")) dbgInterval_ = atoi(e);
    if (dbgInterval_ <= 0) dbgInterval_ = 0;
    // 261025 实时适配：INT_MAX 哨兵＝无界会话——不建逐帧指标数组（各记录点
    // 已有 recs_ 空判护栏；writeMetricsCsv_ 同）
    constexpr int kFcUnbounded = 0x7fffffff;
    if (totalCycles_ > 0 && totalCycles_ != kFcUnbounded)
        recs_.reset(new FcFrameRec[static_cast<size_t>(totalCycles_)]);
    if (busyProbe_) {
        evBusy1aA_.resize(static_cast<size_t>(busyRing_ + 1));   // 环容量=回看距+1（防槽自碰）
        evBusy1aB_.resize(static_cast<size_t>(busyRing_ + 1));
        evBusy1bA_.resize(static_cast<size_t>(busyRing_ + 1));
        evBusy1bB_.resize(static_cast<size_t>(busyRing_ + 1));
        for (int i = 0; i < busyRing_ + 1; ++i) {
            cudaEventCreate(&evBusy1aA_[static_cast<size_t>(i)]);
            cudaEventCreate(&evBusy1aB_[static_cast<size_t>(i)]);
            cudaEventCreate(&evBusy1bA_[static_cast<size_t>(i)]);
            cudaEventCreate(&evBusy1bB_[static_cast<size_t>(i)]);
        }
    }

    h_.fcFuseStreamPtr_ = &fS_;
    gateFed_.store(firstIdx - 1);    // evGate 首帧免等哨兵（槽未占用）
    lmaskFed_.store(firstIdx - 1);   // lmask 首帧免等哨兵

    thReader_ = std::thread([this] { readerLoop(); });
    thUp_ = std::thread([this] { uploadLoop(); });
    // CUDA链1：默认二分 1a/1b（2026-10-21 调优后反超单线程＋正确性 0 diffs）；
    // JMW_FCSPLIT=0 强制回退单线程（病理环境 kill-switch），=1 强制开
    {
        bool split = h.cfg_.fivechain.splitChain1;
        if (const char* e = getenv("JMW_FCSPLIT")) split = (std::string(e) != "0");
        if (split) {
            th1a_ = std::thread([this] { chain1aLoop(); });
            th1b_ = std::thread([this] { chain1bLoop(); });
            std::cout << "[fivechain] CUDA链1 二分 1a/1b（lmask " << lmaskSlots_
                      << " 槽环＋cv 等待）" << std::endl;
        } else {
            th1_ = std::thread([this] { chain1Loop(); });
        }
    }
    for (int i = 0; i < mkLanes_; ++i)
        mkThs_.emplace_back([this, i] { markerLoop(i); });
    thReg_ = std::thread([this] { regLoop(); });
    th2_ = std::thread([this] { chain2Loop(); });
    thFuse_ = std::thread([this] { fuseLoop(); });
}

FiveChainExecutor::~FiveChainExecutor() {
    stopAll();
    // 【R3-F4】无条件设备同步：metricsOn_=0 时 stopAll 后仍可有在飞 D2H/H2D（RLE 环/
    // E 块），随后 cudaFreeHost 释放锁页＝UB——先收尾再释放
    cudaDeviceSynchronize();
    if (getenv("JMW_FCDIAG")) {
        fprintf(stderr, "[fc:diag] post-stopAll err=%s\n",
                cudaGetErrorString(cudaGetLastError()));
    }
    // 指标落盘须在事件销毁前（延迟查询 busy 事件对）
    if (metricsOn_) {
        writeMetricsCsv_();
        if (getenv("JMW_FCDIAG"))
            fprintf(stderr, "[fc:diag] post-csv err=%s\n",
                    cudaGetErrorString(cudaGetLastError()));
    }
    // 指标落盘须在事件销毁前（延迟查询 busy 事件对）
    if (metricsOn_) {
        cudaDeviceSynchronize();   // abort 残留在飞 GPU 工作收尾（查询合法化）
        if (getenv("JMW_FCDIAG"))
            fprintf(stderr, "[fc:diag] post-sync err=%s\n",
                    cudaGetErrorString(cudaGetLastError()));
        writeMetricsCsv_();
        if (getenv("JMW_FCDIAG"))
            fprintf(stderr, "[fc:diag] post-csv err=%s\n",
                    cudaGetErrorString(cudaGetLastError()));
    }
    for (auto& e : evUpl_) if (e) cudaEventDestroy(e);
    for (auto& e : evSlot_) if (e) cudaEventDestroy(e);
    if (evGate_) { cudaEventDestroy(evGate_); evGate_ = nullptr; }
    for (auto& e : evLmaskRing_) if (e) cudaEventDestroy(e);
    evLmaskRing_.clear();
    for (auto& e : evRleRing_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1aA_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1aB_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1bA_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1bB_) if (e) cudaEventDestroy(e);
    if (rleCntRing_) { cudaFreeHost(rleCntRing_); rleCntRing_ = nullptr; }
    if (rleRunsRing_) { cudaFreeHost(rleRunsRing_); rleRunsRing_ = nullptr; }
    for (auto& b : eBlocks_) if (b) cudaFreeHost(b);
    eBlocks_.clear();
    h_.fcActive_ = false;
    h_.fcFuseStreamPtr_ = nullptr;
}

void FiveChainExecutor::stopAll() {
    stop_.store(true);
    chRead_.close(); chGpu_.close(); ch1ab_.close();
    for (auto& q : aQs_) q->close();
    chB_.close(); chC_.close(); chD_.close(); chE_.close(); chF_.close();
    {
        std::lock_guard<std::mutex> lk(fedMtx_);
        fedIdx_ = totalCycles_;   // 解除上传线程 fed 等待
    }
    gateFed_.store(totalCycles_);
    lmaskFed_.store(totalCycles_);
    rleCopied_.store(totalCycles_);
    fedCv_.notify_all();
    gateCv_.notify_all();
    jobCv_.notify_all();
    eCv_.notify_all();
    if (thReader_.joinable()) thReader_.join();
    if (thUp_.joinable()) thUp_.join();
    if (th1_.joinable()) th1_.join();
    if (th1a_.joinable()) th1a_.join();
    if (th1b_.joinable()) th1b_.join();
    for (auto& t : mkThs_) if (t.joinable()) t.join();
    if (thReg_.joinable()) thReg_.join();
    if (th2_.joinable()) th2_.join();
    if (thFuse_.joinable()) thFuse_.join();
}

// ============================================================================
// GPU busy 近似：延迟 8 帧查询事件对 elapsed（时序事件环；查询前 cudaEventQuery
// 就绪才读，未就绪留待下帧/析构兜底）
// ============================================================================
void FiveChainExecutor::recordBusySpan_(int idx, bool oneA) {
    if (!busyProbe_ || !metricsOn_ || !recs_) return;
    const int back = idx - busyRing_;
    if (back < firstIdx_) return;
    auto& evA = oneA ? evBusy1aA_ : evBusy1bA_;
    auto& evB = oneA ? evBusy1aB_ : evBusy1bB_;
    // 环容量=busyRing_+1：回看 busyRing_ 帧必落在不同槽（防查询帧自身记录）
    const size_t s = static_cast<size_t>(back) % static_cast<size_t>(busyRing_ + 1);
    // 探测/计时产生的非致命错误码（NotReady/InvalidResourceHandle——未记录或未
    // 完成事件的合法探测结果）必须自行清账：残留会毒害后续第三方
    // cudaGetLastError() 消费者（flat ensureBuffers 实证）
    if (cudaEventQuery(evB[s]) != cudaSuccess) {
        cudaGetLastError();
        return;   // 未完成——下帧再试
    }
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, evA[s], evB[s]) == cudaSuccess)
        recs_[static_cast<size_t>(back)].v[oneA ? kFcGpu1a : kFcGpu1b]
            .store(static_cast<long long>(ms * 1000.0), std::memory_order_relaxed);   // μs
    cudaGetLastError();
}

// ============================================================================
// 读帧线程（在飞闸＝Job 池空闲；读帧失败＝全程墓碑 Job）
// ============================================================================
void FiveChainExecutor::readerLoop() {
    for (int idx = firstIdx_; idx < totalCycles_; ++idx) {
        FcJob* job = acquireJob();   // 池空＝在飞满（原 slotCv_ 闸门语义）
        if (job == nullptr) break;   // stop
        job->idx = idx;
        job->record = idx >= h_.cfg_.runtime.warmupCycles;
        const auto& cyc = h_.src_->cycles[static_cast<size_t>(idx)];
        cv::Mat gl, gr;
        std::string rerr;
        if (!h_.src_->readFrame(cyc.laserIdx, true, gl, rerr)
            || !h_.src_->readFrame(cyc.laserIdx, false, gr, rerr)) {
            job->readFail = true;
            job->err = rerr.empty() ? "读帧失败" : rerr;
            tombCount_.fetch_add(1);
        } else {
            job->prod.frameGrayL = gl;
            job->prod.frameGrayR = gr;
            // 261025 实时适配：帧元数据注入钩子（07 桥填 frameId/温补快照；
            // 沙盒对拍路径钩子为空＝行为不变）
            if (h_.onFrameMeta) h_.onFrameMeta(idx, job->prod);
        }
        chRead_.push(job);
        chCount_[0].fetch_add(1);
    }
    chRead_.close();
}

// ============================================================================
// 预上传线程（灰度环写槽 k；槽释放双保险＝fed 提交序＋evSlot 执行序）
// ============================================================================
void FiveChainExecutor::uploadLoop() {
    void* v = nullptr;
    while (chRead_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const int slot = j->idx % graySlots_;
        const auto cs = cv::cuda::StreamAccessor::getStream(upS_);
        if (j->idx >= firstIdx_ + graySlots_) {
            std::unique_lock<std::mutex> lk(fedMtx_);
            fedCv_.wait(lk, [&] {
                return stop_.load() || fedIdx_ >= j->idx - graySlots_;
            });
            if (stop_.load()) {
                chGpu_.push(j);   // 排空责任：继续下传防下游永等
                chCount_[1].fetch_add(1);
                continue;
            }
            lk.unlock();
            cudaStreamWaitEvent(cs, evSlot_[static_cast<size_t>(slot)], 0);
        }
        h_.d_grayLanesL_[static_cast<size_t>(slot)].upload(j->prod.frameGrayL, upS_);
        h_.d_grayLanesR_[static_cast<size_t>(slot)].upload(j->prod.frameGrayR, upS_);
        cudaEventRecord(evUpl_[static_cast<size_t>(slot)], cs);
        chGpu_.push(j);
        chCount_[1].fetch_add(1);
    }
    chGpu_.close();
}

// ============================================================================
// CUDA链1 分离段（免上传版——灰度环槽已由预上传就位）
// ============================================================================
void FiveChainExecutor::fcSeparation(FcJob* j) {
    auto& p = j->prod;
    try {
        h_.timeBegin();
        const int slot = j->idx % graySlots_;
        cudaStreamWaitEvent(cv::cuda::StreamAccessor::getStream(gS_),
                            evUpl_[static_cast<size_t>(slot)], 0);
        p.sepL = h_.sepLaserL_->Execute(h_.d_grayLanesL_[static_cast<size_t>(slot)], gS_);
        p.sepR = h_.sepLaserR_->Execute(h_.d_grayLanesR_[static_cast<size_t>(slot)], gS_);
        h_.setStepOk(p, "mask_separation", p.sepL.success && p.sepR.success);
        h_.timeEnd("mask_separation", p, j->record, false);
        if (!h_.stepOk(p, "mask_separation")) {
            gS_.waitForCompletion();
            h_.fail("mask_separation", p, p.sepL.message + " | " + p.sepR.message);
        }
    } catch (const std::exception& e) {
        h_.fail("mask_separation", p, e.what());
    }
}

void FiveChainExecutor::fcV3Front(FcJob* j, bool tubeOdd, cv::cuda::Stream& ls) {
    auto& p = j->prod;
    auto& L = h_.laserOps_;
    if (!h_.stepOk(p, "epipolar_interp_dual")) return;
    try {
        h_.timeBegin();
        if (j->v3Deferred) {
            // 【前段提前提交】完成段：紧缩＋主批 D2H＋拷出（精确 N＝interp 计数；
            // 提交段已在排水前发射——Step0-6 滴流藏进排水等待）
            splitGpu_->FinishFront(p.interpL.interpCount, p.interpR.interpCount,
                                   ls, j->front);
        } else {
            splitGpu_->SetTempTable(tubeOdd ? h_.tableOdd_ : h_.tableEven_);
            splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
            // ══【B4】前段提交图化实验臂（JMW_V3FRONTGRAPH=1；发射相位＝排水后窗口
            //    本函数即排水后调用点——非已证伪的 defer 相位；单 env，禁带 V3DEFER）══
            static const bool fgArmOn_ = [] {
                const char* e = getenv("JMW_V3FRONTGRAPH");
                return e != nullptr && e[0] == '1';
            }();
            bool graphDone = false;
            if (fgArmOn_ && !j->v3Deferred) {
                auto& Lg = h_.laserOps_;
                if (Lg.interpL_ && Lg.interpR_) {
                    cv::cuda::GpuMat gPtsL = Lg.interpL_->GetCapacityPointsView();
                    cv::cuda::GpuMat gIdsL = Lg.interpL_->GetCapacityIdsView();
                    cv::cuda::GpuMat gPtsR = Lg.interpR_->GetCapacityPointsView();
                    cv::cuda::GpuMat gIdsR = Lg.interpR_->GetCapacityIdsView();
                    if (!gPtsL.empty() && !gIdsL.empty() && !gPtsR.empty() && !gIdsR.empty()) {
                        splitGpu_->ExecuteFrontGraph(
                            gPtsL, gIdsL, Lg.interpL_->GetOutputCountPtr(),
                            gPtsR, gIdsR, Lg.interpR_->GetOutputCountPtr(),
                            ls, j->front);
                        if (j->front.success && j->front.leftCount == -1) {
                            splitGpu_->FinishFront(p.interpL.interpCount,
                                                   p.interpR.interpCount, ls, j->front);
                            graphDone = true;
                        }
                    }
                }
            }
            if (!graphDone)
                splitGpu_->Execute(*p.interpL.d_interpPoints, *p.interpL.d_interp_line_ids,
                                   *p.interpR.d_interpPoints, *p.interpR.d_interp_line_ids,
                                   ls, j->front);
        }
        h_.setStepOk(p, "laser_match_scan_v3", j->front.success);
        h_.timeEnd("laser_match_scan_v3", p, j->record, false);
        if (!j->front.success)
            h_.fail("laser_match_scan_v3", p, "前段失败: " + j->front.message);
    } catch (const std::exception& e) {
        h_.fail("laser_match_scan_v3", p, e.what());
    }
}

// 【前段提前提交】v3 提交段（排水前）：容量视图＋设备输出计数尾闸。失败/关闭/
// legacy 臂一律返回 false——执行器走排水后整段旧路径（行为与旧版逐字一致）
bool FiveChainExecutor::submitV3FrontDeferred_(FcJob* j, bool tubeOdd,
                                               cv::cuda::Stream& ls) {
    // ⚠️【2026-10-23 实测证伪·默认关】JMW_V3DEFER=1 才启用（实验臂）。两对 ABBA
    // 实测开启稳定慢 8-12 FPS（126.7/139.9 vs 139.0/148.3）：排水等待在 1b 关键
    // 路径上，把 v3 内核执行塞进排水＝串行化延迟流水线交接；而旧路径的「放行
    // 空隙」部分被并发 1a 内核吸收（§十一 的 89% 空闲≠100% 纯浪费）——H2 假设
    // 在流水线层面不成立。正确性已证（开vs关 399 帧 0 diffs）；代码保留作
    // Linux/TCC 环境（无 WDDM 放行税）复测臂。详见第三轮记录 §十二。
    static const bool on_ = [] {
        const char* e = getenv("JMW_V3DEFER");
        return e != nullptr && e[0] == '1';
    }();
    static const bool legacyArm_ = getenv("JMW_K2A_LEGACY") != nullptr
                                   || getenv("JMW_K2_LEGACY") != nullptr;
    if (!on_ || legacyArm_) return false;
    auto& L = h_.laserOps_;
    if (!L.interpL_ || !L.interpR_) return false;
    cv::cuda::GpuMat ptsL = L.interpL_->GetCapacityPointsView();
    cv::cuda::GpuMat idsL = L.interpL_->GetCapacityIdsView();
    cv::cuda::GpuMat ptsR = L.interpR_->GetCapacityPointsView();
    cv::cuda::GpuMat idsR = L.interpR_->GetCapacityIdsView();
    if (ptsL.empty() || ptsR.empty() || idsL.empty() || idsR.empty()) return false;
    splitGpu_->SetTempTable(tubeOdd ? h_.tableOdd_ : h_.tableEven_);
    splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
    splitGpu_->ExecuteDeferred(ptsL, idsL, L.interpL_->GetOutputCountPtr(),
                               ptsR, idsR, L.interpR_->GetOutputCountPtr(),
                               ls, j->front);
    if (!j->front.success) return false;   // 提交失败——回退旧路径（幂等重做）
    j->v3Deferred = true;
    return true;
}

// ============================================================================
// CUDA链1 单线程（splitChain1=0 回退路径）：分离→RLE 环槽发射→steger→undist→
// interp→v3前段→槽释放→RLE 拷出→推 A/D
// ============================================================================
void FiveChainExecutor::chain1Loop() {
    const bool pin1a = h_.cfg_.fivechain.pinRoles.find("1a") != std::string::npos;
    if (pin1a && h_.cfg_.fivechain.priorityBoost) fcPinThread(0, "fc:1");
    void* v = nullptr;
    while (chGpu_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const long long t0 = steadyNs();
        if (j->readFail) {   // 墓碑：跳过全部 GPU 段（下游按步失败护栏短路）
            aQs_[static_cast<size_t>(j->idx % mkLanes_)]->push(j);
            chD_.push(j);
            continue;
        }
        const bool tubeOdd = h_.src_->cycles[static_cast<size_t>(j->idx)].tubeOdd;
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1aA_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        fcSeparation(j);   // gS_：等 evUpl（上传执行序）→ 分离
        const bool sepOk = h_.stepOk(j->prod, "mask_separation");
        j->sepDone = sepOk;
        const bool hasMarkMask = sepOk && j->prod.sepL.d_markingPointMask
                                 && j->prod.sepR.d_markingPointMask;
        const int rs = j->idx % rleSlots_;
        if (hasMarkMask) {
            // RLE 环槽发射（标记掩膜 sep 内部缓冲——同流 FIFO 防下帧覆写）
            waitCounterGe_(rleCopied_, j->idx - rleSlots_);
            gpuMaskRleBothWindowed(*j->prod.sepL.d_markingPointMask,
                                   *j->prod.sepR.d_markingPointMask,
                                   dRleRingL_[static_cast<size_t>(rs)],
                                   dRleRingR_[static_cast<size_t>(rs)], rleCap_, rleWin_,
                                   rleCntRing_ + 2 * static_cast<size_t>(rs),
                                   rleRunsRing_ + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs),
                                   gS_);
            cudaEventRecord(evRleRing_[static_cast<size_t>(rs)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        }
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1aB_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        recordBusySpan_(j->idx, true);
        // ══ 【E4/E5】提交段（零宿主同步）＋槽释放＋唯一排水＋完成段 ══
        int frontKind = 0;
        if (j->sepDone)
            frontKind = h_.submitLaserFront(h_.laserOps_, j->prod, j->record,
                                            tubeOdd, j->idx % graySlots_, gS_);
        // 【前段提前提交】v3 提交段随提交相位（排水前）——容量视图＋计数尾闸
        //（判据用 interp 结果字段：延迟分支提交段不落 stepOk——那是完成段的事）
        if (frontKind == 1 && j->prod.interpL.success && j->prod.interpR.success)
            submitV3FrontDeferred_(j, tubeOdd, gS_);
        // 灰度环槽释放（d_gray 最后读者＝flat/steger；提交序＋执行序双保险）
        cudaEventRecord(evSlot_[static_cast<size_t>(j->idx % graySlots_)],
                        cv::cuda::StreamAccessor::getStream(gS_));
        {
            std::lock_guard<std::mutex> lk(fedMtx_);
            fedIdx_ = j->idx;
        }
        fedCv_.notify_all();

        if (frontKind != 0) {
            gS_.waitForCompletion();   // 唯一排水点（同流单线程——含 RLE D2H）
            j->prod.gpuDrained = true;
            h_.finishLaserFront(h_.laserOps_, j->prod, j->record, frontKind);
            if (h_.stepOk(j->prod, "steger_fast")
                && h_.stepOk(j->prod, "undistort_cuda")
                && h_.stepOk(j->prod, "epipolar_interp_dual"))
                fcV3Front(j, tubeOdd, gS_);
        }

        // RLE runs 拷出【E3】：正常路径 gpuDrained（flat 排水含 evGate 传递→RLE D2H
        // 必已落地）免同步；早退失败帧（flat 前返回，未排水）保留事件同步兜底
        if (hasMarkMask) {
            if (!j->prod.gpuDrained)
                cudaEventSynchronize(evRleRing_[static_cast<size_t>(rs)]);
            const int* cnt = rleCntRing_ + 2 * static_cast<size_t>(rs);
            const int* runs = rleRunsRing_
                + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs);
            const int nL = cnt[0], nR = cnt[1];
            if (nL > rleCap_ || nR > rleCap_) {
                j->rleOverflow = true;   // A 墓碑（ccl 记失败、配准走 skip）
            } else {
                j->rleNL = nL;
                j->rleNR = nR;
                // 【E7】超窗罕见帧：私有流补拷 [rleWin_, n) 段（同步返回＝store 前完成）
                if (nL > rleWin_ || nR > rleWin_) {
                    ++rleTailCnt_;
                    gpuRleTailCopy(dRleRingL_[static_cast<size_t>(rs)],
                                   dRleRingR_[static_cast<size_t>(rs)],
                                   rleCap_, rleWin_, nL, nR,
                                   const_cast<int*>(runs),
                                   evRleRing_[static_cast<size_t>(rs)]);
                }
                // 【E7】行程数分布统计（fc_summary 落档；供窗口调参）
                rleNSum_ += nL + nR;
                ++rleNCnt_;
                if (nL + nR > rleNMax_) rleNMax_ = nL + nR;
                rleNAll_.push_back(nL + nR);
                j->rleL.assign(runs, runs + 3 * static_cast<size_t>(nL));
                j->rleR.assign(runs + 3 * static_cast<size_t>(rleCap_),
                               runs + 3 * static_cast<size_t>(rleCap_)
                                   + 3 * static_cast<size_t>(nR));
            }
            rleCopied_.store(j->idx);
            notifyGate_();
        }
        const long long tPush0 = steadyNs();
        aQs_[static_cast<size_t>(j->idx % mkLanes_)]->push(j);
        chD_.push(j);
        if (metricsOn_ && recs_) {
            recs_[static_cast<size_t>(j->idx)].v[kFcWall1a]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
            recs_[static_cast<size_t>(j->idx)].v[kFcPush1a]
                .store((steadyNs() - tPush0) / 1000, std::memory_order_relaxed);   // us
        }
    }
    for (auto& q : aQs_) q->close();
    chD_.close();
}

// ============================================================================
// 【P4.5】CUDA链1 二分 1a（流 gS_）：分离 → 激光掩膜 D2D 入环槽 → RLE 内核发射
// → evRle/evGate 记录 → ch1ab 交接（宿主等待全部 cv 阻塞——调优②）
// ============================================================================
void FiveChainExecutor::chain1aLoop() {
    const bool pin1a = h_.cfg_.fivechain.pinRoles.find("1a") != std::string::npos;
    if (pin1a && h_.cfg_.fivechain.priorityBoost) fcPinThread(0, "fc:1a");
    void* v = nullptr;
    while (chGpu_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const long long t0 = steadyNs();
        if (j->readFail) {   // 墓碑直通
            // 【R3-F1】墓碑帧无条件推进水位（无在飞资源——store 无害）；
            // 否则 ≥2 连续读帧失败会卡死 1a 的 lmask 等待（历史缺陷）
            lmaskFed_.store(j->idx);
            rleCopied_.store(j->idx);
            notifyGate_();
            ch1ab_.push(j);
            continue;
        }
        fcSeparation(j);   // gS_：等 evUpl（上传执行序）→ 分离
        const bool sepOk = h_.stepOk(j->prod, "mask_separation");
        j->sepDone = sepOk;
        if (sepOk && j->prod.sepL.d_laserMask && j->prod.sepR.d_laserMask) {
            const auto cs = cv::cuda::StreamAccessor::getStream(gS_);
            // 【调优①】lmask 环槽 D2D：本槽上一代读者＝1b(idx-S)——其 evLmaskRing
            // 记录已提交（提交序计数器 cv 等待）后流级 wait，防 wait 捕获旧记录；
            // 首轮 S 帧（idx-S<firstIdx）事件未记录过＝立即返回
            const int ls = j->idx % lmaskSlots_;
            waitCounterGe_(lmaskFed_, j->idx - lmaskSlots_);
            if (j->idx - lmaskSlots_ >= firstIdx_)
                cudaStreamWaitEvent(cs, evLmaskRing_[static_cast<size_t>(ls)], 0);
            lmaskRingL_[static_cast<size_t>(ls)].create(
                j->prod.sepL.d_laserMask->size(), CV_8UC1);
            lmaskRingR_[static_cast<size_t>(ls)].create(
                j->prod.sepR.d_laserMask->size(), CV_8UC1);
            j->prod.sepL.d_laserMask->copyTo(lmaskRingL_[static_cast<size_t>(ls)], gS_);
            j->prod.sepR.d_laserMask->copyTo(lmaskRingR_[static_cast<size_t>(ls)], gS_);
            // 非持有视图交 1b（生命周期＝环槽闸门保证：本槽复用前 1b 已放手）
            j->prod.sepL.d_laserMask =
                std::shared_ptr<cv::cuda::GpuMat>(&lmaskRingL_[static_cast<size_t>(ls)],
                                                 [](cv::cuda::GpuMat*) {});
            j->prod.sepR.d_laserMask =
                std::shared_ptr<cv::cuda::GpuMat>(&lmaskRingR_[static_cast<size_t>(ls)],
                                                 [](cv::cuda::GpuMat*) {});
            // RLE 内核发射（标记掩膜 sep 内部缓冲——同流 FIFO 防下帧覆写；
            // 目标＝本帧环形槽——1b 拷出计数闸门防越代覆写）
            const int rs = j->idx % rleSlots_;
            waitCounterGe_(rleCopied_, j->idx - rleSlots_);
            gpuMaskRleBothWindowed(*j->prod.sepL.d_markingPointMask,
                                   *j->prod.sepR.d_markingPointMask,
                                   dRleRingL_[static_cast<size_t>(rs)],
                                   dRleRingR_[static_cast<size_t>(rs)], rleCap_, rleWin_,
                                   rleCntRing_ + 2 * static_cast<size_t>(rs),
                                   rleRunsRing_ + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs),
                                   gS_);
            cudaEventRecord(evRleRing_[static_cast<size_t>(rs)], cs);
        } else if (j->sepDone || j->readFail) {
            // 【R3-F1】sep 失败/掩膜缺席帧：本帧无在飞环资源——水位无条件推进，
            // 防 ≥2 连续失败帧卡死 1a 的 lmask/rle 等待（与 readFail 直通同款）
            lmaskFed_.store(j->idx);
            rleCopied_.store(j->idx);
            notifyGate_();
        }
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1aB_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        recordBusySpan_(j->idx, true);
        cudaEventRecord(evGate_, cv::cuda::StreamAccessor::getStream(gS_));
        gateFed_.store(j->idx);
        notifyGate_();
        ch1ab_.push(j);
        if (metricsOn_ && recs_)
            recs_[static_cast<size_t>(j->idx)].v[kFcWall1a]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
    }
    ch1ab_.close();
}

void FiveChainExecutor::chain1bLoop() {
    const bool pin1b = h_.cfg_.fivechain.pinRoles.find("1b") != std::string::npos;
    if (pin1b && h_.cfg_.fivechain.priorityBoost) fcPinThread(1, "fc:1b");
    void* v = nullptr;
    while (ch1ab_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const long long t0 = steadyNs();
        if (j->readFail) {   // 墓碑直通
            aQs_[static_cast<size_t>(j->idx % mkLanes_)]->push(j);
            chD_.push(j);
            continue;
        }
        const bool tubeOdd = h_.src_->cycles[static_cast<size_t>(j->idx)].tubeOdd;
        if (j->sepDone) {
            // 提交序闸门（防 wait 捕获旧记录——cv 阻塞）→ 流级等 1a 的 D2D/RLE 完成
            waitCounterGe_(gateFed_, j->idx);
            cudaStreamWaitEvent(cv::cuda::StreamAccessor::getStream(gS2_), evGate_, 0);
        }
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1bA_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS2_));
        // ══ 【E4/E5】提交段（零宿主同步）：flat(延迟)→undist(容量)→interp(延迟) ══
        int frontKind = 0;
        if (j->sepDone)
            frontKind = h_.submitLaserFront(h_.laserOps_, j->prod, j->record,
                                            tubeOdd, j->idx % graySlots_, gS2_);
        // 【前段提前提交】v3 提交段随提交相位（排水前）——容量视图＋计数尾闸
        //（判据用 interp 结果字段：延迟分支提交段不落 stepOk——那是完成段的事）
        if (frontKind == 1 && j->prod.interpL.success && j->prod.interpR.success)
            submitV3FrontDeferred_(j, tubeOdd, gS2_);
        // 槽释放（无条件——含 sep 失败/未提交帧；事件语义：d_gray/lmask 最后读者=flat，
        // 未提交帧空流上事件立即触发＝正确放行）
        cudaEventRecord(evSlot_[static_cast<size_t>(j->idx % graySlots_)],
                        cv::cuda::StreamAccessor::getStream(gS2_));
        {
            std::lock_guard<std::mutex> lk(fedMtx_);
            fedIdx_ = j->idx;
        }
        fedCv_.notify_all();
        // lmask 环槽释放（1a 第 idx+S 代 D2D 前置——按槽 record）
        cudaEventRecord(evLmaskRing_[static_cast<size_t>(j->idx % lmaskSlots_)],
                        cv::cuda::StreamAccessor::getStream(gS2_));
        lmaskFed_.store(j->idx);
        notifyGate_();

        // ══ 唯一排水点（本帧仅有的宿主同步之一；kind=2 legacy 已内部同步＝空转）══
        if (frontKind != 0) {
            gS2_.waitForCompletion();
            j->prod.gpuDrained = true;   // 【E3】含 evGate 传递→RLE D2H 落地
            h_.finishLaserFront(h_.laserOps_, j->prod, j->record, frontKind);
            // v3 前段（吃精确 ROI；内部自带 sync#1/#2——S4 未做之代价）
            if (h_.stepOk(j->prod, "steger_fast")
                && h_.stepOk(j->prod, "undistort_cuda")
                && h_.stepOk(j->prod, "epipolar_interp_dual"))
                fcV3Front(j, tubeOdd, gS2_);
        }

        if (busyProbe_ && metricsOn_ && recs_) {
            cudaEventRecord(evBusy1bB_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS2_));
            recordBusySpan_(j->idx, false);
        }

        // RLE runs 拷出【E3】：gpuDrained（排水含 evGate 传递→RLE D2H 已落地）免同步；
        // 早退失败帧（flat 前返回）保留事件同步兜底（只等 1a 的 RLE 内核）
        if (j->sepDone && j->prod.sepL.d_markingPointMask) {
            const int rs = j->idx % rleSlots_;
            if (!j->prod.gpuDrained)
                cudaEventSynchronize(evRleRing_[static_cast<size_t>(rs)]);
            const int* cnt = rleCntRing_ + 2 * static_cast<size_t>(rs);
            const int* runs = rleRunsRing_
                + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs);
            const int nL = cnt[0], nR = cnt[1];
            if (nL > rleCap_ || nR > rleCap_) {
                j->rleOverflow = true;   // A 墓碑（ccl 记失败、配准走 skip）
            } else {
                j->rleNL = nL;
                j->rleNR = nR;
                // 【E7】超窗罕见帧：私有流补拷 [rleWin_, n) 段（同步返回＝store 前完成）
                if (nL > rleWin_ || nR > rleWin_) {
                    ++rleTailCnt_;
                    gpuRleTailCopy(dRleRingL_[static_cast<size_t>(rs)],
                                   dRleRingR_[static_cast<size_t>(rs)],
                                   rleCap_, rleWin_, nL, nR,
                                   const_cast<int*>(runs),
                                   evRleRing_[static_cast<size_t>(rs)]);
                }
                // 【E7】行程数分布统计（fc_summary 落档；供窗口调参）
                rleNSum_ += nL + nR;
                ++rleNCnt_;
                if (nL + nR > rleNMax_) rleNMax_ = nL + nR;
                rleNAll_.push_back(nL + nR);
                j->rleL.assign(runs, runs + 3 * static_cast<size_t>(nL));
                j->rleR.assign(runs + 3 * static_cast<size_t>(rleCap_),
                               runs + 3 * static_cast<size_t>(rleCap_)
                                   + 3 * static_cast<size_t>(nR));
            }
            rleCopied_.store(j->idx);
            notifyGate_();
        }
        const long long tPush0 = steadyNs();
        aQs_[static_cast<size_t>(j->idx % mkLanes_)]->push(j);
        chD_.push(j);
        if (metricsOn_ && recs_) {
            recs_[static_cast<size_t>(j->idx)].v[kFcWall1b]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
            recs_[static_cast<size_t>(j->idx)].v[kFcPush1a]
                .store((steadyNs() - tPush0) / 1000, std::memory_order_relaxed);   // us
        }
    }
    for (auto& q : aQs_) q->close();
    chD_.close();
}

// ============================================================================
// CPU链1（P3 ×lanes）：idx%lanes 专属子队列＋邻队窃取（偷 FIFO 首端＝最紧迫；
// 窃取改变"谁处理"不改序约束——B 乱序合法区）。CCL(RLE)→…→conic。
// ============================================================================
void FiveChainExecutor::markerLoop(int laneIdx) {
    // 【E1·修同核冲突】marker 从 roleIdx2 起（原 1+laneIdx 与 1b 的 roleIdx1 撞在
    // 同一逻辑核＝量子切片硬串行——wall1b p90 23-25ms 尖峰的主因，R2-C 定性）
    const bool pinMk = h_.cfg_.fivechain.pinRoles.find("marker") != std::string::npos;
    if (pinMk && h_.cfg_.fivechain.priorityBoost) fcPinThread(2 + laneIdx, "fc:marker");
    BoundedQueue& myQ = *aQs_[static_cast<size_t>(laneIdx)];
    auto& ops = h_.markerLanes_[static_cast<size_t>(laneIdx)];
    auto process = [&](FcJob* j) {
        const long long t0 = steadyNs();
        chCount_[2].fetch_add(1);
        laneFrames_[static_cast<size_t>(laneIdx)].fetch_add(1);
        // 【P5·S1】拖慢注入（marker lane 0；压测背压/无死锁/恢复）
        if (laneIdx == 0 && h_.cfg_.fivechain.slowChainMs > 0)
            std::this_thread::sleep_for(
                std::chrono::milliseconds(h_.cfg_.fivechain.slowChainMs));
        if (j->readFail) {
            h_.fail("ccl", j->prod, "读帧失败墓碑");
        } else if (j->rleOverflow) {
            h_.fail("ccl", j->prod, "RLE 行程溢出（nL/nR>cap）——A 墓碑");
        } else if (j->prod.sepL.d_markingPointMask) {
            h_.runMarkerChainRuns(j->prod, j->record, ops,
                                  j->rleL, j->rleNL, j->rleR, j->rleNR,
                                  j->prod.frameGrayL.cols, j->prod.frameGrayL.rows);
        }
        // 【指标】B 深度＋链3 唤醒时间戳（首入空队者＝唤醒帧；B 是唯一 MPSC 通道）
        const int prev = chB_.pushRetPrevCount(j);
        if (metricsOn_ && recs_ && prev >= 0) {
            recs_[static_cast<size_t>(j->idx)].v[kFcOfoB]
                .store(prev, std::memory_order_relaxed);
            if (prev == 0) j->bPushNs = steadyNs();   // 本帧是唤醒者
        }
        if (metricsOn_ && recs_)
            recs_[static_cast<size_t>(j->idx)].v[kFcWallMk]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
    };
    for (;;) {
        void* v = nullptr;
        if (myQ.pop(v)) {   // 正常路径：本队弹出即处理
            process(static_cast<FcJob*>(v));
            continue;
        }
        // 本队关且空 → 先探窃取（邻队可能仍有积压）再退出
        bool stole = false;
        for (int k = 1; k < mkLanes_ && !stole; ++k) {
            BoundedQueue& nq = *aQs_[static_cast<size_t>((laneIdx + k) % mkLanes_)];
            if (nq.tryPop(v)) {
                process(static_cast<FcJob*>(v));
                stole = true;
            }
        }
        if (stole) continue;
        if (myQ.closed()) break;   // 本队关且空且无处可偷 → 退出
        // 空转等一小段再查（防错过新投喂）
        myQ.popFor(v, 1);
        if (v) process(static_cast<FcJob*>(v));
    }
    // 退出前最后扫一遍全邻队（收尾帧防漏）
    for (int k = 1; k < mkLanes_; ++k) {
        void* v = nullptr;
        BoundedQueue& nq = *aQs_[static_cast<size_t>((laneIdx + k) % mkLanes_)];
        while (nq.tryPop(v)) process(static_cast<FcJob*>(v));
    }
    chB_.closeProducer();
}

// ============================================================================
// CPU链3：急取式重排（B 弹出永不按帧序阻塞——定容环替代 map）→配准→C
// ============================================================================
void FiveChainExecutor::regLoop() {
    // 【H1·reg 绑核＋优先级·2026-10-24】env JMW_FCPIN_REG=N：N=物理核号→绑该
    // 核 LP1＋THREAD_PRIORITY_HIGHEST（fcApplyPin 成对，仿 JMW_FCPIN_CH2 模式；
    // JMW_AFFINITY 设置时仍整体旁路）。不设＝现状浮动 NORMAL（默认等价现状）。
    // 回退：不设 env 即回浮动；彻底回退＝删本段。
    if (const char* e = getenv("JMW_FCPIN_REG")) {
        const int n = atoi(e);
        if (n >= 0) fcPinThreadLP(n, 1, "fc:reg");
    }
    void* v = nullptr;
    while (chB_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        chCount_[3].fetch_add(1);
        // 【指标】链3 唤醒（首入空队帧的 push→pop 时延——§十二.5 P99 门）
        if (metricsOn_ && recs_ && j->bPushNs > 0)
            recs_[static_cast<size_t>(j->idx)].v[kFcRegWake]
                .store((steadyNs() - j->bPushNs) / 1000, std::memory_order_relaxed);   // us
        reorder_[static_cast<size_t>(j->idx) % static_cast<size_t>(pendCap_)] = j;
        while (true) {
            FcJob* r =
                reorder_[static_cast<size_t>(nextRegIdx_) % static_cast<size_t>(pendCap_)];
            if (r == nullptr || r->idx != nextRegIdx_) break;
            reorder_[static_cast<size_t>(nextRegIdx_) % static_cast<size_t>(pendCap_)] = nullptr;
            const long long t0 = steadyNs();
            h_.runRegistration(r->prod, r->record);
            if (r->record)
                h_.trajT_.push_back({r->prod.T[0], r->prod.T[1], r->prod.T[2]});
            ++h_.cycleCount_;
            if (metricsOn_ && recs_)
                recs_[static_cast<size_t>(r->idx)].v[kFcWallReg]
                    .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
            chC_.push(r);
            chCount_[4].fetch_add(1);
            ++nextRegIdx_;
        }
    }
    chC_.close();   // B 关且重排空（墓碑保证每帧必经 B）→ C 关
}

// ============================================================================
// CPU链2：v3 后段（CPU）→CPU 激光重建（直写 E 锁页块）→行残差 host 统计
// （【零分配】ml/mr/dRow/dDisp 线程驻留——容量跨帧复用）
// ============================================================================
void FiveChainExecutor::chain2Loop() {
    // 【E1·2026-10-23 翻转】ch2 绑核：默认核0-LP1（中断核次 LP）。旧默认核2-LP1
    // （1b 的 SMT 兄弟）所依据的「1b 等待型 ~0.5ms/帧」假设已过时——实测 1b 稳态
    // 78% CPU（WDDM 每帧 ~64 次提交的 CPU 税），ch2 与之同物理核互抢；ABBA 实测
    // 移核0 后 wallCh2 −11%、e2e −0.11ms、+2FPS（132.7→134.7，两对全一致）。
    // 逃生门【泛化·2026-10-24】：JMW_FCPIN_CH2=N 绑核N-LP1（N=0/未设＝默认核0；
    // 旧用法 =2 切核2 兼容；任意核号解锁「ch2 独占核5」等组合——核分配专题 §一·补）。
    // 连同 THREAD_PRIORITY_HIGHEST 成对（fcApplyPin 内置）。
    if (h_.cfg_.fivechain.pinRoles.find("ch2") != std::string::npos
        && h_.cfg_.fivechain.priorityBoost) {
        const int site = (getenv("JMW_FCPIN_CH2") != nullptr)
                             ? atoi(getenv("JMW_FCPIN_CH2")) : 0;
        fcPinThreadLP(site > 0 ? site : 0, 1, "fc:ch2");
    }
    // 【插桩·仅测量】JMW_FCBD=1 逐帧后段 7 段耗时串；JMW_FCDIAG 复用于 E 池等待
    // 时长（线程入口一次 getenv，默认零开销；N5：E 等待污染 wallCh2 口径在案）
    const bool fcbd = [] {
        const char* e = getenv("JMW_FCBD");
        return e != nullptr && e[0] == '1';
    }();
    const bool fcdiag = getenv("JMW_FCDIAG") != nullptr;
    std::vector<int> idsScratch(static_cast<size_t>(eCapPts_));
    std::vector<double> dDisp;
    void* v = nullptr;
    while (chD_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const long long t0 = steadyNs();
        chCount_[5].fetch_add(1);
        calib::LaserMatchScanCpuResultV3 cr;
        if (!j->readFail && j->front.success) {
            if (getenv("JMW_FCDIAG")) {
                fprintf(stderr,
                        "[fc:diag] idx=%d front: L=%d R=%d seg=%d cards=%d "
                        "trk=%d over=%d lstat1=%d\n",
                        j->idx, j->front.leftCount, j->front.rightCount,
                        j->front.segCount, j->front.totalCards,
                        static_cast<int>(j->front.leftTracks.size()),
                        j->front.segOverflow,
                        j->front.leftCount > 0 ? j->front.leftStatus[0] : -9);
            }
            h_.timeBegin();
            cr = splitCpu_->Execute(j->front);
            j->prod.laserMatched = cr.matchedCount;
            h_.setStepOk(j->prod, "laser_match_scan_v3", cr.success);
            h_.timeEnd("laser_match_scan_v3", j->prod, j->record, false);
            if (fcbd)   // 【插桩】后段分段串＋front 规模逐帧分布（seg/cards）
                fprintf(stderr, "[fc:bd] idx=%d seg=%d cards=%d bd=%s\n",
                        j->idx, j->front.segCount, j->front.totalCards,
                        splitCpu_->LastTimingBreakdown().c_str());
            if (!cr.success)
                h_.fail("laser_match_scan_v3", j->prod, "后段失败: " + cr.message);
        } else if (!j->readFail) {
            h_.fail("laser_match_scan_v3", j->prod,
                    "前段失败墓碑: " + j->front.message);
        }
        j->prod.laserPoints3D = 0;
        if (cr.success && cr.matchedCount > 0
            && cr.matched_left.total() > 0) {
            h_.timeBegin();
            // 【刀三·免拷贝】Mat(N×1 连续)直传指针（原 430KB 中转拷贝删除——
            // ReconstructInto 本收 const Point2f*；缓冲属后段 Impl 本帧有效）
            const cv::Point2f* ml = cr.matched_left.ptr<cv::Point2f>();
            const cv::Point2f* mr = cr.matched_right.ptr<cv::Point2f>();
            const long long ewT0 = fcdiag ? steadyNs() : 0;   // 【插桩】E 池等待
            const int s = acquireEBlock();
            if (fcdiag)
                fprintf(stderr, "[fc:diag] idx=%d ewait=%.3fms slot=%d\n",
                        j->idx, static_cast<double>(steadyNs() - ewT0) / 1e6, s);
            if (s >= 0) {   // 【R3-F5】stop 路径 -1＝跳过重建与归还
                j->eSlot = s;
                j->eCount = calib::LaserReconstructCpu::ReconstructInto(
                    ml, mr,
                    cr.matched_line_ids.ptr<int>(),
                    cr.matchedCount,
                    // 261025 实时适配：Q 每帧随帧温快照（沙盒路径 metaValid=false
                    // 回退 harness 构造期静态 Q——对拍行为不变）
                    j->prod.metaValid && !j->prod.snapQ.empty()
                        ? j->prod.snapQ : h_.calib_.Q,
                    reconP_.minDepth, reconP_.maxDepth,
                    eBlocks_[static_cast<size_t>(s)], idsScratch.data(), eCapPts_);
                if (cr.matchedCount > eCapPts_)
                    eTruncPts_.fetch_add(cr.matchedCount - eCapPts_);
                j->prod.laserPoints3D = j->eCount;
            }
            h_.setStepOk(j->prod, "laser_reconstruct", true);
            h_.timeEnd("laser_reconstruct", j->prod, j->record, false);
            // 视差统计（host 直算）。【刀三·死代码】rowResidMedPx 按构造恒 0
            //（后段 gather 出右点 y＝左点 y——split_cpu 输出口径，golden 全 0 实证）
            // ——dRow 循环与中位数计算整体删除
            if (!h_.prodProfile_) {
                const size_t n = static_cast<size_t>(cr.matchedCount);
                dDisp.resize(n);
                for (size_t i = 0; i < n; ++i)
                    dDisp[i] = static_cast<double>(ml[i].x - mr[i].x);
                j->prod.rowResidMedPx = 0.0;
                j->prod.dispMedPx = medianOf(dDisp);
            }
        }
        if (metricsOn_ && recs_)
            recs_[static_cast<size_t>(j->idx)].v[kFcWallCh2]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
        chE_.push(j);
        chCount_[6].fetch_add(1);
    }
    chE_.close();
}

// ============================================================================
// CUDA链2：C∩E 按帧号到齐（定容环配对）→ hold 零上传弃 E / 成功帧上传＋融合
// → F（帧序产出）
// ============================================================================
void FiveChainExecutor::fuseLoop() {
    // 【H1/H2·fuse 绑核＋优先级·2026-10-24】env JMW_FCPIN_FUSE=N：N=物理核号→
    // 绑该核 LP1＋THREAD_PRIORITY_HIGHEST（fcApplyPin 成对，仿 JMW_FCPIN_CH2
    // 模式；JMW_AFFINITY 设置时仍整体旁路）。针对：H1 优先级倒置（ch2(HIGHEST)
    // 阻塞等 fuse(NORMAL) 释放 E 块——吞吐被最低优先级线程兜底）＋H2 浮动落
    // 1b 兄弟核。不设＝现状浮动 NORMAL（默认等价现状）。
    // 回退：不设 env 即回浮动；彻底回退＝删本段。
    if (const char* e = getenv("JMW_FCPIN_FUSE")) {
        const int n = atoi(e);
        if (n >= 0) fcPinThreadLP(n, 1, "fc:fuse");
    }
    for (;;) {
        void* v = nullptr;
        if (chC_.tryPop(v)) {
            FcJob* jc = static_cast<FcJob*>(v);
            pendC_[static_cast<size_t>(jc->idx) % static_cast<size_t>(pendCap_)] = jc;
        }
        if (chE_.tryPop(v)) {
            FcJob* je = static_cast<FcJob*>(v);
            pendE_[static_cast<size_t>(je->idx) % static_cast<size_t>(pendCap_)] = je;
        }
        while (true) {
            const size_t slot = static_cast<size_t>(nextFuse_)
                % static_cast<size_t>(pendCap_);
            FcJob* j = pendC_[slot];
            FcJob* je = pendE_[slot];
            if (j == nullptr || je == nullptr || j->idx != nextFuse_
                || je->idx != nextFuse_)
                break;
            pendC_[slot] = nullptr;
            pendE_[slot] = nullptr;
            const std::string& rs = j->prod.regSource;
            // 【P5】holdSkip 演练：全部帧按 hold 处理（零上传弃 E 路径验证）
            const bool fused = h_.cfg_.fivechain.holdSkip == 0
                               && (rs == "first" || rs == "flow" || rs == "fuse")
                               && j->eCount > 0 && j->eSlot >= 0;
            if (fused) {
                const long long t0 = steadyNs();
                h_.timeBegin();
                cv::cuda::GpuMat view =
                    d_pts_(cv::Rect(0, 0, j->eCount, 1));
                cudaMemcpyAsync(view.ptr(),
                                eBlocks_[static_cast<size_t>(j->eSlot)],
                                sizeof(float) * 3ull * static_cast<unsigned>(j->eCount),
                                cudaMemcpyHostToDevice,
                                cv::cuda::StreamAccessor::getStream(fS_));
                auto res = h_.voxelFuseOp_->Execute(
                    view, cv::Matx33d(j->prod.R),
                    cv::Vec3d(j->prod.T[0], j->prod.T[1], j->prod.T[2]), fS_);
                fS_.waitForCompletion();   // H2D 完成后回收 E 块（上传完成事件口径）
                if (!res.success) {
                    std::cout << "[fc:fuse] Execute 失败: " << res.message << std::endl;
                } else {
                    h_.fuseInputTotal_ += static_cast<size_t>(res.inputCount);
                    ++h_.fuseFrameCount_;
                    if (res.rehashHappened) {
                        ++h_.fuseRehashCount_;
                        std::cout << "[fc:fuse] rehash → " << res.hashCapacity
                                  << " 槽 耗时 " << res.rehashTimeMs << "ms" << std::endl;
                    }
                    if (static_cast<double>(res.totalVoxelCount) >
                        h_.cfg_.fivechain.voxelLoadFactor
                            * static_cast<double>(res.hashCapacity))
                        ++h_.fuseOverloadCount_;
                }
                h_.timeEnd("accumulate_global_laser", j->prod, j->record, false);
                if (metricsOn_ && recs_)
                    recs_[static_cast<size_t>(j->idx)].v[kFcWallFuse]
                        .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
            }
            if (j->eSlot >= 0) {
                releaseEBlock(j->eSlot);
                j->eSlot = -1;
            }
            chF_.push(j);
            chCount_[7].fetch_add(1);
            ++nextFuse_;
        }
        if (nextFuse_ >= totalCycles_) break;
        if (stop_.load()) break;
        // 未到齐：轮询等待任一通道（1ms——PairGate 简化版）
        bool got = chC_.popFor(v, 1);
        if (got) {
            FcJob* jc = static_cast<FcJob*>(v);
            pendC_[static_cast<size_t>(jc->idx) % static_cast<size_t>(pendCap_)] = jc;
        }
        got = chE_.popFor(v, 1);
        if (got) {
            FcJob* je = static_cast<FcJob*>(v);
            pendE_[static_cast<size_t>(je->idx) % static_cast<size_t>(pendCap_)] = je;
        }
        if (chC_.closed() && chE_.closed() && !got) {
            // 全链排空但帧未到齐＝协议破损（fail-fast 而非挂起）——含定容环核对
            bool anyPend = false;
            for (int i = nextFuse_; i < totalCycles_; ++i)
                if (pendC_[static_cast<size_t>(i) % static_cast<size_t>(pendCap_)]
                    || pendE_[static_cast<size_t>(i) % static_cast<size_t>(pendCap_)]) {
                    anyPend = true;
                    break;
                }
            if (!anyPend) {
                std::cerr << "[fc:fuse] 协议破损：通道排空而 nextFuse_=" << nextFuse_
                          << " < total=" << totalCycles_ << std::endl;
                break;
            }
        }
    }
    chF_.close();
}

// ============================================================================
// E 锁页池
// ============================================================================
int FiveChainExecutor::acquireEBlock() {
    std::unique_lock<std::mutex> lk(eMtx_);
    eCv_.wait(lk, [&] { return !eFree_.empty() || stop_.load(); });
    if (eFree_.empty()) return -1;   // 【R3-F5】stop 兜底：-1＝未获取（防槽0双持有/重复归还）
    const int s = eFree_.back();
    eFree_.pop_back();
    return s;
}

void FiveChainExecutor::releaseEBlock(int s) {
    {
        std::lock_guard<std::mutex> lk(eMtx_);
        eFree_.push_back(s);
    }
    eCv_.notify_all();
}

// ============================================================================
// collect — 主线程按帧序收割（F 由 nextFuse_ 保证帧序）；收割后 Job 归还池
// ============================================================================
bool FiveChainExecutor::collect(int curIdx, CycleProducts& out, std::string& err) {
    void* v = nullptr;
    if (!chF_.pop(v)) {
        err = "fivechain: F 通道关闭且空（frame=" + std::to_string(curIdx) + "）";
        return false;
    }
    FcJob* j = static_cast<FcJob*>(v);
    if (j->idx != curIdx) {
        err = "fivechain: F 帧序错乱 want=" + std::to_string(curIdx)
              + " got=" + std::to_string(j->idx);
        return false;
    }
    err = j->err;
    out = std::move(j->prod);
    if (metricsOn_ && recs_) {
        const long long now = steadyNs();
        recs_[static_cast<size_t>(curIdx)].v[kFcE2E]
            .store(lastCollectNs_ > 0 ? (now - lastCollectNs_) / 1000 : -1,
                   std::memory_order_relaxed);   // μs
        lastCollectNs_ = now;
        if (dbgInterval_ > 0 && (curIdx - firstIdx_) % dbgInterval_ == 0) {
            const auto& rc = recs_[static_cast<size_t>(curIdx)];
            fprintf(stderr,
                    "[fc:dbg] f=%d e2e=%lldus wall1a=%lldus wall1b=%lldus mk=%lldus "
                    "reg=%lldus ch2=%lldus fuse=%lldus wake=%lldus ofoB=%lld\n",
                    curIdx,
                    rc.v[kFcE2E].load(),
                    rc.v[kFcWall1a].load(),
                    rc.v[kFcWall1b].load(),
                    rc.v[kFcWallMk].load(),
                    rc.v[kFcWallReg].load(),
                    rc.v[kFcWallCh2].load(),
                    rc.v[kFcWallFuse].load(),
                    rc.v[kFcRegWake].load(),
                    rc.v[kFcOfoB].load());
        }
    }
    releaseJob(j);
    return err.empty();
}

// ============================================================================
// 【指标】fc_metrics.csv＋fc_summary.csv 落盘（析构期——main 存活期）
// ============================================================================
void FiveChainExecutor::writeMetricsCsv_() {
    if (!recs_ || totalCycles_ <= 0 || firstIdx_ >= totalCycles_) return;
    // 兜底查询剩余 busy 事件对（析构前已 cudaDeviceSynchronize）
    for (int i = firstIdx_; i < totalCycles_; ++i) {
        if (recs_[static_cast<size_t>(i)].v[kFcGpu1a].load() < 0)
            recordBusySpan_(i + busyRing_, true);
        if (recs_[static_cast<size_t>(i)].v[kFcGpu1b].load() < 0)
            recordBusySpan_(i + busyRing_, false);
    }
    const std::string dir = h_.outDir_;

    // —— fc_metrics.csv（逐帧；μs 口径）——
    {
        FILE* f = nullptr;
        const std::string p1 = dir + "/fc_metrics.csv";
        fopen_s(&f, p1.c_str(), "w");
        if (!f) return;
        fprintf(f,
                "idx,tomb,wall1a_us,wall1b_us,wallMk_us,wallReg_us,wallCh2_us,"
                "wallFuse_us,regWake_us,e2e_us,push1a_us,gpu1a_us,gpu1b_us,ofoB\n");
        for (int i = firstIdx_; i < totalCycles_; ++i) {
            const auto& r = recs_[static_cast<size_t>(i)];
            fprintf(f, "%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                       "%lld\n",
                    i,
                    (r.v[kFcWall1a].load() < 0 && r.v[kFcWall1b].load() < 0) ? 1 : 0,
                    r.v[kFcWall1a].load(),
                    r.v[kFcWall1b].load(),
                    r.v[kFcWallMk].load(),
                    r.v[kFcWallReg].load(),
                    r.v[kFcWallCh2].load(),
                    r.v[kFcWallFuse].load(),
                    r.v[kFcRegWake].load(),
                    r.v[kFcE2E].load(),
                    r.v[kFcPush1a].load(),
                    r.v[kFcGpu1a].load(),
                    r.v[kFcGpu1b].load(),
                    r.v[kFcOfoB].load());
        }
        fclose(f);
    }

    // —— fc_summary.csv（med/P90/P99/max＋门判定＋对账）——
    auto collectMs = [&](FcRecField fld, std::vector<double>& outV) {
        outV.clear();
        for (int i = firstIdx_; i < totalCycles_; ++i) {
            const long long x = recs_[static_cast<size_t>(i)].v[fld].load();
            if (x >= 0) outV.push_back(static_cast<double>(x));
        }
    };
    auto q = [](std::vector<double>& v, double p) {
        if (v.empty()) return -1.0;
        std::sort(v.begin(), v.end());
        const size_t i = std::min(v.size() - 1,
                                  static_cast<size_t>(p * (v.size() - 1)));
        return v[i];
    };
    FILE* f = nullptr;
    const std::string p2 = dir + "/fc_summary.csv";
    fopen_s(&f, p2.c_str(), "w");
    if (!f) return;
    fprintf(f, "metric,med_us,p90_us,p99_us,max_us,gate\n");
    const struct {
        const char* name;
        FcRecField fld;
    } cols[] = {
        {"wall1a", kFcWall1a},   {"wall1b", kFcWall1b},
        {"wallMk", kFcWallMk},   {"wallReg", kFcWallReg},
        {"wallCh2", kFcWallCh2}, {"wallFuse", kFcWallFuse},
        {"regWake", kFcRegWake}, {"e2e", kFcE2E},
        {"push1a", kFcPush1a},   {"gpu1a", kFcGpu1a},
        {"gpu1b", kFcGpu1b},     {"ofoB", kFcOfoB},
    };
    std::vector<double> v;
    double regWakeP99 = -1, pushP99 = -1, e2eMed = -1;
    for (const auto& c : cols) {
        collectMs(c.fld, v);
        const double med = q(v, 0.5), p90 = q(v, 0.90), p99 = q(v, 0.99),
                     mx = v.empty() ? -1.0 : v.back();
        fprintf(f, "%s,%.1f,%.1f,%.1f,%.1f,\n", c.name, med, p90, p99, mx);
        if (c.fld == kFcRegWake) regWakeP99 = p99;
        if (c.fld == kFcPush1a) pushP99 = p99;
        if (c.fld == kFcE2E) e2eMed = med;
    }
    // 门判定（设计 §十四/§十二.5）
    fprintf(f, "gate_regWakeP99,%lld,%d\n",
            static_cast<long long>(h_.cfg_.fivechain.regWakeP99UsMax),
            regWakeP99 >= 0 && regWakeP99 <= h_.cfg_.fivechain.regWakeP99UsMax ? 1 : 0);
    fprintf(f, "gate_pushBudgetP99_us,%.0f,%d\n",
            h_.cfg_.fivechain.pushBudgetMs * 1000.0,
            pushP99 >= 0 && pushP99 <= h_.cfg_.fivechain.pushBudgetMs * 1000.0 ? 1 : 0);
    if (e2eMed > 0)
        fprintf(f, "steady_fps_median,%.1f,\n", 1.0e6 / e2eMed);
    // 对账与直方图
    for (int i = 0; i < 8; ++i)
        fprintf(f, "chCount_%d,%d,\n", i, chCount_[static_cast<size_t>(i)].load());
    fprintf(f, "tombCount,%d,\n", tombCount_.load());
    fprintf(f, "eTruncPts,%d,\n", eTruncPts_.load());
    for (int i = 0; i < mkLanes_ && i < 8; ++i)
        fprintf(f, "laneFrames_%d,%d,\n", i, laneFrames_[static_cast<size_t>(i)].load());
    // 【E7】RLE 按实拷健康度（行程分布＋补拷次数；tail 应为 0）
    if (!rleNAll_.empty()) {
        std::vector<int> srt(rleNAll_);
        std::nth_element(srt.begin(), srt.begin() + static_cast<long>(srt.size() / 2), srt.end());
        fprintf(f, "rle_n_median,%d,\n", srt[srt.size() / 2]);
        fprintf(f, "rle_n_max,%d,\n", rleNMax_);
        fprintf(f, "rle_window,%d,\n", rleWin_);
        fprintf(f, "rle_tail_copies,%ld,\n", rleTailCnt_);
    }
    // GPU busy 近似口径（e2e 归一；三段和可能＞墙钟——流重叠按并集欠估、按和
    // 高估，落档为近似带；nsys 终验另档）
    double g1aMed = -1, g1bMed = -1, fuseMed = -1;
    {
        collectMs(kFcGpu1a, v); g1aMed = q(v, 0.5);
        collectMs(kFcGpu1b, v); g1bMed = q(v, 0.5);
        collectMs(kFcWallFuse, v); fuseMed = q(v, 0.5);
    }
    if (e2eMed > 0 && (g1aMed >= 0 || g1bMed >= 0))
        fprintf(f, "gpu_busy_approach_pct,%.1f,\n",
                100.0 * (std::max(0.0, g1aMed) + std::max(0.0, g1bMed)
                         + std::max(0.0, fuseMed)) / e2eMed);
    fclose(f);
    cudaGetLastError();   // 兜底查询探测残留清账（防毒害后续消费者）
    std::cout << "[fc:metrics] fc_metrics.csv＋fc_summary.csv → " << dir << std::endl;
}

} // namespace fc
} // namespace scr
