// fc_executor.cpp — 【五链 P2】五链流水线执行器实现
// 设计：五链并行流水线重构-设计方案-2026-10-21.md §二/§十 P2。头文件见 fc_executor.h。
// 2026-10-21 调优会话：①二分 1a/1b 默认开（A/B/A/B 实测 131.8-137.2 vs 单线程
// 113.5-118.7 FPS；历史 42.8 病理未复现，诱因模式已改 cv 阻塞＋lmask 多槽环）
// ②Job 池化零分配 ③配对/重排 map→定容环 ④逐帧指标 fc_metrics/fc_summary CSV。
#include "fc_executor.h"
#include "cuda_census.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/core/cuda_stream_accessor.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#include <vector>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>   // 【JMW_PSPIN】_mm_pause 省力轮询
#endif
#endif   // _WIN32

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

FiveChainExecutor::FiveChainExecutor(Harness& h, int firstIdx, int totalCycles,
                                     jmw::probe::FrameProbe* probe)
    : h_(h), firstIdx_(firstIdx), totalCycles_(totalCycles),
      maxInflight_(h.cfg_.fivechain.maxInflight > 0 ? h.cfg_.fivechain.maxInflight : 56),
      graySlots_(std::max(2, h.cfg_.fivechain.graySlots)),
      rleCap_(h.cfg_.fivechain.kRleCapOverride > 0
                  ? std::max(8, h.cfg_.fivechain.kRleCapOverride)
                  : std::max(1024, h.cfg_.fivechain.rleCap)),
      probe_(probe),
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
    // 【B5】掩膜环深度对齐灰槽（同模铁律；配置 lmaskSlots 不再截 2..4——槽深
    // 语义整体交给 graySlots）。lmask 8 槽显存 48MiB（原 4 槽 24MiB，+24MiB；
    // 灰度环 48MiB 同量级已常驻）。evLmaskRing_ 创建随事件合一删除。
    lmaskSlots_ = graySlots_;
    lmaskRingL_.resize(static_cast<size_t>(lmaskSlots_));
    lmaskRingR_.resize(static_cast<size_t>(lmaskSlots_));

    // 【P4.5】RLE 环形槽（设备＋锁页；拷出计数闸门）
    rleSlots_ = std::max(2, std::min(8, h.cfg_.fivechain.rleSlots));
    // 【N-6】前置钉死 rleSlots_==graySlots_（配置钳位 [2,8] 失稳口——rs↔灰环槽
    // 配对恒定是 1a 每槽大图灰度指针可捕的前提；静默取 graySlots_）
    rleSlots_ = graySlots_;
    // 【E7】按实拷窗口：钳 [64, rleCap_]；kRleCapOverride 注入小 cap 时窗口随缩
    //（min-cap=8 → 窗口 64 仍 >cap → 每帧补拷＝欠预测天然压测臂）
    rleWin_ = std::max(64, std::min(h.cfg_.fivechain.rleWindow, rleCap_));
    // 261025 实时适配【261103 集成重放回植】：totalCycles=INT_MAX 表示无界
    // 实时会话——reserve 钳位（原版 INT_MAX 直接 reserve＝8.6GB）
    rleNAll_.reserve(static_cast<size_t>(
        std::max(1, std::min(totalCycles_, 65536))));
    dRleRingL_.resize(static_cast<size_t>(rleSlots_));
    dRleRingR_.resize(static_cast<size_t>(rleSlots_));
    // 【N-2】evRleRing_ 不再创建（并入 evGate_——h 头注释）
    for (int i = 0; i < rleSlots_; ++i) {
        dRleRingL_[static_cast<size_t>(i)].create(1, 3 * rleCap_, CV_32SC1);
        dRleRingR_[static_cast<size_t>(i)].create(1, 3 * rleCap_, CV_32SC1);
    }
    cudaMallocHost(&rleCntRing_, 2ull * static_cast<unsigned long long>(rleSlots_) * sizeof(int));
    cudaMallocHost(&rleRunsRing_, 6ull * rleSlots_
                                 * static_cast<unsigned long long>(rleCap_) * sizeof(int));
    rleCopied_.store(firstIdx - 1);   // 首轮 rleSlots_ 帧免等（"此前全已拷出"哨兵）
    // 【批C】【批H·终态默认开】RLE 槽图（构造器成员读取非 static——gtest 进程内
    // _putenv A/B 时构造器每次重读，K12；JMW_RLEGRAPH=0 显式关）；NOGRAPH 总闸优先
    rleGraphArm_ = [] {
        const char* ng = getenv("JMW_NOGRAPH");
        if (ng != nullptr && ng[0] == '1') return false;
        const char* e = getenv("JMW_RLEGRAPH");
        return !(e != nullptr && e[0] == '0');
    }();
    // 【N-6·终态默认开】1a 三合一大图（JMW_1AGRAPH=0 显式关；NOGRAPH 优先；
    // 构造器读）。ABBA IMPROVED（median +4.78——WDDM 提交 7→1/帧真实收益）
    sepBigArm_ = [] {
        const char* ng = getenv("JMW_NOGRAPH");
        if (ng != nullptr && ng[0] == '1') return false;
        const char* e = getenv("JMW_1AGRAPH");
        return !(e != nullptr && e[0] == '0');
    }();

    eCapPts_ = std::max(1024, h.cfg_.fivechain.eMaxPoints
                                  * std::max(1, std::min(100, h.cfg_.fivechain.eBlockSizePct))
                                  / 100);
    eBlocks_.resize(static_cast<size_t>(std::max(2, h.cfg_.fivechain.ePoolBlocks)));
    // 【离面鬼影修复 2026-10-30·根因②】零拷贝默认关（JMW_EZERO=1 显式开实验臂）。
    // 真因（经多轮代理审核定案，非"复写竞态"）：融合算子图重放稳态曾把建图帧
    // 的旧 E 槽指针写进参数块（laser_cloud_fuse_cuda_impl.cu :596 误传 fuseSig_）
    // ——内核每帧读旧槽，旧槽被 ch2 复写后即"帧 N 的点配帧 M 的位姿"（3-50mm
    // 整帧位移，确定性错误；跑间 53~92% 内点波动是幅度波动非发生概率）。
    // 该行已于 2026-10-30 修复（改传当帧 cur），但零拷贝仍依赖"槽生命周期推理
    // 正确"，拷贝路径（固定基指针 d_pts_＋流序 H2D）是构造性隔离更稳——
    // 依据 N-5 自旋臂同类判例（性能优化正确性存疑时默认关、env 显式开），
    // 默认维持关。代价仅 fuse 链 ~150µs/帧（wall ~2ms ≪ 帧周期 6.96ms，
    // 实测帧率零回退）。历史性能记载：零拷贝曾省 fuse mcaH2D 154→4µs（N-4）。
    // 重开前置：算子级确定性单测（GraphArm.InputRefresh 两喂不同数据）＋
    // 金样对拍。构造器成员读取非 static（K12）。
    eZero_ = [] {
        const char* e = getenv("JMW_EZERO");
        return (e != nullptr && e[0] == '1');
    }();
    eDevBlocks_.assign(eBlocks_.size(), nullptr);
    for (auto& b : eBlocks_) {
        if (eZero_) {
            cudaHostAlloc(reinterpret_cast<void**>(&b),
                          3ull * static_cast<unsigned long long>(eCapPts_) * sizeof(float),
                          cudaHostAllocMapped);
        } else {
            cudaMallocHost(&b, 3ull * static_cast<unsigned long long>(eCapPts_) * sizeof(float));
        }
    }
    if (eZero_) {
        for (size_t i = 0; i < eBlocks_.size(); ++i)
            if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&eDevBlocks_[i]),
                                         eBlocks_[i], 0) != cudaSuccess) {
                eDevBlocks_[i] = nullptr;
                eZero_ = false;   // mapped 不可用（理论不现——UVA 全支持）→整臂回退
                break;
            }
    }
    for (int i = static_cast<int>(eBlocks_.size()) - 1; i >= 0; --i) eFree_.push_back(i);
    d_pts_.create(1, eCapPts_, CV_32FC3);

    // fivechain 专属算子（参数与 harness v3/recon 同源同值——对拍前提）
    const auto v3p = calib::LaserMatchScanParamsV3::fromJson(
        h_.cfg_.opParams("laser_match_scan_v3"));
    splitGpu_ = std::make_unique<calib::LaserMatchScanSplitGpuV3>(v3p);
    splitCpu_ = std::make_unique<calib::LaserMatchScanSplitCpuV3>(v3p);
    reconP_ = calib::LaserReconstructParams::fromJson(
        h_.cfg_.opParams("laser_reconstruct"));
    // 261030 飞线门阈值（laser_reconstruct 节点 linePlaneGateMm——07 侧治理
    // 参数随算子配置下发；默认 0=关；env JMW_FC_LINEPLANE 覆盖）【261103 重放】
    {
        const auto& jn = h_.cfg_.opParams("laser_reconstruct");
        if (jn.contains("linePlaneGateMm"))
            linePlaneGateMm_ = jn.value("linePlaneGateMm", 0);
    }
    if (const char* e = getenv("JMW_FC_LINEPLANE")) linePlaneGateMm_ = atoi(e);
    reconCpu_ = std::make_unique<calib::LaserReconstructCpu>(reconP_);
    // 261031 口径统一：CMTT uR 偏置模型注册（离线拟合产物——FitUrbiasModel
    // 测试消费 checkpoint 生成 config/urbias.json；env JMW_FC_URBIAS=wx,wy,wz
    // 优先）。匹配期逐点补偿（ω 一阶旋转→v 向 CMOS 偏移→dZ），激光深度
    // 出厂即对齐标志点链（P2 原始口径）——live 云与 GBA 重放同享
    //【261103 集成重放；仅 GPU 路——CPU split 为对拍调试路径无偏置机制】
    {
        std::vector<float> ub(3, 0.0f);
        bool has = false;
        if (const char* e = std::getenv("JMW_FC_URBIAS")) {
            if (sscanf(e, "%f,%f,%f", &ub[0], &ub[1], &ub[2]) == 3) has = true;
        } else {
            std::ifstream ubf("config/urbias.json");
            if (ubf.is_open()) {
                try {
                    const auto j = nlohmann::json::parse(ubf);
                    ub = {j.at("wx").get<float>(), j.at("wy").get<float>(),
                          j.at("wz").get<float>()};
                    has = true;
                } catch (...) {}
            }
        }
        if (has) {
            splitGpu_->SetuRBiasModel(ub[0], ub[1], ub[2], 0.0f);
            JMW_LOG_INFO("07-FiveChain",
                "[FC-URBIAS] uR 偏置模型注册: wx={:.6f} wy={:.6f} wz={:.6f}（dtx=0，GPU 路）",
                ub[0], ub[1], ub[2]);
        }
    }
    splitGpu_->Warmup(40000, 40000);   // 冷启动防护（P2 交付项）
    // 【N4·测量卫生】温表预注：双管表构造期注入，把 140MB 级上传＋流同步挪出
    // 首帧计时窗（此前落在 fc 帧 3/4 的 1b 线程——Warmup 只分配缓冲不注表；
    // 两管共表时第二次注入走 skip-fast-path 天然无害）。2026-10-23 第三轮 N4。
    splitGpu_->SetTempTable(h_.tableOdd_);
    splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
    splitGpu_->SetTempTable(h_.tableEven_);
    splitGpu_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);

    // —— 【全GPU链2】默认关（2026-10-27 对齐仓内"默认关+env实验臂"惯例；
    // FG/SP 四轮 ABAB 已证分体更优 80-85 vs 67-73）；JMW_FC_FULLGPU=1 开整体版实验臂 ——
    {
        const char* fg = getenv("JMW_FC_FULLGPU");
        fullGpu_ = (fg != nullptr && fg[0] == '1');
    }
    if (fullGpu_) {
        matchFull_ = std::make_unique<calib::LaserMatchScanCudaV3>(v3p);
        matchFull_->Warmup(40000, 40000);   // 与 splitGpu_ 同容量口径
        matchFull_->SetTempTable(h_.tableOdd_);
        matchFull_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
        matchFull_->SetTempTable(h_.tableEven_);
        matchFull_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
        // 261103 集成重放：整体版同注 uR 偏置（与 splitGpu_ 同参）
        if (const char* e = std::getenv("JMW_FC_URBIAS"); e) {
            float ux = 0, uy = 0, uz = 0;
            if (sscanf(e, "%f,%f,%f", &ux, &uy, &uz) == 3)
                matchFull_->SetuRBiasModel(ux, uy, uz, 0.0f);
        }
        reconGpu_ = std::make_unique<calib::LaserReconstructCuda>(reconP_);
        reconGpu_->Warmup(40000);
        iRingPtsL_.resize(static_cast<size_t>(iRingSlots_));
        iRingIdsL_.resize(static_cast<size_t>(iRingSlots_));
        iRingPtsR_.resize(static_cast<size_t>(iRingSlots_));
        iRingIdsR_.resize(static_cast<size_t>(iRingSlots_));
        for (int i = 0; i < iRingSlots_; ++i) {
            iRingPtsL_[static_cast<size_t>(i)].create(1, iRingCap_, CV_32FC2);
            iRingIdsL_[static_cast<size_t>(i)].create(1, iRingCap_, CV_32SC1);
            iRingPtsR_[static_cast<size_t>(i)].create(1, iRingCap_, CV_32FC2);
            iRingIdsR_[static_cast<size_t>(i)].create(1, iRingCap_, CV_32SC1);
        }
        evIRing_.resize(static_cast<size_t>(iRingSlots_));
        for (int i = 0; i < iRingSlots_; ++i)
            cudaEventCreateWithFlags(&evIRing_[static_cast<size_t>(i)],
                                     cudaEventDisableTiming);
        iRingDone_.store(firstIdx - 1);
        std::cout << "[fivechain] 链2全GPU臂开启（整体版匹配＋GPU重建＋D2H→E 块；"
                     "JMW_FC_FULLGPU=0 回退分体）"
                  << std::endl;
    }

    // 【指标】逐帧记录＋GPU busy 近似事件对（timing 环 8 深，延迟查询）
    metricsOn_ = !(getenv("JMW_FCMETRICS") != nullptr
                   && std::string(getenv("JMW_FCMETRICS")) == "0");
    busyProbe_ = (getenv("JMW_FCBUSY") != nullptr
                  && getenv("JMW_FCBUSY")[0] != '0');   // 【R3-低8】默认关；=0 显式关（全仓约定）
    if (const char* e = getenv("JMW_FCDBG")) dbgInterval_ = atoi(e);
    if (dbgInterval_ <= 0) dbgInterval_ = 0;
    // 261025 实时适配【261103 集成重放回植】：无界会话（INT_MAX 哨兵）免建
    // 逐帧记录（原版 new FcFrameRec[0x7fffffff]＝85GB+ → bad_alloc 即 261103
    // 扫描启动即停的崩点；recs_ 空时全消费点由 metricsOn_&&recs_ 守卫跳过）
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
    // 【B2 修复·低优先级融合流·2026-10-29】nsys 定谳：融合内核与 1b 关键路径
    // 撞车（重叠 82.3%，主撞对象＝TightenCards 77%）＝B2 拆除阻塞回读的意外
    // 节拍器所致。低优先级流让 SM 调度器只在空闲时给融合块——把意外节拍变
    // 硬件自觉，B2 的 API 收益保留。默认开；JMW_FUSELOWPRIO=0 显式关回普通流。
    // 归档：B2机理定谳-隐形节拍器-2026-10-29.md §四b。
    static const bool fuseLowPrioOn_ = [] {
        const char* e = getenv("JMW_FUSELOWPRIO");
        return !(e != nullptr && e[0] == '0');
    }();
    if (fuseLowPrioOn_) {
        int lo = 0, hi = 0;
        if (cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess && lo < hi) {
            cudaStream_t raw = nullptr;
            if (cudaStreamCreateWithPriority(&raw, cudaStreamNonBlocking, lo)
                == cudaSuccess) {
                fS_ = cv::cuda::StreamAccessor::wrapStream(raw);
                fuseLowStream_ = raw;   // wrap 不转移所有权——析构显式销毁
                std::cout << "[fivechain] 融合流低优先级已启用（priority=" << lo
                          << "）" << std::endl;
            } else {
                std::cerr << "[fivechain] 低优先级融合流创建失败——回退普通流"
                          << std::endl;
            }
        }
    }
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
    if (fuseLowStream_) {   // 【B2 修复】低优先级融合流显式销毁（wrap 不拥有）
        cudaStreamDestroy(fuseLowStream_);
        fuseLowStream_ = nullptr;
    }
    for (auto& e : evUpl_) if (e) cudaEventDestroy(e);
    for (auto& e : evSlot_) if (e) cudaEventDestroy(e);
    if (evGate_) { cudaEventDestroy(evGate_); evGate_ = nullptr; }
    // 【B5】evLmaskRing_ 已随事件合一退役（构造期不再创建）
    // 【N-2】evRleRing_ 已随事件合一退役删除（并入 evGate_）
    // 【批C】RLE 槽图销毁（devSync 之后——K4 在飞图销毁 UB 防线）
    for (auto& ge : rleGraphExec_) if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
    for (auto& g : rleGraph_) if (g) { cudaGraphDestroy(g); g = nullptr; }
    // 【N-6】1a 三合一大图销毁（同 K4 防线）
    for (auto& ge : sepBigExec_) if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
    for (auto& g : sepBig_) if (g) { cudaGraphDestroy(g); g = nullptr; }
    for (auto& e : evBusy1aA_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1aB_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1bA_) if (e) cudaEventDestroy(e);
    for (auto& e : evBusy1bB_) if (e) cudaEventDestroy(e);
    if (rleCntRing_) { cudaFreeHost(rleCntRing_); rleCntRing_ = nullptr; }
    if (warmScratchD_) { cudaFree(warmScratchD_); warmScratchD_ = nullptr; }   // 【批1·E5】
    if (rleRunsRing_) { cudaFreeHost(rleRunsRing_); rleRunsRing_ = nullptr; }
    for (auto& b : eBlocks_) if (b) cudaFreeHost(b);
    eBlocks_.clear();
    h_.fcActive_ = false;
    h_.fcFuseStreamPtr_ = nullptr;
}

// 261030 软停（信号段——不含线程 join；stopAll 复用同款信号后追 join）
// 【261103 集成重放】stopRealtime 先软停再 join——析构抽走脚下对象＝空指针
// 成员读崩（0x748）根治
void FiveChainExecutor::requestStop() {
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
    jmw::census::setStage("reader");
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
    jmw::census::setStage("upload");
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
    // —— 【全GPU链2】1b 侧仅做 interp→环槽 D2D 搬运：匹配/重建/D2H 全在 ch2 的
    //    cS_ 流。同流 FIFO 保证"先拷贝后覆写"（源缓冲由下帧 interp 复用）；槽
    //    闸门（iRingDone_ ≥ idx-slots）防 ch2 未消费的槽被越代覆写。
    if (fullGpu_) {
        try {
            const int nl = p.interpL.interpCount;
            const int nr = p.interpR.interpCount;
            const bool ok = nl >= 0 && nr >= 0 && nl <= iRingCap_ && nr <= iRingCap_;
            if (ok) {
                waitCounterGe_(iRingDone_, j->idx - iRingSlots_);
                const int s = j->idx % iRingSlots_;
                cudaStream_t raw = cv::cuda::StreamAccessor::getStream(ls);
                cudaMemcpyAsync(iRingPtsL_[static_cast<size_t>(s)].ptr(),
                                p.interpL.d_interpPoints->ptr(),
                                sizeof(float) * 2ull * static_cast<unsigned>(nl),
                                cudaMemcpyDeviceToDevice, raw);
                cudaMemcpyAsync(iRingIdsL_[static_cast<size_t>(s)].ptr(),
                                p.interpL.d_interp_line_ids->ptr(),
                                sizeof(int) * static_cast<unsigned>(nl),
                                cudaMemcpyDeviceToDevice, raw);
                cudaMemcpyAsync(iRingPtsR_[static_cast<size_t>(s)].ptr(),
                                p.interpR.d_interpPoints->ptr(),
                                sizeof(float) * 2ull * static_cast<unsigned>(nr),
                                cudaMemcpyDeviceToDevice, raw);
                cudaMemcpyAsync(iRingIdsR_[static_cast<size_t>(s)].ptr(),
                                p.interpR.d_interp_line_ids->ptr(),
                                sizeof(int) * static_cast<unsigned>(nr),
                                cudaMemcpyDeviceToDevice, raw);
                // 跨流先后依赖：D2D 完成事件（ch2 的 cS_ 流消费前等待——异步拷贝
                // 入队≠完成，无此事件 ch2 可能读到半写数据＝匹配数漂移）
                cudaEventRecord(evIRing_[static_cast<size_t>(s)], raw);
            }
            j->front.success = ok;
            j->front.leftCount = nl;
            j->front.rightCount = nr;
            if (!ok) {
                j->front.message = "interp 计数越界: "
                                   + std::to_string(nl) + "/" + std::to_string(nr);
                h_.fail("laser_match_scan_v3", p, "interp→环槽失败: " + j->front.message);
            }
        } catch (const std::exception& e) {
            j->front.success = false;
            j->front.message = e.what();
            h_.fail("laser_match_scan_v3", p,
                    std::string("interp→环槽异常: ") + e.what());
        }
        return;
    }
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
            // ══【B4】前段提交图化（【批D·c2】默认开——JMW_V3FRONTGRAPH=0 显式关；
            //    JMW_NOGRAPH=1 总闸优先；发射相位＝排水后窗口，本函数即排水后
            //    调用点——非已证伪的 defer 相位；单 env，禁带 V3DEFER）══
            static const bool fgArmOn_ = [] {
                const char* ng = getenv("JMW_NOGRAPH");
                if (ng != nullptr && ng[0] == '1') return false;
                const char* e = getenv("JMW_V3FRONTGRAPH");
                return !(e != nullptr && e[0] == '0');
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

// 【批1·E3】容量提交核心（相位无关）——V3DEFER（排水前）与 V3POSTDEF（排水后
// 立刻）共用：容量视图＋设备输出计数尾闸，成功置 j->v3Deferred（fcV3Front 走
// FinishFront 收尾）。失败返回 false＝调用方回退旧路径（幂等重做）。
bool FiveChainExecutor::submitV3FrontCapacity_(FcJob* j, bool tubeOdd,
                                               cv::cuda::Stream& ls) {
    auto& L = h_.laserOps_;
    if (fullGpu_) return false;   // 全GPU臂：不产 front——回退路径即环槽搬运
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
    if (!on_ || legacyArm_ || fullGpu_) return false;
    return submitV3FrontCapacity_(j, tubeOdd, ls);
}

// ============================================================================
// 【probe v2】命中帧宿主快照——产出线程 D2H 入队（快照制）
//
// v1 方案（融合线程 E 段直读 GPU 视图）被质量审查否决：
//  - 错帧：undistort/interp/steger/分离掩膜等 GPU 产物是算子持久缓冲的 ROI
//    视图，契约「帧内消费、有效期至下一次 Execute」（undistort_points_cuda_
//    impl.cu:289-292 / epipolar_interp_dual_cuda_impl.cu:1144-1146 /
//    steger_fast_impl.cu:1901-1904 / laser_markingpoint_mask_separation_cuda_
//    impl.cu:1006-1009）——maxInflight=56 下融合线程滞后，E 段读到的是后续
//    帧数据＝错误证据。
//  - 捕获窗禁令：1b 大图 Global 模式捕获窗内任何线程 cudaMalloc/
//    cudaStreamCreate 非法（harness.h 灰环预分配段自写禁令）。
// v2：产出线程在「该帧数据尚未被后续帧覆写」的时机把 D2H 入队进生产流本身
//（同流 FIFO 保证：先产→后拷→再被下一帧覆写），宿主落地由**既有**排水链条
// 覆盖，零新增同步。运行期零 CUDA 分配：只 cudaMemcpy2DAsync 到宿主
// Mat/vector（命中帧仅此一次宿主分配），无 cudaMalloc/建流/GpuMat 扩容——
// 捕获窗（ThreadLocal 大图/sep 内部 Global 图）零不安全 API。
// ============================================================================
void FiveChainExecutor::probeSnapEnsure_(FcJob* j) {
    // 非命中帧：一次 set 查找即返回（零 GPU 操作零分配；probe 关＝一次判空）。
    // 帧首调用（墓碑检查后）——sep 失败帧也要有槽：宿主段（标志点链/CCL/位姿）
    // 照旧落盘，GPU 段缺段 WARN
    if (probe_ == nullptr || j->probeSnap != nullptr) return;
    if (!probe_->hit(j->idx)) return;
    try {
        j->probeSnap = std::make_unique<jmw::probe::ProbeSnapshot>();
    } catch (const std::exception& e) {
        j->probeSnap.reset();
        JMW_LOG_WARN("07-FiveChain", "[fc:probe] idx={} 快照分配失败: {}",
                     j->idx, e.what());
    }
}

void FiveChainExecutor::probeEnqueue1a_(FcJob* j) {
    if (j->probeSnap == nullptr) return;   // 非命中帧零开销（一次判空，无集合查找）
    if (!j->sepDone) return;   // sep 失败帧：无掩膜可拷（宿主段照旧，E 段缺段 WARN）
    auto& p = j->prod;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(gS_);
    jmw::probe::ProbeSnapshot& s = *j->probeSnap;
    auto dlMask = [&](cv::Mat& dst, const std::shared_ptr<cv::cuda::GpuMat>& m,
                      const char* what) {
        if (!m || m->empty()) return;
        dst.create(m->rows, m->cols, m->type());
        if (cudaMemcpy2DAsync(dst.ptr(), dst.step, m->ptr(), m->step,
                              m->cols * m->elemSize(), m->rows,
                              cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            cudaGetLastError();   // 清残留（recordBusySpan_ 同款——防毒害后续消费者）
            dst.release();
            JMW_LOG_WARN("07-FiveChain",
                         "[fc:probe] idx={} 快照下载 {} 失败——该段弃件（宁缺勿错）",
                         j->idx, what);
        }
    };
    // 三对掩膜：d_laserMask＝lmask 环槽视图（split 路已 D2D 入槽——本调用点
    // 在入队流序之后）；marking/binary＝sep 内部缓冲视图（下一覆写者＝本流
    // 下一帧 sep——同流 FIFO 天然防护）
    dlMask(s.laserMaskL, p.sepL.d_laserMask, "laserMaskL");
    dlMask(s.laserMaskR, p.sepR.d_laserMask, "laserMaskR");
    dlMask(s.markingMaskL, p.sepL.d_markingPointMask, "markingMaskL");
    dlMask(s.markingMaskR, p.sepR.d_markingPointMask, "markingMaskR");
    dlMask(s.binaryL, p.sepL.d_step1_binary, "binaryL");
    dlMask(s.binaryR, p.sepR.d_step1_binary, "binaryR");
}

void FiveChainExecutor::probeEnqueue1b_(FcJob* j) {
    if (j->probeSnap == nullptr) return;   // 非命中帧零开销（一次判空，无集合查找）
    auto& p = j->prod;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(gS2_);
    jmw::probe::ProbeSnapshot& s = *j->probeSnap;
    auto dlPts = [&](std::vector<cv::Point2f>& dst,
                     const std::shared_ptr<cv::cuda::GpuMat>& m, const char* what) {
        if (!m || m->empty()) return;
        dst.resize(static_cast<size_t>(m->cols));   // 容量宽下载（延迟算子精确计数
                                                    // 排水后才知）——dump 按计数截断
        if (cudaMemcpy2DAsync(dst.data(), sizeof(cv::Point2f) * m->cols, m->ptr(),
                              m->step, sizeof(cv::Point2f) * m->cols, m->rows,
                              cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            cudaGetLastError();
            dst.clear();
            JMW_LOG_WARN("07-FiveChain",
                         "[fc:probe] idx={} 快照下载 {} 失败——该段弃件（宁缺勿错）",
                         j->idx, what);
        }
    };
    auto dlIds = [&](std::vector<int>& dst,
                     const std::shared_ptr<cv::cuda::GpuMat>& m, const char* what) {
        if (!m || m->empty()) return;
        dst.resize(static_cast<size_t>(m->cols));
        if (cudaMemcpy2DAsync(dst.data(), sizeof(int) * m->cols, m->ptr(), m->step,
                              sizeof(int) * m->cols, m->rows,
                              cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
            cudaGetLastError();
            dst.clear();
            JMW_LOG_WARN("07-FiveChain",
                         "[fc:probe] idx={} 快照下载 {} 失败——该段弃件（宁缺勿错）",
                         j->idx, what);
        }
    };
    // 视图＝本帧 submitLaserFront 刚填的容量壳（下一覆写者＝本流下一帧提交段
    // ——同线程同流 FIFO 天然防护）；success 守卫防「早退帧下载到上一帧残数据」
    if (p.stegerL.success) dlPts(s.stegerL, p.stegerL.d_centerPoints, "stegerL");
    if (p.stegerR.success) dlPts(s.stegerR, p.stegerR.d_centerPoints, "stegerR");
    if (p.undistCudaL.success)
        dlPts(s.undistL, p.undistCudaL.d_rectifiedPoints, "undistL");
    if (p.undistCudaR.success)
        dlPts(s.undistR, p.undistCudaR.d_rectifiedPoints, "undistR");
    if (p.interpL.success) {
        dlPts(s.interpL, p.interpL.d_interpPoints, "interpL");
        dlIds(s.interpLineL, p.interpL.d_interp_line_ids, "interpLineL");
    }
    if (p.interpR.success) {
        dlPts(s.interpR, p.interpR.d_interpPoints, "interpR");
        dlIds(s.interpLineR, p.interpR.d_interp_line_ids, "interpLineR");
    }
}

// ============================================================================
// CUDA链1 单线程（splitChain1=0 回退路径）：分离→RLE 环槽发射→steger→undist→
// interp→v3前段→槽释放→RLE 拷出→推 A/D
// ============================================================================
void FiveChainExecutor::chain1Loop() {
    jmw::census::setStage("ch1");
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
        // 【probe v2】命中判定＋快照槽分配（帧首唯一 set 查找点）
        probeSnapEnsure_(j);
        const bool tubeOdd = h_.src_->cycles[static_cast<size_t>(j->idx)].tubeOdd;
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1aA_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        fcSeparation(j);   // gS_：等 evUpl（上传执行序）→ 分离
        const bool sepOk = h_.stepOk(j->prod, "mask_separation");
        j->sepDone = sepOk;
        const bool hasMarkMask = sepOk && j->prod.sepL.d_markingPointMask
                                 && j->prod.sepR.d_markingPointMask;
        // 261103 逐帧诊断（前 5 帧）：分离产出体检——分离成败＋掩膜非空
        if (j->idx < 5) {
            JMW_LOG_WARN("07-FiveChain",
                "[fc:sepdiag] idx={} sepOk={} hasMask={} laserMask={} markMask={}",
                j->idx, (int)sepOk, (int)hasMarkMask,
                j->prod.sepL.d_laserMask ? "非空" : "空",
                j->prod.sepL.d_markingPointMask ? "非空" : "空");
        }
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
            // 【probe v2】1a 段快照 D2H 入队（gS_）——紧邻 evGate_ 记录之前：
            // 早退帧兜底 cudaEventSynchronize(evGate_) 与正常帧唯一排水点双覆盖；
            // 无标记掩膜帧（理论不可达）不入队＝缺段 WARN（宁缺勿错）
            probeEnqueue1a_(j);
            // 【N-2】evRleRing_ 并入 evGate_（本臂单线程单流，与 1a/1b 线程组互斥
            // ——evGate_ 无他人记录；记录位点与旧 per-slot 事件逐字同位，消费者
            //（早退兜底/尾拷）全在本线程后序，无跨代覆写）
            cudaEventRecord(evGate_, cv::cuda::StreamAccessor::getStream(gS_));
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
        // 【probe v2】1b 段快照 D2H 入队（gS_，同流）——排水（waitForCompletion）
        // 之前入队＝宿主落地由既有排水覆盖；frontKind==0 未提交任何内核，视图
        // 是上帧残数据——跳过（宁缺勿错）
        if (frontKind != 0) probeEnqueue1b_(j);
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
        //（【N-2】改等 evGate_——本臂 record 位点与旧 evRleRing_ 同位，语义等价）
        if (hasMarkMask) {
            if (!j->prod.gpuDrained)
                cudaEventSynchronize(evGate_);
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
                                   evGate_);
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
// → evGate_ 记录（【N-2】evRle 已并入）→ ch1ab 交接（宿主等待全部 cv 阻塞——调优②）
// ============================================================================
// ============================================================================
// 【批C·RLE 入图】槽图发射口（图/逐发统一；回退链核心）＋全毁重建
// ============================================================================
void FiveChainExecutor::rleGraphInvalidateAll_() {
    gS_.waitForCompletion();   // K4：在飞图销毁前排空
    for (size_t i = 0; i < rleGraph_.size(); ++i) {
        if (rleGraphExec_[i]) { cudaGraphExecDestroy(rleGraphExec_[i]); rleGraphExec_[i] = nullptr; }
        if (rleGraph_[i]) { cudaGraphDestroy(rleGraph_[i]); rleGraph_[i] = nullptr; }
        rleGraphSigValid_[i] = false;
    }
}

void FiveChainExecutor::fcEmitRleSlot_(FcJob* j, int rs) {
    auto& maskL = *j->prod.sepL.d_markingPointMask;
    auto& maskR = *j->prod.sepR.d_markingPointMask;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(gS_);
    // done 标签：gateFed_ store 在 ch1ab_.push 之前（E2 happens-before——1b 的
    // waitCounterGe_ 恒即时满足的根据）
    auto done = [this, j] { gateFed_.store(j->idx); notifyGate_(); };
    // 【退守变体定案】事件留图外：图发射后按槽常规 record（与 1B/B4/sep 图同
    // 模式——K1 探针＋并发脆弱性实证见 gpu_accum.h 头注释）
    // 【N-2】evRleRing_ 退役：稳态消费者为零（正常帧 RLE 落地由 1b 排水经
    // evGate_ 传递覆盖），只记 evGate_（同位点单事件——早退兜底/尾拷改等它）
    // ⚠【N-2 红线（审查H）】gateFed_.store 必须后置于 cudaEventRecord(evGate_)——
    // 早退兜底/尾拷"等最近记录"语义的 happens-before 前提；任何把 store 挪到
    // record 之前、或给 1b 增加滞后事件消费者的重构都会静默破坏（ctest 不设防）
    auto recordEvents = [this, cs] {
        cudaEventRecord(evGate_, cs);
    };

    if (rleGraphArm_ && !rleGraphDisabled_ && ++rleGraphWarm_ >= 2) {
        RleGraphSig cur{};
        cur.mL = maskL.ptr<uint8_t>();
        cur.mR = maskR.ptr<uint8_t>();
        cur.rows = maskL.rows;
        cur.cols = maskL.cols;
        cur.stepL = static_cast<long long>(maskL.step);
        cur.stepR = static_cast<long long>(maskR.step);
        cur.cnt = rleCntRing_ + 2 * static_cast<size_t>(rs);
        cur.runs = rleRunsRing_ + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs);
        cur.runBL = dRleRingL_[static_cast<size_t>(rs)].ptr<int>();
        cur.runBR = dRleRingR_[static_cast<size_t>(rs)].ptr<int>();
        // 签名失配＝全毁重建（一槽失配＝槽槽失配——mask 是两眼全局属性；
        // H 变更触发——同时防 gpu_accum 暂存 realloc 后旧图持悬垂指针）
        if (rleGraphSigValid_[static_cast<size_t>(rs)]
            && std::memcmp(&cur, &rleGraphSig_[static_cast<size_t>(rs)], sizeof(cur)) != 0)
            rleGraphInvalidateAll_();
        if (rleGraphExec_[static_cast<size_t>(rs)] && rleGraphSigValid_[static_cast<size_t>(rs)]) {
            if (cudaGraphLaunch(rleGraphExec_[static_cast<size_t>(rs)], cs) == cudaSuccess) {
                recordEvents();
                done();
                return;
            }
            // 稳态发射失败：清账＋全毁＋永久回退（罕见——诊断落 stderr）
            fprintf(stderr, "[fc:rle] steady graphLaunch fail idx=%d: %s\n",
                    j->idx, cudaGetErrorString(cudaGetLastError()));
            rleGraphInvalidateAll_();
            rleGraphDisabled_ = true;
        } else {
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            if (gpuRleBuildSlotGraph(maskL, maskR,
                                     dRleRingL_[static_cast<size_t>(rs)],
                                     dRleRingR_[static_cast<size_t>(rs)],
                                     rleCap_, rleWin_,
                                     const_cast<int*>(cur.cnt), const_cast<int*>(cur.runs),
                                     gS_, &g, &ge)
                && cudaGraphLaunch(ge, cs) == cudaSuccess) {
                rleGraph_[static_cast<size_t>(rs)] = g;
                rleGraphExec_[static_cast<size_t>(rs)] = ge;
                rleGraphSig_[static_cast<size_t>(rs)] = cur;
                rleGraphSigValid_[static_cast<size_t>(rs)] = true;
                recordEvents();
                done();
                return;   // 捕获不执行当帧——launch 即补真跑
            }
            // 构建或首射失败：清残件（流内零残留由构建器保证）＋计数回退
            if (ge) cudaGraphExecDestroy(ge);
            if (g) cudaGraphDestroy(g);
            (void)cudaGetLastError();
            if (++rleGraphAttempts_ > 8) rleGraphDisabled_ = true;
        }
    }
    // 逐发兜底（与原 :877-884 逐字同构；evGate record 随 helper 化收拢至此——
    // 【N-2】evRleRing_ 记录已删）
    gpuMaskRleBothWindowed(*j->prod.sepL.d_markingPointMask,
                           *j->prod.sepR.d_markingPointMask,
                           dRleRingL_[static_cast<size_t>(rs)],
                           dRleRingR_[static_cast<size_t>(rs)], rleCap_, rleWin_,
                           rleCntRing_ + 2 * static_cast<size_t>(rs),
                           rleRunsRing_ + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs),
                           gS_);
    cudaEventRecord(evGate_, cs);
    done();
}

void FiveChainExecutor::sepBigInvalidateAll_() {
    for (int i = 0; i < 8; ++i) {
        if (sepBigExec_[static_cast<size_t>(i)]) {
            cudaGraphExecDestroy(sepBigExec_[static_cast<size_t>(i)]);
            sepBigExec_[static_cast<size_t>(i)] = nullptr;
        }
        if (sepBig_[static_cast<size_t>(i)]) {
            cudaGraphDestroy(sepBig_[static_cast<size_t>(i)]);
            sepBig_[static_cast<size_t>(i)] = nullptr;
        }
        sepBigSigValid_[static_cast<size_t>(i)] = false;
    }
}

// ============================================================================
// 【N-6】1a 三合一大图统一口：sep L/R＋lmask D2D＋RLE per-slot 单图
// （发射体与 legacy 逐段路径同源同序——sep EmitPipelineBody=separateDev 图体/
// lmask copyTo/gpuMaskRleBothWindowed；K1 捕获窗零事件——evGate_ record 图外；
// 暖机 2 帧 legacy 摊平 sep 内部缓冲/形态学暂存后再首捕——C9 同款）
// ============================================================================
bool FiveChainExecutor::trySubmit1aBig_(FcJob* j) {
    auto& p = j->prod;
    const int slot = j->idx % graySlots_;
    const int ls = j->idx % lmaskSlots_;
    const int rs = j->idx % rleSlots_;
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(gS_);
    auto& sepL = h_.sepLaserL_;
    auto& sepR = h_.sepLaserR_;
    auto& gL = h_.d_grayLanesL_[static_cast<size_t>(slot)];
    auto& gR = h_.d_grayLanesR_[static_cast<size_t>(slot)];

    // —— waits（灰环上传执行序＋lmask 上代读者释放＋RLE 环拷出水位——与 legacy
    //    逐段路径逐字同序）——
    cudaStreamWaitEvent(cs, evUpl_[static_cast<size_t>(slot)], 0);
    waitCounterGe_(lmaskFed_, j->idx - lmaskSlots_);
    if (j->idx - lmaskSlots_ >= firstIdx_)
        cudaStreamWaitEvent(cs, evSlot_[static_cast<size_t>(ls)], 0);
    waitCounterGe_(rleCopied_, j->idx - rleSlots_);

    // —— 签名（sep epoch＋维度＋灰环/lmask/RLE 环指针族；一槽失配＝槽槽失配）——
    SepBigSig cur{};
    cur.epSepL = sepL->GetParamsEpoch();
    cur.epSepR = sepR->GetParamsEpoch();
    cur.rows = gL.rows;
    cur.cols = gL.cols;
    cur.grayL = gL.ptr();
    cur.grayR = gR.ptr();
    cur.inBufL = sepL->InputBufferPtr();
    cur.inBufR = sepR->InputBufferPtr();
    cur.lmaskL = lmaskRingL_[static_cast<size_t>(ls)].ptr();
    cur.lmaskR = lmaskRingR_[static_cast<size_t>(ls)].ptr();
    cur.laserL = sepL->LaserMaskPtr();
    cur.laserR = sepR->LaserMaskPtr();
    cur.markL = sepL->MarkingMaskPtr();
    cur.markR = sepR->MarkingMaskPtr();
    cur.cnt = rleCntRing_ + 2 * static_cast<size_t>(rs);
    cur.runs = rleRunsRing_ + 6 * static_cast<size_t>(rleCap_) * static_cast<size_t>(rs);
    cur.runBL = dRleRingL_[static_cast<size_t>(rs)].ptr<int>();
    cur.runBR = dRleRingR_[static_cast<size_t>(rs)].ptr<int>();
    cur.rleCap = rleCap_;
    cur.rleWin = rleWin_;
    if (cur.rows <= 0 || cur.grayL == nullptr) return false;
    if (sepBigSigValid_[static_cast<size_t>(rs)]
        && std::memcmp(&cur, &sepBigSig_[static_cast<size_t>(rs)], sizeof(cur)) != 0)
        sepBigInvalidateAll_();

    auto recordGateAndDone = [this, cs, &j] {
        // ⚠【N-2 红线】evGate_ record 先于 gateFed_.store（1b 早退兜底/尾拷
        // 「等最近记录」happens-before 前提——与 fcEmitRleSlot_ 同构）
        cudaEventRecord(evGate_, cs);
        gateFed_.store(j->idx);
        notifyGate_();
    };
    auto fillProducts = [this, &p, &sepL, &sepR, ls]() {
        p.sepL = sepL->MakeShell();
        p.sepR = sepR->MakeShell();
        // lmask 环视图交 1b（非持有——生命周期＝环槽闸门，与 legacy 同构）
        p.sepL.d_laserMask =
            std::shared_ptr<cv::cuda::GpuMat>(&lmaskRingL_[static_cast<size_t>(ls)],
                                             [](cv::cuda::GpuMat*) {});
        p.sepR.d_laserMask =
            std::shared_ptr<cv::cuda::GpuMat>(&lmaskRingR_[static_cast<size_t>(ls)],
                                             [](cv::cuda::GpuMat*) {});
    };

    // —— 稳态：签名相符→单发 ——
    if (sepBigExec_[static_cast<size_t>(rs)] && sepBigSigValid_[static_cast<size_t>(rs)]) {
        if (cudaGraphLaunch(sepBigExec_[static_cast<size_t>(rs)], cs) == cudaSuccess) {
            fillProducts();
            // 【probe v2】1a 段快照 D2H 入队（gS_，图发射之后＝流序读到本帧图产物；
            // evGate_ 记录之前＝1b 排水经事件传递保证宿主落地）
            probeEnqueue1a_(j);
            recordGateAndDone();
            h_.setStepOk(p, "mask_separation", true);
            return true;
        }
        fprintf(stderr, "[fc:1abig] steady graphLaunch fail idx=%d: %s\n",
                j->idx, cudaGetErrorString(cudaGetLastError()));
        sepBigInvalidateAll_();
        sepBigDisabled_ = true;
        return false;
    }

    // —— 捕获（暖机 2 帧 legacy 由调用侧守卫；捕获前排空本流铁规）——
    // 预建全部 lmask 环槽（每槽惰性首建——未用过的槽首次 create 会落进捕获窗
    // ＝DefaultAllocator 分配非法「operation not permitted when stream is
    // capturing」；全槽同尺寸一次建齐＝纯分配零写、无数据竞态）
    {
        const auto sz = sepL->MakeShell().d_laserMask->size();
        for (auto& m : lmaskRingL_) if (m.empty()) m.create(sz, CV_8UC1);
        for (auto& m : lmaskRingR_) if (m.empty()) m.create(sz, CV_8UC1);
    }
    cudaStreamSynchronize(cs);
    cudaError_t cerr = cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
    bool bodyOk = (cerr == cudaSuccess);
    if (bodyOk) {
        sepL->EmitPipelineBody(gL, gS_);
        sepR->EmitPipelineBody(gR, gS_);
        auto shellL = sepL->MakeShell();
        auto shellR = sepR->MakeShell();
        lmaskRingL_[static_cast<size_t>(ls)].create(shellL.d_laserMask->size(), CV_8UC1);
        lmaskRingR_[static_cast<size_t>(ls)].create(shellR.d_laserMask->size(), CV_8UC1);
        shellL.d_laserMask->copyTo(lmaskRingL_[static_cast<size_t>(ls)], gS_);
        shellR.d_laserMask->copyTo(lmaskRingR_[static_cast<size_t>(ls)], gS_);
        gpuMaskRleBothWindowed(*shellL.d_markingPointMask, *shellR.d_markingPointMask,
                               dRleRingL_[static_cast<size_t>(rs)],
                               dRleRingR_[static_cast<size_t>(rs)], rleCap_, rleWin_,
                               const_cast<int*>(cur.cnt), const_cast<int*>(cur.runs),
                               gS_);
    }
    cudaGraph_t g = nullptr;
    const cudaError_t eerr = cudaStreamEndCapture(cs, &g);
    if (!bodyOk || eerr != cudaSuccess || !g) {
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
        if (++sepBigAttempts_ > 8) sepBigDisabled_ = true;
        return false;   // 本帧 legacy 重做（捕获不执行——幂等）
    }
    cudaGraphExec_t ge = nullptr;
    if (cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess || !ge) {
        if (ge) cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        cudaGetLastError();
        if (++sepBigAttempts_ > 8) sepBigDisabled_ = true;
        return false;
    }
    // 新槽首捕：只清本槽旧件（「全毁」仅留给结构失配路径——per-slot 语义下
    // 全毁新槽＝8 槽永动重捕风暴，151 beginCap 实锤 2026-10-28）
    if (sepBigExec_[static_cast<size_t>(rs)]) {
        cudaGraphExecDestroy(sepBigExec_[static_cast<size_t>(rs)]);
        sepBigExec_[static_cast<size_t>(rs)] = nullptr;
    }
    if (sepBig_[static_cast<size_t>(rs)]) {
        cudaGraphDestroy(sepBig_[static_cast<size_t>(rs)]);
        sepBig_[static_cast<size_t>(rs)] = nullptr;
    }
    sepBig_[static_cast<size_t>(rs)] = g;
    sepBigExec_[static_cast<size_t>(rs)] = ge;
    sepBigSig_[static_cast<size_t>(rs)] = cur;
    sepBigSigValid_[static_cast<size_t>(rs)] = true;
    size_t nn = 0;
    cudaGraphGetNodes(g, nullptr, &nn);
    std::cout << "[fc:1abig] SepBigGraph captured slot=" << rs << " nodes=" << nn << std::endl;
    if (cudaGraphLaunch(ge, cs) == cudaSuccess) {
        fillProducts();
        // 【probe v2】1a 段快照 D2H 入队（同稳态位——图发射后/evGate_ 记录前）
        probeEnqueue1a_(j);
        recordGateAndDone();
        h_.setStepOk(p, "mask_separation", true);
        return true;
    }
    cudaGetLastError();
    sepBigInvalidateAll_();
    sepBigDisabled_ = true;
    return false;
}

void FiveChainExecutor::chain1aLoop() {
    jmw::census::setStage("1a");
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
        // 【probe v2】命中判定＋快照槽分配（帧首唯一 set 查找点）
        probeSnapEnsure_(j);
        // 【N-6】1a 三合一大图（暖机 2 帧 legacy——摊平 sep 内部缓冲/暂存后首捕）
        bool bigDone = false;
        if (sepBigArm_ && !sepBigDisabled_ && sepBigWarm_ >= 2)
            bigDone = trySubmit1aBig_(j);
        else if (sepBigArm_ && !sepBigDisabled_)
            ++sepBigWarm_;
        if (bigDone) {
            j->sepDone = true;
        } else {
        fcSeparation(j);   // gS_：等 evUpl（上传执行序）→ 分离
        const bool sepOk = h_.stepOk(j->prod, "mask_separation");
        j->sepDone = sepOk;
        if (sepOk && j->prod.sepL.d_laserMask && j->prod.sepR.d_laserMask) {
            const auto cs = cv::cuda::StreamAccessor::getStream(gS_);
            // 【调优①＋B5】lmask 环槽 D2D：本槽上一代读者＝1b(idx-S)。【B5】槽深
            // 对齐 graySlots_（%8 同模），wait 改等 evSlot_（1b 无条件记录的双消费
            // 者事件：灰槽复用＋掩膜环覆写）；lmaskFed_ 水位守护其记录已提交（防
            // wait 捕获旧记录——cv 阻塞）；首轮 S 帧（idx-S<firstIdx）未记录过＝
            // 立即返回。readFail/sep 失败帧：水位由 R3-F1 无条件推进，事件无空洞。
            const int ls = j->idx % lmaskSlots_;
            waitCounterGe_(lmaskFed_, j->idx - lmaskSlots_);
            if (j->idx - lmaskSlots_ >= firstIdx_)
                cudaStreamWaitEvent(cs, evSlot_[static_cast<size_t>(ls)], 0);
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
            // 【probe v2】1a 段快照 D2H 入队（gS_，D2D 之后流序读到环槽本帧内容；
            // fcEmitRleSlot_ 的 evGate_ 记录之前＝1b 排水经事件传递保证宿主落地；
            // marking/binary 视图下一覆写者＝下一帧 sep——同流 FIFO 防护）
            probeEnqueue1a_(j);
            // RLE 内核发射（标记掩膜 sep 内部缓冲——同流 FIFO 防下帧覆写；
            // 目标＝本帧环形槽——1b 拷出计数闸门防越代覆写）
            // 【批C】图/逐发统一发射口：图臂含 evGate 图内 record＋gateFed_ 尾
            //（E2：store 在 ch1ab_.push 之前——happens-before 保持，busyProbe 段
            // 留在调用侧原地不参与 helper 化）
            const int rs = j->idx % rleSlots_;
            waitCounterGe_(rleCopied_, j->idx - rleSlots_);
            fcEmitRleSlot_(j, rs);
        } else if (j->sepDone || j->readFail) {
            // 【R3-F1＋红队B·E1 勘误】防御性死分支：sepDone=true 而 mask 缺席由
            // impl:1165-1166 证不可达；sepDone=false/readFail 帧 1b 不 wait
            // evGate（:921 有 j->sepDone 守卫）——RLE 跳过帧安全。水位无条件
            // 推进（gateFed_ 与原 :897 无条件 store 语义对齐）
            lmaskFed_.store(j->idx);
            rleCopied_.store(j->idx);
            gateFed_.store(j->idx);
            notifyGate_();
        }
        }   // 【N-6】legacy 逐段路径结束（bigDone 分支绕过）
        if (busyProbe_ && metricsOn_ && recs_)
            cudaEventRecord(evBusy1aB_[static_cast<size_t>(j->idx) % static_cast<size_t>(busyRing_ + 1)],
                            cv::cuda::StreamAccessor::getStream(gS_));
        recordBusySpan_(j->idx, true);
        ch1ab_.push(j);
        if (metricsOn_ && recs_)
            recs_[static_cast<size_t>(j->idx)].v[kFcWall1a]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
    }
    ch1ab_.close();
}

void FiveChainExecutor::chain1bLoop() {
    jmw::census::setStage("1b");
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
        // 【probe v2】1b 段快照 D2H 入队（gS2_，同流）——入队先于下方排水（evD_/
        // waitForCompletion）＝宿主落地由既有排水覆盖；frontKind==0 未提交任何
        // 内核，prod 视图是上帧残数据——跳过（宁缺勿错）
        if (frontKind != 0) probeEnqueue1b_(j);
        // 槽释放（无条件——含 sep 失败/未提交帧；事件语义：d_gray/lmask 最后读者=flat，
        // 未提交帧空流上事件立即触发＝正确放行）
        cudaEventRecord(evSlot_[static_cast<size_t>(j->idx % graySlots_)],
                        cv::cuda::StreamAccessor::getStream(gS2_));
        {
            std::lock_guard<std::mutex> lk(fedMtx_);
            fedIdx_ = j->idx;
        }
        fedCv_.notify_all();
        // 【B5】lmask 环槽释放已并入上方 evSlot_（:940 按 idx%8 无条件记录——掩膜
        // 环与灰槽同模同槽；该事件双消费者＝upload 灰槽复用＋1a 下一代 D2D 覆写闸）。
        // 独立 evLmaskRing_ record 已删（−1 evRecord/帧）；lmaskFed_ 水位保留
        // （1a 提交序守卫——协议解耦，只共享事件通道）。
        lmaskFed_.store(j->idx);
        notifyGate_();

        // ══ 唯一排水点（本帧仅有的宿主同步之一；kind=2 legacy 已内部同步＝空转）══
        if (frontKind != 0) {
            // 【批1·E5】首任务预热实验臂（JMW_V3WARM=1，默认关）：排水前递一笔
            //    零依赖 16B 清零——递交路径预热，赌「首内核被扣 ~0.3ms」挂在
            //    递交冷启动上（若挂在阻塞同步本身则净零，ABBA 判决）
            static const bool warmOn_ = [] {
                const char* e = getenv("JMW_V3WARM");
                return e != nullptr && e[0] == '1';
            }();
            if (warmOn_) {
                if (!warmScratchD_) cudaMalloc(&warmScratchD_, 4 * sizeof(int));
                if (warmScratchD_)
                    cudaMemsetAsync(warmScratchD_, 0, 4 * sizeof(int),
                                    cv::cuda::StreamAccessor::getStream(gS2_));
            }
            // 【默认开·2026-10-29】排水等待姿势（仅本线程）：
            //   默认＝省力空转（PSPIN：PAUSE 批轮询，帧率 −1.71 噪声级中性）；
            //   JMW_PSPIN=0 显式关回纯自旋；JMW_BSONLY=1 挂起（已证伪 −16.1）。
            //   归档：1b 归因档 §三.6
            static const int drainMode_ = [] {
                const char* b = getenv("JMW_BSONLY");
                if (b != nullptr && b[0] == '1') return 1;
                const char* p = getenv("JMW_PSPIN");
                if (p != nullptr && p[0] == '0') return 0;
                return 2;
            }();
            if (drainMode_ != 0) {
                static cudaEvent_t evD_ = [] {
                    cudaEvent_t ev = nullptr;
                    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                    return ev;
                }();
                static cudaEvent_t evDB_ = [] {
                    cudaEvent_t ev = nullptr;
                    cudaEventCreateWithFlags(&ev, cudaEventBlockingSync);
                    return ev;
                }();
                cudaStream_t raw2_ = cv::cuda::StreamAccessor::getStream(gS2_);
                if (drainMode_ == 1) {
                    cudaEventRecord(evDB_, raw2_);
                    cudaEventSynchronize(evDB_);
                } else {
                    cudaEventRecord(evD_, raw2_);
                    for (;;) {
                        const cudaError_t q = cudaEventQuery(evD_);
                        if (q == cudaSuccess) break;
                        if (q != cudaErrorNotReady) {
                            cudaGetLastError();
                            cudaEventSynchronize(evD_);
                            break;
                        }
                        for (int i = 0; i < 96; ++i) _mm_pause();
                    }
                }
            } else {
                gS2_.waitForCompletion();
            }
            j->prod.gpuDrained = true;   // 【E3】含 evGate 传递→RLE D2H 落地
            // 【批1·E3】后置延迟对实验臂（JMW_V3POSTDEF=1，默认关）：排水后
            // 「立刻」递 v3 前段（容量视图）——GPU 随即开始爬放行坡；随后的
            // finishLaserFront 宿主活（~0.1-0.2ms，原为 GPU 空窗）与 RLE 拷出
            // 塞进 FinishFront 的等待窗。与已证伪 JMW_V3DEFER（排水「前」提交
            // ＝执行塞进排水窗、串行化交接）不同：本臂排水时长分毫不变，只消灭
            // 「排水后到首笔提交」的空窗。与 V3DEFER/FRONTGRAPH 互斥（先占
            // v3Deferred；legacy 臂不用）。
            static const bool postdefOn_ = [] {
                const char* e = getenv("JMW_V3POSTDEF");
                return e != nullptr && e[0] == '1';
            }();
            static const bool postdefLegacy_ = getenv("JMW_K2A_LEGACY") != nullptr
                                               || getenv("JMW_K2_LEGACY") != nullptr;
            if (postdefOn_ && !postdefLegacy_ && frontKind == 1 && !j->v3Deferred
                && j->prod.interpL.success && j->prod.interpR.success)
                submitV3FrontCapacity_(j, tubeOdd, gS2_);
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
        // 早退失败帧（flat 前返回）保留事件同步兜底（【N-2】改等 evGate_——与旧
        // evRleRing_ 同位点记录且在同流更后，等待它⟹RLE 亦完成；触发条件不变：
        // sepDone 帧 1a 必已记 evGate_，readFail 墓碑在函数头直通不达此）
        if (j->sepDone && j->prod.sepL.d_markingPointMask) {
            const int rs = j->idx % rleSlots_;
            if (!j->prod.gpuDrained)
                cudaEventSynchronize(evGate_);
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
                                   evGate_);
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
    { char tag[16]; snprintf(tag, sizeof(tag), "mk%d", laneIdx); jmw::census::setStage(tag); }
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
    jmw::census::setStage("reg");
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
    jmw::census::setStage("ch2");
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
    // 【批0·仪表】前段计时串观测通道（JMW_V3FRT=1）：step1-5/step6/sync1/
    // dl2+sync2/copyout/total 逐帧落 stderr——copyout 段此前无任何观测口径
    static const bool frtOn_ = [] {
        const char* e = getenv("JMW_V3FRT");
        return e != nullptr && e[0] == '1';
    }();
    std::vector<int> idsScratch(static_cast<size_t>(eCapPts_));
    std::vector<double> dDisp;
    void* v = nullptr;
    while (chD_.pop(v)) {
        FcJob* j = static_cast<FcJob*>(v);
        const long long t0 = steadyNs();
        chCount_[5].fetch_add(1);
        if (fullGpu_) {
            ch2FullGpu_(j);
        } else {   // —— 分体旧路径（GPU 前段＋CPU 后段＋CPU 重建）——
        calib::LaserMatchScanCpuResultV3 cr;
        if (!j->readFail && j->front.success) {
            if (frtOn_ && !j->front.timing.empty())
                fprintf(stderr,
                        "[fc:frt] idx=%d N=%d M=%d segs=%d cards=%d %s\n",
                        j->idx, j->front.leftCount, j->front.rightCount,
                        j->front.segCount, j->front.totalCards,
                        j->front.timing.c_str());
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
            j->prod.laserMatched = cr.matchedCount;   // 原值（门关闭时即最终口径）
            // 【离面鬼影修复 2026-10-30】分体后段补帧级视差一致性门（与 v3
            // 整体版 disp_gate 同源同参）：splitCpu 后段在 host 出对，直接
            // host 中位＋原地紧缩。整段错投线号的跨线错配视差系统性偏离主面
            // （实测 −145px → 重建 +120~170mm 离面鬼影片）。重建（ReconstructInto）
            // 消费紧缩后的对集。门在 timeEnd 之前＝耗时计入 laser_match_scan_v3
            // 步骤计时（与串行 harness 门同口径）。
            if (cr.success && cr.matchedCount > 8 && cr.matched_left.total() > 0) {
                static const auto v3pGate = calib::LaserMatchScanParamsV3::fromJson(
                    h_.cfg_.opParams("laser_match_scan_v3"));
                if (v3pGate.disp_gate) {
                    const int n = cr.matchedCount;
                    cv::Point2f* mlg = cr.matched_left.ptr<cv::Point2f>();
                    cv::Point2f* mrg = cr.matched_right.ptr<cv::Point2f>();
                    int* lidg = cr.matched_line_ids.ptr<int>();
                    int hist[1024] = {0};
                    for (int i = 0; i < n; ++i) {
                        const float d = mlg[i].x - mrg[i].x;
                        int b = static_cast<int>(d * 0.25f);
                        if (b < 0) b = 0;
                        if (b > 1023) b = 1023;
                        ++hist[b];
                    }
                    int acc = 0, medBin = 0;
                    for (int b = 0; b < 1024; ++b) {
                        acc += hist[b];
                        if (acc >= n / 2) { medBin = b; break; }
                    }
                    const float med = (medBin + 0.5f) * 4.0f;
                    const float tol = v3pGate.disp_gate_px;
                    int w = 0;
                    for (int i = 0; i < n; ++i) {
                        const float d = mlg[i].x - mrg[i].x;
                        if (std::fabs(d - med) > tol) continue;
                        mlg[w] = mlg[i];
                        mrg[w] = mrg[i];
                        lidg[w] = lidg[i];
                        ++w;
                    }
                    dispGateDropped_.fetch_add(n - w, std::memory_order_relaxed);
                    // 【审核⑧】丢弃占比>40% 告警：超 50% 时帧中位翻转到鬼影簇、
                    // 门会弃好保坏（实测最坏帧占比 20.5%）——留 40% 线提前示警
                    if (n - w > n * 2 / 5)
                        fprintf(stderr,
                                "[fc:dispGate] 警告 idx=%d 丢弃 %d/%d (%.0f%%) 超 40%%"
                                "——数据异常或场景多深度（应关 disp_gate）\n",
                                j->idx, n - w, n, 100.0 * (n - w) / n);
                    j->prod.laserMatched = w;   // 紧缩后口径（覆盖 Execute 原值）
                    cr.matchedCount = w;
                }
            }
            h_.setStepOk(j->prod, "laser_match_scan_v3", cr.success);
            h_.timeEnd("laser_match_scan_v3", j->prod, j->record, false);
            // 【probe v2】命中帧匹配对快照（chain2 段＝纯宿主拷贝——在线
            // prod.matchV3 从不赋值（cr 是局部量），E 段唯一来源即此处快照；
            // 视差门已就地紧缩，matchedCount＝最终口径）
            if (j->probeSnap && cr.success && cr.matchedCount > 0
                && cr.matched_left.total() > 0 && cr.matched_right.total() > 0) {
                jmw::probe::ProbeSnapshot& sPr = *j->probeSnap;
                const size_t n = static_cast<size_t>(cr.matchedCount);
                const cv::Point2f* pl = cr.matched_left.ptr<cv::Point2f>();
                const cv::Point2f* pr = cr.matched_right.ptr<cv::Point2f>();
                sPr.matchL.assign(pl, pl + n);
                sPr.matchR.assign(pr, pr + n);
                if (!cr.matched_line_ids.empty()) {
                    const int* pi = cr.matched_line_ids.ptr<int>();
                    sPr.matchLine.assign(pi, pi + n);   // 逐对线号（L/R 共用）
                }
            }
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
                    cr.matchedCount, h_.calib_.Q,
                    reconP_.minDepth, reconP_.maxDepth,
                    eBlocks_[static_cast<size_t>(s)], idsScratch.data(), eCapPts_);
                if (cr.matchedCount > eCapPts_)
                    eTruncPts_.fetch_add(cr.matchedCount - eCapPts_);
                applyLaserQualityGates_(j, s, idsScratch.data());
                j->prod.laserPoints3D = j->eCount;
                // 【probe v2】命中帧激光 3D 快照（E 块锁页宿主直拷——门后口径；
                // 在线 prod.recon 从不赋值，E 段唯一来源即此处快照）
                if (j->probeSnap && j->eCount > 0) {
                    const float* eb = eBlocks_[static_cast<size_t>(s)];
                    jmw::probe::ProbeSnapshot& sPr = *j->probeSnap;
                    sPr.laser3d.resize(static_cast<size_t>(j->eCount));
                    for (int q = 0; q < j->eCount; ++q)
                        sPr.laser3d[static_cast<size_t>(q)] =
                            cv::Point3f(eb[3 * q], eb[3 * q + 1], eb[3 * q + 2]);
                }
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
        }   // —— 分体旧路径闭合 ——
        if (metricsOn_ && recs_)
            recs_[static_cast<size_t>(j->idx)].v[kFcWallCh2]
                .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
        chE_.push(j);
        chCount_[6].fetch_add(1);
    }
    chE_.close();
}

// ============================================================================
// 【全GPU链2】ch2 消费体：环槽 interp → 整体版 GPU 匹配（cS_ 流）→ GPU 重建
// → D2H 直写 E 锁页块 → 视差统计（小拷贝）→ 环槽放行。
// 匹配/重建算子实例仅本线程串行使用——内部持久结果缓冲跨帧覆写天然安全。
// ============================================================================
// ============================================================================
// 261030 激光帧质量门＋飞线剪枝（两路共用）【261103 集成重放自旧产线】
// Z-IQR 门：错管型/错表匹配产生随机深度弥散帧（白板实证：幻影帧 Z-IQR~90mm
// vs 干净帧 <5mm，交替出现＝tubeOdd 单值时另一管全错）。双判据（IQR 超限且
// 均匀弥散形态）＝整帧弃（eCount 归零——不融合不缓存）。env JMW_FC_ZIQR
//（默认 60mm，0=关）。真实单视场景深离散通常 <40mm——误杀可控。
// 飞线门：真实激光线共面于被扫表面；错线匹配产出「飞线」＝整段错深度但线内
// 稠密。帧主平面（幂迭代）→逐线离面统计：离面占比>50%＝整线剔；否则剔线内
// 离面散点。E 块就地压缩。ids 空＝跳过飞线门（仅 Z 门）。
// ============================================================================
void FiveChainExecutor::applyLaserQualityGates_(FcJob* j, int slot,
                                                const int* ids) {
    if (j->eCount <= 200) return;
    // —— Z-IQR 弥散门 ——
    static const int sZgate = [] {
        const char* e = getenv("JMW_FC_ZIQR");
        return e ? atoi(e) : 60;
    }();
    if (sZgate > 0) {
        zScratch_.resize(static_cast<size_t>(j->eCount));
        const float* eb0 = eBlocks_[static_cast<size_t>(slot)];
        for (int q = 0; q < j->eCount; ++q)
            zScratch_[static_cast<size_t>(q)] = eb0[q * 3 + 2];
        const size_t n = zScratch_.size();
        std::sort(zScratch_.begin(), zScratch_.end());
        const float p25 = zScratch_[n / 4];
        const float p75 = zScratch_[n * 3 / 4];
        const float p5 = zScratch_[n / 20];
        const float p95 = zScratch_[n * 19 / 20];
        const float iqr = p75 - p25;
        const float range = p95 - p5;
        // 双判据：IQR 超限 且 均匀弥散形态（IQR/范围≈0.5＝均匀分布；真实
        // 表面结构有尾部 → 比值低。测试数据深结构场景实测 IQR 49mm 合法
        // ——单 IQR 会误杀，联合判据放行）
        const bool uniform = range > 1.0f && iqr / range > 0.42f;
        if (iqr > static_cast<float>(sZgate) && uniform) {
            zGateDrops_.fetch_add(1, std::memory_order_relaxed);
            if (zGateDrops_.load(std::memory_order_relaxed) % 100 == 1)
                fprintf(stderr,
                        "[fc:zgate] idx=%d 弃帧（IQR=%.0fmm>%dmm "
                        "均匀度=%.2f——错管型/错表弥散帧）累计=%zu\n",
                        j->idx, iqr, sZgate, iqr / range,
                        static_cast<size_t>(zGateDrops_.load()));
            j->eCount = 0;   // 整帧弃：不融合不缓存（E 块照常归还）
            return;
        }
    }
    // —— 帧内逐线离面剪枝（飞线门；ids 空＝跳过）——
    if (linePlaneGateMm_ <= 0 || ids == nullptr) return;
    const int sLinePlane = linePlaneGateMm_;
    float* eb = eBlocks_[static_cast<size_t>(slot)];
    const int n = j->eCount;
    // 帧主平面（质心＋协方差幂迭代）
    double mx = 0, my = 0, mz = 0;
    for (int q = 0; q < n; ++q) {
        mx += eb[q * 3]; my += eb[q * 3 + 1]; mz += eb[q * 3 + 2];
    }
    mx /= n; my /= n; mz /= n;
    double a = 0, b = 0, d2 = 0, e2 = 0, f2 = 0, g = 0;
    for (int q = 0; q < n; ++q) {
        const double x = eb[q * 3] - mx, y = eb[q * 3 + 1] - my,
                     z = eb[q * 3 + 2] - mz;
        a += x * x; b += x * y; d2 += x * z;
        e2 += y * y; f2 += y * z; g += z * z;
    }
    double v1[3] = {1, 0.3, 0.1};
    for (int it = 0; it < 40; ++it) {
        double w[3] = {a * v1[0] + b * v1[1] + d2 * v1[2],
                       b * v1[0] + e2 * v1[1] + f2 * v1[2],
                       d2 * v1[0] + f2 * v1[1] + g * v1[2]};
        const double L = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (L < 1e-12) break;
        v1[0] = w[0] / L; v1[1] = w[1] / L; v1[2] = w[2] / L;
    }
    double v2[3] = {v1[0] + 0.37, v1[1] + 0.61, v1[2] + 0.93};
    for (int it = 0; it < 40; ++it) {
        double w[3] = {a * v2[0] + b * v2[1] + d2 * v2[2],
                       b * v2[0] + e2 * v2[1] + f2 * v2[2],
                       d2 * v2[0] + f2 * v2[1] + g * v2[2]};
        const double dot = v1[0] * w[0] + v1[1] * w[1] + v1[2] * w[2];
        w[0] -= v1[0] * dot; w[1] -= v1[1] * dot;
        w[2] -= v1[2] * dot;
        const double L = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (L < 1e-12) break;
        v2[0] = w[0] / L; v2[1] = w[1] / L; v2[2] = w[2] / L;
    }
    double nv[3] = {v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0]};
    const double nL = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
    if (nL <= 1e-12) return;
    nv[0] /= nL; nv[1] /= nL; nv[2] /= nL;
    const float thr = static_cast<float>(sLinePlane);
    // 逐线统计离面占比
    std::unordered_map<int, std::pair<int, int>> lineStat;
    lineStat.reserve(256);
    for (int q = 0; q < n; ++q) {
        auto& st = lineStat[ids[q]];
        ++st.second;
        const double r = nv[0] * (eb[q * 3] - mx)
                       + nv[1] * (eb[q * 3 + 1] - my)
                       + nv[2] * (eb[q * 3 + 2] - mz);
        if (std::abs(r) > thr) ++st.first;
    }
    // 就地压缩：飞线（离面>50%）整线剔；其余线剔离面散点
    int w = 0;
    int pruned = 0;
    for (int q = 0; q < n; ++q) {
        const auto& st = lineStat[ids[q]];
        const double r = nv[0] * (eb[q * 3] - mx)
                       + nv[1] * (eb[q * 3 + 1] - my)
                       + nv[2] * (eb[q * 3 + 2] - mz);
        const bool drop = (st.first * 2 > st.second) || (std::abs(r) > thr);
        if (drop) {
            ++pruned;
            continue;
        }
        eb[w * 3] = eb[q * 3];
        eb[w * 3 + 1] = eb[q * 3 + 1];
        eb[w * 3 + 2] = eb[q * 3 + 2];
        ++w;
    }
    if (pruned > 0) {
        zLinePruned_.fetch_add(static_cast<size_t>(pruned), std::memory_order_relaxed);
        j->eCount = w;
    }
}

void FiveChainExecutor::ch2FullGpu_(FcJob* j) {
    calib::LaserMatchScanResultV3 mr;
    if (!j->readFail && j->front.success) {
        const int s = j->idx % iRingSlots_;
        const int nl = j->prod.interpL.interpCount;
        const int nr = j->prod.interpR.interpCount;
        cv::cuda::GpuMat vPL(iRingPtsL_[static_cast<size_t>(s)],
                             cv::Rect(0, 0, nl, 1));
        cv::cuda::GpuMat vIL(iRingIdsL_[static_cast<size_t>(s)],
                             cv::Rect(0, 0, nl, 1));
        cv::cuda::GpuMat vPR(iRingPtsR_[static_cast<size_t>(s)],
                             cv::Rect(0, 0, nr, 1));
        cv::cuda::GpuMat vIR(iRingIdsR_[static_cast<size_t>(s)],
                             cv::Rect(0, 0, nr, 1));
        const bool tubeOdd =
            h_.src_->cycles[static_cast<size_t>(j->idx)].tubeOdd;
        // 等 1b 的 D2D 拷贝完成（跨流事件）——之后 cS_ 流才可读环槽
        cudaStreamWaitEvent(cv::cuda::StreamAccessor::getStream(cS_),
                            evIRing_[static_cast<size_t>(s)], 0);
        h_.timeBegin();
        matchFull_->SetTempTable(tubeOdd ? h_.tableOdd_ : h_.tableEven_);
        matchFull_->SetCurrentTemperature(h_.cfg_.runtime.nominalTemperature);
        mr = matchFull_->Execute(vPL, vIL, vPR, vIR, cS_);
        j->prod.laserMatched = mr.matchedCount;
        h_.setStepOk(j->prod, "laser_match_scan_v3", mr.success);
        h_.timeEnd("laser_match_scan_v3", j->prod, j->record, false);
        if (!mr.success)
            h_.fail("laser_match_scan_v3", j->prod,
                    "整体版失败: " + mr.message);
    } else if (!j->readFail) {
        h_.fail("laser_match_scan_v3", j->prod,
                "前段失败墓碑: " + j->front.message);
    }
    // 【probe v2】命中帧匹配对快照（全GPU臂）：整体版 Execute 计数回传已内部
    // 同步——此处同步 cudaMemcpy 与本函数既有 E 块/视差 D2H 同风险面；下一
    // 覆写者＝下一帧 Execute（同线程），数据仍为本帧
    if (j->probeSnap && mr.success && mr.matchedCount > 0) {
        jmw::probe::ProbeSnapshot& sPr = *j->probeSnap;
        const size_t n = static_cast<size_t>(mr.matchedCount);
        bool dlOk = true;
        auto dlSync = [&j, &dlOk](void* dst, const void* src, size_t bytes,
                                  const char* what) {
            if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                cudaGetLastError();   // 清残留（防毒害后续消费者）
                dlOk = false;
                JMW_LOG_WARN("07-FiveChain",
                             "[fc:probe] idx={} 快照下载 {} 失败——该段弃件（宁缺勿错）",
                             j->idx, what);
            }
        };
        if (mr.d_matched_left && mr.d_matched_right) {
            sPr.matchL.resize(n);
            sPr.matchR.resize(n);
            dlSync(sPr.matchL.data(), mr.d_matched_left->ptr(),
                   sizeof(cv::Point2f) * n, "matchL");
            dlSync(sPr.matchR.data(), mr.d_matched_right->ptr(),
                   sizeof(cv::Point2f) * n, "matchR");
            if (dlOk && mr.d_matched_line_ids) {
                sPr.matchLine.resize(n);
                dlSync(sPr.matchLine.data(), mr.d_matched_line_ids->ptr(),
                       sizeof(int) * n, "matchLine");
            }
        }
    }
    j->prod.laserPoints3D = 0;
    if (mr.success && mr.matchedCount > 0) {
        h_.timeBegin();
        calib::LaserReconstructResult rr =
            reconGpu_->Execute(*mr.d_matched_left, *mr.d_matched_right,
                               *mr.d_matched_line_ids, h_.calib_.Q, cS_);
        h_.setStepOk(j->prod, "laser_reconstruct", rr.success);
        if (rr.success && rr.validCount > 0) {
            const int s2 = acquireEBlock();
            if (s2 >= 0) {
                j->eSlot = s2;
                const int n = std::min<int>(rr.validCount, eCapPts_);
                // 重建 Execute 内部计数回传后已同步——D2H 即刻安全；
                // E 块契约＝连续 xyz 浮点（融合只消费位置，与分体路径一致）
                cudaMemcpy(eBlocks_[static_cast<size_t>(s2)],
                           rr.d_points3d->ptr(),
                           sizeof(float) * 3ull * static_cast<unsigned>(n),
                           cudaMemcpyDeviceToHost);
                j->eCount = n;
                j->prod.laserPoints3D = n;   // 截断后实入 E 块口径（与分体一致）
                if (mr.matchedCount > eCapPts_)
                    eTruncPts_.fetch_add(mr.matchedCount - eCapPts_);
                // 飞线门需逐点线号（fullGpu 路线号在设备——按需 D2H）
                const int* fgIds = nullptr;
                if (j->eCount > 200 && linePlaneGateMm_ > 0) {
                    zIdsScratch_.resize(static_cast<size_t>(n));
                    cudaMemcpy(zIdsScratch_.data(),
                               mr.d_matched_line_ids->ptr(),
                               sizeof(int) * static_cast<size_t>(n),
                               cudaMemcpyDeviceToHost);
                    fgIds = zIdsScratch_.data();
                }
                applyLaserQualityGates_(j, s2, fgIds);
                j->prod.laserPoints3D = j->eCount;
                // 【probe v2】命中帧激光 3D 快照（E 块已 D2H——纯宿主拷贝，门后口径）
                if (j->probeSnap && j->eCount > 0) {
                    const float* eb = eBlocks_[static_cast<size_t>(s2)];
                    jmw::probe::ProbeSnapshot& sPr = *j->probeSnap;
                    sPr.laser3d.resize(static_cast<size_t>(j->eCount));
                    for (int q = 0; q < j->eCount; ++q)
                        sPr.laser3d[static_cast<size_t>(q)] =
                            cv::Point3f(eb[3 * q], eb[3 * q + 1], eb[3 * q + 2]);
                }
            }
        } else if (!rr.success) {
            h_.fail("laser_reconstruct", j->prod,
                    "GPU 重建失败: " + rr.message);
        }
        h_.timeEnd("laser_reconstruct", j->prod, j->record, false);
        // 视差统计（host 小拷贝——口径与分体路径一致；rowResid 按构造恒 0）
        if (!h_.prodProfile_) {
            const size_t n = static_cast<size_t>(mr.matchedCount);
            hML_.resize(n);
            hMR_.resize(n);
            cudaMemcpy(hML_.data(), mr.d_matched_left->ptr(),
                       sizeof(cv::Point2f) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(hMR_.data(), mr.d_matched_right->ptr(),
                       sizeof(cv::Point2f) * n, cudaMemcpyDeviceToHost);
            hDisp_.resize(n);
            for (size_t i = 0; i < n; ++i)
                hDisp_[i] = static_cast<double>(hML_[i].x - hMR_[i].x);
            j->prod.rowResidMedPx = 0.0;
            j->prod.dispMedPx = medianOf(hDisp_);
        }
    }
    // 环槽放行（匹配已消费环槽数据——此后 1b 可覆写该槽）
    iRingDone_.store(j->idx);
    notifyGate_();
}

// ============================================================================
// CUDA链2：C∩E 按帧号到齐（定容环配对）→ hold 零上传弃 E / 成功帧上传＋融合
// → F（帧序产出）
// ============================================================================
void FiveChainExecutor::fuseLoop() {
    jmw::census::setStage("fuse");
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
        // 【P0-b】滞后回收：上一帧起的 E 池槽，随上一帧 Finish 的流同步一并释放
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
            // 【查BUG 2026-10-30】JMW_FC_FRAME_DUMPS=<dir>：融合前逐帧落 E 块
            // （相机系 xyz，宿主内存直写）＋R/T＋regSource（五链离面归因；默认关）。
            if (const char* fd2 = getenv("JMW_FC_FRAME_DUMPS")) {
                char dbuf[512];
                std::snprintf(dbuf, sizeof(dbuf), "%s/rt.csv", fd2);
                std::ofstream rt(dbuf, std::ios::app);
                if (rt)
                    rt << j->idx << ',' << rs << ','
                       << j->prod.R[0] << ',' << j->prod.R[1] << ','
                       << j->prod.R[2] << ',' << j->prod.R[3] << ','
                       << j->prod.R[4] << ',' << j->prod.R[5] << ','
                       << j->prod.R[6] << ',' << j->prod.R[7] << ','
                       << j->prod.R[8] << ',' << j->prod.T[0] << ','
                       << j->prod.T[1] << ',' << j->prod.T[2] << '\n';
                if (j->eCount > 0 && j->eSlot >= 0) {
                    std::snprintf(dbuf, sizeof(dbuf), "%s/f%03d.ply", fd2, j->idx);
                    std::ofstream ply(dbuf);
                    const float* eb = eBlocks_[static_cast<size_t>(j->eSlot)];
                    ply << "ply\nformat ascii 1.0\nelement vertex " << j->eCount
                        << "\nproperty float x\nproperty float y\n"
                        << "property float z\nend_header\n" << std::fixed
                        << std::setprecision(3);
                    for (int q = 0; q < j->eCount; ++q)
                        ply << eb[3 * q] << ' ' << eb[3 * q + 1] << ' '
                            << eb[3 * q + 2] << '\n';
                }
            }
            // 【probe v2】E 段钩子（快照制）：probe 开时逐帧 rt 行；命中帧消费
            // 产出线程快照落盘全套（纯宿主读写零 CUDA——GPU 视图过期问题由
            // 快照根治）；落盘后即释放快照（诊断件内存界＝命中帧快照全集峰值）
            if (probe_ != nullptr) {
                probe_->appendRt(j->idx, j->prod);
                if (j->probeSnap) {
                    probe_->maybeDump(j->idx, j->prod, *j->probeSnap);
                    j->probeSnap.reset();
                }
            }
            // 【P5】holdSkip 演练：全部帧按 hold 处理（零上传弃 E 路径验证）
            const bool fused = h_.cfg_.fivechain.holdSkip == 0
                               && (rs == "first" || rs == "flow" || rs == "fuse")
                               && j->eCount > 0 && j->eSlot >= 0;
            if (fused) {
                const long long t0 = steadyNs();
                h_.timeBegin();
                // 【P0-b】滞后记账：先落地上一帧的 Finish（同步近似零等待——
                // 距其入队已过一整帧），再本帧 Deferred（零阻塞发射）。
                if (fusePending_) {
                    auto prev = h_.voxelFuseOp_->Finish();
                    fusePending_ = false;
                    if (prev.success) accountFuseResult_(prev);
                    // Finish 已同步整条 fS_ 流 → 在飞 E 槽的 H2D 全部落定，批量回收
                    drainPendingERelease_();
                }
                const int eSlot = j->eSlot;
                cv::cuda::GpuMat view;
                if (eZero_) {
                    // 【N-4·实验臂（默认关，见 :296 注释）】零拷贝：mapped 设备指
                    // 针直入 fuse（参数块 d_inputXyz 每帧现写当帧指针——:596 曾误
                    // 传建图帧旧指针＝根因②，2026-10-30 已修）；无 H2D。E 块回收
                    // 协议：Finish 同步＝内核完成后才回收（复用必经 release→
                    // acquire 环）——该协议经 2026-10-30 多轮审核确认对"当帧槽"
                    // 防护完好；零拷贝臂残余风险在"正确性依赖此推理成立"本身，
                    // 默认关即为此（拷贝路径构造性隔离）
                    view = cv::cuda::GpuMat(
                        1, j->eCount, CV_32FC3,
                        eDevBlocks_[static_cast<size_t>(eSlot)]);
                } else {
                    view = d_pts_(cv::Rect(0, 0, j->eCount, 1));
                    cudaMemcpyAsync(view.ptr(),
                                    eBlocks_[static_cast<size_t>(eSlot)],
                                    sizeof(float) * 3ull * static_cast<unsigned>(j->eCount),
                                    cudaMemcpyHostToDevice,
                                    cv::cuda::StreamAccessor::getStream(fS_));
                }
                // 【P0-b】E 块生命周期移交 pendingERelease_：下一帧 Finish() 同步
                // 整条 fS_ 流，届时本帧 H2D/内核读必然完成——零事件零额外 API 批量回收
                pendingERelease_.push_back(eSlot);
                j->eSlot = -1;   // 已移交，走通用回收路径之外
                auto res = h_.voxelFuseOp_->ExecuteDeferred(
                    view, cv::Matx33d(j->prod.R),
                    cv::Vec3d(j->prod.T[0], j->prod.T[1], j->prod.T[2]), fS_);
                if (!res.success) {
                    fS_.waitForCompletion();   // 防御：排干后批量回收
                    drainPendingERelease_();
                    std::cout << "[fc:fuse] ExecuteDeferred 失败: " << res.message << std::endl;
                } else {
                    fusePending_ = true;
                }
                h_.timeEnd("accumulate_global_laser", j->prod, j->record, false);
                if (metricsOn_ && recs_)
                    recs_[static_cast<size_t>(j->idx)].v[kFcWallFuse]
                        .store((steadyNs() - t0) / 1000, std::memory_order_relaxed);   // us
            }
            if (j->eSlot >= 0) {
                releaseEBlock(j->eSlot);   // 未融合帧无 H2D，直接回收
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
    // 【P0-b·尾帧 Flush】末帧 Finish 无 N+1 帧触发——显式落地保账本闭合
    // （Σinput==ΣlaserPoints3D 等断言依赖），并强制回收全部在飞 E 槽。
    if (fusePending_) {
        auto last = h_.voxelFuseOp_->Finish();
        fusePending_ = false;
        if (last.success) accountFuseResult_(last);
    }
    drainPendingERelease_();
    chF_.close();
}

// ============================================================================
// 【P0-b】融合滞后记账 + E 块滞后回收
// ============================================================================
void FiveChainExecutor::accountFuseResult_(
    const calib::LaserCloudFuseCudaResult& res) {
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

void FiveChainExecutor::drainPendingERelease_() {
    while (!pendingERelease_.empty()) {
        releaseEBlock(pendingERelease_.front());
        pendingERelease_.pop_front();
    }
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
    fprintf(f, "dispGateDropped,%lld,\n", dispGateDropped_.load());   // 【观测⑦】视差门累计丢弃
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
