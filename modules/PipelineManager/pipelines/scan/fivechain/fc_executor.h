#pragma once
// ============================================================================
// fc_executor.h — 【五链 P2】五链流水线执行器（8 线程骨架）
// ============================================================================
// 设计：五链并行流水线重构-设计方案-2026-10-21.md §二/§十 P2＋实现规范。
// 线程（P2 最小可运行集）：
//   读帧(浮) → 预上传(浮) → CUDA链1（默认二分 1a/1b：分离→RLE ∥ flat→…→v3前段）
//     ├─ A → CPU链1×lanes(CCL→…→conic) → B → CPU链3(急取重排→配准) → C ─┐
//     └─ D → CPU链2(v3后段→CPU激光重建→E锁页块) ────────── E ─┘
//                → CUDA链2(到齐配对→hold弃/成功融合) → F → 主线程收割
// 正确性要点（设计 §六/§七）：
// - 帧号协议：一切通道元素带帧号；CPU链3 急取式重排（B 弹出永不按帧序阻塞）；
//   CUDA链2 C∩E 按帧号到齐；hold 帧零上传弃 E；每帧每通道恰一元素（读帧失败
//   ＝全程墓碑 Job 流到 F，配准走 skip 保 prevState 链不断）。
// - 背压：通道满=压生产者永不丢帧；在飞上限=maxInflight（读帧线程让路）。
// - 灰度环：槽释放双保险（fed 提交序计数器＋evSlot 执行序事件）——d_gray 的
//   读者是分离＋flat 两步（同流 FIFO＋事件跨流）。
// - Job 生命周期：【零分配池化】FcJob 池（容量=在飞上限＋环深余量）——collect
//   收割后归还，rleL/rleR 容量跨帧复用（规范 §五陷阱 2）。
// - 计时/记账复用 Harness 线程安全路径（stepRecMtx_ 护栏；worker 用 thread_local
//   基点）；步骤名与串行/pipe 完全一致（report/timing CSV 口径不变）。
// - 【指标】FcFrameRec 逐帧 POD 记录（链墙钟/唤醒/在飞/GPU busy 近似）→
//   析构落盘 fc_metrics.csv＋fc_summary.csv（JMW_FCMETRICS=0 关闭）。
#include <array>
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "harness.h"
#include "scr_config.h"

#include "laser_match_scan_v3_split.h"
#include "laser_reconstruct_cpu.h"

namespace scr {
namespace fc {

// 【P4】线程绑核＋提优先级（config fivechain.pinRoles/priorityBoost；JMW_AFFINITY=0 强制关）
// 核分配：跳过 0 号物理核（中断核），roleIdx 递增取物理核首逻辑核（pipePinThread 范式）
// 【E1】fcPinThreadLP：绑指定物理核的第 lpOffset 逻辑核（ch2 用——1b 的 SMT 兄弟核）；
// 逻辑核数不足时不绑（绝不回退 LP0——那等于与该核首线程硬串行）
// 【H1/H2·2026-10-24 核分配专题】fuse/reg 默认仍浮动（等价现状），env 旁路（默认全关）：
//   JMW_FCPIN_FUSE=N / JMW_FCPIN_REG=N：fuse/reg 绑物理核 N-LP1＋HIGHEST（仿 JMW_FCPIN_CH2）
//   JMW_PRIO_DECOUPLE=1：marker×3/ch2 降 ABOVE_NORMAL（1a/1b 保持 HIGHEST；fcPrioFor 派生）
void fcPinThread(int roleIdx, const char* who);
void fcPinThreadLP(int coreIdx, int lpOffset, const char* who);

// 通用有界阻塞通道（满=压生产者；空=消费者眠；关闭传染）
// 【零分配】定容环（构造一次分配；运行期零堆分配——替换原 deque 实现的
// 每 2 元素一块的块分配 churn，规范 §五陷阱 2）
// ============================================================================
class BoundedQueue {
public:
    explicit BoundedQueue(int cap, int producers = 1)
        : ring_(static_cast<size_t>(std::max(1, cap))), cap_(std::max(1, cap)),
          producers_(producers) {}
    void addProducer() {
        std::lock_guard<std::mutex> lk(m_);
        ++producers_;
    }

    bool push(void* v) {
        std::unique_lock<std::mutex> lk(m_);
        cvF_.wait(lk, [&] { return closed_ || count_ < cap_; });
        if (closed_) return false;
        ring_[static_cast<size_t>(head_)] = v;
        head_ = (head_ + 1) % cap_;
        ++count_;
        lk.unlock();
        cvN_.notify_all();
        return true;
    }
    bool pop(void*& out) {
        std::unique_lock<std::mutex> lk(m_);
        cvN_.wait(lk, [&] { return closed_ || count_ > 0; });
        if (count_ == 0) return false;   // closed_ && drained
        out = ring_[static_cast<size_t>(tail_)];
        tail_ = (tail_ + 1) % cap_;
        --count_;
        lk.unlock();
        cvF_.notify_all();
        return true;
    }
    /// 谓词等待＋超时（配对轮询用；超时/关闭且空返 false）
    bool popFor(void*& out, int timeoutMs) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cvN_.wait_for(lk, std::chrono::milliseconds(timeoutMs),
                           [&] { return closed_ || count_ > 0; }))
            return false;
        if (count_ == 0) return false;
        out = ring_[static_cast<size_t>(tail_)];
        tail_ = (tail_ + 1) % cap_;
        --count_;
        lk.unlock();
        cvF_.notify_all();
        return true;
    }
    /// push 并返回入队前通道深度（指标：ofoB 深度＋链3 唤醒判定＝首入空队者记时戳）
    int pushRetPrevCount(void* v) {
        std::unique_lock<std::mutex> lk(m_);
        cvF_.wait(lk, [&] { return closed_ || count_ < cap_; });
        if (closed_) return -1;
        const int prev = count_;
        ring_[static_cast<size_t>(head_)] = v;
        head_ = (head_ + 1) % cap_;
        ++count_;
        lk.unlock();
        cvN_.notify_all();
        return prev;
    }
    /// 非阻塞弹（窃取/轮询）
    bool tryPop(void*& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (count_ == 0) return false;
        out = ring_[static_cast<size_t>(tail_)];
        tail_ = (tail_ + 1) % cap_;
        --count_;
        cvF_.notify_all();
        return true;
    }
    int size() {
        std::lock_guard<std::mutex> lk(m_);
        return count_;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        cvN_.notify_all();
        cvF_.notify_all();
    }
    /// MPSC 关闭：各生产者结束各调一次；归零才真关（消费者见"关且空"退出）
    void closeProducer() {
        bool shut = false;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--producers_ <= 0) { closed_ = true; shut = true; }
        }
        if (shut) {
            cvN_.notify_all();
            cvF_.notify_all();
        }
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(m_);
        return closed_;
    }

private:
    std::mutex m_;
    std::condition_variable cvN_, cvF_;
    std::vector<void*> ring_;   // 定容环（构造期分配，运行期零分配）
    int head_ = 0, tail_ = 0, count_ = 0;
    int cap_;
    int producers_ = 1;
    bool closed_ = false;
};

// ============================================================================
// FcJob — 帧上下文（【零分配池化】池取/collect 归还；rle 容量跨帧复用）
// ============================================================================
struct FcJob {
    int idx = -1;
    bool record = true;
    bool readFail = false;          // 读帧失败＝全程墓碑
    bool rleOverflow = false;       // RLE 溢出＝A 墓碑（ccl 记失败、配准走 skip）
    bool sepDone = false;           // 【P4.5】1a 分离成功（1b 事件等待的先决）
    bool v3Deferred = false;        // 【前段提前提交】v3 提交段已发（fcV3Front 走 FinishFront）
    std::string err;

    CycleProducts prod;
    // A 载荷（RLE；GPU lane 同步后拷入）
    std::vector<int> rleL, rleR;
    int rleNL = 0, rleNR = 0;
    // D 载荷（v3 前段值语义产物）
    calib::LaserMatchSplitFrontV3 front;
    // E 载荷（锁页块句柄）
    int eSlot = -1;
    int eCount = 0;
    // 【指标】B 通道入队时间戳（链3 唤醒 P99 分子——生产者侧记录）
    long long bPushNs = 0;

    /// 池归还时复位（容量保留：rleL/rleR/front 内 vector 只 clear 不释放——
    /// 下帧 Execute 的 assign/resize 直接复用容量，零分配）
    void resetForReuse() {
        idx = -1;
        record = true;
        readFail = rleOverflow = sepDone = v3Deferred = false;
        err.clear();
        rleL.clear();
        rleR.clear();
        rleNL = rleNR = 0;
        front.success = false;
        front.message.clear();
        front.timing.clear();
        front.leftCount = front.rightCount = front.segCount = 0;
        front.segOverflow = front.totalCards = 0;
        front.leftPts.clear();
        front.leftTracks.clear();
        front.segmap.clear();
        front.leftStatus.clear();
        front.cardPtr.clear();
        front.cardLid.clear();
        front.cardRx.clear();
        front.cardRidx.clear();
        eSlot = -1;
        eCount = 0;
        bPushNs = 0;
        prod = CycleProducts{};
    }
};

// ============================================================================
// 【指标】逐帧记录（POD；多线程写不同字段——atomic 防撕裂；-1=未记录）
// 字段序＝fc_metrics.csv 列序（写侧与 CSV 侧共用 kFcRec* 枚举）
// ============================================================================
enum FcRecField {
    kFcWall1a = 0,     // 1a 帧 wall（pop→ch1ab 交接）
    kFcWall1b,         // 1b 帧 wall（pop→RLE 拷出完）
    kFcWallMk,         // 标记链处理 wall（处理该帧的 lane 记）
    kFcWallReg,        // 配准 wall
    kFcWallCh2,        // CPU链2 帧 wall
    kFcWallFuse,       // 融合段 wall（上传+Execute+同步）
    kFcRegWake,        // 链3 唤醒：B 入队→pop（μs 口径 P99 门）
    kFcE2E,            // 主线程 F 收割帧间隔
    kFcPush1a,         // 链1 生产者侧 push 合计（A＋D；P5 预算门）
    kFcGpu1a,          // gS_ 段 GPU busy（timing 事件对，近似口径）
    kFcGpu1b,          // gS2_ 段 GPU busy（近似）
    kFcOfoB,           // B 入队时通道深度（在飞观察）
    kFcCount
};
struct FcFrameRec {
    std::atomic<long long> v[kFcCount];
    FcFrameRec() {
        for (auto& x : v) x.store(-1, std::memory_order_relaxed);
    }
};

// ============================================================================
// FiveChainExecutor
// ============================================================================
class FiveChainExecutor {
public:
    FiveChainExecutor(Harness& h, int firstIdx, int totalCycles);
    ~FiveChainExecutor();

    /// 主线程收割第 curIdx 帧产物（F 通道按帧序产出——阻塞等待）
    bool collect(int curIdx, CycleProducts& out, std::string& err);

    // —— 对账计数（P2 验收：七通道 elem 计数 vs totalCycles）——
    // 通道序：0=读 1=上传 2=A 3=B 4=C 5=D 6=E 7=F
    int chCount(int i) const { return chCount_[static_cast<size_t>(i)].load(); }
    int tombCount() const { return tombCount_.load(); }
    int inflightNow() const {
        std::lock_guard<std::mutex> lk(jobMtx_);
        return static_cast<int>(jobs_.size()) - static_cast<int>(jobFree_.size());
    }

private:
    void readerLoop();
    void uploadLoop();
    void chain1Loop();   // CUDA链1 单线程（splitChain1=0）：分离→RLE→steger→前段→RLE 拷出
    void chain1aLoop();   // CUDA链1a（splitChain1=1）：分离→激光掩膜环槽 D2D→RLE 内核发射→evGate 交接
    void chain1bLoop();   // CUDA链1b（splitChain1=1）：flat/undist/interp/v3前段→槽释放→RLE 拷出→推 A/D
    void markerLoop(int laneIdx);   // CPU链1（×lanes，idx%lanes 分派＋窃取）
    void regLoop();        // CPU链3（急取重排）
    void chain2Loop();     // CPU链2（v3 后段＋CPU 重建）
    void fuseLoop();       // CUDA链2（到齐配对＋融合＋F 单点产出）
    void stopAll();

    // CUDA链1 分离段（免上传版——灰度环槽已由预上传就位）
    void fcSeparation(FcJob* j);

    // 【前段提前提交】v3 提交段随 1b 提交相位（排水前）——容量视图＋设备计数
    // 尾闸；false＝未提交（执行器回退排水后整段旧路径）。env：JMW_V3DEFER=0 关
    bool submitV3FrontDeferred_(FcJob* j, bool tubeOdd, cv::cuda::Stream& ls);
    void fcV3Front(FcJob* j, bool tubeOdd, cv::cuda::Stream& ls);

    int acquireEBlock();
    void releaseEBlock(int s);

    // —— Job 池（零分配）——
    FcJob* acquireJob();
    void releaseJob(FcJob* j);
    static long long nowNs();

    // —— 提交序计数器 cv 等待（调优②——见 cpp）——
    void waitCounterGe_(const std::atomic<int>& ctr, int target);
    void notifyGate_();

    // —— 指标 ——
    void writeMetricsCsv_();   // 析构期落盘 fc_metrics/fc_summary

    Harness& h_;
    int firstIdx_;
    int totalCycles_;
    int maxInflight_;
    int graySlots_;
    int rleCap_;

    // —— Job 池 ——
    std::vector<std::unique_ptr<FcJob>> jobs_;   // 池存储（构造期全量建）
    std::vector<FcJob*> jobFree_;                // 空闲栈
    mutable std::mutex jobMtx_;
    std::condition_variable jobCv_;

    BoundedQueue chRead_, chGpu_, chB_, chC_, chD_, chE_, chF_;
    // 【P3】A 通道＝idx%lanes 拆 SPSC 子队列（＋邻队窃取；markerLanes=1 时退化为单队列）
    std::vector<std::unique_ptr<BoundedQueue>> aQs_;
    std::vector<std::thread> mkThs_;
    int mkLanes_ = 1;
    std::array<std::atomic<int>, 8> laneFrames_{};   // per-lane 处理直方图（≤8 lane）

    std::thread thReader_, thUp_, th1_, th1a_, th1b_, thReg_, th2_, thFuse_;
    std::atomic<bool> stop_{false};

    // —— 灰度环（d_grayLanes 复用为环；双保险槽释放——P4.5 后由 1b 侧驱动）——
    cv::cuda::Stream upS_, gS_, fS_;          // 上传流/1a 流/融合流
    cv::cuda::Stream gS2_;                    // 【P4.5】1b 流（flat→undist→interp→前段）
    BoundedQueue ch1ab_;                      // 【P4.5】1a→1b 交接（Job*，容量 4）
    cudaEvent_t evGate_ = nullptr;            // 1a→1b：lmask D2D＋RLE 入队完成（1b 流 wait）
    std::atomic<int> gateFed_{-1};            // evGate_ 提交序（防 wait 捕获旧记录）
    // 【调优①】lmask 单槽→多槽环：1a(N) 写槽 N%S，1b(N-S) 为上一代读者——放松
    // 在飞距 1→S，解耦两流紧耦合（单槽时 1a(N) D2D 须等 1b(N-1) flat 完成）
    std::vector<cv::cuda::GpuMat> lmaskRingL_, lmaskRingR_;
    int lmaskSlots_ = 2;
    std::vector<cudaEvent_t> evLmaskRing_;    // [S]：1b flat 入队后按槽 record（1a D2D 前 wait）
    std::atomic<int> lmaskFed_{-1};           // evLmaskRing_ 提交序（1b 侧）
    // 【调优②】宿主侧等待统一 cv 阻塞（原自旋 yield×THREAD_PRIORITY_HIGHEST 双线程
    // 互踢＝WDDM 乒乓病理疑点——nsys/实测在案）；三个提交序计数器共用一把锁
    mutable std::mutex gateMtx_;
    std::condition_variable gateCv_;
    // 【P4.5】RLE 环形槽（设备＋锁页成环；1b 拷出计数闸门防 1a 越代覆写）
    std::vector<cv::cuda::GpuMat> dRleRingL_, dRleRingR_;
    int* rleCntRing_ = nullptr;               // 锁页 slots×2
    int* rleRunsRing_ = nullptr;              // 锁页 slots×6cap
    std::vector<cudaEvent_t> evRleRing_;      // per-slot（1a 记录→1b 事件同步）
    std::atomic<int> rleCopied_{-1};          // 1b 已拷出的最大帧号（1a 入队闸门）
    int rleSlots_ = 4;
    int rleWin_ = 4096;                       // 【E7】RLE 按实拷窗口（钳 [64, rleCap_]）
    long rleNCnt_ = 0;                        // 【E7】行程数分布统计（落档 fc_summary）
    long long rleNSum_ = 0;
    int rleNMax_ = 0;
    long rleTailCnt_ = 0;                     // 超窗补拷次数（健康度应为 0）
    std::vector<int> rleNAll_;                // 每帧 nL+nR（中位数落档用）
    std::vector<cudaEvent_t> evUpl_, evSlot_; // [graySlots]
    int fedIdx_;                              // 提交序：flat/steger 已入队的最大帧号
    std::mutex fedMtx_;
    std::condition_variable fedCv_;

    // —— E 锁页池 ——
    std::vector<float*> eBlocks_;
    int eCapPts_;
    std::mutex eMtx_;
    std::condition_variable eCv_;
    std::vector<int> eFree_;
    std::atomic<int> eTruncPts_{0};
    cv::cuda::GpuMat d_pts_;                  // 融合上传设备中转（1×eCapPts CV_32FC3）

    // —— fivechain 专属算子（温表/参数与 harness v3 同源）——
    std::unique_ptr<calib::LaserMatchScanSplitGpuV3> splitGpu_;
    std::unique_ptr<calib::LaserMatchScanSplitCpuV3> splitCpu_;
    calib::LaserReconstructParams reconP_;
    std::unique_ptr<calib::LaserReconstructCpu> reconCpu_;

    // —— 配对（CUDA链2；界=在飞上限）——
    // 【零分配】定容环替代 std::map（容量=maxInflight+8；读帧闸保证在飞≤界）
    std::vector<FcJob*> pendC_, pendE_;
    int pendCap_;
    // —— CPU链3 急取重排（同款定容环）——
    std::vector<FcJob*> reorder_;
    int nextFuse_;
    int nextRegIdx_;                          // CPU链3 急取重排游标

    // —— 对账计数 ——
    std::array<std::atomic<int>, 8> chCount_{};
    std::atomic<int> tombCount_{0};

    // —— 【指标】逐帧记录 ——
    std::unique_ptr<FcFrameRec[]> recs_;      // [totalCycles_]（totalCycles>0 时）
    bool metricsOn_ = true;
    bool busyProbe_ = false;                  // 【E2】GPU busy 事件对选通（JMW_FCBUSY=1 开；
                                              // 默认关——节拍线程每帧省 30-80μs 提交税）
    int dbgInterval_ = 0;                     // JMW_FCDBG 实时行（0=off）
    long long lastCollectNs_ = 0;
    // GPU busy 近似：timing 事件对（1a/1b 各 8 深度环；延迟 8 帧查询 elapsed）
    std::vector<cudaEvent_t> evBusy1aA_, evBusy1aB_, evBusy1bA_, evBusy1bB_;
    int busyRing_ = 8;
    void recordBusySpan_(int idx, bool oneA); // 查询 8 帧前事件对 elapsed → recs_
};

} // namespace fc
} // namespace scr
