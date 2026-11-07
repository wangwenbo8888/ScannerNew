// pipeline.cpp — 跨帧流水线执行器实现（架构见 pipeline.h 头注释）
#include "pipeline.h"

#include "ccl_cpu.h"      // runCclCpu/FromRuns（JMW_RLEAB 对拍诊断）
#include "gpu_accum.h"   // gpuMaskRleBoth（GPU-RLE 掩膜压缩）

#include <chrono>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>
#include <intrin.h>   // _tzcnt_u64
#include <vector>
#endif

namespace scr {

// ============================================================================
// 线程绑核（实验性，默认关）
// 实测裁定（2026-10-18 A/B）：拓扑感知绑核（跳过中断核、每线程独占物理核）
// 对 GPU 线程步骤零收益（steger/v3 ±0.1ms）——争用拖累的主因是全核负载掉
// 基频＋内存带宽共享，非超线程兄弟碰撞；标记步骤小赚 ~0.9ms 但非瓶颈。
// 默认关闭；JMW_AFFINITY=1 显式开启（拆分收尾线程后可复试）。
// ============================================================================
void pipePinThread(int coreIdx, const char* who) {
#ifdef _WIN32
    const char* affOn = getenv("JMW_AFFINITY");
    if (!(affOn && affOn[0] == '1')) return;         // 默认关；JMW_AFFINITY=1 显式开
    // 枚举物理核（每核一个 SYSTEM_LOGICAL_PROCESSOR_INFORMATION 条目）
    std::vector<DWORD_PTR> coreFirstLP;              // 各物理核的首个逻辑核位
    {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(64);
        DWORD cb = static_cast<DWORD>(buf.size() * sizeof(buf[0]));
        while (GetLogicalProcessorInformation(buf.data(), &cb) == FALSE) {
            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return;
            buf.resize(cb / sizeof(buf[0]) + 1);
            cb = static_cast<DWORD>(buf.size() * sizeof(buf[0]));
        }
        const size_t n = cb / sizeof(buf[0]);
        for (size_t i = 0; i < n; ++i)
            if (buf[i].Relationship == RelationProcessorCore) {
                DWORD_PTR m = buf[i].ProcessorMask;
                DWORD_PTR first = m & (~(m - 1));    // 最低位＝该核首个 LP
                coreFirstLP.push_back(first);
            }
    }
    if (coreFirstLP.size() < 2) return;              // 拓扑拿不到→不绑（保底）
    // 跳过 0 号物理核（中断核）；coreIdx=0 → 物理核 #1，以此类推
    const size_t target = static_cast<size_t>(coreIdx) + 1;
    if (target >= coreFirstLP.size()) return;        // 核不够→不绑
    const DWORD_PTR mask = coreFirstLP[target];
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        fprintf(stderr, "[affinity] %s 绑核失败: %lu\n", who, GetLastError());
    } else {
        // 报告物理核序号＋LP 位（诊断可读）
        fprintf(stderr, "[affinity] %s -> 物理核#%zu (LP%zu)\n", who, target,
                static_cast<size_t>(_tzcnt_u64(mask)));
    }
#else
    (void)coreIdx; (void)who;
#endif
}

} // namespace scr

#include <opencv2/core/cuda_stream_accessor.hpp>

namespace scr {
namespace pipeprof {
static std::chrono::steady_clock::time_point t0() {
    static const std::chrono::steady_clock::time_point base =
        std::chrono::steady_clock::now();
    return base;
}
static double ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0()).count();
}
// 【诊断】JMW_PIPEDBG=1：gpu/laser lane 阶段打印（stderr 立即刷）
static void dbg(const char* tag, int idx, const char* stage) {
    if (!getenv("JMW_PIPEDBG")) return;
    fprintf(stderr, "[pdbg] %s f%d %s t=%.1f\n", tag, idx, stage, ms());
    fflush(stderr);
}
} // namespace pipeprof
} // namespace scr

namespace scr {

PipelineExecutor::PipelineExecutor(Harness& h, int lanes, int firstIdx, int totalCycles)
    : h_(h), nLanes_(lanes), firstIdx_(firstIdx), totalCycles_(totalCycles),
      nextRegIdx_(firstIdx), nextSubmit_(firstIdx) {
    if (nLanes_ < 1) nLanes_ = 1;
    // 【利用率专项】激光链开关（与原 gpuLoop 内联判定同式）：诊断模式直通闸门
    laserOn_ = !h_.cfg_.runtime.markerOnly && getenv("JMW_NOLASER") == nullptr;
    // 收尾 lane 数（JMW_MLASER；默认 3——正式协议实测热态 91-92 FPS 全程 vs
    // 双 lane 84-88；首跑吞各 lane 冷启动 ~64 FPS，第二跑起稳态）
    if (const char* e = getenv("JMW_MLASER")) {
        kLaserN_ = std::max(1, std::min(4, atoi(e)));
    }
    // 激光收尾 lane 双实例（09 算子非线程安全；组装配失败回落单 lane 共享串行组）
    if (laserOn_ && kLaserN_ > 1) {
        h_.laserLanes_.clear();
        h_.laserLanes_.resize(static_cast<size_t>(kLaserN_));
        bool ok = true;
        std::string lerr;
        for (int i = 0; i < kLaserN_; ++i) {
            h_.laserLanes_[static_cast<size_t>(i)] = h_.makeLaserOps(i, lerr);
            if (!lerr.empty()) { ok = false; break; }
        }
        if (!ok) {
            std::cerr << "[pipeline] 激光收尾 lane 组装配失败（" << lerr
                      << "）——回落单 lane（共享串行算子组）" << std::endl;
            h_.laserLanes_.clear();
            kLaserN_ = 1;
        }
    }
    // 闸门初值对齐首帧前驱（串行头帧不经流水线）；lane 解耦资源
    laserFedLane_.assign(static_cast<size_t>(kLaserN_), firstIdx - 1);
    laserQs_.assign(static_cast<size_t>(kLaserN_), {});
    lmaskL_.assign(static_cast<size_t>(kLaserN_), cv::cuda::GpuMat());
    lmaskR_.assign(static_cast<size_t>(kLaserN_), cv::cuda::GpuMat());
    laserStreams_.resize(static_cast<size_t>(kLaserN_));
    evGate_.assign(static_cast<size_t>(kLaserN_), nullptr);
    evSteg_.assign(static_cast<size_t>(kLaserN_), nullptr);
    evCloud_.assign(static_cast<size_t>(kLaserN_), nullptr);
    for (int i = 0; i < kLaserN_; ++i) {
        cudaEventCreateWithFlags(&evGate_[static_cast<size_t>(i)], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evSteg_[static_cast<size_t>(i)], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evCloud_[static_cast<size_t>(i)], cudaEventDisableTiming);
    }
    h_.ensureGrayLanes(kLaserN_);   // 每 lane 独立 d_gray（跨 lane 解耦）
    // 标记点链算子组：逐 lane 独立实例（09 CPU 算子非线程安全）
    h_.markerLanes_.clear();
    h_.markerLanes_.resize(static_cast<size_t>(nLanes_));
    for (int i = 0; i < nLanes_; ++i) h_.markerLanes_[static_cast<size_t>(i)] = h_.makeMarkerOps();
    gpuTh_ = std::thread([this] { gpuLoop(); });
    laserThs_.resize(static_cast<size_t>(kLaserN_));
    for (int i = 0; i < kLaserN_; ++i)
        laserThs_[static_cast<size_t>(i)] = std::thread([this, i] { laserLoop(i); });
    markerThs_.resize(static_cast<size_t>(nLanes_));
    for (int i = 0; i < nLanes_; ++i)
        markerThs_[static_cast<size_t>(i)] = std::thread([this, i] { markerLoop(i); });
    regTh_ = std::thread([this] { regLoop(); });
    std::cout << "[pipeline] 启动：GPU lane ×1 ＋ 激光收尾 lane ×" << kLaserN_
              << "（深度 " << kLaserQ << "）＋ 标记点 lane ×" << nLanes_
              << " ＋ 配准线程 ×1（在飞上限 " << kMaxInflight << "）" << std::endl;
}

PipelineExecutor::~PipelineExecutor() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    gpuCv_.notify_all();
    laserCv_.notify_all();
    fedCv_.notify_all();
    markerCv_.notify_all();
    regCv_.notify_all();
    runsCv_.notify_all();
    if (gpuTh_.joinable()) gpuTh_.join();
    for (auto& t : laserThs_) if (t.joinable()) t.join();
    for (auto& t : markerThs_) if (t.joinable()) t.join();
    if (regTh_.joinable()) regTh_.join();
    for (auto& e : evGate_) if (e) { cudaEventDestroy(e); e = nullptr; }
    for (auto& e : evSteg_) if (e) { cudaEventDestroy(e); e = nullptr; }
    for (auto& e : evCloud_) if (e) { cudaEventDestroy(e); e = nullptr; }
    if (rleCounts_) { cudaFreeHost(rleCounts_); rleCounts_ = nullptr; }
    if (rleRuns_) { cudaFreeHost(rleRuns_); rleRuns_ = nullptr; }
}

void PipelineExecutor::finishJob(Job* j) {
    if (j->gpuDone.load() && j->regDone.load()) {
        j->done.store(true);
        doneCv_.notify_all();
    }
}

// ============================================================================
// GPU lane：读帧→上传闸门→分离→掩膜快照→派发标记链→RLE 回填→激光链移交收尾线程
//   （【利用率专项】激光链 steger→…→v3→recon→统计→点云快照整段移专职线程——
//    GPU lane 提交完分离即跑下一帧，v3 ~5ms 宿主段期间 GPU 吃下一帧队列）
// ============================================================================
void PipelineExecutor::gpuLoop() {
    pipePinThread(0, "gpu-lane");   // 【150FPS】GPU lane 独占物理核 #1（跳过中断核 0）
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            gpuCv_.wait(lk, [&] { return stop_ || !gpuQ_.empty(); });
            if (stop_ && gpuQ_.empty()) return;
            j = gpuQ_.front();
            gpuQ_.pop();
        }
        const auto& cyc = h_.src_->cycles[static_cast<size_t>(j->idx)];
        cv::Mat grayL, grayR;
        std::string rerr;
        if (!h_.src_->readFrame(cyc.laserIdx, true, grayL, rerr)
            || !h_.src_->readFrame(cyc.laserIdx, false, grayR, rerr)) {
            // 读帧失败＝串行同款语义：整帧失败交还。经 regReady_ 走保序空转
            // （skipReg：reg 线程按序跳过——帧序链不断）
            j->err = rerr;
            j->skipReg = true;
            releaseLaserGate(j->idx);   // 【利用率专项】本帧零上传零消费——放行下一帧闸门
            j->markerDone.store(true);
            j->gpuDone.store(true);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                regReady_[j->idx] = j;
                finishJob(j);   // gpuDone 已置——若 regDone 后置则 reg 线程完单
            }
            regCv_.notify_one();
            continue;
        }
        j->prod.frameGrayL = grayL;
        j->prod.frameGrayR = grayR;
        bool dispatched = false;
        bool handed = false;   // 【利用率专项】激光链已移交收尾线程
        try {
            pipeprof::dbg("gpu", j->idx, "pre-gate");
            // 0) 上传闸门（双保险）：laserFed＝提交序（本 lane 前帧 steger 已入
            //    队——保证 evSteg 已记录到该帧）；evSteg＝执行序（G 流 upload
            //    等该 steger 真正干完——跨流无 FIFO，必须事件显式保序）
            waitLaserFed(j->idx);
            const int lane = j->idx % kLaserN_;
            cudaStream_t gcs = cv::cuda::StreamAccessor::getStream(h_.stream_);
            if (laserOn_)
                cudaStreamWaitEvent(gcs, evSteg_[static_cast<size_t>(lane)], 0);
            pipeprof::dbg("gpu", j->idx, "gate-passed");
            // 1) 分离（上传至本 lane d_gray＋sep；evSepDone_ 亦记录——串行件复用无害）
            h_.runSeparation(grayL, grayR, j->prod, j->record, laserOn_ ? lane : 0);
            pipeprof::dbg("gpu", j->idx, "sep-done");
            // 1b) 激光掩膜快照（【利用率专项·lane 解耦】每 lane 单槽：steger 消费
            //     快照视图——sep 内部缓冲此后仅 GPU lane 生命周期内触碰，他 lane
            //     的收尾不再约束本帧分离）。快照完成后记录 evGate：lane 流的
            //     steger 等它（跨流读 d_gray+lmask 的执行序保证）
            if (laserOn_ && j->prod.sepL.d_laserMask && j->prod.sepR.d_laserMask
                && !j->prod.sepL.d_laserMask->empty()
                && !j->prod.sepR.d_laserMask->empty()) {
                auto& mL = lmaskL_[static_cast<size_t>(lane)];
                auto& mR = lmaskR_[static_cast<size_t>(lane)];
                if (mL.empty()) mL.create(1400, 1688, CV_8UC1);
                if (mR.empty()) mR.create(1400, 1688, CV_8UC1);
                j->prod.sepL.d_laserMask->copyTo(mL, h_.stream_);
                j->prod.sepR.d_laserMask->copyTo(mR, h_.stream_);
                j->prod.sepL.d_laserMask = ringView(mL);
                j->prod.sepR.d_laserMask = ringView(mR);
                cudaEventRecord(evGate_[static_cast<size_t>(lane)], gcs);
            }
            // 2) 标记掩膜快照（D2D；分离算子内部缓冲下一帧覆写——跨帧消费必须
            //    物化）＋ GPU-RLE（【150FPS】替代 lane 侧 2×2.36MB 下载：GPU 行程
            //    编码 KB 级回传；与激光链同流排队，链尾一次同步全就绪）
            cudaStream_t cs = cv::cuda::StreamAccessor::getStream(h_.stream_);
            bool rleOk = false;
            if (j->prod.sepL.d_markingPointMask && j->prod.sepR.d_markingPointMask) {
                // 环形快照（【150FPS】免每帧 cudaMalloc×3——nsys 实锤 ~5ms/帧空闲）
                auto& msL = maskSnap_[maskSnapIdx_];
                auto& msR = maskSnap_[(maskSnapIdx_ + 1) % 4];
                maskSnapIdx_ = (maskSnapIdx_ + 2) % 4;
                if (msL.empty()) msL.create(1400, 1688, CV_8UC1);
                if (msR.empty()) msR.create(1400, 1688, CV_8UC1);
                j->prod.sepL.d_markingPointMask->copyTo(msL, h_.stream_);
                j->prod.sepR.d_markingPointMask->copyTo(msR, h_.stream_);
                j->snapMaskL = ringView(msL);
                j->snapMaskR = ringView(msR);
                if (getenv("JMW_NORLE") != nullptr) {
                    // 【诊断】像素回退对照路径（跳过 GPU-RLE）
                    cudaEventCreateWithFlags(&j->evMaskReady, cudaEventDisableTiming);
                    cudaEventRecord(j->evMaskReady, cs);
                    j->runsPixelFallback = true;
                    j->runsReady.store(true);
                    rleOk = true;
                } else {
                    if (d_rleL_.empty())
                        d_rleL_.create(1, 3 * kRleCap, CV_32SC1);
                    if (d_rleR_.empty())
                        d_rleR_.create(1, 3 * kRleCap, CV_32SC1);
                    if (!rleCounts_) {
                        cudaMallocHost(&rleCounts_, 2 * sizeof(int));
                        cudaMallocHost(&rleRuns_, sizeof(int) * 6 * kRleCap);
                    }
                    rleOk = gpuMaskRleBoth(*j->snapMaskL, *j->snapMaskR,
                                           d_rleL_, d_rleR_, kRleCap, rleCounts_,
                                           rleRuns_, h_.stream_);
                    j->rleW = j->snapMaskL->cols;
                    j->rleH = j->snapMaskL->rows;
                }
            }
            if (!rleOk) {
                // 分离失败（掩膜空）→ 空行程走降级（ccl 步记失败，下游自然跳过）
                j->runsL = std::make_shared<std::vector<int>>();
                j->runsR = std::make_shared<std::vector<int>>();
                j->runsNL = j->runsNR = 0;
                j->runsReady.store(true);
            }
            // 3) 派发标记点链（掩膜就绪即发——不等激光链；工作窃取平衡 lane 负载）
            // 【诊断】JMW_NOMARKER=1：跳过标记点链（量纯 GPU lane 周期）
            if (getenv("JMW_NOMARKER") == nullptr) {
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    markerQ_.push(j);
                }
                markerCv_.notify_one();
            } else {
                j->markerDone.store(true);
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    regReady_[j->idx] = j;
                }
                regCv_.notify_one();
            }
            dispatched = true;
            j->tDispatched = pipeprof::ms();
            // 4) RLE 回填（【利用率专项】提前至激光链移交前：本处同步只排空
            //    分离+RLE＋收尾线程在飞的 v3 尾巴——标记 lane 由此提前放行，
            //    不再等本帧激光链；溢出/空掩膜→像素回退（罕见路径）
            if (!j->runsReady.load()) {
                cudaStreamSynchronize(cs);
                const int nL = rleCounts_[0], nR = rleCounts_[1];
                if (nL <= kRleCap && nR <= kRleCap) {
                    j->runsL = std::make_shared<std::vector<int>>(
                        rleRuns_, rleRuns_ + 3 * (size_t)nL);
                    j->runsR = std::make_shared<std::vector<int>>(
                        rleRuns_ + 3 * kRleCap,
                        rleRuns_ + 3 * kRleCap + 3 * (size_t)nR);
                    j->runsNL = nL;
                    j->runsNR = nR;
                    // 【诊断】JMW_RLEAB=1：GPU-RLE vs 像素 RLE 逐项对拍（前 3 帧）
                    if (getenv("JMW_RLEAB") && j->idx < firstIdx_ + 3
                        && j->snapMaskL && j->snapMaskR) {
                        cv::Mat hL, hR;
                        j->snapMaskL->download(hL, h_.stream_);
                        j->snapMaskR->download(hR, h_.stream_);
                        h_.stream_.waitForCompletion();
                        const auto pxL = runCclCpu(hL, h_.cclMinArea_, h_.cclMaxArea_,
                                                   h_.cclTopX_);
                        const auto pxR = runCclCpu(hR, h_.cclMinArea_, h_.cclMaxArea_,
                                                   h_.cclTopX_);
                        const auto rlL = runCclCpuFromRuns(
                            *j->runsL, j->runsNL, j->rleW, j->rleH, h_.cclMinArea_,
                            h_.cclMaxArea_, h_.cclTopX_);
                        const auto rlR = runCclCpuFromRuns(
                            *j->runsR, j->runsNR, j->rleW, j->rleH, h_.cclMinArea_,
                            h_.cclMaxArea_, h_.cclTopX_);
                        auto compEq = [](const calib::RegionAnalysisResult& a,
                                         const calib::RegionAnalysisResult& b) {
                            if (a.components.size() != b.components.size()) return false;
                            for (size_t i = 0; i < a.components.size(); ++i) {
                                const auto& x = a.components[i];
                                const auto& y = b.components[i];
                                if (x.boundingBoxX != y.boundingBoxX
                                    || x.boundingBoxY != y.boundingBoxY
                                    || x.boundingBoxWidth != y.boundingBoxWidth
                                    || x.boundingBoxHeight != y.boundingBoxHeight
                                    || x.area != y.area)
                                    return false;
                            }
                            return true;
                        };
                        fprintf(stderr,
                                "[rleAB] f%d nRuns L=%d R=%d | 组件 px(L=%d R=%d) "
                                "rle(L=%d R=%d) | 逐项一致 L=%d R=%d\n",
                                j->idx, j->runsNL, j->runsNR,
                                (int)pxL.components.size(), (int)pxR.components.size(),
                                (int)rlL.components.size(), (int)rlR.components.size(),
                                (int)compEq(pxL, rlL), (int)compEq(pxR, rlR));
                    }
                } else {
                    j->runsPixelFallback = true;
                    j->snapMaskL->download(j->pixL, h_.stream_);
                    j->snapMaskR->download(j->pixR, h_.stream_);
                    cudaStreamSynchronize(cs);
                }
                j->runsReady.store(true);
                runsCv_.notify_all();
            }
            // 5) 激光链移交收尾线程（【利用率专项】steger→undist→interp→v3→
            //    recon→统计→点云快照整段）：GPU lane 即刻跑下一帧——v3 两段
            //    ~5ms 宿主期间 GPU 吃下一帧分离/steger 队列（run-ahead 喂 GPU）；
            //    深度 kLaserQ 背压；steger 入队后收尾线程放行 laserFed 闸门
            //    （d_gray／分离激光掩膜覆写安全）。诊断模式（JMW_NOLASER/
            //    markerOnly）无激光消费——不移交，按原语义本线程完单
            if (laserOn_) {
                pipeprof::dbg("gpu", j->idx, "pre-push");
                handed = pushLaser(j);
                pipeprof::dbg("gpu", j->idx, "pushed");
            }
        } catch (const std::exception& e) {
            h_.fail("pipeline_gpu", j->prod, e.what());
        } catch (...) {
            h_.fail("pipeline_gpu", j->prod, "未知异常");
        }
        if (!dispatched) {
            // 异常早退且未派发——补派发保序（标记链跳过，配准按序降级）
            j->markerDone.store(true);
            if (!j->runsReady.load()) j->runsReady.store(true);
            releaseLaserGate(j->idx);   // 激光未移交（无消费者）——放行闸门
            {
                std::lock_guard<std::mutex> lk(mtx_);
                regReady_[j->idx] = j;
            }
            regCv_.notify_one();
        } else if (!j->runsReady.load()) {
            // 异常晚于派发——防标记 lane 永等（空 runs 走降级）
            j->runsReady.store(true);
            runsCv_.notify_all();
        }
        if (!handed) {
            // 未移交（诊断模式/推送前异常/停机丢弃）：本线程完单——gpuDone 在此
            // 置；移交成功时由收尾线程置（点云快照之后）
            releaseLaserGate(j->idx);
            j->gpuDone.store(true);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                finishJob(j);
            }
        }
    }
}

// ============================================================================
// 激光收尾 lane ×kLaserN_（【利用率专项】）：steger→undist→interp→v3→recon→
// 行残差统计→重建点云快照——整段搬离 GPU lane。正确性四支柱：
//   ① 每 lane 独立 LaserOps（09 算子非线程安全；帧号取模确定性映射 lane）；
//   ② 与 GPU lane 上传/分离共用 stream_ 单流 FIFO——steger(N) 先于 upload(N+1)
//      执行由 laserFed 闸门保证（各 lane 就其 parity 帧号就位后才放行上传）；
//   ③ v3 结果 ROI 视图有效期至同实例下一次 Execute——同 lane 帧序串行保证；
//   ④ 点云快照环账本 cloudMtx_（多 lane 共享 8 槽；在飞上限保证槽复用安全）
// ============================================================================
void PipelineExecutor::laserLoop(int laneIdx) {
    Harness::LaserOps& L = laneLaserOps(laneIdx);
    auto& myQ = laserQs_[static_cast<size_t>(laneIdx)];
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            laserCv_.wait(lk, [&] { return stop_ || !myQ.empty(); });
            if (stop_ && myQ.empty()) return;
            j = myQ.front();
            myQ.pop_front();
        }
        laserCv_.notify_all();   // 腾槽——混合等待者（生产者/各 lane 消费者谓词
                                 // 不同），notify_one 可能唤醒错对象＝丢唤醒死锁
        const auto& cyc = h_.src_->cycles[static_cast<size_t>(j->idx)];
        try {
            pipeprof::dbg("lsr", j->idx, "popped");
            // 本 lane 流等待分离链完成（d_gray/lmask 已物化——跨流执行序）
            auto& ls = laserStreams_[static_cast<size_t>(laneIdx)];
            cudaStream_t lcs = cv::cuda::StreamAccessor::getStream(ls);
            cudaStreamWaitEvent(lcs, evGate_[static_cast<size_t>(laneIdx)], 0);
            h_.runLaserSteger(L, j->prod, j->record, cyc.tubeOdd, laneIdx, ls);
            pipeprof::dbg("lsr", j->idx, "steger-done");
            bumpLaserFed(laneIdx, j->idx);   // steger 已入队——放行上传（提交序）
            cudaEventRecord(evSteg_[static_cast<size_t>(laneIdx)], lcs);
            pipeprof::dbg("lsr", j->idx, "fed");
            h_.runLaserTail(L, j->prod, j->record, cyc.tubeOdd, ls);
            pipeprof::dbg("lsr", j->idx, "tail-done");
            // 重建点云快照（accumulateGlobalLaser 由主线程在帧完成后调用，
            // 消费 d_points3d——reconOp 内部缓冲会被后续帧覆写）。环形 16 槽
            // grow-only 复用（免每帧 cudaMalloc）；环账本加锁（多收尾 lane 共享）；
            // D2D 排本 lane 流（读 recon 输出同流保序）＋ evCloud 事件供主线程
            // G 流 accumulate 等待（跨流读槽内容的执行序保证）
            if (j->prod.recon.d_points3d && !j->prod.recon.d_points3d->empty()) {
                {
                    std::lock_guard<std::mutex> lk(cloudMtx_);
                    auto& cs_ = cloudSnap_[cloudSnapIdx_];
                    cloudSnapIdx_ = (cloudSnapIdx_ + 1) % 16;
                    const int need = j->prod.recon.d_points3d->cols;
                    if (cs_.cols < need)
                        cs_.create(1, std::max(need, cs_.cols > 0 ? cs_.cols * 2 : need),
                                   CV_32FC3);
                    const cv::cuda::GpuMat srcRoi =
                        j->prod.recon.d_points3d->colRange(0, need);
                    srcRoi.copyTo(cs_(cv::Rect(0, 0, need, 1)), ls);
                    // 紧致视图（宿主头部 ~ns 级，设备内存走环形槽——无每帧 cudaMalloc）
                    j->prod.recon.d_points3d =
                        std::make_shared<cv::cuda::GpuMat>(cs_(cv::Rect(0, 0, need, 1)));
                }
                cudaEventRecord(evCloud_[static_cast<size_t>(laneIdx)], lcs);
            }
        } catch (const std::exception& e) {
            h_.fail("pipeline_laser", j->prod, e.what());
        } catch (...) {
            h_.fail("pipeline_laser", j->prod, "未知异常");
        }
        bumpLaserFed(laneIdx, j->idx);   // 异常路径保底放行（防上传闸门死等）
        j->gpuDone.store(true);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            finishJob(j);
        }
    }
}

// —— laserFed 闸门四件套（mtx_ 保护；fedCv_ 唤醒）——

Harness::LaserOps& PipelineExecutor::laneLaserOps(int i) {
    return h_.laserLanes_.empty() ? h_.laserOps_ : h_.laserLanes_[static_cast<size_t>(i)];
}

void PipelineExecutor::waitLaserFed(int idx) {
    if (!laserOn_) return;
    // 【lane 解耦】d_gray/激光掩膜均每 lane 独立——只需本 lane 的前帧 steger
    // 已入队（fed ≥ idx-kLaserN）：他 lane 慢不再拖住本帧上传（原全 lane 就位
    // 闸门使双 lane 全流同步互锁成锁步，实测吃掉全部并行收益）
    const int lane = idx % kLaserN_;
    const int need = idx - kLaserN_;
    std::unique_lock<std::mutex> lk(mtx_);
    fedCv_.wait(lk, [&] {
        return stop_ || need < firstIdx_
               || laserFedLane_[static_cast<size_t>(lane)] >= need;
    });
}

void PipelineExecutor::bumpLaserFed(int laneIdx, int idx) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (idx > laserFedLane_[static_cast<size_t>(laneIdx)])
            laserFedLane_[static_cast<size_t>(laneIdx)] = idx;
    }
    fedCv_.notify_all();
}

void PipelineExecutor::releaseLaserGate(int idx) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = 0; i < kLaserN_; ++i)
            if (idx > laserFedLane_[static_cast<size_t>(i)])
                laserFedLane_[static_cast<size_t>(i)] = idx;
    }
    fedCv_.notify_all();
}

bool PipelineExecutor::pushLaser(Job* j) {
    const int lane = j->idx % kLaserN_;
    {
        std::unique_lock<std::mutex> lk(mtx_);
        laserCv_.wait(lk, [&] {
            return stop_ || laserQs_[static_cast<size_t>(lane)].size() < kLaserQ;
        });
        if (stop_) return false;   // 停机丢弃（gpuLoop 按「未移交」完单）
        laserQs_[static_cast<size_t>(lane)].push_back(j);
    }
    laserCv_.notify_all();   // 唤醒对应收尾 lane（混合等待者必须 notify_all——
                             // notify_one 可能唤醒等「不满」的生产者＝丢唤醒死锁）
    return true;
}

// ============================================================================
// 标记点 lane：下载快照掩膜（等就绪事件）→ CPU 全链（ccl→…→conic）
// ============================================================================
void PipelineExecutor::markerLoop(int laneIdx) {
    pipePinThread(1 + laneIdx, "marker-lane");   // 【150FPS】标记 lane 各占核 1..n
    // 本 lane 专用下载流（线程内创建；仅像素回退路径使用）
    cv::cuda::Stream dlStream;
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            markerCv_.wait(lk, [&] { return stop_ || !markerQ_.empty(); });
            if (stop_ && markerQ_.empty()) return;
            j = markerQ_.front();
            markerQ_.pop();
        }
        j->tMkStart = pipeprof::ms();
        // 等 GPU-RLE 回填（GPU lane 链尾同步后；此处起标记点 lane 零 CUDA）
        {
            std::unique_lock<std::mutex> lk(mtx_);
            runsCv_.wait(lk, [&] { return j->runsReady.load() || stop_; });
        }
        try {
            if (j->runsPixelFallback) {
                // 溢出回退：像素路径（快照就绪事件等 D2D 完成）
                h_.runMarkerChain(j->prod, j->record,
                                  h_.markerLanes_[static_cast<size_t>(laneIdx)],
                                  j->snapMaskL, j->snapMaskR, j->evMaskReady, dlStream);
            } else if (j->runsL) {
                h_.runMarkerChainRuns(j->prod, j->record,
                                      h_.markerLanes_[static_cast<size_t>(laneIdx)],
                                      *j->runsL, j->runsNL, *j->runsR, j->runsNR,
                                      j->rleW, j->rleH);
            } else {
                // 分离失败（掩膜空）：与串行同款降级——ccl 步记失败，下游自然跳过
                h_.fail("ccl", j->prod, "标记点掩膜为空（分离失败）");
            }
        } catch (const std::exception& e) {
            h_.fail("pipeline_marker", j->prod, e.what());
        } catch (...) {
            h_.fail("pipeline_marker", j->prod, "未知异常");
        }
        j->tMkDone = pipeprof::ms();
        j->markerDone.store(true);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!j->skipReg) regReady_[j->idx] = j;
            finishJob(j);
        }
        regCv_.notify_one();
    }
}

// ============================================================================
// 配准线程：严格按帧序（prevState 跨帧依赖链——乱序即结果错序）
// ============================================================================
void PipelineExecutor::regLoop() {
    pipePinThread(1 + nLanes_, "reg");   // 【150FPS】配准线程占标记 lane 后一核
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            regCv_.wait(lk, [&] {
                if (stop_) return true;
                return regReady_.count(nextRegIdx_) > 0;
            });
            if (stop_) return;   // 析构仅在全部帧收割后发生——残余必空
            auto it = regReady_.find(nextRegIdx_);
            if (it == regReady_.end()) continue;
            j = it->second;
            regReady_.erase(it);
            ++nextRegIdx_;
        }
        if (!j->skipReg) {
            try {
                h_.runRegistration(j->prod, j->record);
                if (j->record)
                    h_.trajT_.push_back({j->prod.T[0], j->prod.T[1], j->prod.T[2]});
                ++h_.cycleCount_;
            } catch (const std::exception& e) {
                h_.fail("pipeline_reg", j->prod, e.what());
            } catch (...) {
                h_.fail("pipeline_reg", j->prod, "未知异常");
            }
        }
        j->tRegDone = pipeprof::ms();
        j->regDone.store(true);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            finishJob(j);
        }
    }
}

// ============================================================================
// submitAhead / collect — 主线程入口：提前铺帧（在飞窗口）＋按序收割
// （旧 submitAndWait 同步等待＝在飞恒 1、流水线失效——实测破案后重构）
// ============================================================================
void PipelineExecutor::submitAhead(int curIdx) {
    const int target = curIdx + kMaxInflight;
    for (;;) {
        Job* j = nullptr;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (nextSubmit_ > target || nextSubmit_ >= totalCycles_
                || inflight_ >= kMaxInflight)
                return;
            auto job = std::make_unique<Job>();
            job->idx = nextSubmit_;
            job->record = nextSubmit_ >= h_.cfg_.runtime.warmupCycles;
            job->tSubmit = pipeprof::ms();
            j = job.get();
            ++inflight_;
            ++nextSubmit_;
            live_.push_back(std::move(job));
            gpuQ_.push(j);
        }
        gpuCv_.notify_one();
    }
}

bool PipelineExecutor::collect(int curIdx, CycleProducts& out, bool& recordUsed,
                               std::string& err) {
    Job* j = nullptr;
    {
        std::unique_lock<std::mutex> lk(mtx_);
        for (auto& jp : live_)
            if (jp->idx == curIdx) { j = jp.get(); break; }
        if (j == nullptr) {                  // 已收割（重复 collect）或越界
            err = "collect: 帧不在在飞表: " + std::to_string(curIdx);
            return false;
        }
        doneCv_.wait(lk, [&] { return j->done.load(); });
        err = j->err;
        recordUsed = j->record;
        if (getenv("JMW_PIPELOG"))
            fprintf(stderr, "[pipe] f%03d submit=%.1f disp=%.1f mkS=%.1f mkD=%.1f "
                    "regD=%.1f harv=%.1f\n", curIdx, j->tSubmit, j->tDispatched,
                    j->tMkStart, j->tMkDone, j->tRegDone, pipeprof::ms());
        out = std::move(j->prod);            // CycleProducts 移动交还
        if (j->evMaskReady) cudaEventDestroy(j->evMaskReady);   // job 存活期最后触碰
        // 【多流解耦】主线程随后在 G 流排队 accumulate（读点云槽）——先等本帧
        // lane 流的 D2D 完成（跨流执行序；事件未记录过＝立即返回，无害）
        if (laserOn_ && kLaserN_ > 0) {
            cudaStreamWaitEvent(cv::cuda::StreamAccessor::getStream(h_.stream_),
                                evCloud_[static_cast<size_t>(j->idx % kLaserN_)], 0);
        }
        for (auto it = live_.begin(); it != live_.end(); ++it)
            if (it->get() == j) { live_.erase(it); break; }
        --inflight_;
    }
    slotCv_.notify_one();
    return err.empty();                      // 读帧失败等以 err 交还
}

} // namespace scr

