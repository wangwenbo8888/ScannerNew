#pragma once
// ============================================================================
// pipeline.h — 【150FPS·方案 D】跨帧流水线执行器
// ============================================================================
// 架构（参考产线 07 SchedulerRuntime：多 lane 各处理一帧，帧间并行）：
//
//   main ──submit──▶ GPU lane（×1 线程：上传＋分离＋掩膜快照＋RLE 回填）
//                      │ 分离后 D2D 快照标记掩膜＋就绪事件 → 立即派发
//                      ├──────────▶ 标记点 lane ×3（CPU，工作窃取队列）
//                      │                ccl_cpu→split→zern→merge→undist→
//                      │                ellipse→match→conic（逐 lane 独立算子组）
//                      ├─push(深度2)─▶ 激光收尾线程 ×1（【利用率专项】steger→
//                      │                undist→interp→v3→recon→统计→点云快照；
//                      │                laserFed 闸门：steger(N) 入队后放行
//                      │                upload(N+1)——d_gray/激光掩膜覆写安全）
//                      ▼
//              reg 线程（×1，严格按帧序：光流/兜底/hold＋prevState 链）
//                      │
//   main ◀──wait───── 完成合并（gpuDone[收尾线程置] && regDone → 交还主线程）
//
// 正确性要点：
// - 09 算子非线程安全：GPU 链单 lane 串行用原实例组；标记点链每 lane 一组
// - 分离/激光算子结果为内部持久缓冲视图：跨帧消费的产物（标记掩膜、重建点云）
//   在本帧内 D2D 快照到 job 持有 GpuMat；帧内消费（steger→…→recon）同流 FIFO 保序
// - 配准跨帧依赖（prevState）：单独线程按帧序执行——与串行完全同序
// - 全局点云累积仍由主线程在 wait 返回后调用（保序；消费的是 recon 快照）
// - 串行头（warmup＋dumpFullFrames 帧）走原串行路径——逐帧全量 dump 与金样口径
//   逐字节一致
// 背压：在飞帧数上限 kMaxInflight（主线程 submit 阻塞等槽）。
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "harness.h"

namespace scr {

/// 【150FPS】线程绑核（默认开，JMW_AFFINITY=0 关闭）：GPU lane/标记 lane/配准
/// 各占独立物理核首个逻辑核，与系统及其他线程分开——减超线程兄弟互抢＋
/// 减调度迁移（实测标记链抢核拖累 GPU 线程 ~4-5ms/帧）。lp＝逻辑核号。
void pipePinThread(int lp, const char* who);

class PipelineExecutor {
public:
    /// lanes＝标记点链并行线程数（默认 3）
    PipelineExecutor(Harness& h, int lanes, int firstIdx, int totalCycles);
    ~PipelineExecutor();

    /// 非阻塞提交：把 [nextSubmit_, cur+lookahead] 内未提交帧铺满在飞窗口
    /// （背压：在飞满则本轮不提交——collect 后自然腾槽）
    void submitAhead(int curIdx);
    /// 阻塞收割第 curIdx 帧产物（乱序完成→按请求序交还；须按帧序调用）
    bool collect(int curIdx, CycleProducts& out, bool& recordUsed, std::string& err);

private:
    struct Job {
        int idx = -1;
        bool record = false;
        bool skipReg = false;                  // 读帧失败等：跳过标记/配准
        CycleProducts prod;
        std::shared_ptr<cv::cuda::GpuMat> snapMaskL, snapMaskR;   // 标记掩膜快照（像素回退路径）
        std::shared_ptr<cv::cuda::GpuMat> snapCloud;              // 重建点云快照
        cudaEvent_t evMaskReady = nullptr;
        // 【150FPS】GPU-RLE 掩膜行程（GPU lane 同步后回填；marker lane 零 CUDA）
        std::shared_ptr<std::vector<int>> runsL, runsR;
        int runsNL = 0, runsNR = 0, rleW = 0, rleH = 0;
        bool runsPixelFallback = false;      // 溢出/失败→像素路径（GPU lane 已下载）
        cv::Mat pixL, pixR;                  // 像素回退宿主副本
        std::atomic<bool> runsReady{false};
        std::atomic<bool> markerDone{false};
        std::atomic<bool> gpuDone{false};
        std::atomic<bool> regDone{false};
        std::atomic<bool> done{false};         // gpuDone && regDone
        std::string err;
        // 【诊断】JMW_PIPELOG=1 阶段时间戳（ms，pipe 启动为 0 点）
        double tSubmit = 0, tDispatched = 0, tMkStart = 0, tMkDone = 0,
               tRegDone = 0, tHarvest = 0;
    };

    void gpuLoop();
    void markerLoop(int laneIdx);
    void regLoop();
    /// 【利用率专项·收尾 lane】激光链收尾段专职线程（kLaserN_ 条，帧号取模
    /// 定 lane——确定性映射；每 lane 独立 LaserOps 双实例）：steger→undist→
    /// interp→v3→recon→统计→点云快照整段搬离 GPU lane——GPU lane 提交完分离
    /// 即跑下一帧（run-ahead 喂 GPU）；单收尾线程时其逐步全流同步会成为新串
    /// 行瓶颈（实测 12-14ms/帧）——双 lane 交叉在飞，GPU 队列常满
    void laserLoop(int laneIdx);
    /// steger(N) 入队后放行 upload(N+1)（d_gray／分离激光掩膜覆写安全闸门；
    /// 多 lane 口径＝各 lane 就其 parity 帧号就位）
    void waitLaserFed(int idx);
    bool pushLaser(Job* j);
    void bumpLaserFed(int laneIdx, int idx);
    void releaseLaserGate(int idx);           // 非移交路径：全 lane 闸门放行
    Harness::LaserOps& laneLaserOps(int i);
    void finishJob(Job* j);                    // 末位完成者置 done＋通知

    Harness& h_;
    int nLanes_;
    int firstIdx_ = -1;                        // 首个流水线帧号（reg 期望起点）
    int nextRegIdx_ = -1;                      // reg 线程下一期望帧号
    int totalCycles_ = 0;                      // 总帧数（submitAhead 上界）
    int nextSubmit_ = -1;                      // 下一未提交帧号
    static constexpr int kMaxInflight = 16;    // 在飞帧数上限（背压——lane 宿主
                                               // 段期间维持 GPU 队列深度的余量）

    std::thread gpuTh_;
    std::vector<std::thread> markerThs_;
    std::thread regTh_;
    // —— 【利用率专项】激光收尾 lane ×kLaserN_（双实例交叉在飞＋每 lane 独立流）——
    std::vector<std::thread> laserThs_;
    std::vector<std::deque<Job*>> laserQs_;    // per-lane 待收尾帧（深度 kLaserQ）
    std::condition_variable laserCv_, fedCv_;
    std::vector<int> laserFedLane_;            // per-lane steger 已入队最大帧号（mtx_ 保护）
    int kLaserN_ = 3;                          // 收尾 lane 数（JMW_MLASER，1..4）。
                                               // 【修复 2026-10-19】原默认 2 与构造
                                               // 注释「默认 3」不符（改注释漏成员）——
                                               // nsys 实锤仅 2 流有活动；3 lane 提高
                                               // v3 宿主段交错覆盖（利用率主因①）
    bool laserOn_ = true;                      // NOLASER/markerOnly 诊断模式直通
    static constexpr int kLaserQ = 4;          // per-lane 收尾队列深度（run-ahead）
    // 【多流解耦】每 lane 专用 CUDA 流＋三事件链（lane 内全流同步只排空自己；
    // 全流同步会排干整条流水线队列——nsys 实测同步往返带因此占空闲大头）：
    //   evGate_[lane]  G 流记录（lmask D2D(N) 后）→ lane 流 steger(N) 前等待
    //   evSteg_[lane]  lane 流记录（steger 入队后）→ G 流 upload(N+k) 前等待
    //   evCloud_[lane] lane 流记录（点云 D2D 后）→ collect 返回前 G 流等待
    //   （accumulate 由主线程在 G 流排队——事件保证读到 D2D 后的槽内容）
    std::vector<cv::cuda::Stream> laserStreams_;
    std::vector<cudaEvent_t> evGate_, evSteg_, evCloud_;
    std::mutex cloudMtx_;                      // 点云快照环账本（多收尾 lane 共享）
    std::vector<std::unique_ptr<cv::cuda::Stream>> laneStreams_;   // lane 下载流

    std::mutex mtx_;
    std::condition_variable gpuCv_, markerCv_, regCv_, doneCv_, slotCv_, runsCv_;
    std::queue<Job*> gpuQ_;
    std::queue<Job*> markerQ_;
    std::map<int, Job*> regReady_;             // markerDone 的乱序完成集（按帧号）
    std::deque<std::unique_ptr<Job>> live_;    // 在飞 job 所有权
    int inflight_ = 0;
    bool stop_ = false;

    // —— GPU-RLE 常驻资源（GPU lane 线程专用）——
    static constexpr int kRleCap = 65536;      // 行程容量（溢出走像素回退）
    cv::cuda::GpuMat d_rleL_, d_rleR_;         // 设备行程缓冲（1×3*cap CV_32SC1）
    int* rleCounts_ = nullptr;                 // 锁页 [nL,nR]
    int* rleRuns_ = nullptr;                   // 锁页 6*cap ints

    // 【150FPS·快照环形复用】原 make_shared<GpuMat> 每帧 cudaMalloc ×3（掩膜×2＋
    // 点云×1，同步调用 ~1-2ms/块）＝~5ms/帧纯空闲（nsys 时间线实锤：D2D→D2D
    // 2.56ms/帧＋D2H→D2D 2.43ms/帧空档）。掩膜环形 4 槽（同帧 RLE 消费＋罕见
    // 像素回退跨帧读）；点云环形 8 槽（accumulate 消费方=主线程，滞后≤在飞数）
    cv::cuda::GpuMat maskSnap_[4];
    cv::cuda::GpuMat cloudSnap_[16];
    int maskSnapIdx_ = 0, cloudSnapIdx_ = 0;
    // 【利用率专项·lane 解耦】激光掩膜每 lane 快照（单槽——闸门＋FIFO 保序；
    // steger 消费快照视图后 sep 内部缓冲仅 GPU lane 生命周期内触碰）
    std::vector<cv::cuda::GpuMat> lmaskL_, lmaskR_;   // [kLaserN]
    /// 环形槽包装为非持有 shared_ptr（生命周期由流水线成员保证）
    static std::shared_ptr<cv::cuda::GpuMat> ringView(cv::cuda::GpuMat& m) {
        return std::shared_ptr<cv::cuda::GpuMat>(&m, [](cv::cuda::GpuMat*) {});
    }
};

} // namespace scr
