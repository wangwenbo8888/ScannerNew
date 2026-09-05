#pragma once
// ============================================================================
// ScanPipeline.h — C 扫描处理流水线对象（02 工作流 ⑤ 核心引擎；IPipelineObject 落地）
//
// 总装：SchedulerRuntime（E 核编排，X lane）× GrabLatestSource（06 SlotRing 扫描
// 面孔）× ScanChains（GPU/P/E 三钩子，生产装配）× FuseConsumer（融合消费线程）
// + FrameObsAccumulator（GBA 观测）+ GpuPointCloudPool（激光显存块池）。
//
// 生命周期（02 装配序）：
//   ctor → attachRing/attachCalib → configure(deps) → start ⇄ pause/resume → stop
//   （会话私有件：stop 后不可重启，续扫建新对象）
//
// seed 时序（existingMarkers 非空，与 A/B 模式正交）：start() 内、runtime.start()
// 之前 marker 融合适配器 seed() 一次性预填——保证"先于任何扫描帧 fuse"；
// hpGlobalIds 映射简化：把 existingMarkers 的"globalId 语义"按下标 0..n-1 传
// FuseConsumer 的 highPrecisionGlobalIds（09 optical_flow_fuse 首帧以已有点为
// 靶标时匹配点拿 globalId=bestIdx——即已有点下标；后续帧链式传播同源 id。逐点
// 真 globalId 与已有点身份的逐点对应依赖 09 侧配准输出，此处按下标对接为
// 已声明的简化映射）。
//
// pause/resume：runtime 是 drain 后可 restart 的（T8 用例 8 验证；GpuService 每
// 周期重建，GPU 流工厂注入一次即续用）；queue/pool/obs/融合累积器全保留，
// 配准 prevState 锚（ScanChains deps 内）随 chains_ 存续跨 pause 延续。
// **消费水位持久**：pause 记 runtime.lastCounter()（下一待读帧号），resume 以
// startCounter 注入重启——已消费帧不重扫（防错误 R/T 重复污染点云与 obs）。
// **异常停收敛**：runtime 钩子异常即停（自灭）后，isRunning()/pause()/stop()
// 入口惰性收敛 Faulted + Fault(1604) 一次性上报（Sink）；累积数据保留待 stop 收尾。
// pause 期间 ring 写入由 08 停——本对象不碰采集。
//
// 停止顺序（不丢在飞帧）：runtime.requestStop（lane 停抓新帧）→
// drainAndShutdown（在飞帧排空，eFinalize 已 push 队列）→ consumer 置停+join
// （drain 语义：排空队列后退出）。
//
// JMW_BUILD_CUDA=OFF：LaserFuse 适配器不存在 → configure 把 enableLaser 强制
// false + EventSink 一次性 Warning 上报（A 模式降级）。
//
// 测试模式（公共 API，供测试注入假链/假适配器——真算子路径由 T15/T16 覆盖）：
//   attachTestHooks     — 假 LaneHooks 替换 ScanChains 装配（免标定/GPU）
//   attachTestFuseAdapters — 假融合适配器替换真算子（marker 侧含 seed 时序观测）
//   测试模式下 start 自动向 runtime 注入假 GPU 流工厂（假链不触流，无设备依赖）
// ============================================================================
#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "core/common/calib_result_types.h"       // calib::LaserPlaneMapTempTable
#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/FuseConsumer.h"
#include "pipelines/scan/ScanChains.h"
#include "pipelines/scan/ScanTypes.h"
#include "sched/FrameResultQueue.h"
#include "sched/IFrameSource.h"
#include "sched/SchedulerRuntime.h"
#include "SlotRing.h"

namespace Scanner::pipeline {

/// 可 seed 的标记点融合窄接口（真适配器透传 09 MarkerCloudFuseCPU::seed；
/// 测试假实现记录 seed/fuse 时序）。生产/测试统一经此注入 FuseConsumer。
struct ISeedableMarkerFuse : IMarkerFuse {
    virtual Scanner::Result seed(const std::vector<calib::MarkerCloudPoint>& pts) = 0;
};

class ScanPipeline : public IPipelineObject {
public:
    using Hooks = ScanChains::Hooks;

    /// cfg 含 enableLaser/existingMarkers/laserCacheBudgetMB
    explicit ScanPipeline(ScanConfig cfg);
    ~ScanPipeline() override;                      // 安全网：stop()
    ScanPipeline(const ScanPipeline&) = delete;
    ScanPipeline& operator=(const ScanPipeline&) = delete;

    // —— 装配（02 在 ② 构造后、start 前调用）——
    /// 输入源：06 SlotRing（Overwrite，扫描面孔）由外部（02/06 会话件）持有，注入指针
    void attachRing(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring,
                    size_t dropThreshold);
    /// 静态 K/D（过渡契约——逐温档 K/D 待 06 CalibrationRepository 表契约扩展）与
    /// 激光温度表（laser_match_scan 整表 shared_ptr 注入；A 模式可空）
    void attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                     const cv::Mat& D2, int imageWidth, int imageHeight,
                     std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable);
    /// 输出队列（FuseConsumer 消费源，同对象内；02/D 诊断读）
    sched::FrameResultQueue<FrameResult>& outputQueue();
    /// 逐帧观测累加器（D GBA 读）
    FrameObsAccumulator& obs();
    /// 编辑账本访问（05 P4）：标志点融合累积器（未 start/已停=空；窄接口
    /// IMarkerFuse——removePoints/fusedPoints 供编辑会话作用）
    IMarkerFuse* markerFuse() { return markerFuse_; }

    // —— P3 可观测（渲染加固计划）：app 装配喂 HardwareMonitor ——
    /// 已融合消费帧数（FuseConsumer::consumed；未启动=0）
    uint64_t consumedFrames() const;
    /// 输出队列覆盖丢帧累计（队列满整帧丢；消费跟不上时增长）
    uint64_t droppedFrames() const;

    // —— 测试模式注入（configure 前调用；与 attachCalib 互斥）——
    void attachTestHooks(Hooks hooks);
    void attachTestFuseAdapters(ISeedableMarkerFuse* markerFuse
#ifdef JMW_BUILD_CUDA
                                ,
                                ILaserFuse* laserFuse = nullptr
#endif
    );

    // —— IPipelineObject ——
    Scanner::Result configure(const PipelineDeps& deps) override;  // 注入 sink/sceneFeed；
    /// 装配 FuseConsumer（真算子适配器）+ pool + ScanChains hooks
    Scanner::Result start() override;              // seed（若有）→ runtime.start → consumer.start
    void stop() override;                          // 见文件头停止顺序；isRunning false
    bool isRunning() const override;

    // —— 扫描会话控制（⑥ 就绪态再按键回 ③④⑤）——
    void pause();                                  // lane 停+drain；consumer 保活，累积全保留
    Scanner::Result resume();                      // runtime 重启（restart 语义）

    // —— 会话自愈（三件套：看门狗在 runtime 内建；此处 recover＋检查点）——
    /// Faulted 原地恢复：限时 drain（2s；僵尸 lane detach 兜底）→ runtime 按水位
    /// 重启。**consumer/obs/融合累积/prevState 锚全保留**——秒级满血续算。
    /// 3 次上限（防恢复风暴；用尽=会话重建/进程重启）
    Scanner::Result recover();
    /// 会话检查点落盘：obs＋激光缓存（经 FrameObsAccumulator 检查点）＋配准锚
    /// prevState＋消费水位。崩溃重启后新会话 restoreCheckpoint → D 仍可全量
    /// GBA＋重放（C 线融合云不可序列化——由 D 重放重建最终产物）
    Scanner::Result saveCheckpoint(const std::string& path);
    /// 检查点恢复（替换语义）：要求非 Running 态（推荐 configure 后、start 前）
    Scanner::Result restoreCheckpoint(const std::string& path);

private:
    enum class State { Idle, Configured, Running, Paused, Faulted, Stopped };

    std::unique_ptr<PipelineEventSink> makeSink(const PipelineDeps& deps) const;
    sched::SchedConfig scanSchedConfig() const;
    /// 惰性状态收敛：Running 且 runtime lanes 已全退（正常停不会有此态——只有
    /// 异常即停/资源故障）→ Faulted + sink Fault(1604) 一次性上报。
    /// isRunning/pause/stop 入口先调（const：从 const isRunning 惰性触发）
    void syncState() const;

    ScanConfig cfg_;
    mutable State state_ = State::Idle;

    // —— 装配输入 ——
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame>* ring_ = nullptr;
    size_t dropThreshold_ = 0;
    cv::Mat K1_, D1_, K2_, D2_;
    int imageWidth_ = 0, imageHeight_ = 0;
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable_;

    // —— 测试模式 ——
    bool testHooksSet_ = false;
    Hooks testHooks_;
    ISeedableMarkerFuse* testMarkerFuse_ = nullptr;
    bool testMarkerFuseSet_ = false;
#ifdef JMW_BUILD_CUDA
    ILaserFuse* testLaserFuse_ = nullptr;
    bool testLaserFuseSet_ = false;
#endif

    // —— 装配产物（configure 建）——
    std::unique_ptr<PipelineEventSink> sink_;
    std::unique_ptr<ISeedableMarkerFuse> ownedMarkerFuse_;   // 真适配器（自有）
    ISeedableMarkerFuse* markerFuse_ = nullptr;              // 生效指针（真/测试注入）
#ifdef JMW_BUILD_CUDA
    std::unique_ptr<ILaserFuse> ownedLaserFuse_;
    ILaserFuse* laserFuse_ = nullptr;
    std::unique_ptr<GpuPointCloudPool> laserPool_;           // 激光显存块池（生产+enableLaser）
#endif
    std::unique_ptr<ScanChains> chains_;                     // 生产链（pause/resume 存续）
    std::unique_ptr<FuseConsumer> consumer_;
    std::vector<int> hpGlobalIds_;                           // seed 点高精度 id 集（0..n-1）
    AtomicFrameStatePtr testPrevState_;                      // 测试模式配准锚兜底（chains_ 不建时）

    // —— 运行时组件（构造即有）——
    sched::FrameResultQueue<FrameResult> queue_;
    FrameObsAccumulator obs_;
    sched::SchedulerRuntime runtime_;
    std::unique_ptr<sched::GrabLatestSource<Scanner::data::EnhancedFrame>> source_;

    std::atomic<int> fakeStreamSeq_{0};                      // 测试模式假流句柄序号
    bool seeded_ = false;                                    // seed 一次性守卫（start 重试不重复 seed）
    uint64_t pauseCounter_ = 0;                              // pause 时消费水位（resume 注入）
    int recoverAttempts_ = 0;                                // recover 次数（上限 3，防恢复风暴）
    static constexpr int kMaxRecoverAttempts = 3;
};

} // namespace Scanner::pipeline
