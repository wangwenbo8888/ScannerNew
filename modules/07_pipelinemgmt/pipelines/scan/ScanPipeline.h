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
    /// 静态 K/D（过渡契约——逐温档 K/D 待 06 表契约扩展，见设计方案 §5.2）与
    /// 激光温度表（laser_match_scan 整表 shared_ptr 注入；A 模式可空）
    void attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                     const cv::Mat& D2, int imageWidth, int imageHeight,
                     std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable);
    /// 输出队列（FuseConsumer 消费源，同对象内；02/D 诊断读）
    sched::FrameResultQueue<FrameResult>& outputQueue();
    /// 逐帧观测累加器（D GBA 读）
    FrameObsAccumulator& obs();

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

private:
    enum class State { Idle, Configured, Running, Paused, Stopped };

    std::unique_ptr<PipelineEventSink> makeSink(const PipelineDeps& deps) const;
    sched::SchedConfig scanSchedConfig() const;

    ScanConfig cfg_;
    State state_ = State::Idle;

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

    // —— 运行时组件（构造即有）——
    sched::FrameResultQueue<FrameResult> queue_;
    FrameObsAccumulator obs_;
    sched::SchedulerRuntime runtime_;
    std::unique_ptr<sched::GrabLatestSource<Scanner::data::EnhancedFrame>> source_;

    std::atomic<int> fakeStreamSeq_{0};                      // 测试模式假流句柄序号
    bool seeded_ = false;                                    // seed 一次性守卫（start 重试不重复 seed）
};

} // namespace Scanner::pipeline
