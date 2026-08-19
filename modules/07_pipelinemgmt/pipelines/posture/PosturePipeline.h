#pragma once
// ============================================================================
// PosturePipeline.h — A 姿态判断流水线对象（01-⑤ 核心；IPipelineObject 落地）
// ============================================================================
// 总装：SchedulerRuntime（sequential=true 顺序反压）× SequentialSource（06
// SlotRing Backpressure 环）× 三段 LaneHooks × PostureConfirmTable（25 目标
// streak 确认簿记）。
//
// 生命周期（01 装配序）：
//   ctor → attachRing / attachTargets / attachInitialParams / setCompletionHook
//   → configure(deps) → start →（集齐自动收口 | stop）（会话私有件，不重启）
//
// 帧语义（不丢帧）：Backpressure 环写满阻塞写侧（08 采集线程）；
// SequentialSource claim→waitFor→read 顺序领全部周期，grab 时即 done() 腾位。
//
// 三段链（生产装配，形态对齐 ScanChains 先例）：
//   gpuChain（E 核线程持槽）：L/R 各 mask_extract → frame_filter（判 CycleUnit
//     的 markerL/R；maskRatio<阈=激光线帧→返回 false 销毁整周期；不碰
//     laserFrames——激光管帧是随周期保存的"乘客"）→ ccl → 包围盒入 front。
//     【A 模式不调 frontReady：gpuChain 返回后 runtime 兜底提交，天然串行】
//   pChain（P 核 worker）：标记点链 9 算子（image_split→…→point_reconstruct，
//     参数用初始参数组非帧快照）→ 无配准（配准在 E 段）
//   eFinalize（E 核线程）：配准三级降级（optical_flow_fuse 读 prevState 最新
//     快照/首帧 null 初始化 → frame_fuse 兜底 → 再失败沿用快照 R/T 记 Degraded）
//     → store 快照 → pose_estimate（目标表匹配交给 ConfirmTable；无网格参数时
//     算子优雅失败→沿用配准 R/T）→ ConfirmTable.report（新确认→计数/渲染切
//     目标）→ 每帧 pushPostureView（实时姿态+标志点检出数）。
//     ⚠ pushPostureView 可能从多条 lane 并发——ISceneFeed 实现须线程安全。
//     共享状态模型同姿态调度方案 §5：prevState 原子快照（读最新，X 核同时
//     融合）+ ConfirmTable 原子 CAS/细粒度锁。
//
// 收口时序（设计 §4.1：在飞周期判完再收口）：
//   集齐（ConfirmTable onComplete，lane 线程内）→ requestStop（原子置位安全）
//   → 唤醒收口 watcher 线程：drainAndShutdown（join lanes=在飞周期 eFinalize
//   判完）→ deps.acquisition->stopAcquisition()（采集停于排空之后）→
//   continueWithB(table().takeSessionData())（01-⑤→⑥ 叫醒 B）。
//   ⚠ drain 不能在 onComplete 线程内做（join 自身 lane=死锁）——收口序列由
//   专属 watcher 线程执行，onComplete 仅置停+唤醒。
//
// JMW_BUILD_CUDA=OFF：GPU 前段算子不可用 → 生产链 gpuChain 恒 false（运行
// 不支持，同 ScanChains 编译守卫；测试模式假链不受影响）。
//
// 测试模式（公共 API，同 T17）：attachTestHooks 假链注入（configure 免
// attachInitialParams；start 自动配假 GPU 流工厂）。真算子路径由 09 单测覆盖。
// ============================================================================
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

#include "core/scheduler/prev_frame_state.h"     // calib::AtomicFrameState
#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/posture/PostureConfirmTable.h"
#include "pipelines/posture/PostureTypes.h"
#include "sched/IFrameSource.h"
#include "sched/SchedulerRuntime.h"
#include "SlotRing.h"

namespace Scanner::pipeline {

struct PostureLaneOps;                            // per-lane 算子集（.cpp 内定义，头不透明）

// —— 初始参数组（2-6/3-1 严格同组：装配层单源注入，B 重建侧另途同组注入）——
struct PostureInitialParams {
    cv::Mat K1, D1, K2, D2;                       // 双目内参/畸变（undistort_cpu 构造校验）
    cv::Mat R1, R2, P1, P2, Q;                    // 立体矫正矩阵（初始参数，非帧快照）
    int imageWidth = 0, imageHeight = 0;
    double maskRatioThreshold = 0.0;              // frame_filter 激光线帧判定阈
};

// —— GPU 前段产物 + per-lane 算子锚（每 lane 一份，跨帧复用）——
struct PostureFront {
    std::vector<cv::Rect> roisL, roisR;           // ccl 包围盒（pChain 只读）
    std::shared_ptr<PostureLaneOps> ops;          // gpuChain 首帧惰性创建（先于 pChain）
};

// —— P 链输出（每帧一份，与 P 任务共享所有权；eFinalize 消费）——
struct PostureFrameResult {
    std::vector<cv::Point3d> positions;           // 标记点 3D（重建坐标系）
    std::vector<cv::Vec3d> normals;
    std::vector<cv::Point2f> ellipseCentersL, ellipseCentersR;   // 2-7 输出（B 3-1 消费）
};

class PosturePipeline : public IPipelineObject {
public:
    using Hooks = sched::LaneHooks<data::CycleUnit, PostureFront, PostureFrameResult>;

    explicit PosturePipeline(PostureConfirmTable::Config confirmCfg = {});
    ~PosturePipeline() override;                  // 安全网：stop()（幂等）
    PosturePipeline(const PosturePipeline&) = delete;
    PosturePipeline& operator=(const PosturePipeline&) = delete;

    // —— 装配（01 在 ② 装配期、configure 前调用）——
    /// 输入源：06 SlotRing（Backpressure 周期环，06 会话件持有），注入指针
    void attachRing(Scanner::data::SlotRing<Scanner::data::CycleUnit>& ring);
    /// 25×4×4 row-major 目标姿态表（count 须 ≤ PostureSessionData::kTargetCount）
    void attachTargets(const double (*targets)[16], int count);
    /// 初始参数组（内参/畸变/R1/P1——形态对齐标记点链算子参数；生产模式必备）
    void attachInitialParams(const PostureInitialParams& params);

    /// 收口钩子（② 装配期 01 注入）：集齐自动收口后携 SessionData 叫醒 B
    void setCompletionHook(std::function<void(PostureSessionData&&)> continueWithB);

    // —— 测试模式注入（configure 前调用；与 attachInitialParams 互斥）——
    void attachTestHooks(Hooks hooks);

    // —— IPipelineObject ——
    Scanner::Result configure(const PipelineDeps& deps) override;  // sink/sceneFeed/acquisition 接线 + 建表
    /// runtime.start（GPU 1 槽、sequential=true、queue=null）+ 收口 watcher 线程
    Scanner::Result start() override;
    /// 同步 stop：requestStop+drain（不触发 completion、不停采集）
    void stop() override;
    /// 含 lanesExited 惰性收敛（同 T17 syncState 模式；集齐收口路径除外）
    bool isRunning() const override;

    // —— UI/测试 ——
    int collectedCount() const;                   // 已确认姿态数（UI 实时轮询）
    PostureConfirmTable& table();                 // 前置：已 configure（测试假链 report 用）

private:
    enum class State { Idle, Configured, Running, Faulted, Stopped };

    std::unique_ptr<PipelineEventSink> makeSink(const PipelineDeps& deps) const;
    /// 生产三段链装配（形态对齐 ScanChains；失败安全见各钩子内日志）
    Hooks assembleChains();
    std::shared_ptr<PostureLaneOps> makeOps() const;
    /// E 段配准（最新快照模型）：填 liveR/liveT 并按三级降级原子写 prevState_
    void runRegistration(uint64_t cycleId, PostureLaneOps& ops,
                         const std::vector<cv::Point3d>& positions,
                         const std::vector<cv::Vec3d>& normals,
                         double liveR[9], double liveT[3]);
    /// 惰性收敛：Running 且 lanes 全退且非集齐路径 → Faulted + Fault(1701) 一次性
    void syncState() const;
    /// 收口 watcher：等集齐/停信号；集齐 → drain → stopAcquisition → continueWithB
    void watcherLoop();

    PostureConfirmTable::Config confirmCfg_;
    mutable std::atomic<State> state_{State::Idle};

    // —— 装配输入 ——
    Scanner::data::SlotRing<Scanner::data::CycleUnit>* ring_ = nullptr;
    double targets_[PostureSessionData::kTargetCount][16];
    int targetCount_ = 0;
    bool targetsAttached_ = false;
    bool paramsAttached_ = false;
    PostureInitialParams params_;
    std::function<void(PostureSessionData&&)> continueWithB_;

    // —— 测试模式 ——
    bool testHooksSet_ = false;
    Hooks testHooks_;
    std::atomic<int> fakeStreamSeq_{0};

    // —— 装配产物（configure 建）——
    std::unique_ptr<PipelineEventSink> sink_;
    PipelineDeps deps_{};
    std::unique_ptr<PostureConfirmTable> table_;

    // —— 运行时组件 ——
    sched::SchedulerRuntime runtime_;
    std::unique_ptr<sched::SequentialSource<Scanner::data::CycleUnit>> source_;
    std::shared_ptr<calib::AtomicFrameState> prevState_;   // 配准快照锚（null=首帧）

    // —— 收口协调（watcher 线程）——
    std::thread watcher_;
    mutable std::mutex watchMu_;                  // syncState（const 惰性收敛）亦取锁
    std::condition_variable watchCv_;
    bool completionRequested_ = false;            // onComplete 置位（lane 线程）
    bool watcherStop_ = false;                    // stop()/析构置位
    bool completionFired_ = false;                // continueWithB 已调（防重/迟到补发）
};

} // namespace Scanner::pipeline
