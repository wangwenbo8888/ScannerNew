#pragma once
// ============================================================================
// GlobalOptimObject.h — D 全局优化对象（02-⑦ 收尾批处理；IPipelineObject 落地）
//
// 职责（客户端扫描流水线.md §四 / 07 设计方案 §4.4）："GBA + 重融合"成对批算：
//   1 sceneFeed->notifyFreeze(true)（批处理期间 display 保持末帧）
//   2 装配 GlobalBAInput：观测累加器快照逐帧 {R_init, t_init, markerObs[]}；
//     软先验 = setExistingPrior 注入（有先验位置/σ）∪ obs 中 isHighPrecision id
//     （缺先验位置的 id 无法加残差，warn 跳过）
//   3 Ceres GBA（global_ba_cpu，位姿参数化由算子内部完成——R_init/t_init 直喂，
//     算子转四元数参与优化）→ 修正位姿数组
//   4 重融合（成对语义）：marker——新 MarkerCloudFuseCPU 实例按修正位姿重放全部
//     观测；laser——未降级（degradedLaser()==false，kNoLaserSlot 帧=本帧无激光）
//     → 新融合实例重放全部缓存激光帧；降级 → 近似路径：无激光修正重融合（沿
//     C 线初值融合结果不重算激光——简化，记 Degraded）
//   5 cloudRepo->write(tag)（句柄 tag 形态）+ sink 完成事件
//   6 notifyFreeze(false) + pushCloudSnapshot(修正点云)
// GBA 失败 → Fault 上报 + 沿初值位姿重融合兜底（Degraded）；cancel 在 GBA 后/
// 重融合帧间设检查点，取消时安全退出（解冻恢复、不推送不入库）。
//
// 观测来源：C 扫描流水线攒的 FrameObsAccumulator（run 入参 / attachObs 注入）——
// 不能从融合累积器取（体素去重云已丢逐帧结构）。
//
// 生命周期：ctor → setExistingPrior(可选，02 在 ⑦ 前) → configure(deps) →
//   run(阻塞批算) 或 attachObs + start()/stop()（IPipelineObject 适配，内部即
//   后台 run / cancel+join，语义同 B 标定计算流水线）。
// ============================================================================
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pipelines/IPipelineObject.h"
#include "pipelines/ISceneFeed.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"        // ProgressCb/CancelToken
#include "pipelines/globaloptim/GlobalOptimTypes.h"
#include "pipelines/scan/FrameObsAccumulator.h"

namespace Scanner::pipeline {

#ifdef JMW_BUILD_CUDA
/// 激光重融合重放窄接口（host xyz + 修正位姿 R/T）。生产适配器：host→GpuMat
/// 上传 + 新 LaserCloudFuseCuda::Execute（每次 run 新实例=重放语义；法线不随
/// 重放重估计——沿 C 线法线语义）。fuse 返回 false=本帧融合失败/异常——调用方
/// 聚合进质量（>0 帧 → Degraded + 1609 一次）。测试注入假实现免 GPU（注入式
/// 适配形态对齐 FuseConsumer::ILaserFuse）。
struct ILaserReplayFuse {
    virtual ~ILaserReplayFuse() = default;
    virtual bool fuse(const std::vector<float>& xyz, const double R[9], const double T[3]) = 0;
    /// 重放累积器设备上下文（供 output/渲染；默认空）
    virtual calib::LaserCloudFuseDeviceContext deviceContext() const { return {}; }
};
#endif

class GlobalOptimObject : public IPipelineObject {
public:
    struct Config {
        // 阈值类透传 GlobalBAParams（未列字段用 09 默认；软先验 σ 默认=09 默认 0.001）
        int    maxIterations = 200;
        double tolerance = 1e-10;
        bool   enablePoseGraphPreopt = true;
        bool   useSoftPrior = true;
        double defaultPriorSigma = 0.001;
        double sigmaObserved = 0.01;
    };

    /// GBA 执行函数（测试注入假 09 适配；空=真 GlobalBundleAdjustmentCPU::Execute）
    using GbaFn = std::function<calib::GlobalBAResult(const calib::GlobalBAInput&,
                                                      const calib::GlobalBAParams&)>;
#ifdef JMW_BUILD_CUDA
    using LaserReplayFuseFactory = std::function<std::unique_ptr<ILaserReplayFuse>()>;
#endif

    explicit GlobalOptimObject(Config cfg = {});
    ~GlobalOptimObject() override;                   // 安全网：stop()（join 后台线程）
    GlobalOptimObject(const GlobalOptimObject&) = delete;
    GlobalOptimObject& operator=(const GlobalOptimObject&) = delete;

    // —— 装配期注入（02 在 ⑦ 前调用）——
    /// existingMarkers 高精度先验（来源 06 点云仓库——02 侧装配）：
    /// globalIds 已对接（C 侧 hpGlobalIds 下标语义）；xyz3n 为 3n 展平 x,y,z；
    /// sigma 每点一个（空/缺项=09 默认 σ）
    void setExistingPrior(std::vector<int> globalIds, std::vector<double> xyz3n,
                          std::vector<double> sigma);

    // —— IPipelineObject ——
    Scanner::Result configure(const PipelineDeps& deps) override;   // sink/sceneFeed/cloudRepo

    /// 阻塞批算（02 ⑦ 触发）：freeze→GBA→重融合→写仓库→解冻推送（步骤见文件头）
    Scanner::Result run(FrameObsAccumulator& obs, const ProgressCb& cb, CancelToken& cancel);
    const GlobalOptimOutput& output() const;         // 上一次 run 产物（未 run 过为空壳）

    // —— IPipelineObject 适配（内部即 run/取消，见文件头）——
    void attachObs(FrameObsAccumulator& obs);        // start() 异步 run 的观测源
    Scanner::Result start() override;                // 后台线程对 attachObs 观测源 run
    void stop() override;                            // cancel + join
    bool isRunning() const override;

    // —— 测试注入（configure 前调用）——
    void attachTestGba(GbaFn fn);
#ifdef JMW_BUILD_CUDA
    void attachTestLaserFuseFactory(LaserReplayFuseFactory f);
#endif

private:
    Scanner::Result runLocked(FrameObsAccumulator& obs, const ProgressCb& cb,
                              CancelToken& cancel);
    calib::GlobalBAParams gbaParams() const;

    Config cfg_;
    std::vector<int> priorIds_;                      // setExistingPrior（软先验）
    std::vector<double> priorXyz_;                   // 3n 展平
    std::vector<double> priorSigma_;                 // 每点 σ（空/缺=09 默认）
    GbaFn gbaFn_;                                    // 空=真 09 GBA
#ifdef JMW_BUILD_CUDA
    LaserReplayFuseFactory laserFactory_;            // 空=真适配器
    std::unique_ptr<ILaserReplayFuse> replayLaser_;  // 最近一次 run 的重放实例（句柄保活）
#endif
    ICloudRepoWriter* cloudRepo_ = nullptr;          // run 尾自动写（configure 记录）
    ISceneFeed* sceneFeed_ = nullptr;                // 冻结/解冻 + 修正点云推送
    std::unique_ptr<PipelineEventSink> sink_;        // 事件出口（configure 内建）
    bool configured_ = false;

    GlobalOptimOutput out_;
    FrameObsAccumulator* attachedObs_ = nullptr;     // start() 路径观测源
    std::mutex runMutex_;                            // run/start 不可重入
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::shared_ptr<CancelToken> cancelToken_;       // start/stop 路径令牌（start 重建）
};

} // namespace Scanner::pipeline
