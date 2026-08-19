#pragma once
// ============================================================================
// FuseConsumer.h — C 扫描融合消费者（独立线程消费输出队列，不绑核、默认调度）
//
// 循环：pop(100ms 超时空转查 stop) → marker fuse → laser fuse+normal（真算子
// 适配器内，本类不感知）→ 渲染节流 pushCloudSnapshot → obs.push(FrameObs +
// 激光 host 拷贝)。
// drain 语义：stop 只在 pop 超时（=队列已空）后生效——退出前把队列已有帧消费完。
// 单帧异常兜底：processOne 内 catch(...) → spdlog + sink Fault(1602) → 丢帧续跑。
//
// 注入边界：IMarkerFuse/ILaserFuse 为融合算子窄接口（测试注入假实现；生产由
// ScanPipeline 装配真算子适配器——marker 侧走 MarkerCloudFuseCPU::Execute +
// seed()，laser 侧走 LaserCloudFuseCuda::Execute + LaserCloudNormalCuda，适配器
// 内持有激光渲染句柄）。激光 host 下载经 LaserDownloadFn 注入（生产默认
// GpuMat::download 真 CUDA；测试给假函数避免 GPU 数据语义依赖）。
// 渲染句柄：hostMarker=&markerFuse->fusedPoints()（稳定存储）；激光句柄由
// ILaserFuse 适配器持有、窄接口不暴露 → deviceLaser 恒 nullptr（T17 细化）。
//
// 线程模型：单消费线程独占调用 markerFuse/laserFuse/obs/sceneFeed/sink（09 算子
// 每实例非线程安全，单线程串行满足）；queue 为 MPMC。
// ============================================================================
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <unordered_set>
#include <vector>

#include "base/types.h"
#include "pipelines/ISceneFeed.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/ScanTypes.h"
#include "sched/FrameResultQueue.h"

namespace Scanner::pipeline {

/// 标记点融合算子窄接口（测试注入假实现；生产由 ScanPipeline 装配真算子适配器）
struct IMarkerFuse {
    virtual ~IMarkerFuse() = default;
    virtual void fuse(const std::vector<calib::MarkerPoint3D>& markers,
                      const double R[9], const double T[3]) = 0;
    virtual const std::vector<calib::MarkerCloudPoint>& fusedPoints() const = 0;
};

#ifdef JMW_BUILD_CUDA
/// 激光融合算子窄接口（enableLaser=false 时可为 null=A 模式；法线估计在适配器内）
struct ILaserFuse {
    virtual ~ILaserFuse() = default;
    virtual void fuse(const GpuPointCloudBlock& block,
                      const double R[9], const double T[3]) = 0;
};

/// 激光块 → host float xyz 下载函数（生产默认 GpuMat::download；测试假注入）
using LaserDownloadFn = std::function<std::vector<float>(const GpuPointCloudBlock&)>;
#endif

class FuseConsumer {
public:
    struct Deps {
        sched::FrameResultQueue<FrameResult>* queue = nullptr;   // 消费源（必填，start 校验）
        IMarkerFuse* markerFuse = nullptr;                // 必填（start 校验）
#ifdef JMW_BUILD_CUDA
        ILaserFuse* laserFuse = nullptr;                  // 可空=A 模式（跳过激光路径）
        LaserDownloadFn laserDownload;                    // 空=默认真下载（GpuMat::download）
#endif
        ISceneFeed* sceneFeed = nullptr;                  // 可空=不推渲染
        FrameObsAccumulator* obs = nullptr;               // 必填（start 校验）
        PipelineEventSink* sink = nullptr;                // 可空=不上报
        int renderThrottleFrames = 5;                     // 首帧起每 N 帧 push 一次（第 1、N+1…；<=0 按 1）
        const std::vector<int>* highPrecisionGlobalIds = nullptr;  // 可空=全部 false
    };

    explicit FuseConsumer(Deps d);
    ~FuseConsumer();                          // 安全网：requestStop + join
    FuseConsumer(const FuseConsumer&) = delete;
    FuseConsumer& operator=(const FuseConsumer&) = delete;

    Scanner::Result start();                  // 起线程（已在运行/依赖缺失 → fail）
    void requestStop();                       // 置停（异步；drain 语义见文件头）
    void join();                              // 置停 + 等线程退（drain 完成后返回）
    uint64_t consumed() const;                // 已消费帧数

private:
    void loop();                              // 消费主循环
    void processOne(FrameResult& fr);         // 单帧：融合→下载→渲染→观测（catch 兜底丢帧续跑）

    Deps deps_;
    std::unordered_set<int> hpIds_;           // 高精度 globalId 查找集（ctor 拷贝）
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> consumed_{0};
    std::thread thread_;
    bool laserDegradeReported_ = false;       // 降级一次性上报标记（仅消费线程访问）
};

} // namespace Scanner::pipeline
