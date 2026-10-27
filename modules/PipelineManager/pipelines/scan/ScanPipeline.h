// ScanPipeline.h — C 扫描流水线（261027 重构：五链独占版）
//
// 旧 lane（SchedulerRuntime＋ScanChains＋FuseConsumer）已退役删除——生产扫描
// 全链＝五链执行器（新算子：mask_separation/flat_center/steger_fast/
// undistort_cuda/epipolar_interp_dual/laser_match_scan_v3/laser_reconstruct_cpu/
// conic_reconstruct/光流+frame_fuse 三级降级/LaserCloudFuseCuda 体素融合）。
//
// 前端＝06 SlotRing（相机帧，Overwrite 抓最新面孔）；后端＝obs（GBA）＋
// ICloudWarehouse（仓库直写）＋ISceneFeed（点云显示）。
//
// v1 限制：无暂停（pause→停止）、无检查点、无 marker 编辑账本（markerFuse/
// laserFuse 恒空）、无中段模拟注入（attachSimSource 保留为告警桩）。
#pragma once

#include <atomic>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "core/common/calib_result_types.h"       // calib::LaserPlaneMapTempTable
#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/scan/FrameObsAccumulator.h"
#include "pipelines/scan/ScanTypes.h"
#include "sched/FrameResultQueue.h"
#include "SlotRing.h"
#include "EnhancedFrame.h"

namespace Scanner::pipeline {

class ISceneFeed;
class SimScanSource;                            // 调试件（fivechain 无关，02 兼容桩）
class FiveChainAdapter;                          // fivechain/ 目录（同 Scanner::pipeline）

/// 会话状态机（五链版：Paused/Faulted 为兼容保留——pause 即 Stopped）
enum class State { Idle, Configured, Running, Paused, Faulted, Stopped };

/// C 扫描流水线——五链独占（旧 lane 已退役）
class ScanPipeline : public IPipelineObject {
public:
    explicit ScanPipeline(ScanConfig cfg);
    ~ScanPipeline() override;                      // 安全网：stop()
    ScanPipeline(const ScanPipeline&) = delete;
    ScanPipeline& operator=(const ScanPipeline&) = delete;

    // —— 装配（02 在构造后、start 前调用）——
    /// 输入源：06 SlotRing（Overwrite，扫描面孔）由外部持有，注入指针
    void attachRing(Scanner::data::SlotRing<Scanner::data::EnhancedFrame>& ring,
                    size_t dropThreshold);
    /// 静态 K/D＋激光温表（五链 v3 另从 config/ 装载 CMTT 曲线表——见 start）
    void attachCalib(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2,
                     const cv::Mat& D2, int imageWidth, int imageHeight,
                     std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable);
    /// 【退役桩】中段模拟提取源——五链模式忽略（保留 02 兼容）
    void attachSimSource(SimScanSource* src);
    /// 【退役桩】输出队列——恒空队列（旧 FuseConsumer 出口已删，02 兼容保留）
    sched::FrameResultQueue<FrameResult>& outputQueue() { return queue_; }
    /// 逐帧观测累加器（D GBA 读）
    FrameObsAccumulator& obs();
    /// 【退役桩】编辑账本访问（05）——五链 v1 无 marker 编辑账本，恒空
    IMarkerFuse* markerFuse() { return nullptr; }
#ifdef JMW_BUILD_CUDA
    /// 【退役桩】编辑账本访问（05 激光侧）——恒空
    ILaserFuse* laserFuse() { return nullptr; }
#endif

    // —— 生命周期（IPipelineObject：configure/start/stop/isRunning）——
    Scanner::Result configure(const PipelineDeps& deps) override;
    Scanner::Result start() override;
    void stop() override;
    bool isRunning() const override;
    /// 【退役桩】五链无暂停语义——转为停止（已融合数据保留；续扫请新会话）
    void pause();
    Scanner::Result resume();                      // 恒失败（无暂停态）
    Scanner::Result recover();                     // 恒失败（原地恢复退役）
    State state() const { return state_.load(); }

    // —— 【退役桩】旧 lane 遗留接口（02 兼容——五链无对应概念）——
    void setDepthOfField(int mode);                // 景深过滤（旧固定视差补偿件）——no-op
    void setDofRange(double nMin, double nMax, double fMin, double fMax);   // no-op
    uint64_t consumedFrames() const;               // 统计——0
    uint64_t droppedFrames() const;                // 统计——0

private:
    ScanConfig cfg_;
    mutable std::atomic<State> state_{State::Idle};

    // —— 装配输入 ——
    Scanner::data::SlotRing<Scanner::data::EnhancedFrame>* ring_ = nullptr;
    size_t dropThreshold_ = 0;
    cv::Mat K1_, D1_, K2_, D2_;
    int imageWidth_ = 0, imageHeight_ = 0;
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable_;
    Scanner::data::ICloudWarehouse* cloudWarehouse_ = nullptr;   // 仓库直写口
    ISceneFeed* sceneFeed_ = nullptr;                            // 点云显示口

    // —— 装配产物 ——
    std::unique_ptr<PipelineEventSink> sink_;
    sched::FrameResultQueue<FrameResult> queue_{64};   // 退役桩（恒空——02 兼容）

    // —— 五链（唯一执行路径）——
    FiveChainAdapter* fcAdapter_ = nullptr;
    bool fcActive_ = false;
    nlohmann::json fcOpsJson_;                     // 算子参数（config/fivechain_operators.json）
    bool fcOpsLoaded_ = false;

    // —— 运行时观测 ——
    FrameObsAccumulator obs_{1u << 20};            // 五链逐帧 obs 无激光 host 缓存——小预算即可
};

} // namespace Scanner::pipeline
