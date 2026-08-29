#pragma once
// ============================================================================
// ScanChains.h — C 扫描 GPU 链 / P 核链 / E 核终段 算子装配（09 算子 → LaneHooks）
//
// 调用序（客户端扫描流水线文档 §一 + 扫描并行调度260629 §4.3）：
//   GPU 链（E 核线程，持 GpuSlotService::SlotGuard 的 stream 上串行）：
//     mask_separation(L/R) → ccl(L/R d_markingPointMask) →【frontReady() ccl 就绪点】
//     → enableLaser：steger(Flat) → undistort_cuda → epipolar_interp(lineIdCheck=false)
//       → laser_match_scan(查温度表) → laser_reconstruct(Q=snapshot.Q) → 激光块入池
//     → enableLaser=false：ccl 后 guard.reset() 提前归还（A 模式短路）
//   P 核链（PCoreBroker 上，与激光链帧内并行）：
//     image_split → zernike_edge → image_merge → undistort_cpu(R1/R2/P1/P2/Q=snapshot)
//     → ellipse_fit → marker_match → epipolar_intersect → edge_match
//     → point_reconstruct → 配准（ prevState 原子快照：首帧初始化 / optical_flow_fuse
//       失败→frame_fuse 兜底 / 再失败→沿用快照 R/T 记 Degraded → 成功 atomic_store）
//   E 核终段：fut.get() 收 P 链 → 汇合组装 FrameResult（帧号/R/T/markers/laser/quality）
//     → sink push（T8 契约：eFinalize 自行 push）
//
// 线程模型（关键）：09 算子"每实例非线程安全"，而 runtime 多 lane（E 核×N +
// P worker×N-1）并发调用同一组钩子——故算子集按 lane 惰性创建一次、跨帧复用，
// 存放在每 lane 一份的 ScanFront::ops（gpuChain 首帧先于 frontReady() 创建，
// pChain 仅经 frontReady/兜底提交后运行，读到的一定非空且此后不再写）。
// ScanFront 分区（SchedulerRuntime 契约）：roisL/R 为 frontReady 前写入的
// 前段产物（pChain 只读）；laserBlock 为激光分区（仅 gpuChain 激光段写、
// eFinalize 读）；pChain 产出直接写 TResult（FrameResult）。
//
// JMW_BUILD_CUDA 守卫：无 CUDA 构建仅提供编译守卫（gpuChain 返回 false），
// 运行不支持（mask_separation 前段亦需 CUDA，见扫描文档 §3.4）。
// ============================================================================
#include <chrono>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>

#include "base/types.h"
#include "core/common/calib_result_types.h"     // calib::LaserPlaneMapTempTable
#include "core/scheduler/prev_frame_state.h"    // calib::AtomicFrameState
#include "EnhancedFrame.h"                      // Scanner::data::EnhancedFrame（06）
#include "pipelines/scan/GpuPointCloudPool.h"   // GpuPointCloudPool（CUDA 守卫内）
#include "pipelines/scan/ScanTypes.h"           // ScanConfig/FrameResult/GpuPointCloudBlock
#include "sched/FrameResultQueue.h"
#include "sched/GpuSlotService.h"
#include "sched/SchedulerRuntime.h"             // LaneHooks

namespace Scanner::pipeline {

/// 全局配准快照锚（atomic_load/store 自由函数操作，见 prev_frame_state.h；
/// 初始 nullptr=首帧未初始化）
using AtomicFrameStatePtr = std::shared_ptr<calib::AtomicFrameState>;

/// per-lane 算子集（ScanChains.cpp 内定义，头文件保持不透明）
struct ScanLaneOps;

// ============================================================================
// ScanFront — GPU 前段产物 + P 链消费中转（每 lane 一份，跨帧复用）
// ============================================================================
struct ScanFront {
    // —— 前段产物（gpuChain frontReady() 前写入；此后 pChain 只读）——
    std::vector<cv::Rect> roisL;                // ccl 标记点包围盒（host，左）
    std::vector<cv::Rect> roisR;                // 同上（右）
    // —— 激光分区（frontReady() 后仅 gpuChain 激光段写、eFinalize 读）——
#ifdef JMW_BUILD_CUDA
    std::shared_ptr<GpuPointCloudBlock> laserBlock;   // 空=无激光（A 模式/失败/池耗尽）
    bool laserTruncated = false;                      // 本帧激光块因池容量截断（降级用）
#endif
    // —— lane 锚（gpuChain 首帧惰性创建，先于 frontReady()；此后两链只读）——
    std::shared_ptr<ScanLaneOps> ops;
};

// ============================================================================
// ScanChainDeps — 装配依赖（组合注入）
// ============================================================================
struct ScanChainDeps {
    /// 全局配准快照锚（存储即本 deps 拷贝内的 shared_ptr 本体：pChain 经
    /// atomic_load/store 读写；null=首帧未初始化的正确初态，非空=热启动种子）
    AtomicFrameStatePtr prevState;
#ifdef JMW_BUILD_CUDA
    GpuPointCloudPool* laserPool = nullptr;     // 激光块池（enableLaser 时必填）
#endif
    /// 温度补偿激光平面映射表（整表注入 laser_match_scan::SetTempTable，
    /// 逐帧 SetCurrentTemperature(帧温) 由算子内部查最近档）
    std::shared_ptr<const calib::LaserPlaneMapTempTable> laserTable;
    /// 双目内参/畸变（undistort_cpu 构造校验 + undistort_cuda 逐帧 SetParams；
    /// ⚠ 当前 06 CalibSnapshot 不含 K/D，按静态标定值注入——逐温度 K/D 待表契约扩展）
    cv::Mat K1, D1, K2, D2;
    int imageWidth = 0;                         // undistort_cpu 构造校验需要（>0）
    int imageHeight = 0;
    sched::FrameResultQueue<FrameResult>* sink = nullptr;  // eFinalize push（空=不 push）
    std::chrono::milliseconds poolAcquireTimeout{50};      // 激光块池取块超时
};

// ============================================================================
// ScanChains — 三钩子工厂（供 T17 ScanPipeline 装入 SchedulerRuntime）
// ⚠ assemble() 返回的钩子捕获 this：ScanChains 生命周期须覆盖运行期（至 drain）
// ============================================================================
class ScanChains {
public:
    using Hooks = sched::LaneHooks<data::EnhancedFrame, ScanFront, FrameResult>;

    ScanChains(ScanConfig cfg, ScanChainDeps deps);
    ~ScanChains();
    ScanChains(const ScanChains&) = delete;
    ScanChains& operator=(const ScanChains&) = delete;

    /// 三钩子（gpuChain/pChain/eFinalize）；装配期依赖非法时钩子恒 fail（见日志）
    Hooks assemble();

    /// 全局配准快照锚（ prevState 原子读写出口——检查点保存/恢复消费；
    /// 声明序即 deps_ 拷贝内 shared_ptr 本体，原子 load/store 直接操作）
    AtomicFrameStatePtr& prevStateAnchor() { return deps_.prevState; }

private:
    std::shared_ptr<ScanLaneOps> makeOps() const;
    bool runMarkerChain(const data::EnhancedFrame& frame, ScanLaneOps& ops,
                        const ScanFront& front,
                        std::vector<cv::Point3d>& positions,
                        std::vector<cv::Vec3d>& normals) const;   // false=无有效标记点
    void runRegistration(const data::EnhancedFrame& frame, ScanLaneOps& ops,
                         const std::vector<cv::Point3d>& positions,
                         const std::vector<cv::Vec3d>& normals,
                         FrameResult& result);  // 非常量：原子写全局 prevState 快照

    ScanConfig cfg_;
    ScanChainDeps deps_;
    std::string initError_;                     // 装配期依赖校验错误（非空=钩子恒 fail）
};

} // namespace Scanner::pipeline
