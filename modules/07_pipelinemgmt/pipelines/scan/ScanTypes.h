#pragma once
// ============================================================================
// ScanTypes.h — C 扫描流水线自有类型（纯数据，无逻辑）
//
// JMW_BUILD_CUDA 守卫方案：GpuPointCloudBlock 与 FrameResult::laser 仅在
// CUDA 构建下存在；BUILD_CUDA=OFF 时 laser 退化为 int 占位（无 opencv cuda 头依赖）。
// ============================================================================
#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"                                              // Scanner::QualityFlag
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h" // calib::MarkerCloudPoint
#include "core/scheduler/prev_frame_state.h"                         // calib::MarkerPoint3D

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>
#endif

namespace Scanner::pipeline {

// ============================================================================
// ScanConfig — C 扫描配置（单一流水线+开关，不拆两条）
// ============================================================================
struct ScanConfig {
    bool enableLaser = true;                 // A 模式 false：激光链整段跳过
    bool enableFinalBA = true;               // 恒 true（A/B 均必跑 GBA），保留显式
    size_t laserCacheBudgetMB = 2048;        // D 重融合激光帧缓存预算（开放项 6 待实测）
    // existingMarkers：启动扫描时一次性检测装载（非空且仅标记点才加载）——数据形态：
    std::vector<calib::MarkerCloudPoint> existingMarkers;   // 已有点（globalId 语义在 obs 层）
};

#ifdef JMW_BUILD_CUDA
// ============================================================================
// GpuPointCloudBlock — GPU 点云块（池化管理；宿主为 cv::cuda::GpuMat）
// ============================================================================
struct GpuPointCloudBlock {
    cv::cuda::GpuMat points;                 // CV_32FC3 1×N
    int count = 0;
    uint64_t frameId = 0;
    uint32_t slotId = 0;                     // 池槽位号（复用/调试断言用）
};
#endif // JMW_BUILD_CUDA

// ============================================================================
// FrameResult — C 输出队列元素（融合消费前的单帧结果）
// ============================================================================
struct FrameResult {
    uint64_t frameId = 0;
    uint64_t timestamp = 0;
    double temperature = 0.0;
    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double T[3] = {0,0,0};
    std::vector<calib::MarkerPoint3D> markers;           // 标记点 3D（含法线/globalId）
#ifdef JMW_BUILD_CUDA
    std::shared_ptr<GpuPointCloudBlock> laser;           // 空=无激光（A 模式/激光帧无点）
#else
    int laser = 0;                                       // CUDA 关闭占位
#endif
    Scanner::QualityFlag quality = Scanner::QualityFlag::Normal;
};

// ============================================================================
// MarkerObs / FrameObs — GBA 逐帧观测（FrameObsAccumulator 元素）
// ============================================================================
struct MarkerObs {
    double xyz[3] = {0,0,0};                  // 设备系坐标
    int globalId = -1;
    bool isHighPrecision = false;             // existingMarkers 匹配点
};

struct FrameObs {
    static constexpr size_t kNoLaserSlot = static_cast<size_t>(-1);

    uint64_t frameId = 0;
    double R_init[9] = {1,0,0, 0,1,0, 0,0,1};
    double t_init[3] = {0,0,0};
    std::vector<MarkerObs> markerObs;
    size_t laserCacheSlot = kNoLaserSlot;     // 激光帧缓存槽引用
};

} // namespace Scanner::pipeline
