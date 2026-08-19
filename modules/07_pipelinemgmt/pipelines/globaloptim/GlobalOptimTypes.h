#pragma once
// ============================================================================
// GlobalOptimTypes.h — D 全局优化对象自有类型（GlobalOptimOutput，纯数据）
// ============================================================================
#include <string>
#include <vector>

#include "base/types.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"   // MarkerCloudPoint
#include "scanning/global_optim/global_ba_cpu.h"                       // GlobalMarker/GlobalBAStats

#ifdef JMW_BUILD_CUDA
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"  // LaserCloudFuseDeviceContext
#endif

namespace Scanner::pipeline {

// GlobalOptimOutput — D 对象单次 run 的产物（修正位姿 + 修正后点云 + 报告量）
struct GlobalOptimOutput {
    std::vector<Scanner::Pose> poses;                  // 修正位姿数组（序=观测快照序；
                                                       // GBA 失败兜底=初值；取消=已产部分）
    std::vector<calib::MarkerCloudPoint> markerCloud;  // 修正后 marker 点云（重融合重放产物）
    std::vector<calib::GlobalMarker> gbaMarkers;       // GBA 优化后全局点（软先验锚定检视）
    calib::GlobalBAStats gbaStats;                     // GBA 统计（RMSE/迭代数/闭环）

    bool gbaSuccess = false;
    std::string gbaMessage;          // GBA 失败消息（成功时空）
    bool laserReplayed = false;      // 激光重融合真重放（false=无激光帧或降级近似未重放）
    bool laserDegraded = false;      // 激光缓存降级：沿初值融合结果不重算激光（简化近似）
    bool cancelled = false;          // 取消退出（GBA 后/重融合帧间检查点）
    size_t frameCount = 0;           // 参与本轮的观测帧数
    Scanner::QualityFlag quality = Scanner::QualityFlag::Normal;

#ifdef JMW_BUILD_CUDA
    // 激光重放累积器设备上下文（laserReplayed 时有效）。裸设备指针——生命周期随
    // GlobalOptimObject 持有的重放实例，至下一次 run 重建前有效。
    calib::LaserCloudFuseDeviceContext laserCtx{};
#endif
};

} // namespace Scanner::pipeline
