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

#include <opencv2/core.hpp>

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
    bool enableLaser = true;                 // A 模式 false＝纯标记点（不发射激光）
    float voxelSizeMm = 0.0f;                // P-4 体素密度档值（mm）0=默认 0.25 硬
                                             //   编码值档→会话生效（02 装配时注入）
    bool enableFinalBA = true;               // 置 true：A/B 结束走 GBA（终局遍正式
    size_t laserCacheBudgetMB = 2048;        // D 终局遍激光帧缓存预算（最大 6 GB实测）
    int hangTimeoutMs = 10000;               // 看门狗（lane 心跳超时默认值）0=关（见 SchedConfig）
    std::string checkpointPath;              // 非空=stop() 自动落盘（崩溃恢复链）
    // existingMarkers——跨扫描时一次性装配加载（非空且标志点齐全）正常加载（非动态）
    std::vector<calib::MarkerCloudPoint> existingMarkers;   // 标志点（globalId 继续做 obs 侧）

    // 261025 Phase 2：五链执行器开关（默认 false=走旧 lane 调度；true=走五链）
    // 环境变量 JMW_FIVECHAIN=1 亦可开启（app 侧经环境变量切换不动 config）
    bool useFiveChain = false;
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
// SimFrameObs — 中段模拟提取的单帧原始观测（设备系；260917 架构对齐）
// SimScanSource 产出 → ScanChains gpuChain frontReady 前写 ScanFront 分区
// → pChain 以此替换真实提取结果走生产配准（vs prevState 全局锚）→ 激光块
// 覆写 → 生产融合/体素/渲染。纯数据，配准/融合语义全部归生产链。
// ============================================================================
struct SimFrameObs {
    uint64_t frameId = 0;
    std::vector<cv::Point3d> markerPositions;   // 设备系标志点观测（含检测噪声）
    std::vector<cv::Vec3d> markerNormals;       // 法线（场景真值旋转）
    std::vector<cv::Point3f> laser;             // 设备系激光观测（含抖动；可空）
};

// ============================================================================
// IMarkerFuse / ILaserFuse — 编辑账本窄接口（261027 自 FuseConsumer.h 迁入——
// 旧消费线程已退役，接口保留给 05 编辑会话/后续五链账本扩展）
// ============================================================================
struct IMarkerFuse {
    virtual ~IMarkerFuse() = default;
    virtual void fuse(const std::vector<calib::MarkerPoint3D>& markers,
                      const double R[9], const double T[3]) = 0;
    virtual const std::vector<calib::MarkerCloudPoint>& fusedPoints() const = 0;
    /// 编辑账本移除点（05）——默认 fail（由自实现可覆写透传 09 removePoints）
    virtual calib::ResultStatus removePoints(const std::vector<uint32_t>& indices) {
        (void)indices;
        return calib::ResultStatus::fail("IMarkerFuse: removePoints 未实现");
    }
};

#ifdef JMW_BUILD_CUDA
/// 激光融合累积器窄接口（编辑会话作用）
struct ILaserFuse {
    virtual ~ILaserFuse() = default;
    virtual void fuse(const GpuPointCloudBlock& block,
                      const double R[9], const double T[3]) = 0;
    virtual size_t fusedPointCount() const { return 0; }
    virtual std::vector<float> downloadFusedXyz() const { return {}; }
    virtual std::vector<float> downloadFusedXyzTail(size_t fromPts) const {
        auto all = downloadFusedXyz();
        if (fromPts * 3 >= all.size()) return {};
        all.erase(all.begin(),
                  all.begin() + static_cast<std::ptrdiff_t>(fromPts * 3));
        return all;
    }
    virtual calib::ResultStatus removePoints(const std::vector<uint32_t>& indices) {
        (void)indices;
        return calib::ResultStatus::fail("ILaserFuse: removePoints 未实现");
    }
};
#endif // JMW_BUILD_CUDA

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
