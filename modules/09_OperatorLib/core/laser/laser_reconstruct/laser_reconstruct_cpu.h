/**
 * @file laser_reconstruct_cpu.h
 * @brief 激光线三维重建 CPU 算子 - 公开头文件
 *
 * 所属流程：扫描激光链第⑤步（laser_match_scan_v3 之后；五链重构 P0 交付——
 * CPU链2 零 CUDA 消费件）。平台：CPU（无 CUDA 依赖）。
 * 功能：对左右相机匹配的激光中心点对，使用立体矫正 Q 矩阵三维重建，
 *       输出有效点云＋激光线编号（顺序保持）。
 *
 * 数学与 CUDA 版（laser_reconstruct_cuda_impl.cu kernelComputeDisparity＋
 * kernelReconstruct3D）逐行等价，同 float 精度同运算序：
 *   d = left.x - right.x（右点仅 .x 参与运算，.y 不读——与 CUDA 版一致）
 *   W = Q[14]*d + Q[15]；W<=0 弃点
 *   X = (Q[0]*x + Q[3])/W；Y = (Q[5]*y + Q[7])/W；Z = Q[11]/W
 *   Z ∉ [minDepth, maxDepth] 弃点；幸存点按输入顺序紧凑输出
 * 参数复用 LaserReconstructParams（与 CUDA 版共享，config operators.laser_reconstruct
 * 同节喂养；deviceId 字段对本算子无意义，忽略）。
 *
 * 精度容差档次：档次③（亚像素/浮点类）——对拍口径见 tests/test_laser_reconstruct_cpu.cpp
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "laser_reconstruct_cuda.h"   // 共享 LaserReconstructParams

namespace calib {

// ============================================================================
// LaserReconstructCpuResult
// ============================================================================

struct LaserReconstructCpuResult {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    int validCount = 0;     // 幸存点数（= points3d.size() = validLineIds.size()）
    int totalInput = 0;     // 输入点对数

    std::vector<cv::Point3f> points3d;   // 幸存三维点（输入序）
    std::vector<int> validLineIds;       // 对应激光线编号（与 points3d 一一对应）

    LaserReconstructCpuResult() = default;
    ~LaserReconstructCpuResult() = default;
    LaserReconstructCpuResult(LaserReconstructCpuResult&&) = default;
    LaserReconstructCpuResult& operator=(LaserReconstructCpuResult&&) = default;
    LaserReconstructCpuResult(const LaserReconstructCpuResult&) = delete;
    LaserReconstructCpuResult& operator=(const LaserReconstructCpuResult&) = delete;
};

// ============================================================================
// LaserReconstructCpu
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 无状态（输出容器容量跨调用单调保持属 vector 天然行为，非业务状态）
// 说明: Impl 无；Q 按调用传入，params 仅缓存深度门槛。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占；静态方法 ReconstructInto 纯函数线程安全
// ==============================
class LaserReconstructCpu {
public:
    static constexpr const char* kLogTag = "08-LaserReconstructCpu";

    explicit LaserReconstructCpu(const LaserReconstructParams& params = {});
    ~LaserReconstructCpu() = default;

    LaserReconstructCpu(const LaserReconstructCpu&) = delete;
    LaserReconstructCpu& operator=(const LaserReconstructCpu&) = delete;

    /// 三元组契约 Execute：匹配点对＋线编号＋Q（CV_64F 4×4）→ 有效点云（输入序）。
    /// 空输入＝success（与 CUDA 版 "Empty input, no reconstruction" 同语义）。
    LaserReconstructCpuResult Execute(const std::vector<cv::Point2f>& matchedLeft,
                                      const std::vector<cv::Point2f>& matchedRight,
                                      const std::vector<int>& matchedLineIds,
                                      const cv::Mat& Q);

    /// 核心纯函数（线程安全，零分配零异常）：幸存点按序直写调用方定容缓冲。
    /// outXyz 为 3*cap 的 float 数组、outIds 为 cap 的 int 数组；返回幸存点数（≤cap，超cap截断）。
    /// fivechain CPU链2 直写 E 块走此口。
    static int ReconstructInto(const cv::Point2f* left, const cv::Point2f* right,
                               const int* lineIds, int count, const cv::Mat& Q,
                               float minDepth, float maxDepth,
                               float* outXyz, int* outIds, int cap);

    void Warmup(int pointCount);
    void SetParams(const LaserReconstructParams& params);
    const LaserReconstructParams& GetParams() const { return params_; }
    void Destroy();

private:
    LaserReconstructParams params_;
    bool warmedUp_ = false;
};

} // namespace calib
