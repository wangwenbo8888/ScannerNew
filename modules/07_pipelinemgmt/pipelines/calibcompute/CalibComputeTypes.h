#pragma once
// ============================================================================
// CalibComputeTypes.h — B 标定计算流水线自有类型（纯数据 + 轻量取消令牌）
// ============================================================================
// InitialCalibParams ：2-6 与 3-1 严格同组的初始参数（装配层单源注入）
// StereoParams       ：3-2/3-3/3-4 产出（激光链等待的 promise 载荷）
// CalibComputeOutput ：B 输出（→06 仓库落盘）：立体参数 + 5-1/5-2/3-5 三表
// ProgressCb/CancelToken：进度回调 / 原子取消
//
// ⚠ 三温度表形态对齐算子 Result 类型（move-only）→ CalibComputeOutput move-only。
#include <atomic>
#include <functional>
#include <string>

#include <opencv2/core.hpp>

#include "calibration/camera/stereo_rectify_temp_table/stereo_rectify_temp_table_cpu.h"
#include "calibration/laser_calib/plane_map/plane_map_cuda.h"
#include "calibration/laser_calib/plane_map_temp_table/plane_map_temp_table.h"
#include "calibration/laser_calib/projector_joint_calib/projector_joint_calib.h"
#include "calibration/temp/extrinsic_compensate/extrinsic_compensate_cpu.h"
#include "calibration/temp/intrinsic_compensate/intrinsic_compensate_cpu.h"
#include "calibration/temp/laser_extrinsic_compensate/laser_extrinsic_compensate_cpu.h"

namespace Scanner::pipeline {

struct InitialCalibParams {        // 2-6 与 3-1 严格同组（装配层单源注入）
    cv::Mat K1, D1, K2, D2;        // 初始内参/畸变（L/R）
    cv::Mat R1, P1;                // 初始矫正（左，供 3-1 逆变换）
    cv::Mat R2, P2;                // 初始矫正（右）——3-1 右相机实例所需（算子单相机设计）
    cv::Size imageSize;
};

struct StereoParams {              // 3-2/3-3/3-4 产出（激光链等待的 promise 载荷）
    cv::Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;
    cv::Mat R, T;                  // 双目外参
    cv::Mat R1, R2, P1, P2, Q;
    double reprojError = 0.0;
};

struct CalibComputeOutput {        // B 输出（→06 仓库落盘）
    StereoParams stereo;
    calib::IntrinsicCompensateCPUResult intrinsicTempTable;   // 5-1 内参补偿表
    calib::ExtrinsicCompensateCPUResult extrinsicTempTable;   // 5-2 外参补偿表
    calib::StereoRectifyTempTableResult rectifyTempTable;     // 3-5 矫正温度表
    bool laserValid = false;       // 激光半区有效标志（T22 填）
    calib::ProjectorJointCalibResult pjc;                      // PJC 光心+发射曲线（T22）
    calib::PlaneMapResult planeMap;                            // 4-12 激光面映射表（T22）
    calib::PlaneMapTempTableResult planeMapTempTable;          // 4-13 温度补偿映射表（T22）
    calib::LaserExtrinsicCompensateCPUResult laserExtrinsicTempTable; // 5-3 激光外参补偿表（T22）
    struct QualityReport {         // T23 细化
        bool ok = false;
        std::string summary;
    } quality;

    CalibComputeOutput() = default;
    CalibComputeOutput(CalibComputeOutput&&) = default;        // 三表 move-only → 整体 move-only
    CalibComputeOutput& operator=(CalibComputeOutput&&) = default;
    CalibComputeOutput(const CalibComputeOutput&) = delete;
    CalibComputeOutput& operator=(const CalibComputeOutput&) = delete;
};

using ProgressCb = std::function<void(int percent, const std::string& stage)>;

class CancelToken {                // 简单原子取消（不可拷贝/移动：atomic 成员）
public:
    void cancel() { v_.store(true, std::memory_order_relaxed); }
    bool cancelled() const { return v_.load(std::memory_order_relaxed); }
private:
    std::atomic<bool> v_{false};
};

} // namespace Scanner::pipeline
