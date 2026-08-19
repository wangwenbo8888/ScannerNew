#pragma once
// ============================================================================
// LaserChain.h — B 标定计算流水线·激光链（01-⑥，D4 定版 PJC；4-x + 5-3 编排）
// ============================================================================
// in           ：25 组周期帧的激光管帧（PostureSessionData.poses[].cycle.laserFrames，
//                扁平 N*2：偶=L / 奇=R）
// stereoParams ：future 等 StereoParams（相机链 3-4 兑现；4-5 起消费标定后
//                内参/畸变/矫正组/Q）。相机链早期失败 → promise 破诺 → 本链
//                立即 fail 返回（不产出）。
// out          ：激光半区产物（pjc/planeMap/planeMapTempTable/laserExtrinsicTempTable
//                + laserValid；quality 由总装层合成，本链不写）。
//
// 编排（客户端标定流水线.md §1.3）：
//   前段（不依赖相机，全姿态先行）：
//     4-1 mask_extract(L/R) → 4-2 ccl → 4-3 laser_label → 4-4 steger(ByLabel)
//   [前段完成计数——promise 未兑现也可推进至 100%前段]
//   等 future（轮询 100ms 带取消检查；破诺/异常 → 立即 fail）
//   后段（消费 StereoParams）：
//     4-5 undistort_cuda(标定后内参 L:K1/D1/R1/P1 · R:K2/D2/R2/P2)
//     → 4-6 epipolar_interp(lineIdCheck=true) → 4-7 laser_match
//     → 4-8 laser_reconstruct(Q←stereoParams)
//   [编排] 跨姿态聚合 poses(vector<PosePointSet> 按姿态分组——非算子)
//   → PJC projector_joint_calib(f/主点←stereoK(P1 3×3)派生, initialT 机械公差)
//   → 4-12 plane_map(virtualK=f+主点, virtualR=I, virtualT=projectorT,
//                    StereoCalibration(R1/R2/P1/P2/imageSize←3-4))
//   → 4-13 plane_map_temp_table → 5-3 laser_extrinsic_compensate
//          (虚拟→左: R=I/T=projectorT；虚拟→右: R=stereoR, T=stereoR·projectorT+stereoT)
// 进度 50..100（激光半区；相机半区 0..50 归 CameraChain）；取消点同相机链风格
// （算子间；future 等待期轮询取消）。
//
// ⚠ 算子装配（LaserOps）：真实现 makeRealLaserOps() 装配 09 真算子；测试注入
//   假算子组验证编排结构（前段先行/调用序/聚合分组）——两层测试策略（T22 任务书）。
//   4-x 前段算子（mask/ccl/label/steger）均不收内参 → Deps 不携带初始参数组。
#include <future>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"
#include "calibration/laser_calib/plane_map/plane_map_cuda.h"
#include "calibration/laser_calib/plane_map_temp_table/plane_map_temp_table.h"
#include "calibration/laser_calib/projector_joint_calib/projector_joint_calib.h"
#include "calibration/temp/laser_extrinsic_compensate/laser_extrinsic_compensate_cpu.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/posture/PostureTypes.h"

namespace Scanner::pipeline {

// 单幅激光管帧的 Steger 级产物（host 侧；真实现由算子 GPU 结果 download）
struct LaserFramePoints {
    std::vector<cv::Point2f> points;   // 激光中心亚像素点
    std::vector<int> lineIds;          // 逐点线号（4-3 编号）
};

// 单姿态全部激光管帧的前段产物（left/right 各 N 管，下标对应同一管）
struct PoseLaserExtractions {
    std::vector<LaserFramePoints> left, right;
};

// 算子组：编排层只认这组函数签名（依赖注入缝）
struct LaserOps {
    // 4-1→4-2→4-3→4-4（单姿态全部激光管帧 L/R）
    std::function<Scanner::Result(const Scanner::data::CycleUnit& cycle,
                                  PoseLaserExtractions& out)> front;
    // 4-5→4-6→4-7→4-8（单姿态全部管帧对，产出该姿态 3D 点集）
    std::function<Scanner::Result(const PoseLaserExtractions& in,
                                  const StereoParams& sp,
                                  calib::PosePointSet& out)> back;
    // projector_joint_calib（f/主点←stereoK 派生；initialT 机械公差）
    std::function<Scanner::Result(const std::vector<calib::PosePointSet>& poses,
                                  double f, const cv::Point2d& principalPoint,
                                  const cv::Vec3d& initialT,
                                  calib::ProjectorJointCalibResult& out)> pjc;
    // 4-12 plane_map（virtualK=f+主点, virtualR=I, virtualT=projectorT, StereoCalibration）
    std::function<Scanner::Result(const cv::Vec3d& projectorT, double f,
                                  const cv::Point2d& principalPoint,
                                  const StereoParams& sp,
                                  const std::vector<int>& lineIds,
                                  const cv::Size& imageSize,
                                  calib::PlaneMapResult& out)> planeMap;
    // 4-13 plane_map_temp_table（同 4-12 参数 + 温度阶梯默认）
    std::function<Scanner::Result(const cv::Vec3d& projectorT, double f,
                                  const cv::Point2d& principalPoint,
                                  const StereoParams& sp,
                                  const std::vector<int>& lineIds,
                                  const cv::Size& imageSize,
                                  calib::PlaneMapTempTableResult& out)> planeMapTempTable;
    // 5-3 laser_extrinsic_compensate（虚拟→左=PJC；虚拟→右=stereoR/T 推导）
    std::function<Scanner::Result(const cv::Vec3d& projectorT,
                                  const StereoParams& sp,
                                  calib::LaserExtrinsicCompensateCPUResult& out)> laserExtrinsicCompensate;
};

// 真算子装配（09_operatorlib；CUDA 段 JMW_BUILD_CUDA 守卫，无 CUDA 构建返回
// 调用即 fail 的空 ops）。算子逐调用构造（无状态算子，省跨调用生命周期管理）。
LaserOps makeRealLaserOps();

class LaserChain {
public:
    struct Deps {
        cv::Vec3d pjcInitialT{80.0, 3.0, 3.0};   // PJC 初值（机械装配公差，mm，吸引盆小）
        LaserOps ops;                             // 空 → run 时取 makeRealLaserOps()
    };

    explicit LaserChain(const Deps& deps = {});

    // 契约：in 已确认姿态 ≥5（PJC minPoses）且每姿态 laserFrames 为非空偶数张；
    //       stereoParams 由相机链 promise 兑现。
    // 成功：out 激光半区全量写入、laserValid=true（quality 不写——总装层合成）。
    // 失败：fail 返回（含破诺/取消/算子失败；已写入的部分激光产物保留）。
    Scanner::Result run(const PostureSessionData& in,
                        std::future<StereoParams> stereoParams,
                        CalibComputeOutput& out,
                        const ProgressCb& cb,
                        const CancelToken& cancel);

private:
    Deps deps_;
};

} // namespace Scanner::pipeline
