#pragma once
// ============================================================================
// CameraChain.h — B 标定计算流水线·相机链（01-⑥；3-1~3-5 + 5-1/5-2 编排）
// ============================================================================
// in        ：25 姿态椭圆中心（矫正系，PostureSessionData.ellipseCentersL/R）
//             + 标定板物方点（温度补偿后由调用方给）+ 初始参数（与 2-6 同组）
// outStereo ：相机半区立体参数（3-2/3-3/3-4 产出）
// toLaser   ：promise 兑现点（3-4 完成即 set，供激光链 4-5 起跟随；早期失败
//             不 set——future 得 broken_promise，激光链可据此判废）
//
// 编排（客户端标定流水线.md §1.2）：
//   3-1 inverse_distort（每姿态中心 ×L/R 两实例，init 同组参数）
//   → 3-2 intrinsic_calib（L/R）→ 旁支 5-1 intrinsic_compensate（左相机）
//   → 3-3 extrinsic_calib → 旁支 5-2 extrinsic_compensate
//   → 3-4 stereo_rectify → set promise → 旁支 3-5 stereo_rectify_temp_table
// 每算子间查 cancel（cancelled → 安全返回 fail，不写 outStereo）；3-4 定案后、
// 兑现（写 outStereo + set promise）前查最后一次——兑现之后不再查取消：3-5 纯
// CPU 且短，跑完返回 ok 最干净（post-兑现取消不构成失败）。
// 进度 0..50（相机半区；激光半区 50..100 归 T22）。
//
// ⚠ 3-2 算子不从入参收 objectPoints——按棋盘参数自产（含板温膨胀）；编排层
//   从 boardPoints3D 推导 cols/rows/square 并置 plate_temp=20 关闭膨胀，要求
//   boardPoints3D 为规则 Z=0 网格且点序同算子 generateObjectPoints（r 外/c 内）。
#include <future>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/posture/PostureTypes.h"

namespace Scanner::pipeline {

class CameraChain {
public:
    // 契约：in 已确认姿态（collected）须 ≥8（3-3 minViewCount）；
    //       init.K/D/R1/P1/R2/P2 与 2-6 同组；boardPoints3D 规则 Z=0 网格。
    // 成功：outStereo/out 全量写入、promise 已兑现；兑现后取消不构成失败
    //       （照常跑完 3-5 返回 ok）。失败：fail 返回（cancel 路径不写
    //       outStereo；operator 失败路径保留已完成的旁支表）。
    Scanner::Result run(const PostureSessionData& in,
                        const InitialCalibParams& init,
                        const std::vector<cv::Point3f>& boardPoints3D,
                        StereoParams& outStereo,
                        std::promise<StereoParams>& toLaser,
                        CalibComputeOutput& out,
                        const ProgressCb& cb,
                        const CancelToken& cancel);
};

} // namespace Scanner::pipeline
