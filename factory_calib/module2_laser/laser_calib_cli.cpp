// laser_calib_cli.cpp — 模块2 激光标定 CLI（Task 6.2 分阶段实现）
//
// 当前进度: 6.2-d (4-1~4-11 完成; 4-9 endpoint + 4-10 vcp + 4-11 pose_optimize)
//   后续: 6.2-e (5-3 laser_extrinsic_compensate + 4-13 plane_map_temp_table + 写 JSON)
//
// 设计依据: docs/plans/2026-07-18-factory-calib-impl.md Task 6.2 Step 0
// 算子签名以 Step 0.1 速查表为准；原 Step 1 伪代码禁止照抄。

#include "calib_io.h"

#include "mask_extract_cuda.h"
#include "region_analyze_cuda.h"
#include "laser_label_cuda.h"
#include "steger_extract_cuda.h"
#include "undistort_points_cuda.h"
#include "epipolar_interp_cuda.h"
#include "laser_match_cuda.h"
#include "laser_reconstruct_cuda.h"
#include "endpoint_extract_cuda.h"
#include "virtual_camera_pose_cuda.h"
#include "pose_optimize_cuda.h"

#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

using namespace fc;
using namespace calib;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: laser_calib <input_dir> [output_json]\n"
                  << "  input_dir 含 config.json + camera_calib.json + pose_*/L_tube*.png + R_tube*.png\n";
        return 2;
    }
    std::string inDir = argv[1];
    std::string outPath = argc >= 3 ? argv[2] : "laser_calib.json";

    spdlog::info("=== laser_calib (build 6.2-d) ===");

    // ------------------------------------------------------------------
    // 1. 加载输入 + 一致性校验
    // ------------------------------------------------------------------
    auto input = loadLaserInput(inDir);
    if (!input) {
        spdlog::error("load laser input failed");
        return 1;
    }
    const auto& cfg = input->config;
    const auto& h = input->handoff;

    std::string why;
    if (!validateHandoffConsistency(cfg, h, why)) {
        spdlog::error("handoff inconsistent: {}", why);
        return 1;
    }

    spdlog::info("poses={}, imageSize={}x{}, referenceTemp={:.2f}",
                 input->poseFrames.size(),
                 h.imageSize.width, h.imageSize.height,
                 h.referenceTemp);

    // ------------------------------------------------------------------
    // 2. 构造 4-1/4-2/4-3 算子 (L 和 R 各一独立实例)
    //    参数: 当前用默认值或从 cfg 取 deviceId
    //    TODO 6.2-b: 从 config.json 扩展 mask threshold/erodeSize 等可配项
    // ------------------------------------------------------------------
    cv::cuda::Stream stream;

    MaskExtractParams maskParams;
    // maskParams.threshold / erodeSize / ... 用默认值 (Task 6.2-a)
    MaskExtractCUDA maskL(maskParams);
    MaskExtractCUDA maskR(maskParams);
    spdlog::info("4-1 MaskExtractCUDA x2 (L/R) constructed");

    RegionAnalyzerParams cclParams;
    cclParams.deviceId = cfg.deviceId;
    RegionAnalyzerCUDA cclL(cclParams);
    RegionAnalyzerCUDA cclR(cclParams);
    spdlog::info("4-2 RegionAnalyzerCUDA x2 (L/R) constructed");

    LaserLabelParams labelParams;
    labelParams.deviceId = cfg.deviceId;
    LaserLabelerCUDA labelL(labelParams);
    LaserLabelerCUDA labelR(labelParams);
    spdlog::info("4-3 LaserLabelerCUDA x2 (L/R) constructed");

    // ----- 4-4 Steger -----
    // 参数: sigma/threshold 用默认; deviceId 从 cfg
    StegerParams stegerParams;
    stegerParams.deviceId = cfg.deviceId;
    StegerExtractorCUDA stegerL(stegerParams);
    StegerExtractorCUDA stegerR(stegerParams);
    spdlog::info("4-4 StegerExtractorCUDA x2 (L/R) constructed");

    // ----- 4-5 UndistortPoints -----
    // 参数 K/D/R/P 来自 handoff（模块1 输出的内参 + 立体矫正）。
    // L 路: K=K_L, D=D_L, R=R1, P=P1
    // R 路: K=K_R, D=D_R, R=R2, P=P2
    UndistortPointsParams undistL;
    undistL.cameraMatrix = h.cameraMatrixL;
    undistL.distCoeffs   = h.distCoeffsL;
    undistL.R            = h.R1;
    undistL.P            = h.P1;
    undistL.deviceId     = cfg.deviceId;
    undistL.validate();
    UndistortPointsParams undistR;
    undistR.cameraMatrix = h.cameraMatrixR;
    undistR.distCoeffs   = h.distCoeffsR;
    undistR.R            = h.R2;
    undistR.P            = h.P2;
    undistR.deviceId     = cfg.deviceId;
    undistR.validate();
    UndistortPointsCuda undistLOp(undistL);
    UndistortPointsCuda undistROp(undistR);
    spdlog::info("4-5 UndistortPointsCuda x2 (L/R) constructed (R1/P1, R2/P2 from handoff)");

    // ----- 4-6 EpipolarInterp -----
    // lineIdCheck=true 标定模式（按 line_id 同线插值；扫描模式才用 false）
    EpipolarInterpParams epipolarParams;
    epipolarParams.deviceId   = cfg.deviceId;
    epipolarParams.lineIdCheck = true;
    EpipolarInterpCuda epipolarL(epipolarParams);
    EpipolarInterpCuda epipolarR(epipolarParams);
    spdlog::info("4-6 EpipolarInterpCuda x2 (L/R) constructed (lineIdCheck=true)");

    // ----- 4-7 LaserMatch -----
    // 单实例（吃 L+R 两路）。
    LaserMatchParams matchParams;
    matchParams.deviceId = cfg.deviceId;
    LaserMatchCuda matchOp(matchParams);
    spdlog::info("4-7 LaserMatchCuda constructed (single instance, L+R input)");

    // ----- 4-8 LaserReconstruct -----
    // 单实例，Q 矩阵按调用传入（头文件设计如此，避免跨调用累积）。
    LaserReconstructParams reconParams;
    reconParams.minDepth = cfg.depthMin;
    reconParams.maxDepth = cfg.depthMax;
    reconParams.deviceId  = cfg.deviceId;
    LaserReconstructCuda reconOp(reconParams);
    spdlog::info("4-8 LaserReconstructCuda constructed (Q per-call from handoff.Q)");

    // ----- 4-9 EndpointExtract -----
    EndpointExtractParams epParams;
    epParams.deviceId = cfg.deviceId;
    EndpointExtractCuda endpointOp(epParams);
    spdlog::info("4-9 EndpointExtractCuda constructed");

    // ----- 4-10 VirtualCameraPose -----
    VirtualCameraPoseParams vcpParams;
    vcpParams.deviceId = cfg.deviceId;
    VirtualCameraPoseCuda vcpOp(vcpParams);
    spdlog::info("4-10 VirtualCameraPoseCuda constructed");

    // ----- 4-11 PoseOptimize -----
    PoseOptimizeParams poParams;
    poParams.deviceId = cfg.deviceId;
    PoseOptimizeCuda poseOptOp(poParams);
    spdlog::info("4-11 PoseOptimizeCuda constructed");

    // stereoK / stereoR 用 StereoCalibration helper (Step 0 决定 2):
    // stereoK = P1 左上 3×3; stereoR = R (left<-right)
    calib::StereoCalibration sc;
    sc.P = h.P1;  // P1/P2 内参部分一致, stereoRectify 保证
    sc.R = h.R;
    cv::Matx33d stereoK = sc.stereoK();
    cv::Matx33d stereoR = sc.stereoR();
    spdlog::info("stereoK from handoff.P1, stereoR from handoff.R (Step 0 决定 2)");

    // ------------------------------------------------------------------
    // 3. 主循环: pose × tube, 跑 4-1 ~ 4-8 + host 累积
    //    6.2-c: 跑到 reconstruct 并累积 d_points3d 到 host vector
    //    6.2-d/e: 循环结束后用累积结果跑 4-9~4-13
    //    累积策略 (Step 0 决定 1): host 端 vector, 循环末尾统一 upload
    // ------------------------------------------------------------------
    std::vector<cv::Vec3f> host_points3d;
    std::vector<int>       host_line_ids;

    size_t framesOk = 0;
    size_t framesSkip = 0;

    for (size_t pi = 0; pi < input->poseFrames.size(); ++pi) {
        const auto& tubes = input->poseFrames[pi];
        for (size_t ti = 0; ti < tubes.size(); ++ti) {
            const auto& f = tubes[ti];

            // ----- 4-1 mask_extract (L + R) -----
            // Execute(const cv::Mat& gray, Stream&) → MaskExtractResult{d_grayImage, d_cleanedMask, ...}
            auto maskResL = maskL.Execute(f.leftGray, stream);
            auto maskResR = maskR.Execute(f.rightGray, stream);
            if (!maskResL.success || !maskResR.success) {
                spdlog::warn("pose {} tube {}: 4-1 mask failed (L={}, R={}), skip",
                             pi, ti, maskResL.success, maskResR.success);
                ++framesSkip;
                continue;
            }
            if (!maskResL.d_cleanedMask || !maskResR.d_cleanedMask
                || maskResL.d_cleanedMask->empty() || maskResR.d_cleanedMask->empty()) {
                spdlog::warn("pose {} tube {}: 4-1 mask empty, skip", pi, ti);
                ++framesSkip;
                continue;
            }

            // ----- 4-2 region_analyze (L + R) -----
            // Execute(const shared_ptr<GpuMat>& d_mask, Stream&) → RegionAnalysisResult{d_labeledMask CV_32SC1, components}
            auto cclResL = cclL.Execute(maskResL.d_cleanedMask, stream);
            auto cclResR = cclR.Execute(maskResR.d_cleanedMask, stream);
            if (!cclResL.success || !cclResR.success) {
                spdlog::warn("pose {} tube {}: 4-2 ccl failed (L={}, R={}), skip",
                             pi, ti, cclResL.success, cclResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-3 laser_label (L + R) -----
            // Execute(const GpuMat& d_inputMask, Stream&) 输入 CV_32SC1 (来自 4-2)
            // → LaserLabelResult{d_labeledMask (重编号 CV_32SC1)}
            if (!cclResL.d_labeledMask || !cclResR.d_labeledMask) {
                spdlog::warn("pose {} tube {}: 4-2 d_labeledMask null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto labelResL = labelL.Execute(*cclResL.d_labeledMask, stream);
            auto labelResR = labelR.Execute(*cclResR.d_labeledMask, stream);
            if (!labelResL.success || !labelResR.success) {
                spdlog::warn("pose {} tube {}: 4-3 label failed (L={}, R={}), skip",
                             pi, ti, labelResL.success, labelResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-4 steger (L + R) -----
            // Execute(d_gray, d_mask CV_32SC1, stream, GroupMode::ByLabel)
            //   输入: d_grayImage (from 4-1) + d_labeledMask (from 4-3, 重编号)
            //   输出: d_centerPoints (CV_32FC2), d_line_ids (CV_32SC1)
            if (!maskResL.d_grayImage || !maskResR.d_grayImage
                || !labelResL.d_labeledMask || !labelResR.d_labeledMask) {
                spdlog::warn("pose {} tube {}: 4-4 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto stegerResL = stegerL.Execute(*maskResL.d_grayImage,
                                              *labelResL.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            auto stegerResR = stegerR.Execute(*maskResR.d_grayImage,
                                              *labelResR.d_labeledMask,
                                              stream, GroupMode::ByLabel);
            if (!stegerResL.success || !stegerResR.success) {
                spdlog::warn("pose {} tube {}: 4-4 steger failed (L={}, R={}), skip",
                             pi, ti, stegerResL.success, stegerResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-5 undistort (L + R) -----
            // Execute(d_points, d_line_ids, stream)
            //   输入: d_centerPoints + d_line_ids (from 4-4)
            //   输出: d_rectifiedPoints + d_line_ids
            if (!stegerResL.d_centerPoints || !stegerResR.d_centerPoints
                || !stegerResL.d_line_ids    || !stegerResR.d_line_ids) {
                spdlog::warn("pose {} tube {}: 4-5 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto undistResL = undistLOp.Execute(*stegerResL.d_centerPoints,
                                                *stegerResL.d_line_ids, stream);
            auto undistResR = undistROp.Execute(*stegerResR.d_centerPoints,
                                                *stegerResR.d_line_ids, stream);
            if (!undistResL.success || !undistResR.success) {
                spdlog::warn("pose {} tube {}: 4-5 undistort failed (L={}, R={}), skip",
                             pi, ti, undistResL.success, undistResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-6 epipolar_interp (L + R) -----
            // Execute(d_points, d_line_ids, stream)
            //   输入: d_rectifiedPoints + d_line_ids (from 4-5)
            //   输出: d_interpPoints + d_interp_line_ids
            if (!undistResL.d_rectifiedPoints || !undistResR.d_rectifiedPoints
                || !undistResL.d_line_ids      || !undistResR.d_line_ids) {
                spdlog::warn("pose {} tube {}: 4-6 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto epipolarResL = epipolarL.Execute(*undistResL.d_rectifiedPoints,
                                                  *undistResL.d_line_ids, stream);
            auto epipolarResR = epipolarR.Execute(*undistResR.d_rectifiedPoints,
                                                  *undistResR.d_line_ids, stream);
            if (!epipolarResL.success || !epipolarResR.success) {
                spdlog::warn("pose {} tube {}: 4-6 epipolar failed (L={}, R={}), skip",
                             pi, ti, epipolarResL.success, epipolarResR.success);
                ++framesSkip;
                continue;
            }

            // ----- 4-7 laser_match -----
            // Execute(d_left_pts, d_left_ids, d_right_pts, d_right_ids, stream)
            //   输入: L 路和 R 路的 d_interpPoints + d_interp_line_ids (from 4-6)
            //   输出: d_matched_left, d_matched_right, d_matched_line_ids
            if (!epipolarResL.d_interpPoints || !epipolarResR.d_interpPoints
                || !epipolarResL.d_interp_line_ids || !epipolarResR.d_interp_line_ids) {
                spdlog::warn("pose {} tube {}: 4-7 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto matchRes = matchOp.Execute(*epipolarResL.d_interpPoints,
                                            *epipolarResL.d_interp_line_ids,
                                            *epipolarResR.d_interpPoints,
                                            *epipolarResR.d_interp_line_ids,
                                            stream);
            if (!matchRes.success) {
                spdlog::warn("pose {} tube {}: 4-7 match failed ({}), skip",
                             pi, ti, matchRes.message);
                ++framesSkip;
                continue;
            }

            // ----- 4-8 laser_reconstruct -----
            // Execute(d_matched_left, d_matched_right, d_matched_line_ids, Q, stream)
            //   Q = handoff.Q (模块1 输出)
            //   输出: d_points3d (CV_32FC3), d_valid_line_ids (CV_32SC1)
            if (!matchRes.d_matched_left || !matchRes.d_matched_right
                || !matchRes.d_matched_line_ids) {
                spdlog::warn("pose {} tube {}: 4-8 input null, skip", pi, ti);
                ++framesSkip;
                continue;
            }
            auto reconRes = reconOp.Execute(*matchRes.d_matched_left,
                                            *matchRes.d_matched_right,
                                            *matchRes.d_matched_line_ids,
                                            h.Q, stream);
            if (!reconRes.success) {
                spdlog::warn("pose {} tube {}: 4-8 reconstruct failed ({}), skip",
                             pi, ti, reconRes.message);
                ++framesSkip;
                continue;
            }

            // ----- host 累积 (决定 1) -----
            if (reconRes.d_points3d && reconRes.d_valid_line_ids
                && !reconRes.d_points3d->empty()
                && !reconRes.d_valid_line_ids->empty()) {
                cv::Mat h_pts, h_ids;
                reconRes.d_points3d->download(h_pts);
                reconRes.d_valid_line_ids->download(h_ids);
                h_pts = h_pts.reshape(3, 1);   // 强制 1×N CV_32FC3
                h_ids = h_ids.reshape(1, 1);   // 强制 1×N CV_32SC1
                host_points3d.insert(host_points3d.end(),
                                     h_pts.begin<cv::Vec3f>(),
                                     h_pts.end<cv::Vec3f>());
                host_line_ids.insert(host_line_ids.end(),
                                     h_ids.begin<int>(),
                                     h_ids.end<int>());
            }

            ++framesOk;
            spdlog::info("pose {} tube {}: OK (matched={}, reconstructed={}, total_accum={})",
                         pi, ti, matchRes.matchCount, reconRes.validCount,
                         host_points3d.size());
        }
    }

    spdlog::info("loop done: {} ok, {} skipped", framesOk, framesSkip);

    // ------------------------------------------------------------------
    // 3b. 循环结束: 统一 upload 累积的 3D 点（决定 1 后半）
    //     d_all_pts3d, d_all_lids 在 6.2-d 被喂给 4-9/4-11
    // ------------------------------------------------------------------
    cv::cuda::GpuMat d_all_pts3d, d_all_lids;
    if (!host_points3d.empty()) {
        cv::Mat d3d(1, (int)host_points3d.size(), CV_32FC3, host_points3d.data());
        cv::Mat lids(1, (int)host_line_ids.size(), CV_32SC1, host_line_ids.data());
        d_all_pts3d.upload(d3d);
        d_all_lids.upload(lids);
        spdlog::info("accumulated {} 3D points / {} line ids → uploaded",
                     host_points3d.size(), host_line_ids.size());
    } else {
        spdlog::warn("no 3D points accumulated; downstream (4-9+) will be skipped");
    }

    // ------------------------------------------------------------------
    // 3c. 4-9 / 4-10 / 4-11 一次性执行（无 pose×tube 循环）
    //     暂存 finalVirtualK/R/T 供 6.2-e 的 5-3 / 4-13 使用
    // ------------------------------------------------------------------
    cv::Matx33d finalVirtualK = cv::Matx33d::eye();
    cv::Matx33d finalVirtualR = cv::Matx33d::eye();
    cv::Vec3d   finalVirtualT(0, 0, 0);
    bool haveVirtualPose = false;

    if (host_points3d.empty()) {
        spdlog::error("no accumulated 3D points; skip 4-9~4-13");
    } else {
        // ----- 4-9 endpoint_extract -----
        // Execute(d_points3d, d_line_ids, stream) → d_endpoints, d_endpoint_ids, d_line_ids
        auto endpointRes = endpointOp.Execute(d_all_pts3d, d_all_lids, stream);
        if (!endpointRes.success) {
            spdlog::error("4-9 endpoint_extract failed: {}", endpointRes.message);
        } else {
            spdlog::info("4-9 OK: {} endpoints, {} lines",
                         endpointRes.numEndpoints, endpointRes.numLines);

            // ----- 4-10 virtual_camera_pose -----
            // Execute(d_endpoints, d_line_ids, Matx33d& stereoK, Matx33d& stereoR, stream)
            if (!endpointRes.d_endpoints || !endpointRes.d_line_ids) {
                spdlog::error("4-10 input null (endpoint d_endpoints/d_line_ids)");
            } else {
                auto vcpRes = vcpOp.Execute(*endpointRes.d_endpoints,
                                            *endpointRes.d_line_ids,
                                            stereoK, stereoR, stream);
                if (!vcpRes.success) {
                    spdlog::error("4-10 virtual_camera_pose failed: {}", vcpRes.message);
                } else {
                    spdlog::info("4-10 OK: virtualT=({:.3f},{:.3f},{:.3f}), "
                                 "{} lines, fit_err={:.4f}",
                                 vcpRes.virtualT[0], vcpRes.virtualT[1], vcpRes.virtualT[2],
                                 vcpRes.numLines, vcpRes.avgLineFittingError);

                    // ----- 4-11 pose_optimize -----
                    // 决定 4: initialT 直接用 vcpRes.virtualT
                    auto poseRes = poseOptOp.Execute(d_all_pts3d, d_all_lids,
                                                     vcpRes.virtualK, vcpRes.virtualR,
                                                     vcpRes.virtualT, stream);
                    if (!poseRes.success) {
                        spdlog::error("4-11 pose_optimize failed: {}", poseRes.message);
                    } else {
                        spdlog::info("4-11 OK: totalReprojErr={:.4f} "
                                     "(initial={:.4f}), {} line curves",
                                     poseRes.totalReprojectionError,
                                     poseRes.initialReprojectionError,
                                     poseRes.lineCurves.size());
                        finalVirtualK = poseRes.virtualK;
                        finalVirtualR = poseRes.virtualR;
                        finalVirtualT = poseRes.virtualT;
                        haveVirtualPose = true;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. 资源销毁 (算子规范要求析构前显式 Destroy)
    // ------------------------------------------------------------------
    maskL.Destroy();    maskR.Destroy();
    cclL.Destroy();     cclR.Destroy();
    labelL.Destroy();   labelR.Destroy();
    stegerL.Destroy();  stegerR.Destroy();
    undistLOp.Destroy(); undistROp.Destroy();
    epipolarL.Destroy(); epipolarR.Destroy();
    matchOp.Destroy();
    reconOp.Destroy();
    endpointOp.Destroy();
    vcpOp.Destroy();
    poseOptOp.Destroy();

    // ------------------------------------------------------------------
    // 5. 6.2-d 占位输出 (后续 6.2-e 替换为真实 laser_calib.json)
    // ------------------------------------------------------------------
    nlohmann::json j;
    j["schema"]  = "factory_calib.laser_calib.v1";
    j["build"]   = "6.2-d-partial";
    j["posesProcessed"] = input->poseFrames.size();
    j["framesOk"]       = framesOk;
    j["framesSkipped"]  = framesSkip;
    j["accumulatedPoints3D"] = host_points3d.size();
    j["haveVirtualPose"] = haveVirtualPose;
    if (haveVirtualPose) {
        auto matxToArray = [](const cv::Matx33d& m) {
            return std::vector<double>{m(0,0),m(0,1),m(0,2),
                                       m(1,0),m(1,1),m(1,2),
                                       m(2,0),m(2,1),m(2,2)};
        };
        j["virtualK"] = matxToArray(finalVirtualK);
        j["virtualR"] = matxToArray(finalVirtualR);
        j["virtualT"] = std::vector<double>{finalVirtualT[0],
                                            finalVirtualT[1],
                                            finalVirtualT[2]};
    }
    if (!writeJson(outPath, j)) {
        spdlog::error("cannot write output: {}", outPath);
        return 1;
    }
    spdlog::info("laser_calib (6.2-d partial) -> {}", outPath);
    return 0;
}
