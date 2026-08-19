// ============================================================================
// CameraChain.cpp — B 相机链编排实现（3-1~3-5 + 5-1/5-2）
// ============================================================================
// 先例：modules/01_calibration/calib_workflow.cpp runCameraCalibration（3-x 调用方式）。
// 签名以算子头为准；差异清单见任务报告（R2/P2 补齐 / plate_temp=20 关板膨胀）。
#include "pipelines/calibcompute/CameraChain.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>      // cv::CALIB_ZERO_DISPARITY

#include "calibration/camera/extrinsic_calib/extrinsic_calib_cpu.h"
#include "calibration/camera/intrinsic_calib/intrinsic_calib_cpu.h"
#include "calibration/camera/inverse_distort/inverse_distort_cpu.h"
#include "calibration/camera/stereo_rectify/stereo_rectify_cpu.h"
#include "calibration/camera/stereo_rectify_temp_table/stereo_rectify_temp_table_cpu.h"
#include "calibration/temp/extrinsic_compensate/extrinsic_compensate_cpu.h"
#include "calibration/temp/intrinsic_compensate/intrinsic_compensate_cpu.h"

namespace Scanner::pipeline {

namespace {

// 规则 Z=0 网格 → 棋盘参数（cols/rows/square）。3-2 算子按棋盘参数自产物方点
// （r 外/c 内序），推导结果须与 boardPoints3D 点数一致方能对齐。
bool deriveBoardGrid(const std::vector<cv::Point3f>& pts, int& cols, int& rows,
                     double& squareMm) {
    if (pts.empty()) return false;
    auto uniqueSorted = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        std::vector<double> u;
        u.reserve(v.size());
        for (double x : v)
            if (u.empty() || x - u.back() > 1e-4) u.push_back(x);
        return u;
    };
    std::vector<double> xs, ys;
    xs.reserve(pts.size());
    ys.reserve(pts.size());
    for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
    const auto ux = uniqueSorted(std::move(xs));
    const auto uy = uniqueSorted(std::move(ys));
    cols = static_cast<int>(ux.size());
    rows = static_cast<int>(uy.size());
    if (cols < 2 || rows < 2) return false;
    if (static_cast<size_t>(cols) * static_cast<size_t>(rows) != pts.size()) return false;
    squareMm = ux[1] - ux[0];
    for (size_t i = 1; i < ux.size(); ++i) squareMm = std::min(squareMm, ux[i] - ux[i - 1]);
    return squareMm > 1e-6;
}

} // namespace

Scanner::Result CameraChain::run(const PostureSessionData& in,
                                 const InitialCalibParams& init,
                                 const std::vector<cv::Point3f>& boardPoints3D,
                                 StereoParams& outStereo,
                                 std::promise<StereoParams>& toLaser,
                                 CalibComputeOutput& out,
                                 const ProgressCb& cb,
                                 const CancelToken& cancel) {
    auto report = [&](int pct, const std::string& stage) { if (cb) cb(pct, stage); };
    auto cancelled = [&]() { return cancel.cancelled(); };
    constexpr int kMinViews = 8;                  // 3-3 minViewCount 默认值

    try {
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled before start");
        report(0, "camera chain start");

        // —— 收集已确认姿态观测 ——
        std::vector<const PostureData*> views;
        views.reserve(PostureSessionData::kTargetCount);
        for (int i = 0; i < PostureSessionData::kTargetCount; ++i)
            if (in.collected[i]) views.push_back(&in.poses[i]);
        if (static_cast<int>(views.size()) < kMinViews)
            return Scanner::Result::fail("collected poses " +
                                         std::to_string(views.size()) + " < " +
                                         std::to_string(kMinViews));
        const size_t nPts = boardPoints3D.size();
        for (const auto* v : views) {
            if (v->ellipseCentersL.size() != nPts || v->ellipseCentersR.size() != nPts)
                return Scanner::Result::fail("pose point count != board points");
        }

        int cols = 0, rows = 0;
        double squareMm = 0.0;
        if (!deriveBoardGrid(boardPoints3D, cols, rows, squareMm))
            return Scanner::Result::fail("boardPoints3D is not a regular Z=0 grid");

        // ===== 3-1 逆畸变+逆矫正（每姿态中心 ×L/R 两实例，init 同组参数）=====
        report(2, "3-1 inverse_distort");
        calib::InverseDistortParams idpL;
        idpL.cameraMatrix = init.K1; idpL.distCoeffs = init.D1;
        idpL.R1 = init.R1;            idpL.P1 = init.P1;
        calib::InverseDistortParams idpR;
        idpR.cameraMatrix = init.K2; idpR.distCoeffs = init.D2;
        idpR.R1 = init.R2;            idpR.P1 = init.P2;   // 右相机矫正组（算子字段名 R1/P1）
        calib::InverseDistortCPU opL(idpL);
        calib::InverseDistortCPU opR(idpR);

        std::vector<std::vector<cv::Point2f>> origL(views.size()), origR(views.size());
        for (size_t k = 0; k < views.size(); ++k) {
            calib::InverseDistortResult rl;
            if (!opL.Execute(views[k]->ellipseCentersL, rl) || !rl.success)
                return Scanner::Result::fail("3-1 L pose " + std::to_string(k) + ": " + rl.message);
            origL[k] = std::move(rl.originalPoints);
            calib::InverseDistortResult rr;
            if (!opR.Execute(views[k]->ellipseCentersR, rr) || !rr.success)
                return Scanner::Result::fail("3-1 R pose " + std::to_string(k) + ": " + rr.message);
            origR[k] = std::move(rr.originalPoints);
        }
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 3-1");

        // ===== 3-2 双目内参标定 =====
        report(8, "3-2 intrinsic_calib");
        calib::IntrinsicCalibParams ip;
        ip.chessboard_width = cols;
        ip.chessboard_height = rows;
        ip.square_size_mm = squareMm;
        ip.image_width = init.imageSize.width;
        ip.image_height = init.imageSize.height;
        ip.plate_temp = 20.0;         // 关板热膨胀：boardPoints3D 已是温度补偿后物方点
        calib::IntrinsicCalibCPU intrinsicOp(ip);
        calib::IntrinsicCalibResult ires;
        if (!intrinsicOp.Execute(origL, origR, ires) || !ires.success)
            return Scanner::Result::fail("3-2: " + ires.message);

        const cv::Mat KL = ires.left.camera_matrix, DL = ires.left.dist_coeffs;
        const cv::Mat KR = ires.right.camera_matrix, DR = ires.right.dist_coeffs;
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 3-2");

        // ===== 旁支 5-1 内参温度补偿表（左相机参考内参）=====
        report(16, "5-1 intrinsic_compensate");
        calib::CameraIntrinsics ci;
        ci.fx = KL.at<double>(0, 0);  ci.fy = KL.at<double>(1, 1);
        ci.cx = KL.at<double>(0, 2);  ci.cy = KL.at<double>(1, 2);
        ci.referenceTemp = 25.0;
        calib::IntrinsicCompensateCPU icOp(calib::IntrinsicCompensateCPUParams{});
        out.intrinsicTempTable = icOp.Execute(ci);
        if (!out.intrinsicTempTable.success)
            return Scanner::Result::fail("5-1: " + out.intrinsicTempTable.message);
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 5-1");

        // ===== 3-3 双目外参标定（CALIB_FIX_INTRINSIC，用 3-2 内参）=====
        report(22, "3-3 extrinsic_calib");
        calib::ExtrinsicCalibCpuParams ep;
        ep.leftPointsPerView = origL;
        ep.rightPointsPerView = origR;
        ep.objectPoints = boardPoints3D;
        ep.imageSize = init.imageSize;
        calib::ExtrinsicCalibCpu extrinsicOp(ep);
        auto eres = extrinsicOp.Execute(KL, DL, KR, DR);
        if (!eres.success)
            return Scanner::Result::fail("3-3: " + eres.message);
        const cv::Mat R = eres.R.clone(), T = eres.T.clone();
        const double reproj = eres.stereoReprojError;
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 3-3");

        // ===== 旁支 5-2 外参温度补偿表 =====
        report(30, "5-2 extrinsic_compensate");
        calib::CameraExtrinsics ce;
        for (int i = 0; i < 9; ++i) ce.R[i] = R.at<double>(i / 3, i % 3);
        for (int i = 0; i < 3; ++i) ce.T[i] = T.at<double>(i);
        ce.referenceTemp = 25.0;
        calib::ExtrinsicCompensateCPU ecOp(calib::ExtrinsicCompensateCPUParams{});
        out.extrinsicTempTable = ecOp.Execute(ce);
        if (!out.extrinsicTempTable.success)
            return Scanner::Result::fail("5-2: " + out.extrinsicTempTable.message);
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 5-2");

        // ===== 3-4 立体矫正 → promise 兑现点 =====
        report(36, "3-4 stereo_rectify");
        calib::StereoRectifyCpuParams rp;
        rp.cameraMatrixL = KL; rp.distCoeffsL = DL;
        rp.cameraMatrixR = KR; rp.distCoeffsR = DR;
        rp.imageSize = init.imageSize;
        rp.R = R; rp.T = T;
        rp.flags = cv::CALIB_ZERO_DISPARITY;   // 算子头默认 flags=1 与自身 validate 冲突，须显式置
        calib::StereoRectifyCpu rectifyOp(rp);
        auto rres = rectifyOp.Execute();
        if (!rres.success)
            return Scanner::Result::fail("3-4: " + rres.message);

        StereoParams sp;
        sp.cameraMatrixL = KL.clone();  sp.distCoeffsL = DL.clone();
        sp.cameraMatrixR = KR.clone();  sp.distCoeffsR = DR.clone();
        sp.R = R;                        sp.T = T;
        sp.R1 = rres.R1.clone();  sp.R2 = rres.R2.clone();
        sp.P1 = rres.P1.clone();  sp.P2 = rres.P2.clone();
        sp.Q = rres.Q.clone();
        sp.reprojError = reproj;
        outStereo = sp;                 // 写 outStereo（此后失败不再回滚——3-4 已定案）
        out.stereo = sp;
        toLaser.set_value(sp);          // ← promise 兑现：3-4 完成即 set，激光链 4-5 起跟随
        if (cancelled())
            return Scanner::Result::fail("camera chain cancelled after 3-4");

        // ===== 旁支 3-5 温度补偿立体矫正参数表 =====
        report(42, "3-5 stereo_rectify_temp_table");
        calib::StereoRectifyTempTableParams tp;
        tp.cameraMatrixL = KL; tp.distCoeffsL = DL;
        tp.cameraMatrixR = KR; tp.distCoeffsR = DR;
        tp.imageSize = init.imageSize;
        tp.R = R; tp.T = T;             // 温度参数取算子默认（25℃/23.6e-6/0.2 步距/±10℃）
        tp.flags = cv::CALIB_ZERO_DISPARITY;   // 同 3-4：默认 flags=1 过不了 validate
        calib::StereoRectifyTempTableCpu tempTableOp(tp);
        out.rectifyTempTable = tempTableOp.Execute();
        if (!out.rectifyTempTable.success)
            return Scanner::Result::fail("3-5: " + out.rectifyTempTable.message);

        report(50, "camera chain done");
        out.quality.ok = true;
        out.quality.summary = "camera chain ok: stereoReproj=" + std::to_string(reproj) +
                              "px, poses=" + std::to_string(views.size());
        return Scanner::Result::ok(out.quality.summary);
    } catch (const std::exception& e) {
        return Scanner::Result::fail(std::string("camera chain exception: ") + e.what());
    } catch (...) {
        return Scanner::Result::fail("camera chain unknown exception");
    }
}

} // namespace Scanner::pipeline
