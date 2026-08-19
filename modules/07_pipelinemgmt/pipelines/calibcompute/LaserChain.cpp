// ============================================================================
// LaserChain.cpp — B 激光链编排实现（4-x + 5-3）
// ============================================================================
// 先例：docs/流水线/客户端标定流水线.md §1.3（阶段 4 执行链，D4 定版 PJC）。
// 签名以算子头为准；差异清单见任务报告（4-x 前段无内参入参 / PJC f·主点取
// stereoK=P1 3×3 / plane_map 消费 StereoCalibration.R1/R2/P1/P2/imageSize /
// plane_map_temp_table flags 须 0 或 CALIB_ZERO_DISPARITY 且 lineIds 非空）。
#include "pipelines/calibcompute/LaserChain.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>      // cv::CALIB_ZERO_DISPARITY

#ifdef JMW_BUILD_CUDA
#include <opencv2/core/cuda.hpp>

#include "calibration/laser_calib/laser_label/laser_label_cuda.h"
#include "calibration/laser_calib/laser_match/laser_match_cuda.h"
#include "calibration/laser_calib/plane_map/virtual_pixel_gen.h"
#include "core/laser/epipolar_interp/epipolar_interp_cuda.h"
#include "core/laser/laser_reconstruct/laser_reconstruct_cuda.h"
#include "core/laser/steger/steger_extract_cuda.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "core/vision/ccl/region_analyze_cuda.h"
#include "core/vision/mask_extract/mask_extract_cuda.h"
#endif // JMW_BUILD_CUDA

namespace Scanner::pipeline {

namespace {
constexpr int kMinPosesPjc = 5;    // PJC minPoses 默认值
} // namespace

#ifdef JMW_BUILD_CUDA

namespace {

// —— 4-1→4-2→4-3→4-4 单幅图（L 或 R；左右各持独立算子实例）——
bool realFrontOne(calib::MaskExtractCUDA& maskOp, calib::RegionAnalyzerCUDA& cclOp,
                  calib::LaserLabelerCUDA& labelOp, calib::StegerExtractorCUDA& stegerOp,
                  cv::cuda::Stream& stream, const cv::Mat& gray,
                  LaserFramePoints& fp, std::string& err) {
    auto m = maskOp.Execute(gray, stream);                       // 4-1
    if (!m.success || !m.d_cleanedMask || !m.d_grayImage) { err = "4-1: " + m.message; return false; }
    auto c = cclOp.Execute(m.d_cleanedMask, stream);             // 4-2
    if (!c.success || !c.d_labeledMask) { err = "4-2: " + c.message; return false; }
    auto l = labelOp.Execute(*c.d_labeledMask, stream);          // 4-3
    if (!l.success || !l.d_labeledMask) { err = "4-3: " + l.message; return false; }
    auto s = stegerOp.Execute(*m.d_grayImage, *l.d_labeledMask,  // 4-4 ByLabel
                              stream, calib::GroupMode::ByLabel);
    if (!s.success) { err = "4-4: " + s.message; return false; }
    cv::Mat hPts, hIds;
    if (s.d_centerPoints && s.d_line_ids) {
        s.d_centerPoints->download(hPts, stream);
        s.d_line_ids->download(hIds, stream);
        stream.waitForCompletion();
    }
    if (!hPts.empty())
        fp.points.assign(hPts.begin<cv::Point2f>(), hPts.end<cv::Point2f>());
    if (!hIds.empty())
        fp.lineIds.assign(hIds.begin<int>(), hIds.end<int>());
    return true;
}

} // namespace

LaserOps makeRealLaserOps() {
    LaserOps ops;

    // —— 前段 4-1..4-4（全管帧 L/R；无内参入参——不依赖相机链）——
    ops.front = [](const Scanner::data::CycleUnit& cycle,
                   PoseLaserExtractions& out) -> Scanner::Result {
        try {
            const size_t pairs = cycle.laserFrames.size() / 2;
            out.left.assign(pairs, LaserFramePoints{});
            out.right.assign(pairs, LaserFramePoints{});
            cv::cuda::Stream stream;
            calib::MaskExtractCUDA maskL, maskR;                 // 默认参数（阈值 80 等）
            calib::RegionAnalyzerCUDA cclL, cclR;
            calib::LaserLabelerCUDA labelL, labelR;
            calib::StegerExtractorCUDA stegerL, stegerR;
            std::string err;
            for (size_t k = 0; k < pairs; ++k) {
                if (!realFrontOne(maskL, cclL, labelL, stegerL, stream,
                                  cycle.laserFrames[2 * k], out.left[k], err))
                    return Scanner::Result::fail("tube " + std::to_string(k) + " L " + err);
                if (!realFrontOne(maskR, cclR, labelR, stegerR, stream,
                                  cycle.laserFrames[2 * k + 1], out.right[k], err))
                    return Scanner::Result::fail("tube " + std::to_string(k) + " R " + err);
            }
            return Scanner::Result::ok("front " + std::to_string(pairs) + " tube pairs");
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("front exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("front unknown exception");
        }
    };

    // —— 后段 4-5..4-8（标定后内参/矫正组/Q ← StereoParams）——
    ops.back = [](const PoseLaserExtractions& in, const StereoParams& sp,
                  calib::PosePointSet& out) -> Scanner::Result {
        if (in.left.size() != in.right.size())
            return Scanner::Result::fail("L/R tube count mismatch");
        try {
            const size_t pairs = in.left.size();
            cv::cuda::Stream stream;
            // 4-5 参数：标定后内参（L: K1/D1/R1/P1 · R: K2/D2/R2/P2）
            calib::UndistortPointsParams upL;
            upL.cameraMatrix = sp.cameraMatrixL;  upL.distCoeffs = sp.distCoeffsL;
            upL.R = sp.R1;                        upL.P = sp.P1;
            calib::UndistortPointsParams upR;
            upR.cameraMatrix = sp.cameraMatrixR;  upR.distCoeffs = sp.distCoeffsR;
            upR.R = sp.R2;                        upR.P = sp.P2;
            calib::UndistortPointsCuda undL(upL), undR(upR);
            calib::EpipolarInterpParams eip;      // 4-6：标定模式 lineIdCheck=true
            eip.lineIdCheck = true;
            calib::EpipolarInterpCuda interpOp(eip);
            calib::LaserMatchCuda matchOp;        // 4-7：默认参数
            calib::LaserReconstructCuda reconOp;  // 4-8：Q 按调用传入

            int skipped = 0;
            for (size_t k = 0; k < pairs; ++k) {
                const auto& fl = in.left[k];
                const auto& fr = in.right[k];
                if (fl.points.empty() || fr.points.empty()) { ++skipped; continue; }
                cv::Mat hL(1, static_cast<int>(fl.points.size()), CV_32FC2,
                           const_cast<cv::Point2f*>(fl.points.data()));
                cv::Mat hLIds(1, static_cast<int>(fl.lineIds.size()), CV_32SC1,
                              const_cast<int*>(fl.lineIds.data()));
                cv::Mat hR(1, static_cast<int>(fr.points.size()), CV_32FC2,
                           const_cast<cv::Point2f*>(fr.points.data()));
                cv::Mat hRIds(1, static_cast<int>(fr.lineIds.size()), CV_32SC1,
                              const_cast<int*>(fr.lineIds.data()));
                cv::cuda::GpuMat dL, dLIds, dR, dRIds;
                dL.upload(hL, stream);
                dLIds.upload(hLIds, stream);
                dR.upload(hR, stream);
                dRIds.upload(hRIds, stream);

                auto uL = undL.Execute(dL, dLIds, stream);       // 4-5 L
                if (!uL.success || !uL.d_rectifiedPoints || !uL.d_line_ids)
                    return Scanner::Result::fail("4-5 L: " + uL.message);
                auto iL = interpOp.Execute(*uL.d_rectifiedPoints, *uL.d_line_ids, stream);  // 4-6 L
                if (!iL.success) return Scanner::Result::fail("4-6 L: " + iL.message);
                auto uR = undR.Execute(dR, dRIds, stream);       // 4-5 R
                if (!uR.success || !uR.d_rectifiedPoints || !uR.d_line_ids)
                    return Scanner::Result::fail("4-5 R: " + uR.message);
                auto iR = interpOp.Execute(*uR.d_rectifiedPoints, *uR.d_line_ids, stream);  // 4-6 R
                if (!iR.success) return Scanner::Result::fail("4-6 R: " + iR.message);

                if (iL.interpCount == 0 || iR.interpCount == 0 ||
                    !iL.d_interpPoints || !iR.d_interpPoints ||
                    iL.d_interpPoints->empty() || iR.d_interpPoints->empty()) {
                    ++skipped;      // 空插值产出：跳过该管（不构成失败）
                    continue;
                }
                auto m = matchOp.Execute(*iL.d_interpPoints, *iL.d_interp_line_ids,   // 4-7
                                         *iR.d_interpPoints, *iR.d_interp_line_ids, stream);
                if (!m.success) return Scanner::Result::fail("4-7: " + m.message);
                if (m.matchCount == 0 || !m.d_matched_left || m.d_matched_left->empty()) {
                    ++skipped;
                    continue;
                }
                auto r = reconOp.Execute(*m.d_matched_left, *m.d_matched_right,       // 4-8
                                         *m.d_matched_line_ids, sp.Q, stream);
                if (!r.success) return Scanner::Result::fail("4-8: " + r.message);
                if (r.d_points3d && r.d_valid_line_ids && r.validCount > 0) {
                    cv::Mat hPts, hIds;
                    r.d_points3d->download(hPts, stream);
                    r.d_valid_line_ids->download(hIds, stream);
                    stream.waitForCompletion();
                    if (!hPts.empty())
                        out.points3d.insert(out.points3d.end(), hPts.begin<cv::Vec3f>(),
                                            hPts.end<cv::Vec3f>());
                    if (!hIds.empty())
                        out.lineIds.insert(out.lineIds.end(), hIds.begin<int>(), hIds.end<int>());
                }
            }
            return Scanner::Result::ok("back " + std::to_string(pairs) + " pairs, skipped " +
                                       std::to_string(skipped));
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("back exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("back unknown exception");
        }
    };

    // —— PJC（CPU；f/主点←stereoK 派生，initialT 机械公差）——
    ops.pjc = [](const std::vector<calib::PosePointSet>& poses, double f,
                 const cv::Point2d& principalPoint, const cv::Vec3d& initialT,
                 calib::ProjectorJointCalibResult& out) -> Scanner::Result {
        try {
            calib::ProjectorJointCalib op;    // 默认参数（minPoses=5 / minPointsPerPose=50）
            calib::ProjectorJointCalibInput in;
            in.poses = poses;
            in.f = f;
            in.principalPoint = principalPoint;
            in.initialT = initialT;
            auto r = op.Execute(in);
            if (!r.success) return Scanner::Result::fail(r.message);
            out = std::move(r);
            return Scanner::Result::ok("PJC ok: projectorT=(" +
                                       std::to_string(out.projectorT[0]) + "," +
                                       std::to_string(out.projectorT[1]) + "," +
                                       std::to_string(out.projectorT[2]) + ")");
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("PJC exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("PJC unknown exception");
        }
    };

    // —— 4-12 plane_map（virtualK=f+主点, virtualR=I, virtualT=projectorT）——
    ops.planeMap = [](const cv::Vec3d& projectorT, double f, const cv::Point2d& principalPoint,
                      const StereoParams& sp, const std::vector<int>& lineIds,
                      const cv::Size& imageSize,
                      calib::PlaneMapResult& out) -> Scanner::Result {
        try {
            if (lineIds.empty()) return Scanner::Result::fail("lineIds empty");
            if (imageSize.width <= 0 || imageSize.height <= 0)
                return Scanner::Result::fail("invalid imageSize");
            cv::cuda::Stream stream;
            const cv::Matx33d vK(f, 0, principalPoint.x, 0, f, principalPoint.y, 0, 0, 1);
            const cv::Matx33d vR = cv::Matx33d::eye();   // 投影机与左相机绝对轴线平行（§1.3 修正）
            calib::StereoCalibration sc;                  // ← 3-4（R1/R2/P1/P2/imageSize 被消费）
            sc.cameraMatrix = sp.cameraMatrixL;
            sc.distCoeffs = sp.distCoeffsL;
            sc.R = sp.R;
            sc.Q = sp.Q;
            sc.R1 = sp.R1;  sc.R2 = sp.R2;
            sc.P1 = sp.P1;  sc.P2 = sp.P2;
            sc.imageSize = imageSize;
            calib::VirtualPixelGenerator gen;             // 虚拟像素内部生成（默认 gridStep）
            auto g = gen.Execute(vK, imageSize, lineIds, stream);
            if (!g.success) return Scanner::Result::fail("virtual pixels: " + g.message);
            calib::PlaneMapCuda op;
            auto r = op.Execute(g.d_virtualPixels, vK, vR, projectorT, sc, stream);
            if (!r.success) return Scanner::Result::fail(r.message);
            out = std::move(r);
            return Scanner::Result::ok("plane_map ok: totalPairs=" +
                                       std::to_string(out.totalPairs));
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("plane_map exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("plane_map unknown exception");
        }
    };

    // —— 4-13 plane_map_temp_table（温度/网格参数取算子默认）——
    ops.planeMapTempTable = [](const cv::Vec3d& projectorT, double f,
                               const cv::Point2d& principalPoint, const StereoParams& sp,
                               const std::vector<int>& lineIds, const cv::Size& imageSize,
                               calib::PlaneMapTempTableResult& out) -> Scanner::Result {
        try {
            if (lineIds.empty()) return Scanner::Result::fail("lineIds empty");
            if (imageSize.width <= 0 || imageSize.height <= 0)
                return Scanner::Result::fail("invalid imageSize");
            calib::PlaneMapTempTableParams p;
            p.cameraMatrixL = sp.cameraMatrixL;  p.distCoeffsL = sp.distCoeffsL;
            p.cameraMatrixR = sp.cameraMatrixR;  p.distCoeffsR = sp.distCoeffsR;
            p.imageSize = imageSize;
            p.R = sp.R;  p.T = sp.T;
            p.virtualK = cv::Matx33d(f, 0, principalPoint.x, 0, f, principalPoint.y, 0, 0, 1);
            p.virtualR = cv::Matx33d::eye();
            p.virtualT = projectorT;
            p.lineIds = lineIds;
            p.referenceTemp = 25.0;                    // 其余温度参数取默认（±10℃/0.2/23.6e-6）
            p.flags = cv::CALIB_ZERO_DISPARITY;        // 同 3-4（validate 只收 0 或该值）
            calib::PlaneMapTempTable op(p);
            auto r = op.Execute();
            if (!r.success) return Scanner::Result::fail(r.message);
            out = std::move(r);
            return Scanner::Result::ok("plane_map_temp_table ok: tableSize=" +
                                       std::to_string(out.tableSize));
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("plane_map_temp_table exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("plane_map_temp_table unknown exception");
        }
    };

    // —— 5-3 laser_extrinsic_compensate（虚拟→左=PJC；虚拟→右=stereoR/T 推导）——
    ops.laserExtrinsicCompensate = [](const cv::Vec3d& projectorT, const StereoParams& sp,
                                      calib::LaserExtrinsicCompensateCPUResult& out)
                                      -> Scanner::Result {
        try {
            if (sp.R.empty() || sp.T.empty() || sp.R.rows != 3 || sp.R.cols != 3 ||
                sp.T.rows != 3 || sp.T.cols != 1)
                return Scanner::Result::fail("stereo R/T invalid");
            calib::CameraExtrinsics v2l;               // 虚拟→左：R=I, T=projectorT（R=I 修正）
            for (int i = 0; i < 9; ++i) v2l.R[i] = (i % 4 == 0) ? 1.0 : 0.0;
            for (int i = 0; i < 3; ++i) v2l.T[i] = projectorT[i];
            v2l.referenceTemp = 25.0;
            calib::CameraExtrinsics v2r;               // 虚拟→右：p_R = R·p_v + (R·projectorT + T)
            for (int i = 0; i < 9; ++i) v2r.R[i] = sp.R.at<double>(i / 3, i % 3);
            for (int i = 0; i < 3; ++i) {
                double t = sp.T.at<double>(i);
                for (int j = 0; j < 3; ++j) t += sp.R.at<double>(i, j) * projectorT[j];
                v2r.T[i] = t;
            }
            v2r.referenceTemp = 25.0;
            calib::LaserExtrinsicCompensateCPU op;     // 默认参数（±10℃/0.2/23.6e-6）
            auto r = op.Execute(v2l, v2r);
            if (!r.success) return Scanner::Result::fail(r.message);
            out = std::move(r);
            return Scanner::Result::ok("laser_extrinsic_compensate ok");
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("5-3 exception: ") + e.what());
        } catch (...) {
            return Scanner::Result::fail("5-3 unknown exception");
        }
    };

    return ops;
}

#else // !JMW_BUILD_CUDA

LaserOps makeRealLaserOps() {
    return LaserOps{};   // 无 CUDA 构建：空 ops（LaserChain::run 报 "ops unavailable"）
}

#endif // JMW_BUILD_CUDA

// ============================================================================
// LaserChain::run — 编排（前段先行 → promise 跟随 → 后段 → 聚合 → PJC → 4-12/4-13/5-3）
// ============================================================================
LaserChain::LaserChain(const Deps& deps) : deps_(deps) {}

Scanner::Result LaserChain::run(const PostureSessionData& in,
                                std::future<StereoParams> stereoParams,
                                CalibComputeOutput& out,
                                const ProgressCb& cb,
                                const CancelToken& cancel) {
    auto report = [&](int pct, const std::string& stage) { if (cb) cb(pct, stage); };
    auto cancelled = [&]() { return cancel.cancelled(); };

    try {
        if (cancelled())
            return Scanner::Result::fail("laser chain cancelled before start");

        LaserOps ops = deps_.ops;   // 测试注入优先；缺失则真算子
        if (!ops.front || !ops.back || !ops.pjc || !ops.planeMap ||
            !ops.planeMapTempTable || !ops.laserExtrinsicCompensate)
            ops = makeRealLaserOps();
        if (!ops.front || !ops.back || !ops.pjc || !ops.planeMap ||
            !ops.planeMapTempTable || !ops.laserExtrinsicCompensate)
            return Scanner::Result::fail("laser ops unavailable (built without CUDA?)");

        report(50, "laser chain start");

        // —— 收集已确认姿态 + 帧前置检查 ——
        std::vector<const PostureData*> views;
        views.reserve(PostureSessionData::kTargetCount);
        for (int i = 0; i < PostureSessionData::kTargetCount; ++i)
            if (in.collected[i]) views.push_back(&in.poses[i]);
        if (static_cast<int>(views.size()) < kMinPosesPjc)
            return Scanner::Result::fail("collected poses " +
                                         std::to_string(views.size()) + " < " +
                                         std::to_string(kMinPosesPjc) + " (PJC minPoses)");
        cv::Size imageSize(0, 0);
        for (const auto* v : views) {
            const auto& fr = v->cycle.laserFrames;
            if (fr.empty() || fr.size() % 2 != 0)
                return Scanner::Result::fail("pose laserFrames must be non-empty even count");
            if (imageSize.empty()) imageSize = fr.front().size();
        }
        const int n = static_cast<int>(views.size());

        // ===== 前段 4-1..4-4（全姿态先行——不依赖相机链）=====
        std::vector<PoseLaserExtractions> fronts(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            auto r = ops.front(views[i]->cycle, fronts[static_cast<size_t>(i)]);
            if (!r.success)
                return Scanner::Result::fail("laser front pose " + std::to_string(i) +
                                             ": " + r.message);
            if (fronts[static_cast<size_t>(i)].left.size() !=
                fronts[static_cast<size_t>(i)].right.size())
                return Scanner::Result::fail("laser front pose " + std::to_string(i) +
                                             ": L/R tube count mismatch");
            if (cancelled())
                return Scanner::Result::fail("laser chain cancelled in front (pose " +
                                             std::to_string(i) + ")");
            report(50 + static_cast<int>(20.0 * (i + 1) / n),
                   "4-1..4-4 front pose " + std::to_string(i + 1) + "/" + std::to_string(n));
        }
        report(70, "laser front done (" + std::to_string(n) + " poses)");

        // ===== 等 StereoParams（3-4 promise；轮询带取消检查）=====
        StereoParams sp;
        try {
            while (true) {
                if (cancelled())
                    return Scanner::Result::fail("laser chain cancelled while waiting stereo");
                if (stereoParams.wait_for(std::chrono::milliseconds(100)) ==
                    std::future_status::ready)
                    break;
            }
            sp = stereoParams.get();   // 相机链早期失败 → 破诺 → 上方 wait 即抛
        } catch (const std::future_error& e) {
            return Scanner::Result::fail(std::string(
                "laser chain: stereo params promise broken (camera chain failed early): ") +
                e.what());
        } catch (const std::exception& e) {
            return Scanner::Result::fail(std::string("laser chain: stereo future error: ") +
                                         e.what());
        } catch (...) {
            return Scanner::Result::fail("laser chain: stereo future unknown error");
        }
        if (cancelled())
            return Scanner::Result::fail("laser chain cancelled after stereo ready");

        // ===== 后段 4-5..4-8（逐姿态）=====
        std::vector<calib::PosePointSet> poses(static_cast<size_t>(n));   // 按姿态分组（非算子聚合）
        std::set<int> lineSet;
        for (int i = 0; i < n; ++i) {
            auto r = ops.back(fronts[static_cast<size_t>(i)], sp,
                              poses[static_cast<size_t>(i)]);
            if (!r.success)
                return Scanner::Result::fail("laser back pose " + std::to_string(i) +
                                             ": " + r.message);
            for (int id : poses[static_cast<size_t>(i)].lineIds) lineSet.insert(id);
            if (cancelled())
                return Scanner::Result::fail("laser chain cancelled in back (pose " +
                                             std::to_string(i) + ")");
            report(70 + static_cast<int>(20.0 * (i + 1) / n),
                   "4-5..4-8 back pose " + std::to_string(i + 1) + "/" + std::to_string(n));
        }
        if (lineSet.empty())
            return Scanner::Result::fail("no laser lines reconstructed (all poses empty)");
        const std::vector<int> allLineIds(lineSet.begin(), lineSet.end());

        // ===== PJC（f/主点 ← stereoK(P1 3×3) 派生）=====
        report(91, "PJC projector_joint_calib");
        const double f = sp.P1.at<double>(0, 0);
        const cv::Point2d principalPoint(sp.P1.at<double>(0, 2), sp.P1.at<double>(1, 2));
        auto rp = ops.pjc(poses, f, principalPoint, deps_.pjcInitialT, out.pjc);
        if (!rp.success) return Scanner::Result::fail("PJC: " + rp.message);
        if (cancelled()) return Scanner::Result::fail("laser chain cancelled after PJC");

        // ===== 4-12 plane_map =====
        report(93, "4-12 plane_map");
        auto rm = ops.planeMap(out.pjc.projectorT, f, principalPoint, sp, allLineIds,
                               imageSize, out.planeMap);
        if (!rm.success) return Scanner::Result::fail("4-12: " + rm.message);
        if (cancelled()) return Scanner::Result::fail("laser chain cancelled after 4-12");

        // ===== 4-13 plane_map_temp_table =====
        report(96, "4-13 plane_map_temp_table");
        auto rt = ops.planeMapTempTable(out.pjc.projectorT, f, principalPoint, sp,
                                        allLineIds, imageSize, out.planeMapTempTable);
        if (!rt.success) return Scanner::Result::fail("4-13: " + rt.message);
        if (cancelled()) return Scanner::Result::fail("laser chain cancelled after 4-13");

        // ===== 5-3 laser_extrinsic_compensate =====
        report(98, "5-3 laser_extrinsic_compensate");
        auto re = ops.laserExtrinsicCompensate(out.pjc.projectorT, sp,
                                               out.laserExtrinsicTempTable);
        if (!re.success) return Scanner::Result::fail("5-3: " + re.message);

        out.laserValid = true;   // quality 由总装层合成（本链不写——两线程写不同成员）
        report(100, "laser chain done");
        return Scanner::Result::ok("laser chain ok: poses=" + std::to_string(n) +
                                   ", lines=" + std::to_string(allLineIds.size()));
    } catch (const std::exception& e) {
        return Scanner::Result::fail(std::string("laser chain exception: ") + e.what());
    } catch (...) {
        return Scanner::Result::fail("laser chain unknown exception");
    }
}

} // namespace Scanner::pipeline
