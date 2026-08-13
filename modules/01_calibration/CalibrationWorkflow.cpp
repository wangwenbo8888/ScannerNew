// ============================================================================
// CalibrationWorkflow.cpp — 标定工作流实现（接入真实算子）
//
// 流程: 采集N姿态 → 棋盘格检测 → 内参标定 → 外参标定 → 立体矫正 → 温度补偿表
// ============================================================================

#include "CalibrationWorkflow.h"
#include "data/FrameBuffer.h"
#include "data/CalibStore.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

// 标定算子
#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

namespace Scanner::workflow {

CalibrationWorkflow::CalibrationWorkflow(WorkflowContext* ctx) : ctx_(ctx) {}
CalibrationWorkflow::~CalibrationWorkflow() { stop(); }

Result CalibrationWorkflow::initialize() {
    if (!ctx_) return Result::fail("无 WorkflowContext");
    spdlog::info("[CalibWorkflow] 初始化 ({} 姿态, 棋盘 {}x{}, 方格 {}mm)",
                 numPoses_, boardCols_, boardRows_, squareSize_);
    return Result::ok();
}

Result CalibrationWorkflow::start() {
    if (state_ == WorkflowState::Running) return Result::ok();
    state_ = WorkflowState::Running;
    running_ = true;
    calibThread_ = std::thread(&CalibrationWorkflow::calibrationLoop, this);
    return Result::ok();
}

Result CalibrationWorkflow::stop() {
    running_ = false;
    state_ = WorkflowState::Idle;
    if (calibThread_.joinable()) calibThread_.join();
    return Result::ok();
}

Result CalibrationWorkflow::setProgressCallback(WorkflowCallback cb) {
    callback_ = std::move(cb);
    return Result::ok();
}

// ============================================================================
// 标定主循环
// ============================================================================
void CalibrationWorkflow::calibrationLoop() {
    spdlog::info("[CalibWorkflow] 标定流程启动");
    auto t0 = std::chrono::steady_clock::now();

    // === Stage 1: 采集标定图像 ===
    notifyProgress(0, numPoses_, "采集标定图像");
    std::vector<cv::Mat> leftImages, rightImages;

    for (int i = 0; i < numPoses_ && running_; ++i) {
        if (!ctx_->frameBuffer()) break;
        auto frame = ctx_->frameBuffer()->popFrame(std::chrono::milliseconds(5000));
        if (frame && !frame->leftGray.empty() && !frame->rightGray.empty()) {
            leftImages.push_back(frame->leftGray.clone());
            rightImages.push_back(frame->rightGray.clone());
        }
        notifyProgress(i + 1, numPoses_,
                       "采集 " + std::to_string(i + 1) + "/" + std::to_string(numPoses_));
    }

    if (static_cast<int>(leftImages.size()) < 5) {
        result_.success = false;
        result_.message = "标定图像不足 (需≥5, 实际=" + std::to_string(leftImages.size()) + ")";
        state_ = WorkflowState::Error;
        spdlog::error("[CalibWorkflow] {}", result_.message);
        return;
    }

    cv::Size patternSize(boardCols_ - 1, boardRows_ - 1);
    cv::Size imageSize(leftImages[0].cols, leftImages[0].rows);

    // === Stage 2: 棋盘格角点检测 ===
    notifyProgress(0, static_cast<int>(leftImages.size()), "棋盘格角点检测");
    std::vector<std::vector<cv::Point2f>> leftCorners, rightCorners;

    for (size_t i = 0; i < leftImages.size() && running_; ++i) {
        std::vector<cv::Point2f> lc, rc;
        bool foundL = cv::findChessboardCorners(leftImages[i], patternSize, lc,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool foundR = cv::findChessboardCorners(rightImages[i], patternSize, rc,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (foundL && foundR) {
            // 亚像素精化
            cv::cornerSubPix(leftImages[i], lc, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(rightImages[i], rc, cv::Size(5, 5), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));

            leftCorners.push_back(lc);
            rightCorners.push_back(rc);
        }
        notifyProgress(static_cast<int>(i + 1), static_cast<int>(leftImages.size()), "角点检测");
    }

    if (leftCorners.size() < 5) {
        result_.success = false;
        result_.message = "有效角点检测不足 (需≥5, 实际=" + std::to_string(leftCorners.size()) + ")";
        state_ = WorkflowState::Error;
        spdlog::error("[CalibWorkflow] {}", result_.message);
        return;
    }

    spdlog::info("[CalibWorkflow] 有效姿态: {}/{}", leftCorners.size(), leftImages.size());

    // === Stage 3: 内参标定 ===
    notifyProgress(0, 1, "内参标定");
    calib::IntrinsicCalibParams intParams;
    intParams.chessboard_width = boardCols_;
    intParams.chessboard_height = boardRows_;
    intParams.square_size_mm = squareSize_;
    intParams.image_width = imageSize.width;
    intParams.image_height = imageSize.height;

    calib::IntrinsicCalibCPU intrinsicCalib(intParams);
    calib::IntrinsicCalibResult intrinsicResult;
    bool intrinsicOk = intrinsicCalib.Execute(leftCorners, rightCorners, intrinsicResult);

    if (!intrinsicOk || !intrinsicResult.success) {
        result_.success = false;
        result_.message = "内参标定失败: " + intrinsicResult.message;
        state_ = WorkflowState::Error;
        spdlog::error("[CalibWorkflow] {}", result_.message);
        return;
    }

    result_.reprojErrorLeft = intrinsicResult.left.rms_error;
    result_.reprojErrorRight = intrinsicResult.right.rms_error;
    spdlog::info("[CalibWorkflow] 内参标定完成: 左RMS={:.4f} 右RMS={:.4f}",
                 result_.reprojErrorLeft, result_.reprojErrorRight);

    // === Stage 4: 外参标定 ===
    notifyProgress(0, 1, "外参标定");
    calib::ExtrinsicCalibCpuParams extParams;
    extParams.leftPointsPerView = leftCorners;
    extParams.rightPointsPerView = rightCorners;
    extParams.imageSize = imageSize;
    extParams.patternSize = patternSize;
    extParams.squareSize = static_cast<float>(squareSize_);
    extParams.rotateRightImage180 = true;
    extParams.calibrateMono = false;

    // 生成物体坐标点
    for (int r = 0; r < patternSize.height; ++r) {
        for (int c = 0; c < patternSize.width; ++c) {
            extParams.objectPoints.emplace_back(
                c * static_cast<float>(squareSize_),
                r * static_cast<float>(squareSize_), 0.0f);
        }
    }

    calib::ExtrinsicCalibCpu extrinsicCalib(extParams);
    auto extResult = extrinsicCalib.Execute(
        intrinsicResult.left.camera_matrix, intrinsicResult.left.dist_coeffs,
        intrinsicResult.right.camera_matrix, intrinsicResult.right.dist_coeffs);

    if (!extResult.success) {
        result_.success = false;
        result_.message = "外参标定失败: " + extResult.message;
        state_ = WorkflowState::Error;
        spdlog::error("[CalibWorkflow] {}", result_.message);
        return;
    }

    result_.stereoError = extResult.stereoReprojError;
    spdlog::info("[CalibWorkflow] 外参标定完成: 立体RMS={:.4f} 极线误差={:.4f}",
                 extResult.stereoReprojError, extResult.epipolarErrorMean);

    // === Stage 5: 立体矫正 ===
    notifyProgress(0, 1, "立体矫正");
    calib::StereoRectifyCpuParams rectParams;
    rectParams.cameraMatrixL = intrinsicResult.left.camera_matrix;
    rectParams.distCoeffsL = intrinsicResult.left.dist_coeffs;
    rectParams.cameraMatrixR = intrinsicResult.right.camera_matrix;
    rectParams.distCoeffsR = intrinsicResult.right.dist_coeffs;
    rectParams.imageSize = imageSize;
    rectParams.R = extResult.R;
    rectParams.T = extResult.T;
    rectParams.alpha = 0.0;
    rectParams.flags = 1;

    calib::StereoRectifyCpu rectifyCalib(rectParams);
    auto rectResult = rectifyCalib.Execute();

    if (!rectResult.success) {
        result_.success = false;
        result_.message = "立体矫正失败: " + rectResult.message;
        state_ = WorkflowState::Error;
        spdlog::error("[CalibWorkflow] {}", result_.message);
        return;
    }

    spdlog::info("[CalibWorkflow] 立体矫正完成: R1/R2/P1/P2/Q 已生成");

    // === Stage 6: 温度补偿表 ===
    notifyProgress(0, 1, "温度补偿表");
    calib::StereoRectifyTempTableParams tempParams;
    tempParams.cameraMatrixL = intrinsicResult.left.camera_matrix;
    tempParams.distCoeffsL = intrinsicResult.left.dist_coeffs;
    tempParams.cameraMatrixR = intrinsicResult.right.camera_matrix;
    tempParams.distCoeffsR = intrinsicResult.right.dist_coeffs;
    tempParams.imageSize = imageSize;
    tempParams.R = extResult.R;
    tempParams.T = extResult.T;
    tempParams.referenceTemp = 25.0;
    tempParams.cte = 23.6e-6;
    tempParams.tempStep = 0.2;
    tempParams.tempRangeMin = -10.0;
    tempParams.tempRangeMax = 10.0;

    calib::StereoRectifyTempTableCpu tempCalib(tempParams);
    auto tempResult = tempCalib.Execute();

    if (tempResult.success) {
        spdlog::info("[CalibWorkflow] 温度补偿表生成: {} 条目", tempResult.tableSize);
    } else {
        spdlog::warn("[CalibWorkflow] 温度补偿表生成失败 (非致命): {}", tempResult.message);
    }

    // === 完成 ===
    auto t1 = std::chrono::steady_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result_.success = true;
    result_.message = "Calibration complete";
    state_ = WorkflowState::Completed;

    spdlog::info("[CalibWorkflow] === 标定完成 ===");
    spdlog::info("  有效姿态: {}", leftCorners.size());
    spdlog::info("  左RMS: {:.4f}px", result_.reprojErrorLeft);
    spdlog::info("  右RMS: {:.4f}px", result_.reprojErrorRight);
    spdlog::info("  立体RMS: {:.4f}px", result_.stereoError);
    spdlog::info("  耗时: {:.1f}ms", elapsedMs);
    spdlog::info("  温度表: {} 条目", tempResult.success ? tempResult.tableSize : 0);

    // TODO: 保存标定结果到 CalibStore (JSON)
    // intrinsicResult / extResult / rectResult / tempResult → CalibStore
    if (ctx_ && ctx_->calibStore()) {
        auto* store = ctx_->calibStore();
        store->setCameraMatrixL(intrinsicResult.left.camera_matrix);
        store->setCameraMatrixR(intrinsicResult.right.camera_matrix);
        store->setDistCoeffsL(intrinsicResult.left.dist_coeffs);
        store->setDistCoeffsR(intrinsicResult.right.dist_coeffs);
        store->setExtrinsicR(extResult.R);
        store->setExtrinsicT(extResult.T);
        store->setR1(rectResult.R1);
        store->setR2(rectResult.R2);
        store->setP1(rectResult.P1);
        store->setP2(rectResult.P2);
        store->setQ(rectResult.Q);
        store->setImageSize(imageSize);
        store->save("calibration.yml");
        spdlog::info("[CalibWorkflow] 标定参数已写入 CalibStore");
    }
}

void CalibrationWorkflow::notifyProgress(int current, int total, const std::string& stage) {
    spdlog::debug("[CalibWorkflow] {} ({}/{})", stage, current, total);
    if (!callback_) return;
    WorkflowProgress p;
    p.state = state_.load();
    p.currentStage = current;
    p.totalStages = total;
    p.stageName = stage;
    p.progress = total > 0 ? static_cast<float>(current) / total : 0.0f;
    callback_(p);
}

} // namespace Scanner::workflow
