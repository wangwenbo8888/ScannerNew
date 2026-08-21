#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <functional>

namespace calibration {

// ============================================================================
// 固定标定板参数
// ============================================================================
constexpr int CHESSBOARD_COLS = 6;       // 横着6个点
constexpr int CHESSBOARD_ROWS = 7;       // 竖着7个点
constexpr double CHESSBOARD_SQUARE_MM = 15.0;
constexpr int CALIB_FRAME_COUNT = 25;    // 25个姿态

// ============================================================================
// 相机标定
// ============================================================================

struct CameraCalibInput {
    int imageWidth = 2048;
    int imageHeight = 1536;
    // 采集阶段已检测好的角点（每帧一组，至少25帧）
    std::vector<std::vector<cv::Point2f>> leftCorners;
    std::vector<std::vector<cv::Point2f>> rightCorners;
};

struct CameraCalibResult {
    bool success = false;
    std::string message;

    // 内参
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;
    double intrinsicRMS = 0.0;

    // 外参
    cv::Mat R, T, E, F;
    double stereoReprojError = 0.0;
    double epipolarErrorMean = 0.0;

    // 立体矫正
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoiL, validRoiR;

    int validFrameCount = 0;
};

// 完整相机标定流程（内参→外参→矫正）
// 输入：采集阶段已检测的角点
// 输出：CameraCalibResult（内存）
CameraCalibResult runCameraCalibration(const CameraCalibInput& input,
    std::function<void(int, const std::string&)> progress = nullptr);

// ============================================================================
// 激光标定（依赖相机标定结果）
// ============================================================================

struct LaserCalibInput {
    // 采集阶段获取的激光线图像
    cv::Mat leftImage;
    cv::Mat rightImage;
    // 相机标定结果（和相机标定同时采集，共享内外参）
    const CameraCalibResult* cameraCalib = nullptr;
};

struct LaserCalibResult {
    bool success = false;
    std::string message;
    int lineCount = 0;
    int endpointCount = 0;
    // 激光平面映射等结果（具体字段待算子对接补充）
};

// 激光标定流程
LaserCalibResult runLaserCalibration(const LaserCalibInput& input,
    std::function<void(int, const std::string&)> progress = nullptr);

bool saveLaserCalibResult(const std::string& filepath, const LaserCalibResult& result);
bool loadLaserCalibResult(const std::string& filepath, LaserCalibResult& result);

} // namespace calibration
