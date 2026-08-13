#include "calib_workflow.h"

#include "intrinsic_calib_cpu.h"
#include "extrinsic_calib_cpu.h"
#include "stereo_rectify_cpu.h"
#include "stereo_rectify_temp_table_cpu.h"

#include <nlohmann/json.hpp>
#include <fstream>

namespace calibration {

// ============================================================================
// 辅助：生成棋盘格世界坐标
// ============================================================================
static std::vector<cv::Point3f> generateObjectPoints()
{
    std::vector<cv::Point3f> pts;
    for (int r = 0; r < CHESSBOARD_ROWS; ++r)
        for (int c = 0; c < CHESSBOARD_COLS; ++c)
            pts.emplace_back(
                static_cast<float>(c * CHESSBOARD_SQUARE_MM),
                static_cast<float>(r * CHESSBOARD_SQUARE_MM),
                0.0f);
    return pts;
}

// ============================================================================
// 完整相机标定流程（内参→外参→矫正）
// 输入：采集阶段已检测的角点
// ============================================================================
CameraCalibResult runCameraCalibration(
    const CameraCalibInput& input,
    std::function<void(int, const std::string&)> progress)
{
    CameraCalibResult result;
    auto report = [&](int pct, const std::string& step) {
        if (progress) progress(pct, step);
    };

    // 角点数量检查
    int n = std::min(input.leftCorners.size(), input.rightCorners.size());
    if (n < 5) {
        result.message = "valid frames < 5: " + std::to_string(n);
        return result;
    }
    result.validFrameCount = n;

    cv::Size patternSize(CHESSBOARD_COLS, CHESSBOARD_ROWS);
    cv::Size imageSize(input.imageWidth, input.imageHeight);
    auto objectPoints = generateObjectPoints();

    // ===== 1. 内参标定 =====
    report(10, "intrinsic calibration...");

    calib::IntrinsicCalibParams intrinsicParams;
    intrinsicParams.chessboard_width = CHESSBOARD_COLS;
    intrinsicParams.chessboard_height = CHESSBOARD_ROWS;
    intrinsicParams.square_size_mm = CHESSBOARD_SQUARE_MM;
    intrinsicParams.image_width = input.imageWidth;
    intrinsicParams.image_height = input.imageHeight;

    calib::IntrinsicCalibCPU intrinsicCalib(intrinsicParams);
    calib::IntrinsicCalibResult intrinsicResult;

    if (!intrinsicCalib.Execute(input.leftCorners, input.rightCorners, intrinsicResult) ||
        !intrinsicResult.success)
    {
        result.message = "intrinsic failed: " + intrinsicResult.message;
        return result;
    }

    result.cameraMatrixL = intrinsicResult.left.camera_matrix.clone();
    result.distCoeffsL = intrinsicResult.left.dist_coeffs.clone();
    result.cameraMatrixR = intrinsicResult.right.camera_matrix.clone();
    result.distCoeffsR = intrinsicResult.right.dist_coeffs.clone();
    result.intrinsicRMS = intrinsicResult.reproj_error_mean;

    report(40, "intrinsic done, RMS=" + std::to_string(result.intrinsicRMS));

    // ===== 2. 外参标定 =====
    report(45, "extrinsic calibration...");

    calib::ExtrinsicCalibCpuParams extrinsicParams;
    extrinsicParams.leftPointsPerView = input.leftCorners;
    extrinsicParams.rightPointsPerView = input.rightCorners;
    extrinsicParams.objectPoints = objectPoints;
    extrinsicParams.imageSize = imageSize;
    extrinsicParams.patternSize = patternSize;
    extrinsicParams.squareSize = static_cast<float>(CHESSBOARD_SQUARE_MM);

    calib::ExtrinsicCalibCpu extrinsicCalib(extrinsicParams);
    auto extrinsicResult = extrinsicCalib.Execute(
        result.cameraMatrixL, result.distCoeffsL,
        result.cameraMatrixR, result.distCoeffsR);

    if (!extrinsicResult.success) {
        result.message = "extrinsic failed: " + extrinsicResult.message;
        return result;
    }

    result.R = extrinsicResult.R.clone();
    result.T = extrinsicResult.T.clone();
    result.E = extrinsicResult.E.clone();
    result.F = extrinsicResult.F.clone();
    result.stereoReprojError = extrinsicResult.stereoReprojError;
    result.epipolarErrorMean = extrinsicResult.epipolarErrorMean;

    report(70, "extrinsic done");

    // ===== 3. 立体矫正 =====
    report(75, "stereo rectify...");

    calib::StereoRectifyCpuParams rectifyParams;
    rectifyParams.cameraMatrixL = result.cameraMatrixL;
    rectifyParams.distCoeffsL = result.distCoeffsL;
    rectifyParams.cameraMatrixR = result.cameraMatrixR;
    rectifyParams.distCoeffsR = result.distCoeffsR;
    rectifyParams.imageSize = imageSize;
    rectifyParams.R = result.R;
    rectifyParams.T = result.T;

    calib::StereoRectifyCpu rectify(rectifyParams);
    auto rectifyResult = rectify.Execute();

    if (!rectifyResult.success) {
        result.message = "rectify failed: " + rectifyResult.message;
        return result;
    }

    result.R1 = rectifyResult.R1.clone();
    result.R2 = rectifyResult.R2.clone();
    result.P1 = rectifyResult.P1.clone();
    result.P2 = rectifyResult.P2.clone();
    result.Q = rectifyResult.Q.clone();
    result.validRoiL = rectifyResult.validRoiLeft;
    result.validRoiR = rectifyResult.validRoiRight;

    report(100, "calibration done");
    result.success = true;
    return result;
}

// ============================================================================
// 激光标定（依赖相机标定结果）
// ============================================================================
LaserCalibResult runLaserCalibration(const LaserCalibInput& input,
    std::function<void(int, const std::string&)> progress)
{
    LaserCalibResult result;

    if (!input.cameraCalib || !input.cameraCalib->success) {
        result.message = "camera calibration required";
        return result;
    }
    if (input.leftImage.empty() || input.rightImage.empty()) {
        result.message = "laser images empty";
        return result;
    }

    // TODO: 对接激光算子
    // 1. EndpointExtract(input.leftImage, input.rightImage, ...)
    // 2. LaserLabel(...)
    // 3. LaserMatch(...)
    // 4. PlaneMap(...)
    // 5. PoseOptimize(...)
    // 6. VirtualCameraPose(...)
    // 使用 input.cameraCalib->cameraMatrixL/R, distCoeffsL/R, R1/R2/P1/P2/Q

    result.message = "laser calibration: operator integration TODO";
    return result;
}

// ============================================================================
// 保存/加载（JSON）
// ============================================================================
static nlohmann::json matToJson(const cv::Mat& m)
{
    nlohmann::json j;
    j["rows"] = m.rows;
    j["cols"] = m.cols;
    j["type"] = m.type();
    std::vector<double> data;
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c)
            data.push_back(m.at<double>(r, c));
    j["data"] = data;
    return j;
}

static cv::Mat jsonToMat(const nlohmann::json& j)
{
    int rows = j.value("rows", 0);
    int cols = j.value("cols", 0);
    int type = j.value("type", CV_64F);
    cv::Mat m(rows, cols, type);
    const auto& data = j["data"];
    int idx = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<double>(r, c) = data[idx++];
    return m;
}

bool saveCalibResult(const std::string& filepath, const CameraCalibResult& r)
{
    nlohmann::json j;
    j["success"] = r.success;
    j["message"] = r.message;
    j["intrinsicRMS"] = r.intrinsicRMS;
    j["stereoReprojError"] = r.stereoReprojError;
    j["epipolarErrorMean"] = r.epipolarErrorMean;
    j["validFrameCount"] = r.validFrameCount;

    j["cameraMatrixL"] = matToJson(r.cameraMatrixL);
    j["distCoeffsL"] = matToJson(r.distCoeffsL);
    j["cameraMatrixR"] = matToJson(r.cameraMatrixR);
    j["distCoeffsR"] = matToJson(r.distCoeffsR);
    j["R"] = matToJson(r.R);
    j["T"] = matToJson(r.T);
    j["R1"] = matToJson(r.R1);
    j["R2"] = matToJson(r.R2);
    j["P1"] = matToJson(r.P1);
    j["P2"] = matToJson(r.P2);
    j["Q"] = matToJson(r.Q);

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;
    ofs << j.dump(2);
    return true;
}

bool loadCalibResult(const std::string& filepath, CameraCalibResult& r)
{
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) return false;
    nlohmann::json j;
    ifs >> j;

    r.success = j.value("success", false);
    r.message = j.value("message", "");
    r.intrinsicRMS = j.value("intrinsicRMS", 0.0);
    r.stereoReprojError = j.value("stereoReprojError", 0.0);
    r.epipolarErrorMean = j.value("epipolarErrorMean", 0.0);
    r.validFrameCount = j.value("validFrameCount", 0);

    if (j.contains("cameraMatrixL")) r.cameraMatrixL = jsonToMat(j["cameraMatrixL"]);
    if (j.contains("distCoeffsL")) r.distCoeffsL = jsonToMat(j["distCoeffsL"]);
    if (j.contains("cameraMatrixR")) r.cameraMatrixR = jsonToMat(j["cameraMatrixR"]);
    if (j.contains("distCoeffsR")) r.distCoeffsR = jsonToMat(j["distCoeffsR"]);
    if (j.contains("R")) r.R = jsonToMat(j["R"]);
    if (j.contains("T")) r.T = jsonToMat(j["T"]);
    if (j.contains("R1")) r.R1 = jsonToMat(j["R1"]);
    if (j.contains("R2")) r.R2 = jsonToMat(j["R2"]);
    if (j.contains("P1")) r.P1 = jsonToMat(j["P1"]);
    if (j.contains("P2")) r.P2 = jsonToMat(j["P2"]);
    if (j.contains("Q")) r.Q = jsonToMat(j["Q"]);

    return r.success;
}

bool saveLaserCalibResult(const std::string& filepath, const LaserCalibResult& r)
{
    nlohmann::json j;
    j["success"] = r.success;
    j["message"] = r.message;
    j["lineCount"] = r.lineCount;
    j["endpointCount"] = r.endpointCount;
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;
    ofs << j.dump(2);
    return true;
}

bool loadLaserCalibResult(const std::string& filepath, LaserCalibResult& r)
{
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) return false;
    nlohmann::json j;
    ifs >> j;
    r.success = j.value("success", false);
    r.message = j.value("message", "");
    r.lineCount = j.value("lineCount", 0);
    r.endpointCount = j.value("endpointCount", 0);
    return r.success;
}

} // namespace calibration
