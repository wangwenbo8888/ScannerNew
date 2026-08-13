#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <functional>

namespace calibration {

struct CameraCalibInput {
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<cv::Mat> leftImages;
    std::vector<cv::Mat> rightImages;
};

struct CameraCalibResult {
    bool success = false;
    double reprojError = 0.0;
    std::string message;
};

class CameraCalibWorkflow {
public:
    void setProgressCallback(std::function<void(int, const std::string&)> cb) { cb_ = cb; }
    CameraCalibResult run(const CameraCalibInput&) {
        return {true, 0.0, "Stub: camera calibration not implemented"};
    }
private:
    std::function<void(int, const std::string&)> cb_;
};

} // namespace calibration
