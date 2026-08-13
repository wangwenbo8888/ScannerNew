#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <functional>

namespace calibration {

struct LaserCalibInput {
    cv::Mat leftImage;
    cv::Mat rightImage;
};

struct LaserCalibResult {
    bool success = false;
    int lineCount = 0;
    int totalEndpoints = 0;
    std::string message;
};

class LaserCalibWorkflow {
public:
    void setProgressCallback(std::function<void(int, const std::string&)> cb) { cb_ = cb; }
    LaserCalibResult run(const LaserCalibInput&) {
        return {true, 0, 0, "Stub: laser calibration not implemented"};
    }
private:
    std::function<void(int, const std::string&)> cb_;
};

} // namespace calibration
