#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <functional>

namespace calibration {

struct ScanInput {
    // scan parameters
};

struct ScanResult {
    bool success = false;
    int fusedPoints = 0;
    double totalTimeMs = 0.0;
    std::string message;
};

class ScanWorkflow {
public:
    void setProgressCallback(std::function<void(int, const std::string&)> cb) { callback_ = cb; }
    ScanResult processFrameToCloud(const ScanInput&,
                                   std::vector<cv::Point3f>& cloud,
                                   std::vector<cv::Point3f>& normals) {
        return {true, 0, 0.0, "Stub: scan workflow not implemented"};
    }
private:
    std::function<void(int, const std::string&)> callback_;
};

} // namespace calibration
