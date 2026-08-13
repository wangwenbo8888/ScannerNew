#pragma once

#include <opencv2/core.hpp>

class CameraControl
{
public:
    CameraControl(void*) {}
    ~CameraControl() = default;

    bool isScannerCameraOpen() { return false; }
    bool isScannerSnap() { return false; }

    void open_ScannerCamera() {}
    void close_ScannerCamera() {}
    void start_ScannerAcquisition(int) {}
    void close_ScannerAcquisition() {}
    void GetScannerImages(cv::Mat&, cv::Mat&, int) {}

    void open_TrackerCamera() {}
    void close_TrackerCamera() {}
    void start_TrackerAcquisition(int) {}
    void close_TrackerAcquisition() {}
};
