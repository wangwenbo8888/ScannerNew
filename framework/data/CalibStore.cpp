// ============================================================================
// CalibStore.cpp — 标定参数存储实现
// ============================================================================

#include "CalibStore.h"
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>
#include <fstream>

namespace Scanner::data {

CalibStore::CalibStore() {}

bool CalibStore::save(const std::string& path) const {
    std::lock_guard l(mtx_);
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        spdlog::error("[CalibStore] 无法写入: {}", path);
        return false;
    }

    fs << "cameraMatrixL" << cameraMatrixL_;
    fs << "cameraMatrixR" << cameraMatrixR_;
    fs << "distCoeffsL" << distCoeffsL_;
    fs << "distCoeffsR" << distCoeffsR_;
    fs << "R1" << R1_;
    fs << "R2" << R2_;
    fs << "P1" << P1_;
    fs << "P2" << P2_;
    fs << "Q" << Q_;
    fs << "extR" << extR_;
    fs << "extT" << extT_;
    fs << "imageWidth" << imageSize_.width;
    fs << "imageHeight" << imageSize_.height;
    fs.release();

    spdlog::info("[CalibStore] 标定参数已保存: {}", path);
    return true;
}

bool CalibStore::load(const std::string& path) {
    std::lock_guard l(mtx_);
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        spdlog::error("[CalibStore] 无法读取: {}", path);
        return false;
    }

    fs["cameraMatrixL"] >> cameraMatrixL_;
    fs["cameraMatrixR"] >> cameraMatrixR_;
    fs["distCoeffsL"] >> distCoeffsL_;
    fs["distCoeffsR"] >> distCoeffsR_;
    fs["R1"] >> R1_;
    fs["R2"] >> R2_;
    fs["P1"] >> P1_;
    fs["P2"] >> P2_;
    fs["Q"] >> Q_;
    fs["extR"] >> extR_;
    fs["extT"] >> extT_;

    int w = 0, h = 0;
    fs["imageWidth"] >> w;
    fs["imageHeight"] >> h;
    imageSize_ = cv::Size(w, h);
    fs.release();

    hasData_ = !cameraMatrixL_.empty();
    spdlog::info("[CalibStore] 标定参数已加载: {} ({}x{})", path, w, h);
    return hasData_;
}

} // namespace Scanner::data
