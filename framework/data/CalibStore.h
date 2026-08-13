#pragma once
// ============================================================================
// CalibStore.h — 标定参数存储（DataStore）
//
// 持久化标定结果：内参/外参/立体矫正/温度补偿表。
// JSON 格式存储，供 CalibrationWorkflow 写入、ScanWorkflow 读取。
// ============================================================================

#include "common/types.h"
#include <opencv2/core.hpp>
#include <string>
#include <mutex>

namespace Scanner::data {

class CalibStore {
public:
    CalibStore();

    // 保存/加载（JSON + cv::FileStorage）
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    // 设置标定参数
    void setCameraMatrixL(const cv::Mat& m) { std::lock_guard l(mtx_); cameraMatrixL_ = m; hasData_ = true; }
    void setCameraMatrixR(const cv::Mat& m) { std::lock_guard l(mtx_); cameraMatrixR_ = m; }
    void setDistCoeffsL(const cv::Mat& m) { std::lock_guard l(mtx_); distCoeffsL_ = m; }
    void setDistCoeffsR(const cv::Mat& m) { std::lock_guard l(mtx_); distCoeffsR_ = m; }
    void setR1(const cv::Mat& m) { std::lock_guard l(mtx_); R1_ = m; }
    void setR2(const cv::Mat& m) { std::lock_guard l(mtx_); R2_ = m; }
    void setP1(const cv::Mat& m) { std::lock_guard l(mtx_); P1_ = m; }
    void setP2(const cv::Mat& m) { std::lock_guard l(mtx_); P2_ = m; }
    void setQ(const cv::Mat& m) { std::lock_guard l(mtx_); Q_ = m; }
    void setExtrinsicR(const cv::Mat& m) { std::lock_guard l(mtx_); extR_ = m; }
    void setExtrinsicT(const cv::Mat& m) { std::lock_guard l(mtx_); extT_ = m; }
    void setImageSize(const cv::Size& s) { std::lock_guard l(mtx_); imageSize_ = s; }

    // 获取标定参数
    cv::Mat cameraMatrixL() const { std::lock_guard l(mtx_); return cameraMatrixL_; }
    cv::Mat cameraMatrixR() const { std::lock_guard l(mtx_); return cameraMatrixR_; }
    cv::Mat distCoeffsL() const { std::lock_guard l(mtx_); return distCoeffsL_; }
    cv::Mat distCoeffsR() const { std::lock_guard l(mtx_); return distCoeffsR_; }
    cv::Mat R1() const { std::lock_guard l(mtx_); return R1_; }
    cv::Mat R2() const { std::lock_guard l(mtx_); return R2_; }
    cv::Mat P1() const { std::lock_guard l(mtx_); return P1_; }
    cv::Mat P2() const { std::lock_guard l(mtx_); return P2_; }
    cv::Mat Q() const { std::lock_guard l(mtx_); return Q_; }
    cv::Size imageSize() const { std::lock_guard l(mtx_); return imageSize_; }

    bool hasData() const { std::lock_guard l(mtx_); return hasData_; }
    void clear() { std::lock_guard l(mtx_); hasData_ = false; }

private:
    mutable std::mutex mtx_;
    bool hasData_ = false;

    cv::Mat cameraMatrixL_, cameraMatrixR_;
    cv::Mat distCoeffsL_, distCoeffsR_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat extR_, extT_;
    cv::Size imageSize_;
};

} // namespace Scanner::data
