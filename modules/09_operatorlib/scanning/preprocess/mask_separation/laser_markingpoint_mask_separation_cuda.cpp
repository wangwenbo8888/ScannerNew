/**
 * @file laser_markingpoint_mask_separation_cuda.cpp
 * @brief 激光线与标记点掩膜分离算子 - 桥接实现（构造/析构/warmup/setParams）
 *
 * 本文件实现 pImpl 模式的桥接函数，不包含 CUDA 代码。
 */

#include "laser_markingpoint_mask_separation_cuda.h"
#include "laser_markingpoint_mask_separation_cuda_pimpl.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

using namespace calib;

OperatorInfo getLaserMarkingSeparationCUDAInfo() {
    return OperatorInfo{"LaserMarkingSeparationCUDA", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR, OperatorType::CUDA};
}

CALIB_DEFINE_LOG_TAG(LaserMarkingSep, LaserMarkingSeparationCUDA);

LaserMarkingSeparationCUDA::LaserMarkingSeparationCUDA(
    const LaserMarkingSeparationParams& params)
    : pImpl_(std::make_unique<Impl>(params))
{
    CALIB_LOG_INFO("LaserMarkingSeparationCUDA initialized: gaussianSize={}, threshold={}, "
                   "step2_erode={}, step3_erode={}, step4_dilate={}, step6_dilate={}",
                   params.gaussianSize, params.threshold,
                   params.step2_erodeSize, params.step3_erodeSize,
                   params.step4_dilateSize, params.step6_dilateSize);
}

LaserMarkingSeparationCUDA::~LaserMarkingSeparationCUDA() = default;

LaserMarkingSeparationResult LaserMarkingSeparationCUDA::Execute(
    const cv::Mat& grayImage, cv::cuda::Stream& stream)
{
    CALIB_LOG_DEBUG("Execute() called: size={}x{}, type={}",
                    grayImage.cols, grayImage.rows, grayImage.type());

    if (grayImage.empty()) {
        CALIB_LOG_ERROR("Execute() failed: input image is empty");
        LaserMarkingSeparationResult result;
        result.success = false;
        result.message = "Input image is empty";
        return result;
    }

    if (grayImage.type() != CV_8UC1) {
        CALIB_LOG_ERROR("Execute() failed: input must be CV_8UC1, got type {}",
                        grayImage.type());
        LaserMarkingSeparationResult result;
        result.success = false;
        result.message = "Input must be CV_8UC1 grayscale image";
        return result;
    }

    return pImpl_->separate(grayImage, stream);
}

LaserMarkingSeparationResult LaserMarkingSeparationCUDA::Execute(
    const cv::Mat& grayImage)
{
    static thread_local cv::cuda::Stream defaultStream;
    return Execute(grayImage, defaultStream);
}

void LaserMarkingSeparationCUDA::Destroy() {
    pImpl_.reset();
}

void LaserMarkingSeparationCUDA::Warmup(int rows, int cols) {
    CALIB_LOG_INFO("Warmup() called: rows={}, cols={}", rows, cols);
    pImpl_->warmup(rows, cols);
    CALIB_LOG_INFO("Warmup() completed");
}

void LaserMarkingSeparationCUDA::Warmup(const calib::WarmupConfig& config) {
    CALIB_LOG_INFO("Warmup(WarmupConfig) called: rows={}, cols={}",
                   config.rows, config.cols);
    Warmup(config.rows, config.cols);
}

void LaserMarkingSeparationCUDA::SetParams(
    const LaserMarkingSeparationParams& params)
{
    CALIB_LOG_INFO("SetParams() called: threshold={}, step2_erode={}, step3_erode={}",
                   params.threshold, params.step2_erodeSize, params.step3_erodeSize);
    pImpl_->setParams(params);
}

const LaserMarkingSeparationParams& LaserMarkingSeparationCUDA::GetParams() const {
    return pImpl_->getParams();
}
