/**
 * @file laser_match_scan_v3_split_gpu.cpp
 * @brief 激光线匹配扫描 V3 流水线拆分 — GPU 前段门面（组合持有原算子，纯转发）
 */

#include "laser_match_scan_v3_split.h"
#include "common/calib_logging.h"
#include "common/calib_warmup_config.h"
#include <opencv2/core/cuda.hpp>
#include <stdexcept>

namespace calib {

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanSplitGpuV3);

LaserMatchScanSplitGpuV3::LaserMatchScanSplitGpuV3(const LaserMatchScanParamsV3& params)
    : op_(std::make_unique<LaserMatchScanCudaV3>(params))
{
    CALIB_LOG_INFO("LaserMatchScanSplitGpuV3 initialized: deviceId={}, threshold={}",
                   params.deviceId, params.match_threshold);
}

LaserMatchScanSplitGpuV3::~LaserMatchScanSplitGpuV3() = default;

bool LaserMatchScanSplitGpuV3::LoadTempTable(const std::string& jsonPath) {
    return op_ ? op_->LoadTempTable(jsonPath) : false;
}

bool LaserMatchScanSplitGpuV3::SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table) {
    return op_ ? op_->SetTempTable(std::move(table)) : false;
}

void LaserMatchScanSplitGpuV3::SetCurrentTemperature(double temperature) {
    if (op_) op_->SetCurrentTemperature(temperature);
}

void LaserMatchScanSplitGpuV3::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& front)
{
    if (!op_) {
        front = LaserMatchSplitFrontV3{};
        front.success = false;
        front.message = "Execute() after Destroy()";
        return;
    }
    op_->ExecuteFront(d_left_points, d_left_line_ids,
                      d_right_points, d_right_line_ids, stream, front);
    lastBd_ = front.timing;
}

void LaserMatchScanSplitGpuV3::Execute(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    LaserMatchSplitFrontV3& front)
{
    cv::cuda::Stream stream;
    Execute(d_left_points, d_left_line_ids, d_right_points, d_right_line_ids,
            stream, front);
}

void LaserMatchScanSplitGpuV3::ExecuteDeferred(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& front)
{
    if (!op_) {
        front = LaserMatchSplitFrontV3{};
        front.success = false;
        front.message = "ExecuteDeferred() after Destroy()";
        return;
    }
    op_->ExecuteFrontDeferred(d_left_points_cap, d_left_line_ids_cap,
                              d_leftCount, d_right_points_cap,
                              d_right_line_ids_cap, d_rightCount, stream, front);
}

void LaserMatchScanSplitGpuV3::FinishFront(int leftCount, int rightCount,
                                           cv::cuda::Stream& stream,
                                           LaserMatchSplitFrontV3& front)
{
    if (!op_) {
        front = LaserMatchSplitFrontV3{};
        front.success = false;
        front.message = "FinishFront() after Destroy()";
        return;
    }
    op_->FinishFront(leftCount, rightCount, stream, front);
    lastBd_ = front.timing;
}
void LaserMatchScanSplitGpuV3::ExecuteFrontGraph(
    const cv::cuda::GpuMat& d_left_points_cap,
    const cv::cuda::GpuMat& d_left_line_ids_cap,
    const int* d_leftCount,
    const cv::cuda::GpuMat& d_right_points_cap,
    const cv::cuda::GpuMat& d_right_line_ids_cap,
    const int* d_rightCount,
    cv::cuda::Stream& stream,
    LaserMatchSplitFrontV3& front)
{
    if (!op_) {
        front = LaserMatchSplitFrontV3{};
        front.success = false;
        front.message = "ExecuteFrontGraph() after Destroy()";
        return;
    }
    op_->ExecuteFrontGraph(d_left_points_cap, d_left_line_ids_cap,
                           d_leftCount, d_right_points_cap,
                           d_right_line_ids_cap, d_rightCount, stream, front);
}

void LaserMatchScanSplitGpuV3::Destroy() {
    op_.reset();
}

void LaserMatchScanSplitGpuV3::Warmup(int maxLeftPoints, int maxRightPoints) {
    if (op_) op_->Warmup(maxLeftPoints, maxRightPoints);
}

void LaserMatchScanSplitGpuV3::SetParams(const LaserMatchScanParamsV3& params) {
    if (op_) op_->SetParams(params);
}

const LaserMatchScanParamsV3& LaserMatchScanSplitGpuV3::GetParams() const {
    if (!op_)
        throw std::runtime_error("[07-LaserMatchSplitGpuV3] GetParams() after Destroy()");
    return op_->GetParams();
}

void LaserMatchScanSplitGpuV3::SetuRBias(const std::vector<float>& biasPerLid) {
    if (op_) op_->SetuRBias(biasPerLid);
}

void LaserMatchScanSplitGpuV3::SetuRBiasModel(float wx, float wy, float wz, float dtx) {
    if (op_) op_->SetuRBiasModel(wx, wy, wz, dtx);
}

std::string LaserMatchScanSplitGpuV3::LastTimingBreakdown() const {
    return lastBd_;
}

} // namespace calib
