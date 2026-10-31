/**
 * @file undistort_points_cuda_impl.cu
 * @brief 激光中心亚像素点集去畸变+立体矫正算子 CUDA 实现（struct Impl 方法 + GPU Kernel）
 *
 * 算法步骤：
 *   Step 1: 像素坐标 -> 归一化坐标
 *   Step 2: 迭代反向去畸变 (8参数模型: k1,k2,p1,p2,k3,k4,k5,k6)
 *   Step 3: 旋转矫正 (乘以 R)
 *   Step 4: 投影到矫正后像素坐标 (乘以 P)
 */

#include "undistort_points_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <stdexcept>

using namespace calib;


CALIB_DEFINE_LOG_TAG(11, UndistortPointsCuda);

// ============================================================================
// CUDA Kernel
// ============================================================================

__global__ void UndistortRectifyKernel(
    const float2* __restrict__ d_src,
    float2* __restrict__ d_dst,
    int pointCount,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2,
    float k3, float k4, float k5, float k6,
    const float* __restrict__ d_R,
    float fx_p, float fy_p, float cx_p, float cy_p, float tx_p)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pointCount) return;

    float2 pt = d_src[tid];
    float u = pt.x;
    float v = pt.y;

    // Step 1: 像素坐标 -> 归一化坐标
    float x = (u - cx) / fx;
    float y = (v - cy) / fy;

    // Step 2: 迭代反向去畸变
    // 给定畸变归一化坐标 (x, y), 求无畸变坐标 (xu, yu)
    // 使得 distort(xu, yu) == (x, y)
    // 261031 对齐 OpenCV：迭代 5→15（OpenCV 默认 10，边缘大畸变点 5 次不
    // 收敛＝系统性偏差来源）；迭代变量 double 精度（OpenCV 用 double——
    // float 精度在边缘累积 ~0.1px 差＝重建 ~0.5mm 偏差）
    double xu = static_cast<double>(x);
    double yu = static_cast<double>(y);
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);
    const double k1d = k1, k2d = k2, k3d = k3;
    const double k4d = k4, k5d = k5, k6d = k6;
    const double p1d = p1, p2d = p2;

    for (int iter = 0; iter < 15; ++iter) {
        double r2 = xu * xu + yu * yu;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double num = 1.0 + k1d * r2 + k2d * r4 + k3d * r6;
        double den = 1.0 + k4d * r2 + k5d * r4 + k6d * r6;
        double radial = num / den;

        double dx_t = 2.0 * p1d * xu * yu + p2d * (r2 + 2.0 * xu * xu);
        double dy_t = p1d * (r2 + 2.0 * yu * yu) + 2.0 * p2d * xu * yu;

        xu = (xd - dx_t) / radial;
        yu = (yd - dy_t) / radial;
    }

    // Step 3: 旋转矫正（double 精度——与 OpenCV undistortPoints 同口径）
    double xr = static_cast<double>(d_R[0]) * xu + static_cast<double>(d_R[1]) * yu
              + static_cast<double>(d_R[2]);
    double yr = static_cast<double>(d_R[3]) * xu + static_cast<double>(d_R[4]) * yu
              + static_cast<double>(d_R[5]);
    double zr = static_cast<double>(d_R[6]) * xu + static_cast<double>(d_R[7]) * yu
              + static_cast<double>(d_R[8]);

    // Step 4: 投影到矫正后像素坐标（齐次归一化后加主点）
    double inv_z = 1.0 / zr;
    double u_out = static_cast<double>(fx_p) * (xr * inv_z)
                 + static_cast<double>(cx_p) + static_cast<double>(tx_p);
    double v_out = static_cast<double>(fy_p) * (yr * inv_z)
                 + static_cast<double>(cy_p);

    d_dst[tid] = make_float2(static_cast<float>(u_out), static_cast<float>(v_out));
}

// ============================================================================
// ScopedFlag (Debug-only thread safety)
// ============================================================================

#ifndef NDEBUG
class ScopedFlag {
public:
    explicit ScopedFlag(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedFlag() { flag_->store(false); }
    ScopedFlag(const ScopedFlag&) = delete;
    ScopedFlag& operator=(const ScopedFlag&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// Impl Implementation
// ============================================================================

UndistortPointsCuda::Impl::Impl(const UndistortPointsParams& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[11-UndistortPointsCuda] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[11-UndistortPointsCuda] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    extractCalibParams();
}

void UndistortPointsCuda::Impl::extractCalibParams() {
    cv::Mat cm;
    params_.cameraMatrix.convertTo(cm, CV_32F);
    fx_ = static_cast<float>(cm.at<float>(0, 0));
    fy_ = static_cast<float>(cm.at<float>(1, 1));
    cx_ = static_cast<float>(cm.at<float>(0, 2));
    cy_ = static_cast<float>(cm.at<float>(1, 2));

    cv::Mat dc;
    params_.distCoeffs.reshape(1, 1).convertTo(dc, CV_32F);
    k1_ = dc.total() > 0 ? dc.at<float>(0) : 0.0f;
    k2_ = dc.total() > 1 ? dc.at<float>(1) : 0.0f;
    p1_ = dc.total() > 2 ? dc.at<float>(2) : 0.0f;
    p2_ = dc.total() > 3 ? dc.at<float>(3) : 0.0f;
    k3_ = dc.total() > 4 ? dc.at<float>(4) : 0.0f;
    k4_ = dc.total() > 5 ? dc.at<float>(5) : 0.0f;
    k5_ = dc.total() > 6 ? dc.at<float>(6) : 0.0f;
    k6_ = dc.total() > 7 ? dc.at<float>(7) : 0.0f;

    if (!params_.R.empty()) {
        cv::Mat Rf;
        params_.R.convertTo(Rf, CV_32F);
        for (int i = 0; i < 9; ++i)
            R_[i] = Rf.at<float>(i / 3, i % 3);
    } else {
        R_[0] = 1.0f; R_[1] = 0.0f; R_[2] = 0.0f;
        R_[3] = 0.0f; R_[4] = 1.0f; R_[5] = 0.0f;
        R_[6] = 0.0f; R_[7] = 0.0f; R_[8] = 1.0f;
    }

    if (!params_.P.empty()) {
        cv::Mat Pf;
        params_.P.convertTo(Pf, CV_32F);
        fx_p_ = static_cast<float>(Pf.at<float>(0, 0));
        fy_p_ = static_cast<float>(Pf.at<float>(1, 1));
        cx_p_ = static_cast<float>(Pf.at<float>(0, 2));
        cy_p_ = static_cast<float>(Pf.at<float>(1, 2));
        tx_p_ = static_cast<float>(Pf.at<float>(0, 3));
    } else {
        fx_p_ = fx_;
        fy_p_ = fy_;
        cx_p_ = cx_;
        cy_p_ = cy_;
        tx_p_ = 0.0f;
    }
}

void UndistortPointsCuda::Impl::Warmup(int maxPointCount) {
    if (maxPointCount <= 0) {
        CALIB_LOG_WARN("Warmup(): invalid maxPointCount={}, skipping", maxPointCount);
        return;
    }

    if (warmed_up_ && warmup_pointCount_ == maxPointCount) {
        CALIB_LOG_DEBUG("Warmup(): already warmed up for {} points", maxPointCount);
        return;
    }

    d_output_.create(1, maxPointCount, CV_32FC2);

    float h_R[9];
    for (int i = 0; i < 9; ++i) h_R[i] = R_[i];
    d_R_.create(1, 9, CV_32FC1);
    d_R_.upload(cv::Mat(1, 9, CV_32FC1, h_R));

    warmed_up_ = true;
    warmup_pointCount_ = maxPointCount;

    CALIB_LOG_INFO("Warmup(): allocated GPU buffers for {} points", maxPointCount);
}

UndistortPointsResult UndistortPointsCuda::Impl::Execute(
    const cv::cuda::GpuMat& d_points,
    const cv::cuda::GpuMat& d_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlag guard(&inProcess_);
#endif

    UndistortPointsResult result;

    try {
        int pointCount = d_points.rows * d_points.cols;

        // 【速度优化】容量式持久缓冲：只在不够时扩容；R 矩阵只上传一次
        //（显式流）——原先 warmed_up_ 恒假导致每圈重复上传 R（默认流隐式同步）
        if (d_R_.empty()) {
            float h_R[9];
            for (int i = 0; i < 9; ++i) h_R[i] = R_[i];
            d_R_.upload(cv::Mat(1, 9, CV_32FC1, h_R), stream);
            CALIB_LOG_INFO("Execute(): R matrix uploaded once (persistent)");
        }
        if (d_output_.cols < pointCount) {
            d_output_.create(1, pointCount, CV_32FC2);
            CALIB_LOG_INFO("Execute(): output buffer grown to {} points", pointCount);
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

        const float2* d_src_ptr = d_points.ptr<float2>();
        float2* d_dst_ptr = d_output_.ptr<float2>();
        const float* d_R_ptr = d_R_.ptr<float>();

        int blockSize = 256;
        int gridSize = (pointCount + blockSize - 1) / blockSize;

        UndistortRectifyKernel<<<gridSize, blockSize, 0, cuda_stream>>>(
            d_src_ptr, d_dst_ptr, pointCount,
            fx_, fy_, cx_, cy_,
            k1_, k2_, p1_, p2_, k3_, k4_, k5_, k6_,
            d_R_ptr,
            fx_p_, fy_p_, cx_p_, cy_p_, tx_p_);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel launch failed: ") + cudaGetErrorString(err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }

#ifndef NDEBUG
        cudaStreamSynchronize(cuda_stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result.success = false;
            result.message = std::string("Kernel execution failed: ") + cudaGetErrorString(err);
            CALIB_LOG_ERROR("Execute(): {}", result.message);
            return result;
        }
#endif

        result.success = true;
        result.message = "Success";
        // 【速度优化】返回持久缓冲的 ROI 视图（有效期至下一次 Execute/SetParams/
        // Destroy，v3 同款契约）——原 clone() 每圈 cudaMalloc＋D2D 拷贝 ×2
        result.d_rectifiedPoints = std::make_shared<cv::cuda::GpuMat>(
            d_output_.colRange(0, pointCount));

        if (!d_line_ids.empty()) {
            result.d_line_ids = std::make_shared<cv::cuda::GpuMat>(d_line_ids);
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("Execute(): {}", result.message);
    }

    return result;
}

void UndistortPointsCuda::Impl::Destroy() {
    d_output_.release();
    d_R_.release();
    warmed_up_ = false;
    warmup_pointCount_ = 0;
    CALIB_LOG_INFO("Destroy() completed");
}

void UndistortPointsCuda::Impl::SetParams(const UndistortPointsParams& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("SetParams(): called while Execute() is running");
        throw std::runtime_error("[11-UndistortPointsCuda] SetParams() called during Execute()");
    }
#endif

    params.validate();

    bool deviceChanged = (params.deviceId != params_.deviceId);

    params_ = params;
    extractCalibParams();

    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }

    warmed_up_ = false;
    warmup_pointCount_ = 0;
    d_R_.release();   // 【速度优化】参数可能改 R → 下次 Execute 重传一次

    CALIB_LOG_INFO("SetParams(): params updated, warmup reset");
}
