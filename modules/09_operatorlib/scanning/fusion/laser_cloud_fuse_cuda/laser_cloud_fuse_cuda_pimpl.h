#pragma once

#include "laser_cloud_fuse_cuda.h"
#include <cuda_runtime.h>

namespace calib {

// ============================================================
// Impl — 设备状态（含 CUDA 类型，仅供 .cpp / .cu include）
// ============================================================
struct LaserCloudFuseCuda::Impl {
    LaserCloudFuseCUDAParams params_;
    bool warmed_up_ = false;

    // --- 哈希表 SoA（常驻显存，跨帧持久） ---
    unsigned long long* d_keys_      = nullptr;  // voxel key + 1（0 = 空槽）
    unsigned int*       d_fusedIdx_  = nullptr;  // → d_fusedXyz_ 索引
    unsigned int*       d_counts_    = nullptr;  // 饱和计数
    size_t              capacity_    = 0;
    unsigned long long  mask_        = 0;

    // --- 融合点缓冲（常驻显存） ---
    float*        d_fusedXyz_        = nullptr;  // maxFusedPoints_ × 3
    float*        d_fusedNormal_     = nullptr;  // maxFusedPoints_ × 3（法线算子写入）
    unsigned int* d_fusedPointCount_ = nullptr;  // 设备端原子计数器
    size_t        maxFusedPoints_    = 0;

    // --- Host 镜像 ---
    unsigned int h_fusedPointCount_  = 0;
    size_t       h_voxelCount_       = 0;

    // --- 每帧临时缓冲（复用） ---
    int* d_survivingFlags_ = nullptr;   // inputCount
    int* d_statsBuffer_    = nullptr;   // [3]: newVoxel, surviving, deleted
    int  h_statsBuffer_[3] = {0, 0, 0};
    size_t lastInputCount_ = 0;

    // --- 构造 / 析构 ---
    explicit Impl(const LaserCloudFuseCUDAParams& p);
    ~Impl();

    void allocateHash(size_t voxelCount);
    void allocatePoints(size_t maxPoints);
    void ensureTempBuffers(size_t inputCount);
    void freeAll();

    // --- fuse ---
    LaserCloudFuseCudaResult fuseImpl(const cv::cuda::GpuMat& d_points3d,
                  const cv::Matx33d& R, const cv::Vec3d& T,
                  cv::cuda::Stream& stream);

    // --- removePoints（编辑账本，.cu 实现） ---
    ResultStatus removePointsImpl(const std::vector<uint32_t>& indices,
                                  cudaStream_t stream);

    // --- clear ---
    void clearImpl(cudaStream_t stream);
};

} // namespace calib
