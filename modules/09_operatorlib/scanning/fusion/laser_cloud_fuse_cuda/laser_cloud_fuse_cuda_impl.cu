#include "laser_cloud_fuse_cuda_pimpl.h"
#include "common/calib_logging.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <chrono>

using namespace calib;


// ============================================================
// 常量
// ============================================================

constexpr int64_t  VOXEL_BIAS  = (int64_t)1 << 20;
constexpr int      VOXEL_BITS  = 21;
constexpr uint64_t AXIS_MASK   = ((uint64_t)1 << VOXEL_BITS) - 1;
constexpr int      BLOCK_SIZE  = 256;

// ============================================================
// Device 辅助函数
// ============================================================

__device__ __forceinline__ uint64_t hash64Device(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__device__ __forceinline__ uint64_t packVoxelKeyDevice(int ix, int iy, int iz) {
    uint64_t bx = (uint64_t)((int64_t)ix + VOXEL_BIAS);
    uint64_t by = (uint64_t)((int64_t)iy + VOXEL_BIAS);
    uint64_t bz = (uint64_t)((int64_t)iz + VOXEL_BIAS);
    if ((bx | by | bz) > AXIS_MASK) return UINT64_MAX; // 越界标记
    return bx | (by << VOXEL_BITS) | (bz << (2 * VOXEL_BITS));
}

// ============================================================
// RT 变换参数（按值传递给 kernel）
// ============================================================

struct RTParams {
    float R[9];
    float T[3];
};

// ============================================================
// clearKeysKernel — 清空哈希表
// ============================================================

__global__ void clearKeysKernel(unsigned long long* d_keys, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) d_keys[i] = 0ULL;
}

// ============================================================
// fuseKernel — 核心融合 kernel
// ============================================================

__global__ void fuseKernel(
    const float* __restrict__        d_inputXyz,
    int                              inputCount,
    RTParams                         rt,
    float                            invVoxelSize,
    unsigned int                     threshold,
    // --- 哈希表 ---
    unsigned long long* __restrict__ d_keys,
    unsigned int* __restrict__       d_fusedIdx,
    unsigned int* __restrict__       d_counts,
    unsigned long long               mask,
    // --- 融合点缓冲 ---
    float* __restrict__              d_fusedXyz,
    unsigned int                     maxFusedPoints,
    unsigned int* __restrict__       d_fusedPointCount,
    // --- 输出 ---
    int* __restrict__                d_survivingFlags,
    // --- 统计 ---
    int* __restrict__                d_newVoxelCount,
    int* __restrict__                d_survivingCount,
    int* __restrict__                d_deletedCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= inputCount) return;

    // 1. 读取输入点
    float px = d_inputXyz[i * 3 + 0];
    float py = d_inputXyz[i * 3 + 1];
    float pz = d_inputXyz[i * 3 + 2];

    // 2. R*p + T → 全局坐标
    float gx = rt.R[0]*px + rt.R[1]*py + rt.R[2]*pz + rt.T[0];
    float gy = rt.R[3]*px + rt.R[4]*py + rt.R[5]*pz + rt.T[1];
    float gz = rt.R[6]*px + rt.R[7]*py + rt.R[8]*pz + rt.T[2];

    // 3. 体素量化
    int ix = (int)floorf(gx * invVoxelSize);
    int iy = (int)floorf(gy * invVoxelSize);
    int iz = (int)floorf(gz * invVoxelSize);

    uint64_t key = packVoxelKeyDevice(ix, iy, iz);
    if (key == UINT64_MAX) {
        d_survivingFlags[i] = 0;
        atomicAdd(d_deletedCount, 1);
        return;
    }

    uint64_t storedKey = key + 1; // 0 保留为空槽

    // 4. 哈希 + atomicCAS 探测
    uint64_t h = hash64Device(key);
    unsigned long long idx = h & mask;

    int probes = 0;
    while (probes < 4096) {
        unsigned long long old = atomicCAS(
            (unsigned long long*)&d_keys[idx], 0ULL, storedKey);

        if (old == 0ULL || old == storedKey) {
            // === 新体素(old==0) 或 HIT(old==storedKey) ===
            // 统一计数：所有线程都用 atomicAdd，消除竞态
            if (old == 0ULL) {
                // 我们创建了此槽 — 写入代表点数据
                unsigned int fi = atomicAdd(d_fusedPointCount, 1u);
                if (fi >= maxFusedPoints) {
                    d_survivingFlags[i] = 0;
                    atomicAdd(d_deletedCount, 1);
                    return;
                }
                d_fusedXyz[fi * 3 + 0] = gx;
                d_fusedXyz[fi * 3 + 1] = gy;
                d_fusedXyz[fi * 3 + 2] = gz;
                d_fusedIdx[idx] = fi;
                atomicAdd(d_newVoxelCount, 1);
            }

            // 统一饱和判定
            unsigned int oldCount = atomicAdd(&d_counts[idx], 1u);
            if (oldCount < threshold) {
                d_survivingFlags[i] = 1;
                atomicAdd(d_survivingCount, 1);
            } else {
                d_survivingFlags[i] = 0;
                atomicAdd(d_deletedCount, 1);
            }
            return;
        }
        // 碰撞 — 线性探测
        idx = (idx + 1) & mask;
        probes++;
    }

    // 探测次数超限 — 表接近满
    d_survivingFlags[i] = 0;
    atomicAdd(d_deletedCount, 1);
}

// ============================================================
// safeCudaFree
// ============================================================

template <typename T>
static void safeCudaFree(T*& ptr) {
    if (ptr) { cudaFree(ptr); ptr = nullptr; }
}

// ============================================================
// Impl — 构造 / 析构
// ============================================================

LaserCloudFuseCuda::Impl::Impl(const LaserCloudFuseCUDAParams& p) : params_(p) {
    params_.validate();

    size_t slots = params_.reserveVoxelCount;
    // 向上取 2 的幂
    size_t pow2 = 64;
    while (pow2 < slots) pow2 <<= 1;
    slots = pow2;

    allocateHash(slots);
    allocatePoints(slots); // 点数上限 = 哈希槽数

    // 设备端计数器
    cudaMalloc(&d_fusedPointCount_, sizeof(unsigned int));
    cudaMemset(d_fusedPointCount_, 0, sizeof(unsigned int));
}

LaserCloudFuseCuda::Impl::~Impl() {
    cudaDeviceSynchronize();
    freeAll();
    safeCudaFree(d_fusedPointCount_);
}

// ============================================================
// Impl — 内存分配
// ============================================================

void LaserCloudFuseCuda::Impl::allocateHash(size_t voxelCount) {
    size_t pow2 = 64;
    while (pow2 < voxelCount) pow2 <<= 1;
    if (pow2 <= capacity_) return; // 不缩容

    freeAll();

    capacity_ = pow2;
    mask_     = pow2 - 1;

    cudaMalloc(&d_keys_,     capacity_ * sizeof(unsigned long long));
    cudaMalloc(&d_fusedIdx_, capacity_ * sizeof(unsigned int));
    cudaMalloc(&d_counts_,   capacity_ * sizeof(unsigned int));

    cudaMemset(d_keys_,     0, capacity_ * sizeof(unsigned long long));
    cudaMemset(d_fusedIdx_, 0, capacity_ * sizeof(unsigned int));
    cudaMemset(d_counts_,   0, capacity_ * sizeof(unsigned int));

    // 重置设备计数器
    if (d_fusedPointCount_) {
        cudaMemset(d_fusedPointCount_, 0, sizeof(unsigned int));
    }
    h_fusedPointCount_ = 0;
    h_voxelCount_ = 0;
}

void LaserCloudFuseCuda::Impl::allocatePoints(size_t maxPoints) {
    if (maxPoints <= maxFusedPoints_) return;
    safeCudaFree(d_fusedXyz_);
    safeCudaFree(d_fusedNormal_);

    maxFusedPoints_ = maxPoints;
    cudaMalloc(&d_fusedXyz_,    maxPoints * 3 * sizeof(float));
    cudaMalloc(&d_fusedNormal_, maxPoints * 3 * sizeof(float));
    cudaMemset(d_fusedXyz_,    0, maxPoints * 3 * sizeof(float));
    cudaMemset(d_fusedNormal_, 0, maxPoints * 3 * sizeof(float));
}

void LaserCloudFuseCuda::Impl::ensureTempBuffers(size_t inputCount) {
    if (inputCount <= lastInputCount_ && d_survivingFlags_) return;
    safeCudaFree(d_survivingFlags_);
    safeCudaFree(d_statsBuffer_);

    size_t allocCount = inputCount > 0 ? inputCount : 1;
    cudaMalloc(&d_survivingFlags_, allocCount * sizeof(int));
    cudaMalloc(&d_statsBuffer_,    3 * sizeof(int));
    lastInputCount_ = allocCount;
}

void LaserCloudFuseCuda::Impl::freeAll() {
    safeCudaFree(d_keys_);
    safeCudaFree(d_fusedIdx_);
    safeCudaFree(d_counts_);
    safeCudaFree(d_fusedXyz_);
    safeCudaFree(d_fusedNormal_);
    safeCudaFree(d_survivingFlags_);
    safeCudaFree(d_statsBuffer_);
    capacity_ = 0;
    mask_ = 0;
    maxFusedPoints_ = 0;
    lastInputCount_ = 0;
    h_fusedPointCount_ = 0;
    h_voxelCount_ = 0;
}

// ============================================================
// Impl — fuseImpl
// ============================================================

LaserCloudFuseCudaResult LaserCloudFuseCuda::Impl::fuseImpl(
    const cv::cuda::GpuMat& d_points3d,
    const cv::Matx33d& R, const cv::Vec3d& T,
    cv::cuda::Stream& stream)
{
    LaserCloudFuseCudaResult result;

    int inputCount = static_cast<int>(d_points3d.cols);
    if (d_points3d.rows != 1 || d_points3d.type() != CV_32FC3 || inputCount == 0) {
        result.success = false;
        result.message = "Invalid input: expected 1×N CV_32FC3 GpuMat";
        return result;
    }

    auto totalStart = std::chrono::high_resolution_clock::now();

    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

    // 确保临时缓冲够大
    ensureTempBuffers(static_cast<size_t>(inputCount));

    // R/T → float struct（按值传递给 kernel）
    RTParams rt;
    rt.R[0] = (float)R(0,0); rt.R[1] = (float)R(0,1); rt.R[2] = (float)R(0,2);
    rt.R[3] = (float)R(1,0); rt.R[4] = (float)R(1,1); rt.R[5] = (float)R(1,2);
    rt.R[6] = (float)R(2,0); rt.R[7] = (float)R(2,1); rt.R[8] = (float)R(2,2);
    rt.T[0] = (float)T(0); rt.T[1] = (float)T(1); rt.T[2] = (float)T(2);

    // 清零统计
    cudaMemsetAsync(d_statsBuffer_, 0, 3 * sizeof(int), s);

    const float* d_input = d_points3d.ptr<float>();
    float invVoxelSize = 1.0f / params_.voxelSize;
    unsigned int threshold = static_cast<unsigned int>(params_.saturationThreshold);

    int grid = (inputCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fuseKernel<<<grid, BLOCK_SIZE, 0, s>>>(
        d_input, inputCount,
        rt,
        invVoxelSize, threshold,
        d_keys_, d_fusedIdx_, d_counts_, mask_,
        d_fusedXyz_, static_cast<unsigned int>(maxFusedPoints_), d_fusedPointCount_,
        d_survivingFlags_,
        &d_statsBuffer_[0],   // newVoxelCount
        &d_statsBuffer_[1],   // survivingCount
        &d_statsBuffer_[2]);  // deletedCount

    // 读回标量
    cudaMemcpyAsync(&h_fusedPointCount_, d_fusedPointCount_,
                    sizeof(unsigned int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_statsBuffer_, d_statsBuffer_,
                    3 * sizeof(int), cudaMemcpyDeviceToHost, s);

    cudaStreamSynchronize(s);

    auto totalEnd = std::chrono::high_resolution_clock::now();

    // 填充结果
    result.success       = true;
    result.message       = "Frame fusion completed (CUDA)";
    result.inputCount    = inputCount;
    result.newVoxelCount = h_statsBuffer_[0];
    result.survivingCount= h_statsBuffer_[1];
    result.deletedCount  = h_statsBuffer_[2];
    result.totalVoxelCount = static_cast<int>(h_fusedPointCount_);
    result.totalTimeMs   = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    double keepRatio = inputCount > 0
        ? static_cast<double>(result.survivingCount) / inputCount : 0.0;
    result.qualityFlag = keepRatio >= 0.5 ? calib::QualityFlag::Normal :
                         (keepRatio >= 0.1 ? calib::QualityFlag::Degraded :
                                             calib::QualityFlag::Warning);

    h_voxelCount_ = h_fusedPointCount_; // 每个体素一个代表点

    CALIB_LOG_DEBUG("fuseCUDA: in={} kept={} del={} new={} total={} t={:.3f}ms",
                    inputCount, result.survivingCount, result.deletedCount,
                    result.newVoxelCount, result.totalVoxelCount, result.totalTimeMs);

    return result;
}

// ============================================================
// Impl — clearImpl
// ============================================================

void LaserCloudFuseCuda::Impl::clearImpl(cudaStream_t stream) {
    if (!d_keys_) return;
    size_t grid = (capacity_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clearKeysKernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_keys_, capacity_);
    cudaMemsetAsync(d_counts_, 0, capacity_ * sizeof(unsigned int), stream);
    cudaMemsetAsync(d_fusedIdx_, 0, capacity_ * sizeof(unsigned int), stream);
    cudaMemsetAsync(d_fusedXyz_, 0, maxFusedPoints_ * 3 * sizeof(float), stream);
    cudaMemsetAsync(d_fusedNormal_, 0, maxFusedPoints_ * 3 * sizeof(float), stream);
    cudaMemsetAsync(d_fusedPointCount_, 0, sizeof(unsigned int), stream);
    cudaStreamSynchronize(stream);
    h_fusedPointCount_ = 0;
    h_voxelCount_ = 0;
}
