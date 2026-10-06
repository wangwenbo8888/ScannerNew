#include "laser_cloud_normal_cuda_pimpl.h"
#include "common/calib_logging.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <chrono>

using namespace calib;


// ============================================================
// 常量（与融合 kernel 一致）
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
    if ((bx | by | bz) > AXIS_MASK) return UINT64_MAX;
    return bx | (by << VOXEL_BITS) | (bz << (2 * VOXEL_BITS));
}

/// 只读哈希查找 — 返回邻居代表点的 xyz 指针，未找到返回 nullptr
__device__ __forceinline__ const float* lookupVoxelDevice(
    uint64_t key,
    const unsigned long long* d_keys,
    const unsigned int* d_fusedIdx,
    const float* d_fusedXyz,
    unsigned long long mask)
{
    uint64_t storedKey = key + 1;
    uint64_t h = hash64Device(key);
    unsigned long long idx = h & mask;
    int probes = 0;
    while (probes < 1024) {
        unsigned long long existing = d_keys[idx];
        if (existing == 0ULL) return nullptr;
        if (existing == storedKey) {
            return &d_fusedXyz[(size_t)d_fusedIdx[idx] * 3];
        }
        idx = (idx + 1) & mask;
        probes++;
    }
    return nullptr;
}

/// 3×3 对称矩阵 Jacobi 特征分解 — 最小特征值特征向量写入 (outNx, outNy, outNz)
__device__ __forceinline__ void jacobi3x3Device(
    const float cov[6], float& outNx, float& outNy, float& outNz)
{
    float A[3][3] = {
        {cov[0], cov[1], cov[2]},
        {cov[1], cov[3], cov[4]},
        {cov[2], cov[4], cov[5]}
    };
    float V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    for (int sweep = 0; sweep < 12; ++sweep) {
        int p = 0, q = 1;
        float maxOff = fabsf(A[0][1]);
        if (fabsf(A[0][2]) > maxOff) { p = 0; q = 2; maxOff = fabsf(A[0][2]); }
        if (fabsf(A[1][2]) > maxOff) { p = 1; q = 2; maxOff = fabsf(A[1][2]); }
        if (maxOff < 1e-12f) break;

        float app = A[p][p], aqq = A[q][q], apq = A[p][q];
        float tau = (aqq - app) / (2.0f * apq);
        float t = (tau >= 0.0f)
            ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
            : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
        float c = 1.0f / sqrtf(1.0f + t * t);
        float s = t * c;

        for (int i = 0; i < 3; ++i) {
            float aip = A[i][p], aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
        }
        for (int j = 0; j < 3; ++j) {
            float apj = A[p][j], aqj = A[q][j];
            A[p][j] = c * apj - s * aqj;
            A[q][j] = s * apj + c * aqj;
        }
        for (int i = 0; i < 3; ++i) {
            float vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    int minIdx = 0;
    if (A[1][1] < A[0][0]) minIdx = 1;
    if (A[2][2] < A[minIdx][minIdx]) minIdx = 2;

    outNx = V[0][minIdx];
    outNy = V[1][minIdx];
    outNz = V[2][minIdx];
}

// ============================================================
// computeNormalsKernel
// ============================================================

__global__ void computeNormalsKernel(
    const unsigned long long* __restrict__ d_keys,
    const unsigned int* __restrict__       d_fusedIdx,
    const float* __restrict__              d_fusedXyz,
    float* __restrict__                    d_fusedNormal,
    unsigned long long                     mask,
    float                                  invVoxelSize,
    unsigned int                           beginIdx,
    unsigned int                           endIdx,
    int                                    minNeighbors,
    float                                  fallbackNx,
    float                                  fallbackNy,
    float                                  fallbackNz,
    int* d_processedCount,
    int* d_fallbackCount)
{
    unsigned int i = beginIdx + (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= endIdx) return;

    float px = d_fusedXyz[i * 3 + 0];
    float py = d_fusedXyz[i * 3 + 1];
    float pz = d_fusedXyz[i * 3 + 2];

    int ix = (int)floorf(px * invVoxelSize);
    int iy = (int)floorf(py * invVoxelSize);
    int iz = (int)floorf(pz * invVoxelSize);

    // 收集 3×3×3 邻域
    float neighbors[27 * 3];
    int count = 0;

    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        uint64_t key = packVoxelKeyDevice(ix + dx, iy + dy, iz + dz);
        if (key == UINT64_MAX) continue;
        const float* np = lookupVoxelDevice(key, d_keys, d_fusedIdx, d_fusedXyz, mask);
        if (np != nullptr) {
            neighbors[count * 3]     = np[0];
            neighbors[count * 3 + 1] = np[1];
            neighbors[count * 3 + 2] = np[2];
            count++;
        }
    }

    if (count < minNeighbors) {
        d_fusedNormal[i * 3]     = fallbackNx;
        d_fusedNormal[i * 3 + 1] = fallbackNy;
        d_fusedNormal[i * 3 + 2] = fallbackNz;
        atomicAdd(d_fallbackCount, 1);
        return;
    }

    // 质心
    float cx = 0, cy = 0, cz = 0;
    for (int j = 0; j < count; j++) {
        cx += neighbors[j * 3];
        cy += neighbors[j * 3 + 1];
        cz += neighbors[j * 3 + 2];
    }
    float invN = 1.0f / (float)count;
    cx *= invN; cy *= invN; cz *= invN;

    // 协方差
    float cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
    for (int j = 0; j < count; j++) {
        float ddx = neighbors[j * 3]     - cx;
        float ddy = neighbors[j * 3 + 1] - cy;
        float ddz = neighbors[j * 3 + 2] - cz;
        cxx += ddx * ddx; cyy += ddy * ddy; czz += ddz * ddz;
        cxy += ddx * ddy; cxz += ddx * ddz; cyz += ddy * ddz;
    }
    float cov[6] = {cxx * invN, cxy * invN, cxz * invN,
                    cyy * invN, cyz * invN, czz * invN};

    // Jacobi 特征分解
    float nx, ny, nz;
    jacobi3x3Device(cov, nx, ny, nz);

    // 归一化 + 写回
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-10f) {
        float invLen = 1.0f / len;
        d_fusedNormal[i * 3]     = nx * invLen;
        d_fusedNormal[i * 3 + 1] = ny * invLen;
        d_fusedNormal[i * 3 + 2] = nz * invLen;
        atomicAdd(d_processedCount, 1);
    } else {
        d_fusedNormal[i * 3]     = fallbackNx;
        d_fusedNormal[i * 3 + 1] = fallbackNy;
        d_fusedNormal[i * 3 + 2] = fallbackNz;
        atomicAdd(d_fallbackCount, 1);
    }
}

// ============================================================
// Impl::computeImpl
// ============================================================

LaserCloudNormalCudaResult LaserCloudNormalCuda::Impl::computeImpl(
    const LaserCloudFuseDeviceContext& ctx,
    size_t beginIdx, size_t endIdx,
    cv::cuda::Stream& stream)
{
    LaserCloudNormalCudaResult result;

    if (beginIdx >= endIdx) {
        result.success = true;
        result.message = "No new voxels";
        result.qualityFlag = calib::QualityFlag::Warning;
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

    // 清零统计
    cudaMemsetAsync(d_stats_, 0, 2 * sizeof(int), s);

    unsigned int count = static_cast<unsigned int>(endIdx - beginIdx);
    int grid = ((int)count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    computeNormalsKernel<<<grid, BLOCK_SIZE, 0, s>>>(
        ctx.d_keys, ctx.d_fusedIdx, ctx.d_fusedXyz, ctx.d_fusedNormal,
        ctx.mask, ctx.invVoxelSize,
        static_cast<unsigned int>(beginIdx),
        static_cast<unsigned int>(endIdx),
        params_.minNeighbors,
        params_.fallbackNx, params_.fallbackNy, params_.fallbackNz,
        &d_stats_[0],   // processedCount
        &d_stats_[1]);  // fallbackCount

    cudaMemcpyAsync(h_stats_, d_stats_, 2 * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    auto end = std::chrono::high_resolution_clock::now();

    result.success = true;
    result.message = "Normal estimation completed (CUDA)";
    result.processedCount = static_cast<size_t>(h_stats_[0]);
    result.fallbackCount  = static_cast<size_t>(h_stats_[1]);
    result.totalTimeMs    = std::chrono::duration<double, std::milli>(end - start).count();

    size_t total = result.processedCount + result.fallbackCount;
    double fbRate = total > 0
        ? static_cast<double>(result.fallbackCount) / total : 0.0;
    result.qualityFlag = fbRate < 0.1  ? calib::QualityFlag::Normal :
                         fbRate < 0.5  ? calib::QualityFlag::Degraded :
                                         calib::QualityFlag::Warning;

    CALIB_LOG_DEBUG("computeCUDA: voxels={} processed={} fallback={} t={:.3f}ms",
                    endIdx - beginIdx, result.processedCount, result.fallbackCount,
                    result.totalTimeMs);

    return result;
}
