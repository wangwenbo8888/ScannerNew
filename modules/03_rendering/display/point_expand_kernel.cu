#include "point_expand_kernel.h"

#include <cuda_runtime.h>
#include <cmath>

namespace calib {


// ============================================================
// 常量
// ============================================================

constexpr int BLOCK_SIZE = 256;

// ============================================================
// Device 辅助：构建正交基
// ============================================================

__device__ __forceinline__ void buildOrthonormalBasis(
    float Nx, float Ny, float Nz,
    float& Tx, float& Ty, float& Tz,
    float& Bx, float& By, float& Bz)
{
    float ux, uy, uz;
    if (fabsf(Ny) < 0.99f) {
        ux = 0.f; uy = 1.f; uz = 0.f;
    } else {
        ux = 1.f; uy = 0.f; uz = 0.f;
    }

    // T = normalize(cross(up, N))
    Tx = uy * Nz - uz * Ny;
    Ty = uz * Nx - ux * Nz;
    Tz = ux * Ny - uy * Nx;
    float tLen = rsqrtf(Tx*Tx + Ty*Ty + Tz*Tz);
    Tx *= tLen; Ty *= tLen; Tz *= tLen;

    // B = cross(N, T)
    Bx = Ny * Tz - Nz * Ty;
    By = Nz * Tx - Nx * Tz;
    Bz = Nx * Ty - Ny * Tx;
}

// ============================================================
// 展点 Kernel — 激光点云（统一 voxelSize）
// 每线程处理 1 个点，写 4 个顶点到 3 个 SoA 数组
// ============================================================

__global__ void expandLaserPointsKernel(
    float* __restrict__  d_outPos,
    float* __restrict__  d_outUv,
    float* __restrict__  d_outNormal,
    const float* __restrict__ d_xyz,
    const float* __restrict__ d_normal,
    float  halfSize,
    size_t pointCount)
{
    size_t i = (size_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= pointCount) return;

    size_t vBase = i * 4;

    float px = d_xyz[i * 3];
    float py = d_xyz[i * 3 + 1];
    float pz = d_xyz[i * 3 + 2];

    float nx = d_normal[i * 3];
    float ny = d_normal[i * 3 + 1];
    float nz = d_normal[i * 3 + 2];

    float nLen = rsqrtf(nx*nx + ny*ny + nz*nz + 1e-20f);
    nx *= nLen; ny *= nLen; nz *= nLen;

    float Tx, Ty, Tz, Bx, By, Bz;
    buildOrthonormalBasis(nx, ny, nz, Tx, Ty, Tz, Bx, By, Bz);

    float hx = Tx * halfSize, hy = Ty * halfSize, hz = Tz * halfSize;
    float bx = Bx * halfSize, by = By * halfSize, bz = Bz * halfSize;

    // V0: (-1,-1)
    d_outPos[vBase*3   ] = px - hx - bx;
    d_outPos[vBase*3 +1] = py - hy - by;
    d_outPos[vBase*3 +2] = pz - hz - bz;
    d_outUv [vBase*2   ] = -1.f;
    d_outUv [vBase*2 +1] = -1.f;
    d_outNormal[vBase*3   ] = nx;
    d_outNormal[vBase*3 +1] = ny;
    d_outNormal[vBase*3 +2] = nz;

    // V1: (+1,-1)
    d_outPos[(vBase+1)*3   ] = px + hx - bx;
    d_outPos[(vBase+1)*3 +1] = py + hy - by;
    d_outPos[(vBase+1)*3 +2] = pz + hz - bz;
    d_outUv [(vBase+1)*2   ] =  1.f;
    d_outUv [(vBase+1)*2 +1] = -1.f;
    d_outNormal[(vBase+1)*3   ] = nx;
    d_outNormal[(vBase+1)*3 +1] = ny;
    d_outNormal[(vBase+1)*3 +2] = nz;

    // V2: (+1,+1)
    d_outPos[(vBase+2)*3   ] = px + hx + bx;
    d_outPos[(vBase+2)*3 +1] = py + hy + by;
    d_outPos[(vBase+2)*3 +2] = pz + hz + bz;
    d_outUv [(vBase+2)*2   ] =  1.f;
    d_outUv [(vBase+2)*2 +1] =  1.f;
    d_outNormal[(vBase+2)*3   ] = nx;
    d_outNormal[(vBase+2)*3 +1] = ny;
    d_outNormal[(vBase+2)*3 +2] = nz;

    // V3: (-1,+1)
    d_outPos[(vBase+3)*3   ] = px - hx + bx;
    d_outPos[(vBase+3)*3 +1] = py - hy + by;
    d_outPos[(vBase+3)*3 +2] = pz - hz + bz;
    d_outUv [(vBase+3)*2   ] = -1.f;
    d_outUv [(vBase+3)*2 +1] =  1.f;
    d_outNormal[(vBase+3)*3   ] = nx;
    d_outNormal[(vBase+3)*3 +1] = ny;
    d_outNormal[(vBase+3)*3 +2] = nz;
}

// ============================================================
// Host 启动函数
// ============================================================

void launchExpandLaserPoints(
    float*        d_outPos,
    float*        d_outUv,
    float*        d_outNormal,
    const float*  d_xyz,
    const float*  d_normal,
    float         halfSize,
    size_t        pointCount,
    void*         stream)
{
    if (pointCount == 0) return;

    int blocks = (int)((pointCount + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    expandLaserPointsKernel<<<blocks, BLOCK_SIZE, 0, s>>>(
        d_outPos, d_outUv, d_outNormal,
        d_xyz, d_normal, halfSize, pointCount);
}

} // namespace calib
