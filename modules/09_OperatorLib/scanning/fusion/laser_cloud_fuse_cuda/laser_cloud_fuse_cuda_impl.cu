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
// 【2601021 扩容改造】负载因子阈值（>70% 触发固定增量扩容）
constexpr double   LOAD_FACTOR = 0.7;

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
// 【2601021 扩容改造】rehashMoveKernel — 旧表条目原子搬入新表
// ============================================================
// 旧表每槽 key 唯一，但多条目可能探测到新表同一槽 → atomicCAS 抢占。
// 搬移内容：key（key+1 编码）＋ fusedIdx ＋ counts（饱和计数保全）。
__global__ void rehashMoveKernel(
    const unsigned long long* __restrict__ oldKeys,
    const unsigned int*       __restrict__ oldIdx,
    const unsigned int*       __restrict__ oldCnt,
    size_t oldCapacity,
    unsigned long long* __restrict__ newKeys,
    unsigned int*       __restrict__ newIdx,
    unsigned int*       __restrict__ newCnt,
    unsigned long long newCapacity)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= oldCapacity) return;
    unsigned long long storedKey = oldKeys[i];
    if (storedKey == 0ULL) return;                 // 空槽不搬

    uint64_t h = hash64Device(storedKey - 1);      // 还原 packed key 再哈希
    unsigned long long idx = h % newCapacity;
    while (true) {
        unsigned long long old = atomicCAS(
            (unsigned long long*)&newKeys[idx], 0ULL, storedKey);
        if (old == 0ULL) {                         // 抢到槽 → 写负载
            newIdx[idx] = oldIdx[i];
            newCnt[idx] = oldCnt[i];
            return;
        }
        if (old == storedKey) return;              // 不可能（key 唯一），保底
        if (++idx >= newCapacity) idx = 0;
    }
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
    unsigned long long               capacity,
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

    // 4. 哈希 + atomicCAS 探测（取余定位——容量非 2 的幂也可用）
    uint64_t h = hash64Device(key);
    unsigned long long idx = h % capacity;

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
        // 碰撞 — 线性探测（回绕）
        if (++idx >= capacity) idx = 0;
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

    // 【2601021 扩容改造】容量取任意正整数（取余探测，无 2 的幂约束）
    size_t slots = params_.reserveVoxelCount > 64 ? params_.reserveVoxelCount : 64;

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
    size_t want = voxelCount > 64 ? voxelCount : 64;
    if (want <= capacity_) return; // 不缩容

    freeAll();

    capacity_ = want;

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
    maxFusedPoints_ = 0;
    lastInputCount_ = 0;
    h_fusedPointCount_ = 0;
    h_voxelCount_ = 0;
}

// ============================================================
// 【2601021 扩容改造】Impl::rehashGrow — 非破坏性固定增量扩容
// ============================================================
// 负载 >70% 触发：容量 += growSlotCount（默认 1000 万，2601021 人工裁定，
// laser_fuse_stress 压测验证），保留全部已累积数据（对比 Reserve() 清空重来）。
// 步骤：① 新表分配清零 → ② 内核搬移旧槽（key/idx/counts）→
//       ③ 点/法线缓冲扩容（D2D 拷贝保数据）→ ④ 释放旧资源切换指针。
// 计时：全程 wall-clock，末尾 stream 同步后入账 rehashEvents_。
void LaserCloudFuseCuda::Impl::rehashGrow(cudaStream_t stream) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const size_t oldCap = capacity_;
    const size_t newCap = oldCap + params_.growSlotCount;

    // ① 新哈希表三数组（keys 清零即空槽；idx/cnts 由搬移覆写）
    unsigned long long* nKeys = nullptr;
    unsigned int*       nIdx  = nullptr;
    unsigned int*       nCnt  = nullptr;
    cudaMalloc(&nKeys, newCap * sizeof(unsigned long long));
    cudaMalloc(&nIdx,  newCap * sizeof(unsigned int));
    cudaMalloc(&nCnt,  newCap * sizeof(unsigned int));
    cudaMemsetAsync(nKeys, 0, newCap * sizeof(unsigned long long), stream);
    cudaMemsetAsync(nIdx,  0, newCap * sizeof(unsigned int), stream);
    cudaMemsetAsync(nCnt,  0, newCap * sizeof(unsigned int), stream);

    // ② 旧表条目搬入新表（取余定位）
    const unsigned long long newCapacity = (unsigned long long)newCap;
    size_t grid = (oldCap + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rehashMoveKernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_keys_, d_fusedIdx_, d_counts_, oldCap,
        nKeys, nIdx, nCnt, newCapacity);

    // ③ 点/法线缓冲扩到 newCap（保数据：旧 maxFusedPoints_ 区段 D2D 拷贝）
    float* nXyz = nullptr;
    float* nNorm = nullptr;
    cudaMalloc(&nXyz,  newCap * 3 * sizeof(float));
    cudaMalloc(&nNorm, newCap * 3 * sizeof(float));
    const size_t oldPtsBytes = maxFusedPoints_ * 3 * sizeof(float);
    cudaMemcpyAsync(nXyz,  d_fusedXyz_,    oldPtsBytes, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(nNorm, d_fusedNormal_, oldPtsBytes, cudaMemcpyDeviceToDevice, stream);

    // ④ 同步收尾 → 释放旧资源 → 切换
    cudaStreamSynchronize(stream);
    cudaFree(d_keys_);        d_keys_      = nKeys;
    cudaFree(d_fusedIdx_);    d_fusedIdx_  = nIdx;
    cudaFree(d_counts_);      d_counts_    = nCnt;
    cudaFree(d_fusedXyz_);    d_fusedXyz_  = nXyz;
    cudaFree(d_fusedNormal_); d_fusedNormal_ = nNorm;
    capacity_       = newCap;
    maxFusedPoints_ = newCap;

    auto t1 = std::chrono::high_resolution_clock::now();

    RehashEvent ev;
    ev.triggerVoxelCount = h_fusedPointCount_;
    ev.oldCapacity = oldCap;
    ev.newCapacity = newCap;
    ev.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rehashEvents_.push_back(ev);

    CALIB_LOG_INFO("rehashGrow: voxels={} cap {}->{} took {:.3f}ms",
                   ev.triggerVoxelCount, ev.oldCapacity, ev.newCapacity, ev.timeMs);
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
        d_keys_, d_fusedIdx_, d_counts_, (unsigned long long)capacity_,
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

    // 【2601021 扩容改造】负载 >70% → 非破坏性固定增量扩容（耗时单独入账，
    // 不计入融合耗时；laser_fuse_stress 压测：1000 万槽搬移约 17.7ms）
    result.hashCapacity = capacity_;
    if ((double)h_fusedPointCount_ > LOAD_FACTOR * (double)capacity_) {
        rehashGrow(s);
        result.rehashHappened = true;
        result.rehashTimeMs   = rehashEvents_.back().timeMs;
        result.hashCapacity   = capacity_;
    }

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

// ============================================================
// removePoints — 编辑账本（05 D4·激光云侧，实施计划 P2）
// ============================================================

#include <algorithm>   // sort/unique/binary_search（host 侧映射构建）

constexpr unsigned int kLcfRemoved = 0xFFFFFFFFu;

// 单数组压缩：幸存点按 host 顺序生成的 newIdx 写入 dst（确定性；xyz 与
// normal 各跑一次同一 kernel——同映射保证点-法线关联）
__global__ void compactArrayKernel(
    const float* __restrict__ src, float* __restrict__ dst,
    const unsigned int* __restrict__ newIdx, unsigned int oldCount)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= oldCount) return;
    const unsigned int ni = newIdx[i];
    if (ni == kLcfRemoved) return;
    dst[ni * 3 + 0] = src[i * 3 + 0];
    dst[ni * 3 + 1] = src[i * 3 + 1];
    dst[ni * 3 + 2] = src[i * 3 + 2];
}

// 哈希槽重映射：幸存体素槽改指新下标（counts_ 原值保留）；移除点所在槽
// 清键（体素整体摘除——后续同位点重新建格）
__global__ void remapHashKernel(
    unsigned long long* keys, unsigned int* fusedIdx,
    const unsigned int* __restrict__ newIdx, size_t capacity)
{
    const size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= capacity) return;
    if (keys[s] == 0) return;                 // 空槽（key+1 编码，0=空）
    const unsigned int ni = newIdx[fusedIdx[s]];
    if (ni == kLcfRemoved) keys[s] = 0;
    else                   fusedIdx[s] = ni;
}

ResultStatus LaserCloudFuseCuda::Impl::removePointsImpl(
    const std::vector<uint32_t>& indices, cudaStream_t stream)
{
    if (indices.empty()) return ResultStatus::ok();
    const unsigned int n = h_fusedPointCount_;
    if (n == 0) return ResultStatus::fail("removePoints: empty accumulator");

    // 去重排序＋越界校验（整批原子：任一越界 fail 状态不动）
    std::vector<uint32_t> sorted = indices;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    for (uint32_t i : sorted)
        if (i >= n) return ResultStatus::fail("removePoints: index out of range");
    if (sorted.size() >= n) {                 // 全删＝清空
        clearImpl(stream);
        CALIB_LOG_INFO("removePoints: all {} points removed", n);
        return ResultStatus::ok();
    }

    // host 建 old→new 映射（幸存按原序前移——确定性，无原子）
    std::vector<unsigned int> newIdx(n, kLcfRemoved);
    {
        unsigned int w = 0;
        for (unsigned int i = 0; i < n; ++i)
            if (!std::binary_search(sorted.begin(), sorted.end(), i))
                newIdx[i] = w++;
    }
    const unsigned int newCount = n - static_cast<unsigned int>(sorted.size());

    // 单 scratch 两轮复用（控显存峰值）；newIdx 上传
    const size_t ptsBytes = maxFusedPoints_ * 3 * sizeof(float);
    float* d_scratch = nullptr;
    unsigned int* d_newIdx = nullptr;
    if (cudaMalloc(&d_scratch, ptsBytes) != cudaSuccess) {
        return ResultStatus::fail("removePoints: scratch allocation failed");
    }
    if (cudaMalloc(&d_newIdx, n * sizeof(unsigned int)) != cudaSuccess) {
        cudaFree(d_scratch);
        return ResultStatus::fail("removePoints: map allocation failed");
    }
    cudaMemcpyAsync(d_newIdx, newIdx.data(), n * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, stream);

    const unsigned int gridPts = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t gridHash = (capacity_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t keptBytes = newCount * 3 * sizeof(float);

    // ① xyz 压缩 → scratch → 拷回常驻缓冲（DeviceContext 指针保持稳定）
    compactArrayKernel<<<gridPts, BLOCK_SIZE, 0, stream>>>(
        d_fusedXyz_, d_scratch, d_newIdx, n);
    cudaMemcpyAsync(d_fusedXyz_, d_scratch, keptBytes,
                    cudaMemcpyDeviceToDevice, stream);
    // ② normal 同映射压缩（点-法线关联保全）
    compactArrayKernel<<<gridPts, BLOCK_SIZE, 0, stream>>>(
        d_fusedNormal_, d_scratch, d_newIdx, n);
    cudaMemcpyAsync(d_fusedNormal_, d_scratch, keptBytes,
                    cudaMemcpyDeviceToDevice, stream);
    // ③ 哈希槽重映射（幸存改指/移除清键）
    remapHashKernel<<<gridHash, BLOCK_SIZE, 0, stream>>>(
        d_keys_, d_fusedIdx_, d_newIdx, capacity_);
    // ④ 设备端计数器置新值（编辑期无并发 fuse，非原子写安全）
    cudaMemcpyAsync(d_fusedPointCount_, &newCount, sizeof(unsigned int),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);
    cudaFree(d_scratch);
    cudaFree(d_newIdx);

    h_fusedPointCount_ = newCount;
    h_voxelCount_ = newCount;                // 体素↔代表点 1:1

    CALIB_LOG_INFO("removePoints: removed {} points, {} remain (voxels={})",
                   sorted.size(), newCount, h_voxelCount_);
    return ResultStatus::ok();
}
