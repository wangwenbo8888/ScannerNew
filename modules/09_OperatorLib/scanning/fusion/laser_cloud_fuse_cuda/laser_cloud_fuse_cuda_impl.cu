#include "laser_cloud_fuse_cuda_pimpl.h"
#include "common/calib_logging.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cmath>
#include <chrono>
#include <stdexcept>   // 261103 分配全检（失败抛出）
#include <string>

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
// 【批F·单源拆体】body 从参数块指针读全字段（图 Block 壳同源零漂移）；
// 逐参旧签名保留（legacy/eager 路径），栈上打包转发——编译期寄存器化。
// ============================================================

__device__ __forceinline__ void fuseKernelBody(const FuseParamsDev* __restrict__ p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->inputCount) return;   // 烧死 maxGrid 图壳的越界守卫（undistort :242 先例）

    // 1. 读取输入点
    const float* d_inputXyz = p->d_inputXyz;
    float px = d_inputXyz[i * 3 + 0];
    float py = d_inputXyz[i * 3 + 1];
    float pz = d_inputXyz[i * 3 + 2];

    // 2. R*p + T → 全局坐标
    float gx = p->R[0]*px + p->R[1]*py + p->R[2]*pz + p->T[0];
    float gy = p->R[3]*px + p->R[4]*py + p->R[5]*pz + p->T[1];
    float gz = p->R[6]*px + p->R[7]*py + p->R[8]*pz + p->T[2];

    // 3. 体素量化
    int ix = (int)floorf(gx * p->invVoxelSize);
    int iy = (int)floorf(gy * p->invVoxelSize);
    int iz = (int)floorf(gz * p->invVoxelSize);

    uint64_t key = packVoxelKeyDevice(ix, iy, iz);
    if (key == UINT64_MAX) {
        p->d_survivingFlags[i] = 0;
        atomicAdd(p->d_deletedCount, 1);
        return;
    }

    uint64_t storedKey = key + 1; // 0 保留为空槽

    // 4. 哈希 + atomicCAS 探测（取余定位——容量非 2 的幂也可用）
    uint64_t h = hash64Device(key);
    unsigned long long idx = h % p->capacity;

    int probes = 0;
    while (probes < 4096) {
        unsigned long long old = atomicCAS(
            (unsigned long long*)&p->d_keys[idx], 0ULL, storedKey);

        if (old == 0ULL || old == storedKey) {
            // === 新体素(old==0) 或 HIT(old==storedKey) ===
            // 统一计数：所有线程都用 atomicAdd，消除竞态
            if (old == 0ULL) {
                // 我们创建了此槽 — 写入代表点数据
                unsigned int fi = atomicAdd(p->d_fusedPointCount, 1u);
                if (fi >= p->maxFusedPoints) {
                    p->d_survivingFlags[i] = 0;
                    atomicAdd(p->d_deletedCount, 1);
                    return;
                }
                p->d_fusedXyz[fi * 3 + 0] = gx;
                p->d_fusedXyz[fi * 3 + 1] = gy;
                p->d_fusedXyz[fi * 3 + 2] = gz;
                p->d_fusedIdx[idx] = fi;
                atomicAdd(p->d_newVoxelCount, 1);
            }

            // 统一饱和判定
            unsigned int oldCount = atomicAdd(&p->d_counts[idx], 1u);
            if (oldCount < p->threshold) {
                p->d_survivingFlags[i] = 1;
                atomicAdd(p->d_survivingCount, 1);
            } else {
                p->d_survivingFlags[i] = 0;
                atomicAdd(p->d_deletedCount, 1);
            }
            return;
        }
        // 碰撞 — 线性探测（回绕）
        if (++idx >= p->capacity) idx = 0;
        probes++;
    }

    // 探测次数超限 — 表接近满
    p->d_survivingFlags[i] = 0;
    atomicAdd(p->d_deletedCount, 1);
}

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
    FuseParamsDev p;
    p.d_inputXyz = d_inputXyz; p.inputCount = inputCount;
    for (int k = 0; k < 9; ++k) p.R[k] = rt.R[k];
    for (int k = 0; k < 3; ++k) p.T[k] = rt.T[k];
    p.invVoxelSize = invVoxelSize; p.threshold = threshold;
    p.d_keys = d_keys; p.d_fusedIdx = d_fusedIdx; p.d_counts = d_counts;
    p.capacity = capacity;
    p.d_fusedXyz = d_fusedXyz; p.maxFusedPoints = maxFusedPoints;
    p.d_fusedPointCount = d_fusedPointCount;
    p.d_survivingFlags = d_survivingFlags;
    p.d_newVoxelCount = d_newVoxelCount;
    p.d_survivingCount = d_survivingCount;
    p.d_deletedCount = d_deletedCount;
    fuseKernelBody(&p);
}

// 【批F】图 Block 壳：从设备参数块直读（零参数漂移——与上内核同一 body）
__global__ void fuseKernelBlock(const FuseParamsDev* __restrict__ p) {
    fuseKernelBody(p);
}

// 【N-5·自旋标签】图尾标签内核（单代理门铃）：设备计数器 atomicAdd 演进（图
// 节点参数烤死、值运行期自增）；读 scalarPack → threadfence_system → 写 mapped
// {data, token}——token 最后写＝宿主见令牌⟹数据可见（跨代理 PCIe 写序无保证
// 的数据 D2H 门铃竞态解法，2026-10-28 套内竞态实锤后二次设计）
__global__ void fuseDoneLabelKernel(const int* __restrict__ scalarPack,
                                    unsigned long long* __restrict__ counter,
                                    FuseSpinPack* __restrict__ pack) {
    const unsigned long long v = atomicAdd(counter, 1ULL) + 1ULL;
    pack->data[0] = scalarPack[0];
    pack->data[1] = scalarPack[1];
    pack->data[2] = scalarPack[2];
    pack->data[3] = scalarPack[3];
    __threadfence_system();   // 数据写序先于门铃（单代理＝SM——序有效）
    pack->token = v;
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

    // 【B2·16B 标量拼包】设备块＋pinned 镜像，构造期一次终身不 realloc
    // （初始化全零＝count 从 0 起累积；此后 [3] 永不进帧首 memset 范围）
    // 261103 集成重放【分配全检】：pinned 池被会话期 eBlocks/RLE 环占满时
    // cudaHostAlloc 失败＝h_scalarPack_ null → Execute 读 null+0xC 崩
    {
        const cudaError_t e1 = cudaMalloc(&d_scalarPack_, 4 * sizeof(int));
        const cudaError_t e2 = e1 == cudaSuccess
            ? cudaMemset(d_scalarPack_, 0, 4 * sizeof(int)) : cudaSuccess;
        const cudaError_t e3 = e2 == cudaSuccess
            ? cudaHostAlloc((void**)&h_scalarPack_, 4 * sizeof(int), cudaHostAllocDefault)
            : cudaSuccess;
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
            if (d_scalarPack_) { cudaFree(d_scalarPack_); d_scalarPack_ = nullptr; }
            const char* s = e1 != cudaSuccess ? cudaGetErrorString(e1)
                          : e2 != cudaSuccess ? cudaGetErrorString(e2)
                                              : cudaGetErrorString(e3);
            throw std::runtime_error(std::string("scalarPack 分配失败: ") + s);
        }
        h_scalarPack_[0] = h_scalarPack_[1] = h_scalarPack_[2] = h_scalarPack_[3] = 0;
    }
    // 【批F】【批H·终态默认开】fuse 段图（JMW_FUSEGRAPH=0 显式关；NOGRAPH 优先）
    fuseGraphArm_ = [] {
        const char* ng = getenv("JMW_NOGRAPH");
        if (ng != nullptr && ng[0] == '1') return false;
        const char* e = getenv("JMW_FUSEGRAPH");
        return !(e != nullptr && e[0] == '0');
    }();
    // 261103 集成重放【分配全检】（同上——失败抛出，上层降级不崩）
    {
        const cudaError_t e4 = cudaMalloc(&d_fuseParams_, sizeof(FuseParamsDev));
        const cudaError_t e5 = e4 == cudaSuccess
            ? cudaHostAlloc((void**)&h_fuseParams_, sizeof(FuseParamsDev), cudaHostAllocDefault)
            : cudaSuccess;
        if (e4 != cudaSuccess || e5 != cudaSuccess) {
            if (d_fuseParams_) { cudaFree(d_fuseParams_); d_fuseParams_ = nullptr; }
            const char* s = e4 != cudaSuccess ? cudaGetErrorString(e4)
                                              : cudaGetErrorString(e5);
            throw std::runtime_error(std::string("fuseParams 分配失败: ") + s);
        }
    }
    // 【N-5·自旋标签】【默认关 2026-10-28 门裁定】mapped 单代理门铃包；JMW_FSPIN=1
    // 开（实验臂）。ABBA NO-REGRESSION＋env 臂 C2=23.63 已证收益，但套内
    // FuseScalarPack 间歇陈旧读（隔离恒过＝时序依赖残留——单代理门铃后仍有
    // 未明面）——按判决规则「该批翻关」：默认 off 保套稳定，臂随 env 交付
    spinArm_ = [] {
        const char* e = getenv("JMW_FSPIN");
        return e != nullptr && e[0] == '1';
    }();
    if (spinArm_) {
        if (cudaHostAlloc((void**)&h_spinPack_, sizeof(FuseSpinPack),
                          cudaHostAllocMapped) == cudaSuccess
            && cudaHostGetDevicePointer((void**)&d_spinPack_, h_spinPack_, 0)
                   == cudaSuccess
            && cudaMalloc(&d_doneCounter_, sizeof(unsigned long long)) == cudaSuccess) {
            h_spinPack_->token = 0;
            cudaMemset(d_doneCounter_, 0, sizeof(unsigned long long));
        } else {
            h_spinPack_ = nullptr;   // mapped 不可用（UVA 全支持——理论不现）→臂回退
            d_spinPack_ = nullptr;
            if (d_doneCounter_) { cudaFree(d_doneCounter_); d_doneCounter_ = nullptr; }
            spinArm_ = false;
            cudaGetLastError();
        }
    }
}

LaserCloudFuseCuda::Impl::~Impl() {
    cudaDeviceSynchronize();
    freeAll();
    destroyFuseGraph_();
    safeCudaFree(d_scalarPack_);
    if (h_scalarPack_) { cudaFreeHost(h_scalarPack_); h_scalarPack_ = nullptr; }
    if (d_fuseParams_) { cudaFree(d_fuseParams_); d_fuseParams_ = nullptr; }
    if (h_fuseParams_) { cudaFreeHost(h_fuseParams_); h_fuseParams_ = nullptr; }
    if (h_spinPack_) { cudaFreeHost(h_spinPack_); h_spinPack_ = nullptr; }
    if (d_doneCounter_) { cudaFree(d_doneCounter_); d_doneCounter_ = nullptr; }
}

// ============================================================
// Impl — 内存分配
// ============================================================

void LaserCloudFuseCuda::Impl::allocateHash(size_t voxelCount) {
    size_t want = voxelCount > 64 ? voxelCount : 64;
    if (want <= capacity_) return; // 不缩容

    freeAll();

    capacity_ = want;

    // 261103 集成重放【分配全检】：裸 cudaMalloc 失败＝null 指针流入内核
    // ＝null+0xC 访问冲突（exe 全会话显存/pinned 压力下必现——GBA 重放双
    // fuse 并存 ~4GB；沙盒新鲜进程测不出）。失败＝抛出→上层 catch 降级
    {
        const cudaError_t e1 = cudaMalloc(&d_keys_,     capacity_ * sizeof(unsigned long long));
        const cudaError_t e2 = cudaMalloc(&d_fusedIdx_, capacity_ * sizeof(unsigned int));
        const cudaError_t e3 = cudaMalloc(&d_counts_,   capacity_ * sizeof(unsigned int));
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
            safeCudaFree(d_keys_); d_keys_ = nullptr;
            safeCudaFree(d_fusedIdx_); d_fusedIdx_ = nullptr;
            safeCudaFree(d_counts_); d_counts_ = nullptr;
            capacity_ = 0;
            const char* s = e1 != cudaSuccess ? cudaGetErrorString(e1)
                          : e2 != cudaSuccess ? cudaGetErrorString(e2)
                                              : cudaGetErrorString(e3);
            throw std::runtime_error(std::string("allocateHash(")
                                     + std::to_string(want) + " 槽) 失败: " + s);
        }
    }

    cudaMemset(d_keys_,     0, capacity_ * sizeof(unsigned long long));
    cudaMemset(d_fusedIdx_, 0, capacity_ * sizeof(unsigned int));
    cudaMemset(d_counts_,   0, capacity_ * sizeof(unsigned int));

    // 重置计数槽（scalarPack[3]；【B2】拼包块终身不 realloc，只清 count 槽位）
    if (d_scalarPack_) {
        cudaMemset(d_scalarPack_ + 3, 0, sizeof(int));
        h_scalarPack_[3] = 0;
    }
    h_fusedPointCount_ = 0;
    h_voxelCount_ = 0;
    fuseNeedRecapture_ = true;   // 【批F】freeAll 重建指针——签名必失配，下一帧重捕
}

void LaserCloudFuseCuda::Impl::allocatePoints(size_t maxPoints) {
    if (maxPoints <= maxFusedPoints_) return;
    safeCudaFree(d_fusedXyz_);
    safeCudaFree(d_fusedNormal_);

    maxFusedPoints_ = maxPoints;
    // 261103 集成重放【分配全检】（同 allocateHash——d_fusedXyz/Normal 600MB×2
    // 级，显存不足时首当其冲；null 流入下载口＝终版云崩溃）
    {
        const cudaError_t e1 = cudaMalloc(&d_fusedXyz_,    maxPoints * 3 * sizeof(float));
        const cudaError_t e2 = cudaMalloc(&d_fusedNormal_, maxPoints * 3 * sizeof(float));
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            safeCudaFree(d_fusedXyz_); d_fusedXyz_ = nullptr;
            safeCudaFree(d_fusedNormal_); d_fusedNormal_ = nullptr;
            maxFusedPoints_ = 0;
            const char* s = e1 != cudaSuccess ? cudaGetErrorString(e1)
                                              : cudaGetErrorString(e2);
            throw std::runtime_error(std::string("allocatePoints(")
                                     + std::to_string(maxPoints) + " 点) 失败: " + s);
        }
    }
    cudaMemset(d_fusedXyz_,    0, maxPoints * 3 * sizeof(float));
    cudaMemset(d_fusedNormal_, 0, maxPoints * 3 * sizeof(float));
}

void LaserCloudFuseCuda::Impl::ensureTempBuffers(size_t inputCount) {
    if (inputCount <= lastInputCount_ && d_survivingFlags_) return;
    safeCudaFree(d_survivingFlags_);
    fuseNeedRecapture_ = true;   // 【批F】d_survivingFlags 重分配＋maxGrid 变——重捕

    size_t allocCount = inputCount > 0 ? inputCount : 1;
    cudaMalloc(&d_survivingFlags_, allocCount * sizeof(int));
    lastInputCount_ = allocCount;
}

void LaserCloudFuseCuda::Impl::freeAll() {
    safeCudaFree(d_keys_);
    safeCudaFree(d_fusedIdx_);
    safeCudaFree(d_counts_);
    safeCudaFree(d_fusedXyz_);
    safeCudaFree(d_fusedNormal_);
    safeCudaFree(d_survivingFlags_);
    // 【B2】d_scalarPack_/h_scalarPack_ 不在此释放——构造期一次终身持有
    // （allocateHash 重入 freeAll 时拼包存活，仅 count 槽由调用方显式清）
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

// ──【批F】fuse 段图管理─────────────────────────────────────
void LaserCloudFuseCuda::Impl::destroyFuseGraph_() {
    if (fuseGraphExec_) { cudaGraphExecDestroy(fuseGraphExec_); fuseGraphExec_ = nullptr; }
    if (fuseGraph_) { cudaGraphDestroy(fuseGraph_); fuseGraph_ = nullptr; }
    fuseSigValid_ = false;
}

void LaserCloudFuseCuda::Impl::writeFuseParamsPinned_(
    const FuseGraphSig& s, int inputCount, const float R[9], const float T[3],
    float invVoxelSize, unsigned int threshold) {
    FuseParamsDev& p = *h_fuseParams_;
    p.d_inputXyz = s.d_input;
    p.inputCount = inputCount;
    for (int k = 0; k < 9; ++k) p.R[k] = R[k];
    for (int k = 0; k < 3; ++k) p.T[k] = T[k];
    p.invVoxelSize = invVoxelSize;
    p.threshold = threshold;
    p.d_keys = s.keys;
    p.d_fusedIdx = s.fusedIdx;
    p.d_counts = s.counts;
    p.capacity = s.capacity;
    p.d_fusedXyz = s.fusedXyz;
    p.maxFusedPoints = s.maxFusedPoints;
    p.d_fusedPointCount = reinterpret_cast<unsigned int*>(d_scalarPack_ + 3);
    p.d_survivingFlags = s.d_flags;
    p.d_newVoxelCount = d_scalarPack_ + 0;
    p.d_survivingCount = d_scalarPack_ + 1;
    p.d_deletedCount = d_scalarPack_ + 2;
}

bool LaserCloudFuseCuda::Impl::buildFuseGraph_(cudaStream_t cs) {
    if (cudaStreamSynchronize(cs) != cudaSuccess) return false;   // 捕获前排空（WDDM 铁规）
    cudaGraph_t g = nullptr;
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        return false;
    bool ok = true;
    // ① 参数块 H2D（每帧现读 pinned——src 地址恒定）
    if (cudaMemcpyAsync(d_fuseParams_, h_fuseParams_, sizeof(FuseParamsDev),
                        cudaMemcpyHostToDevice, cs) != cudaSuccess) ok = false;
    // ② memset 12B——只清 stats 槽；count [3] 绝不在范围（P2-0 前科；严禁 kernel 自清）
    if (ok && cudaMemsetAsync(d_scalarPack_, 0, 3 * sizeof(int), cs) != cudaSuccess) ok = false;
    // ③ 融合内核 Block 壳（烧死 maxGrid 网格＋kernel 内 inputCount 守卫）
    if (ok) {
        const int grid = (fuseSig_.maxGrid + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fuseKernelBlock<<<grid, BLOCK_SIZE, 0, cs>>>(d_fuseParams_);
    }
    // ④ 16B D2H（B2 拼包）
    if (ok && cudaMemcpyAsync(h_scalarPack_, d_scalarPack_, 4 * sizeof(int),
                              cudaMemcpyDeviceToHost, cs) != cudaSuccess) ok = false;
    // ⑤ 【N-5】图尾标签节点（单代理门铃——数据随标签走；计数器演进式）
    if (ok && spinArm_ && d_spinPack_)
        fuseDoneLabelKernel<<<1, 1, 0, cs>>>(d_scalarPack_, d_doneCounter_,
                                             d_spinPack_);
    cudaError_t endErr = cudaStreamEndCapture(cs, &g);
    if (!ok || endErr != cudaSuccess || !g) {
        if (g) cudaGraphDestroy(g);
        return false;
    }
    cudaGraphExec_t ge = nullptr;
    if (cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0) != cudaSuccess || !ge) {
        if (ge) cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        return false;
    }
    destroyFuseGraph_();   // 覆盖旧图（若有）
    fuseGraph_ = g;
    fuseGraphExec_ = ge;
    fuseSigValid_ = true;
    return true;
}

// ============================================================
// 【N-5·自旋标签】标签发射（逐发路径尾笔；图路径由图尾节点承担）＋令牌镜像
// ============================================================
void LaserCloudFuseCuda::Impl::emitDoneLabel_(cudaStream_t cs) {
    fuseDoneLabelKernel<<<1, 1, 0, cs>>>(d_scalarPack_, d_doneCounter_, d_spinPack_);
    ++launchToken_;
}

LaserCloudFuseCudaResult LaserCloudFuseCuda::Impl::fuseImpl(
    const cv::cuda::GpuMat& d_points3d,
    const cv::Matx33d& R, const cv::Vec3d& T,
    cv::cuda::Stream& stream)
{
    // 【P0-b】同步路径＝两段式组合（07/产线/serial 调用方行为逐位不变）
    auto head = executeDeferredImpl(d_points3d, R, T, stream);
    if (!head.success) return head;
    return finishImpl();
}

// —— 段一：校验＋发射＋异步 D2H（不同步；计数返回 -1 语义）——
LaserCloudFuseCudaResult LaserCloudFuseCuda::Impl::executeDeferredImpl(
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

    deferT0_ = std::chrono::high_resolution_clock::now();

    cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);

    // 确保临时缓冲够大
    ensureTempBuffers(static_cast<size_t>(inputCount));

    // R/T → float struct（按值传递给 kernel）
    RTParams rt;
    rt.R[0] = (float)R(0,0); rt.R[1] = (float)R(0,1); rt.R[2] = (float)R(0,2);
    rt.R[3] = (float)R(1,0); rt.R[4] = (float)R(1,1); rt.R[5] = (float)R(1,2);
    rt.R[6] = (float)R(2,0); rt.R[7] = (float)R(2,1); rt.R[8] = (float)R(2,2);
    rt.T[0] = (float)T(0); rt.T[1] = (float)T(1); rt.T[2] = (float)T(2);

    const float* d_input = d_points3d.ptr<float>();
    float invVoxelSize = 1.0f / params_.voxelSize;
    unsigned int threshold = static_cast<unsigned int>(params_.saturationThreshold);

    // ──【批F】fuse 段图臂（4 节点单发射；失败/失配/暖机回退下方逐发）──
    if (fuseGraphArm_ && !fuseGraphDisabled_ && ++fuseGraphWarm_ >= 2) {
        // rehash/SetParams/Reserve/扩容 触发的下一帧毁图重捕（重捕窗流已空铁规）
        if (fuseNeedRecapture_) { destroyFuseGraph_(); fuseNeedRecapture_ = false; }
        FuseGraphSig cur{};
        cur.d_input = d_input;
        cur.d_flags = d_survivingFlags_;
        cur.keys = d_keys_; cur.fusedIdx = d_fusedIdx_; cur.counts = d_counts_;
        cur.fusedXyz = d_fusedXyz_; cur.scalarPack = d_scalarPack_;
        cur.capacity = capacity_;
        cur.maxFusedPoints = static_cast<unsigned int>(maxFusedPoints_);
        cur.maxGrid = static_cast<int>(lastInputCount_);   // ≥inputCount（ensure 后）
        cur.voxelSize = params_.voxelSize; cur.threshold = threshold;
        if (fuseSigValid_) {
            // 【N-4·E 块零拷贝】【2026-10-30 根因②修复 B'】d_input 经参数块刷新
            // （fuseKernelBlock 壳现读 p.d_inputXyz——launch 时从 pinned 取当帧值）
            // →签名比对掩蔽之：E 块槽轮换（mapped 设备指针轮转）零毁图重捕。
            // ⚠历史缺陷：稳态曾误传 fuseSig_（建图帧旧槽指针）→内核每帧读建图
            // 槽，被复写后即"帧 N 位姿×帧 M 点"（3-50mm 散布，跑间波动）——
            // 修复＝:596 改传 cur（当帧指针）。其余字段（flags/容量/表指针/
            // maxGrid/参数）失配仍全毁——结构/参数变化防线不变
            FuseGraphSig a = cur, b = fuseSig_;
            a.d_input = nullptr;
            b.d_input = nullptr;
            if (std::memcmp(&a, &b, sizeof(a)) != 0)
                destroyFuseGraph_();
        }
        if (fuseGraphExec_ && fuseSigValid_) {
            // 【根因②修复 B'】必须传 cur（当帧输入指针）——传 fuseSig_ 会让
            // 参数块永远指向建图帧的 E 槽（确定性陈旧指针，非竞态）
            writeFuseParamsPinned_(cur, inputCount, rt.R, rt.T, invVoxelSize, threshold);
            if (cudaGraphLaunch(fuseGraphExec_, s) == cudaSuccess) {
                pendingDefer_ = true; deferStream_ = s; deferInput_ = inputCount;
                if (spinArm_ && d_spinPack_) ++launchToken_;   // 【N-5】图尾标签节点在飞
                result.success = true;
                result.message = "Frame fusion deferred (CUDA graph)";
                result.inputCount = inputCount;
                result.totalVoxelCount = -1;
                return result;
            }
            CALIB_LOG_WARN("fuse graphLaunch 失败回退逐发: {}",
                           cudaGetErrorString(cudaGetLastError()));
            destroyFuseGraph_();
            fuseGraphDisabled_ = true;   // 稳态发射失败＝永久回退
        } else {
            fuseSig_ = cur;
            writeFuseParamsPinned_(fuseSig_, inputCount, rt.R, rt.T, invVoxelSize, threshold);
            if (buildFuseGraph_(s) && cudaGraphLaunch(fuseGraphExec_, s) == cudaSuccess) {
                pendingDefer_ = true; deferStream_ = s; deferInput_ = inputCount;
                if (spinArm_ && d_spinPack_) ++launchToken_;   // 【N-5】图尾标签节点在飞
                result.success = true;
                result.message = "Frame fusion deferred (CUDA graph)";
                result.inputCount = inputCount;
                result.totalVoxelCount = -1;
                return result;   // 捕获不执行当帧——launch 即补真跑
            }
            (void)cudaGetLastError();
            destroyFuseGraph_();
            if (++fuseGraphAttempts_ > 8) fuseGraphDisabled_ = true;
        }
    }

    // 清零统计——【B2】只清 scalarPack 前 12B（stats 槽）；
    // count 槽 [3] 跨帧累积，绝不能进 memset 范围（P2-0 脏读前科）
    cudaMemsetAsync(d_scalarPack_, 0, 3 * sizeof(int), s);

    int grid = (inputCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fuseKernel<<<grid, BLOCK_SIZE, 0, s>>>(
        d_input, inputCount,
        rt,
        invVoxelSize, threshold,
        d_keys_, d_fusedIdx_, d_counts_, (unsigned long long)capacity_,
        d_fusedXyz_, static_cast<unsigned int>(maxFusedPoints_),
        reinterpret_cast<unsigned int*>(d_scalarPack_ + 3),   // 【B2】count 共址槽
        d_survivingFlags_,
        d_scalarPack_ + 0,   // newVoxelCount
        d_scalarPack_ + 1,   // survivingCount
        d_scalarPack_ + 2);  // deletedCount

    // 读回标量（异步入队；落地推迟到 Finish——滞后读协议）
    // 【B2】两笔拼一笔：16B pinned D2H（可分页双笔→pinned 单笔，顺带消
    // cudart 分页 staging 的内部路由杂散）
    cudaMemcpyAsync(h_scalarPack_, d_scalarPack_,
                    4 * sizeof(int), cudaMemcpyDeviceToHost, s);
    // 【N-5】逐发路径尾笔标签（图路径由图尾节点承担——两路每 Deferred 恰一笔）
    if (spinArm_ && d_spinPack_) emitDoneLabel_(s);

    pendingDefer_  = true;
    deferStream_   = s;
    deferInput_    = inputCount;
    result.success = true;
    result.message = "Frame fusion deferred (CUDA)";
    result.inputCount    = inputCount;
    result.totalVoxelCount = -1;   // 语义：计数未落地，Finish 后有效
    return result;
}

// —— 段二：同步＋读 pinned＋rehash 判定＋质量旗（滞后一帧调用亦可）——
LaserCloudFuseCudaResult LaserCloudFuseCuda::Impl::finishImpl()
{
    LaserCloudFuseCudaResult result;
    if (!pendingDefer_) {
        result.success = false;
        result.message = "Finish: no pending deferred frame";
        return result;
    }
    pendingDefer_ = false;

    // 【N-5·自旋标签】有界自旋替代 streamSync（单代理门铃：token 到达⟹数据可
    // 见——标签内核 threadfence_system 写序）；超时（50ms≈Finish 等待 p99×5
    // 量级）回退 cudaStreamSynchronize 兜底（回退路径读 D2H 落地的 h_scalarPack_，
    // 正确性保持）。自旋与 1b 争核风险＝ABBA 已盖（yield 式放让超线程兄弟）
    if (spinArm_ && h_spinPack_) {
        const unsigned long long expect = launchToken_;
        volatile const FuseSpinPack* pack = h_spinPack_;
        const auto t0 = std::chrono::high_resolution_clock::now();
        bool done = (pack->token >= expect);   // 先查后等（快路径零自旋）
        while (!done) {
            if (pack->token >= expect) { done = true; break; }
            if (std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - t0).count()
                > 50000000LL) break;   // 50ms 超时——回退 sync 兜底
            std::this_thread::yield();
        }
        if (done) {
            // 门铃数据落地（下游 h_scalarPack_ 读法别名零改动）
            h_scalarPack_[0] = pack->data[0];
            h_scalarPack_[1] = pack->data[1];
            h_scalarPack_[2] = pack->data[2];
            h_scalarPack_[3] = pack->data[3];
        } else {
            cudaStreamSynchronize(deferStream_);
        }
    } else {
        cudaStreamSynchronize(deferStream_);
    }

    const int inputCount = deferInput_;
    auto totalEnd = std::chrono::high_resolution_clock::now();

    // 【B2】落地 16B 拼包（pinned 已同步）——h_fusedPointCount_ 保持读法别名
    h_fusedPointCount_ = static_cast<unsigned int>(h_scalarPack_[3]);

    // 填充结果
    result.success       = true;
    result.message       = "Frame fusion completed (CUDA)";
    result.inputCount    = inputCount;
    result.newVoxelCount = h_scalarPack_[0];
    result.survivingCount= h_scalarPack_[1];
    result.deletedCount  = h_scalarPack_[2];
    result.totalVoxelCount = static_cast<int>(h_fusedPointCount_);
    result.totalTimeMs   = std::chrono::duration<double, std::milli>(totalEnd - deferT0_).count();

    // 【2601021 扩容改造】负载 >70% → 非破坏性固定增量扩容（耗时单独入账，
    // 不计入融合耗时；laser_fuse_stress 压测：1000 万槽搬移约 17.7ms）
    // 【P0-b】滞后读：扩容判定随 Finish 滞后一帧（voxelReserve 12M vs 峰值
    // 5.7M 双倍余量，audit 裁决可接受）
    result.hashCapacity = capacity_;
    if ((double)h_fusedPointCount_ > LOAD_FACTOR * (double)capacity_) {
        rehashGrow(deferStream_);
        result.rehashHappened = true;
        result.rehashTimeMs   = rehashEvents_.back().timeMs;
        result.hashCapacity   = capacity_;
        fuseNeedRecapture_ = true;   // 【批F】rehash 毁图重捕守卫（下一帧 Deferred）
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
    if (pendingDefer_) finishImpl();   // 【P0-b】滞后读新鲜度守卫
    if (!d_keys_) return;
    size_t grid = (capacity_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clearKeysKernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_keys_, capacity_);
    cudaMemsetAsync(d_counts_, 0, capacity_ * sizeof(unsigned int), stream);
    cudaMemsetAsync(d_fusedIdx_, 0, capacity_ * sizeof(unsigned int), stream);
    cudaMemsetAsync(d_fusedXyz_, 0, maxFusedPoints_ * 3 * sizeof(float), stream);
    cudaMemsetAsync(d_fusedNormal_, 0, maxFusedPoints_ * 3 * sizeof(float), stream);
    cudaMemsetAsync(d_scalarPack_ + 3, 0, sizeof(int), stream);   // 【B2】count 共址槽
    cudaStreamSynchronize(stream);
    h_scalarPack_[3] = 0;
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
    if (pendingDefer_) finishImpl();   // 【P0-b】滞后读新鲜度守卫（产线编辑口防脏计数）
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
    // ④ 设备端计数器置新值（编辑期无并发 fuse，非原子写安全）——【B2】count 共址槽
    cudaMemcpyAsync(d_scalarPack_ + 3, &newCount, sizeof(unsigned int),
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
