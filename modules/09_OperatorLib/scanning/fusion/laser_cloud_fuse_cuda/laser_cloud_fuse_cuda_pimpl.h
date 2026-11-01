#pragma once

#include "laser_cloud_fuse_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

namespace calib {

// 【批F·fuse 段图化】图 memcpy 节点每帧现读的参数块（命名空间级——内核可见）
struct FuseParamsDev {
    const float* d_inputXyz; int inputCount;
    float R[9]; float T[3];
    float invVoxelSize; unsigned int threshold;
    unsigned long long* d_keys; unsigned int* d_fusedIdx; unsigned int* d_counts;
    unsigned long long capacity;
    float* d_fusedXyz; unsigned int maxFusedPoints;
    unsigned int* d_fusedPointCount;
    int* d_survivingFlags;
    int* d_newVoxelCount; int* d_survivingCount; int* d_deletedCount;
};

// 【N-5·自旋标签】单代理门铃包（命名空间级——内核可见）
struct FuseSpinPack {
    unsigned long long token;   // 门铃最后写（fence 后）
    int data[4];                // scalarPack 镜像（B2 四槽同构）
};

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
    size_t              capacity_    = 0;        // 槽位数（任意正整数，取余探测）
    // --- 融合点缓冲（常驻显存） ---
    float*        d_fusedXyz_        = nullptr;  // maxFusedPoints_ × 3
    float*        d_fusedNormal_     = nullptr;  // maxFusedPoints_ × 3（法线算子写入）
    size_t        maxFusedPoints_    = 0;

    // --- 【B2·16B 标量拼包】[0..2]=stats(newVoxel/surviving/deleted)、[3]=count ---
    // 设备端共址一块；构造期一次分配，终身不 realloc（rehash/freeAll 不触碰）。
    // 铁律：count 槽 [3] 跨帧原子累积，**永不在帧首 memset 范围内**（只有
    // clear/removePoints/allocateHash 显式写 [3]）——P2-0 脏读前科防线。
    int*  d_scalarPack_ = nullptr;          // 设备 16B
    int*  h_scalarPack_ = nullptr;          // pinned 镜像 16B（cudaHostAlloc）

    // --- 【批F·fuse 段图化】（JMW_FUSEGRAPH=1 实验臂默认关；NOGRAPH 优先）---
    // 4 节点：参数块 H2D（每帧现读 pinned 镜像）→memset 12B（stats 槽；count [3]
    // 永不入清零范围）→fuseKernelBlock（烧死 maxGrid＋kernel 内 inputCount 守卫
    // ——undistort :242 先例）→16B D2H。E 块 H2D 留图外（eSlot 池轮转每帧变，
    // 入图须每帧 SetParams 不赚账——R2 §四；【N-4】已改参数块 d_inputXyz 自愈）。
    // rehash/SetParams/Reserve/扩容触发 fuseNeedRecapture_——毁图重捕发生在
    // rehash 下一帧 Deferred（重捕窗流已空铁规）；签名失配亦全毁重建（【N-4】
    // 掩蔽 d_input——E 块槽轮换零重捕）。attempts>8 永久回退。
    bool fuseGraphArm_ = false;
    bool fuseGraphDisabled_ = false;
    int  fuseGraphAttempts_ = 0;
    int  fuseGraphWarm_ = 0;
    bool fuseNeedRecapture_ = false;
    cudaGraph_t fuseGraph_ = nullptr;
    cudaGraphExec_t fuseGraphExec_ = nullptr;
    FuseParamsDev* d_fuseParams_ = nullptr;   // 设备参数块（构造期一次）
    FuseParamsDev* h_fuseParams_ = nullptr;   // pinned 镜像（每帧 launch 前宿主写）

    // --- 【N-5·自旋标签】Finish 有界自旋替代 streamSync（JMW_FSPIN=0 显式关）---
    // 【单代理门铃 2026-10-28 二次设计】数据 D2H（拷贝引擎）与标志写（SM）跨
    // 代理 PCIe 写序无保证（标志先到＝宿主读陈旧 scalarPack，套内竞态实锤）→
    // 标签内核自带数据：读 d_scalarPack_ → __threadfence_system → 写 mapped
    // {data[4], token}——单代理写序，token 到达⟹数据可见。超时（50ms）回退
    // cudaStreamSynchronize 兜底（回退路径读 D2H 落地的 h_scalarPack_）。自旋
    // 与 1b 争核风险＝ABBA 已盖（NO-REGRESSION）
    FuseSpinPack* h_spinPack_ = nullptr;        // cudaHostAllocMapped
    FuseSpinPack* d_spinPack_ = nullptr;        // 设备侧别名
    unsigned long long* d_doneCounter_ = nullptr;   // 设备侧计数器（图节点参数烤死
                                                    // →运行期 atomicAdd 演进）
    unsigned long long launchToken_ = 0;        // 每次 Deferred 递增（宿主镜像）
    bool spinArm_ = false;                      // JMW_FSPIN（构造器读，K12）
    void emitDoneLabel_(cudaStream_t cs);       // 标签发射（图尾节点/逐发尾笔）
    struct FuseGraphSig {
        const float* d_input; int* d_flags;
        unsigned long long* keys; unsigned int* fusedIdx; unsigned int* counts;
        float* fusedXyz; int* scalarPack;
        unsigned long long capacity; unsigned int maxFusedPoints;
        int maxGrid; float voxelSize; unsigned int threshold;
    };
    FuseGraphSig fuseSig_{};
    bool fuseSigValid_ = false;
    void writeFuseParamsPinned_(const FuseGraphSig& s, int inputCount,
                                const float R[9], const float T[3],
                                float invVoxelSize, unsigned int threshold);
    bool buildFuseGraph_(cudaStream_t cs);
    void destroyFuseGraph_();

    // --- Host 镜像 ---
    unsigned int h_fusedPointCount_  = 0;   // ＝h_scalarPack_[3] 读法别名（消费端零改动）
    size_t       h_voxelCount_       = 0;

    // --- 【2601021 扩容改造】非破坏性扩容（负载>70% 固定增量 +growSlotCount） ---
    std::vector<RehashEvent> rehashEvents_;   // 扩容事件账本（含耗时）
    void rehashGrow(cudaStream_t stream);     // 分配新表搬移旧槽＋点缓冲扩容

    // --- 每帧临时缓冲（复用） ---
    int* d_survivingFlags_ = nullptr;   // inputCount
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

    // 【P0-b·滞后读】两段式：Deferred 只发射＋异步 D2H（不同步）；Finish 落地计数。
    // removePoints/clear 前自动 Finish 保证 h_* 镜像新鲜（产线编辑口防脏读）。
    LaserCloudFuseCudaResult executeDeferredImpl(const cv::cuda::GpuMat& d_points3d,
                  const cv::Matx33d& R, const cv::Vec3d& T,
                  cv::cuda::Stream& stream);
    LaserCloudFuseCudaResult finishImpl();
    bool  pendingDefer_    = false;
    cudaStream_t deferStream_ = nullptr;
    int   deferInput_      = 0;
    std::chrono::high_resolution_clock::time_point deferT0_;

    // --- removePoints（编辑账本，.cu 实现） ---
    ResultStatus removePointsImpl(const std::vector<uint32_t>& indices,
                                  cudaStream_t stream);

    // --- clear ---
    void clearImpl(cudaStream_t stream);
};

} // namespace calib
