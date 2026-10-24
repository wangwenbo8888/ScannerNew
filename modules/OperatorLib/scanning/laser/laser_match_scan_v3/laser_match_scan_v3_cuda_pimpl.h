/**
 * @file laser_match_scan_v3_cuda_pimpl.h
 * @brief 激光线匹配扫描CUDA算子 V2 — 内部桥接头文件（声明 struct Impl 完整结构）
 *
 * 本文件包含 CUDA 类型，仅供内部实现使用。
 * .cpp 和 .cu 均 include 此文件。
 */

#pragma once

#include <opencv2/core/cuda.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <atomic>
#include <vector>
#include <string>

#include <cuda_runtime.h>   // float2/cudaMallocHost（内部头，.cpp/.cu 均可用）

#include "laser_match_scan_v3_cuda.h"

namespace calib {

struct K2Ctx;   // 【v3全GPU化·K2】级联内核缓冲池（impl.cu 内定义）

struct LaserMatchScanCudaV3::Impl {
    static constexpr int MAX_EPIPOLAR_ROWS = 8192;
    static constexpr int MAX_LOOKUP_CANDS = 64;       // 每(行,x)表候选上限（同 v1）
    static constexpr int SMEM_RIGHT_CAP = 256;        // kernelMatchV2 smem 每行右点容量(4KB)；超出置 overflow 标志
    static constexpr unsigned int KEY_SENTINEL = 0xFFFFFFFFu;  // row>=8192 丢弃标记

    LaserMatchScanParamsV3 params_;

    struct TempTableEntry {
        double temperature = 0.0;
        std::vector<float> mapData;
        int entryCount = 0;
        int numLines = 0;
    };
    std::vector<TempTableEntry> tempTable_;
    bool tableLoaded_ = false;

    cv::cuda::GpuMat d_map_table_;
    cv::cuda::GpuMat d_map_line_start_;
    cv::cuda::GpuMat d_map_line_count_;
    cv::cuda::GpuMat d_map_byrow_;          // table re-sorted by (row,xL), CV_32FC4 (xL,uR,lineId,yL)
    cv::cuda::GpuMat d_map_row_start_;       // CSR by epipolar row (MAX_EPIPOLAR_ROWS, CV_32SC1)
    cv::cuda::GpuMat d_map_row_count_;       // CSR row counts (MAX_EPIPOLAR_ROWS, CV_32SC1)
    // 【150FPS·查表桶索引】发卡内核行内二分（13 步串行依赖随机读，冷缓存实测
    // +2.6ms/帧）→ 每行 K 桶下界直达＋短跳走。池 ≈1.35MB 常驻 L2。
    cv::cuda::GpuMat d_bucket_pool_;         // 活跃行 ×(K+1) 桶下界（相对行首偏移，CV_32SC1 平铺）
    cv::cuda::GpuMat d_bucket_row_off_;      // 每行池偏移（MAX_EPIPOLAR_ROWS；空行 0）
    float bucketMinX_ = 0.f, bucketInvW_ = 0.f;
    int bucketK_ = 0;                        // 0＝未建（内核走二分兜底）
    static constexpr int kBucketK = 256;     // 桶数/行（1162px 域 → 4.5px/桶 ≈26 条）
    // 【B3·基建】map/桶池重传代际：每次真实 uploadMapTable（!cacheHit 重传，含
    // 桶池重建）末尾 +1；cacheHit 短路不 +（同表同温同 rowStep 重注入 epoch 不变）。
    // B4 图化签名消费口（MapEpoch() 门面转发）
    unsigned long long mapEpoch_ = 0;

    // ============================================================================
    // 【150FPS·发射图化】WDDM 变尺寸发射开销实测占 g1 的 ~2.9ms（逐帧点数变化→
    // 驱动无法复用命令缓冲，~130µs/发射）。对策：scatter→CSR→发卡五连发捕获为
    // CUDA Graph，逐帧 cudaGraphExecKernelNodeSetParams 刷网格/计数/代际指针后
    // 单次 graphLaunch（~10µs）。内核零改动→语义逐位一致；JMW_V3GRAPH=0 兜底。
    // ============================================================================
    struct FGraph {
        bool captured = false;
        bool disabled = false;               // 捕获/更新失败→永久回退普通发射
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t nScatter = nullptr, nCSR = nullptr, nIssue = nullptr;
        cudaGraphNode_t nMsetRS = nullptr, nMsetRC = nullptr, nMsetSt = nullptr;
        // —— 稳定参数存储（kernelParams 指向这些字段，逐帧改值后 SetParams）——
        // scatter
        const float2* sc_rightPts; int sc_rightCount; int* sc_sortedIdx;
        float2* sc_outPts; int* sc_outLids; const int* sc_rightLids;
        // CSR
        const unsigned int* csr_keys; int csr_rightCount; int csrMaxRows = 0;
        int* csr_rowStart; int* csr_rowCount;
        // memset×3
        void* msRS_ptr; size_t msRS_bytes;
        void* msRC_ptr; size_t msRC_bytes;
        void* msSt_ptr; size_t msSt_bytes;
        // issue（26 参；指针类逐帧可变：左右点/gen 缓冲）
        const float2* ic_leftPts; int ic_leftCount;
        const float4* ic_mapByrow; const int* ic_rowStart; const int* ic_rowCount;
        const float2* ic_rightSorted; const int* ic_rRowStart; const int* ic_rRowCount;
        float ic_rowStep; float ic_vLtol; float ic_matchWin;
        int* ic_lid; float* ic_hitRx; float* ic_res; int* ic_hitRidx; int* ic_count;
        const float* ic_urBias; const float4* ic_biasParams;
        int* ic_leftStatus; int* ic_maxOut;
        const int* ic_bucketPool; const int* ic_bucketRowOff;
        float ic_bMinX; float ic_bInvW; int ic_bK; int ic_useB;
        const int* ic_dN = nullptr;   // 【B3·R6 修】d_n 尾闸第 27 参（process 精确 N 路恒 nullptr）
        // 参数指针数组（构造一次）
        void* scArgs[6]; void* csrArgs[5]; void* icArgs[27];   // 【B3】26→27（对齐内核 27 参——修复既有错位）
        cudaKernelNodeParams scP{}, csrP{}, icP{};
        cudaMemsetParams msRsP{}, msRcP{}, msStP{};
    } fg_;

    // 【150FPS·Run D 扩图】段表回填→投票→选举→exclCount→P6prep→段偏移
    // （13 节点线性；5 个逐帧变参内核 SetParams，8 个固定节点零更新）
    struct FGraphD {
        int attempts = 0;
        bool captured = false, disabled = false;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaGraphNode_t nMapBack{}, nVote{}, nExcl{}, nPtRow{}, nSegOff{};
        // SegMapBack(6)
        int* mb_runrank; int* mb_flags; int* mb_sortedIdx; int mb_leftCount;
        int* mb_map; int* mb_counts2;
        void* mbArgs[6]; cudaKernelNodeParams mbP{};
        // SegVote(8)
        int* sv_map; int* sv_lid; float* sv_res; int* sv_cnt; int sv_leftCount;
        int sv_segCap; int* sv_votes; int* sv_sres;
        void* svArgs[8]; cudaKernelNodeParams svP{};
        // ExclCount(8)
        int* ec_map; int* ec_winner; int* ec_second; int* ec_lid; int* ec_cnt;
        int ec_leftCount; int ec_segCap; int* ec_excl;
        void* ecArgs[8]; cudaKernelNodeParams ecP{};
        // PtRowSegRows(9)
        const float2* pr_pts; int* pr_map; int pr_leftCount; int pr_segCap;
        float pr_rowStep; int* pr_ptRow; unsigned long long* pr_segRows;
        int* pr_counts2; int* pr_viol;
        void* prArgs[9]; cudaKernelNodeParams prP{};
        // SegOffsets(4)
        int* so_flags; int* so_runrank; int so_leftCount; int* so_offsets;
        void* soArgs[4]; cudaKernelNodeParams soP{};
    } fgD_;
    int activeMapCount_ = 0;
    int activeNumLines_ = 0;
    double currentTemp_ = 25.0;
    bool tempSet_ = false;

    // ── 温表注备缓存（全链速度优化 2026-10-16）：同（源表指针, 档温, rowStep）
    //    重复注入 SetTempTable/SetCurrentTemperature 时跳过 O(N) 转换与 140MB×2
    //    上传＋host 分桶排序——逐帧注表的调用方（harness/07 流水线）零成本过闸。
    //    三元组一致 ⇔ uploadMapTable 输入逐位一致 ⇔ GPU 表状态逐位一致（无精度影响）。
    std::shared_ptr<const LaserPlaneMapTempTable> srcHold_;  // 持引用防指针复用误判
    const void* uploadedSrc_ = nullptr;   // 已上传至 GPU 的源表指针（null=无效）
    double uploadedTemp_ = 0.0;           // 已上传档温
    float uploadedRowStep_ = -1.0f;       // 已上传时 rowStep（setParams 改 rowStep 置失效）

    // ── 排序链（32bit 键：row<<19 | xq@1/256px）──
    cv::cuda::GpuMat d_left_keys_in_;        // CV_32SC1 (unsigned int)
    cv::cuda::GpuMat d_left_keys_out_;
    cv::cuda::GpuMat d_right_keys_in_;
    cv::cuda::GpuMat d_right_keys_out_;
    cv::cuda::GpuMat d_left_idx_in_;         // keygen 顺带写 sequence
    cv::cuda::GpuMat d_right_idx_in_;
    cv::cuda::GpuMat d_left_sorted_idx_;
    cv::cuda::GpuMat d_right_sorted_idx_;
    cv::cuda::GpuMat d_left_sorted_pts_;     // CV_32FC2
    cv::cuda::GpuMat d_left_sorted_lids_;    // CV_32SC1
    cv::cuda::GpuMat d_right_sorted_pts_;
    cv::cuda::GpuMat d_right_sorted_lids_;

    cv::cuda::GpuMat d_left_row_start_;      // CSR (MAX_EPIPOLAR_ROWS)
    cv::cuda::GpuMat d_left_row_count_;
    cv::cuda::GpuMat d_right_row_start_;
    cv::cuda::GpuMat d_right_row_count_;

    // ── 查表产物 ──
    // V3 card table (orig space; nearest card per lid)
cv::cuda::GpuMat d_card_lid_;           // leftCount*V3_MAX_CARDS, CV_32SC1
cv::cuda::GpuMat d_card_hitRx_;         // leftCount*V3_MAX_CARDS, CV_32FC1
cv::cuda::GpuMat d_card_count_;         // leftCount, CV_32SC1
cv::cuda::GpuMat d_card_res_;           // leftCount*V3_MAX_CARDS, CV_32FC1（|hitRx-uR|，P3/诊断用）
cv::cuda::GpuMat d_card_hitRidx_;       // leftCount*V3_MAX_CARDS, CV_32SC1（右排序系索引，P4 右占用）
cv::cuda::GpuMat d_seg_res_;            // segCount*V3_MAX_LIDS, CV_32SC1（段内残差定点累加 1/256px）
    cv::cuda::GpuMat d_diag_;               // 4, CV_32SC1（d_diag_[0]=保留段无 winner 卡点数）
    cv::cuda::GpuMat d_ur_bias_;            // V3_MAX_LIDS, CV_32FC1（逐线 uR 补偿；空＝全零）
    bool urBiasActive_ = false;
    cv::cuda::GpuMat d_bias_params_;        // 1x1 CV_32FC4（wx,wy,wz,dtx；4 参数模型激活时 kernel 消费）
    bool urBiasModelActive_ = false;        // 与 urBiasActive_ 互斥（后设者生效）
    // V3 segment pipeline
    cv::cuda::GpuMat d_seg_map_;            // leftCount, CV_32SC1 (orig -> segIdx)
    // 【刀二A+B2 两步法】标签桶分类：域＝interp 簇 label 编码 row*SCAN_SLOTS+slot
    // 上界 262144。hist 帧首 memset 承重（Rank 快径不自清洁）；末槽 262145＝
    // listCnt（memset 一并归零——保险，Rank 末尾自复位为纵深防御）
    cv::cuda::GpuMat d_label_hist_;         // 262146＝domain+2（[262144]=守卫桶；[262145]=listCnt 拼槽）
    cv::cuda::GpuMat d_label_seg_;          // 262145（桶→段号）
    cv::cuda::GpuMat d_label_list_;         // kV3ListCap=8192（非空桶号清单；>cap 核内回落 BaseScan）
    // 【刀二B】右路行桶（替代 GenKeys→radix→Scatter→CSR）：行域 0..8191＋死桶 8192
    cv::cuda::GpuMat d_right_hist_;         // 8193, CV_32SC1（自清洁）
    cv::cuda::GpuMat d_right_cursor_;       // 8192, CV_32SC1（散射游标）
    cv::cuda::GpuMat d_right_members_;      // rightCount, CV_32SC1（行槽 origIdx）
    cv::cuda::GpuMat d_seg_votes_;          // segCount*V3_MAX_LIDS, CV_32SC1
    cv::cuda::GpuMat d_seg_winner_;         // segCount, CV_32SC1
    cv::cuda::GpuMat d_seg_drop_;           // segCount, CV_32SC1
    cv::cuda::GpuMat d_seg_second_;         // segCap, CV_32SC1（GPU 次名选举，P6b exclCount 用）
    cv::cuda::GpuMat d_excl_cnt_;           // segCap, CV_32SC1（P6b rowOrder 预判，免卡下载）
    cv::cuda::GpuMat d_cards_tight_;        // leftCount*stride, CV_32SC1（卡表紧缩下载中转）
    cv::cuda::GpuMat d_seg_flags_;          // leftCount, CV_32SC1（GPU segmap 游程头标记）
    cv::cuda::GpuMat d_seg_runrank_;        // leftCount, CV_32SC1（GPU segmap 游程 rank）
    static constexpr int kSegCap = 4096;    // GPU 段容量（超限回退 host 段表路径）
    cv::cuda::GpuMat d_cell_seg_;           // MAX_EPIPOLAR_ROWS*V3_MAX_LIDS, CV_32SC1
    cv::cuda::GpuMat d_out_flags_;          // leftCount, CV_32SC1
    cv::cuda::GpuMat d_out_left_;           // leftCount, CV_32FC2 (gather temp)
    cv::cuda::GpuMat d_out_right_;          // leftCount, CV_32FC2
    cv::cuda::GpuMat d_out_lid_;            // leftCount, CV_32SC1
    cv::cuda::GpuMat d_out_track_;          // leftCount, CV_32SC1
    void* d_select_temp_ = nullptr;
    size_t select_temp_size_ = 0;
    size_t select_temp_cap_ = 0;
    // matched outputs (capacity-style persistent buffers)
    cv::cuda::GpuMat d_matched_left_cap_;    // leftCount, CV_32FC2
    cv::cuda::GpuMat d_matched_right_cap_;   // leftCount, CV_32FC2
    cv::cuda::GpuMat d_matched_lids_cap_;    // leftCount, CV_32SC1
    cv::cuda::GpuMat d_matched_tracks_cap_;  // leftCount, CV_32SC1

    // ── status ping-pong 双缓冲（免 clone）──
    cv::cuda::GpuMat d_left_status_[2];
    cv::cuda::GpuMat d_right_status_[2];
    int status_gen_ = 0;

    // ── 计数批量回传：[0]=match_count [1]=excl_left [2]=excl_right [3]=overflow ──
    cv::cuda::GpuMat d_counts_;              // 1x4 CV_32SC1
    int* h_counts_ = nullptr;                // pinned host
    bool overflow_warned_ = false;

    void* d_cub_temp_ = nullptr;
    size_t cub_temp_size_ = 0;
    void* d_scan_temp_ = nullptr;        // ExclusiveSum 专用（与排序临时分离——免异步竞态）
    size_t scan_temp_size_ = 0;
    size_t left_idx_filled_ = 0;         // d_left_idx_in_ 已预填 0..N-1 的容量（增容才重填）

    // ── host pinned D2H 缓冲（全链速度优化 2026-10-16）：grow-only，
    //    免逐帧 pageable 分配＋staging（页锁定内存直传）──
    template <typename T>
    struct PinnedBuf {
        T* p = nullptr;
        size_t cap = 0;   // 元素数
        ~PinnedBuf() { if (p) { cudaFreeHost(p); p = nullptr; } }
        T* ensure(size_t n) {
            if (n <= cap) return p;
            if (p) cudaFreeHost(p);
            if (cudaMallocHost(reinterpret_cast<void**>(&p), n * sizeof(T)) != cudaSuccess) {
                p = nullptr; cap = 0; return nullptr;
            }
            cap = n; return p;
        }
    };
    PinnedBuf<int> hpin_tracks_;
    PinnedBuf<int> hpin_votes_;
    PinnedBuf<float2> hpin_left_;       // float2 前置声明见本文件 cuda 类型引入
    PinnedBuf<int> hpin_cards_;
    PinnedBuf<int> hpin_cardcnt_;
    PinnedBuf<int> hpin_leftstat_;
    PinnedBuf<int> hpin_rightstat_;
    PinnedBuf<int> hpin_diag_;           // 4 int：[1]=帧内最大卡数（卡表下载 stride 依据）
    PinnedBuf<int> hpin_counts_;         // 4 int：[2]=GPU segCount、[3]=段容量溢出标志
    PinnedBuf<int> hpin_segmap_;         // GPU segmap D2H（N）
    PinnedBuf<int> hpin_winner_;         // GPU winner 选举 D2H（kSegCap）
    PinnedBuf<int> hpin_second_;         // GPU 次名选举 D2H（kSegCap）
    PinnedBuf<int> hpin_excl_;           // GPU exclCount D2H（kSegCap）
    PinnedBuf<int> hpin_drop_;           // winner/drop pinned H2D 回传（P7 结果）
    // 【速度优化·P6prep GPU 化】行号/行位图/段偏移/排序下标 GPU 产出＋批量回传
    cv::cuda::GpuMat d_pt_row_, d_seg_rows_, d_seg_offsets_;
    PinnedBuf<int> hpin_ptrow_, hpin_segoff_, hpin_sorted_;
    PinnedBuf<unsigned long long> hpin_segrows_;
    // 【速度优化·刀② c7stamp GPU 化】(段,行) 分组 CSR：行号/坐标和/计数＋段基址
    cv::cuda::GpuMat d_grp_row_, d_grp_sum_, d_grp_cnt_, d_grp_off_;
    PinnedBuf<int> hpin_grprow_, hpin_grpcnt_, hpin_grpoff_;
    PinnedBuf<double> hpin_grpsum_;
    // 【速度优化·刀③ c7merge GPU 化】配对结果（S≤512；S² 定容裸缓冲）＋有票标志
    double* d_pair_sum_ = nullptr;          // [512*512]
    long long* d_pair_n_ = nullptr;         // [512*512]
    int* d_seghas_ = nullptr;               // [kSegCap]
    PinnedBuf<double> hpin_pairsum_;
    PinnedBuf<long long> hpin_pairn_;
    // 【按需下载】值域违规计数（GPU 化防御）
    cv::cuda::GpuMat d_viol_;
    PinnedBuf<int> hpin_viol_;
    // ── 【流水线拆分 2026-10-21】前段产物缓冲（executeFront 用；实现见
    //    laser_match_scan_v3_split_front.cu——复用上方既有排序/卡表/段表缓冲）──
    cv::cuda::GpuMat d_card_off_;           // N：ExclusiveSum(card_count)＝每点卡起始偏移
    cv::cuda::GpuMat d_tight_lid_;          // totalCards CV_32SC1（紧缩卡表 SoA：lid）
    cv::cuda::GpuMat d_tight_rx_;           // totalCards CV_32FC1（hitRx）
    cv::cuda::GpuMat d_tight_ridx_;         // totalCards CV_32SC1（右排序系索引）
    PinnedBuf<int> hpin_cardoff_;           // N
    PinnedBuf<int> hpin_tight_lid_;         // T
    PinnedBuf<float> hpin_tight_rx_;        // T
    PinnedBuf<int> hpin_tight_ridx_;        // T
    PinnedBuf<int> hpin_lefttrack_;         // N（左点 track D2H）
    // 【v3全GPU化·Step10 设备化】终局统计设备侧归约（替四路 D2H ~240KB＋宿主扫描）
    cv::cuda::GpuMat d_stats_;             // 1×8 CV_32SC1：[0]droppedSegs [1]noCard
    PinnedBuf<int> hpin_stats_;            //   [2]noMap [3]rightUsed [4]emptySegs [5]noWinnerCard [6]matched
    // 【v3全GPU化·G2 注入】JMW_V3_SEGCAP<N 人为压低段容量——触发溢出重做路径验证
    int segCapEff_ = kSegCap;

    bool warmed_up_ = false;
    int warmup_left_ = 0;
    int warmup_right_ = 0;
    int old_device_id_ = 0;

    // 【测速工具】最近一次 Execute 的阶段耗时串（gpu_stress 聚合用）
    std::string last_breakdown_;

    // 【v3全GPU化】双路径开关（构造读 env JMW_V3FULLGPU；可经门面 setter 运行时切换——
    // 双跑 harness 用；GPU 路径未实现前恒等效 false）。模式：1=GPU 盖章(K1a)+宿主级联；
    // 2=再叠加 K2 级联 GPU 实验路径
    bool fullGpuOn_ = false;
    int fullGpuMode_ = 0;

    // 【v3全GPU化·K2】级联内核宿主侧缓冲池（grow-only）
    K2Ctx* k2Ctx_ = nullptr;
    // 【v3全GPU化·K1a】盖章点并行内核的块私有 scratch（32 块×8192；一次性分配）
    double* stampSum_ = nullptr;
    int* stampCnt_ = nullptr;
    // 【v3全GPU化·K1b】友军合并散布内核组缓冲（grow-only）：
    // dense 512×8192 double（33MB 一次性，仅写在场项免清零）；行桶 CSR（rowOff
    // 偏移＋rowFill 计数/游标复用）；segList 容量按 ΣR≤leftCount 严格上界增长
    double* k1bDense_ = nullptr;        // [512*8192] 段×行均值稠密表
    int* k1bRowOff_ = nullptr;          // [8193] 行桶 CSR 偏移
    int* k1bRowFill_ = nullptr;         // [8192] 行桶计数→填充游标（两用）
    int* k1bSegList_ = nullptr;         // [k1bSegListCap_] 行桶段列表
    size_t k1bSegListCap_ = 0;

    // 【速度优化】P6prep/P7 宿主结构跨帧复用（grow-only 池；赋值序不变→逐位一致）
    std::vector<std::vector<int>> segVotesPool_;
    std::vector<std::vector<unsigned long long>> segRowsPool_;
    std::vector<std::vector<int>> segPtsPool_;
    std::vector<int> winnerPool_;
    std::vector<char> droppedPool_;
    std::vector<int> ptRowPool_;

#ifndef NDEBUG
    std::atomic<bool> inProcess_{false};
#endif

    explicit Impl(const LaserMatchScanParamsV3& params);
    ~Impl();

    bool LoadTempTable(const std::string& jsonPath);
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void SetCurrentTemperature(double temperature);

    LaserMatchScanResultV3 process(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    // 【流水线拆分 2026-10-21】前半段（Step0-6＋紧缩＋D2H＋同步）；
    // 实现于 laser_match_scan_v3_split_front.cu（process() 与内核零改动）
    void executeFront(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream,
        class LaserMatchSplitFrontV3& out);

    // 【前段提前提交·WDDM 滴流根治 2026-10-23】延迟对：executeFrontDeferred
    // 在排水前按容量提交 Step0-6＋卡表前缀（内核带设备计数尾闸、扫描前缀对
    // 容量安全——前缀只受左侧元素影响）；finishFront 在排水后做紧缩＋主批
    // D2H＋宿主拷出（紧缩网格用宿主精确计数，与旧路径同式）。正确性口径：
    // 各聚合内核尾闸挡陈旧数据入桶/入卡池；d_card_off_ 前缀 [0,count) 与精确
    // 版逐位一致。JMW_V3DEFER=0 时执行器不走此对（旧 executeFront 整段）。
    void executeFrontDeferred(
        const cv::cuda::GpuMat& d_left_points_cap,
        const cv::cuda::GpuMat& d_left_line_ids_cap,
        const int* d_leftCount,
        const cv::cuda::GpuMat& d_right_points_cap,
        const cv::cuda::GpuMat& d_right_line_ids_cap,
        const int* d_rightCount,
        cv::cuda::Stream& stream,
        class LaserMatchSplitFrontV3& out);
    void finishFront(int leftCount, int rightCount,
                     cv::cuda::Stream& stream,
                     class LaserMatchSplitFrontV3& out);
    // 延迟上下文（submit→finish 跨调用；deferActive_ 保证配对）
    bool deferActive_ = false;
    int deferCapL_ = 0, deferCapR_ = 0, deferGen_ = 0;
    cv::cuda::GpuMat d_defer_ptsL_, d_defer_idsL_;   // 左点/track 容量视图（浅引用——
                                        // 完成段主批 D2H 的源；同帧生命周期）

    // 【B4·前段提交图化】JMW_V3FRONTGRAPH=1 显式开（默认关）。捕获体＝
    // executeFrontDeferred 提交段（容量视图＝全参数跨帧稳定→零逐帧 SetParams，
    // mask_sep 生产图同款安全形态；探针 26 跑×200 万图发射零崩）。双 gen 槽位
    // （status ping-pong）；签名＝提交段全部触达指针＋容量＋值参数＋表 epoch
    //（mapEpoch_——B3 基建）；失配→毁图重捕（attempts>8→disabled 永久回退逐发）。
    struct FrontGraphSig {
        // 外部输入（interp 容量视图＋设备计数指针——上游重分配→漂移）
        const void *ptsL, *idsL, *ptsR, *idsR, *cntL, *cntR;
        int capL, capR;
        // 内部 grow-only 缓冲数据指针（增长→换址；核内/status 双缓冲按 gen 分图）
        void *rHist, *rCursor, *rMembers, *rRowStart, *rRowCount, *rSortedPts, *rSortedLids;
        void *rStatusSlot, *lStatusSlot;          // ★gen 双缓冲（审核 R1 高危项）
        void *counts, *diag, *segMap, *labelHist, *labelSeg, *labelList, *cardOff;
        void *cardLid, *cardHitRx, *cardRes, *cardHitRidx, *cardCount;
        void *mapByrow, *mapRowStart, *mapRowCount, *bucketPool, *bucketRowOff;
        void *urBias, *biasParams, *scanTemp;
        int *hCnt, *hDia, *hOff, *hCcn;           // pinned D2H 宿（ensure 增长→换址）
        float rowStep, vLtol, matchWin, bMinX, bInvW;
        int bK, useB;
        unsigned long long epoch;                  // mapEpoch_（B3 基建）
    };
    struct FrontGraph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool sigValid = false;
        int attempts = 0;
        FrontGraphSig sig{};
    };
    FrontGraph fgs_[2];
    bool fgFrontDisabled_ = false;
    int fgFrontWarm_ = 0;            // 全局暖机（≥2 才捕获——摊平惰性分配）
    void executeFrontGraph(
        const cv::cuda::GpuMat& d_left_points_cap,
        const cv::cuda::GpuMat& d_left_line_ids_cap,
        const int* d_leftCount,
        const cv::cuda::GpuMat& d_right_points_cap,
        const cv::cuda::GpuMat& d_right_line_ids_cap,
        const int* d_rightCount,
        cv::cuda::Stream& stream,
        class LaserMatchSplitFrontV3& out);

    void warmup(int maxLeftPoints, int maxRightPoints);
    void setParams(const LaserMatchScanParamsV3& params);
    void setuRBias(const std::vector<float>& biasPerLid);
    void setuRBiasModel(float wx, float wy, float wz, float dtx);
    const LaserMatchScanParamsV3& getParams() const { return params_; }
    // 【B3】发卡网格（TP=2 跨步；恒 ≥1——探针公式修正；256＝BLOCK_SIZE，.cu 同值勿漂移）
    // 与 useB 三态统一助手（0=二分 JMW_V3BISECT；1=跳走 JMW_V3ANCHOR=0 旧臂；2=桶锚默认）
    int issueGrid_(int n) const {
        return (n + 2 * 256 - 1) / (2 * 256);
    }
    int issueUseBuckets_() const {
        const bool poolReady = bucketK_ > 0 && !d_bucket_pool_.empty()
                               && !d_bucket_row_off_.empty();
        if (!poolReady || getenv("JMW_V3BISECT") != nullptr) return 0;
        const char* a = getenv("JMW_V3ANCHOR");
        return (a != nullptr && a[0] == '0') ? 1 : 2;
    }
    unsigned long long mapEpoch() const { return mapEpoch_; }   // B4 签名消费口

    bool allocateBuffers(int leftCount, int rightCount);
    void uploadMapTable(const TempTableEntry& entry, cv::cuda::Stream& stream);
};

} // namespace calib
