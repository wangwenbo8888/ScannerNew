/**
 * @file laser_match_scan_v3_cuda_impl.cu
 * @brief 激光线匹配扫描CUDA算子 V3（段级投票+冲突双弃）— CUDA实现（struct Impl 方法）
 *
 * 5 阶段 GPU 管线（算法详见头文件与算子说明文档）：
 *   Step 1-4 右侧排序/CSR：kernelGenKeysIdxV3 → CUB RadixSort → kernelScatterSortedV3 → kernelBuildCSRV3
 *   Step 5   kernelIssueCardsV3：查表发卡（每 lid 就近单卡，±match_window 验卡记 hitRx）
 *   Step 6   [host] 段表构建（D2H 左段 ID → sort/unique → orig→segIdx 映射）
 *   Step 7   kernelSegVoteV3 → kernelSegWinnerV3 → kernelConflictMarkV3
 *            （段内 (段,lid) 票数直方图 → argmax winner → 同行同 winner 双弃 atomicCAS）
 *   Step 8   kernelGatherV3（保留段 winner 卡片收集）→ CUB DeviceSelect ×4 紧凑化
 *   Step 10  [host] pinned 批量回传 + kernelCountStatusV2 段诊断统计
 *
 * 继承自早期 v2 优化版骨架的工程结构（单次主机同步/持久缓冲/ROI 视图）；
 * 旧 kernelMatchV3（v2 逐点唯一命中制）已被上述段级投票 kernel 组取代（2026-09 演进，
 * v1/v2 随 laser_match_scan 退役至 retired/）。
 */

#include "laser_match_scan_v3_cuda_pimpl.h"
#include "common/calib_types.h"
#include "common/calib_logging.h"
#include "common/json_utils.h"
#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <algorithm>
#include <map>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <numeric>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <climits>
#include <functional>
#include <unordered_map>

using namespace calib;

CALIB_DEFINE_LOG_TAG(07, LaserMatchScanCudaV3);

// ============================================================================
// Configuration Constants
// ============================================================================

static constexpr int BLOCK_SIZE = 256;
static constexpr int V3_MAX_CARDS = 32;   // per-point card cap (nearest card per lid)
static constexpr int V3_MAX_LIDS = 32;    // lid bucket stride (lids are 1..25)
static constexpr int MATCH_BLOCK = 32;          // kernelMatchV3: 1 warp per row
static constexpr int MAX_LOOKUP_CANDS = 64;     // 与 Impl::MAX_LOOKUP_CANDS 同值
static constexpr int SMEM_RIGHT_CAP = 256;      // 与 Impl::SMEM_RIGHT_CAP 同值
static constexpr unsigned int KEY_SENTINEL = 0xFFFFFFFFu;
static constexpr unsigned int XQ_ONE = 256u;    // x 量化 1/256 px
static constexpr unsigned int XQ_MASK = (1u << 19) - 1u;

// ============================================================================
// CUDA Error Handling
// ============================================================================

template<typename T>
static inline void safeCudaFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            CALIB_LOG_ERROR("cudaFree failed: {}", cudaGetErrorString(err));
        }
        ptr = nullptr;
    }
}

#ifndef NDEBUG
class ScopedFlagV2 {
public:
    explicit ScopedFlagV2(std::atomic<bool>* flag) : flag_(flag) {
        flag_->store(true);
    }
    ~ScopedFlagV2() { flag_->store(false); }
    ScopedFlagV2(const ScopedFlagV2&) = delete;
    ScopedFlagV2& operator=(const ScopedFlagV2&) = delete;
private:
    std::atomic<bool>* flag_;
};
#endif

// ============================================================================
// 【v3全GPU化·K2】级联判定缓冲池（grow-only；须置于 Impl ctor 之前）
// 注：kWordsU 与下方 kRowWords 一致性由 static_assert 把关（kRowWords 定义更晚）
// ============================================================================
struct calib::K2Ctx {
    static constexpr size_t kMaxU = 128;               // G-1 实测 max 106＋余量
    static constexpr size_t kWordsU = 128;             // 行位图字数＝kRowWords（static_assert 把关）
    static constexpr size_t kMaxRowsU = 8192;          // ＝kMaxRows（static_assert 把关）
    long long* d_orig = nullptr;                       // [U*32]
    unsigned long long* d_rows = nullptr;              // [U*kWordsU]
    int* d_rowOff = nullptr;                           // [U+1]
    int* d_rowIdx = nullptr; double* d_rowXSum = nullptr; int* d_rowCnt = nullptr;
    double* d_meanDense = nullptr;                     // [kMaxU*kMaxRowsU] 稠密行均值（v2 安全检查用）
    int* d_segUnit = nullptr;                          // [S]
    int* d_winner = nullptr; int* d_dropped = nullptr; // [S] out
    int* d_stats = nullptr;                            // [5] det/banned/rounds/iters/fallback
    long long* h_orig = nullptr; unsigned long long* h_rows = nullptr;
    int* h_rowOff = nullptr; int* h_rowIdx = nullptr;
    double* h_rowXSum = nullptr; int* h_rowCnt = nullptr;
    int* h_segUnit = nullptr; int* h_winner = nullptr; int* h_dropped = nullptr;
    int* h_stats = nullptr;
    size_t capRow = 0, capS = 0;
    bool ensured = false;
    bool ensure(size_t rowTotal, size_t S) {           // grow-only（首次后零分配）
        const size_t needRow = rowTotal > capRow ? rowTotal : capRow;
        const size_t needS = S > capS ? S : capS;
        if (ensured && needRow == capRow && needS == capS) return true;
        auto reallocDev = [](void** p, size_t bytes) {
            if (*p) cudaFree(*p);
            return cudaMalloc(p, bytes) == cudaSuccess;
        };
        auto reallocHost = [](void** p, size_t bytes) {
            if (*p) cudaFreeHost(*p);
            return cudaMallocHost(p, bytes) == cudaSuccess;
        };
        if (!reallocDev(reinterpret_cast<void**>(&d_orig), kMaxU * V3_MAX_LIDS * sizeof(long long))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_rows), kMaxU * kWordsU * sizeof(unsigned long long))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_rowOff), (kMaxU + 1) * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_rowIdx), needRow * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_rowXSum), needRow * sizeof(double))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_rowCnt), needRow * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_meanDense), kMaxU * kMaxRowsU * sizeof(double))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_segUnit), needS * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_winner), needS * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_dropped), needS * sizeof(int))) return false;
        if (!reallocDev(reinterpret_cast<void**>(&d_stats), 5 * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_orig), kMaxU * V3_MAX_LIDS * sizeof(long long))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_rows), kMaxU * kWordsU * sizeof(unsigned long long))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_rowOff), (kMaxU + 1) * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_rowIdx), needRow * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_rowXSum), needRow * sizeof(double))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_rowCnt), needRow * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_segUnit), needS * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_winner), needS * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_dropped), needS * sizeof(int))) return false;
        if (!reallocHost(reinterpret_cast<void**>(&h_stats), 5 * sizeof(int))) return false;
        capRow = needRow; capS = needS; ensured = true;
        return true;
    }
    void freeAll() {
        auto f = [](void* p) { if (p) cudaFree(p); };
        auto fh = [](void* p) { if (p) cudaFreeHost(p); };
        f(d_orig); f(d_rows); f(d_rowOff); f(d_rowIdx); f(d_rowXSum); f(d_rowCnt);
        f(d_meanDense); f(d_segUnit); f(d_winner); f(d_dropped); f(d_stats);
        fh(h_orig); fh(h_rows); fh(h_rowOff); fh(h_rowIdx); fh(h_rowXSum); fh(h_rowCnt);
        fh(h_segUnit); fh(h_winner); fh(h_dropped); fh(h_stats);
        d_orig = nullptr; d_rows = nullptr; d_rowOff = nullptr; d_rowIdx = nullptr;
        d_rowXSum = nullptr; d_rowCnt = nullptr; d_meanDense = nullptr; d_segUnit = nullptr;
        d_winner = nullptr; d_dropped = nullptr; d_stats = nullptr;
        h_orig = nullptr; h_rows = nullptr; h_rowOff = nullptr; h_rowIdx = nullptr;
        h_rowXSum = nullptr; h_rowCnt = nullptr; h_segUnit = nullptr;
        h_winner = nullptr; h_dropped = nullptr; h_stats = nullptr;
        capRow = capS = 0; ensured = false;
    }
};
using calib::K2Ctx;   // 供下方匿名命名空间内的内核/包装/级联引用

// ============================================================================
// Impl ctor/dtor
// ============================================================================

LaserMatchScanCudaV3::Impl::Impl(const LaserMatchScanParamsV3& params)
    : params_(params)
{
    params_.validate();

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        throw std::runtime_error("[07-LaserMatchScanCudaV3] No CUDA devices found");
    }
    if (params_.deviceId >= deviceCount) {
        throw std::invalid_argument("[07-LaserMatchScanCudaV3] deviceId >= device count");
    }

    old_device_id_ = 0;
    cudaGetDevice(&old_device_id_);
    if (params_.deviceId != old_device_id_) {
        cudaSetDevice(params_.deviceId);
    }

    if (cudaMallocHost((void**)&h_counts_, 4 * sizeof(int)) != cudaSuccess) {
        h_counts_ = nullptr;
        CALIB_LOG_WARN("pinned counts alloc failed; fallback to staged copy");
    }
    // 【v3全GPU化】双路径开关：env 缺省 0（宿主路径）；JMW_V3FULLGPU=1 GPU 盖章(K1a)＋
    // 宿主级联（生产候选路径）；=2 再叠加 K2 级联 GPU 实验路径
    if (const char* m = getenv("JMW_V3FULLGPU")) {
        fullGpuMode_ = atoi(m);
        if (fullGpuMode_ < 0) fullGpuMode_ = 0;
    }
    fullGpuOn_ = (fullGpuMode_ > 0);
    // 【v3全GPU化·G2 注入】JMW_V3_SEGCAP=N（16≤N<4096）人为压低段容量——常态帧
    // 段数超过即走「溢出重做」宿主路径（回退路径等价性验证用；产线勿开）
    if (const char* c = getenv("JMW_V3_SEGCAP")) {
        const int v = atoi(c);
        if (v >= 16 && v < kSegCap) segCapEff_ = v;
    }
    // 【v3全GPU化·K2】级联内核缓冲池（grow-only；首次 Execute 按需 ensure）
    k2Ctx_ = new K2Ctx();
}

LaserMatchScanCudaV3::Impl::~Impl() {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        CALIB_LOG_ERROR("cudaDeviceSynchronize in destructor failed: {}",
                        cudaGetErrorString(sync_err));
    }
    safeCudaFree(d_cub_temp_);
    safeCudaFree(d_scan_temp_);
    // 【速度优化·刀③】配对/有票裸缓冲释放
    if (d_pair_sum_) { cudaFree(d_pair_sum_); d_pair_sum_ = nullptr; }
    if (d_pair_n_) { cudaFree(d_pair_n_); d_pair_n_ = nullptr; }
    if (d_seghas_) { cudaFree(d_seghas_); d_seghas_ = nullptr; }
    if (h_counts_) { cudaFreeHost(h_counts_); h_counts_ = nullptr; }
    // 【v3全GPU化·K2】级联内核缓冲池释放＋K1a scratch 释放
    if (k2Ctx_) { k2Ctx_->freeAll(); delete k2Ctx_; k2Ctx_ = nullptr; }
    if (stampSum_) { cudaFree(stampSum_); stampSum_ = nullptr; }
    if (stampCnt_) { cudaFree(stampCnt_); stampCnt_ = nullptr; }
    // 【v3全GPU化·K1b】散布内核组缓冲释放
    if (k1bDense_) { cudaFree(k1bDense_); k1bDense_ = nullptr; }
    if (k1bRowOff_) { cudaFree(k1bRowOff_); k1bRowOff_ = nullptr; }
    if (k1bRowFill_) { cudaFree(k1bRowFill_); k1bRowFill_ = nullptr; }
    if (k1bSegList_) { cudaFree(k1bSegList_); k1bSegList_ = nullptr; }
    k1bSegListCap_ = 0;
}

// ============================================================================
// allocateBuffers —— 真·容量式 grow-only（【150FPS】原 create 在尺寸变化时
// 释放重分配＝每帧 ~27 对同步 cudaFree/cudaMalloc ≈1.5-2.7ms——逐帧点数变化
// 时全数触发。改为只在不足时翻倍增长、永不缩小：指针稳定＋零逐帧分配）
// ============================================================================
static void growOnly(cv::cuda::GpuMat& m, int cols, int type) {
    if (m.cols < cols) {
        const int cap = std::max(cols, m.cols > 0 ? m.cols * 2 : cols);
        m.create(1, cap, type);
    }
}

bool LaserMatchScanCudaV3::Impl::allocateBuffers(int leftCount, int rightCount) {
    if (leftCount > 0) {
        growOnly(d_left_keys_in_, leftCount, CV_32SC1);
        growOnly(d_left_keys_out_, leftCount, CV_32SC1);
        growOnly(d_left_idx_in_, leftCount, CV_32SC1);
        growOnly(d_left_sorted_idx_, leftCount, CV_32SC1);
        growOnly(d_left_sorted_pts_, leftCount, CV_32FC2);
        growOnly(d_left_sorted_lids_, leftCount, CV_32SC1);
        growOnly(d_card_lid_, leftCount * V3_MAX_CARDS, CV_32SC1);
        growOnly(d_card_hitRx_, leftCount * V3_MAX_CARDS, CV_32FC1);
        growOnly(d_card_count_, leftCount, CV_32SC1);
        growOnly(d_card_res_, leftCount * V3_MAX_CARDS, CV_32FC1);
        growOnly(d_card_hitRidx_, leftCount * V3_MAX_CARDS, CV_32SC1);
        growOnly(d_out_flags_, leftCount, CV_32SC1);
        growOnly(d_out_left_, leftCount, CV_32FC2);
        growOnly(d_out_right_, leftCount, CV_32FC2);
        growOnly(d_out_lid_, leftCount, CV_32SC1);
        growOnly(d_out_track_, leftCount, CV_32SC1);
        d_cell_seg_.create(1, MAX_EPIPOLAR_ROWS * V3_MAX_LIDS, CV_32SC1);
        growOnly(d_matched_left_cap_, leftCount, CV_32FC2);
        growOnly(d_matched_right_cap_, leftCount, CV_32FC2);
        growOnly(d_matched_lids_cap_, leftCount, CV_32SC1);
        growOnly(d_matched_tracks_cap_, leftCount, CV_32SC1);
        growOnly(d_left_status_[0], leftCount, CV_32SC1);
        growOnly(d_left_status_[1], leftCount, CV_32SC1);
    }
    if (rightCount > 0) {
        growOnly(d_right_keys_in_, rightCount, CV_32SC1);
        growOnly(d_right_keys_out_, rightCount, CV_32SC1);
        growOnly(d_right_idx_in_, rightCount, CV_32SC1);
        growOnly(d_right_sorted_idx_, rightCount, CV_32SC1);
        growOnly(d_right_sorted_pts_, rightCount, CV_32FC2);
        growOnly(d_right_sorted_lids_, rightCount, CV_32SC1);
        growOnly(d_right_status_[0], rightCount, CV_32SC1);
        growOnly(d_right_status_[1], rightCount, CV_32SC1);
    }
    d_counts_.create(1, 4, CV_32SC1);
    d_diag_.create(1, 4, CV_32SC1);

    d_left_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_left_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_start_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);
    d_right_row_count_.create(1, MAX_EPIPOLAR_ROWS, CV_32SC1);

    int maxCount = leftCount > rightCount ? leftCount : rightCount;
    if (maxCount > 0) {
        size_t sort_temp = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp,
            (unsigned int*)nullptr, (unsigned int*)nullptr,
            (int*)nullptr, (int*)nullptr,
            maxCount, 0, sizeof(unsigned int) * 8, 0);
        if (cub_temp_size_ < sort_temp) {
            safeCudaFree(d_cub_temp_);
            cudaMalloc(&d_cub_temp_, sort_temp);
            cub_temp_size_ = sort_temp;
        }
    }

    return true;
}

// ============================================================================
// warmup
// ============================================================================

void LaserMatchScanCudaV3::Impl::warmup(int maxLeftPoints, int maxRightPoints) {
    if (maxLeftPoints <= 0 || maxRightPoints <= 0) {
        CALIB_LOG_WARN("warmup(): invalid sizes (left={}, right={}), skipping",
                       maxLeftPoints, maxRightPoints);
        return;
    }
    allocateBuffers(maxLeftPoints, maxRightPoints);
    warmed_up_ = true;
    warmup_left_ = maxLeftPoints;
    warmup_right_ = maxRightPoints;
    CALIB_LOG_INFO("warmup(): allocated buffers for {} left, {} right points",
                   maxLeftPoints, maxRightPoints);
}

// ============================================================================
// loadTempTable —— JSON 便捷注入（同 v1）
// ============================================================================

bool LaserMatchScanCudaV3::Impl::LoadTempTable(const std::string& jsonPath) {
    try {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            CALIB_LOG_ERROR("loadTempTable(): cannot open file: {}", jsonPath);
            return false;
        }

        nlohmann::json j;
        ifs >> j;

        auto table = std::make_shared<LaserPlaneMapTempTable>();
        const auto& arr = j.at("table");
        for (const auto& entry : arr) {
            const double temp = entry.at("temperature").get<double>();
            const auto& mapData = entry.at("mapData");
            const int n = static_cast<int>(mapData.size());

            LaserPlaneMap planeMap;
            planeMap.temperature = temp;
            planeMap.totalPairs = n;
            cv::Mat m(n, 4, CV_32FC1);
            for (int i = 0; i < n; ++i) {
                const auto& row = mapData[i];
                m.at<float>(i, 0) = row[0].get<float>();
                m.at<float>(i, 1) = row[1].get<float>();
                m.at<float>(i, 2) = row[2].get<float>();
                m.at<float>(i, 3) = row[3].get<float>();
            }
            planeMap.leftToRightMap = std::move(m);
            table->table[temp] = std::move(planeMap);
        }

        if (!SetTempTable(table)) {
            return false;
        }
        CALIB_LOG_INFO("loadTempTable(): loaded {} temperature entries from {}",
                       tempTable_.size(), jsonPath);
        return true;

    } catch (const std::exception& e) {
        CALIB_LOG_ERROR("loadTempTable(): exception: {}", e.what());
        return false;
    }
}

// ============================================================================
// SetTempTable —— 公共类型 → 内部扁平表示（同 v1）
// ============================================================================

bool LaserMatchScanCudaV3::Impl::SetTempTable(
    std::shared_ptr<const LaserPlaneMapTempTable> table)
{
    if (!table || table->empty()) {
        CALIB_LOG_ERROR("SetTempTable(): null or empty temp table");
        return false;
    }

    // 同源重复注入（全链速度优化 2026-10-16）：源指针一致 ⇔ 表内容一致
    // （const 共享只读契约）⇔ 已转换的 tempTable_ 仍有效，跳过 O(N) 重转换。
    if (tableLoaded_ && srcHold_ && srcHold_.get() == table.get()) {
        return true;
    }
    srcHold_ = table;
    uploadedSrc_ = nullptr;   // 新表：GPU 上传状态失效（下次 SetCurrentTemperature 重传）

    tempTable_.clear();

    for (const auto& kv : table->table) {
        const double temp = kv.first;
        const LaserPlaneMap& planeMap = kv.second;
        const cv::Mat& m = planeMap.leftToRightMap;
        if (m.empty()) {
            CALIB_LOG_WARN("SetTempTable(): skip temp={} (empty leftToRightMap)", temp);
            continue;
        }

        TempTableEntry te;
        te.temperature = temp;
        te.entryCount = m.rows;
        te.mapData.resize(te.entryCount * 4);

        int maxLid = -1;
        for (int i = 0; i < te.entryCount; ++i) {
            const float* r = m.ptr<float>(i);
            te.mapData[i * 4 + 0] = r[0];
            te.mapData[i * 4 + 1] = r[1];
            te.mapData[i * 4 + 2] = r[2];
            te.mapData[i * 4 + 3] = r[3];
            const int lid = static_cast<int>(r[3]);
            if (lid > maxLid) maxLid = lid;
        }
        te.numLines = maxLid + 1;
        tempTable_.push_back(std::move(te));
    }

    if (tempTable_.empty()) {
        CALIB_LOG_ERROR("SetTempTable(): no valid entries after conversion");
        tableLoaded_ = false;
        return false;
    }

    std::sort(tempTable_.begin(), tempTable_.end(),
              [](const TempTableEntry& a, const TempTableEntry& b) {
                  return a.temperature < b.temperature;
              });

    tableLoaded_ = true;
    return true;
}

// ============================================================================
// setCurrentTemperature（同 v1）
// ============================================================================

void LaserMatchScanCudaV3::Impl::SetCurrentTemperature(double temperature) {
    if (!tableLoaded_ || tempTable_.empty()) {
        throw std::runtime_error("[07-LaserMatchScanCudaV3] temp table not loaded");
    }

    if (temperature < tempTable_.front().temperature ||
        temperature > tempTable_.back().temperature) {
        CALIB_LOG_WARN("setCurrentTemperature(): temp={} out of range [{}, {}], clamping",
                       temperature,
                       tempTable_.front().temperature,
                       tempTable_.back().temperature);
    }

    auto it = std::lower_bound(tempTable_.begin(), tempTable_.end(), temperature,
        [](const TempTableEntry& e, double val) { return e.temperature < val; });

    if (it == tempTable_.end()) --it;
    if (it != tempTable_.begin()) {
        auto prev = std::prev(it);
        if (std::abs(prev->temperature - temperature) <
            std::abs(it->temperature - temperature)) {
            it = prev;
        }
    }

    currentTemp_ = it->temperature;
    tempSet_ = true;

    // 注备缓存命中（全链速度优化 2026-10-16）：同源表＋同档温＋同 rowStep 的重复
    // 注入跳过 uploadMapTable（O(N log N) host 分桶排序＋140MB×2 上传）。
    const bool cacheHit = (uploadedSrc_ != nullptr
                           && srcHold_ && srcHold_.get() == uploadedSrc_
                           && it->temperature == uploadedTemp_
                           && params_.epipolar_row_step == uploadedRowStep_);
    if (!cacheHit) {
        cv::cuda::Stream stream;
        uploadMapTable(*it, stream);
        stream.waitForCompletion();
        uploadedSrc_ = srcHold_.get();
        uploadedTemp_ = it->temperature;
        uploadedRowStep_ = params_.epipolar_row_step;
    }

    CALIB_LOG_DEBUG("setCurrentTemperature(): temp={} -> selected entry temp={}, {} map entries, {} lines{}",
                   temperature, currentTemp_, it->entryCount, it->numLines,
                   cacheHit ? " (cached)" : "");
}

// ============================================================================
// uploadMapTable（同 v1）
// ============================================================================

void LaserMatchScanCudaV3::Impl::uploadMapTable(
    const TempTableEntry& entry, cv::cuda::Stream& stream)
{
    activeMapCount_ = entry.entryCount;
    activeNumLines_ = entry.numLines;

    cv::Mat h_map(activeMapCount_, 1, CV_32FC4,
                  const_cast<float*>(entry.mapData.data()));
    d_map_table_.upload(h_map, stream);

    std::vector<int> h_line_start(activeNumLines_, 0);
    std::vector<int> h_line_count(activeNumLines_, 0);

    for (int i = 0; i < activeMapCount_; ++i) {
        int lid = static_cast<int>(entry.mapData[i * 4 + 3]);
        if (lid >= 0 && lid < activeNumLines_) {
            if (h_line_count[lid] == 0) {
                h_line_start[lid] = i;
            }
            h_line_count[lid]++;
        }
    }

    cv::Mat h_ls(1, activeNumLines_, CV_32SC1, h_line_start.data());
    cv::Mat h_lc(1, activeNumLines_, CV_32SC1, h_line_count.data());
    d_map_line_start_.upload(h_ls, stream);
    d_map_line_count_.upload(h_lc, stream);

    // Build by-row CSR view for global (scan-mode) lookup
    {
        int N = activeMapCount_;
        std::vector<std::vector<int>> buckets(MAX_EPIPOLAR_ROWS);
        for (int i = 0; i < N; ++i) {
            float yL = entry.mapData[i * 4 + 1];
            int row = (int)std::round(yL / params_.epipolar_row_step);
            if (row < 0) row = 0;
            if (row >= MAX_EPIPOLAR_ROWS) row = MAX_EPIPOLAR_ROWS - 1;
            buckets[row].push_back(i);
        }
        std::vector<float> h_byrow;
        std::vector<int> h_row_start(MAX_EPIPOLAR_ROWS, 0);
        std::vector<int> h_row_count(MAX_EPIPOLAR_ROWS, 0);
        int offset = 0;
        for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
            h_row_start[r] = offset;
            h_row_count[r] = (int)buckets[r].size();
            std::sort(buckets[r].begin(), buckets[r].end(), [&](int a, int b) {
                return entry.mapData[a * 4] < entry.mapData[b * 4];
            });
            for (int idx : buckets[r]) {
                h_byrow.push_back(entry.mapData[idx * 4]);
                h_byrow.push_back(entry.mapData[idx * 4 + 2]);
                h_byrow.push_back(entry.mapData[idx * 4 + 3]);
                h_byrow.push_back(entry.mapData[idx * 4 + 1]);
            }
            offset += h_row_count[r];
        }
        cv::Mat h_byrow_mat(N, 1, CV_32FC4, h_byrow.data());
        d_map_byrow_.upload(h_byrow_mat, stream);
        cv::Mat h_rs(1, MAX_EPIPOLAR_ROWS, CV_32SC1, h_row_start.data());
        cv::Mat h_rc(1, MAX_EPIPOLAR_ROWS, CV_32SC1, h_row_count.data());
        d_map_row_start_.upload(h_rs, stream);
        d_map_row_count_.upload(h_rc, stream);

        // 【150FPS·查表桶索引】行内桶下界池：pool[b]＝行内首个 x≥(minX+b·w) 的
        // 相对偏移。内核桶直达＋跳走替代 13 步二分（锚点逐位一致；JMW_V3BISECT=1 兜底）
        {
            float mnX = FLT_MAX, mxX = -FLT_MAX;
            for (int i = 0; i < N; ++i) {
                const float x = entry.mapData[i * 4];
                if (x < mnX) mnX = x;
                if (x > mxX) mxX = x;
            }
            const float w = (mxX - mnX) / kBucketK;
            if (w > 1e-6f) {
                bucketMinX_ = mnX;
                bucketInvW_ = 1.f / w;
                bucketK_ = kBucketK;
                std::vector<int> h_pool;
                h_pool.reserve(1024 * (kBucketK + 1));
                std::vector<int> h_boff(MAX_EPIPOLAR_ROWS, 0);
                for (int r = 0; r < MAX_EPIPOLAR_ROWS; ++r) {
                    if (h_row_count[r] == 0) continue;
                    h_boff[r] = static_cast<int>(h_pool.size());
                    const int s = h_row_start[r];
                    const int cnt = h_row_count[r];
                    int p = 0;
                    for (int b = 0; b <= kBucketK; ++b) {
                        const float xb = mnX + b * w;
                        while (p < cnt &&
                               h_byrow[static_cast<size_t>(s + p) * 4] < xb)
                            ++p;
                        h_pool.push_back(p);
                    }
                }
                if (!h_pool.empty()) {
                    cv::Mat h_bp(1, static_cast<int>(h_pool.size()), CV_32SC1,
                                 h_pool.data());
                    d_bucket_pool_.upload(h_bp, stream);
                    cv::Mat h_bo(1, MAX_EPIPOLAR_ROWS, CV_32SC1, h_boff.data());
                    d_bucket_row_off_.upload(h_bo, stream);
                } else {
                    bucketK_ = 0;   // 空表防御
                }
            } else {
                bucketK_ = 0;
            }
        }
    }
    // 【B3·基建】map/桶池重传代际 +1（B4 图化签名消费；cacheHit 路径不进本函数
    //＝同表同温重注入 epoch 不变）
    ++mapEpoch_;
}

// ============================================================================
// setParams
// ============================================================================

void LaserMatchScanCudaV3::Impl::setParams(const LaserMatchScanParamsV3& params) {
#ifndef NDEBUG
    if (inProcess_.load()) {
        CALIB_LOG_ERROR("setParams(): called while process() is running");
        throw std::runtime_error("[07-LaserMatchScanCudaV3] setParams() called during process()");
    }
#endif
    params.validate();
    bool deviceChanged = (params.deviceId != params_.deviceId);
    if (params.epipolar_row_step != params_.epipolar_row_step)
        uploadedRowStep_ = -1.0f;   // rowStep 影响分桶 → GPU 表上传状态失效
    params_ = params;
    if (deviceChanged) {
        cudaSetDevice(params_.deviceId);
    }
    warmed_up_ = false;
    warmup_left_ = 0;
    warmup_right_ = 0;
    CALIB_LOG_INFO("setParams(): params updated, warmup reset");
}

void LaserMatchScanCudaV3::Impl::setuRBias(const std::vector<float>& biasPerLid) {
    std::vector<float> bias(V3_MAX_LIDS, 0.0f);
    const size_t n = std::min(biasPerLid.size(), static_cast<size_t>(V3_MAX_LIDS));
    for (size_t i = 0; i < n; ++i) bias[i] = biasPerLid[i];
    cv::Mat m(1, V3_MAX_LIDS, CV_32FC1, bias.data());
    d_ur_bias_.upload(m);
    urBiasActive_ = !biasPerLid.empty();
    if (urBiasActive_) urBiasModelActive_ = false;   // 互斥：逐线表顶替 4 参数模型
    CALIB_LOG_INFO("setuRBias(): {} lids, active={}", n, urBiasActive_);
}

void LaserMatchScanCudaV3::Impl::setuRBiasModel(float wx, float wy, float wz, float dtx) {
    cv::Mat m(1, 1, CV_32FC4);
    m.at<cv::Vec4f>(0, 0) = cv::Vec4f(wx, wy, wz, dtx);
    d_bias_params_.upload(m);
    urBiasModelActive_ = true;
    urBiasActive_ = false;                           // 互斥：模型顶替逐线表
    CALIB_LOG_INFO("setuRBiasModel(): wx={} wy={} wz={} dtx={} (model active, per-lid table off)",
                   wx, wy, wz, dtx);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

struct MatchPairV3 {
    int leftIdx;
    int rightIdx;
};

struct CandPackV3 {
    float uR;
    int lid;
};

// K1: 32bit 复合排序键 + 顺带 idx 序列（省 thrust::sequence）
//     key = (rowKey << 19) | xq,  xq = x*256 取整 [0, 2^19)
//     row >= 8192 → KEY_SENTINEL（排序沉底, CSR build 丢弃, 与 v1 一致）
__global__ void kernelGenKeysIdxV3(
    const float2* __restrict__ d_points,
    int count,
    float epipolar_row_step,
    unsigned int* __restrict__ d_keys,
    int* __restrict__ d_idx_in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float2 pt = d_points[idx];
    int rowKey = static_cast<int>(roundf(pt.y / epipolar_row_step));
    if (rowKey < 0) rowKey = 0;

    unsigned int key;
    if (rowKey >= 8192) {
        key = KEY_SENTINEL;
    } else {
        float x = (pt.x >= 0.0f) ? pt.x : 0.0f;
        int xq = __float2int_rn(x * static_cast<float>(XQ_ONE));
        if (xq < 0) xq = 0;
        if (xq > static_cast<int>(XQ_MASK)) xq = static_cast<int>(XQ_MASK);
        key = (static_cast<unsigned int>(rowKey) << 19) | static_cast<unsigned int>(xq);
    }

    d_keys[idx] = key;
    d_idx_in[idx] = idx;
}

// K2: scatter sorted（同 v1）
__global__ void kernelScatterSortedV3(
    const float2* __restrict__ d_src_pts,
    const int*    __restrict__ d_src_lids,
    const int*    __restrict__ d_sorted_idx,
    int count,
    float2* __restrict__ d_dst_pts,
    int*    __restrict__ d_dst_lids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int src = d_sorted_idx[idx];
    d_dst_pts[idx] = d_src_pts[src];
    d_dst_lids[idx] = d_src_lids[src];
}

// K3: CSR 构建（读 32bit 键, sentinel 丢弃; 同 v1 语义）
__global__ void kernelBuildCSRV3(
    const unsigned int* __restrict__ d_sort_keys,
    int count,
    int numRows,
    int* __restrict__ d_row_start,
    int* __restrict__ d_row_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned int key = d_sort_keys[idx];
    if (key == KEY_SENTINEL) return;
    unsigned int rowKey = key >> 19;
    if (static_cast<int>(rowKey) >= numRows) return;

    atomicMin(&d_row_start[rowKey], idx);
    atomicAdd(reinterpret_cast<unsigned int*>(&d_row_count[rowKey]), 1u);
}

// ============================================================================
// V3 kernels: nearest-card issue + segment voting + conflict drop-both + gather
// ============================================================================

// ── uR 4 参数位置相关偏差模型常数（2026-10-11 方式2）──
// f/B/cx/cy＝本机矫正系内参；tx/ty/tz＝虚拟投影器光心 (mm)（PJC 输出）。
// 来源与 2026-10-11 角度拟合实验（cmtt_lid_analysis/verify_angular 族）同源同值。
static constexpr float URB_F  = 1231.139f;
static constexpr float URB_B  = 133.311f;
static constexpr float URB_CX = 830.523f;
static constexpr float URB_CY = 685.0f;
static constexpr float URB_TX = 39.847f;
static constexpr float URB_TY = 8.087f;
static constexpr float URB_TZ = 10.626f;

// V3-K1: issue cards (nearest entry per lid) + HIT check against right row CSR.
// Runs in ORIGINAL index space (no left sort needed). Card = (lid, hitRx).
// 【速度优化】maxCnt 折入：末尾对 d_maxOut 取 atomicMax（省一发射）
// 【150FPS·占用率实验证伪】__launch_bounds__(256,4) 实测 g1 +0.2ms（强制满占→
// 溢出增多＞延迟隐藏收益）——已回退；低占用假设不成立
__global__ void kernelIssueCardsV3(
    const float2* __restrict__ d_left_pts,        // orig order
    int leftCount,
    const float4* __restrict__ d_map_byrow,       // (xL, uR, lineId, yL) sorted by xL in row
    const int*    __restrict__ d_map_row_start,
    const int*    __restrict__ d_map_row_count,
    const float2* __restrict__ d_right_sorted_pts,
    const int*    __restrict__ d_right_row_start,
    const int*    __restrict__ d_right_row_count,
    float epipolar_row_step,
    float vL_tolerance,
    float match_window,
    int*   __restrict__ d_card_lid,               // [i*V3_MAX_CARDS + c]
    float* __restrict__ d_card_hitRx,
    float* __restrict__ d_card_res,               // |hitRx-uR|（诊断/P3 备用）
    int*   __restrict__ d_card_hitRidx,           // 右排序系索引（P4 右占用）
    int*   __restrict__ d_card_count,
    const float* __restrict__ d_ur_bias,          // 逐线 uR 补偿（V3_MAX_LIDS；空＝全零）
    const float4* __restrict__ d_bias_params,     // 4 参数模型 (wx,wy,wz,dtx)；null＝关
    int*   __restrict__ d_left_status,            // orig: >=1 card / 0 none / -1 no-map
    int*   __restrict__ d_max_out,                // 帧内最大卡数（原子；空＝不统计
    // 【150FPS·查表桶索引】（空/0＝二分兜底）
    const int* __restrict__ d_bucket_pool,        // 行内桶下界（相对行首）
    const int* __restrict__ d_bucket_row_off,     // 每行池偏移
    float bucket_min_x, float bucket_inv_w, int bucket_k, int use_buckets,
    // 【前段提前提交】设备计数尾闸：容量发射时尾部陈旧点不得发卡/写状态
    //（默认 nullptr＝旧调用行为逐字不变）
    const int* __restrict__ d_n = nullptr)
{
    // 【B3·TP2】2 点/线程跨步（ILP 换 TLP；探针 kChamp<false,2> 同构实测 −54%）。
    // 四守卫 return→continue（线程界/尾闸/行界/空行）；d_card_count/d_left_status
    // 清零保持在行/空行守卫之前——仅 i∈[0,leftCount)∩[0,*d_n) 清零，与旧版同域。
    const int nthreads = gridDim.x * blockDim.x;
    #pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + kk * nthreads;
    if (i >= leftCount) continue;
    if (d_n != nullptr && i >= *d_n) continue;

    d_card_count[i] = 0;
    d_left_status[i] = -1;

    const float2 p = d_left_pts[i];
    const int row = (int)roundf(p.y / epipolar_row_step);
    if (row < 0 || row >= 8192) continue;
    const int mapCnt = d_map_row_count[row];
    if (mapCnt == 0) continue;

    const float xL = p.x;
    const int start = d_map_row_start[row];

    // first map entry with x >= xL
    // 【B3·PX-A】use_buckets 三态：0=全行二分（JMW_V3BISECT=1 对照兜底）
    //   1=桶直达＋8 步跳走（JMW_V3ANCHOR=0 旧臂） 2=桶锚＋桶内二分（默认）。
    //   桶锚＝紧凑池每活跃行恰 K+1 条单调下界（末条=行尾哨兵 cnt）——kb 计算式
    //   与钳制同旧版逐字一致→同域覆盖＝同风险包络（探针 px1_256 −30%＋TP2 −54%）；
    //   窗内二分与跳走同收敛点；行尾钳制对齐旧版收尾语义。
    int lo;
    if (use_buckets == 2) {
        int b = (int)((xL - bucket_min_x) * bucket_inv_w);
        if (b < 0) b = 0;
        if (b > bucket_k - 1) b = bucket_k - 1;
        const int poolBase = d_bucket_row_off[row];
        int l = start + d_bucket_pool[poolBase + b];
        int h = start + d_bucket_pool[poolBase + b + 1];
        while (l < h) {
            const int mid = (l + h) >> 1;
            if (d_map_byrow[mid].x < xL) l = mid + 1; else h = mid;
        }
        if (l >= start + mapCnt) l = start + mapCnt - 1;   // 行内无 x≥xL（对齐旧版收尾）
        lo = l;
    } else if (use_buckets == 1) {
        // 桶直达＋8 步跳走（JMW_V3ANCHOR=0 旧臂——逐字保留）
        int b = (int)((xL - bucket_min_x) * bucket_inv_w);
        if (b < 0) b = 0;
        if (b > bucket_k - 1) b = bucket_k - 1;
        const int poolBase = d_bucket_row_off[row];
        int j = start + d_bucket_pool[poolBase + b];
        const int jEnd = start + d_bucket_pool[poolBase + b + 1];
        while (j + 8 <= jEnd && d_map_byrow[j + 7].x < xL) j += 8;
        while (j < jEnd && d_map_byrow[j].x < xL) ++j;
        if (j >= start + mapCnt) j = start + mapCnt - 1;
        lo = j;
    } else {
        // 旧二分（JMW_V3BISECT=1 对照兜底）
        lo = start;
        int hi = start + mapCnt - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (d_map_byrow[mid].x < xL) lo = mid + 1; else hi = mid;
        }
    }

    // nearest entry per lid within vL_tolerance (expand from lo in both dirs)
    int nc = 0;
    int locLid[V3_MAX_CARDS];
    float locU[V3_MAX_CARDS];
    float locDx[V3_MAX_CARDS];
    for (int dir = 0; dir < 2; ++dir) {
        int q = (dir == 0) ? lo : lo - 1;
        const int step = (dir == 0) ? 1 : -1;
        while (q >= start && q < start + mapCnt) {
            const float dx = d_map_byrow[q].x - xL;
            if (fabsf(dx) > vL_tolerance) break;
            const int lid = (int)d_map_byrow[q].z;
            const float uR = d_map_byrow[q].y;
            const float adx = fabsf(dx);
            int found = -1;
            for (int c = 0; c < nc; ++c)
                if (locLid[c] == lid) { found = c; break; }
            if (found < 0) {
                if (nc < V3_MAX_CARDS) {
                    locLid[nc] = lid; locU[nc] = uR; locDx[nc] = adx;
                    ++nc;
                }
            } else if (adx < locDx[found]) {
                locDx[found] = adx;
                locU[found] = uR;
            }
            q += step;
        }
    }

    // HIT check each card: right point within +-match_window（取最左原版语义，P1 最近邻
    // 实测致系统性翻转已弃；res/ridx 照记供诊断与后续 P3）
    // uR 偏差补偿（2026-10-10 逐线表 / 2026-10-11 方式2 4 参数模型，互斥）：
    // 消标定/扫描中心提取约定差——实测偏差 kb>0＝表预测 uR 偏低，需加正补偿。
    // 模型＝拟合实验 corr() 同链反号（拟合最优处 corr≈−kb）：视差→Z→左目 3D→
    // 相对投影器方向 ω 一阶旋转→v 向 CMOS 偏移→深度修正 dZ→bias=f·B·dZ/Z²；
    // 逐点计算（线内不同位置不同补偿），全量修正。dtx 一阶对深度无贡献不参与。
    const int rStart = d_right_row_start[row];
    const int rCnt   = d_right_row_count[row];
    float4 mp = make_float4(0.f, 0.f, 0.f, 0.f);
    if (d_bias_params != nullptr) mp = d_bias_params[0];
    int out = 0;
    for (int c = 0; c < nc; ++c) {
        float bias = 0.0f;
        if (d_bias_params != nullptr) {
            const float Z = URB_F * URB_B / fmaxf(xL - locU[c], 1.0f);  // 表预测视差→近似深度
            const float X = (xL - URB_CX) * Z / URB_F;
            const float Y = (p.y - URB_CY) * Z / URB_F;
            const float dx = X - URB_TX, dy = Y - URB_TY, dz = Z - URB_TZ;
            const float ddy = mp.z * dx - mp.x * dz;   // ω×d 一阶旋转（仅 v 向链进深度）
            const float ddz = mp.x * dy - mp.y * dx;
            const float dv  = URB_F * (ddy * dz - dy * ddz) / (dz * dz);
            const float dZ  = dv * Z / dz;
            bias = URB_F * URB_B * dZ / (Z * Z);
        } else if (d_ur_bias != nullptr) {
            bias = d_ur_bias[locLid[c] - 1];
        }
        const float uR = locU[c] + bias;
        float hitRx = -1.0f;
        int hitRidx = -1;
        if (rCnt > 0 && rStart >= 0) {
            int rlo = rStart, rhi = rStart + rCnt - 1;
            while (rlo < rhi) {
                int mid = (rlo + rhi) / 2;
                if (d_right_sorted_pts[mid].x < uR - match_window) rlo = mid + 1; else rhi = mid;
            }
            if (rlo < rStart + rCnt &&
                fabsf(d_right_sorted_pts[rlo].x - uR) <= match_window) {
                hitRx = d_right_sorted_pts[rlo].x;
                hitRidx = rlo;
            }
        }
        if (hitRidx < 0) continue;                // card fails HIT
        d_card_lid[i * V3_MAX_CARDS + out] = locLid[c];
        d_card_hitRx[i * V3_MAX_CARDS + out] = hitRx;
        d_card_res[i * V3_MAX_CARDS + out] = fabsf(hitRx - uR);
        d_card_hitRidx[i * V3_MAX_CARDS + out] = hitRidx;
        ++out;
    }
    d_card_count[i] = out;
    d_left_status[i] = out > 0 ? 1 : 0;
    if (d_max_out != nullptr) atomicMax(d_max_out, out);
    }   // 【B3·TP2】kk 循环收口
}

// V3-K2: segment vote histogram (orig space; seg map from host)
// 票制＝计数制（每 HIT 卡一票，原版语义）；残差定点累加供 P3/诊断（2026-10-10
// 实测：残差加权/每点一票制引入线间系统偏置，已弃）
// segCap：段容量守卫（seg≥cap 弃票——GPU 段表路径防越界，溢出由 host 重做兜底）
__global__ void kernelSegVoteV3(
    const int* __restrict__ d_seg_map,             // orig -> segIdx
    const int* __restrict__ d_card_lid,
    const float* __restrict__ d_card_res,
    const int* __restrict__ d_card_count,
    int leftCount,
    int segCap,
    int* __restrict__ d_seg_votes,                 // [segIdx * V3_MAX_LIDS + (lid-1)]
    int* __restrict__ d_seg_res)                   // [segIdx * V3_MAX_LIDS + (lid-1)]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int seg = d_seg_map[i];
    if (seg < 0 || seg >= segCap) return;
    const int nc = d_card_count[i];
    for (int c = 0; c < nc; ++c) {
        const int lid = d_card_lid[i * V3_MAX_CARDS + c];
        if (lid >= 1 && lid <= V3_MAX_LIDS) {
            atomicAdd(&d_seg_votes[seg * V3_MAX_LIDS + (lid - 1)], 1);
            atomicAdd(&d_seg_res[seg * V3_MAX_LIDS + (lid - 1)],
                      __float2int_rn(d_card_res[i * V3_MAX_CARDS + c] * 256.0f));
        }
    }
}

// V3-K3: per-segment winner (argmax votes，原版语义；残差平票判决 2026-10-10
// 实测无正贡献，暂不启用——d_seg_res 仅累积供离线分析)
__global__ void kernelSegWinnerV3(
    const int* __restrict__ d_seg_votes,
    const int* __restrict__ d_seg_res,
    int segCount,
    int* __restrict__ d_seg_winner)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount) return;
    // 票制结论（2026-10-10 三帧基准实测，98.29%/79.09% 基线）：
    //   P2 残差最小单票制＝87.26%（拒）；P3 平票残差判决＝87.75%（拒）；
    //   全卡计数＋平票取小 lid 具涌现鲁棒性（选中段内近全点持有线号），保持原版。
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        const int v = d_seg_votes[s * V3_MAX_LIDS + l];
        if (v > best) { best = v; bestLid = l + 1; }
    }
    d_seg_winner[s] = bestLid;                     // -1 = no card at all
    (void)d_seg_res;
}

// V3-K4: conflict detection on (row, winner-lid) + drop-both marking.
__global__ void kernelConflictMarkV3(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_seg_map,
    int leftCount,
    const int*    __restrict__ d_seg_winner,
    float epipolar_row_step,
    int* __restrict__ d_cell_seg,                  // [row * V3_MAX_LIDS + (lid-1)], init -1
    int* __restrict__ d_seg_drop)                  // init 0
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int seg = d_seg_map[i];
    if (seg < 0) return;
    const int winner = d_seg_winner[seg];
    if (winner < 1) return;
    const int row = (int)roundf(d_left_pts[i].y / epipolar_row_step);
    if (row < 0 || row >= 8192) return;
    const int cell = row * V3_MAX_LIDS + (winner - 1);
    const int prev = atomicCAS(&d_cell_seg[cell], -1, seg);
    if (prev != -1 && prev != seg) {
        atomicExch(&d_seg_drop[seg], 1);
        atomicExch(&d_seg_drop[prev], 1);
    }
}

// V3-K5: gather flags + temp output (winner card of kept segments)
// P4: 输出时标记右点占用（d_right_status）；保留段中无 winner 卡的点计入 d_diag[0]
__global__ void kernelGatherV3(
    const float2* __restrict__ d_left_pts,
    const int*    __restrict__ d_left_tracks,
    const int*    __restrict__ d_seg_map,
    const int*    __restrict__ d_seg_winner,
    const int*    __restrict__ d_seg_drop,
    const int*    __restrict__ d_card_lid,
    const float*  __restrict__ d_card_hitRx,
    const int*    __restrict__ d_card_hitRidx,
    const int*    __restrict__ d_card_count,
    int leftCount,
    int* __restrict__ d_out_flags,                 // [leftCount]
    float2* __restrict__ d_out_left,               // temp (uncompacted)
    float2* __restrict__ d_out_right,
    int*   __restrict__ d_out_lid,
    int*   __restrict__ d_out_track,
    int*   __restrict__ d_right_status,
    int*   __restrict__ d_diag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    d_out_flags[i] = 0;
    const int seg = d_seg_map[i];
    if (seg < 0) return;
    if (d_seg_drop[seg] != 0) return;              // dropped segment
    const int winner = d_seg_winner[seg];
    if (winner < 1) return;
    const int nc = d_card_count[i];
    for (int c = 0; c < nc; ++c) {
        if (d_card_lid[i * V3_MAX_CARDS + c] == winner) {
            d_out_flags[i] = 1;
            d_out_left[i] = d_left_pts[i];
            d_out_right[i] = make_float2(d_card_hitRx[i * V3_MAX_CARDS + c],
                                         d_left_pts[i].y);
            d_out_lid[i] = winner;
            d_out_track[i] = d_left_tracks[i];
            d_right_status[d_card_hitRidx[i * V3_MAX_CARDS + c]] = 1;   // P4: 右占用
            return;
        }
    }
    atomicAdd(&d_diag[0], 1);                      // P4: 保留段无 winner 卡（原静默丢弃）
}
// K7: status 计数（输出到 d_counts_[slot]）
__global__ void kernelCountStatusV2(
    const int* __restrict__ d_status,
    int count,
    int targetValue,
    int* __restrict__ d_result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    if (d_status[idx] == targetValue) {
        atomicAdd(d_result, 1);
    }
}

// 速度优化：帧内最大卡数（D2H 后按精确 stride 下载卡表——量随行就市 1/4~1/32）
__global__ void kernelMaxCardCntV3(
    const int* __restrict__ d_card_count,
    int count,
    int* __restrict__ d_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    atomicMax(d_out, d_card_count[idx]);
}

// ── GPU segmap（全链速度优化 2026-10-16）：track id → 段序号（rank）──
// 复用左路排序缓冲：keys=track 副本、idx=0..n-1 序列（kernelCopySeq 生成），
// CUB SortPairs → kernelRunHeadsV3（游程头标记）→ CUB ExclusiveSum（游程 rank）
// → kernelSegMapBackV3（回填 d_seg_map[orig]=rank；末位线程写 segCount）
__global__ void kernelCopySeq(
    const int* __restrict__ src,
    int count,
    int* __restrict__ dstKeys,
    int* __restrict__ dstIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dstKeys[idx] = src[idx];
    dstIdx[idx] = idx;
}

__global__ void kernelRunHeadsV3(
    const int* __restrict__ sortedTracks,
    int count,
    int* __restrict__ flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    flags[idx] = (idx == 0 || sortedTracks[idx] != sortedTracks[idx - 1]) ? 1 : 0;
}

__global__ void kernelSegMapBackV3(
    const int* __restrict__ runRank,      // ExclusiveSum(flags)
    const int* __restrict__ flags,
    const int* __restrict__ sortedIdx,
    int count,
    int* __restrict__ segMap,
    int* __restrict__ segCountOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    // rank：头位＝exSum（本游程头之前的头数）；成员位＝exSum−1（剔除自身游程头）
    segMap[sortedIdx[idx]] = flags[idx] ? runRank[idx] : runRank[idx] - 1;
    if (idx == count - 1) {
        const int total = runRank[count - 1] + flags[count - 1];
        *segCountOut = total;
        // 【P0-2 修复】溢出标志：segCount > kSegCap 时由 host 回退
        // segCountOut 指向 d_counts_[2]，d_counts_[3] 已由 Step0 memset 清零
        segCountOut[1] = (total > 4096) ? 1 : 0;   // [2]=segCount, [3]=overflow flag
    }
}

// V3-K3b：winner＋次名同取（GPU 选举，host P6b/P7 免重算；次名供 exclCount）
__global__ void kernelSegWinnerSecondV3(
    const int* __restrict__ d_seg_votes,
    int segCount,
    int* __restrict__ d_seg_winner,
    int* __restrict__ d_seg_second)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount) return;
    int best = 0, bestLid = -1, second = 0, secondLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        const int v = d_seg_votes[s * V3_MAX_LIDS + l];
        if (v > best) { second = best; secondLid = bestLid; best = v; bestLid = l + 1; }
        else if (v > second) { second = v; secondLid = l + 1; }
    }
    d_seg_winner[s] = bestLid;
    d_seg_second[s] = (bestLid < 1) ? -1 : secondLid;   // 无 winner → 次名无意义
}

// V3-K6b：逐段 exclCount（持 S 不持 W 的点数）——host P6b rowOrder 预判零卡下载
__global__ void kernelExclCountV3(
    const int* __restrict__ d_seg_map,
    const int* __restrict__ d_seg_winner,
    const int* __restrict__ d_seg_second,
    const int* __restrict__ d_card_lid,
    const int* __restrict__ d_card_count,
    int leftCount,
    int segCap,
    int* __restrict__ d_excl_cnt)          // [segCap]，已清零
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int seg = d_seg_map[i];
    if (seg < 0 || seg >= segCap) return;   // 值域守卫（防段表异常值越界读）
    const int W = d_seg_winner[seg];
    if (W < 1) return;
    const int S = d_seg_second[seg];
    if (S < 1) return;
    const int nc = d_card_count[i];
    bool hasW = false, hasS = false;
    for (int c = 0; c < nc; ++c) {
        const int lid = d_card_lid[i * V3_MAX_CARDS + c];
        if (lid == W) { hasW = true; break; }
        if (lid == S) hasS = true;
    }
    if (hasS && !hasW) atomicAdd(&d_excl_cnt[seg], 1);
}

// 卡表紧缩（i*32 → i*stride，供线性 D2H；罕见 cardExcl/分裂路径用）
__global__ void kernelTightenCardsV3(
    const int* __restrict__ d_card_lid,
    const int* __restrict__ d_card_count,
    int leftCount,
    int stride,
    int* __restrict__ d_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int nc = d_card_count[i];
    for (int c = 0; c < nc && c < stride; ++c)
        d_out[i * stride + c] = d_card_lid[i * V3_MAX_CARDS + c];
}

// 【速度优化·P6prep GPU 化】行号计算＋段行位图（替 host 逐点 roundf＋bit-OR）。
// roundf(y/step) 与 host 同式同类型 → 逐位一致；位图 OR 序无关 → atomicOr 合法。
// 守卫对齐 host：seg∈[0,segCap)（segmap 值域即段数内）、row∈[0,kMaxRows)。
__global__ void kernelPtRowSegRowsV3(
    const float2* __restrict__ d_left,
    const int* __restrict__ d_seg_map,
    int leftCount,
    int segCap,
    float rowStep,
    int* __restrict__ d_pt_row,            // [leftCount] 出
    unsigned long long* __restrict__ d_seg_rows,   // [segCap][kRowWords] 已清零
    const int* __restrict__ d_seg_count,   // 设备侧段数（值域防御用）
    int* __restrict__ d_viol)              // 值域违规计数（已清零；>0 → host 重做）
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftCount) return;
    const int row = static_cast<int>(roundf(d_left[i].y / rowStep));
    d_pt_row[i] = row;
    const int seg = d_seg_map[i];
    // 【按需下载】值域防御 GPU 化：seg∉[0,segCount) 计数（≡host min/max 扫描）
    if (seg < 0 || seg >= *d_seg_count) { atomicAdd(d_viol, 1); return; }
    if (row < 0 || row >= 8192) return;
    atomicOr(reinterpret_cast<unsigned long long*>(
                 &d_seg_rows[static_cast<size_t>(seg) * (8192 / 64)
                            + static_cast<size_t>(row) / 64]),
             1ULL << (row % 64));
}

// 【速度优化·P6prep GPU 化】段偏移＝排序布局中各段游程头的位置。
// d_left_sorted_idx_ 按轨迹排序且段内升序 i（稳定排序）→ 偏移即段起点位置。
__global__ void kernelSegOffsetsV3(
    const int* __restrict__ d_seg_flags,   // 游程头标记 [leftCount]
    const int* __restrict__ d_seg_runrank, // 游程 rank（段号）[leftCount]
    int leftCount,
    int* __restrict__ d_seg_offsets)       // [segCap] 出
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= leftCount) return;
    if (d_seg_flags[p] != 1) return;
    d_seg_offsets[d_seg_runrank[p]] = p;
}

// 【速度优化·刀② c7stamp GPU 化①】每段位图 popcount → 段内分组数（供基址前缀）。
__global__ void kernelSegGroupCountsV3(
    const unsigned long long* __restrict__ d_seg_rows,  // [segCap][128]
    int segCount,
    int* __restrict__ d_grp_cnt_per_seg)                // [segCount] 出
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount) return;
    int n = 0;
    const unsigned long long* bits = d_seg_rows + static_cast<size_t>(s) * (8192 / 64);
    for (int w = 0; w < 8192 / 64; ++w) n += __popcll(bits[w]);
    d_grp_cnt_per_seg[s] = n;
}

// 【速度优化·刀② c7stamp GPU 化②】段基址前缀（单块串行——segCount≤4096，µs 级）
__global__ void kernelSegGroupBaseV3(
    const int* __restrict__ d_grp_cnt_per_seg,
    int segCount,
    int* __restrict__ d_grp_off)          // [segCount+1] 出（exSum）
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int run = 0;
    d_grp_off[0] = 0;
    for (int s = 0; s < segCount; ++s) {
        run += d_grp_cnt_per_seg[s];
        d_grp_off[s + 1] = run;
    }
}

// 【速度优化·刀② c7stamp GPU 化③】每（段,行）一工人：沿段内点列（升序 i）过滤
// 本行点、double 顺序累加——与 host c7stamp 同一加法序列（段内升序 i×行命中）；
// 输出槽位＝段基址＋本段位图低位 popcount（升序行秩）。无原子、独占写 → 逐位一致。
__global__ void kernelStampGroupsV3(
    const unsigned long long* __restrict__ d_seg_rows,  // [segCap][128]
    const int* __restrict__ d_seg_offsets,              // [segCount]（排序布局段起点）
    const int* __restrict__ d_grp_off,                  // [segCount+1]（分组基址 exSum）
    const int* __restrict__ d_sorted_idx,               // [leftCount]（段内升序 i）
    const int* __restrict__ d_pt_row,                   // [leftCount]
    const float2* __restrict__ d_left,                  // [leftCount]
    int segCount,
    int leftCount,
    int* __restrict__ d_grp_row,                        // [numGroups] 出（段内升序行）
    double* __restrict__ d_grp_sum,                     // [numGroups] 出
    int* __restrict__ d_grp_cnt)                        // [numGroups] 出
{
    int s = blockIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount || r >= 8192) return;
    const unsigned long long* bits = d_seg_rows + static_cast<size_t>(s) * (8192 / 64);
    if (!((bits[r / 64] >> (r % 64)) & 1ULL)) return;   // 本段未覆盖该行
    // 输出槽位：段基址＋本段位图低位 popcount（升序行秩，确定性）
    int rank = 0;
    for (int w = 0; w < r / 64; ++w) rank += __popcll(bits[w]);
    rank += __popcll(bits[r / 64] & ((r % 64) ? ((1ULL << (r % 64)) - 1ULL) : 0ULL));
    // 沿段点列升序 i 顺序累加本行点（double 提升＋顺序加＝host 同序列）
    const int b = d_seg_offsets[s];
    const int e = (s + 1 < segCount) ? d_seg_offsets[s + 1] : leftCount;
    double sum = 0.0;
    int cnt = 0;
    for (int p = b; p < e; ++p) {
        const int i = d_sorted_idx[p];
        if (d_pt_row[i] != r) continue;
        sum += static_cast<double>(d_left[i].x);
        ++cnt;
    }
    const int outSlot = d_grp_off[s] + rank;
    d_grp_row[outSlot] = r;
    d_grp_sum[outSlot] = sum;
    d_grp_cnt[outSlot] = cnt;
}

// 【速度优化·刀③ c7merge GPU 化①】有票标志（同 host hasVotes：26 票任意>0）。
// [v3-fullGPU K1a] stamp point-parallel: blocks iterate segments; points of a
// segment accumulate atomically (atomic ruling, G0a gate); extract is now
// word-parallel (128 threads = 128 bitmap words, shared popcount prefix scan;
// slot = grpOff[s]+exclusive-prefix(word)+within-word rank → ascending row
// order identical to the serial version). Scratch stays zero between segments
// (accumulated rows ⊆ bitmap rows — same-source invariant of
// kernelPtRowSegRowsV3), so per-frame full zeroing is gone (one memset at
// alloc; error paths re-zero before returning).
__global__ void kernelStampPointsV3(
    const unsigned long long* __restrict__ segRows,   // [segCount*128]
    const int* __restrict__ segOff,                    // [segCount] seg end offsets (last=leftCount)
    const int* __restrict__ grpOff,                    // [segCount+1] group base (prefix sums)
    const int* __restrict__ sortedIdx,                 // [leftCount] sorted pos -> orig i
    const int* __restrict__ ptRow,                     // [leftCount] by orig idx
    const float2* __restrict__ leftPts,                // by orig idx
    double* __restrict__ stampSum,                     // [gridDim*8192] per-block scratch
    int* __restrict__ stampCnt,
    int* __restrict__ grpRow, double* __restrict__ grpSum, int* __restrict__ grpCnt,
    const int segCount, const int leftCount)
{
    const int bid = blockIdx.x;
    double* mySum = stampSum + (size_t)bid * 8192;
    int* myCnt = stampCnt + (size_t)bid * 8192;
    __shared__ int sPref[128];
    for (int s = bid; s < segCount; s += gridDim.x) {
        // offsets[s]=段 s 起始；末段止于 leftCount（与 kernelStampGroupsV3 同口径）
        const int p0 = segOff[s];
        const int p1 = (s + 1 < segCount) ? segOff[s + 1] : leftCount;
        for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
            const int i = sortedIdx[p];
            const int row = ptRow[i];
            if (row < 0 || row >= 8192) continue;
            atomicAdd(mySum + row, static_cast<double>(leftPts[i].x));
            atomicAdd(myCnt + row, 1);
        }
        __syncthreads();
        // 并行提取：每线程一字（w<128），shared popcount 独占前缀＋字内升序位序
        const int w = threadIdx.x;
        int pc = 0;
        unsigned long long wb = 0ULL;
        if (w < 128) {
            wb = segRows[(size_t)s * 128 + w];
            pc = __popcll(wb);
            sPref[w] = pc;
        }
        __syncthreads();
        for (int off = 1; off < 128; off <<= 1) {
            int t = 0;
            if (w < 128 && w >= off) t = sPref[w - off];
            __syncthreads();
            if (w < 128) sPref[w] += t;
            __syncthreads();
        }
        if (w < 128) {
            int base = grpOff[s] + sPref[w] - pc;   // 本字前独占前缀
            unsigned long long b = wb;
            while (b) {
                const int r = w * 64 + (__ffsll(b) - 1);
                grpRow[base] = r;
                grpSum[base] = mySum[r];
                grpCnt[base] = myCnt[r];
                mySum[r] = 0.0;
                myCnt[r] = 0;
                ++base;
                b &= b - 1;
            }
        }
        __syncthreads();
    }
}

__global__ void kernelSegHasVotesV3(
    const int* __restrict__ d_seg_votes,   // [segCount][V3_MAX_LIDS]
    int segCount,
    int* __restrict__ d_has)               // [segCount] 出（0/1）
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= segCount) return;
    int any = 0;
    for (int l = 0; l < V3_MAX_LIDS; ++l)
        if (d_seg_votes[static_cast<size_t>(s) * V3_MAX_LIDS + l] > 0) { any = 1; break; }
    d_has[s] = any;
}

// 【速度优化·刀③ c7merge GPU 化②】每对段一线程：归并走行（升序公共行）逐项
// 累加 |xa−xb|——与 host 稠密路径同一加法序列（行升序×fabs(double 减)×double 加）；
// xbar＝gSum/(double)gCnt 与 host 预算 xbar 同操作数同除法。独占写无原子。
// 有票过滤同 host（无票对写 0/0 → 资格判定必不过＝host 不入 CSR 之效）。
__global__ void kernelPairMergeV3(
    const int* __restrict__ gRow,          // [numGroups]（段内升序行）
    const double* __restrict__ gSum,       // [numGroups]
    const int* __restrict__ gCnt,          // [numGroups]
    const int* __restrict__ gOff,          // [segCount+1]
    const int* __restrict__ d_has,         // [segCount]
    int segCount,
    double* __restrict__ pairSum,          // [S*S] 出（pk=a*S+b）
    long long* __restrict__ pairN)         // [S*S] 出
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= segCount * segCount) return;
    const int a = tid / segCount;
    const int b = tid - a * segCount;
    const size_t pk = static_cast<size_t>(a) * segCount + b;
    if (a >= b) { pairSum[pk] = 0.0; pairN[pk] = 0; return; }
    if (!d_has[a] || !d_has[b]) { pairSum[pk] = 0.0; pairN[pk] = 0; return; }
    int ia = gOff[a], ib = gOff[b];
    const int ea = gOff[a + 1], eb = gOff[b + 1];
    double sum = 0.0;
    long long n = 0;
    while (ia < ea && ib < eb) {
        const int ra = gRow[ia], rb = gRow[ib];
        if (ra < rb) ++ia;
        else if (rb < ra) ++ib;
        else {
            const double xa = gSum[ia] / static_cast<double>(gCnt[ia]);
            const double xb = gSum[ib] / static_cast<double>(gCnt[ib]);
            sum += fabs(xa - xb);
            ++n; ++ia; ++ib;
        }
    }
    pairSum[pk] = sum;
    pairN[pk] = n;
}

// ============================================================================
// 【v3全GPU化·K1b】友军合并 GPU 化（原子版裁定）：行桶散布 pair-CSR 三连——
// 均值稠密化 → 行桶转置 → 对贡献散布。语义对齐宿主行桶稠密路径（c7merge）：
// pairN 整数原子＝精确；pairSum 浮点原子累加序非确定 → 末位漂移（G0a 原子
// 漂移门覆盖：友军门 15.0px 处真值分布空谷 3px/20px，期望零翻转）。
// 资格过滤同 host：仅有票段入桶（hasVotes）；无票/无公共行对恒 0（memset 态）。
// ============================================================================

// K1b-1 均值稠密化：盖章 CSR (grp_sum/cnt) → dense[s][r]=sum/cnt（同操作数除法，
// 与 host 预算 xbar 逐字一致）。仅写在场 (s,r)（CSR 槽位独占写）；读侧只访问
// 在场项 → 无需清零。线程 k 二分 grpOff 定位段（≤10 步；ΣR≤leftCount 上界发射）。
__global__ void kernelSegMeanScatter(
    const int* __restrict__ grpOff,        // [segCount+1]（exSum）
    const int* __restrict__ grpRow,        // [numGroups]（段内升序行）
    const double* __restrict__ grpSum,     // [numGroups]
    const int* __restrict__ grpCnt,        // [numGroups]
    double* __restrict__ dense,            // [512*8192] 出
    const int segCount, const int capGroups)   // numGroups＝grpOff[segCount]（设备侧读）
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= capGroups) return;
    const int numGroups = grpOff[segCount];
    if (k >= numGroups) return;
    int lo = 0, hi = segCount;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (grpOff[mid] <= k) lo = mid; else hi = mid;
    }
    dense[static_cast<size_t>(lo) * 8192 + grpRow[k]] =
        grpSum[k] / static_cast<double>(grpCnt[k]);
}

// K1b-2a 行桶计数：扫段行位图（仅有票段），atomicAdd rowCnt[r]（调用方先清零）。
// 置位总数 ΣR ≤ leftCount（每 (段,行) 组 ≥1 点、组间点集不相交）。
__global__ void kernelRowBucketCount(
    const unsigned long long* __restrict__ segRows,  // [segCap][128]
    const int* __restrict__ dHas,          // [segCount]
    const int segCount,
    int* __restrict__ rowCnt)              // [8192] 出
{
    const int s = blockIdx.y;
    if (s >= segCount || !dHas[s]) return;
    const int w = threadIdx.x;
    if (w >= 128) return;
    unsigned long long b = segRows[static_cast<size_t>(s) * 128 + w];
    while (b) {
        const int r = w * 64 + (__ffsll(b) - 1);
        atomicAdd(&rowCnt[r], 1);
        b &= b - 1;
    }
}

// K1b-2b 行桶偏移：rowCnt → rowOff exSum（单块 1024 线程；每线程 8 元素块内扫）。
// 同时把 rowFill 重写为各行起始游标（rowCnt 读入寄存器后覆写，无双读竞态）。
__global__ void kernelRowBucketScan(
    const int* __restrict__ rowCnt,        // [8192]
    int* __restrict__ rowOff,              // [8193] 出（rowOff[0]=0；[8192]=ΣR）
    int* __restrict__ rowFill)             // [8192] 出（初值=rowOff[r]）
{
    __shared__ int sh[1024];
    const int tid = threadIdx.x;
    constexpr int PER = 8;                 // 8192 = 1024*8
    int v[PER];
    const int base = tid * PER;
    int run = 0;
    #pragma unroll
    for (int i = 0; i < PER; ++i) { v[i] = rowCnt[base + i]; run += v[i]; }
    sh[tid] = run;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int t = (tid >= off) ? sh[tid - off] : 0;
        __syncthreads();
        sh[tid] += t;
        __syncthreads();
    }
    int cur = sh[tid] - run;               // 本线程前缀（独占）
    #pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int r = base + i;
        rowOff[r] = cur;
        rowFill[r] = cur;
        cur += v[i];
    }
    if (tid == 1023) rowOff[8192] = sh[1023];
}

// K1b-2c 行桶散置：段 s 追加入行 r 的列表槽（atomicAdd 游标递增）。桶内次序
// 非确定（原子序）——对集与 pairN 与次序无关（位置对 (i<j) 全枚举＋min/max
// 归一）；pairSum 漂移入 G0a 门。
__global__ void kernelRowBucketScatter(
    const unsigned long long* __restrict__ segRows,  // [segCap][128]
    const int* __restrict__ dHas,          // [segCount]
    const int segCount,
    int* __restrict__ rowFill,             // [8192] 游标（初始=rowOff[r]）
    int* __restrict__ segList)             // [ΣR] 出
{
    const int s = blockIdx.y;
    if (s >= segCount || !dHas[s]) return;
    const int w = threadIdx.x;
    if (w >= 128) return;
    unsigned long long b = segRows[static_cast<size_t>(s) * 128 + w];
    while (b) {
        const int r = w * 64 + (__ffsll(b) - 1);
        segList[atomicAdd(&rowFill[r], 1)] = s;
        b &= b - 1;
    }
}

// K1b-3 对贡献散布：每行一块，桶内两两位置对 (i<j) 线程平分（j-i-1≡tid mod B
// 完备划分）→ (mn,mx) 归一 pk=mn*S+mx；|xa−xb| 与操作数次序无关（IEEE 减法
// 反交换＋fabs）＝host 同值。Σk2 线程量（G-1 实测 1e5-1e6）吞吐形态。
__global__ void kernelPairScatter(
    const int* __restrict__ rowOff,        // [8193]
    const int* __restrict__ segList,       // [ΣR]
    const double* __restrict__ dense,      // [512*8192]
    double* __restrict__ pairSum,          // [S*S]（先清零）
    long long* __restrict__ pairN,         // [S*S]（先清零）
    const int S)
{
    const int r = blockIdx.x;
    const int beg = rowOff[r];
    const int cnt = rowOff[r + 1] - beg;
    if (cnt < 2) return;                   // 单段行无对（cnt≤S≤512）
    __shared__ int sSeg[512];
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) sSeg[i] = segList[beg + i];
    __syncthreads();
    for (int i = 0; i < cnt; ++i) {
        const int a0 = sSeg[i];
        const double xa = dense[static_cast<size_t>(a0) * 8192 + r];
        for (int j = i + 1 + threadIdx.x; j < cnt; j += blockDim.x) {
            const int b0 = sSeg[j];
            const int mn = a0 < b0 ? a0 : b0;
            const int mx = a0 < b0 ? b0 : a0;
            const size_t pk = static_cast<size_t>(mn) * S + mx;
            const double d = fabs(xa - dense[static_cast<size_t>(b0) * 8192 + r]);
            atomicAdd(&pairSum[pk], d);
            // Windows 无 (long long*) atomicAdd 重载——位模式等价走 unsigned long long
            atomicAdd(reinterpret_cast<unsigned long long*>(&pairN[pk]), 1ULL);
        }
    }
}

// ============================================================================
// 【v3全GPU化·Step10 设备化】终局统计设备侧归约（单块 1024 线程，grid-stride
// 五遍）——替代四路 D2H（leftstat/rightstat ~240KB＋drop/winner）＋三段宿主
// 扫描循环。须在 kernelGatherV3＋DeviceSelect 之后发射（右状态/计数终值依赖）。
// ============================================================================
__global__ void kernelStep10ReduceV3(
    const int* __restrict__ segDrop,       // [segCount] 0/1
    const int* __restrict__ segWinner,     // [segCount]
    const int* __restrict__ leftStat,      // [leftCount]
    const int* __restrict__ rightStat,     // [rightCount]
    const int* __restrict__ diag,          // [4]（diag[0]=无 winner 卡点数）
    const int* __restrict__ counts,        // [4]（counts[0]=matched）
    int segCount, int leftCount, int rightCount,
    int* __restrict__ stats)               // [7] 出
{
    __shared__ int s[5];
    const int tid = threadIdx.x;
    if (tid < 5) s[tid] = 0;
    __syncthreads();
    for (int i = tid; i < segCount; i += blockDim.x) {
        if (segDrop[i]) atomicAdd(&s[0], 1);
        if (segWinner[i] < 1) atomicAdd(&s[4], 1);
    }
    for (int i = tid; i < leftCount; i += blockDim.x) {
        const int st = leftStat[i];
        if (st == 0) atomicAdd(&s[1], 1);          // noCard
        else if (st == -1) atomicAdd(&s[2], 1);    // noMap
    }
    for (int i = tid; i < rightCount; i += blockDim.x)
        if (rightStat[i] != 0) atomicAdd(&s[3], 1);   // rightUsed
    __syncthreads();
    if (tid < 5) stats[tid] = s[tid];
    if (tid == 5) stats[5] = diag[0];
    if (tid == 6) stats[6] = counts[0];
}

// ============================================================================
// P5 host 侧冲突重选：winner 选举（原 K3 语义）＋ (row,winner) 撞车时输方重选
// 次优（票数 ≥ max(2, 30%×最优)，剔争议 lid），无合格次优才整段弃。
// 确定性：纯 host 整数运算；迭代求解至无冲突或 10 轮上限。
// ============================================================================

namespace {

constexpr int kMaxRows = 8192;                      // ＝ pimpl MAX_EPIPOLAR_ROWS（host 段自给，避免作用域依赖）
constexpr int kRowWords = kMaxRows / 64;             // 128
static_assert(kRowWords == static_cast<int>(K2Ctx::kWordsU), "K2Ctx row words must equal kRowWords");

int electWinnerArgmax(const int* votes26) {          // votes26[lid-1]，原 K3 语义：平票取小 lid
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        if (votes26[l] > best) { best = votes26[l]; bestLid = l + 1; }
    }
    return bestLid;
}

// 剔除 excludeLid 后的最佳次优；门槛：票数 ≥ max(2, 30%×winnerVotes)
int reelectSecond(const int* votes26, int winnerLid, int winnerVotes) {
    const int threshold = std::max(2, (winnerVotes * 3) / 10);
    int best = 0, bestLid = -1;
    for (int l = 0; l < V3_MAX_LIDS; ++l) {
        if (l + 1 == winnerLid) continue;
        if (votes26[l] > best) { best = votes26[l]; bestLid = l + 1; }
    }
    return (best >= threshold) ? bestLid : -1;
}

// P5 v2：位置定序重选——利用"同行内线号随 x 单调"的结构不变量：
// 输方在共享行上的平均 x 相对 keeper 偏右 → 取 contested+1，偏左 → contested−1；
// 几何证据为主、票数为门槛（≥max(2,15%×争议票)）；位置不显著才回退票数次优。
// 返回重选 lid；-1＝无合格选择（弃）。
int reelectByPosition(int loserSeg, int keeperSeg, int contestedLid,
                      const std::vector<int>& h_segmap,
                      const float2* h_left, int leftCount, float rowStep,
                      const std::vector<std::vector<unsigned long long>>& segRows,
                      const std::vector<std::vector<int>>& segVotes)
{
    std::vector<unsigned long long> shared(kRowWords, 0ULL);
    for (int w = 0; w < kRowWords; ++w)
        shared[w] = segRows[loserSeg][w] & segRows[keeperSeg][w];
    double sumL = 0, sumK = 0;
    long long nL = 0, nK = 0;
    for (int i = 0; i < leftCount; ++i) {
        const int seg = h_segmap[static_cast<size_t>(i)];
        if (seg != loserSeg && seg != keeperSeg) continue;
        const int row = static_cast<int>(roundf(h_left[static_cast<size_t>(i)].y / rowStep));
        if (row < 0 || row >= kMaxRows) continue;
        if (!((shared[static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL)) continue;
        if (seg == loserSeg) { sumL += h_left[static_cast<size_t>(i)].x; ++nL; }
        else { sumK += h_left[static_cast<size_t>(i)].x; ++nK; }
    }
    const int contestedVotes = segVotes[loserSeg][contestedLid - 1];
    const int geoThreshold = std::max(2, (contestedVotes * 15) / 100);
    auto votesOk = [&](int lid) {
        return lid >= 1 && lid <= V3_MAX_LIDS && segVotes[loserSeg][lid - 1] >= geoThreshold;
    };
    if (nL > 0 && nK > 0) {
        const double dx = sumL / nL - sumK / nK;    // 输方相对 keeper 的横向位置
        if (dx > 2.0 && votesOk(contestedLid + 1)) return contestedLid + 1;
        if (dx < -2.0 && votesOk(contestedLid - 1)) return contestedLid - 1;
    }
    return reelectSecond(segVotes[loserSeg].data(), contestedLid, contestedVotes);
}

// P6b：段分裂——检测 interp 串线合并段（票双峰＋次线点独占＋行块连续）并按卡归属
// 拆成两个子段。返回本轮分裂次数（调用方迭代至 0）。会原地更新 segmap/votes/rows/
// winner/dropped，并可能扩张 segCount（各容器按需 resize）。
// 【全链速度优化 2026-10-16】CSR 化：点集经 segPts（段→点，升序 i）遍历，替原
// 每候选段 O(N) 全点扫；byRowPts 8192 向量分配改平铺桶（跨段复用）。访问序与
// 原版逐点升序等价（segPts 按升序 i 构建）→ 求和/累加序不变，结果逐位一致。
int splitBimodalSegments(std::vector<int>& h_segmap,
                         const float2* h_left, int leftCount, float rowStep,
                         int*& h_card_lid, int*& h_card_count, int& cardStride,
                         const int* h_exclCnt,                       // GPU 预备 excl 数（空＝旧扫描）
                         const std::function<void()>& ensureCards,   // 罕见路径现场取卡
                         int& segCount,
                         std::vector<std::vector<int>>& segVotes,
                         std::vector<std::vector<unsigned long long>>& segRows,
                         std::vector<int>& winner,
                         std::vector<char>& dropped,
                         const std::vector<int>& ptRow,
                         std::vector<std::vector<int>>& segPts)
{
    const int segCountIn = segCount;
    int splits = 0;
    // 平铺行桶复用缓冲（原版每段 vector<vector<int>>(8192) 分配）
    std::vector<int> rowCnt(kMaxRows + 1, 0);
    std::vector<int> rowOff(kMaxRows + 1, 0);
    std::vector<int> flatPts;
    std::vector<char> movedFlag(leftCount, 0);
    for (int s = 0; s < segCountIn; ++s) {
        if (dropped[s] || winner[s] < 1) continue;
        const int W = winner[s];
        const int* v = segVotes[s].data();
        // ① 双峰：次线票 ≥ max(30, 30%×winner 票)
        int S = -1, sVotes = 0;
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            if (l + 1 != W && v[l] > sVotes) { sVotes = v[l]; S = l + 1; }
        if (S < 1 || sVotes < std::max(30, (v[W - 1] * 3) / 10)) continue;
        // ② 点可分：优先卡独占（持 S 不持 W）；全表多卡下独占少 → 行序分割兜底
        //    （同行内左点属左线右点属右线——与 P5 同一序不变量，同一适用边界）
        //    excl 数优先用 GPU 预备值（kernelExclCountV3，判据同源）——主路径零卡扫描；
        //    仅 cardExcl（≥80%）或无预备值时才落卡扫描（ensureCards 现场取）。
        std::vector<int> exclPts;
        std::vector<unsigned long long> exclRows(kRowWords, 0ULL);
        int exclSz = -1;
        if (h_exclCnt != nullptr) exclSz = h_exclCnt[s];
        bool needCardScan = (exclSz < 0) || (exclSz >= (sVotes * 8) / 10);
        if (needCardScan) {
            ensureCards();   // 幂等：已就位零开销
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int nc = h_card_count[i];
                bool hasW = false, hasS = false;
                for (int c = 0; c < nc; ++c) {
                    const int lid = h_card_lid[static_cast<size_t>(i) * cardStride + c];
                    if (lid == W) hasW = true;
                    else if (lid == S) hasS = true;
                }
                if (!hasS || hasW) continue;
                exclPts.push_back(i);
                const int row = ptRow[static_cast<size_t>(i)];
                if (row >= 0 && row < kMaxRows)
                    exclRows[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            }
            if (exclSz < 0) exclSz = static_cast<int>(exclPts.size());   // 旧路径自算
        }
        bool byRowOrder = false;
        if (exclSz < (sVotes * 8) / 10) {
            // —— 行序分割：该段每行按 x 排序，S<W 时左点归 S、右点归 W（S>W 反向）——
            byRowOrder = true;
            exclPts.clear();
            std::fill(exclRows.begin(), exclRows.end(), 0ULL);
            // 平铺行桶（行内点序＝升序 i，与原版 byRowPts 一致）；窗口化——
            // 只清/扫该段行范围 [minR,maxR]（窗外行本就 0 点，语义不变）
            int minR = kMaxRows, maxR = -1;
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int row = ptRow[static_cast<size_t>(i)];
                if (row < 0 || row >= kMaxRows) continue;
                if (row < minR) minR = row;
                if (row > maxR) maxR = row;
            }
            if (maxR < 0) {   // 无有效行点（原版同：全跳 → exclPts 空 → 量门弃）
                CALIB_LOG_INFO("P6b reject(seg {}): rowOrder qty=0/{} out of band (W={} S={})",
                               s, sVotes, W, S);
                continue;
            }
            for (int r = minR; r <= maxR; ++r) rowCnt[static_cast<size_t>(r)] = 0;
            for (int i : segPts[static_cast<size_t>(s)]) {
                const int row = ptRow[static_cast<size_t>(i)];
                if (row < 0 || row >= kMaxRows) continue;
                ++rowCnt[static_cast<size_t>(row)];
            }
            // 量门预检：行序分割的量＝「≥2 点的行数」（每行取 1 点）——先数行数，
            // 量不符直接弃，免平铺+逐行排序（与原版同判据同结果，只省被弃路径的工）
            {
                int qtyPre = 0;
                for (int r = minR; r <= maxR; ++r)
                    if (rowCnt[static_cast<size_t>(r)] >= 2) ++qtyPre;
                const int lo2 = sVotes * 6 / 10, hi2 = sVotes * 14 / 10;
                if (qtyPre < lo2 || qtyPre > hi2) {
                    CALIB_LOG_INFO("P6b reject(seg {}): rowOrder qty={}/{} out of band (W={} S={})",
                                   s, qtyPre, sVotes, W, S);
                    continue;
                }
            }
            rowOff[static_cast<size_t>(minR)] = 0;
            for (int r = minR; r < maxR; ++r)
                rowOff[static_cast<size_t>(r) + 1] = rowOff[static_cast<size_t>(r)] + rowCnt[static_cast<size_t>(r)];
            const int total = rowOff[static_cast<size_t>(maxR)] + rowCnt[static_cast<size_t>(maxR)];
            flatPts.assign(static_cast<size_t>(total), -1);
            {
                std::vector<int> fill(rowOff.begin() + minR,
                                      rowOff.begin() + static_cast<long long>(maxR) + 1);   // 窗口副本做游标
                for (int i : segPts[static_cast<size_t>(s)]) {
                    const int row = ptRow[static_cast<size_t>(i)];
                    if (row < 0 || row >= kMaxRows) continue;
                    flatPts[static_cast<size_t>(fill[static_cast<size_t>(row) - minR]++)] = i;
                }
            }
            for (int r = minR; r <= maxR; ++r) {
                const int beg = rowOff[static_cast<size_t>(r)];
                const int cnt = rowCnt[static_cast<size_t>(r)];
                if (cnt < 2) continue;               // 单点行不判（保守归 W）
                int* ps = flatPts.data() + beg;
                std::sort(ps, ps + cnt, [&](int a, int b) {
                    return h_left[static_cast<size_t>(a)].x < h_left[static_cast<size_t>(b)].x;
                });
                const int takeIdx = (S < W) ? ps[0] : ps[cnt - 1];   // S 是左线取最左，反之最右
                exclPts.push_back(takeIdx);
                exclRows[static_cast<size_t>(r) / 64] |= 1ULL << (r % 64);
            }
            // 行序分割的量必须与 sVotes 大体吻合（±40%），否则放弃
            const int lo = sVotes * 6 / 10, hi = sVotes * 14 / 10;
            if (static_cast<int>(exclPts.size()) < lo ||
                static_cast<int>(exclPts.size()) > hi) {
                CALIB_LOG_DEBUG("P6b reject(seg {}): rowOrder qty={}/{} out of band (W={} S={})",
                               s, exclPts.size(), sVotes, W, S);
                continue;
            }
        }
        // ③ 行块连续：行集合排序后最长近连续段（缺口≤25 行）覆盖 ≥70% 且 ≥50 行
        std::vector<int> rows;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = exclRows[w];
            for (int row = w * 64; bits; bits >>= 1, ++row)
                if (bits & 1ULL) rows.push_back(row);
        }
        if (rows.size() < 50) {
            CALIB_LOG_DEBUG("P6b reject(seg {}): gate3a rows={} < 50 (W={} S={})",
                           s, rows.size(), W, S);
            continue;
        }
        int bestRun = 1, run = 1;
        for (size_t k = 1; k < rows.size(); ++k) {
            run = (rows[k] - rows[k - 1] <= 25) ? run + 1 : 1;
            if (run > bestRun) bestRun = run;
        }
        if (bestRun < 50 || bestRun * 10 < static_cast<int>(rows.size()) * 7) {
            CALIB_LOG_DEBUG("P6b reject(seg {}): gate3b bestRun={}/{} (W={} S={})",
                           s, bestRun, rows.size(), W, S);
            continue;
        }
        // —— 执行分裂：独占点移入新段 ——
        const int newSeg = segCount++;
        ensureCards();   // 重建两段票直方图需逐点卡集（rowOrder 分裂路径此前可能零卡）
        for (int idx : exclPts) {
            h_segmap[static_cast<size_t>(idx)] = newSeg;
            movedFlag[static_cast<size_t>(idx)] = 1;
        }
        // segPts 同步：新段得 exclPts，原段剔除（保持升序 i）
        segPts.resize(static_cast<size_t>(segCount));
        segPts[static_cast<size_t>(newSeg)] = exclPts;
        std::vector<int> remain;
        remain.reserve(segPts[static_cast<size_t>(s)].size() - exclPts.size());
        for (int i : segPts[static_cast<size_t>(s)])
            if (!movedFlag[static_cast<size_t>(i)]) remain.push_back(i);
        segPts[static_cast<size_t>(s)] = std::move(remain);
        for (int idx : exclPts) movedFlag[static_cast<size_t>(idx)] = 0;
        // 重建两段的票直方图与行位图（votes 整数累加＋位图 OR——序无关，逐位一致）
        std::vector<int> vA(V3_MAX_LIDS, 0), vB(V3_MAX_LIDS, 0);
        std::vector<unsigned long long> rA(kRowWords, 0ULL), rB(kRowWords, 0ULL);
        for (int i : segPts[static_cast<size_t>(s)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row >= 0 && row < kMaxRows)
                rA[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            const int nc = h_card_count[i];
            for (int c = 0; c < nc; ++c) {
                const int lid = h_card_lid[static_cast<size_t>(i) * cardStride + c];
                if (lid >= 1 && lid <= V3_MAX_LIDS) ++vA[lid - 1];
            }
        }
        for (int i : segPts[static_cast<size_t>(newSeg)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row >= 0 && row < kMaxRows)
                rB[static_cast<size_t>(row) / 64] |= 1ULL << (row % 64);
            const int nc = h_card_count[i];
            for (int c = 0; c < nc; ++c) {
                const int lid = h_card_lid[static_cast<size_t>(i) * cardStride + c];
                if (lid >= 1 && lid <= V3_MAX_LIDS) ++vB[lid - 1];
            }
        }
        segVotes[s] = vA;
        segRows[s] = rA;
        segVotes.push_back(vB);
        segRows.push_back(rB);
        winner[s] = electWinnerArgmax(vA.data());
        winner.push_back(S);
        dropped.push_back(0);
        ++splits;
        CALIB_LOG_DEBUG("P6b split(seg {}): winner {} -> keep lid{} + new seg lid{} "
                       "({} pts, mode={}, sVotes={})",
                       s, W, winner[s], S, exclPts.size(),
                       byRowOrder ? "rowOrder" : "cardExcl", sVotes);
    }
    return splits;
}

// ============================================================================
// P7：帧内级联判定（2026-10-12 原型；离线 399 帧验证口径见 TEST/steger_dedup_replay/会话交接-2026-10-08-激光链调试与匹配优化.md 续篇 §3.6）
// 流程：几何友军合并 → 门槛级联（判定线号从行重叠未定段抹除）→ 撞号/间距安全检查
// （违规单位禁赛重跑）。未判定/禁赛段 winner 置 -1（gather 不输出）。
// ============================================================================

struct CascadeCfg {
    int gateH; float gateShare, gateMargin;
    float friendXpx; int friendMinRows;
    float spacingPx, spacingTol;
};

struct CascadeUnit {
    std::vector<int> members;
    std::vector<long long> origVotes;                  // 成员票和（重建用）
    std::vector<long long> votes;                      // 当前票（级联中被消元）
    std::vector<unsigned long long> rows;              // 行位图（kRowWords）
    std::vector<int> rowIdx;                           // 升序在场行（稀疏，速度优化）
    std::vector<double> rowXSum;                       // 对齐 rowIdx（稀疏）
    std::vector<int> rowXCnt;
    bool determined = false, banned = false;
    int detLid = -1;
};

// ============================================================================
// 【v3全GPU化·K2】级联判定单块持久内核（H7+H8 的 GPU 等价实现）
// 语义对齐宿主 cascadeSolve 级联段（impl:1884-1983）：块内确定规约（sh 严格大→
// mg 严格大→先到者胜＝升序 ui），淘汰/安全检查/禁赛重跑/写回全在单块内完成。
// 工作集：票表驻 shared（≤128×32×8B=32KB）；行位图/稀疏行数据驻 L2（全局）。
// 回退：U>128 由宿主侧不发起；轮数>250 置 stats[4]=1 内核空退（宿主走原路径）。
// ============================================================================
// 单块持久内核：1024 线程＝32 warp；warp 轮扫单元（U≤128）；票表驻 shared。
// v2：淘汰＝warp-per-unit（lane 分字 ballot，best 行直读 L2）；安全检查＝
// 稠密行均值预计算＋位图 AND 升序位迭代（加法序＝宿主共享行升序，逐位一致）。
// v3：全局选优并行化（warp 归约＋严格全序比较器 sh>→mg>→u 小 胜——与宿主
// thread0 串行扫描同语义，确定性归约）。
// stats: [0]=determinedCnt [1]=bannedCnt [2]=rounds [3]=iters [4]=fallback(1=未写回)
__global__ void __launch_bounds__(1024, 1)
kernelCascadeSolveK2(const long long* __restrict__ origVotes,
                     const unsigned long long* __restrict__ rows,
                     const int* __restrict__ rowOff,
                     const int* __restrict__ rowIdx,
                     const double* __restrict__ rowXSum,
                     const int* __restrict__ rowCnt,
                     double* __restrict__ meanDense,      // [U*kMaxRows]（设备侧散建）
                     const int* __restrict__ segUnit,
                     int* __restrict__ winner,
                     int* __restrict__ dropped,
                     int* __restrict__ stats,
                     const int U, const int S,
                     const int gateH, const float gateShare, const float gateMargin,
                     const float spacingPx, const float spacingTol,
                     const int maxRounds,
                     const int diagSkip)
{
    constexpr int LIDS = V3_MAX_LIDS;                  // 32
    constexpr int WORDS = kRowWords;                   // 128
    constexpr int MAXU = 128;
    constexpr int MAXROWS = 8192;
    __shared__ long long sVotes[MAXU][LIDS];
    __shared__ double sSh[MAXU], sMg[MAXU];
    __shared__ int sWl[MAXU], sOk[MAXU];
    __shared__ int sDet[MAXU], sBan[MAXU], sLid[MAXU], sViol[MAXU], sDetList[MAXU];
    __shared__ int sBest, sBestLid, sPhase;            // sPhase: 0=继续轮 1=本轮结束 2=中止
    __shared__ int sRowOff[MAXU + 1];                  // v2：CSR 偏移驻 shared（散建/安全检查用）

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    if (tid == 0) { stats[0] = 0; stats[1] = 0; stats[2] = 0; stats[3] = 0; stats[4] = 0; }
    for (int u = tid; u < U; u += blockDim.x) {
        sBan[u] = 0; sDet[u] = 0; sLid[u] = -1;
        #pragma unroll
        for (int l = 0; l < LIDS; ++l) sVotes[u][l] = origVotes[(size_t)u * LIDS + l];
    }
    for (int i = tid; i <= U; i += blockDim.x) sRowOff[i] = rowOff[i];
    __syncthreads();
    // —— v2：稠密行均值散建（一次；位图/CSR 全程不变）——
    // 仅写在场行（AND 命中的行必在场）→ 无需清零 8MB。除法＝宿主同操作数，逐位一致。
    if ((diagSkip & 4) == 0)
    {
        const int total = sRowOff[U];
        for (int p = tid; p < total; p += blockDim.x) {
            int lo = 0, hi = U;                         // 二分定位所属单元（sRowOff 驻 shared）
            while (lo + 1 < hi) { const int mid = (lo + hi) >> 1; if (sRowOff[mid] <= p) lo = mid; else hi = mid; }
            const int u = lo;
            meanDense[(size_t)u * MAXROWS + rowIdx[p]] =
                rowXSum[p] / static_cast<double>(rowCnt[p]);
        }
    }
    __syncthreads();

    long long bannedCnt = 0;
    int totalRounds = 0;
    bool abort_ = false;

    for (int iter = 0; iter < 10; ++iter) {
        if (tid == 0) stats[3] = iter + 1;
        for (int u = tid; u < U; u += blockDim.x) {    // 重初始化（含首轮）
            sDet[u] = 0; sLid[u] = -1; sViol[u] = 0;
            #pragma unroll
            for (int l = 0; l < LIDS; ++l) sVotes[u][l] = origVotes[(size_t)u * LIDS + l];
        }
        __syncthreads();

        for (int round = 0; ; ++round) {
            if ((diagSkip & 8) != 0) break;                // 【诊断】bit3：跳过整个轮循环
            // —— 候选扫描：warp 轮扫单元（lane↔lid 一一对应）——
            for (int u = warp; u < U; u += (blockDim.x >> 5)) {
                const long long v = sVotes[u][lane];
                long long w = v, second = 0, tot = v;
                int wl = (v > 0) ? (lane + 1) : -1;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const long long ow = __shfl_down_sync(0xffffffffu, w, off);
                    const long long osecond = __shfl_down_sync(0xffffffffu, second, off);
                    const long long otot = __shfl_down_sync(0xffffffffu, tot, off);
                    const int owl = __shfl_down_sync(0xffffffffu, wl, off);
                    long long nsecond = (w < ow) ? w : ow;      // 落败的最大值降入 second 池
                    if (osecond > nsecond) nsecond = osecond;
                    if (second > nsecond) nsecond = second;
                    if (ow > w) { w = ow; wl = owl; }           // 平票保低 lane＝先到先胜
                    second = nsecond;
                    tot += otot;
                }
                if (lane == 0) {
                    if (w < gateH) { sOk[u] = 0; }
                    else {
                        const double sh = tot ? static_cast<double>(w) / static_cast<double>(tot) : 0.0;
                        const double mg = second ? static_cast<double>(w) / static_cast<double>(second) : 99.0;
                        if (sh < static_cast<double>(gateShare) || mg < static_cast<double>(gateMargin)) sOk[u] = 0;
                        else { sOk[u] = 1; sSh[u] = sh; sMg[u] = mg; sWl[u] = wl; }
                    }
                }
            }
            __syncthreads();
            // —— v3 全局最优：块内并行归约（严格全序：sh 大胜→mg 大胜→u 小胜）——
            // 与宿主 thread0 升序扫描先到先胜语义等价（全序下归约次序无关）。
            {
                double bSh = -1.0, bMg = 0.0; int bU = -1, bWl = -1;
                for (int u = tid; u < U; u += blockDim.x) {
                    if (!sOk[u] || sDet[u] || sBan[u]) continue;
                    const double sh = sSh[u], mg = sMg[u];
                    const bool better = (bU < 0) || sh > bSh
                        || (sh == bSh && (mg > bMg || (mg == bMg && u < bU)));
                    if (better) { bSh = sh; bMg = mg; bU = u; bWl = sWl[u]; }
                }
                // warp 归约（低 lane 胜＝u 小者与先到者一致）
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const double oSh = __shfl_down_sync(0xffffffffu, bSh, off);
                    const double oMg = __shfl_down_sync(0xffffffffu, bMg, off);
                    const int oU = __shfl_down_sync(0xffffffffu, bU, off);
                    const int oWl = __shfl_down_sync(0xffffffffu, bWl, off);
                    const bool oBetter = (bU < 0) || oSh > bSh
                        || (oSh == bSh && (oMg > bMg || (oMg == bMg && oU < bU)));
                    if (oBetter || bU < 0) { bSh = oSh; bMg = oMg; bU = oU; bWl = oWl; }
                }
                __shared__ double wSh[32]; __shared__ double wMg[32];
                __shared__ int wU[32], wWl[32];
                if (lane == 0) { wSh[warp] = bSh; wMg[warp] = bMg; wU[warp] = bU; wWl[warp] = bWl; }
                __syncthreads();
                if (tid == 0) {
                    ++totalRounds;
                    if (totalRounds > maxRounds) { stats[4] = 1; sPhase = 2; }
                    else {
                        const int nWarps = static_cast<int>(blockDim.x >> 5);
                        for (int wi = 1; wi < nWarps; ++wi) {   // warp0 胜者与后续 warp 归并
                            const int oU2 = wU[wi];
                            if (oU2 < 0) continue;
                            const bool oBetter2 = (bU < 0) || wSh[wi] > bSh
                                || (wSh[wi] == bSh && (wMg[wi] > bMg || (wMg[wi] == bMg && oU2 < bU)));
                            if (oBetter2 || bU < 0) { bSh = wSh[wi]; bMg = wMg[wi]; bU = oU2; bWl = wWl[wi]; }
                        }
                        if (bU < 0) { sPhase = 1; }
                        else {
                            sDet[bU] = 1; sLid[bU] = bWl;
                            sBest = bU; sBestLid = bWl; sPhase = 0;
                        }
                    }
                }
            }
            __syncthreads();
            if (sPhase != 0) break;                    // 1=本轮尽 2=中止
            // —— v2 淘汰：warp-per-unit，lane 分字＋ballot（best 行直读 L2）——
            if ((diagSkip & 1) == 0) {
                const int li = sBestLid - 1;
                const unsigned long long* bestRow = rows + (size_t)sBest * WORDS;
                for (int u = warp; u < U; u += (blockDim.x >> 5)) {
                    if (u == sBest || sDet[u] || sBan[u]) continue;
                    unsigned hit = 0u;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int w = lane * 4 + k;    // 32 lane×4 字＝128 字全覆盖
                        if (bestRow[w] & rows[(size_t)u * WORDS + w]) hit = 1u;
                    }
                    if (__ballot_sync(0xffffffffu, hit) != 0u && lane == 0
                        && li >= 0 && li < LIDS && sVotes[u][li] > 0)
                        sVotes[u][li] = 0;             // lane0 单写（无竞争）
                }
            }
            __syncthreads();
        }
        if (sPhase == 2) { abort_ = true; break; }

        // —— 事后安全检查（撞号/相邻间距）——
        if (tid == 0) {
            int D = 0;
            for (int u = 0; u < U; ++u)
                if (sDet[u] && !sBan[u]) sDetList[D++] = u;
            sBest = D;                                 // 复用寄存含义：D
        }
        __syncthreads();
        const int D = sBest;
        if ((diagSkip & 2) == 0)
        for (int t = tid; t < D * D; t += blockDim.x) {
            const int i = t / D, j = t % D;
            if (i <= j) continue;
            const int A = sDetList[i], B = sDetList[j];
            int lidDiff = sLid[A] - sLid[B];
            if (lidDiff < 0) lidDiff = -lidDiff;
            if (lidDiff > 1) continue;                 // 非撞号非相邻：恒 good
            // v2：位图 AND 升序位迭代＋稠密均值（加法序＝宿主共享行升序，逐位一致）
            double sum = 0.0; long long n = 0;
            for (int w = 0; w < WORDS; ++w) {
                unsigned long long andv = rows[(size_t)A * WORDS + w] & rows[(size_t)B * WORDS + w];
                while (andv) {
                    const int b = __ffsll(andv) - 1;   // 最低位先行＝行升序
                    const int r = w * 64 + b;
                    sum += fabs(meanDense[(size_t)A * MAXROWS + r]
                                - meanDense[(size_t)B * MAXROWS + r]);
                    ++n;
                    andv &= andv - 1;
                }
            }
            if (n == 0) continue;                      // 无共享行（AND 全零）
            const double mdx = sum / static_cast<double>(n);
            const float rhsF = spacingTol * spacingPx; // float 精度乘法＝宿主口径
            bool bad = false;
            if (sLid[A] == sLid[B]) bad = true;
            else if (lidDiff == 1 && fabs(mdx - static_cast<double>(spacingPx)) > static_cast<double>(rhsF))
                bad = true;
            if (bad) { sViol[A] = 1; sViol[B] = 1; }   // 幂等写
        }
        __syncthreads();
        if (tid == 0) {
            int newBan = 0, anyViol = 0;
            for (int u = 0; u < U; ++u)
                if (sViol[u]) { anyViol = 1; if (!sBan[u]) { sBan[u] = 1; ++newBan; } }
            bannedCnt += newBan;
            sBest = anyViol ? 1 : 0;                   // 复用：1=需重跑
        }
        __syncthreads();
        if (sBest == 0) break;                         // 全部通过→进入写回
        // else：viol 非空，iter++ 重跑（votes/确定性位在 iter 头重初始化）
    }

    if (!abort_) {
        // —— 写回（H8）：段→单元映射，banned→dropped，determined→winner ——
        for (int s = tid; s < S; s += blockDim.x) {
            const int u = segUnit[s];
            winner[s] = -1; dropped[s] = 0;
            if (u < 0) continue;
            if (sBan[u]) { dropped[s] = 1; continue; }
            if (sDet[u]) winner[s] = sLid[u];
        }
        if (tid == 0) {
            int detCnt = 0;
            for (int u = 0; u < U; ++u)
                if (sDet[u] && !sBan[u]) ++detCnt;     // 每单元计一次＝宿主口径
            stats[0] = detCnt; stats[1] = static_cast<int>(bannedCnt);
            stats[2] = totalRounds;
        }
    }
}

struct K2Out { bool ok; int detCnt; int bannedCnt; int rounds; int iters; };

// 打包/上传/发射/回读。返回 false＝不可用或轮数超限回退（宿主侧继续原路径，
// 设备输入只读未被污染——K2 不变量）。evMs＝CUDA 事件口径的纯内核耗时。
static bool runCascadeK2(K2Ctx& k2, cudaStream_t stream,
                         const std::vector<CascadeUnit>& units, size_t uCount,
                         int segCount, const CascadeCfg& cfg,
                         const std::vector<int>& segUnit,
                         std::vector<int>& winner, std::vector<char>& dropped,
                         K2Out& out, float& evMs)
{
    evMs = 0.f;
    const size_t U = uCount;
    if (U == 0 || U > K2Ctx::kMaxU || segCount <= 0) return false;
    size_t rowTotal = 0;
    for (size_t u = 0; u < U; ++u) rowTotal += units[u].rowIdx.size();
    if (!k2.ensure(rowTotal, static_cast<size_t>(segCount))) return false;

    size_t rp = 0;
    for (size_t u = 0; u < U; ++u) {
        const CascadeUnit& un = units[u];
        std::memcpy(k2.h_orig + u * V3_MAX_LIDS, un.origVotes.data(),
                    V3_MAX_LIDS * sizeof(long long));
        std::memcpy(k2.h_rows + u * kRowWords, un.rows.data(),
                    kRowWords * sizeof(unsigned long long));
        k2.h_rowOff[u] = static_cast<int>(rp);
        const size_t n = un.rowIdx.size();
        if (n) {
            std::memcpy(k2.h_rowIdx + rp, un.rowIdx.data(), n * sizeof(int));
            std::memcpy(k2.h_rowXSum + rp, un.rowXSum.data(), n * sizeof(double));
            std::memcpy(k2.h_rowCnt + rp, un.rowXCnt.data(), n * sizeof(int));
        }
        rp += n;
    }
    k2.h_rowOff[U] = static_cast<int>(rp);
    std::memcpy(k2.h_segUnit, segUnit.data(), static_cast<size_t>(segCount) * sizeof(int));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaMemcpyAsync(k2.d_orig, k2.h_orig, U * V3_MAX_LIDS * sizeof(long long),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(k2.d_rows, k2.h_rows, U * kRowWords * sizeof(unsigned long long),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(k2.d_rowOff, k2.h_rowOff, (U + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    const size_t rowBytes = rp * sizeof(int);
    if (rowBytes)
        cudaMemcpyAsync(k2.d_rowIdx, k2.h_rowIdx, rowBytes, cudaMemcpyHostToDevice, stream);
    if (rp * sizeof(double))
        cudaMemcpyAsync(k2.d_rowXSum, k2.h_rowXSum, rp * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (rowBytes)
        cudaMemcpyAsync(k2.d_rowCnt, k2.h_rowCnt, rowBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(k2.d_segUnit, k2.h_segUnit, static_cast<size_t>(segCount) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    cudaEventRecord(ev0, stream);
    static const int k2Diag = (getenv("JMW_K2DIAG") != nullptr ? atoi(getenv("JMW_K2DIAG")) : 0);
    kernelCascadeSolveK2<<<1, 1024, 0, stream>>>(
        k2.d_orig, k2.d_rows, k2.d_rowOff, k2.d_rowIdx, k2.d_rowXSum, k2.d_rowCnt,
        k2.d_meanDense,
        k2.d_segUnit, k2.d_winner, k2.d_dropped, k2.d_stats,
        static_cast<int>(U), segCount,
        cfg.gateH, cfg.gateShare, cfg.gateMargin, cfg.spacingPx, cfg.spacingTol, 250,
        k2Diag);
    cudaEventRecord(ev1, stream);
    cudaMemcpyAsync(k2.h_winner, k2.d_winner, static_cast<size_t>(segCount) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(k2.h_dropped, k2.d_dropped, static_cast<size_t>(segCount) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(k2.h_stats, k2.d_stats, 5 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&evMs, ev0, ev1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    if (cudaGetLastError() != cudaSuccess) return false;
    if (k2.h_stats[4] != 0) return false;              // 轮数超限回退（未写回）
    for (int s = 0; s < segCount; ++s) {
        winner[static_cast<size_t>(s)] = k2.h_winner[s];
        dropped[static_cast<size_t>(s)] = static_cast<char>(k2.h_dropped[s]);
    }
    out.ok = true;
    out.detCnt = k2.h_stats[0]; out.bannedCnt = k2.h_stats[1];
    out.rounds = k2.h_stats[2]; out.iters = k2.h_stats[3];
    return true;
}


// 【全链速度优化 2026-10-16】行稀疏化重写：原版 O(S²×8192) 稠密行扫（友军合并＋
// 安全检查）与 S×8192×12B 稠密 xSum/xCnt 阵列改为「段→升序行集」稀疏结构＋
// 行桶配对累加＋归并游走。累加顺序与原版逐行升序严格一致（同一加法序列），
// 票整数/位图 OR 序无关 → 结果逐位一致。
int cascadeSolve(int segCount,
                 const std::vector<int>& h_segmap,
                 const float2* h_left, int leftCount, float rowStep,
                 const std::vector<int>& ptRow,
                 const std::vector<std::vector<int>>& segPts,
                 const std::vector<std::vector<int>>& segVotes,
                 const std::vector<std::vector<unsigned long long>>& segRows,
                 std::vector<int>& winner,
                 std::vector<char>& dropped,
                 const CascadeCfg& cfg,
                 int* outBanned, int* outMerged,
                 std::string* tbd = nullptr,   // 【测速工具】子阶段明细（可空）
                  // 【速度优化·刀②】GPU 分组 CSR 直取（空＝host 旧法构建）
                  const int* gRow = nullptr, const double* gSum = nullptr,
                  const int* gCnt = nullptr, const int* gOff = nullptr,
                  // 【速度优化·刀③】GPU 配对结果直取（空＝host 旧法构建）
                  const double* gpuPairSum = nullptr, const long long* gpuPairN = nullptr,
                  // 【v3全GPU化·K2】级联判定 GPU 路径（空＝host 原路径）
                  K2Ctx* k2 = nullptr, cudaStream_t k2Stream = nullptr)
{
    // —— 每段稀疏行数据（升序行＋xSum/xCnt；行值来自 ptRow 无 clamp）——
    // 构建序：段内点升序 i 累加（与原版 segXs 逐点累加序一致）→ 逐位一致
    const auto cT0_ = std::chrono::steady_clock::now();
    auto cMark_ = cT0_;
    auto cSince_ = [&cMark_]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - cMark_).count();
        cMark_ = now;
        return ms;
    };
    // 【速度优化】大数组跨帧复用（thread_local 池——v3 有 inProcess_ 单线程守卫）；
    // resize/assign 复用容量，运算序不变 → 逐位一致
    thread_local std::vector<std::vector<int>> segRowIdxP;
    thread_local std::vector<std::vector<double>> segXSumP;
    thread_local std::vector<std::vector<int>> segXCntP;
    thread_local std::vector<double> stampSumP;
    thread_local std::vector<int> stampCntP;
    auto& segRowIdx = segRowIdxP;
    auto& segXSum = segXSumP;
    auto& segXCnt = segXCntP;
    auto& stampSum = stampSumP;
    auto& stampCnt = stampCntP;
    segRowIdx.resize(static_cast<size_t>(segCount));
    segXSum.resize(static_cast<size_t>(segCount));
    segXCnt.resize(static_cast<size_t>(segCount));
    if (gRow != nullptr && gSum != nullptr && gCnt != nullptr && gOff != nullptr) {
        // 【速度优化·刀②】GPU 分组 CSR 直取：切片拷入池（位图驱动的独占工人已在
        // GPU 上按「段内升序 i×行命中」同一加法序列算好 → 逐位一致）
        for (int s = 0; s < segCount; ++s) {
            const int b = gOff[s], e = gOff[s + 1];
            segRowIdx[static_cast<size_t>(s)].assign(gRow + b, gRow + e);
            segXSum[static_cast<size_t>(s)].assign(gSum + b, gSum + e);
            segXCnt[static_cast<size_t>(s)].assign(gCnt + b, gCnt + e);
        }
    } else {
    for (int s = 0; s < segCount; ++s) {
        segRowIdx[static_cast<size_t>(s)].clear();
        segXSum[static_cast<size_t>(s)].clear();
        segXCnt[static_cast<size_t>(s)].clear();
    }
    stampSum.assign(kMaxRows, 0.0);
    stampCnt.assign(kMaxRows, 0);
    for (int s = 0; s < segCount; ++s) {
        for (int i : segPts[static_cast<size_t>(s)]) {
            const int row = ptRow[static_cast<size_t>(i)];
            if (row < 0 || row >= kMaxRows) continue;
            stampSum[static_cast<size_t>(row)] += h_left[static_cast<size_t>(i)].x;
            ++stampCnt[static_cast<size_t>(row)];
        }
            // 行集升序枚举：segRows 位图（与段有效行点一一对应）——免 touched 排序
            const auto& bits_ = segRows[static_cast<size_t>(s)];
            size_t n = 0;
            for (int w = 0; w < kRowWords; ++w) {
                unsigned long long b = bits_[static_cast<size_t>(w)];
                while (b) { n += (b & 1ULL); b >>= 1ULL; }
            }
            if (n == 0) continue;
            segRowIdx[static_cast<size_t>(s)].resize(n);
            segXSum[static_cast<size_t>(s)].resize(n);
            segXCnt[static_cast<size_t>(s)].resize(n);
            size_t k = 0;
            for (int w = 0; w < kRowWords; ++w) {
                unsigned long long b = bits_[static_cast<size_t>(w)];
                for (int row = w * 64; b; b >>= 1ULL, ++row) {
                    if (!(b & 1ULL)) continue;
                    segRowIdx[static_cast<size_t>(s)][k] = row;
                    segXSum[static_cast<size_t>(s)][k] = stampSum[static_cast<size_t>(row)];
                    segXCnt[static_cast<size_t>(s)][k] = stampCnt[static_cast<size_t>(row)];
                    stampSum[static_cast<size_t>(row)] = 0.0;
                    stampCnt[static_cast<size_t>(row)] = 0;
                    ++k;
                }
            }
        }
    }
    const double msStamp_ = cSince_();   // 【测速工具】行数据构建
    auto hasVotes = [&](int s) {
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            if (segVotes[static_cast<size_t>(s)][static_cast<size_t>(l)] > 0) return true;
        return false;
    };
    // —— 友军合并（union-find：共享行≥minRows 且均距≤friendXpx，有票段）——
    // 行桶配对：每行只对「该行在场的有票段」做两两累加（升序行→对 (a,b) 的
    // 累加序列与原版稠密 r 升序扫描逐项一致）；合格对按 (a,b) 字典序合并——
    // 与原版遍历序一致 → union-find 状态逐位一致。
    std::vector<int> parent(static_cast<size_t>(segCount));
    for (int s = 0; s < segCount; ++s) parent[static_cast<size_t>(s)] = s;
    auto find = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    long long mergePairs = 0;
    if (gpuPairSum != nullptr && gpuPairN != nullptr && segCount > 0 && segCount <= 512) {
        // 【速度优化·刀③】GPU 配对结果直取：资格判定＋并查集合并决策留 host
        // （判定式与遍历序与稠密路径逐字相同 → union-find 状态逐位一致）
        const int S = segCount;
        static const bool v3ab_ = (getenv("JMW_V3PAIR_AB") != nullptr);
        if (v3ab_) {
        // 【验证】单元级 A/B：同帧再算 host 稠密路径，逐位比对 GPU 配对值
        {
            std::vector<double> psumAB(static_cast<size_t>(S) * S, 0.0);
            std::vector<long long> pnAB(static_cast<size_t>(S) * S, 0);
            std::vector<int> rsc(kMaxRows + 1, 0), rso(kMaxRows + 1, 0);
            for (int s = 0; s < S; ++s) {
                if (!hasVotes(s)) continue;
                for (int r : segRowIdx[s]) ++rsc[r];
            }
            for (int r = 0; r < kMaxRows; ++r) rso[r + 1] = rso[r] + rsc[r];
            std::vector<int> rsi(rso.begin(), rso.end() - 1);
            std::vector<int> data(rso[kMaxRows]);
            std::vector<double> xb(rso[kMaxRows]);
            for (int s = 0; s < S; ++s) {
                if (!hasVotes(s)) continue;
                for (size_t k = 0; k < segRowIdx[s].size(); ++k) {
                    const size_t sl = rsi[segRowIdx[s][k]]++;
                    data[sl] = s;
                    xb[sl] = segXSum[s][k] / segXCnt[s][k];
                }
            }
            for (int r = 0; r < kMaxRows; ++r) {
                for (int ia = rso[r]; ia < rso[r] + rsc[r]; ++ia) {
                    const int a = data[ia];
                    for (int ib = ia + 1; ib < rso[r] + rsc[r]; ++ib) {
                        const int b = data[ib];
                        psumAB[a * S + b] += std::fabs(xb[ia] - xb[ib]);
                        ++pnAB[a * S + b];
                    }
                }
            }
            long long nDiff = 0, nSumDiff = 0;
            for (size_t k2 = 0; k2 < psumAB.size(); ++k2) {
                if (pnAB[k2] != gpuPairN[k2]) ++nDiff;
                else if (psumAB[k2] != gpuPairSum[k2]) ++nSumDiff;
            }
            fprintf(stderr, "[v3pairAB] S=%d nMismatch=%lld sumBitDiff=%lld (of %zu)\n",
                    S, nDiff, nSumDiff, psumAB.size());
        }
        }
        for (int a = 0; a < S; ++a) {
            for (int b = a + 1; b < S; ++b) {
                const size_t pk = static_cast<size_t>(a) * S + b;
                if (gpuPairN[pk] >= cfg.friendMinRows
                    && gpuPairSum[pk] / gpuPairN[pk] <= cfg.friendXpx) {
                    parent[static_cast<size_t>(find(b))] = find(a); ++mergePairs;
                }
            }
        }
    } else if (segCount > 0 && segCount <= 512) {
        // —— 行主序稠密对累加（S≤512：S²×16B ≤ 4MB，L2 友好）——
        // 每行只遍在场有票段；对 (a,b) 的 sum 累加序列＝行升序（与原版稠密
        // r 升序扫描同一加法序列）→ 逐位一致；合并按 (a,b) 字典序＝原版遍历序。
        // 【速度优化】大数组 thread_local 池复用（assign 归零，容量保留）
        const int S = segCount;
        thread_local std::vector<double> psumP;
        thread_local std::vector<long long> pnP;
        thread_local std::vector<int> rowSegCntP, rowSegOffP, rowSegDataP, curP;
        thread_local std::vector<double> rowSegXbP;
        auto& psum = psumP;
        auto& pn = pnP;
        auto& rowSegCnt = rowSegCntP;
        auto& rowSegOff = rowSegOffP;
        auto& rowSegData = rowSegDataP;
        auto& rowSegXb = rowSegXbP;
        auto& cur = curP;
        psum.assign(static_cast<size_t>(S) * S, 0.0);
        pn.assign(static_cast<size_t>(S) * S, 0);
        rowSegCnt.assign(static_cast<size_t>(kMaxRows) + 1, 0);
        for (int s = 0; s < S; ++s) {
            if (!hasVotes(s)) continue;
            for (int r : segRowIdx[static_cast<size_t>(s)])
                ++rowSegCnt[static_cast<size_t>(r)];
        }
        rowSegOff.assign(static_cast<size_t>(kMaxRows) + 1, 0);
        for (int r = 0; r < kMaxRows; ++r)
            rowSegOff[static_cast<size_t>(r) + 1] = rowSegOff[static_cast<size_t>(r)]
                                                    + rowSegCnt[static_cast<size_t>(r)];
        const int totalEntries = rowSegOff[static_cast<size_t>(kMaxRows)];
        rowSegData.resize(static_cast<size_t>(totalEntries));   // 每槽全量覆写，免清
        rowSegXb.resize(static_cast<size_t>(totalEntries));
        cur.assign(rowSegOff.begin(), rowSegOff.end() - 1);     // 游标副本
        for (int s = 0; s < S; ++s) {
            if (!hasVotes(s)) continue;
            const auto& ridx = segRowIdx[static_cast<size_t>(s)];
            for (size_t k = 0; k < ridx.size(); ++k) {
                const size_t slot = static_cast<size_t>(cur[static_cast<size_t>(ridx[k])]++);
                rowSegData[slot] = s;
                rowSegXb[slot] = segXSum[static_cast<size_t>(s)][k]
                                 / segXCnt[static_cast<size_t>(s)][k];   // 预算 xbar（同操作数）
            }
        }
        for (int r = 0; r < kMaxRows; ++r) {
            const int beg = rowSegOff[static_cast<size_t>(r)];
            const int cnt = rowSegCnt[static_cast<size_t>(r)];
            for (int ia = beg; ia < beg + cnt; ++ia) {
                const int a = rowSegData[static_cast<size_t>(ia)];
                const double xa = rowSegXb[static_cast<size_t>(ia)];
                for (int ib = ia + 1; ib < beg + cnt; ++ib) {
                    const int b = rowSegData[static_cast<size_t>(ib)];
                    const size_t pk = static_cast<size_t>(a) * S + b;
                    psum[pk] += std::fabs(xa - rowSegXb[static_cast<size_t>(ib)]);
                    ++pn[pk];
                }
            }
        }
        for (int a = 0; a < S; ++a) {
            for (int b = a + 1; b < S; ++b) {
                const size_t pk = static_cast<size_t>(a) * S + b;
                if (pn[pk] >= cfg.friendMinRows && psum[pk] / pn[pk] <= cfg.friendXpx) {
                    parent[static_cast<size_t>(find(b))] = find(a); ++mergePairs;
                }
            }
        }
    } else {
        // 对主归并游走（大 S 回退路径；对 (a,b) 字典序＝原版遍历序，共享行升序
        // ＝原版稠密 r 升序扫描等价子列——同一加法序列 → 逐位一致）
        for (int a = 0; a < segCount; ++a) {
            if (!hasVotes(a)) continue;
            for (int b = a + 1; b < segCount; ++b) {
                if (!hasVotes(b)) continue;
                const auto& ra = segRowIdx[static_cast<size_t>(a)];
                const auto& rb = segRowIdx[static_cast<size_t>(b)];
                const auto& sa = segXSum[static_cast<size_t>(a)];
                const auto& sb = segXSum[static_cast<size_t>(b)];
                const auto& ca = segXCnt[static_cast<size_t>(a)];
                const auto& cb = segXCnt[static_cast<size_t>(b)];
                double sum = 0; long long n = 0;
                size_t pa = 0, pb = 0;
                while (pa < ra.size() && pb < rb.size()) {
                    const int r1 = ra[pa], r2 = rb[pb];
                    if (r1 < r2) ++pa;
                    else if (r2 < r1) ++pb;
                    else {
                        sum += std::fabs(sa[pa] / ca[pa] - sb[pb] / cb[pb]);
                        ++n; ++pa; ++pb;
                    }
                }
                if (n >= cfg.friendMinRows && sum / n <= cfg.friendXpx) {
                    parent[static_cast<size_t>(find(b))] = find(a); ++mergePairs;
                }
            }
        }
    }
    *outMerged = static_cast<int>(mergePairs);
    const double msMerge_ = cSince_();   // 【测速工具】友军合并
    // —— 单元组装（成员升序聚合；行数据稀疏化——rowIdx 升序＋对齐 XSum/XCnt，
    //    替原 8192 稠密双数组零初始化。每行加法序＝成员升序，与原版一致）——
    // 【速度优化】std::map→平铺 root→unit 表；单元 thread_local 池（容量复用）；
    // 查找/遍历语义与 map 版一致（map 无序依赖）→ 结果逐位一致
    thread_local std::vector<CascadeUnit> unitsP;
    thread_local std::vector<int> rootToUnitP;
    auto& units = unitsP;
    auto& rootToUnit = rootToUnitP;
    units.reserve(static_cast<size_t>(segCount) + 8);   // 引用不失效保障
    rootToUnit.assign(static_cast<size_t>(segCount), -1);
    size_t uCount = 0;
    for (int s = 0; s < segCount; ++s) {
        if (!hasVotes(s)) continue;
        const int root = find(s);
        int ui = rootToUnit[static_cast<size_t>(root)];
        if (ui < 0) {
            ui = static_cast<int>(uCount);
            rootToUnit[static_cast<size_t>(root)] = ui;
            if (uCount == units.size()) units.emplace_back();
            CascadeUnit& nu = units[uCount];
            nu.members.clear();
            nu.origVotes.assign(V3_MAX_LIDS, 0);
            nu.votes.assign(V3_MAX_LIDS, 0);
            nu.rows.assign(kRowWords, 0ULL);
            nu.determined = false; nu.banned = false; nu.detLid = -1;
            ++uCount;
        }
        CascadeUnit& u = units[static_cast<size_t>(ui)];
        u.members.push_back(s);
        for (int l = 0; l < V3_MAX_LIDS; ++l)
            u.origVotes[static_cast<size_t>(l)] += segVotes[static_cast<size_t>(s)][static_cast<size_t>(l)];
        for (int w = 0; w < kRowWords; ++w)
            u.rows[static_cast<size_t>(w)] |= segRows[static_cast<size_t>(s)][static_cast<size_t>(w)];
    }
    // 稀疏行数据：位图提取升序行集 → 每行按成员升序累加（二分定位成员行数据）
    for (size_t uu = 0; uu < uCount; ++uu) {
        CascadeUnit& u = units[uu];
        u.rowIdx.clear();
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = u.rows[static_cast<size_t>(w)];
            for (int row = w * 64; bits; bits >>= 1, ++row)
                if (bits & 1ULL) u.rowIdx.push_back(row);
        }
        const size_t n = u.rowIdx.size();
        u.rowXSum.assign(n, 0.0);
        u.rowXCnt.assign(n, 0);
        if (u.members.size() == 1) {
            // 快路径（常态：友军合并数 0）：单成员＝行集同源，直拷零搜索
            const int m = u.members.front();
            u.rowXSum = segXSum[static_cast<size_t>(m)];
            u.rowXCnt = segXCnt[static_cast<size_t>(m)];
            continue;
        }
        for (int m : u.members) {
            const auto& ridx = segRowIdx[static_cast<size_t>(m)];
            const auto& xs = segXSum[static_cast<size_t>(m)];
            const auto& xc = segXCnt[static_cast<size_t>(m)];
            for (size_t k = 0; k < ridx.size(); ++k) {
                const size_t slot = static_cast<size_t>(
                    std::lower_bound(u.rowIdx.begin(), u.rowIdx.end(), ridx[k])
                    - u.rowIdx.begin());
                u.rowXSum[slot] += xs[k];   // 行槽加法序＝成员升序（与原版一致）
                u.rowXCnt[slot] += xc[k];
            }
        }
    }
    auto rowsOverlap = [](const std::vector<unsigned long long>& A,
                          const std::vector<unsigned long long>& B) {
        for (size_t w = 0; w < A.size(); ++w) if (A[w] & B[w]) return true;
        return false;
    };
    const double msUnit_ = cSince_();    // 【测速工具】单元组装
    long long bannedCnt = 0;
    // 【v3全GPU化·G-1 埋点】轮数/迭代数分布统计（K2 持久内核容量依据）
    int statRounds = 0, statIters = 0;
    // —— 【v3全GPU化·K2】级联判定 GPU 路径（H7+H8 单块持久内核）——
    // 过渡期混合形态：stamp/merge/unit 仍宿主构建；级联段整体 GPU 化。
    // 成功＝winner/dropped 已由 K2 写回；失败/超限＝落回下方宿主原路径。
    bool k2Done = false; int k2Det = 0; long long k2Banned = 0; float k2EvMs = 0.f;
    if (k2 != nullptr && uCount > 0 && uCount <= K2Ctx::kMaxU && segCount > 0) {
        thread_local std::vector<int> segUnitP;
        segUnitP.assign(static_cast<size_t>(segCount), -1);
        for (int s = 0; s < segCount; ++s) {
            if (!hasVotes(s)) continue;
            const int uiW = rootToUnit[static_cast<size_t>(find(s))];
            segUnitP[static_cast<size_t>(s)] = uiW;    // -1 保留（无单元）
        }
        K2Out k2o{};
        if (runCascadeK2(*k2, k2Stream, units, uCount, segCount, cfg, segUnitP,
                         winner, dropped, k2o, k2EvMs)) {
            k2Done = true; k2Det = k2o.detCnt; k2Banned = k2o.bannedCnt;
            statRounds = k2o.rounds; statIters = k2o.iters;
        }
    }
    // —— 安全迭代级联（宿主原路径：K2 未启用/回退时执行）——
    thread_local std::vector<int> detP, violP;
    auto& det = detP;
    auto& viol = violP;
    if (!k2Done)
    for (int iter = 0; iter < 10; ++iter) {
        ++statIters;
        for (size_t uu = 0; uu < uCount; ++uu) {
            CascadeUnit& u = units[uu];
            u.votes = u.origVotes;
            u.determined = false; u.detLid = -1;
        }
        for (int round = 0; ; ++round) {
            ++statRounds;   // 【v3全GPU化·G-1 埋点】含末次空扫描（与 K2 每轮成本同口径）
            int best = -1; double bestShare = -1, bestMg = 0;
            for (size_t ui = 0; ui < uCount; ++ui) {
                CascadeUnit& u = units[ui];
                if (u.determined || u.banned) continue;
                long long w = 0, tot = 0, second = 0; int wl = -1;
                for (int l = 0; l < V3_MAX_LIDS; ++l) {
                    tot += u.votes[static_cast<size_t>(l)];
                    if (u.votes[static_cast<size_t>(l)] > w) {
                        second = w; w = u.votes[static_cast<size_t>(l)]; wl = l + 1;
                    } else if (u.votes[static_cast<size_t>(l)] > second)
                        second = u.votes[static_cast<size_t>(l)];
                }
                if (w < cfg.gateH) continue;
                const double sh = tot ? static_cast<double>(w) / tot : 0;
                const double mg = second ? static_cast<double>(w) / second : 99.0;
                if (sh < cfg.gateShare || mg < cfg.gateMargin) continue;
                if (sh > bestShare || (sh == bestShare && mg > bestMg)) {
                    bestShare = sh; bestMg = mg; best = static_cast<int>(ui);
                }
            }
            if (best < 0) break;
            CascadeUnit& bu = units[static_cast<size_t>(best)];
            long long w = 0; int wl = -1;
            for (int l = 0; l < V3_MAX_LIDS; ++l)
                if (bu.votes[static_cast<size_t>(l)] > w) { w = bu.votes[static_cast<size_t>(l)]; wl = l + 1; }
            bu.determined = true; bu.detLid = wl;
            for (size_t uu = 0; uu < uCount; ++uu) {
                CascadeUnit& u = units[uu];
                if (u.determined || u.banned || &u == &bu) continue;
                if (rowsOverlap(bu.rows, u.rows) && u.votes[static_cast<size_t>(wl - 1)] > 0)
                    u.votes[static_cast<size_t>(wl - 1)] = 0;
            }
        }
        // ===== 事后安全检查（撞号 / 相邻间距）——行稀疏归并游走 =====
        det.clear();
        for (size_t ui = 0; ui < uCount; ++ui)
            if (units[ui].determined && !units[ui].banned) det.push_back(static_cast<int>(ui));
        viol.clear();
        for (size_t i = 0; i < det.size(); ++i) for (size_t j = i + 1; j < det.size(); ++j) {
            CascadeUnit& A = units[static_cast<size_t>(det[i])];
            CascadeUnit& B = units[static_cast<size_t>(det[j])];
            const int lidDiff = std::abs(A.detLid - B.detLid);
            if (lidDiff > 1) continue;                  // 非撞号非相邻：bad 恒 false，免归并
            if (!rowsOverlap(A.rows, B.rows)) continue;
            // 共享行升序归并（与原版稠密 r 升序＋cnt>0 过滤同一加法序列）
            const auto& ra = A.rowIdx;
            const auto& rb = B.rowIdx;
            double sum = 0; long long n = 0;
            size_t pa = 0, pb = 0;
            while (pa < ra.size() && pb < rb.size()) {
                const int r1 = ra[pa], r2 = rb[pb];
                if (r1 < r2) ++pa;
                else if (r2 < r1) ++pb;
                else {
                    sum += std::fabs(A.rowXSum[pa] / A.rowXCnt[pa]
                                     - B.rowXSum[pb] / B.rowXCnt[pb]);
                    ++n; ++pa; ++pb;
                }
            }
            const double mdx = n ? sum / n : 0;
            bool bad = false;
            if (A.detLid == B.detLid) bad = true;                       // 撞号
            else if (lidDiff == 1                                      // 相邻间距
                     && std::fabs(mdx - cfg.spacingPx) > cfg.spacingTol * cfg.spacingPx)
                bad = true;
            if (bad) { viol.push_back(det[i]); viol.push_back(det[j]); }
        }
        if (viol.empty()) break;
        std::sort(viol.begin(), viol.end());
        viol.erase(std::unique(viol.begin(), viol.end()), viol.end());
        for (int v : viol) { units[static_cast<size_t>(v)].banned = true; ++bannedCnt; }
    }
    if (k2Done)
        *outBanned = static_cast<int>(k2Banned);
    else
        *outBanned = static_cast<int>(bannedCnt);
    const double msCasc_ = cSince_();    // 【测速工具】级联＋安全检查
    // —— 写回 ——
    int determinedCnt = 0;
    if (k2Done) {
        determinedCnt = k2Det;           // K2 已写回 winner/dropped
    } else
    for (int s = 0; s < segCount; ++s) {
        winner[static_cast<size_t>(s)] = -1;
        dropped[static_cast<size_t>(s)] = 0;
        if (!hasVotes(s)) continue;
        const int root = find(s);
        const int uiW = rootToUnit[static_cast<size_t>(root)];
        if (uiW < 0) continue;
        CascadeUnit& u = units[static_cast<size_t>(uiW)];
        if (u.banned) { dropped[static_cast<size_t>(s)] = 1; continue; }
        if (u.determined) {
            winner[static_cast<size_t>(s)] = u.detLid;
            if (std::find(u.members.begin(), u.members.end(), s) == u.members.begin())
                ++determinedCnt;                       // 每单元计一次
        }
    }
    if (tbd) {  // 【测速工具】子阶段明细（写回并入级联段计量）
        *tbd += fmt::format(" c7stamp={:.2f} c7merge={:.2f} c7unit={:.2f} c7casc={:.2f}"
                            " c7segs={} c7units={} c7dets={} c7rounds={} c7iters={}",
                            msStamp_, msMerge_, msUnit_, msCasc_ + cSince_(),
                            segCount, uCount, determinedCnt, statRounds, statIters);
        if (k2Done)
            *tbd += fmt::format(" k2=1 k2ev={:.3f}", k2EvMs);   // 事件口径 ms
    }
    return determinedCnt;
}

// 返回 true=有冲突被处理（需要再迭代）
bool resolveConflictsOnce(int segCount,
                           const std::vector<int>& h_segmap,
                           const float2* h_left, int leftCount, float rowStep,
                           const std::vector<std::vector<int>>& segVotes,
                           const std::vector<std::vector<unsigned long long>>& segRows,
                           std::vector<int>& winner,
                           std::vector<char>& dropped,
                           int* reelectedCnt,
                           int* forcedDropCnt)
{
    // 每 (row,lid) 胞的占用段（最多记 2 个即可判定冲突；第 3+ 个由迭代兜底）
    std::vector<int> cellA(static_cast<size_t>(kMaxRows) * V3_MAX_LIDS, -1);
    std::vector<int> cellB(static_cast<size_t>(kMaxRows) * V3_MAX_LIDS, -1);
    for (int s = 0; s < segCount; ++s) {
        if (dropped[s] || winner[s] < 1) continue;
        const int lidIdx = winner[s] - 1;
        for (int w = 0; w < kRowWords; ++w) {
            unsigned long long bits = segRows[s][w];
            for (int row = w * 64; bits; bits >>= 1, ++row) {
                if (!(bits & 1ULL)) continue;
                const size_t cell = static_cast<size_t>(row) * V3_MAX_LIDS + lidIdx;
                if (cellA[cell] < 0) cellA[cell] = s;
                else if (cellB[cell] < 0 && cellA[cell] != s) cellB[cell] = s;
            }
        }
    }
    // 处理首个冲突胞。v3 序感知裁决（2026-10-10 实测主导模式＝右侧上线的段误投向下）：
    //   共享行上定左右 → 左段保争议 lid、右段改判 contested+1（双向票数门槛）；
    //   位置不显著/门槛不过 → 回退 v1（票多者保、输方 v2 位置重选/弃）。一次一改，改后重扫。
    for (size_t cell = 0; cell < cellA.size(); ++cell) {
        if (cellB[cell] < 0) continue;
        const int sa = cellA[cell], sb = cellB[cell];
        const int contestedLid = static_cast<int>(cell % V3_MAX_LIDS) + 1;
        const int va = segVotes[sa][contestedLid - 1];
        const int vb = segVotes[sb][contestedLid - 1];
        // —— v3b：逐行序投票（段均值在合并大段上失义；逐行比较两段行内均值，多数票定左右）——
        long long rowsALeft = 0, rowsBLeft = 0;
        {
            std::vector<double> sumA(kMaxRows, 0.0), sumB(kMaxRows, 0.0);
            std::vector<int> cntA(kMaxRows, 0), cntB(kMaxRows, 0);
            for (int i = 0; i < leftCount; ++i) {
                const int seg = h_segmap[static_cast<size_t>(i)];
                if (seg != sa && seg != sb) continue;
                const int row = static_cast<int>(roundf(h_left[static_cast<size_t>(i)].y / rowStep));
                if (row < 0 || row >= kMaxRows) continue;
                if (!((segRows[sa][static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL)
                    || !((segRows[sb][static_cast<size_t>(row) / 64] >> (row % 64)) & 1ULL))
                    continue;
                if (seg == sa) { sumA[row] += h_left[static_cast<size_t>(i)].x; ++cntA[row]; }
                else { sumB[row] += h_left[static_cast<size_t>(i)].x; ++cntB[row]; }
            }
            for (int r = 0; r < kMaxRows; ++r) {
                if (cntA[r] == 0 || cntB[r] == 0) continue;
                const double d = sumA[r] / cntA[r] - sumB[r] / cntB[r];
                if (d > 2.0) ++rowsBLeft;            // A 行内均值大 → A 在右
                else if (d < -2.0) ++rowsALeft;
            }
        }
        const bool orderable = (rowsALeft + rowsBLeft >= 5)
                               && (rowsALeft > rowsBLeft * 2 || rowsBLeft > rowsALeft * 2);
        if (orderable) {
            const int leftSeg = (rowsALeft > rowsBLeft) ? sa : sb;
            const int rightSeg = (leftSeg == sa) ? sb : sa;
            const int upLid = contestedLid + 1;
            const auto votesOk = [&](int s, int lid) {
                return lid >= 1 && lid <= V3_MAX_LIDS
                       && segVotes[s][lid - 1] >= std::max(2, (segVotes[s][contestedLid - 1] * 15) / 100);
            };
            CALIB_LOG_INFO("P5 conflict: segs {}/{} lid={} votesA={} votesB={} rowVote {}/{} "
                           "left={} vLeftLid={} vRightUp={}",
                           sa, sb, contestedLid, va, vb, rowsALeft, rowsBLeft, leftSeg,
                           segVotes[leftSeg][contestedLid - 1], segVotes[rightSeg][upLid - 1]);
            if (votesOk(rightSeg, upLid) && votesOk(leftSeg, contestedLid)) {
                winner[leftSeg] = contestedLid;
                winner[rightSeg] = upLid;
                *reelectedCnt += 2;
                return true;
            }
        } else {
            CALIB_LOG_INFO("P5 conflict (no order): segs {}/{} lid={} votesA={} votesB={} "
                           "rowVote {}/{}",
                           sa, sb, contestedLid, va, vb, rowsALeft, rowsBLeft);
        }
        // —— 回退 v1/v2 ——
        const int loser = (va >= vb) ? sb : sa;
        const int keeper = (loser == sa) ? sb : sa;
        const int second = reelectByPosition(loser, keeper, contestedLid,
                                             h_segmap, h_left, leftCount, rowStep,
                                             segRows, segVotes);
        if (second >= 1) {
            winner[loser] = second;
            ++*reelectedCnt;
        } else {
            dropped[loser] = 1;
            ++*forcedDropCnt;
        }
        return true;
    }
    return false;
}

}  // namespace

// ============================================================================
// process — 优化版流水线：仅末尾一次主机同步
// ============================================================================

LaserMatchScanResultV3 LaserMatchScanCudaV3::Impl::process(
    const cv::cuda::GpuMat& d_left_points,
    const cv::cuda::GpuMat& d_left_line_ids,
    const cv::cuda::GpuMat& d_right_points,
    const cv::cuda::GpuMat& d_right_line_ids,
    cv::cuda::Stream& stream)
{
#ifndef NDEBUG
    ScopedFlagV2 guard(&inProcess_);
#endif

    // ── 速度沙箱阶段计时（诊断用；Release 下保留，量级 μs~ms）──
    const auto tT0_ = std::chrono::steady_clock::now();
    auto tMark_ = tT0_;
    auto tSince_ = [&tMark_]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tMark_).count();
        tMark_ = now;
        return ms;
    };
    std::string tBreakdown_;
    // 【仪表税开关】JMW_NOTBD=1：跳过分相计时串拼装＋stamps/pairs 计时事件——
    // 纯工程口径（实测该仪表占宿主开销约 1/3；gpu_stress 对照实验用）
    static const bool noBd_ = (getenv("JMW_NOTBD") != nullptr);
    // 【v3全GPU化·回退记账】fb=0 全热路径；1=溢出重做 2=值域重做 3=分裂作废
    // 4=K1b 分配失败（G3 分桶剔除回退帧用；[v3-time]/LastTimingBreakdown 双口径）
    int fbCode = 0;
    // 【150FPS·g1 拆解】sub＝Step1-5 提交耗时；wait＝count 回传＋sync（含内核执行）
    double g1SubMs_ = 0.0, g1WaitMs_ = 0.0;
    const auto g1Sub0_ = std::chrono::steady_clock::now();

    LaserMatchScanResultV3 result;

    try {
        if (!tempSet_) {
            result.success = false;
            result.message = "Temperature not set: call setCurrentTemperature() first";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        const int leftCount = d_left_points.rows * d_left_points.cols;
        const int rightCount = d_right_points.rows * d_right_points.cols;

        result.totalLeftPoints = leftCount;
        result.totalRightPoints = rightCount;

        if (leftCount == 0 || rightCount == 0) {
            result.success = true;
            result.message = "No points to match";
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.d_left_status = std::make_shared<cv::cuda::GpuMat>();
            result.d_right_status = std::make_shared<cv::cuda::GpuMat>();
            result.matchedCount = 0;
            return result;
        }

        if (!allocateBuffers(leftCount, rightCount)) {
            result.success = false;
            result.message = "GPU buffer allocation failed";
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
        cudaError_t kernel_err;

        // ── Step 0: 计数清零 [match, exclL, exclR, overflow] ＋ diag 清零 ──
        cudaMemsetAsync(d_counts_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);
        cudaMemsetAsync(d_diag_.ptr<int>(), 0, 4 * sizeof(int), cuda_stream);

        // ---- V3 Step 1-4 (right side only): keys/sort/scatter/CSR ----
        {
            int grid = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGenKeysIdxV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_right_points.ptr<float2>(),
                rightCount,
                params_.epipolar_row_step,
                reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
                d_right_idx_in_.ptr<int>());
        }

        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_, cub_temp_size_,
            reinterpret_cast<unsigned int*>(d_right_keys_in_.ptr<int>()),
            reinterpret_cast<unsigned int*>(d_right_keys_out_.ptr<int>()),
            d_right_idx_in_.ptr<int>(),
            d_right_sorted_idx_.ptr<int>(),
            rightCount, 0, sizeof(unsigned int) * 8, cuda_stream);

        // ====【150FPS·发射图化】scatter→CSR→发卡 五连发图 ====
        // 变尺寸发射的 WDDM 重打包开销（~130µs×5）→ 图单发（~10µs）。逐帧
        // SetParams 刷网格/计数/gen 指针；JMW_V3GRAPH=0 或失败→普通发射兜底。
        const int gen = status_gen_ & 1;   // （原 Step5 声明并入——后续步骤仍用）
        {
            // 【裁定】图化实测仅 +1FPS 且间歇崩溃 ~1/8（驱动层，无 stderr）——
            // 默认关（JMW_V3GRAPH=1 显式开）；growOnly 大头收益与此无关
            const bool fgOn = getenv("JMW_V3GRAPH") != nullptr
                              && getenv("JMW_V3GRAPH")[0] == '1' && !fg_.disabled;
            // —— 参数刷新（图/普通路径共用赋值区）——
            auto& fg = fg_;
            fg.sc_rightPts = d_right_points.ptr<float2>();
            fg.sc_rightCount = rightCount;
            fg.sc_rightLids = d_right_line_ids.ptr<int>();
            fg.sc_sortedIdx = d_right_sorted_idx_.ptr<int>();
            fg.sc_outPts = d_right_sorted_pts_.ptr<float2>();
            fg.sc_outLids = d_right_sorted_lids_.ptr<int>();
            fg.csr_keys = reinterpret_cast<const unsigned int*>(d_right_keys_out_.ptr<int>());
            fg.csr_rightCount = rightCount;
            fg.csr_rowStart = d_right_row_start_.ptr<int>();
            fg.csr_rowCount = d_right_row_count_.ptr<int>();
            fg.msRS_ptr = d_right_row_start_.ptr<int>();
            fg.msRS_bytes = MAX_EPIPOLAR_ROWS * sizeof(int);
            fg.msRC_ptr = d_right_row_count_.ptr<int>();
            fg.msRC_bytes = MAX_EPIPOLAR_ROWS * sizeof(int);
            fg.msSt_ptr = d_right_status_[gen].ptr<int>();
            fg.msSt_bytes = (size_t)rightCount * sizeof(int);
            fg.ic_leftPts = d_left_points.ptr<float2>();
            fg.ic_leftCount = leftCount;
            fg.ic_mapByrow = d_map_byrow_.ptr<float4>();
            fg.ic_rowStart = d_map_row_start_.ptr<int>();
            fg.ic_rowCount = d_map_row_count_.ptr<int>();
            fg.ic_rightSorted = d_right_sorted_pts_.ptr<float2>();
            fg.ic_rRowStart = d_right_row_start_.ptr<int>();
            fg.ic_rRowCount = d_right_row_count_.ptr<int>();
            fg.ic_rowStep = params_.epipolar_row_step;
            fg.ic_vLtol = params_.vL_tolerance;
            fg.ic_matchWin = params_.match_threshold;
            fg.ic_lid = d_card_lid_.ptr<int>();
            fg.ic_hitRx = d_card_hitRx_.ptr<float>();
            fg.ic_res = d_card_res_.ptr<float>();
            fg.ic_hitRidx = d_card_hitRidx_.ptr<int>();
            fg.ic_count = d_card_count_.ptr<int>();
            fg.ic_urBias = urBiasActive_ ? d_ur_bias_.ptr<float>() : nullptr;
            fg.ic_biasParams = urBiasModelActive_ ? d_bias_params_.ptr<float4>() : nullptr;
            fg.ic_leftStatus = d_left_status_[gen].ptr<int>();
            fg.ic_maxOut = d_diag_.ptr<int>() + 1;
            fg.ic_bucketPool = (bucketK_ > 0 && !d_bucket_pool_.empty())
                                   ? d_bucket_pool_.ptr<int>() : nullptr;
            fg.ic_bucketRowOff = (bucketK_ > 0 && !d_bucket_row_off_.empty())
                                     ? d_bucket_row_off_.ptr<int>() : nullptr;
            fg.ic_bMinX = bucketMinX_;
            fg.ic_bInvW = bucketInvW_;
            fg.ic_bK = bucketK_;
            fg.ic_useB = issueUseBuckets_();

            const int gridR = (rightCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int gridL = issueGrid_(leftCount);

            bool usedGraph = false;
            if (fgOn) {
                if (!fg.captured) {
                    // 首帧：捕获（WDDM 教训：捕获前必须排空流——先同步）
                    cudaStreamSynchronize(cuda_stream);
                    if (cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeThreadLocal)
                        == cudaSuccess) {
                    } else { fg.disabled = true; }
                }
                if (!fg.disabled) {
                    // 捕获帧：顺序发射（进图）；已捕获帧：SetParams＋单发
                    if (!fg.captured) {
                        // —— 捕获路径：按当前帧参数发射（被捕获进图）——
                        kernelScatterSortedV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                            d_right_points.ptr<float2>(),
                            d_right_line_ids.ptr<int>(),
                            d_right_sorted_idx_.ptr<int>(),
                            rightCount,
                            d_right_sorted_pts_.ptr<float2>(),
                            d_right_sorted_lids_.ptr<int>());
                        cudaMemsetAsync(fg.msRS_ptr, 0x7F, fg.msRS_bytes, cuda_stream);
                        cudaMemsetAsync(fg.msRC_ptr, 0, fg.msRC_bytes, cuda_stream);
                        kernelBuildCSRV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                            reinterpret_cast<const unsigned int*>(d_right_keys_out_.ptr<int>()),
                            rightCount, MAX_EPIPOLAR_ROWS,
                            d_right_row_start_.ptr<int>(),
                            d_right_row_count_.ptr<int>());
                        cudaMemsetAsync(fg.msSt_ptr, 0, fg.msSt_bytes, cuda_stream);
                        kernelIssueCardsV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
                            d_left_points.ptr<float2>(), leftCount,
                            d_map_byrow_.ptr<float4>(),
                            d_map_row_start_.ptr<int>(), d_map_row_count_.ptr<int>(),
                            d_right_sorted_pts_.ptr<float2>(),
                            d_right_row_start_.ptr<int>(), d_right_row_count_.ptr<int>(),
                            params_.epipolar_row_step, params_.vL_tolerance,
                            params_.match_threshold,
                            d_card_lid_.ptr<int>(), d_card_hitRx_.ptr<float>(),
                            d_card_res_.ptr<float>(), d_card_hitRidx_.ptr<int>(),
                            d_card_count_.ptr<int>(),
                            fg.ic_urBias, fg.ic_biasParams,
                            d_left_status_[gen].ptr<int>(), d_diag_.ptr<int>() + 1,
                            fg.ic_bucketPool, fg.ic_bucketRowOff,
                            fg.ic_bMinX, fg.ic_bInvW, fg.ic_bK, fg.ic_useB);
                        cudaGraph_t g = nullptr;
                        if (cudaStreamEndCapture(cuda_stream, &g) == cudaSuccess && g) {
                            fg.graph = g;
                            if (cudaGraphInstantiateWithFlags(&fg.exec, g, 0)
                                == cudaSuccess) {
                                // 取节点句柄（按拓扑序：scatter,msRS,msRC,CSR,msSt,issue）
                                size_t nn = 0;
                                cudaGraphGetNodes(g, nullptr, &nn);
                                std::vector<cudaGraphNode_t> nodes(nn);
                                cudaGraphGetNodes(g, nodes.data(), &nn);
                                if (nn == 6) {
                                    fg.nScatter = nodes[0]; fg.nMsetRS = nodes[1];
                                    fg.nMsetRC = nodes[2]; fg.nCSR = nodes[3];
                                    fg.nMsetSt = nodes[4]; fg.nIssue = nodes[5];
                                    // 构建稳定参数描述（kernelParams 指向 fg_ 字段）
                                    fg.scArgs[0]=&fg.sc_rightPts; fg.scArgs[1]=&fg.sc_rightLids;
                                    fg.scArgs[2]=&fg.sc_sortedIdx; fg.scArgs[3]=&fg.sc_rightCount;
                                    fg.scArgs[4]=&fg.sc_outPts; fg.scArgs[5]=&fg.sc_outLids;
                                    fg.csrArgs[0]=&fg.csr_keys; fg.csrArgs[1]=&fg.csr_rightCount;
                                    fg.csrMaxRows = MAX_EPIPOLAR_ROWS;
                                    fg.csrArgs[2] = &fg.csrMaxRows;
                                    fg.csrArgs[3]=&fg.csr_rowStart; fg.csrArgs[4]=&fg.csr_rowCount;
                                    fg.icArgs[0]=&fg.ic_leftPts; fg.icArgs[1]=&fg.ic_leftCount;
                                    fg.icArgs[2]=&fg.ic_mapByrow; fg.icArgs[3]=&fg.ic_rowStart;
                                    fg.icArgs[4]=&fg.ic_rowCount; fg.icArgs[5]=&fg.ic_rightSorted;
                                    fg.icArgs[6]=&fg.ic_rRowStart; fg.icArgs[7]=&fg.ic_rRowCount;
                                    fg.icArgs[8]=&fg.ic_rowStep; fg.icArgs[9]=&fg.ic_vLtol;
                                    fg.icArgs[10]=&fg.ic_matchWin; fg.icArgs[11]=&fg.ic_lid;
                                    fg.icArgs[12]=&fg.ic_hitRx; fg.icArgs[13]=&fg.ic_res;
                                    fg.icArgs[14]=&fg.ic_hitRidx; fg.icArgs[15]=&fg.ic_count;
                                    fg.icArgs[16]=&fg.ic_urBias; fg.icArgs[17]=&fg.ic_biasParams;
                                    fg.icArgs[18]=&fg.ic_leftStatus; fg.icArgs[19]=&fg.ic_maxOut;
        fg.icArgs[20]=&fg.ic_bucketPool; fg.icArgs[21]=&fg.ic_bucketRowOff;
        fg.icArgs[22]=&fg.ic_bMinX; fg.icArgs[23]=&fg.ic_bInvW;
        fg.icArgs[24]=&fg.ic_bK; fg.icArgs[25]=&fg.ic_useB;
        fg.icArgs[26]=&fg.ic_dN;   // 【B3·R6 修】27 参对齐（d_n 尾闸；process 精确
                                   // N 路恒 nullptr＝d_n 引入前语义——修复既有错位）
                                    fg.scP.func = (void*)kernelScatterSortedV3;
                                    fg.scP.gridDim = dim3(gridR); fg.scP.blockDim = dim3(BLOCK_SIZE);
                                    fg.scP.sharedMemBytes = 0; fg.scP.kernelParams = fg.scArgs;
                                    fg.csrP.func = (void*)kernelBuildCSRV3;
                                    fg.csrP.gridDim = dim3(gridR); fg.csrP.blockDim = dim3(BLOCK_SIZE);
                                    fg.csrP.sharedMemBytes = 0; fg.csrP.kernelParams = fg.csrArgs;
                                    fg.icP.func = (void*)kernelIssueCardsV3;
                                    fg.icP.gridDim = dim3(gridL); fg.icP.blockDim = dim3(BLOCK_SIZE);
                                    fg.icP.sharedMemBytes = 0; fg.icP.kernelParams = fg.icArgs;
                                    // memset 参数描述（二维风格：elementSize=1 字节填充）
                                    fg.msRsP.dst = fg.msRS_ptr; fg.msRsP.value = 0x7F;
                                    fg.msRsP.elementSize = 1; fg.msRsP.pitch = 0;
                                    fg.msRsP.width = fg.msRS_bytes; fg.msRsP.height = 1;
                                    fg.msRcP.dst = fg.msRC_ptr; fg.msRcP.value = 0;
                                    fg.msRcP.elementSize = 1; fg.msRcP.pitch = 0;
                                    fg.msRcP.width = fg.msRC_bytes; fg.msRcP.height = 1;
                                    fg.msStP.dst = fg.msSt_ptr; fg.msStP.value = 0;
                                    fg.msStP.elementSize = 1; fg.msStP.pitch = 0;
                                    fg.msStP.width = fg.msSt_bytes; fg.msStP.height = 1;
                                    fg.captured = true;
                                } else {
                                    fg.disabled = true;
                                }
                            } else { fg.disabled = true; }
                        } else { fg.disabled = true; }
                        // 捕获不执行——首帧也要真跑：下面统一 graphLaunch
                        if (fg.captured) {
                            cudaGraphLaunch(fg.exec, cuda_stream);
                            usedGraph = true;
                        }
                    } else {
                        // —— 常态帧：刷新参数＋单发 ——
                        fg.scP.gridDim = dim3(gridR);
                        fg.csrP.gridDim = dim3(gridR);
                        fg.icP.gridDim = dim3(gridL);
                        fg.msStP.width = fg.msSt_bytes;
                        fg.msStP.dst = fg.msSt_ptr;
                        bool ok = cudaGraphExecKernelNodeSetParams(fg.exec, fg.nScatter, &fg.scP) == cudaSuccess
                               && cudaGraphExecMemsetNodeSetParams(fg.exec, fg.nMsetRS, &fg.msRsP) == cudaSuccess
                               && cudaGraphExecMemsetNodeSetParams(fg.exec, fg.nMsetRC, &fg.msRcP) == cudaSuccess
                               && cudaGraphExecKernelNodeSetParams(fg.exec, fg.nCSR, &fg.csrP) == cudaSuccess
                               && cudaGraphExecMemsetNodeSetParams(fg.exec, fg.nMsetSt, &fg.msStP) == cudaSuccess
                               && cudaGraphExecKernelNodeSetParams(fg.exec, fg.nIssue, &fg.icP) == cudaSuccess;
                        if (ok) {
                            cudaGraphLaunch(fg.exec, cuda_stream);
                            usedGraph = true;
                        } else {
                            fg.disabled = true;
                            cudaGetLastError();
                        }
                    }
                }
            }
            if (!usedGraph) {
                // —— 普通发射兜底（图关闭/失败）——
                {
                    kernelScatterSortedV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_right_points.ptr<float2>(),
                        d_right_line_ids.ptr<int>(),
                        d_right_sorted_idx_.ptr<int>(),
                        rightCount,
                        d_right_sorted_pts_.ptr<float2>(),
                        d_right_sorted_lids_.ptr<int>());
                }

                cudaMemsetAsync(d_right_row_start_.ptr<int>(), 0x7F,
                                MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
                cudaMemsetAsync(d_right_row_count_.ptr<int>(), 0,
                                MAX_EPIPOLAR_ROWS * sizeof(int), cuda_stream);
                {
                    kernelBuildCSRV3<<<gridR, BLOCK_SIZE, 0, cuda_stream>>>(
                        reinterpret_cast<const unsigned int*>(d_right_keys_out_.ptr<int>()),
                        rightCount, MAX_EPIPOLAR_ROWS,
                        d_right_row_start_.ptr<int>(),
                        d_right_row_count_.ptr<int>());
                }
                cudaMemsetAsync(d_right_status_[gen].ptr<int>(), 0, rightCount * sizeof(int), cuda_stream);
                {
                    kernelIssueCardsV3<<<gridL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_left_points.ptr<float2>(),
                        leftCount,
                        d_map_byrow_.ptr<float4>(),
                        d_map_row_start_.ptr<int>(),
                        d_map_row_count_.ptr<int>(),
                        d_right_sorted_pts_.ptr<float2>(),
                        d_right_row_start_.ptr<int>(),
                        d_right_row_count_.ptr<int>(),
                        params_.epipolar_row_step,
                        params_.vL_tolerance,
                        params_.match_threshold,
                        d_card_lid_.ptr<int>(),
                        d_card_hitRx_.ptr<float>(),
                        d_card_res_.ptr<float>(),
                        d_card_hitRidx_.ptr<int>(),
                        d_card_count_.ptr<int>(),
                        urBiasActive_ ? d_ur_bias_.ptr<float>() : nullptr,
                        urBiasModelActive_ ? d_bias_params_.ptr<float4>() : nullptr,
                        d_left_status_[gen].ptr<int>(),
                        d_diag_.ptr<int>() + 1,
                        bucketK_ > 0 && !d_bucket_pool_.empty()
                            ? d_bucket_pool_.ptr<int>() : nullptr,
                        bucketK_ > 0 && !d_bucket_row_off_.empty()
                            ? d_bucket_row_off_.ptr<int>() : nullptr,
                        bucketMinX_, bucketInvW_, bucketK_,
                        issueUseBuckets_());
                }
            }
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("sort/CSR stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            return result;
        }

        // ---- V3 Step 6+7 (GPU)：段表（CUB segmap）＋投票＋winner/second 选举＋
        //      exclCount 预备——全链速度优化 2026-10-16 单同步重构：
        //      段表/选举/预判全部上 GPU，host 一次批量 D2H 后零往返进 P6b/P7 ----
        int segCount = 0;                       // 非 const：P6b 分裂可扩
        std::vector<int> h_segmap;
        int* h_cards = nullptr;
        int* h_cardcnt = nullptr;
        int cardStride = 0;
        bool cardsReady_ = false;
        int* h_votesH = nullptr;
        float2* h_leftH = nullptr;
        int* h_diagI = nullptr;
        int* h_exclI = nullptr;                 // GPU exclCount（kSegCap）
        int* h_winnerI = nullptr;
            int* h_secondI = nullptr;  // 【瘦身】死传输已删——d_seg_second_ 仅设备侧消费
        int* h_cntI = nullptr;
        auto ensureCards_ = [&]() {             // 罕见路径：紧缩＋线性 D2H（rowOrder 分裂执行/cardExcl 用）
            if (cardsReady_) return;
            int ms = h_diagI ? h_diagI[1] : V3_MAX_CARDS;
            if (ms < 1) ms = 1;
            if (ms > V3_MAX_CARDS) ms = V3_MAX_CARDS;
            d_cards_tight_.create(1, leftCount * ms, CV_32SC1);
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelTightenCardsV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_card_lid_.ptr<int>(), d_card_count_.ptr<int>(), leftCount, ms,
                d_cards_tight_.ptr<int>());
            h_cards = hpin_cards_.ensure(static_cast<size_t>(leftCount)
                                         * static_cast<size_t>(ms));
            h_cardcnt = hpin_cardcnt_.ensure(static_cast<size_t>(leftCount));
            cudaMemcpyAsync(h_cards, d_cards_tight_.ptr<int>(),
                            static_cast<size_t>(leftCount) * ms * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_cardcnt, d_card_count_.ptr<int>(),
                            static_cast<size_t>(leftCount) * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            cardStride = ms;
            cardsReady_ = true;
        };
        {
            // —— GPU segmap：track id 排序＋游程 rank 回填（映射＝host sort/unique
            //    lower_bound 逐位一致：sorted-unique rank）——
            growOnly(d_seg_flags_, leftCount, CV_32SC1);
            growOnly(d_seg_runrank_, leftCount, CV_32SC1);
            growOnly(d_seg_map_, leftCount, CV_32SC1);
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelCopySeq<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_line_ids.ptr<int>(), leftCount,
                reinterpret_cast<int*>(d_left_keys_in_.ptr<int>()),
                d_left_idx_in_.ptr<int>());
            cub::DeviceRadixSort::SortPairs(
                d_cub_temp_, cub_temp_size_,
                reinterpret_cast<unsigned int*>(d_left_keys_in_.ptr<int>()),
                reinterpret_cast<unsigned int*>(d_left_keys_out_.ptr<int>()),
                d_left_idx_in_.ptr<int>(),
                d_left_sorted_idx_.ptr<int>(),
                leftCount, 0, sizeof(unsigned int) * 8, cuda_stream);
            kernelRunHeadsV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<const int*>(d_left_keys_out_.ptr<int>()),
                leftCount, d_seg_flags_.ptr<int>());
            size_t scanTemp = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scanTemp,
                (const int*)nullptr, (int*)nullptr, leftCount, cuda_stream);
            if (scan_temp_size_ < scanTemp || d_scan_temp_ == nullptr) {
                if (d_scan_temp_) cudaFree(d_scan_temp_);   // 独立缓冲，无在途排序使用
                cudaMalloc(&d_scan_temp_, scanTemp);
                scan_temp_size_ = scanTemp;
            }
            cub::DeviceScan::ExclusiveSum(d_scan_temp_, scan_temp_size_,
                d_seg_flags_.ptr<int>(), d_seg_runrank_.ptr<int>(),
                leftCount, cuda_stream);
            // ====【150FPS·Run D 扩图】段表回填→…→段偏移 13 连发图 ====
            // 先无条件执行容量预备（首帧分配/翻倍——捕获区外，防 cudaMalloc 破坏捕获）
            d_seg_votes_.create(1, kSegCap * V3_MAX_LIDS, CV_32SC1);
            d_seg_res_.create(1, kSegCap * V3_MAX_LIDS, CV_32SC1);
            d_seg_winner_.create(1, kSegCap, CV_32SC1);
            d_seg_drop_.create(1, kSegCap, CV_32SC1);
            d_seg_second_.create(1, kSegCap, CV_32SC1);
            d_excl_cnt_.create(1, kSegCap, CV_32SC1);
            growOnly(d_pt_row_, leftCount, CV_32SC1);
            d_seg_rows_.create(1, kSegCap * kRowWords * 8, CV_8UC1);
            d_seg_offsets_.create(1, kSegCap, CV_32SC1);
            d_viol_.create(1, 1, CV_32SC1);
            {
                auto& D = fgD_;
                const int gL = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                const int gCap = (kSegCap + BLOCK_SIZE - 1) / BLOCK_SIZE;
                // 参数刷新（图/普通共用）
                D.mb_runrank = d_seg_runrank_.ptr<int>();
                D.mb_flags = d_seg_flags_.ptr<int>();
                D.mb_sortedIdx = d_left_sorted_idx_.ptr<int>();
                D.mb_leftCount = leftCount;
                D.mb_map = d_seg_map_.ptr<int>();
                D.mb_counts2 = d_counts_.ptr<int>() + 2;
                D.sv_map = d_seg_map_.ptr<int>();
                D.sv_lid = d_card_lid_.ptr<int>();
                D.sv_res = d_card_res_.ptr<float>();
                D.sv_cnt = d_card_count_.ptr<int>();
                D.sv_leftCount = leftCount; D.sv_segCap = kSegCap;
                D.sv_votes = d_seg_votes_.ptr<int>();
                D.sv_sres = d_seg_res_.ptr<int>();
                D.ec_map = d_seg_map_.ptr<int>();
                D.ec_winner = d_seg_winner_.ptr<int>();
                D.ec_second = d_seg_second_.ptr<int>();
                D.ec_lid = d_card_lid_.ptr<int>();
                D.ec_cnt = d_card_count_.ptr<int>();
                D.ec_leftCount = leftCount; D.ec_segCap = kSegCap;
                D.ec_excl = d_excl_cnt_.ptr<int>();
                D.pr_pts = d_left_points.ptr<float2>();
                D.pr_map = d_seg_map_.ptr<int>();
                D.pr_leftCount = leftCount; D.pr_segCap = kSegCap;
                D.pr_rowStep = params_.epipolar_row_step;
                D.pr_ptRow = d_pt_row_.ptr<int>();
                D.pr_segRows = reinterpret_cast<unsigned long long*>(d_seg_rows_.ptr());
                D.pr_counts2 = d_counts_.ptr<int>() + 2;
                D.pr_viol = d_viol_.ptr<int>();
                D.so_flags = d_seg_flags_.ptr<int>();
                D.so_runrank = d_seg_runrank_.ptr<int>();
                D.so_leftCount = leftCount;
                D.so_offsets = d_seg_offsets_.ptr<int>();

                const bool dOn = getenv("JMW_V3GRAPH") != nullptr
                    && getenv("JMW_V3GRAPH")[0] == '1'
                    && !D.disabled && params_.conflict_reelect;
                bool usedD = false;
                if (dOn) {
                    if (!D.captured) {
                        if (++D.attempts <= 8) {
                            cudaStreamSynchronize(cuda_stream);   // WDDM：捕获前排空
                            if (cudaStreamBeginCapture(cuda_stream,
                                                       cudaStreamCaptureModeThreadLocal)
                                == cudaSuccess) {
                                kernelSegMapBackV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_seg_runrank_.ptr<int>(), d_seg_flags_.ptr<int>(),
                                    d_left_sorted_idx_.ptr<int>(), leftCount,
                                    d_seg_map_.ptr<int>(), d_counts_.ptr<int>() + 2);
                                cudaMemsetAsync(d_seg_votes_.ptr<int>(), 0,
                                                kSegCap * V3_MAX_LIDS * sizeof(int), cuda_stream);
                                cudaMemsetAsync(d_seg_res_.ptr<int>(), 0,
                                                kSegCap * V3_MAX_LIDS * sizeof(int), cuda_stream);
                                cudaMemsetAsync(d_seg_winner_.ptr<int>(), -1,
                                                kSegCap * sizeof(int), cuda_stream);
                                cudaMemsetAsync(d_seg_drop_.ptr<int>(), 0,
                                                kSegCap * sizeof(int), cuda_stream);
                                kernelSegVoteV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_seg_map_.ptr<int>(), d_card_lid_.ptr<int>(),
                                    d_card_res_.ptr<float>(), d_card_count_.ptr<int>(),
                                    leftCount, kSegCap,
                                    d_seg_votes_.ptr<int>(), d_seg_res_.ptr<int>());
                                kernelSegWinnerSecondV3<<<gCap, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_seg_votes_.ptr<int>(), kSegCap,
                                    d_seg_winner_.ptr<int>(), d_seg_second_.ptr<int>());
                                cudaMemsetAsync(d_excl_cnt_.ptr<int>(), 0,
                                                kSegCap * sizeof(int), cuda_stream);
                                kernelExclCountV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_seg_map_.ptr<int>(), d_seg_winner_.ptr<int>(),
                                    d_seg_second_.ptr<int>(),
                                    d_card_lid_.ptr<int>(), d_card_count_.ptr<int>(),
                                    leftCount, kSegCap, d_excl_cnt_.ptr<int>());
                                cudaMemsetAsync(d_seg_rows_.ptr(), 0,
                                                static_cast<size_t>(kSegCap) * kRowWords * 8,
                                                cuda_stream);
                                cudaMemsetAsync(d_viol_.ptr<int>(), 0, sizeof(int), cuda_stream);
                                kernelPtRowSegRowsV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_left_points.ptr<float2>(), d_seg_map_.ptr<int>(),
                                    leftCount, kSegCap, params_.epipolar_row_step,
                                    d_pt_row_.ptr<int>(),
                                    reinterpret_cast<unsigned long long*>(d_seg_rows_.ptr()),
                                    d_counts_.ptr<int>() + 2, d_viol_.ptr<int>());
                                kernelSegOffsetsV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_seg_flags_.ptr<int>(), d_seg_runrank_.ptr<int>(),
                                    leftCount, d_seg_offsets_.ptr<int>());
                                cudaGraph_t g = nullptr;
                                if (cudaStreamEndCapture(cuda_stream, &g) == cudaSuccess && g) {
                                    size_t nn = 0;
                                    cudaGraphGetNodes(g, nullptr, &nn);
                                    if (nn == 13
                                        && cudaGraphInstantiateWithFlags(&D.exec, g, 0)
                                               == cudaSuccess) {
                                        std::vector<cudaGraphNode_t> nd(nn);
                                        cudaGraphGetNodes(g, nd.data(), &nn);
                                        D.graph = g;
                                        D.nMapBack = nd[0]; D.nVote = nd[5];
                                        D.nExcl = nd[8]; D.nPtRow = nd[11];
                                        D.nSegOff = nd[12];
                                        // 参数描述（仅 5 变参节点需逐帧 SetParams）
                                        D.mbArgs[0]=&D.mb_runrank; D.mbArgs[1]=&D.mb_flags;
                                        D.mbArgs[2]=&D.mb_sortedIdx; D.mbArgs[3]=&D.mb_leftCount;
                                        D.mbArgs[4]=&D.mb_map; D.mbArgs[5]=&D.mb_counts2;
                                        D.mbP.func=(void*)kernelSegMapBackV3;
                                        D.mbP.gridDim=dim3(gL); D.mbP.blockDim=dim3(BLOCK_SIZE);
                                        D.mbP.sharedMemBytes=0; D.mbP.kernelParams=D.mbArgs;
                                        D.svArgs[0]=&D.sv_map; D.svArgs[1]=&D.sv_lid;
                                        D.svArgs[2]=&D.sv_res; D.svArgs[3]=&D.sv_cnt;
                                        D.svArgs[4]=&D.sv_leftCount; D.svArgs[5]=&D.sv_segCap;
                                        D.svArgs[6]=&D.sv_votes; D.svArgs[7]=&D.sv_sres;
                                        D.svP.func=(void*)kernelSegVoteV3;
                                        D.svP.gridDim=dim3(gL); D.svP.blockDim=dim3(BLOCK_SIZE);
                                        D.svP.sharedMemBytes=0; D.svP.kernelParams=D.svArgs;
                                        D.ecArgs[0]=&D.ec_map; D.ecArgs[1]=&D.ec_winner;
                                        D.ecArgs[2]=&D.ec_second; D.ecArgs[3]=&D.ec_lid;
                                        D.ecArgs[4]=&D.ec_cnt; D.ecArgs[5]=&D.ec_leftCount;
                                        D.ecArgs[6]=&D.ec_segCap; D.ecArgs[7]=&D.ec_excl;
                                        D.ecP.func=(void*)kernelExclCountV3;
                                        D.ecP.gridDim=dim3(gL); D.ecP.blockDim=dim3(BLOCK_SIZE);
                                        D.ecP.sharedMemBytes=0; D.ecP.kernelParams=D.ecArgs;
                                        D.prArgs[0]=&D.pr_pts; D.prArgs[1]=&D.pr_map;
                                        D.prArgs[2]=&D.pr_leftCount; D.prArgs[3]=&D.pr_segCap;
                                        D.prArgs[4]=&D.pr_rowStep; D.prArgs[5]=&D.pr_ptRow;
                                        D.prArgs[6]=&D.pr_segRows; D.prArgs[7]=&D.pr_counts2;
                                        D.prArgs[8]=&D.pr_viol;
                                        D.prP.func=(void*)kernelPtRowSegRowsV3;
                                        D.prP.gridDim=dim3(gL); D.prP.blockDim=dim3(BLOCK_SIZE);
                                        D.prP.sharedMemBytes=0; D.prP.kernelParams=D.prArgs;
                                        D.soArgs[0]=&D.so_flags; D.soArgs[1]=&D.so_runrank;
                                        D.soArgs[2]=&D.so_leftCount; D.soArgs[3]=&D.so_offsets;
                                        D.soP.func=(void*)kernelSegOffsetsV3;
                                        D.soP.gridDim=dim3(gL); D.soP.blockDim=dim3(BLOCK_SIZE);
                                        D.soP.sharedMemBytes=0; D.soP.kernelParams=D.soArgs;
                                        D.captured = true;
                                    }
                                }
                                // 捕获不执行——捕获成功则图发射，失败则下文普通发射重跑
                                if (D.captured) {
                                    cudaGraphLaunch(D.exec, cuda_stream);
                                    usedD = true;
                                } else if (g) {
                                    cudaGraphDestroy(g);
                                }
                            }
                        } else {
                            D.disabled = true;
                        }
                    } else {
                        D.mbP.gridDim = dim3(gL);
                        D.svP.gridDim = dim3(gL);
                        D.ecP.gridDim = dim3(gL);
                        D.prP.gridDim = dim3(gL);
                        D.soP.gridDim = dim3(gL);
                        const bool ok =
                            cudaGraphExecKernelNodeSetParams(D.exec, D.nMapBack, &D.mbP) == cudaSuccess
                            && cudaGraphExecKernelNodeSetParams(D.exec, D.nVote, &D.svP) == cudaSuccess
                            && cudaGraphExecKernelNodeSetParams(D.exec, D.nExcl, &D.ecP) == cudaSuccess
                            && cudaGraphExecKernelNodeSetParams(D.exec, D.nPtRow, &D.prP) == cudaSuccess
                            && cudaGraphExecKernelNodeSetParams(D.exec, D.nSegOff, &D.soP) == cudaSuccess;
                        if (ok) {
                            cudaGraphLaunch(D.exec, cuda_stream);
                            usedD = true;
                        } else {
                            D.disabled = true;
                            cudaGetLastError();
                        }
                    }
                }
                if (!usedD) {
                    // —— 普通发射兜底（图关/捕获失败帧；前 6 恒发，后 7 随 reelect）——
                    kernelSegMapBackV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_runrank_.ptr<int>(), d_seg_flags_.ptr<int>(),
                        d_left_sorted_idx_.ptr<int>(), leftCount,
                        d_seg_map_.ptr<int>(), d_counts_.ptr<int>() + 2);
                    cudaMemsetAsync(d_seg_votes_.ptr<int>(), 0,
                                    kSegCap * V3_MAX_LIDS * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_res_.ptr<int>(), 0,
                                    kSegCap * V3_MAX_LIDS * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_winner_.ptr<int>(), -1,
                                    kSegCap * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_drop_.ptr<int>(), 0,
                                    kSegCap * sizeof(int), cuda_stream);
                    kernelSegVoteV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_map_.ptr<int>(), d_card_lid_.ptr<int>(),
                        d_card_res_.ptr<float>(), d_card_count_.ptr<int>(),
                        leftCount, kSegCap,
                        d_seg_votes_.ptr<int>(), d_seg_res_.ptr<int>());
                    if (params_.conflict_reelect) {
                    kernelSegWinnerSecondV3<<<gCap, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_votes_.ptr<int>(), kSegCap,
                        d_seg_winner_.ptr<int>(), d_seg_second_.ptr<int>());
                    cudaMemsetAsync(d_excl_cnt_.ptr<int>(), 0,
                                    kSegCap * sizeof(int), cuda_stream);
                    kernelExclCountV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_map_.ptr<int>(), d_seg_winner_.ptr<int>(),
                        d_seg_second_.ptr<int>(),
                        d_card_lid_.ptr<int>(), d_card_count_.ptr<int>(),
                        leftCount, kSegCap, d_excl_cnt_.ptr<int>());
                    cudaMemsetAsync(d_seg_rows_.ptr(), 0,
                                    static_cast<size_t>(kSegCap) * kRowWords * 8, cuda_stream);
                    cudaMemsetAsync(d_viol_.ptr<int>(), 0, sizeof(int), cuda_stream);
                    kernelPtRowSegRowsV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_left_points.ptr<float2>(), d_seg_map_.ptr<int>(),
                        leftCount, kSegCap, params_.epipolar_row_step,
                        d_pt_row_.ptr<int>(),
                        reinterpret_cast<unsigned long long*>(d_seg_rows_.ptr()),
                        d_counts_.ptr<int>() + 2, d_viol_.ptr<int>());
                    kernelSegOffsetsV3<<<gL, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_flags_.ptr<int>(), d_seg_runrank_.ptr<int>(),
                        leftCount, d_seg_offsets_.ptr<int>());
                    }
                }
            }
        }   // （段表预备块闭合——copySeq/scan/SegMapBack..SegOffsets）

        if (params_.conflict_reelect) {
            // （winner/excl/P6prep/segoffsets 发射已并入上方 Run D 图化段——此处仅保留
            //  counts 批量回传＋sync#1＋分段计时）
            // —— 【瘦身·两阶段】先下 counts/diag(32B)→sync→知段数→按精确尺寸下其余 ——
            // （【级联·同步①②合并】评估结论 2026-10-19：小数组定容可行，但 segrows
            //  ＝段数×128 字×8B 按段上限 4096 定容＝4MB/帧（+0.6ms D2H）吃掉合并
            //  收益——前置条件＝GPU 侧位图压缩为 (段,行) 紧凑对再下载，另专项实施）
            h_cntI = hpin_counts_.ensure(4);
            h_diagI = hpin_diag_.ensure(4);
            g1SubMs_ = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - g1Sub0_).count();
            const auto g1Wait0_ = std::chrono::steady_clock::now();
            cudaMemcpyAsync(h_cntI, d_counts_.ptr<int>(), 4 * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaMemcpyAsync(h_diagI, d_diag_.ptr<int>(), 4 * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            g1WaitMs_ = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - g1Wait0_).count();
            // 【测速工具】g1＝原前半段（排序/分组/投票/选举＋counts 回传＋sync#1）
            if (!noBd_) tBreakdown_ += fmt::format(" g1={:.2f}(sub{:.2f}/wait{:.2f})", tSince_(),
                                        g1SubMs_, g1WaitMs_);
            static thread_local cudaEvent_t evTkA_ = nullptr, evTkB_ = nullptr,
                                           evPkA_ = nullptr, evPkB_ = nullptr;
            if (!evTkA_) {
                cudaEventCreate(&evTkA_); cudaEventCreate(&evTkB_);
                cudaEventCreate(&evPkA_); cudaEventCreate(&evPkB_);
            }
            segCount = h_cntI[2];
            if (segCount < 0) segCount = 0;
            if (segCount > kSegCap) segCount = kSegCap;
            int* h_segmapI = nullptr;   // 块外声明（值域防御路径复用）
            // 【速度优化·P6prep GPU 化】块外声明（预备段消费）；兜底重做帧弃用
            int* h_ptRowI = nullptr;
            int* h_sortedI = nullptr;
            int* h_segoffI = nullptr;
            unsigned long long* h_segrowsI = nullptr;
            bool prepFromGpu = false;
            // 【v3全GPU化·Step6 删回传】热路径跳过 sorted/segoff 下载（segPts 池仅
            // P6b 候选帧消费，按需补取）；segPtsDeferred＝本帧 segPts 池未建
            bool segPtsDeferred = false;
            // 【v3全GPU化·Step6 排障】三件套（left/segmap/ptRow）是否已下载——重做
            // 路径宿主按点重建与 P6b 分裂均依赖，未下载须现场补取（堵陈旧 pinned 读）
            bool trioFetched = false;
            // 【速度优化·刀② c7stamp GPU 化】分组 CSR 回传（cascadeSolve 消费）
            int* h_grpRowI = nullptr;
            double* h_grpSumI = nullptr;
            int* h_grpCntI = nullptr;
            int* h_grpOffI = nullptr;
            bool stampsFromGpu = false;
            // 【速度优化·刀③ c7merge GPU 化】配对结果回传
            double* h_pairSumI = nullptr;
            long long* h_pairNI = nullptr;
            bool pairFromGpu = false;
            // 【v3全GPU化·K1b】散布内核组已发射（pair 数据留设备）。stamps 下载
            // 默认保留（2026-10-20 数据裁定：直取 0.08ms ≪ 宿主回建 0.30ms，单步
            // 3.35 vs 3.85ms——方案 §10-5「跳过下载」假设被 ALL 399 实测推翻）；
            // JMW_K1B_SKIPSTAMPS=1 跳过下载走宿主回建（A/B 对照保留）
            bool k1bOn = false;
            static const bool k1bSkipStamps = (getenv("JMW_K1B_SKIPSTAMPS") != nullptr);
            {
                // —— 刀②③回退开关：JMW_V3GPU=1 才走 GPU 路径。内核级实测：两内核组
                //    对 GPU 不友好（配对仅 ~2k 线程延迟受限走链 2.42ms、行求和重复
                //    扫描 1.04ms）——比被替换的宿主版（1.22/0.44ms）更慢 → 默认宿主。
                //    基础设施保留（刀①/按需下载/预筛不受影响）
                static const bool v3GpuOn = (getenv("JMW_V3GPU") != nullptr);
                const bool okStamps =
                    (v3GpuOn || fullGpuOn_) && (h_cntI[3] == 0) && segCount > 0 && segCount <= kSegCap;
                if (okStamps) {
                    if (!noBd_) cudaEventRecord(evTkA_, cuda_stream);   // 【测速工具】stamps 内核组起
                    growOnly(d_grp_row_, leftCount, CV_32SC1);
                    growOnly(d_grp_sum_, leftCount, CV_64FC1);
                    growOnly(d_grp_cnt_, leftCount, CV_32SC1);
                    d_grp_off_.create(1, segCount + 1, CV_32SC1);
                    {
                        int grid = (segCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        kernelSegGroupCountsV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                            reinterpret_cast<const unsigned long long*>(d_seg_rows_.ptr()),
                            segCount, d_grp_cnt_.ptr<int>());
                    }
                    kernelSegGroupBaseV3<<<1, 1, 0, cuda_stream>>>(
                        d_grp_cnt_.ptr<int>(), segCount, d_grp_off_.ptr<int>());
                    if (fullGpuOn_ && !v3GpuOn) {
                        // 【v3全GPU化·K1a】盖章点并行内核（32 块轮扫段；块私有 scratch）
                        // 【stampsK 优化】scratch 恒零不变量（提取复位）→ 分配时清一次即可
                        if (!stampSum_) {
                            cudaMalloc(&stampSum_, 32ull * 8192 * sizeof(double));
                            cudaMalloc(&stampCnt_, 32ull * 8192 * sizeof(int));
                            cudaMemset(stampSum_, 0, 32ull * 8192 * sizeof(double));
                            cudaMemset(stampCnt_, 0, 32ull * 8192 * sizeof(int));
                        }
                        const int g = segCount < 32 ? segCount : 32;
                        kernelStampPointsV3<<<g, BLOCK_SIZE, 0, cuda_stream>>>(
                            reinterpret_cast<const unsigned long long*>(d_seg_rows_.ptr()),
                            d_seg_offsets_.ptr<int>(), d_grp_off_.ptr<int>(),
                            d_left_sorted_idx_.ptr<int>(), d_pt_row_.ptr<int>(),
                            d_left_points.ptr<float2>(),
                            stampSum_, stampCnt_,
                            d_grp_row_.ptr<int>(),
                            reinterpret_cast<double*>(d_grp_sum_.ptr()),
                            d_grp_cnt_.ptr<int>(), segCount, leftCount);
                    } else {
                        dim3 blk(BLOCK_SIZE);
                        dim3 grd((8192 + BLOCK_SIZE - 1) / BLOCK_SIZE, segCount);
                        kernelStampGroupsV3<<<grd, blk, 0, cuda_stream>>>(
                            reinterpret_cast<const unsigned long long*>(d_seg_rows_.ptr()),
                            d_seg_offsets_.ptr<int>(), d_grp_off_.ptr<int>(),
                            d_left_sorted_idx_.ptr<int>(), d_pt_row_.ptr<int>(),
                            d_left_points.ptr<float2>(), segCount, leftCount,
                            d_grp_row_.ptr<int>(),
                            reinterpret_cast<double*>(d_grp_sum_.ptr()),
                            d_grp_cnt_.ptr<int>());
                    }
                }
                stampsFromGpu = okStamps;
                // —— 刀③：有票标志＋配对（S≤512 走 GPU；>512 host 旧法）——
                // 【v3全GPU化·K1b】fullGpuMode_>=1 走行桶散布内核组（原子版：
                // pairN 精确/pairSum 末位漂移入 G0a 门）；旧 JMW_V3GPU 仍走每对
                // 归并内核（逐位一致版，兼容保留）。
                if (okStamps && segCount > 0 && segCount <= 512) {
                    if (!noBd_) {
                        cudaEventRecord(evTkB_, cuda_stream);   // stamps 组止／pairs 组起
                        cudaEventRecord(evPkA_, cuda_stream);
                    }
                    if (!d_seghas_)
                        cudaMalloc(&d_seghas_, kSegCap * sizeof(int));
                    if (!d_pair_sum_) {
                        cudaMalloc(&d_pair_sum_, 512ull * 512 * sizeof(double));
                        cudaMalloc(&d_pair_n_, 512ull * 512 * sizeof(long long));
                    }
                    {
                        int grid = (segCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        kernelSegHasVotesV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                            d_seg_votes_.ptr<int>(), segCount, d_seghas_);
                    }
                    if (v3GpuOn) {
                        const int total = segCount * segCount;
                        int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        kernelPairMergeV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                            d_grp_row_.ptr<int>(),
                            reinterpret_cast<const double*>(d_grp_sum_.ptr()),
                            d_grp_cnt_.ptr<int>(), d_grp_off_.ptr<int>(),
                            d_seghas_, segCount, d_pair_sum_, d_pair_n_);
                        pairFromGpu = true;
                    } else if (fullGpuOn_) {
                        // 【v3全GPU化·K1b】行桶散布 pair-CSR 三连＋均值稠密化。
                        // 失败（显存等）→ 静默落宿主 merge（行为＝宿主路径）。
                        bool k1bOk = true;
                        if (!k1bDense_)
                            k1bOk = (cudaMalloc(&k1bDense_, 512ull * 8192 * sizeof(double))
                                     == cudaSuccess);
                        if (k1bOk && !k1bRowOff_) {
                            k1bOk = (cudaMalloc(&k1bRowOff_, 8193 * sizeof(int))
                                     == cudaSuccess)
                                    && (cudaMalloc(&k1bRowFill_, 8192 * sizeof(int))
                                        == cudaSuccess);
                        }
                        if (k1bOk && k1bSegListCap_ < static_cast<size_t>(leftCount)) {
                            if (k1bSegList_) cudaFree(k1bSegList_);
                            const size_t cap = static_cast<size_t>(leftCount) * 2;   // grow-only 余量
                            k1bOk = (cudaMalloc(&k1bSegList_, cap * sizeof(int)) == cudaSuccess);
                            if (k1bOk) k1bSegListCap_ = cap;
                            else k1bSegListCap_ = 0;
                        }
                        if (k1bOk) {
                            const size_t sq = static_cast<size_t>(segCount) * segCount;
                            cudaMemsetAsync(d_pair_sum_, 0, sq * sizeof(double), cuda_stream);
                            cudaMemsetAsync(d_pair_n_, 0, sq * sizeof(long long), cuda_stream);
                            cudaMemsetAsync(k1bRowFill_, 0, 8192 * sizeof(int), cuda_stream);
                            {
                                const int g = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                                kernelSegMeanScatter<<<g, BLOCK_SIZE, 0, cuda_stream>>>(
                                    d_grp_off_.ptr<int>(), d_grp_row_.ptr<int>(),
                                    reinterpret_cast<const double*>(d_grp_sum_.ptr()),
                                    d_grp_cnt_.ptr<int>(), k1bDense_, segCount, leftCount);
                            }
                            kernelRowBucketCount<<<dim3(1, segCount), 128, 0, cuda_stream>>>(
                                reinterpret_cast<const unsigned long long*>(d_seg_rows_.ptr()),
                                d_seghas_, segCount, k1bRowFill_);
                            kernelRowBucketScan<<<1, 1024, 0, cuda_stream>>>(
                                k1bRowFill_, k1bRowOff_, k1bRowFill_);
                            kernelRowBucketScatter<<<dim3(1, segCount), 128, 0, cuda_stream>>>(
                                reinterpret_cast<const unsigned long long*>(d_seg_rows_.ptr()),
                                d_seghas_, segCount, k1bRowFill_, k1bSegList_);
                            kernelPairScatter<<<8192, BLOCK_SIZE, 0, cuda_stream>>>(
                                k1bRowOff_, k1bSegList_, k1bDense_,
                                d_pair_sum_, d_pair_n_, segCount);
                            pairFromGpu = true;
                            k1bOn = true;
                        } else {
                            fbCode = 4;   // 【回退记账】K1b 缓冲分配失败
                            CALIB_LOG_WARN("K1b 缓冲分配失败——友军合并落宿主路径");
                        }
                    }
                    if (!noBd_) cudaEventRecord(evPkB_, cuda_stream);   // 【测速工具】pairs 组止
                } else if (okStamps) {
                    if (!noBd_) cudaEventRecord(evTkB_, cuda_stream);   // stamps 组止（无 pairs）
                }
            }
            {
                const size_t sc = static_cast<size_t>(segCount > 0 ? segCount : 1);
                h_votesH = hpin_votes_.ensure(sc * V3_MAX_LIDS);
                h_leftH = hpin_left_.ensure(static_cast<size_t>(leftCount));
                h_winnerI = hpin_winner_.ensure(sc);
                h_exclI = hpin_excl_.ensure(sc);
                cudaMemcpyAsync(h_votesH, d_seg_votes_.ptr<int>(),
                                sc * V3_MAX_LIDS * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
            // 【按需下载】三件大传输（点坐标 262KB/行号 130KB/段表 130KB）仅在下述
            // 情形无条件下发：P5 老配置（冲突重选每帧要用）或兜底帧（host 旧法构建
            // 要用）或 K1b 跳过 stamps 下载帧（宿主回建行数据要用）。常态帧（级联开
            // ＋GPU 直取成功）跳过——分裂/诊断等罕见消费者按需再取（见 ensureTrioFetched）
            const bool k1bRebuild = k1bOn && k1bSkipStamps;   // stamps 不下载→宿主回建
            const bool trioNeededNow =
                !params_.cascade_elim || !stampsFromGpu || k1bRebuild;
            if (trioNeededNow) {
                h_leftH = hpin_left_.ensure(static_cast<size_t>(leftCount));
                cudaMemcpyAsync(h_leftH, d_left_points.ptr<float2>(),
                                static_cast<size_t>(leftCount) * sizeof(float2),
                                cudaMemcpyDeviceToHost, cuda_stream);
                h_segmapI = hpin_segmap_.ensure(static_cast<size_t>(leftCount));
                cudaMemcpyAsync(h_segmapI, d_seg_map_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                h_ptRowI = hpin_ptrow_.ensure(static_cast<size_t>(leftCount));
                cudaMemcpyAsync(h_ptRowI, d_pt_row_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                trioFetched = true;
            }
            cudaMemcpyAsync(h_winnerI, d_seg_winner_.ptr<int>(), sc * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            // h_secondI D2H 已删（死传输）
            cudaMemcpyAsync(h_exclI, d_excl_cnt_.ptr<int>(), sc * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            // 【速度优化·P6prep GPU 化】行位图恒回传（H6 单元组装宿主裁决消费）；
            // 【v3全GPU化·Step6 删回传】sorted/segoff 仅非热路径回传（segPts 池
            // 推迟到 P6b 候选帧按需补建，省 leftCount×8B/帧）
            const bool deferSegPts = params_.cascade_elim && stampsFromGpu && !k1bRebuild;
            h_segrowsI = hpin_segrows_.ensure(static_cast<size_t>(sc) * kRowWords);
            cudaMemcpyAsync(h_segrowsI, d_seg_rows_.ptr(),
                            static_cast<size_t>(sc) * kRowWords * 8,
                            cudaMemcpyDeviceToHost, cuda_stream);
            if (!deferSegPts) {
                h_sortedI = hpin_sorted_.ensure(static_cast<size_t>(leftCount));
                h_segoffI = hpin_segoff_.ensure(sc);
                cudaMemcpyAsync(h_sortedI, d_left_sorted_idx_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_segoffI, d_seg_offsets_.ptr<int>(), sc * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
            } else {
                segPtsDeferred = true;
            }
            // 【按需下载】值域违规计数（替 host min/max 扫描）
            int* h_violI = hpin_viol_.ensure(1);
            cudaMemcpyAsync(h_violI, d_viol_.ptr<int>(), sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
                // 【速度优化·刀②】分组 CSR 回传（同批同步；分组数≤leftCount 定容下载）
                // 【v3全GPU化·K1b】默认保留下载（cascadeSolve 直取 0.08ms）；实验开关
                // JMW_K1B_SKIPSTAMPS=1 跳过（宿主回建 0.30ms——实测更慢，留对照）
                if (stampsFromGpu && !k1bRebuild) {
                    h_grpRowI = hpin_grprow_.ensure(static_cast<size_t>(leftCount));
                    h_grpSumI = hpin_grpsum_.ensure(static_cast<size_t>(leftCount));
                    h_grpCntI = hpin_grpcnt_.ensure(static_cast<size_t>(leftCount));
                    h_grpOffI = hpin_grpoff_.ensure(static_cast<size_t>(segCount) + 1);
                    cudaMemcpyAsync(h_grpRowI, d_grp_row_.ptr<int>(),
                                    static_cast<size_t>(leftCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_grpSumI, d_grp_sum_.ptr(),
                                    static_cast<size_t>(leftCount) * sizeof(double),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_grpCntI, d_grp_cnt_.ptr<int>(),
                                    static_cast<size_t>(leftCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_grpOffI, d_grp_off_.ptr<int>(),
                                    (static_cast<size_t>(segCount) + 1) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                }
                // 【速度优化·刀③】配对结果回传（S²×16B ≤ 32KB）
                if (pairFromGpu) {
                    const size_t sq = static_cast<size_t>(segCount) * segCount;
                    h_pairSumI = hpin_pairsum_.ensure(sq);
                    h_pairNI = hpin_pairn_.ensure(sq);
                    cudaMemcpyAsync(h_pairSumI, d_pair_sum_, sq * sizeof(double),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_pairNI, d_pair_n_, sq * sizeof(long long),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                }
                cudaStreamSynchronize(cuda_stream);
                prepFromGpu = true;   // 本批下载有效（兜底重做会再置 false）
            }
            if (!noBd_) {
                tBreakdown_ += fmt::format(" g2={:.2f}", tSince_());   // stamps/pairs 发射＋批量回传＋sync#2
                if (stampsFromGpu) {   // 【测速工具】内核组真实 GPU 时长（未记录勿读）
                    float msT = 0.f, msP = 0.f;
                    cudaEventElapsedTime(&msT, evTkA_, evTkB_);
                    if (pairFromGpu) cudaEventElapsedTime(&msP, evPkA_, evPkB_);
                    tBreakdown_ += fmt::format(" stampsK={:.2f} pairsK={:.2f}", msT, msP);
                }
            }

            segCount = h_cntI[2];
            // 【v3全GPU化·G2 注入】JMW_V3_SEGCAP<N：人为制造「段数超容」（真实溢出
            // h_cntI[3]≠0 在产线数据从不触发）——走同一溢出重做路径验证等价性
            const bool g2Inject = (segCapEff_ < kSegCap) && (h_cntI[2] > segCapEff_);
            bool g2Overflow = g2Inject;
            if (h_cntI[3] != 0 && segCount > kSegCap) g2Overflow = true;   // 真实溢出同路
            if (g2Overflow) {
                fbCode = 1;   // 【回退记账】溢出重做帧
                // 溢出兜底（段数 > cap）：host 段表重做（原路径，逐位一致）
                prepFromGpu = false;   // 【速度优化】GPU 预备数据作废 → host 重建
                stampsFromGpu = false; // 【v3全GPU化】段表已重建 → GPU stamps 过时同废
                pairFromGpu = false;   // 【v3全GPU化】配对数据同废
                int* h_tracks = hpin_tracks_.ensure(static_cast<size_t>(leftCount));
                // 【Step6 排障】宿主重建按点循环需 left/ptRow（热路径三件套未下）——
                // 现场补取（搭本块同步顺风车；此前读陈旧 pinned ＝潜在错值）
                h_leftH = hpin_left_.ensure(static_cast<size_t>(leftCount));
                h_ptRowI = hpin_ptrow_.ensure(static_cast<size_t>(leftCount));
                cudaMemcpyAsync(h_tracks, d_left_line_ids.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_leftH, d_left_points.ptr<float2>(),
                                static_cast<size_t>(leftCount) * sizeof(float2),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_ptRowI, d_pt_row_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                trioFetched = true;
                cudaStreamSynchronize(cuda_stream);
                std::vector<int> uniqIds(h_tracks, h_tracks + leftCount);
                std::sort(uniqIds.begin(), uniqIds.end());
                uniqIds.erase(std::unique(uniqIds.begin(), uniqIds.end()), uniqIds.end());
                segCount = static_cast<int>(uniqIds.size());
                h_segmap.assign(static_cast<size_t>(leftCount), -1);
                for (int i = 0; i < leftCount; ++i)
                    h_segmap[static_cast<size_t>(i)] = static_cast<int>(
                        std::lower_bound(uniqIds.begin(), uniqIds.end(), h_tracks[i]) - uniqIds.begin());
                {
                    cv::Mat sm(1, leftCount, CV_32SC1, h_segmap.data());
                    d_seg_map_.upload(sm, stream);
                }
                d_seg_votes_.create(1, segCount * V3_MAX_LIDS, CV_32SC1);
                d_seg_res_.create(1, segCount * V3_MAX_LIDS, CV_32SC1);
                d_seg_winner_.create(1, segCount, CV_32SC1);
                d_seg_drop_.create(1, segCount, CV_32SC1);
                cudaMemsetAsync(d_seg_votes_.ptr<int>(), 0, segCount * V3_MAX_LIDS * sizeof(int), cuda_stream);
                cudaMemsetAsync(d_seg_res_.ptr<int>(), 0, segCount * V3_MAX_LIDS * sizeof(int), cuda_stream);
                cudaMemsetAsync(d_seg_winner_.ptr<int>(), -1, segCount * sizeof(int), cuda_stream);
                cudaMemsetAsync(d_seg_drop_.ptr<int>(), 0, segCount * sizeof(int), cuda_stream);
                int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernelSegVoteV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_seg_map_.ptr<int>(), d_card_lid_.ptr<int>(), d_card_res_.ptr<float>(),
                    d_card_count_.ptr<int>(), leftCount, segCount,
                    d_seg_votes_.ptr<int>(), d_seg_res_.ptr<int>());
                h_votesH = hpin_votes_.ensure(static_cast<size_t>(segCount) * V3_MAX_LIDS);
                h_winnerI = hpin_winner_.ensure(segCount);
                h_secondI = hpin_second_.ensure(segCount);
                h_exclI = nullptr;   // 溢出路径：excl 走旧扫描（卡数据先备好）
                cudaMemcpyAsync(h_votesH, d_seg_votes_.ptr<int>(),
                                static_cast<size_t>(segCount) * V3_MAX_LIDS * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_winnerI, d_seg_winner_.ptr<int>(), segCount * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaStreamSynchronize(cuda_stream);
            } else {
                // —— 值域防御（GPU 化）：kernelPtRowSegRowsV3 已对 seg∉[0,segCount)
                //    原子计数（≡host min/max 扫描判据）；异常 → host 段表重做。
                //    【按需下载】段表常态不回传——重做前现场取
                if (h_segmapI != nullptr) {
                    h_segmap.assign(h_segmapI, h_segmapI + leftCount);
                }
                const int violCnt = *hpin_viol_.ensure(1);
                if (violCnt > 0) {
                    fbCode = 2;   // 【回退记账】值域重做帧
                    CALIB_LOG_WARN("GPU segmap 值域异常（violation={} vs segCount={}）—— host 重做",
                                   violCnt, segCount);
                    prepFromGpu = false;   // 【速度优化】GPU 预备数据作废 → host 重建
                    stampsFromGpu = false; // 【刀②】分组数据同废（映射已重建）
                    pairFromGpu = false;   // 【刀③】配对数据同废
                    // 【Step6 排障】宿主重建需 left/ptRow——热路径未下则现场补取
                    h_leftH = hpin_left_.ensure(static_cast<size_t>(leftCount));
                    h_ptRowI = hpin_ptrow_.ensure(static_cast<size_t>(leftCount));
                    cudaMemcpyAsync(h_leftH, d_left_points.ptr<float2>(),
                                    static_cast<size_t>(leftCount) * sizeof(float2),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_ptRowI, d_pt_row_.ptr<int>(),
                                    static_cast<size_t>(leftCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    trioFetched = true;
                    if (h_segmapI == nullptr) {   // 【按需下载】现场取段表
                        h_segmapI = hpin_segmap_.ensure(static_cast<size_t>(leftCount));
                        cudaMemcpyAsync(h_segmapI, d_seg_map_.ptr<int>(),
                                        static_cast<size_t>(leftCount) * sizeof(int),
                                        cudaMemcpyDeviceToHost, cuda_stream);
                        cudaStreamSynchronize(cuda_stream);
                        h_segmap.assign(h_segmapI, h_segmapI + leftCount);
                    }
                    int* h_tracks2 = hpin_tracks_.ensure(static_cast<size_t>(leftCount));
                    cudaMemcpyAsync(h_tracks2, d_left_line_ids.ptr<int>(),
                                    static_cast<size_t>(leftCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaStreamSynchronize(cuda_stream);
                    std::vector<int> uniqIds2(h_tracks2, h_tracks2 + leftCount);
                    std::sort(uniqIds2.begin(), uniqIds2.end());
                    uniqIds2.erase(std::unique(uniqIds2.begin(), uniqIds2.end()),
                                   uniqIds2.end());
                    segCount = static_cast<int>(uniqIds2.size());
                    for (int i = 0; i < leftCount; ++i)
                        h_segmap[static_cast<size_t>(i)] = static_cast<int>(
                            std::lower_bound(uniqIds2.begin(), uniqIds2.end(),
                                             h_tracks2[i]) - uniqIds2.begin());
                    {
                        cv::Mat sm(1, leftCount, CV_32SC1, h_segmap.data());
                        d_seg_map_.upload(sm, stream);
                    }
                    // 重跑投票（原票算在异常段表上）——精确尺寸
                    d_seg_votes_.create(1, segCount * V3_MAX_LIDS, CV_32SC1);
                    d_seg_res_.create(1, segCount * V3_MAX_LIDS, CV_32SC1);
                    d_seg_winner_.create(1, segCount, CV_32SC1);
                    d_seg_drop_.create(1, segCount, CV_32SC1);
                    cudaMemsetAsync(d_seg_votes_.ptr<int>(), 0,
                                    segCount * V3_MAX_LIDS * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_res_.ptr<int>(), 0,
                                    segCount * V3_MAX_LIDS * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_winner_.ptr<int>(), -1,
                                    segCount * sizeof(int), cuda_stream);
                    cudaMemsetAsync(d_seg_drop_.ptr<int>(), 0,
                                    segCount * sizeof(int), cuda_stream);
                    int grid2 = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    kernelSegVoteV3<<<grid2, BLOCK_SIZE, 0, cuda_stream>>>(
                        d_seg_map_.ptr<int>(), d_card_lid_.ptr<int>(), d_card_res_.ptr<float>(),
                        d_card_count_.ptr<int>(), leftCount, segCount,
                        d_seg_votes_.ptr<int>(), d_seg_res_.ptr<int>());
                    // GPU excl 预备失效 → 旧扫描路径（host 选举＋卡现场取）
                    h_exclI = nullptr;
                    h_votesH = hpin_votes_.ensure(static_cast<size_t>(segCount) * V3_MAX_LIDS);
                    h_winnerI = hpin_winner_.ensure(segCount);
                    cudaMemcpyAsync(h_votesH, d_seg_votes_.ptr<int>(),
                                    static_cast<size_t>(segCount) * V3_MAX_LIDS * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaStreamSynchronize(cuda_stream);
                }
            }
            if (getenv("JMW_V3TIME") && h_segmap.size() >= static_cast<size_t>(leftCount)) {
                int mx = -1, mn = 9999;
                for (int i = 0; i < leftCount; ++i) {
                    if (h_segmap[i] > mx) mx = h_segmap[i];
                    if (h_segmap[i] < mn) mn = h_segmap[i];
                }
                std::vector<int> allF(static_cast<size_t>(leftCount));
                cudaMemcpy(allF.data(), d_seg_flags_.ptr<int>(),
                           static_cast<size_t>(leftCount) * 4, cudaMemcpyDeviceToHost);
                int headSum = 0;
                for (int i = 0; i < leftCount; ++i) headSum += allF[i];
                int lastR = 0, lastF = 0;
                cudaMemcpy(&lastR, d_seg_runrank_.ptr<int>() + leftCount - 1, 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastF, d_seg_flags_.ptr<int>() + leftCount - 1, 4, cudaMemcpyDeviceToHost);
                fprintf(stderr, "[sdbg] segCnt=%d mapRange=[%d,%d] headSum=%d lastR=%d lastF=%d\n",
                        segCount, mn, mx, headSum, lastR, lastF);
            }

            // ---- host 求解预备：CSR（ptRow/segPts/segRows）＋票直方图＋winner ----
            // 【速度优化】局部 vector<vector> 改 Impl 持久池：assign/clear 复用容量，
            // 运算序与原版逐位一致（每帧覆写，无跨帧残留读）
            const int* h_votes = h_votesH;
            float2* h_left = h_leftH;   // 【按需下载】可空；分裂候选帧现场补取后回填
            auto& segVotes = segVotesPool_;
            auto& segRows = segRowsPool_;
            auto& winner = winnerPool_;
            auto& dropped = droppedPool_;
            auto& ptRow = ptRowPool_;
            auto& segPts = segPtsPool_;
            segVotes.resize(static_cast<size_t>(segCount));
            for (int s = 0; s < segCount; ++s)
                segVotes[static_cast<size_t>(s)].assign(V3_MAX_LIDS, 0);
            winner.assign(static_cast<size_t>(segCount), -1);
            dropped.assign(static_cast<size_t>(segCount), 0);
            if (prepFromGpu && h_segrowsI) {
                // 【速度优化·P6prep GPU 直取】行号/行位图/段偏移/排序下标 GPU 已产：
                // 位图按段整拷（OR 序无关≡host 逐点 OR）、segPts 按段切片（排序布局
                // 段内升序 i ≡ host 过滤升序）——删 host 逐点循环。
                // 【按需下载】常态帧行号表不回传——池保上帧容量（分裂罕见路径按需取）
                // 【v3全GPU化·Step6】segPts 池热路径延迟（sorted/segoff 未下载时留陈
                // 旧——级联直取不读；P6b 候选帧按需补建）
                segRows.resize(static_cast<size_t>(segCount));
                for (int s = 0; s < segCount; ++s)
                    segRows[static_cast<size_t>(s)].assign(
                        h_segrowsI + static_cast<size_t>(s) * kRowWords,
                        h_segrowsI + (static_cast<size_t>(s) + 1) * kRowWords);
                if (h_ptRowI != nullptr)
                    ptRow.assign(h_ptRowI, h_ptRowI + leftCount);
                else
                    ptRow.resize(static_cast<size_t>(leftCount));
                segPts.resize(static_cast<size_t>(segCount));
                if (h_sortedI && h_segoffI) {
                    for (int s = 0; s < segCount; ++s) {
                        const int b = h_segoffI[s];
                        const int e = (s + 1 < segCount) ? h_segoffI[s + 1] : leftCount;
                        segPts[static_cast<size_t>(s)].assign(h_sortedI + b, h_sortedI + e);
                    }
                    segPtsDeferred = false;
                }   // else：延迟态维持 segPtsDeferred（P6b 候选帧补建）
            } else {
                segRows.resize(static_cast<size_t>(segCount));
                for (int s = 0; s < segCount; ++s)
                    segRows[static_cast<size_t>(s)].assign(kRowWords, 0ULL);
                ptRow.resize(static_cast<size_t>(leftCount));
                segPts.resize(static_cast<size_t>(segCount));
                for (int s = 0; s < segCount; ++s)
                    segPts[static_cast<size_t>(s)].clear();
                for (int i = 0; i < leftCount; ++i) {
                    const int row = static_cast<int>(roundf(h_left[static_cast<size_t>(i)].y
                                                            / params_.epipolar_row_step));
                    ptRow[static_cast<size_t>(i)] = row;
                    const int seg = h_segmap[static_cast<size_t>(i)];
                    if (seg >= 0 && seg < segCount)
                        segPts[static_cast<size_t>(seg)].push_back(i);
                    if (seg >= 0 && seg < segCount && row >= 0 && row < kMaxRows)
                        segRows[static_cast<size_t>(seg)][static_cast<size_t>(row) / 64] |=
                            1ULL << (static_cast<size_t>(row) % 64);
                }
                segPtsDeferred = false;   // 宿主重建已产 segPts
            }
            for (int s = 0; s < segCount; ++s)
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    segVotes[static_cast<size_t>(s)][static_cast<size_t>(l)] =
                        h_votes[static_cast<size_t>(s) * V3_MAX_LIDS + l];
            if (h_winnerI != nullptr && h_exclI != nullptr) {
                // GPU 选举结果直取（与 electWinnerArgmax 同判定序，逐位一致）
                for (int s = 0; s < segCount; ++s)
                    winner[static_cast<size_t>(s)] = h_winnerI[s];
            } else {
                for (int s = 0; s < segCount; ++s)
                    winner[static_cast<size_t>(s)] =
                        electWinnerArgmax(segVotes[static_cast<size_t>(s)].data());
            }

            // ---- P6b：段分裂（excl 数 GPU 预备；罕见路径 ensureCards_ 现场取卡）----
            const double tP6prep_ = tSince_();   // 【测速工具】CSR 预备段
            // 【按需下载＋预筛】双峰门槛①与 splitBimodalSegments 同判据——常态帧
            // 全段被拒（cand=0）→ 整个 P6b 跳过（行为等价：全拒时函数零副作用）；
            // 有候选帧才现场取三件套（点坐标/行号/段表，罕见路径付一次小同步）
            int p6bCandidates = 0;
            for (int s = 0; s < segCount; ++s) {
                if (dropped[static_cast<size_t>(s)] || winner[static_cast<size_t>(s)] < 1) continue;
                const int W = winner[static_cast<size_t>(s)];
                const int* v = segVotes[static_cast<size_t>(s)].data();
                int sv = 0;
                for (int l = 0; l < V3_MAX_LIDS; ++l)
                    if (l + 1 != W && v[l] > sv) sv = v[l];
                if (sv >= std::max(30, (v[W - 1] * 3) / 10)) { ++p6bCandidates; break; }
            }
            if (params_.segment_split && p6bCandidates > 0 && (!trioFetched || segPtsDeferred)) {
                // 【按需下载】现场取三件套（罕见帧：+一次小同步）。
                // 【v3全GPU化·Step6】三件套未下（trioFetched=false）或 segPts 池
                // 延迟（sorted/segoff 热路径未回传）——分裂前补齐（此前 h_left
                // 恒非空致本分支死代码、分裂帧读陈旧数据＝潜在错值，一并修复）
                h_leftH = hpin_left_.ensure(static_cast<size_t>(leftCount));
                h_segmapI = hpin_segmap_.ensure(static_cast<size_t>(leftCount));
                h_ptRowI = hpin_ptrow_.ensure(static_cast<size_t>(leftCount));
                if (segPtsDeferred) {
                    h_sortedI = hpin_sorted_.ensure(static_cast<size_t>(leftCount));
                    h_segoffI = hpin_segoff_.ensure(static_cast<size_t>(segCount));
                }
                cudaMemcpyAsync(h_leftH, d_left_points.ptr<float2>(),
                                static_cast<size_t>(leftCount) * sizeof(float2),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_segmapI, d_seg_map_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                cudaMemcpyAsync(h_ptRowI, d_pt_row_.ptr<int>(),
                                static_cast<size_t>(leftCount) * sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream);
                if (segPtsDeferred) {
                    cudaMemcpyAsync(h_sortedI, d_left_sorted_idx_.ptr<int>(),
                                    static_cast<size_t>(leftCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                    cudaMemcpyAsync(h_segoffI, d_seg_offsets_.ptr<int>(),
                                    static_cast<size_t>(segCount) * sizeof(int),
                                    cudaMemcpyDeviceToHost, cuda_stream);
                }
                cudaStreamSynchronize(cuda_stream);
                h_segmap.assign(h_segmapI, h_segmapI + leftCount);
                ptRow.assign(h_ptRowI, h_ptRowI + leftCount);
                h_left = h_leftH;
                if (segPtsDeferred) {
                    segPts.resize(static_cast<size_t>(segCount));
                    for (int s = 0; s < segCount; ++s) {
                        const int b = h_segoffI[s];
                        const int e = (s + 1 < segCount) ? h_segoffI[s + 1] : leftCount;
                        segPts[static_cast<size_t>(s)].assign(h_sortedI + b, h_sortedI + e);
                    }
                    segPtsDeferred = false;
                }
            }
            if (params_.segment_split && p6bCandidates > 0) {
                int splitRounds = 0;
                const int* h_exclForP6b_ = h_exclI;
                for (int r = 0; r < 3; ++r) {
                    const int n = splitBimodalSegments(
                        h_segmap, h_left, leftCount, params_.epipolar_row_step,
                        h_cards, h_cardcnt, cardStride,
                        h_exclForP6b_, ensureCards_,
                        segCount, segVotes, segRows, winner, dropped, ptRow, segPts);
                    if (n == 0) break;
                    ++splitRounds;
                    h_exclForP6b_ = nullptr;   // 旧扫描路径（卡已由分裂执行 ensure）
                }
                if (splitRounds > 0) {   // segmap 变了 → 回传
                    fbCode = 3;   // 【回退记账】分裂作废帧
                    cv::Mat sm(1, leftCount, CV_32SC1, h_segmap.data());
                    d_seg_map_.upload(sm, stream);
                    // 【v3全GPU化·K1b 排障】段表已分裂（段数/票/位图已变）→ GPU
                    // stamps/pair 数据全部过时，作废落宿主回建＋宿主 merge（罕见
                    // 帧，原路径语义）。此前缺此守卫＝分裂帧用过时 CSR 直取（潜在
                    // 错数据，刀②③/K1a/K1b 同病）
                    stampsFromGpu = false;
                    pairFromGpu = false;
                }
            }
            if (!noBd_) tBreakdown_ += fmt::format(" P6prep={:.2f} P6b={:.2f}", tP6prep_, tSince_());

            if (params_.cascade_elim) {
                // ---- P7：级联判定（替代 P5 冲突循环；P6b 分裂已在上方完成）----
                    CascadeCfg cp;
                cp.gateH = params_.cascade_gate_h;
                cp.gateShare = params_.cascade_gate_share;
                cp.gateMargin = params_.cascade_gate_margin;
                cp.friendXpx = params_.cascade_friend_xpx;
                cp.friendMinRows = params_.cascade_friend_minrows;
                cp.spacingPx = params_.cascade_spacing_px;
                cp.spacingTol = params_.cascade_spacing_tol;
                int banned = 0, merged = 0;
                const int determined = cascadeSolve(segCount, h_segmap, h_left, leftCount,
                                                    params_.epipolar_row_step, ptRow, segPts,
                                                    segVotes, segRows,
                                                    winner, dropped, cp, &banned, &merged,
                                                    noBd_ ? nullptr : &tBreakdown_,
                                                    stampsFromGpu ? h_grpRowI : nullptr,
                                                    stampsFromGpu ? h_grpSumI : nullptr,
                                                    stampsFromGpu ? h_grpCntI : nullptr,
                                                    stampsFromGpu ? h_grpOffI : nullptr,
                                                    pairFromGpu ? h_pairSumI : nullptr,
                                                    pairFromGpu ? h_pairNI : nullptr,
                                                    (fullGpuMode_ >= 2) ? k2Ctx_ : nullptr,
                                                    (fullGpuMode_ >= 2)
                                                        ? cuda_stream
                                                        : static_cast<cudaStream_t>(nullptr));
                CALIB_LOG_INFO("P7 cascade: segs={} determined={} merged={} banned={}",
                               segCount, determined, merged, banned);
            } else {
                int conflictsResolved = 0, reelected = 0, forcedDrop = 0;
                for (int iter = 0; iter < 10; ++iter)
                    if (!resolveConflictsOnce(segCount, h_segmap, h_left, leftCount,
                                              params_.epipolar_row_step, segVotes, segRows,
                                              winner, dropped, &reelected, &forcedDrop))
                        break;
                    else ++conflictsResolved;
                CALIB_LOG_INFO("P5/P6b: segs={} conflictsResolved={} reelected={} forcedDrop={}",
                               segCount, conflictsResolved, reelected, forcedDrop);
            }
            if (!noBd_) tBreakdown_ += fmt::format(" P7orP5={:.2f}", tSince_());
            {
                if (d_seg_winner_.cols < segCount)
                    d_seg_winner_.create(1, segCount, CV_32SC1);
                if (d_seg_drop_.cols < segCount)
                    d_seg_drop_.create(1, segCount, CV_32SC1);
                // 【瘦身】pageable→pinned（cv::Mat upload 有驱动 staging＋隐式同步）
                int* wPin = hpin_winner_.ensure(segCount);
                for (int s = 0; s < segCount; ++s)
                    wPin[s] = winner[static_cast<size_t>(s)];
                const cv::Mat w(1, segCount, CV_32SC1, wPin);
                d_seg_winner_.upload(w, stream);
                int* dPin = hpin_drop_.ensure(segCount);
                for (int s = 0; s < segCount; ++s)
                    dPin[s] = dropped[static_cast<size_t>(s)] ? 1 : 0;
                const cv::Mat d(1, segCount, CV_32SC1, dPin);
                d_seg_drop_.upload(d, stream);
            }
        } else {
            // 【P0-1 修复】非 relect 路径：segCount 从 GPU 回读（此前恒 0 → grid=0 崩）
            h_cntI = hpin_counts_.ensure(4);
            cudaMemcpyAsync(h_cntI, d_counts_.ptr<int>(), 4 * sizeof(int),
                            cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            if (segCount == 0) segCount = h_cntI[2];
            if (segCount <= 0) segCount = 1;   // 防御
            // 扩容 winner/drop（GPU segmap 可能产出多于 kSegCap 的段）
            if (d_seg_winner_.cols < segCount)
                d_seg_winner_.create(1, segCount, CV_32SC1);
            if (d_seg_drop_.cols < segCount)
                d_seg_drop_.create(1, segCount, CV_32SC1);
            cudaMemsetAsync(d_cell_seg_.ptr<int>(), -1,
                            MAX_EPIPOLAR_ROWS * V3_MAX_LIDS * sizeof(int), cuda_stream);
            {
                int grid = (segCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernelSegWinnerV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_seg_votes_.ptr<int>(),
                    d_seg_res_.ptr<int>(),
                    segCount,
                    d_seg_winner_.ptr<int>());
            }
            {
                int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernelConflictMarkV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                    d_left_points.ptr<float2>(),
                    d_seg_map_.ptr<int>(),
                    leftCount,
                    d_seg_winner_.ptr<int>(),
                    params_.epipolar_row_step,
                    d_cell_seg_.ptr<int>(),
                    d_seg_drop_.ptr<int>());
            }
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("segment stage failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            // 【stampsK 优化】scratch 恒零不变量兜底：失败帧可能残留累加值——复位
            if (stampSum_) {
                cudaMemsetAsync(stampSum_, 0, 32ull * 8192 * sizeof(double), cuda_stream);
                cudaMemsetAsync(stampCnt_, 0, 32ull * 8192 * sizeof(int), cuda_stream);
            }
            return result;
        }

        // ---- V3 Step 8: gather winner cards of kept segments ----
        if (getenv("JMW_V3TIME") && !h_segmap.empty()) {
            int mx = -1;
            for (int q = 0; q < leftCount; ++q)
                if (h_segmap[static_cast<size_t>(q)] > mx) mx = h_segmap[static_cast<size_t>(q)];
            fprintf(stderr, "[gdbg] N=%d segCnt=%d maxMap=%d win.cols=%d drop.cols=%d\n",
                    leftCount, segCount, mx, d_seg_winner_.cols, d_seg_drop_.cols);
        }
        {
            int grid = (leftCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernelGatherV3<<<grid, BLOCK_SIZE, 0, cuda_stream>>>(
                d_left_points.ptr<float2>(),
                d_left_line_ids.ptr<int>(),
                d_seg_map_.ptr<int>(),
                d_seg_winner_.ptr<int>(),
                d_seg_drop_.ptr<int>(),
                d_card_lid_.ptr<int>(),
                d_card_hitRx_.ptr<float>(),
                d_card_hitRidx_.ptr<int>(),
                d_card_count_.ptr<int>(),
                leftCount,
                d_out_flags_.ptr<int>(),
                d_out_left_.ptr<float2>(),
                d_out_right_.ptr<float2>(),
                d_out_lid_.ptr<int>(),
                d_out_track_.ptr<int>(),
                d_right_status_[gen].ptr<int>(),
                d_diag_.ptr<int>());
        }

        if ((kernel_err = cudaGetLastError()) != cudaSuccess) {
            result.success = false;
            result.message = std::string("kernelGatherV3 failed: ") + cudaGetErrorString(kernel_err);
            CALIB_LOG_ERROR("process(): {}", result.message);
            if (stampSum_) {   // 【stampsK 优化】恒零不变量兜底（同上）
                cudaMemsetAsync(stampSum_, 0, 32ull * 8192 * sizeof(double), cuda_stream);
                cudaMemsetAsync(stampCnt_, 0, 32ull * 8192 * sizeof(int), cuda_stream);
            }
            return result;
        }

        // ---- V3 Step 9: compact flagged outputs (CUB DeviceSelect x4) ----
        {
            size_t t1 = 0;
            cub::DeviceSelect::Flagged(nullptr, t1,
                (const float2*)nullptr, (const int*)nullptr,
                (float2*)nullptr, (int*)nullptr, leftCount, cuda_stream);
            if (select_temp_cap_ < t1 || d_select_temp_ == nullptr) {
                safeCudaFree(d_select_temp_);
                cudaError_t err = cudaMalloc(&d_select_temp_, t1);
                if (err != cudaSuccess) {
                    result.success = false;
                    result.message = "select temp alloc failed";
                    CALIB_LOG_ERROR("process(): {}", result.message);
                    return result;
                }
                select_temp_size_ = t1;
                select_temp_cap_ = t1;
            }
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const float2*)d_out_left_.ptr<float2>(), d_out_flags_.ptr<int>(),
                d_matched_left_cap_.ptr<float2>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const float2*)d_out_right_.ptr<float2>(), d_out_flags_.ptr<int>(),
                d_matched_right_cap_.ptr<float2>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const int*)d_out_lid_.ptr<int>(), d_out_flags_.ptr<int>(),
                d_matched_lids_cap_.ptr<int>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
            cub::DeviceSelect::Flagged(d_select_temp_, select_temp_size_,
                (const int*)d_out_track_.ptr<int>(), d_out_flags_.ptr<int>(),
                d_matched_tracks_cap_.ptr<int>(), d_counts_.ptr<int>(), leftCount, cuda_stream);
        }

        // ---- V3 Step 10: single host sync (count + diagnostics) ----
        // 【v3全GPU化·Step10 设备化】统计归约上 GPU（kernelStep10ReduceV3）——
        // 出口纪律不变：返回前一次下载＋同步（matchedCount 供下游定宽）。
        // 删四路 D2H（leftstat/rightstat ~240KB＋drop/winner）＋三段宿主扫描。
        if (d_stats_.empty())
            d_stats_.create(1, 8, CV_32SC1);
        kernelStep10ReduceV3<<<1, 1024, 0, cuda_stream>>>(
            d_seg_drop_.ptr<int>(), d_seg_winner_.ptr<int>(),
            d_left_status_[gen].ptr<int>(), d_right_status_[gen].ptr<int>(),
            d_diag_.ptr<int>(), d_counts_.ptr<int>(),
            segCount, leftCount, rightCount, d_stats_.ptr<int>());
        int* h_stats = hpin_stats_.ensure(8);
        cudaMemcpyAsync(h_stats, d_stats_.ptr<int>(), 7 * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        if (!noBd_)
            tBreakdown_ += fmt::format(" step8-10={:.2f} total={:.2f} fb={}",
                                        tSince_(),
                                        std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - tT0_).count(),
                                        fbCode);
        last_breakdown_ = tBreakdown_;   // 【测速工具】暴露给 gpu_stress 聚合
        if (getenv("JMW_V3TIME"))   // 速度沙箱诊断（--quiet 下仍可用）
            fprintf(stderr, "[v3-time]%s\n", tBreakdown_.c_str());

        const int droppedSegs = h_stats[0];
        const int noCard = h_stats[1];
        const int noMap = h_stats[2];
        const int rightUsed = h_stats[3];
        const int emptySegs = h_stats[4];
        const int h_final = h_stats[6];

        result.matchedCount = h_final;
        result.excludedLeftCount = noCard;
        result.excludedRightCount = rightCount - rightUsed;   // P4: 右点未消费数（原恒 0）
        result.noMapLeftCount = noMap;
        result.noWinnerCardCount = h_stats[5];
        result.emptyWinnerSegCount = emptySegs;
        result.trackTotal = segCount;
        result.trackDropped = droppedSegs;

        if (h_final > 0) {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>(
                d_matched_left_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>(
                d_matched_right_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>(
                d_matched_lids_cap_, cv::Rect(0, 0, h_final, 1));
            result.d_matched_track_ids = std::make_shared<cv::cuda::GpuMat>(
                d_matched_tracks_cap_, cv::Rect(0, 0, h_final, 1));
        } else {
            result.d_matched_left = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_right = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_line_ids = std::make_shared<cv::cuda::GpuMat>();
            result.d_matched_track_ids = std::make_shared<cv::cuda::GpuMat>();
        }
        result.d_left_status = std::make_shared<cv::cuda::GpuMat>(d_left_status_[gen]);
        result.d_right_status = std::make_shared<cv::cuda::GpuMat>(d_right_status_[gen]);

        status_gen_++;

        result.success = true;
        result.message = fmt::format(
            "OK: {} matched, {}/{} tracks kept, no-card={} no-map={} exclR={} "
            "noWinnerCard={} emptySeg={}",
            h_final, segCount - droppedSegs, segCount, noCard, noMap,
            rightCount - rightUsed, h_stats[5], emptySegs);

        CALIB_LOG_DEBUG("process(): {}", result.message);
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Exception: ") + e.what();
        CALIB_LOG_ERROR("process(): {}", result.message);
    } catch (...) {
        result.success = false;
        result.message = "Unknown exception";
        CALIB_LOG_ERROR("process(): {}", result.message);
    }

    return result;
}



