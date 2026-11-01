/**
 * @file laser_match_scan_v3_split.h
 * @brief 激光线匹配扫描 V3 流水线拆分算子（GPU 前段＋CPU 后段）— 公开接口
 *
 * 2026-10-21 将 LaserMatchScanCudaV3 的 10 步流水线拆为两个可独立调用的算子：
 *   - LaserMatchScanSplitGpuV3（GPU 前段）：Step0-6（右排序/CSR＋发卡＋GPU 段表）
 *     ＋卡表紧缩＋中间产物批量 D2H＋同步（Step 6.5），产出 LaserMatchSplitFrontV3
 *   - LaserMatchScanSplitCpuV3（CPU 后段）：Step7-10（段内投票/winner 选举/冲突处理
 *     P5/P6b/P7 全路径/收集/统计），纯 CPU 无 CUDA 调用，消费 FrontV3
 *
 * 拆分动机：GPU 算 N+1 帧前段时 CPU 收 N 帧后段（错帧并行），消除尾部串行同步。
 * 一致性：[前段＋后段] 与原 GPU 算子同输入同参数逐点一致（A/B 回放验证）。
 * 设计/评审/实施记录见 TEST/speed_optimize/laser_match_v3_split_设计方案.md。
 *
 * ⚠ 沙箱定位：不替换产线 GPU 算子，不入 mod_operatorlib。
 *
 * 使用契约：
 *   - 前后段必须持有同一份 LaserMatchScanParamsV3（rowStep＋conflict 三开关＋
 *     cascade 七参；分裂/重选判定发生在后段）
 *   - FrontV3 为值语义（内部 vector 拷贝），可跨帧持有、与下一次 ExecuteFront 并行
 *   - 后段 Execute 入口检查 front.success，失败短路返回
 *   - <2ms 硬指标口径＝后段 Execute（SoA→AoS 组装＋计算＋输出），D2H 计入前段
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "laser_match_scan_v3_cuda.h"   // LaserMatchScanParamsV3（单一事实源，CUDA-free）
#include "laser_match_scan_v3_cpu.h"    // LaserMatchScanCpuResultV3（后段结果复用）

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

// ============================================================================
// LaserMatchSplitFrontV3 — 前段产物（host 值语义）
// ============================================================================

struct LaserMatchSplitFrontV3 {
    bool success = false;
    std::string message;

    int leftCount = 0;
    int rightCount = 0;
    int segCount = 0;        // GPU 段表段数（counts[2] 原值，未钳位）
    int segOverflow = 0;     // counts[3]：segCount>4096 标志（透传；后段无段上限）
    int totalCards = 0;      // 全帧有效卡总数（＝cardPtr[N]）

    std::vector<cv::Point2f> leftPts;    // N 左点原序（后段消费序）
    std::vector<int> leftTracks;         // N 左点 track id（原序）
    std::vector<int> segmap;             // N orig→segIdx（GPU 游程 rank）
    std::vector<int> leftStatus;         // N：-1 noMap / 0 noCard / 1 有卡（GPU Step5 产物）
    std::vector<int> cardPtr;            // N+1 每点卡起始偏移（末位＝totalCards）
    std::vector<int> cardLid;            // totalCards 紧缩卡表 SoA：lid
    std::vector<float> cardRx;           // totalCards：hitRx（命中右点 x）
    std::vector<int> cardRidx;           // totalCards：右排序系索引

    std::string timing;      // 前段阶段耗时串（观测用）
};

// ============================================================================
// LaserMatchScanSplitGpuV3 — GPU 前段算子（组合持有 LaserMatchScanCudaV3）
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有（温表/温度经注入；内部组合原算子实例）
// 并发策略: 每实例非线程安全（同原算子契约）
// ==============================

class LaserMatchScanSplitGpuV3 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchSplitGpuV3";

    explicit LaserMatchScanSplitGpuV3(const LaserMatchScanParamsV3& params = {});
    ~LaserMatchScanSplitGpuV3();

    LaserMatchScanSplitGpuV3(const LaserMatchScanSplitGpuV3&) = delete;
    LaserMatchScanSplitGpuV3& operator=(const LaserMatchScanSplitGpuV3&) = delete;

    bool LoadTempTable(const std::string& jsonPath);
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);
    void SetCurrentTemperature(double temperature);

    /// 前段执行：Step0-6＋紧缩＋D2H＋同步 → front（值语义，可跨帧持有）
    void Execute(const cv::cuda::GpuMat& d_left_points,
                 const cv::cuda::GpuMat& d_left_line_ids,
                 const cv::cuda::GpuMat& d_right_points,
                 const cv::cuda::GpuMat& d_right_line_ids,
                 cv::cuda::Stream& stream,
                 LaserMatchSplitFrontV3& front);
    void Execute(const cv::cuda::GpuMat& d_left_points,
                 const cv::cuda::GpuMat& d_left_line_ids,
                 const cv::cuda::GpuMat& d_right_points,
                 const cv::cuda::GpuMat& d_right_line_ids,
                 LaserMatchSplitFrontV3& front);

    ///【前段提前提交】延迟对（转发 LaserMatchScanCudaV3；语义见彼处注释）
    void ExecuteDeferred(const cv::cuda::GpuMat& d_left_points_cap,
                         const cv::cuda::GpuMat& d_left_line_ids_cap,
                         const int* d_leftCount,
                         const cv::cuda::GpuMat& d_right_points_cap,
                         const cv::cuda::GpuMat& d_right_line_ids_cap,
                         const int* d_rightCount,
                         cv::cuda::Stream& stream,
                         LaserMatchSplitFrontV3& front);
    void FinishFront(int leftCount, int rightCount,
                     cv::cuda::Stream& stream,
                     LaserMatchSplitFrontV3& front);

    ///【B4·前段提交图化】JMW_V3FRONTGRAPH=1（默认关）——转发 ExecuteFrontGraph
    void ExecuteFrontGraph(const cv::cuda::GpuMat& d_left_points_cap,
                           const cv::cuda::GpuMat& d_left_line_ids_cap,
                           const int* d_leftCount,
                           const cv::cuda::GpuMat& d_right_points_cap,
                           const cv::cuda::GpuMat& d_right_line_ids_cap,
                           const int* d_rightCount,
                           cv::cuda::Stream& stream,
                           LaserMatchSplitFrontV3& front);

    void Destroy();
    void Warmup(int maxLeftPoints, int maxRightPoints);
    void SetParams(const LaserMatchScanParamsV3& params);
    const LaserMatchScanParamsV3& GetParams() const;
    void SetuRBias(const std::vector<float>& biasPerLid);
    void SetuRBiasModel(float wx, float wy, float wz, float dtx);

    /// 前段最近一次阶段耗时串（同 front.timing）
    std::string LastTimingBreakdown() const;

private:
    std::unique_ptr<LaserMatchScanCudaV3> op_;
    std::string lastBd_;
};

// ============================================================================
// LaserMatchScanSplitCpuV3 — CPU 后段算子（纯 CPU，无 CUDA 依赖）
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有（参数注入；结果缓冲容量式复用）
// 并发策略: 每实例非线程安全，多实例并行各自独占（单线程执行）
// ==============================

class LaserMatchScanSplitCpuV3 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchSplitCpuV3";

    explicit LaserMatchScanSplitCpuV3(const LaserMatchScanParamsV3& params = {});
    ~LaserMatchScanSplitCpuV3();

    LaserMatchScanSplitCpuV3(const LaserMatchScanSplitCpuV3&) = delete;
    LaserMatchScanSplitCpuV3& operator=(const LaserMatchScanSplitCpuV3&) = delete;

    /// 后段执行（Step7-10）：消费前段产物 → 与原算子同构的结果（值语义）。
    /// 入口检查 front.success，失败原样透传失败结果。
    LaserMatchScanCpuResultV3 Execute(const LaserMatchSplitFrontV3& front);

    void Destroy();
    void SetParams(const LaserMatchScanParamsV3& params);
    const LaserMatchScanParamsV3& GetParams() const;

    /// 后段阶段耗时串（" 组名=毫秒 …"；下一次 Execute 覆写）
    const std::string& LastTimingBreakdown() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
