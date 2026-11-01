/**
 * @file laser_match_scan_v3_cpu.h
 * @brief 激光线匹配扫描 V3 — 全 CPU 单线程备用算子（沙箱专用）
 *
 * 2026-10-21 为 GPU 算子 LaserMatchScanCudaV3 制作的全 CPU 备用实现：
 *   - 功能以 GPU 版为基准（段级投票＋冲突双弃＋P5 冲突重选/P6b 段分裂/P7 级联全路径）
 *   - 算法移植逐级对齐 laser_match_scan_v3_cuda_impl.cu（含桶索引查表锚点、
 *     量化键右行稳定序、双弃净语义），单线程确定性执行
 *   - 设计/评审记录见 TEST/speed_optimize/laser_match_v3_cpu_设计方案.md
 *
 * ⚠ 备用/沙箱定位：不替代产线 GPU 算子，不入 mod_operatorlib。
 *
 * 与 GPU 版的差异（有意为之）：
 *   - 输入/输出为 cv::Mat / std::vector（无 GpuMat 视图生命周期问题）
 *   - 无段数上限（GPU kSegCap=4096 溢出兜底路径在 CPU 自然不需要）
 *   - uRBias 结果字段 GPU 从不回填，本类不设
 *   - maxRightPerRow 参数仅 validate（GPU V3 同样不消费）
 */

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>
#include "laser_match_scan_v3_cuda.h"   // 复用 LaserMatchScanParamsV3（单一事实源；该头 CUDA-free）

namespace calib {

struct WarmupConfig;

// ============================================================================
// LaserMatchScanCpuResultV3 — 值语义结果（无 ROI 视图契约）
// ============================================================================

struct LaserMatchScanCpuResultV3 {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    cv::Mat matched_left;        // N×1 CV_32FC2（匹配左点，原输入坐标）
    cv::Mat matched_right;       // N×1 CV_32FC2（hitRx, 左点 y）——y＝左点 y（与 GPU 对齐）
    cv::Mat matched_line_ids;    // N×1 CV_32SC1（winner lid）
    cv::Mat matched_track_ids;   // N×1 CV_32SC1（左点 track id）
    int matchedCount = 0;

    // status 语义同 GPU：left -1=noMap（行无表覆盖/越界） 0=noCard 1=有 HIT 卡；
    // right 为右行排序系索引空间，1＝被匹配消费；每帧覆写
    std::vector<int> left_status;
    std::vector<int> right_status;

    int totalLeftPoints = 0;
    int totalRightPoints = 0;
    int excludedLeftCount = 0;     // ＝ noCard 数（不含 noMap，与 GPU 口径一致）
    int excludedRightCount = 0;    // ＝ rightCount − rightUsed
    int noMapLeftCount = 0;
    int noWinnerCardCount = 0;
    int emptyWinnerSegCount = 0;

    int trackTotal = 0;            // 段数（P6b 分裂后）
    int trackDropped = 0;          // 被弃段数
};

// ============================================================================
// LaserMatchScanCpuV3
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有（同 GPU 版：温表/温度注入式；结果为容量式持久缓冲覆写）
// 并发策略: 每实例非线程安全，多实例并行各自独占（单线程执行）
// ==============================

class LaserMatchScanCpuV3 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchScanCpuV3";

    explicit LaserMatchScanCpuV3(const LaserMatchScanParamsV3& params = {});
    ~LaserMatchScanCpuV3();

    LaserMatchScanCpuV3(const LaserMatchScanCpuV3&) = delete;
    LaserMatchScanCpuV3& operator=(const LaserMatchScanCpuV3&) = delete;

    bool LoadTempTable(const std::string& jsonPath);

    /// 只读共享温表注入（与 GPU 版同源跳过缓存：同源指针＋同温＋同 rowStep）
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);

    void SetCurrentTemperature(double temperature);

    /// 输入与 GPU 版同构：N×1（或 1×N）CV_32FC2 点 ＋ N×1 CV_32SC1 track id
    LaserMatchScanCpuResultV3 Execute(
        const cv::Mat& left_points,
        const cv::Mat& left_line_ids,
        const cv::Mat& right_points,
        const cv::Mat& right_line_ids);

    void Destroy();

    void Warmup(int maxLeftPoints, int maxRightPoints);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserMatchScanParamsV3& params);
    const LaserMatchScanParamsV3& GetParams() const;

    /// uR 逐线偏差补偿（与 GPU 互斥语义一致：后设者生效）
    void SetuRBias(const std::vector<float>& biasPerLid);

    /// uR 4 参数位置相关偏差模型（公式与常数逐字对齐 GPU 版）
    void SetuRBiasModel(float wx, float wy, float wz, float dtx);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMatchScanCpuV3Info();

} // namespace calib
