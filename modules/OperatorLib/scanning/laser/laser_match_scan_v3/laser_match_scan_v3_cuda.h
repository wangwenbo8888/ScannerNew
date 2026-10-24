/**
 * @file laser_match_scan_v3_cuda.h
 * @brief 激光线匹配扫描CUDA算子 V3（段级投票 + 冲突双弃）— 公开头文件
 *
 * 2026-09-05 自 factory_calib 同步入库并接替退役的 v1/v2（laser_match_scan → retired/）。
 * 算法（与旧 v2 逐点唯一命中制不同）：
 *   - 发卡制：每左点查表，每 lid 取最近条目（单卡），±match_window 验卡记 hitRx
 *   - 段级投票：输入按段组织（上游 epipolar_interp_dual Scan 模式段 ID），
 *     段内 (段,lid) 票数直方图 → argmax winner；同行两段同 winner 双弃（atomicCAS）
 *   - 保留段取 winner 卡片输出配对；金标准 wrong=0（factory_calib pose_06）
 * 性能结构（继承自早期 v2 优化版骨架）：
 *   - 32bit 排序键（row 13bit + x 19bit@1/256px 量化）、CUB 排序
 *   - 全程单次主机同步（末尾 pinned 16B 批量回传）
 *   - 结果缓冲持久化 + ROI 视图返回（有效期至下一次 Execute/Destroy/SetParams）
 *   - status 双缓冲 ping-pong（无 clone）
 *
 * ⚠ 生命周期契约：result 中所有 GpuMat 为内部持久缓冲的视图，
 *    有效期至该实例下一次 Execute() / Destroy() / setParams()。
 * 详细说明：docs/算子说明文档/scanning/laser/laser_match_scan_v3/
 */

#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <stdexcept>
#include "common/calib_types.h"
#include "common/calib_result_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace cv { namespace cuda { class GpuMat; class Stream; } }

namespace calib {

struct WarmupConfig;

// ============================================================================
// LaserMatchScanParamsV3
// ============================================================================

struct LaserMatchScanParamsV3 {
    float match_threshold = 1.0f;
    float epipolar_row_step = 0.5f;
    int max_right_per_row = 1024;   // 语义上限保留；V2 kernelMatch smem 容量另见 SMEM_RIGHT_CAP
    float vL_tolerance = 0.01f;
    int deviceId = 0;
    // P5（2026-10-10）：冲突重选再弃——(row,winner) 撞车时输方在自身票直方图中
    // 剔争议 lid 重选次优（次优票数 ≥ max(2, 30%×最优)），无合格次优才整段弃；
    // host 侧确定性求解。false＝原版双弃语义（产线默认）。
    bool conflict_reelect = false;
    // P6b（2026-10-10）：段分裂——interp 串线把两条物理线并进一个段时（票直方图双峰），
    // 三门槛（双峰票≥30%、次线票主来自独占点≥80%、次线点行块连续）满足则按卡归属
    // 拆成两个子段各自成 winner。依赖 conflict_reelect 的 host 求解路径。
    bool segment_split = false;
    // P7（2026-10-12 原型·默认关）：帧内级联判定——P6b 分裂后替代 P5 冲突循环。
    // 流程：几何友军合并（同帧+共享行+横向连续）→ 门槛级联（过门槛的最高票段先判，
    // 判定线号从行重叠未定段抹除，循环）→ 事后安全检查（撞号/相邻间距，违规禁赛重跑）。
    // 离线 399 帧验证（无补偿口径）：判错 0 / 帧间线号稳定 98.9% / 点数口径未定 14%
    // （对比无补偿现状：缺失 21% 且有错标）。详见 TEST/steger_dedup_replay/会话交接-2026-10-08-激光链调试与匹配优化.md（续篇）§3.6。
    // cascade_elim=true 隐含 conflict_reelect=true（host 路径）。
    bool cascade_elim = false;
    int cascade_gate_h = 30;            // 门槛：winner 票数下限
    float cascade_gate_share = 0.70f;   // 门槛：winner 占段总票比例下限
    float cascade_gate_margin = 2.0f;   // 门槛：winner/次名 下限
    float cascade_friend_xpx = 15.0f;   // 友军合并：共享行横向均距上限（px）
    int cascade_friend_minrows = 3;     // 友军合并：最少共享行数
    float cascade_spacing_px = 36.9f;   // 间距检查：相邻线距标定值（px，随机型标定）
    float cascade_spacing_tol = 0.55f;  // 间距检查：容差比例（×spacing_px）

    void validate() const {
        if (match_threshold <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::match_threshold must be > 0");
        if (epipolar_row_step <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::epipolar_row_step must be > 0");
        if (max_right_per_row <= 0)
            throw std::invalid_argument("LaserMatchScanParamsV3::max_right_per_row must be > 0");
        if (vL_tolerance < 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::vL_tolerance must be >= 0");
        if (deviceId < 0)
            throw std::invalid_argument("LaserMatchScanParamsV3::deviceId must be >= 0");
        if (cascade_gate_h < 1 || cascade_gate_h > 100000)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_gate_h out of range");
        if (cascade_gate_share <= 0.0f || cascade_gate_share >= 1.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_gate_share must be in (0,1)");
        if (cascade_gate_margin < 1.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_gate_margin must be >= 1");
        if (cascade_friend_xpx <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_friend_xpx must be > 0");
        if (cascade_friend_minrows < 1)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_friend_minrows must be >= 1");
        if (cascade_spacing_px <= 0.0f || cascade_spacing_tol <= 0.0f)
            throw std::invalid_argument("LaserMatchScanParamsV3::cascade_spacing_px/tol must be > 0");
    }

    nlohmann::json toJson() const {
        return {
            {"match_threshold", match_threshold},
            {"epipolar_row_step", epipolar_row_step},
            {"max_right_per_row", max_right_per_row},
            {"vL_tolerance", vL_tolerance},
            {"deviceId", deviceId},
            {"conflict_reelect", conflict_reelect},
            {"segment_split", segment_split},
            {"cascade_elim", cascade_elim},
            {"cascade_gate_h", cascade_gate_h},
            {"cascade_gate_share", cascade_gate_share},
            {"cascade_gate_margin", cascade_gate_margin},
            {"cascade_friend_xpx", cascade_friend_xpx},
            {"cascade_friend_minrows", cascade_friend_minrows},
            {"cascade_spacing_px", cascade_spacing_px},
            {"cascade_spacing_tol", cascade_spacing_tol}
        };
    }

    static LaserMatchScanParamsV3 fromJson(const nlohmann::json& j) {
        LaserMatchScanParamsV3 p;
        if (j.contains("match_threshold"))
            p.match_threshold = j.at("match_threshold").get<float>();
        if (j.contains("epipolar_row_step"))
            p.epipolar_row_step = j.at("epipolar_row_step").get<float>();
        if (j.contains("max_right_per_row"))
            p.max_right_per_row = j.at("max_right_per_row").get<int>();
        if (j.contains("vL_tolerance"))
            p.vL_tolerance = j.at("vL_tolerance").get<float>();
        if (j.contains("deviceId"))
            p.deviceId = j.at("deviceId").get<int>();
        if (j.contains("conflict_reelect"))
            p.conflict_reelect = j.at("conflict_reelect").get<bool>();
        if (j.contains("segment_split"))
            p.segment_split = j.at("segment_split").get<bool>();
        if (j.contains("cascade_elim"))
            p.cascade_elim = j.at("cascade_elim").get<bool>();
        if (j.contains("cascade_gate_h"))
            p.cascade_gate_h = j.at("cascade_gate_h").get<int>();
        if (j.contains("cascade_gate_share"))
            p.cascade_gate_share = j.at("cascade_gate_share").get<float>();
        if (j.contains("cascade_gate_margin"))
            p.cascade_gate_margin = j.at("cascade_gate_margin").get<float>();
        if (j.contains("cascade_friend_xpx"))
            p.cascade_friend_xpx = j.at("cascade_friend_xpx").get<float>();
        if (j.contains("cascade_friend_minrows"))
            p.cascade_friend_minrows = j.at("cascade_friend_minrows").get<int>();
        if (j.contains("cascade_spacing_px"))
            p.cascade_spacing_px = j.at("cascade_spacing_px").get<float>();
        if (j.contains("cascade_spacing_tol"))
            p.cascade_spacing_tol = j.at("cascade_spacing_tol").get<float>();
        if (p.cascade_elim && !p.conflict_reelect)
            p.conflict_reelect = true;   // 级联依赖 host 求解路径，隐含开启
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchScanResultV3
// ============================================================================

struct LaserMatchScanResultV3 {
    bool success = false;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    // ⚠ 内部持久缓冲的 ROI 视图，有效期至下一次 Execute/Destroy/setParams
    std::shared_ptr<cv::cuda::GpuMat> d_matched_left;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_right;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_line_ids;
    std::shared_ptr<cv::cuda::GpuMat> d_matched_track_ids;   // V3: per-pair track id
    int matchedCount = 0;

    // status 为 ping-pong 双缓冲当前帧视图（有效期同上）
    std::shared_ptr<cv::cuda::GpuMat> d_left_status;
    std::shared_ptr<cv::cuda::GpuMat> d_right_status;

    int totalLeftPoints = 0;
    int totalRightPoints = 0;
    int excludedLeftCount = 0;
    int excludedRightCount = 0;

    // V3 segment diagnostics
    int trackTotal = 0;        // segments seen this frame
    int trackDropped = 0;      // segments dropped by row-conflict rule

    // V3 细分诊断（P4 实装：原静默路径可见化）
    int noMapLeftCount = 0;      // status=-1 左点（行无表覆盖/越界，原 exclL 黑洞）
    int noWinnerCardCount = 0;   // 保留段中不持 winner 卡的点（原 gather 静默丢弃）
    int emptyWinnerSegCount = 0; // winner=-1 段数（整段无任何 HIT 卡）

    // uR 逐线偏差补偿（2026-10-10 归因分析产出：+1.2→+4.7px 随线号单调增长）
    // 非空时 kernelIssueCardsV3 在验卡前对预测 uR 加上 bias[lid-1]
    // 产线默认为空（不补偿）；实验/补偿场景经 SetuRBias 注入
    std::vector<float> uRBias;   // size=0 不补偿；size>=V3_MAX_LIDS 时取前 25 个

    LaserMatchScanResultV3() = default;
    ~LaserMatchScanResultV3() = default;

    LaserMatchScanResultV3(LaserMatchScanResultV3&&) = default;
    LaserMatchScanResultV3& operator=(LaserMatchScanResultV3&&) = default;

    LaserMatchScanResultV3(const LaserMatchScanResultV3&) = delete;
    LaserMatchScanResultV3& operator=(const LaserMatchScanResultV3&) = delete;
};

// ============================================================================
// LaserMatchScanCudaV3
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有
// 说明: 温度补偿映射表与当前温度由调用方经 SetTempTable/SetCurrentTemperature 注入；
//       结果缓冲为容量式持久缓冲，每次 Execute 覆写并返回视图。
// 重置接口: N/A
// 并发策略: 每实例非线程安全（§1.4），多实例并行各自独占
// ==============================

class SCANNER_API LaserMatchScanCudaV3 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchScanCudaV3";

    explicit LaserMatchScanCudaV3(const LaserMatchScanParamsV3& params = {});
    ~LaserMatchScanCudaV3();

    LaserMatchScanCudaV3(const LaserMatchScanCudaV3&) = delete;
    LaserMatchScanCudaV3& operator=(const LaserMatchScanCudaV3&) = delete;

    bool LoadTempTable(const std::string& jsonPath);

    // 算子规范 §3.6：只读共享资源（温度补偿映射表）经注入传入，算子内部不写入。
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);

    void SetCurrentTemperature(double temperature);

    LaserMatchScanResultV3 Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids,
        cv::cuda::Stream& stream);

    LaserMatchScanResultV3 Execute(
        const cv::cuda::GpuMat& d_left_points,
        const cv::cuda::GpuMat& d_left_line_ids,
        const cv::cuda::GpuMat& d_right_points,
        const cv::cuda::GpuMat& d_right_line_ids);

    // 【流水线拆分 2026-10-21】只跑前半段：Step0-6（右排序/CSR＋发卡＋GPU 段表）
    // ＋卡表紧缩＋中间产物批量 D2H＋同步（＝Step 6.5 取段数收尾），产物写入 out
    // （值语义 host 结构，定义见 laser_match_scan_v3_split.h）。后半段（投票/冲突/
    // 收集/统计）由 CPU 算子 LaserMatchScanSplitCpuV3 消费 out 完成。
    // 前半发射与 process() 逐字同源（普通发射，不走 CUDA Graph）；status 视图契约同
    // Execute（有效期至下一次 Execute/ExecuteFront/Destroy/setParams）。
    void ExecuteFront(const cv::cuda::GpuMat& d_left_points,
                      const cv::cuda::GpuMat& d_left_line_ids,
                      const cv::cuda::GpuMat& d_right_points,
                      const cv::cuda::GpuMat& d_right_line_ids,
                      cv::cuda::Stream& stream,
                      class LaserMatchSplitFrontV3& out);

    /// 【前段提前提交·WDDM 滴流根治 2026-10-23】延迟对（fivechain 用）：提交段
    /// 在排水前按容量发射 Step0-6＋卡表前缀（聚合内核设备计数尾闸挡陈尾）；
    /// FinishFront 在排水后以精确 N 收尾（紧缩＋主批 D2H＋拷出）。两调用间不得
    /// 夹其它本算子调用；配对失败/容量视图空→success=false（执行器回退 ExecuteFront）。
    /// JMW_K2A_LEGACY/JMW_K2_LEGACY 环境由执行器侧判断（legacy 臂走旧整段）。
    void ExecuteFrontDeferred(
        const cv::cuda::GpuMat& d_left_points_cap,
        const cv::cuda::GpuMat& d_left_line_ids_cap,
        const int* d_leftCount,
        const cv::cuda::GpuMat& d_right_points_cap,
        const cv::cuda::GpuMat& d_right_line_ids_cap,
        const int* d_rightCount,
        cv::cuda::Stream& stream,
        class LaserMatchSplitFrontV3& out);

    /// 【B4·前段提交图化】JMW_V3FRONTGRAPH=1 显式开（默认关）。容量视图单图发射
    ///（~19 笔→1 笔）；语义与 ExecuteFrontDeferred 完全同协议（leftCount==-1 哨兵
    /// ＋FinishFront 收尾）；失败/图关内部回退逐发。禁与 JMW_K2*_LEGACY 并用。
    void ExecuteFrontGraph(
        const cv::cuda::GpuMat& d_left_points_cap,
        const cv::cuda::GpuMat& d_left_line_ids_cap,
        const int* d_leftCount,
        const cv::cuda::GpuMat& d_right_points_cap,
        const cv::cuda::GpuMat& d_right_line_ids_cap,
        const int* d_rightCount,
        cv::cuda::Stream& stream,
        class LaserMatchSplitFrontV3& out);
    void FinishFront(int leftCount, int rightCount,
                     cv::cuda::Stream& stream,
                     class LaserMatchSplitFrontV3& out);

    void Destroy();

    void Warmup(int maxLeftPoints, int maxRightPoints);
    void Warmup(const WarmupConfig& config);
    void SetParams(const LaserMatchScanParamsV3& params);
    const LaserMatchScanParamsV3& GetParams() const;

    /// uR 逐线偏差补偿注入（2026-10-10）：非空 vector 时验卡前 uR += bias[lid-1]
    /// 传空 vector 或不调用＝不补偿（产线默认）。长度 < V3_MAX_LIDS 时补零。
    void SetuRBias(const std::vector<float>& biasPerLid);

    /// uR 4 参数位置相关偏差模型注入（2026-10-11 方式2，SetuRBias 的升级替代）：
    /// 表预测视差→近似深度 Z→左目 3D 点→相对虚拟投影器方向的一阶旋转修正
    /// （v 向 CMOS 偏移→深度修正 dZ）→ bias = f·B·dZ/Z²，逐点计算（同一条线
    /// 内不同位置补偿量不同）。与 2026-10-11 角度拟合实验（verify_angular 族）
    /// 同链同式：拟合最优处 corr≈−kb（实测偏差 kb>0＝表预测 uR 偏低），故本
    /// 接口施加的是 −corr 方向（＝+kb 全量修正）。dtx 在扇形发射曲线一阶近似
    /// 下对深度无贡献（u 向 CMOS 偏移不改深度），当前为保留占位、不参与计算。
    /// 与 SetuRBias 互斥：非空调用使逐线表失效（反之亦然），后调用者生效。
    /// 不调用＝不补偿（产线默认）。
    void SetuRBiasModel(float wx, float wy, float wz, float dtx);

    /// 【测速工具】最近一次 Execute 的内部阶段耗时串（" 相名=毫秒 …"），
    /// 供 gpu_stress 聚合；失效条件：下一次 Execute 覆写
    std::string LastTimingBreakdown() const;

    /// 【v3全GPU化】双路径开关（运行时切换，双跑 harness 用）。GPU 路径
    /// 未实现前恒等效宿主路径。env JMW_V3FULLGPU=1 为构造期缺省值。
    void SetFullGpuPath(bool on);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

OperatorInfo getLaserMatchScanCudaV3Info();

} // namespace calib
