/**
 * @file laser_match_scan_v4.h
 * @brief 激光线匹配扫描 V4 — 裁判型后段算子（V3 拆分流水线之上的增强版）— 公开接口
 *
 * 定位（REA 对标激光匹配 v4）：复用 v3 GPU 前段零改动（LaserMatchScanSplitGpuV3 /
 * LaserMatchSplitFrontV3 原样消费），本算子替换 CPU 后段，在 v3 判定链之上叠加三件
 * 抗污染机制：
 *   1. 双分数体系（quality_elect）：整数票仓之外并行一套「逐卡残差分数」票仓
 *      （cntQ/sumRes＝|hitRx−表预测 uR|）；争议段的 winner 可被残差最优 lid 翻转，
 *      受分离门 quality_separate_px 约束。
 *   2. 自适应窗（adapt_window）：匹配有效窗随相邻线间距自适应收紧
 *      （w_eff＝max(floor, min(mt, 0.5×间距))）；超有效窗的卡被 mask（不进分数票仓、
 *      不作 winner 输出卡）。
 *   3. 弃权门（abstain）：争议段既无双分数分离、又过不了 margin/share 双门 → 整段
 *      弃权（dropped），不再被迫输出错线号。
 *
 * 2026-10-10 残差加权证伪史的三条防护（本次铁律）：
 *   - 仅争议段介入——非争议段（winner 票数占优明显）逐字节放行＝零漂移；
 *   - 分离门——最优与次优残差差距须超过 quality_separate_px 才允许翻转（分得开），
 *     分不开则弃权或维持，杜绝「微差翻转」重演；
 *   - 票仓不动——v3 整数票仓（segVotes）与全部判定序列逐字保留；分数票仓为独立
 *      数组（cntQ_/sumRes_），永不写回 votes。
 *
 * 附加门：card_depth_gate（4×4 重投影 Q 视差深度门，深度区间 [min,max]mm，经
 * SetStereoQ 注入 Q，未注入时该门自动关闭）；quality_lid_norm（帧内逐 lid 残差中位
 * 归一，去除线号系统性偏差后再比较）。
 *
 * L0 恒等契约：v4 四开关全关（或温表/温度未就绪）→ v4 通道整体跳过，与
 * LaserMatchScanSplitCpuV3 同输入同参数输出逐位一致（除 V4 结果类型新增的零值字段）。
 *
 * 261115 回迁产线：自 TEST/fullgpu_optimize 沙盒移植（V5 的回退臂；沙盒归档/记录文档/ 有专题档案）。
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <opencv2/core.hpp>             // cv::Mat（SetStereoQ）
#include <nlohmann/json.hpp>            // toJson/fromJson
#include "laser_match_scan_v3_split.h"  // 带出 ParamsV3 / ResultV3 / FrontV3 / TempTable 类型

namespace calib {

// ============================================================================
// LaserMatchScanParamsV4 — v3 组合＋v4 新开关（默认值＝v4 推荐配置）
// ============================================================================

struct LaserMatchScanParamsV4 {
    LaserMatchScanParamsV3 v3;    // 组合持有（非继承：v3 参数为单一事实源，整体注入）

    // —— 双分数体系（quality elect）——
    bool quality_elect = true;          // 争议段残差分数重选
    float quality_separate_px = 1.0f;   // 分离门：最优−次优 meanRes 差距阈值（px）
    int quality_cnt_veto = 2;           // 分数票仓候选门槛（cntQ ≥ veto 才可当选）
    bool quality_lid_norm = false;      // 帧内逐 lid 残差中位归一（去线号系统偏差）
    bool res_channel = false;           // 逐点残差输出通道（matched_res/resMedianPx；
                                        // 标定观测用——每输出点一次查表，默认关）

    // —— 自适应窗 ——
    bool adapt_window = true;           // 有效窗随相邻线间距收紧
    float adapt_window_floor = 2.0f;    // 有效窗下限（px，防间距估计过小全 mask）

    // —— 弃权门 ——
    bool abstain = true;                // 争议不可分 → 整段弃权
    float abstain_margin = 1.3f;        // W ≥ margin×S 视为票数安全
    float abstain_share = 0.60f;        // W ≥ share×T 视为份额安全
    float abstain_gap_factor = 2.0f;    // R2b 路径：gap < factor×mt 视为间距可疑

    // —— 深度门 ——
    bool card_depth_gate = false;       // 视差深度门（需 SetStereoQ 注入 Q）
    float card_depth_min = 200.0f;      // 深度区间下限（mm）
    float card_depth_max = 1500.0f;     // 深度区间上限（mm）

    // —— P6b 拆段门槛（v4 参数化；原硬编码 max(30, 30%)）——
    int p6b_min_votes = 30;             // 双峰次名票数下限（绝对值）
    int p6b_vote_pct = 30;              // 双峰次名票数下限（winner 的百分比）

    void validate() const {
        v3.validate();   // v3 全量校验在前
        if (adapt_window_floor <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::adapt_window_floor must be > 0");
        if (abstain_share <= 0.0f || abstain_share >= 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::abstain_share must be in (0,1)");
        if (abstain_margin <= 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::abstain_margin must be > 1.0");
        if (abstain_gap_factor <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::abstain_gap_factor must be > 0");
        if (quality_cnt_veto < 1)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::quality_cnt_veto must be >= 1");
        if (quality_separate_px <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::quality_separate_px must be > 0");
        if (card_depth_min >= card_depth_max)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::card_depth_min must be < card_depth_max");
        if (p6b_min_votes < 1 || p6b_vote_pct < 1 || p6b_vote_pct > 100)
            throw std::invalid_argument(
                "LaserMatchScanParamsV4::p6b gate out of range (min>=1, pct 1..100)");
    }

    nlohmann::json toJson() const {
        nlohmann::json j = v3.toJson();   // 平铺合并 v3 键（与 fromJson 同一 node 契约）
        j["quality_elect"] = quality_elect;
        j["quality_separate_px"] = quality_separate_px;
        j["quality_cnt_veto"] = quality_cnt_veto;
        j["quality_lid_norm"] = quality_lid_norm;
        j["res_channel"] = res_channel;
        j["adapt_window"] = adapt_window;
        j["adapt_window_floor"] = adapt_window_floor;
        j["abstain"] = abstain;
        j["abstain_margin"] = abstain_margin;
        j["abstain_share"] = abstain_share;
        j["abstain_gap_factor"] = abstain_gap_factor;
        j["card_depth_gate"] = card_depth_gate;
        j["card_depth_min"] = card_depth_min;
        j["card_depth_max"] = card_depth_max;
        j["p6b_min_votes"] = p6b_min_votes;
        j["p6b_vote_pct"] = p6b_vote_pct;
        return j;
    }

    static LaserMatchScanParamsV4 fromJson(const nlohmann::json& j) {
        LaserMatchScanParamsV4 p;
        // v3 子对象用 LaserMatchScanParamsV3::fromJson(同一 node) 解析
        //（含自身的键存在性容错＋隐含开关归一＋validate）
        p.v3 = LaserMatchScanParamsV3::fromJson(j);
        // 新键用 value("键", 默认) 容错（缺键→默认值）
        p.quality_elect = j.value("quality_elect", p.quality_elect);
        p.quality_separate_px = j.value("quality_separate_px", p.quality_separate_px);
        p.quality_cnt_veto = j.value("quality_cnt_veto", p.quality_cnt_veto);
        p.quality_lid_norm = j.value("quality_lid_norm", p.quality_lid_norm);
        p.res_channel = j.value("res_channel", p.res_channel);
        p.adapt_window = j.value("adapt_window", p.adapt_window);
        p.adapt_window_floor = j.value("adapt_window_floor", p.adapt_window_floor);
        p.abstain = j.value("abstain", p.abstain);
        p.abstain_margin = j.value("abstain_margin", p.abstain_margin);
        p.abstain_share = j.value("abstain_share", p.abstain_share);
        p.abstain_gap_factor = j.value("abstain_gap_factor", p.abstain_gap_factor);
        p.card_depth_gate = j.value("card_depth_gate", p.card_depth_gate);
        p.card_depth_min = j.value("card_depth_min", p.card_depth_min);
        p.card_depth_max = j.value("card_depth_max", p.card_depth_max);
        p.p6b_min_votes = j.value("p6b_min_votes", p.p6b_min_votes);
        p.p6b_vote_pct = j.value("p6b_vote_pct", p.p6b_vote_pct);
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchScanCpuResultV4 — v3 结果＋v4 诊断（值语义）
// ============================================================================

struct LaserMatchScanCpuResultV4 : public LaserMatchScanCpuResultV3 {
    // v4 裁判计数（v4 通道关闭时全零）
    int qualityFlipCount = 0;      // 双分数分离门通过的 winner 翻转段数
    int abstainSegCount = 0;       // 弃权段数（分离失败/无候选/R2b 双门不过）
    int windowShrinkDropped = 0;   // 自适应窗收紧 mask 的卡数
    int cardDepthDropped = 0;      // 深度门 mask 的卡数（含非正视差 d<=0）
    int adaptNoCardPts = 0;        // winner 卡被 mask 而按无卡处理的点数
    int spacingMissCount = 0;      // 卡 lid 查表缺失（残差回退 mt；R2b 缺样本同计）
    int singleVoteVetoCount = 0;   // 分数票仓中 cntQ<veto 被 veto 的 (段,lid) 计数
    float resMedianPx = -1.0f;     // 输出对残差中位（px；无输出对＝-1）

    // 输出对残差流（与 matched_left/right/line_ids/track_ids 同序同长；
    // v4 通道关闭时为空）
    std::vector<float> matched_res;
};

// ============================================================================
// LaserMatchScanSplitCpuV4 — v4 CPU 后段算子（纯 CPU，无 CUDA 依赖）
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有（v3 参数＋v4 开关注入；温表/温度/Q 经注入；结果缓冲容量式复用）
// 并发策略: 每实例非线程安全，多实例并行各自独占（单线程执行）
// ==============================

class LaserMatchScanSplitCpuV4 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchSplitCpuV4";

    explicit LaserMatchScanSplitCpuV4(const LaserMatchScanParamsV4& params = {});
    ~LaserMatchScanSplitCpuV4();

    LaserMatchScanSplitCpuV4(const LaserMatchScanSplitCpuV4&) = delete;
    LaserMatchScanSplitCpuV4& operator=(const LaserMatchScanSplitCpuV4&) = delete;

    /// 温表注入（双槽缓存：同源指针命中免重转换；最近两份表可热切换）
    bool SetTempTable(std::shared_ptr<const LaserPlaneMapTempTable> table);

    /// 设定当前温度（未载表 throw；就近解析温档，缓存命中免重建）
    void SetCurrentTemperature(double temperature);

    /// 4×4 重投影 Q（CV_64F）注入，供深度门使用；空/非 4×4/非 CV_64F →
    /// 深度门自动关（hasQ_=false）。Z＝Q(2,3)/(Q(3,2)·d＋Q(3,3))，d＝xL−hitRx。
    void SetStereoQ(const cv::Mat& Q);

    /// 后段执行：消费 v3 前段产物 → v4 结果（值语义）。
    /// 入口检查 front.success，失败原样透传失败结果。
    /// v4 四开关全关或温表/温度未就绪 → 与 v3 后段逐位一致（L0 恒等）。
    LaserMatchScanCpuResultV4 Execute(const LaserMatchSplitFrontV3& front);

    void Destroy();
    void SetParams(const LaserMatchScanParamsV4& params);
    const LaserMatchScanParamsV4& GetParams() const;

    /// 后段阶段耗时串（" 组名=毫秒 …"；v4 通道活跃时追加 " v4judge=X.Xms"；
    /// 下一次 Execute 覆写）
    const std::string& LastTimingBreakdown() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
