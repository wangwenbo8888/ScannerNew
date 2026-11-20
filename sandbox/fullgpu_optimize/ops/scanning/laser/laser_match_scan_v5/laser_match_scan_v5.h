/**
 * @file laser_match_scan_v5.h
 * @brief 激光线匹配扫描 V5 — 人工 8 步骨架 CPU 后段算子（纯 CPU，无 CUDA 依赖）
 *
 * 设计定稿：归档/记录文档/专题-激光匹配算子laser_match_scan_v5-设计方案-2026-11-11.md（v2.1）。
 * 骨架＝人工 8 步（一字不动）：支持度 A 门备选 → 票数比 B 门级联消元 →
 * D×B 误差裁决 → 裁决后回环级联。补丁①~⑫ 见设计方案 §五（①~⑧ 自主修补，
 * ⑨~⑫ 对标 ZG laser_stripe_match 采纳）。
 *
 * 与 V3/V4 的关系：
 *   - 消费 V3 前段产物 LaserMatchSplitFrontV3（复用发卡，零改动；唯一新依赖＝
 *     front.cardRes 带符号残差列——前段发射开关 JMW_V3CARDRES/SetEmitCardRes）
 *   - 不移植 V4 的 P5/P6b/P7（骨架不含拆段/合并/冲突重选——v2.1 撤销清单）
 *   - 后段零查表（误差用前段预存值，V5 增量①的直接兑现）；温表/温度无需注入
 *
 * 定稿原则（设计方案 §一）：P1 宁可放掉也不错误匹配；P2 镂空工件必须可扫
 * （votes_floor=4 / 全票段下限 1）；P3 尽量简化、速度优先；P4 骨架不动只打补丁。
 *
 * 安全设计：
 *   - cardRes 空态（发射关）＝配置矛盾 → Execute 直接失败（fail-safe）
 *   - 深度门（补丁⑩）需 SetStereoQ 注入 4×4 Q，未注入自动关（V4 同契约）
 *
 * ⚠ 沙箱定位：TEST/fullgpu_optimize 实验算子，不入 mod_operatorlib，不入产线。
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <opencv2/core.hpp>             // cv::Mat（SetStereoQ）
#include <nlohmann/json.hpp>            // toJson/fromJson
#include "laser_match_scan_v3_split.h"  // 带出 ParamsV3 / FrontV3 / CpuResultV3

namespace calib {

// ============================================================================
// LaserMatchScanParamsV5 — v3 组合＋v5 新键（默认值＝v2.1 §六 定稿；
// 未验证机制（深度门/残差清洗）代码默认关，config 显式开——D7 安全原则）
// ============================================================================

struct LaserMatchScanParamsV5 {
    LaserMatchScanParamsV3 v3;    // 组合持有（非继承：v3 参数单一事实源，整体注入）
                                  //（后段实际消费 epipolar_row_step；其余为前段配置同节装载）

    // —— 骨架四参数（人工定）——
    float share_fast = 0.8f;          // 第2步：支持度门 A（> 此值的线号入备选）
    float margin_min = 2.0f;          // 第4步：票数比门 B（≥ 此值才可定）
    float quality_e_min = 1.9f;       // 第7步：E=D×B 融合分门（≥ 此值才判）
    int votes_floor = 4;              // 第4步：有第二名时第一名票数下限（P2 镂空件）
    // —— 补丁新增 ——
    int votes_floor_solo = 2;         // 补丁③(b)：全票无第二名段下限。【261113 D″】1→2
                                      //   （1 点孤立串零复核、实测 2/3 是垃圾、零覆盖风险——
                                      //   与入口显式丢单点配套）
    int big_seg_min_pts = 60;         // 【补丁⑬】大段阈值（≥此值＝大段，优先判）
    int small_seg_max_pts = 4;        // 【D″·261113】小段上限（≤此值＝小段，三阶段用）。
                                      //   阶段分桶：大≥big_seg_min_pts / 中(small+1..big-1) /
                                      //   小(2..small)。1 点串在入口显式丢弃（kDropTinySeg）。
    bool res_gate = false;            // 【D″·261113】逐点残差门（代码默认关 D7；配置开）。
                                      //   小段（赢家卡数≤4＝⑪豁免同界）定号后逐点查
                                      //   赢家卡|带符号残差|>τ → 丢该点；幸存输出点<2 → 整段弃。
    float res_gate_k = 2.7f;          //   τ = max(k×大段赢家卡|res|中位, 0.4×res_clean_thresh)
                                      //   扫参 2.2~3.0（仿真拐点 1.8px≈k2.7：清 50% 垃圾，真点存活 93%）
    bool margin_cand_scope = false;   // 【补丁⑭·人工 261112】B 分母候选集豁免：true＝第二名只在
                                      //   候选集内取（无候选对手→B=∞ 直接过投票门）；false＝全量口径（原版）。
                                      //   动机：非候选对手（A≤share）永远不可能赢得本段，其票数不应行使
                                      //   否决权——0.8/2.0 臂 53 个被弃段中 52 个死于此（对手 75~78% 支持、
                                      //   进不了候选集，却把 B 压到 <2 挡住 100% 支持的第一名）。
    bool judge_rival_admit = false;   // 【补丁⑮·人工 261112】裁决道对手准入：true＝B<margin 的段进
                                      //   裁决时，可比集＝票数前两名（第二名不管是否候选都可受审）；
                                      //   裁决赢家若为非候选第二名 → 整段弃（宁弃不错——没资格的线号
                                      //   不许借裁判上位）。false＝原版（可比集限候选内，不足两名拒收）。
                                      //   动机（修正后数据）：4731 个接近段中 90% 残差清晰支持第一名；
                                      //   190 段（4%）票多线残差 1.47px vs 票少线 0.82px＝大段自信错投
                                      //   形态——基线投票道（B≥1.05）从不查它们的残差＝残差重尾坏帧病灶。
    float quality_sep_px = 0.5f;      // 补丁⑥：分离门（次小−最小平均误差 ≥ 此值才许判）
    bool vote_weight_log = false;     // 补丁⑨：true＝B 因子取 ln(c1)/ln(c2)（ZG 形态，
                                      //   ln(1)=0 单票自动否决）；false＝线性 c1/c2（人工原式）
    bool depth_gate = false;          // 补丁⑩：卡片级深度门（需 Q；代码默认关，D7）
    float depth_min = 200.0f;         //   深度区间下限（mm）
    float depth_max = 1500.0f;        //   深度区间上限（mm）
    bool res_clean = false;           // 补丁⑪：输出前按段残差众数清洗（代码默认关，D7）
    float res_clean_thresh = 1.0f;    //   自适应门基准（×1.5，钳 [0.4,1.2]px——ZG LineFilter 同构）
    bool res_output = true;           // 补丁⑫：逐点带符号残差输出（默认开，零成本）
    float no_support_warn_ratio = 0.30f;  // D1：无支持段占比超此值告警"建议重新校准"
    // —— 补丁⑰·line_clean（线连续性清洗；三轮代理方案 v3·261118）——
    // 与⑪/D″分层独立 OR：⑪管对表错配（大幅），⑰管线内高频抖动。基准＝线内
    // 相对形貌（赢家卡残差序列 r(idx) 的滑窗 Hampel），对温漂（平行/旋转）
    // 机制性免疫——中位/MAD 均序统计量，斜坡被滑窗中值吸收、常数被差分消除。
    bool line_clean = false;          // 总开关（D7 默认关）。挂点＝趟B 后 gather 前
                                      //   （吃幸存口径），lcSkip_ 惰性标记（⑪同款）
    float line_clean_k = 3.5f;        // 剔门 ×σ̂（3.5σ 高斯单侧 2.3e-4，20k 点≈5 点/帧）
    int line_clean_gap_rows = 3;      // gap 断缝阈：相邻行号差＞此值切 run（统计不跨缝）
    float line_clean_cap_ratio = 0.05f;  // 单线剔点率帽：超出→该线本帧全撤（故障安全
                                        //   ——部分放行无统计依据；防串线/系统性误杀）
    // ⑰ 常量（方案定稿，有意不设键——P3）：win=5（半窗 2）、min_run=5（与⑪ n<5
    //   同界）、σ_floor=0.3px（防 MAD 坍缩）。推回（阶段2）整体搁置：代表点制融合
    //   下推回＝固化合成表面；复活条件见三轮方案（GBA 非回归＋台阶真值＋契约扩展）。

    void validate() const {
        v3.validate();   // v3 全量校验在前
        if (votes_floor < 1)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::votes_floor must be >= 1");
        if (votes_floor_solo < 1)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::votes_floor_solo must be >= 1");
        if (big_seg_min_pts < 0)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::big_seg_min_pts must be >= 0 (0=off)");
        if (small_seg_max_pts < 0)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::small_seg_max_pts must be >= 0");
        if (big_seg_min_pts > 0 && small_seg_max_pts >= big_seg_min_pts)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::small_seg_max_pts must be < big_seg_min_pts");
        if (res_gate_k <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::res_gate_k must be > 0");
        if (margin_min < 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::margin_min must be >= 1");
        if (quality_e_min < 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::quality_e_min must be >= 1");
        if (share_fast <= 0.0f || share_fast > 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::share_fast must be in (0,1]");
        if (quality_sep_px < 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::quality_sep_px must be >= 0");
        if (depth_min >= depth_max)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::depth_min must be < depth_max");
        if (res_clean_thresh <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::res_clean_thresh must be > 0");
        if (no_support_warn_ratio <= 0.0f || no_support_warn_ratio > 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::no_support_warn_ratio must be in (0,1]");
        if (line_clean_k <= 0.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::line_clean_k must be > 0");
        if (line_clean_gap_rows < 1)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::line_clean_gap_rows must be >= 1");
        if (line_clean_cap_ratio <= 0.0f || line_clean_cap_ratio > 1.0f)
            throw std::invalid_argument(
                "LaserMatchScanParamsV5::line_clean_cap_ratio must be in (0,1]");
    }

    nlohmann::json toJson() const {
        nlohmann::json j = v3.toJson();   // 平铺合并 v3 键（与 fromJson 同一 node 契约）
        j["share_fast"] = share_fast;
        j["margin_min"] = margin_min;
        j["quality_e_min"] = quality_e_min;
        j["votes_floor"] = votes_floor;
        j["votes_floor_solo"] = votes_floor_solo;
        j["big_seg_min_pts"] = big_seg_min_pts;
        j["small_seg_max_pts"] = small_seg_max_pts;
        j["res_gate"] = res_gate;
        j["res_gate_k"] = res_gate_k;
        j["margin_cand_scope"] = margin_cand_scope;
        j["judge_rival_admit"] = judge_rival_admit;
        j["quality_sep_px"] = quality_sep_px;
        j["vote_weight_log"] = vote_weight_log;
        j["depth_gate"] = depth_gate;
        j["depth_min"] = depth_min;
        j["depth_max"] = depth_max;
        j["res_clean"] = res_clean;
        j["res_clean_thresh"] = res_clean_thresh;
        j["res_output"] = res_output;
        j["no_support_warn_ratio"] = no_support_warn_ratio;
        j["line_clean"] = line_clean;
        j["line_clean_k"] = line_clean_k;
        j["line_clean_gap_rows"] = line_clean_gap_rows;
        j["line_clean_cap_ratio"] = line_clean_cap_ratio;
        return j;
    }

    static LaserMatchScanParamsV5 fromJson(const nlohmann::json& j) {
        LaserMatchScanParamsV5 p;
        p.v3 = LaserMatchScanParamsV3::fromJson(j);   // 同 node 解析 v3 基（含容错＋校验）
        p.share_fast = j.value("share_fast", p.share_fast);
        p.margin_min = j.value("margin_min", p.margin_min);
        p.quality_e_min = j.value("quality_e_min", p.quality_e_min);
        p.votes_floor = j.value("votes_floor", p.votes_floor);
        p.votes_floor_solo = j.value("votes_floor_solo", p.votes_floor_solo);
        p.big_seg_min_pts = j.value("big_seg_min_pts", p.big_seg_min_pts);
        p.small_seg_max_pts = j.value("small_seg_max_pts", p.small_seg_max_pts);
        p.res_gate = j.value("res_gate", p.res_gate);
        p.res_gate_k = j.value("res_gate_k", p.res_gate_k);
        p.margin_cand_scope = j.value("margin_cand_scope", p.margin_cand_scope);
        p.judge_rival_admit = j.value("judge_rival_admit", p.judge_rival_admit);
        p.quality_sep_px = j.value("quality_sep_px", p.quality_sep_px);
        p.vote_weight_log = j.value("vote_weight_log", p.vote_weight_log);
        p.depth_gate = j.value("depth_gate", p.depth_gate);
        p.depth_min = j.value("depth_min", p.depth_min);
        p.depth_max = j.value("depth_max", p.depth_max);
        p.res_clean = j.value("res_clean", p.res_clean);
        p.res_clean_thresh = j.value("res_clean_thresh", p.res_clean_thresh);
        p.res_output = j.value("res_output", p.res_output);
        p.no_support_warn_ratio = j.value("no_support_warn_ratio", p.no_support_warn_ratio);
        p.line_clean = j.value("line_clean", p.line_clean);
        p.line_clean_k = j.value("line_clean_k", p.line_clean_k);
        p.line_clean_gap_rows = j.value("line_clean_gap_rows", p.line_clean_gap_rows);
        p.line_clean_cap_ratio = j.value("line_clean_cap_ratio", p.line_clean_cap_ratio);
        p.validate();
        return p;
    }
};

// ============================================================================
// LaserMatchScanCpuResultV5 — v3 结果＋v5 诊断（值语义）
// ============================================================================

struct LaserMatchScanCpuResultV5 : public LaserMatchScanCpuResultV3 {
    // ── 入口三分计数（补丁①/D1）──
    int noCardSegCount = 0;           // 整段无卡（A 分母为 0——不算设备误差信号）
    int noSupportSegCount = 0;        // 候选集空（所有线号 A ≤ share_fast——设备误差大信号）
    bool noSupportWarned = false;     // 占比超 no_support_warn_ratio 触发告警（观测）
    // ── D″新增诊断 ──
    int tinyDroppedSegCount = 0;      // 【D″】入口丢单点段（carded==1，零复核——独立计数）
    int elimCandSegCount = 0;         // 【D″/F2】候选被消元清空但票有残留的即时弃段
    int survivorDroppedSegCount = 0;  // 【D″】逐点门后幸存<2 整段弃（防单点复活）
    int resGateDroppedPts = 0;        // 【D″】逐点门丢弃的输出点数
    float resGateTauPx = -1.0f;       // 【D″】本帧逐点门 τ（-1=门关/无大段）
    // ── 循环/裁决计数 ──
    int eliminatedAllVotesSegCount = 0;  // 票被消元删光（线号已被邻居合法占用）
    int exitDroppedSegCount = 0;         // 兜底弃（第7步不入口或 E/分离门不达标）
    int qualityJudgedCount = 0;          // 第7步实际判定成功的段数（回环轮次计）
    int qualityFlipCount = 0;            // 裁决翻案数（判定线号 ≠ 投票第一名）
    // ── 补丁⑩⑪⑫计数 ──
    int depthDroppedCards = 0;        // 深度门掩蔽卡数（机制关时恒 0）
    int resCleanDropped = 0;          // 残差清洗丢弃的输出点数（机制关时恒 0）
    float resMedianPx = -1.0f;        // 输出对带符号残差中位（px；无输出对＝-1）
    // ── 补丁⑫：输出对带符号残差流（>0＝实际右点在表预测右侧＝表预测偏低）──
    std::vector<float> matched_res;   // 与 matched_* 同序同长；res_output=false 时为空
    // ── 补丁⑰·line_clean 诊断（机制关恒 0）──
    int lineCleanDroppedPts = 0;      // 净剔点数（率帽撤销后实际生效）
    int lineCleanFallbackLines = 0;   // 率帽回退线数（该线本帧全撤——故障安全）
};

// ============================================================================
// LaserMatchScanSplitCpuV5 — v5 CPU 后段算子（纯 CPU，无 CUDA 依赖）
// ============================================================================

// ===== 算子规范 §4 状态模型 =====
// 状态类别: 调用方持有（v3+v5 参数注入；Q 经注入；结果缓冲容量式复用）
// 并发策略: 每实例非线程安全，多实例并行各自独占（单线程执行）
// ==============================

class LaserMatchScanSplitCpuV5 {
public:
    static constexpr const char* kLogTag = "07-LaserMatchSplitCpuV5";

    explicit LaserMatchScanSplitCpuV5(const LaserMatchScanParamsV5& params = {});
    ~LaserMatchScanSplitCpuV5();

    LaserMatchScanSplitCpuV5(const LaserMatchScanSplitCpuV5&) = delete;
    LaserMatchScanSplitCpuV5& operator=(const LaserMatchScanSplitCpuV5&) = delete;

    // 注：无 SetTempTable/SetCurrentTemperature——V5 后段零查表（cardRes 前段预存，
    // 增量①的目的）；温表由前段算子持有。

    /// 4×4 重投影 Q（CV_64F）注入，供深度门（补丁⑩）使用；空/非 4×4/非 CV_64F →
    /// 深度门自动关。Z＝Q(2,3)/(Q(3,2)·d＋Q(3,3))，d＝xL−hitRx（V4 同式）。
    /// 设计方案记 ZG 廉价形式 ρ=P/(disp−D)，但 K2 无该组标定常数——Q 式一次除法
    /// 同价且 Q 在 stereoRectifyTempTable 现成，故实现取 Q 式（等价数学）。
    void SetStereoQ(const cv::Mat& Q);

    /// 后段执行：消费 v3 前段产物 → v5 结果（值语义）。
    /// 入口检查 front.success；cardRes 为空（前段发射关）＝配置矛盾 → 失败返回。
    LaserMatchScanCpuResultV5 Execute(const LaserMatchSplitFrontV3& front);

    void Destroy();
    void SetParams(const LaserMatchScanParamsV5& params);
    const LaserMatchScanParamsV5& GetParams() const;

    /// 后段阶段耗时串（" 组名=毫秒 …"；下一次 Execute 覆写）
    const std::string& LastTimingBreakdown() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace calib
